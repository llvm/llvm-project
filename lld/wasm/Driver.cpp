//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "Config.h"
#include "InputGlobal.h"
#include "MarkLive.h"
#include "SymbolTable.h"
#include "Writer.h"
#include "lld/Common/Args.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Threads.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::wasm;

using namespace lld;
using namespace lld::wasm;

Configuration *lld::wasm::Config;

namespace {

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11, _12) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

class LinkerDriver {
public:
  void link(ArrayRef<const char *> ArgsArr);

private:
  void createFiles(opt::InputArgList &Args);
  void addFile(StringRef Path);
  void addLibrary(StringRef Name);
  std::vector<InputFile *> Files;
  llvm::wasm::WasmGlobal StackPointerGlobal;
};
} // anonymous namespace

bool lld::wasm::link(ArrayRef<const char *> Args, bool CanExitEarly,
                     raw_ostream &Error) {
  errorHandler().LogName = Args[0];
  errorHandler().ErrorOS = &Error;
  errorHandler().ColorDiagnostics = Error.has_colors();
  errorHandler().ErrorLimitExceededMsg =
      "too many errors emitted, stopping now (use "
      "-error-limit=0 to see all errors)";

  Config = make<Configuration>();
  Symtab = make<SymbolTable>();

  LinkerDriver().link(Args);

  // Exit immediately if we don't need to return to the caller.
  // This saves time because the overhead of calling destructors
  // for all globally-allocated objects is not negligible.
  if (CanExitEarly)
    exitLld(errorCount() ? 1 : 0);

  freeArena();
  return !errorCount();
}

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const opt::OptTable::Info OptInfo[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X7, X8, X9, X10, X11, X12)      \
  {X1, X2, X10,         X11,         OPT_##ID, opt::Option::KIND##Class,       \
   X9, X8, OPT_##GROUP, OPT_##ALIAS, X7,       X12},
#include "Options.inc"
#undef OPTION
};

class WasmOptTable : public llvm::opt::OptTable {
public:
  WasmOptTable() : OptTable(OptInfo) {}
  opt::InputArgList parse(ArrayRef<const char *> Argv);
};

// Set color diagnostics according to -color-diagnostics={auto,always,never}
// or -no-color-diagnostics flags.
static void handleColorDiagnostics(opt::InputArgList &Args) {
  auto *Arg = Args.getLastArg(OPT_color_diagnostics, OPT_color_diagnostics_eq,
                              OPT_no_color_diagnostics);
  if (!Arg)
    return;

  if (Arg->getOption().getID() == OPT_color_diagnostics)
    errorHandler().ColorDiagnostics = true;
  else if (Arg->getOption().getID() == OPT_no_color_diagnostics)
    errorHandler().ColorDiagnostics = false;
  else {
    StringRef S = Arg->getValue();
    if (S == "always")
      errorHandler().ColorDiagnostics = true;
    if (S == "never")
      errorHandler().ColorDiagnostics = false;
    if (S != "auto")
      error("unknown option: -color-diagnostics=" + S);
  }
}

// Find a file by concatenating given paths.
static Optional<std::string> findFile(StringRef Path1, const Twine &Path2) {
  SmallString<128> S;
  path::append(S, Path1, Path2);
  if (fs::exists(S))
    return S.str().str();
  return None;
}

opt::InputArgList WasmOptTable::parse(ArrayRef<const char *> Argv) {
  SmallVector<const char *, 256> Vec(Argv.data(), Argv.data() + Argv.size());

  unsigned MissingIndex;
  unsigned MissingCount;
  opt::InputArgList Args = this->ParseArgs(Vec, MissingIndex, MissingCount);

  handleColorDiagnostics(Args);
  for (auto *Arg : Args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + Arg->getSpelling());
  return Args;
}

// Currently we allow a ".imports" to live alongside a library. This can
// be used to specify a list of symbols which can be undefined at link
// time (imported from the environment.  For example libc.a include an
// import file that lists the syscall functions it relies on at runtime.
// In the long run this information would be better stored as a symbol
// attribute/flag in the object file itself.
// See: https://github.com/WebAssembly/tool-conventions/issues/35
static void readImportFile(StringRef Filename) {
  if (Optional<MemoryBufferRef> Buf = readFile(Filename))
    for (StringRef Sym : args::getLines(*Buf))
      Config->AllowUndefinedSymbols.insert(Sym);
}

void LinkerDriver::addFile(StringRef Path) {
  Optional<MemoryBufferRef> Buffer = readFile(Path);
  if (!Buffer.hasValue())
    return;
  MemoryBufferRef MBRef = *Buffer;

  if (identify_magic(MBRef.getBuffer()) == file_magic::archive) {
    SmallString<128> ImportFile = Path;
    path::replace_extension(ImportFile, ".imports");
    if (fs::exists(ImportFile))
      readImportFile(ImportFile.str());

    Files.push_back(make<ArchiveFile>(MBRef));
    return;
  }

  Files.push_back(make<ObjFile>(MBRef));
}

// Add a given library by searching it from input search paths.
void LinkerDriver::addLibrary(StringRef Name) {
  for (StringRef Dir : Config->SearchPaths) {
    if (Optional<std::string> S = findFile(Dir, "lib" + Name + ".a")) {
      addFile(*S);
      return;
    }
  }

  error("unable to find library -l" + Name);
}

void LinkerDriver::createFiles(opt::InputArgList &Args) {
  for (auto *Arg : Args) {
    switch (Arg->getOption().getUnaliasedOption().getID()) {
    case OPT_l:
      addLibrary(Arg->getValue());
      break;
    case OPT_INPUT:
      addFile(Arg->getValue());
      break;
    }
  }
}

static StringRef getEntry(opt::InputArgList &Args, StringRef Default) {
  auto *Arg = Args.getLastArg(OPT_entry, OPT_no_entry);
  if (!Arg)
    return Default;
  if (Arg->getOption().getID() == OPT_no_entry)
    return "";
  return Arg->getValue();
}

void LinkerDriver::link(ArrayRef<const char *> ArgsArr) {
  WasmOptTable Parser;
  opt::InputArgList Args = Parser.parse(ArgsArr.slice(1));

  // Handle --help
  if (Args.hasArg(OPT_help)) {
    Parser.PrintHelp(outs(), ArgsArr[0], "LLVM Linker", false);
    return;
  }

  // Handle --version
  if (Args.hasArg(OPT_version) || Args.hasArg(OPT_v)) {
    outs() << getLLDVersion() << "\n";
    return;
  }

  // Parse and evaluate -mllvm options.
  std::vector<const char *> V;
  V.push_back("wasm-ld (LLVM option parsing)");
  for (auto *Arg : Args.filtered(OPT_mllvm))
    V.push_back(Arg->getValue());
  cl::ParseCommandLineOptions(V.size(), V.data());

  errorHandler().ErrorLimit = args::getInteger(Args, OPT_error_limit, 20);

  Config->AllowUndefined = Args.hasArg(OPT_allow_undefined);
  Config->CheckSignatures =
      Args.hasFlag(OPT_check_signatures, OPT_no_check_signatures, false);
  Config->Entry = getEntry(Args, Args.hasArg(OPT_relocatable) ? "" : "_start");
  Config->ImportMemory = Args.hasArg(OPT_import_memory);
  Config->OutputFile = Args.getLastArgValue(OPT_o);
  Config->Relocatable = Args.hasArg(OPT_relocatable);
  Config->GcSections =
      Args.hasFlag(OPT_gc_sections, OPT_no_gc_sections, !Config->Relocatable);
  Config->PrintGcSections =
      Args.hasFlag(OPT_print_gc_sections, OPT_no_print_gc_sections, false);
  Config->SearchPaths = args::getStrings(Args, OPT_L);
  Config->StripAll = Args.hasArg(OPT_strip_all);
  Config->StripDebug = Args.hasArg(OPT_strip_debug);
  errorHandler().Verbose = Args.hasArg(OPT_verbose);
  ThreadsEnabled = Args.hasFlag(OPT_threads, OPT_no_threads, true);

  Config->InitialMemory = args::getInteger(Args, OPT_initial_memory, 0);
  Config->GlobalBase = args::getInteger(Args, OPT_global_base, 1024);
  Config->MaxMemory = args::getInteger(Args, OPT_max_memory, 0);
  Config->ZStackSize =
      args::getZOptionValue(Args, OPT_z, "stack-size", WasmPageSize);

  if (auto *Arg = Args.getLastArg(OPT_allow_undefined_file))
    readImportFile(Arg->getValue());

  if (!Args.hasArg(OPT_INPUT)) {
    error("no input files");
    return;
  }

  if (Config->OutputFile.empty())
    error("no output file specified");

  if (Config->Relocatable) {
    if (!Config->Entry.empty())
      error("entry point specified for relocatable output file");
    if (Config->GcSections)
      error("-r and --gc-sections may not be used together");
    if (Args.hasArg(OPT_undefined))
      error("-r -and --undefined may not be used together");
  }

  Symbol *EntrySym = nullptr;
  if (!Config->Relocatable) {
    // Can't export the SP right now because it's mutable, and mutable
    // globals aren't yet supported in the official binary format.
    // TODO(sbc): Remove WASM_SYMBOL_VISIBILITY_HIDDEN if/when the
    // "mutable global" proposal is accepted.
    StackPointerGlobal.Type = {WASM_TYPE_I32, true};
    StackPointerGlobal.InitExpr.Value.Int32 = 0;
    StackPointerGlobal.InitExpr.Opcode = WASM_OPCODE_I32_CONST;
    InputGlobal *StackPointer = make<InputGlobal>(StackPointerGlobal);
    StackPointer->Live = true;

    static WasmSignature NullSignature = {{}, WASM_TYPE_NORESULT};
    // Add synthetic symbols before any others
    WasmSym::CallCtors = Symtab->addSyntheticFunction(
        "__wasm_call_ctors", &NullSignature, WASM_SYMBOL_VISIBILITY_HIDDEN);
    WasmSym::StackPointer = Symtab->addSyntheticGlobal(
        "__stack_pointer", WASM_SYMBOL_VISIBILITY_HIDDEN, StackPointer);
    WasmSym::HeapBase = Symtab->addSyntheticDataSymbol("__heap_base", 0);
    WasmSym::DsoHandle = Symtab->addSyntheticDataSymbol(
        "__dso_handle", WASM_SYMBOL_VISIBILITY_HIDDEN);
    WasmSym::DataEnd = Symtab->addSyntheticDataSymbol("__data_end", 0);

    if (!Config->Entry.empty())
      EntrySym = Symtab->addUndefinedFunction(Config->Entry, 0, nullptr,
                                              &NullSignature);

    // Handle the `--undefined <sym>` options.
    for (auto *Arg : Args.filtered(OPT_undefined))
      Symtab->addUndefinedFunction(Arg->getValue(), 0, nullptr, nullptr);
  }

  createFiles(Args);
  if (errorCount())
    return;

  // Add all files to the symbol table. This will add almost all
  // symbols that we need to the symbol table.
  for (InputFile *F : Files)
    Symtab->addFile(F);

  // Make sure we have resolved all symbols.
  if (!Config->Relocatable && !Config->AllowUndefined) {
    Symtab->reportRemainingUndefines();
  } else {
    // When we allow undefined symbols we cannot include those defined in
    // -u/--undefined since these undefined symbols have only names and no
    // function signature, which means they cannot be written to the final
    // output.
    for (auto *Arg : Args.filtered(OPT_undefined)) {
      Symbol *Sym = Symtab->find(Arg->getValue());
      if (!Sym->isDefined())
        error("function forced with --undefined not found: " + Sym->getName());
    }
  }
  if (errorCount())
    return;

  // Handle --export.
  for (auto *Arg : Args.filtered(OPT_export)) {
    StringRef Name = Arg->getValue();
    Symbol *Sym = Symtab->find(Name);
    if (Sym && Sym->isDefined())
      Sym->setHidden(false);
    else if (!Config->AllowUndefined)
      error("symbol exported via --export not found: " + Name);
  }

  if (EntrySym)
    EntrySym->setHidden(false);

  if (errorCount())
    return;

  // Do size optimizations: garbage collection
  markLive();

  // Write the result to the file.
  writeResult();
}
