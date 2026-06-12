//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool executes a sequence of steps required to link device code in SYCL
// device images. SYCL device code linking requires a complex sequence of steps
// that include linking of llvm bitcode files, linking bitcode library files
// with the fully linked source bitcode file(s), running several SYCL specific
// post-link steps on the fully linked bitcode file(s), and finally generating
// target-specific device code.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OffloadArch.h"
#include "clang/Basic/Version.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/IRSymtab.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/SplitModuleByCategory.h"

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::object;
using namespace clang;

/// Print commands with arguments without executing.
static bool DryRun = false;

/// Print verbose output.
static bool Verbose = false;

/// Filename of the output being created.
static StringRef OutputFile;

/// Directory to dump SPIR-V IR if requested by user.
static SmallString<128> SPIRVDumpDir;

using OffloadingImage = OffloadBinary::OffloadingImage;

static void printVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-sycl-linker") << '\n';
}

/// The value of `argv[0]` when run.
static const char *Executable;

/// Temporary files to be cleaned up.
static SmallVector<SmallString<128>> TempFiles;

namespace {
// Must not overlap with llvm::opt::DriverFlag.
enum LinkerFlags { LinkerOnlyOption = (1 << 4) };

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "SYCLLinkOpts.inc"
  LastOption
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "SYCLLinkOpts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "SYCLLinkOpts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "SYCLLinkOpts.inc"
#undef OPTION
};

class LinkerOptTable : public opt::GenericOptTable {
public:
  LinkerOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};
} // namespace

static const OptTable &getOptTable() {
  static const LinkerOptTable *Table = []() {
    auto Result = std::make_unique<LinkerOptTable>();
    return Result.release();
  }();
  return *Table;
}

[[noreturn]] static void reportError(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), Executable));
  exit(EXIT_FAILURE);
}

static std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

static Expected<StringRef>
createTempFile(const ArgList &Args, const Twine &Prefix, StringRef Extension) {
  SmallString<128> Path;
  if (Args.hasArg(OPT_save_temps) || DryRun) {
    // Generate a unique path name without creating a file
    sys::fs::createUniquePath(Prefix + "-%%%%%%." + Extension, Path,
                              /*MakeAbsolute=*/false);
  } else {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(Prefix, Extension, Path))
      return createFileError(Path, EC);
  }

  TempFiles.emplace_back(std::move(Path));
  return TempFiles.back();
}

static Expected<std::string> findProgram(const ArgList &Args, StringRef Name,
                                         ArrayRef<StringRef> Paths) {
  if (DryRun)
    return Name.str();
  ErrorOr<std::string> Path = sys::findProgramByName(Name, Paths);
  if (!Path)
    Path = sys::findProgramByName(Name);
  if (!Path)
    return createStringError(Path.getError(),
                             "unable to find '" + Name + "' in path");
  return *Path;
}

static void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  llvm::errs() << llvm::join(std::next(CmdArgs.begin()), CmdArgs.end(), " ")
               << "\n";
}

/// Execute the command \p ExecutablePath with the arguments \p Args.
static Error executeCommands(StringRef ExecutablePath,
                             ArrayRef<StringRef> Args) {
  if (Verbose || DryRun)
    printCommands(Args);

  if (DryRun)
    return Error::success();

  if (sys::ExecuteAndWait(ExecutablePath, Args))
    return createStringError("'%s' failed",
                             sys::path::filename(ExecutablePath).str().c_str());
  return Error::success();
}

namespace {
/// A minimal symbol interface used to drive archive member extraction. Only the
/// flags required by the symbol-resolution fixed-point loop are tracked.
struct Symbol {
  enum Flags {
    None = 0,
    Undefined = 1 << 0,
    Weak = 1 << 1,
  };

  Symbol() : SymFlags(None) {}
  Symbol(Symbol::Flags F) : SymFlags(F) {}
  Symbol(const irsymtab::Reader::SymbolRef Sym) : SymFlags(0) {
    if (Sym.isUndefined())
      SymFlags |= Undefined;
    if (Sym.isWeak())
      SymFlags |= Weak;
  }

  bool isWeak() const { return SymFlags & Weak; }
  bool isUndefined() const { return SymFlags & Undefined; }

  uint32_t SymFlags;
};

/// Description of a single input (positional file or -l library).
struct InputDesc {
  enum class Kind { File, Library };

  StringRef Value; // File path, or library name for -l (the value after -l).
  Kind InputKind = Kind::File;
  bool WholeArchive = false; // --whole-archive state in effect at this input.
};

/// An input buffer pending archive-member resolution, together with its parsed
/// IR symbol table. The symbol table is parsed once and reused across all
/// fixed-point passes so members are not re-parsed on every pass.
struct PendingInput {
  std::unique_ptr<MemoryBuffer> Buffer;
  bool IsLazy = false;
  bool FromArchive = false;
  IRSymtabFile Symtab;
};

/// Resolved input buffers and their target triple.
struct ResolvedInputs {
  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  llvm::Triple TargetTriple;
  StringRef TripleSource; // Source of the triple (--triple= or filename)
};
} // namespace

static std::optional<std::string> findFile(StringRef Dir, const Twine &Name) {
  SmallString<128> Path;
  sys::path::append(Path, Dir, Name);
  // Skip directories so a directory whose name matches the requested library
  // does not stop the search; a later -L path may hold the real archive.
  if (sys::fs::exists(Path) && !sys::fs::is_directory(Path))
    return static_cast<std::string>(Path);
  return std::nullopt;
}

static std::optional<std::string>
findFromSearchPaths(StringRef Name, ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (std::optional<std::string> File = findFile(Dir, Name))
      return File;
  return std::nullopt;
}

/// Search for static libraries in the linker's library path given input like
/// `-lfoo`, `-l:libfoo.a`, or `-l/absolute/path/to/lib.a`.
static std::optional<std::string>
searchLibrary(StringRef Input, ArrayRef<StringRef> SearchPaths) {
  // An absolute path is taken as-is; -L paths are only consulted for relative
  // names.
  if (sys::path::is_absolute(Input)) {
    if (sys::fs::exists(Input) && !sys::fs::is_directory(Input))
      return Input.str();
    return std::nullopt;
  }

  if (Input.starts_with(":"))
    return findFromSearchPaths(Input.drop_front(), SearchPaths);
  SmallString<128> LibName("lib");
  LibName += Input;
  LibName += ".a";
  return findFromSearchPaths(LibName, SearchPaths);
}

/// Scan a member's pre-parsed IR symbol table against \p LinkerSymtab and
/// return true if the member should be extracted: it is non-lazy, or it defines
/// a symbol that resolves a currently-undefined reference. Mirrors a linker's
/// archive member selection.
static bool scanSymbols(const IRSymtabFile &MemberSymtab,
                        StringMap<Symbol> &LinkerSymtab, bool IsLazy) {
  bool Extracted = !IsLazy;
  StringMap<Symbol> PendingSymbols;
  for (unsigned ModIdx = 0; ModIdx != MemberSymtab.Mods.size(); ++ModIdx) {
    for (const auto &IRSym : MemberSymtab.TheReader.module_symbols(ModIdx)) {
      if (IRSym.isFormatSpecific() || !IRSym.isGlobal())
        continue;

      bool IsNewSymbol = IsLazy && !LinkerSymtab.count(IRSym.getName());
      StringMap<Symbol> &Target = IsNewSymbol ? PendingSymbols : LinkerSymtab;
      Symbol Sym(IRSym);
      auto [It, Inserted] = Target.try_emplace(IRSym.getName(), Sym);
      // A freshly inserted entry has no prior symbol to resolve or upgrade, so
      // it cannot trigger extraction.
      if (Inserted)
        continue;

      Symbol &OldSym = It->second;
      bool ResolvesReference =
          !Sym.isUndefined() &&
          (OldSym.isUndefined() || (OldSym.isWeak() && !Sym.isWeak())) &&
          !(OldSym.isWeak() && OldSym.isUndefined() && IsLazy);
      Extracted |= ResolvesReference;

      if (ResolvesReference)
        OldSym = Sym;
    }
  }
  if (Extracted && IsLazy)
    for (const auto &[Name, Sym] : PendingSymbols)
      LinkerSymtab[Name] = Sym;
  return Extracted;
}

/// Parse \p Buffer's IR symbol table and append it to \p Inputs. Errors if the
/// buffer is not LLVM bitcode (the only member type the SYCL linker supports).
static Error addBitcodeInput(SmallVector<PendingInput> &Inputs,
                             std::unique_ptr<MemoryBuffer> Buffer, bool IsLazy,
                             bool FromArchive) {
  if (identify_magic(Buffer->getBuffer()) != file_magic::bitcode)
    return createStringError("unsupported file type: '" +
                             Buffer->getBufferIdentifier() + "'");
  Expected<IRSymtabFile> SymtabOrErr = readIRSymtab(Buffer->getMemBufferRef());
  if (!SymtabOrErr)
    return SymtabOrErr.takeError();
  Inputs.push_back(
      {std::move(Buffer), IsLazy, FromArchive, std::move(*SymtabOrErr)});
  return Error::success();
}

/// Resolve archive members from the given inputs using a symbol-driven
/// fixed-point algorithm. For each input:
/// - If it's a Library, search for lib<name>.a or :<name> in SearchPaths
/// - If it's a File, use the path directly
/// - Archives are expanded and members are lazily extracted based on symbol
///   references unless WholeArchive is true
/// - Non-archive bitcode inputs are always included
///
/// Returns the buffers to link, in extraction order, along with the resolved
/// target triple. All returned buffers have compatible target triples;
/// incompatible archive members are filtered during resolution.
static Expected<ResolvedInputs> resolveArchiveMembers(
    ArrayRef<InputDesc> Order, ArrayRef<StringRef> SearchPaths,
    ArrayRef<StringRef> ForcedUndefs, StringRef TargetTripleArgValue) {
  // Collect every candidate member, parsing each one's IR symbol table once.
  SmallVector<PendingInput> Inputs;

  for (const InputDesc &Desc : Order) {
    std::optional<std::string> Filename;

    if (Desc.InputKind == InputDesc::Kind::Library) {
      Filename = searchLibrary(Desc.Value, SearchPaths);
      if (!Filename)
        return createStringError("unable to find library -l" + Desc.Value);
    } else {
      if (!sys::fs::exists(Desc.Value))
        return createStringError("input file not found: '" + Desc.Value + "'");
      if (sys::fs::is_directory(Desc.Value))
        return createStringError("'" + Desc.Value + "': is a directory");
      Filename = Desc.Value.str();
    }

    auto BufferOrErr =
        errorOrToExpected(MemoryBuffer::getFileOrSTDIN(*Filename));
    if (!BufferOrErr)
      return createFileError(*Filename, BufferOrErr.takeError());

    MemoryBufferRef Buffer = (*BufferOrErr)->getMemBufferRef();
    switch (identify_magic(Buffer.getBuffer())) {
    case file_magic::bitcode:
      if (Error Err = addBitcodeInput(Inputs, std::move(*BufferOrErr),
                                      /*IsLazy=*/false, /*FromArchive=*/false))
        return Err;
      break;
    case file_magic::archive: {
      Expected<std::unique_ptr<object::Archive>> LibFile =
          object::Archive::create(Buffer);
      if (!LibFile)
        return LibFile.takeError();
      Error Err = Error::success();
      for (auto Child : (*LibFile)->children(Err)) {
        auto ChildBufferOrErr = Child.getMemoryBufferRef();
        if (!ChildBufferOrErr)
          return ChildBufferOrErr.takeError();
        // Include archive name in buffer identifier for better diagnostics.
        std::string BufferIdentifier =
            (*Filename + "(" + ChildBufferOrErr->getBufferIdentifier() + ")")
                .str();
        std::unique_ptr<MemoryBuffer> ChildBuffer =
            MemoryBuffer::getMemBufferCopy(ChildBufferOrErr->getBuffer(),
                                           BufferIdentifier);
        if (Error E = addBitcodeInput(Inputs, std::move(ChildBuffer),
                                      !Desc.WholeArchive, /*FromArchive=*/true))
          return E;
      }
      if (Err)
        return Err;
      break;
    }
    default:
      return createStringError("unsupported file type: '" + *Filename + "'");
    }
  }

  // Resolve the target triple: use --triple= if provided, otherwise infer from
  // the first non-archive input with a non-empty triple.
  llvm::Triple TargetTriple(TargetTripleArgValue);
  StringRef TripleSource = TargetTriple.empty() ? "" : "--triple=";

  if (TargetTriple.empty()) {
    for (const PendingInput &In : Inputs) {
      if (!In.FromArchive && In.Symtab.Mods.size() > 0) {
        StringRef Triple = In.Symtab.TheReader.getTargetTriple();
        if (!Triple.empty()) {
          TargetTriple = llvm::Triple(Triple);
          TripleSource = In.Buffer->getBufferIdentifier();
          break;
        }
      }
    }
  }

  // Seed symbol table with forced undefined symbols.
  StringMap<Symbol> SymTab;
  for (StringRef Sym : ForcedUndefs)
    SymTab[Sym] = Symbol(Symbol::Undefined);

  // Fixed-point loop to extract archive members. Each pass may resolve symbols
  // that unlock further members; iterate until no new member is extracted.
  SmallVector<std::unique_ptr<MemoryBuffer>> Resolved;
  bool KeepExtracting = true;
  while (KeepExtracting) {
    KeepExtracting = false;
    for (PendingInput &In : Inputs) {
      if (!In.Buffer)
        continue;

      // Filter archive members by target triple before symbol scanning.
      // Members built for a different target are silently skipped, matching how
      // a real linker treats device libraries built for other architectures.
      if (In.FromArchive) {
        StringRef MemberTriple = In.Symtab.TheReader.getTargetTriple();
        if (!MemberTriple.empty() &&
            llvm::Triple(MemberTriple) != TargetTriple) {
          if (Verbose)
            errs() << formatv(
                "archive resolution: skipping {0}: triple {1} != {2}\n",
                In.Buffer->getBufferIdentifier(), MemberTriple,
                TargetTriple.str());
          In.Buffer.reset();
          In.Symtab = {};
          continue;
        }
      }

      if (!scanSymbols(In.Symtab, SymTab, In.IsLazy))
        continue;
      KeepExtracting = true;
      Resolved.push_back(std::move(In.Buffer));
    }
  }

  return ResolvedInputs{std::move(Resolved), std::move(TargetTriple),
                        TripleSource};
}

static Expected<ResolvedInputs> getInput(const ArgList &Args) {
  // Build input descriptors for the archive resolver.
  SmallVector<InputDesc> InputDescs;
  bool WholeArchive = false;
  for (const opt::Arg *Arg : Args.filtered(
           OPT_INPUT, OPT_library, OPT_whole_archive, OPT_no_whole_archive)) {
    if (Arg->getOption().matches(OPT_whole_archive) ||
        Arg->getOption().matches(OPT_no_whole_archive)) {
      WholeArchive = Arg->getOption().matches(OPT_whole_archive);
      continue;
    }

    InputDesc Desc;
    Desc.Value = Arg->getValue();
    Desc.InputKind = Arg->getOption().matches(OPT_library)
                         ? InputDesc::Kind::Library
                         : InputDesc::Kind::File;
    Desc.WholeArchive = WholeArchive;
    InputDescs.push_back(Desc);
  }

  if (InputDescs.empty())
    return createStringError("no input files provided");

  // Gather search paths and forced undefined symbols.
  SmallVector<StringRef> LibraryPaths;
  for (const opt::Arg *Arg : Args.filtered(OPT_library_path))
    LibraryPaths.push_back(Arg->getValue());

  // getAllArgValues returns a temporary vector; retain it so the StringRefs
  // remain valid through the resolveArchiveMembers call.
  std::vector<std::string> ForcedUndefStorage = Args.getAllArgValues(OPT_u);
  SmallVector<StringRef> ForcedUndefs(ForcedUndefStorage.begin(),
                                      ForcedUndefStorage.end());

  // Get target triple from command line if specified.
  StringRef TargetTripleStr = Args.getLastArgValue(OPT_triple_EQ);

  Expected<ResolvedInputs> ResolvedOrErr = resolveArchiveMembers(
      InputDescs, LibraryPaths, ForcedUndefs, TargetTripleStr);
  if (!ResolvedOrErr)
    return ResolvedOrErr.takeError();

  if (ResolvedOrErr->Buffers.empty())
    return createStringError("no input files could be resolved");

  if (ResolvedOrErr->TargetTriple.empty())
    return createStringError(
        "target triple must be specified or inferable from inputs");

  return std::move(*ResolvedOrErr);
}

namespace {
struct LinkResult {
  std::unique_ptr<Module> LinkedModule;
  SmallString<256> BitcodeFile;
  llvm::Triple TargetTriple;
};
} // namespace

/// Link all resolved input bitcode images into one module. All resolved inputs
/// are guaranteed to have compatible target triples (incompatible archive
/// members are filtered during archive resolution). Triple conflicts between
/// regular (non-archive) inputs are hard errors caught before running
/// linkInModule.
static Expected<LinkResult>
linkInputs(ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs,
           const llvm::Triple &TargetTriple, StringRef TripleSource,
           const ArgList &Args, LLVMContext &C) {
  llvm::TimeTraceScope TimeScope("Link code");

  assert(Inputs.size() && "No inputs to link");

  // Create a new file to write the linked file to.
  auto BitcodeOutput =
      createTempFile(Args, sys::path::filename(OutputFile), "bc");
  if (!BitcodeOutput)
    return BitcodeOutput.takeError();

  if (Verbose) {
    std::string InputList =
        llvm::join(llvm::map_range(Inputs,
                                   [](const auto &Buffer) {
                                     return Buffer->getBufferIdentifier();
                                   }),
                   ", ");
    errs() << formatv("link: inputs: {0} output: {1}\n", InputList,
                      *BitcodeOutput);
  }

  auto LinkerOutput = std::make_unique<Module>("linker-output", C);
  Linker L(*LinkerOutput);

  for (const auto &Buffer : Inputs) {
    auto ModOrErr = parseBitcodeFile(Buffer->getMemBufferRef(), C);
    if (!ModOrErr)
      return ModOrErr.takeError();

    const llvm::Triple &T = (*ModOrErr)->getTargetTriple();
    if (!T.empty() && T != TargetTriple) {
      // All incompatible archive members should have been filtered during
      // resolution, so this is a conflict between regular inputs.
      return createStringError("conflicting target triples: '" +
                               TargetTriple.str() + "' (from " + TripleSource +
                               ") vs '" + T.str() + "' (from " +
                               Buffer->getBufferIdentifier() + ")");
    }

    if (L.linkInModule(std::move(*ModOrErr)))
      return createStringError("could not link IR");
  }

  // Dump linked output for testing.
  if (Args.hasArg(OPT_print_linked_module))
    outs() << *LinkerOutput;

  // Write the final output into 'BitcodeOutput' file.
  if (!DryRun) {
    int FD = -1;
    if (std::error_code EC = sys::fs::openFileForWrite(*BitcodeOutput, FD))
      return errorCodeToError(EC);
    llvm::raw_fd_ostream OS(FD, true);
    WriteBitcodeToFile(*LinkerOutput, OS);
  }

  return LinkResult{std::move(LinkerOutput), SmallString<256>(*BitcodeOutput),
                    std::move(TargetTriple)};
}

/// Run Code Generation using LLVM backend.
/// \param File The input LLVM IR bitcode file.
/// \param TargetTriple The resolved target triple.
/// \param Args encompasses all arguments required for linking device code and
/// will be parsed to generate options required to be passed into the backend.
/// \param OutputFile The output file name.
/// \param C The LLVM context.
static Error runCodeGen(StringRef File, const llvm::Triple &TargetTriple,
                        const ArgList &Args, StringRef OutputFile,
                        LLVMContext &C) {
  llvm::TimeTraceScope TimeScope("Code generation");

  if (Verbose || DryRun)
    errs() << formatv("LLVM backend: input: {0}, output: {1}\n", File,
                      OutputFile);

  if (DryRun)
    return Error::success();

  // Parse input module.
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(File, Err, C);
  if (!M)
    return createStringError(Err.getMessage());

  if (Error MatErr = M->materializeAll())
    return MatErr;

  M->setTargetTriple(TargetTriple);

  // Get a handle to a target backend.
  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M->getTargetTriple(), Msg);
  if (!T)
    return createStringError(Msg + ": " + M->getTargetTriple().str());

  // Allocate target machine.
  TargetOptions Options;
  std::optional<Reloc::Model> RM;
  std::optional<CodeModel::Model> CM;
  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M->getTargetTriple(), /*CPU=*/"",
                             /*Features=*/"", Options, RM, CM));
  if (!TM)
    return createStringError("could not allocate target machine");

  // Set data layout if needed.
  if (M->getDataLayout().isDefault())
    M->setDataLayout(TM->createDataLayout());

  // Open output file for writing.
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(OutputFile, FD))
    return errorCodeToError(EC);
  auto OS = std::make_unique<llvm::raw_fd_ostream>(FD, true);

  legacy::PassManager CodeGenPasses;
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS, nullptr,
                              CodeGenFileType::ObjectFile))
    return createStringError("failed to execute LLVM backend");
  CodeGenPasses.run(*M);

  return Error::success();
}

/// Run AOT compilation for Intel CPU.
/// Calls opencl-aot tool to generate device code for the Intel OpenCL CPU
/// Runtime.
/// \param InputFile The input SPIR-V file.
/// \param OutputFile The output file name.
/// \param Args Encompasses all arguments required for linking and wrapping
/// device code and will be parsed to generate options required to be passed
/// into the AOT compilation step.
static Error runAOTCompileIntelCPU(StringRef InputFile, StringRef OutputFile,
                                   const ArgList &Args) {
  SmallVector<StringRef, 8> CmdArgs;
  Expected<std::string> OpenCLAOTPath =
      findProgram(Args, "opencl-aot", {getMainExecutable("opencl-aot")});
  if (!OpenCLAOTPath)
    return OpenCLAOTPath.takeError();

  CmdArgs.push_back(*OpenCLAOTPath);
  CmdArgs.push_back("--device=cpu");
  StringRef ExtraArgs = Args.getLastArgValue(OPT_opencl_aot_options_EQ);
  ExtraArgs.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(OutputFile);
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*OpenCLAOTPath, CmdArgs))
    return Err;
  return Error::success();
}

/// Run AOT compilation for Intel GPU.
/// Calls ocloc tool to generate device code for the Intel Graphics Compute
/// Runtime.
/// \param InputFile The input SPIR-V file.
/// \param OutputFile The output file name.
/// \param Args Encompasses all arguments required for linking and wrapping
/// device code and will be parsed to generate options required to be passed
/// into the AOT compilation step.
static Error runAOTCompileIntelGPU(StringRef InputFile, StringRef OutputFile,
                                   const ArgList &Args) {
  SmallVector<StringRef, 8> CmdArgs;
  Expected<std::string> OclocPath =
      findProgram(Args, "ocloc", {getMainExecutable("ocloc")});
  if (!OclocPath)
    return OclocPath.takeError();

  CmdArgs.push_back(*OclocPath);
  // The next line prevents ocloc from modifying the image name
  CmdArgs.push_back("-output_no_suffix");
  CmdArgs.push_back("-spirv_input");

  StringRef Arch(Args.getLastArgValue(OPT_arch_EQ));
  assert(!Arch.empty() && "Arch must be specified for AOT compilation");
  CmdArgs.push_back("-device");
  CmdArgs.push_back(Arch);

  StringRef ExtraArgs = Args.getLastArgValue(OPT_ocloc_options_EQ);
  ExtraArgs.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  CmdArgs.push_back("-output");
  CmdArgs.push_back(OutputFile);
  CmdArgs.push_back("-file");
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*OclocPath, CmdArgs))
    return Err;
  return Error::success();
}

/// Run AOT compilation for Intel CPU/GPU.
/// \param InputFile The input SPIR-V file.
/// \param OutputFile The output file name.
/// \param Args Encompasses all arguments required for linking and wrapping
/// device code and will be parsed to generate options required to be passed
/// into the AOT compilation step.
static Error runAOTCompile(StringRef InputFile, StringRef OutputFile,
                           const ArgList &Args) {
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);
  OffloadArch OA = StringToOffloadArch(Arch);
  if (IsIntelGPUOffloadArch(OA))
    return runAOTCompileIntelGPU(InputFile, OutputFile, Args);
  if (IsIntelCPUOffloadArch(OA))
    return runAOTCompileIntelCPU(InputFile, OutputFile, Args);

  llvm_unreachable("runAOTCompile dispatched on unsupported arch");
}

static constexpr char AttrSYCLModuleId[] = "sycl-module-id";

namespace {
/// SYCL device code module split mode.
enum class IRSplitMode {
  SPLIT_PER_TU,     // one module per translation unit
  SPLIT_PER_KERNEL, // one module per kernel
  SPLIT_NONE        // no splitting
};
} // namespace

/// Parses the value of \p --module-split-mode.
static std::optional<IRSplitMode> convertStringToSplitMode(StringRef S) {
  return StringSwitch<std::optional<IRSplitMode>>(S)
      .Case("source", IRSplitMode::SPLIT_PER_TU)
      .Case("kernel", IRSplitMode::SPLIT_PER_KERNEL)
      .Case("none", IRSplitMode::SPLIT_NONE)
      .Default(std::nullopt);
}

static StringRef splitModeToString(IRSplitMode Mode) {
  switch (Mode) {
  case IRSplitMode::SPLIT_PER_TU:
    return "source";
  case IRSplitMode::SPLIT_PER_KERNEL:
    return "kernel";
  case IRSplitMode::SPLIT_NONE:
    return "none";
  }
  llvm_unreachable("bad split mode");
}

namespace {
/// Result of splitting a device module: the bitcode file path and the
/// serialized symbol table for each device image.
struct SplitModule {
  SmallString<256> ModuleFilePath;
  SmallString<0> Symbols;
};
} // namespace

static bool isEntryPoint(const Function &F, bool EmitOnlyKernelsAsEntryPoints) {
  if (F.isDeclaration())
    return false;
  if (F.hasKernelCallingConv())
    return true;
  if (EmitOnlyKernelsAsEntryPoints)
    return false;
  // sycl_external functions carry the "sycl-module-id" attribute.
  return F.hasFnAttribute(AttrSYCLModuleId);
}

/// Collect entry point names from \p M and serialize them into a symbol table.
static SmallString<0> collectEntryPoints(const Module &M,
                                         bool EmitOnlyKernelsAsEntryPoints) {
  SmallVector<StringRef> Names;
  for (const Function &F : M)
    if (isEntryPoint(F, EmitOnlyKernelsAsEntryPoints))
      Names.push_back(F.getName());
  SmallString<0> SymbolData;
  llvm::offloading::sycl::writeSymbolTable(Names, SymbolData);
  return SymbolData;
}

namespace {
/// Functor passed to splitModuleTransitiveFromEntryPoints. For each input
/// function \p F, returns a numeric group ID (if \p F is an entry point)
/// determining which device image it lands in, or std::nullopt (for
/// non-entry-points). SPLIT_PER_KERNEL \p Mode gives each kernel its own ID;
/// SPLIT_PER_TU \p Mode groups kernels by their "sycl-module-id" attribute
/// value.
class EntryPointCategorizer {
public:
  EntryPointCategorizer(IRSplitMode Mode, bool EmitOnlyKernelsAsEntryPoints)
      : Mode(Mode), OnlyKernelsAreEntryPoints(EmitOnlyKernelsAsEntryPoints) {}

  std::optional<int> operator()(const Function &F) {
    if (!isEntryPoint(F, OnlyKernelsAreEntryPoints))
      return std::nullopt;

    std::string Key;
    switch (Mode) {
    case IRSplitMode::SPLIT_PER_KERNEL:
      Key = F.getName().str();
      break;
    case IRSplitMode::SPLIT_PER_TU:
      Key = F.getFnAttribute(AttrSYCLModuleId).getValueAsString().str();
      break;
    case IRSplitMode::SPLIT_NONE:
      llvm_unreachable("categorizer cannot be used for SPLIT_NONE");
    }

    auto [It, Inserted] =
        StrToId.try_emplace(std::move(Key), static_cast<int>(StrToId.size()));
    return It->second;
  }

private:
  IRSplitMode Mode;
  bool OnlyKernelsAreEntryPoints;
  llvm::StringMap<int> StrToId;
};
} // namespace

/// Splits the fully linked device \p M into one bitcode file per device image
/// according to \p Mode and returns the list of split images with their symbol
/// tables. The module is split transitively from entry points; each part is
/// written to a fresh temporary bitcode file.
static Expected<SmallVector<SplitModule, 0>>
splitDeviceCode(std::unique_ptr<Module> M, StringRef LinkedBitcodeFile,
                IRSplitMode Mode, bool EmitOnlyKernelsAsEntryPoints,
                const ArgList &Args) {
  assert(Mode != IRSplitMode::SPLIT_NONE && "SPLIT_NONE is unsupported");

  SmallVector<SplitModule, 0> SplitModules;
  EntryPointCategorizer Categorizer(Mode, EmitOnlyKernelsAsEntryPoints);

  auto SplitCallback = [&](std::unique_ptr<Module> Part) -> Error {
    Expected<StringRef> BitcodeFileOrErr =
        createTempFile(Args, sys::path::filename(OutputFile), "bc");
    if (!BitcodeFileOrErr)
      return BitcodeFileOrErr.takeError();

    if (!DryRun) {
      int FD = -1;
      if (std::error_code EC = sys::fs::openFileForWrite(*BitcodeFileOrErr, FD))
        return errorCodeToError(EC);
      raw_fd_ostream OS(FD, /*shouldClose=*/true);
      WriteBitcodeToFile(*Part, OS);
    }

    SplitModules.push_back(
        {SmallString<256>(*BitcodeFileOrErr),
         collectEntryPoints(*Part, EmitOnlyKernelsAsEntryPoints)});
    return Error::success();
  };

  if (Error Err = splitModuleTransitiveFromEntryPoints(
          std::move(M), Categorizer, SplitCallback))
    return Err;

  if (Verbose) {
    errs() << formatv("sycl-module-split: input: {0}, mode: {1}\n",
                      LinkedBitcodeFile, splitModeToString(Mode));
    for (const SplitModule &SI : SplitModules) {
      errs() << formatv("{0} [", SI.ModuleFilePath);
      llvm::offloading::sycl::forEachSymbol(
          SI.Symbols, [](StringRef Name) { errs() << Name << " "; });
      errs() << "]\n";
    }
  }

  return SplitModules;
}

/// Returns true if module splitting can be skipped: either \p Mode is
/// SPLIT_NONE, or \p M contains no entry points (nothing to split from).
static bool canSkipModuleSplit(IRSplitMode Mode, const Module &M,
                               bool EmitOnlyKernelsAsEntryPoints) {
  if (Mode == IRSplitMode::SPLIT_NONE)
    return true;
  return llvm::none_of(M.functions(), [&](const Function &F) {
    return isEntryPoint(F, EmitOnlyKernelsAsEntryPoints);
  });
}

/// Performs the following steps:
/// 1. Link all input bitcode files together with library files.
/// 2. Optionally split the linked module according to the requested
///    IRSplitMode.
/// 3. Run SPIR-V code generation on each (split) module.
/// 4. Optionally run AOT compilation when targeting an Intel HW arch.
/// 5. Pack the resulting images into a single OffloadBinary written to the
///    output file.
static Error runSYCLLink(ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs,
                         const llvm::Triple &TargetTriple,
                         StringRef TripleSource, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL linking");

  LLVMContext C;

  // Link all input bitcode files and library files.
  Expected<LinkResult> LinkedOrErr =
      linkInputs(Inputs, TargetTriple, TripleSource, Args, C);
  if (!LinkedOrErr)
    return LinkedOrErr.takeError();
  LinkResult &Result = *LinkedOrErr;

  // Determine the requested module split mode.
  IRSplitMode SplitMode = IRSplitMode::SPLIT_PER_TU;
  if (Arg *A = Args.getLastArg(OPT_module_split_mode_EQ)) {
    std::optional<IRSplitMode> ModeOrNone =
        convertStringToSplitMode(A->getValue());
    if (!ModeOrNone)
      return createStringError(formatv(
          "module-split-mode value isn't recognized: {0}", A->getValue()));
    SplitMode = *ModeOrNone;
  }

  // TODO: Expose this as a command-line option and default it to false when
  // device-image dynamic linking is supported, so that sycl_external functions
  // can be called across device image boundaries.
  bool EmitOnlyKernelsAsEntryPoints = true;

  SmallVector<SplitModule, 0> SplitModules;
  if (canSkipModuleSplit(SplitMode, *Result.LinkedModule,
                         EmitOnlyKernelsAsEntryPoints)) {
    SplitModules.push_back({SmallString<256>(Result.BitcodeFile),
                            collectEntryPoints(*Result.LinkedModule,
                                               EmitOnlyKernelsAsEntryPoints)});
  } else {
    Expected<SmallVector<SplitModule, 0>> SplitModulesOrErr =
        splitDeviceCode(std::move(Result.LinkedModule), Result.BitcodeFile,
                        SplitMode, EmitOnlyKernelsAsEntryPoints, Args);
    if (!SplitModulesOrErr)
      return SplitModulesOrErr.takeError();

    SplitModules = std::move(*SplitModulesOrErr);
  }

  bool IsAOTCompileNeeded = IsIntelOffloadArch(
      StringToOffloadArch(Args.getLastArgValue(OPT_arch_EQ)));

  StringRef OutputFileNameExt = ".spv";

  // Code generation step.
  for (size_t I = 0, E = SplitModules.size(); I != E; ++I) {
    StringRef Stem = OutputFile.rsplit('.').first;
    std::string CodeGenFile = (Stem + "_" + Twine(I) + OutputFileNameExt).str();

    if (Error Err = runCodeGen(SplitModules[I].ModuleFilePath,
                               Result.TargetTriple, Args, CodeGenFile, C))
      return Err;

    if (!SPIRVDumpDir.empty() && !DryRun) {
      SmallString<128> DumpFile(SPIRVDumpDir);
      sys::path::append(DumpFile, sys::path::filename(CodeGenFile));
      if (std::error_code EC = sys::fs::copy_file(CodeGenFile, DumpFile))
        return createFileError(DumpFile, EC);
    }

    SplitModules[I].ModuleFilePath = CodeGenFile;
    if (IsAOTCompileNeeded) {
      std::string AOTFile = (Stem + "_" + Twine(I) + ".out").str();
      if (Error Err = runAOTCompile(CodeGenFile, AOTFile, Args))
        return Err;
      SplitModules[I].ModuleFilePath = AOTFile;
    }
  }

  // Collect all images to be packed into a single OffloadBinary.
  SmallVector<OffloadingImage> Images;
  for (SplitModule &SI : SplitModules) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
        DryRun ? llvm::MemoryBuffer::getMemBuffer("")
               : llvm::MemoryBuffer::getFileOrSTDIN(SI.ModuleFilePath);
    if (!FileOrErr)
      return createFileError(SI.ModuleFilePath, FileOrErr.getError());

    OffloadingImage TheImage{};
    TheImage.TheImageKind = IsAOTCompileNeeded ? IMG_Object : IMG_SPIRV;
    TheImage.TheOffloadKind = OFK_SYCL;
    TheImage.StringData["triple"] =
        Args.MakeArgString(Result.TargetTriple.str());
    TheImage.StringData["arch"] =
        Args.MakeArgString(Args.getLastArgValue(OPT_arch_EQ));
    TheImage.StringData["symbols"] = SI.Symbols;
    TheImage.Image = std::move(*FileOrErr);
    Images.emplace_back(std::move(TheImage));
  }

  if (Verbose) {
    for (const OffloadingImage &Image : Images)
      errs() << formatv(
          "sycl-bundle: image kind: {0}, triple: {1}, arch: {2}\n",
          getImageKindName(Image.TheImageKind),
          Image.StringData.lookup("triple"), Image.StringData.lookup("arch"));
  }

  llvm::SmallString<0> Buffer = OffloadBinary::write(Images);
  if (Buffer.size() % OffloadBinary::getAlignment() != 0)
    return createStringError("offload binary has invalid size alignment");

  if (DryRun)
    return Error::success();

  auto OutputOrErr = FileOutputBuffer::create(OutputFile, Buffer.size());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  llvm::copy(Buffer, (*OutputOrErr)->getBufferStart());
  return (*OutputOrErr)->commit();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  Executable = argv[0];
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  const OptTable &Tbl = getOptTable();
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  auto Args = Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [](StringRef Err) {
    reportError(createStringError(Err));
  });

  if (Args.hasArg(OPT_help) || Args.hasArg(OPT_help_hidden)) {
    Tbl.printHelp(
        outs(), "clang-sycl-linker [options] <input bitcode files>",
        "A utility that wraps around the SYCL device code linking process.\n"
        "This enables LLVM IR linking, post-linking and code generation for "
        "SPIR-V JIT and AOT targets.",
        Args.hasArg(OPT_help_hidden), Args.hasArg(OPT_help_hidden));
    return EXIT_SUCCESS;
  }

  if (Args.hasArg(OPT_version)) {
    printVersion(outs());
    return EXIT_SUCCESS;
  }

  Verbose = Args.hasArg(OPT_verbose);
  DryRun = Args.hasArg(OPT_dry_run);

  if (!Args.hasArg(OPT_o))
    reportError(createStringError("output file must be specified"));
  OutputFile = Args.getLastArgValue(OPT_o);

  // Get the input buffers to pass to the linking stage.
  auto ResolvedInputsOrErr = getInput(Args);
  if (!ResolvedInputsOrErr)
    reportError(ResolvedInputsOrErr.takeError());

  if (auto *A = Args.getLastArg(OPT_spirv_dump_device_code_EQ)) {
    StringRef V = A->getValue();
    if (V.empty())
      reportError(createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "--spirv-dump-device-code= requires a non-empty path"));
    SPIRVDumpDir = V;
    // The directory is shared across all split modules, which use the
    // "<output-stem>_<index>.spv" naming scheme. Concurrent invocations
    // sharing a dump dir may overwrite each other's files.
    if (!DryRun)
      if (std::error_code EC = sys::fs::create_directories(SPIRVDumpDir))
        reportError(createStringError(
            EC, "cannot create SPIR-V dump directory '" + SPIRVDumpDir + "'"));
  }

  // Run SYCL linking process on the generated inputs.
  if (Error Err = runSYCLLink(ResolvedInputsOrErr->Buffers,
                              ResolvedInputsOrErr->TargetTriple,
                              ResolvedInputsOrErr->TripleSource, Args))
    reportError(std::move(Err));

  // Remove the temporary files created.
  if (!Args.hasArg(OPT_save_temps) && !DryRun)
    for (const auto &TempFile : TempFiles)
      if (std::error_code EC = sys::fs::remove(TempFile))
        reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}
