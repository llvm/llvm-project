//===- Driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputElement.h"
#include "MarkLive.h"
#include "SymbolTable.h"
#include "Writer.h"
#include "lld/Common/Args.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Filesystem.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Reproduce.h"
#include "lld/Common/Strings.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace llvm::wasm;

namespace lld::wasm {
Ctx ctx;

void errorOrWarn(const llvm::Twine &msg) {
  if (ctx.arg.noinhibitExec)
    warn(msg);
  else
    error(msg);
}

Ctx::Ctx() {}

void Ctx::reset() {
  arg.~Config();
  new (&arg) Config();
  objectFiles.clear();
  stubFiles.clear();
  sharedFiles.clear();
  bitcodeFiles.clear();
  lazyBitcodeFiles.clear();
  syntheticFunctions.clear();
  syntheticGlobals.clear();
  syntheticTables.clear();
  whyExtractRecords.clear();
  isPic = false;
  legacyFunctionTable = false;
  emitBssSegments = false;
  sym = WasmSym{};
}

namespace {

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

// This function is called on startup. We need this for LTO since
// LTO calls LLVM functions to compile bitcode files to native code.
// Technically this can be delayed until we read bitcode files, but
// we don't bother to do lazily because the initialization is fast.
static void initLLVM() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
}

class LinkerDriver {
public:
  LinkerDriver(Ctx &);
  void linkerMain(ArrayRef<const char *> argsArr);

private:
  void createFiles(opt::InputArgList &args);
  void addFile(StringRef path);
  void addLibrary(StringRef name);

  Ctx &ctx;

  // True if we are in --whole-archive and --no-whole-archive.
  bool inWholeArchive = false;

  // True if we are in --start-lib and --end-lib.
  bool inLib = false;

  std::vector<InputFile *> files;
};

static bool hasZOption(opt::InputArgList &args, StringRef key) {
  bool ret = false;
  for (const auto *arg : args.filtered(OPT_z))
    if (key == arg->getValue()) {
      ret = true;
      arg->claim();
    }
  return ret;
}
} // anonymous namespace

bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
  // This driver-specific context will be freed later by unsafeLldMain().
  auto *context = new CommonLinkerContext;

  context->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
  context->e.cleanupCallback = []() { ctx.reset(); };
  context->e.logName = args::getFilenameWithoutExe(args[0]);
  context->e.errorLimitExceededMsg =
      "too many errors emitted, stopping now (use "
      "-error-limit=0 to see all errors)";

  symtab = make<SymbolTable>();

  initLLVM();
  LinkerDriver(ctx).linkerMain(args);

  return errorCount() == 0;
}

#define OPTTABLE_STR_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

// Create table mapping all options defined in Options.td
static constexpr opt::OptTable::Info optInfo[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS, METAVAR,     \
               VALUES)                                                         \
  {PREFIX,                                                                     \
   NAME,                                                                       \
   HELPTEXT,                                                                   \
   HELPTEXTSFORVARIANTS,                                                       \
   METAVAR,                                                                    \
   OPT_##ID,                                                                   \
   opt::Option::KIND##Class,                                                   \
   PARAM,                                                                      \
   FLAGS,                                                                      \
   VISIBILITY,                                                                 \
   OPT_##GROUP,                                                                \
   OPT_##ALIAS,                                                                \
   ALIASARGS,                                                                  \
   VALUES},
#include "Options.inc"
#undef OPTION
};

namespace {
class WasmOptTable : public opt::GenericOptTable {
public:
  WasmOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, optInfo) {}
  opt::InputArgList parse(ArrayRef<const char *> argv);
};
} // namespace

// Set color diagnostics according to -color-diagnostics={auto,always,never}
// or -no-color-diagnostics flags.
static void handleColorDiagnostics(opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_color_diagnostics, OPT_color_diagnostics_eq,
                              OPT_no_color_diagnostics);
  if (!arg)
    return;
  auto &errs = errorHandler().errs();
  if (arg->getOption().getID() == OPT_color_diagnostics) {
    errs.enable_colors(true);
  } else if (arg->getOption().getID() == OPT_no_color_diagnostics) {
    errs.enable_colors(false);
  } else {
    StringRef s = arg->getValue();
    if (s == "always")
      errs.enable_colors(true);
    else if (s == "never")
      errs.enable_colors(false);
    else if (s != "auto")
      error("unknown option: --color-diagnostics=" + s);
  }
}

static cl::TokenizerCallback getQuotingStyle(opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_rsp_quoting)) {
    StringRef s = arg->getValue();
    if (s != "windows" && s != "posix")
      error("invalid response file quoting: " + s);
    if (s == "windows")
      return cl::TokenizeWindowsCommandLine;
    return cl::TokenizeGNUCommandLine;
  }
  if (Triple(sys::getProcessTriple()).isOSWindows())
    return cl::TokenizeWindowsCommandLine;
  return cl::TokenizeGNUCommandLine;
}

// Find a file by concatenating given paths.
static std::optional<std::string> findFile(StringRef path1,
                                           const Twine &path2) {
  SmallString<128> s;
  path::append(s, path1, path2);
  if (fs::exists(s))
    return std::string(s);
  return std::nullopt;
}

opt::InputArgList WasmOptTable::parse(ArrayRef<const char *> argv) {
  SmallVector<const char *, 256> vec(argv.data(), argv.data() + argv.size());

  unsigned missingIndex;
  unsigned missingCount;

  // We need to get the quoting style for response files before parsing all
  // options so we parse here before and ignore all the options but
  // --rsp-quoting.
  opt::InputArgList args = this->ParseArgs(vec, missingIndex, missingCount);

  // Expand response files (arguments in the form of @<filename>)
  // and then parse the argument again.
  cl::ExpandResponseFiles(saver(), getQuotingStyle(args), vec);
  args = this->ParseArgs(vec, missingIndex, missingCount);

  handleColorDiagnostics(args);
  if (missingCount)
    error(Twine(args.getArgString(missingIndex)) + ": missing argument");

  for (auto *arg : args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + arg->getAsString(args));
  return args;
}

// Currently we allow a ".imports" to live alongside a library. This can
// be used to specify a list of symbols which can be undefined at link
// time (imported from the environment.  For example libc.a include an
// import file that lists the syscall functions it relies on at runtime.
// In the long run this information would be better stored as a symbol
// attribute/flag in the object file itself.
// See: https://github.com/WebAssembly/tool-conventions/issues/35
static void readImportFile(StringRef filename) {
  if (std::optional<MemoryBufferRef> buf = readFile(filename))
    for (StringRef sym : args::getLines(*buf))
      ctx.arg.allowUndefinedSymbols.insert(sym);
}

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
std::vector<std::pair<MemoryBufferRef, uint64_t>> static getArchiveMembers(
    MemoryBufferRef mb) {
  std::unique_ptr<Archive> file =
      CHECK(Archive::create(mb),
            mb.getBufferIdentifier() + ": failed to parse archive");

  std::vector<std::pair<MemoryBufferRef, uint64_t>> v;
  Error err = Error::success();
  for (const Archive::Child &c : file->children(err)) {
    MemoryBufferRef mbref =
        CHECK(c.getMemoryBufferRef(),
              mb.getBufferIdentifier() +
                  ": could not get the buffer for a child of the archive");
    v.push_back(std::make_pair(mbref, c.getChildOffset()));
  }
  if (err)
    fatal(mb.getBufferIdentifier() +
          ": Archive::children failed: " + toString(std::move(err)));

  // Take ownership of memory buffers created for members of thin archives.
  for (std::unique_ptr<MemoryBuffer> &mb : file->takeThinBuffers())
    make<std::unique_ptr<MemoryBuffer>>(std::move(mb));

  return v;
}

void LinkerDriver::addFile(StringRef path) {
  std::optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return;
  MemoryBufferRef mbref = *buffer;

  switch (identify_magic(mbref.getBuffer())) {
  case file_magic::archive: {
    SmallString<128> importFile = path;
    path::replace_extension(importFile, ".imports");
    if (fs::exists(importFile))
      readImportFile(importFile.str());

    auto members = getArchiveMembers(mbref);

    // Handle -whole-archive.
    if (inWholeArchive) {
      for (const auto &[m, offset] : members) {
        auto *object = createObjectFile(m, path, offset);
        files.push_back(object);
      }

      return;
    }

    std::unique_ptr<Archive> file =
        CHECK(Archive::create(mbref), path + ": failed to parse archive");

    for (const auto &[m, offset] : members) {
      auto magic = identify_magic(m.getBuffer());
      if (magic == file_magic::wasm_object || magic == file_magic::bitcode)
        files.push_back(createObjectFile(m, path, offset, true));
      else
        warn(path + ": archive member '" + m.getBufferIdentifier() +
             "' is neither Wasm object file nor LLVM bitcode");
    }

    return;
  }
  case file_magic::bitcode:
  case file_magic::wasm_object: {
    auto obj = createObjectFile(mbref, "", 0, inLib);
    if (ctx.arg.isStatic && isa<SharedFile>(obj)) {
      error("attempted static link of dynamic object " + path);
      break;
    }
    files.push_back(obj);
    break;
  }
  case file_magic::unknown:
    if (mbref.getBuffer().starts_with("#STUB")) {
      files.push_back(make<StubFile>(mbref));
      break;
    }
    [[fallthrough]];
  default:
    error("unknown file type: " + mbref.getBufferIdentifier());
  }
}

static std::optional<std::string> findFromSearchPaths(StringRef path) {
  for (StringRef dir : ctx.arg.searchPaths)
    if (std::optional<std::string> s = findFile(dir, path))
      return s;
  return std::nullopt;
}

// This is for -l<basename>. We'll look for lib<basename>.a from
// search paths.
static std::optional<std::string> searchLibraryBaseName(StringRef name) {
  for (StringRef dir : ctx.arg.searchPaths) {
    if (!ctx.arg.isStatic)
      if (std::optional<std::string> s = findFile(dir, "lib" + name + ".so"))
        return s;
    if (std::optional<std::string> s = findFile(dir, "lib" + name + ".a"))
      return s;
  }
  return std::nullopt;
}

// This is for -l<namespec>.
static std::optional<std::string> searchLibrary(StringRef name) {
  if (name.starts_with(":"))
    return findFromSearchPaths(name.substr(1));
  return searchLibraryBaseName(name);
}

// Add a given library by searching it from input search paths.
void LinkerDriver::addLibrary(StringRef name) {
  if (std::optional<std::string> path = searchLibrary(name))
    addFile(saver().save(*path));
  else
    error("unable to find library -l" + name, ErrorTag::LibNotFound, {name});
}

void LinkerDriver::createFiles(opt::InputArgList &args) {
  for (auto *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_library:
      addLibrary(arg->getValue());
      break;
    case OPT_INPUT:
      addFile(arg->getValue());
      break;
    case OPT_Bstatic:
      ctx.arg.isStatic = true;
      break;
    case OPT_Bdynamic:
      ctx.arg.isStatic = false;
      break;
    case OPT_whole_archive:
      inWholeArchive = true;
      break;
    case OPT_no_whole_archive:
      inWholeArchive = false;
      break;
    case OPT_start_lib:
      if (inLib)
        error("nested --start-lib");
      inLib = true;
      break;
    case OPT_end_lib:
      if (!inLib)
        error("stray --end-lib");
      inLib = false;
      break;
    }
  }
  if (files.empty() && errorCount() == 0)
    error("no input files");
}

static StringRef getAliasSpelling(opt::Arg *arg) {
  if (const opt::Arg *alias = arg->getAlias())
    return alias->getSpelling();
  return arg->getSpelling();
}

static std::pair<StringRef, StringRef> getOldNewOptions(opt::InputArgList &args,
                                                        unsigned id) {
  auto *arg = args.getLastArg(id);
  if (!arg)
    return {"", ""};

  StringRef s = arg->getValue();
  std::pair<StringRef, StringRef> ret = s.split(';');
  if (ret.second.empty())
    error(getAliasSpelling(arg) + " expects 'old;new' format, but got " + s);
  return ret;
}

// Parse options of the form "old;new[;extra]".
static std::tuple<StringRef, StringRef, StringRef>
getOldNewOptionsExtra(opt::InputArgList &args, unsigned id) {
  auto [oldDir, second] = getOldNewOptions(args, id);
  auto [newDir, extraDir] = second.split(';');
  return {oldDir, newDir, extraDir};
}

static StringRef getEntry(opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_entry, OPT_no_entry);
  if (!arg) {
    if (args.hasArg(OPT_relocatable))
      return "";
    if (args.hasArg(OPT_shared))
      return "__wasm_call_ctors";
    return "_start";
  }
  if (arg->getOption().getID() == OPT_no_entry)
    return "";
  return arg->getValue();
}

// Determines what we should do if there are remaining unresolved
// symbols after the name resolution.
static UnresolvedPolicy getUnresolvedSymbolPolicy(opt::InputArgList &args) {
  UnresolvedPolicy errorOrWarn = args.hasFlag(OPT_error_unresolved_symbols,
                                              OPT_warn_unresolved_symbols, true)
                                     ? UnresolvedPolicy::ReportError
                                     : UnresolvedPolicy::Warn;

  if (auto *arg = args.getLastArg(OPT_unresolved_symbols)) {
    StringRef s = arg->getValue();
    if (s == "ignore-all")
      return UnresolvedPolicy::Ignore;
    if (s == "import-dynamic")
      return UnresolvedPolicy::ImportDynamic;
    if (s == "report-all")
      return errorOrWarn;
    error("unknown --unresolved-symbols value: " + s);
  }

  return errorOrWarn;
}

// Parse --build-id or --build-id=<style>. We handle "tree" as a
// synonym for "sha1" because all our hash functions including
// -build-id=sha1 are actually tree hashes for performance reasons.
static std::pair<BuildIdKind, SmallVector<uint8_t, 0>>
getBuildId(opt::InputArgList &args) {
  auto *arg = args.getLastArg(OPT_build_id, OPT_build_id_eq);
  if (!arg)
    return {BuildIdKind::None, {}};

  if (arg->getOption().getID() == OPT_build_id)
    return {BuildIdKind::Fast, {}};

  StringRef s = arg->getValue();
  if (s == "fast")
    return {BuildIdKind::Fast, {}};
  if (s == "sha1" || s == "tree")
    return {BuildIdKind::Sha1, {}};
  if (s == "uuid")
    return {BuildIdKind::Uuid, {}};
  if (s.starts_with("0x"))
    return {BuildIdKind::Hexstring, parseHex(s.substr(2))};

  if (s != "none")
    error("unknown --build-id style: " + s);
  return {BuildIdKind::None, {}};
}

// Initializes Config members by the command line options.
static void readConfigs(opt::InputArgList &args) {
  ctx.arg.allowMultipleDefinition =
      hasZOption(args, "muldefs") ||
      args.hasFlag(OPT_allow_multiple_definition,
                   OPT_no_allow_multiple_definition, false);
  ctx.arg.bsymbolic = args.hasArg(OPT_Bsymbolic);
  ctx.arg.checkFeatures =
      args.hasFlag(OPT_check_features, OPT_no_check_features, true);
  ctx.arg.compressRelocations = args.hasArg(OPT_compress_relocations);
  ctx.arg.demangle = args.hasFlag(OPT_demangle, OPT_no_demangle, true);
  ctx.arg.disableVerify = args.hasArg(OPT_disable_verify);
  ctx.arg.emitRelocs = args.hasArg(OPT_emit_relocs);
  ctx.arg.experimentalPic = args.hasArg(OPT_experimental_pic);
  ctx.arg.entry = getEntry(args);
  ctx.arg.exportAll = args.hasArg(OPT_export_all);
  ctx.arg.exportTable = args.hasArg(OPT_export_table);
  ctx.arg.growableTable = args.hasArg(OPT_growable_table);
  ctx.arg.noinhibitExec = args.hasArg(OPT_noinhibit_exec);

  if (args.hasArg(OPT_import_memory_with_name)) {
    ctx.arg.memoryImport =
        args.getLastArgValue(OPT_import_memory_with_name).split(",");
  } else if (args.hasArg(OPT_import_memory)) {
    ctx.arg.memoryImport =
        std::pair<llvm::StringRef, llvm::StringRef>(defaultModule, memoryName);
  } else {
    ctx.arg.memoryImport =
        std::optional<std::pair<llvm::StringRef, llvm::StringRef>>();
  }

  if (args.hasArg(OPT_export_memory_with_name)) {
    ctx.arg.memoryExport = args.getLastArgValue(OPT_export_memory_with_name);
  } else if (args.hasArg(OPT_export_memory)) {
    ctx.arg.memoryExport = memoryName;
  } else {
    ctx.arg.memoryExport = std::optional<llvm::StringRef>();
  }

  ctx.arg.sharedMemory = args.hasArg(OPT_shared_memory);
  ctx.arg.soName = args.getLastArgValue(OPT_soname);
  ctx.arg.importTable = args.hasArg(OPT_import_table);
  ctx.arg.importUndefined = args.hasArg(OPT_import_undefined);
  ctx.arg.ltoo = args::getInteger(args, OPT_lto_O, 2);
  if (ctx.arg.ltoo > 3)
    error("invalid optimization level for LTO: " + Twine(ctx.arg.ltoo));
  unsigned ltoCgo =
      args::getInteger(args, OPT_lto_CGO, args::getCGOptLevel(ctx.arg.ltoo));
  if (auto level = CodeGenOpt::getLevel(ltoCgo))
    ctx.arg.ltoCgo = *level;
  else
    error("invalid codegen optimization level for LTO: " + Twine(ltoCgo));
  ctx.arg.ltoPartitions = args::getInteger(args, OPT_lto_partitions, 1);
  ctx.arg.ltoObjPath = args.getLastArgValue(OPT_lto_obj_path_eq);
  ctx.arg.ltoDebugPassManager = args.hasArg(OPT_lto_debug_pass_manager);
  ctx.arg.mapFile = args.getLastArgValue(OPT_Map);
  ctx.arg.optimize = args::getInteger(args, OPT_O, 1);
  ctx.arg.outputFile = args.getLastArgValue(OPT_o);
  ctx.arg.relocatable = args.hasArg(OPT_relocatable);
  ctx.arg.rpath = args::getStrings(args, OPT_rpath);
  ctx.arg.gcSections =
      args.hasFlag(OPT_gc_sections, OPT_no_gc_sections, !ctx.arg.relocatable);
  for (auto *arg : args.filtered(OPT_keep_section))
    ctx.arg.keepSections.insert(arg->getValue());
  ctx.arg.mergeDataSegments =
      args.hasFlag(OPT_merge_data_segments, OPT_no_merge_data_segments,
                   !ctx.arg.relocatable);
  ctx.arg.pie = args.hasFlag(OPT_pie, OPT_no_pie, false);
  ctx.arg.printGcSections =
      args.hasFlag(OPT_print_gc_sections, OPT_no_print_gc_sections, false);
  ctx.arg.saveTemps = args.hasArg(OPT_save_temps);
  ctx.arg.searchPaths = args::getStrings(args, OPT_library_path);
  ctx.arg.shared = args.hasArg(OPT_shared);
  ctx.arg.shlibSigCheck = !args.hasArg(OPT_no_shlib_sigcheck);
  ctx.arg.stripAll = args.hasArg(OPT_strip_all);
  ctx.arg.stripDebug = args.hasArg(OPT_strip_debug);
  ctx.arg.stackFirst = args.hasArg(OPT_stack_first);
  ctx.arg.trace = args.hasArg(OPT_trace);
  ctx.arg.thinLTOCacheDir = args.getLastArgValue(OPT_thinlto_cache_dir);
  ctx.arg.thinLTOCachePolicy = CHECK(
      parseCachePruningPolicy(args.getLastArgValue(OPT_thinlto_cache_policy)),
      "--thinlto-cache-policy: invalid cache policy");
  ctx.arg.thinLTOEmitImportsFiles = args.hasArg(OPT_thinlto_emit_imports_files);
  ctx.arg.thinLTOEmitIndexFiles = args.hasArg(OPT_thinlto_emit_index_files) ||
                                  args.hasArg(OPT_thinlto_index_only) ||
                                  args.hasArg(OPT_thinlto_index_only_eq);
  ctx.arg.thinLTOIndexOnly = args.hasArg(OPT_thinlto_index_only) ||
                             args.hasArg(OPT_thinlto_index_only_eq);
  ctx.arg.thinLTOIndexOnlyArg = args.getLastArgValue(OPT_thinlto_index_only_eq);
  ctx.arg.thinLTOObjectSuffixReplace =
      getOldNewOptions(args, OPT_thinlto_object_suffix_replace_eq);
  std::tie(ctx.arg.thinLTOPrefixReplaceOld, ctx.arg.thinLTOPrefixReplaceNew,
           ctx.arg.thinLTOPrefixReplaceNativeObject) =
      getOldNewOptionsExtra(args, OPT_thinlto_prefix_replace_eq);
  if (ctx.arg.thinLTOEmitIndexFiles && !ctx.arg.thinLTOIndexOnly) {
    if (args.hasArg(OPT_thinlto_object_suffix_replace_eq))
      error("--thinlto-object-suffix-replace is not supported with "
            "--thinlto-emit-index-files");
    else if (args.hasArg(OPT_thinlto_prefix_replace_eq))
      error("--thinlto-prefix-replace is not supported with "
            "--thinlto-emit-index-files");
  }
  if (!ctx.arg.thinLTOPrefixReplaceNativeObject.empty() &&
      ctx.arg.thinLTOIndexOnlyArg.empty()) {
    error("--thinlto-prefix-replace=old_dir;new_dir;obj_dir must be used with "
          "--thinlto-index-only=");
  }
  ctx.arg.unresolvedSymbols = getUnresolvedSymbolPolicy(args);
  ctx.arg.whyExtract = args.getLastArgValue(OPT_why_extract);
  errorHandler().verbose = args.hasArg(OPT_verbose);
  LLVM_DEBUG(errorHandler().verbose = true);

  ctx.arg.tableBase = args::getInteger(args, OPT_table_base, 0);
  ctx.arg.globalBase = args::getInteger(args, OPT_global_base, 0);
  ctx.arg.initialHeap = args::getInteger(args, OPT_initial_heap, 0);
  ctx.arg.initialMemory = args::getInteger(args, OPT_initial_memory, 0);
  ctx.arg.maxMemory = args::getInteger(args, OPT_max_memory, 0);
  ctx.arg.noGrowableMemory = args.hasArg(OPT_no_growable_memory);
  ctx.arg.zStackSize =
      args::getZOptionValue(args, OPT_z, "stack-size", WasmDefaultPageSize);
  ctx.arg.pageSize = args::getInteger(args, OPT_page_size, WasmDefaultPageSize);
  if (ctx.arg.pageSize != 1 && ctx.arg.pageSize != WasmDefaultPageSize)
    error("--page_size=N must be either 1 or 65536");

  // -Bdynamic by default if -pie or -shared is specified.
  if (ctx.arg.pie || ctx.arg.shared)
    ctx.arg.isStatic = false;

  if (ctx.arg.maxMemory != 0 && ctx.arg.noGrowableMemory) {
    // Erroring out here is simpler than defining precedence rules.
    error("--max-memory is incompatible with --no-growable-memory");
  }

  // Default value of exportDynamic depends on `-shared`
  ctx.arg.exportDynamic =
      args.hasFlag(OPT_export_dynamic, OPT_no_export_dynamic, ctx.arg.shared);

  // Parse wasm32/64.
  if (auto *arg = args.getLastArg(OPT_m)) {
    StringRef s = arg->getValue();
    if (s == "wasm32")
      ctx.arg.is64 = false;
    else if (s == "wasm64")
      ctx.arg.is64 = true;
    else
      error("invalid target architecture: " + s);
  }

  // --threads= takes a positive integer and provides the default value for
  // --thinlto-jobs=.
  if (auto *arg = args.getLastArg(OPT_threads)) {
    StringRef v(arg->getValue());
    unsigned threads = 0;
    if (!llvm::to_integer(v, threads, 0) || threads == 0)
      error(arg->getSpelling() + ": expected a positive integer, but got '" +
            arg->getValue() + "'");
    parallel::strategy = hardware_concurrency(threads);
    ctx.arg.thinLTOJobs = v;
  }
  if (auto *arg = args.getLastArg(OPT_thinlto_jobs))
    ctx.arg.thinLTOJobs = arg->getValue();

  if (auto *arg = args.getLastArg(OPT_features)) {
    ctx.arg.features =
        std::optional<std::vector<std::string>>(std::vector<std::string>());
    for (StringRef s : arg->getValues())
      ctx.arg.features->push_back(std::string(s));
  }

  if (auto *arg = args.getLastArg(OPT_extra_features)) {
    ctx.arg.extraFeatures =
        std::optional<std::vector<std::string>>(std::vector<std::string>());
    for (StringRef s : arg->getValues())
      ctx.arg.extraFeatures->push_back(std::string(s));
  }

  // Legacy --allow-undefined flag which is equivalent to
  // --unresolve-symbols=ignore + --import-undefined
  if (args.hasArg(OPT_allow_undefined)) {
    ctx.arg.importUndefined = true;
    ctx.arg.unresolvedSymbols = UnresolvedPolicy::Ignore;
  }

  if (args.hasArg(OPT_print_map))
    ctx.arg.mapFile = "-";

  std::tie(ctx.arg.buildId, ctx.arg.buildIdVector) = getBuildId(args);
}

// Some Config members do not directly correspond to any particular
// command line options, but computed based on other Config values.
// This function initialize such members. See Config.h for the details
// of these values.
static void setConfigs() {
  ctx.isPic = ctx.arg.pie || ctx.arg.shared;

  if (ctx.isPic) {
    if (ctx.arg.exportTable)
      error("-shared/-pie is incompatible with --export-table");
    ctx.arg.importTable = true;
  } else {
    // Default table base.  Defaults to 1, reserving 0 for the NULL function
    // pointer.
    if (!ctx.arg.tableBase)
      ctx.arg.tableBase = 1;
    // The default offset for static/global data, for when --global-base is
    // not specified on the command line.  The precise value of 1024 is
    // somewhat arbitrary, and pre-dates wasm-ld (Its the value that
    // emscripten used prior to wasm-ld).
    if (!ctx.arg.globalBase && !ctx.arg.relocatable && !ctx.arg.stackFirst)
      ctx.arg.globalBase = 1024;
  }

  if (ctx.arg.relocatable) {
    if (ctx.arg.exportTable)
      error("--relocatable is incompatible with --export-table");
    if (ctx.arg.growableTable)
      error("--relocatable is incompatible with --growable-table");
    // Ignore any --import-table, as it's redundant.
    ctx.arg.importTable = true;
  }

  if (ctx.arg.shared) {
    if (ctx.arg.memoryExport.has_value()) {
      error("--export-memory is incompatible with --shared");
    }
    if (!ctx.arg.memoryImport.has_value()) {
      ctx.arg.memoryImport = std::pair<llvm::StringRef, llvm::StringRef>(
          defaultModule, memoryName);
    }
  }

  // If neither export-memory nor import-memory is specified, default to
  // exporting memory under its default name.
  if (!ctx.arg.memoryExport.has_value() && !ctx.arg.memoryImport.has_value()) {
    ctx.arg.memoryExport = memoryName;
  }
}

// Some command line options or some combinations of them are not allowed.
// This function checks for such errors.
static void checkOptions(opt::InputArgList &args) {
  if (!ctx.arg.stripDebug && !ctx.arg.stripAll && ctx.arg.compressRelocations)
    error("--compress-relocations is incompatible with output debug"
          " information. Please pass --strip-debug or --strip-all");

  if (ctx.arg.ltoPartitions == 0)
    error("--lto-partitions: number of threads must be > 0");
  if (!get_threadpool_strategy(ctx.arg.thinLTOJobs))
    error("--thinlto-jobs: invalid job count: " + ctx.arg.thinLTOJobs);

  if (ctx.arg.pie && ctx.arg.shared)
    error("-shared and -pie may not be used together");

  if (ctx.arg.outputFile.empty() && !ctx.arg.thinLTOIndexOnly)
    error("no output file specified");

  if (ctx.arg.importTable && ctx.arg.exportTable)
    error("--import-table and --export-table may not be used together");

  if (ctx.arg.relocatable) {
    if (!ctx.arg.entry.empty())
      error("entry point specified for relocatable output file");
    if (ctx.arg.gcSections)
      error("-r and --gc-sections may not be used together");
    if (ctx.arg.compressRelocations)
      error("-r -and --compress-relocations may not be used together");
    if (args.hasArg(OPT_undefined))
      error("-r -and --undefined may not be used together");
    if (ctx.arg.pie)
      error("-r and -pie may not be used together");
    if (ctx.arg.sharedMemory)
      error("-r and --shared-memory may not be used together");
    if (ctx.arg.globalBase)
      error("-r and --global-base may not by used together");
  }

  // To begin to prepare for Module Linking-style shared libraries, start
  // warning about uses of `-shared` and related flags outside of Experimental
  // mode, to give anyone using them a heads-up that they will be changing.
  //
  // Also, warn about flags which request explicit exports.
  if (!ctx.arg.experimentalPic) {
    // -shared will change meaning when Module Linking is implemented.
    if (ctx.arg.shared) {
      warn("creating shared libraries, with -shared, is not yet stable");
    }

    // -pie will change meaning when Module Linking is implemented.
    if (ctx.arg.pie) {
      warn("creating PIEs, with -pie, is not yet stable");
    }

    if (ctx.arg.unresolvedSymbols == UnresolvedPolicy::ImportDynamic) {
      warn("dynamic imports are not yet stable "
           "(--unresolved-symbols=import-dynamic)");
    }
  }

  if (ctx.arg.bsymbolic && !ctx.arg.shared) {
    warn("-Bsymbolic is only meaningful when combined with -shared");
  }

  if (ctx.isPic) {
    if (ctx.arg.globalBase)
      error("--global-base may not be used with -shared/-pie");
    if (ctx.arg.tableBase)
      error("--table-base may not be used with -shared/-pie");
  }
}

static const char *getReproduceOption(opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_reproduce))
    return arg->getValue();
  return getenv("LLD_REPRODUCE");
}

// Force Sym to be entered in the output. Used for -u or equivalent.
static Symbol *handleUndefined(StringRef name, const char *option) {
  Symbol *sym = symtab->find(name);
  if (!sym)
    return nullptr;

  // Since symbol S may not be used inside the program, LTO may
  // eliminate it. Mark the symbol as "used" to prevent it.
  sym->isUsedInRegularObj = true;

  if (auto *lazySym = dyn_cast<LazySymbol>(sym)) {
    lazySym->extract();
    if (!ctx.arg.whyExtract.empty())
      ctx.whyExtractRecords.emplace_back(option, sym->getFile(), *sym);
  }

  return sym;
}

static void handleLibcall(StringRef name) {
  Symbol *sym = symtab->find(name);
  if (sym && sym->isLazy() && isa<BitcodeFile>(sym->getFile())) {
    if (!ctx.arg.whyExtract.empty())
      ctx.whyExtractRecords.emplace_back("<libcall>", sym->getFile(), *sym);
    cast<LazySymbol>(sym)->extract();
  }
}

static void writeWhyExtract() {
  if (ctx.arg.whyExtract.empty())
    return;

  std::error_code ec;
  raw_fd_ostream os(ctx.arg.whyExtract, ec, sys::fs::OF_None);
  if (ec) {
    error("cannot open --why-extract= file " + ctx.arg.whyExtract + ": " +
          ec.message());
    return;
  }

  os << "reference\textracted\tsymbol\n";
  for (auto &entry : ctx.whyExtractRecords) {
    os << std::get<0>(entry) << '\t' << toString(std::get<1>(entry)) << '\t'
       << toString(std::get<2>(entry)) << '\n';
  }
}

// Equivalent of demote demoteSharedAndLazySymbols() in the ELF linker
static void demoteLazySymbols() {
  for (Symbol *sym : symtab->symbols()) {
    if (auto* s = dyn_cast<LazySymbol>(sym)) {
      if (s->signature) {
        LLVM_DEBUG(llvm::dbgs()
                   << "demoting lazy func: " << s->getName() << "\n");
        replaceSymbol<UndefinedFunction>(s, s->getName(), std::nullopt,
                                         std::nullopt, WASM_SYMBOL_BINDING_WEAK,
                                         s->getFile(), s->signature);
      }
    }
  }
}

static UndefinedGlobal *
createUndefinedGlobal(StringRef name, llvm::wasm::WasmGlobalType *type) {
  auto *sym = cast<UndefinedGlobal>(symtab->addUndefinedGlobal(
      name, std::nullopt, std::nullopt, WASM_SYMBOL_UNDEFINED, nullptr, type));
  ctx.arg.allowUndefinedSymbols.insert(sym->getName());
  sym->isUsedInRegularObj = true;
  return sym;
}

static InputGlobal *createGlobal(StringRef name, bool isMutable) {
  llvm::wasm::WasmGlobal wasmGlobal;
  bool is64 = ctx.arg.is64.value_or(false);
  wasmGlobal.Type = {uint8_t(is64 ? WASM_TYPE_I64 : WASM_TYPE_I32), isMutable};
  wasmGlobal.InitExpr = intConst(0, is64);
  wasmGlobal.SymbolName = name;
  return make<InputGlobal>(wasmGlobal, nullptr);
}

static GlobalSymbol *createGlobalVariable(StringRef name, bool isMutable) {
  InputGlobal *g = createGlobal(name, isMutable);
  return symtab->addSyntheticGlobal(name, WASM_SYMBOL_VISIBILITY_HIDDEN, g);
}

static GlobalSymbol *createOptionalGlobal(StringRef name, bool isMutable) {
  InputGlobal *g = createGlobal(name, isMutable);
  return symtab->addOptionalGlobalSymbol(name, g);
}

// Create ABI-defined synthetic symbols
static void createSyntheticSymbols() {
  if (ctx.arg.relocatable)
    return;

  static WasmSignature nullSignature = {{}, {}};
  static WasmSignature i32ArgSignature = {{}, {ValType::I32}};
  static WasmSignature i64ArgSignature = {{}, {ValType::I64}};
  static llvm::wasm::WasmGlobalType globalTypeI32 = {WASM_TYPE_I32, false};
  static llvm::wasm::WasmGlobalType globalTypeI64 = {WASM_TYPE_I64, false};
  static llvm::wasm::WasmGlobalType mutableGlobalTypeI32 = {WASM_TYPE_I32,
                                                            true};
  static llvm::wasm::WasmGlobalType mutableGlobalTypeI64 = {WASM_TYPE_I64,
                                                            true};
  ctx.sym.callCtors = symtab->addSyntheticFunction(
      "__wasm_call_ctors", WASM_SYMBOL_VISIBILITY_HIDDEN,
      make<SyntheticFunction>(nullSignature, "__wasm_call_ctors"));

  bool is64 = ctx.arg.is64.value_or(false);

  if (ctx.isPic) {
    ctx.sym.stackPointer =
        createUndefinedGlobal("__stack_pointer", ctx.arg.is64.value_or(false)
                                                     ? &mutableGlobalTypeI64
                                                     : &mutableGlobalTypeI32);
    // For PIC code, we import two global variables (__memory_base and
    // __table_base) from the environment and use these as the offset at
    // which to load our static data and function table.
    // See:
    // https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md
    auto *globalType = is64 ? &globalTypeI64 : &globalTypeI32;
    ctx.sym.memoryBase = createUndefinedGlobal("__memory_base", globalType);
    ctx.sym.tableBase = createUndefinedGlobal("__table_base", globalType);
    ctx.sym.memoryBase->markLive();
    ctx.sym.tableBase->markLive();
  } else {
    // For non-PIC code
    ctx.sym.stackPointer = createGlobalVariable("__stack_pointer", true);
    ctx.sym.stackPointer->markLive();
  }

  if (ctx.arg.sharedMemory) {
    ctx.sym.tlsBase = createGlobalVariable("__tls_base", true);
    ctx.sym.tlsSize = createGlobalVariable("__tls_size", false);
    ctx.sym.tlsAlign = createGlobalVariable("__tls_align", false);
    ctx.sym.initTLS = symtab->addSyntheticFunction(
        "__wasm_init_tls", WASM_SYMBOL_VISIBILITY_HIDDEN,
        make<SyntheticFunction>(is64 ? i64ArgSignature : i32ArgSignature,
                                "__wasm_init_tls"));
  }
}

static void createOptionalSymbols() {
  if (ctx.arg.relocatable)
    return;

  ctx.sym.dsoHandle = symtab->addOptionalDataSymbol("__dso_handle");

  if (!ctx.arg.shared)
    ctx.sym.dataEnd = symtab->addOptionalDataSymbol("__data_end");

  if (!ctx.isPic) {
    ctx.sym.stackLow = symtab->addOptionalDataSymbol("__stack_low");
    ctx.sym.stackHigh = symtab->addOptionalDataSymbol("__stack_high");
    ctx.sym.globalBase = symtab->addOptionalDataSymbol("__global_base");
    ctx.sym.heapBase = symtab->addOptionalDataSymbol("__heap_base");
    ctx.sym.heapEnd = symtab->addOptionalDataSymbol("__heap_end");
    ctx.sym.definedMemoryBase = symtab->addOptionalDataSymbol("__memory_base");
    ctx.sym.definedTableBase = symtab->addOptionalDataSymbol("__table_base");
  }

  ctx.sym.firstPageEnd = symtab->addOptionalDataSymbol("__wasm_first_page_end");
  if (ctx.sym.firstPageEnd)
    ctx.sym.firstPageEnd->setVA(ctx.arg.pageSize);

  // For non-shared memory programs we still need to define __tls_base since we
  // allow object files built with TLS to be linked into single threaded
  // programs, and such object files can contain references to this symbol.
  //
  // However, in this case __tls_base is immutable and points directly to the
  // start of the `.tdata` static segment.
  //
  // __tls_size and __tls_align are not needed in this case since they are only
  // needed for __wasm_init_tls (which we do not create in this case).
  if (!ctx.arg.sharedMemory)
    ctx.sym.tlsBase = createOptionalGlobal("__tls_base", false);
}

static void processStubLibrariesPreLTO() {
  log("-- processStubLibrariesPreLTO");
  for (auto &stub_file : ctx.stubFiles) {
    LLVM_DEBUG(llvm::dbgs()
               << "processing stub file: " << stub_file->getName() << "\n");
    for (auto [name, deps]: stub_file->symbolDependencies) {
      auto* sym = symtab->find(name);
      // If the symbol is not present at all (yet), or if it is present but
      // undefined, then mark the dependent symbols as used by a regular
      // object so they will be preserved and exported by the LTO process.
      if (!sym || sym->isUndefined()) {
        for (const auto dep : deps) {
          auto* needed = symtab->find(dep);
          if (needed ) {
            needed->isUsedInRegularObj = true;
            // Like with handleLibcall we have to extract any LTO archive
            // members that might need to be exported due to stub library
            // symbols being referenced.  Without this the LTO object could be
            // extracted during processStubLibraries, which is too late since
            // LTO has already being performed at that point.
            if (needed->isLazy() && isa<BitcodeFile>(needed->getFile())) {
              if (!ctx.arg.whyExtract.empty())
                ctx.whyExtractRecords.emplace_back(toString(stub_file),
                                                   needed->getFile(), *needed);
              cast<LazySymbol>(needed)->extract();
            }
          }
        }
      }
    }
  }
}

static bool addStubSymbolDeps(const StubFile *stub_file, Symbol *sym,
                              ArrayRef<StringRef> deps) {
  // The first stub library to define a given symbol sets this and
  // definitions in later stub libraries are ignored.
  if (sym->forceImport)
    return false; // Already handled
  sym->forceImport = true;
  if (sym->traced)
    message(toString(stub_file) + ": importing " + sym->getName());
  else
    LLVM_DEBUG(llvm::dbgs() << toString(stub_file) << ": importing "
                            << sym->getName() << "\n");
  bool depsAdded = false;
  for (const auto dep : deps) {
    auto *needed = symtab->find(dep);
    if (!needed) {
      error(toString(stub_file) + ": undefined symbol: " + dep +
            ". Required by " + toString(*sym));
    } else if (needed->isUndefined()) {
      error(toString(stub_file) + ": undefined symbol: " + toString(*needed) +
            ". Required by " + toString(*sym));
    } else {
      if (needed->traced)
        message(toString(stub_file) + ": exported " + toString(*needed) +
                " due to import of " + sym->getName());
      else
        LLVM_DEBUG(llvm::dbgs()
                   << "force export: " << toString(*needed) << "\n");
      needed->forceExport = true;
      if (auto *lazy = dyn_cast<LazySymbol>(needed)) {
        depsAdded = true;
        lazy->extract();
        if (!ctx.arg.whyExtract.empty())
          ctx.whyExtractRecords.emplace_back(toString(stub_file),
                                             sym->getFile(), *sym);
      }
    }
  }
  return depsAdded;
}

static void processStubLibraries() {
  log("-- processStubLibraries");
  bool depsAdded = false;
  do {
    depsAdded = false;
    for (auto &stub_file : ctx.stubFiles) {
      LLVM_DEBUG(llvm::dbgs()
                 << "processing stub file: " << stub_file->getName() << "\n");

      // First look for any imported symbols that directly match
      // the names of the stub imports
      for (auto [name, deps]: stub_file->symbolDependencies) {
        auto* sym = symtab->find(name);
        if (sym && sym->isUndefined()) {
          depsAdded |= addStubSymbolDeps(stub_file, sym, deps);
        } else {
          if (sym && sym->traced)
            message(toString(stub_file) + ": stub symbol not needed: " + name);
          else
            LLVM_DEBUG(llvm::dbgs()
                       << "stub symbol not needed: `" << name << "`\n");
        }
      }

      // Secondly looks for any symbols with an `importName` that matches
      for (Symbol *sym : symtab->symbols()) {
        if (sym->isUndefined() && sym->importName.has_value()) {
          auto it = stub_file->symbolDependencies.find(sym->importName.value());
          if (it != stub_file->symbolDependencies.end()) {
            depsAdded |= addStubSymbolDeps(stub_file, sym, it->second);
          }
        }
      }
    }
  } while (depsAdded);

  log("-- done processStubLibraries");
}

// Reconstructs command line arguments so that so that you can re-run
// the same command with the same inputs. This is for --reproduce.
static std::string createResponseFile(const opt::InputArgList &args) {
  SmallString<0> data;
  raw_svector_ostream os(data);

  // Copy the command line to the output while rewriting paths.
  for (auto *arg : args) {
    switch (arg->getOption().getID()) {
    case OPT_reproduce:
      break;
    case OPT_INPUT:
      os << quote(relativeToRoot(arg->getValue())) << "\n";
      break;
    case OPT_o:
      // If -o path contains directories, "lld @response.txt" will likely
      // fail because the archive we are creating doesn't contain empty
      // directories for the output path (-o doesn't create directories).
      // Strip directories to prevent the issue.
      os << "-o " << quote(sys::path::filename(arg->getValue())) << "\n";
      break;
    default:
      os << toString(*arg) << "\n";
    }
  }
  return std::string(data);
}

// The --wrap option is a feature to rename symbols so that you can write
// wrappers for existing functions. If you pass `-wrap=foo`, all
// occurrences of symbol `foo` are resolved to `wrap_foo` (so, you are
// expected to write `wrap_foo` function as a wrapper). The original
// symbol becomes accessible as `real_foo`, so you can call that from your
// wrapper.
//
// This data structure is instantiated for each -wrap option.
struct WrappedSymbol {
  Symbol *sym;
  Symbol *real;
  Symbol *wrap;
};

static Symbol *addUndefined(StringRef name) {
  return symtab->addUndefinedFunction(name, std::nullopt, std::nullopt,
                                      WASM_SYMBOL_UNDEFINED, nullptr, nullptr,
                                      false);
}

// Handles -wrap option.
//
// This function instantiates wrapper symbols. At this point, they seem
// like they are not being used at all, so we explicitly set some flags so
// that LTO won't eliminate them.
static std::vector<WrappedSymbol> addWrappedSymbols(opt::InputArgList &args) {
  std::vector<WrappedSymbol> v;
  DenseSet<StringRef> seen;

  for (auto *arg : args.filtered(OPT_wrap)) {
    StringRef name = arg->getValue();
    if (!seen.insert(name).second)
      continue;

    Symbol *sym = symtab->find(name);
    if (!sym)
      continue;

    Symbol *real = addUndefined(saver().save("__real_" + name));
    Symbol *wrap = addUndefined(saver().save("__wrap_" + name));
    v.push_back({sym, real, wrap});

    // We want to tell LTO not to inline symbols to be overwritten
    // because LTO doesn't know the final symbol contents after renaming.
    real->canInline = false;
    sym->canInline = false;

    // Tell LTO not to eliminate these symbols.
    sym->isUsedInRegularObj = true;
    wrap->isUsedInRegularObj = true;
    real->isUsedInRegularObj = false;
  }
  return v;
}

// Do renaming for -wrap by updating pointers to symbols.
//
// When this function is executed, only InputFiles and symbol table
// contain pointers to symbol objects. We visit them to replace pointers,
// so that wrapped symbols are swapped as instructed by the command line.
static void wrapSymbols(ArrayRef<WrappedSymbol> wrapped) {
  DenseMap<Symbol *, Symbol *> map;
  for (const WrappedSymbol &w : wrapped) {
    map[w.sym] = w.wrap;
    map[w.real] = w.sym;
  }

  // Update pointers in input files.
  parallelForEach(ctx.objectFiles, [&](InputFile *file) {
    MutableArrayRef<Symbol *> syms = file->getMutableSymbols();
    for (Symbol *&sym : syms)
      if (Symbol *s = map.lookup(sym))
        sym = s;
  });

  // Update pointers in the symbol table.
  for (const WrappedSymbol &w : wrapped)
    symtab->wrap(w.sym, w.real, w.wrap);
}

static void splitSections() {
  // splitIntoPieces needs to be called on each MergeInputChunk
  // before calling finalizeContents().
  LLVM_DEBUG(llvm::dbgs() << "splitSections\n");
  parallelForEach(ctx.objectFiles, [](ObjFile *file) {
    for (InputChunk *seg : file->segments) {
      if (auto *s = dyn_cast<MergeInputChunk>(seg))
        s->splitIntoPieces();
    }
    for (InputChunk *sec : file->customSections) {
      if (auto *s = dyn_cast<MergeInputChunk>(sec))
        s->splitIntoPieces();
    }
  });
}

static bool isKnownZFlag(StringRef s) {
  // For now, we only support a very limited set of -z flags
  return s.starts_with("stack-size=") || s.starts_with("muldefs");
}

// Report a warning for an unknown -z option.
static void checkZOptions(opt::InputArgList &args) {
  for (auto *arg : args.filtered(OPT_z))
    if (!isKnownZFlag(arg->getValue()))
      warn("unknown -z value: " + StringRef(arg->getValue()));
}

LinkerDriver::LinkerDriver(Ctx &ctx) : ctx(ctx) {}

void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
  WasmOptTable parser;
  opt::InputArgList args = parser.parse(argsArr.slice(1));

  // Interpret these flags early because error()/warn() depend on them.
  auto &errHandler = errorHandler();
  errHandler.errorLimit = args::getInteger(args, OPT_error_limit, 20);
  errHandler.fatalWarnings =
      args.hasFlag(OPT_fatal_warnings, OPT_no_fatal_warnings, false);
  checkZOptions(args);

  // Handle --help
  if (args.hasArg(OPT_help)) {
    parser.printHelp(errHandler.outs(),
                     (std::string(argsArr[0]) + " [options] file...").c_str(),
                     "LLVM Linker", false);
    return;
  }

  // Handle -v or -version.
  if (args.hasArg(OPT_v) || args.hasArg(OPT_version))
    errHandler.outs() << getLLDVersion() << "\n";

  // Handle --reproduce
  if (const char *path = getReproduceOption(args)) {
    Expected<std::unique_ptr<TarWriter>> errOrWriter =
        TarWriter::create(path, path::stem(path));
    if (errOrWriter) {
      tar = std::move(*errOrWriter);
      tar->append("response.txt", createResponseFile(args));
      tar->append("version.txt", getLLDVersion() + "\n");
    } else {
      error("--reproduce: " + toString(errOrWriter.takeError()));
    }
  }

  // Parse and evaluate -mllvm options.
  std::vector<const char *> v;
  v.push_back("wasm-ld (LLVM option parsing)");
  for (auto *arg : args.filtered(OPT_mllvm))
    v.push_back(arg->getValue());
  cl::ResetAllOptionOccurrences();
  cl::ParseCommandLineOptions(v.size(), v.data());

  readConfigs(args);
  setConfigs();

  // The behavior of -v or --version is a bit strange, but this is
  // needed for compatibility with GNU linkers.
  if (args.hasArg(OPT_v) && !args.hasArg(OPT_INPUT))
    return;
  if (args.hasArg(OPT_version))
    return;

  createFiles(args);
  if (errorCount())
    return;

  checkOptions(args);
  if (errorCount())
    return;

  if (auto *arg = args.getLastArg(OPT_allow_undefined_file))
    readImportFile(arg->getValue());

  // Fail early if the output file or map file is not writable. If a user has a
  // long link, e.g. due to a large LTO link, they do not wish to run it and
  // find that it failed because there was a mistake in their command-line.
  if (auto e = tryCreateFile(ctx.arg.outputFile))
    error("cannot open output file " + ctx.arg.outputFile + ": " + e.message());
  if (auto e = tryCreateFile(ctx.arg.mapFile))
    error("cannot open map file " + ctx.arg.mapFile + ": " + e.message());
  if (errorCount())
    return;

  // Handle --trace-symbol.
  for (auto *arg : args.filtered(OPT_trace_symbol))
    symtab->trace(arg->getValue());

  for (auto *arg : args.filtered(OPT_export_if_defined))
    ctx.arg.exportedSymbols.insert(arg->getValue());

  for (auto *arg : args.filtered(OPT_export)) {
    ctx.arg.exportedSymbols.insert(arg->getValue());
    ctx.arg.requiredExports.push_back(arg->getValue());
  }

  createSyntheticSymbols();

  // Add all files to the symbol table. This will add almost all
  // symbols that we need to the symbol table.
  for (InputFile *f : files)
    symtab->addFile(f);
  if (errorCount())
    return;

  // Handle the `--undefined <sym>` options.
  for (auto *arg : args.filtered(OPT_undefined))
    handleUndefined(arg->getValue(), "<internal>");

  // Handle the `--export <sym>` options
  // This works like --undefined but also exports the symbol if its found
  for (auto &iter : ctx.arg.exportedSymbols)
    handleUndefined(iter.first(), "--export");

  Symbol *entrySym = nullptr;
  if (!ctx.arg.relocatable && !ctx.arg.entry.empty()) {
    entrySym = handleUndefined(ctx.arg.entry, "--entry");
    if (entrySym && entrySym->isDefined())
      entrySym->forceExport = true;
    else
      error("entry symbol not defined (pass --no-entry to suppress): " +
            ctx.arg.entry);
  }

  // If the user code defines a `__wasm_call_dtors` function, remember it so
  // that we can call it from the command export wrappers. Unlike
  // `__wasm_call_ctors` which we synthesize, `__wasm_call_dtors` is defined
  // by libc/etc., because destructors are registered dynamically with
  // `__cxa_atexit` and friends.
  if (!ctx.arg.relocatable && !ctx.arg.shared &&
      !ctx.sym.callCtors->isUsedInRegularObj &&
      ctx.sym.callCtors->getName() != ctx.arg.entry &&
      !ctx.arg.exportedSymbols.count(ctx.sym.callCtors->getName())) {
    if (Symbol *callDtors =
            handleUndefined("__wasm_call_dtors", "<internal>")) {
      if (auto *callDtorsFunc = dyn_cast<DefinedFunction>(callDtors)) {
        if (callDtorsFunc->signature &&
            (!callDtorsFunc->signature->Params.empty() ||
             !callDtorsFunc->signature->Returns.empty())) {
          error("__wasm_call_dtors must have no argument or return values");
        }
        ctx.sym.callDtors = callDtorsFunc;
      } else {
        error("__wasm_call_dtors must be a function");
      }
    }
  }

  if (errorCount())
    return;

  // Create wrapped symbols for -wrap option.
  std::vector<WrappedSymbol> wrapped = addWrappedSymbols(args);

  // If any of our inputs are bitcode files, the LTO code generator may create
  // references to certain library functions that might not be explicit in the
  // bitcode file's symbol table. If any of those library functions are defined
  // in a bitcode file in an archive member, we need to arrange to use LTO to
  // compile those archive members by adding them to the link beforehand.
  //
  // We only need to add libcall symbols to the link before LTO if the symbol's
  // definition is in bitcode. Any other required libcall symbols will be added
  // to the link after LTO when we add the LTO object file to the link.
  if (!ctx.bitcodeFiles.empty()) {
    llvm::Triple TT(ctx.bitcodeFiles.front()->obj->getTargetTriple());
    for (auto *s : lto::LTO::getRuntimeLibcallSymbols(TT))
      handleLibcall(s);
  }
  if (errorCount())
    return;

  // We process the stub libraries once beofore LTO to ensure that any possible
  // required exports are preserved by the LTO process.
  processStubLibrariesPreLTO();

  // Do link-time optimization if given files are LLVM bitcode files.
  // This compiles bitcode files into real object files.
  symtab->compileBitcodeFiles();
  if (errorCount())
    return;

  // The LTO process can generate new undefined symbols, specifically libcall
  // functions.  Because those symbols might be declared in a stub library we
  // need the process the stub libraries once again after LTO to handle all
  // undefined symbols, including ones that didn't exist prior to LTO.
  processStubLibraries();

  writeWhyExtract();

  // Bail out if normal linked output is skipped due to LTO.
  if (ctx.arg.thinLTOIndexOnly)
    return;

  createOptionalSymbols();

  // Resolve any variant symbols that were created due to signature
  // mismatchs.
  symtab->handleSymbolVariants();
  if (errorCount())
    return;

  // Apply symbol renames for -wrap.
  if (!wrapped.empty())
    wrapSymbols(wrapped);

  for (auto &iter : ctx.arg.exportedSymbols) {
    Symbol *sym = symtab->find(iter.first());
    if (sym && sym->isDefined())
      sym->forceExport = true;
  }

  if (!ctx.arg.relocatable && !ctx.isPic) {
    // Add synthetic dummies for weak undefined functions.  Must happen
    // after LTO otherwise functions may not yet have signatures.
    symtab->handleWeakUndefines();
  }

  if (entrySym)
    entrySym->setHidden(false);

  if (errorCount())
    return;

  // Split WASM_SEG_FLAG_STRINGS sections into pieces in preparation for garbage
  // collection.
  splitSections();

  // Any remaining lazy symbols should be demoted to Undefined
  demoteLazySymbols();

  // Do size optimizations: garbage collection
  markLive();

  // Provide the indirect function table if needed.
  ctx.sym.indirectFunctionTable =
      symtab->resolveIndirectFunctionTable(/*required =*/false);

  if (errorCount())
    return;

  // Write the result to the file.
  writeResult();
}

} // namespace lld::wasm
