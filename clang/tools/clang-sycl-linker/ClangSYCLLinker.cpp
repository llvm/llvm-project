//=-------- clang-sycl-linker/ClangSYCLLinker.cpp - SYCL Linker util -------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This tool executes a sequence of steps required to link device code in SYCL
// device images. SYCL device code linking requires a complex sequence of steps
// that include linking of llvm bitcode files, linking bitcode library files
// with the fully linked source bitcode file(s), running several SYCL specific
// post-link steps on the fully linked bitcode file(s), and finally generating
// target-specific device code.
//
//===---------------------------------------------------------------------===//

#include "clang/Basic/OffloadArch.h"
#include "clang/Basic/Version.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
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

/// Print commands/steps with arguments without executing.
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

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "SYCLLinkOpts.inc"
#undef OPTION
};

class LinkerOptTable : public opt::GenericOptTable {
public:
  LinkerOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};

const OptTable &getOptTable() {
  static const LinkerOptTable *Table = []() {
    auto Result = std::make_unique<LinkerOptTable>();
    return Result.release();
  }();
  return *Table;
}

[[noreturn]] void reportError(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), Executable));
  exit(EXIT_FAILURE);
}

std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

Expected<StringRef> createTempFile(const ArgList &Args, const Twine &Prefix,
                                   StringRef Extension) {
  SmallString<128> OutputFile;
  if (Args.hasArg(OPT_save_temps)) {
    // Generate a unique path name without creating a file
    sys::fs::createUniquePath(Prefix + "-%%%%%%." + Extension, OutputFile,
                              /*MakeAbsolute=*/false);
  } else {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(Prefix, Extension, OutputFile))
      return createFileError(OutputFile, EC);
  }

  TempFiles.emplace_back(std::move(OutputFile));
  return TempFiles.back();
}

Expected<std::string> findProgram(const ArgList &Args, StringRef Name,
                                  ArrayRef<StringRef> Paths) {
  if (Args.hasArg(OPT_dry_run))
    return Name.str();
  ErrorOr<std::string> Path = sys::findProgramByName(Name, Paths);
  if (!Path)
    Path = sys::findProgramByName(Name);
  if (!Path)
    return createStringError(Path.getError(),
                             "Unable to find '" + Name + "' in path");
  return *Path;
}

void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  llvm::errs() << llvm::join(std::next(CmdArgs.begin()), CmdArgs.end(), " ")
               << "\n";
}

/// Execute the command \p ExecutablePath with the arguments \p Args.
Error executeCommands(StringRef ExecutablePath, ArrayRef<StringRef> Args) {
  if (Verbose || DryRun)
    printCommands(Args);

  if (!DryRun)
    if (sys::ExecuteAndWait(ExecutablePath, Args))
      return createStringError(
          "'%s' failed", sys::path::filename(ExecutablePath).str().c_str());
  return Error::success();
}

Expected<SmallVector<std::string>> getInput(const ArgList &Args) {
  // Collect all input bitcode files to be passed to the linking stage.
  SmallVector<std::string> BitcodeFiles;
  for (const opt::Arg *Arg : Args.filtered(OPT_INPUT)) {
    std::optional<std::string> Filename = std::string(Arg->getValue());
    if (!Filename || !sys::fs::exists(*Filename) ||
        sys::fs::is_directory(*Filename))
      continue;
    file_magic Magic;
    if (auto EC = identify_magic(*Filename, Magic))
      return createStringError("Failed to open file " + *Filename);
    // TODO: Current use case involves LLVM IR bitcode files as input.
    // This will be extended to support SPIR-V IR files.
    if (Magic != file_magic::bitcode)
      return createStringError("Unsupported file type");
    BitcodeFiles.push_back(*Filename);
  }
  return BitcodeFiles;
}

/// Handle cases where input file is a LLVM IR bitcode file.
/// When clang-sycl-linker is called via clang-linker-wrapper tool, input files
/// are LLVM IR bitcode files.
// TODO: Support SPIR-V IR files.
Expected<std::unique_ptr<Module>> getBitcodeModule(StringRef File,
                                                   LLVMContext &C) {
  SMDiagnostic Err;

  auto M = getLazyIRFileModule(File, Err, C);
  if (M)
    return std::move(M);
  return createStringError(Err.getMessage());
}

std::optional<std::string> findFile(StringRef Dir, const Twine &Name) {
  SmallString<128> Path(Dir);
  llvm::sys::path::append(Path, Name);
  if (sys::fs::exists(Path) && !sys::fs::is_directory(Path))
    return std::string(Path);
  return std::nullopt;
}

std::optional<std::string> searchLibrary(StringRef Name,
                                         ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (std::optional<std::string> File = findFile(Dir, Name))
      return File;
  return std::nullopt;
}

/// Gather all library files. The list of files and its location are passed from
/// driver.
Expected<SmallVector<std::string>> getBCLibraryNames(const ArgList &Args) {
  SmallVector<StringRef> LibraryPaths;
  for (const opt::Arg *Arg : Args.filtered(OPT_library_path))
    LibraryPaths.push_back(Arg->getValue());

  SmallVector<std::string> LibraryFiles;
  for (const opt::Arg *Arg : Args.filtered(OPT_bc_library)) {
    std::optional<std::string> LibName =
        searchLibrary(Arg->getValue(), LibraryPaths);
    if (!LibName)
      return createStringError("'" + Twine(Arg->getValue()) +
                               "' library file not found");
    LibraryFiles.push_back(std::move(*LibName));
  }

  return LibraryFiles;
}

struct LinkResult {
  std::unique_ptr<Module> LinkedModule;
  SmallString<256> BitcodeFile;
  llvm::Triple TargetTriple;
};

/// Following tasks are performed:
/// 1. Resolve the target triple: use --triple= when given, otherwise take the
/// first input that supplies a triple as canonical. Issue an error if any
/// triple inputs disagree.
/// 2. Link all input bitcode images into one image using the linkInModule API.
/// 3. Gather all library bitcode images.
/// 4. Link all the images gathered in Step 3 with the output of Step 2 using
/// linkInModule API. LinkOnlyNeeded flag is used.
Expected<LinkResult> linkInputs(ArrayRef<std::string> InputFiles,
                                const ArgList &Args, LLVMContext &C) {
  llvm::TimeTraceScope TimeScope("Link code");

  assert(InputFiles.size() && "No inputs to link");

  // Get all library files.
  Expected<SmallVector<std::string>> BCLibFiles = getBCLibraryNames(Args);
  if (!BCLibFiles)
    return BCLibFiles.takeError();

  // Create a new file to write the linked file to.
  auto BitcodeOutput =
      createTempFile(Args, sys::path::filename(OutputFile), "bc");
  if (!BitcodeOutput)
    return BitcodeOutput.takeError();

  if (Verbose || DryRun) {
    std::string Inputs = llvm::join(InputFiles.begin(), InputFiles.end(), ", ");
    std::string LibInputs =
        llvm::join((*BCLibFiles).begin(), (*BCLibFiles).end(), ", ");
    errs() << formatv("link: inputs: {0} libfiles: {1} output: {2}\n", Inputs,
                      LibInputs, *BitcodeOutput);
  }

  // Link input files. Resolve the target triple.
  llvm::Triple TargetTriple(Args.getLastArgValue(OPT_triple_EQ));
  StringRef TripleSource = TargetTriple.empty() ? "" : "--triple=";
  auto LinkerOutput = std::make_unique<Module>("linker-output", C);
  Linker L(*LinkerOutput);

  for (auto &File : InputFiles) {
    auto ModOrErr = getBitcodeModule(File, C);
    if (!ModOrErr)
      return ModOrErr.takeError();

    const llvm::Triple &T = (*ModOrErr)->getTargetTriple();
    if (!T.empty() && T != TargetTriple) {
      if (TargetTriple.empty()) {
        TargetTriple = T;
        TripleSource = File;
      } else {
        return createStringError(
            "conflicting target triples: '" + TargetTriple.str() + "' (from " +
            TripleSource + ") vs '" + T.str() + "' (from " + File + ")");
      }
    }

    if (L.linkInModule(std::move(*ModOrErr)))
      return createStringError("Could not link IR");
  }

  if (TargetTriple.empty())
    return createStringError(
        "Target triple must be specified or inferable from inputs");

  // Link in library files.
  for (auto &File : *BCLibFiles) {
    auto LibMod = getBitcodeModule(File, C);
    if (!LibMod)
      return LibMod.takeError();
    if ((*LibMod)->getTargetTriple() == TargetTriple) {
      unsigned Flags = Linker::Flags::LinkOnlyNeeded;
      if (L.linkInModule(std::move(*LibMod), Flags))
        return createStringError("Could not link IR");
    }
  }

  // Dump linked output for testing.
  if (Args.hasArg(OPT_print_linked_module))
    outs() << *LinkerOutput;

  // Write the final output into 'BitcodeOutput' file.
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(*BitcodeOutput, FD))
    return errorCodeToError(EC);
  llvm::raw_fd_ostream OS(FD, true);
  WriteBitcodeToFile(*LinkerOutput, OS);

  return LinkResult{std::move(LinkerOutput), SmallString<256>(*BitcodeOutput),
                    std::move(TargetTriple)};
}

/// Run Code Generation using LLVM backend.
/// \param 'File' The input LLVM IR bitcode file.
/// \param 'TargetTriple' The resolved target triple.
/// \param 'Args' encompasses all arguments required for linking device code and
/// will be parsed to generate options required to be passed into the backend.
/// \param 'OutputFile' The output file name.
/// \param 'C' The LLVM context.
static Error runCodeGen(StringRef File, const llvm::Triple &TargetTriple,
                        const ArgList &Args, StringRef OutputFile,
                        LLVMContext &C) {
  llvm::TimeTraceScope TimeScope("Code generation");

  if (Verbose || DryRun)
    errs() << formatv("LLVM backend: input: {0}, output: {1}\n", File,
                      OutputFile);

  // Parse input module.
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(File, Err, C);
  if (!M)
    return createStringError(Err.getMessage());

  if (Error Err = M->materializeAll())
    return Err;

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
      T->createTargetMachine(M->getTargetTriple(), /* CPU */ "",
                             /* Features */ "", Options, RM, CM));
  if (!TM)
    return createStringError("Could not allocate target machine!");

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
    return createStringError("Failed to execute LLVM backend");
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

/// SYCL device code module split mode.
enum class IRSplitMode {
  SPLIT_PER_TU,     // one module per translation unit
  SPLIT_PER_KERNEL, // one module per kernel
  SPLIT_NONE        // no splitting
};

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

/// Result of splitting a device module: the bitcode file path and the
/// serialized symbol table for each device image.
struct SplitModule {
  SmallString<256> ModuleFilePath;
  SmallString<0> Symbols;
};

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

    int FD = -1;
    if (std::error_code EC = sys::fs::openFileForWrite(*BitcodeFileOrErr, FD))
      return errorCodeToError(EC);
    raw_fd_ostream OS(FD, /*shouldClose=*/true);
    WriteBitcodeToFile(*Part, OS);

    SplitModules.push_back(
        {SmallString<256>(*BitcodeFileOrErr),
         collectEntryPoints(*Part, EmitOnlyKernelsAsEntryPoints)});
    return Error::success();
  };

  if (Error Err = splitModuleTransitiveFromEntryPoints(
          std::move(M), Categorizer, SplitCallback))
    return Err;

  if (Verbose || DryRun) {
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
Error runSYCLLink(ArrayRef<std::string> Files, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL linking");

  LLVMContext C;

  // Link all input bitcode files and library files.
  Expected<LinkResult> LinkedOrErr = linkInputs(Files, Args, C);
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
        llvm::MemoryBuffer::getFileOrSTDIN(SI.ModuleFilePath);
    if (std::error_code EC = FileOrErr.getError()) {
      if (DryRun)
        FileOrErr = MemoryBuffer::getMemBuffer("");
      else
        return createFileError(SI.ModuleFilePath, EC);
    }
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

  llvm::SmallString<0> Buffer = OffloadBinary::write(Images);
  if (Buffer.size() % OffloadBinary::getAlignment() != 0)
    return createStringError("Offload binary has invalid size alignment");

  auto OutputOrErr = FileOutputBuffer::create(OutputFile, Buffer.size());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  llvm::copy(Buffer, (*OutputOrErr)->getBufferStart());
  return (*OutputOrErr)->commit();
}

} // namespace

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

  if (Args.hasArg(OPT_version))
    printVersion(outs());

  Verbose = Args.hasArg(OPT_verbose);
  DryRun = Args.hasArg(OPT_dry_run);

  if (!Args.hasArg(OPT_o))
    reportError(createStringError("Output file must be specified"));
  OutputFile = Args.getLastArgValue(OPT_o);

  if (Args.hasArg(OPT_spirv_dump_device_code_EQ)) {
    Arg *A = Args.getLastArg(OPT_spirv_dump_device_code_EQ);
    SmallString<128> Dir(A->getValue());
    if (Dir.empty())
      llvm::sys::path::native(Dir = "./");
    else
      Dir.append(llvm::sys::path::get_separator());

    SPIRVDumpDir = Dir;
  }

  // Get the input files to pass to the linking stage.
  auto FilesOrErr = getInput(Args);
  if (!FilesOrErr)
    reportError(FilesOrErr.takeError());

  // Run SYCL linking process on the generated inputs.
  if (Error Err = runSYCLLink(*FilesOrErr, Args))
    reportError(std::move(Err));

  // Remove the temporary files created.
  if (!Args.hasArg(OPT_save_temps))
    for (const auto &TempFile : TempFiles)
      if (std::error_code EC = sys::fs::remove(TempFile))
        reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}
