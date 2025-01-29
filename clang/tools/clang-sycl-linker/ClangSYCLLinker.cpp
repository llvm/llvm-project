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
// that include linking of llvm bitcode files, linking device library files
// with the fully linked source bitcode file(s), running several SYCL specific
// post-link steps on the fully linked bitcode file(s), and finally generating
// target-specific device code.
//===---------------------------------------------------------------------===//

#include "clang/Basic/Version.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::object;

/// Save intermediary results.
static bool SaveTemps = false;

/// Print arguments without executing.
static bool DryRun = false;

/// Print verbose output.
static bool Verbose = false;

/// Filename of the output being created.
static StringRef OutputFile;

/// Directory to dump SPIR-V IR if requested by user.
static SmallString<128> SPIRVDumpDir;

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
  // Collect all input bitcode files to be passed to llvm-link.
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
    // This will be extended to support objects and SPIR-V IR files.
    if (Magic != file_magic::bitcode)
      return createStringError("Unsupported file type");
    BitcodeFiles.push_back(*Filename);
  }
  return BitcodeFiles;
}

/// Link all SYCL device input files into one before adding device library
/// files. Device linking is performed using llvm-link tool.
/// 'InputFiles' is the list of all LLVM IR device input files.
/// 'Args' encompasses all arguments required for linking device code and will
/// be parsed to generate options required to be passed into llvm-link.
Expected<StringRef> linkDeviceInputFiles(ArrayRef<std::string> InputFiles,
                                         const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL LinkDeviceInputFiles");

  assert(InputFiles.size() && "No inputs to llvm-link");
  // Early check to see if there is only one input.
  if (InputFiles.size() < 2)
    return InputFiles[0];

  Expected<std::string> LLVMLinkPath =
      findProgram(Args, "llvm-link", {getMainExecutable("llvm-link")});
  if (!LLVMLinkPath)
    return LLVMLinkPath.takeError();

  SmallVector<StringRef> CmdArgs;
  CmdArgs.push_back(*LLVMLinkPath);
  for (auto &File : InputFiles)
    CmdArgs.push_back(File);
  // Create a new file to write the linked device file to.
  auto OutFileOrErr =
      createTempFile(Args, sys::path::filename(OutputFile), "bc");
  if (!OutFileOrErr)
    return OutFileOrErr.takeError();
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*OutFileOrErr);
  CmdArgs.push_back("--suppress-warnings");
  if (Error Err = executeCommands(*LLVMLinkPath, CmdArgs))
    return std::move(Err);
  return Args.MakeArgString(*OutFileOrErr);
}

// This utility function is used to gather all SYCL device library files that
// will be linked with input device files.
// The list of files and its location are passed from driver.
Expected<SmallVector<std::string>> getSYCLDeviceLibs(const ArgList &Args) {
  SmallVector<std::string> DeviceLibFiles;
  StringRef LibraryPath;
  if (Arg *A = Args.getLastArg(OPT_library_path_EQ))
    LibraryPath = A->getValue();
  if (LibraryPath.empty())
    return DeviceLibFiles;
  if (Arg *A = Args.getLastArg(OPT_device_libs_EQ)) {
    if (A->getValues().size() == 0)
      return createStringError(
          inconvertibleErrorCode(),
          "Number of device library files cannot be zero.");
    for (StringRef Val : A->getValues()) {
      SmallString<128> LibName(LibraryPath);
      llvm::sys::path::append(LibName, Val);
      if (llvm::sys::fs::exists(LibName))
        DeviceLibFiles.push_back(std::string(LibName));
      else
        return createStringError(inconvertibleErrorCode(),
                                 "\'" + std::string(LibName) + "\'" +
                                     " SYCL device library file is not found.");
    }
  }
  return DeviceLibFiles;
}

/// Link all device library files and input file into one LLVM IR file. This
/// linking is performed using llvm-link tool.
/// 'InputFiles' is the list of all LLVM IR device input files.
/// 'Args' encompasses all arguments required for linking device code and will
/// be parsed to generate options required to be passed into llvm-link tool.
static Expected<StringRef> linkDeviceLibFiles(StringRef InputFile,
                                              const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("LinkDeviceLibraryFiles");

  auto SYCLDeviceLibFiles = getSYCLDeviceLibs(Args);
  if (!SYCLDeviceLibFiles)
    return SYCLDeviceLibFiles.takeError();
  if ((*SYCLDeviceLibFiles).empty())
    return InputFile;

  Expected<std::string> LLVMLinkPath =
      findProgram(Args, "llvm-link", {getMainExecutable("llvm-link")});
  if (!LLVMLinkPath)
    return LLVMLinkPath.takeError();

  // Create a new file to write the linked device file to.
  auto OutFileOrErr =
      createTempFile(Args, sys::path::filename(OutputFile), "bc");
  if (!OutFileOrErr)
    return OutFileOrErr.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*LLVMLinkPath);
  CmdArgs.push_back("-only-needed");
  CmdArgs.push_back(InputFile);
  for (auto &File : *SYCLDeviceLibFiles)
    CmdArgs.push_back(File);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*OutFileOrErr);
  CmdArgs.push_back("--suppress-warnings");
  if (Error Err = executeCommands(*LLVMLinkPath, CmdArgs))
    return std::move(Err);
  return *OutFileOrErr;
}

/// Add any llvm-spirv option that relies on a specific Triple in addition
/// to user supplied options.
static void getSPIRVTransOpts(const ArgList &Args,
                              SmallVector<StringRef, 8> &TranslatorArgs,
                              const llvm::Triple Triple) {
  // Enable NonSemanticShaderDebugInfo.200 for non-Windows
  const bool IsWindowsMSVC =
      Triple.isWindowsMSVCEnvironment() || Args.hasArg(OPT_is_windows_msvc_env);
  const bool EnableNonSemanticDebug = !IsWindowsMSVC;
  if (EnableNonSemanticDebug) {
    TranslatorArgs.push_back(
        "-spirv-debug-info-version=nonsemantic-shader-200");
  } else {
    TranslatorArgs.push_back("-spirv-debug-info-version=ocl-100");
    // Prevent crash in the translator if input IR contains DIExpression
    // operations which don't have mapping to OpenCL.DebugInfo.100 spec.
    TranslatorArgs.push_back("-spirv-allow-extra-diexpressions");
  }
  std::string UnknownIntrinsics("-spirv-allow-unknown-intrinsics=llvm.genx.");

  TranslatorArgs.push_back(Args.MakeArgString(UnknownIntrinsics));

  // Disable all the extensions by default
  std::string ExtArg("-spirv-ext=-all");
  std::string DefaultExtArg =
      ",+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max"
      ",+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls"
      ",+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr";
  std::string INTELExtArg =
      ",+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io"
      ",+SPV_INTEL_device_side_avc_motion_estimation"
      ",+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_unstructured_loop_controls"
      ",+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes"
      ",+SPV_INTEL_function_pointers,+SPV_INTEL_kernel_attributes"
      ",+SPV_INTEL_io_pipes,+SPV_INTEL_inline_assembly"
      ",+SPV_INTEL_arbitrary_precision_integers"
      ",+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute"
      ",+SPV_INTEL_fast_composite"
      ",+SPV_INTEL_arbitrary_precision_fixed_point"
      ",+SPV_INTEL_arbitrary_precision_floating_point"
      ",+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode"
      ",+SPV_INTEL_long_constant_composite"
      ",+SPV_INTEL_arithmetic_fence"
      ",+SPV_INTEL_global_variable_decorations"
      ",+SPV_INTEL_cache_controls"
      ",+SPV_INTEL_fpga_buffer_location"
      ",+SPV_INTEL_fpga_argument_interfaces"
      ",+SPV_INTEL_fpga_invocation_pipelining_attributes"
      ",+SPV_INTEL_fpga_latency_control"
      ",+SPV_INTEL_task_sequence"
      ",+SPV_KHR_shader_clock"
      ",+SPV_INTEL_bindless_images";
  ExtArg = ExtArg + DefaultExtArg + INTELExtArg;
  ExtArg += ",+SPV_INTEL_token_type"
            ",+SPV_INTEL_bfloat16_conversion"
            ",+SPV_INTEL_joint_matrix"
            ",+SPV_INTEL_hw_thread_queries"
            ",+SPV_KHR_uniform_group_instructions"
            ",+SPV_INTEL_masked_gather_scatter"
            ",+SPV_INTEL_tensor_float32_conversion"
            ",+SPV_INTEL_optnone"
            ",+SPV_KHR_non_semantic_info"
            ",+SPV_KHR_cooperative_matrix";
  TranslatorArgs.push_back(Args.MakeArgString(ExtArg));
}

/// Run LLVM to SPIR-V translation.
/// Converts 'File' from LLVM bitcode to SPIR-V format using llvm-spirv tool.
/// 'Args' encompasses all arguments required for linking device code and will
/// be parsed to generate options required to be passed into llvm-spirv tool.
static Expected<StringRef> runLLVMToSPIRVTranslation(StringRef File,
                                                     const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("LLVMToSPIRVTranslation");
  StringRef LLVMSPIRVPath = Args.getLastArgValue(OPT_llvm_spirv_path_EQ);
  Expected<std::string> LLVMToSPIRVProg =
      findProgram(Args, "llvm-spirv", {LLVMSPIRVPath});
  if (!LLVMToSPIRVProg)
    return LLVMToSPIRVProg.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*LLVMToSPIRVProg);
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple));
  getSPIRVTransOpts(Args, CmdArgs, Triple);
  StringRef LLVMToSPIRVOptions;
  if (Arg *A = Args.getLastArg(OPT_llvm_spirv_options_EQ))
    LLVMToSPIRVOptions = A->getValue();
  LLVMToSPIRVOptions.split(CmdArgs, " ", /* MaxSplit = */ -1,
                           /* KeepEmpty = */ false);
  CmdArgs.append({"-o", OutputFile});
  CmdArgs.push_back(File);
  if (Error Err = executeCommands(*LLVMToSPIRVProg, CmdArgs))
    return std::move(Err);

  if (!SPIRVDumpDir.empty()) {
    std::error_code EC =
        llvm::sys::fs::create_directory(SPIRVDumpDir, /*IgnoreExisting*/ true);
    if (EC)
      return createStringError(
          EC,
          formatv("failed to create dump directory. path: {0}, error_code: {1}",
                  SPIRVDumpDir, EC.value()));

    StringRef Path = OutputFile;
    StringRef Filename = llvm::sys::path::filename(Path);
    SmallString<128> CopyPath = SPIRVDumpDir;
    CopyPath.append(Filename);
    EC = llvm::sys::fs::copy_file(Path, CopyPath);
    if (EC)
      return createStringError(
          EC,
          formatv(
              "failed to copy file. original: {0}, copy: {1}, error_code: {2}",
              Path, CopyPath, EC.value()));
  }

  return OutputFile;
}

Error runSYCLLink(ArrayRef<std::string> Files, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCLDeviceLink");
  // First llvm-link step
  auto LinkedFile = linkDeviceInputFiles(Files, Args);
  if (!LinkedFile)
    reportError(LinkedFile.takeError());

  // second llvm-link step
  auto DeviceLinkedFile = linkDeviceLibFiles(*LinkedFile, Args);
  if (!DeviceLinkedFile)
    reportError(DeviceLinkedFile.takeError());

  // LLVM to SPIR-V translation step
  auto SPVFile = runLLVMToSPIRVTranslation(*DeviceLinkedFile, Args);
  if (!SPVFile)
    return SPVFile.takeError();
  return Error::success();
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  Executable = argv[0];
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  const OptTable &Tbl = getOptTable();
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  auto Args = Tbl.parseArgs(argc, argv, OPT_INVALID, Saver, [&](StringRef Err) {
    reportError(createStringError(inconvertibleErrorCode(), Err));
  });

  if (Args.hasArg(OPT_help) || Args.hasArg(OPT_help_hidden)) {
    Tbl.printHelp(
        outs(), "clang-sycl-linker [options] <options to sycl link steps>",
        "A utility that wraps around several steps required to link SYCL "
        "device files.\n"
        "This enables LLVM IR linking, post-linking and code generation for "
        "SYCL targets.",
        Args.hasArg(OPT_help_hidden), Args.hasArg(OPT_help_hidden));
    return EXIT_SUCCESS;
  }

  if (Args.hasArg(OPT_version))
    printVersion(outs());

  Verbose = Args.hasArg(OPT_verbose);
  DryRun = Args.hasArg(OPT_dry_run);
  SaveTemps = Args.hasArg(OPT_save_temps);

  OutputFile = "a.spv";
  if (Args.hasArg(OPT_o))
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
