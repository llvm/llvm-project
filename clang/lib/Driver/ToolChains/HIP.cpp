//===--- HIP.cpp - HIP Tool and ToolChain Implementations -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIP.h"
#include "AMDGPU.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/TargetID.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetParser.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

#if defined(_WIN32) || defined(_WIN64)
#define NULL_FILE "nul"
#else
#define NULL_FILE "/dev/null"
#endif

namespace {
const unsigned HIPCodeObjectAlign = 4096;

static void addBCLib(const Driver &D, const ArgList &Args,
                     ArgStringList &CmdArgs, ArgStringList LibraryPaths,
                     StringRef BCName) {
  StringRef FullName;
  for (std::string LibraryPath : LibraryPaths) {
    SmallString<128> Path(LibraryPath);
    llvm::sys::path::append(Path, BCName);
    FullName = Path;
    if (llvm::sys::fs::exists(FullName)) {
      CmdArgs.push_back("-mlink-builtin-bitcode");
      CmdArgs.push_back(Args.MakeArgString(FullName));
      return;
    }
  }
  D.Diag(diag::err_drv_no_such_file) << BCName;
}
} // namespace

void AMDGCN::Linker::constructLldCommand(Compilation &C, const JobAction &JA,
                                          const InputInfoList &Inputs,
                                          const InputInfo &Output,
                                          const llvm::opt::ArgList &Args) const {
  // Construct lld command.
  // The output from ld.lld is an HSA code object file.
  ArgStringList LldArgs{"-flavor", "gnu", "--no-undefined", "-shared",
                        "-plugin-opt=-amdgpu-internalize-symbols"};

  auto &TC = getToolChain();
  auto &D = TC.getDriver();
  assert(!Inputs.empty() && "Must have at least one input.");
  addLTOOptions(TC, Args, LldArgs, Output, Inputs[0],
                D.getLTOMode() == LTOK_Thin);

  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  amdgpu::getAMDGPUTargetFeatures(D, TC.getTriple(), Args, Features);

  // Add features to mattr such as cumode
  std::string MAttrString = "-plugin-opt=-mattr=";
  for (auto OneFeature : unifyTargetFeatures(Features)) {
    MAttrString.append(Args.MakeArgString(OneFeature));
    if (OneFeature != Features.back())
      MAttrString.append(",");
  }
  if (!Features.empty())
    LldArgs.push_back(Args.MakeArgString(MAttrString));

  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    LldArgs.push_back(
        Args.MakeArgString(Twine("-plugin-opt=") + A->getValue(0)));
  }

  if (C.getDriver().isSaveTempsEnabled())
    LldArgs.push_back("-save-temps");

  addLinkerCompressDebugSectionsOption(TC, Args, LldArgs);

  LldArgs.append({"-o", Output.getFilename()});
  for (auto Input : Inputs)
    LldArgs.push_back(Input.getFilename());
  const char *Lld = Args.MakeArgString(getToolChain().GetProgramPath("lld"));
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Lld, LldArgs, Inputs));
}

// Construct a clang-offload-bundler command to bundle code objects for
// different GPU's into a HIP fat binary.
void AMDGCN::constructHIPFatbinCommand(Compilation &C, const JobAction &JA,
                  StringRef OutputFileName, const InputInfoList &Inputs,
                  const llvm::opt::ArgList &Args, const Tool& T) {
  // Construct clang-offload-bundler command to bundle object files for
  // for different GPU archs.
  ArgStringList BundlerArgs;
  BundlerArgs.push_back(Args.MakeArgString("-type=o"));
  BundlerArgs.push_back(
      Args.MakeArgString("-bundle-align=" + Twine(HIPCodeObjectAlign)));

  // ToDo: Remove the dummy host binary entry which is required by
  // clang-offload-bundler.
  std::string BundlerTargetArg = "-targets=host-x86_64-unknown-linux";
  std::string BundlerInputArg = "-inputs=" NULL_FILE;

  for (const auto &II : Inputs) {
    const auto* A = II.getAction();
    BundlerTargetArg = BundlerTargetArg + ",hip-amdgcn-amd-amdhsa-" +
                       StringRef(A->getOffloadingArch()).str();
    BundlerInputArg = BundlerInputArg + "," + II.getFilename();
  }
  BundlerArgs.push_back(Args.MakeArgString(BundlerTargetArg));
  BundlerArgs.push_back(Args.MakeArgString(BundlerInputArg));

  auto BundlerOutputArg = Args.MakeArgString(
      std::string("-outputs=").append(std::string(OutputFileName)));
  BundlerArgs.push_back(BundlerOutputArg);

  const char *Bundler = Args.MakeArgString(
      T.getToolChain().GetProgramPath("clang-offload-bundler"));
  C.addCommand(std::make_unique<Command>(JA, T, ResponseFileSupport::None(),
                                         Bundler, BundlerArgs, Inputs));
}

/// Add Generated HIP Object File which has device images embedded into the
/// host to the argument list for linking. Using MC directives, embed the
/// device code and also define symbols required by the code generation so that
/// the image can be retrieved at runtime.
void AMDGCN::Linker::constructGenerateObjFileFromHIPFatBinary(
    Compilation &C, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args,
    const JobAction &JA) const {
  const ToolChain &TC = getToolChain();
  std::string Name =
      std::string(llvm::sys::path::stem(Output.getFilename()));

  // Create Temp Object File Generator,
  // Offload Bundled file and Bundled Object file.
  // Keep them if save-temps is enabled.
  const char *McinFile;
  const char *BundleFile;
  if (C.getDriver().isSaveTempsEnabled()) {
    McinFile = C.getArgs().MakeArgString(Name + ".mcin");
    BundleFile = C.getArgs().MakeArgString(Name + ".hipfb");
  } else {
    auto TmpNameMcin = C.getDriver().GetTemporaryPath(Name, "mcin");
    McinFile = C.addTempFile(C.getArgs().MakeArgString(TmpNameMcin));
    auto TmpNameFb = C.getDriver().GetTemporaryPath(Name, "hipfb");
    BundleFile = C.addTempFile(C.getArgs().MakeArgString(TmpNameFb));
  }
  constructHIPFatbinCommand(C, JA, BundleFile, Inputs, Args, *this);

  // Create a buffer to write the contents of the temp obj generator.
  std::string ObjBuffer;
  llvm::raw_string_ostream ObjStream(ObjBuffer);

  // Add MC directives to embed target binaries. We ensure that each
  // section and image is 16-byte aligned. This is not mandatory, but
  // increases the likelihood of data to be aligned with a cache block
  // in several main host machines.
  ObjStream << "#       HIP Object Generator\n";
  ObjStream << "# *** Automatically generated by Clang ***\n";
  ObjStream << "  .type __hip_fatbin,@object\n";
  ObjStream << "  .section .hip_fatbin,\"aMS\",@progbits,1\n";
  ObjStream << "  .data\n";
  ObjStream << "  .globl __hip_fatbin\n";
  ObjStream << "  .p2align " << llvm::Log2(llvm::Align(HIPCodeObjectAlign))
            << "\n";
  ObjStream << "__hip_fatbin:\n";
  ObjStream << "  .incbin \"" << BundleFile << "\"\n";
  ObjStream.flush();

  // Dump the contents of the temp object file gen if the user requested that.
  // We support this option to enable testing of behavior with -###.
  if (C.getArgs().hasArg(options::OPT_fhip_dump_offload_linker_script))
    llvm::errs() << ObjBuffer;

  // Open script file and write the contents.
  std::error_code EC;
  llvm::raw_fd_ostream Objf(McinFile, EC, llvm::sys::fs::OF_None);

  if (EC) {
    C.getDriver().Diag(clang::diag::err_unable_to_make_temp) << EC.message();
    return;
  }

  Objf << ObjBuffer;

  ArgStringList McArgs{"-o",      Output.getFilename(),
                       McinFile,  "--filetype=obj"};
  const char *Mc = Args.MakeArgString(TC.GetProgramPath("llvm-mc"));
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Mc, McArgs, Inputs));
}

// For amdgcn the inputs of the linker job are device bitcode and output is
// object file. It calls llvm-link, opt, llc, then lld steps.
void AMDGCN::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  if (Inputs.size() > 0 &&
      Inputs[0].getType() == types::TY_Image &&
      JA.getType() == types::TY_Object)
    return constructGenerateObjFileFromHIPFatBinary(C, Output, Inputs, Args, JA);

  if (JA.getType() == types::TY_HIP_FATBIN)
    return constructHIPFatbinCommand(C, JA, Output.getFilename(), Inputs, Args, *this);

  return constructLldCommand(C, JA, Inputs, Output, Args);
}

HIPToolChain::HIPToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ROCMToolChain(D, Triple, Args), HostTC(HostTC) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void HIPToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  StringRef GpuArch = getGPUArch(DriverArgs);
  assert(!GpuArch.empty() && "Must have an explicit GPU arch.");
  (void) GpuArch;
  assert(DeviceOffloadingKind == Action::OFK_HIP &&
         "Only HIP offloading kinds are supported for GPUs.");
  auto Kind = llvm::AMDGPU::parseArchAMDGCN(GpuArch);
  const StringRef CanonArch = llvm::AMDGPU::getArchNameAMDGCN(Kind);

  CC1Args.push_back("-fcuda-is-device");

  if (DriverArgs.hasFlag(options::OPT_fcuda_approx_transcendentals,
                         options::OPT_fno_cuda_approx_transcendentals, false))
    CC1Args.push_back("-fcuda-approx-transcendentals");

  if (DriverArgs.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                         false))
    CC1Args.push_back("-fgpu-rdc");
  else
    CC1Args.append({"-mllvm", "-amdgpu-internalize-symbols"});

  StringRef MaxThreadsPerBlock =
      DriverArgs.getLastArgValue(options::OPT_gpu_max_threads_per_block_EQ);
  if (!MaxThreadsPerBlock.empty()) {
    std::string ArgStr =
        std::string("--gpu-max-threads-per-block=") + MaxThreadsPerBlock.str();
    CC1Args.push_back(DriverArgs.MakeArgStringRef(ArgStr));
  }

  if (DriverArgs.hasFlag(options::OPT_fgpu_allow_device_init,
                         options::OPT_fno_gpu_allow_device_init, false))
    CC1Args.push_back("-fgpu-allow-device-init");

  CC1Args.push_back("-fcuda-allow-variadic-functions");

  // Default to "hidden" visibility, as object level linking will not be
  // supported for the foreseeable future.
  if (!DriverArgs.hasArg(options::OPT_fvisibility_EQ,
                         options::OPT_fvisibility_ms_compat)) {
    CC1Args.append({"-fvisibility", "hidden"});
    CC1Args.push_back("-fapply-global-visibility-to-externs");
  }

  if (DriverArgs.hasArg(options::OPT_nogpulib))
    return;
  ArgStringList LibraryPaths;

  // Find in --hip-device-lib-path and HIP_LIBRARY_PATH.
  for (auto Path : RocmInstallation.getRocmDeviceLibPathArg())
    LibraryPaths.push_back(DriverArgs.MakeArgString(Path));

  addDirectoryList(DriverArgs, LibraryPaths, "", "HIP_DEVICE_LIB_PATH");

  // Maintain compatability with --hip-device-lib.
  auto BCLibs = DriverArgs.getAllArgValues(options::OPT_hip_device_lib_EQ);
  if (!BCLibs.empty()) {
    for (auto Lib : BCLibs)
      addBCLib(getDriver(), DriverArgs, CC1Args, LibraryPaths, Lib);
  } else {
    if (!RocmInstallation.hasDeviceLibrary()) {
      getDriver().Diag(diag::err_drv_no_rocm_device_lib) << 0;
      return;
    }

    std::string LibDeviceFile = RocmInstallation.getLibDeviceFile(CanonArch);
    if (LibDeviceFile.empty()) {
      getDriver().Diag(diag::err_drv_no_rocm_device_lib) << 1 << GpuArch;
      return;
    }

    // If --hip-device-lib is not set, add the default bitcode libraries.
    // TODO: There are way too many flags that change this. Do we need to check
    // them all?
    bool DAZ = DriverArgs.hasFlag(options::OPT_fcuda_flush_denormals_to_zero,
                                  options::OPT_fno_cuda_flush_denormals_to_zero,
                                  getDefaultDenormsAreZeroForTarget(Kind));
    // TODO: Check standard C++ flags?
    bool FiniteOnly = false;
    bool UnsafeMathOpt = false;
    bool FastRelaxedMath = false;
    bool CorrectSqrt = true;
    bool Wave64 = isWave64(DriverArgs, Kind);

    // Add the HIP specific bitcode library.
    CC1Args.push_back("-mlink-builtin-bitcode");
    CC1Args.push_back(DriverArgs.MakeArgString(RocmInstallation.getHIPPath()));

    // Add the generic set of libraries.
    RocmInstallation.addCommonBitcodeLibCC1Args(
      DriverArgs, CC1Args, LibDeviceFile, Wave64, DAZ, FiniteOnly,
      UnsafeMathOpt, FastRelaxedMath, CorrectSqrt);

    // Add instrument lib.
    auto InstLib =
        DriverArgs.getLastArgValue(options::OPT_gpu_instrument_lib_EQ);
    if (InstLib.empty())
      return;
    if (llvm::sys::fs::exists(InstLib)) {
      CC1Args.push_back("-mlink-builtin-bitcode");
      CC1Args.push_back(DriverArgs.MakeArgString(InstLib));
    } else
      getDriver().Diag(diag::err_drv_no_such_file) << InstLib;
  }
}

llvm::opt::DerivedArgList *
HIPToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  for (Arg *A : Args) {
    if (!shouldSkipArgument(A))
      DAL->append(A);
  }

  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_mcpu_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_mcpu_EQ), BoundArch);
    checkTargetID(*DAL);
  }

  return DAL;
}

Tool *HIPToolChain::buildLinker() const {
  assert(getTriple().getArch() == llvm::Triple::amdgcn);
  return new tools::AMDGCN::Linker(*this);
}

void HIPToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
HIPToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void HIPToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void HIPToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

void HIPToolChain::AddIAMCUIncludeArgs(const ArgList &Args,
                                        ArgStringList &CC1Args) const {
  HostTC.AddIAMCUIncludeArgs(Args, CC1Args);
}

void HIPToolChain::AddHIPIncludeArgs(const ArgList &DriverArgs,
                                     ArgStringList &CC1Args) const {
  RocmInstallation.AddHIPIncludeArgs(DriverArgs, CC1Args);
}

SanitizerMask HIPToolChain::getSupportedSanitizers() const {
  // The HIPToolChain only supports sanitizers in the sense that it allows
  // sanitizer arguments on the command line if they are supported by the host
  // toolchain. The HIPToolChain will actually ignore any command line
  // arguments for any of these "supported" sanitizers. That means that no
  // sanitization of device code is actually supported at this time.
  //
  // This behavior is necessary because the host and device toolchains
  // invocations often share the command line, so the device toolchain must
  // tolerate flags meant only for the host toolchain.
  return HostTC.getSupportedSanitizers();
}

VersionTuple HIPToolChain::computeMSVCVersion(const Driver *D,
                                               const ArgList &Args) const {
  return HostTC.computeMSVCVersion(D, Args);
}
