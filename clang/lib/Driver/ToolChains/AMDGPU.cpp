//===--- AMDGPU.cpp - AMDGPU ToolChain Implementations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

RocmInstallationDetector::RocmInstallationDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args)
    : D(D) {
  struct Candidate {
    std::string Path;
    bool StrictChecking;

    Candidate(std::string Path, bool StrictChecking = false)
        : Path(Path), StrictChecking(StrictChecking) {}
  };

  SmallVector<Candidate, 4> Candidates;

  if (Args.hasArg(clang::driver::options::OPT_rocm_path_EQ)) {
    Candidates.emplace_back(
        Args.getLastArgValue(clang::driver::options::OPT_rocm_path_EQ).str());
  } else {
    // Try to find relative to the compiler binary.
    const char *InstallDir = D.getInstalledDir();

    // Check both a normal Unix prefix position of the clang binary, as well as
    // the Windows-esque layout the ROCm packages use with the host architecture
    // subdirectory of bin.

    StringRef ParentDir = llvm::sys::path::parent_path(InstallDir);
    if (ParentDir == HostTriple.getArchName())
      ParentDir = llvm::sys::path::parent_path(ParentDir);

    if (ParentDir == "bin") {
      Candidates.emplace_back(llvm::sys::path::parent_path(ParentDir).str(),
                              /*StrictChecking=*/true);
    }

    Candidates.emplace_back(D.SysRoot + "/opt/rocm");
  }

  bool NoBuiltinLibs = Args.hasArg(options::OPT_nogpulib);

  for (const auto &Candidate : Candidates) {
    InstallPath = Candidate.Path;
    if (InstallPath.empty() || !D.getVFS().exists(InstallPath))
      continue;

    // FIXME: The install path situation is a real mess.

    // For a cmake install, these are placed directly in
    // ${INSTALL_PREFIX}/lib

    // In the separate OpenCL builds, the bitcode libraries are placed in
    // ${OPENCL_ROOT}/lib/x86_64/bitcode/*

    // For the rocm installed packages, these are placed at
    // /opt/rocm/opencl/lib/x86_64/bitcode

    // An additional copy is installed, in scattered locations between
    // /opt/rocm/hcc/rocdl/oclc
    // /opt/rocm/hcc/rocdl/ockl
    // /opt/rocm/hcc/rocdl/lib
    //
    // Yet another complete set is installed to
    // /opt/rocm/hcc/rocdl/lib

    // For now just recognize the opencl package layout.

    // BinPath = InstallPath + "/bin";
    llvm::sys::path::append(IncludePath, InstallPath, "include");
    llvm::sys::path::append(LibDevicePath, InstallPath, "lib");

    auto &FS = D.getVFS();

    // We don't need the include path for OpenCL, since clang already ships with
    // the default header.

    bool CheckLibDevice = (!NoBuiltinLibs || Candidate.StrictChecking);
    if (CheckLibDevice && !FS.exists(LibDevicePath))
      continue;

    const StringRef Suffix(".amdgcn.bc");

    std::error_code EC;
    for (llvm::sys::fs::directory_iterator LI(LibDevicePath, EC), LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef FilePath = LI->path();
      StringRef FileName = llvm::sys::path::filename(FilePath);
      if (!FileName.endswith(Suffix))
        continue;

      StringRef BaseName = FileName.drop_back(Suffix.size());

      if (BaseName == "ocml") {
        OCML = FilePath;
      } else if (BaseName == "ockl") {
        OCKL = FilePath;
      } else if (BaseName == "opencl") {
        OpenCL = FilePath;
      } else if (BaseName == "hip") {
        HIP = FilePath;
      } else if (BaseName == "oclc_finite_only_off") {
        FiniteOnly.Off = FilePath;
      } else if (BaseName == "oclc_finite_only_on") {
        FiniteOnly.On = FilePath;
      } else if (BaseName == "oclc_daz_opt_on") {
        DenormalsAreZero.On = FilePath;
      } else if (BaseName == "oclc_daz_opt_off") {
        DenormalsAreZero.Off = FilePath;
      } else if (BaseName == "oclc_correctly_rounded_sqrt_on") {
        CorrectlyRoundedSqrt.On = FilePath;
      } else if (BaseName == "oclc_correctly_rounded_sqrt_off") {
        CorrectlyRoundedSqrt.Off = FilePath;
      } else if (BaseName == "oclc_unsafe_math_on") {
        UnsafeMath.On = FilePath;
      } else if (BaseName == "oclc_unsafe_math_off") {
        UnsafeMath.Off = FilePath;
      } else if (BaseName == "oclc_wavefrontsize64_on") {
        WavefrontSize64.On = FilePath;
      } else if (BaseName == "oclc_wavefrontsize64_off") {
        WavefrontSize64.Off = FilePath;
      } else {
        // Process all bitcode filenames that look like
        // ocl_isa_version_XXX.amdgcn.bc
        const StringRef DeviceLibPrefix = "oclc_isa_version_";
        if (!BaseName.startswith(DeviceLibPrefix))
          continue;

        StringRef IsaVersionNumber =
            BaseName.drop_front(DeviceLibPrefix.size());

        llvm::Twine GfxName = Twine("gfx") + IsaVersionNumber;
        SmallString<8> Tmp;
        LibDeviceMap.insert(
            std::make_pair(GfxName.toStringRef(Tmp), FilePath.str()));
      }
    }

    if (!NoBuiltinLibs) {
      // Check that the required non-target libraries are all available.
      if (!allGenericLibsValid())
        continue;

      // Check that we have found at least one libdevice that we can link in if
      // -nobuiltinlib hasn't been specified.
      if (LibDeviceMap.empty())
        continue;
    }

    IsValid = true;
    break;
  }
}

void RocmInstallationDetector::print(raw_ostream &OS) const {
  if (isValid())
    OS << "Found ROCm installation: " << InstallPath << '\n';
}

void amdgpu::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {

  std::string Linker = getToolChain().GetProgramPath(getShortName());
  ArgStringList CmdArgs;
  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);
  CmdArgs.push_back("-shared");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  C.addCommand(std::make_unique<Command>(JA, *this, Args.MakeArgString(Linker),
                                          CmdArgs, Inputs));
}

void amdgpu::getAMDGPUTargetFeatures(const Driver &D,
                                     const llvm::opt::ArgList &Args,
                                     std::vector<StringRef> &Features) {
  if (const Arg *dAbi = Args.getLastArg(options::OPT_mamdgpu_debugger_abi))
    D.Diag(diag::err_drv_clang_unsupported) << dAbi->getAsString(Args);

  if (Args.getLastArg(options::OPT_mwavefrontsize64)) {
    Features.push_back("-wavefrontsize16");
    Features.push_back("-wavefrontsize32");
    Features.push_back("+wavefrontsize64");
  }
  if (Args.getLastArg(options::OPT_mno_wavefrontsize64)) {
    Features.push_back("-wavefrontsize16");
    Features.push_back("+wavefrontsize32");
    Features.push_back("-wavefrontsize64");
  }

  handleTargetFeaturesGroup(
    Args, Features, options::OPT_m_amdgpu_Features_Group);
}

/// AMDGPU Toolchain
AMDGPUToolChain::AMDGPUToolChain(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args),
      OptionsDefault({{options::OPT_O, "3"},
                      {options::OPT_cl_std_EQ, "CL1.2"}}) {}

Tool *AMDGPUToolChain::buildLinker() const {
  return new tools::amdgpu::Linker(*this);
}

DerivedArgList *
AMDGPUToolChain::TranslateArgs(const DerivedArgList &Args, StringRef BoundArch,
                               Action::OffloadKind DeviceOffloadKind) const {

  DerivedArgList *DAL =
      Generic_ELF::TranslateArgs(Args, BoundArch, DeviceOffloadKind);

  // Do nothing if not OpenCL (-x cl)
  if (!Args.getLastArgValue(options::OPT_x).equals("cl"))
    return DAL;

  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());
  for (auto *A : Args)
    DAL->append(A);

  const OptTable &Opts = getDriver().getOpts();

  // Phase 1 (.cl -> .bc)
  if (Args.hasArg(options::OPT_c) && Args.hasArg(options::OPT_emit_llvm)) {
    DAL->AddFlagArg(nullptr, Opts.getOption(getTriple().isArch64Bit()
                                                ? options::OPT_m64
                                                : options::OPT_m32));

    // Have to check OPT_O4, OPT_O0 & OPT_Ofast separately
    // as they defined that way in Options.td
    if (!Args.hasArg(options::OPT_O, options::OPT_O0, options::OPT_O4,
                     options::OPT_Ofast))
      DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_O),
                        getOptionDefault(options::OPT_O));
  }

  return DAL;
}

bool AMDGPUToolChain::getDefaultDenormsAreZeroForTarget(
    llvm::AMDGPU::GPUKind Kind) {

  // Assume nothing without a specific target.
  if (Kind == llvm::AMDGPU::GK_NONE)
    return false;

  const unsigned ArchAttr = llvm::AMDGPU::getArchAttrAMDGCN(Kind);

  // Default to enabling f32 denormals by default on subtargets where fma is
  // fast with denormals
  const bool BothDenormAndFMAFast =
      (ArchAttr & llvm::AMDGPU::FEATURE_FAST_FMA_F32) &&
      (ArchAttr & llvm::AMDGPU::FEATURE_FAST_DENORMAL_F32);
  return !BothDenormAndFMAFast;
}

llvm::DenormalMode AMDGPUToolChain::getDefaultDenormalModeForType(
    const llvm::opt::ArgList &DriverArgs, const JobAction &JA,
    const llvm::fltSemantics *FPType) const {
  // Denormals should always be enabled for f16 and f64.
  if (!FPType || FPType != &llvm::APFloat::IEEEsingle())
    return llvm::DenormalMode::getIEEE();

  if (JA.getOffloadingDeviceKind() == Action::OFK_HIP ||
      JA.getOffloadingDeviceKind() == Action::OFK_Cuda) {
    auto Kind = llvm::AMDGPU::parseArchAMDGCN(JA.getOffloadingArch());
    if (FPType && FPType == &llvm::APFloat::IEEEsingle() &&
        DriverArgs.hasFlag(options::OPT_fcuda_flush_denormals_to_zero,
                           options::OPT_fno_cuda_flush_denormals_to_zero,
                           getDefaultDenormsAreZeroForTarget(Kind)))
      return llvm::DenormalMode::getPreserveSign();

    return llvm::DenormalMode::getIEEE();
  }

  const StringRef GpuArch = DriverArgs.getLastArgValue(options::OPT_mcpu_EQ);
  auto Kind = llvm::AMDGPU::parseArchAMDGCN(GpuArch);

  // TODO: There are way too many flags that change this. Do we need to check
  // them all?
  bool DAZ = DriverArgs.hasArg(options::OPT_cl_denorms_are_zero) ||
             getDefaultDenormsAreZeroForTarget(Kind);

  // Outputs are flushed to zero (FTZ), preserving sign. Denormal inputs are
  // also implicit treated as zero (DAZ).
  return DAZ ? llvm::DenormalMode::getPreserveSign() :
               llvm::DenormalMode::getIEEE();
}

/// ROCM Toolchain
ROCMToolChain::ROCMToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ArgList &Args)
  : AMDGPUToolChain(D, Triple, Args),
    RocmInstallation(D, Triple, Args) { }

void AMDGPUToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  // Default to "hidden" visibility, as object level linking will not be
  // supported for the foreseeable future.
  if (!DriverArgs.hasArg(options::OPT_fvisibility_EQ,
                         options::OPT_fvisibility_ms_compat)) {
    CC1Args.push_back("-fvisibility");
    CC1Args.push_back("hidden");
    CC1Args.push_back("-fapply-global-visibility-to-externs");
  }
}

void ROCMToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  AMDGPUToolChain::addClangTargetOptions(DriverArgs, CC1Args,
                                         DeviceOffloadingKind);

  if (DriverArgs.hasArg(options::OPT_nogpulib))
    return;

  if (!RocmInstallation.isValid()) {
    getDriver().Diag(diag::err_drv_no_rocm_installation);
    return;
  }

  // Get the device name and canonicalize it
  const StringRef GpuArch = DriverArgs.getLastArgValue(options::OPT_mcpu_EQ);
  auto Kind = llvm::AMDGPU::parseArchAMDGCN(GpuArch);
  const StringRef CanonArch = llvm::AMDGPU::getArchNameAMDGCN(Kind);
  std::string LibDeviceFile = RocmInstallation.getLibDeviceFile(CanonArch);
  if (LibDeviceFile.empty()) {
    getDriver().Diag(diag::err_drv_no_rocm_device_lib) << GpuArch;
    return;
  }

  const unsigned ArchAttr = llvm::AMDGPU::getArchAttrAMDGCN(Kind);
  static bool HasWave32 = (ArchAttr & llvm::AMDGPU::FEATURE_WAVE32);

  bool Wave64 = !HasWave32 || DriverArgs.hasFlag(
    options::OPT_mwavefrontsize64, options::OPT_mno_wavefrontsize64, false);

  // TODO: There are way too many flags that change this. Do we need to check
  // them all?
  bool DAZ = DriverArgs.hasArg(options::OPT_cl_denorms_are_zero) ||
             getDefaultDenormsAreZeroForTarget(Kind);
  bool FiniteOnly = DriverArgs.hasArg(options::OPT_cl_finite_math_only);

  bool UnsafeMathOpt =
      DriverArgs.hasArg(options::OPT_cl_unsafe_math_optimizations);
  bool FastRelaxedMath = DriverArgs.hasArg(options::OPT_cl_fast_relaxed_math);
  bool CorrectSqrt =
      DriverArgs.hasArg(options::OPT_cl_fp32_correctly_rounded_divide_sqrt);

  // Add the OpenCL specific bitcode library.
  CC1Args.push_back("-mlink-builtin-bitcode");
  CC1Args.push_back(DriverArgs.MakeArgString(RocmInstallation.getOpenCLPath()));

  // Add the generic set of libraries.
  RocmInstallation.addCommonBitcodeLibCC1Args(
      DriverArgs, CC1Args, LibDeviceFile, Wave64, DAZ, FiniteOnly,
      UnsafeMathOpt, FastRelaxedMath, CorrectSqrt);
}

void RocmInstallationDetector::addCommonBitcodeLibCC1Args(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    StringRef LibDeviceFile, bool Wave64, bool DAZ, bool FiniteOnly,
    bool UnsafeMathOpt, bool FastRelaxedMath, bool CorrectSqrt) const {
  static const char LinkBitcodeFlag[] = "-mlink-builtin-bitcode";

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(DriverArgs.MakeArgString(getOCMLPath()));

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(DriverArgs.MakeArgString(getOCKLPath()));

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(DriverArgs.MakeArgString(getDenormalsAreZeroPath(DAZ)));

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(DriverArgs.MakeArgString(
      getUnsafeMathPath(UnsafeMathOpt || FastRelaxedMath)));

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(DriverArgs.MakeArgString(
      getFiniteOnlyPath(FiniteOnly || FastRelaxedMath)));

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(
      DriverArgs.MakeArgString(getCorrectlyRoundedSqrtPath(CorrectSqrt)));

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(DriverArgs.MakeArgString(getWavefrontSize64Path(Wave64)));

  CC1Args.push_back(LinkBitcodeFlag);
  CC1Args.push_back(DriverArgs.MakeArgString(LibDeviceFile));
}
