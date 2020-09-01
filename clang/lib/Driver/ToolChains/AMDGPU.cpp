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
#include "clang/Basic/TargetID.h"
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

void RocmInstallationDetector::scanLibDevicePath(llvm::StringRef Path) {
  assert(!Path.empty());

  const StringRef Suffix(".bc");
  const StringRef Suffix2(".amdgcn.bc");

  std::error_code EC;
  for (llvm::vfs::directory_iterator LI = D.getVFS().dir_begin(Path, EC), LE;
       !EC && LI != LE; LI = LI.increment(EC)) {
    StringRef FilePath = LI->path();
    StringRef FileName = llvm::sys::path::filename(FilePath);
    if (!FileName.endswith(Suffix))
      continue;

    StringRef BaseName;
    if (FileName.endswith(Suffix2))
      BaseName = FileName.drop_back(Suffix2.size());
    else if (FileName.endswith(Suffix))
      BaseName = FileName.drop_back(Suffix.size());

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
}

void RocmInstallationDetector::ParseHIPVersionFile(llvm::StringRef V) {
  SmallVector<StringRef, 4> VersionParts;
  V.split(VersionParts, '\n');
  unsigned Major;
  unsigned Minor;
  for (auto Part : VersionParts) {
    auto Splits = Part.split('=');
    if (Splits.first == "HIP_VERSION_MAJOR")
      Splits.second.getAsInteger(0, Major);
    else if (Splits.first == "HIP_VERSION_MINOR")
      Splits.second.getAsInteger(0, Minor);
    else if (Splits.first == "HIP_VERSION_PATCH")
      VersionPatch = Splits.second.str();
  }
  VersionMajorMinor = llvm::VersionTuple(Major, Minor);
  DetectedVersion =
      (Twine(Major) + "." + Twine(Minor) + "." + VersionPatch).str();
}

// For candidate specified by --rocm-path we do not do strict check.
SmallVector<RocmInstallationDetector::Candidate, 4>
RocmInstallationDetector::getInstallationPathCandidates() {
  SmallVector<Candidate, 4> Candidates;
  if (!RocmPathArg.empty()) {
    Candidates.emplace_back(RocmPathArg.str());
    return Candidates;
  }

  // Try to find relative to the compiler binary.
  const char *InstallDir = D.getInstalledDir();

  // Check both a normal Unix prefix position of the clang binary, as well as
  // the Windows-esque layout the ROCm packages use with the host architecture
  // subdirectory of bin.

  // Strip off directory (usually bin)
  StringRef ParentDir = llvm::sys::path::parent_path(InstallDir);
  StringRef ParentName = llvm::sys::path::filename(ParentDir);

  // Some builds use bin/{host arch}, so go up again.
  if (ParentName == "bin") {
    ParentDir = llvm::sys::path::parent_path(ParentDir);
    ParentName = llvm::sys::path::filename(ParentDir);
  }

  // Some versions of the rocm llvm package install to /opt/rocm/llvm/bin
  if (ParentName == "llvm")
    ParentDir = llvm::sys::path::parent_path(ParentDir);

  Candidates.emplace_back(ParentDir.str(), /*StrictChecking=*/true);

  // Device library may be installed in clang resource directory.
  Candidates.emplace_back(D.ResourceDir, /*StrictChecking=*/true);

  Candidates.emplace_back(D.SysRoot + "/opt/rocm", /*StrictChecking=*/true);
  return Candidates;
}

RocmInstallationDetector::RocmInstallationDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args, bool DetectHIPRuntime, bool DetectDeviceLib)
    : D(D) {
  RocmPathArg = Args.getLastArgValue(clang::driver::options::OPT_rocm_path_EQ);
  RocmDeviceLibPathArg =
      Args.getAllArgValues(clang::driver::options::OPT_rocm_device_lib_path_EQ);
  if (auto *A = Args.getLastArg(clang::driver::options::OPT_hip_version_EQ)) {
    HIPVersionArg = A->getValue();
    unsigned Major = 0;
    unsigned Minor = 0;
    SmallVector<StringRef, 3> Parts;
    HIPVersionArg.split(Parts, '.');
    if (Parts.size())
      Parts[0].getAsInteger(0, Major);
    if (Parts.size() > 1)
      Parts[1].getAsInteger(0, Minor);
    if (Parts.size() > 2)
      VersionPatch = Parts[2].str();
    if (VersionPatch.empty())
      VersionPatch = "0";
    if (Major == 0 || Minor == 0)
      D.Diag(diag::err_drv_invalid_value)
          << A->getAsString(Args) << HIPVersionArg;

    VersionMajorMinor = llvm::VersionTuple(Major, Minor);
    DetectedVersion =
        (Twine(Major) + "." + Twine(Minor) + "." + VersionPatch).str();
  } else {
    VersionPatch = DefaultVersionPatch;
    VersionMajorMinor =
        llvm::VersionTuple(DefaultVersionMajor, DefaultVersionMinor);
    DetectedVersion = (Twine(DefaultVersionMajor) + "." +
                       Twine(DefaultVersionMinor) + "." + VersionPatch)
                          .str();
  }

  if (DetectHIPRuntime)
    detectHIPRuntime();
  if (DetectDeviceLib)
    detectDeviceLibrary();
}

void RocmInstallationDetector::detectDeviceLibrary() {
  assert(LibDevicePath.empty());

  if (!RocmDeviceLibPathArg.empty())
    LibDevicePath = RocmDeviceLibPathArg[RocmDeviceLibPathArg.size() - 1];
  else if (const char *LibPathEnv = ::getenv("HIP_DEVICE_LIB_PATH"))
    LibDevicePath = LibPathEnv;

  auto &FS = D.getVFS();
  if (!LibDevicePath.empty()) {
    // Maintain compatability with HIP flag/envvar pointing directly at the
    // bitcode library directory. This points directly at the library path instead
    // of the rocm root installation.
    if (!FS.exists(LibDevicePath))
      return;

    scanLibDevicePath(LibDevicePath);
    HasDeviceLibrary = allGenericLibsValid() && !LibDeviceMap.empty();
    return;
  }

  // The install path situation in old versions of ROCm is a real mess, and
  // use a different install layout. Multiple copies of the device libraries
  // exist for each frontend project, and differ depending on which build
  // system produced the packages. Standalone OpenCL builds also have a
  // different directory structure from the ROCm OpenCL package.
  auto Candidates = getInstallationPathCandidates();
  for (const auto &Candidate : Candidates) {
    auto CandidatePath = Candidate.Path;

    // Check device library exists at the given path.
    auto CheckDeviceLib = [&](StringRef Path) {
      bool CheckLibDevice = (!NoBuiltinLibs || Candidate.StrictChecking);
      if (CheckLibDevice && !FS.exists(Path))
        return false;

      scanLibDevicePath(Path);

      if (!NoBuiltinLibs) {
        // Check that the required non-target libraries are all available.
        if (!allGenericLibsValid())
          return false;

        // Check that we have found at least one libdevice that we can link in
        // if -nobuiltinlib hasn't been specified.
        if (LibDeviceMap.empty())
          return false;
      }
      return true;
    };

    // The possible structures are:
    // - ${ROCM_ROOT}/amdgcn/bitcode/*
    // - ${ROCM_ROOT}/lib/*
    // - ${ROCM_ROOT}/lib/bitcode/*
    // so try to detect these layouts.
    static constexpr std::array<const char *, 2> SubDirsList[] = {
        {"amdgcn", "bitcode"},
        {"lib", ""},
        {"lib", "bitcode"},
    };

    // Make a path by appending sub-directories to InstallPath.
    auto MakePath = [&](const llvm::ArrayRef<const char *> &SubDirs) {
      auto Path = CandidatePath;
      for (auto SubDir : SubDirs)
        llvm::sys::path::append(Path, SubDir);
      return Path;
    };

    for (auto SubDirs : SubDirsList) {
      LibDevicePath = MakePath(SubDirs);
      HasDeviceLibrary = CheckDeviceLib(LibDevicePath);
      if (HasDeviceLibrary)
        return;
    }
  }
}

void RocmInstallationDetector::detectHIPRuntime() {
  auto Candidates = getInstallationPathCandidates();
  auto &FS = D.getVFS();

  for (const auto &Candidate : Candidates) {
    InstallPath = Candidate.Path;
    if (InstallPath.empty() || !FS.exists(InstallPath))
      continue;

    BinPath = InstallPath;
    llvm::sys::path::append(BinPath, "bin");
    IncludePath = InstallPath;
    llvm::sys::path::append(IncludePath, "include");
    LibPath = InstallPath;
    llvm::sys::path::append(LibPath, "lib");

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> VersionFile =
        FS.getBufferForFile(BinPath + "/.hipVersion");
    if (!VersionFile && Candidate.StrictChecking)
      continue;

    if (HIPVersionArg.empty() && VersionFile)
      ParseHIPVersionFile((*VersionFile)->getBuffer());

    HasHIPRuntime = true;
    return;
  }
  HasHIPRuntime = false;
}

void RocmInstallationDetector::print(raw_ostream &OS) const {
  if (hasHIPRuntime())
    OS << "Found HIP installation: " << InstallPath << ", version "
       << DetectedVersion << '\n';
}

void RocmInstallationDetector::AddHIPIncludeArgs(const ArgList &DriverArgs,
                                                 ArgStringList &CC1Args) const {
  bool UsesRuntimeWrapper = VersionMajorMinor > llvm::VersionTuple(3, 5);

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    // HIP header includes standard library wrapper headers under clang
    // cuda_wrappers directory. Since these wrapper headers include_next
    // standard C++ headers, whereas libc++ headers include_next other clang
    // headers. The include paths have to follow this order:
    // - wrapper include path
    // - standard C++ include path
    // - other clang include path
    // Since standard C++ and other clang include paths are added in other
    // places after this function, here we only need to make sure wrapper
    // include path is added.
    //
    // ROCm 3.5 does not fully support the wrapper headers. Therefore it needs
    // a workaround.
    SmallString<128> P(D.ResourceDir);
    if (UsesRuntimeWrapper)
      llvm::sys::path::append(P, "include", "cuda_wrappers");
    CC1Args.push_back("-internal-isystem");
    CC1Args.push_back(DriverArgs.MakeArgString(P));
  }

  if (DriverArgs.hasArg(options::OPT_nogpuinc))
    return;

  if (!hasHIPRuntime()) {
    D.Diag(diag::err_drv_no_hip_runtime);
    return;
  }

  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(getIncludePath()));
  if (UsesRuntimeWrapper)
    CC1Args.append({"-include", "__clang_hip_runtime_wrapper.h"});
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
  C.addCommand(
      std::make_unique<Command>(JA, *this, ResponseFileSupport::AtFileCurCP(),
                                Args.MakeArgString(Linker), CmdArgs, Inputs));
}

void amdgpu::getAMDGPUTargetFeatures(const Driver &D,
                                     const llvm::Triple &Triple,
                                     const llvm::opt::ArgList &Args,
                                     std::vector<StringRef> &Features) {
  if (const Arg *dAbi = Args.getLastArg(options::OPT_mamdgpu_debugger_abi))
    D.Diag(diag::err_drv_clang_unsupported) << dAbi->getAsString(Args);

  // Add target ID features to -target-feature options. No diagnostics should
  // be emitted here since invalid target ID is diagnosed at other places.
  StringRef TargetID = Args.getLastArgValue(options::OPT_mcpu_EQ);
  if (!TargetID.empty()) {
    llvm::StringMap<bool> FeatureMap;
    auto OptionalGpuArch = parseTargetID(Triple, TargetID, &FeatureMap);
    if (OptionalGpuArch) {
      StringRef GpuArch = OptionalGpuArch.getValue();
      // Iterate through all possible target ID features for the given GPU.
      // If it is mapped to true, add +feature.
      // If it is mapped to false, add -feature.
      // If it is not in the map (default), do not add it
      for (auto &&Feature : getAllPossibleTargetIDFeatures(Triple, GpuArch)) {
        auto Pos = FeatureMap.find(Feature);
        if (Pos == FeatureMap.end())
          continue;
        Features.push_back(Args.MakeArgStringRef(
            (Twine(Pos->second ? "+" : "-") + Feature).str()));
      }
    }
  }

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

  const OptTable &Opts = getDriver().getOpts();

  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());
  for (auto *A : Args)
    DAL->append(A);

  if (!Args.getLastArgValue(options::OPT_x).equals("cl"))
    return DAL;

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
    auto Arch = getProcessorFromTargetID(getTriple(), JA.getOffloadingArch());
    auto Kind = llvm::AMDGPU::parseArchAMDGCN(Arch);
    if (FPType && FPType == &llvm::APFloat::IEEEsingle() &&
        DriverArgs.hasFlag(options::OPT_fcuda_flush_denormals_to_zero,
                           options::OPT_fno_cuda_flush_denormals_to_zero,
                           getDefaultDenormsAreZeroForTarget(Kind)))
      return llvm::DenormalMode::getPreserveSign();

    return llvm::DenormalMode::getIEEE();
  }

  const StringRef GpuArch = getGPUArch(DriverArgs);
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

bool AMDGPUToolChain::isWave64(const llvm::opt::ArgList &DriverArgs,
                               llvm::AMDGPU::GPUKind Kind) {
  const unsigned ArchAttr = llvm::AMDGPU::getArchAttrAMDGCN(Kind);
  static bool HasWave32 = (ArchAttr & llvm::AMDGPU::FEATURE_WAVE32);

  return !HasWave32 || DriverArgs.hasFlag(
    options::OPT_mwavefrontsize64, options::OPT_mno_wavefrontsize64, false);
}


/// ROCM Toolchain
ROCMToolChain::ROCMToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ArgList &Args)
    : AMDGPUToolChain(D, Triple, Args) {
  RocmInstallation.detectDeviceLibrary();
}

void AMDGPUToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  // Allow using target ID in -mcpu.
  translateTargetID(DriverArgs, CC1Args);
  // Default to "hidden" visibility, as object level linking will not be
  // supported for the foreseeable future.
  if (!DriverArgs.hasArg(options::OPT_fvisibility_EQ,
                         options::OPT_fvisibility_ms_compat)) {
    CC1Args.push_back("-fvisibility");
    CC1Args.push_back("hidden");
    CC1Args.push_back("-fapply-global-visibility-to-externs");
  }
}

StringRef
AMDGPUToolChain::getGPUArch(const llvm::opt::ArgList &DriverArgs) const {
  return getProcessorFromTargetID(
      getTriple(), DriverArgs.getLastArgValue(options::OPT_mcpu_EQ));
}

StringRef
AMDGPUToolChain::translateTargetID(const llvm::opt::ArgList &DriverArgs,
                                   llvm::opt::ArgStringList &CC1Args) const {
  StringRef TargetID = DriverArgs.getLastArgValue(options::OPT_mcpu_EQ);
  if (TargetID.empty())
    return StringRef();

  llvm::StringMap<bool> FeatureMap;
  auto OptionalGpuArch = parseTargetID(getTriple(), TargetID, &FeatureMap);
  if (!OptionalGpuArch) {
    getDriver().Diag(clang::diag::err_drv_bad_target_id) << TargetID;
    return StringRef();
  }

  return OptionalGpuArch.getValue();
}

void ROCMToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  AMDGPUToolChain::addClangTargetOptions(DriverArgs, CC1Args,
                                         DeviceOffloadingKind);

  // For the OpenCL case where there is no offload target, accept -nostdlib to
  // disable bitcode linking.
  if (DeviceOffloadingKind == Action::OFK_None &&
      DriverArgs.hasArg(options::OPT_nostdlib))
    return;

  if (DriverArgs.hasArg(options::OPT_nogpulib))
    return;

  if (!RocmInstallation.hasDeviceLibrary()) {
    getDriver().Diag(diag::err_drv_no_rocm_device_lib) << 0;
    return;
  }

  // Get the device name and canonicalize it
  const StringRef GpuArch = getGPUArch(DriverArgs);
  auto Kind = llvm::AMDGPU::parseArchAMDGCN(GpuArch);
  const StringRef CanonArch = llvm::AMDGPU::getArchNameAMDGCN(Kind);
  std::string LibDeviceFile = RocmInstallation.getLibDeviceFile(CanonArch);
  if (LibDeviceFile.empty()) {
    getDriver().Diag(diag::err_drv_no_rocm_device_lib) << 1 << GpuArch;
    return;
  }

  bool Wave64 = isWave64(DriverArgs, Kind);

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
