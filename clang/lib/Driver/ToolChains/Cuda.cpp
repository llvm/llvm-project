//===--- Cuda.cpp - Cuda Tool and ToolChain Implementations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Cuda.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Distro.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

// Parses the contents of version.txt in an CUDA installation.  It should
// contain one line of the from e.g. "CUDA Version 7.5.2".
static CudaVersion ParseCudaVersionFile(llvm::StringRef V) {
  if (!V.startswith("CUDA Version "))
    return CudaVersion::UNKNOWN;
  V = V.substr(strlen("CUDA Version "));
  int Major = -1, Minor = -1;
  auto First = V.split('.');
  auto Second = First.second.split('.');
  if (First.first.getAsInteger(10, Major) ||
      Second.first.getAsInteger(10, Minor))
    return CudaVersion::UNKNOWN;

  if (Major == 7 && Minor == 0) {
    // This doesn't appear to ever happen -- version.txt doesn't exist in the
    // CUDA 7 installs I've seen.  But no harm in checking.
    return CudaVersion::CUDA_70;
  }
  if (Major == 7 && Minor == 5)
    return CudaVersion::CUDA_75;
  if (Major == 8 && Minor == 0)
    return CudaVersion::CUDA_80;
  if (Major == 9 && Minor == 0)
    return CudaVersion::CUDA_90;
  if (Major == 9 && Minor == 1)
    return CudaVersion::CUDA_91;
  return CudaVersion::UNKNOWN;
}

CudaInstallationDetector::CudaInstallationDetector(
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

  // In decreasing order so we prefer newer versions to older versions.
  std::initializer_list<const char *> Versions = {"8.0", "7.5", "7.0"};

  if (Args.hasArg(clang::driver::options::OPT_cuda_path_EQ)) {
    Candidates.emplace_back(
        Args.getLastArgValue(clang::driver::options::OPT_cuda_path_EQ).str());
  } else if (HostTriple.isOSWindows()) {
    for (const char *Ver : Versions)
      Candidates.emplace_back(
          D.SysRoot + "/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v" +
          Ver);
  } else {
    if (!Args.hasArg(clang::driver::options::OPT_cuda_path_ignore_env)) {
      // Try to find ptxas binary. If the executable is located in a directory
      // called 'bin/', its parent directory might be a good guess for a valid
      // CUDA installation.
      // However, some distributions might installs 'ptxas' to /usr/bin. In that
      // case the candidate would be '/usr' which passes the following checks
      // because '/usr/include' exists as well. To avoid this case, we always
      // check for the directory potentially containing files for libdevice,
      // even if the user passes -nocudalib.
      if (llvm::ErrorOr<std::string> ptxas =
              llvm::sys::findProgramByName("ptxas")) {
        SmallString<256> ptxasAbsolutePath;
        llvm::sys::fs::real_path(*ptxas, ptxasAbsolutePath);

        StringRef ptxasDir = llvm::sys::path::parent_path(ptxasAbsolutePath);
        if (llvm::sys::path::filename(ptxasDir) == "bin")
          Candidates.emplace_back(llvm::sys::path::parent_path(ptxasDir),
                                  /*StrictChecking=*/true);
      }
    }

    Candidates.emplace_back(D.SysRoot + "/usr/local/cuda");
    for (const char *Ver : Versions)
      Candidates.emplace_back(D.SysRoot + "/usr/local/cuda-" + Ver);

    if (Distro(D.getVFS()).IsDebian())
      // Special case for Debian to have nvidia-cuda-toolkit work
      // out of the box. More info on http://bugs.debian.org/882505
      Candidates.emplace_back(D.SysRoot + "/usr/lib/cuda");
  }

  bool NoCudaLib = Args.hasArg(options::OPT_nocudalib);

  for (const auto &Candidate : Candidates) {
    InstallPath = Candidate.Path;
    if (InstallPath.empty() || !D.getVFS().exists(InstallPath))
      continue;

    BinPath = InstallPath + "/bin";
    IncludePath = InstallPath + "/include";
    LibDevicePath = InstallPath + "/nvvm/libdevice";

    auto &FS = D.getVFS();
    if (!(FS.exists(IncludePath) && FS.exists(BinPath)))
      continue;
    bool CheckLibDevice = (!NoCudaLib || Candidate.StrictChecking);
    if (CheckLibDevice && !FS.exists(LibDevicePath))
      continue;

    // On Linux, we have both lib and lib64 directories, and we need to choose
    // based on our triple.  On MacOS, we have only a lib directory.
    //
    // It's sufficient for our purposes to be flexible: If both lib and lib64
    // exist, we choose whichever one matches our triple.  Otherwise, if only
    // lib exists, we use it.
    if (HostTriple.isArch64Bit() && FS.exists(InstallPath + "/lib64"))
      LibPath = InstallPath + "/lib64";
    else if (FS.exists(InstallPath + "/lib"))
      LibPath = InstallPath + "/lib";
    else
      continue;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> VersionFile =
        FS.getBufferForFile(InstallPath + "/version.txt");
    if (!VersionFile) {
      // CUDA 7.0 doesn't have a version.txt, so guess that's our version if
      // version.txt isn't present.
      Version = CudaVersion::CUDA_70;
    } else {
      Version = ParseCudaVersionFile((*VersionFile)->getBuffer());
    }

    if (Version >= CudaVersion::CUDA_90) {
      // CUDA-9+ uses single libdevice file for all GPU variants.
      std::string FilePath = LibDevicePath + "/libdevice.10.bc";
      if (FS.exists(FilePath)) {
        for (const char *GpuArchName :
             {"sm_20", "sm_30", "sm_32", "sm_35", "sm_50", "sm_52", "sm_53",
                   "sm_60", "sm_61", "sm_62", "sm_70", "sm_72"}) {
          const CudaArch GpuArch = StringToCudaArch(GpuArchName);
          if (Version >= MinVersionForCudaArch(GpuArch) &&
              Version <= MaxVersionForCudaArch(GpuArch))
            LibDeviceMap[GpuArchName] = FilePath;
        }
      }
    } else {
      std::error_code EC;
      for (llvm::sys::fs::directory_iterator LI(LibDevicePath, EC), LE;
           !EC && LI != LE; LI = LI.increment(EC)) {
        StringRef FilePath = LI->path();
        StringRef FileName = llvm::sys::path::filename(FilePath);
        // Process all bitcode filenames that look like
        // libdevice.compute_XX.YY.bc
        const StringRef LibDeviceName = "libdevice.";
        if (!(FileName.startswith(LibDeviceName) && FileName.endswith(".bc")))
          continue;
        StringRef GpuArch = FileName.slice(
            LibDeviceName.size(), FileName.find('.', LibDeviceName.size()));
        LibDeviceMap[GpuArch] = FilePath.str();
        // Insert map entries for specifc devices with this compute
        // capability. NVCC's choice of the libdevice library version is
        // rather peculiar and depends on the CUDA version.
        if (GpuArch == "compute_20") {
          LibDeviceMap["sm_20"] = FilePath;
          LibDeviceMap["sm_21"] = FilePath;
          LibDeviceMap["sm_32"] = FilePath;
        } else if (GpuArch == "compute_30") {
          LibDeviceMap["sm_30"] = FilePath;
          if (Version < CudaVersion::CUDA_80) {
            LibDeviceMap["sm_50"] = FilePath;
            LibDeviceMap["sm_52"] = FilePath;
            LibDeviceMap["sm_53"] = FilePath;
          }
          LibDeviceMap["sm_60"] = FilePath;
          LibDeviceMap["sm_61"] = FilePath;
          LibDeviceMap["sm_62"] = FilePath;
        } else if (GpuArch == "compute_35") {
          LibDeviceMap["sm_35"] = FilePath;
          LibDeviceMap["sm_37"] = FilePath;
        } else if (GpuArch == "compute_50") {
          if (Version >= CudaVersion::CUDA_80) {
            LibDeviceMap["sm_50"] = FilePath;
            LibDeviceMap["sm_52"] = FilePath;
            LibDeviceMap["sm_53"] = FilePath;
          }
        }
      }
    }

    // Check that we have found at least one libdevice that we can link in if
    // -nocudalib hasn't been specified.
    if (LibDeviceMap.empty() && !NoCudaLib)
      continue;

    IsValid = true;
    break;
  }
}

void CudaInstallationDetector::AddCudaIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    // Add cuda_wrappers/* to our system include path.  This lets us wrap
    // standard library headers.
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    llvm::sys::path::append(P, "cuda_wrappers");
    CC1Args.push_back("-internal-isystem");
    CC1Args.push_back(DriverArgs.MakeArgString(P));
  }

  if (DriverArgs.hasArg(options::OPT_nocudainc))
    return;

  if (!isValid()) {
    D.Diag(diag::err_drv_no_cuda_installation);
    return;
  }

  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(getIncludePath()));
  CC1Args.push_back("-include");
  CC1Args.push_back("__clang_cuda_runtime_wrapper.h");
}

void CudaInstallationDetector::CheckCudaVersionSupportsArch(
    CudaArch Arch) const {
  if (Arch == CudaArch::UNKNOWN || Version == CudaVersion::UNKNOWN ||
      ArchsWithBadVersion.count(Arch) > 0)
    return;

  auto MinVersion = MinVersionForCudaArch(Arch);
  auto MaxVersion = MaxVersionForCudaArch(Arch);
  if (Version < MinVersion || Version > MaxVersion) {
    ArchsWithBadVersion.insert(Arch);
    D.Diag(diag::err_drv_cuda_version_unsupported)
        << CudaArchToString(Arch) << CudaVersionToString(MinVersion)
        << CudaVersionToString(MaxVersion) << InstallPath
        << CudaVersionToString(Version);
  }
}

void CudaInstallationDetector::print(raw_ostream &OS) const {
  if (isValid())
    OS << "Found CUDA installation: " << InstallPath << ", version "
       << CudaVersionToString(Version) << "\n";
}

void NVPTX::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::CudaToolChain &>(getToolChain());
  assert(TC.getTriple().isNVPTX() && "Wrong platform");

  StringRef GPUArchName;
  // If this is an OpenMP action we need to extract the device architecture
  // from the -march=arch option. This option may come from -Xopenmp-target
  // flag or the default value.
  if (JA.isDeviceOffloading(Action::OFK_OpenMP)) {
    GPUArchName = Args.getLastArgValue(options::OPT_march_EQ);
    assert(!GPUArchName.empty() && "Must have an architecture passed in.");
  } else
    GPUArchName = JA.getOffloadingArch();

  // Obtain architecture from the action.
  CudaArch gpu_arch = StringToCudaArch(GPUArchName);
  assert(gpu_arch != CudaArch::UNKNOWN &&
         "Device action expected to have an architecture.");

  // Check that our installation's ptxas supports gpu_arch.
  if (!Args.hasArg(options::OPT_no_cuda_version_check)) {
    TC.CudaInstallation.CheckCudaVersionSupportsArch(gpu_arch);
  }

  ArgStringList CmdArgs;
  CmdArgs.push_back(TC.getTriple().isArch64Bit() ? "-m64" : "-m32");
  if (Args.hasFlag(options::OPT_cuda_noopt_device_debug,
                   options::OPT_no_cuda_noopt_device_debug, false)) {
    // ptxas does not accept -g option if optimization is enabled, so
    // we ignore the compiler's -O* options if we want debug info.
    CmdArgs.push_back("-g");
    CmdArgs.push_back("--dont-merge-basicblocks");
    CmdArgs.push_back("--return-at-end");
  } else if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    // Map the -O we received to -O{0,1,2,3}.
    //
    // TODO: Perhaps we should map host -O2 to ptxas -O3. -O3 is ptxas's
    // default, so it may correspond more closely to the spirit of clang -O2.

    // -O3 seems like the least-bad option when -Osomething is specified to
    // clang but it isn't handled below.
    StringRef OOpt = "3";
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    else if (A->getOption().matches(options::OPT_O)) {
      // -Os, -Oz, and -O(anything else) map to -O2, for lack of better options.
      OOpt = llvm::StringSwitch<const char *>(A->getValue())
                 .Case("1", "1")
                 .Case("2", "2")
                 .Case("3", "3")
                 .Case("s", "2")
                 .Case("z", "2")
                 .Default("2");
    }
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("-O") + OOpt));
  } else {
    // If no -O was passed, pass -O0 to ptxas -- no opt flag should correspond
    // to no optimizations, but ptxas's default is -O3.
    CmdArgs.push_back("-O0");
  }

  // Pass -v to ptxas if it was passed to the driver.
  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-v");

  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(Args.MakeArgString(CudaArchToString(gpu_arch)));
  CmdArgs.push_back("--output-file");
  CmdArgs.push_back(Args.MakeArgString(TC.getInputFilename(Output)));
  for (const auto& II : Inputs)
    CmdArgs.push_back(Args.MakeArgString(II.getFilename()));

  for (const auto& A : Args.getAllArgValues(options::OPT_Xcuda_ptxas))
    CmdArgs.push_back(Args.MakeArgString(A));

  bool Relocatable = false;
  if (JA.isOffloading(Action::OFK_OpenMP))
    // In OpenMP we need to generate relocatable code.
    Relocatable = Args.hasFlag(options::OPT_fopenmp_relocatable_target,
                               options::OPT_fnoopenmp_relocatable_target,
                               /*Default=*/true);
  else if (JA.isOffloading(Action::OFK_Cuda))
    Relocatable = Args.hasFlag(options::OPT_fcuda_rdc,
                               options::OPT_fno_cuda_rdc, /*Default=*/false);

  if (Relocatable)
    CmdArgs.push_back("-c");

  const char *Exec;
  if (Arg *A = Args.getLastArg(options::OPT_ptxas_path_EQ))
    Exec = A->getValue();
  else
    Exec = Args.MakeArgString(TC.GetProgramPath("ptxas"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

// All inputs to this linker must be from CudaDeviceActions, as we need to look
// at the Inputs' Actions in order to figure out which GPU architecture they
// correspond to.
void NVPTX::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::CudaToolChain &>(getToolChain());
  assert(TC.getTriple().isNVPTX() && "Wrong platform");

  ArgStringList CmdArgs;
  CmdArgs.push_back("--cuda");
  CmdArgs.push_back(TC.getTriple().isArch64Bit() ? "-64" : "-32");
  CmdArgs.push_back(Args.MakeArgString("--create"));
  CmdArgs.push_back(Args.MakeArgString(Output.getFilename()));

  for (const auto& II : Inputs) {
    auto *A = II.getAction();
    assert(A->getInputs().size() == 1 &&
           "Device offload action is expected to have a single input");
    const char *gpu_arch_str = A->getOffloadingArch();
    assert(gpu_arch_str &&
           "Device action expected to have associated a GPU architecture!");
    CudaArch gpu_arch = StringToCudaArch(gpu_arch_str);

    // We need to pass an Arch of the form "sm_XX" for cubin files and
    // "compute_XX" for ptx.
    const char *Arch =
        (II.getType() == types::TY_PP_Asm)
            ? CudaVirtualArchToString(VirtualArchForCudaArch(gpu_arch))
            : gpu_arch_str;
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("--image=profile=") +
                                         Arch + ",file=" + II.getFilename()));
  }

  for (const auto& A : Args.getAllArgValues(options::OPT_Xcuda_fatbinary))
    CmdArgs.push_back(Args.MakeArgString(A));

  const char *Exec = Args.MakeArgString(TC.GetProgramPath("fatbinary"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void NVPTX::OpenMPLinker::ConstructJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::CudaToolChain &>(getToolChain());
  assert(TC.getTriple().isNVPTX() && "Wrong platform");

  ArgStringList CmdArgs;

  // OpenMP uses nvlink to link cubin files. The result will be embedded in the
  // host binary by the host linker.
  assert(!JA.isHostOffloading(Action::OFK_OpenMP) &&
         "CUDA toolchain not expected for an OpenMP host device.");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else
    assert(Output.isNothing() && "Invalid output.");
  if (Args.hasArg(options::OPT_g_Flag))
    CmdArgs.push_back("-g");

  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-v");

  StringRef GPUArch =
      Args.getLastArgValue(options::OPT_march_EQ);
  assert(!GPUArch.empty() && "At least one GPU Arch required for ptxas.");

  CmdArgs.push_back("-arch");
  CmdArgs.push_back(Args.MakeArgString(GPUArch));

  // Add paths specified in LIBRARY_PATH environment variable as -L options.
  addDirectoryList(Args, CmdArgs, "-L", "LIBRARY_PATH");

  // Add paths for the default clang library path.
  SmallString<256> DefaultLibPath =
      llvm::sys::path::parent_path(TC.getDriver().Dir);
  llvm::sys::path::append(DefaultLibPath, "lib" CLANG_LIBDIR_SUFFIX);
  CmdArgs.push_back(Args.MakeArgString(Twine("-L") + DefaultLibPath));

  // Add linking against library implementing OpenMP calls on NVPTX target.
  CmdArgs.push_back("-lomptarget-nvptx");

  for (const auto &II : Inputs) {
    if (II.getType() == types::TY_LLVM_IR ||
        II.getType() == types::TY_LTO_IR ||
        II.getType() == types::TY_LTO_BC ||
        II.getType() == types::TY_LLVM_BC) {
      C.getDriver().Diag(diag::err_drv_no_linker_llvm_support)
          << getToolChain().getTripleString();
      continue;
    }

    // Currently, we only pass the input files to the linker, we do not pass
    // any libraries that may be valid only for the host.
    if (!II.isFilename())
      continue;

    const char *CubinF = C.addTempFile(
        C.getArgs().MakeArgString(getToolChain().getInputFilename(II)));

    CmdArgs.push_back(CubinF);
  }

  AddOpenMPLinkerScript(getToolChain(), C, Output, Inputs, Args, CmdArgs, JA);

  const char *Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("nvlink"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

/// CUDA toolchain.  Our assembler is ptxas, and our "linker" is fatbinary,
/// which isn't properly a linker but nonetheless performs the step of stitching
/// together object files from the assembler into a single blob.

CudaToolChain::CudaToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args,
                             const Action::OffloadKind OK)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      CudaInstallation(D, HostTC.getTriple(), Args), OK(OK) {
  if (CudaInstallation.isValid())
    getProgramPaths().push_back(CudaInstallation.getBinPath());
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

std::string CudaToolChain::getInputFilename(const InputInfo &Input) const {
  // Only object files are changed, for example assembly files keep their .s
  // extensions. CUDA also continues to use .o as they don't use nvlink but
  // fatbinary.
  if (!(OK == Action::OFK_OpenMP && Input.getType() == types::TY_Object))
    return ToolChain::getInputFilename(Input);

  // Replace extension for object files with cubin because nvlink relies on
  // these particular file names.
  SmallString<256> Filename(ToolChain::getInputFilename(Input));
  llvm::sys::path::replace_extension(Filename, "cubin");
  return Filename.str();
}

void CudaToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  StringRef GpuArch = DriverArgs.getLastArgValue(options::OPT_march_EQ);
  assert(!GpuArch.empty() && "Must have an explicit GPU arch.");
  assert((DeviceOffloadingKind == Action::OFK_OpenMP ||
          DeviceOffloadingKind == Action::OFK_Cuda) &&
         "Only OpenMP or CUDA offloading kinds are supported for NVIDIA GPUs.");

  if (DeviceOffloadingKind == Action::OFK_Cuda) {
    CC1Args.push_back("-fcuda-is-device");

    if (DriverArgs.hasFlag(options::OPT_fcuda_flush_denormals_to_zero,
                           options::OPT_fno_cuda_flush_denormals_to_zero, false))
      CC1Args.push_back("-fcuda-flush-denormals-to-zero");

    if (DriverArgs.hasFlag(options::OPT_fcuda_approx_transcendentals,
                           options::OPT_fno_cuda_approx_transcendentals, false))
      CC1Args.push_back("-fcuda-approx-transcendentals");

    if (DriverArgs.hasFlag(options::OPT_fcuda_rdc, options::OPT_fno_cuda_rdc,
                           false))
      CC1Args.push_back("-fcuda-rdc");
  }

  if (DriverArgs.hasArg(options::OPT_nocudalib))
    return;

  std::string LibDeviceFile = CudaInstallation.getLibDeviceFile(GpuArch);

  if (LibDeviceFile.empty()) {
    if (DeviceOffloadingKind == Action::OFK_OpenMP &&
        DriverArgs.hasArg(options::OPT_S))
      return;

    getDriver().Diag(diag::err_drv_no_cuda_libdevice) << GpuArch;
    return;
  }

  CC1Args.push_back("-mlink-cuda-bitcode");
  CC1Args.push_back(DriverArgs.MakeArgString(LibDeviceFile));

  if (CudaInstallation.version() >= CudaVersion::CUDA_90) {
    // CUDA-9 uses new instructions that are only available in PTX6.0
    CC1Args.push_back("-target-feature");
    CC1Args.push_back("+ptx60");
  } else {
    // Libdevice in CUDA-7.0 requires PTX version that's more recent
    // than LLVM defaults to. Use PTX4.2 which is the PTX version that
    // came with CUDA-7.0.
    CC1Args.push_back("-target-feature");
    CC1Args.push_back("+ptx42");
  }

  if (DeviceOffloadingKind == Action::OFK_OpenMP) {
    SmallVector<StringRef, 8> LibraryPaths;
    // Add path to lib and/or lib64 folders.
    SmallString<256> DefaultLibPath =
      llvm::sys::path::parent_path(getDriver().Dir);
    llvm::sys::path::append(DefaultLibPath,
        Twine("lib") + CLANG_LIBDIR_SUFFIX);
    LibraryPaths.emplace_back(DefaultLibPath.c_str());

    // Add user defined library paths from LIBRARY_PATH.
    llvm::Optional<std::string> LibPath =
        llvm::sys::Process::GetEnv("LIBRARY_PATH");
    if (LibPath) {
      SmallVector<StringRef, 8> Frags;
      const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
      llvm::SplitString(*LibPath, Frags, EnvPathSeparatorStr);
      for (StringRef Path : Frags)
        LibraryPaths.emplace_back(Path.trim());
    }

    std::string LibOmpTargetName =
      "libomptarget-nvptx-" + GpuArch.str() + ".bc";
    bool FoundBCLibrary = false;
    for (StringRef LibraryPath : LibraryPaths) {
      SmallString<128> LibOmpTargetFile(LibraryPath);
      llvm::sys::path::append(LibOmpTargetFile, LibOmpTargetName);
      if (llvm::sys::fs::exists(LibOmpTargetFile)) {
        CC1Args.push_back("-mlink-cuda-bitcode");
        CC1Args.push_back(DriverArgs.MakeArgString(LibOmpTargetFile));
        FoundBCLibrary = true;
        break;
      }
    }
    if (!FoundBCLibrary)
      getDriver().Diag(diag::warn_drv_omp_offload_target_missingbcruntime)
          << LibOmpTargetName;
  }
}

void CudaToolChain::AddCudaIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  // Check our CUDA version if we're going to include the CUDA headers.
  if (!DriverArgs.hasArg(options::OPT_nocudainc) &&
      !DriverArgs.hasArg(options::OPT_no_cuda_version_check)) {
    StringRef Arch = DriverArgs.getLastArgValue(options::OPT_march_EQ);
    assert(!Arch.empty() && "Must have an explicit GPU arch.");
    CudaInstallation.CheckCudaVersionSupportsArch(StringToCudaArch(Arch));
  }
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

llvm::opt::DerivedArgList *
CudaToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  // For OpenMP device offloading, append derived arguments. Make sure
  // flags are not duplicated.
  // Also append the compute capability.
  if (DeviceOffloadKind == Action::OFK_OpenMP) {
    for (Arg *A : Args) {
      bool IsDuplicate = false;
      for (Arg *DALArg : *DAL) {
        if (A == DALArg) {
          IsDuplicate = true;
          break;
        }
      }
      if (!IsDuplicate)
        DAL->append(A);
    }

    StringRef Arch = DAL->getLastArgValue(options::OPT_march_EQ);
    if (Arch.empty())
      DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                        CLANG_OPENMP_NVPTX_DEFAULT_ARCH);

    return DAL;
  }

  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT_Xarch__)) {
      // Skip this argument unless the architecture matches BoundArch
      if (BoundArch.empty() || A->getValue(0) != BoundArch)
        continue;

      unsigned Index = Args.getBaseArgs().MakeIndex(A->getValue(1));
      unsigned Prev = Index;
      std::unique_ptr<Arg> XarchArg(Opts.ParseOneArg(Args, Index));

      // If the argument parsing failed or more than one argument was
      // consumed, the -Xarch_ argument's parameter tried to consume
      // extra arguments. Emit an error and ignore.
      //
      // We also want to disallow any options which would alter the
      // driver behavior; that isn't going to work in our model. We
      // use isDriverOption() as an approximation, although things
      // like -O4 are going to slip through.
      if (!XarchArg || Index > Prev + 1) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_with_args)
            << A->getAsString(Args);
        continue;
      } else if (XarchArg->getOption().hasFlag(options::DriverOption)) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_isdriver)
            << A->getAsString(Args);
        continue;
      }
      XarchArg->setBaseArg(A);
      A = XarchArg.release();
      DAL->AddSynthesizedArg(A);
    }
    DAL->append(A);
  }

  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ), BoundArch);
  }
  return DAL;
}

Tool *CudaToolChain::buildAssembler() const {
  return new tools::NVPTX::Assembler(*this);
}

Tool *CudaToolChain::buildLinker() const {
  if (OK == Action::OFK_OpenMP)
    return new tools::NVPTX::OpenMPLinker(*this);
  return new tools::NVPTX::Linker(*this);
}

void CudaToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
CudaToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void CudaToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void CudaToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

void CudaToolChain::AddIAMCUIncludeArgs(const ArgList &Args,
                                        ArgStringList &CC1Args) const {
  HostTC.AddIAMCUIncludeArgs(Args, CC1Args);
}

SanitizerMask CudaToolChain::getSupportedSanitizers() const {
  // The CudaToolChain only supports sanitizers in the sense that it allows
  // sanitizer arguments on the command line if they are supported by the host
  // toolchain. The CudaToolChain will actually ignore any command line
  // arguments for any of these "supported" sanitizers. That means that no
  // sanitization of device code is actually supported at this time.
  //
  // This behavior is necessary because the host and device toolchains
  // invocations often share the command line, so the device toolchain must
  // tolerate flags meant only for the host toolchain.
  return HostTC.getSupportedSanitizers();
}

VersionTuple CudaToolChain::computeMSVCVersion(const Driver *D,
                                               const ArgList &Args) const {
  return HostTC.computeMSVCVersion(D, Args);
}
