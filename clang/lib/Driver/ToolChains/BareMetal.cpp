//===-- BareMetal.cpp - Bare Metal ToolChain --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BareMetal.h"

#include "Gnu.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/InputInfo.h"

#include "Arch/AArch64.h"
#include "Arch/ARM.h"
#include "Arch/RISCV.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/MultilibBuilder.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <sstream>

using namespace llvm::opt;
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;

static bool isRISCVBareMetal(const llvm::Triple &Triple) {
  if (!Triple.isRISCV())
    return false;

  if (Triple.getVendor() != llvm::Triple::UnknownVendor)
    return false;

  if (Triple.getOS() != llvm::Triple::UnknownOS)
    return false;

  return Triple.getEnvironmentName() == "elf";
}

/// Is the triple powerpc[64][le]-*-none-eabi?
static bool isPPCBareMetal(const llvm::Triple &Triple) {
  return Triple.isPPC() && Triple.getOS() == llvm::Triple::UnknownOS &&
         Triple.getEnvironment() == llvm::Triple::EABI;
}

static bool findRISCVMultilibs(const Driver &D,
                               const llvm::Triple &TargetTriple,
                               const ArgList &Args, DetectedMultilibs &Result) {
  Multilib::flags_list Flags;
  std::string Arch = riscv::getRISCVArch(Args, TargetTriple);
  StringRef Abi = tools::riscv::getRISCVABI(Args, TargetTriple);

  if (TargetTriple.isRISCV64()) {
    MultilibBuilder Imac =
        MultilibBuilder().flag("-march=rv64imac").flag("-mabi=lp64");
    MultilibBuilder Imafdc = MultilibBuilder("/rv64imafdc/lp64d")
                                 .flag("-march=rv64imafdc")
                                 .flag("-mabi=lp64d");

    // Multilib reuse
    bool UseImafdc =
        (Arch == "rv64imafdc") || (Arch == "rv64gc"); // gc => imafdc

    addMultilibFlag((Arch == "rv64imac"), "-march=rv64imac", Flags);
    addMultilibFlag(UseImafdc, "-march=rv64imafdc", Flags);
    addMultilibFlag(Abi == "lp64", "-mabi=lp64", Flags);
    addMultilibFlag(Abi == "lp64d", "-mabi=lp64d", Flags);

    Result.Multilibs =
        MultilibSetBuilder().Either(Imac, Imafdc).makeMultilibSet();
    return Result.Multilibs.select(D, Flags, Result.SelectedMultilibs);
  }
  if (TargetTriple.isRISCV32()) {
    MultilibBuilder Imac =
        MultilibBuilder().flag("-march=rv32imac").flag("-mabi=ilp32");
    MultilibBuilder I = MultilibBuilder("/rv32i/ilp32")
                            .flag("-march=rv32i")
                            .flag("-mabi=ilp32");
    MultilibBuilder Im = MultilibBuilder("/rv32im/ilp32")
                             .flag("-march=rv32im")
                             .flag("-mabi=ilp32");
    MultilibBuilder Iac = MultilibBuilder("/rv32iac/ilp32")
                              .flag("-march=rv32iac")
                              .flag("-mabi=ilp32");
    MultilibBuilder Imafc = MultilibBuilder("/rv32imafc/ilp32f")
                                .flag("-march=rv32imafc")
                                .flag("-mabi=ilp32f");

    // Multilib reuse
    bool UseI = (Arch == "rv32i") || (Arch == "rv32ic");    // ic => i
    bool UseIm = (Arch == "rv32im") || (Arch == "rv32imc"); // imc => im
    bool UseImafc = (Arch == "rv32imafc") || (Arch == "rv32imafdc") ||
                    (Arch == "rv32gc"); // imafdc,gc => imafc

    addMultilibFlag(UseI, "-march=rv32i", Flags);
    addMultilibFlag(UseIm, "-march=rv32im", Flags);
    addMultilibFlag((Arch == "rv32iac"), "-march=rv32iac", Flags);
    addMultilibFlag((Arch == "rv32imac"), "-march=rv32imac", Flags);
    addMultilibFlag(UseImafc, "-march=rv32imafc", Flags);
    addMultilibFlag(Abi == "ilp32", "-mabi=ilp32", Flags);
    addMultilibFlag(Abi == "ilp32f", "-mabi=ilp32f", Flags);

    Result.Multilibs =
        MultilibSetBuilder().Either(I, Im, Iac, Imac, Imafc).makeMultilibSet();
    return Result.Multilibs.select(D, Flags, Result.SelectedMultilibs);
  }
  return false;
}

static std::string computeClangRuntimesSysRoot(const Driver &D,
                                               bool IncludeTriple) {
  if (!D.SysRoot.empty())
    return D.SysRoot;

  SmallString<128> SysRootDir(D.Dir);
  llvm::sys::path::append(SysRootDir, "..", "lib", "clang-runtimes");

  if (IncludeTriple)
    llvm::sys::path::append(SysRootDir, D.getTargetTriple());

  return std::string(SysRootDir);
}

// Only consider the GCC toolchain based on the values provided through the
// `--gcc-toolchain` and `--gcc-install-dir` flags. The function below returns
// whether the GCC toolchain was initialized successfully.
bool BareMetal::initGCCInstallation(const llvm::Triple &Triple,
                                    const llvm::opt::ArgList &Args) {
  if (Args.getLastArg(options::OPT_gcc_toolchain) ||
      Args.getLastArg(clang::driver::options::OPT_gcc_install_dir_EQ)) {
    GCCInstallation.init(Triple, Args);
    return GCCInstallation.isValid();
  }
  return false;
}

// This logic is adapted from RISCVToolChain.cpp as part of the ongoing effort
// to merge RISCVToolChain into the Baremetal toolchain. It infers the presence
// of a valid GCC toolchain by checking whether the `crt0.o` file exists in the
// `bin/../<target-triple>/lib` directory.
static bool detectGCCToolchainAdjacent(const Driver &D) {
  SmallString<128> GCCDir;
  llvm::sys::path::append(GCCDir, D.Dir, "..", D.getTargetTriple(),
                          "lib/crt0.o");
  return llvm::sys::fs::exists(GCCDir);
}

// If no sysroot is provided the driver will first attempt to infer it from the
// values of `--gcc-install-dir` or `--gcc-toolchain`, which specify the
// location of a GCC toolchain.
// If neither flag is used, the sysroot defaults to either:
//    - `bin/../<target-triple>`
//    - `bin/../lib/clang-runtimes/<target-triple>`
//
// To use the `clang-runtimes` path, ensure that `../<target-triple>/lib/crt0.o`
// does not exist relative to the driver.
std::string BareMetal::computeSysRoot() const {
  // Use Baremetal::sysroot if it has already been set.
  if (!SysRoot.empty())
    return SysRoot;

  // Use the sysroot specified via the `--sysroot` command-line flag, if
  // provided.
  const Driver &D = getDriver();
  if (!D.SysRoot.empty())
    return D.SysRoot;

  // Attempt to infer sysroot from a valid GCC installation.
  // If no valid GCC installation, check for a GCC toolchain alongside Clang.
  SmallString<128> inferredSysRoot;
  if (IsGCCInstallationValid) {
    llvm::sys::path::append(inferredSysRoot, GCCInstallation.getParentLibPath(),
                            "..", GCCInstallation.getTriple().str());
  } else if (detectGCCToolchainAdjacent(D)) {
    // Use the triple as provided to the driver. Unlike the parsed triple
    // this has not been normalized to always contain every field.
    llvm::sys::path::append(inferredSysRoot, D.Dir, "..", D.getTargetTriple());
  }
  // If a valid sysroot was inferred and exists, use it
  if (!inferredSysRoot.empty() && llvm::sys::fs::exists(inferredSysRoot))
    return std::string(inferredSysRoot);

  // Use the clang-runtimes path.
  return computeClangRuntimesSysRoot(D, /*IncludeTriple*/ true);
}

std::string BareMetal::getCompilerRTPath() const {
  const Driver &D = getDriver();
  if (IsGCCInstallationValid || detectGCCToolchainAdjacent(getDriver())) {
    SmallString<128> Path(D.ResourceDir);
    llvm::sys::path::append(Path, "lib");
    return std::string(Path.str());
  }
  return ToolChain::getCompilerRTPath();
}

static void addMultilibsFilePaths(const Driver &D, const MultilibSet &Multilibs,
                                  const Multilib &Multilib,
                                  StringRef InstallPath,
                                  ToolChain::path_list &Paths) {
  if (const auto &PathsCallback = Multilibs.filePathsCallback())
    for (const auto &Path : PathsCallback(Multilib))
      addPathIfExists(D, InstallPath + Path, Paths);
}

// GCC mutltilibs will only work for those targets that have their multlib
// structure encoded into GCCInstallation. Baremetal toolchain supports ARM,
// AArch64, RISCV and PPC and of these only RISCV have GCC multilibs hardcoded
// in GCCInstallation.
BareMetal::BareMetal(const Driver &D, const llvm::Triple &Triple,
                     const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  IsGCCInstallationValid = initGCCInstallation(Triple, Args);
  std::string ComputedSysRoot = computeSysRoot();
  if (IsGCCInstallationValid) {
    if (!isRISCVBareMetal(Triple))
      D.Diag(clang::diag::warn_drv_multilib_not_available_for_target);

    Multilibs = GCCInstallation.getMultilibs();
    SelectedMultilibs.assign({GCCInstallation.getMultilib()});

    path_list &Paths = getFilePaths();
    // Add toolchain/multilib specific file paths.
    addMultilibsFilePaths(D, Multilibs, SelectedMultilibs.back(),
                          GCCInstallation.getInstallPath(), Paths);
    // Adding filepath for locating crt{begin,end}.o files.
    Paths.push_back(GCCInstallation.getInstallPath().str());
    // Adding filepath for locating crt0.o file.
    Paths.push_back(ComputedSysRoot + "/lib");

    ToolChain::path_list &PPaths = getProgramPaths();
    // Multilib cross-compiler GCC installations put ld in a triple-prefixed
    // directory off of the parent of the GCC installation.
    PPaths.push_back(Twine(GCCInstallation.getParentLibPath() + "/../" +
                           GCCInstallation.getTriple().str() + "/bin")
                         .str());
    PPaths.push_back((GCCInstallation.getParentLibPath() + "/../bin").str());
  } else {
    getProgramPaths().push_back(getDriver().Dir);
    findMultilibs(D, Triple, Args);
    const SmallString<128> SysRootDir(computeSysRoot());
    if (!SysRootDir.empty()) {
      for (const Multilib &M : getOrderedMultilibs()) {
        SmallString<128> Dir(SysRootDir);
        llvm::sys::path::append(Dir, M.osSuffix(), "lib");
        getFilePaths().push_back(std::string(Dir));
        getLibraryPaths().push_back(std::string(Dir));
      }
    }
  }
}

static void
findMultilibsFromYAML(const ToolChain &TC, const Driver &D,
                      StringRef MultilibPath, const ArgList &Args,
                      DetectedMultilibs &Result,
                      SmallVector<StringRef> &CustomFlagsMacroDefines) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MB =
      D.getVFS().getBufferForFile(MultilibPath);
  if (!MB)
    return;
  Multilib::flags_list Flags = TC.getMultilibFlags(Args);
  llvm::ErrorOr<MultilibSet> ErrorOrMultilibSet =
      MultilibSet::parseYaml(*MB.get());
  if (ErrorOrMultilibSet.getError())
    return;
  Result.Multilibs = ErrorOrMultilibSet.get();
  if (Result.Multilibs.select(D, Flags, Result.SelectedMultilibs,
                              &CustomFlagsMacroDefines))
    return;
  D.Diag(clang::diag::warn_drv_missing_multilib) << llvm::join(Flags, " ");
  std::stringstream ss;

  // If multilib selection didn't complete successfully, report a list
  // of all the configurations the user could have provided.
  for (const Multilib &Multilib : Result.Multilibs)
    if (!Multilib.isError())
      ss << "\n" << llvm::join(Multilib.flags(), " ");
  D.Diag(clang::diag::note_drv_available_multilibs) << ss.str();

  // Now report any custom error messages requested by the YAML. We do
  // this after displaying the list of available multilibs, because
  // that list is probably large, and (in interactive use) risks
  // scrolling the useful error message off the top of the user's
  // terminal.
  for (const Multilib &Multilib : Result.SelectedMultilibs)
    if (Multilib.isError())
      D.Diag(clang::diag::err_drv_multilib_custom_error)
          << Multilib.getErrorMessage();

  // If there was an error, clear the SelectedMultilibs vector, in
  // case it contains partial data.
  Result.SelectedMultilibs.clear();
}

static constexpr llvm::StringLiteral MultilibFilename = "multilib.yaml";

static std::optional<llvm::SmallString<128>>
getMultilibConfigPath(const Driver &D, const llvm::Triple &Triple,
                      const ArgList &Args) {
  llvm::SmallString<128> MultilibPath;
  if (Arg *ConfigFileArg = Args.getLastArg(options::OPT_multi_lib_config)) {
    MultilibPath = ConfigFileArg->getValue();
    if (!D.getVFS().exists(MultilibPath)) {
      D.Diag(clang::diag::err_drv_no_such_file) << MultilibPath.str();
      return {};
    }
  } else {
    MultilibPath = computeClangRuntimesSysRoot(D, /*IncludeTriple=*/false);
    llvm::sys::path::append(MultilibPath, MultilibFilename);
  }
  return MultilibPath;
}

void BareMetal::findMultilibs(const Driver &D, const llvm::Triple &Triple,
                              const ArgList &Args) {
  DetectedMultilibs Result;
  // Look for a multilib.yaml before trying target-specific hardwired logic.
  // If it exists, always do what it specifies.
  std::optional<llvm::SmallString<128>> MultilibPath =
      getMultilibConfigPath(D, Triple, Args);
  if (!MultilibPath)
    return;
  if (D.getVFS().exists(*MultilibPath)) {
    // If multilib.yaml is found, update sysroot so it doesn't use a target
    // specific suffix
    SysRoot = computeClangRuntimesSysRoot(D, /*IncludeTriple=*/false);
    SmallVector<StringRef> CustomFlagMacroDefines;
    findMultilibsFromYAML(*this, D, *MultilibPath, Args, Result,
                          CustomFlagMacroDefines);
    SelectedMultilibs = Result.SelectedMultilibs;
    Multilibs = Result.Multilibs;
    MultilibMacroDefines.append(CustomFlagMacroDefines.begin(),
                                CustomFlagMacroDefines.end());
  } else if (isRISCVBareMetal(Triple) && !detectGCCToolchainAdjacent(D)) {
    if (findRISCVMultilibs(D, Triple, Args, Result)) {
      SelectedMultilibs = Result.SelectedMultilibs;
      Multilibs = Result.Multilibs;
    }
  }
}

bool BareMetal::handlesTarget(const llvm::Triple &Triple) {
  return arm::isARMEABIBareMetal(Triple) ||
         aarch64::isAArch64BareMetal(Triple) || isRISCVBareMetal(Triple) ||
         isPPCBareMetal(Triple);
}

Tool *BareMetal::buildLinker() const {
  return new tools::baremetal::Linker(*this);
}

Tool *BareMetal::buildStaticLibTool() const {
  return new tools::baremetal::StaticLibTool(*this);
}

BareMetal::OrderedMultilibs BareMetal::getOrderedMultilibs() const {
  // Get multilibs in reverse order because they're ordered most-specific last.
  if (!SelectedMultilibs.empty())
    return llvm::reverse(SelectedMultilibs);

  // No multilibs selected so return a single default multilib.
  static const llvm::SmallVector<Multilib> Default = {Multilib()};
  return llvm::reverse(Default);
}

ToolChain::CXXStdlibType BareMetal::GetDefaultCXXStdlibType() const {
  if (getTriple().isRISCV() && IsGCCInstallationValid)
    return ToolChain::CST_Libstdcxx;
  return ToolChain::CST_Libcxx;
}

ToolChain::RuntimeLibType BareMetal::GetDefaultRuntimeLibType() const {
  if (getTriple().isRISCV() && IsGCCInstallationValid)
    return ToolChain::RLT_Libgcc;
  return ToolChain::RLT_CompilerRT;
}

// TODO: Add a validity check for GCCInstallation.
//       If valid, use `UNW_Libgcc`; otherwise, use `UNW_None`.
ToolChain::UnwindLibType
BareMetal::GetUnwindLibType(const llvm::opt::ArgList &Args) const {
  if (getTriple().isRISCV())
    return ToolChain::UNW_None;

  return ToolChain::GetUnwindLibType(Args);
}

void BareMetal::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                          ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> Dir(getDriver().ResourceDir);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (std::optional<std::string> Path = getStdlibIncludePath())
    addSystemInclude(DriverArgs, CC1Args, *Path);

  const SmallString<128> SysRootDir(computeSysRoot());
  if (!SysRootDir.empty()) {
    for (const Multilib &M : getOrderedMultilibs()) {
      SmallString<128> Dir(SysRootDir);
      llvm::sys::path::append(Dir, M.includeSuffix());
      llvm::sys::path::append(Dir, "include");
      addSystemInclude(DriverArgs, CC1Args, Dir.str());
    }
  }
}

void BareMetal::addClangTargetOptions(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args,
                                      Action::OffloadKind) const {
  CC1Args.push_back("-nostdsysteminc");
}

void BareMetal::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  if (!IsGCCInstallationValid)
    return;
  const GCCVersion &Version = GCCInstallation.getVersion();
  StringRef TripleStr = GCCInstallation.getTriple().str();
  const Multilib &Multilib = GCCInstallation.getMultilib();
  addLibStdCXXIncludePaths(computeSysRoot() + "/include/c++/" + Version.Text,
                           TripleStr, Multilib.includeSuffix(), DriverArgs,
                           CC1Args);
}

void BareMetal::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                             ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc, options::OPT_nostdlibinc,
                        options::OPT_nostdincxx))
    return;

  const Driver &D = getDriver();
  std::string Target = getTripleString();

  auto AddCXXIncludePath = [&](StringRef Path) {
    std::string Version = detectLibcxxVersion(Path);
    if (Version.empty())
      return;

    {
      // First the per-target include dir: include/<target>/c++/v1.
      SmallString<128> TargetDir(Path);
      llvm::sys::path::append(TargetDir, Target, "c++", Version);
      addSystemInclude(DriverArgs, CC1Args, TargetDir);
    }

    {
      // Then the generic dir: include/c++/v1.
      SmallString<128> Dir(Path);
      llvm::sys::path::append(Dir, "c++", Version);
      addSystemInclude(DriverArgs, CC1Args, Dir);
    }
  };

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx: {
    SmallString<128> P(D.Dir);
    llvm::sys::path::append(P, "..", "include");
    AddCXXIncludePath(P);
    break;
  }
  case ToolChain::CST_Libstdcxx:
    addLibStdCxxIncludePaths(DriverArgs, CC1Args);
    break;
  }

  std::string SysRootDir(computeSysRoot());
  if (SysRootDir.empty())
    return;

  for (const Multilib &M : getOrderedMultilibs()) {
    SmallString<128> Dir(SysRootDir);
    llvm::sys::path::append(Dir, M.gccSuffix());
    switch (GetCXXStdlibType(DriverArgs)) {
    case ToolChain::CST_Libcxx: {
      // First check sysroot/usr/include/c++/v1 if it exists.
      SmallString<128> TargetDir(Dir);
      llvm::sys::path::append(TargetDir, "usr", "include", "c++", "v1");
      if (D.getVFS().exists(TargetDir)) {
        addSystemInclude(DriverArgs, CC1Args, TargetDir.str());
        break;
      }
      // Add generic path if nothing else succeeded so far.
      llvm::sys::path::append(Dir, "include", "c++", "v1");
      addSystemInclude(DriverArgs, CC1Args, Dir.str());
      break;
    }
    case ToolChain::CST_Libstdcxx: {
      llvm::sys::path::append(Dir, "include", "c++");
      std::error_code EC;
      Generic_GCC::GCCVersion Version = {"", -1, -1, -1, "", "", ""};
      // Walk the subdirs, and find the one with the newest gcc version:
      for (llvm::vfs::directory_iterator
               LI = D.getVFS().dir_begin(Dir.str(), EC),
               LE;
           !EC && LI != LE; LI = LI.increment(EC)) {
        StringRef VersionText = llvm::sys::path::filename(LI->path());
        auto CandidateVersion = Generic_GCC::GCCVersion::Parse(VersionText);
        if (CandidateVersion.Major == -1)
          continue;
        if (CandidateVersion <= Version)
          continue;
        Version = CandidateVersion;
      }
      if (Version.Major != -1) {
        llvm::sys::path::append(Dir, Version.Text);
        addSystemInclude(DriverArgs, CC1Args, Dir.str());
      }
      break;
    }
    }
  }
}

void baremetal::StaticLibTool::ConstructJob(Compilation &C, const JobAction &JA,
                                            const InputInfo &Output,
                                            const InputInfoList &Inputs,
                                            const ArgList &Args,
                                            const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);
  // Silence warnings when linking C code with a C++ '-stdlib' argument.
  Args.ClaimAllArgs(options::OPT_stdlib_EQ);

  // ar tool command "llvm-ar <options> <output_file> <input_files>".
  ArgStringList CmdArgs;
  // Create and insert file members with a deterministic index.
  CmdArgs.push_back("rcsD");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs) {
    if (II.isFilename()) {
      CmdArgs.push_back(II.getFilename());
    }
  }

  // Delete old output archive file if it already exists before generating a new
  // archive file.
  const char *OutputFileName = Output.getFilename();
  if (Output.isFilename() && llvm::sys::fs::exists(OutputFileName)) {
    if (std::error_code EC = llvm::sys::fs::remove(OutputFileName)) {
      D.Diag(diag::err_drv_unable_to_remove_file) << EC.message();
      return;
    }
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetStaticLibToolPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

void baremetal::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  auto &TC = static_cast<const toolchains::BareMetal &>(getToolChain());
  const Driver &D = getToolChain().getDriver();
  const llvm::Triple::ArchType Arch = TC.getArch();
  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();
  const bool IsStaticPIE = getStaticPIE(Args, TC);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  CmdArgs.push_back("-Bstatic");
  if (IsStaticPIE) {
    CmdArgs.push_back("-pie");
    CmdArgs.push_back("--no-dynamic-linker");
    CmdArgs.push_back("-z");
    CmdArgs.push_back("text");
  }

  if (const char *LDMOption = getLDMOption(TC.getTriple(), Args)) {
    CmdArgs.push_back("-m");
    CmdArgs.push_back(LDMOption);
  } else {
    D.Diag(diag::err_target_unknown_triple) << Triple.str();
    return;
  }

  if (Triple.isRISCV()) {
    CmdArgs.push_back("-X");
    if (Args.hasArg(options::OPT_mno_relax))
      CmdArgs.push_back("--no-relax");
  }

  if (Triple.isARM() || Triple.isThumb()) {
    bool IsBigEndian = arm::isARMBigEndian(Triple, Args);
    if (IsBigEndian)
      arm::appendBE8LinkFlag(Args, CmdArgs, Triple);
    CmdArgs.push_back(IsBigEndian ? "-EB" : "-EL");
  } else if (Triple.isAArch64()) {
    CmdArgs.push_back(Arch == llvm::Triple::aarch64_be ? "-EB" : "-EL");
  }

  bool NeedCRTs =
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles);

  const char *CRTBegin, *CRTEnd;
  if (NeedCRTs) {
    if (!Args.hasArg(options::OPT_r)) {
      const char *crt = "crt0.o";
      if (IsStaticPIE)
        crt = "rcrt1.o";
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath(crt)));
    }
    if (TC.hasValidGCCInstallation() || detectGCCToolchainAdjacent(D)) {
      auto RuntimeLib = TC.GetRuntimeLibType(Args);
      switch (RuntimeLib) {
      case (ToolChain::RLT_Libgcc): {
        CRTBegin = IsStaticPIE ? "crtbeginS.o" : "crtbegin.o";
        CRTEnd = IsStaticPIE ? "crtendS.o" : "crtend.o";
        break;
      }
      case (ToolChain::RLT_CompilerRT): {
        CRTBegin =
            TC.getCompilerRTArgString(Args, "crtbegin", ToolChain::FT_Object);
        CRTEnd =
            TC.getCompilerRTArgString(Args, "crtend", ToolChain::FT_Object);
        break;
      }
      }
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath(CRTBegin)));
    }
  }

  Args.addAllArgs(CmdArgs,
                  {options::OPT_L, options::OPT_u, options::OPT_T_Group,
                   options::OPT_s, options::OPT_t, options::OPT_r});

  TC.AddFilePathLibArgs(Args, CmdArgs);

  for (const auto &LibPath : TC.getLibraryPaths())
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("-L", LibPath)));

  if (D.isUsingLTO())
    addLTOOptions(TC, Args, CmdArgs, Output, Inputs,
                  D.getLTOMode() == LTOK_Thin);

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  if (TC.ShouldLinkCXXStdlib(Args)) {
    bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                               !Args.hasArg(options::OPT_static);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
    CmdArgs.push_back("-lm");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    CmdArgs.push_back("--start-group");
    AddRunTimeLibs(TC, D, CmdArgs, Args);
    if (!Args.hasArg(options::OPT_nolibc))
      CmdArgs.push_back("-lc");
    if (TC.hasValidGCCInstallation() || detectGCCToolchainAdjacent(D))
      CmdArgs.push_back("-lgloss");
    CmdArgs.push_back("--end-group");
  }

  if ((TC.hasValidGCCInstallation() || detectGCCToolchainAdjacent(D)) &&
      NeedCRTs)
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath(CRTEnd)));

  // The R_ARM_TARGET2 relocation must be treated as R_ARM_REL32 on arm*-*-elf
  // and arm*-*-eabi (the default is R_ARM_GOT_PREL, used on arm*-*-linux and
  // arm*-*-*bsd).
  if (arm::isARMEABIBareMetal(TC.getTriple()))
    CmdArgs.push_back("--target2=rel");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(),
      Args.MakeArgString(TC.GetLinkerPath()), CmdArgs, Inputs, Output));
}

// BareMetal toolchain allows all sanitizers where the compiler generates valid
// code, ignoring all runtime library support issues on the assumption that
// baremetal targets typically implement their own runtime support.
SanitizerMask BareMetal::getSupportedSanitizers() const {
  const bool IsX86_64 = getTriple().getArch() == llvm::Triple::x86_64;
  const bool IsAArch64 = getTriple().getArch() == llvm::Triple::aarch64 ||
                         getTriple().getArch() == llvm::Triple::aarch64_be;
  const bool IsRISCV64 = getTriple().getArch() == llvm::Triple::riscv64;
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::KernelAddress;
  Res |= SanitizerKind::PointerCompare;
  Res |= SanitizerKind::PointerSubtract;
  Res |= SanitizerKind::Fuzzer;
  Res |= SanitizerKind::FuzzerNoLink;
  Res |= SanitizerKind::Vptr;
  Res |= SanitizerKind::SafeStack;
  Res |= SanitizerKind::Thread;
  Res |= SanitizerKind::Scudo;
  if (IsX86_64 || IsAArch64 || IsRISCV64) {
    Res |= SanitizerKind::HWAddress;
    Res |= SanitizerKind::KernelHWAddress;
  }
  return Res;
}

SmallVector<std::string>
BareMetal::getMultilibMacroDefinesStr(llvm::opt::ArgList &Args) const {
  return MultilibMacroDefines;
}
