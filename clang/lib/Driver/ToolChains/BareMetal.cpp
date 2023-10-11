//===-- BareMetal.cpp - Bare Metal ToolChain --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BareMetal.h"

#include "CommonArgs.h"
#include "Gnu.h"
#include "clang/Driver/InputInfo.h"

#include "Arch/ARM.h"
#include "Arch/RISCV.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/MultilibBuilder.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

using namespace llvm::opt;
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;

static bool findRISCVMultilibs(const Driver &D,
                               const llvm::Triple &TargetTriple,
                               const ArgList &Args, DetectedMultilibs &Result) {
  Multilib::flags_list Flags;
  StringRef Arch = riscv::getRISCVArch(Args, TargetTriple);
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
    return Result.Multilibs.select(Flags, Result.SelectedMultilibs);
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
    return Result.Multilibs.select(Flags, Result.SelectedMultilibs);
  }
  return false;
}

BareMetal::BareMetal(const Driver &D, const llvm::Triple &Triple,
                     const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  findMultilibs(D, Triple, Args);
  SmallString<128> SysRoot(computeSysRoot());
  if (!SysRoot.empty()) {
    for (const Multilib &M : getOrderedMultilibs()) {
      SmallString<128> Dir(SysRoot);
      llvm::sys::path::append(Dir, M.osSuffix(), "lib");
      getFilePaths().push_back(std::string(Dir));
      getLibraryPaths().push_back(std::string(Dir));
    }
  }
}

/// Is the triple {arm,armeb,thumb,thumbeb}-none-none-{eabi,eabihf} ?
static bool isARMBareMetal(const llvm::Triple &Triple) {
  if (Triple.getArch() != llvm::Triple::arm &&
      Triple.getArch() != llvm::Triple::thumb &&
      Triple.getArch() != llvm::Triple::armeb &&
      Triple.getArch() != llvm::Triple::thumbeb)
    return false;

  if (Triple.getVendor() != llvm::Triple::UnknownVendor)
    return false;

  if (Triple.getOS() != llvm::Triple::UnknownOS)
    return false;

  if (Triple.getEnvironment() != llvm::Triple::EABI &&
      Triple.getEnvironment() != llvm::Triple::EABIHF)
    return false;

  return true;
}

/// Is the triple {aarch64.aarch64_be}-none-elf?
static bool isAArch64BareMetal(const llvm::Triple &Triple) {
  if (Triple.getArch() != llvm::Triple::aarch64 &&
      Triple.getArch() != llvm::Triple::aarch64_be)
    return false;

  if (Triple.getVendor() != llvm::Triple::UnknownVendor)
    return false;

  if (Triple.getOS() != llvm::Triple::UnknownOS)
    return false;

  return Triple.getEnvironmentName() == "elf";
}

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

static void findMultilibsFromYAML(const ToolChain &TC, const Driver &D,
                                  StringRef MultilibPath, const ArgList &Args,
                                  DetectedMultilibs &Result) {
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
  if (Result.Multilibs.select(Flags, Result.SelectedMultilibs))
    return;
  D.Diag(clang::diag::warn_drv_missing_multilib) << llvm::join(Flags, " ");
  std::stringstream ss;
  for (const Multilib &Multilib : Result.Multilibs)
    ss << "\n" << llvm::join(Multilib.flags(), " ");
  D.Diag(clang::diag::note_drv_available_multilibs) << ss.str();
}

static constexpr llvm::StringLiteral MultilibFilename = "multilib.yaml";

// Get the sysroot, before multilib takes effect.
static std::string computeBaseSysRoot(const Driver &D,
                                      const llvm::Triple &Triple) {
  if (!D.SysRoot.empty())
    return D.SysRoot;

  SmallString<128> SysRootDir(D.Dir);
  llvm::sys::path::append(SysRootDir, "..", "lib", "clang-runtimes");

  SmallString<128> MultilibPath(SysRootDir);
  llvm::sys::path::append(MultilibPath, MultilibFilename);

  // New behaviour: if multilib.yaml is found then use clang-runtimes as the
  // sysroot.
  if (D.getVFS().exists(MultilibPath))
    return std::string(SysRootDir);

  // Otherwise fall back to the old behaviour of appending the target triple.
  llvm::sys::path::append(SysRootDir, D.getTargetTriple());
  return std::string(SysRootDir);
}

void BareMetal::findMultilibs(const Driver &D, const llvm::Triple &Triple,
                              const ArgList &Args) {
  DetectedMultilibs Result;
  if (isRISCVBareMetal(Triple)) {
    if (findRISCVMultilibs(D, Triple, Args, Result)) {
      SelectedMultilibs = Result.SelectedMultilibs;
      Multilibs = Result.Multilibs;
    }
  } else {
    llvm::SmallString<128> MultilibPath(computeBaseSysRoot(D, Triple));
    llvm::sys::path::append(MultilibPath, MultilibFilename);
    findMultilibsFromYAML(*this, D, MultilibPath, Args, Result);
    SelectedMultilibs = Result.SelectedMultilibs;
    Multilibs = Result.Multilibs;
  }
}

bool BareMetal::handlesTarget(const llvm::Triple &Triple) {
  return isARMBareMetal(Triple) || isAArch64BareMetal(Triple) ||
         isRISCVBareMetal(Triple) || isPPCBareMetal(Triple);
}

Tool *BareMetal::buildLinker() const {
  return new tools::baremetal::Linker(*this);
}

Tool *BareMetal::buildStaticLibTool() const {
  return new tools::baremetal::StaticLibTool(*this);
}

std::string BareMetal::computeSysRoot() const {
  return computeBaseSysRoot(getDriver(), getTriple());
}

BareMetal::OrderedMultilibs BareMetal::getOrderedMultilibs() const {
  // Get multilibs in reverse order because they're ordered most-specific last.
  if (!SelectedMultilibs.empty())
    return llvm::reverse(SelectedMultilibs);

  // No multilibs selected so return a single default multilib.
  static const llvm::SmallVector<Multilib> Default = {Multilib()};
  return llvm::reverse(Default);
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

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc)) {
    const SmallString<128> SysRoot(computeSysRoot());
    if (!SysRoot.empty()) {
      for (const Multilib &M : getOrderedMultilibs()) {
        SmallString<128> Dir(SysRoot);
        llvm::sys::path::append(Dir, M.includeSuffix());
        llvm::sys::path::append(Dir, "include");
        addSystemInclude(DriverArgs, CC1Args, Dir.str());
      }
    }
  }
}

void BareMetal::addClangTargetOptions(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args,
                                      Action::OffloadKind) const {
  CC1Args.push_back("-nostdsysteminc");
}

void BareMetal::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                             ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  const Driver &D = getDriver();
  std::string SysRoot(computeSysRoot());
  if (SysRoot.empty())
    return;

  for (const Multilib &M : getOrderedMultilibs()) {
    SmallString<128> Dir(SysRoot);
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

void BareMetal::AddCXXStdlibLibArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    if (Args.hasArg(options::OPT_fexperimental_library))
      CmdArgs.push_back("-lc++experimental");
    CmdArgs.push_back("-lc++abi");
    break;
  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    CmdArgs.push_back("-lsupc++");
    break;
  }
  CmdArgs.push_back("-lunwind");
}

void BareMetal::AddLinkRuntimeLib(const ArgList &Args,
                                  ArgStringList &CmdArgs) const {
  ToolChain::RuntimeLibType RLT = GetRuntimeLibType(Args);
  switch (RLT) {
  case ToolChain::RLT_CompilerRT: {
    const std::string FileName = getCompilerRT(Args, "builtins");
    llvm::StringRef BaseName = llvm::sys::path::filename(FileName);
    BaseName.consume_front("lib");
    BaseName.consume_back(".a");
    CmdArgs.push_back(Args.MakeArgString("-l" + BaseName));
    return;
  }
  case ToolChain::RLT_Libgcc:
    CmdArgs.push_back("-lgcc");
    return;
  }
  llvm_unreachable("Unhandled RuntimeLibType.");
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
  const llvm::Triple::ArchType Arch = TC.getArch();
  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  CmdArgs.push_back("-Bstatic");

  if (Triple.isARM() || Triple.isThumb()) {
    bool IsBigEndian = arm::isARMBigEndian(Triple, Args);
    if (IsBigEndian)
      arm::appendBE8LinkFlag(Args, CmdArgs, Triple);
    CmdArgs.push_back(IsBigEndian ? "-EB" : "-EL");
  } else if (Triple.isAArch64()) {
    CmdArgs.push_back(Arch == llvm::Triple::aarch64_be ? "-EB" : "-EL");
  }

  Args.addAllArgs(CmdArgs,
                  {options::OPT_L, options::OPT_T_Group, options::OPT_s,
                   options::OPT_t, options::OPT_Z_Flag, options::OPT_r});

  TC.AddFilePathLibArgs(Args, CmdArgs);

  for (const auto &LibPath : TC.getLibraryPaths())
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("-L", LibPath)));

  const std::string FileName = TC.getCompilerRT(Args, "builtins");
  llvm::SmallString<128> PathBuf{FileName};
  llvm::sys::path::remove_filename(PathBuf);
  CmdArgs.push_back(Args.MakeArgString("-L" + PathBuf));

  if (TC.ShouldLinkCXXStdlib(Args))
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lm");

    TC.AddLinkRuntimeLib(Args, CmdArgs);
  }

  if (TC.getTriple().isRISCV())
    CmdArgs.push_back("-X");

  // The R_ARM_TARGET2 relocation must be treated as R_ARM_REL32 on arm*-*-elf
  // and arm*-*-eabi (the default is R_ARM_GOT_PREL, used on arm*-*-linux and
  // arm*-*-*bsd).
  if (isARMBareMetal(TC.getTriple()))
    CmdArgs.push_back("--target2=rel");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(),
      Args.MakeArgString(TC.GetLinkerPath()), CmdArgs, Inputs, Output));
}
