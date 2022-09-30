//===--- Fuchsia.cpp - Fuchsia ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Fuchsia.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

using tools::addMultilibFlag;

void fuchsia::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  const toolchains::Fuchsia &ToolChain =
      static_cast<const toolchains::Fuchsia &>(getToolChain());
  const Driver &D = ToolChain.getDriver();

  const llvm::Triple &Triple = ToolChain.getEffectiveTriple();

  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  CmdArgs.push_back("-z");
  CmdArgs.push_back("max-page-size=4096");

  CmdArgs.push_back("-z");
  CmdArgs.push_back("now");

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  if (llvm::sys::path::filename(Exec).equals_insensitive("ld.lld") ||
      llvm::sys::path::stem(Exec).equals_insensitive("ld.lld")) {
    CmdArgs.push_back("-z");
    CmdArgs.push_back("rodynamic");
    CmdArgs.push_back("-z");
    CmdArgs.push_back("separate-loadable-segments");
    CmdArgs.push_back("-z");
    CmdArgs.push_back("rel");
    CmdArgs.push_back("--pack-dyn-relocs=relr");
  } else {
    // The following are already the default in lld
    CmdArgs.push_back("-z");
    CmdArgs.push_back("combreloc");
    CmdArgs.push_back("-z");
    CmdArgs.push_back("text");
  }

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (!Args.hasArg(options::OPT_shared) && !Args.hasArg(options::OPT_r))
    CmdArgs.push_back("-pie");

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  if (Args.hasArg(options::OPT_r)) {
    CmdArgs.push_back("-r");
  } else {
    CmdArgs.push_back("--build-id");
    CmdArgs.push_back("--hash-style=gnu");
  }

  if (ToolChain.getArch() == llvm::Triple::aarch64) {
    CmdArgs.push_back("--execute-only");

    std::string CPU = getCPUName(D, Args, Triple);
    if (CPU.empty() || CPU == "generic" || CPU == "cortex-a53")
      CmdArgs.push_back("--fix-cortex-a53-843419");
  }

  CmdArgs.push_back("--eh-frame-hdr");

  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-Bstatic");
  else if (Args.hasArg(options::OPT_shared))
    CmdArgs.push_back("-shared");

  const SanitizerArgs &SanArgs = ToolChain.getSanitizerArgs(Args);

  if (!Args.hasArg(options::OPT_shared) && !Args.hasArg(options::OPT_r)) {
    std::string Dyld = D.DyldPrefix;
    if (SanArgs.needsAsanRt() && SanArgs.needsSharedRt())
      Dyld += "asan/";
    if (SanArgs.needsHwasanRt() && SanArgs.needsSharedRt())
      Dyld += "hwasan/";
    if (SanArgs.needsTsanRt() && SanArgs.needsSharedRt())
      Dyld += "tsan/";
    Dyld += "ld.so.1";
    CmdArgs.push_back("-dynamic-linker");
    CmdArgs.push_back(Args.MakeArgString(Dyld));
  }

  if (ToolChain.getArch() == llvm::Triple::riscv64)
    CmdArgs.push_back("-X");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("Scrt1.o")));
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (D.isUsingLTO()) {
    assert(!Inputs.empty() && "Must have at least one input.");
    addLTOOptions(ToolChain, Args, CmdArgs, Output, Inputs[0],
                  D.getLTOMode() == LTOK_Thin);
  }

  addLinkerCompressDebugSectionsOption(ToolChain, Args, CmdArgs);
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    if (Args.hasArg(options::OPT_static))
      CmdArgs.push_back("-Bdynamic");

    if (D.CCCIsCXX()) {
      if (ToolChain.ShouldLinkCXXStdlib(Args)) {
        bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                                   !Args.hasArg(options::OPT_static);
        CmdArgs.push_back("--push-state");
        CmdArgs.push_back("--as-needed");
        if (OnlyLibstdcxxStatic)
          CmdArgs.push_back("-Bstatic");
        ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
        if (OnlyLibstdcxxStatic)
          CmdArgs.push_back("-Bdynamic");
        CmdArgs.push_back("-lm");
        CmdArgs.push_back("--pop-state");
      }
    }

    // Note that Fuchsia never needs to link in sanitizer runtime deps.  Any
    // sanitizer runtimes with system dependencies use the `.deplibs` feature
    // instead.
    addSanitizerRuntimes(ToolChain, Args, CmdArgs);

    addXRayRuntime(ToolChain, Args, CmdArgs);

    ToolChain.addProfileRTLibs(Args, CmdArgs);

    AddRunTimeLibs(ToolChain, D, CmdArgs, Args);

    if (Args.hasArg(options::OPT_pthread) ||
        Args.hasArg(options::OPT_pthreads))
      CmdArgs.push_back("-lpthread");

    if (Args.hasArg(options::OPT_fsplit_stack))
      CmdArgs.push_back("--wrap=pthread_create");

    if (!Args.hasArg(options::OPT_nolibc))
      CmdArgs.push_back("-lc");
  }

  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Inputs, Output));
}

/// Fuchsia - Fuchsia tool chain which can call as(1) and ld(1) directly.

Fuchsia::Fuchsia(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != D.Dir)
    getProgramPaths().push_back(D.Dir);

  if (!D.SysRoot.empty()) {
    SmallString<128> P(D.SysRoot);
    llvm::sys::path::append(P, "lib");
    getFilePaths().push_back(std::string(P.str()));
  }

  auto FilePaths = [&](const Multilib &M) -> std::vector<std::string> {
    std::vector<std::string> FP;
    for (const std::string &Path : getStdlibPaths()) {
      SmallString<128> P(Path);
      llvm::sys::path::append(P, M.gccSuffix());
      FP.push_back(std::string(P.str()));
    }
    return FP;
  };

  Multilibs.push_back(Multilib());
  // Use the noexcept variant with -fno-exceptions to avoid the extra overhead.
  Multilibs.push_back(Multilib("noexcept", {}, {}, 1)
                          .flag("-fexceptions")
                          .flag("+fno-exceptions"));
  // ASan has higher priority because we always want the instrumentated version.
  Multilibs.push_back(Multilib("asan", {}, {}, 2)
                          .flag("+fsanitize=address"));
  // Use the asan+noexcept variant with ASan and -fno-exceptions.
  Multilibs.push_back(Multilib("asan+noexcept", {}, {}, 3)
                          .flag("+fsanitize=address")
                          .flag("-fexceptions")
                          .flag("+fno-exceptions"));
  // HWASan has higher priority because we always want the instrumentated
  // version.
  Multilibs.push_back(
      Multilib("hwasan", {}, {}, 4).flag("+fsanitize=hwaddress"));
  // Use the hwasan+noexcept variant with HWASan and -fno-exceptions.
  Multilibs.push_back(Multilib("hwasan+noexcept", {}, {}, 5)
                          .flag("+fsanitize=hwaddress")
                          .flag("-fexceptions")
                          .flag("+fno-exceptions"));
  // Use the relative vtables ABI.
  // TODO: Remove these multilibs once relative vtables are enabled by default
  // for Fuchsia.
  Multilibs.push_back(Multilib("relative-vtables", {}, {}, 6)
                          .flag("+fexperimental-relative-c++-abi-vtables"));
  Multilibs.push_back(Multilib("relative-vtables+noexcept", {}, {}, 7)
                          .flag("+fexperimental-relative-c++-abi-vtables")
                          .flag("-fexceptions")
                          .flag("+fno-exceptions"));
  Multilibs.push_back(Multilib("relative-vtables+asan", {}, {}, 8)
                          .flag("+fexperimental-relative-c++-abi-vtables")
                          .flag("+fsanitize=address"));
  Multilibs.push_back(Multilib("relative-vtables+asan+noexcept", {}, {}, 9)
                          .flag("+fexperimental-relative-c++-abi-vtables")
                          .flag("+fsanitize=address")
                          .flag("-fexceptions")
                          .flag("+fno-exceptions"));
  Multilibs.push_back(Multilib("relative-vtables+hwasan", {}, {}, 10)
                          .flag("+fexperimental-relative-c++-abi-vtables")
                          .flag("+fsanitize=hwaddress"));
  Multilibs.push_back(Multilib("relative-vtables+hwasan+noexcept", {}, {}, 11)
                          .flag("+fexperimental-relative-c++-abi-vtables")
                          .flag("+fsanitize=hwaddress")
                          .flag("-fexceptions")
                          .flag("+fno-exceptions"));
  // Use Itanium C++ ABI for the compat multilib.
  Multilibs.push_back(Multilib("compat", {}, {}, 12).flag("+fc++-abi=itanium"));

  Multilibs.FilterOut([&](const Multilib &M) {
    std::vector<std::string> RD = FilePaths(M);
    return llvm::all_of(RD, [&](std::string P) { return !getVFS().exists(P); });
  });

  Multilib::flags_list Flags;
  addMultilibFlag(
      Args.hasFlag(options::OPT_fexceptions, options::OPT_fno_exceptions, true),
      "fexceptions", Flags);
  addMultilibFlag(getSanitizerArgs(Args).needsAsanRt(), "fsanitize=address",
                  Flags);
  addMultilibFlag(getSanitizerArgs(Args).needsHwasanRt(), "fsanitize=hwaddress",
                  Flags);

  addMultilibFlag(
      Args.hasFlag(options::OPT_fexperimental_relative_cxx_abi_vtables,
                   options::OPT_fno_experimental_relative_cxx_abi_vtables,
                   /*default=*/false),
      "fexperimental-relative-c++-abi-vtables", Flags);
  addMultilibFlag(Args.getLastArgValue(options::OPT_fcxx_abi_EQ) == "itanium",
                  "fc++-abi=itanium", Flags);

  Multilibs.setFilePathsCallback(FilePaths);

  if (Multilibs.select(Flags, SelectedMultilib))
    if (!SelectedMultilib.isDefault())
      if (const auto &PathsCallback = Multilibs.filePathsCallback())
        for (const auto &Path : PathsCallback(SelectedMultilib))
          // Prepend the multilib path to ensure it takes the precedence.
          getFilePaths().insert(getFilePaths().begin(), Path);
}

std::string Fuchsia::ComputeEffectiveClangTriple(const ArgList &Args,
                                                 types::ID InputType) const {
  llvm::Triple Triple(ComputeLLVMTriple(Args, InputType));
  return Triple.str();
}

Tool *Fuchsia::buildLinker() const {
  return new tools::fuchsia::Linker(*this);
}

ToolChain::RuntimeLibType Fuchsia::GetRuntimeLibType(
    const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(clang::driver::options::OPT_rtlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value != "compiler-rt")
      getDriver().Diag(clang::diag::err_drv_invalid_rtlib_name)
          << A->getAsString(Args);
  }

  return ToolChain::RLT_CompilerRT;
}

ToolChain::CXXStdlibType
Fuchsia::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value != "libc++")
      getDriver().Diag(diag::err_drv_invalid_stdlib_name)
        << A->getAsString(Args);
  }

  return ToolChain::CST_Libcxx;
}

void Fuchsia::addClangTargetOptions(const ArgList &DriverArgs,
                                    ArgStringList &CC1Args,
                                    Action::OffloadKind) const {
  if (!DriverArgs.hasFlag(options::OPT_fuse_init_array,
                          options::OPT_fno_use_init_array, true))
    CC1Args.push_back("-fno-use-init-array");
}

void Fuchsia::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                        ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? "" : StringRef(D.SysRoot);
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  if (!D.SysRoot.empty()) {
    SmallString<128> P(D.SysRoot);
    llvm::sys::path::append(P, "include");
    addExternCSystemInclude(DriverArgs, CC1Args, P.str());
  }
}

void Fuchsia::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  const Driver &D = getDriver();
  std::string Target = getTripleString();

  auto AddCXXIncludePath = [&](StringRef Path) {
    std::string Version = detectLibcxxVersion(Path);
    if (Version.empty())
      return;

    // First add the per-target include path.
    SmallString<128> TargetDir(Path);
    llvm::sys::path::append(TargetDir, Target, "c++", Version);
    if (getVFS().exists(TargetDir))
      addSystemInclude(DriverArgs, CC1Args, TargetDir);

    // Second add the generic one.
    SmallString<128> Dir(Path);
    llvm::sys::path::append(Dir, "c++", Version);
    addSystemInclude(DriverArgs, CC1Args, Dir);
  };

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx: {
    SmallString<128> P(D.Dir);
    llvm::sys::path::append(P, "..", "include");
    AddCXXIncludePath(P);
    break;
  }

  default:
    llvm_unreachable("invalid stdlib name");
  }
}

void Fuchsia::AddCXXStdlibLibArgs(const ArgList &Args,
                                  ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    if (Args.hasArg(options::OPT_fexperimental_library))
      CmdArgs.push_back("-lc++experimental");
    break;

  case ToolChain::CST_Libstdcxx:
    llvm_unreachable("invalid stdlib name");
  }
}

SanitizerMask Fuchsia::getSupportedSanitizers() const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::HWAddress;
  Res |= SanitizerKind::PointerCompare;
  Res |= SanitizerKind::PointerSubtract;
  Res |= SanitizerKind::Fuzzer;
  Res |= SanitizerKind::FuzzerNoLink;
  Res |= SanitizerKind::Leak;
  Res |= SanitizerKind::SafeStack;
  Res |= SanitizerKind::Scudo;
  Res |= SanitizerKind::Thread;
  return Res;
}

SanitizerMask Fuchsia::getDefaultSanitizers() const {
  SanitizerMask Res;
  switch (getTriple().getArch()) {
  case llvm::Triple::aarch64:
    Res |= SanitizerKind::ShadowCallStack;
    break;
  case llvm::Triple::x86_64:
    Res |= SanitizerKind::SafeStack;
    break;
  default:
    // TODO: Enable SafeStack on RISC-V once tested.
    break;
  }
  return Res;
}
