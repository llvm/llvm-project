//===--- ToolChain.cpp - Collections of tools for one platform ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/ToolChain.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/HostInfo.h"
#include "clang/Driver/ObjCRuntime.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang::driver;
using namespace clang;

ToolChain::ToolChain(const HostInfo &H, const llvm::Triple &T)
  : Host(H), Triple(T) {
}

ToolChain::~ToolChain() {
}

const Driver &ToolChain::getDriver() const {
 return Host.getDriver();
}

std::string ToolChain::GetFilePath(const char *Name) const {
  return Host.getDriver().GetFilePath(Name, *this);

}

std::string ToolChain::GetProgramPath(const char *Name, bool WantFile) const {
  return Host.getDriver().GetProgramPath(Name, *this, WantFile);
}

types::ID ToolChain::LookupTypeForExtension(const char *Ext) const {
  return types::lookupTypeForExtension(Ext);
}

bool ToolChain::HasNativeLLVMSupport() const {
  return false;
}

void ToolChain::configureObjCRuntime(ObjCRuntime &runtime) const {
  switch (runtime.getKind()) {
  case ObjCRuntime::NeXT:
    // Assume a minimal NeXT runtime.
    runtime.HasARC = false;
    runtime.HasWeak = false;
    runtime.HasTerminate = false;
    return;

  case ObjCRuntime::GNU:
    // Assume a maximal GNU runtime.
    runtime.HasARC = true;
    runtime.HasWeak = true;
    runtime.HasTerminate = false; // to be added
    return;
  }
  llvm_unreachable("invalid runtime kind!");
}

/// getARMTargetCPU - Get the (LLVM) name of the ARM cpu we are targeting.
//
// FIXME: tblgen this.
static const char *getARMTargetCPU(const ArgList &Args,
                                   const llvm::Triple &Triple) {
  // FIXME: Warn on inconsistent use of -mcpu and -march.

  // If we have -mcpu=, use that.
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    return A->getValue(Args);

  StringRef MArch;
  if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    // Otherwise, if we have -march= choose the base CPU for that arch.
    MArch = A->getValue(Args);
  } else {
    // Otherwise, use the Arch from the triple.
    MArch = Triple.getArchName();
  }

  if (MArch == "armv2" || MArch == "armv2a")
    return "arm2";
  if (MArch == "armv3")
    return "arm6";
  if (MArch == "armv3m")
    return "arm7m";
  if (MArch == "armv4" || MArch == "armv4t")
    return "arm7tdmi";
  if (MArch == "armv5" || MArch == "armv5t")
    return "arm10tdmi";
  if (MArch == "armv5e" || MArch == "armv5te")
    return "arm1026ejs";
  if (MArch == "armv5tej")
    return "arm926ej-s";
  if (MArch == "armv6" || MArch == "armv6k")
    return "arm1136jf-s";
  if (MArch == "armv6j")
    return "arm1136j-s";
  if (MArch == "armv6z" || MArch == "armv6zk")
    return "arm1176jzf-s";
  if (MArch == "armv6t2")
    return "arm1156t2-s";
  if (MArch == "armv7" || MArch == "armv7a" || MArch == "armv7-a")
    return "cortex-a8";
  if (MArch == "armv7r" || MArch == "armv7-r")
    return "cortex-r4";
  if (MArch == "armv7m" || MArch == "armv7-m")
    return "cortex-m3";
  if (MArch == "ep9312")
    return "ep9312";
  if (MArch == "iwmmxt")
    return "iwmmxt";
  if (MArch == "xscale")
    return "xscale";
  if (MArch == "armv6m" || MArch == "armv6-m")
    return "cortex-m0";

  // If all else failed, return the most base CPU LLVM supports.
  return "arm7tdmi";
}

/// getLLVMArchSuffixForARM - Get the LLVM arch name to use for a particular
/// CPU.
//
// FIXME: This is redundant with -mcpu, why does LLVM use this.
// FIXME: tblgen this, or kill it!
static const char *getLLVMArchSuffixForARM(StringRef CPU) {
  if (CPU == "arm7tdmi" || CPU == "arm7tdmi-s" || CPU == "arm710t" ||
      CPU == "arm720t" || CPU == "arm9" || CPU == "arm9tdmi" ||
      CPU == "arm920" || CPU == "arm920t" || CPU == "arm922t" ||
      CPU == "arm940t" || CPU == "ep9312")
    return "v4t";

  if (CPU == "arm10tdmi" || CPU == "arm1020t")
    return "v5";

  if (CPU == "arm9e" || CPU == "arm926ej-s" || CPU == "arm946e-s" ||
      CPU == "arm966e-s" || CPU == "arm968e-s" || CPU == "arm10e" ||
      CPU == "arm1020e" || CPU == "arm1022e" || CPU == "xscale" ||
      CPU == "iwmmxt")
    return "v5e";

  if (CPU == "arm1136j-s" || CPU == "arm1136jf-s" || CPU == "arm1176jz-s" ||
      CPU == "arm1176jzf-s" || CPU == "mpcorenovfp" || CPU == "mpcore")
    return "v6";

  if (CPU == "arm1156t2-s" || CPU == "arm1156t2f-s")
    return "v6t2";

  if (CPU == "cortex-a8" || CPU == "cortex-a9")
    return "v7";

  if (CPU == "cortex-m3")
    return "v7m";

  if (CPU == "cortex-m0")
    return "v6m";

  return "";
}

std::string ToolChain::ComputeLLVMTriple(const ArgList &Args, 
                                         types::ID InputType) const {
  switch (getTriple().getArch()) {
  default:
    return getTripleString();

  case llvm::Triple::arm:
  case llvm::Triple::thumb: {
    // FIXME: Factor into subclasses.
    llvm::Triple Triple = getTriple();

    // Thumb2 is the default for V7 on Darwin.
    //
    // FIXME: Thumb should just be another -target-feaure, not in the triple.
    StringRef Suffix =
      getLLVMArchSuffixForARM(getARMTargetCPU(Args, Triple));
    bool ThumbDefault = (Suffix == "v7" && getTriple().isOSDarwin());
    std::string ArchName = "arm";

    // Assembly files should start in ARM mode.
    if (InputType != types::TY_PP_Asm &&
        Args.hasFlag(options::OPT_mthumb, options::OPT_mno_thumb, ThumbDefault))
      ArchName = "thumb";
    Triple.setArchName(ArchName + Suffix.str());

    return Triple.getTriple();
  }
  }
}

std::string ToolChain::ComputeEffectiveClangTriple(const ArgList &Args, 
                                                   types::ID InputType) const {
  // Diagnose use of Darwin OS deployment target arguments on non-Darwin.
  if (Arg *A = Args.getLastArg(options::OPT_mmacosx_version_min_EQ,
                               options::OPT_miphoneos_version_min_EQ,
                               options::OPT_mios_simulator_version_min_EQ))
    getDriver().Diag(diag::err_drv_clang_unsupported)
      << A->getAsString(Args);

  return ComputeLLVMTriple(Args, InputType);
}

void ToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                          ArgStringList &CC1Args) const {
  // Each toolchain should provide the appropriate include flags.
}

ToolChain::CXXStdlibType ToolChain::GetCXXStdlibType(const ArgList &Args) const{
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue(Args);
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    getDriver().Diag(diag::err_drv_invalid_stdlib_name)
      << A->getAsString(Args);
  }

  return ToolChain::CST_Libstdcxx;
}

void ToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                             ArgStringList &CC1Args) const {
  // Header search paths should be handled by each of the subclasses.
  // Historically, they have not been, and instead have been handled inside of
  // the CC1-layer frontend. As the logic is hoisted out, this generic function
  // will slowly stop being called.
  //
  // While it is being called, replicate a bit of a hack to propagate the
  // '-stdlib=' flag down to CC1 so that it can in turn customize the C++
  // header search paths with it. Once all systems are overriding this
  // function, the CC1 flag and this line can be removed.
  DriverArgs.AddAllArgs(CC1Args, options::OPT_stdlib_EQ);
}

void ToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  CXXStdlibType Type = GetCXXStdlibType(Args);

  switch (Type) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    break;

  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    break;
  }
}

void ToolChain::AddCCKextLibArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-lcc_kext");
}
