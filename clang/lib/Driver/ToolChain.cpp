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
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "clang/Basic/ObjCRuntime.h"
using namespace clang::driver;
using namespace clang;

ToolChain::ToolChain(const Driver &D, const llvm::Triple &T)
  : D(D), Triple(T) {
}

ToolChain::~ToolChain() {
}

const Driver &ToolChain::getDriver() const {
 return D;
}

std::string ToolChain::getDefaultUniversalArchName() const {
  // In universal driver terms, the arch name accepted by -arch isn't exactly
  // the same as the ones that appear in the triple. Roughly speaking, this is
  // an inverse of the darwin::getArchTypeForDarwinArchName() function, but the
  // only interesting special case is powerpc.
  switch (Triple.getArch()) {
  case llvm::Triple::ppc:
    return "ppc";
  case llvm::Triple::ppc64:
    return "ppc64";
  default:
    return Triple.getArchName();
  }
}

bool ToolChain::IsUnwindTablesDefault() const {
  return false;
}

std::string ToolChain::GetFilePath(const char *Name) const {
  return D.GetFilePath(Name, *this);

}

std::string ToolChain::GetProgramPath(const char *Name) const {
  return D.GetProgramPath(Name, *this);
}

types::ID ToolChain::LookupTypeForExtension(const char *Ext) const {
  return types::lookupTypeForExtension(Ext);
}

bool ToolChain::HasNativeLLVMSupport() const {
  return false;
}

ObjCRuntime ToolChain::getDefaultObjCRuntime(bool isNonFragile) const {
  return ObjCRuntime(isNonFragile ? ObjCRuntime::GNUstep : ObjCRuntime::GCC,
                     VersionTuple());
}

/// getARMTargetCPU - Get the (LLVM) name of the ARM cpu we are targeting.
//
// FIXME: tblgen this.
static const char *getARMTargetCPU(const ArgList &Args,
                                   const llvm::Triple &Triple) {
  // For Darwin targets, the -arch option (which is translated to a
  // corresponding -march option) should determine the architecture
  // (and the Mach-O slice) regardless of any -mcpu options.
  if (!Triple.isOSDarwin()) {
    // FIXME: Warn on inconsistent use of -mcpu and -march.
    // If we have -mcpu=, use that.
    if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      return A->getValue();
  }

  StringRef MArch;
  if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    // Otherwise, if we have -march= choose the base CPU for that arch.
    MArch = A->getValue();
  } else {
    // Otherwise, use the Arch from the triple.
    MArch = Triple.getArchName();
  }

  return llvm::StringSwitch<const char *>(MArch)
    .Cases("armv2", "armv2a","arm2")
    .Case("armv3", "arm6")
    .Case("armv3m", "arm7m")
    .Cases("armv4", "armv4t", "arm7tdmi")
    .Cases("armv5", "armv5t", "arm10tdmi")
    .Cases("armv5e", "armv5te", "arm1026ejs")
    .Case("armv5tej", "arm926ej-s")
    .Cases("armv6", "armv6k", "arm1136jf-s")
    .Case("armv6j", "arm1136j-s")
    .Cases("armv6z", "armv6zk", "arm1176jzf-s")
    .Case("armv6t2", "arm1156t2-s")
    .Cases("armv7", "armv7a", "armv7-a", "cortex-a8")
    .Cases("armv7f", "armv7-f", "cortex-a9-mp")
    .Cases("armv7s", "armv7-s", "swift")
    .Cases("armv7r", "armv7-r", "cortex-r4")
    .Cases("armv7m", "armv7-m", "cortex-m3")
    .Case("ep9312", "ep9312")
    .Case("iwmmxt", "iwmmxt")
    .Case("xscale", "xscale")
    .Cases("armv6m", "armv6-m", "cortex-m0")
    // If all else failed, return the most base CPU LLVM supports.
    .Default("arm7tdmi");
}

/// getLLVMArchSuffixForARM - Get the LLVM arch name to use for a particular
/// CPU.
//
// FIXME: This is redundant with -mcpu, why does LLVM use this.
// FIXME: tblgen this, or kill it!
static const char *getLLVMArchSuffixForARM(StringRef CPU) {
  return llvm::StringSwitch<const char *>(CPU)
    .Cases("arm7tdmi", "arm7tdmi-s", "arm710t", "v4t")
    .Cases("arm720t", "arm9", "arm9tdmi", "v4t")
    .Cases("arm920", "arm920t", "arm922t", "v4t")
    .Cases("arm940t", "ep9312","v4t")
    .Cases("arm10tdmi",  "arm1020t", "v5")
    .Cases("arm9e",  "arm926ej-s",  "arm946e-s", "v5e")
    .Cases("arm966e-s",  "arm968e-s",  "arm10e", "v5e")
    .Cases("arm1020e",  "arm1022e",  "xscale", "iwmmxt", "v5e")
    .Cases("arm1136j-s",  "arm1136jf-s",  "arm1176jz-s", "v6")
    .Cases("arm1176jzf-s",  "mpcorenovfp",  "mpcore", "v6")
    .Cases("arm1156t2-s",  "arm1156t2f-s", "v6t2")
    .Cases("cortex-a8", "cortex-a9", "cortex-a15", "v7")
    .Case("cortex-m3", "v7m")
    .Case("cortex-m4", "v7m")
    .Case("cortex-m0", "v6m")
    .Case("cortex-a9-mp", "v7f")
    .Case("swift", "v7s")
    .Default("");
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
    bool ThumbDefault = (Suffix.startswith("v7") && getTriple().isOSDarwin());
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

void ToolChain::addClangTargetOptions(ArgStringList &CC1Args) const {
}

ToolChain::RuntimeLibType ToolChain::GetRuntimeLibType(
  const ArgList &Args) const
{
  if (Arg *A = Args.getLastArg(options::OPT_rtlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "compiler-rt")
      return ToolChain::RLT_CompilerRT;
    if (Value == "libgcc")
      return ToolChain::RLT_Libgcc;
    getDriver().Diag(diag::err_drv_invalid_rtlib_name)
      << A->getAsString(Args);
  }

  return GetDefaultRuntimeLibType();
}

ToolChain::CXXStdlibType ToolChain::GetCXXStdlibType(const ArgList &Args) const{
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    getDriver().Diag(diag::err_drv_invalid_stdlib_name)
      << A->getAsString(Args);
  }

  return ToolChain::CST_Libstdcxx;
}

/// \brief Utility function to add a system include directory to CC1 arguments.
/*static*/ void ToolChain::addSystemInclude(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args,
                                            const Twine &Path) {
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(Path));
}

/// \brief Utility function to add a system include directory with extern "C"
/// semantics to CC1 arguments.
///
/// Note that this should be used rarely, and only for directories that
/// historically and for legacy reasons are treated as having implicit extern
/// "C" semantics. These semantics are *ignored* by and large today, but its
/// important to preserve the preprocessor changes resulting from the
/// classification.
/*static*/ void ToolChain::addExternCSystemInclude(const ArgList &DriverArgs,
                                                   ArgStringList &CC1Args,
                                                   const Twine &Path) {
  CC1Args.push_back("-internal-externc-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(Path));
}

/// \brief Utility function to add a list of system include directories to CC1.
/*static*/ void ToolChain::addSystemIncludes(const ArgList &DriverArgs,
                                             ArgStringList &CC1Args,
                                             ArrayRef<StringRef> Paths) {
  for (ArrayRef<StringRef>::iterator I = Paths.begin(), E = Paths.end();
       I != E; ++I) {
    CC1Args.push_back("-internal-isystem");
    CC1Args.push_back(DriverArgs.MakeArgString(*I));
  }
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

bool ToolChain::AddFastMathRuntimeIfAvailable(const ArgList &Args,
                                              ArgStringList &CmdArgs) const {
  // Check if -ffast-math or -funsafe-math is enabled.
  Arg *A = Args.getLastArg(options::OPT_ffast_math,
                           options::OPT_fno_fast_math,
                           options::OPT_funsafe_math_optimizations,
                           options::OPT_fno_unsafe_math_optimizations);

  if (!A || A->getOption().getID() == options::OPT_fno_fast_math ||
      A->getOption().getID() == options::OPT_fno_unsafe_math_optimizations)
    return false;

  // If crtfastmath.o exists add it to the arguments.
  std::string Path = GetFilePath("crtfastmath.o");
  if (Path == "crtfastmath.o") // Not found.
    return false;

  CmdArgs.push_back(Args.MakeArgString(Path));
  return true;
}
