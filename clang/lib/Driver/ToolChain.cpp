//===--- ToolChain.cpp - Collections of tools for one platform ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Tools.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

ToolChain::ToolChain(const Driver &D, const llvm::Triple &T,
                     const ArgList &A)
  : D(D), Triple(T), Args(A) {
}

ToolChain::~ToolChain() {
}

const Driver &ToolChain::getDriver() const {
 return D;
}

bool ToolChain::useIntegratedAs() const {
  return Args.hasFlag(options::OPT_integrated_as,
                      options::OPT_no_integrated_as,
                      IsIntegratedAssemblerDefault());
}

const SanitizerArgs& ToolChain::getSanitizerArgs() const {
  if (!SanitizerArguments.get())
    SanitizerArguments.reset(new SanitizerArgs(*this, Args));
  return *SanitizerArguments.get();
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
  case llvm::Triple::ppc64le:
    return "ppc64le";
  default:
    return Triple.getArchName();
  }
}

bool ToolChain::IsUnwindTablesDefault() const {
  return false;
}

Tool *ToolChain::getClang() const {
  if (!Clang)
    Clang.reset(new tools::Clang(*this));
  return Clang.get();
}

Tool *ToolChain::buildAssembler() const {
  return new tools::ClangAs(*this);
}

Tool *ToolChain::buildLinker() const {
  llvm_unreachable("Linking is not supported by this toolchain");
}

Tool *ToolChain::getAssemble() const {
  if (!Assemble)
    Assemble.reset(buildAssembler());
  return Assemble.get();
}

Tool *ToolChain::getClangAs() const {
  if (!Assemble)
    Assemble.reset(new tools::ClangAs(*this));
  return Assemble.get();
}

Tool *ToolChain::getLink() const {
  if (!Link)
    Link.reset(buildLinker());
  return Link.get();
}

Tool *ToolChain::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::AssembleJobClass:
    return getAssemble();

  case Action::LinkJobClass:
    return getLink();

  case Action::InputClass:
  case Action::BindArchClass:
  case Action::LipoJobClass:
  case Action::DsymutilJobClass:
  case Action::VerifyJobClass:
    llvm_unreachable("Invalid tool kind.");

  case Action::CompileJobClass:
  case Action::PrecompileJobClass:
  case Action::PreprocessJobClass:
  case Action::AnalyzeJobClass:
  case Action::MigrateJobClass:
    return getClang();
  }

  llvm_unreachable("Invalid tool kind.");
}

Tool *ToolChain::SelectTool(const JobAction &JA) const {
  if (getDriver().ShouldUseClangCompiler(JA))
    return getClang();
  Action::ActionClass AC = JA.getKind();
  if (AC == Action::AssembleJobClass && useIntegratedAs())
    return getClangAs();
  return getTool(AC);
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

  if (Triple.getOS() == llvm::Triple::NetBSD) {
    if (MArch == "armv6")
      return "arm1176jzf-s";
  }

  const char *result = llvm::StringSwitch<const char *>(MArch)
    .Cases("armv2", "armv2a","arm2")
    .Case("armv3", "arm6")
    .Case("armv3m", "arm7m")
    .Case("armv4", "strongarm")
    .Case("armv4t", "arm7tdmi")
    .Cases("armv5", "armv5t", "arm10tdmi")
    .Cases("armv5e", "armv5te", "arm1026ejs")
    .Case("armv5tej", "arm926ej-s")
    .Cases("armv6", "armv6k", "arm1136jf-s")
    .Case("armv6j", "arm1136j-s")
    .Cases("armv6z", "armv6zk", "arm1176jzf-s")
    .Case("armv6t2", "arm1156t2-s")
    .Cases("armv6m", "armv6-m", "cortex-m0")
    .Cases("armv7", "armv7a", "armv7-a", "cortex-a8")
    .Cases("armv7l", "armv7-l", "cortex-a8")
    .Cases("armv7f", "armv7-f", "cortex-a9-mp")
    .Cases("armv7s", "armv7-s", "swift")
    .Cases("armv7r", "armv7-r", "cortex-r4")
    .Cases("armv7m", "armv7-m", "cortex-m3")
    .Cases("armv7em", "armv7e-m", "cortex-m4")
    .Cases("armv8", "armv8a", "armv8-a", "cortex-a53")
    .Case("ep9312", "ep9312")
    .Case("iwmmxt", "iwmmxt")
    .Case("xscale", "xscale")
    // If all else failed, return the most base CPU with thumb interworking
    // supported by LLVM.
    .Default(0);

  if (result)
    return result;

  return
    Triple.getEnvironment() == llvm::Triple::GNUEABIHF
      ? "arm1176jzf-s"
      : "arm7tdmi";
}

/// getLLVMArchSuffixForARM - Get the LLVM arch name to use for a particular
/// CPU.
//
// FIXME: This is redundant with -mcpu, why does LLVM use this.
// FIXME: tblgen this, or kill it!
static const char *getLLVMArchSuffixForARM(StringRef CPU) {
  return llvm::StringSwitch<const char *>(CPU)
    .Case("strongarm", "v4")
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
    .Cases("cortex-a5", "cortex-a7", "cortex-a8", "v7")
    .Cases("cortex-a9", "cortex-a12", "cortex-a15", "v7")
    .Cases("cortex-r4", "cortex-r5", "v7r")
    .Case("cortex-m0", "v6m")
    .Case("cortex-m3", "v7m")
    .Case("cortex-m4", "v7em")
    .Case("cortex-a9-mp", "v7f")
    .Case("swift", "v7s")
    .Cases("cortex-a53", "cortex-a57", "v8")
    .Default("");
}

std::string ToolChain::ComputeLLVMTriple(const ArgList &Args,
                                         types::ID InputType) const {
  switch (getTriple().getArch()) {
  default:
    return getTripleString();

  case llvm::Triple::x86_64: {
    llvm::Triple Triple = getTriple();
    if (!Triple.isOSDarwin())
      return getTripleString();

    if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
      // x86_64h goes in the triple. Other -march options just use the
      // vanilla triple we already have.
      StringRef MArch = A->getValue();
      if (MArch == "x86_64h")
        Triple.setArchName(MArch);
    }
    return Triple.getTriple();
  }
  case llvm::Triple::arm:
  case llvm::Triple::thumb: {
    // FIXME: Factor into subclasses.
    llvm::Triple Triple = getTriple();

    // Thumb2 is the default for V7 on Darwin.
    //
    // FIXME: Thumb should just be another -target-feaure, not in the triple.
    StringRef Suffix =
      getLLVMArchSuffixForARM(getARMTargetCPU(Args, Triple));
    bool ThumbDefault = Suffix.startswith("v6m") ||
      (Suffix.startswith("v7") && getTriple().isOSDarwin());
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

void ToolChain::addClangTargetOptions(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
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

void ToolChain::addExternCSystemIncludeIfExists(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args,
                                                const Twine &Path) {
  if (llvm::sys::fs::exists(Path))
    addExternCSystemInclude(DriverArgs, CC1Args, Path);
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
