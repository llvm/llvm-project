//===--- Targets.cpp - Implement -arch option and targets -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements construction of a TargetInfo object from a
// target triple.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Type.h"
#include <algorithm>
using namespace clang;

//===----------------------------------------------------------------------===//
//  Common code shared among targets.
//===----------------------------------------------------------------------===//

/// DefineStd - Define a macro name and standard variants.  For example if
/// MacroName is "unix", then this will define "__unix", "__unix__", and "unix"
/// when in GNU mode.
static void DefineStd(MacroBuilder &Builder, StringRef MacroName,
                      const LangOptions &Opts) {
  assert(MacroName[0] != '_' && "Identifier should be in the user's namespace");

  // If in GNU mode (e.g. -std=gnu99 but not -std=c99) define the raw identifier
  // in the user's namespace.
  if (Opts.GNUMode)
    Builder.defineMacro(MacroName);

  // Define __unix.
  Builder.defineMacro("__" + MacroName);

  // Define __unix__.
  Builder.defineMacro("__" + MacroName + "__");
}

static void defineCPUMacros(MacroBuilder &Builder, StringRef CPUName,
                            bool Tuning = true) {
  Builder.defineMacro("__" + CPUName);
  Builder.defineMacro("__" + CPUName + "__");
  if (Tuning)
    Builder.defineMacro("__tune_" + CPUName + "__");
}

//===----------------------------------------------------------------------===//
// Defines specific to certain operating systems.
//===----------------------------------------------------------------------===//

namespace {
template<typename TgtInfo>
class OSTargetInfo : public TgtInfo {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const=0;
public:
  OSTargetInfo(const std::string& triple) : TgtInfo(triple) {}
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    TgtInfo::getTargetDefines(Opts, Builder);
    getOSDefines(Opts, TgtInfo::getTriple(), Builder);
  }

};
} // end anonymous namespace


static void getDarwinDefines(MacroBuilder &Builder, const LangOptions &Opts,
                             const llvm::Triple &Triple,
                             StringRef &PlatformName,
                             VersionTuple &PlatformMinVersion) {
  Builder.defineMacro("__APPLE_CC__", "5621");
  Builder.defineMacro("__APPLE__");
  Builder.defineMacro("__MACH__");
  Builder.defineMacro("OBJC_NEW_PROPERTIES");

  if (!Opts.ObjCAutoRefCount) {
    // __weak is always defined, for use in blocks and with objc pointers.
    Builder.defineMacro("__weak", "__attribute__((objc_gc(weak)))");

    // Darwin defines __strong even in C mode (just to nothing).
    if (Opts.getGC() != LangOptions::NonGC)
      Builder.defineMacro("__strong", "__attribute__((objc_gc(strong)))");
    else
      Builder.defineMacro("__strong", "");

    // __unsafe_unretained is defined to nothing in non-ARC mode. We even
    // allow this in C, since one might have block pointers in structs that
    // are used in pure C code and in Objective-C ARC.
    Builder.defineMacro("__unsafe_unretained", "");
  }

  if (Opts.Static)
    Builder.defineMacro("__STATIC__");
  else
    Builder.defineMacro("__DYNAMIC__");

  if (Opts.POSIXThreads)
    Builder.defineMacro("_REENTRANT");

  // Get the platform type and version number from the triple.
  unsigned Maj, Min, Rev;
  if (Triple.isMacOSX()) {
    Triple.getMacOSXVersion(Maj, Min, Rev);
    PlatformName = "macosx";
  } else {
    Triple.getOSVersion(Maj, Min, Rev);
    PlatformName = llvm::Triple::getOSTypeName(Triple.getOS());
  }

  // If -target arch-pc-win32-macho option specified, we're
  // generating code for Win32 ABI. No need to emit
  // __ENVIRONMENT_XX_OS_VERSION_MIN_REQUIRED__.
  if (PlatformName == "win32") {
    PlatformMinVersion = VersionTuple(Maj, Min, Rev);
    return;
  }

  // Set the appropriate OS version define.
  if (Triple.getOS() == llvm::Triple::IOS) {
    assert(Maj < 10 && Min < 100 && Rev < 100 && "Invalid version!");
    char Str[6];
    Str[0] = '0' + Maj;
    Str[1] = '0' + (Min / 10);
    Str[2] = '0' + (Min % 10);
    Str[3] = '0' + (Rev / 10);
    Str[4] = '0' + (Rev % 10);
    Str[5] = '\0';
    Builder.defineMacro("__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__", Str);
  } else {
    // Note that the Driver allows versions which aren't representable in the
    // define (because we only get a single digit for the minor and micro
    // revision numbers). So, we limit them to the maximum representable
    // version.
    assert(Triple.getEnvironmentName().empty() && "Invalid environment!");
    assert(Maj < 100 && Min < 100 && Rev < 100 && "Invalid version!");
    char Str[5];
    Str[0] = '0' + (Maj / 10);
    Str[1] = '0' + (Maj % 10);
    Str[2] = '0' + std::min(Min, 9U);
    Str[3] = '0' + std::min(Rev, 9U);
    Str[4] = '\0';
    Builder.defineMacro("__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__", Str);
  }

  PlatformMinVersion = VersionTuple(Maj, Min, Rev);
}

namespace {
template<typename Target>
class DarwinTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    getDarwinDefines(Builder, Opts, Triple, this->PlatformName,
                     this->PlatformMinVersion);
  }

public:
  DarwinTargetInfo(const std::string& triple) :
    OSTargetInfo<Target>(triple) {
      llvm::Triple T = llvm::Triple(triple);
      this->TLSSupported = T.isMacOSX() && !T.isMacOSXVersionLT(10,7);
      this->MCountName = "\01mcount";
    }

  virtual std::string isValidSectionSpecifier(StringRef SR) const {
    // Let MCSectionMachO validate this.
    StringRef Segment, Section;
    unsigned TAA, StubSize;
    bool HasTAA;
    return llvm::MCSectionMachO::ParseSectionSpecifier(SR, Segment, Section,
                                                       TAA, HasTAA, StubSize);
  }

  virtual const char *getStaticInitSectionSpecifier() const {
    // FIXME: We should return 0 when building kexts.
    return "__TEXT,__StaticInit,regular,pure_instructions";
  }

  /// Darwin does not support protected visibility.  Darwin's "default"
  /// is very similar to ELF's "protected";  Darwin requires a "weak"
  /// attribute on declarations that can be dynamically replaced.
  virtual bool hasProtectedVisibility() const {
    return false;
  }
};


// DragonFlyBSD Target
template<typename Target>
class DragonFlyBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // DragonFly defines; list based off of gcc output
    Builder.defineMacro("__DragonFly__");
    Builder.defineMacro("__DragonFly_cc_version", "100001");
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__KPRINTF_ATTRIBUTE__");
    Builder.defineMacro("__tune_i386__");
    DefineStd(Builder, "unix", Opts);
  }
public:
  DragonFlyBSDTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";

      llvm::Triple Triple(triple);
      switch (Triple.getArch()) {
        default:
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
          this->MCountName = ".mcount";
          break;
      }
  }
};

// FreeBSD Target
template<typename Target>
class FreeBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // FreeBSD defines; list based off of gcc output

    unsigned Release = Triple.getOSMajorVersion();
    if (Release == 0U)
      Release = 8;

    Builder.defineMacro("__FreeBSD__", Twine(Release));
    Builder.defineMacro("__FreeBSD_cc_version", Twine(Release * 100000U + 1U));
    Builder.defineMacro("__KPRINTF_ATTRIBUTE__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
  }
public:
  FreeBSDTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";

      llvm::Triple Triple(triple);
      switch (Triple.getArch()) {
        default:
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
          this->MCountName = ".mcount";
          break;
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
        case llvm::Triple::ppc:
        case llvm::Triple::ppc64:
          this->MCountName = "_mcount";
          break;
        case llvm::Triple::arm:
          this->MCountName = "__mcount";
          break;
      }

    }
};

// Minix Target
template<typename Target>
class MinixTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // Minix defines

    Builder.defineMacro("__minix", "3");
    Builder.defineMacro("_EM_WSIZE", "4");
    Builder.defineMacro("_EM_PSIZE", "4");
    Builder.defineMacro("_EM_SSIZE", "2");
    Builder.defineMacro("_EM_LSIZE", "4");
    Builder.defineMacro("_EM_FSIZE", "4");
    Builder.defineMacro("_EM_DSIZE", "8");
    Builder.defineMacro("__ELF__");
    DefineStd(Builder, "unix", Opts);
  }
public:
  MinixTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";
    }
};

// Linux target
template<typename Target>
class LinuxTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // Linux defines; list based off of gcc output
    DefineStd(Builder, "unix", Opts);
    DefineStd(Builder, "linux", Opts);
    Builder.defineMacro("__gnu_linux__");
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
  }
public:
  LinuxTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
    this->WIntType = TargetInfo::UnsignedInt;
  }

  virtual const char *getStaticInitSectionSpecifier() const {
    return ".text.startup";
  }
};

// NetBSD Target
template<typename Target>
class NetBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // NetBSD defines; list based off of gcc output
    Builder.defineMacro("__NetBSD__");
    Builder.defineMacro("__unix__");
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_POSIX_THREADS");
  }
public:
  NetBSDTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";
    }
};

// OpenBSD Target
template<typename Target>
class OpenBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // OpenBSD defines; list based off of gcc output

    Builder.defineMacro("__OpenBSD__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
  }
public:
  OpenBSDTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";

      llvm::Triple Triple(triple);
      switch (Triple.getArch()) {
        default:
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
        case llvm::Triple::arm:
	case llvm::Triple::sparc:
          this->MCountName = "__mcount";
          break;
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
        case llvm::Triple::ppc:
	case llvm::Triple::sparcv9:
          this->MCountName = "_mcount";
          break;
      }
  }
};

// PSP Target
template<typename Target>
class PSPTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // PSP defines; list based on the output of the pspdev gcc toolchain.
    Builder.defineMacro("PSP");
    Builder.defineMacro("_PSP");
    Builder.defineMacro("__psp__");
    Builder.defineMacro("__ELF__");
  }
public:
  PSPTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
  }
};

// PS3 PPU Target
template<typename Target>
class PS3PPUTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // PS3 PPU defines.
    Builder.defineMacro("__PPC__");
    Builder.defineMacro("__PPU__");
    Builder.defineMacro("__CELLOS_LV2__");
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__LP32__");
    Builder.defineMacro("_ARCH_PPC64");
    Builder.defineMacro("__powerpc64__");
  }
public:
  PS3PPUTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
    this->LongWidth = this->LongAlign = 32;
    this->PointerWidth = this->PointerAlign = 32;
    this->IntMaxType = TargetInfo::SignedLongLong;
    this->UIntMaxType = TargetInfo::UnsignedLongLong;
    this->Int64Type = TargetInfo::SignedLongLong;
    this->SizeType = TargetInfo::UnsignedInt;
    this->DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                              "i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32";
  }
};

// FIXME: Need a real SPU target.
// PS3 SPU Target
template<typename Target>
class PS3SPUTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // PS3 PPU defines.
    Builder.defineMacro("__SPU__");
    Builder.defineMacro("__ELF__");
  }
public:
  PS3SPUTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
  }
};

// AuroraUX target
template<typename Target>
class AuroraUXTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    DefineStd(Builder, "sun", Opts);
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__svr4__");
    Builder.defineMacro("__SVR4");
  }
public:
  AuroraUXTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
    this->WCharType = this->SignedLong;
    // FIXME: WIntType should be SignedLong
  }
};

// Solaris target
template<typename Target>
class SolarisTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    DefineStd(Builder, "sun", Opts);
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__svr4__");
    Builder.defineMacro("__SVR4");
    // Solaris headers require _XOPEN_SOURCE to be set to 600 for C99 and
    // newer, but to 500 for everything else.  feature_test.h has a check to
    // ensure that you are not using C99 with an old version of X/Open or C89
    // with a new version.  
    if (Opts.C99 || Opts.C11)
      Builder.defineMacro("_XOPEN_SOURCE", "600");
    else
      Builder.defineMacro("_XOPEN_SOURCE", "500");
    if (Opts.CPlusPlus)
      Builder.defineMacro("__C99FEATURES__");
    Builder.defineMacro("_LARGEFILE_SOURCE");
    Builder.defineMacro("_LARGEFILE64_SOURCE");
    Builder.defineMacro("__EXTENSIONS__");
    Builder.defineMacro("_REENTRANT");
  }
public:
  SolarisTargetInfo(const std::string& triple)
    : OSTargetInfo<Target>(triple) {
    this->UserLabelPrefix = "";
    this->WCharType = this->SignedInt;
    // FIXME: WIntType should be SignedLong
  }
};

// Windows target
template<typename Target>
class WindowsTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    Builder.defineMacro("_WIN32");
  }
  void getVisualStudioDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    if (Opts.CPlusPlus) {
      if (Opts.RTTI)
        Builder.defineMacro("_CPPRTTI");

      if (Opts.Exceptions)
        Builder.defineMacro("_CPPUNWIND");
    }

    if (!Opts.CharIsSigned)
      Builder.defineMacro("_CHAR_UNSIGNED");

    // FIXME: POSIXThreads isn't exactly the option this should be defined for,
    //        but it works for now.
    if (Opts.POSIXThreads)
      Builder.defineMacro("_MT");

    if (Opts.MSCVersion != 0)
      Builder.defineMacro("_MSC_VER", Twine(Opts.MSCVersion));

    if (Opts.MicrosoftExt) {
      Builder.defineMacro("_MSC_EXTENSIONS");

      if (Opts.CPlusPlus0x) {
        Builder.defineMacro("_RVALUE_REFERENCES_V2_SUPPORTED");
        Builder.defineMacro("_RVALUE_REFERENCES_SUPPORTED");
        Builder.defineMacro("_NATIVE_NULLPTR_SUPPORTED");
      }
    }

    Builder.defineMacro("_INTEGRAL_MAX_BITS", "64");
  }

public:
  WindowsTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {}
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// Specific target implementations.
//===----------------------------------------------------------------------===//

namespace {
// PPC abstract base class
class PPCTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
  static const char * const GCCRegNames[];
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
public:
  PPCTargetInfo(const std::string& triple) : TargetInfo(triple) {
    LongDoubleWidth = LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::PPCDoubleDouble;
  }

  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = BuiltinInfo;
    NumRecords = clang::PPC::LastTSBuiltin-Builtin::FirstTSBuiltin;
  }

  virtual bool isCLZForZeroUndef() const { return false; }

  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const;

  virtual bool hasFeature(StringRef Feature) const;
  
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
    switch (*Name) {
    default: return false;
    case 'O': // Zero
      break;
    case 'b': // Base register
    case 'f': // Floating point register
      Info.setAllowsRegister();
      break;
    // FIXME: The following are added to allow parsing.
    // I just took a guess at what the actions should be.
    // Also, is more specific checking needed?  I.e. specific registers?
    case 'd': // Floating point register (containing 64-bit value)
    case 'v': // Altivec vector register
      Info.setAllowsRegister();
      break;
    case 'w':
      switch (Name[1]) {
        case 'd':// VSX vector register to hold vector double data
        case 'f':// VSX vector register to hold vector float data
        case 's':// VSX vector register to hold scalar float data
        case 'a':// Any VSX register
          break;
        default:
          return false;
      }
      Info.setAllowsRegister();
      Name++; // Skip over 'w'.
      break;
    case 'h': // `MQ', `CTR', or `LINK' register
    case 'q': // `MQ' register
    case 'c': // `CTR' register
    case 'l': // `LINK' register
    case 'x': // `CR' register (condition register) number 0
    case 'y': // `CR' register (condition register)
    case 'z': // `XER[CA]' carry bit (part of the XER register)
      Info.setAllowsRegister();
      break;
    case 'I': // Signed 16-bit constant
    case 'J': // Unsigned 16-bit constant shifted left 16 bits
              //  (use `L' instead for SImode constants)
    case 'K': // Unsigned 16-bit constant
    case 'L': // Signed 16-bit constant shifted left 16 bits
    case 'M': // Constant larger than 31
    case 'N': // Exact power of 2
    case 'P': // Constant whose negation is a signed 16-bit constant
    case 'G': // Floating point constant that can be loaded into a
              // register with one instruction per word
    case 'H': // Integer/Floating point constant that can be loaded
              // into a register using three instructions
      break;
    case 'm': // Memory operand. Note that on PowerPC targets, m can
              // include addresses that update the base register. It
              // is therefore only safe to use `m' in an asm statement
              // if that asm statement accesses the operand exactly once.
              // The asm statement must also use `%U<opno>' as a
              // placeholder for the "update" flag in the corresponding
              // load or store instruction. For example:
              // asm ("st%U0 %1,%0" : "=m" (mem) : "r" (val));
              // is correct but:
              // asm ("st %1,%0" : "=m" (mem) : "r" (val));
              // is not. Use es rather than m if you don't want the base
              // register to be updated.
    case 'e':
      if (Name[1] != 's')
          return false;
              // es: A "stable" memory operand; that is, one which does not
              // include any automodification of the base register. Unlike
              // `m', this constraint can be used in asm statements that
              // might access the operand several times, or that might not
              // access it at all.
      Info.setAllowsMemory();
      Name++; // Skip over 'e'.
      break;
    case 'Q': // Memory operand that is an offset from a register (it is
              // usually better to use `m' or `es' in asm statements)
    case 'Z': // Memory operand that is an indexed or indirect from a
              // register (it is usually better to use `m' or `es' in
              // asm statements)
      Info.setAllowsMemory();
      Info.setAllowsRegister();
      break;
    case 'R': // AIX TOC entry
    case 'a': // Address operand that is an indexed or indirect from a
              // register (`p' is preferable for asm statements)
    case 'S': // Constant suitable as a 64-bit mask operand
    case 'T': // Constant suitable as a 32-bit mask operand
    case 'U': // System V Release 4 small data area reference
    case 't': // AND masks that can be performed by two rldic{l, r}
              // instructions
    case 'W': // Vector constant that does not require memory
    case 'j': // Vector constant that is all zeros.
      break;
    // End FIXME.
    }
    return true;
  }
  virtual const char *getClobbers() const {
    return "";
  }
};

const Builtin::Info PPCTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, ALL_LANGUAGES },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER,\
                                              ALL_LANGUAGES },
#include "clang/Basic/BuiltinsPPC.def"
};


/// PPCTargetInfo::getTargetDefines - Return a set of the PowerPC-specific
/// #defines that are not tied to a specific subtarget.
void PPCTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  Builder.defineMacro("__ppc__");
  Builder.defineMacro("_ARCH_PPC");
  Builder.defineMacro("__powerpc__");
  Builder.defineMacro("__POWERPC__");
  if (PointerWidth == 64) {
    Builder.defineMacro("_ARCH_PPC64");
    Builder.defineMacro("_LP64");
    Builder.defineMacro("__LP64__");
    Builder.defineMacro("__powerpc64__");
    Builder.defineMacro("__ppc64__");
  } else {
    Builder.defineMacro("__ppc__");
  }

  // Target properties.
  if (getTriple().getOS() != llvm::Triple::NetBSD)
    Builder.defineMacro("_BIG_ENDIAN");
  Builder.defineMacro("__BIG_ENDIAN__");

  // Subtarget options.
  Builder.defineMacro("__NATURAL_ALIGNMENT__");
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // FIXME: Should be controlled by command line option.
  Builder.defineMacro("__LONG_DOUBLE_128__");

  if (Opts.AltiVec) {
    Builder.defineMacro("__VEC__", "10206");
    Builder.defineMacro("__ALTIVEC__");
  }
}

bool PPCTargetInfo::hasFeature(StringRef Feature) const {
  return Feature == "powerpc";
}

  
const char * const PPCTargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
  "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",
  "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
  "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
  "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
  "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
  "mq", "lr", "ctr", "ap",
  "cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7",
  "xer",
  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
  "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
  "vrsave", "vscr",
  "spe_acc", "spefscr",
  "sfp"
};

void PPCTargetInfo::getGCCRegNames(const char * const *&Names,
                                   unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias PPCTargetInfo::GCCRegAliases[] = {
  // While some of these aliases do map to different registers
  // they still share the same register name.
  { { "0" }, "r0" },
  { { "1"}, "r1" },
  { { "2" }, "r2" },
  { { "3" }, "r3" },
  { { "4" }, "r4" },
  { { "5" }, "r5" },
  { { "6" }, "r6" },
  { { "7" }, "r7" },
  { { "8" }, "r8" },
  { { "9" }, "r9" },
  { { "10" }, "r10" },
  { { "11" }, "r11" },
  { { "12" }, "r12" },
  { { "13" }, "r13" },
  { { "14" }, "r14" },
  { { "15" }, "r15" },
  { { "16" }, "r16" },
  { { "17" }, "r17" },
  { { "18" }, "r18" },
  { { "19" }, "r19" },
  { { "20" }, "r20" },
  { { "21" }, "r21" },
  { { "22" }, "r22" },
  { { "23" }, "r23" },
  { { "24" }, "r24" },
  { { "25" }, "r25" },
  { { "26" }, "r26" },
  { { "27" }, "r27" },
  { { "28" }, "r28" },
  { { "29" }, "r29" },
  { { "30" }, "r30" },
  { { "31" }, "r31" },
  { { "fr0" }, "f0" },
  { { "fr1" }, "f1" },
  { { "fr2" }, "f2" },
  { { "fr3" }, "f3" },
  { { "fr4" }, "f4" },
  { { "fr5" }, "f5" },
  { { "fr6" }, "f6" },
  { { "fr7" }, "f7" },
  { { "fr8" }, "f8" },
  { { "fr9" }, "f9" },
  { { "fr10" }, "f10" },
  { { "fr11" }, "f11" },
  { { "fr12" }, "f12" },
  { { "fr13" }, "f13" },
  { { "fr14" }, "f14" },
  { { "fr15" }, "f15" },
  { { "fr16" }, "f16" },
  { { "fr17" }, "f17" },
  { { "fr18" }, "f18" },
  { { "fr19" }, "f19" },
  { { "fr20" }, "f20" },
  { { "fr21" }, "f21" },
  { { "fr22" }, "f22" },
  { { "fr23" }, "f23" },
  { { "fr24" }, "f24" },
  { { "fr25" }, "f25" },
  { { "fr26" }, "f26" },
  { { "fr27" }, "f27" },
  { { "fr28" }, "f28" },
  { { "fr29" }, "f29" },
  { { "fr30" }, "f30" },
  { { "fr31" }, "f31" },
  { { "cc" }, "cr0" },
};

void PPCTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                     unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}
} // end anonymous namespace.

namespace {
class PPC32TargetInfo : public PPCTargetInfo {
public:
  PPC32TargetInfo(const std::string &triple) : PPCTargetInfo(triple) {
    DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32";

    switch (getTriple().getOS()) {
    case llvm::Triple::Linux:
    case llvm::Triple::FreeBSD:
    case llvm::Triple::NetBSD:
      SizeType = UnsignedInt;
      PtrDiffType = SignedInt;
      IntPtrType = SignedInt;
      break;
    default:
      break;
    }

    if (getTriple().getOS() == llvm::Triple::FreeBSD) {
      LongDoubleWidth = LongDoubleAlign = 64;
      LongDoubleFormat = &llvm::APFloat::IEEEdouble;
    }
  }

  virtual const char *getVAListDeclaration() const {
    // This is the ELF definition, and is overridden by the Darwin sub-target
    return "typedef struct __va_list_tag {"
           "  unsigned char gpr;"
           "  unsigned char fpr;"
           "  unsigned short reserved;"
           "  void* overflow_arg_area;"
           "  void* reg_save_area;"
           "} __builtin_va_list[1];";
  }
};
} // end anonymous namespace.

namespace {
class PPC64TargetInfo : public PPCTargetInfo {
public:
  PPC64TargetInfo(const std::string& triple) : PPCTargetInfo(triple) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    IntMaxType = SignedLong;
    UIntMaxType = UnsignedLong;
    Int64Type = SignedLong;
    DescriptionString = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64";

    if (getTriple().getOS() == llvm::Triple::FreeBSD) {
      LongDoubleWidth = LongDoubleAlign = 64;
      LongDoubleFormat = &llvm::APFloat::IEEEdouble;
    }
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
};
} // end anonymous namespace.


namespace {
class DarwinPPC32TargetInfo :
  public DarwinTargetInfo<PPC32TargetInfo> {
public:
  DarwinPPC32TargetInfo(const std::string& triple)
    : DarwinTargetInfo<PPC32TargetInfo>(triple) {
    HasAlignMac68kSupport = true;
    BoolWidth = BoolAlign = 32; //XXX support -mone-byte-bool?
    LongLongAlign = 32;
    SuitableAlign = 128;
    DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:64:64-v128:128:128-n32";
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
};

class DarwinPPC64TargetInfo :
  public DarwinTargetInfo<PPC64TargetInfo> {
public:
  DarwinPPC64TargetInfo(const std::string& triple)
    : DarwinTargetInfo<PPC64TargetInfo>(triple) {
    HasAlignMac68kSupport = true;
    SuitableAlign = 128;
  }
};
} // end anonymous namespace.

namespace {
  static const unsigned PTXAddrSpaceMap[] = {
    0,    // opencl_global
    4,    // opencl_local
    1     // opencl_constant
  };
  class PTXTargetInfo : public TargetInfo {
    static const char * const GCCRegNames[];
    static const Builtin::Info BuiltinInfo[];
    std::vector<llvm::StringRef> AvailableFeatures;
  public:
    PTXTargetInfo(const std::string& triple) : TargetInfo(triple) {
      BigEndian = false;
      TLSSupported = false;
      LongWidth = LongAlign = 64;
      AddrSpaceMap = &PTXAddrSpaceMap;
      // Define available target features
      // These must be defined in sorted order!      
      AvailableFeatures.push_back("compute10");
      AvailableFeatures.push_back("compute11");
      AvailableFeatures.push_back("compute12");
      AvailableFeatures.push_back("compute13");
      AvailableFeatures.push_back("compute20");
      AvailableFeatures.push_back("double");
      AvailableFeatures.push_back("no-fma");
      AvailableFeatures.push_back("ptx20");
      AvailableFeatures.push_back("ptx21");
      AvailableFeatures.push_back("ptx22");
      AvailableFeatures.push_back("ptx23");
      AvailableFeatures.push_back("sm10");
      AvailableFeatures.push_back("sm11");
      AvailableFeatures.push_back("sm12");
      AvailableFeatures.push_back("sm13");
      AvailableFeatures.push_back("sm20");
      AvailableFeatures.push_back("sm21");
      AvailableFeatures.push_back("sm22");
      AvailableFeatures.push_back("sm23");
    }
    virtual void getTargetDefines(const LangOptions &Opts,
                                  MacroBuilder &Builder) const {
      Builder.defineMacro("__PTX__");
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {
      Records = BuiltinInfo;
      NumRecords = clang::PTX::LastTSBuiltin-Builtin::FirstTSBuiltin;
    }
    virtual bool hasFeature(StringRef Feature) const {
      return Feature == "ptx";
    }
    
    virtual void getGCCRegNames(const char * const *&Names,
                                unsigned &NumNames) const;
    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                  unsigned &NumAliases) const {
      // No aliases.
      Aliases = 0;
      NumAliases = 0;
    }
    virtual bool validateAsmConstraint(const char *&Name,
                                       TargetInfo::ConstraintInfo &info) const {
      // FIXME: implement
      return true;
    }
    virtual const char *getClobbers() const {
      // FIXME: Is this really right?
      return "";
    }
    virtual const char *getVAListDeclaration() const {
      // FIXME: implement
      return "typedef char* __builtin_va_list;";
    }

    virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                   StringRef Name,
                                   bool Enabled) const;
  };

  const Builtin::Info PTXTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, ALL_LANGUAGES },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER,\
                                              ALL_LANGUAGES },
#include "clang/Basic/BuiltinsPTX.def"
  };

  const char * const PTXTargetInfo::GCCRegNames[] = {
    "r0"
  };

  void PTXTargetInfo::getGCCRegNames(const char * const *&Names,
                                     unsigned &NumNames) const {
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }

  bool PTXTargetInfo::setFeatureEnabled(llvm::StringMap<bool> &Features,
                                        StringRef Name,
                                        bool Enabled) const {
    if(std::binary_search(AvailableFeatures.begin(), AvailableFeatures.end(),
                          Name)) {
      Features[Name] = Enabled;
      return true;
    } else {
      return false;
    }
  }

  class PTX32TargetInfo : public PTXTargetInfo {
  public:
  PTX32TargetInfo(const std::string& triple) : PTXTargetInfo(triple) {
      PointerWidth = PointerAlign = 32;
      SizeType = PtrDiffType = IntPtrType = TargetInfo::UnsignedInt;
      DescriptionString
        = "e-p:32:32-i64:64:64-f64:64:64-n1:8:16:32:64";
    }
  };

  class PTX64TargetInfo : public PTXTargetInfo {
  public:
  PTX64TargetInfo(const std::string& triple) : PTXTargetInfo(triple) {
      PointerWidth = PointerAlign = 64;
      SizeType = PtrDiffType = IntPtrType = TargetInfo::UnsignedLongLong;
      DescriptionString
        = "e-p:64:64-i64:64:64-f64:64:64-n1:8:16:32:64";
    }
  };
}

namespace {
// MBlaze abstract base class
class MBlazeTargetInfo : public TargetInfo {
  static const char * const GCCRegNames[];
  static const TargetInfo::GCCRegAlias GCCRegAliases[];

public:
  MBlazeTargetInfo(const std::string& triple) : TargetInfo(triple) {
    DescriptionString = "E-p:32:32:32-i8:8:8-i16:16:16";
  }

  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    // FIXME: Implement.
    Records = 0;
    NumRecords = 0;
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const;

  virtual bool hasFeature(StringRef Feature) const {
    return Feature == "mblaze";
  }
  
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
  virtual const char *getTargetPrefix() const {
    return "mblaze";
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
    switch (*Name) {
    default: return false;
    case 'O': // Zero
      return true;
    case 'b': // Base register
    case 'f': // Floating point register
      Info.setAllowsRegister();
      return true;
    }
  }
  virtual const char *getClobbers() const {
    return "";
  }
};

/// MBlazeTargetInfo::getTargetDefines - Return a set of the MBlaze-specific
/// #defines that are not tied to a specific subtarget.
void MBlazeTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  Builder.defineMacro("__microblaze__");
  Builder.defineMacro("_ARCH_MICROBLAZE");
  Builder.defineMacro("__MICROBLAZE__");

  // Target properties.
  Builder.defineMacro("_BIG_ENDIAN");
  Builder.defineMacro("__BIG_ENDIAN__");

  // Subtarget options.
  Builder.defineMacro("__REGISTER_PREFIX__", "");
}


const char * const MBlazeTargetInfo::GCCRegNames[] = {
  "r0",   "r1",   "r2",   "r3",   "r4",   "r5",   "r6",   "r7",
  "r8",   "r9",   "r10",  "r11",  "r12",  "r13",  "r14",  "r15",
  "r16",  "r17",  "r18",  "r19",  "r20",  "r21",  "r22",  "r23",
  "r24",  "r25",  "r26",  "r27",  "r28",  "r29",  "r30",  "r31",
  "$f0",  "$f1",  "$f2",  "$f3",  "$f4",  "$f5",  "$f6",  "$f7",
  "$f8",  "$f9",  "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
  "$f16", "$f17", "$f18", "$f19", "$f20", "$f21", "$f22", "$f23",
  "$f24", "$f25", "$f26", "$f27", "$f28", "$f29", "$f30", "$f31",
  "hi",   "lo",   "accum","rmsr", "$fcc1","$fcc2","$fcc3","$fcc4",
  "$fcc5","$fcc6","$fcc7","$ap",  "$rap", "$frp"
};

void MBlazeTargetInfo::getGCCRegNames(const char * const *&Names,
                                   unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias MBlazeTargetInfo::GCCRegAliases[] = {
  { {"f0"},  "r0" },
  { {"f1"},  "r1" },
  { {"f2"},  "r2" },
  { {"f3"},  "r3" },
  { {"f4"},  "r4" },
  { {"f5"},  "r5" },
  { {"f6"},  "r6" },
  { {"f7"},  "r7" },
  { {"f8"},  "r8" },
  { {"f9"},  "r9" },
  { {"f10"}, "r10" },
  { {"f11"}, "r11" },
  { {"f12"}, "r12" },
  { {"f13"}, "r13" },
  { {"f14"}, "r14" },
  { {"f15"}, "r15" },
  { {"f16"}, "r16" },
  { {"f17"}, "r17" },
  { {"f18"}, "r18" },
  { {"f19"}, "r19" },
  { {"f20"}, "r20" },
  { {"f21"}, "r21" },
  { {"f22"}, "r22" },
  { {"f23"}, "r23" },
  { {"f24"}, "r24" },
  { {"f25"}, "r25" },
  { {"f26"}, "r26" },
  { {"f27"}, "r27" },
  { {"f28"}, "r28" },
  { {"f29"}, "r29" },
  { {"f30"}, "r30" },
  { {"f31"}, "r31" },
};

void MBlazeTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                     unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}
} // end anonymous namespace.

namespace {
// Namespace for x86 abstract base class
const Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, ALL_LANGUAGES },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER,\
                                              ALL_LANGUAGES },
#include "clang/Basic/BuiltinsX86.def"
};

static const char* const GCCRegNames[] = {
  "ax", "dx", "cx", "bx", "si", "di", "bp", "sp",
  "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)",
  "argp", "flags", "fpcr", "fpsr", "dirflag", "frame",
  "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
  "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
  "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
};

const TargetInfo::AddlRegName AddlRegNames[] = {
  { { "al", "ah", "eax", "rax" }, 0 },
  { { "bl", "bh", "ebx", "rbx" }, 3 },
  { { "cl", "ch", "ecx", "rcx" }, 2 },
  { { "dl", "dh", "edx", "rdx" }, 1 },
  { { "esi", "rsi" }, 4 },
  { { "edi", "rdi" }, 5 },
  { { "esp", "rsp" }, 7 },
  { { "ebp", "rbp" }, 6 },
};

// X86 target abstract base class; x86-32 and x86-64 are very close, so
// most of the implementation can be shared.
class X86TargetInfo : public TargetInfo {
  enum X86SSEEnum {
    NoSSE, SSE1, SSE2, SSE3, SSSE3, SSE41, SSE42, AVX, AVX2
  } SSELevel;
  enum MMX3DNowEnum {
    NoMMX3DNow, MMX, AMD3DNow, AMD3DNowAthlon
  } MMX3DNowLevel;

  bool HasAES;
  bool HasLZCNT;
  bool HasBMI;
  bool HasBMI2;
  bool HasPOPCNT;
  bool HasFMA4;

  /// \brief Enumeration of all of the X86 CPUs supported by Clang.
  ///
  /// Each enumeration represents a particular CPU supported by Clang. These
  /// loosely correspond to the options passed to '-march' or '-mtune' flags.
  enum CPUKind {
    CK_Generic,

    /// \name i386
    /// i386-generation processors.
    //@{
    CK_i386,
    //@}

    /// \name i486
    /// i486-generation processors.
    //@{
    CK_i486,
    CK_WinChipC6,
    CK_WinChip2,
    CK_C3,
    //@}

    /// \name i586
    /// i586-generation processors, P5 microarchitecture based.
    //@{
    CK_i586,
    CK_Pentium,
    CK_PentiumMMX,
    //@}

    /// \name i686
    /// i686-generation processors, P6 / Pentium M microarchitecture based.
    //@{
    CK_i686,
    CK_PentiumPro,
    CK_Pentium2,
    CK_Pentium3,
    CK_Pentium3M,
    CK_PentiumM,
    CK_C3_2,

    /// This enumerator is a bit odd, as GCC no longer accepts -march=yonah.
    /// Clang however has some logic to suport this.
    // FIXME: Warn, deprecate, and potentially remove this.
    CK_Yonah,
    //@}

    /// \name Netburst
    /// Netburst microarchitecture based processors.
    //@{
    CK_Pentium4,
    CK_Pentium4M,
    CK_Prescott,
    CK_Nocona,
    //@}

    /// \name Core
    /// Core microarchitecture based processors.
    //@{
    CK_Core2,

    /// This enumerator, like \see CK_Yonah, is a bit odd. It is another
    /// codename which GCC no longer accepts as an option to -march, but Clang
    /// has some logic for recognizing it.
    // FIXME: Warn, deprecate, and potentially remove this.
    CK_Penryn,
    //@}

    /// \name Atom
    /// Atom processors
    //@{
    CK_Atom,
    //@}

    /// \name Nehalem
    /// Nehalem microarchitecture based processors.
    //@{
    CK_Corei7,
    CK_Corei7AVX,
    CK_CoreAVXi,
    CK_CoreAVX2,
    //@}

    /// \name K6
    /// K6 architecture processors.
    //@{
    CK_K6,
    CK_K6_2,
    CK_K6_3,
    //@}

    /// \name K7
    /// K7 architecture processors.
    //@{
    CK_Athlon,
    CK_AthlonThunderbird,
    CK_Athlon4,
    CK_AthlonXP,
    CK_AthlonMP,
    //@}

    /// \name K8
    /// K8 architecture processors.
    //@{
    CK_Athlon64,
    CK_Athlon64SSE3,
    CK_AthlonFX,
    CK_K8,
    CK_K8SSE3,
    CK_Opteron,
    CK_OpteronSSE3,
    CK_AMDFAM10,
    //@}

    /// \name Bobcat
    /// Bobcat architecture processors.
    //@{
    CK_BTVER1,
    //@}

    /// \name Bulldozer
    /// Bulldozer architecture processors.
    //@{
    CK_BDVER1,
    CK_BDVER2,
    //@}

    /// This specification is deprecated and will be removed in the future.
    /// Users should prefer \see CK_K8.
    // FIXME: Warn on this when the CPU is set to it.
    CK_x86_64,
    //@}

    /// \name Geode
    /// Geode processors.
    //@{
    CK_Geode
    //@}
  } CPU;

public:
  X86TargetInfo(const std::string& triple)
    : TargetInfo(triple), SSELevel(NoSSE), MMX3DNowLevel(NoMMX3DNow),
      HasAES(false), HasLZCNT(false), HasBMI(false), HasBMI2(false),
      HasPOPCNT(false), HasFMA4(false), CPU(CK_Generic) {
    BigEndian = false;
    LongDoubleFormat = &llvm::APFloat::x87DoubleExtended;
  }
  virtual unsigned getFloatEvalMethod() const {
    // X87 evaluates with 80 bits "long double" precision.
    return SSELevel == NoSSE ? 2 : 0;
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = BuiltinInfo;
    NumRecords = clang::X86::LastTSBuiltin-Builtin::FirstTSBuiltin;
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const {
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const {
    Aliases = 0;
    NumAliases = 0;
  }
  virtual void getGCCAddlRegNames(const AddlRegName *&Names,
				  unsigned &NumNames) const {
    Names = AddlRegNames;
    NumNames = llvm::array_lengthof(AddlRegNames);
  }
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const;
  virtual std::string convertConstraint(const char *&Constraint) const;
  virtual const char *getClobbers() const {
    return "~{dirflag},~{fpsr},~{flags}";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const;
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 StringRef Name,
                                 bool Enabled) const;
  virtual void getDefaultFeatures(llvm::StringMap<bool> &Features) const;
  virtual bool hasFeature(StringRef Feature) const;
  virtual void HandleTargetFeatures(std::vector<std::string> &Features);
  virtual const char* getABI() const {
    if (PointerWidth == 64 && SSELevel >= AVX)
      return "avx";
    else if (PointerWidth == 32 && MMX3DNowLevel == NoMMX3DNow)
      return "no-mmx";
    return "";
  }
  virtual bool setCPU(const std::string &Name) {
    CPU = llvm::StringSwitch<CPUKind>(Name)
      .Case("i386", CK_i386)
      .Case("i486", CK_i486)
      .Case("winchip-c6", CK_WinChipC6)
      .Case("winchip2", CK_WinChip2)
      .Case("c3", CK_C3)
      .Case("i586", CK_i586)
      .Case("pentium", CK_Pentium)
      .Case("pentium-mmx", CK_PentiumMMX)
      .Case("i686", CK_i686)
      .Case("pentiumpro", CK_PentiumPro)
      .Case("pentium2", CK_Pentium2)
      .Case("pentium3", CK_Pentium3)
      .Case("pentium3m", CK_Pentium3M)
      .Case("pentium-m", CK_PentiumM)
      .Case("c3-2", CK_C3_2)
      .Case("yonah", CK_Yonah)
      .Case("pentium4", CK_Pentium4)
      .Case("pentium4m", CK_Pentium4M)
      .Case("prescott", CK_Prescott)
      .Case("nocona", CK_Nocona)
      .Case("core2", CK_Core2)
      .Case("penryn", CK_Penryn)
      .Case("atom", CK_Atom)
      .Case("corei7", CK_Corei7)
      .Case("corei7-avx", CK_Corei7AVX)
      .Case("core-avx-i", CK_CoreAVXi)
      .Case("core-avx2", CK_CoreAVX2)
      .Case("k6", CK_K6)
      .Case("k6-2", CK_K6_2)
      .Case("k6-3", CK_K6_3)
      .Case("athlon", CK_Athlon)
      .Case("athlon-tbird", CK_AthlonThunderbird)
      .Case("athlon-4", CK_Athlon4)
      .Case("athlon-xp", CK_AthlonXP)
      .Case("athlon-mp", CK_AthlonMP)
      .Case("athlon64", CK_Athlon64)
      .Case("athlon64-sse3", CK_Athlon64SSE3)
      .Case("athlon-fx", CK_AthlonFX)
      .Case("k8", CK_K8)
      .Case("k8-sse3", CK_K8SSE3)
      .Case("opteron", CK_Opteron)
      .Case("opteron-sse3", CK_OpteronSSE3)
      .Case("amdfam10", CK_AMDFAM10)
      .Case("btver1", CK_BTVER1)
      .Case("bdver1", CK_BDVER1)
      .Case("bdver2", CK_BDVER2)
      .Case("x86-64", CK_x86_64)
      .Case("geode", CK_Geode)
      .Default(CK_Generic);

    // Perform any per-CPU checks necessary to determine if this CPU is
    // acceptable.
    // FIXME: This results in terrible diagnostics. Clang just says the CPU is
    // invalid without explaining *why*.
    switch (CPU) {
    case CK_Generic:
      // No processor selected!
      return false;

    case CK_i386:
    case CK_i486:
    case CK_WinChipC6:
    case CK_WinChip2:
    case CK_C3:
    case CK_i586:
    case CK_Pentium:
    case CK_PentiumMMX:
    case CK_i686:
    case CK_PentiumPro:
    case CK_Pentium2:
    case CK_Pentium3:
    case CK_Pentium3M:
    case CK_PentiumM:
    case CK_Yonah:
    case CK_C3_2:
    case CK_Pentium4:
    case CK_Pentium4M:
    case CK_Prescott:
    case CK_K6:
    case CK_K6_2:
    case CK_K6_3:
    case CK_Athlon:
    case CK_AthlonThunderbird:
    case CK_Athlon4:
    case CK_AthlonXP:
    case CK_AthlonMP:
    case CK_Geode:
      // Only accept certain architectures when compiling in 32-bit mode.
      if (PointerWidth != 32)
        return false;

      // Fallthrough
    case CK_Nocona:
    case CK_Core2:
    case CK_Penryn:
    case CK_Atom:
    case CK_Corei7:
    case CK_Corei7AVX:
    case CK_CoreAVXi:
    case CK_CoreAVX2:
    case CK_Athlon64:
    case CK_Athlon64SSE3:
    case CK_AthlonFX:
    case CK_K8:
    case CK_K8SSE3:
    case CK_Opteron:
    case CK_OpteronSSE3:
    case CK_AMDFAM10:
    case CK_BTVER1:
    case CK_BDVER1:
    case CK_BDVER2:
    case CK_x86_64:
      return true;
    }
    llvm_unreachable("Unhandled CPU kind");
  }
};

void X86TargetInfo::getDefaultFeatures(llvm::StringMap<bool> &Features) const {
  // FIXME: This should not be here.
  Features["3dnow"] = false;
  Features["3dnowa"] = false;
  Features["mmx"] = false;
  Features["sse"] = false;
  Features["sse2"] = false;
  Features["sse3"] = false;
  Features["ssse3"] = false;
  Features["sse41"] = false;
  Features["sse42"] = false;
  Features["sse4a"] = false;
  Features["aes"] = false;
  Features["avx"] = false;
  Features["avx2"] = false;
  Features["lzcnt"] = false;
  Features["bmi"] = false;
  Features["bmi2"] = false;
  Features["popcnt"] = false;
  Features["fma4"] = false;

  // FIXME: This *really* should not be here.

  // X86_64 always has SSE2.
  if (PointerWidth == 64)
    Features["sse2"] = Features["sse"] = Features["mmx"] = true;

  switch (CPU) {
  case CK_Generic:
  case CK_i386:
  case CK_i486:
  case CK_i586:
  case CK_Pentium:
  case CK_i686:
  case CK_PentiumPro:
    break;
  case CK_PentiumMMX:
  case CK_Pentium2:
    setFeatureEnabled(Features, "mmx", true);
    break;
  case CK_Pentium3:
  case CK_Pentium3M:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse", true);
    break;
  case CK_PentiumM:
  case CK_Pentium4:
  case CK_Pentium4M:
  case CK_x86_64:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse2", true);
    break;
  case CK_Yonah:
  case CK_Prescott:
  case CK_Nocona:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse3", true);
    break;
  case CK_Core2:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "ssse3", true);
    break;
  case CK_Penryn:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse4.1", true);
    break;
  case CK_Atom:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "ssse3", true);
    break;
  case CK_Corei7:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse4", true);
    setFeatureEnabled(Features, "aes", true);
    break;
  case CK_Corei7AVX:
  case CK_CoreAVXi:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse4", true);
    setFeatureEnabled(Features, "aes", true);
    //setFeatureEnabled(Features, "avx", true);
    break;
  case CK_CoreAVX2:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse4", true);
    setFeatureEnabled(Features, "aes", true);
    setFeatureEnabled(Features, "lzcnt", true);
    setFeatureEnabled(Features, "bmi", true);
    setFeatureEnabled(Features, "bmi2", true);
    //setFeatureEnabled(Features, "avx2", true);
    break;
  case CK_K6:
  case CK_WinChipC6:
    setFeatureEnabled(Features, "mmx", true);
    break;
  case CK_K6_2:
  case CK_K6_3:
  case CK_WinChip2:
  case CK_C3:
    setFeatureEnabled(Features, "3dnow", true);
    break;
  case CK_Athlon:
  case CK_AthlonThunderbird:
  case CK_Geode:
    setFeatureEnabled(Features, "3dnowa", true);
    break;
  case CK_Athlon4:
  case CK_AthlonXP:
  case CK_AthlonMP:
    setFeatureEnabled(Features, "sse", true);
    setFeatureEnabled(Features, "3dnowa", true);
    break;
  case CK_K8:
  case CK_Opteron:
  case CK_Athlon64:
  case CK_AthlonFX:
    setFeatureEnabled(Features, "sse2", true);
    setFeatureEnabled(Features, "3dnowa", true);
    break;
  case CK_K8SSE3:
  case CK_OpteronSSE3:
  case CK_Athlon64SSE3:
    setFeatureEnabled(Features, "sse3", true);
    setFeatureEnabled(Features, "3dnowa", true);
    break;
  case CK_AMDFAM10:
    setFeatureEnabled(Features, "sse3", true);
    setFeatureEnabled(Features, "sse4a", true);
    setFeatureEnabled(Features, "3dnowa", true);
    break;
  case CK_BTVER1:
    setFeatureEnabled(Features, "ssse3", true);
    setFeatureEnabled(Features, "sse4a", true);
  case CK_BDVER1:
  case CK_BDVER2:
    setFeatureEnabled(Features, "sse4", true);
    setFeatureEnabled(Features, "sse4a", true);
    setFeatureEnabled(Features, "aes", true);
    break;
  case CK_C3_2:
    setFeatureEnabled(Features, "mmx", true);
    setFeatureEnabled(Features, "sse", true);
    break;
  }
}

bool X86TargetInfo::setFeatureEnabled(llvm::StringMap<bool> &Features,
                                      StringRef Name,
                                      bool Enabled) const {
  // FIXME: This *really* should not be here.  We need some way of translating
  // options into llvm subtarget features.
  if (!Features.count(Name) &&
      (Name != "sse4" && Name != "sse4.2" && Name != "sse4.1"))
    return false;

  // FIXME: this should probably use a switch with fall through.

  if (Enabled) {
    if (Name == "mmx")
      Features["mmx"] = true;
    else if (Name == "sse")
      Features["mmx"] = Features["sse"] = true;
    else if (Name == "sse2")
      Features["mmx"] = Features["sse"] = Features["sse2"] = true;
    else if (Name == "sse3")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        true;
    else if (Name == "ssse3")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["ssse3"] = true;
    else if (Name == "sse4" || Name == "sse4.2")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["ssse3"] = Features["sse41"] = Features["sse42"] =
        Features["popcnt"] = true;
    else if (Name == "sse4.1")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["ssse3"] = Features["sse41"] = true;
    else if (Name == "3dnow")
      Features["mmx"] = Features["3dnow"] = true;
    else if (Name == "3dnowa")
      Features["mmx"] = Features["3dnow"] = Features["3dnowa"] = true;
    else if (Name == "aes")
      Features["aes"] = true;
    else if (Name == "avx")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["ssse3"] = Features["sse41"] = Features["sse42"] =
        Features["popcnt"] = Features["avx"] = true;
    else if (Name == "avx2")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["ssse3"] = Features["sse41"] = Features["sse42"] =
        Features["popcnt"] = Features["avx"] = Features["avx2"] = true;
    else if (Name == "fma4")
        Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["ssse3"] = Features["sse41"] = Features["sse42"] =
        Features["popcnt"] = Features["avx"] = Features["fma4"] = true;
    else if (Name == "sse4a")
      Features["mmx"] = Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["lzcnt"] = Features["popcnt"] = Features["sse4a"] = true;
    else if (Name == "lzcnt")
      Features["lzcnt"] = true;
    else if (Name == "bmi")
      Features["bmi"] = true;
    else if (Name == "bmi2")
      Features["bmi2"] = true;
    else if (Name == "popcnt")
      Features["popcnt"] = true;
  } else {
    if (Name == "mmx")
      Features["mmx"] = Features["3dnow"] = Features["3dnowa"] = false;
    else if (Name == "sse")
      Features["sse"] = Features["sse2"] = Features["sse3"] =
        Features["ssse3"] = Features["sse41"] = Features["sse42"] =
        Features["sse4a"] = false;
    else if (Name == "sse2")
      Features["sse2"] = Features["sse3"] = Features["ssse3"] =
        Features["sse41"] = Features["sse42"] = Features["sse4a"] = false;
    else if (Name == "sse3")
      Features["sse3"] = Features["ssse3"] = Features["sse41"] =
        Features["sse42"] = Features["sse4a"] = false;
    else if (Name == "ssse3")
      Features["ssse3"] = Features["sse41"] = Features["sse42"] = false;
    else if (Name == "sse4" || Name == "sse4.1")
      Features["sse41"] = Features["sse42"] = false;
    else if (Name == "sse4.2")
      Features["sse42"] = false;
    else if (Name == "3dnow")
      Features["3dnow"] = Features["3dnowa"] = false;
    else if (Name == "3dnowa")
      Features["3dnowa"] = false;
    else if (Name == "aes")
      Features["aes"] = false;
    else if (Name == "avx")
      Features["avx"] = Features["avx2"] = Features["fma4"] = false;
    else if (Name == "avx2")
      Features["avx2"] = false;
    else if (Name == "sse4a")
      Features["sse4a"] = false;
    else if (Name == "lzcnt")
      Features["lzcnt"] = false;
    else if (Name == "bmi")
      Features["bmi"] = false;
    else if (Name == "bmi2")
      Features["bmi2"] = false;
    else if (Name == "popcnt")
      Features["popcnt"] = false;
    else if (Name == "fma4")
      Features["fma4"] = false;
  }

  return true;
}

/// HandleTargetOptions - Perform initialization based on the user
/// configured set of features.
void X86TargetInfo::HandleTargetFeatures(std::vector<std::string> &Features) {
  // Remember the maximum enabled sselevel.
  for (unsigned i = 0, e = Features.size(); i !=e; ++i) {
    // Ignore disabled features.
    if (Features[i][0] == '-')
      continue;

    StringRef Feature = StringRef(Features[i]).substr(1);

    if (Feature == "aes") {
      HasAES = true;
      continue;
    }

    if (Feature == "lzcnt") {
      HasLZCNT = true;
      continue;
    }

    if (Feature == "bmi") {
      HasBMI = true;
      continue;
    }

    if (Feature == "bmi2") {
      HasBMI2 = true;
      continue;
    }

    if (Feature == "popcnt") {
      HasPOPCNT = true;
      continue;
    }

    if (Feature == "fma4") {
      HasFMA4 = true;
      continue;
    }

    assert(Features[i][0] == '+' && "Invalid target feature!");
    X86SSEEnum Level = llvm::StringSwitch<X86SSEEnum>(Feature)
      .Case("avx2", AVX2)
      .Case("avx", AVX)
      .Case("sse42", SSE42)
      .Case("sse41", SSE41)
      .Case("ssse3", SSSE3)
      .Case("sse3", SSE3)
      .Case("sse2", SSE2)
      .Case("sse", SSE1)
      .Default(NoSSE);
    SSELevel = std::max(SSELevel, Level);

    MMX3DNowEnum ThreeDNowLevel =
      llvm::StringSwitch<MMX3DNowEnum>(Feature)
        .Case("3dnowa", AMD3DNowAthlon)
        .Case("3dnow", AMD3DNow)
        .Case("mmx", MMX)
        .Default(NoMMX3DNow);

    MMX3DNowLevel = std::max(MMX3DNowLevel, ThreeDNowLevel);
  }

  // Don't tell the backend if we're turning off mmx; it will end up disabling
  // SSE, which we don't want.
  std::vector<std::string>::iterator it;
  it = std::find(Features.begin(), Features.end(), "-mmx");
  if (it != Features.end())
    Features.erase(it);
}

/// X86TargetInfo::getTargetDefines - Return the set of the X86-specific macro
/// definitions for this particular subtarget.
void X86TargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  if (PointerWidth == 64) {
    if (getLongWidth() == 64) {
      Builder.defineMacro("_LP64");
      Builder.defineMacro("__LP64__");
    }
    Builder.defineMacro("__amd64__");
    Builder.defineMacro("__amd64");
    Builder.defineMacro("__x86_64");
    Builder.defineMacro("__x86_64__");
  } else {
    DefineStd(Builder, "i386", Opts);
  }

  // Subtarget options.
  // FIXME: We are hard-coding the tune parameters based on the CPU, but they
  // truly should be based on -mtune options.
  switch (CPU) {
  case CK_Generic:
    break;
  case CK_i386:
    // The rest are coming from the i386 define above.
    Builder.defineMacro("__tune_i386__");
    break;
  case CK_i486:
  case CK_WinChipC6:
  case CK_WinChip2:
  case CK_C3:
    defineCPUMacros(Builder, "i486");
    break;
  case CK_PentiumMMX:
    Builder.defineMacro("__pentium_mmx__");
    Builder.defineMacro("__tune_pentium_mmx__");
    // Fallthrough
  case CK_i586:
  case CK_Pentium:
    defineCPUMacros(Builder, "i586");
    defineCPUMacros(Builder, "pentium");
    break;
  case CK_Pentium3:
  case CK_Pentium3M:
  case CK_PentiumM:
    Builder.defineMacro("__tune_pentium3__");
    // Fallthrough
  case CK_Pentium2:
  case CK_C3_2:
    Builder.defineMacro("__tune_pentium2__");
    // Fallthrough
  case CK_PentiumPro:
    Builder.defineMacro("__tune_i686__");
    Builder.defineMacro("__tune_pentiumpro__");
    // Fallthrough
  case CK_i686:
    Builder.defineMacro("__i686");
    Builder.defineMacro("__i686__");
    // Strangely, __tune_i686__ isn't defined by GCC when CPU == i686.
    Builder.defineMacro("__pentiumpro");
    Builder.defineMacro("__pentiumpro__");
    break;
  case CK_Pentium4:
  case CK_Pentium4M:
    defineCPUMacros(Builder, "pentium4");
    break;
  case CK_Yonah:
  case CK_Prescott:
  case CK_Nocona:
    defineCPUMacros(Builder, "nocona");
    break;
  case CK_Core2:
  case CK_Penryn:
    defineCPUMacros(Builder, "core2");
    break;
  case CK_Atom:
    defineCPUMacros(Builder, "atom");
    break;
  case CK_Corei7:
  case CK_Corei7AVX:
  case CK_CoreAVXi:
  case CK_CoreAVX2:
    defineCPUMacros(Builder, "corei7");
    break;
  case CK_K6_2:
    Builder.defineMacro("__k6_2__");
    Builder.defineMacro("__tune_k6_2__");
    // Fallthrough
  case CK_K6_3:
    if (CPU != CK_K6_2) {  // In case of fallthrough
      // FIXME: GCC may be enabling these in cases where some other k6
      // architecture is specified but -m3dnow is explicitly provided. The
      // exact semantics need to be determined and emulated here.
      Builder.defineMacro("__k6_3__");
      Builder.defineMacro("__tune_k6_3__");
    }
    // Fallthrough
  case CK_K6:
    defineCPUMacros(Builder, "k6");
    break;
  case CK_Athlon:
  case CK_AthlonThunderbird:
  case CK_Athlon4:
  case CK_AthlonXP:
  case CK_AthlonMP:
    defineCPUMacros(Builder, "athlon");
    if (SSELevel != NoSSE) {
      Builder.defineMacro("__athlon_sse__");
      Builder.defineMacro("__tune_athlon_sse__");
    }
    break;
  case CK_K8:
  case CK_K8SSE3:
  case CK_x86_64:
  case CK_Opteron:
  case CK_OpteronSSE3:
  case CK_Athlon64:
  case CK_Athlon64SSE3:
  case CK_AthlonFX:
    defineCPUMacros(Builder, "k8");
    break;
  case CK_AMDFAM10:
    defineCPUMacros(Builder, "amdfam10");
    break;
  case CK_BTVER1:
    defineCPUMacros(Builder, "btver1");
    break;
  case CK_BDVER1:
    defineCPUMacros(Builder, "bdver1");
    break;
  case CK_BDVER2:
    defineCPUMacros(Builder, "bdver2");
    break;
  case CK_Geode:
    defineCPUMacros(Builder, "geode");
    break;
  }

  // Target properties.
  Builder.defineMacro("__LITTLE_ENDIAN__");
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // Define __NO_MATH_INLINES on linux/x86 so that we don't get inline
  // functions in glibc header files that use FP Stack inline asm which the
  // backend can't deal with (PR879).
  Builder.defineMacro("__NO_MATH_INLINES");

  if (HasAES)
    Builder.defineMacro("__AES__");

  if (HasLZCNT)
    Builder.defineMacro("__LZCNT__");

  if (HasBMI)
    Builder.defineMacro("__BMI__");

  if (HasBMI2)
    Builder.defineMacro("__BMI2__");

  if (HasPOPCNT)
    Builder.defineMacro("__POPCNT__");

  if (HasFMA4)
    Builder.defineMacro("__FMA4__");

  // Each case falls through to the previous one here.
  switch (SSELevel) {
  case AVX2:
    Builder.defineMacro("__AVX2__");
  case AVX:
    Builder.defineMacro("__AVX__");
  case SSE42:
    Builder.defineMacro("__SSE4_2__");
  case SSE41:
    Builder.defineMacro("__SSE4_1__");
  case SSSE3:
    Builder.defineMacro("__SSSE3__");
  case SSE3:
    Builder.defineMacro("__SSE3__");
  case SSE2:
    Builder.defineMacro("__SSE2__");
    Builder.defineMacro("__SSE2_MATH__");  // -mfp-math=sse always implied.
  case SSE1:
    Builder.defineMacro("__SSE__");
    Builder.defineMacro("__SSE_MATH__");   // -mfp-math=sse always implied.
  case NoSSE:
    break;
  }

  if (Opts.MicrosoftExt && PointerWidth == 32) {
    switch (SSELevel) {
    case AVX2:
    case AVX:
    case SSE42:
    case SSE41:
    case SSSE3:
    case SSE3:
    case SSE2:
      Builder.defineMacro("_M_IX86_FP", Twine(2));
      break;
    case SSE1:
      Builder.defineMacro("_M_IX86_FP", Twine(1));
      break;
    default:
      Builder.defineMacro("_M_IX86_FP", Twine(0));
    }
  }

  // Each case falls through to the previous one here.
  switch (MMX3DNowLevel) {
  case AMD3DNowAthlon:
    Builder.defineMacro("__3dNOW_A__");
  case AMD3DNow:
    Builder.defineMacro("__3dNOW__");
  case MMX:
    Builder.defineMacro("__MMX__");
  case NoMMX3DNow:
    break;
  }
}

bool X86TargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("aes", HasAES)
      .Case("avx", SSELevel >= AVX)
      .Case("avx2", SSELevel >= AVX2)
      .Case("bmi", HasBMI)
      .Case("bmi2", HasBMI2)
      .Case("fma4", HasFMA4)
      .Case("lzcnt", HasLZCNT)
      .Case("mm3dnow", MMX3DNowLevel >= AMD3DNow)
      .Case("mm3dnowa", MMX3DNowLevel >= AMD3DNowAthlon)
      .Case("mmx", MMX3DNowLevel >= MMX)
      .Case("popcnt", HasPOPCNT)
      .Case("sse", SSELevel >= SSE1)
      .Case("sse2", SSELevel >= SSE2)
      .Case("sse3", SSELevel >= SSE3)
      .Case("ssse3", SSELevel >= SSSE3)
      .Case("sse41", SSELevel >= SSE41)
      .Case("sse42", SSELevel >= SSE42)
      .Case("x86", true)
      .Case("x86_32", PointerWidth == 32)
      .Case("x86_64", PointerWidth == 64)
      .Default(false);
}

bool
X86TargetInfo::validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
  switch (*Name) {
  default: return false;
  case 'Y': // first letter of a pair:
    switch (*(Name+1)) {
    default: return false;
    case '0':  // First SSE register.
    case 't':  // Any SSE register, when SSE2 is enabled.
    case 'i':  // Any SSE register, when SSE2 and inter-unit moves enabled.
    case 'm':  // any MMX register, when inter-unit moves enabled.
      break;   // falls through to setAllowsRegister.
  }
  case 'a': // eax.
  case 'b': // ebx.
  case 'c': // ecx.
  case 'd': // edx.
  case 'S': // esi.
  case 'D': // edi.
  case 'A': // edx:eax.
  case 'f': // any x87 floating point stack register.
  case 't': // top of floating point stack.
  case 'u': // second from top of floating point stack.
  case 'q': // Any register accessible as [r]l: a, b, c, and d.
  case 'y': // Any MMX register.
  case 'x': // Any SSE register.
  case 'Q': // Any register accessible as [r]h: a, b, c, and d.
  case 'R': // "Legacy" registers: ax, bx, cx, dx, di, si, sp, bp.
  case 'l': // "Index" registers: any general register that can be used as an
            // index in a base+index memory access.
    Info.setAllowsRegister();
    return true;
  case 'C': // SSE floating point constant.
  case 'G': // x87 floating point constant.
  case 'e': // 32-bit signed integer constant for use with zero-extending
            // x86_64 instructions.
  case 'Z': // 32-bit unsigned integer constant for use with zero-extending
            // x86_64 instructions.
    return true;
  }
}


std::string
X86TargetInfo::convertConstraint(const char *&Constraint) const {
  switch (*Constraint) {
  case 'a': return std::string("{ax}");
  case 'b': return std::string("{bx}");
  case 'c': return std::string("{cx}");
  case 'd': return std::string("{dx}");
  case 'S': return std::string("{si}");
  case 'D': return std::string("{di}");
  case 'p': // address
    return std::string("im");
  case 't': // top of floating point stack.
    return std::string("{st}");
  case 'u': // second from top of floating point stack.
    return std::string("{st(1)}"); // second from top of floating point stack.
  default:
    return std::string(1, *Constraint);
  }
}
} // end anonymous namespace

namespace {
// X86-32 generic target
class X86_32TargetInfo : public X86TargetInfo {
public:
  X86_32TargetInfo(const std::string& triple) : X86TargetInfo(triple) {
    DoubleAlign = LongLongAlign = 32;
    LongDoubleWidth = 96;
    LongDoubleAlign = 32;
    SuitableAlign = 128;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:32:32-n8:16:32-S128";
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
    RegParmMax = 3;

    // Use fpret for all types.
    RealTypeUsesObjCFPRet = ((1 << TargetInfo::Float) |
                             (1 << TargetInfo::Double) |
                             (1 << TargetInfo::LongDouble));

    // x86-32 has atomics up to 8 bytes
    // FIXME: Check that we actually have cmpxchg8b before setting
    // MaxAtomicInlineWidth. (cmpxchg8b is an i586 instruction.)
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }

  int getEHDataRegisterNumber(unsigned RegNo) const {
    if (RegNo == 0) return 0;
    if (RegNo == 1) return 2;
    return -1;
  }
};
} // end anonymous namespace

namespace {
class NetBSDI386TargetInfo : public NetBSDTargetInfo<X86_32TargetInfo> {
public:
  NetBSDI386TargetInfo(const std::string &triple) :
    NetBSDTargetInfo<X86_32TargetInfo>(triple) {
  }

  virtual unsigned getFloatEvalMethod() const {
    // NetBSD defaults to "double" rounding
    return 1;
  }
};
} // end anonymous namespace

namespace {
class OpenBSDI386TargetInfo : public OpenBSDTargetInfo<X86_32TargetInfo> {
public:
  OpenBSDI386TargetInfo(const std::string& triple) :
    OpenBSDTargetInfo<X86_32TargetInfo>(triple) {
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    PtrDiffType = SignedLong;
  }
};
} // end anonymous namespace

namespace {
class DarwinI386TargetInfo : public DarwinTargetInfo<X86_32TargetInfo> {
public:
  DarwinI386TargetInfo(const std::string& triple) :
    DarwinTargetInfo<X86_32TargetInfo>(triple) {
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    SuitableAlign = 128;
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:128:128-n8:16:32-S128";
    HasAlignMac68kSupport = true;
  }

};
} // end anonymous namespace

namespace {
// x86-32 Windows target
class WindowsX86_32TargetInfo : public WindowsTargetInfo<X86_32TargetInfo> {
public:
  WindowsX86_32TargetInfo(const std::string& triple)
    : WindowsTargetInfo<X86_32TargetInfo>(triple) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    DoubleAlign = LongLongAlign = 64;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-"
                        "v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsTargetInfo<X86_32TargetInfo>::getTargetDefines(Opts, Builder);
  }
};
} // end anonymous namespace

namespace {

// x86-32 Windows Visual Studio target
class VisualStudioWindowsX86_32TargetInfo : public WindowsX86_32TargetInfo {
public:
  VisualStudioWindowsX86_32TargetInfo(const std::string& triple)
    : WindowsX86_32TargetInfo(triple) {
    LongDoubleWidth = LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble;
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_32TargetInfo::getTargetDefines(Opts, Builder);
    WindowsX86_32TargetInfo::getVisualStudioDefines(Opts, Builder);
    // The value of the following reflects processor type.
    // 300=386, 400=486, 500=Pentium, 600=Blend (default)
    // We lost the original triple, so we use the default.
    Builder.defineMacro("_M_IX86", "600");
  }
};
} // end anonymous namespace

namespace {
// x86-32 MinGW target
class MinGWX86_32TargetInfo : public WindowsX86_32TargetInfo {
public:
  MinGWX86_32TargetInfo(const std::string& triple)
    : WindowsX86_32TargetInfo(triple) {
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_32TargetInfo::getTargetDefines(Opts, Builder);
    DefineStd(Builder, "WIN32", Opts);
    DefineStd(Builder, "WINNT", Opts);
    Builder.defineMacro("_X86_");
    Builder.defineMacro("__MSVCRT__");
    Builder.defineMacro("__MINGW32__");

    // mingw32-gcc provides __declspec(a) as alias of __attribute__((a)).
    // In contrast, clang-cc1 provides __declspec(a) with -fms-extensions.
    if (Opts.MicrosoftExt)
      // Provide "as-is" __declspec.
      Builder.defineMacro("__declspec", "__declspec");
    else
      // Provide alias of __attribute__ like mingw32-gcc.
      Builder.defineMacro("__declspec(a)", "__attribute__((a))");
  }
};
} // end anonymous namespace

namespace {
// x86-32 Cygwin target
class CygwinX86_32TargetInfo : public X86_32TargetInfo {
public:
  CygwinX86_32TargetInfo(const std::string& triple)
    : X86_32TargetInfo(triple) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    DoubleAlign = LongLongAlign = 64;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                        "a0:0:64-f80:32:32-n8:16:32-S32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__CYGWIN__");
    Builder.defineMacro("__CYGWIN32__");
    DefineStd(Builder, "unix", Opts);
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
  }
};
} // end anonymous namespace

namespace {
// x86-32 Haiku target
class HaikuX86_32TargetInfo : public X86_32TargetInfo {
public:
  HaikuX86_32TargetInfo(const std::string& triple)
    : X86_32TargetInfo(triple) {
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    PtrDiffType = SignedLong;
    this->UserLabelPrefix = "";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__INTEL__");
    Builder.defineMacro("__HAIKU__");
  }
};
} // end anonymous namespace

// RTEMS Target
template<typename Target>
class RTEMSTargetInfo : public OSTargetInfo<Target> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    // RTEMS defines; list based off of gcc output

    Builder.defineMacro("__rtems__");
    Builder.defineMacro("__ELF__");
  }
public:
  RTEMSTargetInfo(const std::string &triple)
    : OSTargetInfo<Target>(triple) {
      this->UserLabelPrefix = "";

      llvm::Triple Triple(triple);
      switch (Triple.getArch()) {
        default:
        case llvm::Triple::x86:
          // this->MCountName = ".mcount";
          break;
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
        case llvm::Triple::ppc:
        case llvm::Triple::ppc64:
          // this->MCountName = "_mcount";
          break;
        case llvm::Triple::arm:
          // this->MCountName = "__mcount";
          break;
      }

    }
};

namespace {
// x86-32 RTEMS target
class RTEMSX86_32TargetInfo : public X86_32TargetInfo {
public:
  RTEMSX86_32TargetInfo(const std::string& triple)
    : X86_32TargetInfo(triple) {
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    PtrDiffType = SignedLong;
    this->UserLabelPrefix = "";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__INTEL__");
    Builder.defineMacro("__rtems__");
  }
};
} // end anonymous namespace

namespace {
// x86-64 generic target
class X86_64TargetInfo : public X86TargetInfo {
public:
  X86_64TargetInfo(const std::string &triple) : X86TargetInfo(triple) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    LargeArrayMinWidth = 128;
    LargeArrayAlign = 128;
    SuitableAlign = 128;
    IntMaxType = SignedLong;
    UIntMaxType = UnsignedLong;
    Int64Type = SignedLong;
    RegParmMax = 6;

    DescriptionString = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                        "a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128";

    // Use fpret only for long double.
    RealTypeUsesObjCFPRet = (1 << TargetInfo::LongDouble);

    // Use fp2ret for _Complex long double.
    ComplexLongDoubleUsesFP2Ret = true;

    // x86-64 has atomics up to 16 bytes.
    // FIXME: Once the backend is fixed, increase MaxAtomicInlineWidth to 128
    // on CPUs with cmpxchg16b
    MaxAtomicPromoteWidth = 128;
    MaxAtomicInlineWidth = 64;
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef struct __va_list_tag {"
           "  unsigned gp_offset;"
           "  unsigned fp_offset;"
           "  void* overflow_arg_area;"
           "  void* reg_save_area;"
           "} __va_list_tag;"
           "typedef __va_list_tag __builtin_va_list[1];";
  }

  int getEHDataRegisterNumber(unsigned RegNo) const {
    if (RegNo == 0) return 0;
    if (RegNo == 1) return 1;
    return -1;
  }
};
} // end anonymous namespace

namespace {
// x86-64 Windows target
class WindowsX86_64TargetInfo : public WindowsTargetInfo<X86_64TargetInfo> {
public:
  WindowsX86_64TargetInfo(const std::string& triple)
    : WindowsTargetInfo<X86_64TargetInfo>(triple) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    LongWidth = LongAlign = 32;
    DoubleAlign = LongLongAlign = 64;
    IntMaxType = SignedLongLong;
    UIntMaxType = UnsignedLongLong;
    Int64Type = SignedLongLong;
    SizeType = UnsignedLongLong;
    PtrDiffType = SignedLongLong;
    IntPtrType = SignedLongLong;
    this->UserLabelPrefix = "";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsTargetInfo<X86_64TargetInfo>::getTargetDefines(Opts, Builder);
    Builder.defineMacro("_WIN64");
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
};
} // end anonymous namespace

namespace {
// x86-64 Windows Visual Studio target
class VisualStudioWindowsX86_64TargetInfo : public WindowsX86_64TargetInfo {
public:
  VisualStudioWindowsX86_64TargetInfo(const std::string& triple)
    : WindowsX86_64TargetInfo(triple) {
    LongDoubleWidth = LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble;
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_64TargetInfo::getTargetDefines(Opts, Builder);
    WindowsX86_64TargetInfo::getVisualStudioDefines(Opts, Builder);
    Builder.defineMacro("_M_X64");
    Builder.defineMacro("_M_AMD64");
  }
};
} // end anonymous namespace

namespace {
// x86-64 MinGW target
class MinGWX86_64TargetInfo : public WindowsX86_64TargetInfo {
public:
  MinGWX86_64TargetInfo(const std::string& triple)
    : WindowsX86_64TargetInfo(triple) {
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    WindowsX86_64TargetInfo::getTargetDefines(Opts, Builder);
    DefineStd(Builder, "WIN64", Opts);
    Builder.defineMacro("__MSVCRT__");
    Builder.defineMacro("__MINGW32__");
    Builder.defineMacro("__MINGW64__");

    // mingw32-gcc provides __declspec(a) as alias of __attribute__((a)).
    // In contrast, clang-cc1 provides __declspec(a) with -fms-extensions.
    if (Opts.MicrosoftExt)
      // Provide "as-is" __declspec.
      Builder.defineMacro("__declspec", "__declspec");
    else
      // Provide alias of __attribute__ like mingw32-gcc.
      Builder.defineMacro("__declspec(a)", "__attribute__((a))");
  }
};
} // end anonymous namespace

namespace {
class DarwinX86_64TargetInfo : public DarwinTargetInfo<X86_64TargetInfo> {
public:
  DarwinX86_64TargetInfo(const std::string& triple)
      : DarwinTargetInfo<X86_64TargetInfo>(triple) {
    Int64Type = SignedLongLong;
  }
};
} // end anonymous namespace

namespace {
class OpenBSDX86_64TargetInfo : public OpenBSDTargetInfo<X86_64TargetInfo> {
public:
  OpenBSDX86_64TargetInfo(const std::string& triple)
      : OpenBSDTargetInfo<X86_64TargetInfo>(triple) {
    IntMaxType = SignedLongLong;
    UIntMaxType = UnsignedLongLong;
    Int64Type = SignedLongLong;
  }
};
} // end anonymous namespace

namespace {
class ARMTargetInfo : public TargetInfo {
  // Possible FPU choices.
  enum FPUMode {
    NoFPU,
    VFP2FPU,
    VFP3FPU,
    NeonFPU
  };

  static bool FPUModeIsVFP(FPUMode Mode) {
    return Mode >= VFP2FPU && Mode <= NeonFPU;
  }

  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char * const GCCRegNames[];

  std::string ABI, CPU;

  unsigned FPU : 3;

  unsigned IsThumb : 1;

  // Initialized via features.
  unsigned SoftFloat : 1;
  unsigned SoftFloatABI : 1;

  static const Builtin::Info BuiltinInfo[];

public:
  ARMTargetInfo(const std::string &TripleStr)
    : TargetInfo(TripleStr), ABI("aapcs-linux"), CPU("arm1136j-s")
  {
    BigEndian = false;
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    // AAPCS 7.1.1, ARM-Linux ABI 2.4: type of wchar_t is unsigned int.
    WCharType = UnsignedInt;

    // {} in inline assembly are neon specifiers, not assembly variant
    // specifiers.
    NoAsmVariants = true;

    // FIXME: Should we just treat this as a feature?
    IsThumb = getTriple().getArchName().startswith("thumb");
    if (IsThumb) {
      // Thumb1 add sp, #imm requires the immediate value be multiple of 4,
      // so set preferred for small types to 32.
      DescriptionString = ("e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-"
                           "v64:64:64-v128:64:128-a0:0:32-n32-S64");
    } else {
      DescriptionString = ("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                           "i64:64:64-f32:32:32-f64:64:64-"
                           "v64:64:64-v128:64:128-a0:0:64-n32-S64");
    }

    // ARM targets default to using the ARM C++ ABI.
    CXXABI = CXXABI_ARM;

    // ARM has atomics up to 8 bytes
    // FIXME: Set MaxAtomicInlineWidth if we have the feature v6e
    MaxAtomicPromoteWidth = 64;

    // Do force alignment of members that follow zero length bitfields.  If
    // the alignment of the zero-length bitfield is greater than the member 
    // that follows it, `bar', `bar' will be aligned as the  type of the 
    // zero length bitfield.
    UseZeroLengthBitfieldAlignment = true;
  }
  virtual const char *getABI() const { return ABI.c_str(); }
  virtual bool setABI(const std::string &Name) {
    ABI = Name;

    // The defaults (above) are for AAPCS, check if we need to change them.
    //
    // FIXME: We need support for -meabi... we could just mangle it into the
    // name.
    if (Name == "apcs-gnu") {
      DoubleAlign = LongLongAlign = LongDoubleAlign = SuitableAlign = 32;
      SizeType = UnsignedLong;

      // Revert to using SignedInt on apcs-gnu to comply with existing behaviour.
      WCharType = SignedInt;

      // Do not respect the alignment of bit-field types when laying out
      // structures. This corresponds to PCC_BITFIELD_TYPE_MATTERS in gcc.
      UseBitFieldTypeAlignment = false;

      /// gcc forces the alignment to 4 bytes, regardless of the type of the
      /// zero length bitfield.  This corresponds to EMPTY_FIELD_BOUNDARY in
      /// gcc.
      ZeroLengthBitfieldBoundary = 32;

      if (IsThumb) {
        // Thumb1 add sp, #imm requires the immediate value be multiple of 4,
        // so set preferred for small types to 32.
        DescriptionString = ("e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-"
                             "i64:32:64-f32:32:32-f64:32:64-"
                             "v64:32:64-v128:32:128-a0:0:32-n32-S32");
      } else {
        DescriptionString = ("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                             "i64:32:64-f32:32:32-f64:32:64-"
                             "v64:32:64-v128:32:128-a0:0:32-n32-S32");
      }

      // FIXME: Override "preferred align" for double and long long.
    } else if (Name == "aapcs") {
      // FIXME: Enumerated types are variable width in straight AAPCS.
    } else if (Name == "aapcs-linux") {
      ;
    } else
      return false;

    return true;
  }

  void getDefaultFeatures(llvm::StringMap<bool> &Features) const {
    if (CPU == "arm1136jf-s" || CPU == "arm1176jzf-s" || CPU == "mpcore")
      Features["vfp2"] = true;
    else if (CPU == "cortex-a8" || CPU == "cortex-a9")
      Features["neon"] = true;
  }

  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 StringRef Name,
                                 bool Enabled) const {
    if (Name == "soft-float" || Name == "soft-float-abi" ||
        Name == "vfp2" || Name == "vfp3" || Name == "neon" || Name == "d16" ||
        Name == "neonfp") {
      Features[Name] = Enabled;
    } else
      return false;

    return true;
  }

  virtual void HandleTargetFeatures(std::vector<std::string> &Features) {
    FPU = NoFPU;
    SoftFloat = SoftFloatABI = false;
    for (unsigned i = 0, e = Features.size(); i != e; ++i) {
      if (Features[i] == "+soft-float")
        SoftFloat = true;
      else if (Features[i] == "+soft-float-abi")
        SoftFloatABI = true;
      else if (Features[i] == "+vfp2")
        FPU = VFP2FPU;
      else if (Features[i] == "+vfp3")
        FPU = VFP3FPU;
      else if (Features[i] == "+neon")
        FPU = NeonFPU;
    }

    // Remove front-end specific options which the backend handles differently.
    std::vector<std::string>::iterator it;
    it = std::find(Features.begin(), Features.end(), "+soft-float");
    if (it != Features.end())
      Features.erase(it);
    it = std::find(Features.begin(), Features.end(), "+soft-float-abi");
    if (it != Features.end())
      Features.erase(it);
  }

  virtual bool hasFeature(StringRef Feature) const {
    return llvm::StringSwitch<bool>(Feature)
        .Case("arm", true)
        .Case("softfloat", SoftFloat)
        .Case("thumb", IsThumb)
        .Case("neon", FPU == NeonFPU && !SoftFloat && 
              StringRef(getCPUDefineSuffix(CPU)).startswith("7"))    
        .Default(false);
  }
  static const char *getCPUDefineSuffix(StringRef Name) {
    return llvm::StringSwitch<const char*>(Name)
      .Cases("arm8", "arm810", "4")
      .Cases("strongarm", "strongarm110", "strongarm1100", "strongarm1110", "4")
      .Cases("arm7tdmi", "arm7tdmi-s", "arm710t", "arm720t", "arm9", "4T")
      .Cases("arm9tdmi", "arm920", "arm920t", "arm922t", "arm940t", "4T")
      .Case("ep9312", "4T")
      .Cases("arm10tdmi", "arm1020t", "5T")
      .Cases("arm9e", "arm946e-s", "arm966e-s", "arm968e-s", "5TE")
      .Case("arm926ej-s", "5TEJ")
      .Cases("arm10e", "arm1020e", "arm1022e", "5TE")
      .Cases("xscale", "iwmmxt", "5TE")
      .Case("arm1136j-s", "6J")
      .Cases("arm1176jz-s", "arm1176jzf-s", "6ZK")
      .Cases("arm1136jf-s", "mpcorenovfp", "mpcore", "6K")
      .Cases("arm1156t2-s", "arm1156t2f-s", "6T2")
      .Cases("cortex-a8", "cortex-a9", "7A")
      .Case("cortex-m3", "7M")
      .Case("cortex-m4", "7M")
      .Case("cortex-m0", "6M")
      .Default(0);
  }
  virtual bool setCPU(const std::string &Name) {
    if (!getCPUDefineSuffix(Name))
      return false;

    CPU = Name;
    return true;
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    // Target identification.
    Builder.defineMacro("__arm");
    Builder.defineMacro("__arm__");

    // Target properties.
    Builder.defineMacro("__ARMEL__");
    Builder.defineMacro("__LITTLE_ENDIAN__");
    Builder.defineMacro("__REGISTER_PREFIX__", "");

    StringRef CPUArch = getCPUDefineSuffix(CPU);
    Builder.defineMacro("__ARM_ARCH_" + CPUArch + "__");

    // Subtarget options.

    // FIXME: It's more complicated than this and we don't really support
    // interworking.
    if ('5' <= CPUArch[0] && CPUArch[0] <= '7')
      Builder.defineMacro("__THUMB_INTERWORK__");

    if (ABI == "aapcs" || ABI == "aapcs-linux")
      Builder.defineMacro("__ARM_EABI__");

    if (SoftFloat)
      Builder.defineMacro("__SOFTFP__");

    if (CPU == "xscale")
      Builder.defineMacro("__XSCALE__");

    bool IsARMv7 = CPUArch.startswith("7");
    if (IsThumb) {
      Builder.defineMacro("__THUMBEL__");
      Builder.defineMacro("__thumb__");
      if (CPUArch == "6T2" || IsARMv7)
        Builder.defineMacro("__thumb2__");
    }

    // Note, this is always on in gcc, even though it doesn't make sense.
    Builder.defineMacro("__APCS_32__");

    if (FPUModeIsVFP((FPUMode) FPU))
      Builder.defineMacro("__VFP_FP__");

    // This only gets set when Neon instructions are actually available, unlike
    // the VFP define, hence the soft float and arch check. This is subtly
    // different from gcc, we follow the intent which was that it should be set
    // when Neon instructions are actually available.
    if (FPU == NeonFPU && !SoftFloat && IsARMv7)
      Builder.defineMacro("__ARM_NEON__");
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = BuiltinInfo;
    NumRecords = clang::ARM::LastTSBuiltin-Builtin::FirstTSBuiltin;
  }
  virtual bool isCLZForZeroUndef() const { return false; }
  virtual const char *getVAListDeclaration() const {
    return "typedef void* __builtin_va_list;";
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
    // FIXME: Check if this is complete
    switch (*Name) {
    default:
    case 'l': // r0-r7
    case 'h': // r8-r15
    case 'w': // VFP Floating point register single precision
    case 'P': // VFP Floating point register double precision
      Info.setAllowsRegister();
      return true;
    case 'Q': // A memory address that is a single base register.
      Info.setAllowsMemory();
      return true;
    case 'U': // a memory reference...
      switch (Name[1]) {
      case 'q': // ...ARMV4 ldrsb
      case 'v': // ...VFP load/store (reg+constant offset)
      case 'y': // ...iWMMXt load/store
      case 't': // address valid for load/store opaque types wider
	        // than 128-bits
      case 'n': // valid address for Neon doubleword vector load/store
      case 'm': // valid address for Neon element and structure load/store
      case 's': // valid address for non-offset loads/stores of quad-word
	        // values in four ARM registers
        Info.setAllowsMemory();
        Name++;
        return true;
      }
    }
    return false;
  }
  virtual std::string convertConstraint(const char *&Constraint) const {
    std::string R;
    switch (*Constraint) {
    case 'U':   // Two-character constraint; add "^" hint for later parsing.
      R = std::string("^") + std::string(Constraint, 2);
      Constraint++;
      break;
    case 'p': // 'p' should be translated to 'r' by default.
      R = std::string("r");
      break;
    default:
      return std::string(1, *Constraint);
    }
    return R;
  }
  virtual const char *getClobbers() const {
    // FIXME: Is this really right?
    return "";
  }
};

const char * const ARMTargetInfo::GCCRegNames[] = {
  // Integer registers
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "sp", "lr", "pc",

  // Float registers
  "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
  "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
  "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",

  // Double registers
  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",

  // Quad registers
  "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
  "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
};

void ARMTargetInfo::getGCCRegNames(const char * const *&Names,
                                   unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias ARMTargetInfo::GCCRegAliases[] = {
  { { "a1" }, "r0" },
  { { "a2" }, "r1" },
  { { "a3" }, "r2" },
  { { "a4" }, "r3" },
  { { "v1" }, "r4" },
  { { "v2" }, "r5" },
  { { "v3" }, "r6" },
  { { "v4" }, "r7" },
  { { "v5" }, "r8" },
  { { "v6", "rfp" }, "r9" },
  { { "sl" }, "r10" },
  { { "fp" }, "r11" },
  { { "ip" }, "r12" },
  { { "r13" }, "sp" },
  { { "r14" }, "lr" },
  { { "r15" }, "pc" },
  // The S, D and Q registers overlap, but aren't really aliases; we
  // don't want to substitute one of these for a different-sized one.
};

void ARMTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                       unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}

const Builtin::Info ARMTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, ALL_LANGUAGES },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER,\
                                              ALL_LANGUAGES },
#include "clang/Basic/BuiltinsARM.def"
};
} // end anonymous namespace.

namespace {
class DarwinARMTargetInfo :
  public DarwinTargetInfo<ARMTargetInfo> {
protected:
  virtual void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                            MacroBuilder &Builder) const {
    getDarwinDefines(Builder, Opts, Triple, PlatformName, PlatformMinVersion);
  }

public:
  DarwinARMTargetInfo(const std::string& triple)
    : DarwinTargetInfo<ARMTargetInfo>(triple) {
    HasAlignMac68kSupport = true;
    // iOS always has 64-bit atomic instructions.
    // FIXME: This should be based off of the target features in ARMTargetInfo.
    MaxAtomicInlineWidth = 64;
  }
};
} // end anonymous namespace.


namespace {
// Hexagon abstract base class
class HexagonTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
  static const char * const GCCRegNames[];
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  std::string CPU;
public:
  HexagonTargetInfo(const std::string& triple) : TargetInfo(triple)  {
    BigEndian = false;
    DescriptionString = ("e-p:32:32:32-"
                         "i64:64:64-i32:32:32-"
                         "i16:16:16-i1:32:32-a:0:0");

    // {} in inline assembly are packet specifiers, not assembly variant
    // specifiers.
    NoAsmVariants = true;
  }

  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = BuiltinInfo;
    NumRecords = clang::Hexagon::LastTSBuiltin-Builtin::FirstTSBuiltin;
  }

  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
    return true;
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const;

  virtual bool hasFeature(StringRef Feature) const {
    return Feature == "hexagon";
  }
  
  virtual const char *getVAListDeclaration() const {
    return "typedef char* __builtin_va_list;";
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
  virtual const char *getClobbers() const {
    return "";
  }

  static const char *getHexagonCPUSuffix(StringRef Name) {
    return llvm::StringSwitch<const char*>(Name)
      .Case("hexagonv2", "2")
      .Case("hexagonv3", "3")
      .Case("hexagonv4", "4")
      .Default(0);
  }

  virtual bool setCPU(const std::string &Name) {
    if (!getHexagonCPUSuffix(Name))
      return false;

    CPU = Name;
    return true;
  }
};

void HexagonTargetInfo::getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
  Builder.defineMacro("qdsp6");
  Builder.defineMacro("__qdsp6", "1");
  Builder.defineMacro("__qdsp6__", "1");

  Builder.defineMacro("hexagon");
  Builder.defineMacro("__hexagon", "1");
  Builder.defineMacro("__hexagon__", "1");

  if(CPU == "hexagonv1") {
    Builder.defineMacro("__HEXAGON_V1__");
    Builder.defineMacro("__HEXAGON_ARCH__", "1");
    if(Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V1__");
      Builder.defineMacro("__QDSP6_ARCH__", "1");
    }
  }
  else if(CPU == "hexagonv2") {
    Builder.defineMacro("__HEXAGON_V2__");
    Builder.defineMacro("__HEXAGON_ARCH__", "2");
    if(Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V2__");
      Builder.defineMacro("__QDSP6_ARCH__", "2");
    }
  }
  else if(CPU == "hexagonv3") {
    Builder.defineMacro("__HEXAGON_V3__");
    Builder.defineMacro("__HEXAGON_ARCH__", "3");
    if(Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V3__");
      Builder.defineMacro("__QDSP6_ARCH__", "3");
    }
  }
  else if(CPU == "hexagonv4") {
    Builder.defineMacro("__HEXAGON_V4__");
    Builder.defineMacro("__HEXAGON_ARCH__", "4");
    if(Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V4__");
      Builder.defineMacro("__QDSP6_ARCH__", "4");
    }
  }
}

const char * const HexagonTargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
  "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",
  "p0", "p1", "p2", "p3",
  "sa0", "lc0", "sa1", "lc1", "m0", "m1", "usr", "ugp"
};

void HexagonTargetInfo::getGCCRegNames(const char * const *&Names,
                                   unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}


const TargetInfo::GCCRegAlias HexagonTargetInfo::GCCRegAliases[] = {
  { { "sp" }, "r29" },
  { { "fp" }, "r30" },
  { { "lr" }, "r31" },
 };

void HexagonTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                     unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}


const Builtin::Info HexagonTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS, 0, ALL_LANGUAGES },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) { #ID, TYPE, ATTRS, HEADER,\
                                              ALL_LANGUAGES },
#include "clang/Basic/BuiltinsHexagon.def"
};
}


namespace {
class SparcV8TargetInfo : public TargetInfo {
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char * const GCCRegNames[];
  bool SoftFloat;
public:
  SparcV8TargetInfo(const std::string& triple) : TargetInfo(triple) {
    // FIXME: Support Sparc quad-precision long double?
    BigEndian = false;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32";
  }
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 StringRef Name,
                                 bool Enabled) const {
    if (Name == "soft-float")
      Features[Name] = Enabled;
    else
      return false;

    return true;
  }
  virtual void HandleTargetFeatures(std::vector<std::string> &Features) {
    SoftFloat = false;
    for (unsigned i = 0, e = Features.size(); i != e; ++i)
      if (Features[i] == "+soft-float")
        SoftFloat = true;
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "sparc", Opts);
    Builder.defineMacro("__sparcv8");
    Builder.defineMacro("__REGISTER_PREFIX__", "");

    if (SoftFloat)
      Builder.defineMacro("SOFT_FLOAT", "1");
  }
  
  virtual bool hasFeature(StringRef Feature) const {
    return llvm::StringSwitch<bool>(Feature)
             .Case("softfloat", SoftFloat)
             .Case("sparc", true)
             .Default(false);
  }
  
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    // FIXME: Implement!
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef void* __builtin_va_list;";
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const {
    // FIXME: Implement!
    return false;
  }
  virtual const char *getClobbers() const {
    // FIXME: Implement!
    return "";
  }
};

const char * const SparcV8TargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
  "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"
};

void SparcV8TargetInfo::getGCCRegNames(const char * const *&Names,
                                       unsigned &NumNames) const {
  Names = GCCRegNames;
  NumNames = llvm::array_lengthof(GCCRegNames);
}

const TargetInfo::GCCRegAlias SparcV8TargetInfo::GCCRegAliases[] = {
  { { "g0" }, "r0" },
  { { "g1" }, "r1" },
  { { "g2" }, "r2" },
  { { "g3" }, "r3" },
  { { "g4" }, "r4" },
  { { "g5" }, "r5" },
  { { "g6" }, "r6" },
  { { "g7" }, "r7" },
  { { "o0" }, "r8" },
  { { "o1" }, "r9" },
  { { "o2" }, "r10" },
  { { "o3" }, "r11" },
  { { "o4" }, "r12" },
  { { "o5" }, "r13" },
  { { "o6", "sp" }, "r14" },
  { { "o7" }, "r15" },
  { { "l0" }, "r16" },
  { { "l1" }, "r17" },
  { { "l2" }, "r18" },
  { { "l3" }, "r19" },
  { { "l4" }, "r20" },
  { { "l5" }, "r21" },
  { { "l6" }, "r22" },
  { { "l7" }, "r23" },
  { { "i0" }, "r24" },
  { { "i1" }, "r25" },
  { { "i2" }, "r26" },
  { { "i3" }, "r27" },
  { { "i4" }, "r28" },
  { { "i5" }, "r29" },
  { { "i6", "fp" }, "r30" },
  { { "i7" }, "r31" },
};

void SparcV8TargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                         unsigned &NumAliases) const {
  Aliases = GCCRegAliases;
  NumAliases = llvm::array_lengthof(GCCRegAliases);
}
} // end anonymous namespace.

namespace {
class AuroraUXSparcV8TargetInfo : public AuroraUXTargetInfo<SparcV8TargetInfo> {
public:
  AuroraUXSparcV8TargetInfo(const std::string& triple) :
      AuroraUXTargetInfo<SparcV8TargetInfo>(triple) {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
  }
};
class SolarisSparcV8TargetInfo : public SolarisTargetInfo<SparcV8TargetInfo> {
public:
  SolarisSparcV8TargetInfo(const std::string& triple) :
      SolarisTargetInfo<SparcV8TargetInfo>(triple) {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
  }
};
} // end anonymous namespace.

namespace {
  class MSP430TargetInfo : public TargetInfo {
    static const char * const GCCRegNames[];
  public:
    MSP430TargetInfo(const std::string& triple) : TargetInfo(triple) {
      BigEndian = false;
      TLSSupported = false;
      IntWidth = 16; IntAlign = 16;
      LongWidth = 32; LongLongWidth = 64;
      LongAlign = LongLongAlign = 16;
      PointerWidth = 16; PointerAlign = 16;
      SuitableAlign = 16;
      SizeType = UnsignedInt;
      IntMaxType = SignedLong;
      UIntMaxType = UnsignedLong;
      IntPtrType = SignedShort;
      PtrDiffType = SignedInt;
      SigAtomicType = SignedLong;
      DescriptionString = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16";
   }
    virtual void getTargetDefines(const LangOptions &Opts,
                                  MacroBuilder &Builder) const {
      Builder.defineMacro("MSP430");
      Builder.defineMacro("__MSP430__");
      // FIXME: defines for different 'flavours' of MCU
    }
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {
     // FIXME: Implement.
      Records = 0;
      NumRecords = 0;
    }
    virtual bool hasFeature(StringRef Feature) const {
      return Feature == "msp430";
    }
    virtual void getGCCRegNames(const char * const *&Names,
                                unsigned &NumNames) const;
    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                  unsigned &NumAliases) const {
      // No aliases.
      Aliases = 0;
      NumAliases = 0;
    }
    virtual bool validateAsmConstraint(const char *&Name,
                                       TargetInfo::ConstraintInfo &info) const {
      // No target constraints for now.
      return false;
    }
    virtual const char *getClobbers() const {
      // FIXME: Is this really right?
      return "";
    }
    virtual const char *getVAListDeclaration() const {
      // FIXME: implement
      return "typedef char* __builtin_va_list;";
   }
  };

  const char * const MSP430TargetInfo::GCCRegNames[] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
  };

  void MSP430TargetInfo::getGCCRegNames(const char * const *&Names,
                                        unsigned &NumNames) const {
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }
}

namespace {

  // LLVM and Clang cannot be used directly to output native binaries for
  // target, but is used to compile C code to llvm bitcode with correct
  // type and alignment information.
  //
  // TCE uses the llvm bitcode as input and uses it for generating customized
  // target processor and program binary. TCE co-design environment is
  // publicly available in http://tce.cs.tut.fi

  static const unsigned TCEOpenCLAddrSpaceMap[] = {
      3, // opencl_global
      4, // opencl_local
      5  // opencl_constant
  };

  class TCETargetInfo : public TargetInfo{
  public:
    TCETargetInfo(const std::string& triple) : TargetInfo(triple) {
      TLSSupported = false;
      IntWidth = 32;
      LongWidth = LongLongWidth = 32;
      PointerWidth = 32;
      IntAlign = 32;
      LongAlign = LongLongAlign = 32;
      PointerAlign = 32;
      SuitableAlign = 32;
      SizeType = UnsignedInt;
      IntMaxType = SignedLong;
      UIntMaxType = UnsignedLong;
      IntPtrType = SignedInt;
      PtrDiffType = SignedInt;
      FloatWidth = 32;
      FloatAlign = 32;
      DoubleWidth = 32;
      DoubleAlign = 32;
      LongDoubleWidth = 32;
      LongDoubleAlign = 32;
      FloatFormat = &llvm::APFloat::IEEEsingle;
      DoubleFormat = &llvm::APFloat::IEEEsingle;
      LongDoubleFormat = &llvm::APFloat::IEEEsingle;
      DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:32-"
                          "i16:16:32-i32:32:32-i64:32:32-"
                          "f32:32:32-f64:32:32-v64:32:32-"
                          "v128:32:32-a0:0:32-n32";
      AddrSpaceMap = &TCEOpenCLAddrSpaceMap;
    }

    virtual void getTargetDefines(const LangOptions &Opts,
                                  MacroBuilder &Builder) const {
      DefineStd(Builder, "tce", Opts);
      Builder.defineMacro("__TCE__");
      Builder.defineMacro("__TCE_V1__");
    }
    virtual bool hasFeature(StringRef Feature) const {
      return Feature == "tce";
    }
    
    virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                   unsigned &NumRecords) const {}
    virtual const char *getClobbers() const {
      return "";
    }
    virtual const char *getVAListDeclaration() const {
      return "typedef void* __builtin_va_list;";
    }
    virtual void getGCCRegNames(const char * const *&Names,
                                unsigned &NumNames) const {}
    virtual bool validateAsmConstraint(const char *&Name,
                                       TargetInfo::ConstraintInfo &info) const {
      return true;
    }
    virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                  unsigned &NumAliases) const {}
  };
}

namespace {
class MipsTargetInfoBase : public TargetInfo {
  std::string CPU;
  bool SoftFloat;
  bool SingleFloat;

protected:
  std::string ABI;

public:
  MipsTargetInfoBase(const std::string& triple,
                     const std::string& ABIStr,
                     const std::string& CPUStr)
    : TargetInfo(triple),
      CPU(CPUStr),
      SoftFloat(false), SingleFloat(false),
      ABI(ABIStr)
  {}

  virtual const char *getABI() const { return ABI.c_str(); }
  virtual bool setABI(const std::string &Name) = 0;
  virtual bool setCPU(const std::string &Name) {
    CPU = Name;
    return true;
  }
  void getDefaultFeatures(llvm::StringMap<bool> &Features) const {
    Features[ABI] = true;
    Features[CPU] = true;
  }

  virtual void getArchDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    if (SoftFloat)
      Builder.defineMacro("__mips_soft_float", Twine(1));
    else if (SingleFloat)
      Builder.defineMacro("__mips_single_float", Twine(1));
    else if (!SoftFloat && !SingleFloat)
      Builder.defineMacro("__mips_hard_float", Twine(1));
    else
      llvm_unreachable("Invalid float ABI for Mips.");

    Builder.defineMacro("_MIPS_SZPTR", Twine(getPointerWidth(0)));
    Builder.defineMacro("_MIPS_SZINT", Twine(getIntWidth()));
    Builder.defineMacro("_MIPS_SZLONG", Twine(getLongWidth()));
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const = 0;
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    // FIXME: Implement!
  }
  virtual bool hasFeature(StringRef Feature) const {
    return Feature == "mips";
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef void* __builtin_va_list;";
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const {
    static const char * const GCCRegNames[] = {
      // CPU register names
      // Must match second column of GCCRegAliases
      "$0",   "$1",   "$2",   "$3",   "$4",   "$5",   "$6",   "$7",
      "$8",   "$9",   "$10",  "$11",  "$12",  "$13",  "$14",  "$15",
      "$16",  "$17",  "$18",  "$19",  "$20",  "$21",  "$22",  "$23",
      "$24",  "$25",  "$26",  "$27",  "$28",  "$29",  "$30",  "$31",
      // Floating point register names
      "$f0",  "$f1",  "$f2",  "$f3",  "$f4",  "$f5",  "$f6",  "$f7",
      "$f8",  "$f9",  "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
      "$f16", "$f17", "$f18", "$f19", "$f20", "$f21", "$f22", "$f23",
      "$f24", "$f25", "$f26", "$f27", "$f28", "$f29", "$f30", "$f31",
      // Hi/lo and condition register names
      "hi",   "lo",   "",     "$fcc0","$fcc1","$fcc2","$fcc3","$fcc4",
      "$fcc5","$fcc6","$fcc7"
    };
    Names = GCCRegNames;
    NumNames = llvm::array_lengthof(GCCRegNames);
  }
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const = 0;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
    switch (*Name) {
    default:
      return false;
        
    case 'r': // CPU registers.
    case 'd': // Equivalent to "r" unless generating MIPS16 code.
    case 'y': // Equivalent to "r", backwards compatibility only.
    case 'f': // floating-point registers.
    case 'c': // $25 for indirect jumps
    case 'l': // lo register
    case 'x': // hilo register pair
      Info.setAllowsRegister();
      return true;
    }
  }

  virtual const char *getClobbers() const {
    // FIXME: Implement!
    return "";
  }

  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 StringRef Name,
                                 bool Enabled) const {
    if (Name == "soft-float" || Name == "single-float" ||
        Name == "o32" || Name == "n32" || Name == "n64" || Name == "eabi" ||
        Name == "mips32" || Name == "mips32r2" ||
        Name == "mips64" || Name == "mips64r2") {
      Features[Name] = Enabled;
      return true;
    }
    return false;
  }

  virtual void HandleTargetFeatures(std::vector<std::string> &Features) {
    SoftFloat = false;
    SingleFloat = false;

    for (std::vector<std::string>::iterator it = Features.begin(),
         ie = Features.end(); it != ie; ++it) {
      if (*it == "+single-float") {
        SingleFloat = true;
        break;
      }

      if (*it == "+soft-float") {
        SoftFloat = true;
        // This option is front-end specific.
        // Do not need to pass it to the backend.
        Features.erase(it);
        break;
      }
    }
  }
};

class Mips32TargetInfoBase : public MipsTargetInfoBase {
public:
  Mips32TargetInfoBase(const std::string& triple) :
    MipsTargetInfoBase(triple, "o32", "mips32") {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
  }
  virtual bool setABI(const std::string &Name) {
    if ((Name == "o32") || (Name == "eabi")) {
      ABI = Name;
      return true;
    } else
      return false;
  }
  virtual void getArchDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    MipsTargetInfoBase::getArchDefines(Opts, Builder);

    if (ABI == "o32") {
      Builder.defineMacro("__mips_o32");
      Builder.defineMacro("_ABIO32", "1");
      Builder.defineMacro("_MIPS_SIM", "_ABIO32");
    }
    else if (ABI == "eabi")
      Builder.defineMacro("__mips_eabi");
    else
      llvm_unreachable("Invalid ABI for Mips32.");
  }
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const {
    static const TargetInfo::GCCRegAlias GCCRegAliases[] = {
      { { "at" },  "$1" },
      { { "v0" },  "$2" },
      { { "v1" },  "$3" },
      { { "a0" },  "$4" },
      { { "a1" },  "$5" },
      { { "a2" },  "$6" },
      { { "a3" },  "$7" },
      { { "t0" },  "$8" },
      { { "t1" },  "$9" },
      { { "t2" }, "$10" },
      { { "t3" }, "$11" },
      { { "t4" }, "$12" },
      { { "t5" }, "$13" },
      { { "t6" }, "$14" },
      { { "t7" }, "$15" },
      { { "s0" }, "$16" },
      { { "s1" }, "$17" },
      { { "s2" }, "$18" },
      { { "s3" }, "$19" },
      { { "s4" }, "$20" },
      { { "s5" }, "$21" },
      { { "s6" }, "$22" },
      { { "s7" }, "$23" },
      { { "t8" }, "$24" },
      { { "t9" }, "$25" },
      { { "k0" }, "$26" },
      { { "k1" }, "$27" },
      { { "gp" }, "$28" },
      { { "sp","$sp" }, "$29" },
      { { "fp","$fp" }, "$30" },
      { { "ra" }, "$31" }
    };
    Aliases = GCCRegAliases;
    NumAliases = llvm::array_lengthof(GCCRegAliases);
  }
};

class Mips32EBTargetInfo : public Mips32TargetInfoBase {
public:
  Mips32EBTargetInfo(const std::string& triple) : Mips32TargetInfoBase(triple) {
    DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "mips", Opts);
    Builder.defineMacro("_mips");
    DefineStd(Builder, "MIPSEB", Opts);
    Builder.defineMacro("_MIPSEB");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
    getArchDefines(Opts, Builder);
  }
};

class Mips32ELTargetInfo : public Mips32TargetInfoBase {
public:
  Mips32ELTargetInfo(const std::string& triple) : Mips32TargetInfoBase(triple) {
    BigEndian = false;
    DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "mips", Opts);
    Builder.defineMacro("_mips");
    DefineStd(Builder, "MIPSEL", Opts);
    Builder.defineMacro("_MIPSEL");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
    getArchDefines(Opts, Builder);
  }
};

class Mips64TargetInfoBase : public MipsTargetInfoBase {
  virtual void SetDescriptionString(const std::string &Name) = 0;
public:
  Mips64TargetInfoBase(const std::string& triple) :
    MipsTargetInfoBase(triple, "n64", "mips64") {
    LongWidth = LongAlign = 64;
    PointerWidth = PointerAlign = 64;
    LongDoubleWidth = LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad;
    SuitableAlign = 128;
  }
  virtual bool setABI(const std::string &Name) {
    SetDescriptionString(Name);

    if (Name != "n32" && Name != "n64")
      return false;

    ABI = Name;

    if (Name == "n32") {
      LongWidth = LongAlign = 32;
      PointerWidth = PointerAlign = 32;
    }

    return true;
  }
  virtual void getArchDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    MipsTargetInfoBase::getArchDefines(Opts, Builder);

    if (ABI == "n32") {
      Builder.defineMacro("__mips_n32");
      Builder.defineMacro("_ABIN32", "2");
      Builder.defineMacro("_MIPS_SIM", "_ABIN32");
    }
    else if (ABI == "n64") {
      Builder.defineMacro("__mips_n64");
      Builder.defineMacro("_ABI64", "3");
      Builder.defineMacro("_MIPS_SIM", "_ABI64");
    }
    else
      llvm_unreachable("Invalid ABI for Mips64.");
  }
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const {
    static const TargetInfo::GCCRegAlias GCCRegAliases[] = {
      { { "at" },  "$1" },
      { { "v0" },  "$2" },
      { { "v1" },  "$3" },
      { { "a0" },  "$4" },
      { { "a1" },  "$5" },
      { { "a2" },  "$6" },
      { { "a3" },  "$7" },
      { { "a4" },  "$8" },
      { { "a5" },  "$9" },
      { { "a6" }, "$10" },
      { { "a7" }, "$11" },
      { { "t0" }, "$12" },
      { { "t1" }, "$13" },
      { { "t2" }, "$14" },
      { { "t3" }, "$15" },
      { { "s0" }, "$16" },
      { { "s1" }, "$17" },
      { { "s2" }, "$18" },
      { { "s3" }, "$19" },
      { { "s4" }, "$20" },
      { { "s5" }, "$21" },
      { { "s6" }, "$22" },
      { { "s7" }, "$23" },
      { { "t8" }, "$24" },
      { { "t9" }, "$25" },
      { { "k0" }, "$26" },
      { { "k1" }, "$27" },
      { { "gp" }, "$28" },
      { { "sp","$sp" }, "$29" },
      { { "fp","$fp" }, "$30" },
      { { "ra" }, "$31" }
    };
    Aliases = GCCRegAliases;
    NumAliases = llvm::array_lengthof(GCCRegAliases);
  }
};

class Mips64EBTargetInfo : public Mips64TargetInfoBase {
  virtual void SetDescriptionString(const std::string &Name) {
    // Change DescriptionString only if ABI is n32.  
    if (Name == "n32")
      DescriptionString = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                          "i64:64:64-f32:32:32-f64:64:64-f128:128:128-"
                          "v64:64:64-n32";      
  }
public:
  Mips64EBTargetInfo(const std::string& triple) : Mips64TargetInfoBase(triple) {
    // Default ABI is n64.  
    DescriptionString = "E-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-f128:128:128-"
                        "v64:64:64-n32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "mips", Opts);
    Builder.defineMacro("_mips");
    DefineStd(Builder, "MIPSEB", Opts);
    Builder.defineMacro("_MIPSEB");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
    getArchDefines(Opts, Builder);
  }
};

class Mips64ELTargetInfo : public Mips64TargetInfoBase {
  virtual void SetDescriptionString(const std::string &Name) {
    // Change DescriptionString only if ABI is n32.  
    if (Name == "n32")
      DescriptionString = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                          "i64:64:64-f32:32:32-f64:64:64-f128:128:128"
                          "-v64:64:64-n32";      
  }
public:
  Mips64ELTargetInfo(const std::string& triple) : Mips64TargetInfoBase(triple) {
    // Default ABI is n64.
    BigEndian = false;
    DescriptionString = "e-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-"
                        "i64:64:64-f32:32:32-f64:64:64-f128:128:128-"
                        "v64:64:64-n32";
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "mips", Opts);
    Builder.defineMacro("_mips");
    DefineStd(Builder, "MIPSEL", Opts);
    Builder.defineMacro("_MIPSEL");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
    getArchDefines(Opts, Builder);
  }
};
} // end anonymous namespace.

namespace {
class PNaClTargetInfo : public TargetInfo {
public:
  PNaClTargetInfo(const std::string& triple) : TargetInfo(triple) {
    BigEndian = false;
    this->UserLabelPrefix = "";
    this->LongAlign = 32;
    this->LongWidth = 32;
    this->PointerAlign = 32;
    this->PointerWidth = 32;
    this->IntMaxType = TargetInfo::SignedLongLong;
    this->UIntMaxType = TargetInfo::UnsignedLongLong;
    this->Int64Type = TargetInfo::SignedLongLong;
    this->DoubleAlign = 64;
    this->LongDoubleWidth = 64;
    this->LongDoubleAlign = 64;
    this->SizeType = TargetInfo::UnsignedInt;
    this->PtrDiffType = TargetInfo::SignedInt;
    this->IntPtrType = TargetInfo::SignedInt;
    this->RegParmMax = 2;
    DescriptionString = "e-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
                        "f32:32:32-f64:64:64-p:32:32:32-v128:32:32";
  }

  void getDefaultFeatures(llvm::StringMap<bool> &Features) const {
  }
  virtual void getArchDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    Builder.defineMacro("__le32__");
    Builder.defineMacro("__pnacl__");
  }
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");

    Builder.defineMacro("__LITTLE_ENDIAN__");
    Builder.defineMacro("__native_client__");
    getArchDefines(Opts, Builder);
  }
  virtual bool hasFeature(StringRef Feature) const {
    return Feature == "pnacl";
  }
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
  }
  virtual const char *getVAListDeclaration() const {
    return "typedef int __builtin_va_list[4];";
  }
  virtual void getGCCRegNames(const char * const *&Names,
                              unsigned &NumNames) const;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases,
                                unsigned &NumAliases) const;
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
    return false;
  }

  virtual const char *getClobbers() const {
    return "";
  }
};

void PNaClTargetInfo::getGCCRegNames(const char * const *&Names,
                                     unsigned &NumNames) const {
  Names = NULL;
  NumNames = 0;
}

void PNaClTargetInfo::getGCCRegAliases(const GCCRegAlias *&Aliases,
                                       unsigned &NumAliases) const {
  Aliases = NULL;
  NumAliases = 0;
}
} // end anonymous namespace.


//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

static TargetInfo *AllocateTarget(const std::string &T) {
  llvm::Triple Triple(T);
  llvm::Triple::OSType os = Triple.getOS();

  switch (Triple.getArch()) {
  default:
    return NULL;

  case llvm::Triple::hexagon:
    return new HexagonTargetInfo(T);

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    if (Triple.isOSDarwin())
      return new DarwinARMTargetInfo(T);

    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<ARMTargetInfo>(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<ARMTargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<ARMTargetInfo>(T);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<ARMTargetInfo>(T);
    default:
      return new ARMTargetInfo(T);
    }

  case llvm::Triple::msp430:
    return new MSP430TargetInfo(T);

  case llvm::Triple::mips:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<Mips32EBTargetInfo>(T);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<Mips32EBTargetInfo>(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<Mips32EBTargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<Mips32EBTargetInfo>(T);
    default:
      return new Mips32EBTargetInfo(T);
    }

  case llvm::Triple::mipsel:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<Mips32ELTargetInfo>(T);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<Mips32ELTargetInfo>(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<Mips32ELTargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<Mips32ELTargetInfo>(T);
    default:
      return new Mips32ELTargetInfo(T);
    }

  case llvm::Triple::mips64:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<Mips64EBTargetInfo>(T);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<Mips64EBTargetInfo>(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<Mips64EBTargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<Mips64EBTargetInfo>(T);
    default:
      return new Mips64EBTargetInfo(T);
    }

  case llvm::Triple::mips64el:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<Mips64ELTargetInfo>(T);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<Mips64ELTargetInfo>(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<Mips64ELTargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<Mips64ELTargetInfo>(T);
    default:
      return new Mips64ELTargetInfo(T);
    }

  case llvm::Triple::le32:
    switch (os) {
      case llvm::Triple::NativeClient:
        return new PNaClTargetInfo(T);
      default:
        return NULL;
    }

  case llvm::Triple::ppc:
    if (Triple.isOSDarwin())
      return new DarwinPPC32TargetInfo(T);
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<PPC32TargetInfo>(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<PPC32TargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<PPC32TargetInfo>(T);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<PPC32TargetInfo>(T);
    default:
      return new PPC32TargetInfo(T);
    }

  case llvm::Triple::ppc64:
    if (Triple.isOSDarwin())
      return new DarwinPPC64TargetInfo(T);
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<PPC64TargetInfo>(T);
    case llvm::Triple::Lv2:
      return new PS3PPUTargetInfo<PPC64TargetInfo>(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<PPC64TargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<PPC64TargetInfo>(T);
    default:
      return new PPC64TargetInfo(T);
    }

  case llvm::Triple::ptx32:
    return new PTX32TargetInfo(T);
  case llvm::Triple::ptx64:
    return new PTX64TargetInfo(T);

  case llvm::Triple::mblaze:
    return new MBlazeTargetInfo(T);

  case llvm::Triple::sparc:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<SparcV8TargetInfo>(T);
    case llvm::Triple::AuroraUX:
      return new AuroraUXSparcV8TargetInfo(T);
    case llvm::Triple::Solaris:
      return new SolarisSparcV8TargetInfo(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<SparcV8TargetInfo>(T);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<SparcV8TargetInfo>(T);
    default:
      return new SparcV8TargetInfo(T);
    }

  // FIXME: Need a real SPU target.
  case llvm::Triple::cellspu:
    return new PS3SPUTargetInfo<PPC64TargetInfo>(T);

  case llvm::Triple::tce:
    return new TCETargetInfo(T);

  case llvm::Triple::x86:
    if (Triple.isOSDarwin())
      return new DarwinI386TargetInfo(T);

    switch (os) {
    case llvm::Triple::AuroraUX:
      return new AuroraUXTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::DragonFly:
      return new DragonFlyBSDTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDI386TargetInfo(T);
    case llvm::Triple::OpenBSD:
      return new OpenBSDI386TargetInfo(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::Minix:
      return new MinixTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<X86_32TargetInfo>(T);
    case llvm::Triple::Cygwin:
      return new CygwinX86_32TargetInfo(T);
    case llvm::Triple::MinGW32:
      return new MinGWX86_32TargetInfo(T);
    case llvm::Triple::Win32:
      return new VisualStudioWindowsX86_32TargetInfo(T);
    case llvm::Triple::Haiku:
      return new HaikuX86_32TargetInfo(T);
    case llvm::Triple::RTEMS:
      return new RTEMSX86_32TargetInfo(T);
    default:
      return new X86_32TargetInfo(T);
    }

  case llvm::Triple::x86_64:
    if (Triple.isOSDarwin() || Triple.getEnvironment() == llvm::Triple::MachO)
      return new DarwinX86_64TargetInfo(T);

    switch (os) {
    case llvm::Triple::AuroraUX:
      return new AuroraUXTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::DragonFly:
      return new DragonFlyBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::OpenBSD:
      return new OpenBSDX86_64TargetInfo(T);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<X86_64TargetInfo>(T);
    case llvm::Triple::MinGW32:
      return new MinGWX86_64TargetInfo(T);
    case llvm::Triple::Win32:   // This is what Triple.h supports now.
      return new VisualStudioWindowsX86_64TargetInfo(T);
    default:
      return new X86_64TargetInfo(T);
    }
  }
}

/// CreateTargetInfo - Return the target info object for the specified target
/// triple.
TargetInfo *TargetInfo::CreateTargetInfo(DiagnosticsEngine &Diags,
                                         TargetOptions &Opts) {
  llvm::Triple Triple(Opts.Triple);

  // Construct the target
  OwningPtr<TargetInfo> Target(AllocateTarget(Triple.str()));
  if (!Target) {
    Diags.Report(diag::err_target_unknown_triple) << Triple.str();
    return 0;
  }

  // Set the target CPU if specified.
  if (!Opts.CPU.empty() && !Target->setCPU(Opts.CPU)) {
    Diags.Report(diag::err_target_unknown_cpu) << Opts.CPU;
    return 0;
  }

  // Set the target ABI if specified.
  if (!Opts.ABI.empty() && !Target->setABI(Opts.ABI)) {
    Diags.Report(diag::err_target_unknown_abi) << Opts.ABI;
    return 0;
  }

  // Set the target C++ ABI.
  if (!Opts.CXXABI.empty() && !Target->setCXXABI(Opts.CXXABI)) {
    Diags.Report(diag::err_target_unknown_cxxabi) << Opts.CXXABI;
    return 0;
  }

  // Compute the default target features, we need the target to handle this
  // because features may have dependencies on one another.
  llvm::StringMap<bool> Features;
  Target->getDefaultFeatures(Features);

  // Apply the user specified deltas.
  // First the enables.
  for (std::vector<std::string>::const_iterator it = Opts.Features.begin(),
         ie = Opts.Features.end(); it != ie; ++it) {
    const char *Name = it->c_str();

    if (Name[0] != '+')
      continue;

    // Apply the feature via the target.
    if (!Target->setFeatureEnabled(Features, Name + 1, true)) {
      Diags.Report(diag::err_target_invalid_feature) << Name;
      return 0;
    }
  }

  // Then the disables.
  for (std::vector<std::string>::const_iterator it = Opts.Features.begin(),
         ie = Opts.Features.end(); it != ie; ++it) {
    const char *Name = it->c_str();

    if (Name[0] == '+')
      continue;

    // Apply the feature via the target.
    if (Name[0] != '-' ||
        !Target->setFeatureEnabled(Features, Name + 1, false)) {
      Diags.Report(diag::err_target_invalid_feature) << Name;
      return 0;
    }
  }

  // Add the features to the compile options.
  //
  // FIXME: If we are completely confident that we have the right set, we only
  // need to pass the minuses.
  Opts.Features.clear();
  for (llvm::StringMap<bool>::const_iterator it = Features.begin(),
         ie = Features.end(); it != ie; ++it)
    Opts.Features.push_back((it->second ? "+" : "-") + it->first().str());
  Target->HandleTargetFeatures(Opts.Features);

  return Target.take();
}
