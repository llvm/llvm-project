//===--- Targets.cpp - Implement target feature support -------------------===//
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

#include "clang/Basic/Builtins.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetParser.h"
#include <algorithm>
#include <memory>

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

static TargetInfo *AllocateTarget(const llvm::Triple &Triple,
                                  const TargetOptions &Opts);

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
  OSTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TgtInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    TgtInfo::getTargetDefines(Opts, Builder);
    getOSDefines(Opts, TgtInfo::getTriple(), Builder);
  }

};

// CloudABI Target
template <typename Target>
class CloudABITargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    Builder.defineMacro("__CloudABI__");
    Builder.defineMacro("__ELF__");

    // CloudABI uses ISO/IEC 10646:2012 for wchar_t, char16_t and char32_t.
    Builder.defineMacro("__STDC_ISO_10646__", "201206L");
    Builder.defineMacro("__STDC_UTF_16__");
    Builder.defineMacro("__STDC_UTF_32__");
  }

public:
  CloudABITargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {}
};

// Ananas target
template<typename Target>
class AnanasTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // Ananas defines
    Builder.defineMacro("__Ananas__");
    Builder.defineMacro("__ELF__");
  }
public:
  AnanasTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {}
};

static void getDarwinDefines(MacroBuilder &Builder, const LangOptions &Opts,
                             const llvm::Triple &Triple,
                             StringRef &PlatformName,
                             VersionTuple &PlatformMinVersion) {
  Builder.defineMacro("__APPLE_CC__", "6000");
  Builder.defineMacro("__APPLE__");
  Builder.defineMacro("__STDC_NO_THREADS__");
  Builder.defineMacro("OBJC_NEW_PROPERTIES");
  // AddressSanitizer doesn't play well with source fortification, which is on
  // by default on Darwin.
  if (Opts.Sanitize.has(SanitizerKind::Address))
    Builder.defineMacro("_FORTIFY_SOURCE", "0");

  // Darwin defines __weak, __strong, and __unsafe_unretained even in C mode.
  if (!Opts.ObjC1) {
    // __weak is always defined, for use in blocks and with objc pointers.
    Builder.defineMacro("__weak", "__attribute__((objc_gc(weak)))");
    Builder.defineMacro("__strong", "");
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
    PlatformName = "macos";
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
  if (Triple.isiOS()) {
    assert(Maj < 100 && Min < 100 && Rev < 100 && "Invalid version!");
    char Str[7];
    if (Maj < 10) {
      Str[0] = '0' + Maj;
      Str[1] = '0' + (Min / 10);
      Str[2] = '0' + (Min % 10);
      Str[3] = '0' + (Rev / 10);
      Str[4] = '0' + (Rev % 10);
      Str[5] = '\0';
    } else {
      // Handle versions >= 10.
      Str[0] = '0' + (Maj / 10);
      Str[1] = '0' + (Maj % 10);
      Str[2] = '0' + (Min / 10);
      Str[3] = '0' + (Min % 10);
      Str[4] = '0' + (Rev / 10);
      Str[5] = '0' + (Rev % 10);
      Str[6] = '\0';
    }
    if (Triple.isTvOS())
      Builder.defineMacro("__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__", Str);
    else
      Builder.defineMacro("__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__",
                          Str);

  } else if (Triple.isWatchOS()) {
    assert(Maj < 10 && Min < 100 && Rev < 100 && "Invalid version!");
    char Str[6];
    Str[0] = '0' + Maj;
    Str[1] = '0' + (Min / 10);
    Str[2] = '0' + (Min % 10);
    Str[3] = '0' + (Rev / 10);
    Str[4] = '0' + (Rev % 10);
    Str[5] = '\0';
    Builder.defineMacro("__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__", Str);
  } else if (Triple.isMacOSX()) {
    // Note that the Driver allows versions which aren't representable in the
    // define (because we only get a single digit for the minor and micro
    // revision numbers). So, we limit them to the maximum representable
    // version.
    assert(Maj < 100 && Min < 100 && Rev < 100 && "Invalid version!");
    char Str[7];
    if (Maj < 10 || (Maj == 10 && Min < 10)) {
      Str[0] = '0' + (Maj / 10);
      Str[1] = '0' + (Maj % 10);
      Str[2] = '0' + std::min(Min, 9U);
      Str[3] = '0' + std::min(Rev, 9U);
      Str[4] = '\0';
    } else {
      // Handle versions > 10.9.
      Str[0] = '0' + (Maj / 10);
      Str[1] = '0' + (Maj % 10);
      Str[2] = '0' + (Min / 10);
      Str[3] = '0' + (Min % 10);
      Str[4] = '0' + (Rev / 10);
      Str[5] = '0' + (Rev % 10);
      Str[6] = '\0';
    }
    Builder.defineMacro("__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__", Str);
  }

  // Tell users about the kernel if there is one.
  if (Triple.isOSDarwin())
    Builder.defineMacro("__MACH__");

  // The Watch ABI uses Dwarf EH.
  if(Triple.isWatchABI())
    Builder.defineMacro("__ARM_DWARF_EH__");

  PlatformMinVersion = VersionTuple(Maj, Min, Rev);
}

template<typename Target>
class DarwinTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    getDarwinDefines(Builder, Opts, Triple, this->PlatformName,
                     this->PlatformMinVersion);
  }

public:
  DarwinTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    // By default, no TLS, and we whitelist permitted architecture/OS
    // combinations.
    this->TLSSupported = false;

    if (Triple.isMacOSX())
      this->TLSSupported = !Triple.isMacOSXVersionLT(10, 7);
    else if (Triple.isiOS()) {
      // 64-bit iOS supported it from 8 onwards, 32-bit from 9 onwards.
      if (Triple.getArch() == llvm::Triple::x86_64 ||
          Triple.getArch() == llvm::Triple::aarch64)
        this->TLSSupported = !Triple.isOSVersionLT(8);
      else if (Triple.getArch() == llvm::Triple::x86 ||
               Triple.getArch() == llvm::Triple::arm ||
               Triple.getArch() == llvm::Triple::thumb)
        this->TLSSupported = !Triple.isOSVersionLT(9);
    } else if (Triple.isWatchOS())
      this->TLSSupported = !Triple.isOSVersionLT(2);

    this->MCountName = "\01mcount";
  }

  std::string isValidSectionSpecifier(StringRef SR) const override {
    // Let MCSectionMachO validate this.
    StringRef Segment, Section;
    unsigned TAA, StubSize;
    bool HasTAA;
    return llvm::MCSectionMachO::ParseSectionSpecifier(SR, Segment, Section,
                                                       TAA, HasTAA, StubSize);
  }

  const char *getStaticInitSectionSpecifier() const override {
    // FIXME: We should return 0 when building kexts.
    return "__TEXT,__StaticInit,regular,pure_instructions";
  }

  /// Darwin does not support protected visibility.  Darwin's "default"
  /// is very similar to ELF's "protected";  Darwin requires a "weak"
  /// attribute on declarations that can be dynamically replaced.
  bool hasProtectedVisibility() const override {
    return false;
  }

  unsigned getExnObjectAlignment() const override {
    // The alignment of an exception object is 8-bytes for darwin since
    // libc++abi doesn't declare _Unwind_Exception with __attribute__((aligned))
    // and therefore doesn't guarantee 16-byte alignment.
    return  64;
  }
};


// DragonFlyBSD Target
template<typename Target>
class DragonFlyBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // DragonFly defines; list based off of gcc output
    Builder.defineMacro("__DragonFly__");
    Builder.defineMacro("__DragonFly_cc_version", "100001");
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__KPRINTF_ATTRIBUTE__");
    Builder.defineMacro("__tune_i386__");
    DefineStd(Builder, "unix", Opts);
  }
public:
  DragonFlyBSDTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    switch (Triple.getArch()) {
    default:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      this->MCountName = ".mcount";
      break;
    }
  }
};

#ifndef FREEBSD_CC_VERSION
#define FREEBSD_CC_VERSION 0U
#endif

// FreeBSD Target
template<typename Target>
class FreeBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // FreeBSD defines; list based off of gcc output

    unsigned Release = Triple.getOSMajorVersion();
    if (Release == 0U)
      Release = 8U;
    unsigned CCVersion = FREEBSD_CC_VERSION;
    if (CCVersion == 0U)
      CCVersion = Release * 100000U + 1U;

    Builder.defineMacro("__FreeBSD__", Twine(Release));
    Builder.defineMacro("__FreeBSD_cc_version", Twine(CCVersion));
    Builder.defineMacro("__KPRINTF_ATTRIBUTE__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");

    // On FreeBSD, wchar_t contains the number of the code point as
    // used by the character set of the locale. These character sets are
    // not necessarily a superset of ASCII.
    //
    // FIXME: This is wrong; the macro refers to the numerical values
    // of wchar_t *literals*, which are not locale-dependent. However,
    // FreeBSD systems apparently depend on us getting this wrong, and
    // setting this to 1 is conforming even if all the basic source
    // character literals have the same encoding as char and wchar_t.
    Builder.defineMacro("__STDC_MB_MIGHT_NEQ_WC__", "1");
  }
public:
  FreeBSDTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
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
    case llvm::Triple::ppc64le:
      this->MCountName = "_mcount";
      break;
    case llvm::Triple::arm:
      this->MCountName = "__mcount";
      break;
    }
  }
};

// GNU/kFreeBSD Target
template<typename Target>
class KFreeBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // GNU/kFreeBSD defines; list based off of gcc output

    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__FreeBSD_kernel__");
    Builder.defineMacro("__GLIBC__");
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
  }
public:
  KFreeBSDTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {}
};

// Haiku Target
template<typename Target>
class HaikuTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // Haiku defines; list based off of gcc output
    Builder.defineMacro("__HAIKU__");
    Builder.defineMacro("__ELF__");
    DefineStd(Builder, "unix", Opts);
  }
public:
  HaikuTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->SizeType = TargetInfo::UnsignedLong;
    this->IntPtrType = TargetInfo::SignedLong;
    this->PtrDiffType = TargetInfo::SignedLong;
    this->ProcessIDType = TargetInfo::SignedLong;
    this->TLSSupported = false;

  }
};

// Minix Target
template<typename Target>
class MinixTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
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
  MinixTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {}
};

// Linux target
template<typename Target>
class LinuxTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // Linux defines; list based off of gcc output
    DefineStd(Builder, "unix", Opts);
    DefineStd(Builder, "linux", Opts);
    Builder.defineMacro("__gnu_linux__");
    Builder.defineMacro("__ELF__");
    if (Triple.isAndroid()) {
      Builder.defineMacro("__ANDROID__", "1");
      unsigned Maj, Min, Rev;
      Triple.getEnvironmentVersion(Maj, Min, Rev);
      this->PlatformName = "android";
      this->PlatformMinVersion = VersionTuple(Maj, Min, Rev);
      if (Maj)
        Builder.defineMacro("__ANDROID_API__", Twine(Maj));
    }
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
    if (this->HasFloat128)
      Builder.defineMacro("__FLOAT128__");
  }
public:
  LinuxTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->WIntType = TargetInfo::UnsignedInt;

    switch (Triple.getArch()) {
    default:
      break;
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
    case llvm::Triple::ppc64le:
      this->MCountName = "_mcount";
      break;
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
    case llvm::Triple::systemz:
      this->HasFloat128 = true;
      break;
    }
  }

  const char *getStaticInitSectionSpecifier() const override {
    return ".text.startup";
  }
};

// NetBSD Target
template<typename Target>
class NetBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // NetBSD defines; list based off of gcc output
    Builder.defineMacro("__NetBSD__");
    Builder.defineMacro("__unix__");
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");

    switch (Triple.getArch()) {
    default:
      break;
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      Builder.defineMacro("__ARM_DWARF_EH__");
      break;
    }
  }
public:
  NetBSDTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->MCountName = "_mcount";
  }
};

// OpenBSD Target
template<typename Target>
class OpenBSDTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // OpenBSD defines; list based off of gcc output

    Builder.defineMacro("__OpenBSD__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    if (this->HasFloat128)
      Builder.defineMacro("__FLOAT128__");
  }
public:
  OpenBSDTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
      switch (Triple.getArch()) {
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
          this->HasFloat128 = true;
          // FALLTHROUGH
        default:
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

// Bitrig Target
template<typename Target>
class BitrigTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // Bitrig defines; list based off of gcc output

    Builder.defineMacro("__Bitrig__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");

    switch (Triple.getArch()) {
    default:
      break;
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      Builder.defineMacro("__ARM_DWARF_EH__");
      break;
    }
  }
public:
  BitrigTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->MCountName = "__mcount";
  }
};

// PSP Target
template<typename Target>
class PSPTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // PSP defines; list based on the output of the pspdev gcc toolchain.
    Builder.defineMacro("PSP");
    Builder.defineMacro("_PSP");
    Builder.defineMacro("__psp__");
    Builder.defineMacro("__ELF__");
  }
public:
  PSPTargetInfo(const llvm::Triple &Triple) : OSTargetInfo<Target>(Triple) {}
};

// PS3 PPU Target
template<typename Target>
class PS3PPUTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
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
  PS3PPUTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->LongWidth = this->LongAlign = 32;
    this->PointerWidth = this->PointerAlign = 32;
    this->IntMaxType = TargetInfo::SignedLongLong;
    this->Int64Type = TargetInfo::SignedLongLong;
    this->SizeType = TargetInfo::UnsignedInt;
    this->resetDataLayout("E-m:e-p:32:32-i64:64-n32:64");
  }
};

template <typename Target>
class PS4OSTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    Builder.defineMacro("__FreeBSD__", "9");
    Builder.defineMacro("__FreeBSD_cc_version", "900001");
    Builder.defineMacro("__KPRINTF_ATTRIBUTE__");
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__ORBIS__");
  }
public:
  PS4OSTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->WCharType = this->UnsignedShort;

    // On PS4, TLS variable cannot be aligned to more than 32 bytes (256 bits).
    this->MaxTLSAlign = 256;

    // On PS4, do not honor explicit bit field alignment,
    // as in "__attribute__((aligned(2))) int b : 1;".
    this->UseExplicitBitFieldAlignment = false;

    switch (Triple.getArch()) {
    default:
    case llvm::Triple::x86_64:
      this->MCountName = ".mcount";
      break;
    }
  }
};

// Solaris target
template<typename Target>
class SolarisTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    DefineStd(Builder, "sun", Opts);
    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__svr4__");
    Builder.defineMacro("__SVR4");
    // Solaris headers require _XOPEN_SOURCE to be set to 600 for C99 and
    // newer, but to 500 for everything else.  feature_test.h has a check to
    // ensure that you are not using C99 with an old version of X/Open or C89
    // with a new version.
    if (Opts.C99)
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
  SolarisTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->WCharType = this->SignedInt;
    // FIXME: WIntType should be SignedLong
  }
};

// Windows target
template<typename Target>
class WindowsTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    Builder.defineMacro("_WIN32");
  }
  void getVisualStudioDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    if (Opts.CPlusPlus) {
      if (Opts.RTTIData)
        Builder.defineMacro("_CPPRTTI");

      if (Opts.CXXExceptions)
        Builder.defineMacro("_CPPUNWIND");
    }

    if (Opts.Bool)
      Builder.defineMacro("__BOOL_DEFINED");

    if (!Opts.CharIsSigned)
      Builder.defineMacro("_CHAR_UNSIGNED");

    // FIXME: POSIXThreads isn't exactly the option this should be defined for,
    //        but it works for now.
    if (Opts.POSIXThreads)
      Builder.defineMacro("_MT");

    if (Opts.MSCompatibilityVersion) {
      Builder.defineMacro("_MSC_VER",
                          Twine(Opts.MSCompatibilityVersion / 100000));
      Builder.defineMacro("_MSC_FULL_VER", Twine(Opts.MSCompatibilityVersion));
      // FIXME We cannot encode the revision information into 32-bits
      Builder.defineMacro("_MSC_BUILD", Twine(1));

      if (Opts.CPlusPlus11 && Opts.isCompatibleWithMSVC(LangOptions::MSVC2015))
        Builder.defineMacro("_HAS_CHAR16_T_LANGUAGE_SUPPORT", Twine(1));

      if (Opts.isCompatibleWithMSVC(LangOptions::MSVC2015)) {
        if (Opts.CPlusPlus1z)
          Builder.defineMacro("_MSVC_LANG", "201403L");
        else if (Opts.CPlusPlus14)
          Builder.defineMacro("_MSVC_LANG", "201402L");
      }
    }

    if (Opts.MicrosoftExt) {
      Builder.defineMacro("_MSC_EXTENSIONS");

      if (Opts.CPlusPlus11) {
        Builder.defineMacro("_RVALUE_REFERENCES_V2_SUPPORTED");
        Builder.defineMacro("_RVALUE_REFERENCES_SUPPORTED");
        Builder.defineMacro("_NATIVE_NULLPTR_SUPPORTED");
      }
    }

    Builder.defineMacro("_INTEGRAL_MAX_BITS", "64");
  }

public:
  WindowsTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {}
};

template <typename Target>
class NaClTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");

    DefineStd(Builder, "unix", Opts);
    Builder.defineMacro("__ELF__");
    Builder.defineMacro("__native_client__");
  }

public:
  NaClTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->LongAlign = 32;
    this->LongWidth = 32;
    this->PointerAlign = 32;
    this->PointerWidth = 32;
    this->IntMaxType = TargetInfo::SignedLongLong;
    this->Int64Type = TargetInfo::SignedLongLong;
    this->DoubleAlign = 64;
    this->LongDoubleWidth = 64;
    this->LongDoubleAlign = 64;
    this->LongLongWidth = 64;
    this->LongLongAlign = 64;
    this->SizeType = TargetInfo::UnsignedInt;
    this->PtrDiffType = TargetInfo::SignedInt;
    this->IntPtrType = TargetInfo::SignedInt;
    // RegParmMax is inherited from the underlying architecture.
    this->LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    if (Triple.getArch() == llvm::Triple::arm) {
      // Handled in ARM's setABI().
    } else if (Triple.getArch() == llvm::Triple::x86) {
      this->resetDataLayout("e-m:e-p:32:32-i64:64-n8:16:32-S128");
    } else if (Triple.getArch() == llvm::Triple::x86_64) {
      this->resetDataLayout("e-m:e-p:32:32-i64:64-n8:16:32:64-S128");
    } else if (Triple.getArch() == llvm::Triple::mipsel) {
      // Handled on mips' setDataLayout.
    } else {
      assert(Triple.getArch() == llvm::Triple::le32);
      this->resetDataLayout("e-p:32:32-i64:64");
    }
  }
};

// Fuchsia Target
template<typename Target>
class FuchsiaTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    Builder.defineMacro("__Fuchsia__");
    Builder.defineMacro("__ELF__");
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    // Required by the libc++ locale support.
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
  }
public:
  FuchsiaTargetInfo(const llvm::Triple &Triple,
                    const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->MCountName = "__mcount";
  }
};

// WebAssembly target
template <typename Target>
class WebAssemblyOSTargetInfo : public OSTargetInfo<Target> {
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const final {
    // A common platform macro.
    if (Opts.POSIXThreads)
      Builder.defineMacro("_REENTRANT");
    // Follow g++ convention and predefine _GNU_SOURCE for C++.
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
  }

  // As an optimization, group static init code together in a section.
  const char *getStaticInitSectionSpecifier() const final {
    return ".text.__startup";
  }

public:
  explicit WebAssemblyOSTargetInfo(const llvm::Triple &Triple,
                                   const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    this->MCountName = "__mcount";
    this->TheCXXABI.set(TargetCXXABI::WebAssembly);
  }
};

//===----------------------------------------------------------------------===//
// Specific target implementations.
//===----------------------------------------------------------------------===//

// PPC abstract base class
class PPCTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
  static const char * const GCCRegNames[];
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  std::string CPU;

  // Target cpu features.
  bool HasAltivec;
  bool HasVSX;
  bool HasP8Vector;
  bool HasP8Crypto;
  bool HasDirectMove;
  bool HasQPX;
  bool HasHTM;
  bool HasBPERMD;
  bool HasExtDiv;
  bool HasP9Vector;

protected:
  std::string ABI;

public:
  PPCTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
    : TargetInfo(Triple), HasAltivec(false), HasVSX(false), HasP8Vector(false),
      HasP8Crypto(false), HasDirectMove(false), HasQPX(false), HasHTM(false),
      HasBPERMD(false), HasExtDiv(false), HasP9Vector(false) {
    SuitableAlign = 128;
    SimdDefaultAlign = 128;
    LongDoubleWidth = LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::PPCDoubleDouble();
  }

  /// \brief Flags for architecture specific defines.
  typedef enum {
    ArchDefineNone  = 0,
    ArchDefineName  = 1 << 0, // <name> is substituted for arch name.
    ArchDefinePpcgr = 1 << 1,
    ArchDefinePpcsq = 1 << 2,
    ArchDefine440   = 1 << 3,
    ArchDefine603   = 1 << 4,
    ArchDefine604   = 1 << 5,
    ArchDefinePwr4  = 1 << 6,
    ArchDefinePwr5  = 1 << 7,
    ArchDefinePwr5x = 1 << 8,
    ArchDefinePwr6  = 1 << 9,
    ArchDefinePwr6x = 1 << 10,
    ArchDefinePwr7  = 1 << 11,
    ArchDefinePwr8  = 1 << 12,
    ArchDefinePwr9  = 1 << 13,
    ArchDefineA2    = 1 << 14,
    ArchDefineA2q   = 1 << 15
  } ArchDefineTypes;

  // Set the language option for altivec based on our value.
  void adjust(LangOptions &Opts) override {
    if (HasAltivec)
      Opts.AltiVec = 1;
    TargetInfo::adjust(Opts);
  }

  // Note: GCC recognizes the following additional cpus:
  //  401, 403, 405, 405fp, 440fp, 464, 464fp, 476, 476fp, 505, 740, 801,
  //  821, 823, 8540, 8548, e300c2, e300c3, e500mc64, e6500, 860, cell,
  //  titan, rs64.
  bool setCPU(const std::string &Name) override {
    bool CPUKnown = llvm::StringSwitch<bool>(Name)
      .Case("generic", true)
      .Case("440", true)
      .Case("450", true)
      .Case("601", true)
      .Case("602", true)
      .Case("603", true)
      .Case("603e", true)
      .Case("603ev", true)
      .Case("604", true)
      .Case("604e", true)
      .Case("620", true)
      .Case("630", true)
      .Case("g3", true)
      .Case("7400", true)
      .Case("g4", true)
      .Case("7450", true)
      .Case("g4+", true)
      .Case("750", true)
      .Case("970", true)
      .Case("g5", true)
      .Case("a2", true)
      .Case("a2q", true)
      .Case("e500mc", true)
      .Case("e5500", true)
      .Case("power3", true)
      .Case("pwr3", true)
      .Case("power4", true)
      .Case("pwr4", true)
      .Case("power5", true)
      .Case("pwr5", true)
      .Case("power5x", true)
      .Case("pwr5x", true)
      .Case("power6", true)
      .Case("pwr6", true)
      .Case("power6x", true)
      .Case("pwr6x", true)
      .Case("power7", true)
      .Case("pwr7", true)
      .Case("power8", true)
      .Case("pwr8", true)
      .Case("power9", true)
      .Case("pwr9", true)
      .Case("powerpc", true)
      .Case("ppc", true)
      .Case("powerpc64", true)
      .Case("ppc64", true)
      .Case("powerpc64le", true)
      .Case("ppc64le", true)
      .Default(false);

    if (CPUKnown)
      CPU = Name;

    return CPUKnown;
  }


  StringRef getABI() const override { return ABI; }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                             clang::PPC::LastTSBuiltin-Builtin::FirstTSBuiltin);
  }

  bool isCLZForZeroUndef() const override { return false; }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override;

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;
  bool hasFeature(StringRef Feature) const override;
  void setFeatureEnabled(llvm::StringMap<bool> &Features, StringRef Name,
                         bool Enabled) const override;

  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
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
        case 'c':// An individual CR bit
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
  std::string convertConstraint(const char *&Constraint) const override {
    std::string R;
    switch (*Constraint) {
    case 'e':
    case 'w':
      // Two-character constraint; add "^" hint for later parsing.
      R = std::string("^") + std::string(Constraint, 2);
      Constraint++;
      break;
    default:
      return TargetInfo::convertConstraint(Constraint);
    }
    return R;
  }
  const char *getClobbers() const override {
    return "";
  }
  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0) return 3;
    if (RegNo == 1) return 4;
    return -1;
  }

  bool hasSjLjLowering() const override {
    return true;
  }

  bool useFloat128ManglingForLongDouble() const override {
    return LongDoubleWidth == 128 &&
           LongDoubleFormat == &llvm::APFloat::PPCDoubleDouble() &&
           getTriple().isOSBinFormatELF();
  }
};

const Builtin::Info PPCTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsPPC.def"
};

/// handleTargetFeatures - Perform initialization based on the user
/// configured set of features.
bool PPCTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                         DiagnosticsEngine &Diags) {
  for (const auto &Feature : Features) {
    if (Feature == "+altivec") {
      HasAltivec = true;
    } else if (Feature == "+vsx") {
      HasVSX = true;
    } else if (Feature == "+bpermd") {
      HasBPERMD = true;
    } else if (Feature == "+extdiv") {
      HasExtDiv = true;
    } else if (Feature == "+power8-vector") {
      HasP8Vector = true;
    } else if (Feature == "+crypto") {
      HasP8Crypto = true;
    } else if (Feature == "+direct-move") {
      HasDirectMove = true;
    } else if (Feature == "+qpx") {
      HasQPX = true;
    } else if (Feature == "+htm") {
      HasHTM = true;
    } else if (Feature == "+float128") {
      HasFloat128 = true;
    } else if (Feature == "+power9-vector") {
      HasP9Vector = true;
    }
    // TODO: Finish this list and add an assert that we've handled them
    // all.
  }

  return true;
}

/// PPCTargetInfo::getTargetDefines - Return a set of the PowerPC-specific
/// #defines that are not tied to a specific subtarget.
void PPCTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  Builder.defineMacro("__ppc__");
  Builder.defineMacro("__PPC__");
  Builder.defineMacro("_ARCH_PPC");
  Builder.defineMacro("__powerpc__");
  Builder.defineMacro("__POWERPC__");
  if (PointerWidth == 64) {
    Builder.defineMacro("_ARCH_PPC64");
    Builder.defineMacro("__powerpc64__");
    Builder.defineMacro("__ppc64__");
    Builder.defineMacro("__PPC64__");
  }

  // Target properties.
  if (getTriple().getArch() == llvm::Triple::ppc64le) {
    Builder.defineMacro("_LITTLE_ENDIAN");
  } else {
    if (getTriple().getOS() != llvm::Triple::NetBSD &&
        getTriple().getOS() != llvm::Triple::OpenBSD)
      Builder.defineMacro("_BIG_ENDIAN");
  }

  // ABI options.
  if (ABI == "elfv1" || ABI == "elfv1-qpx")
    Builder.defineMacro("_CALL_ELF", "1");
  if (ABI == "elfv2")
    Builder.defineMacro("_CALL_ELF", "2");

  // This typically is only for a new enough linker (bfd >= 2.16.2 or gold), but
  // our suppport post-dates this and it should work on all 64-bit ppc linux
  // platforms. It is guaranteed to work on all elfv2 platforms.
  if (getTriple().getOS() == llvm::Triple::Linux && PointerWidth == 64)
    Builder.defineMacro("_CALL_LINUX", "1");

  // Subtarget options.
  Builder.defineMacro("__NATURAL_ALIGNMENT__");
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // FIXME: Should be controlled by command line option.
  if (LongDoubleWidth == 128) {
    Builder.defineMacro("__LONG_DOUBLE_128__");
    Builder.defineMacro("__LONGDOUBLE128");
  }

  // Define this for elfv2 (64-bit only) or 64-bit darwin.
  if (ABI == "elfv2" ||
      (getTriple().getOS() == llvm::Triple::Darwin && PointerWidth == 64))
    Builder.defineMacro("__STRUCT_PARM_ALIGN__", "16");

  // CPU identification.
  ArchDefineTypes defs =
      (ArchDefineTypes)llvm::StringSwitch<int>(CPU)
          .Case("440", ArchDefineName)
          .Case("450", ArchDefineName | ArchDefine440)
          .Case("601", ArchDefineName)
          .Case("602", ArchDefineName | ArchDefinePpcgr)
          .Case("603", ArchDefineName | ArchDefinePpcgr)
          .Case("603e", ArchDefineName | ArchDefine603 | ArchDefinePpcgr)
          .Case("603ev", ArchDefineName | ArchDefine603 | ArchDefinePpcgr)
          .Case("604", ArchDefineName | ArchDefinePpcgr)
          .Case("604e", ArchDefineName | ArchDefine604 | ArchDefinePpcgr)
          .Case("620", ArchDefineName | ArchDefinePpcgr)
          .Case("630", ArchDefineName | ArchDefinePpcgr)
          .Case("7400", ArchDefineName | ArchDefinePpcgr)
          .Case("7450", ArchDefineName | ArchDefinePpcgr)
          .Case("750", ArchDefineName | ArchDefinePpcgr)
          .Case("970", ArchDefineName | ArchDefinePwr4 | ArchDefinePpcgr |
                           ArchDefinePpcsq)
          .Case("a2", ArchDefineA2)
          .Case("a2q", ArchDefineName | ArchDefineA2 | ArchDefineA2q)
          .Case("pwr3", ArchDefinePpcgr)
          .Case("pwr4", ArchDefineName | ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("pwr5", ArchDefineName | ArchDefinePwr4 | ArchDefinePpcgr |
                            ArchDefinePpcsq)
          .Case("pwr5x", ArchDefineName | ArchDefinePwr5 | ArchDefinePwr4 |
                             ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("pwr6", ArchDefineName | ArchDefinePwr5x | ArchDefinePwr5 |
                            ArchDefinePwr4 | ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("pwr6x", ArchDefineName | ArchDefinePwr6 | ArchDefinePwr5x |
                             ArchDefinePwr5 | ArchDefinePwr4 | ArchDefinePpcgr |
                             ArchDefinePpcsq)
          .Case("pwr7", ArchDefineName | ArchDefinePwr6x | ArchDefinePwr6 |
                            ArchDefinePwr5x | ArchDefinePwr5 | ArchDefinePwr4 |
                            ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("pwr8", ArchDefineName | ArchDefinePwr7 | ArchDefinePwr6x |
                            ArchDefinePwr6 | ArchDefinePwr5x | ArchDefinePwr5 |
                            ArchDefinePwr4 | ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("pwr9", ArchDefineName | ArchDefinePwr8 | ArchDefinePwr7 |
                            ArchDefinePwr6x | ArchDefinePwr6 | ArchDefinePwr5x |
                            ArchDefinePwr5 | ArchDefinePwr4 | ArchDefinePpcgr |
                            ArchDefinePpcsq)
          .Case("power3", ArchDefinePpcgr)
          .Case("power4", ArchDefinePwr4 | ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("power5", ArchDefinePwr5 | ArchDefinePwr4 | ArchDefinePpcgr |
                              ArchDefinePpcsq)
          .Case("power5x", ArchDefinePwr5x | ArchDefinePwr5 | ArchDefinePwr4 |
                               ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("power6", ArchDefinePwr6 | ArchDefinePwr5x | ArchDefinePwr5 |
                              ArchDefinePwr4 | ArchDefinePpcgr |
                              ArchDefinePpcsq)
          .Case("power6x", ArchDefinePwr6x | ArchDefinePwr6 | ArchDefinePwr5x |
                               ArchDefinePwr5 | ArchDefinePwr4 |
                               ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("power7", ArchDefinePwr7 | ArchDefinePwr6x | ArchDefinePwr6 |
                              ArchDefinePwr5x | ArchDefinePwr5 |
                              ArchDefinePwr4 | ArchDefinePpcgr |
                              ArchDefinePpcsq)
          .Case("power8", ArchDefinePwr8 | ArchDefinePwr7 | ArchDefinePwr6x |
                              ArchDefinePwr6 | ArchDefinePwr5x |
                              ArchDefinePwr5 | ArchDefinePwr4 |
                              ArchDefinePpcgr | ArchDefinePpcsq)
          .Case("power9", ArchDefinePwr9 | ArchDefinePwr8 | ArchDefinePwr7 |
                              ArchDefinePwr6x | ArchDefinePwr6 |
                              ArchDefinePwr5x | ArchDefinePwr5 |
                              ArchDefinePwr4 | ArchDefinePpcgr |
                              ArchDefinePpcsq)
          // powerpc64le automatically defaults to at least power8.
          .Case("ppc64le", ArchDefinePwr8 | ArchDefinePwr7 | ArchDefinePwr6x |
                               ArchDefinePwr6 | ArchDefinePwr5x |
                               ArchDefinePwr5 | ArchDefinePwr4 |
                               ArchDefinePpcgr | ArchDefinePpcsq)
          .Default(ArchDefineNone);

  if (defs & ArchDefineName)
    Builder.defineMacro(Twine("_ARCH_", StringRef(CPU).upper()));
  if (defs & ArchDefinePpcgr)
    Builder.defineMacro("_ARCH_PPCGR");
  if (defs & ArchDefinePpcsq)
    Builder.defineMacro("_ARCH_PPCSQ");
  if (defs & ArchDefine440)
    Builder.defineMacro("_ARCH_440");
  if (defs & ArchDefine603)
    Builder.defineMacro("_ARCH_603");
  if (defs & ArchDefine604)
    Builder.defineMacro("_ARCH_604");
  if (defs & ArchDefinePwr4)
    Builder.defineMacro("_ARCH_PWR4");
  if (defs & ArchDefinePwr5)
    Builder.defineMacro("_ARCH_PWR5");
  if (defs & ArchDefinePwr5x)
    Builder.defineMacro("_ARCH_PWR5X");
  if (defs & ArchDefinePwr6)
    Builder.defineMacro("_ARCH_PWR6");
  if (defs & ArchDefinePwr6x)
    Builder.defineMacro("_ARCH_PWR6X");
  if (defs & ArchDefinePwr7)
    Builder.defineMacro("_ARCH_PWR7");
  if (defs & ArchDefinePwr8)
    Builder.defineMacro("_ARCH_PWR8");
  if (defs & ArchDefinePwr9)
    Builder.defineMacro("_ARCH_PWR9");
  if (defs & ArchDefineA2)
    Builder.defineMacro("_ARCH_A2");
  if (defs & ArchDefineA2q) {
    Builder.defineMacro("_ARCH_A2Q");
    Builder.defineMacro("_ARCH_QP");
  }

  if (getTriple().getVendor() == llvm::Triple::BGQ) {
    Builder.defineMacro("__bg__");
    Builder.defineMacro("__THW_BLUEGENE__");
    Builder.defineMacro("__bgq__");
    Builder.defineMacro("__TOS_BGQ__");
  }

  if (HasAltivec) {
    Builder.defineMacro("__VEC__", "10206");
    Builder.defineMacro("__ALTIVEC__");
  }
  if (HasVSX)
    Builder.defineMacro("__VSX__");
  if (HasP8Vector)
    Builder.defineMacro("__POWER8_VECTOR__");
  if (HasP8Crypto)
    Builder.defineMacro("__CRYPTO__");
  if (HasHTM)
    Builder.defineMacro("__HTM__");
  if (HasFloat128)
    Builder.defineMacro("__FLOAT128__");
  if (HasP9Vector)
    Builder.defineMacro("__POWER9_VECTOR__");

  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
  if (PointerWidth == 64)
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");

  // We have support for the bswap intrinsics so we can define this.
  Builder.defineMacro("__HAVE_BSWAP__", "1");

  // FIXME: The following are not yet generated here by Clang, but are
  //        generated by GCC:
  //
  //   _SOFT_FLOAT_
  //   __RECIP_PRECISION__
  //   __APPLE_ALTIVEC__
  //   __RECIP__
  //   __RECIPF__
  //   __RSQRTE__
  //   __RSQRTEF__
  //   _SOFT_DOUBLE_
  //   __NO_LWSYNC__
  //   __CMODEL_MEDIUM__
  //   __CMODEL_LARGE__
  //   _CALL_SYSV
  //   _CALL_DARWIN
  //   __NO_FPRS__
}

// Handle explicit options being passed to the compiler here: if we've
// explicitly turned off vsx and turned on any of:
// - power8-vector
// - direct-move
// - float128
// - power9-vector
// then go ahead and error since the customer has expressed an incompatible
// set of options.
static bool ppcUserFeaturesCheck(DiagnosticsEngine &Diags,
                                 const std::vector<std::string> &FeaturesVec) {

  if (std::find(FeaturesVec.begin(), FeaturesVec.end(), "-vsx") !=
      FeaturesVec.end()) {
    if (std::find(FeaturesVec.begin(), FeaturesVec.end(), "+power8-vector") !=
        FeaturesVec.end()) {
      Diags.Report(diag::err_opt_not_valid_with_opt) << "-mpower8-vector"
                                                     << "-mno-vsx";
      return false;
    }

    if (std::find(FeaturesVec.begin(), FeaturesVec.end(), "+direct-move") !=
        FeaturesVec.end()) {
      Diags.Report(diag::err_opt_not_valid_with_opt) << "-mdirect-move"
                                                     << "-mno-vsx";
      return false;
    }

    if (std::find(FeaturesVec.begin(), FeaturesVec.end(), "+float128") !=
        FeaturesVec.end()) {
      Diags.Report(diag::err_opt_not_valid_with_opt) << "-mfloat128"
                                                     << "-mno-vsx";
      return false;
    }

    if (std::find(FeaturesVec.begin(), FeaturesVec.end(), "+power9-vector") !=
        FeaturesVec.end()) {
      Diags.Report(diag::err_opt_not_valid_with_opt) << "-mpower9-vector"
                                                     << "-mno-vsx";
      return false;
    }
  }

  return true;
}

bool PPCTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  Features["altivec"] = llvm::StringSwitch<bool>(CPU)
    .Case("7400", true)
    .Case("g4", true)
    .Case("7450", true)
    .Case("g4+", true)
    .Case("970", true)
    .Case("g5", true)
    .Case("pwr6", true)
    .Case("pwr7", true)
    .Case("pwr8", true)
    .Case("pwr9", true)
    .Case("ppc64", true)
    .Case("ppc64le", true)
    .Default(false);

  Features["qpx"] = (CPU == "a2q");
  Features["power9-vector"] = (CPU == "pwr9");
  Features["crypto"] = llvm::StringSwitch<bool>(CPU)
    .Case("ppc64le", true)
    .Case("pwr9", true)
    .Case("pwr8", true)
    .Default(false);
  Features["power8-vector"] = llvm::StringSwitch<bool>(CPU)
    .Case("ppc64le", true)
    .Case("pwr9", true)
    .Case("pwr8", true)
    .Default(false);
  Features["bpermd"] = llvm::StringSwitch<bool>(CPU)
    .Case("ppc64le", true)
    .Case("pwr9", true)
    .Case("pwr8", true)
    .Case("pwr7", true)
    .Default(false);
  Features["extdiv"] = llvm::StringSwitch<bool>(CPU)
    .Case("ppc64le", true)
    .Case("pwr9", true)
    .Case("pwr8", true)
    .Case("pwr7", true)
    .Default(false);
  Features["direct-move"] = llvm::StringSwitch<bool>(CPU)
    .Case("ppc64le", true)
    .Case("pwr9", true)
    .Case("pwr8", true)
    .Default(false);
  Features["vsx"] = llvm::StringSwitch<bool>(CPU)
    .Case("ppc64le", true)
    .Case("pwr9", true)
    .Case("pwr8", true)
    .Case("pwr7", true)
    .Default(false);
  Features["htm"] = llvm::StringSwitch<bool>(CPU)
    .Case("ppc64le", true)
    .Case("pwr9", true)
    .Case("pwr8", true)
    .Default(false);

  if (!ppcUserFeaturesCheck(Diags, FeaturesVec))
    return false;

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

bool PPCTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("powerpc", true)
      .Case("altivec", HasAltivec)
      .Case("vsx", HasVSX)
      .Case("power8-vector", HasP8Vector)
      .Case("crypto", HasP8Crypto)
      .Case("direct-move", HasDirectMove)
      .Case("qpx", HasQPX)
      .Case("htm", HasHTM)
      .Case("bpermd", HasBPERMD)
      .Case("extdiv", HasExtDiv)
      .Case("float128", HasFloat128)
      .Case("power9-vector", HasP9Vector)
      .Default(false);
}

void PPCTargetInfo::setFeatureEnabled(llvm::StringMap<bool> &Features,
                                      StringRef Name, bool Enabled) const {
  if (Enabled) {
    // If we're enabling any of the vsx based features then enable vsx and
    // altivec. We'll diagnose any problems later.
    bool FeatureHasVSX = llvm::StringSwitch<bool>(Name)
                             .Case("vsx", true)
                             .Case("direct-move", true)
                             .Case("power8-vector", true)
                             .Case("power9-vector", true)
                             .Case("float128", true)
                             .Default(false);
    if (FeatureHasVSX)
      Features["vsx"] = Features["altivec"] = true;
    if (Name == "power9-vector")
      Features["power8-vector"] = true;
    Features[Name] = true;
  } else {
    // If we're disabling altivec or vsx go ahead and disable all of the vsx
    // features.
    if ((Name == "altivec") || (Name == "vsx"))
      Features["vsx"] = Features["direct-move"] = Features["power8-vector"] =
          Features["float128"] = Features["power9-vector"] = false;
    if (Name == "power8-vector")
      Features["power9-vector"] = false;
    Features[Name] = false;
  }
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

ArrayRef<const char*> PPCTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
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

ArrayRef<TargetInfo::GCCRegAlias> PPCTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

class PPC32TargetInfo : public PPCTargetInfo {
public:
  PPC32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : PPCTargetInfo(Triple, Opts) {
    resetDataLayout("E-m:e-p:32:32-i64:64-n32");

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
      LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    }

    // PPC32 supports atomics up to 4 bytes.
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 32;
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    // This is the ELF definition, and is overridden by the Darwin sub-target
    return TargetInfo::PowerABIBuiltinVaList;
  }
};

// Note: ABI differences may eventually require us to have a separate
// TargetInfo for little endian.
class PPC64TargetInfo : public PPCTargetInfo {
public:
  PPC64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : PPCTargetInfo(Triple, Opts) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    IntMaxType = SignedLong;
    Int64Type = SignedLong;

    if ((Triple.getArch() == llvm::Triple::ppc64le)) {
      resetDataLayout("e-m:e-i64:64-n32:64");
      ABI = "elfv2";
    } else {
      resetDataLayout("E-m:e-i64:64-n32:64");
      ABI = "elfv1";
    }

    switch (getTriple().getOS()) {
    case llvm::Triple::FreeBSD:
      LongDoubleWidth = LongDoubleAlign = 64;
      LongDoubleFormat = &llvm::APFloat::IEEEdouble();
      break;
    case llvm::Triple::NetBSD:
      IntMaxType = SignedLongLong;
      Int64Type = SignedLongLong;
      break;
    default:
      break;
    }

    // PPC64 supports atomics up to 8 bytes.
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
  // PPC64 Linux-specific ABI options.
  bool setABI(const std::string &Name) override {
    if (Name == "elfv1" || Name == "elfv1-qpx" || Name == "elfv2") {
      ABI = Name;
      return true;
    }
    return false;
  }
};

class DarwinPPC32TargetInfo : public DarwinTargetInfo<PPC32TargetInfo> {
public:
  DarwinPPC32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<PPC32TargetInfo>(Triple, Opts) {
    HasAlignMac68kSupport = true;
    BoolWidth = BoolAlign = 32; //XXX support -mone-byte-bool?
    PtrDiffType = SignedInt; // for http://llvm.org/bugs/show_bug.cgi?id=15726
    LongLongAlign = 32;
    resetDataLayout("E-m:o-p:32:32-f64:32:64-n32");
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
};

class DarwinPPC64TargetInfo : public DarwinTargetInfo<PPC64TargetInfo> {
public:
  DarwinPPC64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<PPC64TargetInfo>(Triple, Opts) {
    HasAlignMac68kSupport = true;
    resetDataLayout("E-m:o-i64:64-n32:64");
  }
};

static const unsigned NVPTXAddrSpaceMap[] = {
    0, // Default
    1, // opencl_global
    3, // opencl_local
    4, // opencl_constant
    // FIXME: generic has to be added to the target
    0, // opencl_generic
    1, // cuda_device
    4, // cuda_constant
    3, // cuda_shared
};

class NVPTXTargetInfo : public TargetInfo {
  static const char *const GCCRegNames[];
  static const Builtin::Info BuiltinInfo[];
  CudaArch GPU;
  std::unique_ptr<TargetInfo> HostTarget;

public:
  NVPTXTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts,
                  unsigned TargetPointerWidth)
      : TargetInfo(Triple) {
    assert((TargetPointerWidth == 32 || TargetPointerWidth == 64) &&
           "NVPTX only supports 32- and 64-bit modes.");

    TLSSupported = false;
    AddrSpaceMap = &NVPTXAddrSpaceMap;
    UseAddrSpaceMapMangling = true;

    // Define available target features
    // These must be defined in sorted order!
    NoAsmVariants = true;
    GPU = CudaArch::SM_20;

    if (TargetPointerWidth == 32)
      resetDataLayout("e-p:32:32-i64:64-v16:16-v32:32-n16:32:64");
    else
      resetDataLayout("e-i64:64-v16:16-v32:32-n16:32:64");

    // If possible, get a TargetInfo for our host triple, so we can match its
    // types.
    llvm::Triple HostTriple(Opts.HostTriple);
    if (!HostTriple.isNVPTX())
      HostTarget.reset(AllocateTarget(llvm::Triple(Opts.HostTriple), Opts));

    // If no host target, make some guesses about the data layout and return.
    if (!HostTarget) {
      LongWidth = LongAlign = TargetPointerWidth;
      PointerWidth = PointerAlign = TargetPointerWidth;
      switch (TargetPointerWidth) {
      case 32:
        SizeType = TargetInfo::UnsignedInt;
        PtrDiffType = TargetInfo::SignedInt;
        IntPtrType = TargetInfo::SignedInt;
        break;
      case 64:
        SizeType = TargetInfo::UnsignedLong;
        PtrDiffType = TargetInfo::SignedLong;
        IntPtrType = TargetInfo::SignedLong;
        break;
      default:
        llvm_unreachable("TargetPointerWidth must be 32 or 64");
      }
      return;
    }

    // Copy properties from host target.
    PointerWidth = HostTarget->getPointerWidth(/* AddrSpace = */ 0);
    PointerAlign = HostTarget->getPointerAlign(/* AddrSpace = */ 0);
    BoolWidth = HostTarget->getBoolWidth();
    BoolAlign = HostTarget->getBoolAlign();
    IntWidth = HostTarget->getIntWidth();
    IntAlign = HostTarget->getIntAlign();
    HalfWidth = HostTarget->getHalfWidth();
    HalfAlign = HostTarget->getHalfAlign();
    FloatWidth = HostTarget->getFloatWidth();
    FloatAlign = HostTarget->getFloatAlign();
    DoubleWidth = HostTarget->getDoubleWidth();
    DoubleAlign = HostTarget->getDoubleAlign();
    LongWidth = HostTarget->getLongWidth();
    LongAlign = HostTarget->getLongAlign();
    LongLongWidth = HostTarget->getLongLongWidth();
    LongLongAlign = HostTarget->getLongLongAlign();
    MinGlobalAlign = HostTarget->getMinGlobalAlign();
    NewAlign = HostTarget->getNewAlign();
    DefaultAlignForAttributeAligned =
        HostTarget->getDefaultAlignForAttributeAligned();
    SizeType = HostTarget->getSizeType();
    IntMaxType = HostTarget->getIntMaxType();
    PtrDiffType = HostTarget->getPtrDiffType(/* AddrSpace = */ 0);
    IntPtrType = HostTarget->getIntPtrType();
    WCharType = HostTarget->getWCharType();
    WIntType = HostTarget->getWIntType();
    Char16Type = HostTarget->getChar16Type();
    Char32Type = HostTarget->getChar32Type();
    Int64Type = HostTarget->getInt64Type();
    SigAtomicType = HostTarget->getSigAtomicType();
    ProcessIDType = HostTarget->getProcessIDType();

    UseBitFieldTypeAlignment = HostTarget->useBitFieldTypeAlignment();
    UseZeroLengthBitfieldAlignment =
        HostTarget->useZeroLengthBitfieldAlignment();
    UseExplicitBitFieldAlignment = HostTarget->useExplicitBitFieldAlignment();
    ZeroLengthBitfieldBoundary = HostTarget->getZeroLengthBitfieldBoundary();

    // This is a bit of a lie, but it controls __GCC_ATOMIC_XXX_LOCK_FREE, and
    // we need those macros to be identical on host and device, because (among
    // other things) they affect which standard library classes are defined, and
    // we need all classes to be defined on both the host and device.
    MaxAtomicInlineWidth = HostTarget->getMaxAtomicInlineWidth();

    // Properties intentionally not copied from host:
    // - LargeArrayMinWidth, LargeArrayAlign: Not visible across the
    //   host/device boundary.
    // - SuitableAlign: Not visible across the host/device boundary, and may
    //   correctly be different on host/device, e.g. if host has wider vector
    //   types than device.
    // - LongDoubleWidth, LongDoubleAlign: nvptx's long double type is the same
    //   as its double type, but that's not necessarily true on the host.
    //   TODO: nvcc emits a warning when using long double on device; we should
    //   do the same.
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__PTX__");
    Builder.defineMacro("__NVPTX__");
    if (Opts.CUDAIsDevice) {
      // Set __CUDA_ARCH__ for the GPU specified.
      std::string CUDAArchCode = [this] {
        switch (GPU) {
        case CudaArch::UNKNOWN:
          assert(false && "No GPU arch when compiling CUDA device code.");
          return "";
        case CudaArch::SM_20:
          return "200";
        case CudaArch::SM_21:
          return "210";
        case CudaArch::SM_30:
          return "300";
        case CudaArch::SM_32:
          return "320";
        case CudaArch::SM_35:
          return "350";
        case CudaArch::SM_37:
          return "370";
        case CudaArch::SM_50:
          return "500";
        case CudaArch::SM_52:
          return "520";
        case CudaArch::SM_53:
          return "530";
        case CudaArch::SM_60:
          return "600";
        case CudaArch::SM_61:
          return "610";
        case CudaArch::SM_62:
          return "620";
        }
        llvm_unreachable("unhandled CudaArch");
      }();
      Builder.defineMacro("__CUDA_ARCH__", CUDAArchCode);
    }
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                         clang::NVPTX::LastTSBuiltin - Builtin::FirstTSBuiltin);
  }
  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override {
    Features["satom"] = GPU >= CudaArch::SM_60;
    return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
  }

  bool hasFeature(StringRef Feature) const override {
    return llvm::StringSwitch<bool>(Feature)
        .Cases("ptx", "nvptx", true)
        .Case("satom", GPU >= CudaArch::SM_60)  // Atomics w/ scope.
        .Default(false);
  }

  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    // No aliases.
    return None;
  }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      return false;
    case 'c':
    case 'h':
    case 'r':
    case 'l':
    case 'f':
    case 'd':
      Info.setAllowsRegister();
      return true;
    }
  }
  const char *getClobbers() const override {
    // FIXME: Is this really right?
    return "";
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    // FIXME: implement
    return TargetInfo::CharPtrBuiltinVaList;
  }
  bool setCPU(const std::string &Name) override {
    GPU = StringToCudaArch(Name);
    return GPU != CudaArch::UNKNOWN;
  }
  void setSupportedOpenCLOpts() override {
    auto &Opts = getSupportedOpenCLOpts();
    Opts.support("cl_clang_storage_class_specifiers");
    Opts.support("cl_khr_gl_sharing");
    Opts.support("cl_khr_icd");

    Opts.support("cl_khr_fp64");
    Opts.support("cl_khr_byte_addressable_store");
    Opts.support("cl_khr_global_int32_base_atomics");
    Opts.support("cl_khr_global_int32_extended_atomics");
    Opts.support("cl_khr_local_int32_base_atomics");
    Opts.support("cl_khr_local_int32_extended_atomics");
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    // CUDA compilations support all of the host's calling conventions.
    //
    // TODO: We should warn if you apply a non-default CC to anything other than
    // a host function.
    if (HostTarget)
      return HostTarget->checkCallingConvention(CC);
    return CCCR_Warning;
  }
};

const Builtin::Info NVPTXTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE },
#include "clang/Basic/BuiltinsNVPTX.def"
};

const char *const NVPTXTargetInfo::GCCRegNames[] = {"r0"};

ArrayRef<const char *> NVPTXTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

static const LangAS::Map AMDGPUPrivIsZeroDefIsGenMap = {
    4, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    4, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};
static const LangAS::Map AMDGPUGenIsZeroDefIsGenMap = {
    0, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    0, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};
static const LangAS::Map AMDGPUPrivIsZeroDefIsPrivMap = {
    0, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    4, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};
static const LangAS::Map AMDGPUGenIsZeroDefIsPrivMap = {
    5, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    0, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};

// If you edit the description strings, make sure you update
// getPointerWidthV().

static const char *const DataLayoutStringR600 =
  "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
  "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64";

static const char *const DataLayoutStringSIPrivateIsZero =
  "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32"
  "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
  "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64";

static const char *const DataLayoutStringSIGenericIsZero =
  "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32"
  "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
  "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5";

class AMDGPUTargetInfo final : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
  static const char * const GCCRegNames[];

  struct AddrSpace {
    unsigned Generic, Global, Local, Constant, Private;
    AddrSpace(bool IsGenericZero_ = false){
      if (IsGenericZero_) {
        Generic   = 0;
        Global    = 1;
        Local     = 3;
        Constant  = 2;
        Private   = 5;
      } else {
        Generic   = 4;
        Global    = 1;
        Local     = 3;
        Constant  = 2;
        Private   = 0;
      }
    }
  };

  /// \brief The GPU profiles supported by the AMDGPU target.
  enum GPUKind {
    GK_NONE,
    GK_R600,
    GK_R600_DOUBLE_OPS,
    GK_R700,
    GK_R700_DOUBLE_OPS,
    GK_EVERGREEN,
    GK_EVERGREEN_DOUBLE_OPS,
    GK_NORTHERN_ISLANDS,
    GK_CAYMAN,
    GK_GFX6,
    GK_GFX7,
    GK_GFX8,
    GK_GFX9
  } GPU;

  bool hasFP64:1;
  bool hasFMAF:1;
  bool hasLDEXPF:1;
  const AddrSpace AS;

  static bool hasFullSpeedFMAF32(StringRef GPUName) {
    return parseAMDGCNName(GPUName) >= GK_GFX9;
  }

  static bool isAMDGCN(const llvm::Triple &TT) {
    return TT.getArch() == llvm::Triple::amdgcn;
  }

  static bool isGenericZero(const llvm::Triple &TT) {
    return TT.getEnvironmentName() == "amdgiz" ||
        TT.getEnvironmentName() == "amdgizcl";
  }
public:
  AMDGPUTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
    : TargetInfo(Triple) ,
      GPU(isAMDGCN(Triple) ? GK_GFX6 : parseR600Name(Opts.CPU)),
      hasFP64(false),
      hasFMAF(false),
      hasLDEXPF(false),
      AS(isGenericZero(Triple)){
    if (getTriple().getArch() == llvm::Triple::amdgcn) {
      hasFP64 = true;
      hasFMAF = true;
      hasLDEXPF = true;
    }
    if (getTriple().getArch() == llvm::Triple::r600) {
      if (GPU == GK_EVERGREEN_DOUBLE_OPS || GPU == GK_CAYMAN) {
        hasFMAF = true;
      }
    }

    auto IsGenericZero = isGenericZero(Triple);
    resetDataLayout(getTriple().getArch() == llvm::Triple::amdgcn ?
                    (IsGenericZero ? DataLayoutStringSIGenericIsZero :
                        DataLayoutStringSIPrivateIsZero)
                    : DataLayoutStringR600);
    assert(DataLayout->getAllocaAddrSpace() == AS.Private);

    setAddressSpaceMap(Triple.getOS() == llvm::Triple::Mesa3D ||
                       Triple.getEnvironment() == llvm::Triple::OpenCL ||
                       Triple.getEnvironmentName() == "amdgizcl" ||
                       !isAMDGCN(Triple));
    UseAddrSpaceMapMangling = true;

    // Set pointer width and alignment for target address space 0.
    PointerWidth = PointerAlign = DataLayout->getPointerSizeInBits();
    if (getMaxPointerWidth() == 64) {
      LongWidth = LongAlign = 64;
      SizeType = UnsignedLong;
      PtrDiffType = SignedLong;
      IntPtrType = SignedLong;
    }
  }

  void setAddressSpaceMap(bool DefaultIsPrivate) {
    if (isGenericZero(getTriple())) {
      AddrSpaceMap = DefaultIsPrivate ? &AMDGPUGenIsZeroDefIsPrivMap
                                      : &AMDGPUGenIsZeroDefIsGenMap;
    } else {
      AddrSpaceMap = DefaultIsPrivate ? &AMDGPUPrivIsZeroDefIsPrivMap
                                      : &AMDGPUPrivIsZeroDefIsGenMap;
    }
  }

  void adjust(LangOptions &Opts) override {
    TargetInfo::adjust(Opts);
    setAddressSpaceMap(Opts.OpenCL || !isAMDGCN(getTriple()));
  }

  uint64_t getPointerWidthV(unsigned AddrSpace) const override {
    if (GPU <= GK_CAYMAN)
      return 32;

    if (AddrSpace == AS.Private || AddrSpace == AS.Local) {
      return 32;
    }
    return 64;
  }

  uint64_t getPointerAlignV(unsigned AddrSpace) const override {
    return getPointerWidthV(AddrSpace);
  }

  uint64_t getMaxPointerWidth() const override {
    return getTriple().getArch() == llvm::Triple::amdgcn ? 64 : 32;
  }

  const char * getClobbers() const override {
    return "";
  }

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default: break;
    case 'v': // vgpr
    case 's': // sgpr
      Info.setAllowsRegister();
      return true;
    }
    return false;
  }

  bool initFeatureMap(llvm::StringMap<bool> &Features,
                      DiagnosticsEngine &Diags, StringRef CPU,
                      const std::vector<std::string> &FeatureVec) const override;

  void adjustTargetOptions(const CodeGenOptions &CGOpts,
                           TargetOptions &TargetOpts) const override {
    bool hasFP32Denormals = false;
    bool hasFP64Denormals = false;
    for (auto &I : TargetOpts.FeaturesAsWritten) {
      if (I == "+fp32-denormals" || I == "-fp32-denormals")
        hasFP32Denormals = true;
      if (I == "+fp64-fp16-denormals" || I == "-fp64-fp16-denormals")
        hasFP64Denormals = true;
    }
    if (!hasFP32Denormals)
      TargetOpts.Features.push_back(
          (Twine(hasFullSpeedFMAF32(TargetOpts.CPU) &&
          !CGOpts.FlushDenorm ? '+' : '-') + Twine("fp32-denormals")).str());
    // Always do not flush fp64 or fp16 denorms.
    if (!hasFP64Denormals && hasFP64)
      TargetOpts.Features.push_back("+fp64-fp16-denormals");
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                        clang::AMDGPU::LastTSBuiltin - Builtin::FirstTSBuiltin);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    if (getTriple().getArch() == llvm::Triple::amdgcn)
      Builder.defineMacro("__AMDGCN__");
    else
      Builder.defineMacro("__R600__");

    if (hasFMAF)
      Builder.defineMacro("__HAS_FMAF__");
    if (hasLDEXPF)
      Builder.defineMacro("__HAS_LDEXPF__");
    if (hasFP64)
      Builder.defineMacro("__HAS_FP64__");
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }

  static GPUKind parseR600Name(StringRef Name) {
    return llvm::StringSwitch<GPUKind>(Name)
      .Case("r600" ,    GK_R600)
      .Case("rv610",    GK_R600)
      .Case("rv620",    GK_R600)
      .Case("rv630",    GK_R600)
      .Case("rv635",    GK_R600)
      .Case("rs780",    GK_R600)
      .Case("rs880",    GK_R600)
      .Case("rv670",    GK_R600_DOUBLE_OPS)
      .Case("rv710",    GK_R700)
      .Case("rv730",    GK_R700)
      .Case("rv740",    GK_R700_DOUBLE_OPS)
      .Case("rv770",    GK_R700_DOUBLE_OPS)
      .Case("palm",     GK_EVERGREEN)
      .Case("cedar",    GK_EVERGREEN)
      .Case("sumo",     GK_EVERGREEN)
      .Case("sumo2",    GK_EVERGREEN)
      .Case("redwood",  GK_EVERGREEN)
      .Case("juniper",  GK_EVERGREEN)
      .Case("hemlock",  GK_EVERGREEN_DOUBLE_OPS)
      .Case("cypress",  GK_EVERGREEN_DOUBLE_OPS)
      .Case("barts",    GK_NORTHERN_ISLANDS)
      .Case("turks",    GK_NORTHERN_ISLANDS)
      .Case("caicos",   GK_NORTHERN_ISLANDS)
      .Case("cayman",   GK_CAYMAN)
      .Case("aruba",    GK_CAYMAN)
      .Default(GK_NONE);
  }

  static GPUKind parseAMDGCNName(StringRef Name) {
    return llvm::StringSwitch<GPUKind>(Name)
      .Case("tahiti",    GK_GFX6)
      .Case("pitcairn",  GK_GFX6)
      .Case("verde",     GK_GFX6)
      .Case("oland",     GK_GFX6)
      .Case("hainan",    GK_GFX6)
      .Case("bonaire",   GK_GFX7)
      .Case("kabini",    GK_GFX7)
      .Case("kaveri",    GK_GFX7)
      .Case("hawaii",    GK_GFX7)
      .Case("mullins",   GK_GFX7)
      .Case("gfx700",    GK_GFX7)
      .Case("gfx701",    GK_GFX7)
      .Case("gfx702",    GK_GFX7)
      .Case("tonga",     GK_GFX8)
      .Case("iceland",   GK_GFX8)
      .Case("carrizo",   GK_GFX8)
      .Case("fiji",      GK_GFX8)
      .Case("stoney",    GK_GFX8)
      .Case("polaris10", GK_GFX8)
      .Case("polaris11", GK_GFX8)
      .Case("gfx800",    GK_GFX8)
      .Case("gfx801",    GK_GFX8)
      .Case("gfx802",    GK_GFX8)
      .Case("gfx803",    GK_GFX8)
      .Case("gfx804",    GK_GFX8)
      .Case("gfx810",    GK_GFX8)
      .Case("gfx900",    GK_GFX9)
      .Case("gfx901",    GK_GFX9)
      .Default(GK_NONE);
  }

  bool setCPU(const std::string &Name) override {
    if (getTriple().getArch() == llvm::Triple::amdgcn)
      GPU = parseAMDGCNName(Name);
    else
      GPU = parseR600Name(Name);

    return GPU != GK_NONE;
  }

  void setSupportedOpenCLOpts() override {
    auto &Opts = getSupportedOpenCLOpts();
    Opts.support("cl_clang_storage_class_specifiers");
    Opts.support("cl_khr_icd");

    if (hasFP64)
      Opts.support("cl_khr_fp64");
    if (GPU >= GK_EVERGREEN) {
      Opts.support("cl_khr_byte_addressable_store");
      Opts.support("cl_khr_global_int32_base_atomics");
      Opts.support("cl_khr_global_int32_extended_atomics");
      Opts.support("cl_khr_local_int32_base_atomics");
      Opts.support("cl_khr_local_int32_extended_atomics");
    }
    if (GPU >= GK_GFX6) {
      Opts.support("cl_khr_fp16");
      Opts.support("cl_khr_int64_base_atomics");
      Opts.support("cl_khr_int64_extended_atomics");
      Opts.support("cl_khr_mipmap_image");
      Opts.support("cl_khr_subgroups");
      Opts.support("cl_khr_3d_image_writes");
      Opts.support("cl_amd_media_ops");
      Opts.support("cl_amd_media_ops2");
    }
  }

  LangAS::ID getOpenCLImageAddrSpace() const override {
    return LangAS::opencl_constant;
  }

  llvm::Optional<unsigned> getConstantAddressSpace() const override {
    return LangAS::FirstTargetAddressSpace + AS.Constant;
  }

  /// \returns Target specific vtbl ptr address space.
  unsigned getVtblPtrAddressSpace() const override { return AS.Constant; }

  /// \returns If a target requires an address within a target specific address
  /// space \p AddressSpace to be converted in order to be used, then return the
  /// corresponding target specific DWARF address space.
  ///
  /// \returns Otherwise return None and no conversion will be emitted in the
  /// DWARF.
  Optional<unsigned> getDWARFAddressSpace(
      unsigned AddressSpace) const override {
    const unsigned DWARF_Private = 1;
    const unsigned DWARF_Local   = 2;
    if (AddressSpace == AS.Private) {
      return DWARF_Private;
    } else if (AddressSpace == AS.Local) {
      return DWARF_Local;
    } else {
      return None;
    }
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
      default:
        return CCCR_Warning;
      case CC_C:
      case CC_OpenCLKernel:
        return CCCR_OK;
    }
  }

  // In amdgcn target the null pointer in global, constant, and generic
  // address space has value 0 but in private and local address space has
  // value ~0.
  uint64_t getNullPointerValue(unsigned AS) const override {
    return AS == LangAS::opencl_local ? ~0 : 0;
  }
};

const Builtin::Info AMDGPUTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE },
#include "clang/Basic/BuiltinsAMDGPU.def"
};
const char * const AMDGPUTargetInfo::GCCRegNames[] = {
  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
  "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
  "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39",
  "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47",
  "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55",
  "v56", "v57", "v58", "v59", "v60", "v61", "v62", "v63",
  "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71",
  "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79",
  "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87",
  "v88", "v89", "v90", "v91", "v92", "v93", "v94", "v95",
  "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103",
  "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111",
  "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119",
  "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127",
  "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135",
  "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143",
  "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151",
  "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159",
  "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167",
  "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175",
  "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183",
  "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191",
  "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199",
  "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207",
  "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215",
  "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223",
  "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231",
  "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239",
  "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247",
  "v248", "v249", "v250", "v251", "v252", "v253", "v254", "v255",
  "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
  "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
  "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
  "s32", "s33", "s34", "s35", "s36", "s37", "s38", "s39",
  "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47",
  "s48", "s49", "s50", "s51", "s52", "s53", "s54", "s55",
  "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63",
  "s64", "s65", "s66", "s67", "s68", "s69", "s70", "s71",
  "s72", "s73", "s74", "s75", "s76", "s77", "s78", "s79",
  "s80", "s81", "s82", "s83", "s84", "s85", "s86", "s87",
  "s88", "s89", "s90", "s91", "s92", "s93", "s94", "s95",
  "s96", "s97", "s98", "s99", "s100", "s101", "s102", "s103",
  "s104", "s105", "s106", "s107", "s108", "s109", "s110", "s111",
  "s112", "s113", "s114", "s115", "s116", "s117", "s118", "s119",
  "s120", "s121", "s122", "s123", "s124", "s125", "s126", "s127",
  "exec", "vcc", "scc", "m0", "flat_scratch", "exec_lo", "exec_hi",
  "vcc_lo", "vcc_hi", "flat_scratch_lo", "flat_scratch_hi"
};

ArrayRef<const char *> AMDGPUTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

bool AMDGPUTargetInfo::initFeatureMap(
  llvm::StringMap<bool> &Features,
  DiagnosticsEngine &Diags, StringRef CPU,
  const std::vector<std::string> &FeatureVec) const {

  // XXX - What does the member GPU mean if device name string passed here?
  if (getTriple().getArch() == llvm::Triple::amdgcn) {
    if (CPU.empty())
      CPU = "tahiti";

    switch (parseAMDGCNName(CPU)) {
    case GK_GFX6:
    case GK_GFX7:
      break;

    case GK_GFX9:
      Features["gfx9-insts"] = true;
      LLVM_FALLTHROUGH;
    case GK_GFX8:
      Features["s-memrealtime"] = true;
      Features["16-bit-insts"] = true;
      Features["dpp"] = true;
      break;

    case GK_NONE:
      return false;
    default:
      llvm_unreachable("unhandled subtarget");
    }
  } else {
    if (CPU.empty())
      CPU = "r600";

    switch (parseR600Name(CPU)) {
    case GK_R600:
    case GK_R700:
    case GK_EVERGREEN:
    case GK_NORTHERN_ISLANDS:
      break;
    case GK_R600_DOUBLE_OPS:
    case GK_R700_DOUBLE_OPS:
    case GK_EVERGREEN_DOUBLE_OPS:
    case GK_CAYMAN:
      Features["fp64"] = true;
      break;
    case GK_NONE:
      return false;
    default:
      llvm_unreachable("unhandled subtarget");
    }
  }

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeatureVec);
}

const Builtin::Info BuiltinInfoX86[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE },
#define TARGET_HEADER_BUILTIN(ID, TYPE, ATTRS, HEADER, LANGS, FEATURE)         \
  { #ID, TYPE, ATTRS, HEADER, LANGS, FEATURE },
#include "clang/Basic/BuiltinsX86.def"

#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)         \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE },
#define TARGET_HEADER_BUILTIN(ID, TYPE, ATTRS, HEADER, LANGS, FEATURE)         \
  { #ID, TYPE, ATTRS, HEADER, LANGS, FEATURE },
#include "clang/Basic/BuiltinsX86_64.def"
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
  "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22", "xmm23",
  "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31",
  "ymm16", "ymm17", "ymm18", "ymm19", "ymm20", "ymm21", "ymm22", "ymm23",
  "ymm24", "ymm25", "ymm26", "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
  "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
  "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
  "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
  "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
  "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
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
  { { "r8d", "r8w", "r8b" }, 38 },
  { { "r9d", "r9w", "r9b" }, 39 },
  { { "r10d", "r10w", "r10b" }, 40 },
  { { "r11d", "r11w", "r11b" }, 41 },
  { { "r12d", "r12w", "r12b" }, 42 },
  { { "r13d", "r13w", "r13b" }, 43 },
  { { "r14d", "r14w", "r14b" }, 44 },
  { { "r15d", "r15w", "r15b" }, 45 },
};

// X86 target abstract base class; x86-32 and x86-64 are very close, so
// most of the implementation can be shared.
class X86TargetInfo : public TargetInfo {
  enum X86SSEEnum {
    NoSSE, SSE1, SSE2, SSE3, SSSE3, SSE41, SSE42, AVX, AVX2, AVX512F
  } SSELevel = NoSSE;
  enum MMX3DNowEnum {
    NoMMX3DNow, MMX, AMD3DNow, AMD3DNowAthlon
  } MMX3DNowLevel = NoMMX3DNow;
  enum XOPEnum {
    NoXOP,
    SSE4A,
    FMA4,
    XOP
  } XOPLevel = NoXOP;

  bool HasAES = false;
  bool HasPCLMUL = false;
  bool HasLZCNT = false;
  bool HasRDRND = false;
  bool HasFSGSBASE = false;
  bool HasBMI = false;
  bool HasBMI2 = false;
  bool HasPOPCNT = false;
  bool HasRTM = false;
  bool HasPRFCHW = false;
  bool HasRDSEED = false;
  bool HasADX = false;
  bool HasTBM = false;
  bool HasLWP = false;
  bool HasFMA = false;
  bool HasF16C = false;
  bool HasAVX512CD = false;
  bool HasAVX512VPOPCNTDQ = false;
  bool HasAVX512ER = false;
  bool HasAVX512PF = false;
  bool HasAVX512DQ = false;
  bool HasAVX512BW = false;
  bool HasAVX512VL = false;
  bool HasAVX512VBMI = false;
  bool HasAVX512IFMA = false;
  bool HasSHA = false;
  bool HasMPX = false;
  bool HasSGX = false;
  bool HasCX16 = false;
  bool HasFXSR = false;
  bool HasXSAVE = false;
  bool HasXSAVEOPT = false;
  bool HasXSAVEC = false;
  bool HasXSAVES = false;
  bool HasMWAITX = false;
  bool HasCLZERO = false;
  bool HasPKU = false;
  bool HasCLFLUSHOPT = false;
  bool HasCLWB = false;
  bool HasMOVBE = false;
  bool HasPREFETCHWT1 = false;

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
    /// Clang however has some logic to support this.
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
    CK_Bonnell,
    CK_Silvermont,
    CK_Goldmont,
    //@}

    /// \name Nehalem
    /// Nehalem microarchitecture based processors.
    CK_Nehalem,

    /// \name Westmere
    /// Westmere microarchitecture based processors.
    CK_Westmere,

    /// \name Sandy Bridge
    /// Sandy Bridge microarchitecture based processors.
    CK_SandyBridge,

    /// \name Ivy Bridge
    /// Ivy Bridge microarchitecture based processors.
    CK_IvyBridge,

    /// \name Haswell
    /// Haswell microarchitecture based processors.
    CK_Haswell,

    /// \name Broadwell
    /// Broadwell microarchitecture based processors.
    CK_Broadwell,

    /// \name Skylake Client
    /// Skylake client microarchitecture based processors.
    CK_SkylakeClient,

    /// \name Skylake Server
    /// Skylake server microarchitecture based processors.
    CK_SkylakeServer,

    /// \name Cannonlake Client
    /// Cannonlake client microarchitecture based processors.
    CK_Cannonlake,

    /// \name Knights Landing
    /// Knights Landing processor.
    CK_KNL,

    /// \name Lakemont
    /// Lakemont microarchitecture based processors.
    CK_Lakemont,

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
    CK_BTVER2,
    //@}

    /// \name Bulldozer
    /// Bulldozer architecture processors.
    //@{
    CK_BDVER1,
    CK_BDVER2,
    CK_BDVER3,
    CK_BDVER4,
    //@}

    /// \name zen
    /// Zen architecture processors.
    //@{
    CK_ZNVER1,
    //@}

    /// This specification is deprecated and will be removed in the future.
    /// Users should prefer \see CK_K8.
    // FIXME: Warn on this when the CPU is set to it.
    //@{
    CK_x86_64,
    //@}

    /// \name Geode
    /// Geode processors.
    //@{
    CK_Geode
    //@}
  } CPU = CK_Generic;

  CPUKind getCPUKind(StringRef CPU) const {
    return llvm::StringSwitch<CPUKind>(CPU)
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
        .Case("bonnell", CK_Bonnell)
        .Case("atom", CK_Bonnell) // Legacy name.
        .Case("silvermont", CK_Silvermont)
        .Case("slm", CK_Silvermont) // Legacy name.
        .Case("goldmont", CK_Goldmont)
        .Case("nehalem", CK_Nehalem)
        .Case("corei7", CK_Nehalem) // Legacy name.
        .Case("westmere", CK_Westmere)
        .Case("sandybridge", CK_SandyBridge)
        .Case("corei7-avx", CK_SandyBridge) // Legacy name.
        .Case("ivybridge", CK_IvyBridge)
        .Case("core-avx-i", CK_IvyBridge) // Legacy name.
        .Case("haswell", CK_Haswell)
        .Case("core-avx2", CK_Haswell) // Legacy name.
        .Case("broadwell", CK_Broadwell)
        .Case("skylake", CK_SkylakeClient)
        .Case("skylake-avx512", CK_SkylakeServer)
        .Case("skx", CK_SkylakeServer) // Legacy name.
        .Case("cannonlake", CK_Cannonlake)
        .Case("knl", CK_KNL)
        .Case("lakemont", CK_Lakemont)
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
        .Case("barcelona", CK_AMDFAM10)
        .Case("amdfam10", CK_AMDFAM10)
        .Case("btver1", CK_BTVER1)
        .Case("btver2", CK_BTVER2)
        .Case("bdver1", CK_BDVER1)
        .Case("bdver2", CK_BDVER2)
        .Case("bdver3", CK_BDVER3)
        .Case("bdver4", CK_BDVER4)
        .Case("znver1", CK_ZNVER1)
        .Case("x86-64", CK_x86_64)
        .Case("geode", CK_Geode)
        .Default(CK_Generic);
  }

  enum FPMathKind {
    FP_Default,
    FP_SSE,
    FP_387
  } FPMath = FP_Default;

public:
  X86TargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    LongDoubleFormat = &llvm::APFloat::x87DoubleExtended();
  }
  unsigned getFloatEvalMethod() const override {
    // X87 evaluates with 80 bits "long double" precision.
    return SSELevel == NoSSE ? 2 : 0;
  }
  ArrayRef<const char *> getGCCRegNames() const override {
    return llvm::makeArrayRef(GCCRegNames);
  }
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }
  ArrayRef<TargetInfo::AddlRegName> getGCCAddlRegNames() const override {
    return llvm::makeArrayRef(AddlRegNames);
  }
  bool validateCpuSupports(StringRef Name) const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override;

  bool validateGlobalRegisterVariable(StringRef RegName,
                                      unsigned RegSize,
                                      bool &HasSizeMismatch) const override {
    // esp and ebp are the only 32-bit registers the x86 backend can currently
    // handle.
    if (RegName.equals("esp") || RegName.equals("ebp")) {
      // Check that the register size is 32-bit.
      HasSizeMismatch = RegSize != 32;
      return true;
    }

    return false;
  }

  bool validateOutputSize(StringRef Constraint, unsigned Size) const override;

  bool validateInputSize(StringRef Constraint, unsigned Size) const override;

  virtual bool validateOperandSize(StringRef Constraint, unsigned Size) const;

  std::string convertConstraint(const char *&Constraint) const override;
  const char *getClobbers() const override {
    return "~{dirflag},~{fpsr},~{flags}";
  }

  StringRef getConstraintRegister(const StringRef &Constraint,
                                  const StringRef &Expression) const override {
    StringRef::iterator I, E;
    for (I = Constraint.begin(), E = Constraint.end(); I != E; ++I) {
      if (isalpha(*I))
        break;
    }
    if (I == E)
      return "";
    switch (*I) {
    // For the register constraints, return the matching register name
    case 'a':
      return "ax";
    case 'b':
      return "bx";
    case 'c':
      return "cx";
    case 'd':
      return "dx";
    case 'S':
      return "si";
    case 'D':
      return "di";
    // In case the constraint is 'r' we need to return Expression
    case 'r':
      return Expression;
    default:
      // Default value if there is no constraint for the register
      return "";
    }
    return "";
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
  static void setSSELevel(llvm::StringMap<bool> &Features, X86SSEEnum Level,
                          bool Enabled);
  static void setMMXLevel(llvm::StringMap<bool> &Features, MMX3DNowEnum Level,
                          bool Enabled);
  static void setXOPLevel(llvm::StringMap<bool> &Features, XOPEnum Level,
                          bool Enabled);
  void setFeatureEnabled(llvm::StringMap<bool> &Features,
                         StringRef Name, bool Enabled) const override {
    setFeatureEnabledImpl(Features, Name, Enabled);
  }
  // This exists purely to cut down on the number of virtual calls in
  // initFeatureMap which calls this repeatedly.
  static void setFeatureEnabledImpl(llvm::StringMap<bool> &Features,
                                    StringRef Name, bool Enabled);
  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override;
  bool hasFeature(StringRef Feature) const override;
  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;
  StringRef getABI() const override {
    if (getTriple().getArch() == llvm::Triple::x86_64 && SSELevel >= AVX512F)
      return "avx512";
    if (getTriple().getArch() == llvm::Triple::x86_64 && SSELevel >= AVX)
      return "avx";
    if (getTriple().getArch() == llvm::Triple::x86 &&
             MMX3DNowLevel == NoMMX3DNow)
      return "no-mmx";
    return "";
  }
  bool setCPU(const std::string &Name) override {
    CPU = getCPUKind(Name);

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
    case CK_Lakemont:
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
      if (getTriple().getArch() != llvm::Triple::x86)
        return false;

      // Fallthrough
    case CK_Nocona:
    case CK_Core2:
    case CK_Penryn:
    case CK_Bonnell:
    case CK_Silvermont:
    case CK_Goldmont:
    case CK_Nehalem:
    case CK_Westmere:
    case CK_SandyBridge:
    case CK_IvyBridge:
    case CK_Haswell:
    case CK_Broadwell:
    case CK_SkylakeClient:
    case CK_SkylakeServer:
    case CK_Cannonlake:
    case CK_KNL:
    case CK_Athlon64:
    case CK_Athlon64SSE3:
    case CK_AthlonFX:
    case CK_K8:
    case CK_K8SSE3:
    case CK_Opteron:
    case CK_OpteronSSE3:
    case CK_AMDFAM10:
    case CK_BTVER1:
    case CK_BTVER2:
    case CK_BDVER1:
    case CK_BDVER2:
    case CK_BDVER3:
    case CK_BDVER4:
    case CK_ZNVER1:
    case CK_x86_64:
      return true;
    }
    llvm_unreachable("Unhandled CPU kind");
  }

  bool setFPMath(StringRef Name) override;

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    // Most of the non-ARM calling conventions are i386 conventions.
    switch (CC) {
    case CC_X86ThisCall:
    case CC_X86FastCall:
    case CC_X86StdCall:
    case CC_X86VectorCall:
    case CC_X86RegCall:
    case CC_C:
    case CC_Swift:
    case CC_X86Pascal:
    case CC_IntelOclBicc:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }

  CallingConv getDefaultCallingConv(CallingConvMethodType MT) const override {
    return MT == CCMT_Member ? CC_X86ThisCall : CC_C;
  }

  bool hasSjLjLowering() const override {
    return true;
  }

  void setSupportedOpenCLOpts() override {
    getSupportedOpenCLOpts().supportAll();
  }
};

bool X86TargetInfo::setFPMath(StringRef Name) {
  if (Name == "387") {
    FPMath = FP_387;
    return true;
  }
  if (Name == "sse") {
    FPMath = FP_SSE;
    return true;
  }
  return false;
}

bool X86TargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  // FIXME: This *really* should not be here.
  // X86_64 always has SSE2.
  if (getTriple().getArch() == llvm::Triple::x86_64)
    setFeatureEnabledImpl(Features, "sse2", true);

  const CPUKind Kind = getCPUKind(CPU);

  // Enable X87 for all X86 processors but Lakemont.
  if (Kind != CK_Lakemont)
    setFeatureEnabledImpl(Features, "x87", true);

  switch (Kind) {
  case CK_Generic:
  case CK_i386:
  case CK_i486:
  case CK_i586:
  case CK_Pentium:
  case CK_i686:
  case CK_PentiumPro:
  case CK_Lakemont:
    break;
  case CK_PentiumMMX:
  case CK_Pentium2:
  case CK_K6:
  case CK_WinChipC6:
    setFeatureEnabledImpl(Features, "mmx", true);
    break;
  case CK_Pentium3:
  case CK_Pentium3M:
  case CK_C3_2:
    setFeatureEnabledImpl(Features, "sse", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    break;
  case CK_PentiumM:
  case CK_Pentium4:
  case CK_Pentium4M:
  case CK_x86_64:
    setFeatureEnabledImpl(Features, "sse2", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    break;
  case CK_Yonah:
  case CK_Prescott:
  case CK_Nocona:
    setFeatureEnabledImpl(Features, "sse3", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    break;
  case CK_Core2:
    setFeatureEnabledImpl(Features, "ssse3", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    break;
  case CK_Penryn:
    setFeatureEnabledImpl(Features, "sse4.1", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    break;
  case CK_Cannonlake:
    setFeatureEnabledImpl(Features, "avx512ifma", true);
    setFeatureEnabledImpl(Features, "avx512vbmi", true);
    setFeatureEnabledImpl(Features, "sha", true);
    LLVM_FALLTHROUGH;
  case CK_SkylakeServer:
    setFeatureEnabledImpl(Features, "avx512f", true);
    setFeatureEnabledImpl(Features, "avx512cd", true);
    setFeatureEnabledImpl(Features, "avx512dq", true);
    setFeatureEnabledImpl(Features, "avx512bw", true);
    setFeatureEnabledImpl(Features, "avx512vl", true);
    setFeatureEnabledImpl(Features, "pku", true);
    setFeatureEnabledImpl(Features, "clwb", true);
    LLVM_FALLTHROUGH;
  case CK_SkylakeClient:
    setFeatureEnabledImpl(Features, "xsavec", true);
    setFeatureEnabledImpl(Features, "xsaves", true);
    setFeatureEnabledImpl(Features, "mpx", true);
    setFeatureEnabledImpl(Features, "sgx", true);
    setFeatureEnabledImpl(Features, "clflushopt", true);
    setFeatureEnabledImpl(Features, "rtm", true);
    LLVM_FALLTHROUGH;
  case CK_Broadwell:
    setFeatureEnabledImpl(Features, "rdseed", true);
    setFeatureEnabledImpl(Features, "adx", true);
    LLVM_FALLTHROUGH;
  case CK_Haswell:
    setFeatureEnabledImpl(Features, "avx2", true);
    setFeatureEnabledImpl(Features, "lzcnt", true);
    setFeatureEnabledImpl(Features, "bmi", true);
    setFeatureEnabledImpl(Features, "bmi2", true);
    setFeatureEnabledImpl(Features, "fma", true);
    setFeatureEnabledImpl(Features, "movbe", true);
    LLVM_FALLTHROUGH;
  case CK_IvyBridge:
    setFeatureEnabledImpl(Features, "rdrnd", true);
    setFeatureEnabledImpl(Features, "f16c", true);
    setFeatureEnabledImpl(Features, "fsgsbase", true);
    LLVM_FALLTHROUGH;
  case CK_SandyBridge:
    setFeatureEnabledImpl(Features, "avx", true);
    setFeatureEnabledImpl(Features, "xsave", true);
    setFeatureEnabledImpl(Features, "xsaveopt", true);
    LLVM_FALLTHROUGH;
  case CK_Westmere:
    setFeatureEnabledImpl(Features, "aes", true);
    setFeatureEnabledImpl(Features, "pclmul", true);
    LLVM_FALLTHROUGH;
  case CK_Nehalem:
    setFeatureEnabledImpl(Features, "sse4.2", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    break;
  case CK_Goldmont:
    setFeatureEnabledImpl(Features, "sha", true);
    setFeatureEnabledImpl(Features, "rdrnd", true);
    setFeatureEnabledImpl(Features, "rdseed", true);
    setFeatureEnabledImpl(Features, "xsave", true);
    setFeatureEnabledImpl(Features, "xsaveopt", true);
    setFeatureEnabledImpl(Features, "xsavec", true);
    setFeatureEnabledImpl(Features, "xsaves", true);
    setFeatureEnabledImpl(Features, "clflushopt", true);
    setFeatureEnabledImpl(Features, "mpx", true);
    LLVM_FALLTHROUGH;
  case CK_Silvermont:
    setFeatureEnabledImpl(Features, "aes", true);
    setFeatureEnabledImpl(Features, "pclmul", true);
    setFeatureEnabledImpl(Features, "sse4.2", true);
    LLVM_FALLTHROUGH;
  case CK_Bonnell:
    setFeatureEnabledImpl(Features, "movbe", true);
    setFeatureEnabledImpl(Features, "ssse3", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    break;
  case CK_KNL:
    setFeatureEnabledImpl(Features, "avx512f", true);
    setFeatureEnabledImpl(Features, "avx512cd", true);
    setFeatureEnabledImpl(Features, "avx512er", true);
    setFeatureEnabledImpl(Features, "avx512pf", true);
    setFeatureEnabledImpl(Features, "prefetchwt1", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "rdseed", true);
    setFeatureEnabledImpl(Features, "adx", true);
    setFeatureEnabledImpl(Features, "lzcnt", true);
    setFeatureEnabledImpl(Features, "bmi", true);
    setFeatureEnabledImpl(Features, "bmi2", true);
    setFeatureEnabledImpl(Features, "rtm", true);
    setFeatureEnabledImpl(Features, "fma", true);
    setFeatureEnabledImpl(Features, "rdrnd", true);
    setFeatureEnabledImpl(Features, "f16c", true);
    setFeatureEnabledImpl(Features, "fsgsbase", true);
    setFeatureEnabledImpl(Features, "aes", true);
    setFeatureEnabledImpl(Features, "pclmul", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    setFeatureEnabledImpl(Features, "xsaveopt", true);
    setFeatureEnabledImpl(Features, "xsave", true);
    setFeatureEnabledImpl(Features, "movbe", true);
    break;
  case CK_K6_2:
  case CK_K6_3:
  case CK_WinChip2:
  case CK_C3:
    setFeatureEnabledImpl(Features, "3dnow", true);
    break;
  case CK_Athlon:
  case CK_AthlonThunderbird:
  case CK_Geode:
    setFeatureEnabledImpl(Features, "3dnowa", true);
    break;
  case CK_Athlon4:
  case CK_AthlonXP:
  case CK_AthlonMP:
    setFeatureEnabledImpl(Features, "sse", true);
    setFeatureEnabledImpl(Features, "3dnowa", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    break;
  case CK_K8:
  case CK_Opteron:
  case CK_Athlon64:
  case CK_AthlonFX:
    setFeatureEnabledImpl(Features, "sse2", true);
    setFeatureEnabledImpl(Features, "3dnowa", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    break;
  case CK_AMDFAM10:
    setFeatureEnabledImpl(Features, "sse4a", true);
    setFeatureEnabledImpl(Features, "lzcnt", true);
    setFeatureEnabledImpl(Features, "popcnt", true);
    LLVM_FALLTHROUGH;
  case CK_K8SSE3:
  case CK_OpteronSSE3:
  case CK_Athlon64SSE3:
    setFeatureEnabledImpl(Features, "sse3", true);
    setFeatureEnabledImpl(Features, "3dnowa", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    break;
  case CK_BTVER2:
    setFeatureEnabledImpl(Features, "avx", true);
    setFeatureEnabledImpl(Features, "aes", true);
    setFeatureEnabledImpl(Features, "pclmul", true);
    setFeatureEnabledImpl(Features, "bmi", true);
    setFeatureEnabledImpl(Features, "f16c", true);
    setFeatureEnabledImpl(Features, "xsaveopt", true);
    setFeatureEnabledImpl(Features, "movbe", true);
    LLVM_FALLTHROUGH;
  case CK_BTVER1:
    setFeatureEnabledImpl(Features, "ssse3", true);
    setFeatureEnabledImpl(Features, "sse4a", true);
    setFeatureEnabledImpl(Features, "lzcnt", true);
    setFeatureEnabledImpl(Features, "popcnt", true);
    setFeatureEnabledImpl(Features, "prfchw", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    break;
  case CK_ZNVER1:
    setFeatureEnabledImpl(Features, "adx", true);
    setFeatureEnabledImpl(Features, "aes", true);
    setFeatureEnabledImpl(Features, "avx2", true);
    setFeatureEnabledImpl(Features, "bmi", true);
    setFeatureEnabledImpl(Features, "bmi2", true);
    setFeatureEnabledImpl(Features, "clflushopt", true);
    setFeatureEnabledImpl(Features, "clzero", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    setFeatureEnabledImpl(Features, "f16c", true);
    setFeatureEnabledImpl(Features, "fma", true);
    setFeatureEnabledImpl(Features, "fsgsbase", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "lzcnt", true);
    setFeatureEnabledImpl(Features, "mwaitx", true);
    setFeatureEnabledImpl(Features, "movbe", true);
    setFeatureEnabledImpl(Features, "pclmul", true);
    setFeatureEnabledImpl(Features, "popcnt", true);
    setFeatureEnabledImpl(Features, "prfchw", true);
    setFeatureEnabledImpl(Features, "rdrnd", true);
    setFeatureEnabledImpl(Features, "rdseed", true);
    setFeatureEnabledImpl(Features, "sha", true);
    setFeatureEnabledImpl(Features, "sse4a", true);
    setFeatureEnabledImpl(Features, "xsave", true);
    setFeatureEnabledImpl(Features, "xsavec", true);
    setFeatureEnabledImpl(Features, "xsaveopt", true);
    setFeatureEnabledImpl(Features, "xsaves", true);
    break;
  case CK_BDVER4:
    setFeatureEnabledImpl(Features, "avx2", true);
    setFeatureEnabledImpl(Features, "bmi2", true);
    setFeatureEnabledImpl(Features, "mwaitx", true);
    LLVM_FALLTHROUGH;
  case CK_BDVER3:
    setFeatureEnabledImpl(Features, "fsgsbase", true);
    setFeatureEnabledImpl(Features, "xsaveopt", true);
    LLVM_FALLTHROUGH;
  case CK_BDVER2:
    setFeatureEnabledImpl(Features, "bmi", true);
    setFeatureEnabledImpl(Features, "fma", true);
    setFeatureEnabledImpl(Features, "f16c", true);
    setFeatureEnabledImpl(Features, "tbm", true);
    LLVM_FALLTHROUGH;
  case CK_BDVER1:
    // xop implies avx, sse4a and fma4.
    setFeatureEnabledImpl(Features, "xop", true);
    setFeatureEnabledImpl(Features, "lwp", true);
    setFeatureEnabledImpl(Features, "lzcnt", true);
    setFeatureEnabledImpl(Features, "aes", true);
    setFeatureEnabledImpl(Features, "pclmul", true);
    setFeatureEnabledImpl(Features, "prfchw", true);
    setFeatureEnabledImpl(Features, "cx16", true);
    setFeatureEnabledImpl(Features, "fxsr", true);
    setFeatureEnabledImpl(Features, "xsave", true);
    break;
  }
  if (!TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec))
    return false;

  // Can't do this earlier because we need to be able to explicitly enable
  // or disable these features and the things that they depend upon.

  // Enable popcnt if sse4.2 is enabled and popcnt is not explicitly disabled.
  auto I = Features.find("sse4.2");
  if (I != Features.end() && I->getValue() &&
      std::find(FeaturesVec.begin(), FeaturesVec.end(), "-popcnt") ==
          FeaturesVec.end())
    Features["popcnt"] = true;

  // Enable prfchw if 3DNow! is enabled and prfchw is not explicitly disabled.
  I = Features.find("3dnow");
  if (I != Features.end() && I->getValue() &&
      std::find(FeaturesVec.begin(), FeaturesVec.end(), "-prfchw") ==
          FeaturesVec.end())
    Features["prfchw"] = true;

  // Additionally, if SSE is enabled and mmx is not explicitly disabled,
  // then enable MMX.
  I = Features.find("sse");
  if (I != Features.end() && I->getValue() &&
      std::find(FeaturesVec.begin(), FeaturesVec.end(), "-mmx") ==
          FeaturesVec.end())
    Features["mmx"] = true;

  return true;
}

void X86TargetInfo::setSSELevel(llvm::StringMap<bool> &Features,
                                X86SSEEnum Level, bool Enabled) {
  if (Enabled) {
    switch (Level) {
    case AVX512F:
      Features["avx512f"] = true;
      LLVM_FALLTHROUGH;
    case AVX2:
      Features["avx2"] = true;
      LLVM_FALLTHROUGH;
    case AVX:
      Features["avx"] = true;
      Features["xsave"] = true;
      LLVM_FALLTHROUGH;
    case SSE42:
      Features["sse4.2"] = true;
      LLVM_FALLTHROUGH;
    case SSE41:
      Features["sse4.1"] = true;
      LLVM_FALLTHROUGH;
    case SSSE3:
      Features["ssse3"] = true;
      LLVM_FALLTHROUGH;
    case SSE3:
      Features["sse3"] = true;
      LLVM_FALLTHROUGH;
    case SSE2:
      Features["sse2"] = true;
      LLVM_FALLTHROUGH;
    case SSE1:
      Features["sse"] = true;
      LLVM_FALLTHROUGH;
    case NoSSE:
      break;
    }
    return;
  }

  switch (Level) {
  case NoSSE:
  case SSE1:
    Features["sse"] = false;
    LLVM_FALLTHROUGH;
  case SSE2:
    Features["sse2"] = Features["pclmul"] = Features["aes"] =
      Features["sha"] = false;
    LLVM_FALLTHROUGH;
  case SSE3:
    Features["sse3"] = false;
    setXOPLevel(Features, NoXOP, false);
    LLVM_FALLTHROUGH;
  case SSSE3:
    Features["ssse3"] = false;
    LLVM_FALLTHROUGH;
  case SSE41:
    Features["sse4.1"] = false;
    LLVM_FALLTHROUGH;
  case SSE42:
    Features["sse4.2"] = false;
    LLVM_FALLTHROUGH;
  case AVX:
    Features["fma"] = Features["avx"] = Features["f16c"] = Features["xsave"] =
      Features["xsaveopt"] = false;
    setXOPLevel(Features, FMA4, false);
    LLVM_FALLTHROUGH;
  case AVX2:
    Features["avx2"] = false;
    LLVM_FALLTHROUGH;
  case AVX512F:
    Features["avx512f"] = Features["avx512cd"] = Features["avx512er"] =
        Features["avx512pf"] = Features["avx512dq"] = Features["avx512bw"] =
            Features["avx512vl"] = Features["avx512vbmi"] =
                Features["avx512ifma"] = Features["avx512vpopcntdq"] = false;
    break;
  }
}

void X86TargetInfo::setMMXLevel(llvm::StringMap<bool> &Features,
                                MMX3DNowEnum Level, bool Enabled) {
  if (Enabled) {
    switch (Level) {
    case AMD3DNowAthlon:
      Features["3dnowa"] = true;
      LLVM_FALLTHROUGH;
    case AMD3DNow:
      Features["3dnow"] = true;
      LLVM_FALLTHROUGH;
    case MMX:
      Features["mmx"] = true;
      LLVM_FALLTHROUGH;
    case NoMMX3DNow:
      break;
    }
    return;
  }

  switch (Level) {
  case NoMMX3DNow:
  case MMX:
    Features["mmx"] = false;
    LLVM_FALLTHROUGH;
  case AMD3DNow:
    Features["3dnow"] = false;
    LLVM_FALLTHROUGH;
  case AMD3DNowAthlon:
    Features["3dnowa"] = false;
    break;
  }
}

void X86TargetInfo::setXOPLevel(llvm::StringMap<bool> &Features, XOPEnum Level,
                                bool Enabled) {
  if (Enabled) {
    switch (Level) {
    case XOP:
      Features["xop"] = true;
      LLVM_FALLTHROUGH;
    case FMA4:
      Features["fma4"] = true;
      setSSELevel(Features, AVX, true);
      LLVM_FALLTHROUGH;
    case SSE4A:
      Features["sse4a"] = true;
      setSSELevel(Features, SSE3, true);
      LLVM_FALLTHROUGH;
    case NoXOP:
      break;
    }
    return;
  }

  switch (Level) {
  case NoXOP:
  case SSE4A:
    Features["sse4a"] = false;
    LLVM_FALLTHROUGH;
  case FMA4:
    Features["fma4"] = false;
    LLVM_FALLTHROUGH;
  case XOP:
    Features["xop"] = false;
    break;
  }
}

void X86TargetInfo::setFeatureEnabledImpl(llvm::StringMap<bool> &Features,
                                          StringRef Name, bool Enabled) {
  // This is a bit of a hack to deal with the sse4 target feature when used
  // as part of the target attribute. We handle sse4 correctly everywhere
  // else. See below for more information on how we handle the sse4 options.
  if (Name != "sse4")
    Features[Name] = Enabled;

  if (Name == "mmx") {
    setMMXLevel(Features, MMX, Enabled);
  } else if (Name == "sse") {
    setSSELevel(Features, SSE1, Enabled);
  } else if (Name == "sse2") {
    setSSELevel(Features, SSE2, Enabled);
  } else if (Name == "sse3") {
    setSSELevel(Features, SSE3, Enabled);
  } else if (Name == "ssse3") {
    setSSELevel(Features, SSSE3, Enabled);
  } else if (Name == "sse4.2") {
    setSSELevel(Features, SSE42, Enabled);
  } else if (Name == "sse4.1") {
    setSSELevel(Features, SSE41, Enabled);
  } else if (Name == "3dnow") {
    setMMXLevel(Features, AMD3DNow, Enabled);
  } else if (Name == "3dnowa") {
    setMMXLevel(Features, AMD3DNowAthlon, Enabled);
  } else if (Name == "aes") {
    if (Enabled)
      setSSELevel(Features, SSE2, Enabled);
  } else if (Name == "pclmul") {
    if (Enabled)
      setSSELevel(Features, SSE2, Enabled);
  } else if (Name == "avx") {
    setSSELevel(Features, AVX, Enabled);
  } else if (Name == "avx2") {
    setSSELevel(Features, AVX2, Enabled);
  } else if (Name == "avx512f") {
    setSSELevel(Features, AVX512F, Enabled);
  } else if (Name == "avx512cd" || Name == "avx512er" || Name == "avx512pf" ||
             Name == "avx512dq" || Name == "avx512bw" || Name == "avx512vl" ||
             Name == "avx512vbmi" || Name == "avx512ifma" ||
             Name == "avx512vpopcntdq") {
    if (Enabled)
      setSSELevel(Features, AVX512F, Enabled);
    // Enable BWI instruction if VBMI is being enabled.
    if (Name == "avx512vbmi" && Enabled)
      Features["avx512bw"] = true;
    // Also disable VBMI if BWI is being disabled.
    if (Name == "avx512bw" && !Enabled)
      Features["avx512vbmi"] = false;
  } else if (Name == "fma") {
    if (Enabled)
      setSSELevel(Features, AVX, Enabled);
  } else if (Name == "fma4") {
    setXOPLevel(Features, FMA4, Enabled);
  } else if (Name == "xop") {
    setXOPLevel(Features, XOP, Enabled);
  } else if (Name == "sse4a") {
    setXOPLevel(Features, SSE4A, Enabled);
  } else if (Name == "f16c") {
    if (Enabled)
      setSSELevel(Features, AVX, Enabled);
  } else if (Name == "sha") {
    if (Enabled)
      setSSELevel(Features, SSE2, Enabled);
  } else if (Name == "sse4") {
    // We can get here via the __target__ attribute since that's not controlled
    // via the -msse4/-mno-sse4 command line alias. Handle this the same way
    // here - turn on the sse4.2 if enabled, turn off the sse4.1 level if
    // disabled.
    if (Enabled)
      setSSELevel(Features, SSE42, Enabled);
    else
      setSSELevel(Features, SSE41, Enabled);
  } else if (Name == "xsave") {
    if (!Enabled)
      Features["xsaveopt"] = false;
  } else if (Name == "xsaveopt" || Name == "xsavec" || Name == "xsaves") {
    if (Enabled)
      Features["xsave"] = true;
  }
}

/// handleTargetFeatures - Perform initialization based on the user
/// configured set of features.
bool X86TargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                         DiagnosticsEngine &Diags) {
  for (const auto &Feature : Features) {
    if (Feature[0] != '+')
      continue;

    if (Feature == "+aes") {
      HasAES = true;
    } else if (Feature == "+pclmul") {
      HasPCLMUL = true;
    } else if (Feature == "+lzcnt") {
      HasLZCNT = true;
    } else if (Feature == "+rdrnd") {
      HasRDRND = true;
    } else if (Feature == "+fsgsbase") {
      HasFSGSBASE = true;
    } else if (Feature == "+bmi") {
      HasBMI = true;
    } else if (Feature == "+bmi2") {
      HasBMI2 = true;
    } else if (Feature == "+popcnt") {
      HasPOPCNT = true;
    } else if (Feature == "+rtm") {
      HasRTM = true;
    } else if (Feature == "+prfchw") {
      HasPRFCHW = true;
    } else if (Feature == "+rdseed") {
      HasRDSEED = true;
    } else if (Feature == "+adx") {
      HasADX = true;
    } else if (Feature == "+tbm") {
      HasTBM = true;
    } else if (Feature == "+lwp") {
      HasLWP = true;
    } else if (Feature == "+fma") {
      HasFMA = true;
    } else if (Feature == "+f16c") {
      HasF16C = true;
    } else if (Feature == "+avx512cd") {
      HasAVX512CD = true;
    } else if (Feature == "+avx512vpopcntdq") {
      HasAVX512VPOPCNTDQ = true;
    } else if (Feature == "+avx512er") {
      HasAVX512ER = true;
    } else if (Feature == "+avx512pf") {
      HasAVX512PF = true;
    } else if (Feature == "+avx512dq") {
      HasAVX512DQ = true;
    } else if (Feature == "+avx512bw") {
      HasAVX512BW = true;
    } else if (Feature == "+avx512vl") {
      HasAVX512VL = true;
    } else if (Feature == "+avx512vbmi") {
      HasAVX512VBMI = true;
    } else if (Feature == "+avx512ifma") {
      HasAVX512IFMA = true;
    } else if (Feature == "+sha") {
      HasSHA = true;
    } else if (Feature == "+mpx") {
      HasMPX = true;
    } else if (Feature == "+movbe") {
      HasMOVBE = true;
    } else if (Feature == "+sgx") {
      HasSGX = true;
    } else if (Feature == "+cx16") {
      HasCX16 = true;
    } else if (Feature == "+fxsr") {
      HasFXSR = true;
    } else if (Feature == "+xsave") {
      HasXSAVE = true;
    } else if (Feature == "+xsaveopt") {
      HasXSAVEOPT = true;
    } else if (Feature == "+xsavec") {
      HasXSAVEC = true;
    } else if (Feature == "+xsaves") {
      HasXSAVES = true;
    } else if (Feature == "+mwaitx") {
      HasMWAITX = true;
    } else if (Feature == "+pku") {
      HasPKU = true;
    } else if (Feature == "+clflushopt") {
      HasCLFLUSHOPT = true;
    } else if (Feature == "+clwb") {
      HasCLWB = true;
    } else if (Feature == "+prefetchwt1") {
      HasPREFETCHWT1 = true;
    } else if (Feature == "+clzero") {
      HasCLZERO = true;
    }

    X86SSEEnum Level = llvm::StringSwitch<X86SSEEnum>(Feature)
      .Case("+avx512f", AVX512F)
      .Case("+avx2", AVX2)
      .Case("+avx", AVX)
      .Case("+sse4.2", SSE42)
      .Case("+sse4.1", SSE41)
      .Case("+ssse3", SSSE3)
      .Case("+sse3", SSE3)
      .Case("+sse2", SSE2)
      .Case("+sse", SSE1)
      .Default(NoSSE);
    SSELevel = std::max(SSELevel, Level);

    MMX3DNowEnum ThreeDNowLevel =
      llvm::StringSwitch<MMX3DNowEnum>(Feature)
        .Case("+3dnowa", AMD3DNowAthlon)
        .Case("+3dnow", AMD3DNow)
        .Case("+mmx", MMX)
        .Default(NoMMX3DNow);
    MMX3DNowLevel = std::max(MMX3DNowLevel, ThreeDNowLevel);

    XOPEnum XLevel = llvm::StringSwitch<XOPEnum>(Feature)
        .Case("+xop", XOP)
        .Case("+fma4", FMA4)
        .Case("+sse4a", SSE4A)
        .Default(NoXOP);
    XOPLevel = std::max(XOPLevel, XLevel);
  }

  // LLVM doesn't have a separate switch for fpmath, so only accept it if it
  // matches the selected sse level.
  if ((FPMath == FP_SSE && SSELevel < SSE1) ||
      (FPMath == FP_387 && SSELevel >= SSE1)) {
    Diags.Report(diag::err_target_unsupported_fpmath) <<
      (FPMath == FP_SSE ? "sse" : "387");
    return false;
  }

  SimdDefaultAlign =
      hasFeature("avx512f") ? 512 : hasFeature("avx") ? 256 : 128;
  return true;
}

/// X86TargetInfo::getTargetDefines - Return the set of the X86-specific macro
/// definitions for this particular subtarget.
void X86TargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  if (getTriple().getArch() == llvm::Triple::x86_64) {
    Builder.defineMacro("__amd64__");
    Builder.defineMacro("__amd64");
    Builder.defineMacro("__x86_64");
    Builder.defineMacro("__x86_64__");
    if (getTriple().getArchName() == "x86_64h") {
      Builder.defineMacro("__x86_64h");
      Builder.defineMacro("__x86_64h__");
    }
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
    LLVM_FALLTHROUGH;
  case CK_i586:
  case CK_Pentium:
    defineCPUMacros(Builder, "i586");
    defineCPUMacros(Builder, "pentium");
    break;
  case CK_Pentium3:
  case CK_Pentium3M:
  case CK_PentiumM:
    Builder.defineMacro("__tune_pentium3__");
    LLVM_FALLTHROUGH;
  case CK_Pentium2:
  case CK_C3_2:
    Builder.defineMacro("__tune_pentium2__");
    LLVM_FALLTHROUGH;
  case CK_PentiumPro:
    Builder.defineMacro("__tune_i686__");
    Builder.defineMacro("__tune_pentiumpro__");
    LLVM_FALLTHROUGH;
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
  case CK_Bonnell:
    defineCPUMacros(Builder, "atom");
    break;
  case CK_Silvermont:
    defineCPUMacros(Builder, "slm");
    break;
  case CK_Goldmont:
    defineCPUMacros(Builder, "goldmont");
    break;
  case CK_Nehalem:
  case CK_Westmere:
  case CK_SandyBridge:
  case CK_IvyBridge:
  case CK_Haswell:
  case CK_Broadwell:
  case CK_SkylakeClient:
    // FIXME: Historically, we defined this legacy name, it would be nice to
    // remove it at some point. We've never exposed fine-grained names for
    // recent primary x86 CPUs, and we should keep it that way.
    defineCPUMacros(Builder, "corei7");
    break;
  case CK_SkylakeServer:
    defineCPUMacros(Builder, "skx");
    break;
  case CK_Cannonlake:
    break;
  case CK_KNL:
    defineCPUMacros(Builder, "knl");
    break;
  case CK_Lakemont:
    Builder.defineMacro("__tune_lakemont__");
    break;
  case CK_K6_2:
    Builder.defineMacro("__k6_2__");
    Builder.defineMacro("__tune_k6_2__");
    LLVM_FALLTHROUGH;
  case CK_K6_3:
    if (CPU != CK_K6_2) {  // In case of fallthrough
      // FIXME: GCC may be enabling these in cases where some other k6
      // architecture is specified but -m3dnow is explicitly provided. The
      // exact semantics need to be determined and emulated here.
      Builder.defineMacro("__k6_3__");
      Builder.defineMacro("__tune_k6_3__");
    }
    LLVM_FALLTHROUGH;
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
  case CK_BTVER2:
    defineCPUMacros(Builder, "btver2");
    break;
  case CK_BDVER1:
    defineCPUMacros(Builder, "bdver1");
    break;
  case CK_BDVER2:
    defineCPUMacros(Builder, "bdver2");
    break;
  case CK_BDVER3:
    defineCPUMacros(Builder, "bdver3");
    break;
  case CK_BDVER4:
    defineCPUMacros(Builder, "bdver4");
    break;
  case CK_ZNVER1:
    defineCPUMacros(Builder, "znver1");
    break;
  case CK_Geode:
    defineCPUMacros(Builder, "geode");
    break;
  }

  // Target properties.
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // Define __NO_MATH_INLINES on linux/x86 so that we don't get inline
  // functions in glibc header files that use FP Stack inline asm which the
  // backend can't deal with (PR879).
  Builder.defineMacro("__NO_MATH_INLINES");

  if (HasAES)
    Builder.defineMacro("__AES__");

  if (HasPCLMUL)
    Builder.defineMacro("__PCLMUL__");

  if (HasLZCNT)
    Builder.defineMacro("__LZCNT__");

  if (HasRDRND)
    Builder.defineMacro("__RDRND__");

  if (HasFSGSBASE)
    Builder.defineMacro("__FSGSBASE__");

  if (HasBMI)
    Builder.defineMacro("__BMI__");

  if (HasBMI2)
    Builder.defineMacro("__BMI2__");

  if (HasPOPCNT)
    Builder.defineMacro("__POPCNT__");

  if (HasRTM)
    Builder.defineMacro("__RTM__");

  if (HasPRFCHW)
    Builder.defineMacro("__PRFCHW__");

  if (HasRDSEED)
    Builder.defineMacro("__RDSEED__");

  if (HasADX)
    Builder.defineMacro("__ADX__");

  if (HasTBM)
    Builder.defineMacro("__TBM__");

  if (HasLWP)
    Builder.defineMacro("__LWP__");

  if (HasMWAITX)
    Builder.defineMacro("__MWAITX__");

  switch (XOPLevel) {
  case XOP:
    Builder.defineMacro("__XOP__");
    LLVM_FALLTHROUGH;
  case FMA4:
    Builder.defineMacro("__FMA4__");
    LLVM_FALLTHROUGH;
  case SSE4A:
    Builder.defineMacro("__SSE4A__");
    LLVM_FALLTHROUGH;
  case NoXOP:
    break;
  }

  if (HasFMA)
    Builder.defineMacro("__FMA__");

  if (HasF16C)
    Builder.defineMacro("__F16C__");

  if (HasAVX512CD)
    Builder.defineMacro("__AVX512CD__");
  if (HasAVX512VPOPCNTDQ)
    Builder.defineMacro("__AVX512VPOPCNTDQ__");
  if (HasAVX512ER)
    Builder.defineMacro("__AVX512ER__");
  if (HasAVX512PF)
    Builder.defineMacro("__AVX512PF__");
  if (HasAVX512DQ)
    Builder.defineMacro("__AVX512DQ__");
  if (HasAVX512BW)
    Builder.defineMacro("__AVX512BW__");
  if (HasAVX512VL)
    Builder.defineMacro("__AVX512VL__");
  if (HasAVX512VBMI)
    Builder.defineMacro("__AVX512VBMI__");
  if (HasAVX512IFMA)
    Builder.defineMacro("__AVX512IFMA__");

  if (HasSHA)
    Builder.defineMacro("__SHA__");

  if (HasFXSR)
    Builder.defineMacro("__FXSR__");
  if (HasXSAVE)
    Builder.defineMacro("__XSAVE__");
  if (HasXSAVEOPT)
    Builder.defineMacro("__XSAVEOPT__");
  if (HasXSAVEC)
    Builder.defineMacro("__XSAVEC__");
  if (HasXSAVES)
    Builder.defineMacro("__XSAVES__");
  if (HasPKU)
    Builder.defineMacro("__PKU__");
  if (HasCX16)
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16");
  if (HasCLFLUSHOPT)
    Builder.defineMacro("__CLFLUSHOPT__");
  if (HasCLWB)
    Builder.defineMacro("__CLWB__");
  if (HasMPX)
    Builder.defineMacro("__MPX__");
  if (HasSGX)
    Builder.defineMacro("__SGX__");
  if (HasPREFETCHWT1)
    Builder.defineMacro("__PREFETCHWT1__");
  if (HasCLZERO)
    Builder.defineMacro("__CLZERO__");

  // Each case falls through to the previous one here.
  switch (SSELevel) {
  case AVX512F:
    Builder.defineMacro("__AVX512F__");
    LLVM_FALLTHROUGH;
  case AVX2:
    Builder.defineMacro("__AVX2__");
    LLVM_FALLTHROUGH;
  case AVX:
    Builder.defineMacro("__AVX__");
    LLVM_FALLTHROUGH;
  case SSE42:
    Builder.defineMacro("__SSE4_2__");
    LLVM_FALLTHROUGH;
  case SSE41:
    Builder.defineMacro("__SSE4_1__");
    LLVM_FALLTHROUGH;
  case SSSE3:
    Builder.defineMacro("__SSSE3__");
    LLVM_FALLTHROUGH;
  case SSE3:
    Builder.defineMacro("__SSE3__");
    LLVM_FALLTHROUGH;
  case SSE2:
    Builder.defineMacro("__SSE2__");
    Builder.defineMacro("__SSE2_MATH__");  // -mfp-math=sse always implied.
    LLVM_FALLTHROUGH;
  case SSE1:
    Builder.defineMacro("__SSE__");
    Builder.defineMacro("__SSE_MATH__");   // -mfp-math=sse always implied.
    LLVM_FALLTHROUGH;
  case NoSSE:
    break;
  }

  if (Opts.MicrosoftExt && getTriple().getArch() == llvm::Triple::x86) {
    switch (SSELevel) {
    case AVX512F:
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
      break;
    }
  }

  // Each case falls through to the previous one here.
  switch (MMX3DNowLevel) {
  case AMD3DNowAthlon:
    Builder.defineMacro("__3dNOW_A__");
    LLVM_FALLTHROUGH;
  case AMD3DNow:
    Builder.defineMacro("__3dNOW__");
    LLVM_FALLTHROUGH;
  case MMX:
    Builder.defineMacro("__MMX__");
    LLVM_FALLTHROUGH;
  case NoMMX3DNow:
    break;
  }

  if (CPU >= CK_i486) {
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
  }
  if (CPU >= CK_i586)
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");

  if (HasFloat128)
    Builder.defineMacro("__SIZEOF_FLOAT128__", "16");
}

bool X86TargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("aes", HasAES)
      .Case("avx", SSELevel >= AVX)
      .Case("avx2", SSELevel >= AVX2)
      .Case("avx512f", SSELevel >= AVX512F)
      .Case("avx512cd", HasAVX512CD)
      .Case("avx512vpopcntdq", HasAVX512VPOPCNTDQ)
      .Case("avx512er", HasAVX512ER)
      .Case("avx512pf", HasAVX512PF)
      .Case("avx512dq", HasAVX512DQ)
      .Case("avx512bw", HasAVX512BW)
      .Case("avx512vl", HasAVX512VL)
      .Case("avx512vbmi", HasAVX512VBMI)
      .Case("avx512ifma", HasAVX512IFMA)
      .Case("bmi", HasBMI)
      .Case("bmi2", HasBMI2)
      .Case("clflushopt", HasCLFLUSHOPT)
      .Case("clwb", HasCLWB)
      .Case("clzero", HasCLZERO)
      .Case("cx16", HasCX16)
      .Case("f16c", HasF16C)
      .Case("fma", HasFMA)
      .Case("fma4", XOPLevel >= FMA4)
      .Case("fsgsbase", HasFSGSBASE)
      .Case("fxsr", HasFXSR)
      .Case("lzcnt", HasLZCNT)
      .Case("mm3dnow", MMX3DNowLevel >= AMD3DNow)
      .Case("mm3dnowa", MMX3DNowLevel >= AMD3DNowAthlon)
      .Case("mmx", MMX3DNowLevel >= MMX)
      .Case("movbe", HasMOVBE)
      .Case("mpx", HasMPX)
      .Case("pclmul", HasPCLMUL)
      .Case("pku", HasPKU)
      .Case("popcnt", HasPOPCNT)
      .Case("prefetchwt1", HasPREFETCHWT1)
      .Case("prfchw", HasPRFCHW)
      .Case("rdrnd", HasRDRND)
      .Case("rdseed", HasRDSEED)
      .Case("rtm", HasRTM)
      .Case("sgx", HasSGX)
      .Case("sha", HasSHA)
      .Case("sse", SSELevel >= SSE1)
      .Case("sse2", SSELevel >= SSE2)
      .Case("sse3", SSELevel >= SSE3)
      .Case("ssse3", SSELevel >= SSSE3)
      .Case("sse4.1", SSELevel >= SSE41)
      .Case("sse4.2", SSELevel >= SSE42)
      .Case("sse4a", XOPLevel >= SSE4A)
      .Case("tbm", HasTBM)
      .Case("lwp", HasLWP)
      .Case("x86", true)
      .Case("x86_32", getTriple().getArch() == llvm::Triple::x86)
      .Case("x86_64", getTriple().getArch() == llvm::Triple::x86_64)
      .Case("xop", XOPLevel >= XOP)
      .Case("xsave", HasXSAVE)
      .Case("xsavec", HasXSAVEC)
      .Case("xsaves", HasXSAVES)
      .Case("xsaveopt", HasXSAVEOPT)
      .Default(false);
}

// We can't use a generic validation scheme for the features accepted here
// versus subtarget features accepted in the target attribute because the
// bitfield structure that's initialized in the runtime only supports the
// below currently rather than the full range of subtarget features. (See
// X86TargetInfo::hasFeature for a somewhat comprehensive list).
bool X86TargetInfo::validateCpuSupports(StringRef FeatureStr) const {
  return llvm::StringSwitch<bool>(FeatureStr)
      .Case("cmov", true)
      .Case("mmx", true)
      .Case("popcnt", true)
      .Case("sse", true)
      .Case("sse2", true)
      .Case("sse3", true)
      .Case("ssse3", true)
      .Case("sse4.1", true)
      .Case("sse4.2", true)
      .Case("avx", true)
      .Case("avx2", true)
      .Case("sse4a", true)
      .Case("fma4", true)
      .Case("xop", true)
      .Case("fma", true)
      .Case("avx512f", true)
      .Case("bmi", true)
      .Case("bmi2", true)
      .Case("aes", true)
      .Case("pclmul", true)
      .Case("avx512vl", true)
      .Case("avx512bw", true)
      .Case("avx512dq", true)
      .Case("avx512cd", true)
      .Case("avx512vpopcntdq", true)
      .Case("avx512er", true)
      .Case("avx512pf", true)
      .Case("avx512vbmi", true)
      .Case("avx512ifma", true)
      .Default(false);
}

bool
X86TargetInfo::validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &Info) const {
  switch (*Name) {
  default: return false;
  // Constant constraints.
  case 'e': // 32-bit signed integer constant for use with sign-extending x86_64
            // instructions.
  case 'Z': // 32-bit unsigned integer constant for use with zero-extending
            // x86_64 instructions.
  case 's':
    Info.setRequiresImmediate();
    return true;
  case 'I':
    Info.setRequiresImmediate(0, 31);
    return true;
  case 'J':
    Info.setRequiresImmediate(0, 63);
    return true;
  case 'K':
    Info.setRequiresImmediate(-128, 127);
    return true;
  case 'L':
    Info.setRequiresImmediate({ int(0xff), int(0xffff), int(0xffffffff) });
    return true;
  case 'M':
    Info.setRequiresImmediate(0, 3);
    return true;
  case 'N':
    Info.setRequiresImmediate(0, 255);
    return true;
  case 'O':
    Info.setRequiresImmediate(0, 127);
    return true;
  // Register constraints.
  case 'Y': // 'Y' is the first character for several 2-character constraints.
    // Shift the pointer to the second character of the constraint.
    Name++;
    switch (*Name) {
    default:
      return false;
    case '0': // First SSE register.
    case 't': // Any SSE register, when SSE2 is enabled.
    case 'i': // Any SSE register, when SSE2 and inter-unit moves enabled.
    case 'm': // Any MMX register, when inter-unit moves enabled.
    case 'k': // AVX512 arch mask registers: k1-k7.
      Info.setAllowsRegister();
      return true;
    }
  case 'f': // Any x87 floating point stack register.
    // Constraint 'f' cannot be used for output operands.
    if (Info.ConstraintStr[0] == '=')
      return false;
    Info.setAllowsRegister();
    return true;
  case 'a': // eax.
  case 'b': // ebx.
  case 'c': // ecx.
  case 'd': // edx.
  case 'S': // esi.
  case 'D': // edi.
  case 'A': // edx:eax.
  case 't': // Top of floating point stack.
  case 'u': // Second from top of floating point stack.
  case 'q': // Any register accessible as [r]l: a, b, c, and d.
  case 'y': // Any MMX register.
  case 'v': // Any {X,Y,Z}MM register (Arch & context dependent)
  case 'x': // Any SSE register.
  case 'k': // Any AVX512 mask register (same as Yk, additionaly allows k0
            // for intermideate k reg operations).
  case 'Q': // Any register accessible as [r]h: a, b, c, and d.
  case 'R': // "Legacy" registers: ax, bx, cx, dx, di, si, sp, bp.
  case 'l': // "Index" registers: any general register that can be used as an
            // index in a base+index memory access.
    Info.setAllowsRegister();
    return true;
  // Floating point constant constraints.
  case 'C': // SSE floating point constant.
  case 'G': // x87 floating point constant.
    return true;
  }
}

bool X86TargetInfo::validateOutputSize(StringRef Constraint,
                                       unsigned Size) const {
  // Strip off constraint modifiers.
  while (Constraint[0] == '=' ||
         Constraint[0] == '+' ||
         Constraint[0] == '&')
    Constraint = Constraint.substr(1);

  return validateOperandSize(Constraint, Size);
}

bool X86TargetInfo::validateInputSize(StringRef Constraint,
                                      unsigned Size) const {
  return validateOperandSize(Constraint, Size);
}

bool X86TargetInfo::validateOperandSize(StringRef Constraint,
                                        unsigned Size) const {
  switch (Constraint[0]) {
  default: break;
  case 'k':
  // Registers k0-k7 (AVX512) size limit is 64 bit.
  case 'y':
    return Size <= 64;
  case 'f':
  case 't':
  case 'u':
    return Size <= 128;
  case 'v':
  case 'x':
    if (SSELevel >= AVX512F)
      // 512-bit zmm registers can be used if target supports AVX512F.
      return Size <= 512U;
    else if (SSELevel >= AVX)
      // 256-bit ymm registers can be used if target supports AVX.
      return Size <= 256U;
    return Size <= 128U;
  case 'Y':
    // 'Y' is the first character for several 2-character constraints.
    switch (Constraint[1]) {
    default: break;
    case 'm':
      // 'Ym' is synonymous with 'y'.
    case 'k':
      return Size <= 64;
    case 'i':
    case 't':
      // 'Yi' and 'Yt' are synonymous with 'x' when SSE2 is enabled.
      if (SSELevel >= AVX512F)
        return Size <= 512U;
      else if (SSELevel >= AVX)
        return Size <= 256U;
      return SSELevel >= SSE2 && Size <= 128U;
    }

  }

  return true;
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
  case 'Y':
    switch (Constraint[1]) {
    default:
      // Break from inner switch and fall through (copy single char),
      // continue parsing after copying the current constraint into 
      // the return string.
      break;
    case 'k':
      // "^" hints llvm that this is a 2 letter constraint.
      // "Constraint++" is used to promote the string iterator 
      // to the next constraint.
      return std::string("^") + std::string(Constraint++, 2);
    } 
    LLVM_FALLTHROUGH;
  default:
    return std::string(1, *Constraint);
  }
}

// X86-32 generic target
class X86_32TargetInfo : public X86TargetInfo {
public:
  X86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : X86TargetInfo(Triple, Opts) {
    DoubleAlign = LongLongAlign = 32;
    LongDoubleWidth = 96;
    LongDoubleAlign = 32;
    SuitableAlign = 128;
    resetDataLayout("e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128");
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
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0) return 0;
    if (RegNo == 1) return 2;
    return -1;
  }
  bool validateOperandSize(StringRef Constraint,
                           unsigned Size) const override {
    switch (Constraint[0]) {
    default: break;
    case 'R':
    case 'q':
    case 'Q':
    case 'a':
    case 'b':
    case 'c':
    case 'd':
    case 'S':
    case 'D':
      return Size <= 32;
    case 'A':
      return Size <= 64;
    }

    return X86TargetInfo::validateOperandSize(Constraint, Size);
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfoX86, clang::X86::LastX86CommonBuiltin -
                                                  Builtin::FirstTSBuiltin + 1);
  }
};

class NetBSDI386TargetInfo : public NetBSDTargetInfo<X86_32TargetInfo> {
public:
  NetBSDI386TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : NetBSDTargetInfo<X86_32TargetInfo>(Triple, Opts) {}

  unsigned getFloatEvalMethod() const override {
    unsigned Major, Minor, Micro;
    getTriple().getOSVersion(Major, Minor, Micro);
    // New NetBSD uses the default rounding mode.
    if (Major >= 7 || (Major == 6 && Minor == 99 && Micro >= 26) || Major == 0)
      return X86_32TargetInfo::getFloatEvalMethod();
    // NetBSD before 6.99.26 defaults to "double" rounding.
    return 1;
  }
};

class OpenBSDI386TargetInfo : public OpenBSDTargetInfo<X86_32TargetInfo> {
public:
  OpenBSDI386TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OpenBSDTargetInfo<X86_32TargetInfo>(Triple, Opts) {
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    PtrDiffType = SignedLong;
  }
};

class BitrigI386TargetInfo : public BitrigTargetInfo<X86_32TargetInfo> {
public:
  BitrigI386TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : BitrigTargetInfo<X86_32TargetInfo>(Triple, Opts) {
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    PtrDiffType = SignedLong;
  }
};

class DarwinI386TargetInfo : public DarwinTargetInfo<X86_32TargetInfo> {
public:
  DarwinI386TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<X86_32TargetInfo>(Triple, Opts) {
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    SuitableAlign = 128;
    MaxVectorAlign = 256;
    // The watchOS simulator uses the builtin bool type for Objective-C.
    llvm::Triple T = llvm::Triple(Triple);
    if (T.isWatchOS())
      UseSignedCharForObjCBool = false;
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    resetDataLayout("e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128");
    HasAlignMac68kSupport = true;
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    if (!DarwinTargetInfo<X86_32TargetInfo>::handleTargetFeatures(Features,
                                                                  Diags))
      return false;
    // We now know the features we have: we can decide how to align vectors.
    MaxVectorAlign =
        hasFeature("avx512f") ? 512 : hasFeature("avx") ? 256 : 128;
    return true;
  }
};

// x86-32 Windows target
class WindowsX86_32TargetInfo : public WindowsTargetInfo<X86_32TargetInfo> {
public:
  WindowsX86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsTargetInfo<X86_32TargetInfo>(Triple, Opts) {
    WCharType = UnsignedShort;
    DoubleAlign = LongLongAlign = 64;
    bool IsWinCOFF =
        getTriple().isOSWindows() && getTriple().isOSBinFormatCOFF();
    resetDataLayout(IsWinCOFF
                        ? "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
                        : "e-m:e-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsTargetInfo<X86_32TargetInfo>::getTargetDefines(Opts, Builder);
  }
};

// x86-32 Windows Visual Studio target
class MicrosoftX86_32TargetInfo : public WindowsX86_32TargetInfo {
public:
  MicrosoftX86_32TargetInfo(const llvm::Triple &Triple,
                            const TargetOptions &Opts)
      : WindowsX86_32TargetInfo(Triple, Opts) {
    LongDoubleWidth = LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsX86_32TargetInfo::getTargetDefines(Opts, Builder);
    WindowsX86_32TargetInfo::getVisualStudioDefines(Opts, Builder);
    // The value of the following reflects processor type.
    // 300=386, 400=486, 500=Pentium, 600=Blend (default)
    // We lost the original triple, so we use the default.
    Builder.defineMacro("_M_IX86", "600");
  }
};

static void addCygMingDefines(const LangOptions &Opts, MacroBuilder &Builder) {
  // Mingw and cygwin define __declspec(a) to __attribute__((a)).  Clang
  // supports __declspec natively under -fms-extensions, but we define a no-op
  // __declspec macro anyway for pre-processor compatibility.
  if (Opts.MicrosoftExt)
    Builder.defineMacro("__declspec", "__declspec");
  else
    Builder.defineMacro("__declspec(a)", "__attribute__((a))");

  if (!Opts.MicrosoftExt) {
    // Provide macros for all the calling convention keywords.  Provide both
    // single and double underscore prefixed variants.  These are available on
    // x64 as well as x86, even though they have no effect.
    const char *CCs[] = {"cdecl", "stdcall", "fastcall", "thiscall", "pascal"};
    for (const char *CC : CCs) {
      std::string GCCSpelling = "__attribute__((__";
      GCCSpelling += CC;
      GCCSpelling += "__))";
      Builder.defineMacro(Twine("_") + CC, GCCSpelling);
      Builder.defineMacro(Twine("__") + CC, GCCSpelling);
    }
  }
}

static void addMinGWDefines(const LangOptions &Opts, MacroBuilder &Builder) {
  Builder.defineMacro("__MSVCRT__");
  Builder.defineMacro("__MINGW32__");
  addCygMingDefines(Opts, Builder);
}

// x86-32 MinGW target
class MinGWX86_32TargetInfo : public WindowsX86_32TargetInfo {
public:
  MinGWX86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsX86_32TargetInfo(Triple, Opts) {
    HasFloat128 = true;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsX86_32TargetInfo::getTargetDefines(Opts, Builder);
    DefineStd(Builder, "WIN32", Opts);
    DefineStd(Builder, "WINNT", Opts);
    Builder.defineMacro("_X86_");
    addMinGWDefines(Opts, Builder);
  }
};

// x86-32 Cygwin target
class CygwinX86_32TargetInfo : public X86_32TargetInfo {
public:
  CygwinX86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : X86_32TargetInfo(Triple, Opts) {
    WCharType = UnsignedShort;
    DoubleAlign = LongLongAlign = 64;
    resetDataLayout("e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("_X86_");
    Builder.defineMacro("__CYGWIN__");
    Builder.defineMacro("__CYGWIN32__");
    addCygMingDefines(Opts, Builder);
    DefineStd(Builder, "unix", Opts);
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
  }
};

// x86-32 Haiku target
class HaikuX86_32TargetInfo : public HaikuTargetInfo<X86_32TargetInfo> {
public:
  HaikuX86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
    : HaikuTargetInfo<X86_32TargetInfo>(Triple, Opts) {
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    HaikuTargetInfo<X86_32TargetInfo>::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__INTEL__");
  }
};

// X86-32 MCU target
class MCUX86_32TargetInfo : public X86_32TargetInfo {
public:
  MCUX86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : X86_32TargetInfo(Triple, Opts) {
    LongDoubleWidth = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    resetDataLayout("e-m:e-p:32:32-i64:32-f64:32-f128:32-n8:16:32-a:0:32-S32");
    WIntType = UnsignedInt;
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    // On MCU we support only C calling convention.
    return CC == CC_C ? CCCR_OK : CCCR_Warning;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__iamcu");
    Builder.defineMacro("__iamcu__");
  }

  bool allowsLargerPreferedTypeAlignment() const override {
    return false;
  }
};

// RTEMS Target
template<typename Target>
class RTEMSTargetInfo : public OSTargetInfo<Target> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    // RTEMS defines; list based off of gcc output

    Builder.defineMacro("__rtems__");
    Builder.defineMacro("__ELF__");
  }

public:
  RTEMSTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OSTargetInfo<Target>(Triple, Opts) {
    switch (Triple.getArch()) {
    default:
    case llvm::Triple::x86:
      // this->MCountName = ".mcount";
      break;
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
    case llvm::Triple::ppc64le:
      // this->MCountName = "_mcount";
      break;
    case llvm::Triple::arm:
      // this->MCountName = "__mcount";
      break;
    }
  }
};

// x86-32 RTEMS target
class RTEMSX86_32TargetInfo : public X86_32TargetInfo {
public:
  RTEMSX86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : X86_32TargetInfo(Triple, Opts) {
    SizeType = UnsignedLong;
    IntPtrType = SignedLong;
    PtrDiffType = SignedLong;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    X86_32TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__INTEL__");
    Builder.defineMacro("__rtems__");
  }
};

// x86-64 generic target
class X86_64TargetInfo : public X86TargetInfo {
public:
  X86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : X86TargetInfo(Triple, Opts) {
    const bool IsX32 = getTriple().getEnvironment() == llvm::Triple::GNUX32;
    bool IsWinCOFF =
        getTriple().isOSWindows() && getTriple().isOSBinFormatCOFF();
    LongWidth = LongAlign = PointerWidth = PointerAlign = IsX32 ? 32 : 64;
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    LargeArrayMinWidth = 128;
    LargeArrayAlign = 128;
    SuitableAlign = 128;
    SizeType    = IsX32 ? UnsignedInt      : UnsignedLong;
    PtrDiffType = IsX32 ? SignedInt        : SignedLong;
    IntPtrType  = IsX32 ? SignedInt        : SignedLong;
    IntMaxType  = IsX32 ? SignedLongLong   : SignedLong;
    Int64Type   = IsX32 ? SignedLongLong   : SignedLong;
    RegParmMax = 6;

    // Pointers are 32-bit in x32.
    resetDataLayout(IsX32
                        ? "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128"
                        : IsWinCOFF ? "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
                                    : "e-m:e-i64:64-f80:128-n8:16:32:64-S128");

    // Use fpret only for long double.
    RealTypeUsesObjCFPRet = (1 << TargetInfo::LongDouble);

    // Use fp2ret for _Complex long double.
    ComplexLongDoubleUsesFP2Ret = true;

    // Make __builtin_ms_va_list available.
    HasBuiltinMSVaList = true;

    // x86-64 has atomics up to 16 bytes.
    MaxAtomicPromoteWidth = 128;
    MaxAtomicInlineWidth = 128;
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::X86_64ABIBuiltinVaList;
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0) return 0;
    if (RegNo == 1) return 1;
    return -1;
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_C:
    case CC_Swift:
    case CC_X86VectorCall:
    case CC_IntelOclBicc:
    case CC_Win64:
    case CC_PreserveMost:
    case CC_PreserveAll:
    case CC_X86RegCall:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }

  CallingConv getDefaultCallingConv(CallingConvMethodType MT) const override {
    return CC_C;
  }

  // for x32 we need it here explicitly
  bool hasInt128Type() const override { return true; }
  unsigned getUnwindWordWidth() const override { return 64; }
  unsigned getRegisterWidth() const override { return 64; }

  bool validateGlobalRegisterVariable(StringRef RegName,
                                      unsigned RegSize,
                                      bool &HasSizeMismatch) const override {
    // rsp and rbp are the only 64-bit registers the x86 backend can currently
    // handle.
    if (RegName.equals("rsp") || RegName.equals("rbp")) {
      // Check that the register size is 64-bit.
      HasSizeMismatch = RegSize != 64;
      return true;
    }

    // Check if the register is a 32-bit register the backend can handle.
    return X86TargetInfo::validateGlobalRegisterVariable(RegName, RegSize,
                                                         HasSizeMismatch);
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfoX86,
                              X86::LastTSBuiltin - Builtin::FirstTSBuiltin);
  }
};

// x86-64 Windows target
class WindowsX86_64TargetInfo : public WindowsTargetInfo<X86_64TargetInfo> {
public:
  WindowsX86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsTargetInfo<X86_64TargetInfo>(Triple, Opts) {
    WCharType = UnsignedShort;
    LongWidth = LongAlign = 32;
    DoubleAlign = LongLongAlign = 64;
    IntMaxType = SignedLongLong;
    Int64Type = SignedLongLong;
    SizeType = UnsignedLongLong;
    PtrDiffType = SignedLongLong;
    IntPtrType = SignedLongLong;
  }

  void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const override {
    WindowsTargetInfo<X86_64TargetInfo>::getTargetDefines(Opts, Builder);
    Builder.defineMacro("_WIN64");
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_X86StdCall:
    case CC_X86ThisCall:
    case CC_X86FastCall:
      return CCCR_Ignore;
    case CC_C:
    case CC_X86VectorCall:
    case CC_IntelOclBicc:
    case CC_X86_64SysV:
    case CC_Swift:
    case CC_X86RegCall:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }
};

// x86-64 Windows Visual Studio target
class MicrosoftX86_64TargetInfo : public WindowsX86_64TargetInfo {
public:
  MicrosoftX86_64TargetInfo(const llvm::Triple &Triple,
                            const TargetOptions &Opts)
      : WindowsX86_64TargetInfo(Triple, Opts) {
    LongDoubleWidth = LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsX86_64TargetInfo::getTargetDefines(Opts, Builder);
    WindowsX86_64TargetInfo::getVisualStudioDefines(Opts, Builder);
    Builder.defineMacro("_M_X64", "100");
    Builder.defineMacro("_M_AMD64", "100");
  }
};

// x86-64 MinGW target
class MinGWX86_64TargetInfo : public WindowsX86_64TargetInfo {
public:
  MinGWX86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsX86_64TargetInfo(Triple, Opts) {
    // Mingw64 rounds long double size and alignment up to 16 bytes, but sticks
    // with x86 FP ops. Weird.
    LongDoubleWidth = LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::x87DoubleExtended();
    HasFloat128 = true;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsX86_64TargetInfo::getTargetDefines(Opts, Builder);
    DefineStd(Builder, "WIN64", Opts);
    Builder.defineMacro("__MINGW64__");
    addMinGWDefines(Opts, Builder);

    // GCC defines this macro when it is using __gxx_personality_seh0.
    if (!Opts.SjLjExceptions)
      Builder.defineMacro("__SEH__");
  }
};

// x86-64 Cygwin target
class CygwinX86_64TargetInfo : public X86_64TargetInfo {
public:
  CygwinX86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : X86_64TargetInfo(Triple, Opts) {
    TLSSupported = false;
    WCharType = UnsignedShort;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    X86_64TargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__x86_64__");
    Builder.defineMacro("__CYGWIN__");
    Builder.defineMacro("__CYGWIN64__");
    addCygMingDefines(Opts, Builder);
    DefineStd(Builder, "unix", Opts);
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");

    // GCC defines this macro when it is using __gxx_personality_seh0.
    if (!Opts.SjLjExceptions)
      Builder.defineMacro("__SEH__");
  }
};

class DarwinX86_64TargetInfo : public DarwinTargetInfo<X86_64TargetInfo> {
public:
  DarwinX86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<X86_64TargetInfo>(Triple, Opts) {
    Int64Type = SignedLongLong;
    // The 64-bit iOS simulator uses the builtin bool type for Objective-C.
    llvm::Triple T = llvm::Triple(Triple);
    if (T.isiOS())
      UseSignedCharForObjCBool = false;
    resetDataLayout("e-m:o-i64:64-f80:128-n8:16:32:64-S128");
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    if (!DarwinTargetInfo<X86_64TargetInfo>::handleTargetFeatures(Features,
                                                                  Diags))
      return false;
    // We now know the features we have: we can decide how to align vectors.
    MaxVectorAlign =
        hasFeature("avx512f") ? 512 : hasFeature("avx") ? 256 : 128;
    return true;
  }
};

class OpenBSDX86_64TargetInfo : public OpenBSDTargetInfo<X86_64TargetInfo> {
public:
  OpenBSDX86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : OpenBSDTargetInfo<X86_64TargetInfo>(Triple, Opts) {
    IntMaxType = SignedLongLong;
    Int64Type = SignedLongLong;
  }
};

class BitrigX86_64TargetInfo : public BitrigTargetInfo<X86_64TargetInfo> {
public:
  BitrigX86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : BitrigTargetInfo<X86_64TargetInfo>(Triple, Opts) {
    IntMaxType = SignedLongLong;
    Int64Type = SignedLongLong;
  }
};

class ARMTargetInfo : public TargetInfo {
  // Possible FPU choices.
  enum FPUMode {
    VFP2FPU = (1 << 0),
    VFP3FPU = (1 << 1),
    VFP4FPU = (1 << 2),
    NeonFPU = (1 << 3),
    FPARMV8 = (1 << 4)
  };

  // Possible HWDiv features.
  enum HWDivMode {
    HWDivThumb = (1 << 0),
    HWDivARM = (1 << 1)
  };

  static bool FPUModeIsVFP(FPUMode Mode) {
    return Mode & (VFP2FPU | VFP3FPU | VFP4FPU | NeonFPU | FPARMV8);
  }

  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char * const GCCRegNames[];

  std::string ABI, CPU;

  StringRef CPUProfile;
  StringRef CPUAttr;

  enum {
    FP_Default,
    FP_VFP,
    FP_Neon
  } FPMath;

  unsigned ArchISA;
  unsigned ArchKind = llvm::ARM::AK_ARMV4T;
  unsigned ArchProfile;
  unsigned ArchVersion;

  unsigned FPU : 5;

  unsigned IsAAPCS : 1;
  unsigned HWDiv : 2;

  // Initialized via features.
  unsigned SoftFloat : 1;
  unsigned SoftFloatABI : 1;

  unsigned CRC : 1;
  unsigned Crypto : 1;
  unsigned DSP : 1;
  unsigned Unaligned : 1;

  enum {
    LDREX_B = (1 << 0), /// byte (8-bit)
    LDREX_H = (1 << 1), /// half (16-bit)
    LDREX_W = (1 << 2), /// word (32-bit)
    LDREX_D = (1 << 3), /// double (64-bit)
  };

  uint32_t LDREX;

  // ACLE 6.5.1 Hardware floating point
  enum {
    HW_FP_HP = (1 << 1), /// half (16-bit)
    HW_FP_SP = (1 << 2), /// single (32-bit)
    HW_FP_DP = (1 << 3), /// double (64-bit)
  };
  uint32_t HW_FP;

  static const Builtin::Info BuiltinInfo[];

  void setABIAAPCS() {
    IsAAPCS = true;

    DoubleAlign = LongLongAlign = LongDoubleAlign = SuitableAlign = 64;
    const llvm::Triple &T = getTriple();

    // size_t is unsigned long on MachO-derived environments, NetBSD,
    // OpenBSD and Bitrig.
    if (T.isOSBinFormatMachO() || T.getOS() == llvm::Triple::NetBSD ||
        T.getOS() == llvm::Triple::OpenBSD ||
        T.getOS() == llvm::Triple::Bitrig)
      SizeType = UnsignedLong;
    else
      SizeType = UnsignedInt;

    switch (T.getOS()) {
    case llvm::Triple::NetBSD:
    case llvm::Triple::OpenBSD:
      WCharType = SignedInt;
      break;
    case llvm::Triple::Win32:
      WCharType = UnsignedShort;
      break;
    case llvm::Triple::Linux:
    default:
      // AAPCS 7.1.1, ARM-Linux ABI 2.4: type of wchar_t is unsigned int.
      WCharType = UnsignedInt;
      break;
    }

    UseBitFieldTypeAlignment = true;

    ZeroLengthBitfieldBoundary = 0;

    // Thumb1 add sp, #imm requires the immediate value be multiple of 4,
    // so set preferred for small types to 32.
    if (T.isOSBinFormatMachO()) {
      resetDataLayout(BigEndian
                          ? "E-m:o-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
                          : "e-m:o-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64");
    } else if (T.isOSWindows()) {
      assert(!BigEndian && "Windows on ARM does not support big endian");
      resetDataLayout("e"
                      "-m:w"
                      "-p:32:32"
                      "-i64:64"
                      "-v128:64:128"
                      "-a:0:32"
                      "-n32"
                      "-S64");
    } else if (T.isOSNaCl()) {
      assert(!BigEndian && "NaCl on ARM does not support big endian");
      resetDataLayout("e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S128");
    } else {
      resetDataLayout(BigEndian
                          ? "E-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
                          : "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64");
    }

    // FIXME: Enumerated types are variable width in straight AAPCS.
  }

  void setABIAPCS(bool IsAAPCS16) {
    const llvm::Triple &T = getTriple();

    IsAAPCS = false;

    if (IsAAPCS16)
      DoubleAlign = LongLongAlign = LongDoubleAlign = SuitableAlign = 64;
    else
      DoubleAlign = LongLongAlign = LongDoubleAlign = SuitableAlign = 32;

    // size_t is unsigned int on FreeBSD.
    if (T.getOS() == llvm::Triple::FreeBSD)
      SizeType = UnsignedInt;
    else
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

    if (T.isOSBinFormatMachO() && IsAAPCS16) {
      assert(!BigEndian && "AAPCS16 does not support big-endian");
      resetDataLayout("e-m:o-p:32:32-i64:64-a:0:32-n32-S128");
    } else if (T.isOSBinFormatMachO())
      resetDataLayout(
          BigEndian
              ? "E-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
              : "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32");
    else
      resetDataLayout(
          BigEndian
              ? "E-m:e-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
              : "e-m:e-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32");

    // FIXME: Override "preferred align" for double and long long.
  }

  void setArchInfo() {
    StringRef ArchName = getTriple().getArchName();

    ArchISA     = llvm::ARM::parseArchISA(ArchName);
    CPU         = llvm::ARM::getDefaultCPU(ArchName);
    unsigned AK = llvm::ARM::parseArch(ArchName);
    if (AK != llvm::ARM::AK_INVALID)
      ArchKind = AK;
    setArchInfo(ArchKind);
  }

  void setArchInfo(unsigned Kind) {
    StringRef SubArch;

    // cache TargetParser info
    ArchKind    = Kind;
    SubArch     = llvm::ARM::getSubArch(ArchKind);
    ArchProfile = llvm::ARM::parseArchProfile(SubArch);
    ArchVersion = llvm::ARM::parseArchVersion(SubArch);

    // cache CPU related strings
    CPUAttr    = getCPUAttr();
    CPUProfile = getCPUProfile();
  }

  void setAtomic() {
    // when triple does not specify a sub arch,
    // then we are not using inline atomics
    bool ShouldUseInlineAtomic =
                   (ArchISA == llvm::ARM::IK_ARM   && ArchVersion >= 6) ||
                   (ArchISA == llvm::ARM::IK_THUMB && ArchVersion >= 7);
    // Cortex M does not support 8 byte atomics, while general Thumb2 does.
    if (ArchProfile == llvm::ARM::PK_M) {
      MaxAtomicPromoteWidth = 32;
      if (ShouldUseInlineAtomic)
        MaxAtomicInlineWidth = 32;
    }
    else {
      MaxAtomicPromoteWidth = 64;
      if (ShouldUseInlineAtomic)
        MaxAtomicInlineWidth = 64;
    }
  }

  bool isThumb() const {
    return (ArchISA == llvm::ARM::IK_THUMB);
  }

  bool supportsThumb() const {
    return CPUAttr.count('T') || ArchVersion >= 6;
  }

  bool supportsThumb2() const {
    return CPUAttr.equals("6T2") ||
           (ArchVersion >= 7 && !CPUAttr.equals("8M_BASE"));
  }

  StringRef getCPUAttr() const {
    // For most sub-arches, the build attribute CPU name is enough.
    // For Cortex variants, it's slightly different.
    switch(ArchKind) {
    default:
      return llvm::ARM::getCPUAttr(ArchKind);
    case llvm::ARM::AK_ARMV6M:
      return "6M";
    case llvm::ARM::AK_ARMV7S:
      return "7S";
    case llvm::ARM::AK_ARMV7A:
      return "7A";
    case llvm::ARM::AK_ARMV7R:
      return "7R";
    case llvm::ARM::AK_ARMV7M:
      return "7M";
    case llvm::ARM::AK_ARMV7EM:
      return "7EM";
    case llvm::ARM::AK_ARMV7VE:
      return "7VE";
    case llvm::ARM::AK_ARMV8A:
      return "8A";
    case llvm::ARM::AK_ARMV8_1A:
      return "8_1A";
    case llvm::ARM::AK_ARMV8_2A:
      return "8_2A";
    case llvm::ARM::AK_ARMV8MBaseline:
      return "8M_BASE";
    case llvm::ARM::AK_ARMV8MMainline:
      return "8M_MAIN";
    case llvm::ARM::AK_ARMV8R:
      return "8R";
    }
  }

  StringRef getCPUProfile() const {
    switch(ArchProfile) {
    case llvm::ARM::PK_A:
      return "A";
    case llvm::ARM::PK_R:
      return "R";
    case llvm::ARM::PK_M:
      return "M";
    default:
      return "";
    }
  }

public:
  ARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TargetInfo(Triple), FPMath(FP_Default), IsAAPCS(true), LDREX(0),
        HW_FP(0) {

    switch (getTriple().getOS()) {
    case llvm::Triple::NetBSD:
    case llvm::Triple::OpenBSD:
      PtrDiffType = SignedLong;
      break;
    default:
      PtrDiffType = SignedInt;
      break;
    }

    // Cache arch related info.
    setArchInfo();

    // {} in inline assembly are neon specifiers, not assembly variant
    // specifiers.
    NoAsmVariants = true;

    // FIXME: This duplicates code from the driver that sets the -target-abi
    // option - this code is used if -target-abi isn't passed and should
    // be unified in some way.
    if (Triple.isOSBinFormatMachO()) {
      // The backend is hardwired to assume AAPCS for M-class processors, ensure
      // the frontend matches that.
      if (Triple.getEnvironment() == llvm::Triple::EABI ||
          Triple.getOS() == llvm::Triple::UnknownOS ||
          ArchProfile == llvm::ARM::PK_M) {
        setABI("aapcs");
      } else if (Triple.isWatchABI()) {
        setABI("aapcs16");
      } else {
        setABI("apcs-gnu");
      }
    } else if (Triple.isOSWindows()) {
      // FIXME: this is invalid for WindowsCE
      setABI("aapcs");
    } else {
      // Select the default based on the platform.
      switch (Triple.getEnvironment()) {
      case llvm::Triple::Android:
      case llvm::Triple::GNUEABI:
      case llvm::Triple::GNUEABIHF:
      case llvm::Triple::MuslEABI:
      case llvm::Triple::MuslEABIHF:
        setABI("aapcs-linux");
        break;
      case llvm::Triple::EABIHF:
      case llvm::Triple::EABI:
        setABI("aapcs");
        break;
      case llvm::Triple::GNU:
        setABI("apcs-gnu");
      break;
      default:
        if (Triple.getOS() == llvm::Triple::NetBSD)
          setABI("apcs-gnu");
        else if (Triple.getOS() == llvm::Triple::OpenBSD)
          setABI("aapcs-linux");
        else
          setABI("aapcs");
        break;
      }
    }

    // ARM targets default to using the ARM C++ ABI.
    TheCXXABI.set(TargetCXXABI::GenericARM);

    // ARM has atomics up to 8 bytes
    setAtomic();

    // Maximum alignment for ARM NEON data types should be 64-bits (AAPCS)
    if (IsAAPCS && (Triple.getEnvironment() != llvm::Triple::Android))
       MaxVectorAlign = 64;

    // Do force alignment of members that follow zero length bitfields.  If
    // the alignment of the zero-length bitfield is greater than the member
    // that follows it, `bar', `bar' will be aligned as the  type of the
    // zero length bitfield.
    UseZeroLengthBitfieldAlignment = true;

    if (Triple.getOS() == llvm::Triple::Linux ||
        Triple.getOS() == llvm::Triple::UnknownOS)
      this->MCountName =
          Opts.EABIVersion == llvm::EABI::GNU ? "\01__gnu_mcount_nc" : "\01mcount";
  }

  StringRef getABI() const override { return ABI; }

  bool setABI(const std::string &Name) override {
    ABI = Name;

    // The defaults (above) are for AAPCS, check if we need to change them.
    //
    // FIXME: We need support for -meabi... we could just mangle it into the
    // name.
    if (Name == "apcs-gnu" || Name == "aapcs16") {
      setABIAPCS(Name == "aapcs16");
      return true;
    }
    if (Name == "aapcs" || Name == "aapcs-vfp" || Name == "aapcs-linux") {
      setABIAAPCS();
      return true;
    }
    return false;
  }

  // FIXME: This should be based on Arch attributes, not CPU names.
  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override {

    std::vector<StringRef> TargetFeatures;
    unsigned Arch = llvm::ARM::parseArch(getTriple().getArchName());

    // get default FPU features
    unsigned FPUKind = llvm::ARM::getDefaultFPU(CPU, Arch);
    llvm::ARM::getFPUFeatures(FPUKind, TargetFeatures);

    // get default Extension features
    unsigned Extensions = llvm::ARM::getDefaultExtensions(CPU, Arch);
    llvm::ARM::getExtensionFeatures(Extensions, TargetFeatures);

    for (auto Feature : TargetFeatures)
      if (Feature[0] == '+')
        Features[Feature.drop_front(1)] = true;

    // Enable or disable thumb-mode explicitly per function to enable mixed
    // ARM and Thumb code generation.
    if (isThumb())
      Features["thumb-mode"] = true;
    else
      Features["thumb-mode"] = false;

    // Convert user-provided arm and thumb GNU target attributes to
    // [-|+]thumb-mode target features respectively.
    std::vector<std::string> UpdatedFeaturesVec(FeaturesVec);
    for (auto &Feature : UpdatedFeaturesVec) {
      if (Feature.compare("+arm") == 0)
        Feature = "-thumb-mode";
      else if (Feature.compare("+thumb") == 0)
        Feature = "+thumb-mode";
    }

    return TargetInfo::initFeatureMap(Features, Diags, CPU, UpdatedFeaturesVec);
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    FPU = 0;
    CRC = 0;
    Crypto = 0;
    DSP = 0;
    Unaligned = 1;
    SoftFloat = SoftFloatABI = false;
    HWDiv = 0;

    // This does not diagnose illegal cases like having both
    // "+vfpv2" and "+vfpv3" or having "+neon" and "+fp-only-sp".
    uint32_t HW_FP_remove = 0;
    for (const auto &Feature : Features) {
      if (Feature == "+soft-float") {
        SoftFloat = true;
      } else if (Feature == "+soft-float-abi") {
        SoftFloatABI = true;
      } else if (Feature == "+vfp2") {
        FPU |= VFP2FPU;
        HW_FP |= HW_FP_SP | HW_FP_DP;
      } else if (Feature == "+vfp3") {
        FPU |= VFP3FPU;
        HW_FP |= HW_FP_SP | HW_FP_DP;
      } else if (Feature == "+vfp4") {
        FPU |= VFP4FPU;
        HW_FP |= HW_FP_SP | HW_FP_DP | HW_FP_HP;
      } else if (Feature == "+fp-armv8") {
        FPU |= FPARMV8;
        HW_FP |= HW_FP_SP | HW_FP_DP | HW_FP_HP;
      } else if (Feature == "+neon") {
        FPU |= NeonFPU;
        HW_FP |= HW_FP_SP | HW_FP_DP;
      } else if (Feature == "+hwdiv") {
        HWDiv |= HWDivThumb;
      } else if (Feature == "+hwdiv-arm") {
        HWDiv |= HWDivARM;
      } else if (Feature == "+crc") {
        CRC = 1;
      } else if (Feature == "+crypto") {
        Crypto = 1;
      } else if (Feature == "+dsp") {
        DSP = 1;
      } else if (Feature == "+fp-only-sp") {
        HW_FP_remove |= HW_FP_DP;
      } else if (Feature == "+strict-align") {
        Unaligned = 0;
      } else if (Feature == "+fp16") {
        HW_FP |= HW_FP_HP;
      }
    }
    HW_FP &= ~HW_FP_remove;

    switch (ArchVersion) {
    case 6:
      if (ArchProfile == llvm::ARM::PK_M)
        LDREX = 0;
      else if (ArchKind == llvm::ARM::AK_ARMV6K)
        LDREX = LDREX_D | LDREX_W | LDREX_H | LDREX_B ;
      else
        LDREX = LDREX_W;
      break;
    case 7:
      if (ArchProfile == llvm::ARM::PK_M)
        LDREX = LDREX_W | LDREX_H | LDREX_B ;
      else
        LDREX = LDREX_D | LDREX_W | LDREX_H | LDREX_B ;
      break;
    case 8:
      LDREX = LDREX_D | LDREX_W | LDREX_H | LDREX_B ;
    }

    if (!(FPU & NeonFPU) && FPMath == FP_Neon) {
      Diags.Report(diag::err_target_unsupported_fpmath) << "neon";
      return false;
    }

    if (FPMath == FP_Neon)
      Features.push_back("+neonfp");
    else if (FPMath == FP_VFP)
      Features.push_back("-neonfp");

    // Remove front-end specific options which the backend handles differently.
    auto Feature =
        std::find(Features.begin(), Features.end(), "+soft-float-abi");
    if (Feature != Features.end())
      Features.erase(Feature);

    return true;
  }

  bool hasFeature(StringRef Feature) const override {
    return llvm::StringSwitch<bool>(Feature)
        .Case("arm", true)
        .Case("aarch32", true)
        .Case("softfloat", SoftFloat)
        .Case("thumb", isThumb())
        .Case("neon", (FPU & NeonFPU) && !SoftFloat)
        .Case("vfp", FPU && !SoftFloat)
        .Case("hwdiv", HWDiv & HWDivThumb)
        .Case("hwdiv-arm", HWDiv & HWDivARM)
        .Default(false);
  }

  bool setCPU(const std::string &Name) override {
    if (Name != "generic")
      setArchInfo(llvm::ARM::parseCPUArch(Name));

    if (ArchKind == llvm::ARM::AK_INVALID)
      return false;
    setAtomic();
    CPU = Name;
    return true;
  }

  bool setFPMath(StringRef Name) override;

  void getTargetDefinesARMV81A(const LangOptions &Opts,
                               MacroBuilder &Builder) const {
    Builder.defineMacro("__ARM_FEATURE_QRDMX", "1");
  }

  void getTargetDefinesARMV82A(const LangOptions &Opts,
                               MacroBuilder &Builder) const {
    // Also include the ARMv8.1-A defines
    getTargetDefinesARMV81A(Opts, Builder);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    // Target identification.
    Builder.defineMacro("__arm");
    Builder.defineMacro("__arm__");
    // For bare-metal none-eabi.
    if (getTriple().getOS() == llvm::Triple::UnknownOS &&
        (getTriple().getEnvironment() == llvm::Triple::EABI ||
         getTriple().getEnvironment() == llvm::Triple::EABIHF))
      Builder.defineMacro("__ELF__");


    // Target properties.
    Builder.defineMacro("__REGISTER_PREFIX__", "");

    // Unfortunately, __ARM_ARCH_7K__ is now more of an ABI descriptor. The CPU
    // happens to be Cortex-A7 though, so it should still get __ARM_ARCH_7A__.
    if (getTriple().isWatchABI())
      Builder.defineMacro("__ARM_ARCH_7K__", "2");

    if (!CPUAttr.empty())
      Builder.defineMacro("__ARM_ARCH_" + CPUAttr + "__");

    // ACLE 6.4.1 ARM/Thumb instruction set architecture
    // __ARM_ARCH is defined as an integer value indicating the current ARM ISA
    Builder.defineMacro("__ARM_ARCH", Twine(ArchVersion));

    if (ArchVersion >= 8) {
      // ACLE 6.5.7 Crypto Extension
      if (Crypto)
        Builder.defineMacro("__ARM_FEATURE_CRYPTO", "1");
      // ACLE 6.5.8 CRC32 Extension
      if (CRC)
        Builder.defineMacro("__ARM_FEATURE_CRC32", "1");
      // ACLE 6.5.10 Numeric Maximum and Minimum
      Builder.defineMacro("__ARM_FEATURE_NUMERIC_MAXMIN", "1");
      // ACLE 6.5.9 Directed Rounding
      Builder.defineMacro("__ARM_FEATURE_DIRECTED_ROUNDING", "1");
    }

    // __ARM_ARCH_ISA_ARM is defined to 1 if the core supports the ARM ISA.  It
    // is not defined for the M-profile.
    // NOTE that the default profile is assumed to be 'A'
    if (CPUProfile.empty() || ArchProfile != llvm::ARM::PK_M)
      Builder.defineMacro("__ARM_ARCH_ISA_ARM", "1");

    // __ARM_ARCH_ISA_THUMB is defined to 1 if the core supports the original
    // Thumb ISA (including v6-M and v8-M Baseline).  It is set to 2 if the
    // core supports the Thumb-2 ISA as found in the v6T2 architecture and all
    // v7 and v8 architectures excluding v8-M Baseline.
    if (supportsThumb2())
      Builder.defineMacro("__ARM_ARCH_ISA_THUMB", "2");
    else if (supportsThumb())
      Builder.defineMacro("__ARM_ARCH_ISA_THUMB", "1");

    // __ARM_32BIT_STATE is defined to 1 if code is being generated for a 32-bit
    // instruction set such as ARM or Thumb.
    Builder.defineMacro("__ARM_32BIT_STATE", "1");

    // ACLE 6.4.2 Architectural Profile (A, R, M or pre-Cortex)

    // __ARM_ARCH_PROFILE is defined as 'A', 'R', 'M' or 'S', or unset.
    if (!CPUProfile.empty())
      Builder.defineMacro("__ARM_ARCH_PROFILE", "'" + CPUProfile + "'");

    // ACLE 6.4.3 Unaligned access supported in hardware
    if (Unaligned)
      Builder.defineMacro("__ARM_FEATURE_UNALIGNED", "1");

    // ACLE 6.4.4 LDREX/STREX
    if (LDREX)
      Builder.defineMacro("__ARM_FEATURE_LDREX", "0x" + llvm::utohexstr(LDREX));

    // ACLE 6.4.5 CLZ
    if (ArchVersion == 5 ||
       (ArchVersion == 6 && CPUProfile != "M") ||
        ArchVersion >  6)
      Builder.defineMacro("__ARM_FEATURE_CLZ", "1");

    // ACLE 6.5.1 Hardware Floating Point
    if (HW_FP)
      Builder.defineMacro("__ARM_FP", "0x" + llvm::utohexstr(HW_FP));

    // ACLE predefines.
    Builder.defineMacro("__ARM_ACLE", "200");

    // FP16 support (we currently only support IEEE format).
    Builder.defineMacro("__ARM_FP16_FORMAT_IEEE", "1");
    Builder.defineMacro("__ARM_FP16_ARGS", "1");

    // ACLE 6.5.3 Fused multiply-accumulate (FMA)
    if (ArchVersion >= 7 && (FPU & VFP4FPU))
      Builder.defineMacro("__ARM_FEATURE_FMA", "1");

    // Subtarget options.

    // FIXME: It's more complicated than this and we don't really support
    // interworking.
    // Windows on ARM does not "support" interworking
    if (5 <= ArchVersion && ArchVersion <= 8 && !getTriple().isOSWindows())
      Builder.defineMacro("__THUMB_INTERWORK__");

    if (ABI == "aapcs" || ABI == "aapcs-linux" || ABI == "aapcs-vfp") {
      // Embedded targets on Darwin follow AAPCS, but not EABI.
      // Windows on ARM follows AAPCS VFP, but does not conform to EABI.
      if (!getTriple().isOSBinFormatMachO() && !getTriple().isOSWindows())
        Builder.defineMacro("__ARM_EABI__");
      Builder.defineMacro("__ARM_PCS", "1");
    }

    if ((!SoftFloat && !SoftFloatABI) || ABI == "aapcs-vfp" ||
        ABI == "aapcs16")
      Builder.defineMacro("__ARM_PCS_VFP", "1");

    if (SoftFloat)
      Builder.defineMacro("__SOFTFP__");

    if (ArchKind == llvm::ARM::AK_XSCALE)
      Builder.defineMacro("__XSCALE__");

    if (isThumb()) {
      Builder.defineMacro("__THUMBEL__");
      Builder.defineMacro("__thumb__");
      if (supportsThumb2())
        Builder.defineMacro("__thumb2__");
    }

    // ACLE 6.4.9 32-bit SIMD instructions
    if (ArchVersion >= 6 && (CPUProfile != "M" || CPUAttr == "7EM"))
      Builder.defineMacro("__ARM_FEATURE_SIMD32", "1");

    // ACLE 6.4.10 Hardware Integer Divide
    if (((HWDiv & HWDivThumb) && isThumb()) ||
        ((HWDiv & HWDivARM) && !isThumb())) {
      Builder.defineMacro("__ARM_FEATURE_IDIV", "1");
      Builder.defineMacro("__ARM_ARCH_EXT_IDIV__", "1");
    }

    // Note, this is always on in gcc, even though it doesn't make sense.
    Builder.defineMacro("__APCS_32__");

    if (FPUModeIsVFP((FPUMode) FPU)) {
      Builder.defineMacro("__VFP_FP__");
      if (FPU & VFP2FPU)
        Builder.defineMacro("__ARM_VFPV2__");
      if (FPU & VFP3FPU)
        Builder.defineMacro("__ARM_VFPV3__");
      if (FPU & VFP4FPU)
        Builder.defineMacro("__ARM_VFPV4__");
      if (FPU & FPARMV8)
        Builder.defineMacro("__ARM_FPV5__");
    }

    // This only gets set when Neon instructions are actually available, unlike
    // the VFP define, hence the soft float and arch check. This is subtly
    // different from gcc, we follow the intent which was that it should be set
    // when Neon instructions are actually available.
    if ((FPU & NeonFPU) && !SoftFloat && ArchVersion >= 7) {
      Builder.defineMacro("__ARM_NEON", "1");
      Builder.defineMacro("__ARM_NEON__");
      // current AArch32 NEON implementations do not support double-precision
      // floating-point even when it is present in VFP.
      Builder.defineMacro("__ARM_NEON_FP",
                          "0x" + llvm::utohexstr(HW_FP & ~HW_FP_DP));
    }

    Builder.defineMacro("__ARM_SIZEOF_WCHAR_T",
                        Opts.ShortWChar ? "2" : "4");

    Builder.defineMacro("__ARM_SIZEOF_MINIMAL_ENUM",
                        Opts.ShortEnums ? "1" : "4");

    if (ArchVersion >= 6 && CPUAttr != "6M" && CPUAttr != "8M_BASE") {
      Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
      Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
      Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
      Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
    }

    // ACLE 6.4.7 DSP instructions
    if (DSP) {
      Builder.defineMacro("__ARM_FEATURE_DSP", "1");
    }

    // ACLE 6.4.8 Saturation instructions
    bool SAT = false;
    if ((ArchVersion == 6 && CPUProfile != "M") || ArchVersion > 6 ) {
      Builder.defineMacro("__ARM_FEATURE_SAT", "1");
      SAT = true;
    }

    // ACLE 6.4.6 Q (saturation) flag
    if (DSP || SAT)
      Builder.defineMacro("__ARM_FEATURE_QBIT", "1");

    if (Opts.UnsafeFPMath)
      Builder.defineMacro("__ARM_FP_FAST", "1");

    switch(ArchKind) {
    default: break;
    case llvm::ARM::AK_ARMV8_1A:
      getTargetDefinesARMV81A(Opts, Builder);
      break;
    case llvm::ARM::AK_ARMV8_2A:
      getTargetDefinesARMV82A(Opts, Builder);
      break;
    }
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                             clang::ARM::LastTSBuiltin-Builtin::FirstTSBuiltin);
  }
  bool isCLZForZeroUndef() const override { return false; }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return IsAAPCS
               ? AAPCSABIBuiltinVaList
               : (getTriple().isWatchABI() ? TargetInfo::CharPtrBuiltinVaList
                                           : TargetInfo::VoidPtrBuiltinVaList);
  }
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default: break;
    case 'l': // r0-r7
    case 'h': // r8-r15
    case 't': // VFP Floating point register single precision
    case 'w': // VFP Floating point register double precision
      Info.setAllowsRegister();
      return true;
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
      // FIXME
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
  std::string convertConstraint(const char *&Constraint) const override {
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
  bool
  validateConstraintModifier(StringRef Constraint, char Modifier, unsigned Size,
                             std::string &SuggestedModifier) const override {
    bool isOutput = (Constraint[0] == '=');
    bool isInOut = (Constraint[0] == '+');

    // Strip off constraint modifiers.
    while (Constraint[0] == '=' ||
           Constraint[0] == '+' ||
           Constraint[0] == '&')
      Constraint = Constraint.substr(1);

    switch (Constraint[0]) {
    default: break;
    case 'r': {
      switch (Modifier) {
      default:
        return (isInOut || isOutput || Size <= 64);
      case 'q':
        // A register of size 32 cannot fit a vector type.
        return false;
      }
    }
    }

    return true;
  }
  const char *getClobbers() const override {
    // FIXME: Is this really right?
    return "";
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_AAPCS:
    case CC_AAPCS_VFP:
    case CC_Swift:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0) return 0;
    if (RegNo == 1) return 1;
    return -1;
  }

  bool hasSjLjLowering() const override {
    return true;
  }
};

bool ARMTargetInfo::setFPMath(StringRef Name) {
  if (Name == "neon") {
    FPMath = FP_Neon;
    return true;
  } else if (Name == "vfp" || Name == "vfp2" || Name == "vfp3" ||
             Name == "vfp4") {
    FPMath = FP_VFP;
    return true;
  }
  return false;
}

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

ArrayRef<const char *> ARMTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
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

ArrayRef<TargetInfo::GCCRegAlias> ARMTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

const Builtin::Info ARMTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsNEON.def"

#define BUILTIN(ID, TYPE, ATTRS) \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LANGBUILTIN(ID, TYPE, ATTRS, LANG) \
  { #ID, TYPE, ATTRS, nullptr, LANG, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#define TARGET_HEADER_BUILTIN(ID, TYPE, ATTRS, HEADER, LANGS, FEATURE) \
  { #ID, TYPE, ATTRS, HEADER, LANGS, FEATURE },
#include "clang/Basic/BuiltinsARM.def"
};

class ARMleTargetInfo : public ARMTargetInfo {
public:
  ARMleTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : ARMTargetInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__ARMEL__");
    ARMTargetInfo::getTargetDefines(Opts, Builder);
  }
};

class ARMbeTargetInfo : public ARMTargetInfo {
public:
  ARMbeTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : ARMTargetInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__ARMEB__");
    Builder.defineMacro("__ARM_BIG_ENDIAN");
    ARMTargetInfo::getTargetDefines(Opts, Builder);
  }
};

class WindowsARMTargetInfo : public WindowsTargetInfo<ARMleTargetInfo> {
  const llvm::Triple Triple;
public:
  WindowsARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsTargetInfo<ARMleTargetInfo>(Triple, Opts), Triple(Triple) {
    WCharType = UnsignedShort;
    SizeType = UnsignedInt;
  }
  void getVisualStudioDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    WindowsTargetInfo<ARMleTargetInfo>::getVisualStudioDefines(Opts, Builder);

    // FIXME: this is invalid for WindowsCE
    Builder.defineMacro("_M_ARM_NT", "1");
    Builder.defineMacro("_M_ARMT", "_M_ARM");
    Builder.defineMacro("_M_THUMB", "_M_ARM");

    assert((Triple.getArch() == llvm::Triple::arm ||
            Triple.getArch() == llvm::Triple::thumb) &&
           "invalid architecture for Windows ARM target info");
    unsigned Offset = Triple.getArch() == llvm::Triple::arm ? 4 : 6;
    Builder.defineMacro("_M_ARM", Triple.getArchName().substr(Offset));

    // TODO map the complete set of values
    // 31: VFPv3 40: VFPv4
    Builder.defineMacro("_M_ARM_FP", "31");
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_X86StdCall:
    case CC_X86ThisCall:
    case CC_X86FastCall:
    case CC_X86VectorCall:
      return CCCR_Ignore;
    case CC_C:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }
};

// Windows ARM + Itanium C++ ABI Target
class ItaniumWindowsARMleTargetInfo : public WindowsARMTargetInfo {
public:
  ItaniumWindowsARMleTargetInfo(const llvm::Triple &Triple,
                                const TargetOptions &Opts)
      : WindowsARMTargetInfo(Triple, Opts) {
    TheCXXABI.set(TargetCXXABI::GenericARM);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsARMTargetInfo::getTargetDefines(Opts, Builder);

    if (Opts.MSVCCompat)
      WindowsARMTargetInfo::getVisualStudioDefines(Opts, Builder);
  }
};

// Windows ARM, MS (C++) ABI
class MicrosoftARMleTargetInfo : public WindowsARMTargetInfo {
public:
  MicrosoftARMleTargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts)
      : WindowsARMTargetInfo(Triple, Opts) {
    TheCXXABI.set(TargetCXXABI::Microsoft);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsARMTargetInfo::getTargetDefines(Opts, Builder);
    WindowsARMTargetInfo::getVisualStudioDefines(Opts, Builder);
  }
};

// ARM MinGW target
class MinGWARMTargetInfo : public WindowsARMTargetInfo {
public:
  MinGWARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsARMTargetInfo(Triple, Opts) {
    TheCXXABI.set(TargetCXXABI::GenericARM);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsARMTargetInfo::getTargetDefines(Opts, Builder);
    DefineStd(Builder, "WIN32", Opts);
    DefineStd(Builder, "WINNT", Opts);
    Builder.defineMacro("_ARM_");
    addMinGWDefines(Opts, Builder);
  }
};

// ARM Cygwin target
class CygwinARMTargetInfo : public ARMleTargetInfo {
public:
  CygwinARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : ARMleTargetInfo(Triple, Opts) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    DoubleAlign = LongLongAlign = 64;
    resetDataLayout("e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    ARMleTargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("_ARM_");
    Builder.defineMacro("__CYGWIN__");
    Builder.defineMacro("__CYGWIN32__");
    DefineStd(Builder, "unix", Opts);
    if (Opts.CPlusPlus)
      Builder.defineMacro("_GNU_SOURCE");
  }
};

class DarwinARMTargetInfo : public DarwinTargetInfo<ARMleTargetInfo> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    getDarwinDefines(Builder, Opts, Triple, PlatformName, PlatformMinVersion);
  }

public:
  DarwinARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<ARMleTargetInfo>(Triple, Opts) {
    HasAlignMac68kSupport = true;
    // iOS always has 64-bit atomic instructions.
    // FIXME: This should be based off of the target features in
    // ARMleTargetInfo.
    MaxAtomicInlineWidth = 64;

    if (Triple.isWatchABI()) {
      // Darwin on iOS uses a variant of the ARM C++ ABI.
      TheCXXABI.set(TargetCXXABI::WatchOS);

      // The 32-bit ABI is silent on what ptrdiff_t should be, but given that
      // size_t is long, it's a bit weird for it to be int.
      PtrDiffType = SignedLong;

      // BOOL should be a real boolean on the new ABI
      UseSignedCharForObjCBool = false;
    } else
      TheCXXABI.set(TargetCXXABI::iOS);
  }
};

class AArch64TargetInfo : public TargetInfo {
  virtual void setDataLayout() = 0;
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char *const GCCRegNames[];

  enum FPUModeEnum {
    FPUMode,
    NeonMode = (1 << 0),
    SveMode = (1 << 1)
  };

  unsigned FPU;
  unsigned CRC;
  unsigned Crypto;
  unsigned Unaligned;
  unsigned HasFullFP16;
  llvm::AArch64::ArchKind ArchKind;

  static const Builtin::Info BuiltinInfo[];

  std::string ABI;

public:
  AArch64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TargetInfo(Triple), ABI("aapcs") {
    if (getTriple().getOS() == llvm::Triple::NetBSD ||
        getTriple().getOS() == llvm::Triple::OpenBSD) {
      WCharType = SignedInt;

      // NetBSD apparently prefers consistency across ARM targets to consistency
      // across 64-bit targets.
      Int64Type = SignedLongLong;
      IntMaxType = SignedLongLong;
    } else {
      WCharType = UnsignedInt;
      Int64Type = SignedLong;
      IntMaxType = SignedLong;
    }

    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    MaxVectorAlign = 128;
    MaxAtomicInlineWidth = 128;
    MaxAtomicPromoteWidth = 128;

    LongDoubleWidth = LongDoubleAlign = SuitableAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();

    // Make __builtin_ms_va_list available.
    HasBuiltinMSVaList = true;

    // {} in inline assembly are neon specifiers, not assembly variant
    // specifiers.
    NoAsmVariants = true;

    // AAPCS gives rules for bitfields. 7.1.7 says: "The container type
    // contributes to the alignment of the containing aggregate in the same way
    // a plain (non bit-field) member of that type would, without exception for
    // zero-sized or anonymous bit-fields."
    assert(UseBitFieldTypeAlignment && "bitfields affect type alignment");
    UseZeroLengthBitfieldAlignment = true;

    // AArch64 targets default to using the ARM C++ ABI.
    TheCXXABI.set(TargetCXXABI::GenericAArch64);

    if (Triple.getOS() == llvm::Triple::Linux)
      this->MCountName = "\01_mcount";
    else if (Triple.getOS() == llvm::Triple::UnknownOS)
      this->MCountName = Opts.EABIVersion == llvm::EABI::GNU ? "\01_mcount" : "mcount";
  }

  StringRef getABI() const override { return ABI; }
  bool setABI(const std::string &Name) override {
    if (Name != "aapcs" && Name != "darwinpcs")
      return false;

    ABI = Name;
    return true;
  }

  bool setCPU(const std::string &Name) override {
    return Name == "generic" ||
           llvm::AArch64::parseCPUArch(Name) !=
           static_cast<unsigned>(llvm::AArch64::ArchKind::AK_INVALID);
  }

  void getTargetDefinesARMV81A(const LangOptions &Opts,
                        MacroBuilder &Builder) const {
    Builder.defineMacro("__ARM_FEATURE_QRDMX", "1");
  }

  void getTargetDefinesARMV82A(const LangOptions &Opts,
                        MacroBuilder &Builder) const {
    // Also include the ARMv8.1 defines
    getTargetDefinesARMV81A(Opts, Builder);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    // Target identification.
    Builder.defineMacro("__aarch64__");
    // For bare-metal none-eabi.
    if (getTriple().getOS() == llvm::Triple::UnknownOS &&
        (getTriple().getEnvironment() == llvm::Triple::EABI ||
         getTriple().getEnvironment() == llvm::Triple::EABIHF))
      Builder.defineMacro("__ELF__");

    // Target properties.
    Builder.defineMacro("_LP64");
    Builder.defineMacro("__LP64__");

    // ACLE predefines. Many can only have one possible value on v8 AArch64.
    Builder.defineMacro("__ARM_ACLE", "200");
    Builder.defineMacro("__ARM_ARCH", "8");
    Builder.defineMacro("__ARM_ARCH_PROFILE", "'A'");

    Builder.defineMacro("__ARM_64BIT_STATE", "1");
    Builder.defineMacro("__ARM_PCS_AAPCS64", "1");
    Builder.defineMacro("__ARM_ARCH_ISA_A64", "1");

    Builder.defineMacro("__ARM_FEATURE_CLZ", "1");
    Builder.defineMacro("__ARM_FEATURE_FMA", "1");
    Builder.defineMacro("__ARM_FEATURE_LDREX", "0xF");
    Builder.defineMacro("__ARM_FEATURE_IDIV", "1"); // As specified in ACLE
    Builder.defineMacro("__ARM_FEATURE_DIV");  // For backwards compatibility
    Builder.defineMacro("__ARM_FEATURE_NUMERIC_MAXMIN", "1");
    Builder.defineMacro("__ARM_FEATURE_DIRECTED_ROUNDING", "1");

    Builder.defineMacro("__ARM_ALIGN_MAX_STACK_PWR", "4");

    // 0xe implies support for half, single and double precision operations.
    Builder.defineMacro("__ARM_FP", "0xE");

    // PCS specifies this for SysV variants, which is all we support. Other ABIs
    // may choose __ARM_FP16_FORMAT_ALTERNATIVE.
    Builder.defineMacro("__ARM_FP16_FORMAT_IEEE", "1");
    Builder.defineMacro("__ARM_FP16_ARGS", "1");

    if (Opts.UnsafeFPMath)
      Builder.defineMacro("__ARM_FP_FAST", "1");

    Builder.defineMacro("__ARM_SIZEOF_WCHAR_T", Opts.ShortWChar ? "2" : "4");

    Builder.defineMacro("__ARM_SIZEOF_MINIMAL_ENUM",
                        Opts.ShortEnums ? "1" : "4");

    if (FPU & NeonMode) {
      Builder.defineMacro("__ARM_NEON", "1");
      // 64-bit NEON supports half, single and double precision operations.
      Builder.defineMacro("__ARM_NEON_FP", "0xE");
    }

    if (FPU & SveMode)
      Builder.defineMacro("__ARM_FEATURE_SVE", "1");

    if (CRC)
      Builder.defineMacro("__ARM_FEATURE_CRC32", "1");

    if (Crypto)
      Builder.defineMacro("__ARM_FEATURE_CRYPTO", "1");

    if (Unaligned)
      Builder.defineMacro("__ARM_FEATURE_UNALIGNED", "1");

    switch(ArchKind) {
    default: break;
    case llvm::AArch64::ArchKind::AK_ARMV8_1A:
      getTargetDefinesARMV81A(Opts, Builder);
      break;
    case llvm::AArch64::ArchKind::AK_ARMV8_2A:
      getTargetDefinesARMV82A(Opts, Builder);
      break;
    }

    // All of the __sync_(bool|val)_compare_and_swap_(1|2|4|8) builtins work.
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                       clang::AArch64::LastTSBuiltin - Builtin::FirstTSBuiltin);
  }

  bool hasFeature(StringRef Feature) const override {
    return Feature == "aarch64" ||
      Feature == "arm64" ||
      Feature == "arm" ||
      (Feature == "neon" && (FPU & NeonMode)) ||
      (Feature == "sve" && (FPU & SveMode));
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    FPU = FPUMode;
    CRC = 0;
    Crypto = 0;
    Unaligned = 1;
    HasFullFP16 = 0;
    ArchKind = llvm::AArch64::ArchKind::AK_ARMV8A;

    for (const auto &Feature : Features) {
      if (Feature == "+neon")
        FPU |= NeonMode;
      if (Feature == "+sve")
        FPU |= SveMode;
      if (Feature == "+crc")
        CRC = 1;
      if (Feature == "+crypto")
        Crypto = 1;
      if (Feature == "+strict-align")
        Unaligned = 0;
      if (Feature == "+v8.1a")
        ArchKind = llvm::AArch64::ArchKind::AK_ARMV8_1A;
      if (Feature == "+v8.2a")
        ArchKind = llvm::AArch64::ArchKind::AK_ARMV8_2A;
      if (Feature == "+fullfp16")
        HasFullFP16 = 1;
    }

    setDataLayout();

    return true;
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_C:
    case CC_Swift:
    case CC_PreserveMost:
    case CC_PreserveAll:
    case CC_OpenCLKernel:
    case CC_Win64:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }

  bool isCLZForZeroUndef() const override { return false; }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::AArch64ABIBuiltinVaList;
  }

  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      return false;
    case 'w': // Floating point and SIMD registers (V0-V31)
      Info.setAllowsRegister();
      return true;
    case 'I': // Constant that can be used with an ADD instruction
    case 'J': // Constant that can be used with a SUB instruction
    case 'K': // Constant that can be used with a 32-bit logical instruction
    case 'L': // Constant that can be used with a 64-bit logical instruction
    case 'M': // Constant that can be used as a 32-bit MOV immediate
    case 'N': // Constant that can be used as a 64-bit MOV immediate
    case 'Y': // Floating point constant zero
    case 'Z': // Integer constant zero
      return true;
    case 'Q': // A memory reference with base register and no offset
      Info.setAllowsMemory();
      return true;
    case 'S': // A symbolic address
      Info.setAllowsRegister();
      return true;
    case 'U':
      // Ump: A memory address suitable for ldp/stp in SI, DI, SF and DF modes.
      // Utf: A memory address suitable for ldp/stp in TF mode.
      // Usa: An absolute symbolic address.
      // Ush: The high part (bits 32:12) of a pc-relative symbolic address.
      llvm_unreachable("FIXME: Unimplemented support for U* constraints.");
    case 'z': // Zero register, wzr or xzr
      Info.setAllowsRegister();
      return true;
    case 'x': // Floating point and SIMD registers (V0-V15)
      Info.setAllowsRegister();
      return true;
    }
    return false;
  }

  bool
  validateConstraintModifier(StringRef Constraint, char Modifier, unsigned Size,
                             std::string &SuggestedModifier) const override {
    // Strip off constraint modifiers.
    while (Constraint[0] == '=' || Constraint[0] == '+' || Constraint[0] == '&')
      Constraint = Constraint.substr(1);

    switch (Constraint[0]) {
    default:
      return true;
    case 'z':
    case 'r': {
      switch (Modifier) {
      case 'x':
      case 'w':
        // For now assume that the person knows what they're
        // doing with the modifier.
        return true;
      default:
        // By default an 'r' constraint will be in the 'x'
        // registers.
        if (Size == 64)
          return true;

        SuggestedModifier = "w";
        return false;
      }
    }
    }
  }

  const char *getClobbers() const override { return ""; }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0)
      return 0;
    if (RegNo == 1)
      return 1;
    return -1;
  }
};

const char *const AArch64TargetInfo::GCCRegNames[] = {
  // 32-bit Integer registers
  "w0",  "w1",  "w2",  "w3",  "w4",  "w5",  "w6",  "w7",  "w8",  "w9",  "w10",
  "w11", "w12", "w13", "w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21",
  "w22", "w23", "w24", "w25", "w26", "w27", "w28", "w29", "w30", "wsp",

  // 64-bit Integer registers
  "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",  "x8",  "x9",  "x10",
  "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21",
  "x22", "x23", "x24", "x25", "x26", "x27", "x28", "fp",  "lr",  "sp",

  // 32-bit floating point regsisters
  "s0",  "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",  "s8",  "s9",  "s10",
  "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
  "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",

  // 64-bit floating point regsisters
  "d0",  "d1",  "d2",  "d3",  "d4",  "d5",  "d6",  "d7",  "d8",  "d9",  "d10",
  "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
  "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",

  // Vector registers
  "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",  "v8",  "v9",  "v10",
  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
  "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
};

ArrayRef<const char *> AArch64TargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias AArch64TargetInfo::GCCRegAliases[] = {
  { { "w31" }, "wsp" },
  { { "x29" }, "fp" },
  { { "x30" }, "lr" },
  { { "x31" }, "sp" },
  // The S/D/Q and W/X registers overlap, but aren't really aliases; we
  // don't want to substitute one of these for a different-sized one.
};

ArrayRef<TargetInfo::GCCRegAlias> AArch64TargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

const Builtin::Info AArch64TargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsNEON.def"

#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsAArch64.def"
};

class AArch64leTargetInfo : public AArch64TargetInfo {
  void setDataLayout() override {
    if (getTriple().isOSBinFormatMachO())
      resetDataLayout("e-m:o-i64:64-i128:128-n32:64-S128");
    else
      resetDataLayout("e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  }

public:
  AArch64leTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : AArch64TargetInfo(Triple, Opts) {
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__AARCH64EL__");
    AArch64TargetInfo::getTargetDefines(Opts, Builder);
  }
};

class MicrosoftARM64TargetInfo
    : public WindowsTargetInfo<AArch64leTargetInfo> {
  const llvm::Triple Triple;

public:
  MicrosoftARM64TargetInfo(const llvm::Triple &Triple,
                             const TargetOptions &Opts)
      : WindowsTargetInfo<AArch64leTargetInfo>(Triple, Opts), Triple(Triple) {

    // This is an LLP64 platform.
    // int:4, long:4, long long:8, long double:8.
    WCharType = UnsignedShort;
    IntWidth = IntAlign = 32;
    LongWidth = LongAlign = 32;
    DoubleAlign = LongLongAlign = 64;
    LongDoubleWidth = LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    IntMaxType = SignedLongLong;
    Int64Type = SignedLongLong;
    SizeType = UnsignedLongLong;
    PtrDiffType = SignedLongLong;
    IntPtrType = SignedLongLong;

    TheCXXABI.set(TargetCXXABI::Microsoft);
  }

  void setDataLayout() override {
    resetDataLayout("e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128");
  }

  void getVisualStudioDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    WindowsTargetInfo<AArch64leTargetInfo>::getVisualStudioDefines(Opts,
                                                                   Builder);
    Builder.defineMacro("_WIN32", "1");
    Builder.defineMacro("_WIN64", "1");
    Builder.defineMacro("_M_ARM64", "1");
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsTargetInfo::getTargetDefines(Opts, Builder);
    getVisualStudioDefines(Opts, Builder);
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
};

class AArch64beTargetInfo : public AArch64TargetInfo {
  void setDataLayout() override {
    assert(!getTriple().isOSBinFormatMachO());
    resetDataLayout("E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  }

public:
  AArch64beTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : AArch64TargetInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__AARCH64EB__");
    Builder.defineMacro("__AARCH_BIG_ENDIAN");
    Builder.defineMacro("__ARM_BIG_ENDIAN");
    AArch64TargetInfo::getTargetDefines(Opts, Builder);
  }
};

class DarwinAArch64TargetInfo : public DarwinTargetInfo<AArch64leTargetInfo> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    Builder.defineMacro("__AARCH64_SIMD__");
    Builder.defineMacro("__ARM64_ARCH_8__");
    Builder.defineMacro("__ARM_NEON__");
    Builder.defineMacro("__LITTLE_ENDIAN__");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
    Builder.defineMacro("__arm64", "1");
    Builder.defineMacro("__arm64__", "1");

    getDarwinDefines(Builder, Opts, Triple, PlatformName, PlatformMinVersion);
  }

public:
  DarwinAArch64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<AArch64leTargetInfo>(Triple, Opts) {
    Int64Type = SignedLongLong;
    WCharType = SignedInt;
    UseSignedCharForObjCBool = false;

    LongDoubleWidth = LongDoubleAlign = SuitableAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();

    TheCXXABI.set(TargetCXXABI::iOS64);
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
};

// Hexagon abstract base class
class HexagonTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
  static const char * const GCCRegNames[];
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  std::string CPU;
  bool HasHVX, HasHVXDouble;
  bool UseLongCalls;

public:
  HexagonTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    // Specify the vector alignment explicitly. For v512x1, the calculated
    // alignment would be 512*alignment(i1), which is 512 bytes, instead of
    // the required minimum of 64 bytes.
    resetDataLayout("e-m:e-p:32:32:32-a:0-n16:32-"
        "i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-"
        "v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048");
    SizeType    = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType  = SignedInt;

    // {} in inline assembly are packet specifiers, not assembly variant
    // specifiers.
    NoAsmVariants = true;

    LargeArrayMinWidth = 64;
    LargeArrayAlign = 64;
    UseBitFieldTypeAlignment = true;
    ZeroLengthBitfieldBoundary = 32;
    HasHVX = HasHVXDouble = false;
    UseLongCalls = false;
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                         clang::Hexagon::LastTSBuiltin-Builtin::FirstTSBuiltin);
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
      case 'v':
      case 'q':
        if (HasHVX) {
          Info.setAllowsRegister();
          return true;
        }
        break;
      case 's':
        // Relocatable constant.
        return true;
    }
    return false;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool isCLZForZeroUndef() const override { return false; }

  bool hasFeature(StringRef Feature) const override {
    return llvm::StringSwitch<bool>(Feature)
      .Case("hexagon", true)
      .Case("hvx", HasHVX)
      .Case("hvx-double", HasHVXDouble)
      .Case("long-calls", UseLongCalls)
      .Default(false);
  }

  bool initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
        StringRef CPU, const std::vector<std::string> &FeaturesVec)
        const override;

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;

  void setFeatureEnabled(llvm::StringMap<bool> &Features, StringRef Name,
                         bool Enabled) const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  const char *getClobbers() const override {
    return "";
  }

  static const char *getHexagonCPUSuffix(StringRef Name) {
    return llvm::StringSwitch<const char*>(Name)
      .Case("hexagonv4", "4")
      .Case("hexagonv5", "5")
      .Case("hexagonv55", "55")
      .Case("hexagonv60", "60")
      .Case("hexagonv62", "62")
      .Default(nullptr);
  }

  bool setCPU(const std::string &Name) override {
    if (!getHexagonCPUSuffix(Name))
      return false;
    CPU = Name;
    return true;
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    return RegNo < 2 ? RegNo : -1;
  }
};

void HexagonTargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  Builder.defineMacro("__qdsp6__", "1");
  Builder.defineMacro("__hexagon__", "1");

  if (CPU == "hexagonv4") {
    Builder.defineMacro("__HEXAGON_V4__");
    Builder.defineMacro("__HEXAGON_ARCH__", "4");
    if (Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V4__");
      Builder.defineMacro("__QDSP6_ARCH__", "4");
    }
  } else if (CPU == "hexagonv5") {
    Builder.defineMacro("__HEXAGON_V5__");
    Builder.defineMacro("__HEXAGON_ARCH__", "5");
    if(Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V5__");
      Builder.defineMacro("__QDSP6_ARCH__", "5");
    }
  } else if (CPU == "hexagonv55") {
    Builder.defineMacro("__HEXAGON_V55__");
    Builder.defineMacro("__HEXAGON_ARCH__", "55");
    Builder.defineMacro("__QDSP6_V55__");
    Builder.defineMacro("__QDSP6_ARCH__", "55");
  } else if (CPU == "hexagonv60") {
    Builder.defineMacro("__HEXAGON_V60__");
    Builder.defineMacro("__HEXAGON_ARCH__", "60");
    Builder.defineMacro("__QDSP6_V60__");
    Builder.defineMacro("__QDSP6_ARCH__", "60");
  } else if (CPU == "hexagonv62") {
    Builder.defineMacro("__HEXAGON_V62__");
    Builder.defineMacro("__HEXAGON_ARCH__", "62");
  }

  if (hasFeature("hvx")) {
    Builder.defineMacro("__HVX__");
    if (hasFeature("hvx-double"))
      Builder.defineMacro("__HVXDBL__");
  }
}

bool HexagonTargetInfo::initFeatureMap(llvm::StringMap<bool> &Features,
      DiagnosticsEngine &Diags, StringRef CPU,
      const std::vector<std::string> &FeaturesVec) const {
  // Default for v60: -hvx, -hvx-double.
  Features["hvx"] = false;
  Features["hvx-double"] = false;
  Features["long-calls"] = false;

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

bool HexagonTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                             DiagnosticsEngine &Diags) {
  for (auto &F : Features) {
    if (F == "+hvx")
      HasHVX = true;
    else if (F == "-hvx")
      HasHVX = HasHVXDouble = false;
    else if (F == "+hvx-double")
      HasHVX = HasHVXDouble = true;
    else if (F == "-hvx-double")
      HasHVXDouble = false;

    if (F == "+long-calls")
      UseLongCalls = true;
    else if (F == "-long-calls")
      UseLongCalls = false;
  }
  return true;
}

void HexagonTargetInfo::setFeatureEnabled(llvm::StringMap<bool> &Features,
      StringRef Name, bool Enabled) const {
  if (Enabled) {
    if (Name == "hvx-double")
      Features["hvx"] = true;
  } else {
    if (Name == "hvx")
      Features["hvx-double"] = false;
  }
  Features[Name] = Enabled;
}

const char *const HexagonTargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
  "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",
  "p0", "p1", "p2", "p3",
  "sa0", "lc0", "sa1", "lc1", "m0", "m1", "usr", "ugp"
};

ArrayRef<const char*> HexagonTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias HexagonTargetInfo::GCCRegAliases[] = {
  { { "sp" }, "r29" },
  { { "fp" }, "r30" },
  { { "lr" }, "r31" },
};

ArrayRef<TargetInfo::GCCRegAlias> HexagonTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}


const Builtin::Info HexagonTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsHexagon.def"
};

class LanaiTargetInfo : public TargetInfo {
  // Class for Lanai (32-bit).
  // The CPU profiles supported by the Lanai backend
  enum CPUKind {
    CK_NONE,
    CK_V11,
  } CPU;

  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char *const GCCRegNames[];

public:
  LanaiTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    // Description string has to be kept in sync with backend.
    resetDataLayout("E"        // Big endian
                    "-m:e"     // ELF name manging
                    "-p:32:32" // 32 bit pointers, 32 bit aligned
                    "-i64:64"  // 64 bit integers, 64 bit aligned
                    "-a:0:32"  // 32 bit alignment of objects of aggregate type
                    "-n32"     // 32 bit native integer width
                    "-S64"     // 64 bit natural stack alignment
                    );

    // Setting RegParmMax equal to what mregparm was set to in the old
    // toolchain
    RegParmMax = 4;

    // Set the default CPU to V11
    CPU = CK_V11;

    // Temporary approach to make everything at least word-aligned and allow for
    // safely casting between pointers with different alignment requirements.
    // TODO: Remove this when there are no more cast align warnings on the
    // firmware.
    MinGlobalAlign = 32;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    // Define __lanai__ when building for target lanai.
    Builder.defineMacro("__lanai__");

    // Set define for the CPU specified.
    switch (CPU) {
    case CK_V11:
      Builder.defineMacro("__LANAI_V11__");
      break;
    case CK_NONE:
      llvm_unreachable("Unhandled target CPU");
    }
  }

  bool setCPU(const std::string &Name) override {
    CPU = llvm::StringSwitch<CPUKind>(Name)
              .Case("v11", CK_V11)
              .Default(CK_NONE);

    return CPU != CK_NONE;
  }

  bool hasFeature(StringRef Feature) const override {
    return llvm::StringSwitch<bool>(Feature).Case("lanai", true).Default(false);
  }

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override { return None; }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return false;
  }

  const char *getClobbers() const override { return ""; }
};

const char *const LanaiTargetInfo::GCCRegNames[] = {
    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",  "r9",  "r10",
    "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21",
    "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"};

ArrayRef<const char *> LanaiTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias LanaiTargetInfo::GCCRegAliases[] = {
    {{"pc"}, "r2"},
    {{"sp"}, "r4"},
    {{"fp"}, "r5"},
    {{"rv"}, "r8"},
    {{"rr1"}, "r10"},
    {{"rr2"}, "r11"},
    {{"rca"}, "r15"},
};

ArrayRef<TargetInfo::GCCRegAlias> LanaiTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

// Shared base class for SPARC v8 (32-bit) and SPARC v9 (64-bit).
class SparcTargetInfo : public TargetInfo {
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char * const GCCRegNames[];
  bool SoftFloat;
public:
  SparcTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple), SoftFloat(false) {}

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0) return 24;
    if (RegNo == 1) return 25;
    return -1;
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    // Check if software floating point is enabled
    auto Feature = std::find(Features.begin(), Features.end(), "+soft-float");
    if (Feature != Features.end()) {
      SoftFloat = true;
    }
    return true;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "sparc", Opts);
    Builder.defineMacro("__REGISTER_PREFIX__", "");

    if (SoftFloat)
      Builder.defineMacro("SOFT_FLOAT", "1");
  }

  bool hasFeature(StringRef Feature) const override {
    return llvm::StringSwitch<bool>(Feature)
             .Case("softfloat", SoftFloat)
             .Case("sparc", true)
             .Default(false);
  }

  bool hasSjLjLowering() const override {
    return true;
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    // FIXME: Implement!
    return None;
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    // FIXME: Implement!
    switch (*Name) {
    case 'I': // Signed 13-bit constant
    case 'J': // Zero
    case 'K': // 32-bit constant with the low 12 bits clear
    case 'L': // A constant in the range supported by movcc (11-bit signed imm)
    case 'M': // A constant in the range supported by movrcc (19-bit signed imm)
    case 'N': // Same as 'K' but zext (required for SIMode)
    case 'O': // The constant 4096
      return true;

    case 'f':
    case 'e':
      info.setAllowsRegister();
      return true;
    }
    return false;
  }
  const char *getClobbers() const override {
    // FIXME: Implement!
    return "";
  }

  // No Sparc V7 for now, the backend doesn't support it anyway.
  enum CPUKind {
    CK_GENERIC,
    CK_V8,
    CK_SUPERSPARC,
    CK_SPARCLITE,
    CK_F934,
    CK_HYPERSPARC,
    CK_SPARCLITE86X,
    CK_SPARCLET,
    CK_TSC701,
    CK_V9,
    CK_ULTRASPARC,
    CK_ULTRASPARC3,
    CK_NIAGARA,
    CK_NIAGARA2,
    CK_NIAGARA3,
    CK_NIAGARA4,
    CK_MYRIAD2100,
    CK_MYRIAD2150,
    CK_MYRIAD2450,
    CK_LEON2,
    CK_LEON2_AT697E,
    CK_LEON2_AT697F,
    CK_LEON3,
    CK_LEON3_UT699,
    CK_LEON3_GR712RC,
    CK_LEON4,
    CK_LEON4_GR740
  } CPU = CK_GENERIC;

  enum CPUGeneration {
    CG_V8,
    CG_V9,
  };

  CPUGeneration getCPUGeneration(CPUKind Kind) const {
    switch (Kind) {
    case CK_GENERIC:
    case CK_V8:
    case CK_SUPERSPARC:
    case CK_SPARCLITE:
    case CK_F934:
    case CK_HYPERSPARC:
    case CK_SPARCLITE86X:
    case CK_SPARCLET:
    case CK_TSC701:
    case CK_MYRIAD2100:
    case CK_MYRIAD2150:
    case CK_MYRIAD2450:
    case CK_LEON2:
    case CK_LEON2_AT697E:
    case CK_LEON2_AT697F:
    case CK_LEON3:
    case CK_LEON3_UT699:
    case CK_LEON3_GR712RC:
    case CK_LEON4:
    case CK_LEON4_GR740:
      return CG_V8;
    case CK_V9:
    case CK_ULTRASPARC:
    case CK_ULTRASPARC3:
    case CK_NIAGARA:
    case CK_NIAGARA2:
    case CK_NIAGARA3:
    case CK_NIAGARA4:
      return CG_V9;
    }
    llvm_unreachable("Unexpected CPU kind");
  }

  CPUKind getCPUKind(StringRef Name) const {
    return llvm::StringSwitch<CPUKind>(Name)
        .Case("v8", CK_V8)
        .Case("supersparc", CK_SUPERSPARC)
        .Case("sparclite", CK_SPARCLITE)
        .Case("f934", CK_F934)
        .Case("hypersparc", CK_HYPERSPARC)
        .Case("sparclite86x", CK_SPARCLITE86X)
        .Case("sparclet", CK_SPARCLET)
        .Case("tsc701", CK_TSC701)
        .Case("v9", CK_V9)
        .Case("ultrasparc", CK_ULTRASPARC)
        .Case("ultrasparc3", CK_ULTRASPARC3)
        .Case("niagara", CK_NIAGARA)
        .Case("niagara2", CK_NIAGARA2)
        .Case("niagara3", CK_NIAGARA3)
        .Case("niagara4", CK_NIAGARA4)
        .Case("ma2100", CK_MYRIAD2100)
        .Case("ma2150", CK_MYRIAD2150)
        .Case("ma2450", CK_MYRIAD2450)
        // FIXME: the myriad2[.n] spellings are obsolete,
        // but a grace period is needed to allow updating dependent builds.
        .Case("myriad2", CK_MYRIAD2100)
        .Case("myriad2.1", CK_MYRIAD2100)
        .Case("myriad2.2", CK_MYRIAD2150)
        .Case("leon2", CK_LEON2)
        .Case("at697e", CK_LEON2_AT697E)
        .Case("at697f", CK_LEON2_AT697F)
        .Case("leon3", CK_LEON3)
        .Case("ut699", CK_LEON3_UT699)
        .Case("gr712rc", CK_LEON3_GR712RC)
        .Case("leon4", CK_LEON4)
        .Case("gr740", CK_LEON4_GR740)
        .Default(CK_GENERIC);
  }

  bool setCPU(const std::string &Name) override {
    CPU = getCPUKind(Name);
    return CPU != CK_GENERIC;
  }
};

const char * const SparcTargetInfo::GCCRegNames[] = {
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
  "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"
};

ArrayRef<const char *> SparcTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias SparcTargetInfo::GCCRegAliases[] = {
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

ArrayRef<TargetInfo::GCCRegAlias> SparcTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

// SPARC v8 is the 32-bit mode selected by Triple::sparc.
class SparcV8TargetInfo : public SparcTargetInfo {
public:
  SparcV8TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : SparcTargetInfo(Triple, Opts) {
    resetDataLayout("E-m:e-p:32:32-i64:64-f128:64-n32-S64");
    // NetBSD / OpenBSD use long (same as llvm default); everyone else uses int.
    switch (getTriple().getOS()) {
    default:
      SizeType = UnsignedInt;
      IntPtrType = SignedInt;
      PtrDiffType = SignedInt;
      break;
    case llvm::Triple::NetBSD:
    case llvm::Triple::OpenBSD:
      SizeType = UnsignedLong;
      IntPtrType = SignedLong;
      PtrDiffType = SignedLong;
      break;
    }
    // Up to 32 bits are lock-free atomic, but we're willing to do atomic ops
    // on up to 64 bits.
    MaxAtomicPromoteWidth = 64;
    MaxAtomicInlineWidth = 32;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    SparcTargetInfo::getTargetDefines(Opts, Builder);
    switch (getCPUGeneration(CPU)) {
    case CG_V8:
      Builder.defineMacro("__sparcv8");
      if (getTriple().getOS() != llvm::Triple::Solaris)
        Builder.defineMacro("__sparcv8__");
      break;
    case CG_V9:
      Builder.defineMacro("__sparcv9");
      if (getTriple().getOS() != llvm::Triple::Solaris) {
        Builder.defineMacro("__sparcv9__");
        Builder.defineMacro("__sparc_v9__");
      }
      break;
    }
    if (getTriple().getVendor() == llvm::Triple::Myriad) {
      std::string MyriadArchValue, Myriad2Value;
      Builder.defineMacro("__sparc_v8__");
      Builder.defineMacro("__leon__");
      switch (CPU) {
      case CK_MYRIAD2150:
        MyriadArchValue = "__ma2150";
        Myriad2Value = "2";
        break;
      case CK_MYRIAD2450:
        MyriadArchValue = "__ma2450";
        Myriad2Value = "2";
        break;
      default:
        MyriadArchValue = "__ma2100";
        Myriad2Value = "1";
        break;
      }
      Builder.defineMacro(MyriadArchValue, "1");
      Builder.defineMacro(MyriadArchValue+"__", "1");
      Builder.defineMacro("__myriad2__", Myriad2Value);
      Builder.defineMacro("__myriad2", Myriad2Value);
    }
  }

  bool hasSjLjLowering() const override {
    return true;
  }
};

// SPARCV8el is the 32-bit little-endian mode selected by Triple::sparcel.
class SparcV8elTargetInfo : public SparcV8TargetInfo {
 public:
   SparcV8elTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
       : SparcV8TargetInfo(Triple, Opts) {
     resetDataLayout("e-m:e-p:32:32-i64:64-f128:64-n32-S64");
  }
};

// SPARC v9 is the 64-bit mode selected by Triple::sparcv9.
class SparcV9TargetInfo : public SparcTargetInfo {
public:
  SparcV9TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : SparcTargetInfo(Triple, Opts) {
    // FIXME: Support Sparc quad-precision long double?
    resetDataLayout("E-m:e-i64:64-n32:64-S128");
    // This is an LP64 platform.
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;

    // OpenBSD uses long long for int64_t and intmax_t.
    if (getTriple().getOS() == llvm::Triple::OpenBSD)
      IntMaxType = SignedLongLong;
    else
      IntMaxType = SignedLong;
    Int64Type = IntMaxType;

    // The SPARCv8 System V ABI has long double 128-bits in size, but 64-bit
    // aligned. The SPARCv9 SCD 2.4.1 says 16-byte aligned.
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    SparcTargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__sparcv9");
    Builder.defineMacro("__arch64__");
    // Solaris doesn't need these variants, but the BSDs do.
    if (getTriple().getOS() != llvm::Triple::Solaris) {
      Builder.defineMacro("__sparc64__");
      Builder.defineMacro("__sparc_v9__");
      Builder.defineMacro("__sparcv9__");
    }
  }

  bool setCPU(const std::string &Name) override {
    if (!SparcTargetInfo::setCPU(Name))
      return false;
    return getCPUGeneration(CPU) == CG_V9;
  }
};

class SystemZTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
  static const char *const GCCRegNames[];
  std::string CPU;
  int ISARevision;
  bool HasTransactionalExecution;
  bool HasVector;

public:
  SystemZTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple), CPU("z10"), ISARevision(8),
        HasTransactionalExecution(false), HasVector(false) {
    IntMaxType = SignedLong;
    Int64Type = SignedLong;
    TLSSupported = true;
    IntWidth = IntAlign = 32;
    LongWidth = LongLongWidth = LongAlign = LongLongAlign = 64;
    PointerWidth = PointerAlign = 64;
    LongDoubleWidth = 128;
    LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
    DefaultAlignForAttributeAligned = 64;
    MinGlobalAlign = 16;
    resetDataLayout("E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64");
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__s390__");
    Builder.defineMacro("__s390x__");
    Builder.defineMacro("__zarch__");
    Builder.defineMacro("__LONG_DOUBLE_128__");

    Builder.defineMacro("__ARCH__", Twine(ISARevision));

    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");

    if (HasTransactionalExecution)
      Builder.defineMacro("__HTM__");
    if (HasVector)
      Builder.defineMacro("__VX__");
    if (Opts.ZVector)
      Builder.defineMacro("__VEC__", "10302");
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                         clang::SystemZ::LastTSBuiltin-Builtin::FirstTSBuiltin);
  }

  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    // No aliases.
    return None;
  }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override;
  const char *getClobbers() const override {
    // FIXME: Is this really right?
    return "";
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::SystemZBuiltinVaList;
  }
  int getISARevision(const StringRef &Name) const {
    return llvm::StringSwitch<int>(Name)
      .Cases("arch8", "z10", 8)
      .Cases("arch9", "z196", 9)
      .Cases("arch10", "zEC12", 10)
      .Cases("arch11", "z13", 11)
      .Cases("arch12", "z14", 12)
      .Default(-1);
  }
  bool setCPU(const std::string &Name) override {
    CPU = Name;
    ISARevision = getISARevision(CPU);
    return ISARevision != -1;
  }
  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override {
    int ISARevision = getISARevision(CPU);
    if (ISARevision >= 10)
      Features["transactional-execution"] = true;
    if (ISARevision >= 11)
      Features["vector"] = true;
    if (ISARevision >= 12)
      Features["vector-enhancements-1"] = true;
    return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    HasTransactionalExecution = false;
    HasVector = false;
    for (const auto &Feature : Features) {
      if (Feature == "+transactional-execution")
        HasTransactionalExecution = true;
      else if (Feature == "+vector")
        HasVector = true;
    }
    // If we use the vector ABI, vector types are 64-bit aligned.
    if (HasVector) {
      MaxVectorAlign = 64;
      resetDataLayout("E-m:e-i1:8:16-i8:8:16-i64:64-f128:64"
                      "-v128:64-a:8:16-n32:64");
    }
    return true;
  }

  bool hasFeature(StringRef Feature) const override {
    return llvm::StringSwitch<bool>(Feature)
        .Case("systemz", true)
        .Case("arch8", ISARevision >= 8)
        .Case("arch9", ISARevision >= 9)
        .Case("arch10", ISARevision >= 10)
        .Case("arch11", ISARevision >= 11)
        .Case("arch12", ISARevision >= 12)
        .Case("htm", HasTransactionalExecution)
        .Case("vx", HasVector)
        .Default(false);
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_C:
    case CC_Swift:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }

  StringRef getABI() const override {
    if (HasVector)
      return "vector";
    return "";
  }

  bool useFloat128ManglingForLongDouble() const override {
    return true;
  }
};

const Builtin::Info SystemZTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE },
#include "clang/Basic/BuiltinsSystemZ.def"
};

const char *const SystemZTargetInfo::GCCRegNames[] = {
  "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",
  "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
  "f0",  "f2",  "f4",  "f6",  "f1",  "f3",  "f5",  "f7",
  "f8",  "f10", "f12", "f14", "f9",  "f11", "f13", "f15"
};

ArrayRef<const char *> SystemZTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

bool SystemZTargetInfo::
validateAsmConstraint(const char *&Name,
                      TargetInfo::ConstraintInfo &Info) const {
  switch (*Name) {
  default:
    return false;

  case 'a': // Address register
  case 'd': // Data register (equivalent to 'r')
  case 'f': // Floating-point register
    Info.setAllowsRegister();
    return true;

  case 'I': // Unsigned 8-bit constant
  case 'J': // Unsigned 12-bit constant
  case 'K': // Signed 16-bit constant
  case 'L': // Signed 20-bit displacement (on all targets we support)
  case 'M': // 0x7fffffff
    return true;

  case 'Q': // Memory with base and unsigned 12-bit displacement
  case 'R': // Likewise, plus an index
  case 'S': // Memory with base and signed 20-bit displacement
  case 'T': // Likewise, plus an index
    Info.setAllowsMemory();
    return true;
  }
}

class MSP430TargetInfo : public TargetInfo {
  static const char *const GCCRegNames[];

public:
  MSP430TargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    TLSSupported = false;
    IntWidth = 16;
    IntAlign = 16;
    LongWidth = 32;
    LongLongWidth = 64;
    LongAlign = LongLongAlign = 16;
    PointerWidth = 16;
    PointerAlign = 16;
    SuitableAlign = 16;
    SizeType = UnsignedInt;
    IntMaxType = SignedLongLong;
    IntPtrType = SignedInt;
    PtrDiffType = SignedInt;
    SigAtomicType = SignedLong;
    resetDataLayout("e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("MSP430");
    Builder.defineMacro("__MSP430__");
    // FIXME: defines for different 'flavours' of MCU
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    // FIXME: Implement.
    return None;
  }
  bool hasFeature(StringRef Feature) const override {
    return Feature == "msp430";
  }
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    // No aliases.
    return None;
  }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    // FIXME: implement
    switch (*Name) {
    case 'K': // the constant 1
    case 'L': // constant -1^20 .. 1^19
    case 'M': // constant 1-4:
      return true;
    }
    // No target constraints for now.
    return false;
  }
  const char *getClobbers() const override {
    // FIXME: Is this really right?
    return "";
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    // FIXME: implement
    return TargetInfo::CharPtrBuiltinVaList;
  }
};

const char *const MSP430TargetInfo::GCCRegNames[] = {
    "r0", "r1", "r2",  "r3",  "r4",  "r5",  "r6",  "r7",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};

ArrayRef<const char *> MSP430TargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

// LLVM and Clang cannot be used directly to output native binaries for
// target, but is used to compile C code to llvm bitcode with correct
// type and alignment information.
//
// TCE uses the llvm bitcode as input and uses it for generating customized
// target processor and program binary. TCE co-design environment is
// publicly available in http://tce.cs.tut.fi

static const unsigned TCEOpenCLAddrSpaceMap[] = {
    0, // Default
    3, // opencl_global
    4, // opencl_local
    5, // opencl_constant
    // FIXME: generic has to be added to the target
    0, // opencl_generic
    0, // cuda_device
    0, // cuda_constant
    0  // cuda_shared
};

class TCETargetInfo : public TargetInfo {
public:
  TCETargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
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
    IntPtrType = SignedInt;
    PtrDiffType = SignedInt;
    FloatWidth = 32;
    FloatAlign = 32;
    DoubleWidth = 32;
    DoubleAlign = 32;
    LongDoubleWidth = 32;
    LongDoubleAlign = 32;
    FloatFormat = &llvm::APFloat::IEEEsingle();
    DoubleFormat = &llvm::APFloat::IEEEsingle();
    LongDoubleFormat = &llvm::APFloat::IEEEsingle();
    resetDataLayout("E-p:32:32:32-i1:8:8-i8:8:32-"
                    "i16:16:32-i32:32:32-i64:32:32-"
                    "f32:32:32-f64:32:32-v64:32:32-"
                    "v128:32:32-v256:32:32-v512:32:32-"
                    "v1024:32:32-a0:0:32-n32");
    AddrSpaceMap = &TCEOpenCLAddrSpaceMap;
    UseAddrSpaceMapMangling = true;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "tce", Opts);
    Builder.defineMacro("__TCE__");
    Builder.defineMacro("__TCE_V1__");
  }
  bool hasFeature(StringRef Feature) const override { return Feature == "tce"; }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override { return None; }
  const char *getClobbers() const override { return ""; }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const override { return None; }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return true;
  }
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }
};

class TCELETargetInfo : public TCETargetInfo {
public:
  TCELETargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TCETargetInfo(Triple, Opts) {
    BigEndian = false;

    resetDataLayout("e-p:32:32:32-i1:8:8-i8:8:32-"
                    "i16:16:32-i32:32:32-i64:32:32-"
                    "f32:32:32-f64:32:32-v64:32:32-"
                    "v128:32:32-v256:32:32-v512:32:32-"
                    "v1024:32:32-a0:0:32-n32");

  }

  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const {
    DefineStd(Builder, "tcele", Opts);
    Builder.defineMacro("__TCE__");
    Builder.defineMacro("__TCE_V1__");
    Builder.defineMacro("__TCELE__");
    Builder.defineMacro("__TCELE_V1__");
  }

};

class BPFTargetInfo : public TargetInfo {
public:
  BPFTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    SizeType    = UnsignedLong;
    PtrDiffType = SignedLong;
    IntPtrType  = SignedLong;
    IntMaxType  = SignedLong;
    Int64Type   = SignedLong;
    RegParmMax = 5;
    if (Triple.getArch() == llvm::Triple::bpfeb) {
      resetDataLayout("E-m:e-p:64:64-i64:64-n32:64-S128");
    } else {
      resetDataLayout("e-m:e-p:64:64-i64:64-n32:64-S128");
    }
    MaxAtomicPromoteWidth = 64;
    MaxAtomicInlineWidth = 64;
    TLSSupported = false;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "bpf", Opts);
    Builder.defineMacro("__BPF__");
  }
  bool hasFeature(StringRef Feature) const override {
    return Feature == "bpf";
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override { return None; }
  const char *getClobbers() const override {
    return "";
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const override {
    return None;
  }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return true;
  }
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }
  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
      default:
        return CCCR_Warning;
      case CC_C:
      case CC_OpenCLKernel:
        return CCCR_OK;
    }
  }
};

class Nios2TargetInfo : public TargetInfo {
  void setDataLayout() {
    if (BigEndian)
      resetDataLayout("E-p:32:32:32-i8:8:32-i16:16:32-n32");
    else
      resetDataLayout("e-p:32:32:32-i8:8:32-i16:16:32-n32");
  }

  static const Builtin::Info BuiltinInfo[];
  std::string CPU;
  std::string ABI;

public:
  Nios2TargetInfo(const llvm::Triple &triple, const TargetOptions &opts)
      : TargetInfo(triple), CPU(opts.CPU), ABI(opts.ABI) {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 32;
    setDataLayout();
  }

  StringRef getABI() const override { return ABI; }
  bool setABI(const std::string &Name) override {
    if (Name == "o32" || Name == "eabi") {
      ABI = Name;
      return true;
    }
    return false;
  }

  bool setCPU(const std::string &Name) override {
    if (Name == "nios2r1" || Name == "nios2r2") {
      CPU = Name;
      return true;
    }
    return false;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "nios2", Opts);
    DefineStd(Builder, "NIOS2", Opts);

    Builder.defineMacro("__nios2");
    Builder.defineMacro("__NIOS2");
    Builder.defineMacro("__nios2__");
    Builder.defineMacro("__NIOS2__");
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo, clang::Nios2::LastTSBuiltin -
                                               Builtin::FirstTSBuiltin);
  }

  bool isFeatureSupportedByCPU(StringRef Feature, StringRef CPU) const {
    const bool isR2 = CPU == "nios2r2";
    return llvm::StringSwitch<bool>(Feature)
        .Case("nios2r2mandatory", isR2)
        .Case("nios2r2bmx", isR2)
        .Case("nios2r2mpx", isR2)
        .Case("nios2r2cdx", isR2)
        .Default(false);
  }

  bool initFeatureMap(llvm::StringMap<bool> &Features,
                      DiagnosticsEngine &Diags, StringRef CPU,
                      const std::vector<std::string> &FeatureVec) const override {
    static const char *allFeatures[] = {
      "nios2r2mandatory", "nios2r2bmx", "nios2r2mpx", "nios2r2cdx"
    };
    for (const char *feature : allFeatures) {
        Features[feature] = isFeatureSupportedByCPU(feature, CPU);
    }
    return true;
  }

  bool hasFeature(StringRef Feature) const override {
    return isFeatureSupportedByCPU(Feature, CPU);
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  ArrayRef<const char *> getGCCRegNames() const override {
    static const char *const GCCRegNames[] = {
      // CPU register names
      // Must match second column of GCCRegAliases
      "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10",
      "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20",
      "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30",
      "r31",
      // Floating point register names
      "ctl0", "ctl1", "ctl2", "ctl3", "ctl4", "ctl5", "ctl6", "ctl7", "ctl8",
      "ctl9", "ctl10", "ctl11", "ctl12", "ctl13", "ctl14", "ctl15"
    };
    return llvm::makeArrayRef(GCCRegNames);
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
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

  const char *getClobbers() const override { return ""; }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    static const TargetInfo::GCCRegAlias aliases[] = {
        {{"zero"}, "r0"},      {{"at"}, "r1"},          {{"et"}, "r24"},
        {{"bt"}, "r25"},       {{"gp"}, "r26"},         {{"sp"}, "r27"},
        {{"fp"}, "r28"},       {{"ea"}, "r29"},         {{"ba"}, "r30"},
        {{"ra"}, "r31"},       {{"status"}, "ctl0"},    {{"estatus"}, "ctl1"},
        {{"bstatus"}, "ctl2"}, {{"ienable"}, "ctl3"},   {{"ipending"}, "ctl4"},
        {{"cpuid"}, "ctl5"},   {{"exception"}, "ctl7"}, {{"pteaddr"}, "ctl8"},
        {{"tlbacc"}, "ctl9"},  {{"tlbmisc"}, "ctl10"},  {{"badaddr"}, "ctl12"},
        {{"config"}, "ctl13"}, {{"mpubase"}, "ctl14"},  {{"mpuacc"}, "ctl15"},
    };
    return llvm::makeArrayRef(aliases);
  }
};

const Builtin::Info Nios2TargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE},
#include "clang/Basic/BuiltinsNios2.def"
};

class MipsTargetInfo : public TargetInfo {
  void setDataLayout() {
    StringRef Layout;

    if (ABI == "o32")
      Layout = "m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64";
    else if (ABI == "n32")
      Layout = "m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32:64-S128";
    else if (ABI == "n64")
      Layout = "m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128";
    else
      llvm_unreachable("Invalid ABI");

    if (BigEndian)
      resetDataLayout(("E-" + Layout).str());
    else
      resetDataLayout(("e-" + Layout).str());
  }


  static const Builtin::Info BuiltinInfo[];
  std::string CPU;
  bool IsMips16;
  bool IsMicromips;
  bool IsNan2008;
  bool IsSingleFloat;
  bool IsNoABICalls;
  bool CanUseBSDABICalls;
  enum MipsFloatABI {
    HardFloat, SoftFloat
  } FloatABI;
  enum DspRevEnum {
    NoDSP, DSP1, DSP2
  } DspRev;
  bool HasMSA;
  bool DisableMadd4;

protected:
  bool HasFP64;
  std::string ABI;

public:
  MipsTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple), IsMips16(false), IsMicromips(false),
        IsNan2008(false), IsSingleFloat(false), IsNoABICalls(false),
        CanUseBSDABICalls(false), FloatABI(HardFloat), DspRev(NoDSP),
        HasMSA(false), DisableMadd4(false), HasFP64(false) {
    TheCXXABI.set(TargetCXXABI::GenericMIPS);

    setABI((getTriple().getArch() == llvm::Triple::mips ||
            getTriple().getArch() == llvm::Triple::mipsel)
               ? "o32"
               : "n64");

    CPU = ABI == "o32" ? "mips32r2" : "mips64r2";

    CanUseBSDABICalls = Triple.getOS() == llvm::Triple::FreeBSD ||
                        Triple.getOS() == llvm::Triple::OpenBSD;
  }

  bool isNaN2008Default() const {
    return CPU == "mips32r6" || CPU == "mips64r6";
  }

  bool isFP64Default() const {
    return CPU == "mips32r6" || ABI == "n32" || ABI == "n64" || ABI == "64";
  }

  bool isNan2008() const override {
    return IsNan2008;
  }

  bool processorSupportsGPR64() const {
    return llvm::StringSwitch<bool>(CPU)
        .Case("mips3", true)
        .Case("mips4", true)
        .Case("mips5", true)
        .Case("mips64", true)
        .Case("mips64r2", true)
        .Case("mips64r3", true)
        .Case("mips64r5", true)
        .Case("mips64r6", true)
        .Case("octeon", true)
        .Default(false);
    return false;
  }

  StringRef getABI() const override { return ABI; }
  bool setABI(const std::string &Name) override {
    if (Name == "o32") {
      setO32ABITypes();
      ABI = Name;
      return true;
    }

    if (Name == "n32") {
      setN32ABITypes();
      ABI = Name;
      return true;
    }
    if (Name == "n64") {
      setN64ABITypes();
      ABI = Name;
      return true;
    }
    return false;
  }

  void setO32ABITypes() {
    Int64Type = SignedLongLong;
    IntMaxType = Int64Type;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    LongDoubleWidth = LongDoubleAlign = 64;
    LongWidth = LongAlign = 32;
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 32;
    PointerWidth = PointerAlign = 32;
    PtrDiffType = SignedInt;
    SizeType = UnsignedInt;
    SuitableAlign = 64;
  }

  void setN32N64ABITypes() {
    LongDoubleWidth = LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
    if (getTriple().getOS() == llvm::Triple::FreeBSD) {
      LongDoubleWidth = LongDoubleAlign = 64;
      LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    }
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
    SuitableAlign = 128;
  }

  void setN64ABITypes() {
    setN32N64ABITypes();
    if (getTriple().getOS() == llvm::Triple::OpenBSD) {
      Int64Type = SignedLongLong;
    } else {
      Int64Type = SignedLong;
    }
    IntMaxType = Int64Type;
    LongWidth = LongAlign = 64;
    PointerWidth = PointerAlign = 64;
    PtrDiffType = SignedLong;
    SizeType = UnsignedLong;
  }

  void setN32ABITypes() {
    setN32N64ABITypes();
    Int64Type = SignedLongLong;
    IntMaxType = Int64Type;
    LongWidth = LongAlign = 32;
    PointerWidth = PointerAlign = 32;
    PtrDiffType = SignedInt;
    SizeType = UnsignedInt;
  }

  bool setCPU(const std::string &Name) override {
    CPU = Name;
    return llvm::StringSwitch<bool>(Name)
        .Case("mips1", true)
        .Case("mips2", true)
        .Case("mips3", true)
        .Case("mips4", true)
        .Case("mips5", true)
        .Case("mips32", true)
        .Case("mips32r2", true)
        .Case("mips32r3", true)
        .Case("mips32r5", true)
        .Case("mips32r6", true)
        .Case("mips64", true)
        .Case("mips64r2", true)
        .Case("mips64r3", true)
        .Case("mips64r5", true)
        .Case("mips64r6", true)
        .Case("octeon", true)
        .Case("p5600", true)
        .Default(false);
  }
  const std::string& getCPU() const { return CPU; }
  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override {
    if (CPU.empty())
      CPU = getCPU();
    if (CPU == "octeon")
      Features["mips64r2"] = Features["cnmips"] = true;
    else
      Features[CPU] = true;
    return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    if (BigEndian) {
      DefineStd(Builder, "MIPSEB", Opts);
      Builder.defineMacro("_MIPSEB");
    } else {
      DefineStd(Builder, "MIPSEL", Opts);
      Builder.defineMacro("_MIPSEL");
    }

    Builder.defineMacro("__mips__");
    Builder.defineMacro("_mips");
    if (Opts.GNUMode)
      Builder.defineMacro("mips");

    if (ABI == "o32") {
      Builder.defineMacro("__mips", "32");
      Builder.defineMacro("_MIPS_ISA", "_MIPS_ISA_MIPS32");
    } else {
      Builder.defineMacro("__mips", "64");
      Builder.defineMacro("__mips64");
      Builder.defineMacro("__mips64__");
      Builder.defineMacro("_MIPS_ISA", "_MIPS_ISA_MIPS64");
    }

    const std::string ISARev = llvm::StringSwitch<std::string>(getCPU())
                                   .Cases("mips32", "mips64", "1")
                                   .Cases("mips32r2", "mips64r2", "2")
                                   .Cases("mips32r3", "mips64r3", "3")
                                   .Cases("mips32r5", "mips64r5", "5")
                                   .Cases("mips32r6", "mips64r6", "6")
                                   .Default("");
    if (!ISARev.empty())
      Builder.defineMacro("__mips_isa_rev", ISARev);

    if (ABI == "o32") {
      Builder.defineMacro("__mips_o32");
      Builder.defineMacro("_ABIO32", "1");
      Builder.defineMacro("_MIPS_SIM", "_ABIO32");
    } else if (ABI == "n32") {
      Builder.defineMacro("__mips_n32");
      Builder.defineMacro("_ABIN32", "2");
      Builder.defineMacro("_MIPS_SIM", "_ABIN32");
    } else if (ABI == "n64") {
      Builder.defineMacro("__mips_n64");
      Builder.defineMacro("_ABI64", "3");
      Builder.defineMacro("_MIPS_SIM", "_ABI64");
    } else
      llvm_unreachable("Invalid ABI.");

    if (!IsNoABICalls) {
      Builder.defineMacro("__mips_abicalls");
      if (CanUseBSDABICalls)
        Builder.defineMacro("__ABICALLS__");
    }

    Builder.defineMacro("__REGISTER_PREFIX__", "");

    switch (FloatABI) {
    case HardFloat:
      Builder.defineMacro("__mips_hard_float", Twine(1));
      break;
    case SoftFloat:
      Builder.defineMacro("__mips_soft_float", Twine(1));
      break;
    }

    if (IsSingleFloat)
      Builder.defineMacro("__mips_single_float", Twine(1));

    Builder.defineMacro("__mips_fpr", HasFP64 ? Twine(64) : Twine(32));
    Builder.defineMacro("_MIPS_FPSET",
                        Twine(32 / (HasFP64 || IsSingleFloat ? 1 : 2)));

    if (IsMips16)
      Builder.defineMacro("__mips16", Twine(1));

    if (IsMicromips)
      Builder.defineMacro("__mips_micromips", Twine(1));

    if (IsNan2008)
      Builder.defineMacro("__mips_nan2008", Twine(1));

    switch (DspRev) {
    default:
      break;
    case DSP1:
      Builder.defineMacro("__mips_dsp_rev", Twine(1));
      Builder.defineMacro("__mips_dsp", Twine(1));
      break;
    case DSP2:
      Builder.defineMacro("__mips_dsp_rev", Twine(2));
      Builder.defineMacro("__mips_dspr2", Twine(1));
      Builder.defineMacro("__mips_dsp", Twine(1));
      break;
    }

    if (HasMSA)
      Builder.defineMacro("__mips_msa", Twine(1));

    if (DisableMadd4)
      Builder.defineMacro("__mips_no_madd4", Twine(1));

    Builder.defineMacro("_MIPS_SZPTR", Twine(getPointerWidth(0)));
    Builder.defineMacro("_MIPS_SZINT", Twine(getIntWidth()));
    Builder.defineMacro("_MIPS_SZLONG", Twine(getLongWidth()));

    Builder.defineMacro("_MIPS_ARCH", "\"" + CPU + "\"");
    Builder.defineMacro("_MIPS_ARCH_" + StringRef(CPU).upper());

    // These shouldn't be defined for MIPS-I but there's no need to check
    // for that since MIPS-I isn't supported.
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");

    // 32-bit MIPS processors don't have the necessary lld/scd instructions
    // found in 64-bit processors. In the case of O32 on a 64-bit processor,
    // the instructions exist but using them violates the ABI since they
    // require 64-bit GPRs and O32 only supports 32-bit GPRs.
    if (ABI == "n32" || ABI == "n64")
      Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                          clang::Mips::LastTSBuiltin - Builtin::FirstTSBuiltin);
  }
  bool hasFeature(StringRef Feature) const override {
    return llvm::StringSwitch<bool>(Feature)
      .Case("mips", true)
      .Case("fp64", HasFP64)
      .Default(false);
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const override {
    static const char *const GCCRegNames[] = {
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
      "$fcc5","$fcc6","$fcc7","$ac1hi","$ac1lo","$ac2hi","$ac2lo",
      "$ac3hi","$ac3lo",
      // MSA register names
      "$w0",  "$w1",  "$w2",  "$w3",  "$w4",  "$w5",  "$w6",  "$w7",
      "$w8",  "$w9",  "$w10", "$w11", "$w12", "$w13", "$w14", "$w15",
      "$w16", "$w17", "$w18", "$w19", "$w20", "$w21", "$w22", "$w23",
      "$w24", "$w25", "$w26", "$w27", "$w28", "$w29", "$w30", "$w31",
      // MSA control register names
      "$msair",      "$msacsr", "$msaaccess", "$msasave", "$msamodify",
      "$msarequest", "$msamap", "$msaunmap"
    };
    return llvm::makeArrayRef(GCCRegNames);
  }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      return false;
    case 'r': // CPU registers.
    case 'd': // Equivalent to "r" unless generating MIPS16 code.
    case 'y': // Equivalent to "r", backward compatibility only.
    case 'f': // floating-point registers.
    case 'c': // $25 for indirect jumps
    case 'l': // lo register
    case 'x': // hilo register pair
      Info.setAllowsRegister();
      return true;
    case 'I': // Signed 16-bit constant
    case 'J': // Integer 0
    case 'K': // Unsigned 16-bit constant
    case 'L': // Signed 32-bit constant, lower 16-bit zeros (for lui)
    case 'M': // Constants not loadable via lui, addiu, or ori
    case 'N': // Constant -1 to -65535
    case 'O': // A signed 15-bit constant
    case 'P': // A constant between 1 go 65535
      return true;
    case 'R': // An address that can be used in a non-macro load or store
      Info.setAllowsMemory();
      return true;
    case 'Z':
      if (Name[1] == 'C') { // An address usable by ll, and sc.
        Info.setAllowsMemory();
        Name++; // Skip over 'Z'.
        return true;
      }
      return false;
    }
  }

  std::string convertConstraint(const char *&Constraint) const override {
    std::string R;
    switch (*Constraint) {
    case 'Z': // Two-character constraint; add "^" hint for later parsing.
      if (Constraint[1] == 'C') {
        R = std::string("^") + std::string(Constraint, 2);
        Constraint++;
        return R;
      }
      break;
    }
    return TargetInfo::convertConstraint(Constraint);
  }

  const char *getClobbers() const override {
    // In GCC, $1 is not widely used in generated code (it's used only in a few
    // specific situations), so there is no real need for users to add it to
    // the clobbers list if they want to use it in their inline assembly code.
    //
    // In LLVM, $1 is treated as a normal GPR and is always allocatable during
    // code generation, so using it in inline assembly without adding it to the
    // clobbers list can cause conflicts between the inline assembly code and
    // the surrounding generated code.
    //
    // Another problem is that LLVM is allowed to choose $1 for inline assembly
    // operands, which will conflict with the ".set at" assembler option (which
    // we use only for inline assembly, in order to maintain compatibility with
    // GCC) and will also conflict with the user's usage of $1.
    //
    // The easiest way to avoid these conflicts and keep $1 as an allocatable
    // register for generated code is to automatically clobber $1 for all inline
    // assembly code.
    //
    // FIXME: We should automatically clobber $1 only for inline assembly code
    // which actually uses it. This would allow LLVM to use $1 for inline
    // assembly operands if the user's assembly code doesn't use it.
    return "~{$1}";
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    IsMips16 = false;
    IsMicromips = false;
    IsNan2008 = isNaN2008Default();
    IsSingleFloat = false;
    FloatABI = HardFloat;
    DspRev = NoDSP;
    HasFP64 = isFP64Default();

    for (const auto &Feature : Features) {
      if (Feature == "+single-float")
        IsSingleFloat = true;
      else if (Feature == "+soft-float")
        FloatABI = SoftFloat;
      else if (Feature == "+mips16")
        IsMips16 = true;
      else if (Feature == "+micromips")
        IsMicromips = true;
      else if (Feature == "+dsp")
        DspRev = std::max(DspRev, DSP1);
      else if (Feature == "+dspr2")
        DspRev = std::max(DspRev, DSP2);
      else if (Feature == "+msa")
        HasMSA = true;
      else if (Feature == "+nomadd4")
        DisableMadd4 = true;
      else if (Feature == "+fp64")
        HasFP64 = true;
      else if (Feature == "-fp64")
        HasFP64 = false;
      else if (Feature == "+nan2008")
        IsNan2008 = true;
      else if (Feature == "-nan2008")
        IsNan2008 = false;
      else if (Feature == "+noabicalls")
        IsNoABICalls = true;
    }

    setDataLayout();

    return true;
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0) return 4;
    if (RegNo == 1) return 5;
    return -1;
  }

  bool isCLZForZeroUndef() const override { return false; }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    static const TargetInfo::GCCRegAlias O32RegAliases[] = {
        {{"at"}, "$1"},  {{"v0"}, "$2"},         {{"v1"}, "$3"},
        {{"a0"}, "$4"},  {{"a1"}, "$5"},         {{"a2"}, "$6"},
        {{"a3"}, "$7"},  {{"t0"}, "$8"},         {{"t1"}, "$9"},
        {{"t2"}, "$10"}, {{"t3"}, "$11"},        {{"t4"}, "$12"},
        {{"t5"}, "$13"}, {{"t6"}, "$14"},        {{"t7"}, "$15"},
        {{"s0"}, "$16"}, {{"s1"}, "$17"},        {{"s2"}, "$18"},
        {{"s3"}, "$19"}, {{"s4"}, "$20"},        {{"s5"}, "$21"},
        {{"s6"}, "$22"}, {{"s7"}, "$23"},        {{"t8"}, "$24"},
        {{"t9"}, "$25"}, {{"k0"}, "$26"},        {{"k1"}, "$27"},
        {{"gp"}, "$28"}, {{"sp", "$sp"}, "$29"}, {{"fp", "$fp"}, "$30"},
        {{"ra"}, "$31"}};
    static const TargetInfo::GCCRegAlias NewABIRegAliases[] = {
        {{"at"}, "$1"},  {{"v0"}, "$2"},         {{"v1"}, "$3"},
        {{"a0"}, "$4"},  {{"a1"}, "$5"},         {{"a2"}, "$6"},
        {{"a3"}, "$7"},  {{"a4"}, "$8"},         {{"a5"}, "$9"},
        {{"a6"}, "$10"}, {{"a7"}, "$11"},        {{"t0"}, "$12"},
        {{"t1"}, "$13"}, {{"t2"}, "$14"},        {{"t3"}, "$15"},
        {{"s0"}, "$16"}, {{"s1"}, "$17"},        {{"s2"}, "$18"},
        {{"s3"}, "$19"}, {{"s4"}, "$20"},        {{"s5"}, "$21"},
        {{"s6"}, "$22"}, {{"s7"}, "$23"},        {{"t8"}, "$24"},
        {{"t9"}, "$25"}, {{"k0"}, "$26"},        {{"k1"}, "$27"},
        {{"gp"}, "$28"}, {{"sp", "$sp"}, "$29"}, {{"fp", "$fp"}, "$30"},
        {{"ra"}, "$31"}};
    if (ABI == "o32")
      return llvm::makeArrayRef(O32RegAliases);
    return llvm::makeArrayRef(NewABIRegAliases);
  }

  bool hasInt128Type() const override {
    return ABI == "n32" || ABI == "n64";
  }

  bool validateTarget(DiagnosticsEngine &Diags) const override {
    // FIXME: It's valid to use O32 on a 64-bit CPU but the backend can't handle
    //        this yet. It's better to fail here than on the backend assertion.
    if (processorSupportsGPR64() && ABI == "o32") {
      Diags.Report(diag::err_target_unsupported_abi) << ABI << CPU;
      return false;
    }

    // 64-bit ABI's require 64-bit CPU's.
    if (!processorSupportsGPR64() && (ABI == "n32" || ABI == "n64")) {
      Diags.Report(diag::err_target_unsupported_abi) << ABI << CPU;
      return false;
    }

    // FIXME: It's valid to use O32 on a mips64/mips64el triple but the backend
    //        can't handle this yet. It's better to fail here than on the
    //        backend assertion.
    if ((getTriple().getArch() == llvm::Triple::mips64 ||
         getTriple().getArch() == llvm::Triple::mips64el) &&
        ABI == "o32") {
      Diags.Report(diag::err_target_unsupported_abi_for_triple)
          << ABI << getTriple().str();
      return false;
    }

    // FIXME: It's valid to use N32/N64 on a mips/mipsel triple but the backend
    //        can't handle this yet. It's better to fail here than on the
    //        backend assertion.
    if ((getTriple().getArch() == llvm::Triple::mips ||
         getTriple().getArch() == llvm::Triple::mipsel) &&
        (ABI == "n32" || ABI == "n64")) {
      Diags.Report(diag::err_target_unsupported_abi_for_triple)
          << ABI << getTriple().str();
      return false;
    }

    return true;
  }
};

const Builtin::Info MipsTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsMips.def"
};

class PNaClTargetInfo : public TargetInfo {
public:
  PNaClTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TargetInfo(Triple) {
    this->LongAlign = 32;
    this->LongWidth = 32;
    this->PointerAlign = 32;
    this->PointerWidth = 32;
    this->IntMaxType = TargetInfo::SignedLongLong;
    this->Int64Type = TargetInfo::SignedLongLong;
    this->DoubleAlign = 64;
    this->LongDoubleWidth = 64;
    this->LongDoubleAlign = 64;
    this->SizeType = TargetInfo::UnsignedInt;
    this->PtrDiffType = TargetInfo::SignedInt;
    this->IntPtrType = TargetInfo::SignedInt;
    this->RegParmMax = 0; // Disallow regparm
  }

  void getArchDefines(const LangOptions &Opts, MacroBuilder &Builder) const {
    Builder.defineMacro("__le32__");
    Builder.defineMacro("__pnacl__");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    getArchDefines(Opts, Builder);
  }
  bool hasFeature(StringRef Feature) const override {
    return Feature == "pnacl";
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override { return None; }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::PNaClABIBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    return false;
  }

  const char *getClobbers() const override {
    return "";
  }
};

ArrayRef<const char *> PNaClTargetInfo::getGCCRegNames() const {
  return None;
}

ArrayRef<TargetInfo::GCCRegAlias> PNaClTargetInfo::getGCCRegAliases() const {
  return None;
}

// We attempt to use PNaCl (le32) frontend and Mips32EL backend.
class NaClMips32TargetInfo : public MipsTargetInfo {
public:
  NaClMips32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : MipsTargetInfo(Triple, Opts) {}

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::PNaClABIBuiltinVaList;
  }
};

class Le64TargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];

public:
  Le64TargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    NoAsmVariants = true;
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
    resetDataLayout("e-m:e-v128:32-v16:16-v32:32-v96:32-n8:16:32:64-S128");
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "unix", Opts);
    defineCPUMacros(Builder, "le64", /*Tuning=*/false);
    Builder.defineMacro("__ELF__");
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                          clang::Le64::LastTSBuiltin - Builtin::FirstTSBuiltin);
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::PNaClABIBuiltinVaList;
  }
  const char *getClobbers() const override { return ""; }
  ArrayRef<const char *> getGCCRegNames() const override {
    return None;
  }
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    return false;
  }

  bool hasProtectedVisibility() const override { return false; }
};

class WebAssemblyTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];

  enum SIMDEnum {
    NoSIMD,
    SIMD128,
  } SIMDLevel;

public:
  explicit WebAssemblyTargetInfo(const llvm::Triple &T, const TargetOptions &)
      : TargetInfo(T), SIMDLevel(NoSIMD) {
    NoAsmVariants = true;
    SuitableAlign = 128;
    LargeArrayMinWidth = 128;
    LargeArrayAlign = 128;
    SimdDefaultAlign = 128;
    SigAtomicType = SignedLong;
    LongDoubleWidth = LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
  }

protected:
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    defineCPUMacros(Builder, "wasm", /*Tuning=*/false);
    if (SIMDLevel >= SIMD128)
      Builder.defineMacro("__wasm_simd128__");
  }

private:
  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override {
    if (CPU == "bleeding-edge")
      Features["simd128"] = true;
    return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
  }
  bool hasFeature(StringRef Feature) const final {
    return llvm::StringSwitch<bool>(Feature)
        .Case("simd128", SIMDLevel >= SIMD128)
        .Default(false);
  }
  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) final {
    for (const auto &Feature : Features) {
      if (Feature == "+simd128") {
        SIMDLevel = std::max(SIMDLevel, SIMD128);
        continue;
      }
      if (Feature == "-simd128") {
        SIMDLevel = std::min(SIMDLevel, SIMDEnum(SIMD128 - 1));
        continue;
      }

      Diags.Report(diag::err_opt_not_valid_with_opt) << Feature
                                                     << "-target-feature";
      return false;
    }
    return true;
  }
  bool setCPU(const std::string &Name) final {
    return llvm::StringSwitch<bool>(Name)
              .Case("mvp",           true)
              .Case("bleeding-edge", true)
              .Case("generic",       true)
              .Default(false);
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const final {
    return llvm::makeArrayRef(BuiltinInfo,
                   clang::WebAssembly::LastTSBuiltin - Builtin::FirstTSBuiltin);
  }
  BuiltinVaListKind getBuiltinVaListKind() const final {
    return VoidPtrBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const final {
    return None;
  }
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const final {
    return None;
  }
  bool
  validateAsmConstraint(const char *&Name,
                        TargetInfo::ConstraintInfo &Info) const final {
    return false;
  }
  const char *getClobbers() const final { return ""; }
  bool isCLZForZeroUndef() const final { return false; }
  bool hasInt128Type() const final { return true; }
  IntType getIntTypeByWidth(unsigned BitWidth,
                            bool IsSigned) const final {
    // WebAssembly prefers long long for explicitly 64-bit integers.
    return BitWidth == 64 ? (IsSigned ? SignedLongLong : UnsignedLongLong)
                          : TargetInfo::getIntTypeByWidth(BitWidth, IsSigned);
  }
  IntType getLeastIntTypeByWidth(unsigned BitWidth,
                                 bool IsSigned) const final {
    // WebAssembly uses long long for int_least64_t and int_fast64_t.
    return BitWidth == 64
               ? (IsSigned ? SignedLongLong : UnsignedLongLong)
               : TargetInfo::getLeastIntTypeByWidth(BitWidth, IsSigned);
  }
};

const Builtin::Info WebAssemblyTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsWebAssembly.def"
};

class WebAssembly32TargetInfo : public WebAssemblyTargetInfo {
public:
  explicit WebAssembly32TargetInfo(const llvm::Triple &T,
                                   const TargetOptions &Opts)
      : WebAssemblyTargetInfo(T, Opts) {
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
    resetDataLayout("e-m:e-p:32:32-i64:64-n32:64-S128");
  }

protected:
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WebAssemblyTargetInfo::getTargetDefines(Opts, Builder);
    defineCPUMacros(Builder, "wasm32", /*Tuning=*/false);
  }
};

class WebAssembly64TargetInfo : public WebAssemblyTargetInfo {
public:
  explicit WebAssembly64TargetInfo(const llvm::Triple &T,
                                   const TargetOptions &Opts)
      : WebAssemblyTargetInfo(T, Opts) {
    LongAlign = LongWidth = 64;
    PointerAlign = PointerWidth = 64;
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
    SizeType = UnsignedLong;
    PtrDiffType = SignedLong;
    IntPtrType = SignedLong;
    resetDataLayout("e-m:e-p:64:64-i64:64-n32:64-S128");
  }

protected:
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WebAssemblyTargetInfo::getTargetDefines(Opts, Builder);
    defineCPUMacros(Builder, "wasm64", /*Tuning=*/false);
  }
};

const Builtin::Info Le64TargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsLe64.def"
};

static const unsigned SPIRAddrSpaceMap[] = {
    0, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    4, // opencl_generic
    0, // cuda_device
    0, // cuda_constant
    0  // cuda_shared
};
class SPIRTargetInfo : public TargetInfo {
public:
  SPIRTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    assert(getTriple().getOS() == llvm::Triple::UnknownOS &&
           "SPIR target must use unknown OS");
    assert(getTriple().getEnvironment() == llvm::Triple::UnknownEnvironment &&
           "SPIR target must use unknown environment type");
    TLSSupported = false;
    LongWidth = LongAlign = 64;
    AddrSpaceMap = &SPIRAddrSpaceMap;
    UseAddrSpaceMapMangling = true;
    // Define available target features
    // These must be defined in sorted order!
    NoAsmVariants = true;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "SPIR", Opts);
  }
  bool hasFeature(StringRef Feature) const override {
    return Feature == "spir";
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override { return None; }
  const char *getClobbers() const override { return ""; }
  ArrayRef<const char *> getGCCRegNames() const override { return None; }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return true;
  }
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    return (CC == CC_SpirFunction || CC == CC_OpenCLKernel) ? CCCR_OK
                                                            : CCCR_Warning;
  }

  CallingConv getDefaultCallingConv(CallingConvMethodType MT) const override {
    return CC_SpirFunction;
  }

  void setSupportedOpenCLOpts() override {
    // Assume all OpenCL extensions and optional core features are supported
    // for SPIR since it is a generic target.
    getSupportedOpenCLOpts().supportAll();
  }
};

class SPIR32TargetInfo : public SPIRTargetInfo {
public:
  SPIR32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : SPIRTargetInfo(Triple, Opts) {
    PointerWidth = PointerAlign = 32;
    SizeType = TargetInfo::UnsignedInt;
    PtrDiffType = IntPtrType = TargetInfo::SignedInt;
    resetDataLayout("e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-"
                    "v96:128-v192:256-v256:256-v512:512-v1024:1024");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "SPIR32", Opts);
  }
};

class SPIR64TargetInfo : public SPIRTargetInfo {
public:
  SPIR64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : SPIRTargetInfo(Triple, Opts) {
    PointerWidth = PointerAlign = 64;
    SizeType = TargetInfo::UnsignedLong;
    PtrDiffType = IntPtrType = TargetInfo::SignedLong;
    resetDataLayout("e-i64:64-v16:16-v24:32-v32:32-v48:64-"
                    "v96:128-v192:256-v256:256-v512:512-v1024:1024");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "SPIR64", Opts);
  }
};

class XCoreTargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];
public:
  XCoreTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    NoAsmVariants = true;
    LongLongAlign = 32;
    SuitableAlign = 32;
    DoubleAlign = LongDoubleAlign = 32;
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
    WCharType = UnsignedChar;
    WIntType = UnsignedInt;
    UseZeroLengthBitfieldAlignment = true;
    resetDataLayout("e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32"
                    "-f64:32-a:0:32-n32");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__XS1B__");
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo,
                           clang::XCore::LastTSBuiltin-Builtin::FirstTSBuiltin);
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }
  const char *getClobbers() const override {
    return "";
  }
  ArrayRef<const char *> getGCCRegNames() const override {
    static const char * const GCCRegNames[] = {
      "r0",   "r1",   "r2",   "r3",   "r4",   "r5",   "r6",   "r7",
      "r8",   "r9",   "r10",  "r11",  "cp",   "dp",   "sp",   "lr"
    };
    return llvm::makeArrayRef(GCCRegNames);
  }
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    return false;
  }
  int getEHDataRegisterNumber(unsigned RegNo) const override {
    // R0=ExceptionPointerRegister R1=ExceptionSelectorRegister
    return (RegNo < 2)? RegNo : -1;
  }
  bool allowsLargerPreferedTypeAlignment() const override {
    return false;
  }
};

const Builtin::Info XCoreTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER) \
  { #ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr },
#include "clang/Basic/BuiltinsXCore.def"
};

// x86_32 Android target
class AndroidX86_32TargetInfo : public LinuxTargetInfo<X86_32TargetInfo> {
public:
  AndroidX86_32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : LinuxTargetInfo<X86_32TargetInfo>(Triple, Opts) {
    SuitableAlign = 32;
    LongDoubleWidth = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
  }
};

// x86_64 Android target
class AndroidX86_64TargetInfo : public LinuxTargetInfo<X86_64TargetInfo> {
public:
  AndroidX86_64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : LinuxTargetInfo<X86_64TargetInfo>(Triple, Opts) {
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
  }

  bool useFloat128ManglingForLongDouble() const override {
    return true;
  }
};

// 32-bit RenderScript is armv7 with width and align of 'long' set to 8-bytes
class RenderScript32TargetInfo : public ARMleTargetInfo {
public:
  RenderScript32TargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts)
      : ARMleTargetInfo(llvm::Triple("armv7", Triple.getVendorName(),
                                     Triple.getOSName(),
                                     Triple.getEnvironmentName()),
                        Opts) {
    IsRenderScriptTarget = true;
    LongWidth = LongAlign = 64;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__RENDERSCRIPT__");
    ARMleTargetInfo::getTargetDefines(Opts, Builder);
  }
};

// 64-bit RenderScript is aarch64
class RenderScript64TargetInfo : public AArch64leTargetInfo {
public:
  RenderScript64TargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts)
      : AArch64leTargetInfo(llvm::Triple("aarch64", Triple.getVendorName(),
                                         Triple.getOSName(),
                                         Triple.getEnvironmentName()),
                            Opts) {
    IsRenderScriptTarget = true;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__RENDERSCRIPT__");
    AArch64leTargetInfo::getTargetDefines(Opts, Builder);
  }
};

/// Information about a specific microcontroller.
struct MCUInfo {
  const char *Name;
  const char *DefineName;
};

// This list should be kept up-to-date with AVRDevices.td in LLVM.
static ArrayRef<MCUInfo> AVRMcus = {
  { "at90s1200", "__AVR_AT90S1200__" },
  { "attiny11", "__AVR_ATtiny11__" },
  { "attiny12", "__AVR_ATtiny12__" },
  { "attiny15", "__AVR_ATtiny15__" },
  { "attiny28", "__AVR_ATtiny28__" },
  { "at90s2313", "__AVR_AT90S2313__" },
  { "at90s2323", "__AVR_AT90S2323__" },
  { "at90s2333", "__AVR_AT90S2333__" },
  { "at90s2343", "__AVR_AT90S2343__" },
  { "attiny22", "__AVR_ATtiny22__" },
  { "attiny26", "__AVR_ATtiny26__" },
  { "at86rf401", "__AVR_AT86RF401__" },
  { "at90s4414", "__AVR_AT90S4414__" },
  { "at90s4433", "__AVR_AT90S4433__" },
  { "at90s4434", "__AVR_AT90S4434__" },
  { "at90s8515", "__AVR_AT90S8515__" },
  { "at90c8534", "__AVR_AT90c8534__" },
  { "at90s8535", "__AVR_AT90S8535__" },
  { "ata5272", "__AVR_ATA5272__" },
  { "attiny13", "__AVR_ATtiny13__" },
  { "attiny13a", "__AVR_ATtiny13A__" },
  { "attiny2313", "__AVR_ATtiny2313__" },
  { "attiny2313a", "__AVR_ATtiny2313A__" },
  { "attiny24", "__AVR_ATtiny24__" },
  { "attiny24a", "__AVR_ATtiny24A__" },
  { "attiny4313", "__AVR_ATtiny4313__" },
  { "attiny44", "__AVR_ATtiny44__" },
  { "attiny44a", "__AVR_ATtiny44A__" },
  { "attiny84", "__AVR_ATtiny84__" },
  { "attiny84a", "__AVR_ATtiny84A__" },
  { "attiny25", "__AVR_ATtiny25__" },
  { "attiny45", "__AVR_ATtiny45__" },
  { "attiny85", "__AVR_ATtiny85__" },
  { "attiny261", "__AVR_ATtiny261__" },
  { "attiny261a", "__AVR_ATtiny261A__" },
  { "attiny461", "__AVR_ATtiny461__" },
  { "attiny461a", "__AVR_ATtiny461A__" },
  { "attiny861", "__AVR_ATtiny861__" },
  { "attiny861a", "__AVR_ATtiny861A__" },
  { "attiny87", "__AVR_ATtiny87__" },
  { "attiny43u", "__AVR_ATtiny43U__" },
  { "attiny48", "__AVR_ATtiny48__" },
  { "attiny88", "__AVR_ATtiny88__" },
  { "attiny828", "__AVR_ATtiny828__" },
  { "at43usb355", "__AVR_AT43USB355__" },
  { "at76c711", "__AVR_AT76C711__" },
  { "atmega103", "__AVR_ATmega103__" },
  { "at43usb320", "__AVR_AT43USB320__" },
  { "attiny167", "__AVR_ATtiny167__" },
  { "at90usb82", "__AVR_AT90USB82__" },
  { "at90usb162", "__AVR_AT90USB162__" },
  { "ata5505", "__AVR_ATA5505__" },
  { "atmega8u2", "__AVR_ATmega8U2__" },
  { "atmega16u2", "__AVR_ATmega16U2__" },
  { "atmega32u2", "__AVR_ATmega32U2__" },
  { "attiny1634", "__AVR_ATtiny1634__" },
  { "atmega8", "__AVR_ATmega8__" },
  { "ata6289", "__AVR_ATA6289__" },
  { "atmega8a", "__AVR_ATmega8A__" },
  { "ata6285", "__AVR_ATA6285__" },
  { "ata6286", "__AVR_ATA6286__" },
  { "atmega48", "__AVR_ATmega48__" },
  { "atmega48a", "__AVR_ATmega48A__" },
  { "atmega48pa", "__AVR_ATmega48PA__" },
  { "atmega48p", "__AVR_ATmega48P__" },
  { "atmega88", "__AVR_ATmega88__" },
  { "atmega88a", "__AVR_ATmega88A__" },
  { "atmega88p", "__AVR_ATmega88P__" },
  { "atmega88pa", "__AVR_ATmega88PA__" },
  { "atmega8515", "__AVR_ATmega8515__" },
  { "atmega8535", "__AVR_ATmega8535__" },
  { "atmega8hva", "__AVR_ATmega8HVA__" },
  { "at90pwm1", "__AVR_AT90PWM1__" },
  { "at90pwm2", "__AVR_AT90PWM2__" },
  { "at90pwm2b", "__AVR_AT90PWM2B__" },
  { "at90pwm3", "__AVR_AT90PWM3__" },
  { "at90pwm3b", "__AVR_AT90PWM3B__" },
  { "at90pwm81", "__AVR_AT90PWM81__" },
  { "ata5790", "__AVR_ATA5790__" },
  { "ata5795", "__AVR_ATA5795__" },
  { "atmega16", "__AVR_ATmega16__" },
  { "atmega16a", "__AVR_ATmega16A__" },
  { "atmega161", "__AVR_ATmega161__" },
  { "atmega162", "__AVR_ATmega162__" },
  { "atmega163", "__AVR_ATmega163__" },
  { "atmega164a", "__AVR_ATmega164A__" },
  { "atmega164p", "__AVR_ATmega164P__" },
  { "atmega164pa", "__AVR_ATmega164PA__" },
  { "atmega165", "__AVR_ATmega165__" },
  { "atmega165a", "__AVR_ATmega165A__" },
  { "atmega165p", "__AVR_ATmega165P__" },
  { "atmega165pa", "__AVR_ATmega165PA__" },
  { "atmega168", "__AVR_ATmega168__" },
  { "atmega168a", "__AVR_ATmega168A__" },
  { "atmega168p", "__AVR_ATmega168P__" },
  { "atmega168pa", "__AVR_ATmega168PA__" },
  { "atmega169", "__AVR_ATmega169__" },
  { "atmega169a", "__AVR_ATmega169A__" },
  { "atmega169p", "__AVR_ATmega169P__" },
  { "atmega169pa", "__AVR_ATmega169PA__" },
  { "atmega32", "__AVR_ATmega32__" },
  { "atmega32a", "__AVR_ATmega32A__" },
  { "atmega323", "__AVR_ATmega323__" },
  { "atmega324a", "__AVR_ATmega324A__" },
  { "atmega324p", "__AVR_ATmega324P__" },
  { "atmega324pa", "__AVR_ATmega324PA__" },
  { "atmega325", "__AVR_ATmega325__" },
  { "atmega325a", "__AVR_ATmega325A__" },
  { "atmega325p", "__AVR_ATmega325P__" },
  { "atmega325pa", "__AVR_ATmega325PA__" },
  { "atmega3250", "__AVR_ATmega3250__" },
  { "atmega3250a", "__AVR_ATmega3250A__" },
  { "atmega3250p", "__AVR_ATmega3250P__" },
  { "atmega3250pa", "__AVR_ATmega3250PA__" },
  { "atmega328", "__AVR_ATmega328__" },
  { "atmega328p", "__AVR_ATmega328P__" },
  { "atmega329", "__AVR_ATmega329__" },
  { "atmega329a", "__AVR_ATmega329A__" },
  { "atmega329p", "__AVR_ATmega329P__" },
  { "atmega329pa", "__AVR_ATmega329PA__" },
  { "atmega3290", "__AVR_ATmega3290__" },
  { "atmega3290a", "__AVR_ATmega3290A__" },
  { "atmega3290p", "__AVR_ATmega3290P__" },
  { "atmega3290pa", "__AVR_ATmega3290PA__" },
  { "atmega406", "__AVR_ATmega406__" },
  { "atmega64", "__AVR_ATmega64__" },
  { "atmega64a", "__AVR_ATmega64A__" },
  { "atmega640", "__AVR_ATmega640__" },
  { "atmega644", "__AVR_ATmega644__" },
  { "atmega644a", "__AVR_ATmega644A__" },
  { "atmega644p", "__AVR_ATmega644P__" },
  { "atmega644pa", "__AVR_ATmega644PA__" },
  { "atmega645", "__AVR_ATmega645__" },
  { "atmega645a", "__AVR_ATmega645A__" },
  { "atmega645p", "__AVR_ATmega645P__" },
  { "atmega649", "__AVR_ATmega649__" },
  { "atmega649a", "__AVR_ATmega649A__" },
  { "atmega649p", "__AVR_ATmega649P__" },
  { "atmega6450", "__AVR_ATmega6450__" },
  { "atmega6450a", "__AVR_ATmega6450A__" },
  { "atmega6450p", "__AVR_ATmega6450P__" },
  { "atmega6490", "__AVR_ATmega6490__" },
  { "atmega6490a", "__AVR_ATmega6490A__" },
  { "atmega6490p", "__AVR_ATmega6490P__" },
  { "atmega64rfr2", "__AVR_ATmega64RFR2__" },
  { "atmega644rfr2", "__AVR_ATmega644RFR2__" },
  { "atmega16hva", "__AVR_ATmega16HVA__" },
  { "atmega16hva2", "__AVR_ATmega16HVA2__" },
  { "atmega16hvb", "__AVR_ATmega16HVB__" },
  { "atmega16hvbrevb", "__AVR_ATmega16HVBREVB__" },
  { "atmega32hvb", "__AVR_ATmega32HVB__" },
  { "atmega32hvbrevb", "__AVR_ATmega32HVBREVB__" },
  { "atmega64hve", "__AVR_ATmega64HVE__" },
  { "at90can32", "__AVR_AT90CAN32__" },
  { "at90can64", "__AVR_AT90CAN64__" },
  { "at90pwm161", "__AVR_AT90PWM161__" },
  { "at90pwm216", "__AVR_AT90PWM216__" },
  { "at90pwm316", "__AVR_AT90PWM316__" },
  { "atmega32c1", "__AVR_ATmega32C1__" },
  { "atmega64c1", "__AVR_ATmega64C1__" },
  { "atmega16m1", "__AVR_ATmega16M1__" },
  { "atmega32m1", "__AVR_ATmega32M1__" },
  { "atmega64m1", "__AVR_ATmega64M1__" },
  { "atmega16u4", "__AVR_ATmega16U4__" },
  { "atmega32u4", "__AVR_ATmega32U4__" },
  { "atmega32u6", "__AVR_ATmega32U6__" },
  { "at90usb646", "__AVR_AT90USB646__" },
  { "at90usb647", "__AVR_AT90USB647__" },
  { "at90scr100", "__AVR_AT90SCR100__" },
  { "at94k", "__AVR_AT94K__" },
  { "m3000", "__AVR_AT000__" },
  { "atmega128", "__AVR_ATmega128__" },
  { "atmega128a", "__AVR_ATmega128A__" },
  { "atmega1280", "__AVR_ATmega1280__" },
  { "atmega1281", "__AVR_ATmega1281__" },
  { "atmega1284", "__AVR_ATmega1284__" },
  { "atmega1284p", "__AVR_ATmega1284P__" },
  { "atmega128rfa1", "__AVR_ATmega128RFA1__" },
  { "atmega128rfr2", "__AVR_ATmega128RFR2__" },
  { "atmega1284rfr2", "__AVR_ATmega1284RFR2__" },
  { "at90can128", "__AVR_AT90CAN128__" },
  { "at90usb1286", "__AVR_AT90USB1286__" },
  { "at90usb1287", "__AVR_AT90USB1287__" },
  { "atmega2560", "__AVR_ATmega2560__" },
  { "atmega2561", "__AVR_ATmega2561__" },
  { "atmega256rfr2", "__AVR_ATmega256RFR2__" },
  { "atmega2564rfr2", "__AVR_ATmega2564RFR2__" },
  { "atxmega16a4", "__AVR_ATxmega16A4__" },
  { "atxmega16a4u", "__AVR_ATxmega16a4U__" },
  { "atxmega16c4", "__AVR_ATxmega16C4__" },
  { "atxmega16d4", "__AVR_ATxmega16D4__" },
  { "atxmega32a4", "__AVR_ATxmega32A4__" },
  { "atxmega32a4u", "__AVR_ATxmega32A4U__" },
  { "atxmega32c4", "__AVR_ATxmega32C4__" },
  { "atxmega32d4", "__AVR_ATxmega32D4__" },
  { "atxmega32e5", "__AVR_ATxmega32E5__" },
  { "atxmega16e5", "__AVR_ATxmega16E5__" },
  { "atxmega8e5", "__AVR_ATxmega8E5__" },
  { "atxmega32x1", "__AVR_ATxmega32X1__" },
  { "atxmega64a3", "__AVR_ATxmega64A3__" },
  { "atxmega64a3u", "__AVR_ATxmega64A3U__" },
  { "atxmega64a4u", "__AVR_ATxmega64A4U__" },
  { "atxmega64b1", "__AVR_ATxmega64B1__" },
  { "atxmega64b3", "__AVR_ATxmega64B3__" },
  { "atxmega64c3", "__AVR_ATxmega64C3__" },
  { "atxmega64d3", "__AVR_ATxmega64D3__" },
  { "atxmega64d4", "__AVR_ATxmega64D4__" },
  { "atxmega64a1", "__AVR_ATxmega64A1__" },
  { "atxmega64a1u", "__AVR_ATxmega64A1U__" },
  { "atxmega128a3", "__AVR_ATxmega128A3__" },
  { "atxmega128a3u", "__AVR_ATxmega128A3U__" },
  { "atxmega128b1", "__AVR_ATxmega128B1__" },
  { "atxmega128b3", "__AVR_ATxmega128B3__" },
  { "atxmega128c3", "__AVR_ATxmega128C3__" },
  { "atxmega128d3", "__AVR_ATxmega128D3__" },
  { "atxmega128d4", "__AVR_ATxmega128D4__" },
  { "atxmega192a3", "__AVR_ATxmega192A3__" },
  { "atxmega192a3u", "__AVR_ATxmega192A3U__" },
  { "atxmega192c3", "__AVR_ATxmega192C3__" },
  { "atxmega192d3", "__AVR_ATxmega192D3__" },
  { "atxmega256a3", "__AVR_ATxmega256A3__" },
  { "atxmega256a3u", "__AVR_ATxmega256A3U__" },
  { "atxmega256a3b", "__AVR_ATxmega256A3B__" },
  { "atxmega256a3bu", "__AVR_ATxmega256A3BU__" },
  { "atxmega256c3", "__AVR_ATxmega256C3__" },
  { "atxmega256d3", "__AVR_ATxmega256D3__" },
  { "atxmega384c3", "__AVR_ATxmega384C3__" },
  { "atxmega384d3", "__AVR_ATxmega384D3__" },
  { "atxmega128a1", "__AVR_ATxmega128A1__" },
  { "atxmega128a1u", "__AVR_ATxmega128A1U__" },
  { "atxmega128a4u", "__AVR_ATxmega128a4U__" },
  { "attiny4", "__AVR_ATtiny4__" },
  { "attiny5", "__AVR_ATtiny5__" },
  { "attiny9", "__AVR_ATtiny9__" },
  { "attiny10", "__AVR_ATtiny10__" },
  { "attiny20", "__AVR_ATtiny20__" },
  { "attiny40", "__AVR_ATtiny40__" },
  { "attiny102", "__AVR_ATtiny102__" },
  { "attiny104", "__AVR_ATtiny104__" },
};

// AVR Target
class AVRTargetInfo : public TargetInfo {
public:
  AVRTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    TLSSupported = false;
    PointerWidth = 16;
    PointerAlign = 8;
    IntWidth = 16;
    IntAlign = 8;
    LongWidth = 32;
    LongAlign = 8;
    LongLongWidth = 64;
    LongLongAlign = 8;
    SuitableAlign = 8;
    DefaultAlignForAttributeAligned = 8;
    HalfWidth = 16;
    HalfAlign = 8;
    FloatWidth = 32;
    FloatAlign = 8;
    DoubleWidth = 32;
    DoubleAlign = 8;
    DoubleFormat = &llvm::APFloat::IEEEsingle();
    LongDoubleWidth = 32;
    LongDoubleAlign = 8;
    LongDoubleFormat = &llvm::APFloat::IEEEsingle();
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
    Char16Type = UnsignedInt;
    WCharType = SignedInt;
    WIntType = SignedInt;
    Char32Type = UnsignedLong;
    SigAtomicType = SignedChar;
    resetDataLayout("e-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8");
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("AVR");
    Builder.defineMacro("__AVR");
    Builder.defineMacro("__AVR__");

    if (!this->CPU.empty()) {
      auto It = std::find_if(AVRMcus.begin(), AVRMcus.end(),
        [&](const MCUInfo &Info) { return Info.Name == this->CPU; });

      if (It != AVRMcus.end())
        Builder.defineMacro(It->DefineName);
    }
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return None;
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  const char *getClobbers() const override {
    return "";
  }

  ArrayRef<const char *> getGCCRegNames() const override {
    static const char * const GCCRegNames[] = {
      "r0",   "r1",   "r2",   "r3",   "r4",   "r5",   "r6",   "r7",
      "r8",   "r9",   "r10",  "r11",  "r12",  "r13",  "r14",  "r15",
      "r16",  "r17",  "r18",  "r19",  "r20",  "r21",  "r22",  "r23",
      "r24",  "r25",  "X",    "Y",    "Z",    "SP"
    };
    return llvm::makeArrayRef(GCCRegNames);
  }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }

  ArrayRef<TargetInfo::AddlRegName> getGCCAddlRegNames() const override {
    static const TargetInfo::AddlRegName AddlRegNames[] = {
      { { "r26", "r27"}, 26 },
      { { "r28", "r29"}, 27 },
      { { "r30", "r31"}, 28 },
      { { "SPL", "SPH"}, 29 },
    };
    return llvm::makeArrayRef(AddlRegNames);
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    // There aren't any multi-character AVR specific constraints.
    if (StringRef(Name).size() > 1) return false;

    switch (*Name) {
      default: return false;
      case 'a': // Simple upper registers
      case 'b': // Base pointer registers pairs
      case 'd': // Upper register
      case 'l': // Lower registers
      case 'e': // Pointer register pairs
      case 'q': // Stack pointer register
      case 'r': // Any register
      case 'w': // Special upper register pairs
      case 't': // Temporary register
      case 'x': case 'X': // Pointer register pair X
      case 'y': case 'Y': // Pointer register pair Y
      case 'z': case 'Z': // Pointer register pair Z
        Info.setAllowsRegister();
        return true;
      case 'I': // 6-bit positive integer constant
        Info.setRequiresImmediate(0, 63);
        return true;
      case 'J': // 6-bit negative integer constant
        Info.setRequiresImmediate(-63, 0);
        return true;
      case 'K': // Integer constant (Range: 2)
        Info.setRequiresImmediate(2);
        return true;
      case 'L': // Integer constant (Range: 0)
        Info.setRequiresImmediate(0);
        return true;
      case 'M': // 8-bit integer constant
        Info.setRequiresImmediate(0, 0xff);
        return true;
      case 'N': // Integer constant (Range: -1)
        Info.setRequiresImmediate(-1);
        return true;
      case 'O': // Integer constant (Range: 8, 16, 24)
        Info.setRequiresImmediate({8, 16, 24});
        return true;
      case 'P': // Integer constant (Range: 1)
        Info.setRequiresImmediate(1);
        return true;
      case 'R': // Integer constant (Range: -6 to 5)
        Info.setRequiresImmediate(-6, 5);
        return true;
      case 'G': // Floating point constant
      case 'Q': // A memory address based on Y or Z pointer with displacement.
        return true;
    }

    return false;
  }

  IntType getIntTypeByWidth(unsigned BitWidth,
                            bool IsSigned) const final {
    // AVR prefers int for 16-bit integers.
    return BitWidth == 16 ? (IsSigned ? SignedInt : UnsignedInt)
                          : TargetInfo::getIntTypeByWidth(BitWidth, IsSigned);
  }

  IntType getLeastIntTypeByWidth(unsigned BitWidth,
                                 bool IsSigned) const final {
    // AVR uses int for int_least16_t and int_fast16_t.
    return BitWidth == 16
               ? (IsSigned ? SignedInt : UnsignedInt)
               : TargetInfo::getLeastIntTypeByWidth(BitWidth, IsSigned);
  }

  bool setCPU(const std::string &Name) override {
    bool IsFamily = llvm::StringSwitch<bool>(Name)
      .Case("avr1", true)
      .Case("avr2", true)
      .Case("avr25", true)
      .Case("avr3", true)
      .Case("avr31", true)
      .Case("avr35", true)
      .Case("avr4", true)
      .Case("avr5", true)
      .Case("avr51", true)
      .Case("avr6", true)
      .Case("avrxmega1", true)
      .Case("avrxmega2", true)
      .Case("avrxmega3", true)
      .Case("avrxmega4", true)
      .Case("avrxmega5", true)
      .Case("avrxmega6", true)
      .Case("avrxmega7", true)
      .Case("avrtiny", true)
      .Default(false);

    if (IsFamily) this->CPU = Name;

    bool IsMCU = std::find_if(AVRMcus.begin(), AVRMcus.end(),
      [&](const MCUInfo &Info) { return Info.Name == Name; }) != AVRMcus.end();

    if (IsMCU) this->CPU = Name;

    return IsFamily || IsMCU;
  }

protected:
  std::string CPU;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

static TargetInfo *AllocateTarget(const llvm::Triple &Triple,
                                  const TargetOptions &Opts) {
  llvm::Triple::OSType os = Triple.getOS();

  switch (Triple.getArch()) {
  default:
    return nullptr;

  case llvm::Triple::xcore:
    return new XCoreTargetInfo(Triple, Opts);

  case llvm::Triple::hexagon:
    return new HexagonTargetInfo(Triple, Opts);

  case llvm::Triple::lanai:
    return new LanaiTargetInfo(Triple, Opts);

  case llvm::Triple::aarch64:
    if (Triple.isOSDarwin())
      return new DarwinAArch64TargetInfo(Triple, Opts);

    switch (os) {
    case llvm::Triple::CloudABI:
      return new CloudABITargetInfo<AArch64leTargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<AArch64leTargetInfo>(Triple, Opts);
    case llvm::Triple::Fuchsia:
      return new FuchsiaTargetInfo<AArch64leTargetInfo>(Triple, Opts);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<AArch64leTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<AArch64leTargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<AArch64leTargetInfo>(Triple, Opts);
    case llvm::Triple::Win32:
      return new MicrosoftARM64TargetInfo(Triple, Opts);
    default:
      return new AArch64leTargetInfo(Triple, Opts);
    }

  case llvm::Triple::aarch64_be:
    switch (os) {
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<AArch64beTargetInfo>(Triple, Opts);
    case llvm::Triple::Fuchsia:
      return new FuchsiaTargetInfo<AArch64beTargetInfo>(Triple, Opts);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<AArch64beTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<AArch64beTargetInfo>(Triple, Opts);
    default:
      return new AArch64beTargetInfo(Triple, Opts);
    }

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    if (Triple.isOSBinFormatMachO())
      return new DarwinARMTargetInfo(Triple, Opts);

    switch (os) {
    case llvm::Triple::CloudABI:
      return new CloudABITargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::Bitrig:
      return new BitrigTargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::NaCl:
      return new NaClTargetInfo<ARMleTargetInfo>(Triple, Opts);
    case llvm::Triple::Win32:
      switch (Triple.getEnvironment()) {
      case llvm::Triple::Cygnus:
        return new CygwinARMTargetInfo(Triple, Opts);
      case llvm::Triple::GNU:
        return new MinGWARMTargetInfo(Triple, Opts);
      case llvm::Triple::Itanium:
        return new ItaniumWindowsARMleTargetInfo(Triple, Opts);
      case llvm::Triple::MSVC:
      default: // Assume MSVC for unknown environments
        return new MicrosoftARMleTargetInfo(Triple, Opts);
      }
    default:
      return new ARMleTargetInfo(Triple, Opts);
    }

  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb:
    if (Triple.isOSDarwin())
      return new DarwinARMTargetInfo(Triple, Opts);

    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<ARMbeTargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<ARMbeTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<ARMbeTargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<ARMbeTargetInfo>(Triple, Opts);
    case llvm::Triple::Bitrig:
      return new BitrigTargetInfo<ARMbeTargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<ARMbeTargetInfo>(Triple, Opts);
    case llvm::Triple::NaCl:
      return new NaClTargetInfo<ARMbeTargetInfo>(Triple, Opts);
    default:
      return new ARMbeTargetInfo(Triple, Opts);
    }

  case llvm::Triple::avr:
    return new AVRTargetInfo(Triple, Opts);
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
    return new BPFTargetInfo(Triple, Opts);

  case llvm::Triple::msp430:
    return new MSP430TargetInfo(Triple, Opts);

  case llvm::Triple::nios2:
    return new LinuxTargetInfo<Nios2TargetInfo>(Triple, Opts);

  case llvm::Triple::mips:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    default:
      return new MipsTargetInfo(Triple, Opts);
    }

  case llvm::Triple::mipsel:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::NaCl:
      return new NaClTargetInfo<NaClMips32TargetInfo>(Triple, Opts);
    default:
      return new MipsTargetInfo(Triple, Opts);
    }

  case llvm::Triple::mips64:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    default:
      return new MipsTargetInfo(Triple, Opts);
    }

  case llvm::Triple::mips64el:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<MipsTargetInfo>(Triple, Opts);
    default:
      return new MipsTargetInfo(Triple, Opts);
    }

  case llvm::Triple::le32:
    switch (os) {
    case llvm::Triple::NaCl:
      return new NaClTargetInfo<PNaClTargetInfo>(Triple, Opts);
    default:
      return nullptr;
    }

  case llvm::Triple::le64:
    return new Le64TargetInfo(Triple, Opts);

  case llvm::Triple::ppc:
    if (Triple.isOSDarwin())
      return new DarwinPPC32TargetInfo(Triple, Opts);
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<PPC32TargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<PPC32TargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<PPC32TargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<PPC32TargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<PPC32TargetInfo>(Triple, Opts);
    default:
      return new PPC32TargetInfo(Triple, Opts);
    }

  case llvm::Triple::ppc64:
    if (Triple.isOSDarwin())
      return new DarwinPPC64TargetInfo(Triple, Opts);
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<PPC64TargetInfo>(Triple, Opts);
    case llvm::Triple::Lv2:
      return new PS3PPUTargetInfo<PPC64TargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<PPC64TargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<PPC64TargetInfo>(Triple, Opts);
    default:
      return new PPC64TargetInfo(Triple, Opts);
    }

  case llvm::Triple::ppc64le:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<PPC64TargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<PPC64TargetInfo>(Triple, Opts);
    default:
      return new PPC64TargetInfo(Triple, Opts);
    }

  case llvm::Triple::nvptx:
    return new NVPTXTargetInfo(Triple, Opts, /*TargetPointerWidth=*/32);
  case llvm::Triple::nvptx64:
    return new NVPTXTargetInfo(Triple, Opts, /*TargetPointerWidth=*/64);

  case llvm::Triple::amdgcn:
  case llvm::Triple::r600:
    return new AMDGPUTargetInfo(Triple, Opts);

  case llvm::Triple::sparc:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<SparcV8TargetInfo>(Triple, Opts);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<SparcV8TargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<SparcV8TargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<SparcV8TargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<SparcV8TargetInfo>(Triple, Opts);
    default:
      return new SparcV8TargetInfo(Triple, Opts);
    }

  // The 'sparcel' architecture copies all the above cases except for Solaris.
  case llvm::Triple::sparcel:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<SparcV8elTargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<SparcV8elTargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<SparcV8elTargetInfo>(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSTargetInfo<SparcV8elTargetInfo>(Triple, Opts);
    default:
      return new SparcV8elTargetInfo(Triple, Opts);
    }

  case llvm::Triple::sparcv9:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<SparcV9TargetInfo>(Triple, Opts);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<SparcV9TargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<SparcV9TargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDTargetInfo<SparcV9TargetInfo>(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<SparcV9TargetInfo>(Triple, Opts);
    default:
      return new SparcV9TargetInfo(Triple, Opts);
    }

  case llvm::Triple::systemz:
    switch (os) {
    case llvm::Triple::Linux:
      return new LinuxTargetInfo<SystemZTargetInfo>(Triple, Opts);
    default:
      return new SystemZTargetInfo(Triple, Opts);
    }

  case llvm::Triple::tce:
    return new TCETargetInfo(Triple, Opts);

  case llvm::Triple::tcele:
    return new TCELETargetInfo(Triple, Opts);

  case llvm::Triple::x86:
    if (Triple.isOSDarwin())
      return new DarwinI386TargetInfo(Triple, Opts);

    switch (os) {
    case llvm::Triple::Ananas:
      return new AnanasTargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::CloudABI:
      return new CloudABITargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::Linux: {
      switch (Triple.getEnvironment()) {
      default:
        return new LinuxTargetInfo<X86_32TargetInfo>(Triple, Opts);
      case llvm::Triple::Android:
        return new AndroidX86_32TargetInfo(Triple, Opts);
      }
    }
    case llvm::Triple::DragonFly:
      return new DragonFlyBSDTargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDI386TargetInfo(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDI386TargetInfo(Triple, Opts);
    case llvm::Triple::Bitrig:
      return new BitrigI386TargetInfo(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::KFreeBSD:
      return new KFreeBSDTargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::Minix:
      return new MinixTargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::Win32: {
      switch (Triple.getEnvironment()) {
      case llvm::Triple::Cygnus:
        return new CygwinX86_32TargetInfo(Triple, Opts);
      case llvm::Triple::GNU:
        return new MinGWX86_32TargetInfo(Triple, Opts);
      case llvm::Triple::Itanium:
      case llvm::Triple::MSVC:
      default: // Assume MSVC for unknown environments
        return new MicrosoftX86_32TargetInfo(Triple, Opts);
      }
    }
    case llvm::Triple::Haiku:
      return new HaikuX86_32TargetInfo(Triple, Opts);
    case llvm::Triple::RTEMS:
      return new RTEMSX86_32TargetInfo(Triple, Opts);
    case llvm::Triple::NaCl:
      return new NaClTargetInfo<X86_32TargetInfo>(Triple, Opts);
    case llvm::Triple::ELFIAMCU:
      return new MCUX86_32TargetInfo(Triple, Opts);
    default:
      return new X86_32TargetInfo(Triple, Opts);
    }

  case llvm::Triple::x86_64:
    if (Triple.isOSDarwin() || Triple.isOSBinFormatMachO())
      return new DarwinX86_64TargetInfo(Triple, Opts);

    switch (os) {
    case llvm::Triple::Ananas:
      return new AnanasTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::CloudABI:
      return new CloudABITargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::Linux: {
      switch (Triple.getEnvironment()) {
      default:
        return new LinuxTargetInfo<X86_64TargetInfo>(Triple, Opts);
      case llvm::Triple::Android:
        return new AndroidX86_64TargetInfo(Triple, Opts);
      }
    }
    case llvm::Triple::DragonFly:
      return new DragonFlyBSDTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::NetBSD:
      return new NetBSDTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::OpenBSD:
      return new OpenBSDX86_64TargetInfo(Triple, Opts);
    case llvm::Triple::Bitrig:
      return new BitrigX86_64TargetInfo(Triple, Opts);
    case llvm::Triple::FreeBSD:
      return new FreeBSDTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::Fuchsia:
      return new FuchsiaTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::KFreeBSD:
      return new KFreeBSDTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::Solaris:
      return new SolarisTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::Win32: {
      switch (Triple.getEnvironment()) {
      case llvm::Triple::Cygnus:
        return new CygwinX86_64TargetInfo(Triple, Opts);
      case llvm::Triple::GNU:
        return new MinGWX86_64TargetInfo(Triple, Opts);
      case llvm::Triple::MSVC:
      default: // Assume MSVC for unknown environments
        return new MicrosoftX86_64TargetInfo(Triple, Opts);
      }
    }
    case llvm::Triple::Haiku:
      return new HaikuTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::NaCl:
      return new NaClTargetInfo<X86_64TargetInfo>(Triple, Opts);
    case llvm::Triple::PS4:
      return new PS4OSTargetInfo<X86_64TargetInfo>(Triple, Opts);
    default:
      return new X86_64TargetInfo(Triple, Opts);
    }

  case llvm::Triple::spir: {
    if (Triple.getOS() != llvm::Triple::UnknownOS ||
        Triple.getEnvironment() != llvm::Triple::UnknownEnvironment)
      return nullptr;
    return new SPIR32TargetInfo(Triple, Opts);
  }
  case llvm::Triple::spir64: {
    if (Triple.getOS() != llvm::Triple::UnknownOS ||
        Triple.getEnvironment() != llvm::Triple::UnknownEnvironment)
      return nullptr;
    return new SPIR64TargetInfo(Triple, Opts);
  }
  case llvm::Triple::wasm32:
    if (Triple.getSubArch() != llvm::Triple::NoSubArch ||
        Triple.getVendor() != llvm::Triple::UnknownVendor ||
        Triple.getOS() != llvm::Triple::UnknownOS ||
        Triple.getEnvironment() != llvm::Triple::UnknownEnvironment ||
        !(Triple.isOSBinFormatELF() || Triple.isOSBinFormatWasm()))
      return nullptr;
    return new WebAssemblyOSTargetInfo<WebAssembly32TargetInfo>(Triple, Opts);
  case llvm::Triple::wasm64:
    if (Triple.getSubArch() != llvm::Triple::NoSubArch ||
        Triple.getVendor() != llvm::Triple::UnknownVendor ||
        Triple.getOS() != llvm::Triple::UnknownOS ||
        Triple.getEnvironment() != llvm::Triple::UnknownEnvironment ||
        !(Triple.isOSBinFormatELF() || Triple.isOSBinFormatWasm()))
      return nullptr;
    return new WebAssemblyOSTargetInfo<WebAssembly64TargetInfo>(Triple, Opts);

  case llvm::Triple::renderscript32:
    return new LinuxTargetInfo<RenderScript32TargetInfo>(Triple, Opts);
  case llvm::Triple::renderscript64:
    return new LinuxTargetInfo<RenderScript64TargetInfo>(Triple, Opts);
  }
}

/// CreateTargetInfo - Return the target info object for the specified target
/// options.
TargetInfo *
TargetInfo::CreateTargetInfo(DiagnosticsEngine &Diags,
                             const std::shared_ptr<TargetOptions> &Opts) {
  llvm::Triple Triple(Opts->Triple);

  // Construct the target
  std::unique_ptr<TargetInfo> Target(AllocateTarget(Triple, *Opts));
  if (!Target) {
    Diags.Report(diag::err_target_unknown_triple) << Triple.str();
    return nullptr;
  }
  Target->TargetOpts = Opts;

  // Set the target CPU if specified.
  if (!Opts->CPU.empty() && !Target->setCPU(Opts->CPU)) {
    Diags.Report(diag::err_target_unknown_cpu) << Opts->CPU;
    return nullptr;
  }

  // Set the target ABI if specified.
  if (!Opts->ABI.empty() && !Target->setABI(Opts->ABI)) {
    Diags.Report(diag::err_target_unknown_abi) << Opts->ABI;
    return nullptr;
  }

  // Set the fp math unit.
  if (!Opts->FPMath.empty() && !Target->setFPMath(Opts->FPMath)) {
    Diags.Report(diag::err_target_unknown_fpmath) << Opts->FPMath;
    return nullptr;
  }

  // Compute the default target features, we need the target to handle this
  // because features may have dependencies on one another.
  llvm::StringMap<bool> Features;
  if (!Target->initFeatureMap(Features, Diags, Opts->CPU,
                              Opts->FeaturesAsWritten))
      return nullptr;

  // Add the features to the compile options.
  Opts->Features.clear();
  for (const auto &F : Features)
    Opts->Features.push_back((F.getValue() ? "+" : "-") + F.getKey().str());

  if (!Target->handleTargetFeatures(Opts->Features, Diags))
    return nullptr;

  Target->setSupportedOpenCLOpts();
  Target->setOpenCLExtensionOpts();

  if (!Target->validateTarget(Diags))
    return nullptr;

  return Target.release();
}
