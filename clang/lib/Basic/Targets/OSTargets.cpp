//===--- OSTargets.cpp - Implement OS target feature support --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements OS specific TargetInfo types.
//===----------------------------------------------------------------------===//

#include "OSTargets.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;
using namespace clang::targets;

namespace clang {
namespace targets {

void getDarwinDefines(MacroBuilder &Builder, const LangOptions &Opts,
                      const llvm::Triple &Triple, StringRef &PlatformName,
                      VersionTuple &PlatformMinVersion) {
  Builder.defineMacro("__APPLE_CC__", "6000");
  Builder.defineMacro("__APPLE__");
  Builder.defineMacro("__STDC_NO_THREADS__");

  // AddressSanitizer doesn't play well with source fortification, which is on
  // by default on Darwin.
  if (Opts.Sanitize.has(SanitizerKind::Address))
    Builder.defineMacro("_FORTIFY_SOURCE", "0");

  // Darwin defines __weak, __strong, and __unsafe_unretained even in C mode.
  if (!Opts.ObjC) {
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
    if (PlatformName == "ios" && Triple.isMacCatalystEnvironment())
      PlatformName = "maccatalyst";
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

  PlatformMinVersion = VersionTuple(Maj, Min, Rev);
}

static void addMinGWDefines(const llvm::Triple &Triple, const LangOptions &Opts,
                            MacroBuilder &Builder) {
  DefineStd(Builder, "WIN32", Opts);
  DefineStd(Builder, "WINNT", Opts);
  if (Triple.isArch64Bit()) {
    DefineStd(Builder, "WIN64", Opts);
    Builder.defineMacro("__MINGW64__");
  }
  Builder.defineMacro("__MSVCRT__");
  Builder.defineMacro("__MINGW32__");
  addCygMingDefines(Opts, Builder);
}

static void addVisualCDefines(const LangOptions &Opts, MacroBuilder &Builder) {
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
      if (Opts.CPlusPlus2b)
        Builder.defineMacro("_MSVC_LANG", "202004L");
      else if (Opts.CPlusPlus20)
        Builder.defineMacro("_MSVC_LANG", "202002L");
      else if (Opts.CPlusPlus17)
        Builder.defineMacro("_MSVC_LANG", "201703L");
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

  // Starting with VS 2022 17.1, MSVC predefines the below macro to inform
  // users of the execution character set defined at compile time.
  // The value given is the Windows Code Page Identifier:
  // https://docs.microsoft.com/en-us/windows/win32/intl/code-page-identifiers
  //
  // Clang currently only supports UTF-8, so we'll use 65001
  Builder.defineMacro("_MSVC_EXECUTION_CHARACTER_SET", "65001");
}

void addWindowsDefines(const llvm::Triple &Triple, const LangOptions &Opts,
                       MacroBuilder &Builder) {
  Builder.defineMacro("_WIN32");
  if (Triple.isArch64Bit())
    Builder.defineMacro("_WIN64");
  if (Triple.isWindowsGNUEnvironment())
    addMinGWDefines(Triple, Opts, Builder);
  else if (Triple.isKnownWindowsMSVCEnvironment() ||
           (Triple.isWindowsItaniumEnvironment() && Opts.MSVCCompat))
    addVisualCDefines(Opts, Builder);
}

} // namespace targets
} // namespace clang
