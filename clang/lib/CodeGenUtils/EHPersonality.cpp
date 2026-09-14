//===--- EHPersonality.cpp - Shared EH personality descriptions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/EHPersonality.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"

namespace clang::CodeGenUtils {

const EHPersonality EHPersonality::GNU_C = {"__gcc_personality_v0", nullptr};
const EHPersonality EHPersonality::GNU_C_SJLJ = {"__gcc_personality_sj0",
                                                 nullptr};
const EHPersonality EHPersonality::GNU_C_SEH = {"__gcc_personality_seh0",
                                                nullptr};
const EHPersonality EHPersonality::NeXT_ObjC = {"__objc_personality_v0",
                                                nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus = {"__gxx_personality_v0",
                                                    nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SJLJ = {
    "__gxx_personality_sj0", nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SEH = {
    "__gxx_personality_seh0", nullptr};
const EHPersonality EHPersonality::GNU_ObjC = {"__gnu_objc_personality_v0",
                                               "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SJLJ = {
    "__gnu_objc_personality_sj0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SEH = {
    "__gnu_objc_personality_seh0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjCXX = {
    "__gnustep_objcxx_personality_v0", nullptr};
const EHPersonality EHPersonality::GNUstep_ObjC = {
    "__gnustep_objc_personality_v0", nullptr};
const EHPersonality EHPersonality::MSVC_except_handler = {"_except_handler3",
                                                          nullptr};
const EHPersonality EHPersonality::MSVC_C_specific_handler = {
    "__C_specific_handler", nullptr};
const EHPersonality EHPersonality::MSVC_CxxFrameHandler3 = {
    "__CxxFrameHandler3", nullptr};
const EHPersonality EHPersonality::GNU_Wasm_CPlusPlus = {
    "__gxx_wasm_personality_v0", nullptr};
const EHPersonality EHPersonality::XL_CPlusPlus = {"__xlcxx_personality_v1",
                                                   nullptr};
const EHPersonality EHPersonality::ZOS_CPlusPlus = {"__zos_cxx_personality_v2",
                                                    nullptr};

static const EHPersonality &getCPersonality(const TargetInfo &Target,
                                            const CodeGenOptions &CGOpts) {
  const llvm::Triple &T = Target.getTriple();
  if (T.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (CGOpts.hasSjLjExceptions())
    return EHPersonality::GNU_C_SJLJ;
  if (CGOpts.hasDWARFExceptions())
    return EHPersonality::GNU_C;
  if (CGOpts.hasSEHExceptions())
    return EHPersonality::GNU_C_SEH;
  return EHPersonality::GNU_C;
}

static const EHPersonality &getObjCPersonality(const TargetInfo &Target,
                                               const CodeGenOptions &CGOpts,
                                               const LangOptions &L) {
  const llvm::Triple &T = Target.getTriple();
  if (T.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (T.isWasm())
    return EHPersonality::GNU_Wasm_CPlusPlus;

  switch (L.ObjCRuntime.getKind()) {
  case ObjCRuntime::FragileMacOSX:
    return getCPersonality(Target, CGOpts);
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return EHPersonality::NeXT_ObjC;
  case ObjCRuntime::GNUstep:
    if (T.isOSCygMing())
      return EHPersonality::GNU_CPlusPlus_SEH;
    if (L.ObjCRuntime.getVersion() >= VersionTuple(1, 7))
      return EHPersonality::GNUstep_ObjC;
    [[fallthrough]];
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    if (CGOpts.hasSjLjExceptions())
      return EHPersonality::GNU_ObjC_SJLJ;
    if (CGOpts.hasSEHExceptions())
      return EHPersonality::GNU_ObjC_SEH;
    return EHPersonality::GNU_ObjC;
  }
  llvm_unreachable("bad runtime kind");
}

const EHPersonality &getCXXEHPersonality(const TargetInfo &Target,
                                         const CodeGenOptions &CGOpts) {
  const llvm::Triple &T = Target.getTriple();
  if (T.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (T.isOSAIX())
    return EHPersonality::XL_CPlusPlus;
  if (CGOpts.hasSjLjExceptions())
    return EHPersonality::GNU_CPlusPlus_SJLJ;
  if (CGOpts.hasDWARFExceptions())
    return EHPersonality::GNU_CPlusPlus;
  if (CGOpts.hasSEHExceptions())
    return EHPersonality::GNU_CPlusPlus_SEH;
  if (CGOpts.hasWasmExceptions())
    return EHPersonality::GNU_Wasm_CPlusPlus;
  if (T.isOSzOS())
    return EHPersonality::ZOS_CPlusPlus;
  return EHPersonality::GNU_CPlusPlus;
}

/// Determines the personality function to use when both C++
/// and Objective-C exceptions are being caught.
static const EHPersonality &getObjCXXPersonality(const TargetInfo &Target,
                                                 const CodeGenOptions &CGOpts,
                                                 const LangOptions &L) {
  auto Triple = Target.getTriple();
  if (Triple.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (Triple.isWasm())
    return EHPersonality::GNU_Wasm_CPlusPlus;

  switch (L.ObjCRuntime.getKind()) {
  // In the fragile ABI, just use C++ exception handling and hope
  // they're not doing crazy exception mixing.
  case ObjCRuntime::FragileMacOSX:
    return getCXXEHPersonality(Target, CGOpts);

  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return getObjCPersonality(Target, CGOpts, L);

  case ObjCRuntime::GNUstep:
    if (Triple.isOSCygMing())
      return EHPersonality::GNU_CPlusPlus_SEH;
    return EHPersonality::GNU_ObjCXX;

  // The GCC runtime's personality function inherently doesn't support
  // mixed EH.  Use the ObjC personality just to avoid returning null.
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    return getObjCPersonality(Target, CGOpts, L);
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getSEHPersonalityMSVC(const llvm::Triple &T) {
  if (T.getArch() == llvm::Triple::x86)
    return EHPersonality::MSVC_except_handler;
  return EHPersonality::MSVC_C_specific_handler;
}

const EHPersonality &getEHPersonality(const TargetInfo &Target,
                                      const LangOptions &LangOpts,
                                      const CodeGenOptions &CGOpts,
                                      const FunctionDecl *FD) {
  // Functions using SEH get an SEH personality.
  if (FD && FD->usesSEHTry())
    return getSEHPersonalityMSVC(Target.getTriple());

  if (LangOpts.ObjC)
    return LangOpts.CPlusPlus ? getObjCXXPersonality(Target, CGOpts, LangOpts)
                              : getObjCPersonality(Target, CGOpts, LangOpts);
  return LangOpts.CPlusPlus ? getCXXEHPersonality(Target, CGOpts)
                            : getCPersonality(Target, CGOpts);
}

} // namespace clang::CodeGenUtils
