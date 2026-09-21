//===--- EHPersonality.h - Shared EH personality descriptions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the exception personality functions that both classic
// CodeGen and CIR CodeGen can select for a function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_EHPERSONALITY_H
#define LLVM_CLANG_CODEGENUTILS_EHPERSONALITY_H

namespace clang {
class CodeGenOptions;
class FunctionDecl;
class LangOptions;
class TargetInfo;
} // namespace clang

namespace clang::CodeGenUtils {

/// The exceptions personality for a function.
struct EHPersonality {
  const char *PersonalityFn = nullptr;

  // If this is non-null, this personality requires a non-standard
  // function for rethrowing an exception after a catchall cleanup.
  // This function must have prototype void(void*).
  const char *CatchallRethrowFn = nullptr;

  static const EHPersonality GNU_C;
  static const EHPersonality GNU_C_SJLJ;
  static const EHPersonality GNU_C_SEH;
  static const EHPersonality GNU_ObjC;
  static const EHPersonality GNU_ObjC_SJLJ;
  static const EHPersonality GNU_ObjC_SEH;
  static const EHPersonality GNUstep_ObjC;
  static const EHPersonality GNU_ObjCXX;
  static const EHPersonality NeXT_ObjC;
  static const EHPersonality GNU_CPlusPlus;
  static const EHPersonality GNU_CPlusPlus_SJLJ;
  static const EHPersonality GNU_CPlusPlus_SEH;
  static const EHPersonality MSVC_except_handler;
  static const EHPersonality MSVC_C_specific_handler;
  static const EHPersonality MSVC_CxxFrameHandler3;
  static const EHPersonality GNU_Wasm_CPlusPlus;
  static const EHPersonality XL_CPlusPlus;
  static const EHPersonality ZOS_CPlusPlus;

  /// Does this personality use landingpads or the family of pad instructions
  /// designed to form funclets?
  bool usesFuncletPads() const {
    return isMSVCPersonality() || isWasmPersonality();
  }

  bool isMSVCPersonality() const {
    return this == &MSVC_except_handler || this == &MSVC_C_specific_handler ||
           this == &MSVC_CxxFrameHandler3;
  }

  bool isWasmPersonality() const { return this == &GNU_Wasm_CPlusPlus; }

  bool isMSVCXXPersonality() const { return this == &MSVC_CxxFrameHandler3; }
};

/// Selects the personality function to use for \p FD, or for the translation
/// unit as a whole when \p FD is null.
const EHPersonality &getEHPersonality(const TargetInfo &Target,
                                      const LangOptions &LangOpts,
                                      const CodeGenOptions &CGOpts,
                                      const FunctionDecl *FD);

/// Selects the personality function that plain C++ would use.  ObjC++ consults
/// this to decide whether it can fall back on the C++ personality.
const EHPersonality &getCXXEHPersonality(const TargetInfo &Target,
                                         const CodeGenOptions &CGOpts);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_EHPERSONALITY_H
