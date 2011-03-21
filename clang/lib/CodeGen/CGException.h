//===-- CGException.h - Classes for exceptions IR generation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes support the generation of LLVM IR for exceptions in
// C++ and Objective C.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGEXCEPTION_H
#define CLANG_CODEGEN_CGEXCEPTION_H

#include "llvm/ADT/StringRef.h"

namespace clang {
class LangOptions;

namespace CodeGen {

/// The exceptions personality for a function.  When 
class EHPersonality {
  llvm::StringRef PersonalityFn;

  // If this is non-null, this personality requires a non-standard
  // function for rethrowing an exception after a catchall cleanup.
  // This function must have prototype void(void*).
  llvm::StringRef CatchallRethrowFn;

  EHPersonality(llvm::StringRef PersonalityFn,
                llvm::StringRef CatchallRethrowFn = llvm::StringRef())
    : PersonalityFn(PersonalityFn),
      CatchallRethrowFn(CatchallRethrowFn) {}

public:
  static const EHPersonality &get(const LangOptions &Lang);
  static const EHPersonality GNU_C;
  static const EHPersonality GNU_C_SJLJ;
  static const EHPersonality GNU_ObjC;
  static const EHPersonality GNU_ObjCXX;
  static const EHPersonality NeXT_ObjC;
  static const EHPersonality GNU_CPlusPlus;
  static const EHPersonality GNU_CPlusPlus_SJLJ;

  llvm::StringRef getPersonalityFnName() const { return PersonalityFn; }
  llvm::StringRef getCatchallRethrowFnName() const { return CatchallRethrowFn; }
};

}
}

#endif
