//===- VariadicMacroSupport.h - scope-guards etc. -*- C++ -*---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines support types to help with preprocessing variadic macro 
// (i.e. macros that use: ellipses __VA_ARGS__ ) definitions and 
// expansions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_VARIADICMACROSUPPORT_H
#define LLVM_CLANG_LEX_VARIADICMACROSUPPORT_H

#include "clang/Lex/Preprocessor.h"

namespace clang {

/// An RAII class that tracks when the Preprocessor starts and stops lexing the
/// definition of a (ISO C/C++) variadic macro.  As an example, this is useful
/// for unpoisoning and repoisoning certain identifiers (such as __VA_ARGS__)
/// that are only allowed in this context.  Also, being a friend of the
/// Preprocessor class allows it to access PP's cached identifiers directly (as
/// opposed to performing a lookup each time).
class VariadicMacroScopeGuard {
  const Preprocessor &PP;
  IdentifierInfo &Ident__VA_ARGS__;

public:
  VariadicMacroScopeGuard(const Preprocessor &P)
      : PP(P), Ident__VA_ARGS__(*PP.Ident__VA_ARGS__) {
    assert(Ident__VA_ARGS__.isPoisoned() && "__VA_ARGS__ should be poisoned "
                                            "outside an ISO C/C++ variadic "
                                            "macro definition!");
  }

  /// Client code should call this function just before the Preprocessor is
  /// about to Lex tokens from the definition of a variadic (ISO C/C++) macro.
  void enterScope() { Ident__VA_ARGS__.setIsPoisoned(false); }

  /// Client code should call this function as soon as the Preprocessor has
  /// either completed lexing the macro's definition tokens, or an error occured
  /// and the context is being exited.  This function is idempotent (might be
  /// explicitly called, and then reinvoked via the destructor).
  void exitScope() { Ident__VA_ARGS__.setIsPoisoned(true); }
  
  ~VariadicMacroScopeGuard() { exitScope(); }
};

}  // end namespace clang

#endif
