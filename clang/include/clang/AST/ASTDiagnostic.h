//===--- ASTDiagnostic.h - Diagnostics for the AST library ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIAGNOSTICAST_H
#define LLVM_CLANG_DIAGNOSTICAST_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag {
    enum {
#define DIAG(ENUM,FLAGS,DEFAULT_MAPPING,DESC,GROUP,\
             SFINAE,ACCESS,NOWERROR,SHOWINSYSHEADER,CATEGORY) ENUM,
#define ASTSTART
#include "clang/Basic/DiagnosticASTKinds.inc"
#undef DIAG
      NUM_BUILTIN_AST_DIAGNOSTICS
    };
  }  // end namespace diag
  
  /// \brief DiagnosticsEngine argument formatting function for diagnostics that
  /// involve AST nodes.
  ///
  /// This function formats diagnostic arguments for various AST nodes, 
  /// including types, declaration names, nested name specifiers, and
  /// declaration contexts, into strings that can be printed as part of
  /// diagnostics. It is meant to be used as the argument to
  /// \c DiagnosticsEngine::SetArgToStringFn(), where the cookie is an \c
  /// ASTContext pointer.
  void FormatASTNodeDiagnosticArgument(
      DiagnosticsEngine::ArgumentKind Kind,
      intptr_t Val,
      const char *Modifier,
      unsigned ModLen,
      const char *Argument,
      unsigned ArgLen,
      const DiagnosticsEngine::ArgumentValue *PrevArgs,
      unsigned NumPrevArgs,
      SmallVectorImpl<char> &Output,
      void *Cookie,
      ArrayRef<intptr_t> QualTypeVals);
}  // end namespace clang

#endif
