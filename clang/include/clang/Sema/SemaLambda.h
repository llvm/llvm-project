//===--- SemaLambda.h - Lambda Helper Functions --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides some common utility functions for processing
/// Lambdas.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMALAMBDA_H
#define LLVM_CLANG_SEMA_SEMALAMBDA_H
#include "clang/AST/ASTLambda.h"
#include "clang/Sema/ScopeInfo.h"
namespace clang {
class Sema;

/// \brief Examines the FunctionScopeInfo stack to determine the nearest
/// enclosing lambda (to the current lambda) that is 'capture-capable' for 
/// the variable referenced in the current lambda (i.e. \p VarToCapture).
/// If successful, returns the index into Sema's FunctionScopeInfo stack
/// of the capture-capable lambda's LambdaScopeInfo. 
/// See Implementation for more detailed comments. 

Optional<unsigned> getStackIndexOfNearestEnclosingCaptureCapableLambda(
    ArrayRef<const sema::FunctionScopeInfo *> FunctionScopes,
    VarDecl *VarToCapture, Sema &S);

} // clang

#endif
