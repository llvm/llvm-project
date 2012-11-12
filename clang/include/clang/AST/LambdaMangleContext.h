//===--- LambdaMangleContext.h - Context for mangling lambdas ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LambdaMangleContext interface, which keeps track of
//  the Itanium C++ ABI mangling numbers for lambda expressions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LAMBDAMANGLECONTEXT_H
#define LLVM_CLANG_LAMBDAMANGLECONTEXT_H

#include "llvm/ADT/DenseMap.h"

namespace clang {

class CXXMethodDecl;
class FunctionProtoType;

/// \brief Keeps track of the mangled names of lambda expressions within a
/// particular context.
class LambdaMangleContext {
  llvm::DenseMap<const FunctionProtoType *, unsigned> ManglingNumbers;
  
public:
  /// \brief Retrieve the mangling number of a new lambda expression with the
  /// given call operator within this lambda context.
  unsigned getManglingNumber(CXXMethodDecl *CallOperator);
};
  
} // end namespace clang
#endif
