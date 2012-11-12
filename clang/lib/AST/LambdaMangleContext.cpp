//===--- LambdaMangleContext.cpp - Context for mangling lambdas -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LambdaMangleContext class, which keeps track of
//  the Itanium C++ ABI mangling numbers for lambda expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/LambdaMangleContext.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;

unsigned LambdaMangleContext::getManglingNumber(CXXMethodDecl *CallOperator) {
  const FunctionProtoType *Proto
    = CallOperator->getType()->getAs<FunctionProtoType>();
  ASTContext &Context = CallOperator->getASTContext();
  
  QualType Key = Context.getFunctionType(Context.VoidTy, 
                                         Proto->arg_type_begin(),
                                         Proto->getNumArgs(),
                                         FunctionProtoType::ExtProtoInfo());
  Key = Context.getCanonicalType(Key);
  return ++ManglingNumbers[Key->castAs<FunctionProtoType>()];
}
