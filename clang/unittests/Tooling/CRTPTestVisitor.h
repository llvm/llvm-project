//===--- TestVisitor.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines a CRTP-based RecursiveASTVisitor helper for tests.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_TOOLING_CRTPTESTVISITOR_H
#define LLVM_CLANG_UNITTESTS_TOOLING_CRTPTESTVISITOR_H

#include "TestVisitor.h"
#include "clang/AST/RecursiveASTVisitor.h"

// CRTP versions of the visitors in TestVisitor.h.
namespace clang {
template <typename T>
class CRTPTestVisitor : public RecursiveASTVisitor<T>,
                        public detail::TestVisitorHelper {
public:
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

  void InvokeTraverseDecl(TranslationUnitDecl *D) override {
    RecursiveASTVisitor<T>::TraverseDecl(D);
  }
};

template <typename T>
class CRTPExpectedLocationVisitor
    : public CRTPTestVisitor<T>,
      public detail::ExpectedLocationVisitorHelper {
  ASTContext *getASTContext() override { return this->Context; }
};
} // namespace clang

#endif // LLVM_CLANG_UNITTESTS_TOOLING_CRTPTESTVISITOR_H
