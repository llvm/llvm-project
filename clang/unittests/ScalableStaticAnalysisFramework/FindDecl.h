//===- FindDecl.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_FINDDECL_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_FINDDECL_H

#include "clang/AST/Decl.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"

namespace clang::ssaf {

template <typename SomeDecl = clang::NamedDecl>
const SomeDecl *findDeclByName(StringRef Name, ASTContext &Ctx) {
  class NamedDeclFinder : public DynamicRecursiveASTVisitor {
  public:
    StringRef SearchingName;
    const NamedDecl *FoundDecl = nullptr;

    NamedDeclFinder(StringRef SearchingName) : SearchingName(SearchingName) {}

    bool VisitDecl(Decl *D) override {
      if (const auto *ND = dyn_cast<SomeDecl>(D)) {
        if (ND->getNameAsString() == SearchingName) {
          FoundDecl = ND;
          return false;
        }
      }
      return true;
    }
  };

  NamedDeclFinder Finder(Name);

  Finder.TraverseDecl(Ctx.getTranslationUnitDecl());
  return dyn_cast_or_null<SomeDecl>(Finder.FoundDecl);
}

inline const FunctionDecl *findFnByName(StringRef Name, ASTContext &Ctx) {
  return findDeclByName<FunctionDecl>(Name, Ctx);
}

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_FINDDECL_H
