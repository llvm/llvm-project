//===- Attribute.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which adds an an annotation to class declarations
// with a microsoft style '[example]' attribute.
//
// This plugin is used by clang/test/Frontend/ms-attributes tests.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Attributes.h"

using namespace clang;

namespace {

struct ExampleAttrInfo : public ParsedAttrInfo {
  ExampleAttrInfo() {
    // Can take up to 1 optional argument.
    OptArgs = 1;

    // Only Microsoft-style [ms_example] supported.
    // e.g.:
    // [ms_example] class C {};
    // [ms_example] void f() {}
    static constexpr Spelling S[] = {{ParsedAttr::AS_Microsoft, "ms_example"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute can be applied to any declaration.
    if (!isa<CXXRecordDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "classes";
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    // Add an annotation to the declaration.
    D->addAttr(AnnotateAttr::Create(S.Context, "ms_example", nullptr, 0,
                                    Attr.getRange()));
    return AttributeApplied;
  }
};

} // namespace

static ParsedAttrInfoRegistry::Add<ExampleAttrInfo> Y("microsoft-example", "");
