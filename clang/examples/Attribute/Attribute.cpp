//===- Attribute.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which adds an an annotation to file-scope declarations
// with the 'example' attribute.
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
    // Can take an optional string argument (the check that the argument
    // actually is a string happens in handleDeclAttribute).
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {{ParsedAttr::AS_GNU, "example"},
                                     {ParsedAttr::AS_CXX11, "example"},
                                     {ParsedAttr::AS_CXX11, "plugin::example"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    // Check if the decl is at file scope.
    if (!D->getDeclContext()->isFileContext()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'example' attribute only allowed at file scope");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    // Check if we have an optional string argument.
    StringRef Str = "";
    if (Attr.getNumArgs() > 0) {
      Expr *ArgExpr = Attr.getArgAsExpr(0);
      StringLiteral *Literal =
          dyn_cast<StringLiteral>(ArgExpr->IgnoreParenCasts());
      if (Literal) {
        Str = Literal->getString();
      } else {
        S.Diag(ArgExpr->getExprLoc(), diag::err_attribute_argument_type)
            << Attr.getAttrName() << AANT_ArgumentString;
        return AttributeNotApplied;
      }
    }
    // Attach an annotate attribute to the Decl.
    D->addAttr(AnnotateAttr::Create(S.Context, "example(" + Str.str() + ")",
                                    Attr.getRange()));
    return AttributeApplied;
  }
};

} // namespace

static ParsedAttrInfoRegistry::Add<ExampleAttrInfo> X("example", "");
