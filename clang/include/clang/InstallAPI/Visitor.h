//===- InstallAPI/Visitor.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// ASTVisitor Interface for InstallAPI frontend operations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_VISITOR_H
#define LLVM_CLANG_INSTALLAPI_VISITOR_H

#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/InstallAPI/Context.h"
#include "llvm/ADT/Twine.h"

namespace clang {
namespace installapi {

/// ASTVisitor for collecting declarations that represent global symbols.
class InstallAPIVisitor final : public ASTConsumer,
                                public RecursiveASTVisitor<InstallAPIVisitor> {
public:
  InstallAPIVisitor(ASTContext &ASTCtx, InstallAPIContext &Ctx,
                    SourceManager &SrcMgr, Preprocessor &PP)
      : Ctx(Ctx), SrcMgr(SrcMgr), PP(PP),
        MC(ItaniumMangleContext::create(ASTCtx, ASTCtx.getDiagnostics())),
        Layout(ASTCtx.getTargetInfo().getDataLayoutString()) {}
  void HandleTranslationUnit(ASTContext &ASTCtx) override;

  /// Collect global variables.
  bool VisitVarDecl(const VarDecl *D);

  /// Collect Objective-C Interface declarations.
  /// Every Objective-C class has an interface declaration that lists all the
  /// ivars, properties, and methods of the class.
  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D);

private:
  std::string getMangledName(const NamedDecl *D) const;
  std::string getBackendMangledName(llvm::Twine Name) const;
  std::optional<HeaderType> getAccessForDecl(const NamedDecl *D) const;

  InstallAPIContext &Ctx;
  SourceManager &SrcMgr;
  Preprocessor &PP;
  std::unique_ptr<clang::ItaniumMangleContext> MC;
  StringRef Layout;
};

} // namespace installapi
} // namespace clang

#endif // LLVM_CLANG_INSTALLAPI_VISITOR_H
