//===-- Mapper.h - ClangDoc Mapper ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mapper piece of the clang-doc tool. It implements
// a RecursiveASTVisitor to look at each declaration and populate the info
// into the internal representation. Each seen declaration is serialized to
// to bitcode and written out to the ExecutionContext as a KV pair where the
// key is the declaration's USR and the value is the serialized bitcode.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MAPPER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MAPPER_H

#include "Representation.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/Execution.h"

using namespace clang::comments;
using namespace clang::tooling;

namespace clang {
namespace doc {

class MapASTVisitor : public clang::RecursiveASTVisitor<MapASTVisitor>,
                      public ASTConsumer {
public:
  explicit MapASTVisitor(ASTContext *Ctx, ClangDocContext CDCtx)
      : CDCtx(CDCtx) {}

  void HandleTranslationUnit(ASTContext &Context) override;
  bool VisitNamespaceDecl(const NamespaceDecl *D);
  bool VisitRecordDecl(const RecordDecl *D);
  bool VisitEnumDecl(const EnumDecl *D);
  bool VisitCXXMethodDecl(const CXXMethodDecl *D);
  bool VisitFunctionDecl(const FunctionDecl *D);

private:
  template <typename T> bool mapDecl(const T *D);

  int getLine(const NamedDecl *D, const ASTContext &Context) const;
  StringRef getFile(const NamedDecl *D, const ASTContext &Context) const;
  comments::FullComment *getComment(const NamedDecl *D,
                                    const ASTContext &Context) const;

  ClangDocContext CDCtx;
};

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MAPPER_H
