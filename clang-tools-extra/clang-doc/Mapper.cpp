//===-- Mapper.cpp - ClangDoc Mapper ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Mapper.h"
#include "BitcodeWriter.h"
#include "Serialize.h"
#include "clang/AST/Comment.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

using clang::comments::FullComment;

namespace clang {
namespace doc {

void MapASTVisitor::HandleTranslationUnit(ASTContext &Context) {
  TraverseDecl(Context.getTranslationUnitDecl());
}

template <typename T> bool MapASTVisitor::mapDecl(const T *D) {
  // If we're looking a decl not in user files, skip this decl.
  if (D->getASTContext().getSourceManager().isInSystemHeader(D->getLocation()))
    return true;

  // Skip function-internal decls.
  if (D->getParentFunctionOrMethod())
    return true;

  llvm::SmallString<128> USR;
  // If there is an error generating a USR for the decl, skip this decl.
  if (index::generateUSRForDecl(D, USR))
    return true;

  auto I = serialize::emitInfo(
      D, getComment(D, D->getASTContext()), getLine(D, D->getASTContext()),
      getFile(D, D->getASTContext()), CDCtx.PublicOnly);

  // A null in place of I indicates that the serializer is skipping this decl
  // for some reason (e.g. we're only reporting public decls).
  if (I)
    CDCtx.ECtx->reportResult(llvm::toHex(llvm::toStringRef(I->USR)),
                       serialize::serialize(I));
  return true;
}

bool MapASTVisitor::VisitNamespaceDecl(const NamespaceDecl *D) {
  return mapDecl(D);
}

bool MapASTVisitor::VisitRecordDecl(const RecordDecl *D) { return mapDecl(D); }

bool MapASTVisitor::VisitEnumDecl(const EnumDecl *D) { return mapDecl(D); }

bool MapASTVisitor::VisitCXXMethodDecl(const CXXMethodDecl *D) {
  return mapDecl(D);
}

bool MapASTVisitor::VisitFunctionDecl(const FunctionDecl *D) {
  // Don't visit CXXMethodDecls twice
  if (dyn_cast<CXXMethodDecl>(D))
    return true;
  return mapDecl(D);
}

comments::FullComment *
MapASTVisitor::getComment(const NamedDecl *D, const ASTContext &Context) const {
  RawComment *Comment = Context.getRawCommentForDeclNoCache(D);
  // FIXME: Move setAttached to the initial comment parsing.
  if (Comment) {
    Comment->setAttached();
    return Comment->parse(Context, nullptr, D);
  }
  return nullptr;
}

int MapASTVisitor::getLine(const NamedDecl *D,
                           const ASTContext &Context) const {
  return Context.getSourceManager().getPresumedLoc(D->getBeginLoc()).getLine();
}

llvm::StringRef MapASTVisitor::getFile(const NamedDecl *D,
                                       const ASTContext &Context) const {
  return Context.getSourceManager()
      .getPresumedLoc(D->getBeginLoc())
      .getFilename();
}

} // namespace doc
} // namespace clang
