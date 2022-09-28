//===-- Mapper.cpp - ClangDoc Mapper ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Mapper.h"
#include "BitcodeWriter.h"
#include "Serialize.h"
#include "clang/AST/Comment.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

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
  bool IsFileInRootDir;
  llvm::SmallString<128> File =
      getFile(D, D->getASTContext(), CDCtx.SourceRoot, IsFileInRootDir);
  auto I = serialize::emitInfo(D, getComment(D, D->getASTContext()),
                               getLine(D, D->getASTContext()), File,
                               IsFileInRootDir, CDCtx.PublicOnly);

  // A null in place of I indicates that the serializer is skipping this decl
  // for some reason (e.g. we're only reporting public decls).
  if (I.first)
    CDCtx.ECtx->reportResult(llvm::toHex(llvm::toStringRef(I.first->USR)),
                             serialize::serialize(I.first));
  if (I.second)
    CDCtx.ECtx->reportResult(llvm::toHex(llvm::toStringRef(I.second->USR)),
                             serialize::serialize(I.second));
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
  if (isa<CXXMethodDecl>(D))
    return true;
  return mapDecl(D);
}

bool MapASTVisitor::VisitTypedefDecl(const TypedefDecl *D) {
  return mapDecl(D);
}

bool MapASTVisitor::VisitTypeAliasDecl(const TypeAliasDecl *D) {
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

llvm::SmallString<128> MapASTVisitor::getFile(const NamedDecl *D,
                                              const ASTContext &Context,
                                              llvm::StringRef RootDir,
                                              bool &IsFileInRootDir) const {
  llvm::SmallString<128> File(Context.getSourceManager()
                                  .getPresumedLoc(D->getBeginLoc())
                                  .getFilename());
  IsFileInRootDir = false;
  if (RootDir.empty() || !File.startswith(RootDir))
    return File;
  IsFileInRootDir = true;
  llvm::SmallString<128> Prefix(RootDir);
  // replace_path_prefix removes the exact prefix provided. The result of
  // calling that function on ("A/B/C.c", "A/B", "") would be "/C.c", which
  // starts with a / that is not needed. This is why we fix Prefix so it always
  // ends with a / and the result has the desired format.
  if (!llvm::sys::path::is_separator(Prefix.back()))
    Prefix += llvm::sys::path::get_separator();
  llvm::sys::path::replace_path_prefix(File, Prefix, "");
  return File;
}

} // namespace doc
} // namespace clang
