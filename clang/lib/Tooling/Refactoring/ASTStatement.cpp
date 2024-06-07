//===--- ASTStatement.cpp - Clang refactoring library ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/ASTStatement.h"
#include "clang/AST/LexicallyOrderedRecursiveASTVisitor.h"

using namespace clang;
using namespace tooling;

namespace {

class ASTStatementFinder
    : public LexicallyOrderedRecursiveASTVisitor<ASTStatementFinder> {
public:
  ASTStatementFinder(SourceLocation Location, FileID TargetFile,
                     const ASTContext &Context)
      : LexicallyOrderedRecursiveASTVisitor(Context.getSourceManager()),
        Location(std::move(Location)),
        TargetFile(TargetFile), Context(Context) {}

  bool TraverseStmt(Stmt *Statement) {
    if (!Statement)
        return true;
    const SourceManager &SM = Context.getSourceManager();
    if (SM.isPointWithin(Location, Statement->getBeginLoc(), Statement->getEndLoc())) {
      this->Statement = Statement;
    }
    LexicallyOrderedRecursiveASTVisitor::TraverseStmt(Statement);
    return true;
  }

  Stmt *getOuterStatement() {
    return Statement;
  }
private:
  const SourceLocation Location;
  FileID TargetFile;
  const ASTContext &Context;
  Stmt *Statement = nullptr;
};

} // end anonymous namespace

Stmt *
clang::tooling::findOuterStmt(const ASTContext &Context, SourceLocation Location) {
  assert(Location.isValid() && Location.isFileID() && "Expected a file location");

  FileID TargetFile = Context.getSourceManager().getFileID(Location);

  ASTStatementFinder Visitor(Location, TargetFile, Context);
  Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  return Visitor.getOuterStatement();
}
