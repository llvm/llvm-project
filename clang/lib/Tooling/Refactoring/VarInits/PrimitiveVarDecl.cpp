//===--- PrimitiveVarDecl.cpp - Clang refactoring library ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/AST/LexicallyOrderedRecursiveASTVisitor.h"
#include "clang/Tooling/Refactoring/RefactoringActionRuleRequirements.h"
#include "clang/Tooling/Refactoring/VarInits/PrimitiveVarDecl.h"

using namespace clang;
using namespace tooling;

namespace {

class VarDeclFinder
    : public LexicallyOrderedRecursiveASTVisitor<VarDeclFinder> {
public:
  VarDeclFinder(SourceLocation Location, FileID TargetFile,
                const ASTContext &AST)
      : LexicallyOrderedRecursiveASTVisitor(AST.getSourceManager()),
        Location(Location), TargetFile(TargetFile), AST(AST) {}

  bool VisitDeclRefExpr(DeclRefExpr *Ref) {
    const SourceManager &SM = AST.getSourceManager();
    if (SM.isPointWithin(Location, Ref->getBeginLoc(),
                         Ref->getEndLoc())) {
      this->VariableReference = Ref;
      return false;
    }
    return true;
  }

  DeclRefExpr *getDeclRefExpr() { return VariableReference; }

private:
  const SourceLocation Location;
  FileID TargetFile;
  const ASTContext &AST;
  DeclRefExpr *VariableReference = nullptr;
};

} // end anonymous namespace

DeclRefExpr *
clang::tooling::getDeclRefExprFromSourceLocation(ASTContext &AST,
                                                 SourceLocation Location) {

  FileID TargetFile = AST.getSourceManager().getFileID(Location);

  VarDeclFinder Visitor(Location, TargetFile, AST);
  Visitor.TraverseAST(AST);
  return Visitor.getDeclRefExpr();
}
