//===-- LoopConvert/VariableNaming.h - Gererate variable names --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the VariableNamer class, which
/// is responsible for generating new variable names and ensuring that they do
/// not conflict with existing ones.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_VARIABLE_NAMING_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_VARIABLE_NAMING_H

#include "StmtAncestor.h"
#include "clang/AST/ASTContext.h"

/// \brief Create names for generated variables within a particular statement.
///
/// VariableNamer uses a DeclContext as a reference point, checking for any
/// conflicting declarations higher up in the context or within SourceStmt.
/// It creates a variable name using hints from a source container and the old
/// index, if they exist.
class VariableNamer {
 public:
  VariableNamer(
      StmtGeneratedVarNameMap *GeneratedDecls, const StmtParentMap *ReverseAST,
      const clang::Stmt *SourceStmt, const clang::VarDecl *OldIndex,
      const clang::VarDecl *TheContainer, const clang::ASTContext *Context)
      : GeneratedDecls(GeneratedDecls), ReverseAST(ReverseAST),
        SourceStmt(SourceStmt), OldIndex(OldIndex), TheContainer(TheContainer),
        Context(Context) {}

  /// \brief Generate a new index name.
  ///
  /// Generates the name to be used for an inserted iterator. It relies on
  /// declarationExists() to determine that there are no naming conflicts, and
  /// tries to use some hints from the container name and the old index name.
  std::string createIndexName();

 private:
  StmtGeneratedVarNameMap *GeneratedDecls;
  const StmtParentMap *ReverseAST;
  const clang::Stmt *SourceStmt;
  const clang::VarDecl *OldIndex;
  const clang::VarDecl *TheContainer;
  const clang::ASTContext *Context;

  // Determine whether or not a declaration that would conflict with Symbol
  // exists in an outer context or in any statement contained in SourceStmt.
  bool declarationExists(llvm::StringRef Symbol);
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_VARIABLE_NAMING_H
