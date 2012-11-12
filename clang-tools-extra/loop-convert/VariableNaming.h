//===-- loop-convert/VariableNaming.h - Gererate variable names -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the VariableNamer class, which is
// responsible for generating new variable names and ensuring that they do not
// conflict with existing ones.
//
//===----------------------------------------------------------------------===//
#ifndef _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_VARIABLE_NAMING_H_
#define _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_VARIABLE_NAMING_H_

#include "StmtAncestor.h"
#include "clang/AST/ASTContext.h"

namespace clang {
namespace loop_migrate {

/// \brief Create names for generated variables within a particular statement.
///
/// VariableNamer uses a DeclContext as a reference point, checking for any
/// conflicting declarations higher up in the context or within SourceStmt.
/// It creates a variable name using hints from a source container and the old
/// index, if they exist.
class VariableNamer {
 public:
  VariableNamer(StmtGeneratedVarNameMap *GeneratedDecls,
                const StmtParentMap *ReverseAST, const Stmt *SourceStmt,
                const VarDecl *OldIndex, const VarDecl *TheContainer) :
  GeneratedDecls(GeneratedDecls), ReverseAST(ReverseAST),
  SourceStmt(SourceStmt), OldIndex(OldIndex), TheContainer(TheContainer) { }

  /// \brief Generate a new index name.
  ///
  /// Generates the name to be used for an inserted iterator. It relies on
  /// declarationExists() to determine that there are no naming conflicts, and
  /// tries to use some hints from the container name and the old index name.
  std::string createIndexName();

 private:
  StmtGeneratedVarNameMap *GeneratedDecls;
  const StmtParentMap *ReverseAST;
  const Stmt *SourceStmt;
  const VarDecl *OldIndex;
  const VarDecl *TheContainer;

  // Determine whether or not a declaration that would conflict with Symbol
  // exists in an outer context or in any statement contained in SourceStmt.
  bool declarationExists(const StringRef Symbol);
};

} // namespace loop_migrate
} // namespace clang
#endif // _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_VARIABLE_NAMING_H_
