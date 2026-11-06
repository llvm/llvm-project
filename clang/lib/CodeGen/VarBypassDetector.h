//===--- VarBypassDetector.h - Bypass jumps detector --------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains VarBypassDetector class, which is used to detect
// local variable declarations which can be bypassed by jumps.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_VARBYPASSDETECTOR_H
#define LLVM_CLANG_LIB_CODEGEN_VARBYPASSDETECTOR_H

#include "CodeGenModule.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace CodeGen {

/// The class detects jumps which bypass local variables declaration:
///    goto L;
///    int a;
///  L:
///
/// This is simplified version of JumpScopeChecker. Primary differences:
///  * Detects only jumps into the scope local variables.
///  * Does not detect jumps out of the scope of local variables.
///  * Not limited to variables with initializers, JumpScopeChecker is limited.
class VarBypassDetector {
  struct ScopeEntry {
    unsigned ParentScope;
    // Scope index of the immediately enclosing CompoundStmt block.
    unsigned EnclosingBlockScope = 0;
    llvm::PointerUnion<const VarDecl *, const CompoundLiteralExpr *> Item;
  };
  // Scope information. Contains a parent scope and related variable
  // declaration or compound literal.
  llvm::SmallVector<ScopeEntry, 48> Scopes;
  // List of jumps with scopes.
  llvm::SmallVector<std::pair<const Stmt *, unsigned>, 16> FromScopes;
  // Lookup map to find scope for destinations.
  llvm::DenseMap<const Stmt *, unsigned> ToScopes;
  // Set of variables which were bypassed by some forward jump.
  llvm::DenseSet<const VarDecl *> Bypasses;
  // Set of C local variables crossed by a backward jump within their enclosing
  // lexical block.
  llvm::DenseSet<const VarDecl *> BackwardBypassedVars;
  // Set of C block-scoped compound literals crossed by a backward jump within
  // their enclosing lexical block.
  llvm::DenseSet<const CompoundLiteralExpr *> BackwardBypassedCLs;
  // Map from a bypassing jump (goto/asm goto/switch) to the variable
  // declarations it bypasses. Used to reinitialize those variables at the jump.
  llvm::DenseMap<const Stmt *, llvm::SmallSetVector<const VarDecl *, 4>>
      BypassedVarsAtSource;
  unsigned CurEnclosingBlockScope = 0;
  // If true assume that all variables are being bypassed.
  bool AlwaysBypassed = false;
  bool Initialized = false;

public:
  void Init(CodeGenModule &CGM, const Stmt *Body);

  /// Returns true if the variable declaration was bypassed by any forward goto,
  /// asm goto, or switch statement.
  bool IsBypassed(const VarDecl *D) const {
    return AlwaysBypassed || Bypasses.contains(D);
  }

  /// Returns true if a backward jump branches from within \p D's live scope to
  /// an earlier label inside \p D's enclosing lexical block.
  bool IsBackwardBypassed(const VarDecl *D) const {
    return !Initialized || AlwaysBypassed || BackwardBypassedVars.contains(D);
  }

  /// Returns true if a backward jump branches from within \p CLE's live scope
  /// to an earlier label inside \p CLE's enclosing lexical block.
  bool IsBackwardBypassed(const CompoundLiteralExpr *CLE) const {
    return !Initialized || AlwaysBypassed || BackwardBypassedCLs.contains(CLE);
  }

  /// Returns true if jump sources cannot be determined (e.g. computed gotos),
  /// so all variables must be treated as bypassed.
  bool isAlwaysBypassed() const { return AlwaysBypassed; }

  /// Returns the variables bypassed by jumps from the given source statement,
  /// or nullptr if it bypasses none.
  const llvm::SmallSetVector<const VarDecl *, 4> *
  getBypassedVarsForSource(const Stmt *Source) const {
    auto It = BypassedVarsAtSource.find(Source);
    if (It == BypassedVarsAtSource.end())
      return nullptr;
    return &It->second;
  }

private:
  bool BuildScopeInformation(CodeGenModule &CGM, QualType Ty,
                             unsigned &ParentScope);
  bool BuildScopeInformation(CodeGenModule &CGM, const Decl *D,
                             unsigned &ParentScope);
  bool BuildScopeInformation(CodeGenModule &CGM, const Stmt *S,
                             unsigned &origParentScope);
  void Detect(bool IsCPlusPlus);
  void Detect(bool IsCPlusPlus, unsigned From, unsigned To, const Stmt *Source);
};
} // namespace CodeGen
} // namespace clang

#endif
