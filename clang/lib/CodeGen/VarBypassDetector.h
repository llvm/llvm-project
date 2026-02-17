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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class Decl;
class Stmt;
class VarDecl;

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
  // Scope information. Contains a parent scope and related variable
  // declaration.
  llvm::SmallVector<std::pair<unsigned, const VarDecl *>, 48> Scopes;
  // List of jumps with scopes.
  llvm::SmallVector<std::pair<const Stmt *, unsigned>, 16> FromScopes;
  // Lookup map to find scope for destinations.
  llvm::DenseMap<const Stmt *, unsigned> ToScopes;
  // Set of variables which were bypassed by some jump.
  llvm::DenseSet<const VarDecl *> Bypasses;
  // Map from a bypassing jump (goto/switch case) to the variable declarations
  // it bypasses. Used to reinitialize those variables at the jump (C++ only).
  llvm::DenseMap<const Stmt *, llvm::DenseSet<const VarDecl *>>
      BypassedVarsAtSource;
  // If true assume that all variables are being bypassed.
  bool AlwaysBypassed = false;

public:
  void Init(CodeGenModule &CGM, const Stmt *Body);

  /// Returns true if the variable declaration was by bypassed by any goto or
  /// switch statement.
  bool IsBypassed(const VarDecl *D) const {
    return AlwaysBypassed || Bypasses.contains(D);
  }

  /// Returns true if jump sources cannot be determined (e.g. computed gotos),
  /// so all variables must be treated as bypassed.
  bool isAlwaysBypassed() const { return AlwaysBypassed; }

  /// Returns the variables bypassed by jumps from the given source statement,
  /// or nullptr if it bypasses none.
  const llvm::DenseSet<const VarDecl *> *
  getBypassedVarsForSource(const Stmt *Source) const {
    auto It = BypassedVarsAtSource.find(Source);
    if (It == BypassedVarsAtSource.end())
      return nullptr;
    return &It->second;
  }

private:
  bool BuildScopeInformation(CodeGenModule &CGM, const Decl *D,
                             unsigned &ParentScope);
  bool BuildScopeInformation(CodeGenModule &CGM, const Stmt *S,
                             unsigned &origParentScope);
  void Detect();
  void Detect(unsigned From, unsigned To, const Stmt *Source);
};
}
}

#endif
