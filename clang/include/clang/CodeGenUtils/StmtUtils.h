//===--- StmtUtils.h - Shared statement emission queries ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST queries about statements that both classic CodeGen
// and CIR CodeGen need while emitting statements.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_STMTUTILS_H
#define LLVM_CLANG_CODEGENUTILS_STMTUTILS_H

#include "clang/AST/Stmt.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang {
class ASTContext;
class Expr;
} // namespace clang

namespace clang::CodeGenUtils {

// [C++26][stmt.iter.general] (DR)
// A trivially empty iteration statement is an iteration statement matching
// one of the following forms:
//  - while ( expression ) ;
//  - while ( expression ) { }
//  - do ; while ( expression ) ;
//  - do { } while ( expression ) ;
//  - for ( init-statement expression(opt); ) ;
//  - for ( init-statement expression(opt); ) { }
template <typename LoopStmt> bool hasEmptyLoopBody(const LoopStmt &S) {
  if constexpr (std::is_same_v<LoopStmt, ForStmt>) {
    if (S.getInc())
      return false;
  }
  const Stmt *Body = S.getBody();
  if (!Body || isa<NullStmt>(Body))
    return true;
  if (const auto *Compound = dyn_cast<CompoundStmt>(Body))
    return Compound->body_empty();
  return false;
}

/// Returns true if a loop must make progress, which means the mustprogress
/// attribute can be added to it. \p HasEmptyBody indicates whether the loop
/// is trivially empty (see hasEmptyLoopBody()).
///
/// A loop that is both trivially empty and has a constant-true controlling
/// expression (e.g. `while (true) {}`) is a trivial infinite loop, which is
/// exempt from the forward-progress guarantee: [C++26][stmt.iter.general]
/// (DR). When such a loop is found, \p RemoveMustProgress is called so the
/// caller can delete the 'mustprogress' attribute it may have
/// speculatively added to the enclosing function.
bool checkIfLoopMustProgress(const LangOptions &LangOpts,
                             const CodeGenOptions &CGOpts, ASTContext &Ctx,
                             const Expr *ControllingExpression,
                             bool HasEmptyBody,
                             llvm::function_ref<void()> RemoveMustProgress);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_STMTUTILS_H
