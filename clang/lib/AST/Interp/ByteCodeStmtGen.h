//===--- ByteCodeStmtGen.h - Code generator for expressions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the constexpr bytecode compiler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_BYTECODESTMTGEN_H
#define LLVM_CLANG_AST_INTERP_BYTECODESTMTGEN_H

#include "ByteCodeEmitter.h"
#include "ByteCodeExprGen.h"
#include "EvalEmitter.h"
#include "PrimType.h"
#include "clang/AST/StmtVisitor.h"

namespace clang {
namespace interp {

template <class Emitter> class LoopScope;
template <class Emitter> class SwitchScope;
template <class Emitter> class LabelScope;

/// Compilation context for statements.
template <class Emitter>
class ByteCodeStmtGen final : public ByteCodeExprGen<Emitter> {
  using LabelTy = typename Emitter::LabelTy;
  using AddrTy = typename Emitter::AddrTy;
  using OptLabelTy = std::optional<LabelTy>;
  using CaseMap = llvm::DenseMap<const SwitchCase *, LabelTy>;

public:
  template<typename... Tys>
  ByteCodeStmtGen(Tys&&... Args)
      : ByteCodeExprGen<Emitter>(std::forward<Tys>(Args)...) {}

protected:
  bool visitFunc(const FunctionDecl *F) override;

private:
  friend class LabelScope<Emitter>;
  friend class LoopScope<Emitter>;
  friend class SwitchScope<Emitter>;

  // Statement visitors.
  bool visitStmt(const Stmt *S);
  bool visitCompoundStmt(const CompoundStmt *S);
  bool visitLoopBody(const Stmt *S);
  bool visitDeclStmt(const DeclStmt *DS);
  bool visitReturnStmt(const ReturnStmt *RS);
  bool visitIfStmt(const IfStmt *IS);
  bool visitWhileStmt(const WhileStmt *S);
  bool visitDoStmt(const DoStmt *S);
  bool visitForStmt(const ForStmt *S);
  bool visitCXXForRangeStmt(const CXXForRangeStmt *S);
  bool visitBreakStmt(const BreakStmt *S);
  bool visitContinueStmt(const ContinueStmt *S);
  bool visitSwitchStmt(const SwitchStmt *S);
  bool visitCaseStmt(const CaseStmt *S);
  bool visitDefaultStmt(const DefaultStmt *S);
  bool visitAsmStmt(const AsmStmt *S);
  bool visitAttributedStmt(const AttributedStmt *S);

  bool emitLambdaStaticInvokerBody(const CXXMethodDecl *MD);

  /// Type of the expression returned by the function.
  std::optional<PrimType> ReturnType;

  /// Switch case mapping.
  CaseMap CaseLabels;

  /// Point to break to.
  OptLabelTy BreakLabel;
  /// Point to continue to.
  OptLabelTy ContinueLabel;
  /// Default case label.
  OptLabelTy DefaultLabel;
};

extern template class ByteCodeExprGen<EvalEmitter>;

} // namespace interp
} // namespace clang

#endif
