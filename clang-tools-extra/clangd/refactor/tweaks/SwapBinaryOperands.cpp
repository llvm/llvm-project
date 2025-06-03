//===--- SwapBinaryOperands.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ParsedAST.h"
#include "Protocol.h"
#include "Selection.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>
#include <utility>

namespace clang {
namespace clangd {
namespace {
/// Check whether it makes logical sense to swap operands to an operator.
/// Assignment or member access operators are rarely swappable
/// while keeping the meaning intact, whereas comparison operators, mathematical
/// operators, etc. are often desired to be swappable for readability, avoiding
/// bugs by assigning to nullptr when comparison was desired, etc.
bool isOpSwappable(const BinaryOperatorKind Opcode) {
  switch (Opcode) {
  case BinaryOperatorKind::BO_Mul:
  case BinaryOperatorKind::BO_Add:
  case BinaryOperatorKind::BO_LT:
  case BinaryOperatorKind::BO_GT:
  case BinaryOperatorKind::BO_LE:
  case BinaryOperatorKind::BO_GE:
  case BinaryOperatorKind::BO_EQ:
  case BinaryOperatorKind::BO_NE:
  case BinaryOperatorKind::BO_And:
  case BinaryOperatorKind::BO_Xor:
  case BinaryOperatorKind::BO_Or:
  case BinaryOperatorKind::BO_LAnd:
  case BinaryOperatorKind::BO_LOr:
  case BinaryOperatorKind::BO_Comma:
    return true;
  // Noncommutative operators:
  case BinaryOperatorKind::BO_Div:
  case BinaryOperatorKind::BO_Sub:
  case BinaryOperatorKind::BO_Shl:
  case BinaryOperatorKind::BO_Shr:
  case BinaryOperatorKind::BO_Rem:
  // <=> is noncommutative
  case BinaryOperatorKind::BO_Cmp:
  // Member access:
  case BinaryOperatorKind::BO_PtrMemD:
  case BinaryOperatorKind::BO_PtrMemI:
  // Assignment:
  case BinaryOperatorKind::BO_Assign:
  case BinaryOperatorKind::BO_MulAssign:
  case BinaryOperatorKind::BO_DivAssign:
  case BinaryOperatorKind::BO_RemAssign:
  case BinaryOperatorKind::BO_AddAssign:
  case BinaryOperatorKind::BO_SubAssign:
  case BinaryOperatorKind::BO_ShlAssign:
  case BinaryOperatorKind::BO_ShrAssign:
  case BinaryOperatorKind::BO_AndAssign:
  case BinaryOperatorKind::BO_XorAssign:
  case BinaryOperatorKind::BO_OrAssign:
    return false;
  }
  return false;
}

/// Some operators are asymmetric and need to be flipped when swapping their
/// operands
/// @param[out] Opcode the opcode to potentially swap
/// If the opcode does not need to be swapped or is not swappable, does nothing
BinaryOperatorKind swapOperator(const BinaryOperatorKind Opcode) {
  switch (Opcode) {
  case BinaryOperatorKind::BO_LT:
    return BinaryOperatorKind::BO_GT;

  case BinaryOperatorKind::BO_GT:
    return BinaryOperatorKind::BO_LT;

  case BinaryOperatorKind::BO_LE:
    return BinaryOperatorKind::BO_GE;

  case BinaryOperatorKind::BO_GE:
    return BinaryOperatorKind::BO_LE;

  case BinaryOperatorKind::BO_Mul:
  case BinaryOperatorKind::BO_Add:
  case BinaryOperatorKind::BO_Cmp:
  case BinaryOperatorKind::BO_EQ:
  case BinaryOperatorKind::BO_NE:
  case BinaryOperatorKind::BO_And:
  case BinaryOperatorKind::BO_Xor:
  case BinaryOperatorKind::BO_Or:
  case BinaryOperatorKind::BO_LAnd:
  case BinaryOperatorKind::BO_LOr:
  case BinaryOperatorKind::BO_Comma:
  case BinaryOperatorKind::BO_Div:
  case BinaryOperatorKind::BO_Sub:
  case BinaryOperatorKind::BO_Shl:
  case BinaryOperatorKind::BO_Shr:
  case BinaryOperatorKind::BO_Rem:
  case BinaryOperatorKind::BO_PtrMemD:
  case BinaryOperatorKind::BO_PtrMemI:
  case BinaryOperatorKind::BO_Assign:
  case BinaryOperatorKind::BO_MulAssign:
  case BinaryOperatorKind::BO_DivAssign:
  case BinaryOperatorKind::BO_RemAssign:
  case BinaryOperatorKind::BO_AddAssign:
  case BinaryOperatorKind::BO_SubAssign:
  case BinaryOperatorKind::BO_ShlAssign:
  case BinaryOperatorKind::BO_ShrAssign:
  case BinaryOperatorKind::BO_AndAssign:
  case BinaryOperatorKind::BO_XorAssign:
  case BinaryOperatorKind::BO_OrAssign:
    return Opcode;
  }
  llvm_unreachable("Unknown BinaryOperatorKind enum");
}

/// Swaps the operands to a binary operator
/// Before:
///   x != nullptr
///   ^    ^^^^^^^
/// After:
///   nullptr != x
class SwapBinaryOperands : public Tweak {
public:
  const char *id() const final;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override {
    return llvm::formatv("Swap operands to {0}",
                         Op ? Op->getOpcodeStr() : "binary operator");
  }
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  bool hidden() const override { return false; }

private:
  const BinaryOperator *Op;
};

REGISTER_TWEAK(SwapBinaryOperands)

bool SwapBinaryOperands::prepare(const Selection &Inputs) {
  for (const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
       N && !Op; N = N->Parent) {
    // Stop once we hit a block, e.g. a lambda in one of the operands.
    // This makes sure that the selection point is in the 'scope' of the binary
    // operator, not from somewhere inside a lambda for example
    // (5 < [](){ ^return 1; })
    if (llvm::isa_and_nonnull<CompoundStmt>(N->ASTNode.get<Stmt>()))
      return false;
    Op = dyn_cast_or_null<BinaryOperator>(N->ASTNode.get<Stmt>());
    // If we hit upon a nonswappable binary operator, ignore and keep going
    if (Op && !isOpSwappable(Op->getOpcode())) {
      Op = nullptr;
    }
  }
  return Op != nullptr;
}

Expected<Tweak::Effect> SwapBinaryOperands::apply(const Selection &Inputs) {
  const auto &Ctx = Inputs.AST->getASTContext();
  const auto &SrcMgr = Inputs.AST->getSourceManager();

  const auto LHSRng = toHalfOpenFileRange(SrcMgr, Ctx.getLangOpts(),
                                          Op->getLHS()->getSourceRange());
  if (!LHSRng)
    return error(
        "Could not obtain range of the 'lhs' of the operator. Macros?");
  const auto RHSRng = toHalfOpenFileRange(SrcMgr, Ctx.getLangOpts(),
                                          Op->getRHS()->getSourceRange());
  if (!RHSRng)
    return error(
        "Could not obtain range of the 'rhs' of the operator. Macros?");
  const auto OpRng =
      toHalfOpenFileRange(SrcMgr, Ctx.getLangOpts(), Op->getOperatorLoc());
  if (!OpRng)
    return error("Could not obtain range of the operator itself. Macros?");

  const auto LHSCode = toSourceCode(SrcMgr, *LHSRng);
  const auto RHSCode = toSourceCode(SrcMgr, *RHSRng);
  const auto OperatorCode = toSourceCode(SrcMgr, *OpRng);

  tooling::Replacements Result;
  if (auto Err = Result.add(tooling::Replacement(
          Ctx.getSourceManager(), LHSRng->getBegin(), LHSCode.size(), RHSCode)))
    return std::move(Err);
  if (auto Err = Result.add(tooling::Replacement(
          Ctx.getSourceManager(), RHSRng->getBegin(), RHSCode.size(), LHSCode)))
    return std::move(Err);
  const auto SwappedOperator = swapOperator(Op->getOpcode());
  if (auto Err = Result.add(tooling::Replacement(
          Ctx.getSourceManager(), OpRng->getBegin(), OperatorCode.size(),
          Op->getOpcodeStr(SwappedOperator))))
    return std::move(Err);
  return Effect::mainFileEdit(SrcMgr, std::move(Result));
}

} // namespace
} // namespace clangd
} // namespace clang
