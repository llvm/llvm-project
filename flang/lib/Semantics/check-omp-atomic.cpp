//===-- lib/Semantics/check-omp-atomic.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic checks related to the ATOMIC construct.
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"

#include "flang/Common/indirection.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <list>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace Fortran::semantics {

using namespace Fortran::semantics::omp;

namespace operation = Fortran::evaluate::operation;

template <typename T, typename U>
static bool operator!=(const evaluate::Expr<T> &e, const evaluate::Expr<U> &f) {
  return !(e == f);
}

struct AnalyzedCondStmt {
  SomeExpr cond{evaluate::NullPointer{}}; // Default ctor is deleted
  parser::CharBlock source;
  SourcedActionStmt ift, iff;
};

// Compute the `evaluate::Assignment` from parser::ActionStmt. The assumption
// is that the ActionStmt will be either an assignment or a pointer-assignment,
// otherwise return std::nullopt.
// Note: This function can return std::nullopt on [Pointer]AssignmentStmt where
// the "typedAssignment" is unset. This can happen if there are semantic errors
// in the purported assignment.
static std::optional<evaluate::Assignment> GetEvaluateAssignment(
    const parser::ActionStmt *x) {
  if (x == nullptr) {
    return std::nullopt;
  }

  using AssignmentStmt = common::Indirection<parser::AssignmentStmt>;
  using PointerAssignmentStmt =
      common::Indirection<parser::PointerAssignmentStmt>;
  using TypedAssignment = parser::AssignmentStmt::TypedAssignment;

  return common::visit(
      [](auto &&s) -> std::optional<evaluate::Assignment> {
        using BareS = llvm::remove_cvref_t<decltype(s)>;
        if constexpr (std::is_same_v<BareS, AssignmentStmt> ||
            std::is_same_v<BareS, PointerAssignmentStmt>) {
          const TypedAssignment &typed{s.value().typedAssignment};
          // ForwardOwningPointer                 typedAssignment
          // `- GenericAssignmentWrapper          ^.get()
          //    `- std::optional<Assignment>      ^->v
          return typed.get()->v;
        } else {
          return std::nullopt;
        }
      },
      x->u);
}

static std::optional<AnalyzedCondStmt> AnalyzeConditionalStmt(
    const parser::ExecutionPartConstruct *x) {
  if (x == nullptr) {
    return std::nullopt;
  }

  // Extract the evaluate::Expr from ScalarLogicalExpr.
  auto getFromLogical{[](const parser::ScalarLogicalExpr &logical) {
    // ScalarLogicalExpr is Scalar<Logical<common::Indirection<Expr>>>
    const parser::Expr &expr{logical.thing.thing.value()};
    return GetEvaluateExpr(expr);
  }};

  // Recognize either
  // ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> IfStmt, or
  // ExecutionPartConstruct -> ExecutableConstruct -> IfConstruct.

  if (auto &&action{GetActionStmt(x)}) {
    if (auto *ifs{std::get_if<common::Indirection<parser::IfStmt>>(
            &action.stmt->u)}) {
      const parser::IfStmt &s{ifs->value()};
      auto &&maybeCond{
          getFromLogical(std::get<parser::ScalarLogicalExpr>(s.t))};
      auto &thenStmt{
          std::get<parser::UnlabeledStatement<parser::ActionStmt>>(s.t)};
      if (maybeCond) {
        return AnalyzedCondStmt{std::move(*maybeCond), action.source,
            SourcedActionStmt{&thenStmt.statement, thenStmt.source},
            SourcedActionStmt{}};
      }
    }
    return std::nullopt;
  }

  if (auto *exec{std::get_if<parser::ExecutableConstruct>(&x->u)}) {
    if (auto *ifc{
            std::get_if<common::Indirection<parser::IfConstruct>>(&exec->u)}) {
      using ElseBlock = parser::IfConstruct::ElseBlock;
      using ElseIfBlock = parser::IfConstruct::ElseIfBlock;
      const parser::IfConstruct &s{ifc->value()};

      if (!std::get<std::list<ElseIfBlock>>(s.t).empty()) {
        // Not expecting any else-if statements.
        return std::nullopt;
      }
      auto &stmt{std::get<parser::Statement<parser::IfThenStmt>>(s.t)};
      auto &&maybeCond{getFromLogical(
          std::get<parser::ScalarLogicalExpr>(stmt.statement.t))};
      if (!maybeCond) {
        return std::nullopt;
      }

      if (auto &maybeElse{std::get<std::optional<ElseBlock>>(s.t)}) {
        AnalyzedCondStmt result{std::move(*maybeCond), stmt.source,
            GetActionStmt(std::get<parser::Block>(s.t)),
            GetActionStmt(std::get<parser::Block>(maybeElse->t))};
        if (result.ift.stmt && result.iff.stmt) {
          return result;
        }
      } else {
        AnalyzedCondStmt result{std::move(*maybeCond), stmt.source,
            GetActionStmt(std::get<parser::Block>(s.t)), SourcedActionStmt{}};
        if (result.ift.stmt) {
          return result;
        }
      }
    }
    return std::nullopt;
  }

  return std::nullopt;
}

static std::pair<parser::CharBlock, parser::CharBlock> SplitAssignmentSource(
    parser::CharBlock source) {
  // Find => in the range, if not found, find = that is not a part of
  // <=, >=, ==, or /=.
  auto trim{[](std::string_view v) {
    const char *begin{v.data()};
    const char *end{begin + v.size()};
    while (*begin == ' ' && begin != end) {
      ++begin;
    }
    while (begin != end && end[-1] == ' ') {
      --end;
    }
    assert(begin != end && "Source should not be empty");
    return parser::CharBlock(begin, end - begin);
  }};

  std::string_view sv(source.begin(), source.size());

  if (auto where{sv.find("=>")}; where != sv.npos) {
    std::string_view lhs(sv.data(), where);
    std::string_view rhs(sv.data() + where + 2, sv.size() - where - 2);
    return std::make_pair(trim(lhs), trim(rhs));
  }

  // Go backwards, since all the exclusions above end with a '='.
  for (size_t next{source.size()}; next > 1; --next) {
    if (sv[next - 1] == '=' && !llvm::is_contained("<>=/", sv[next - 2])) {
      std::string_view lhs(sv.data(), next - 1);
      std::string_view rhs(sv.data() + next, sv.size() - next);
      return std::make_pair(trim(lhs), trim(rhs));
    }
  }
  llvm_unreachable("Could not find assignment operator");
}

static bool IsCheckForAssociated(const SomeExpr &cond) {
  return GetTopLevelOperationIgnoreResizing(cond).first ==
      operation::Operator::Associated;
}

static bool IsMaybeAtomicWrite(const evaluate::Assignment &assign) {
  // This ignores function calls, so it will accept "f(x) = f(x) + 1"
  // for example.
  return HasStorageOverlap(assign.lhs, assign.rhs) == nullptr;
}

static void SetExpr(parser::TypedExpr &expr, MaybeExpr value) {
  if (value) {
    expr.Reset(new evaluate::GenericExprWrapper(std::move(value)),
        evaluate::GenericExprWrapper::Deleter);
  }
}

static void SetAssignment(parser::AssignmentStmt::TypedAssignment &assign,
    std::optional<evaluate::Assignment> value) {
  if (value) {
    assign.Reset(new evaluate::GenericAssignmentWrapper(std::move(value)),
        evaluate::GenericAssignmentWrapper::Deleter);
  }
}

static parser::OpenMPAtomicConstruct::Analysis::Op MakeAtomicAnalysisOp(
    int what,
    const std::optional<evaluate::Assignment> &maybeAssign = std::nullopt) {
  parser::OpenMPAtomicConstruct::Analysis::Op operation;
  operation.what = what;
  SetAssignment(operation.assign, maybeAssign);
  return operation;
}

static parser::OpenMPAtomicConstruct::Analysis MakeAtomicAnalysis(
    const SomeExpr &atom, const MaybeExpr &cond,
    parser::OpenMPAtomicConstruct::Analysis::Op &&op0,
    parser::OpenMPAtomicConstruct::Analysis::Op &&op1) {
  // Defined in flang/include/flang/Parser/parse-tree.h
  //
  // struct Analysis {
  //   struct Kind {
  //     static constexpr int None = 0;
  //     static constexpr int Read = 1;
  //     static constexpr int Write = 2;
  //     static constexpr int Update = Read | Write;
  //     static constexpr int Action = 3; // Bits containing N, R, W, U
  //     static constexpr int IfTrue = 4;
  //     static constexpr int IfFalse = 8;
  //     static constexpr int Condition = 12; // Bits containing IfTrue, IfFalse
  //   };
  //   struct Op {
  //     int what;
  //     TypedAssignment assign;
  //   };
  //   TypedExpr atom, cond;
  //   Op op0, op1;
  // };

  parser::OpenMPAtomicConstruct::Analysis an;
  SetExpr(an.atom, atom);
  SetExpr(an.cond, cond);
  an.op0 = std::move(op0);
  an.op1 = std::move(op1);
  return an;
}

/// Check if `expr` satisfies the following conditions for x and v:
///
/// [6.0:189:10-12]
/// - x and v (as applicable) are either scalar variables or
///   function references with scalar data pointer result of non-character
///   intrinsic type or variables that are non-polymorphic scalar pointers
///   and any length type parameter must be constant.
void OmpStructureChecker::CheckAtomicType(
    SymbolRef sym, parser::CharBlock source, std::string_view name) {
  const DeclTypeSpec *typeSpec{sym->GetType()};
  if (!typeSpec) {
    return;
  }

  if (!IsPointer(sym)) {
    using Category = DeclTypeSpec::Category;
    Category cat{typeSpec->category()};
    if (cat == Category::Character) {
      context_.Say(source,
          "Atomic variable %s cannot have CHARACTER type"_err_en_US, name);
    } else if (cat != Category::Numeric && cat != Category::Logical) {
      context_.Say(source,
          "Atomic variable %s should have an intrinsic type"_err_en_US, name);
    }
    return;
  }

  // Variable is a pointer.
  if (typeSpec->IsPolymorphic()) {
    context_.Say(source,
        "Atomic variable %s cannot be a pointer to a polymorphic type"_err_en_US,
        name);
    return;
  }

  // Go over all length parameters, if any, and check if they are
  // explicit.
  if (const DerivedTypeSpec *derived{typeSpec->AsDerived()}) {
    if (llvm::any_of(derived->parameters(), [](auto &&entry) {
          // "entry" is a map entry
          return entry.second.isLen() && !entry.second.isExplicit();
        })) {
      context_.Say(source,
          "Atomic variable %s is a pointer to a type with non-constant length parameter"_err_en_US,
          name);
    }
  }
}

void OmpStructureChecker::CheckAtomicVariable(
    const SomeExpr &atom, parser::CharBlock source) {
  if (atom.Rank() != 0) {
    context_.Say(source, "Atomic variable %s should be a scalar"_err_en_US,
        atom.AsFortran());
  }

  std::vector<SomeExpr> dsgs{GetAllDesignators(atom)};
  assert(dsgs.size() == 1 && "Should have a single top-level designator");
  evaluate::SymbolVector syms{evaluate::GetSymbolVector(dsgs.front())};

  CheckAtomicType(syms.back(), source, atom.AsFortran());

  if (IsAllocatable(syms.back()) && !IsArrayElement(atom)) {
    context_.Say(source, "Atomic variable %s cannot be ALLOCATABLE"_err_en_US,
        atom.AsFortran());
  }
}

void OmpStructureChecker::CheckStorageOverlap(const SomeExpr &base,
    llvm::ArrayRef<evaluate::Expr<evaluate::SomeType>> exprs,
    parser::CharBlock source) {
  if (auto *expr{HasStorageOverlap(base, exprs)}) {
    context_.Say(source,
        "Within atomic operation %s and %s access the same storage"_warn_en_US,
        base.AsFortran(), expr->AsFortran());
  }
}

void OmpStructureChecker::ErrorShouldBeVariable(
    const MaybeExpr &expr, parser::CharBlock source) {
  if (expr) {
    context_.Say(source, "Atomic expression %s should be a variable"_err_en_US,
        expr->AsFortran());
  } else {
    context_.Say(source, "Atomic expression should be a variable"_err_en_US);
  }
}

std::pair<const parser::ExecutionPartConstruct *,
    const parser::ExecutionPartConstruct *>
OmpStructureChecker::CheckUpdateCapture(
    const parser::ExecutionPartConstruct *ec1,
    const parser::ExecutionPartConstruct *ec2, parser::CharBlock source) {
  // Decide which statement is the atomic update and which is the capture.
  //
  // The two allowed cases are:
  //   x = ...      atomic-var = ...
  //   ... = x      capture-var = atomic-var (with optional converts)
  // or
  //   ... = x      capture-var = atomic-var (with optional converts)
  //   x = ...      atomic-var = ...
  //
  // The case of 'a = b; b = a' is ambiguous, so pick the first one as capture
  // (which makes more sense, as it captures the original value of the atomic
  // variable).
  //
  // If the two statements don't fit these criteria, return a pair of default-
  // constructed values.
  using ReturnTy = std::pair<const parser::ExecutionPartConstruct *,
      const parser::ExecutionPartConstruct *>;

  SourcedActionStmt act1{GetActionStmt(ec1)};
  SourcedActionStmt act2{GetActionStmt(ec2)};
  auto maybeAssign1{GetEvaluateAssignment(act1.stmt)};
  auto maybeAssign2{GetEvaluateAssignment(act2.stmt)};
  if (!maybeAssign1 || !maybeAssign2) {
    if (!IsAssignment(act1.stmt) || !IsAssignment(act2.stmt)) {
      context_.Say(source,
          "ATOMIC UPDATE operation with CAPTURE should contain two assignments"_err_en_US);
    }
    return std::make_pair(nullptr, nullptr);
  }

  auto as1{*maybeAssign1}, as2{*maybeAssign2};

  auto isUpdateCapture{
      [](const evaluate::Assignment &u, const evaluate::Assignment &c) {
        return IsSameOrConvertOf(c.rhs, u.lhs);
      }};

  // Do some checks that narrow down the possible choices for the update
  // and the capture statements. This will help to emit better diagnostics.
  // 1. An assignment could be an update (cbu) if the left-hand side is a
  //    subexpression of the right-hand side.
  // 2. An assignment could be a capture (cbc) if the right-hand side is
  //    a variable (or a function ref), with potential type conversions.
  bool cbu1{IsVarSubexpressionOf(as1.lhs, as1.rhs)}; // Can as1 be an update?
  bool cbu2{IsVarSubexpressionOf(as2.lhs, as2.rhs)}; // Can as2 be an update?
  bool cbc1{IsVarOrFunctionRef(GetConvertInput(as1.rhs))}; // Can 1 be capture?
  bool cbc2{IsVarOrFunctionRef(GetConvertInput(as2.rhs))}; // Can 2 be capture?

  // We want to diagnose cases where both assignments cannot be an update,
  // or both cannot be a capture, as well as cases where either assignment
  // cannot be any of these two.
  //
  // If we organize these boolean values into a matrix
  //   |cbu1 cbu2|
  //   |cbc1 cbc2|
  // then we want to diagnose cases where the matrix has a zero (i.e. "false")
  // row or column, including the case where everything is zero. All these
  // cases correspond to the determinant of the matrix being 0, which suggests
  // that checking the det may be a convenient diagnostic check. There is only
  // one additional case where the det is 0, which is when the matrix is all 1
  // ("true"). The "all true" case represents the situation where both
  // assignments could be an update as well as a capture. On the other hand,
  // whenever det != 0, the roles of the update and the capture can be
  // unambiguously assigned to as1 and as2 [1].
  //
  // [1] This can be easily verified by hand: there are 10 2x2 matrices with
  // det = 0, leaving 6 cases where det != 0:
  //   0 1   0 1   1 0   1 0   1 1   1 1
  //   1 0   1 1   0 1   1 1   0 1   1 0
  // In each case the classification is unambiguous.

  //     |cbu1 cbu2|
  // det |cbc1 cbc2| = cbu1*cbc2 - cbu2*cbc1
  int det{int(cbu1) * int(cbc2) - int(cbu2) * int(cbc1)};

  auto errorCaptureShouldRead{[&](const parser::CharBlock &source,
                                  const std::string &expr) {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE the right-hand side of the capture assignment should read %s"_err_en_US,
        expr);
  }};

  auto errorNeitherWorks{[&]() {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE neither statement could be the update or the capture"_err_en_US);
  }};

  auto makeSelectionFromDet{[&](int det) -> ReturnTy {
    // If det != 0, then the checks unambiguously suggest a specific
    // categorization.
    // If det == 0, then this function should be called only if the
    // checks haven't ruled out any possibility, i.e. when both assignments
    // could still be either updates or captures.
    if (det > 0) {
      // as1 is update, as2 is capture
      if (isUpdateCapture(as1, as2)) {
        return std::make_pair(/*Update=*/ec1, /*Capture=*/ec2);
      } else {
        errorCaptureShouldRead(act2.source, as1.lhs.AsFortran());
        return std::make_pair(nullptr, nullptr);
      }
    } else if (det < 0) {
      // as2 is update, as1 is capture
      if (isUpdateCapture(as2, as1)) {
        return std::make_pair(/*Update=*/ec2, /*Capture=*/ec1);
      } else {
        errorCaptureShouldRead(act1.source, as2.lhs.AsFortran());
        return std::make_pair(nullptr, nullptr);
      }
    } else {
      bool updateFirst{isUpdateCapture(as1, as2)};
      bool captureFirst{isUpdateCapture(as2, as1)};
      if (updateFirst && captureFirst) {
        // If both assignment could be the update and both could be the
        // capture, emit a warning about the ambiguity.
        context_.Say(act1.source,
            "In ATOMIC UPDATE operation with CAPTURE either statement could be the update and the capture, assuming the first one is the capture statement"_warn_en_US);
        return std::make_pair(/*Update=*/ec2, /*Capture=*/ec1);
      }
      if (updateFirst != captureFirst) {
        const parser::ExecutionPartConstruct *upd{updateFirst ? ec1 : ec2};
        const parser::ExecutionPartConstruct *cap{captureFirst ? ec1 : ec2};
        return std::make_pair(upd, cap);
      }
      assert(!updateFirst && !captureFirst);
      errorNeitherWorks();
      return std::make_pair(nullptr, nullptr);
    }
  }};

  if (det != 0 || (cbu1 && cbu2 && cbc1 && cbc2)) {
    return makeSelectionFromDet(det);
  }
  assert(det == 0 && "Prior checks should have covered det != 0");

  // If neither of the statements is an RMW update, it could still be a
  // "write" update. Pretty much any assignment can be a write update, so
  // recompute det with cbu1 = cbu2 = true.
  if (int writeDet{int(cbc2) - int(cbc1)}; writeDet || (cbc1 && cbc2)) {
    return makeSelectionFromDet(writeDet);
  }

  // It's only errors from here on.

  if (!cbu1 && !cbu2 && !cbc1 && !cbc2) {
    errorNeitherWorks();
    return std::make_pair(nullptr, nullptr);
  }

  // The remaining cases are that
  // - no candidate for update, or for capture,
  // - one of the assignments cannot be anything.

  if (!cbu1 && !cbu2) {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE neither statement could be the update"_err_en_US);
    return std::make_pair(nullptr, nullptr);
  } else if (!cbc1 && !cbc2) {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE neither statement could be the capture"_err_en_US);
    return std::make_pair(nullptr, nullptr);
  }

  if ((!cbu1 && !cbc1) || (!cbu2 && !cbc2)) {
    auto &src = (!cbu1 && !cbc1) ? act1.source : act2.source;
    context_.Say(src,
        "In ATOMIC UPDATE operation with CAPTURE the statement could be neither the update nor the capture"_err_en_US);
    return std::make_pair(nullptr, nullptr);
  }

  // All cases should have been covered.
  llvm_unreachable("Unchecked condition");
}

void OmpStructureChecker::CheckAtomicCaptureAssignment(
    const evaluate::Assignment &capture, const SomeExpr &atom,
    parser::CharBlock source) {
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};
  const SomeExpr &cap{capture.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, rsrc);
  } else {
    CheckAtomicVariable(atom, rsrc);
    // This part should have been checked prior to calling this function.
    assert(*GetConvertInput(capture.rhs) == atom &&
        "This cannot be a capture assignment");
    CheckStorageOverlap(atom, {cap}, source);
  }
}

void OmpStructureChecker::CheckAtomicReadAssignment(
    const evaluate::Assignment &read, parser::CharBlock source) {
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};

  if (auto maybe{GetConvertInput(read.rhs)}) {
    const SomeExpr &atom{*maybe};

    if (!IsVarOrFunctionRef(atom)) {
      ErrorShouldBeVariable(atom, rsrc);
    } else {
      CheckAtomicVariable(atom, rsrc);
      CheckStorageOverlap(atom, {read.lhs}, source);
    }
  } else {
    ErrorShouldBeVariable(read.rhs, rsrc);
  }
}

void OmpStructureChecker::CheckAtomicWriteAssignment(
    const evaluate::Assignment &write, parser::CharBlock source) {
  // [6.0:190:13-15]
  // A write structured block is write-statement, a write statement that has
  // one of the following forms:
  //   x = expr
  //   x => expr
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};
  const SomeExpr &atom{write.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, rsrc);
  } else {
    CheckAtomicVariable(atom, lsrc);
    CheckStorageOverlap(atom, {write.rhs}, source);
  }
}

void OmpStructureChecker::CheckAtomicUpdateAssignment(
    const evaluate::Assignment &update, parser::CharBlock source) {
  // [6.0:191:1-7]
  // An update structured block is update-statement, an update statement
  // that has one of the following forms:
  //   x = x operator expr
  //   x = expr operator x
  //   x = intrinsic-procedure-name (x)
  //   x = intrinsic-procedure-name (x, expr-list)
  //   x = intrinsic-procedure-name (expr-list, x)
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};
  const SomeExpr &atom{update.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, rsrc);
    // Skip other checks.
    return;
  }

  CheckAtomicVariable(atom, lsrc);

  std::pair<operation::Operator, std::vector<SomeExpr>> top{
      operation::Operator::Unknown, {}};
  if (auto &&maybeInput{GetConvertInput(update.rhs)}) {
    top = GetTopLevelOperationIgnoreResizing(*maybeInput);
  }
  switch (top.first) {
  case operation::Operator::Add:
  case operation::Operator::Sub:
  case operation::Operator::Mul:
  case operation::Operator::Div:
  case operation::Operator::And:
  case operation::Operator::Or:
  case operation::Operator::Eqv:
  case operation::Operator::Neqv:
  case operation::Operator::Min:
  case operation::Operator::Max:
  case operation::Operator::Identity:
    break;
  case operation::Operator::Call:
    context_.Say(source,
        "A call to this function is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  case operation::Operator::Convert:
    context_.Say(source,
        "An implicit or explicit type conversion is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  case operation::Operator::Intrinsic:
    context_.Say(source,
        "This intrinsic function is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  case operation::Operator::Constant:
  case operation::Operator::Unknown:
    context_.Say(
        source, "This is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  default:
    assert(
        top.first != operation::Operator::Identity && "Handle this separately");
    context_.Say(source,
        "The %s operator is not a valid ATOMIC UPDATE operation"_err_en_US,
        operation::ToString(top.first));
    return;
  }
  // Check how many times `atom` occurs as an argument, if it's a subexpression
  // of an argument, and collect the non-atom arguments.
  std::vector<SomeExpr> nonAtom;
  MaybeExpr subExpr;
  auto atomCount{[&]() {
    int count{0};
    for (const SomeExpr &arg : top.second) {
      if (IsSameOrConvertOf(arg, atom)) {
        ++count;
      } else {
        if (!subExpr && evaluate::IsVarSubexpressionOf(atom, arg)) {
          subExpr = arg;
        }
        nonAtom.push_back(arg);
      }
    }
    return count;
  }()};

  bool hasError{false};
  if (subExpr) {
    context_.Say(rsrc,
        "The atomic variable %s cannot be a proper subexpression of an argument (here: %s) in the update operation"_err_en_US,
        atom.AsFortran(), subExpr->AsFortran());
    hasError = true;
  }
  if (top.first == operation::Operator::Identity) {
    // This is "x = y".
    assert((atomCount == 0 || atomCount == 1) && "Unexpected count");
    if (atomCount == 0) {
      context_.Say(rsrc,
          "The atomic variable %s should appear as an argument in the update operation"_err_en_US,
          atom.AsFortran());
      hasError = true;
    }
  } else {
    if (atomCount == 0) {
      context_.Say(rsrc,
          "The atomic variable %s should appear as an argument of the top-level %s operator"_err_en_US,
          atom.AsFortran(), operation::ToString(top.first));
      hasError = true;
    } else if (atomCount > 1) {
      context_.Say(rsrc,
          "The atomic variable %s should be exactly one of the arguments of the top-level %s operator"_err_en_US,
          atom.AsFortran(), operation::ToString(top.first));
      hasError = true;
    }
  }

  if (!hasError) {
    CheckStorageOverlap(atom, nonAtom, source);
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdateAssignment(
    const SomeExpr &cond, parser::CharBlock condSource,
    const evaluate::Assignment &assign, parser::CharBlock assignSource) {
  auto [alsrc, arsrc]{SplitAssignmentSource(assignSource)};
  const SomeExpr &atom{assign.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, arsrc);
    // Skip other checks.
    return;
  }

  CheckAtomicVariable(atom, alsrc);

  auto top{GetTopLevelOperationIgnoreResizing(cond)};
  // Missing arguments to operations would have been diagnosed by now.

  switch (top.first) {
  case operation::Operator::Associated:
    if (atom != top.second.front()) {
      context_.Say(assignSource,
          "The pointer argument to ASSOCIATED must be same as the target of the assignment"_err_en_US);
    }
    break;
  // x equalop e | e equalop x  (allowing "e equalop x" is an extension)
  case operation::Operator::Eq:
  case operation::Operator::Eqv:
  // x ordop expr | expr ordop x
  case operation::Operator::Lt:
  case operation::Operator::Gt: {
    const SomeExpr &arg0{top.second[0]};
    const SomeExpr &arg1{top.second[1]};
    if (IsSameOrConvertOf(arg0, atom)) {
      CheckStorageOverlap(atom, {arg1}, condSource);
    } else if (IsSameOrConvertOf(arg1, atom)) {
      CheckStorageOverlap(atom, {arg0}, condSource);
    } else {
      assert(top.first != operation::Operator::Identity &&
          "Handle this separately");
      context_.Say(assignSource,
          "An argument of the %s operator should be the target of the assignment"_err_en_US,
          operation::ToString(top.first));
    }
    break;
  }
  case operation::Operator::Identity:
  case operation::Operator::True:
  case operation::Operator::False:
    break;
  default:
    assert(
        top.first != operation::Operator::Identity && "Handle this separately");
    context_.Say(condSource,
        "The %s operator is not a valid condition for ATOMIC operation"_err_en_US,
        operation::ToString(top.first));
    break;
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdateStmt(
    const AnalyzedCondStmt &update, parser::CharBlock source) {
  // The condition/statements must be:
  // - cond: x equalop e      ift: x =  d     iff: -
  // - cond: x ordop expr     ift: x =  expr  iff: -  (+ commute ordop)
  // - cond: associated(x)    ift: x => expr  iff: -
  // - cond: associated(x, e) ift: x => expr  iff: -

  // The if-true statement must be present, and must be an assignment.
  auto maybeAssign{GetEvaluateAssignment(update.ift.stmt)};
  if (!maybeAssign) {
    if (update.ift.stmt && !IsAssignment(update.ift.stmt)) {
      context_.Say(update.ift.source,
          "In ATOMIC UPDATE COMPARE the update statement should be an assignment"_err_en_US);
    } else {
      context_.Say(
          source, "Invalid body of ATOMIC UPDATE COMPARE operation"_err_en_US);
    }
    return;
  }
  const evaluate::Assignment assign{*maybeAssign};
  const SomeExpr &atom{assign.lhs};

  CheckAtomicConditionalUpdateAssignment(
      update.cond, update.source, assign, update.ift.source);

  CheckStorageOverlap(atom, {assign.rhs}, update.ift.source);

  if (update.iff) {
    context_.Say(update.iff.source,
        "In ATOMIC UPDATE COMPARE the update statement should not have an ELSE branch"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicUpdateOnly(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  if (body.size() == 1) {
    SourcedActionStmt action{GetActionStmt(&body.front())};
    if (auto maybeUpdate{GetEvaluateAssignment(action.stmt)}) {
      const SomeExpr &atom{maybeUpdate->lhs};
      CheckAtomicUpdateAssignment(*maybeUpdate, action.source);

      using Analysis = parser::OpenMPAtomicConstruct::Analysis;
      x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
          MakeAtomicAnalysisOp(Analysis::Update, maybeUpdate),
          MakeAtomicAnalysisOp(Analysis::None));
    } else if (!IsAssignment(action.stmt)) {
      context_.Say(
          source, "ATOMIC UPDATE operation should be an assignment"_err_en_US);
    }
  } else {
    context_.Say(x.source,
        "ATOMIC UPDATE operation should have a single statement"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdate(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  // Allowable forms are (single-statement):
  // - if ...
  // - x = (... ? ... : x)
  // and two-statement:
  // - r = cond ; if (r) ...

  const parser::ExecutionPartConstruct *ust{nullptr}; // update
  const parser::ExecutionPartConstruct *cst{nullptr}; // condition

  if (body.size() == 1) {
    ust = &body.front();
  } else if (body.size() == 2) {
    cst = &body.front();
    ust = &body.back();
  } else {
    context_.Say(source,
        "ATOMIC UPDATE COMPARE operation should contain one or two statements"_err_en_US);
    return;
  }

  // Flang doesn't support conditional-expr yet, so all update statements
  // are if-statements.

  // IfStmt:        if (...) ...
  // IfConstruct:   if (...) then ... endif
  auto maybeUpdate{AnalyzeConditionalStmt(ust)};
  if (!maybeUpdate) {
    context_.Say(source,
        "In ATOMIC UPDATE COMPARE the update statement should be a conditional statement"_err_en_US);
    return;
  }

  AnalyzedCondStmt &update{*maybeUpdate};

  if (SourcedActionStmt action{GetActionStmt(cst)}) {
    // The "condition" statement must be `r = cond`.
    if (auto maybeCond{GetEvaluateAssignment(action.stmt)}) {
      if (maybeCond->lhs != update.cond) {
        context_.Say(update.source,
            "In ATOMIC UPDATE COMPARE the conditional statement must use %s as the condition"_err_en_US,
            maybeCond->lhs.AsFortran());
      } else {
        // If it's "r = ...; if (r) ..." then put the original condition
        // in `update`.
        update.cond = maybeCond->rhs;
      }
    } else {
      context_.Say(action.source,
          "In ATOMIC UPDATE COMPARE with two statements the first statement should compute the condition"_err_en_US);
    }
  }

  evaluate::Assignment assign{*GetEvaluateAssignment(update.ift.stmt)};

  CheckAtomicConditionalUpdateStmt(update, source);
  if (IsCheckForAssociated(update.cond)) {
    if (!IsPointerAssignment(assign)) {
      context_.Say(source,
          "The assignment should be a pointer-assignment when the condition is ASSOCIATED"_err_en_US);
    }
  } else {
    if (IsPointerAssignment(assign)) {
      context_.Say(source,
          "The assignment cannot be a pointer-assignment except when the condition is ASSOCIATED"_err_en_US);
    }
  }

  using Analysis = parser::OpenMPAtomicConstruct::Analysis;
  x.analysis = MakeAtomicAnalysis(assign.lhs, update.cond,
      MakeAtomicAnalysisOp(Analysis::Update | Analysis::IfTrue, assign),
      MakeAtomicAnalysisOp(Analysis::None));
}

void OmpStructureChecker::CheckAtomicUpdateCapture(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  if (body.size() != 2) {
    context_.Say(source,
        "ATOMIC UPDATE operation with CAPTURE should contain two statements"_err_en_US);
    return;
  }

  auto [uec, cec]{CheckUpdateCapture(&body.front(), &body.back(), source)};
  if (!uec || !cec) {
    // Diagnostics already emitted.
    return;
  }
  SourcedActionStmt uact{GetActionStmt(uec)};
  SourcedActionStmt cact{GetActionStmt(cec)};
  // The "dereferences" of std::optional are guaranteed to be valid after
  // CheckUpdateCapture.
  evaluate::Assignment update{*GetEvaluateAssignment(uact.stmt)};
  evaluate::Assignment capture{*GetEvaluateAssignment(cact.stmt)};

  const SomeExpr &atom{update.lhs};

  using Analysis = parser::OpenMPAtomicConstruct::Analysis;
  int action;

  if (IsMaybeAtomicWrite(update)) {
    action = Analysis::Write;
    CheckAtomicWriteAssignment(update, uact.source);
  } else {
    action = Analysis::Update;
    CheckAtomicUpdateAssignment(update, uact.source);
  }
  CheckAtomicCaptureAssignment(capture, atom, cact.source);

  if (IsPointerAssignment(update) != IsPointerAssignment(capture)) {
    context_.Say(cact.source,
        "The update and capture assignments should both be pointer-assignments or both be non-pointer-assignments"_err_en_US);
    return;
  }

  if (GetActionStmt(&body.front()).stmt == uact.stmt) {
    x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
        MakeAtomicAnalysisOp(action, update),
        MakeAtomicAnalysisOp(Analysis::Read, capture));
  } else {
    x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
        MakeAtomicAnalysisOp(Analysis::Read, capture),
        MakeAtomicAnalysisOp(action, update));
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdateCapture(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  // There are two different variants of this:
  // (1) conditional-update and capture separately:
  //     This form only allows single-statement updates, i.e. the update
  //     form "r = cond; if (r) ..." is not allowed.
  // (2) conditional-update combined with capture in a single statement:
  //     This form does allow the condition to be calculated separately,
  //     i.e. "r = cond; if (r) ...".
  // Regardless of what form it is, the actual update assignment is a
  // proper write, i.e. "x = d", where d does not depend on x.

  AnalyzedCondStmt update;
  SourcedActionStmt capture;
  bool captureAlways{true}, captureFirst{true};

  auto extractCapture{[&]() {
    capture = update.iff;
    captureAlways = false;
    update.iff = SourcedActionStmt{};
  }};

  auto classifyNonUpdate{[&](const SourcedActionStmt &action) {
    // The non-update statement is either "r = cond" or the capture.
    if (auto maybeAssign{GetEvaluateAssignment(action.stmt)}) {
      if (update.cond == maybeAssign->lhs) {
        // If this is "r = cond; if (r) ...", then update the condition.
        update.cond = maybeAssign->rhs;
        update.source = action.source;
        // In this form, the update and the capture are combined into
        // an IF-THEN-ELSE statement.
        extractCapture();
      } else {
        // Assume this is the capture-statement.
        capture = action;
      }
    }
  }};

  if (body.size() == 2) {
    // This could be
    // - capture; conditional-update (in any order), or
    // - r = cond; if (r) capture-update
    const parser::ExecutionPartConstruct *st1{&body.front()};
    const parser::ExecutionPartConstruct *st2{&body.back()};
    // In either case, the conditional statement can be analyzed by
    // AnalyzeConditionalStmt, whereas the other statement cannot.
    if (auto maybeUpdate1{AnalyzeConditionalStmt(st1)}) {
      update = *maybeUpdate1;
      classifyNonUpdate(GetActionStmt(st2));
      captureFirst = false;
    } else if (auto maybeUpdate2{AnalyzeConditionalStmt(st2)}) {
      update = *maybeUpdate2;
      classifyNonUpdate(GetActionStmt(st1));
    } else {
      // None of the statements are conditional, this rules out the
      // "r = cond; if (r) ..." and the "capture + conditional-update"
      // variants. This could still be capture + write (which is classified
      // as conditional-update-capture in the spec).
      auto [uec, cec]{CheckUpdateCapture(st1, st2, source)};
      if (!uec || !cec) {
        // Diagnostics already emitted.
        return;
      }
      SourcedActionStmt uact{GetActionStmt(uec)};
      SourcedActionStmt cact{GetActionStmt(cec)};
      update.ift = uact;
      capture = cact;
      if (uec == st1) {
        captureFirst = false;
      }
    }
  } else if (body.size() == 1) {
    if (auto maybeUpdate{AnalyzeConditionalStmt(&body.front())}) {
      update = *maybeUpdate;
      // This is the form with update and capture combined into an IF-THEN-ELSE
      // statement. The capture-statement is always the ELSE branch.
      extractCapture();
    } else {
      goto invalid;
    }
  } else {
    context_.Say(source,
        "ATOMIC UPDATE COMPARE CAPTURE operation should contain one or two statements"_err_en_US);
    return;
  invalid:
    context_.Say(source,
        "Invalid body of ATOMIC UPDATE COMPARE CAPTURE operation"_err_en_US);
    return;
  }

  // The update must have a form `x = d` or `x => d`.
  if (auto maybeWrite{GetEvaluateAssignment(update.ift.stmt)}) {
    const SomeExpr &atom{maybeWrite->lhs};
    CheckAtomicWriteAssignment(*maybeWrite, update.ift.source);
    if (auto maybeCapture{GetEvaluateAssignment(capture.stmt)}) {
      CheckAtomicCaptureAssignment(*maybeCapture, atom, capture.source);

      if (IsPointerAssignment(*maybeWrite) !=
          IsPointerAssignment(*maybeCapture)) {
        context_.Say(capture.source,
            "The update and capture assignments should both be pointer-assignments or both be non-pointer-assignments"_err_en_US);
        return;
      }
    } else {
      if (!IsAssignment(capture.stmt)) {
        context_.Say(capture.source,
            "In ATOMIC UPDATE COMPARE CAPTURE the capture statement should be an assignment"_err_en_US);
      }
      return;
    }
  } else {
    if (!IsAssignment(update.ift.stmt)) {
      context_.Say(update.ift.source,
          "In ATOMIC UPDATE COMPARE CAPTURE the update statement should be an assignment"_err_en_US);
    }
    return;
  }

  // update.iff should be empty here, the capture statement should be
  // stored in "capture".

  // Fill out the analysis in the AST node.
  using Analysis = parser::OpenMPAtomicConstruct::Analysis;
  bool condUnused{std::visit(
      [](auto &&s) {
        using BareS = llvm::remove_cvref_t<decltype(s)>;
        if constexpr (std::is_same_v<BareS, evaluate::NullPointer>) {
          return true;
        } else {
          return false;
        }
      },
      update.cond.u)};

  int updateWhen{!condUnused ? Analysis::IfTrue : 0};
  int captureWhen{!captureAlways ? Analysis::IfFalse : 0};

  evaluate::Assignment updAssign{*GetEvaluateAssignment(update.ift.stmt)};
  evaluate::Assignment capAssign{*GetEvaluateAssignment(capture.stmt)};

  if (captureFirst) {
    x.analysis = MakeAtomicAnalysis(updAssign.lhs, update.cond,
        MakeAtomicAnalysisOp(Analysis::Read | captureWhen, capAssign),
        MakeAtomicAnalysisOp(Analysis::Write | updateWhen, updAssign));
  } else {
    x.analysis = MakeAtomicAnalysis(updAssign.lhs, update.cond,
        MakeAtomicAnalysisOp(Analysis::Write | updateWhen, updAssign),
        MakeAtomicAnalysisOp(Analysis::Read | captureWhen, capAssign));
  }
}

void OmpStructureChecker::CheckAtomicRead(
    const parser::OpenMPAtomicConstruct &x) {
  // [6.0:190:5-7]
  // A read structured block is read-statement, a read statement that has one
  // of the following forms:
  //   v = x
  //   v => x
  auto &block{std::get<parser::Block>(x.t)};

  // Read cannot be conditional or have a capture statement.
  if (x.IsCompare() || x.IsCapture()) {
    context_.Say(x.BeginDir().source,
        "ATOMIC READ cannot have COMPARE or CAPTURE clauses"_err_en_US);
    return;
  }

  const parser::Block &body{GetInnermostExecPart(block)};

  if (body.size() == 1) {
    SourcedActionStmt action{GetActionStmt(&body.front())};
    if (auto maybeRead{GetEvaluateAssignment(action.stmt)}) {
      CheckAtomicReadAssignment(*maybeRead, action.source);

      if (auto maybe{GetConvertInput(maybeRead->rhs)}) {
        const SomeExpr &atom{*maybe};
        using Analysis = parser::OpenMPAtomicConstruct::Analysis;
        x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
            MakeAtomicAnalysisOp(Analysis::Read, maybeRead),
            MakeAtomicAnalysisOp(Analysis::None));
      }
    } else if (!IsAssignment(action.stmt)) {
      context_.Say(
          x.source, "ATOMIC READ operation should be an assignment"_err_en_US);
    }
  } else {
    context_.Say(x.source,
        "ATOMIC READ operation should have a single statement"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicWrite(
    const parser::OpenMPAtomicConstruct &x) {
  auto &block{std::get<parser::Block>(x.t)};

  // Write cannot be conditional or have a capture statement.
  if (x.IsCompare() || x.IsCapture()) {
    context_.Say(x.BeginDir().source,
        "ATOMIC WRITE cannot have COMPARE or CAPTURE clauses"_err_en_US);
    return;
  }

  const parser::Block &body{GetInnermostExecPart(block)};

  if (body.size() == 1) {
    SourcedActionStmt action{GetActionStmt(&body.front())};
    if (auto maybeWrite{GetEvaluateAssignment(action.stmt)}) {
      const SomeExpr &atom{maybeWrite->lhs};
      CheckAtomicWriteAssignment(*maybeWrite, action.source);

      using Analysis = parser::OpenMPAtomicConstruct::Analysis;
      x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
          MakeAtomicAnalysisOp(Analysis::Write, maybeWrite),
          MakeAtomicAnalysisOp(Analysis::None));
    } else if (!IsAssignment(action.stmt)) {
      context_.Say(
          x.source, "ATOMIC WRITE operation should be an assignment"_err_en_US);
    }
  } else {
    context_.Say(x.source,
        "ATOMIC WRITE operation should have a single statement"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicUpdate(
    const parser::OpenMPAtomicConstruct &x) {
  auto &block{std::get<parser::Block>(x.t)};

  bool isConditional{x.IsCompare()};
  bool isCapture{x.IsCapture()};
  const parser::Block &body{GetInnermostExecPart(block)};

  if (isConditional && isCapture) {
    CheckAtomicConditionalUpdateCapture(x, body, x.source);
  } else if (isConditional) {
    CheckAtomicConditionalUpdate(x, body, x.source);
  } else if (isCapture) {
    CheckAtomicUpdateCapture(x, body, x.source);
  } else { // update-only
    CheckAtomicUpdateOnly(x, body, x.source);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPAtomicConstruct &x) {
  if (visitedAtomicSource_.empty())
    visitedAtomicSource_ = x.source;

  // All of the following groups have the "exclusive" property, i.e. at
  // most one clause from each group is allowed.
  // The exclusivity-checking code should eventually be unified for all
  // clauses, with clause groups defined in OMP.td.
  std::array atomic{llvm::omp::Clause::OMPC_read,
      llvm::omp::Clause::OMPC_update, llvm::omp::Clause::OMPC_write};
  std::array memoryOrder{llvm::omp::Clause::OMPC_acq_rel,
      llvm::omp::Clause::OMPC_acquire, llvm::omp::Clause::OMPC_relaxed,
      llvm::omp::Clause::OMPC_release, llvm::omp::Clause::OMPC_seq_cst};

  auto checkExclusive{[&](llvm::ArrayRef<llvm::omp::Clause> group,
                          std::string_view name,
                          const parser::OmpClauseList &clauses) {
    const parser::OmpClause *present{nullptr};
    for (const parser::OmpClause &clause : clauses.v) {
      llvm::omp::Clause id{clause.Id()};
      if (!llvm::is_contained(group, id)) {
        continue;
      }
      if (present == nullptr) {
        present = &clause;
        continue;
      } else if (id == present->Id()) {
        // Ignore repetitions of the same clause, those will be diagnosed
        // separately.
        continue;
      }
      parser::MessageFormattedText txt(
          "At most one clause from the '%s' group is allowed on ATOMIC construct"_err_en_US,
          name.data());
      parser::Message message(clause.source, txt);
      message.Attach(present->source,
          "Previous clause from this group provided here"_en_US);
      context_.Say(std::move(message));
      return;
    }
  }};

  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  auto &dir{std::get<parser::OmpDirectiveName>(dirSpec.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_atomic);
  llvm::omp::Clause kind{x.GetKind()};

  checkExclusive(atomic, "atomic", dirSpec.Clauses());
  checkExclusive(memoryOrder, "memory-order", dirSpec.Clauses());

  switch (kind) {
  case llvm::omp::Clause::OMPC_read:
    CheckAtomicRead(x);
    break;
  case llvm::omp::Clause::OMPC_write:
    CheckAtomicWrite(x);
    break;
  case llvm::omp::Clause::OMPC_update:
    CheckAtomicUpdate(x);
    break;
  default:
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPAtomicConstruct &) {
  dirContext_.pop_back();
}

} // namespace Fortran::semantics
