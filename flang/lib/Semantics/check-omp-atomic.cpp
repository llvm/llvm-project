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
#include "flang/Common/template.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/match.h"
#include "flang/Evaluate/rewrite.h"
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

static MaybeExpr PostSemaRewrite(const SomeExpr &atom, const SomeExpr &expr);

template <typename T, typename U>
static bool operator!=(const evaluate::Expr<T> &e, const evaluate::Expr<U> &f) {
  return !(e == f);
}

namespace {
template <typename...> struct IsIntegral {
  static constexpr bool value{false};
};

template <common::TypeCategory C, int K>
struct IsIntegral<evaluate::Type<C, K>> {
  static constexpr bool value{//
      C == common::TypeCategory::Integer ||
      C == common::TypeCategory::Unsigned};
};

template <typename T> constexpr bool is_integral_v{IsIntegral<T>::value};

template <typename...> struct IsFloatingPoint {
  static constexpr bool value{false};
};

template <common::TypeCategory C, int K>
struct IsFloatingPoint<evaluate::Type<C, K>> {
  static constexpr bool value{//
      C == common::TypeCategory::Real || C == common::TypeCategory::Complex};
};

template <typename T>
constexpr bool is_floating_point_v{IsFloatingPoint<T>::value};

template <typename T>
constexpr bool is_numeric_v{is_integral_v<T> || is_floating_point_v<T>};

template <typename...> struct IsLogical {
  static constexpr bool value{false};
};

template <common::TypeCategory C, int K>
struct IsLogical<evaluate::Type<C, K>> {
  static constexpr bool value{C == common::TypeCategory::Logical};
};

template <typename T> constexpr bool is_logical_v{IsLogical<T>::value};

template <typename T, typename Op0, typename Op1>
using ReassocOpBase = evaluate::match::AnyOfPattern< //
    evaluate::match::Add<T, Op0, Op1>, //
    evaluate::match::Mul<T, Op0, Op1>, //
    evaluate::match::LogicalOp<common::LogicalOperator::And, T, Op0, Op1>,
    evaluate::match::LogicalOp<common::LogicalOperator::Or, T, Op0, Op1>,
    evaluate::match::LogicalOp<common::LogicalOperator::Eqv, T, Op0, Op1>,
    evaluate::match::LogicalOp<common::LogicalOperator::Neqv, T, Op0, Op1>>;

template <typename T, typename Op0, typename Op1>
struct ReassocOp : public ReassocOpBase<T, Op0, Op1> {
  using Base = ReassocOpBase<T, Op0, Op1>;
  using Base::Base;
};

template <typename T, typename Op0, typename Op1>
ReassocOp<T, Op0, Op1> reassocOp(const Op0 &op0, const Op1 &op1) {
  return ReassocOp<T, Op0, Op1>(op0, op1);
}
} // namespace

struct ReassocRewriter : public evaluate::rewrite::Identity {
  using Id = evaluate::rewrite::Identity;
  struct NonIntegralTag {};

  ReassocRewriter(const SomeExpr &atom, const SemanticsContext &context)
      : atom_(atom), context_(context) {}

  // Try to find cases where the input expression is of the form
  // (1) (a . b) . c, or
  // (2) a . (b . c),
  // where . denotes an associative operation, and a, b, c are some
  // subexpresions.
  // If one of the operands in the nested operation is the atomic variable
  // (with some possible type conversions applied to it), bring it to the
  // top-level operation, and move the top-level operand into the nested
  // operation.
  // For example, assuming x is the atomic variable:
  //   (a + x) + b  ->  (a + b) + x,  i.e. (conceptually) swap x and b.
  template <typename T, typename U,
      typename = std::enable_if_t<is_numeric_v<T> || is_logical_v<T>>>
  evaluate::Expr<T> operator()(evaluate::Expr<T> &&x, const U &u) {
    if constexpr (is_floating_point_v<T>) {
      if (!context_.langOptions().AssociativeMath) {
        return Id::operator()(std::move(x), u);
      }
    }
    // As per the above comment, there are 3 subexpressions involved in this
    // transformation. A match::Expr<T> will match evaluate::Expr<U> when T is
    // same as U, plus it will store a pointer (ref) to the matched expression.
    // When the match is successful, the sub[i].ref will point to a, b, x (in
    // some order) from the example above.
    evaluate::match::Expr<T> sub[3];
    auto inner{reassocOp<T>(sub[0], sub[1])};
    auto outer1{reassocOp<T>(inner, sub[2])}; // inner . something
    auto outer2{reassocOp<T>(sub[2], inner)}; // something . inner
#if !defined(__clang__) && !defined(_MSC_VER) && \
    (__GNUC__ < 8 || (__GNUC__ == 8 && __GNUC_MINOR__ < 5))
    // If GCC version < 8.5, use this definition. For the other definition
    // (which is equivalent), GCC 7.5 emits a somewhat cryptic error:
    //    use of ‘outer1’ before deduction of ‘auto’
    // inside of the visitor function in common::visit.
    // Since this works with clang, MSVC and at least GCC 8.5, I'm assuming
    // that this is some kind of a GCC issue.
    using MatchTypes = std::tuple<evaluate::Add<T>, evaluate::Multiply<T>>;
#else
    using MatchTypes = typename decltype(outer1)::MatchTypes;
#endif
    // There is no way to ensure that the outer operation is the same as
    // the inner one. They are matched independently, so we need to compare
    // the index in the member variant that represents the matched type.
    if ((match(outer1, x) && outer1.ref.index() == inner.ref.index()) ||
        (match(outer2, x) && outer2.ref.index() == inner.ref.index())) {
      size_t atomIdx{[&]() { // sub[atomIdx] will be the atom.
        size_t idx;
        for (idx = 0; idx != 3; ++idx) {
          if (IsAtom(*sub[idx].ref)) {
            break;
          }
        }
        return idx;
      }()};

      if (atomIdx > 2) {
        return Id::operator()(std::move(x), u);
      }
      return common::visit(
          [&](auto &&s) {
            // Build the new expression from the matched components.
            return Reconstruct<T, MatchTypes>(s, *sub[atomIdx].ref,
                *sub[(atomIdx + 1) % 3].ref, *sub[(atomIdx + 2) % 3].ref);
          },
          evaluate::match::deparen(x).u);
    }
    return Id::operator()(std::move(x), u);
  }

  template <typename T, typename U,
      typename = std::enable_if_t<!is_numeric_v<T> && !is_logical_v<T>>>
  evaluate::Expr<T> operator()(
      evaluate::Expr<T> &&x, const U &u, NonIntegralTag = {}) {
    return Id::operator()(std::move(x), u);
  }

private:
  template <typename T, typename MatchTypes, typename S>
  evaluate::Expr<T> Reconstruct(const S &op, evaluate::Expr<T> atom,
      evaluate::Expr<T> op1, evaluate::Expr<T> op2) {
    using TypeS = llvm::remove_cvref_t<decltype(op)>;
    // This function has to be semantically correct for all possible types
    // of S even though at runtime s will only be one of the matched types.
    // Limit the construction to the operation types that we tried to match
    // (otherwise TypeS(op1, op2) would fail for non-binary operations).
    if constexpr (!common::HasMember<TypeS, MatchTypes>) {
      return evaluate::Expr<T>(TypeS(op));
    } else if constexpr (is_logical_v<T>) {
      constexpr int K{T::kind};
      if constexpr (std::is_same_v<TypeS, evaluate::LogicalOperation<K>>) {
        // Logical operators take an extra argument in their constructor,
        // so they need their own reconstruction code.
        common::LogicalOperator opCode{op.logicalOperator};
        return evaluate::Expr<T>(TypeS( //
            opCode, std::move(atom),
            evaluate::Expr<T>(TypeS( //
                opCode, std::move(op1), std::move(op2)))));
      }
    } else {
      // Generic reconstruction.
      return evaluate::Expr<T>(TypeS( //
          std::move(atom),
          evaluate::Expr<T>(TypeS( //
              std::move(op1), std::move(op2)))));
    }
  }

  template <typename T> bool IsAtom(const evaluate::Expr<T> &x) const {
    return IsSameOrConvertOf(evaluate::AsGenericExpr(AsRvalue(x)), atom_);
  }

  const SomeExpr &atom_;
  const SemanticsContext &context_;
};

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

static std::vector<SomeExpr> GetNonAtomExpressions(
    const SomeExpr &atom, const std::vector<SomeExpr> &exprs) {
  std::vector<SomeExpr> nonAtom;
  for (const SomeExpr &e : exprs) {
    if (!IsSameOrConvertOf(e, atom)) {
      nonAtom.push_back(e);
    }
  }
  return nonAtom;
}

static std::vector<SomeExpr> GetNonAtomArguments(
    const SomeExpr &atom, const SomeExpr &expr) {
  if (auto &&maybe{GetConvertInput(expr)}) {
    return GetNonAtomExpressions(
        atom, GetTopLevelOperationIgnoreResizing(*maybe).second);
  }
  return {};
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

namespace {
struct AtomicAnalysis {
  AtomicAnalysis(const SomeExpr &atom, const MaybeExpr &cond = std::nullopt)
      : atom_(atom), cond_(cond) {}

  AtomicAnalysis &addOp0(int what,
      const std::optional<evaluate::Assignment> &maybeAssign = std::nullopt) {
    return addOp(op0_, what, maybeAssign);
  }
  AtomicAnalysis &addOp1(int what,
      const std::optional<evaluate::Assignment> &maybeAssign = std::nullopt) {
    return addOp(op1_, what, maybeAssign);
  }

  operator parser::OpenMPAtomicConstruct::Analysis() const {
    // Defined in flang/include/flang/Parser/parse-tree.h
    //
    // struct Analysis {
    //   struct Kind {
    //     static constexpr int None = 0;
    //     static constexpr int Read = 1;
    //     static constexpr int Write = 2;
    //     static constexpr int Update = Read | Write;
    //     static constexpr int Action = 3; // Bits containing None, Read,
    //                                      // Write, Update
    //     static constexpr int IfTrue = 4;
    //     static constexpr int IfFalse = 8;
    //     static constexpr int Condition = 12; // Bits containing IfTrue,
    //                                          // IfFalse
    //   };
    //   struct Op {
    //     int what;
    //     TypedAssignment assign;
    //   };
    //   TypedExpr atom, cond;
    //   Op op0, op1;
    // };

    parser::OpenMPAtomicConstruct::Analysis an;
    SetExpr(an.atom, atom_);
    SetExpr(an.cond, cond_);
    an.op0 = std::move(op0_);
    an.op1 = std::move(op1_);
    return an;
  }

private:
  struct Op {
    operator parser::OpenMPAtomicConstruct::Analysis::Op() const {
      parser::OpenMPAtomicConstruct::Analysis::Op op;
      op.what = what;
      SetAssignment(op.assign, assign);
      return op;
    }

    int what;
    std::optional<evaluate::Assignment> assign;
  };

  AtomicAnalysis &addOp(Op &op, int what,
      const std::optional<evaluate::Assignment> &maybeAssign) {
    op.what = what;
    if (maybeAssign) {
      if (MaybeExpr rewritten{PostSemaRewrite(atom_, maybeAssign->rhs)}) {
        op.assign = evaluate::Assignment(
            AsRvalue(maybeAssign->lhs), std::move(*rewritten));
        op.assign->u = std::move(maybeAssign->u);
      } else {
        op.assign = *maybeAssign;
      }
    }
    return *this;
  }

  const SomeExpr &atom_;
  const MaybeExpr &cond_;
  Op op0_, op1_;
};
} // namespace

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
  (void)lsrc;
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
  (void)lsrc;

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

std::optional<evaluate::Assignment>
OmpStructureChecker::CheckAtomicUpdateAssignment(
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
    return std::nullopt;
  }

  CheckAtomicVariable(atom, lsrc);

  auto [hasErrors, tryReassoc]{CheckAtomicUpdateAssignmentRhs(
      atom, update.rhs, source, /*suppressDiagnostics=*/true)};

  if (!hasErrors) {
    CheckStorageOverlap(atom, GetNonAtomArguments(atom, update.rhs), source);
    return std::nullopt;
  } else if (tryReassoc) {
    ReassocRewriter ra(atom, context_);
    SomeExpr raRhs{evaluate::rewrite::Mutator(ra)(update.rhs)};

    std::tie(hasErrors, tryReassoc) = CheckAtomicUpdateAssignmentRhs(
        atom, raRhs, source, /*suppressDiagnostics=*/true);
    if (!hasErrors) {
      CheckStorageOverlap(atom, GetNonAtomArguments(atom, raRhs), source);

      evaluate::Assignment raAssign(update);
      raAssign.rhs = raRhs;
      return raAssign;
    }
  }

  // This is guaranteed to report errors.
  CheckAtomicUpdateAssignmentRhs(
      atom, update.rhs, source, /*suppressDiagnostics=*/false);
  return std::nullopt;
}

std::pair<bool, bool> OmpStructureChecker::CheckAtomicUpdateAssignmentRhs(
    const SomeExpr &atom, const SomeExpr &rhs, parser::CharBlock source,
    bool suppressDiagnostics) {
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};
  (void)lsrc;

  std::pair<operation::Operator, std::vector<SomeExpr>> top{
      operation::Operator::Unknown, {}};
  if (auto &&maybeInput{GetConvertInput(rhs)}) {
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
    if (!suppressDiagnostics) {
      context_.Say(source,
          "A call to this function is not a valid ATOMIC UPDATE operation"_err_en_US);
    }
    return std::make_pair(true, false);
  case operation::Operator::Convert:
    if (!suppressDiagnostics) {
      context_.Say(source,
          "An implicit or explicit type conversion is not a valid ATOMIC UPDATE operation"_err_en_US);
    }
    return std::make_pair(true, false);
  case operation::Operator::Intrinsic:
    if (!suppressDiagnostics) {
      context_.Say(source,
          "This intrinsic function is not a valid ATOMIC UPDATE operation"_err_en_US);
    }
    return std::make_pair(true, false);
  case operation::Operator::Constant:
  case operation::Operator::Unknown:
    if (!suppressDiagnostics) {
      context_.Say(
          source, "This is not a valid ATOMIC UPDATE operation"_err_en_US);
    }
    return std::make_pair(true, false);
  default:
    assert(
        top.first != operation::Operator::Identity && "Handle this separately");
    if (!suppressDiagnostics) {
      context_.Say(source,
          "The %s operator is not a valid ATOMIC UPDATE operation"_err_en_US,
          operation::ToString(top.first));
    }
    return std::make_pair(true, false);
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

  bool hasError{false}, tryReassoc{false};
  if (subExpr) {
    if (!suppressDiagnostics) {
      context_.Say(rsrc,
          "The atomic variable %s cannot be a proper subexpression of an argument (here: %s) in the update operation"_err_en_US,
          atom.AsFortran(), subExpr->AsFortran());
    }
    hasError = true;
  }
  if (top.first == operation::Operator::Identity) {
    // This is "x = y".
    assert((atomCount == 0 || atomCount == 1) && "Unexpected count");
    if (atomCount == 0) {
      if (!suppressDiagnostics) {
        context_.Say(rsrc,
            "The atomic variable %s should appear as an argument in the update operation"_err_en_US,
            atom.AsFortran());
      }
      hasError = true;
    }
  } else {
    if (atomCount == 0) {
      if (!suppressDiagnostics) {
        context_.Say(rsrc,
            "The atomic variable %s should appear as an argument of the top-level %s operator"_err_en_US,
            atom.AsFortran(), operation::ToString(top.first));
      }
      // If `atom` is a proper subexpression, and it not present as an
      // argument on its own, reassociation may be able to help.
      tryReassoc = subExpr.has_value();
      hasError = true;
    } else if (atomCount > 1) {
      if (!suppressDiagnostics) {
        context_.Say(rsrc,
            "The atomic variable %s should be exactly one of the arguments of the top-level %s operator"_err_en_US,
            atom.AsFortran(), operation::ToString(top.first));
      }
      hasError = true;
    }
  }

  return std::make_pair(hasError, tryReassoc);
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
      auto maybeAssign{
          CheckAtomicUpdateAssignment(*maybeUpdate, action.source)};
      auto &updateAssign{maybeAssign.has_value() ? maybeAssign : maybeUpdate};

      using Analysis = parser::OpenMPAtomicConstruct::Analysis;
      x.analysis = AtomicAnalysis(atom)
                       .addOp0(Analysis::Update, updateAssign)
                       .addOp1(Analysis::None);
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
  const SomeExpr &atom{assign.lhs};

  x.analysis = AtomicAnalysis(atom, update.cond)
                   .addOp0(Analysis::Update | Analysis::IfTrue, assign)
                   .addOp1(Analysis::None);
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

  std::optional<evaluate::Assignment> updateAssign{update};
  if (IsMaybeAtomicWrite(update)) {
    action = Analysis::Write;
    CheckAtomicWriteAssignment(update, uact.source);
  } else {
    action = Analysis::Update;
    if (auto &&maybe{CheckAtomicUpdateAssignment(update, uact.source)}) {
      updateAssign = maybe;
    }
  }
  CheckAtomicCaptureAssignment(capture, atom, cact.source);

  if (IsPointerAssignment(*updateAssign) != IsPointerAssignment(capture)) {
    context_.Say(cact.source,
        "The update and capture assignments should both be pointer-assignments or both be non-pointer-assignments"_err_en_US);
    return;
  }

  if (GetActionStmt(&body.front()).stmt == uact.stmt) {
    x.analysis = AtomicAnalysis(atom)
                     .addOp0(action, updateAssign)
                     .addOp1(Analysis::Read, capture);
  } else {
    x.analysis = AtomicAnalysis(atom)
                     .addOp0(Analysis::Read, capture)
                     .addOp1(action, updateAssign);
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
  const SomeExpr &atom{updAssign.lhs};

  if (captureFirst) {
    x.analysis = AtomicAnalysis(atom, update.cond)
                     .addOp0(Analysis::Read | captureWhen, capAssign)
                     .addOp1(Analysis::Write | updateWhen, updAssign);
  } else {
    x.analysis = AtomicAnalysis(atom, update.cond)
                     .addOp0(Analysis::Write | updateWhen, updAssign)
                     .addOp1(Analysis::Read | captureWhen, capAssign);
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
        x.analysis = AtomicAnalysis(atom)
                         .addOp0(Analysis::Read, maybeRead)
                         .addOp1(Analysis::None);
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
      x.analysis = AtomicAnalysis(atom)
                       .addOp0(Analysis::Write, maybeWrite)
                       .addOp1(Analysis::None);
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

// Rewrite min/max:
// Min and max intrinsics in Fortran take an arbitrary number of arguments
// (two or more). The first two are mandatory, the rest is optional. That
// means that arguments beyond the first two may be optional dummy argument
// from the caller. In that case, a reference to such an argument will
// cause presence test to be emitted, which cannot go inside of the atomic
// operation. Since the atom operand must be present, rewrite the min/max
// operation in a way that avoid the presence tests in the atomic code.
// For example, in
//   subroutine f(atom, x, y, z)
//     integer :: atom, x
//     integer, optional :: y, z
//     !$omp atomic update
//     atom = min(atom, x, y, z)
//   end
// the min operation will become
//   atom = min(atom, min(x, y, z))
// and in the final code
//   // Presence check is fine here.
//   tmp = min(x, y, z)
//   atomic update {
//     // Both operands are mandatory, no presence check needed.
//     atom = min(atom, tmp)
//   }
struct MinMaxRewriter : public evaluate::rewrite::Identity {
  using Id = evaluate::rewrite::Identity;
  using Id::operator();

  MinMaxRewriter(const SomeExpr &atom) : atom_(atom) {}

  static bool IsMinMax(const evaluate::ProcedureDesignator &p) {
    if (auto *intrin{p.GetSpecificIntrinsic()}) {
      return intrin->name == "min" || intrin->name == "max";
    }
    return false;
  }

  // Take a list of arguments to a min/max operation, e.g. [a0, a1, ...]
  // One of the a_i's, say a_t, must be the atom.
  // Generate
  //   min/max(a_t, min/max(a0, a1, ... [except a_t]))
  template <typename T>
  evaluate::Expr<T> operator()(
      evaluate::Expr<T> &&x, const evaluate::FunctionRef<T> &f) {
    const evaluate::ProcedureDesignator &proc = f.proc();
    if (!IsMinMax(proc) || f.arguments().size() <= 2) {
      return Id::operator()(std::move(x), f);
    }

    // Collect arguments as SomeExpr's and find out which argument
    // corresponds to atom.
    const SomeExpr *atomArg{nullptr};
    std::vector<const SomeExpr *> args;
    for (const std::optional<evaluate::ActualArgument> &a : f.arguments()) {
      if (!a) {
        continue;
      }
      if (const SomeExpr *e{a->UnwrapExpr()}) {
        if (evaluate::IsSameOrConvertOf(*e, atom_)) {
          atomArg = e;
        }
        args.push_back(e);
      }
    }
    if (!atomArg) {
      return Id::operator()(std::move(x), f);
    }

    evaluate::ActualArguments nonAtoms;

    auto AsActual = [](const SomeExpr &z) {
      SomeExpr copy = z;
      return evaluate::ActualArgument(std::move(copy));
    };
    // Semantic checks guarantee that the "atom" shows exactly once in the
    // argument list (with potential conversions around it).
    // For the first two (non-optional) arguments, if "atom" is among them,
    // replace it with another occurrence of the other non-optional argument.
    if (atomArg == args[0]) {
      // (atom, x, y...) -> (x, x, y...)
      nonAtoms.push_back(AsActual(*args[1]));
      nonAtoms.push_back(AsActual(*args[1]));
    } else if (atomArg == args[1]) {
      // (x, atom, y...) -> (x, x, y...)
      nonAtoms.push_back(AsActual(*args[0]));
      nonAtoms.push_back(AsActual(*args[0]));
    } else {
      // (x, y, z...) -> unchanged
      nonAtoms.push_back(AsActual(*args[0]));
      nonAtoms.push_back(AsActual(*args[1]));
    }

    // The rest of arguments are optional, so we can just skip "atom".
    for (size_t i = 2, e = args.size(); i != e; ++i) {
      if (atomArg != args[i])
        nonAtoms.push_back(AsActual(*args[i]));
    }

    SomeExpr tmp = evaluate::AsGenericExpr(
        evaluate::FunctionRef<T>(AsRvalue(proc), AsRvalue(nonAtoms)));

    return evaluate::Expr<T>(evaluate::FunctionRef<T>(
        AsRvalue(proc), {AsActual(*atomArg), AsActual(tmp)}));
  }

private:
  const SomeExpr &atom_;
};

static MaybeExpr PostSemaRewrite(const SomeExpr &atom, const SomeExpr &expr) {
  MinMaxRewriter rewriter(atom);
  return evaluate::rewrite::Mutator(rewriter)(expr);
}

} // namespace Fortran::semantics
