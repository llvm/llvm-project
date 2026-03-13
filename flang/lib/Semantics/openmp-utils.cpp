//===-- lib/Semantics/openmp-utils.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common utilities used in OpenMP semantic checks.
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/openmp-utils.h"

#include "flang/Common/Fortran-consts.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Common/reference.h"
#include "flang/Common/visit.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/match.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Evaluate/variable.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace Fortran::semantics::omp {
using namespace Fortran::parser::omp;

const Scope &GetScopingUnit(const Scope &scope) {
  const Scope *iter{&scope};
  for (; !iter->IsTopLevel(); iter = &iter->parent()) {
    switch (iter->kind()) {
    case Scope::Kind::BlockConstruct:
    case Scope::Kind::BlockData:
    case Scope::Kind::DerivedType:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Module:
    case Scope::Kind::Subprogram:
      return *iter;
    default:
      break;
    }
  }
  return *iter;
}

const Scope &GetProgramUnit(const Scope &scope) {
  const Scope *unit{nullptr};
  for (const Scope *iter{&scope}; !iter->IsTopLevel(); iter = &iter->parent()) {
    switch (iter->kind()) {
    case Scope::Kind::BlockData:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Module:
      return *iter;
    case Scope::Kind::Subprogram:
      // Ignore subprograms that are nested.
      unit = iter;
      break;
    default:
      break;
    }
  }
  assert(unit && "Scope not in a program unit");
  return *unit;
}

SourcedActionStmt GetActionStmt(const parser::ExecutionPartConstruct *x) {
  if (x == nullptr) {
    return SourcedActionStmt{};
  }
  if (auto *exec{std::get_if<parser::ExecutableConstruct>(&x->u)}) {
    using ActionStmt = parser::Statement<parser::ActionStmt>;
    if (auto *stmt{std::get_if<ActionStmt>(&exec->u)}) {
      return SourcedActionStmt{&stmt->statement, stmt->source};
    }
  }
  return SourcedActionStmt{};
}

SourcedActionStmt GetActionStmt(const parser::Block &block) {
  if (block.size() == 1) {
    return GetActionStmt(&block.front());
  }
  return SourcedActionStmt{};
}

std::string ThisVersion(unsigned version) {
  std::string tv{
      std::to_string(version / 10) + "." + std::to_string(version % 10)};
  return "OpenMP v" + tv;
}

std::string TryVersion(unsigned version) {
  return "try -fopenmp-version=" + std::to_string(version);
}

const parser::Designator *GetDesignatorFromObj(
    const parser::OmpObject &object) {
  return std::get_if<parser::Designator>(&object.u);
}

const parser::DataRef *GetDataRefFromObj(const parser::OmpObject &object) {
  if (auto *desg{GetDesignatorFromObj(object)}) {
    return std::get_if<parser::DataRef>(&desg->u);
  }
  return nullptr;
}

const parser::ArrayElement *GetArrayElementFromObj(
    const parser::OmpObject &object) {
  if (auto *dataRef{GetDataRefFromObj(object)}) {
    using ElementIndirection = common::Indirection<parser::ArrayElement>;
    if (auto *ind{std::get_if<ElementIndirection>(&dataRef->u)}) {
      return &ind->value();
    }
  }
  return nullptr;
}

const Symbol *GetObjectSymbol(const parser::OmpObject &object) {
  // Some symbols may be missing if the resolution failed, e.g. when an
  // undeclared name is used with implicit none.
  if (auto *name{std::get_if<parser::Name>(&object.u)}) {
    return name->symbol ? &name->symbol->GetUltimate() : nullptr;
  } else if (auto *desg{std::get_if<parser::Designator>(&object.u)}) {
    auto &last{GetLastName(*desg)};
    return last.symbol ? &GetLastName(*desg).symbol->GetUltimate() : nullptr;
  }
  return nullptr;
}

std::optional<parser::CharBlock> GetObjectSource(
    const parser::OmpObject &object) {
  if (auto *name{std::get_if<parser::Name>(&object.u)}) {
    return name->source;
  } else if (auto *desg{std::get_if<parser::Designator>(&object.u)}) {
    return GetLastName(*desg).source;
  }
  return std::nullopt;
}

const Symbol *GetArgumentSymbol(const parser::OmpArgument &argument) {
  if (auto *locator{std::get_if<parser::OmpLocator>(&argument.u)}) {
    if (auto *object{std::get_if<parser::OmpObject>(&locator->u)}) {
      return GetObjectSymbol(*object);
    }
  }
  return nullptr;
}

const parser::OmpObject *GetArgumentObject(
    const parser::OmpArgument &argument) {
  if (auto *locator{std::get_if<parser::OmpLocator>(&argument.u)}) {
    return std::get_if<parser::OmpObject>(&locator->u);
  }
  return nullptr;
}

bool IsCommonBlock(const Symbol &sym) {
  return sym.detailsIf<CommonBlockDetails>() != nullptr;
}

bool IsVariableListItem(const Symbol &sym) {
  return evaluate::IsVariable(sym) || sym.attrs().test(Attr::POINTER);
}

bool IsExtendedListItem(const Symbol &sym) {
  return IsVariableListItem(sym) || sym.IsSubprogram();
}

bool IsTypeParamInquiry(const Symbol &sym) {
  return common::visit( //
      common::visitors{
          [&](const MiscDetails &d) {
            return d.kind() == MiscDetails::Kind::KindParamInquiry ||
                d.kind() == MiscDetails::Kind::LenParamInquiry;
          },
          [&](const TypeParamDetails &s) { return true; },
          [&](auto &&) { return false; },
      },
      sym.details());
}

bool IsStructureComponent(const Symbol &sym) {
  return sym.owner().kind() == Scope::Kind::DerivedType;
}

bool IsVarOrFunctionRef(const MaybeExpr &expr) {
  if (expr) {
    return evaluate::UnwrapProcedureRef(*expr) != nullptr ||
        evaluate::IsVariable(*expr);
  } else {
    return false;
  }
}

bool IsWholeAssumedSizeArray(const parser::OmpObject &object) {
  if (auto *sym{GetObjectSymbol(object)}; sym && IsAssumedSizeArray(*sym)) {
    return !GetArrayElementFromObj(object);
  }
  return false;
}

bool IsMapEnteringType(parser::OmpMapType::Value type) {
  switch (type) {
  case parser::OmpMapType::Value::Alloc:
  case parser::OmpMapType::Value::Storage:
  case parser::OmpMapType::Value::To:
  case parser::OmpMapType::Value::Tofrom:
    return true;
  default:
    return false;
  }
}

bool IsMapExitingType(parser::OmpMapType::Value type) {
  switch (type) {
  case parser::OmpMapType::Value::Delete:
  case parser::OmpMapType::Value::From:
  case parser::OmpMapType::Value::Release:
  case parser::OmpMapType::Value::Storage:
  case parser::OmpMapType::Value::Tofrom:
    return true;
  default:
    return false;
  }
}

static MaybeExpr GetEvaluateExprFromTyped(const parser::TypedExpr &typedExpr) {
  // ForwardOwningPointer           typedExpr
  // `- GenericExprWrapper          ^.get()
  //    `- std::optional<Expr>      ^->v
  if (auto *wrapper{typedExpr.get()}) {
    return wrapper->v;
  }
  return std::nullopt;
}

MaybeExpr GetEvaluateExpr(const parser::Expr &parserExpr) {
  return GetEvaluateExprFromTyped(parserExpr.typedExpr);
}

std::optional<evaluate::DynamicType> GetDynamicType(
    const parser::Expr &parserExpr) {
  if (auto maybeExpr{GetEvaluateExpr(parserExpr)}) {
    return maybeExpr->GetType();
  } else {
    return std::nullopt;
  }
}

namespace {
struct LogicalConstantVistor : public evaluate::Traverse<LogicalConstantVistor,
                                   std::optional<bool>, false> {
  using Result = std::optional<bool>;
  using Base = evaluate::Traverse<LogicalConstantVistor, Result, false>;
  LogicalConstantVistor() : Base(*this) {}

  Result Default() const { return std::nullopt; }

  using Base::operator();

  template <typename T> //
  Result operator()(const evaluate::Constant<T> &x) const {
    if constexpr (T::category == common::TypeCategory::Logical) {
      return llvm::transformOptional(
          x.GetScalarValue(), [](auto &&v) { return v.IsTrue(); });
    } else {
      return std::nullopt;
    }
  }

  template <typename... Rs> //
  Result Combine(Result &&result, Rs &&...results) const {
    if constexpr (sizeof...(results) == 0) {
      return result;
    } else {
      if (result.has_value()) {
        return result;
      } else {
        return Combine(std::move(results)...);
      }
    }
  }
};
} // namespace

std::optional<bool> GetLogicalValue(const SomeExpr &expr) {
  return LogicalConstantVistor{}(expr);
}

namespace {
struct ContiguousHelper {
  ContiguousHelper(SemanticsContext &context)
      : fctx_(context.foldingContext()) {}

  template <typename Contained>
  std::optional<bool> Visit(const common::Indirection<Contained> &x) {
    return Visit(x.value());
  }
  template <typename Contained>
  std::optional<bool> Visit(const common::Reference<Contained> &x) {
    return Visit(x.get());
  }
  template <typename T> std::optional<bool> Visit(const evaluate::Expr<T> &x) {
    return common::visit([&](auto &&s) { return Visit(s); }, x.u);
  }
  template <typename T>
  std::optional<bool> Visit(const evaluate::Designator<T> &x) {
    return common::visit(
        [this](auto &&s) { return evaluate::IsContiguous(s, fctx_); }, x.u);
  }
  template <typename T> std::optional<bool> Visit(const T &) {
    // Everything else.
    return std::nullopt;
  }

private:
  evaluate::FoldingContext &fctx_;
};
} // namespace

// Return values:
// - std::optional<bool>{true} if the object is known to be contiguous
// - std::optional<bool>{false} if the object is known not to be contiguous
// - std::nullopt if the object contiguity cannot be determined
std::optional<bool> IsContiguous(
    SemanticsContext &semaCtx, const parser::OmpObject &object) {
  return common::visit( //
      common::visitors{//
          [&](const parser::Name &x) {
            // Any member of a common block must be contiguous.
            return std::optional<bool>{true};
          },
          [&](const parser::Designator &x) {
            evaluate::ExpressionAnalyzer ea{semaCtx};
            if (MaybeExpr maybeExpr{ea.Analyze(x)}) {
              return ContiguousHelper{semaCtx}.Visit(*maybeExpr);
            }
            return std::optional<bool>{};
          },
          [&](const parser::OmpObject::Invalid &) {
            return std::optional<bool>{};
          }},
      object.u);
}

struct DesignatorCollector : public evaluate::Traverse<DesignatorCollector,
                                 std::vector<SomeExpr>, false> {
  using Result = std::vector<SomeExpr>;
  using Base = evaluate::Traverse<DesignatorCollector, Result, false>;
  DesignatorCollector() : Base(*this) {}

  Result Default() const { return {}; }

  using Base::operator();

  template <typename T> //
  Result operator()(const evaluate::Designator<T> &x) const {
    // Once in a designator, don't traverse it any further (i.e. only
    // collect top-level designators).
    auto copy{x};
    return Result{AsGenericExpr(std::move(copy))};
  }

  template <typename... Rs> //
  Result Combine(Result &&result, Rs &&...results) const {
    Result v(std::move(result));
    auto moveAppend{[](auto &accum, auto &&other) {
      for (auto &&s : other) {
        accum.push_back(std::move(s));
      }
    }};
    (moveAppend(v, std::move(results)), ...);
    return v;
  }
};

std::vector<SomeExpr> GetAllDesignators(const SomeExpr &expr) {
  return DesignatorCollector{}(expr);
}

static bool HasCommonDesignatorSymbols(
    const evaluate::SymbolVector &baseSyms, const SomeExpr &other) {
  // Compare the designators used in "other" with the designators whose
  // symbols are given in baseSyms.
  // This is a part of the check if these two expressions can access the same
  // storage: if the designators used in them are different enough, then they
  // will be assumed not to access the same memory.
  //
  // Consider an (array element) expression x%y(w%z), the corresponding symbol
  // vector will be {x, y, w, z} (i.e. the symbols for these names).
  // Check whether this exact sequence appears anywhere in any the symbol
  // vector for "other". This will be true for x(y) and x(y+1), so this is
  // not a sufficient condition, but can be used to eliminate candidates
  // before doing more exhaustive checks.
  //
  // If any of the symbols in this sequence are function names, assume that
  // there is no storage overlap, mostly because it would be impossible in
  // general to determine what storage the function will access.
  // Note: if f is pure, then two calls to f will access the same storage
  // when called with the same arguments. This check is not done yet.

  if (llvm::any_of(
          baseSyms, [](const SymbolRef &s) { return s->IsSubprogram(); })) {
    // If there is a function symbol in the chain then we can't infer much
    // about the accessed storage.
    return false;
  }

  auto isSubsequence{// Is u a subsequence of v.
      [](const evaluate::SymbolVector &u, const evaluate::SymbolVector &v) {
        size_t us{u.size()}, vs{v.size()};
        if (us > vs) {
          return false;
        }
        for (size_t off{0}; off != vs - us + 1; ++off) {
          bool same{true};
          for (size_t i{0}; i != us; ++i) {
            if (u[i] != v[off + i]) {
              same = false;
              break;
            }
          }
          if (same) {
            return true;
          }
        }
        return false;
      }};

  evaluate::SymbolVector otherSyms{evaluate::GetSymbolVector(other)};
  return isSubsequence(baseSyms, otherSyms);
}

static bool HasCommonTopLevelDesignators(
    const std::vector<SomeExpr> &baseDsgs, const SomeExpr &other) {
  // Compare designators directly as expressions. This will ensure
  // that x(y) and x(y+1) are not flagged as overlapping, whereas
  // the symbol vectors for both of these would be identical.
  std::vector<SomeExpr> otherDsgs{GetAllDesignators(other)};

  for (auto &s : baseDsgs) {
    if (llvm::any_of(otherDsgs, [&](auto &&t) { return s == t; })) {
      return true;
    }
  }
  return false;
}

const SomeExpr *HasStorageOverlap(
    const SomeExpr &base, llvm::ArrayRef<SomeExpr> exprs) {
  evaluate::SymbolVector baseSyms{evaluate::GetSymbolVector(base)};
  std::vector<SomeExpr> baseDsgs{GetAllDesignators(base)};

  for (const SomeExpr &expr : exprs) {
    if (!HasCommonDesignatorSymbols(baseSyms, expr)) {
      continue;
    }
    if (HasCommonTopLevelDesignators(baseDsgs, expr)) {
      return &expr;
    }
  }
  return nullptr;
}

// Check if the ActionStmt is actually a [Pointer]AssignmentStmt. This is
// to separate cases where the source has something that looks like an
// assignment, but is semantically wrong (diagnosed by general semantic
// checks), and where the source has some other statement (which we want
// to report as "should be an assignment").
bool IsAssignment(const parser::ActionStmt *x) {
  if (x == nullptr) {
    return false;
  }

  using AssignmentStmt = common::Indirection<parser::AssignmentStmt>;
  using PointerAssignmentStmt =
      common::Indirection<parser::PointerAssignmentStmt>;

  return common::visit(
      [](auto &&s) -> bool {
        using BareS = llvm::remove_cvref_t<decltype(s)>;
        return std::is_same_v<BareS, AssignmentStmt> ||
            std::is_same_v<BareS, PointerAssignmentStmt>;
      },
      x->u);
}

bool IsPointerAssignment(const evaluate::Assignment &x) {
  return std::holds_alternative<evaluate::Assignment::BoundsSpec>(x.u) ||
      std::holds_alternative<evaluate::Assignment::BoundsRemapping>(x.u);
}

MaybeExpr MakeEvaluateExpr(const parser::OmpStylizedInstance &inp) {
  auto &instance = std::get<parser::OmpStylizedInstance::Instance>(inp.t);

  return common::visit( //
      common::visitors{
          [&](const parser::AssignmentStmt &s) -> MaybeExpr {
            return GetEvaluateExpr(std::get<parser::Expr>(s.t));
          },
          [&](const parser::CallStmt &s) -> MaybeExpr {
            assert(s.typedCall && "Expecting typedCall");
            const auto &procRef = *s.typedCall;
            return SomeExpr(procRef);
          },
          [&](const common::Indirection<parser::Expr> &s) -> MaybeExpr {
            return GetEvaluateExpr(s.value());
          },
      },
      instance.u);
}

bool IsLoopTransforming(llvm::omp::Directive dir) {
  switch (dir) {
  // TODO case llvm::omp::Directive::OMPD_flatten:
  case llvm::omp::Directive::OMPD_fuse:
  case llvm::omp::Directive::OMPD_interchange:
  case llvm::omp::Directive::OMPD_nothing:
  case llvm::omp::Directive::OMPD_reverse:
  // TODO case llvm::omp::Directive::OMPD_split:
  case llvm::omp::Directive::OMPD_stripe:
  case llvm::omp::Directive::OMPD_tile:
  case llvm::omp::Directive::OMPD_unroll:
    return true;
  default:
    return false;
  }
}

bool IsFullUnroll(const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};

  if (beginSpec.DirName().v == llvm::omp::Directive::OMPD_unroll) {
    return parser::omp::FindClause(
               beginSpec, llvm::omp::Clause::OMPC_partial) == nullptr;
  }
  return false;
}

namespace {
// Helper class to check if a given evaluate::Expr is an array expression.
// This does not check any proper subexpressions of the expression (except
// parentheses).
struct ArrayExpressionRecognizer {
  template <TypeCategory C>
  static bool isArrayExpression(
      const evaluate::Expr<evaluate::SomeKind<C>> &x) {
    return common::visit([](auto &&s) { return isArrayExpression(s); }, x.u);
  }

  template <TypeCategory C, int K>
  static bool isArrayExpression(const evaluate::Expr<evaluate::Type<C, K>> &x) {
    return common::visit([](auto &&s) { return isArrayExpression(s); },
        evaluate::match::deparen(x).u);
  }

  template <typename T>
  static bool isArrayExpression(const evaluate::Designator<T> &x) {
    if (auto *sym{std::get_if<SymbolRef>(&x.u)}) {
      return (*sym)->Rank() != 0;
    }
    if (auto *array{std::get_if<evaluate::ArrayRef>(&x.u)}) {
      return llvm::any_of(array->subscript(), [](const evaluate::Subscript &s) {
        // A vector subscript will not be a Triplet, but will have rank > 0.
        return std::holds_alternative<evaluate::Triplet>(s.u) || s.Rank() > 0;
      });
    }
    return false;
  }

  template <typename T> static bool isArrayExpression(const T &x) {
    return false;
  }

  static bool isArrayExpression(const evaluate::Expr<evaluate::SomeType> &x) {
    return common::visit([](auto &&s) { return isArrayExpression(s); }, x.u);
  }
};

/// Helper class to check if a given evaluate::Expr contains a subexpression
/// (not necessarily proper) that is an array expression.
struct ArrayExpressionFinder
    : public evaluate::AnyTraverse<ArrayExpressionFinder> {
  using Base = evaluate::AnyTraverse<ArrayExpressionFinder>;
  using Base::operator();
  ArrayExpressionFinder() : Base(*this) {}

  template <typename T>
  bool operator()(const evaluate::Designator<T> &x) const {
    return ArrayExpressionRecognizer::isArrayExpression(x);
  }
};

/// Helper class to check if any array expressions contained in the given
/// evaluate::Expr satisfy the criteria for being in "intervening code".
struct ArrayExpressionChecker {
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::Expr &parserExpr) {
    // If we have found a prohibited expression, skip the rest of the
    // traversal.
    if (!rejected) {
      if (auto expr{GetEvaluateExpr(parserExpr)}) {
        rejected = ArrayExpressionFinder{}(*expr);
      }
    }
    return !rejected;
  }

  bool rejected{false};
};
} // namespace

static bool ContainsInvalidArrayExpression(
    const parser::ExecutionPartConstruct &x) {
  ArrayExpressionChecker checker;
  parser::Walk(x, checker);
  return checker.rejected;
}

/// Checks if the given construct `x` satisfied OpenMP requirements for
/// intervening-code. Excludes CYCLE/EXIT statements as well as constructs
/// likely to result in a runtime loop, e.g. FORALL, WHERE, etc.
bool IsValidInterveningCode(const parser::ExecutionPartConstruct &x) {
  static auto isScalar = [](const parser::Variable &variable) {
    if (auto expr{GetEvaluateExprFromTyped(variable.typedExpr)}) {
      return expr->Rank() == 0;
    }
    return false;
  };

  auto *exec{parser::Unwrap<parser::ExecutableConstruct>(x)};
  if (!exec) {
    // DATA, ENTRY, FORMAT, NAMELIST are not explicitly prohibited in a CLN
    // although they are likely disallowed due to other requirements.
    // Return true, they should be rejected elsewhere if necessary.
    return true;
  }

  if (auto *action{parser::Unwrap<parser::ActionStmt>(exec->u)}) {
    if (parser::Unwrap<parser::CycleStmt>(action->u) ||
        parser::Unwrap<parser::ExitStmt>(action->u) ||
        parser::Unwrap<parser::ForallStmt>(action->u) ||
        parser::Unwrap<parser::WhereStmt>(action->u)) {
      return false;
    }
    if (auto *assign{parser::Unwrap<parser::AssignmentStmt>(&action->u)}) {
      if (!isScalar(std::get<parser::Variable>(assign->t))) {
        return false;
      }
    }
  } else { // Not ActionStmt
    if (parser::Unwrap<parser::LabelDoStmt>(exec->u) ||
        parser::Unwrap<parser::DoConstruct>(exec->u) ||
        parser::Unwrap<parser::ForallConstruct>(exec->u) ||
        parser::Unwrap<parser::WhereConstruct>(exec->u)) {
      return false;
    }
    if (auto *omp{parser::Unwrap<parser::OpenMPConstruct>(exec->u)}) {
      auto dirName{GetOmpDirectiveName(*omp)};
      if (llvm::omp::getDirectiveCategory(dirName.v) ==
          llvm::omp::Category::Executable) {
        return false;
      }
    }
  }

  if (ContainsInvalidArrayExpression(x)) {
    return false;
  }

  return true;
}

/// Checks if the given construct `x` preserves perfect nesting of a loop,
/// when placed adjacent to the loop in the enclosing (parent) loop.
/// CONTINUE statements are no-ops, and thus are considered transparent.
/// Non-OpenMP compiler directives are also considered transparent to
/// allow legacy applications to pass the semantic checks.
bool IsTransparentInterveningCode(const parser::ExecutionPartConstruct &x) {
  // Tolerate compiler directives in perfect nests.
  return parser::Unwrap<parser::CompilerDirective>(x) ||
      parser::Unwrap<parser::ContinueStmt>(x);
}

bool IsTransformableLoop(const parser::DoConstruct &loop) {
  return loop.IsDoNormal();
}

bool IsTransformableLoop(const parser::OpenMPLoopConstruct &omp) {
  return IsLoopTransforming(omp.BeginDir().DirId());
}

bool IsTransformableLoop(const parser::ExecutionPartConstruct &epc) {
  if (auto *loop{parser::Unwrap<parser::DoConstruct>(epc)}) {
    return IsTransformableLoop(*loop);
  }
  if (auto *omp{parser::Unwrap<parser::OpenMPLoopConstruct>(epc)}) {
    return IsTransformableLoop(*omp);
  }
  return false;
}

LoopSequence::LoopSequence(
    const parser::ExecutionPartConstruct &root, bool allowAllLoops)
    : allowAllLoops_(allowAllLoops) {
  entry_ = createConstructEntry(root);
  assert(entry_ && "Expecting loop like code");

  createChildrenFromRange(entry_->location);
  length_ = calculateLength();
}

LoopSequence::LoopSequence(std::unique_ptr<Construct> entry, bool allowAllLoops)
    : allowAllLoops_(allowAllLoops), entry_(std::move(entry)) {
  createChildrenFromRange(entry_->location);
  length_ = calculateLength();
}

std::unique_ptr<LoopSequence::Construct> LoopSequence::createConstructEntry(
    const parser::ExecutionPartConstruct &code) {
  if (auto *loop{parser::Unwrap<parser::DoConstruct>(code)}) {
    if (allowAllLoops_ || IsTransformableLoop(*loop)) {
      auto &body{std::get<parser::Block>(loop->t)};
      return std::make_unique<Construct>(body, &code);
    }
  } else if (auto *omp{parser::Unwrap<parser::OpenMPLoopConstruct>(code)}) {
    if (IsTransformableLoop(*omp)) {
      auto &body{std::get<parser::Block>(omp->t)};
      return std::make_unique<Construct>(body, &code);
    }
  }

  return nullptr;
}

void LoopSequence::createChildrenFromRange(
    ExecutionPartIterator::IteratorType begin,
    ExecutionPartIterator::IteratorType end) {
  for (auto &code : BlockRange(begin, end, BlockRange::Step::Over)) {
    if (auto entry{createConstructEntry(code)}) {
      children_.push_back(LoopSequence(std::move(entry), allowAllLoops_));
    }
  }
}

std::optional<int64_t> LoopSequence::calculateLength() const {
  if (!entry_->owner) {
    return sumOfChildrenLengths();
  }
  if (parser::Unwrap<parser::DoConstruct>(entry_->owner)) {
    return 1;
  }

  auto &omp{DEREF(parser::Unwrap<parser::OpenMPLoopConstruct>(*entry_->owner))};
  const parser::OmpDirectiveSpecification &beginSpec{omp.BeginDir()};
  llvm::omp::Directive dir{beginSpec.DirId()};
  if (!IsLoopTransforming(dir)) {
    return 0;
  }

  // TODO: Handle split, apply.
  if (IsFullUnroll(omp)) {
    return std::nullopt;
  }

  auto nestedCount{sumOfChildrenLengths()};

  if (dir == llvm::omp::Directive::OMPD_fuse) {
    // If there are no loops nested inside of FUSE, then the construct is
    // invalid. This case will be diagnosed when analyzing the body of the FUSE
    // construct itself, not when checking a construct in which the FUSE is
    // nested.
    // Returning std::nullopt prevents error messages caused by the same
    // problem from being emitted for every enclosing loop construct, for
    // example:
    //   !$omp do         ! error: this should contain a loop (superfluous)
    //   !$omp fuse       ! error: this should contain a loop
    //   !$omp end fuse
    if (!nestedCount || *nestedCount == 0) {
      return std::nullopt;
    }
    auto *clause{
        parser::omp::FindClause(beginSpec, llvm::omp::Clause::OMPC_looprange)};
    if (!clause) {
      return 1;
    }

    auto *loopRange{parser::Unwrap<parser::OmpLooprangeClause>(*clause)};
    std::optional<int64_t> count{GetIntValue(std::get<1>(loopRange->t))};
    if (!count || *count <= 0) {
      return std::nullopt;
    }
    if (*count <= *nestedCount) {
      return 1 + *nestedCount - *count;
    }
    return std::nullopt;
  }

  if (dir == llvm::omp::Directive::OMPD_nothing) {
    return nestedCount;
  }

  // For every other loop construct return 1.
  return 1;
}

std::optional<int64_t> LoopSequence::sumOfChildrenLengths() const {
  int64_t sum{0};
  for (auto &seq : children_) {
    if (auto len{seq.length()}) {
      sum += *len;
    } else {
      return std::nullopt;
    }
  }
  return sum;
}
} // namespace Fortran::semantics::omp
