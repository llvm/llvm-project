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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#include <array>
#include <cinttypes>
#include <list>
#include <memory>
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

bool IsPrivatizable(const Symbol &sym) {
  auto *misc{sym.detailsIf<MiscDetails>()};
  return IsVariableName(sym) && !IsProcedure(sym) && !IsStmtFunction(sym) &&
      !IsNamedConstant(sym) &&
      ( // OpenMP 5.2, 5.1.1: Assumed-size arrays are shared
          !semantics::IsAssumedSizeArray(sym) ||
          // If CrayPointer is among the DSA list then the
          // CrayPointee is Privatizable
          sym.test(Symbol::Flag::CrayPointee)) &&
      !sym.owner().IsDerivedType() &&
      sym.owner().kind() != Scope::Kind::ImpliedDos &&
      sym.owner().kind() != Scope::Kind::Forall &&
      !sym.detailsIf<semantics::AssocEntityDetails>() &&
      !sym.detailsIf<semantics::NamelistDetails>() &&
      (!misc ||
          (misc->kind() != MiscDetails::Kind::ComplexPartRe &&
              misc->kind() != MiscDetails::Kind::ComplexPartIm &&
              misc->kind() != MiscDetails::Kind::KindParamInquiry &&
              misc->kind() != MiscDetails::Kind::LenParamInquiry &&
              misc->kind() != MiscDetails::Kind::ConstructName));
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

  template <typename T>
  Result operator()(const evaluate::ConditionalExpr<T> &) const {
    // A conditional expression is not treated as a constant logical value.
    return std::nullopt;
  }
};
} // namespace

std::optional<bool> GetLogicalValue(const SomeExpr &expr) {
  return LogicalConstantVistor{}(expr);
}

std::optional<int64_t> GetIntValueFromExpr(
    const parser::Expr &parserExpr, SemanticsContext *semaCtx) {
  if (auto value{GetIntValue(parserExpr)}) {
    return value;
  }
  if (semaCtx) {
    if (auto expr{evaluate::ExpressionAnalyzer{*semaCtx}.Analyze(parserExpr)}) {
      return evaluate::ToInt64(expr);
    }
  }
  return std::nullopt;
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

std::vector<SomeExpr> GetTopLevelDesignators(const SomeExpr &expr) {
  return DesignatorCollector{}(expr);
}

static bool HasCommonDesignatorSymbols(
    const SymbolVector &baseSyms, const SomeExpr &other) {
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

  // Is u a subsequence of v.
  auto isSubsequence{[](const SymbolVector &u, const SymbolVector &v) {
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

  SymbolVector otherSyms{evaluate::GetSymbolVector(other)};
  return isSubsequence(baseSyms, otherSyms);
}

static bool HasCommonTopLevelDesignators(
    const std::vector<SomeExpr> &baseDsgs, const SomeExpr &other) {
  // Compare designators directly as expressions. This will ensure
  // that x(y) and x(y+1) are not flagged as overlapping, whereas
  // the symbol vectors for both of these would be identical.
  std::vector<SomeExpr> otherDsgs{GetTopLevelDesignators(other)};

  for (auto &s : baseDsgs) {
    if (llvm::any_of(otherDsgs, [&](auto &&t) { return s == t; })) {
      return true;
    }
  }
  return false;
}

const SomeExpr *HasStorageOverlap(
    const SomeExpr &base, llvm::ArrayRef<SomeExpr> exprs) {
  SymbolVector baseSyms{evaluate::GetSymbolVector(base)};
  std::vector<SomeExpr> baseDsgs{GetTopLevelDesignators(base)};

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

bool IsFullUnroll(const parser::OmpDirectiveSpecification &spec) {
  if (spec.DirId() == llvm::omp::Directive::OMPD_unroll) {
    return !parser::omp::FindClause(spec, llvm::omp::Clause::OMPC_partial);
  }
  return false;
}

static bool IsTransformableLoop(const parser::OmpDirectiveSpecification &spec) {
  return !IsFullUnroll(spec) && IsLoopTransforming(spec.DirId());
}

static bool IsTransformableLoop(const parser::ExecutionPartConstruct &epc) {
  if (auto *loop{parser::Unwrap<parser::DoConstruct>(epc)}) {
    return loop->IsDoNormal();
  }
  if (auto *omp{parser::Unwrap<parser::OpenMPLoopConstruct>(epc)}) {
    return IsTransformableLoop(omp->BeginDir());
  }
  return false;
}

LoopControl::LoopControl(const parser::LoopControl::Bounds &x) {
  iv = x.Name().thing.symbol;
  lbound = fromParserExpr(parser::UnwrapRef<parser::Expr>(x.Lower()));
  ubound = fromParserExpr(parser::UnwrapRef<parser::Expr>(x.Upper()));
  if (auto &inc{x.Step()}) {
    step = fromParserExpr(parser::UnwrapRef<parser::Expr>(*inc));
  }
}

LoopControl::LoopControl(const parser::ConcurrentControl &x) {
  auto &[name, lower, upper, inc]{x.t};
  iv = name.symbol;
  lbound = fromParserExpr(parser::UnwrapRef<parser::Expr>(lower));
  ubound = fromParserExpr(parser::UnwrapRef<parser::Expr>(upper));
  if (inc) {
    step = fromParserExpr(parser::UnwrapRef<parser::Expr>(inc));
  }
}

WithSource<MaybeExpr> LoopControl::fromParserExpr(const parser::Expr &x) {
  return WithSource<MaybeExpr>(GetEvaluateExpr(x), x.source);
}

std::vector<LoopControl> GetLoopControls(const parser::DoConstruct &x) {
  std::vector<LoopControl> controls;
  if (x.IsDoNormal()) {
    const parser::LoopControl &control{*x.GetLoopControl()};
    controls.emplace_back(std::get<parser::LoopControl::Bounds>(control.u));
  } else if (x.IsDoConcurrent()) {
    const parser::LoopControl &control{*x.GetLoopControl()};
    auto &header{parser::UnwrapRef<parser::ConcurrentHeader>(control)};
    for (auto &cc : std::get<std::list<parser::ConcurrentControl>>(header.t)) {
      controls.emplace_back(cc);
    }
  }
  return controls;
}

static const auto MsgNotValidAffectedLoop{
    "%s is not a valid affected loop"_because_en_US};
static const auto MsgClauseAbsentAssume{
    "%s clause was not specified, %s is assumed"_because_en_US};
static const auto MsgConstructDoesNotResult{
    "%s does not result in %s"_because_en_US};

Reason::Reason(const Reason &other) { //
  CopyFrom(other);
}

Reason &Reason::operator=(const Reason &other) {
  if (this != &other) {
    msgs.clear();
    CopyFrom(other);
  }
  return *this;
}

void Reason::CopyFrom(const Reason &other) {
  for (auto &msg : other.msgs.messages()) {
    msgs.Say(parser::Message(msg));
  }
}

parser::Message &Reason::AttachTo(parser::Message &msg) {
  msgs.AttachTo(msg);
  return msg;
}

/// From `vars` select the subsequence of symbols that are used in `expr`
/// either directly, or via some kind of association.
static SymbolVector SelectUsedSymbols(
    const SymbolVector &vars, const SomeExpr &expr) {
  llvm::DenseSet<const Symbol *> uses;
  for (SymbolRef s : evaluate::GetSymbolVector(expr)) {
    uses.insert(&s->GetUltimate());
  }

  SymbolVector deps;
  for (SymbolRef s : vars) {
    if (uses.count(&s->GetUltimate())) {
      deps.push_back(s);
    }
  }
  return deps;
}

WithReason<int64_t> GetArgumentValueWithReason(
    const parser::OmpDirectiveSpecification &spec, llvm::omp::Clause clauseId,
    unsigned version, SemanticsContext *semaCtx) {
  if (auto *clause{parser::omp::FindClause(spec, clauseId)}) {
    if (auto *expr{parser::Unwrap<parser::Expr>(clause->u)}) {
      if (auto value{GetIntValueFromExpr(*expr, semaCtx)}) {
        std::string name{GetUpperName(clauseId, version)};
        Reason reason;
        reason.Say(clause->source,
            "%s clause was specified with argument %" PRId64 ""_because_en_US,
            name, *value);
        return {*value, std::move(reason)};
      }
    }
  }
  return {};
}

template <typename T>
static WithReason<int64_t> GetNumArgumentsWithReasonForType(
    const parser::OmpClause &clause, const std::string &name) {
  if (auto *args{parser::Unwrap<std::list<T>>(clause.u)}) {
    auto num{static_cast<int64_t>(args->size())};
    Reason reason;
    reason.Say(clause.source,
        "%s clause was specified with %" PRId64 " arguments"_because_en_US,
        name, num);
    return {num, std::move(reason)};
  }
  return {};
}

WithReason<int64_t> GetNumArgumentsWithReason(
    const parser::OmpDirectiveSpecification &spec, llvm::omp::Clause clauseId,
    unsigned version, SemanticsContext *semaCtx) {
  if (auto *clause{parser::omp::FindClause(spec, clauseId)}) {
    std::string name{GetUpperName(clauseId, version)};
    // Try the types used for list items.
    {
      using Ty = parser::ScalarIntExpr;
      if (auto n{GetNumArgumentsWithReasonForType<Ty>(*clause, name)}) {
        return n;
      }
    }
    {
      using Ty = parser::ScalarIntConstantExpr;
      if (auto n{GetNumArgumentsWithReasonForType<Ty>(*clause, name)}) {
        return n;
      }
    }
  }
  return {};
}

WithReason<int64_t> GetHeightWithReason(
    const parser::OmpDirectiveSpecification &spec, unsigned version,
    SemanticsContext *semaCtx) {
  bool isFullUnroll{IsFullUnroll(spec)};

  if (!isFullUnroll && !IsTransformableLoop(spec)) {
    Reason reason;
    reason.Say(spec.DirName().source,
        "This construct is not a DO-loop or a loop-transformation construct"_because_en_US);
    return {0, reason};
  }

  switch (spec.DirId()) {
  // These generate loop sequences.
  case llvm::omp::Directive::OMPD_fuse:
  case llvm::omp::Directive::OMPD_split:
    return {0, Reason()};
  case llvm::omp::Directive::OMPD_flatten:
  case llvm::omp::Directive::OMPD_interchange:
  case llvm::omp::Directive::OMPD_nothing:
  case llvm::omp::Directive::OMPD_reverse:
  case llvm::omp::Directive::OMPD_stripe:
  case llvm::omp::Directive::OMPD_tile:
  case llvm::omp::Directive::OMPD_unroll: {
    auto [cons, _1]{GetAffectedNestDepthWithReason(spec, version, semaCtx)};
    auto [prod, _2]{GetGeneratedNestDepthWithReason(spec, version, semaCtx)};
    if (cons && prod) {
      return WithReason<int64_t>{*prod.value - *cons.value,
          Reason().Append(cons.reason).Append(prod.reason)};
    }
    return {};
  }
  default:
    llvm_unreachable("Expecting loop-transforming construct");
  }
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

template <typename T,
    typename = std::enable_if_t<std::is_arithmetic_v<llvm::remove_cvref_t<T>>>>
WithReason<T> operator+(const WithReason<T> &a, const WithReason<T> &b) {
  if (a.value && b.value) {
    return WithReason<T>{
        *a.value + *b.value, Reason().Append(a.reason).Append(b.reason)};
  }
  return WithReason<T>();
}

template <typename T,
    typename = std::enable_if_t<std::is_arithmetic_v<llvm::remove_cvref_t<T>>>>
WithReason<T> operator+(T a, const WithReason<T> &b) {
  return WithReason<T>{a, Reason()} + b;
}

/// Return the depth of the affected nest(s):
///   {affected-depth, must-be-perfect-nest}.
std::pair<WithReason<int64_t>, bool> GetAffectedNestDepthWithReason(
    const parser::OmpDirectiveSpecification &spec, unsigned version,
    SemanticsContext *semaCtx) {
  llvm::omp::Directive dir{spec.DirId()};
  bool allowsCollapse{llvm::omp::isAllowedClauseForDirective(
      dir, llvm::omp::Clause::OMPC_collapse, version)};
  bool allowsOrdered{llvm::omp::isAllowedClauseForDirective(
      dir, llvm::omp::Clause::OMPC_ordered, version)};

  if (allowsCollapse || allowsOrdered) {
    auto [ccount, creason]{GetArgumentValueWithReason(
        spec, llvm::omp::Clause::OMPC_collapse, version, semaCtx)};
    auto [ocount, oreason]{GetArgumentValueWithReason(
        spec, llvm::omp::Clause::OMPC_ordered, version, semaCtx)};
    // Ignore invalid arguments.
    if (ccount <= 0) {
      ccount = std::nullopt;
      creason = Reason();
    }
    if (ocount <= 0) {
      ocount = std::nullopt;
      oreason = Reason();
    }
    if (ccount < ocount) {
      // `ocount` cannot be std::nullopt here (C++ std guarantee).
      return {{ocount.value_or(1), std::move(oreason)}, true};
    }
    return {{ccount.value_or(1), std::move(creason)}, true};
  }

  if (IsLoopTransforming(dir)) {
    switch (dir) {
    case llvm::omp::Directive::OMPD_flatten:
      if (auto &&value{GetArgumentValueWithReason(
              spec, llvm::omp::Clause::OMPC_depth, version, semaCtx)}) {
        // FLATTEN DEPTH(n) replaces n loops with 1.
        return {std::move(value), true};
      } else {
        Reason reason;
        reason.Say(spec.DirName().source, MsgClauseAbsentAssume,
            GetUpperName(llvm::omp::Clause::OMPC_depth, version),
            "a depth of 2");
        return {{2, std::move(reason)}, true};
      }
      break;
    case llvm::omp::Directive::OMPD_interchange: {
      // Get the length of the argument list to PERMUTATION.
      if (parser::omp::FindClause(spec, llvm::omp::Clause::OMPC_permutation)) {
        auto [num, reason]{GetNumArgumentsWithReason(
            spec, llvm::omp::Clause::OMPC_permutation, version, semaCtx)};
        return {{num, std::move(reason)}, true};
      }
      // PERMUTATION not specified, assume PERMUTATION(2, 1).
      std::string name{
          GetUpperName(llvm::omp::Clause::OMPC_permutation, version)};
      Reason reason;
      reason.Say(
          spec.source, MsgClauseAbsentAssume, name, "a permutation (2, 1)");
      return {{2, std::move(reason)}, true};
    }
    case llvm::omp::Directive::OMPD_nothing:
      return {WithReason<int64_t>(0), false};
    case llvm::omp::Directive::OMPD_stripe:
    case llvm::omp::Directive::OMPD_tile: {
      // Get the length of the argument list to SIZES.
      auto [num, reason]{GetNumArgumentsWithReason(
          spec, llvm::omp::Clause::OMPC_sizes, version, semaCtx)};
      return {{num, std::move(reason)}, true};
    }
    case llvm::omp::Directive::OMPD_fuse: {
      // Get the value from the argument to DEPTH.
      if (parser::omp::FindClause(spec, llvm::omp::Clause::OMPC_depth)) {
        auto [count, reason]{GetArgumentValueWithReason(
            spec, llvm::omp::Clause::OMPC_depth, version, semaCtx)};
        return {{count, std::move(reason)}, true};
      }
      std::string name{GetUpperName(llvm::omp::Clause::OMPC_depth, version)};
      Reason reason;
      reason.Say(spec.source, MsgClauseAbsentAssume, name, "a value of 1");
      return {{1, std::move(reason)}, true};
    }
    case llvm::omp::Directive::OMPD_reverse:
    case llvm::omp::Directive::OMPD_split:
    case llvm::omp::Directive::OMPD_unroll:
      return {WithReason<int64_t>(1), false};
    default:
      break;
    }
  }

  return {{}, false};
}

/// Return the depth of the generated nest(s)
///   {generated-depth, is-perfect-nest}
std::pair<WithReason<int64_t>, bool> GetGeneratedNestDepthWithReason(
    const parser::OmpDirectiveSpecification &spec, unsigned version,
    SemanticsContext *semaCtx) {
  llvm::omp::Directive dir{spec.DirId()};
  if (!IsLoopTransforming(dir)) {
    return {{}, false};
  }

  auto [depth, _]{GetAffectedNestDepthWithReason(spec, version, semaCtx)};

  switch (dir) {
  case llvm::omp::Directive::OMPD_flatten:
    return {WithReason<int64_t>(1), true};
  case llvm::omp::Directive::OMPD_fuse:
  case llvm::omp::Directive::OMPD_split:
    // These result in loop sequences.
    return {{}, false};
  case llvm::omp::Directive::OMPD_interchange:
  case llvm::omp::Directive::OMPD_nothing:
  case llvm::omp::Directive::OMPD_reverse:
    return {depth, true};
  case llvm::omp::Directive::OMPD_stripe:
  case llvm::omp::Directive::OMPD_tile:
    if (depth) {
      return {
          WithReason<int64_t>(2 * *depth.value, std::move(depth.reason)), true};
    }
    return {{}, true};
  case llvm::omp::Directive::OMPD_unroll:
    if (IsFullUnroll(spec)) {
      return {WithReason<int64_t>(0), false};
    }
    return {WithReason<int64_t>(1), true};
  default:
    return {{}, false};
  }
}

/// Return the range of the affected nests in the sequence:
///   {first, count}
WithReason<std::pair<int64_t, int64_t>> GetAffectedLoopRangeWithReason(
    const parser::OmpDirectiveSpecification &spec, unsigned version,
    SemanticsContext *semaCtx) {
  llvm::omp::Directive dir{spec.DirId()};

  if (dir == llvm::omp::Directive::OMPD_fuse) {
    std::string name{GetUpperName(llvm::omp::Clause::OMPC_looprange, version)};
    if (auto *clause{
            parser::omp::FindClause(spec, llvm::omp::Clause::OMPC_looprange)}) {
      auto &range{DEREF(parser::Unwrap<parser::OmpLooprangeClause>(clause->u))};
      std::optional<int64_t> first{
          GetIntValueFromExpr(std::get<0>(range.t), semaCtx)};
      std::optional<int64_t> count{
          GetIntValueFromExpr(std::get<1>(range.t), semaCtx)};
      if (!first || !count || *first <= 0 || *count <= 0) {
        return {};
      }
      Reason reason;
      reason.Say(clause->source,
          "%s clause was specified with a count of %" PRId64
          " starting at loop %" PRId64 ""_because_en_US,
          name, *count, *first);
      return {std::make_pair(*first, *count), std::move(reason)};
    }
    // If LOOPRANGE was not found, return {1, -1}, where -1 means "the whole
    // associated sequence".
    Reason reason;
    reason.Say(
        spec.source, MsgClauseAbsentAssume, name, "the entire loop sequence");
    return {std::make_pair(1, -1), std::move(reason)};
  }

  assert(llvm::omp::getDirectiveAssociation(dir) ==
          llvm::omp::Association::LoopNest &&
      "Expecting loop-nest-associated construct");
  // For loop-nest constructs, a single loop-nest is affected.
  return {std::make_pair(1, 1), Reason()};
}

WithReason<int64_t> GetRectangularNestDepthWithReason(
    const parser::OmpDirectiveSpecification &spec, unsigned version,
    SemanticsContext *semaCtx) {
  auto [depth, _]{GetAffectedNestDepthWithReason(spec, version, semaCtx)};
  if (!depth) {
    return {};
  }

  // Remove the reasons for the affected depth. Reasons for needing
  // rectangular loops will be added instead.
  depth.reason.msgs.clear();

  static const std::array directives{
      llvm::omp::Directive::OMPD_interchange,
      llvm::omp::Directive::OMPD_stripe,
      llvm::omp::Directive::OMPD_tile,
  };

  llvm::omp::Directive dirId{spec.DirId()};
  if (llvm::is_contained(directives, dirId)) {
    depth.reason.Say(spec.DirName().source,
        "None of the loops affected by %s can be non-rectangular"_because_en_US,
        GetUpperName(dirId, version));
    return std::move(depth);
  }

  static const std::array clauses{
      llvm::omp::Clause::OMPC_dist_schedule,
      llvm::omp::Clause::OMPC_grainsize,
      llvm::omp::Clause::OMPC_induction,
      llvm::omp::Clause::OMPC_linear,
      llvm::omp::Clause::OMPC_schedule,
  };

  auto clauseAt{
      llvm::find_if(spec.Clauses().v, [&](const parser::OmpClause &c) {
        llvm::omp::Clause clauseId{c.Id()};
        return llvm::is_contained(clauses, clauseId) &&
            llvm::omp::isAllowedClauseForDirective(dirId, clauseId, version);
      })};
  if (clauseAt != spec.Clauses().v.end()) {
    depth.reason.Say(clauseAt->source,
        "When %s clause is present, none of the loops affected by %s can be non-rectangular"_because_en_US,
        GetUpperName(clauseAt->Id(), version), GetUpperName(dirId, version));
    return std::move(depth);
  }

  // No restrictions.
  return {0, Reason()};
}

std::optional<int64_t> GetMinimumSequenceCount(
    std::optional<int64_t> first, std::optional<int64_t> count) {
  if (first && count && *first > 0) {
    if (*count > 0) {
      return *first + *count - 1;
    } else if (*count == -1) {
      return -1;
    }
  }
  return std::nullopt;
}

std::optional<int64_t> GetMinimumSequenceCount(
    std::optional<std::pair<int64_t, int64_t>> range) {
  if (range) {
    return GetMinimumSequenceCount(range->first, range->second);
  }
  return GetMinimumSequenceCount(std::nullopt, std::nullopt);
}

#ifdef EXPENSIVE_CHECKS
namespace {
/// Check that for every value x of type T, there will be a "source" member
/// somewhere in x. This is to specifically make sure that parser::GetSource
/// will return something for any parser::ExecutionPartConstruct.

template <typename...> struct HasSourceT {
  static constexpr bool value{false};
};

template <typename T> struct HasSourceT<T> {
private:
  using U = llvm::remove_cvref_t<T>;

  static constexpr bool check() {
    if constexpr (parser::HasSource<U>::value) {
      return true;
    } else if constexpr (ConstraintTrait<U>) {
      return HasSourceT<decltype(U::thing)>::value;
    } else if constexpr (WrapperTrait<U>) {
      return HasSourceT<decltype(U::v)>::value;
    } else if constexpr (TupleTrait<U>) {
      return HasSourceT<decltype(U::t)>::value;
    } else if constexpr (UnionTrait<U>) {
      return HasSourceT<decltype(U::u)>::value;
    } else {
      return false;
    }
  }

public:
  static constexpr bool value{check()};
};

template <> struct HasSourceT<parser::ErrorRecovery> {
  static constexpr bool value{true};
};

template <typename T> struct HasSourceT<common::Indirection<T>> {
  static constexpr bool value{HasSourceT<T>::value};
};

template <typename... Ts> struct HasSourceT<std::tuple<Ts...>> {
  static constexpr bool value{(HasSourceT<Ts>::value || ...)};
};

template <typename... Ts> struct HasSourceT<std::variant<Ts...>> {
  static constexpr bool value{(HasSourceT<Ts>::value && ...)};
};

static_assert(HasSourceT<parser::ExecutionPartConstruct>::value);
} // namespace
#endif // EXPENSIVE_CHECKS

LoopSequence::LoopSequence(const parser::ExecutionPartConstruct &root,
    unsigned version, bool allowAllLoops, SemanticsContext *semaCtx)
    : version_(version), allowAllLoops_(allowAllLoops), semaCtx_(semaCtx) {
  entry_ = createConstructEntry(root);
  assert(entry_ && "Expecting loop like code");

  createChildrenFromRange(entry_->location);
  precalculate();
}

LoopSequence::LoopSequence(std::unique_ptr<Construct> entry, unsigned version,
    bool allowAllLoops, SemanticsContext *semaCtx)
    : version_(version), allowAllLoops_(allowAllLoops),
      entry_(std::move(entry)), semaCtx_(semaCtx) {
  createChildrenFromRange(entry_->location);
  precalculate();
}

std::unique_ptr<LoopSequence::Construct> LoopSequence::createConstructEntry(
    const parser::ExecutionPartConstruct &code) {
  if (auto *loop{parser::Unwrap<parser::DoConstruct>(code)}) {
    if (allowAllLoops_ || IsTransformableLoop(code)) {
      auto &body{std::get<parser::Block>(loop->t)};
      return std::make_unique<Construct>(body, &code);
    }
  } else if (auto *omp{parser::Unwrap<parser::OpenMPLoopConstruct>(code)}) {
    // Allow all loop constructs. This helps with better diagnostics, e.g.
    // "this is not a loop-transforming construct", insted of just "this is
    // not a valid intervening code".
    auto &body{std::get<parser::Block>(omp->t)};
    return std::make_unique<Construct>(body, &code);
  }

  return nullptr;
}

void LoopSequence::createChildrenFromRange(
    ExecutionPartIterator::IteratorType begin,
    ExecutionPartIterator::IteratorType end) {
  bool invalidWithEntry{false};
  // Create children. If there is zero or one, this LoopSequence could be
  // a nest. If there are more, it could be a proper sequence. In the latter
  // case any code between consecutive children must be "transparent".
  for (auto &code : BlockRange(begin, end, BlockRange::Step::Over)) {
    if (auto entry{createConstructEntry(code)}) {
      children_.push_back(
          LoopSequence(std::move(entry), version_, allowAllLoops_, semaCtx_));
      // Even when DO WHILE et al are allowed to have entries, still treat
      // them as invalid intervening code.
      // Give it priority over other kinds of invalid interveninig code.
      if (!invalidWithEntry && !IsTransformableLoop(code)) {
        invalidIC_ = &code;
        invalidWithEntry = true;
      }
    } else {
      if (!invalidIC_ && !IsValidInterveningCode(code)) {
        invalidIC_ = &code;
      }
      if (!opaqueIC_ && !IsTransparentInterveningCode(code)) {
        opaqueIC_ = &code;
      }
    }
  }
}

const LoopSequence *LoopSequence::getNestedDoConcurrent() const {
  // DO CONCURRENT loops are considered invalid code, even though they
  // can be allowed in some circumstances.
  if (!invalidIC_) {
    return nullptr;
  }
  // The invalidIC_ will point to the DO CONCURRENT if that's the only
  // invalid loop construct, but it may also point to DO WHILE.
  for (auto &sequence : children()) {
    auto &owner{DEREF(sequence.entry_->owner)};
    if (auto *loop{parser::Unwrap<parser::DoConstruct>(owner)}) {
      if (loop->IsDoConcurrent()) {
        return &sequence;
      }
    }
  }
  return nullptr;
}

std::vector<LoopControl> LoopSequence::getLoopControls() const {
  if (!entry_->owner) {
    return {};
  }

  if (auto *loop{parser::Unwrap<parser::DoConstruct>(*entry_->owner)}) {
    return GetLoopControls(*loop);
  }
  return {};
}

void LoopSequence::precalculate() {
  // Calculate length before depths.
  length_ = calculateLength();
  depth_ = calculateDepths();
  height_ = calculateHeight();
}

WithReason<int64_t> LoopSequence::calculateLength() const {
  if (!entry_->owner) {
    return getNestedLength();
  }
  if (parser::Unwrap<parser::DoConstruct>(entry_->owner)) {
    return WithReason<int64_t>(1);
  }

  auto &omp{DEREF(parser::Unwrap<parser::OpenMPLoopConstruct>(*entry_->owner))};
  const parser::OmpDirectiveSpecification &beginSpec{omp.BeginDir()};
  llvm::omp::Directive dir{beginSpec.DirId()};
  if (!IsLoopTransforming(dir)) {
    Reason reason;
    reason.Say(beginSpec.DirName().source, MsgConstructDoesNotResult,
        GetUpperName(dir, version_), "a loop nest or a loop sequence");
    return {0, std::move(reason)};
  }

  // TODO: Handle split, apply.
  if (IsFullUnroll(beginSpec)) {
    return {};
  }

  auto nestedLength{getNestedLength()};

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
    if (!nestedLength.value || *nestedLength.value == 0) {
      return {};
    }
    auto *clause{
        parser::omp::FindClause(beginSpec, llvm::omp::Clause::OMPC_looprange)};
    if (!clause) {
      Reason reason;
      reason.Say(beginSpec.DirName().source, MsgClauseAbsentAssume,
          GetUpperName(llvm::omp::Clause::OMPC_looprange, version_),
          "the entire loop sequence");
      return {1, std::move(reason)};
    }

    auto *loopRange{parser::Unwrap<parser::OmpLooprangeClause>(*clause)};
    std::optional<int64_t> count{
        GetIntValueFromExpr(std::get<1>(loopRange->t), semaCtx_)};
    if (!count || *count <= 0) {
      return {};
    }
    if (*count <= *nestedLength.value) {
      int64_t result{1 + *nestedLength.value - *count};
      Reason reason;
      reason.Say(beginSpec.DirName().source,
          "Out of %" PRId64 " loops, %" PRId64 " are fused"_because_en_US,
          *nestedLength.value, *count);
      return {result, std::move(reason)};
    }
    return {};
  }

  if (dir == llvm::omp::Directive::OMPD_nothing) {
    return nestedLength;
  }

  // For every other loop construct return 1.
  return {1, Reason()};
}

WithReason<int64_t> LoopSequence::getNestedLength() const {
  WithReason<int64_t> sum(0);
  for (auto &seq : children_) {
    if (const auto &len{seq.length()}) {
      sum = sum + len;
    } else {
      return {};
    }
  }
  return sum;
}

static void ResetIfPositiveWithReason(
    WithReason<int64_t> &quantity, const Reason &reason) {
  if (quantity.value > 0) {
    quantity.value = 0;
    quantity.reason.Append(reason);
  }
}

static void ResetIfPositiveWithReason(WithReason<int64_t> &quantity,
    parser::CharBlock source, parser::MessageFixedText msg) {
  if (quantity.value > 0) {
    quantity.value = 0;
    quantity.reason.Say(source, msg);
  }
}

static Reason WhyNotWellFormed(
    const parser::ExecutionPartConstruct &badCode, bool isSequence);

LoopSequence::Depth LoopSequence::calculateDepths() const {
  // Get the length of the nested sequence. The invalidIC_ and opaqueIC_
  // members do not include sibling canonical loop nests, but there can
  // only be one for depth to make sense.
  WithReason<int64_t> nestedLength{getNestedLength()};
  // Get the depths of the code nested in this sequence (e.g. contained in
  // entry_), and use it as the basis for the depths of entry_->owner.
  auto [semaDepth, perfDepth]{getNestedDepths()};
  if (invalidIC_) {
    auto whyNot{WhyNotWellFormed(*invalidIC_, false)};
    ResetIfPositiveWithReason(semaDepth, whyNot);
    ResetIfPositiveWithReason(perfDepth, whyNot);
  } else if (opaqueIC_) {
    auto message{"This code prevents perfect nesting"_because_en_US};
    parser::CharBlock source{*parser::GetSource(*opaqueIC_)};
    ResetIfPositiveWithReason(perfDepth, source, message);
  }
  if (nestedLength.value.value_or(0) != 1) {
    // This may simply be the bottom of the loop nest. Only emit messages
    // if the depths are reset back to 0.
    if (entry_->owner) {
      auto message{"This construct does not contain a loop nest"_because_en_US};
      parser::CharBlock source{*parser::GetSource(*entry_->owner)};
      ResetIfPositiveWithReason(semaDepth, source, message);
      ResetIfPositiveWithReason(perfDepth, source, message);
    }
    semaDepth.value = perfDepth.value = 0;
  }

  if (!entry_->owner) {
    return Depth{semaDepth, perfDepth};
  }
  if (parser::Unwrap<parser::DoConstruct>(entry_->owner)) {
    return Depth{int64_t(1) + semaDepth, int64_t(1) + perfDepth};
  }

  auto &omp{DEREF(parser::Unwrap<parser::OpenMPLoopConstruct>(*entry_->owner))};
  const parser::OmpDirectiveSpecification &beginSpec{omp.BeginDir()};
  llvm::omp::Directive dir{beginSpec.DirId()};
  bool isFullUnroll{IsFullUnroll(beginSpec)};

  // Check full unroll separately.
  if (!isFullUnroll && !IsTransformableLoop(beginSpec)) {
    Reason reason;
    reason.Say(beginSpec.DirName().source,
        "This construct is not a DO-loop or a loop-nest-generating construct"_because_en_US);
    return Depth{{0, reason}, {0, reason}};
  }

  switch (dir) {
  // TODO: case llvm::omp::Directive::OMPD_split:
  // TODO: case llvm::omp::Directive::OMPD_flatten:
  case llvm::omp::Directive::OMPD_fuse:
    if (auto *clause{parser::omp::FindClause(
            beginSpec, llvm::omp::Clause::OMPC_depth)}) {
      auto &expr{parser::UnwrapRef<parser::Expr>(clause->u)};
      auto value{GetIntValueFromExpr(expr, semaCtx_)};
      // The result is a perfect nest only if all loop in the sequence
      // are fused.
      if (value && nestedLength.value) {
        auto range{
            GetAffectedLoopRangeWithReason(beginSpec, version_, semaCtx_)};
        if (auto required{GetMinimumSequenceCount(range.value)}) {
          if (*required == -1 || *required == *nestedLength.value) {
            return Depth{value, value};
          }
          std::string name{
              GetUpperName(llvm::omp::Directive::OMPD_fuse, version_)};
          Reason reason(std::move(range.reason));
          reason.Say(beginSpec.DirName().source, MsgConstructDoesNotResult,
              "This " + name + " construct",
              "a loop nest, but a proper loop sequence");
          return Depth{{1, reason}, {1, reason}};
        }
      }
      return Depth{};
    }
    // FUSE cannot create a nest of depth > 1 without DEPTH clause.
    return Depth{WithReason<int64_t>(1), WithReason<int64_t>(1)};
  case llvm::omp::Directive::OMPD_interchange:
  case llvm::omp::Directive::OMPD_nothing:
  case llvm::omp::Directive::OMPD_reverse:
    return {semaDepth, perfDepth};
  case llvm::omp::Directive::OMPD_stripe:
  case llvm::omp::Directive::OMPD_tile:
    // Look for SIZES clause.
    if (auto *clause{parser::omp::FindClause(
            beginSpec, llvm::omp::Clause::OMPC_sizes)}) {
      // Return the number of arguments in the SIZES clause
      size_t num{
          parser::UnwrapRef<parser::OmpClause::Sizes>(clause->u).v.size()};
      return Depth{//
          static_cast<int64_t>(num) + semaDepth,
          static_cast<int64_t>(num) + perfDepth};
    }
    // The SIZES clause is mandatory, if it's missing the result is unknown.
    return Depth{};
  case llvm::omp::Directive::OMPD_unroll:
    if (isFullUnroll) {
      Reason reason;
      reason.Say(beginSpec.DirName().source, MsgConstructDoesNotResult,
          "Fully unrolled loop", "a loop nest");
      return Depth{{0, reason}, {0, reason}};
    }
    // If this is not a full unroll then look for a PARTIAL clause.
    if (auto *clause{parser::omp::FindClause(
            beginSpec, llvm::omp::Clause::OMPC_partial)}) {
      std::optional<int64_t> factor;
      if (auto *expr{parser::Unwrap<parser::Expr>(clause->u)}) {
        factor = GetIntValueFromExpr(*expr, semaCtx_);
      }
      // If it's a partial unroll, and the unroll count is 1, then this
      // construct is a no-op.
      if (factor && *factor == 1) {
        return Depth{semaDepth, perfDepth};
      }
      // If it's a proper partial unroll, then the resulting loop cannot
      // have either depth greater than 1: if it had a loop nested in it,
      // then after unroll it will have at least two copies it it, making
      // it a final loop.
      Reason reason;
      reason.Say(beginSpec.DirName().source,
          "Partially unrolled loop cannot form a nest of depth > 1"_because_en_US);
      return {{1, reason}, {1, reason}};
    }
    return Depth{};
  default:
    llvm_unreachable("Expecting loop-transforming construct");
  }
}

LoopSequence::Depth LoopSequence::getNestedDepths() const {
  if (!isNest()) {
    // If the current sequence is not a nest, it can still be a part of
    // an enclosing nest.
    return Depth{WithReason<int64_t>(0), WithReason<int64_t>(0)};
  } else if (children_.empty()) {
    // No children, but length == 1.
    assert(entry_->owner &&
        parser::Unwrap<parser::DoConstruct>(entry_->owner) &&
        "Expecting DO construct");
    return Depth{WithReason<int64_t>(0), WithReason<int64_t>(0)};
  }
  return children_.front().depth_;
}

WithReason<int64_t> LoopSequence::calculateHeight() const {
  if (!entry_->owner) {
    return {0, Reason()};
  }
  if (parser::Unwrap<parser::DoConstruct>(*entry_->owner)) {
    return {1, Reason()};
  }
  if (auto *omp{parser::Unwrap<parser::OpenMPLoopConstruct>(*entry_->owner)}) {
    const parser::OmpDirectiveSpecification &beginSpec{omp->BeginDir()};
    if (IsLoopTransforming(beginSpec.DirId())) {
      return GetHeightWithReason(beginSpec, version_, semaCtx_);
    }
    return {0, Reason()};
  }
  return {};
}

static bool IsDoConcurrent(const parser::ExecutionPartConstruct &x) {
  if (auto *loop{parser::Unwrap<parser::DoConstruct>(x)}) {
    return loop->IsDoConcurrent();
  }
  return false;
}

static Reason WhyNotWellFormed(
    const parser::ExecutionPartConstruct &badCode, bool isSequence) {
  Reason reason;
  parser::CharBlock source{*parser::GetSource(badCode)};
  if (auto *omp{parser::Unwrap<parser::OpenMPLoopConstruct>(badCode)}) {
    const parser::OmpDirectiveSpecification &beginSpec{omp->BeginDir()};
    if (IsFullUnroll(beginSpec)) {
      reason.Say(source, MsgConstructDoesNotResult, "Fully unrolled loop",
          isSequence ? "a loop nest or a loop sequence" : "a loop nest");
    } else if (!IsLoopTransforming(beginSpec.DirId())) {
      reason.Say(source,
          "Only loop-transforming constructs are allowed inside loop constructs"_because_en_US);
    }
    return reason;
  }

  if (auto *loop{parser::Unwrap<parser::DoConstruct>(badCode)}) {
    if (loop->IsDoWhile()) {
      reason.Say(source, MsgNotValidAffectedLoop, "DO WHILE loop");
    } else if (loop->IsDoConcurrent()) {
      reason.Say(source, MsgNotValidAffectedLoop, "DO CONCURRENT loop");
    } else if (!loop->GetLoopControl()) {
      reason.Say(
          source, MsgNotValidAffectedLoop, "DO loop without loop control");
    }
    if (reason) {
      return reason;
    }
  }
  reason.Say(source,
      "The %s contains code that prevents it from being canonical at this nesting level"_because_en_US,
      isSequence ? "sequence" : "nest");
  return reason;
}

WithReason<bool> LoopSequence::isWellFormedSequence() const {
  const parser::ExecutionPartConstruct *badCode{
      invalidIC_ ? invalidIC_ : opaqueIC_};
  if (badCode) {
    return {false, WhyNotWellFormed(*badCode, true)};
  }
  return {true, Reason()};
}

WithReason<bool> LoopSequence::isWellFormedNest() const {
  // DO CONCURRENT is allowed at the top level in OpenMP 6.0+.
  if (invalidIC_) {
    if (!IsDoConcurrentLegal(version_) || !IsDoConcurrent(*invalidIC_)) {
      return {false, WhyNotWellFormed(*invalidIC_, false)};
    }
  }
  return {true, Reason()};
}

static std::string JoinSymbolNames(const SymbolVector &syms) {
  std::vector<std::string> names;
  for (SymbolRef s : syms) {
    names.push_back("'" + s->name().ToString() + "'");
  }
  return llvm::join(names, ", ");
}

static void CheckSymbolExprOverlap(WithReason<bool> &result,
    const SymbolVector &syms, const SomeExpr &expr, std::string exprName,
    parser::CharBlock exprSource) {
  if (auto used{SelectUsedSymbols(syms, expr)}; !used.empty()) {
    result.value = false;
    result.reason.Say(exprSource,
        "The %s of the affected loop uses iteration variables of enclosing loops: %s"_because_en_US,
        exprName, JoinSymbolNames(used));
  }
}

WithReason<bool> LoopSequence::isRectangular(
    const std::vector<const LoopSequence *> &outer) const {
  assert(entry_->owner && "Must have owner construct");
  auto *loop{parser::Unwrap<parser::DoConstruct>(*entry_->owner)};
  if (!loop) {
    // Can "rectangular" property be computed for a loop-nest-generating
    // construct? What if the loops in the nest are not rectangular with
    // respect to each other?
    return {};
  }

  SymbolVector outerIVs;
  for (auto *sequence : llvm::reverse(outer)) {
    for (auto &control : sequence->getLoopControls()) {
      if (control.iv) {
        outerIVs.emplace_back(*control.iv);
      }
    }
  }

  WithReason<bool> result(true);

  for (auto &control : getLoopControls()) {
    if (!control.iv || !control.lbound.value || !control.ubound.value) {
      continue;
    }
    CheckSymbolExprOverlap(result, outerIVs, *control.lbound.value,
        "lower bound", control.lbound.source);
    CheckSymbolExprOverlap(result, outerIVs, *control.ubound.value,
        "upper bound", control.ubound.source);
    if (control.step.value) {
      CheckSymbolExprOverlap(result, outerIVs, *control.step.value,
          "iteration step", control.step.source);
    }
  }

  return result;
}
} // namespace Fortran::semantics::omp
