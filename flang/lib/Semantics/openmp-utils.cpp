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
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Evaluate/variable.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"

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

bool IsVarOrFunctionRef(const MaybeExpr &expr) {
  if (expr) {
    return evaluate::UnwrapProcedureRef(*expr) != nullptr ||
        evaluate::IsVariable(*expr);
  } else {
    return false;
  }
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

std::optional<SomeExpr> GetEvaluateExpr(const parser::Expr &parserExpr) {
  const parser::TypedExpr &typedExpr{parserExpr.typedExpr};
  // ForwardOwningPointer           typedExpr
  // `- GenericExprWrapper          ^.get()
  //    `- std::optional<Expr>      ^->v
  return DEREF(typedExpr.get()).v;
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

/// parser::Block is a list of executable constructs, parser::BlockConstruct
/// is Fortran's BLOCK/ENDBLOCK construct.
/// Strip the outermost BlockConstructs, return the reference to the Block
/// in the executable part of the innermost of the stripped constructs.
/// Specifically, if the given `block` has a single entry (it's a list), and
/// the entry is a BlockConstruct, get the Block contained within. Repeat
/// this step as many times as possible.
const parser::Block &GetInnermostExecPart(const parser::Block &block) {
  const parser::Block *iter{&block};
  while (iter->size() == 1) {
    const parser::ExecutionPartConstruct &ep{iter->front()};
    if (auto *bc{GetFortranBlockConstruct(ep)}) {
      iter = &std::get<parser::Block>(bc->t);
    } else {
      break;
    }
  }
  return *iter;
}

bool IsStrictlyStructuredBlock(const parser::Block &block) {
  if (block.size() == 1) {
    return GetFortranBlockConstruct(block.front()) != nullptr;
  } else {
    return false;
  }
}

} // namespace Fortran::semantics::omp
