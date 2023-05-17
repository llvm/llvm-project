//===-- include/flang/Semantics/expression.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_EXPRESSION_H_
#define FORTRAN_SEMANTICS_EXPRESSION_H_

#include "semantics.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/indirection.h"
#include "flang/Common/restorer.h"
#include "flang/Common/visit.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include <map>
#include <optional>
#include <type_traits>
#include <variant>

using namespace Fortran::parser::literals;

namespace Fortran::parser {
struct SourceLocationFindingVisitor {
  template <typename A> bool Pre(const A &x) {
    if constexpr (HasSource<A>::value) {
      source.ExtendToCover(x.source);
      return false;
    } else {
      return true;
    }
  }
  template <typename A> void Post(const A &) {}
  void Post(const CharBlock &at) { source.ExtendToCover(at); }

  CharBlock source;
};

template <typename A> CharBlock FindSourceLocation(const A &x) {
  SourceLocationFindingVisitor visitor;
  Walk(x, visitor);
  return visitor.source;
}
} // namespace Fortran::parser

using namespace Fortran::parser::literals;

// The expression semantic analysis code has its implementation in
// namespace Fortran::evaluate, but the exposed API to it is in the
// namespace Fortran::semantics (below).
//
// The ExpressionAnalyzer wraps a SemanticsContext reference
// and implements constraint checking on expressions using the
// parse tree node wrappers that mirror the grammar annotations used
// in the Fortran standard (i.e., scalar-, constant-, &c.).

namespace Fortran::evaluate {

class IntrinsicProcTable;

struct SetExprHelper {
  explicit SetExprHelper(GenericExprWrapper &&expr) : expr_{std::move(expr)} {}
  void Set(parser::TypedExpr &x) {
    x.Reset(new GenericExprWrapper{std::move(expr_)},
        evaluate::GenericExprWrapper::Deleter);
  }
  template <typename T> void Set(const common::Indirection<T> &x) {
    Set(x.value());
  }
  template <typename T> void Set(const T &x) {
    if constexpr (parser::HasTypedExpr<T>::value) {
      Set(x.typedExpr);
    } else if constexpr (ConstraintTrait<T>) {
      Set(x.thing);
    } else if constexpr (WrapperTrait<T>) {
      Set(x.v);
    }
  }

  GenericExprWrapper expr_;
};

template <typename T> void ResetExpr(const T &x) {
  SetExprHelper{GenericExprWrapper{/* error indicator */}}.Set(x);
}

template <typename T> void SetExpr(const T &x, Expr<SomeType> &&expr) {
  SetExprHelper{GenericExprWrapper{std::move(expr)}}.Set(x);
}

class ExpressionAnalyzer {
public:
  using MaybeExpr = std::optional<Expr<SomeType>>;

  explicit ExpressionAnalyzer(semantics::SemanticsContext &sc) : context_{sc} {}
  ExpressionAnalyzer(semantics::SemanticsContext &sc, FoldingContext &fc)
      : context_{sc}, foldingContext_{fc} {}
  ExpressionAnalyzer(const ExpressionAnalyzer &) = default;

  semantics::SemanticsContext &context() const { return context_; }
  bool inWhereBody() const { return inWhereBody_; }
  void set_inWhereBody(bool yes = true) { inWhereBody_ = yes; }
  bool inDataStmtObject() const { return inDataStmtObject_; }
  void set_inDataStmtObject(bool yes = true) { inDataStmtObject_ = yes; }

  FoldingContext &GetFoldingContext() const { return foldingContext_; }

  parser::ContextualMessages &GetContextualMessages() {
    return foldingContext_.messages();
  }

  template <typename... A> parser::Message *Say(A &&...args) {
    return GetContextualMessages().Say(std::forward<A>(args)...);
  }

  template <typename T, typename... A>
  parser::Message *SayAt(const T &parsed, A &&...args) {
    return Say(parser::FindSourceLocation(parsed), std::forward<A>(args)...);
  }

  int GetDefaultKind(common::TypeCategory);
  DynamicType GetDefaultKindOfType(common::TypeCategory);

  // Return false and emit error if these checks fail:
  bool CheckIntrinsicKind(TypeCategory, std::int64_t kind);
  bool CheckIntrinsicSize(TypeCategory, std::int64_t size);

  // Manage a set of active implied DO loops.
  bool AddImpliedDo(parser::CharBlock, int kind);
  void RemoveImpliedDo(parser::CharBlock);

  // When the argument is the name of an active implied DO index, returns
  // its INTEGER kind type parameter.
  std::optional<int> IsImpliedDo(parser::CharBlock) const;

  common::Restorer<bool> DoNotUseSavedTypedExprs() {
    return common::ScopedSet(useSavedTypedExprs_, false);
  }

  Expr<SubscriptInteger> AnalyzeKindSelector(common::TypeCategory category,
      const std::optional<parser::KindSelector> &);

  MaybeExpr Analyze(const parser::Expr &);
  MaybeExpr Analyze(const parser::Variable &);
  MaybeExpr Analyze(const parser::Selector &);
  MaybeExpr Analyze(const parser::Designator &);
  MaybeExpr Analyze(const parser::DataStmtValue &);
  MaybeExpr Analyze(const parser::AllocateObject &);
  MaybeExpr Analyze(const parser::PointerObject &);

  template <typename A> MaybeExpr Analyze(const common::Indirection<A> &x) {
    return Analyze(x.value());
  }
  template <typename A> MaybeExpr Analyze(const std::optional<A> &x) {
    if (x) {
      return Analyze(*x);
    } else {
      return std::nullopt;
    }
  }

  // Implement constraint-checking wrappers from the Fortran grammar.
  template <typename A> MaybeExpr Analyze(const parser::Scalar<A> &x) {
    auto result{Analyze(x.thing)};
    if (result) {
      if (int rank{result->Rank()}; rank != 0) {
        SayAt(x, "Must be a scalar value, but is a rank-%d array"_err_en_US,
            rank);
        ResetExpr(x);
        return std::nullopt;
      }
    }
    return result;
  }
  template <typename A> MaybeExpr Analyze(const parser::Constant<A> &x) {
    auto restorer{
        GetFoldingContext().messages().SetLocation(FindSourceLocation(x))};
    auto result{Analyze(x.thing)};
    if (result) {
      *result = Fold(std::move(*result));
      if (!IsConstantExpr(*result)) { //  C886, C887, C713
        SayAt(x, "Must be a constant value"_err_en_US);
        ResetExpr(x);
        return std::nullopt;
      } else {
        // Save folded expression for later use
        SetExpr(x, common::Clone(*result));
      }
    }
    return result;
  }
  template <typename A> MaybeExpr Analyze(const parser::Integer<A> &x) {
    auto result{Analyze(x.thing)};
    if (!EnforceTypeConstraint(
            parser::FindSourceLocation(x), result, TypeCategory::Integer)) {
      ResetExpr(x);
      return std::nullopt;
    }
    return result;
  }
  template <typename A> MaybeExpr Analyze(const parser::Logical<A> &x) {
    auto result{Analyze(x.thing)};
    if (!EnforceTypeConstraint(
            parser::FindSourceLocation(x), result, TypeCategory::Logical)) {
      ResetExpr(x);
      return std::nullopt;
    }
    return result;
  }
  template <typename A> MaybeExpr Analyze(const parser::DefaultChar<A> &x) {
    auto result{Analyze(x.thing)};
    if (!EnforceTypeConstraint(parser::FindSourceLocation(x), result,
            TypeCategory::Character, true /* default kind */)) {
      ResetExpr(x);
      return std::nullopt;
    }
    return result;
  }

  MaybeExpr Analyze(const parser::Name &);
  MaybeExpr Analyze(const parser::DataRef &dr) {
    return Analyze<parser::DataRef>(dr);
  }
  MaybeExpr Analyze(const parser::StructureComponent &);
  MaybeExpr Analyze(const parser::SignedIntLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedRealLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedComplexLiteralConstant &);
  MaybeExpr Analyze(const parser::StructureConstructor &);
  MaybeExpr Analyze(const parser::InitialDataTarget &);
  MaybeExpr Analyze(const parser::NullInit &);
  MaybeExpr Analyze(const parser::StmtFunctionStmt &);

  void Analyze(const parser::CallStmt &);
  const Assignment *Analyze(const parser::AssignmentStmt &);
  const Assignment *Analyze(const parser::PointerAssignmentStmt &);

  // Builds a typed Designator from an untyped DataRef
  MaybeExpr Designate(DataRef &&);

protected:
  int IntegerTypeSpecKind(const parser::IntegerTypeSpec &);

private:
  // Allows a whole assumed-size array to appear for the lifetime of
  // the returned value.
  common::Restorer<bool> AllowWholeAssumedSizeArray() {
    return common::ScopedSet(isWholeAssumedSizeArrayOk_, true);
  }

  // Allows an Expr to be a null pointer.
  common::Restorer<bool> AllowNullPointer() {
    return common::ScopedSet(isNullPointerOk_, true);
  }

  MaybeExpr Analyze(const parser::IntLiteralConstant &, bool negated = false);
  MaybeExpr Analyze(const parser::RealLiteralConstant &);
  MaybeExpr Analyze(const parser::ComplexPart &);
  MaybeExpr Analyze(const parser::ComplexLiteralConstant &);
  MaybeExpr Analyze(const parser::LogicalLiteralConstant &);
  MaybeExpr Analyze(const parser::CharLiteralConstant &);
  MaybeExpr Analyze(const parser::HollerithLiteralConstant &);
  MaybeExpr Analyze(const parser::BOZLiteralConstant &);
  MaybeExpr Analyze(const parser::NamedConstant &);
  MaybeExpr Analyze(const parser::DataStmtConstant &);
  MaybeExpr Analyze(const parser::Substring &);
  MaybeExpr Analyze(const parser::ArrayElement &);
  MaybeExpr Analyze(const parser::CoindexedNamedObject &);
  MaybeExpr Analyze(const parser::CharLiteralConstantSubstring &);
  MaybeExpr Analyze(const parser::SubstringInquiry &);
  MaybeExpr Analyze(const parser::ArrayConstructor &);
  MaybeExpr Analyze(const parser::FunctionReference &,
      std::optional<parser::StructureConstructor> * = nullptr);
  MaybeExpr Analyze(const parser::Expr::Parentheses &);
  MaybeExpr Analyze(const parser::Expr::UnaryPlus &);
  MaybeExpr Analyze(const parser::Expr::Negate &);
  MaybeExpr Analyze(const parser::Expr::NOT &);
  MaybeExpr Analyze(const parser::Expr::PercentLoc &);
  MaybeExpr Analyze(const parser::Expr::DefinedUnary &);
  MaybeExpr Analyze(const parser::Expr::Power &);
  MaybeExpr Analyze(const parser::Expr::Multiply &);
  MaybeExpr Analyze(const parser::Expr::Divide &);
  MaybeExpr Analyze(const parser::Expr::Add &);
  MaybeExpr Analyze(const parser::Expr::Subtract &);
  MaybeExpr Analyze(const parser::Expr::ComplexConstructor &);
  MaybeExpr Analyze(const parser::Expr::Concat &);
  MaybeExpr Analyze(const parser::Expr::LT &);
  MaybeExpr Analyze(const parser::Expr::LE &);
  MaybeExpr Analyze(const parser::Expr::EQ &);
  MaybeExpr Analyze(const parser::Expr::NE &);
  MaybeExpr Analyze(const parser::Expr::GE &);
  MaybeExpr Analyze(const parser::Expr::GT &);
  MaybeExpr Analyze(const parser::Expr::AND &);
  MaybeExpr Analyze(const parser::Expr::OR &);
  MaybeExpr Analyze(const parser::Expr::EQV &);
  MaybeExpr Analyze(const parser::Expr::NEQV &);
  MaybeExpr Analyze(const parser::Expr::DefinedBinary &);
  template <typename A> MaybeExpr Analyze(const A &x) {
    return Analyze(x.u); // default case
  }
  template <typename... As> MaybeExpr Analyze(const std::variant<As...> &u) {
    return common::visit([&](const auto &x) { return Analyze(x); }, u);
  }

  // Analysis subroutines
  int AnalyzeKindParam(
      const std::optional<parser::KindParam> &, int defaultKind);
  template <typename PARSED>
  MaybeExpr ExprOrVariable(const PARSED &, parser::CharBlock source);
  template <typename PARSED>
  MaybeExpr IntLiteralConstant(const PARSED &, bool negated = false);
  MaybeExpr AnalyzeString(std::string &&, int kind);
  std::optional<Expr<SubscriptInteger>> AsSubscript(MaybeExpr &&);
  std::optional<Expr<SubscriptInteger>> TripletPart(
      const std::optional<parser::Subscript> &);
  std::optional<Subscript> AnalyzeSectionSubscript(
      const parser::SectionSubscript &);
  std::vector<Subscript> AnalyzeSectionSubscripts(
      const std::list<parser::SectionSubscript> &);
  std::optional<Component> CreateComponent(
      DataRef &&, const Symbol &, const semantics::Scope &);
  MaybeExpr CompleteSubscripts(ArrayRef &&);
  MaybeExpr ApplySubscripts(DataRef &&, std::vector<Subscript> &&);
  void CheckConstantSubscripts(ArrayRef &);
  bool CheckRanks(const DataRef &); // Return false if error exists.
  bool CheckPolymorphic(const DataRef &); // ditto
  bool CheckDataRef(const DataRef &); // ditto
  std::optional<Expr<SubscriptInteger>> GetSubstringBound(
      const std::optional<parser::ScalarIntExpr> &);
  MaybeExpr AnalyzeDefinedOp(const parser::Name &, ActualArguments &&);
  MaybeExpr FixMisparsedSubstring(const parser::Designator &);

  struct CalleeAndArguments {
    // A non-component function reference may constitute a misparsed
    // structure constructor, in which case its derived type's Symbol
    // will appear here.
    std::variant<ProcedureDesignator, SymbolRef> u;
    ActualArguments arguments;
  };

  std::optional<CalleeAndArguments> AnalyzeProcedureComponentRef(
      const parser::ProcComponentRef &, ActualArguments &&, bool isSubroutine);
  std::optional<characteristics::Procedure> CheckCall(
      parser::CharBlock, const ProcedureDesignator &, ActualArguments &);
  bool CheckPPCIntrinsic(const ProcedureDesignator &, ActualArguments &);
  using AdjustActuals =
      std::optional<std::function<bool(const Symbol &, ActualArguments &)>>;
  bool ResolveForward(const Symbol &);
  std::pair<const Symbol *, bool /* failure due ambiguity */> ResolveGeneric(
      const Symbol &, const ActualArguments &, const AdjustActuals &,
      bool isSubroutine, bool mightBeStructureConstructor = false);
  void EmitGenericResolutionError(
      const Symbol &, bool dueToNullActuals, bool isSubroutine);
  const Symbol &AccessSpecific(
      const Symbol &originalGeneric, const Symbol &specific);
  std::optional<CalleeAndArguments> GetCalleeAndArguments(const parser::Name &,
      ActualArguments &&, bool isSubroutine = false,
      bool mightBeStructureConstructor = false);
  std::optional<CalleeAndArguments> GetCalleeAndArguments(
      const parser::ProcedureDesignator &, ActualArguments &&,
      bool isSubroutine, bool mightBeStructureConstructor = false);
  void CheckBadExplicitType(const SpecificCall &, const Symbol &);
  void CheckForBadRecursion(parser::CharBlock, const semantics::Symbol &);
  bool EnforceTypeConstraint(parser::CharBlock, const MaybeExpr &, TypeCategory,
      bool defaultKind = false);
  MaybeExpr MakeFunctionRef(
      parser::CharBlock, ProcedureDesignator &&, ActualArguments &&);
  MaybeExpr MakeFunctionRef(parser::CharBlock intrinsic, ActualArguments &&);
  template <typename T> T Fold(T &&expr) {
    return evaluate::Fold(foldingContext_, std::move(expr));
  }
  bool CheckIsValidForwardReference(const semantics::DerivedTypeSpec &);
  MaybeExpr AnalyzeComplex(MaybeExpr &&re, MaybeExpr &&im, const char *what);

  MaybeExpr IterativelyAnalyzeSubexpressions(const parser::Expr &);

  semantics::SemanticsContext &context_;
  FoldingContext &foldingContext_{context_.foldingContext()};
  std::map<parser::CharBlock, int> impliedDos_; // values are INTEGER kinds
  std::map<parser::CharBlock,
      std::pair<parser::CharBlock, evaluate::characteristics::Procedure>>
      implicitInterfaces_;
  bool isWholeAssumedSizeArrayOk_{false};
  bool isNullPointerOk_{false};
  bool useSavedTypedExprs_{true};
  bool inWhereBody_{false};
  bool inDataStmtObject_{false};
  bool inDataStmtConstant_{false};
  bool inStmtFunctionDefinition_{false};
  bool iterativelyAnalyzingSubexpressions_{false};
  friend class ArgumentAnalyzer;
};

inline bool AreConformable(int leftRank, int rightRank) {
  return leftRank == 0 || rightRank == 0 || leftRank == rightRank;
}

template <typename L, typename R>
bool AreConformable(const L &left, const R &right) {
  return AreConformable(left.Rank(), right.Rank());
}

template <typename L, typename R>
void ConformabilityCheck(
    parser::ContextualMessages &context, const L &left, const R &right) {
  if (!AreConformable(left, right)) {
    context.Say("left operand has rank %d, right operand has rank %d"_err_en_US,
        left.Rank(), right.Rank());
  }
}
} // namespace Fortran::evaluate

namespace Fortran::semantics {

// Semantic analysis of one expression, variable, selector, designator, &c.
template <typename A>
std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &context, const A &expr) {
  return evaluate::ExpressionAnalyzer{context}.Analyze(expr);
}

// Semantic analysis of an intrinsic type's KIND parameter expression.
evaluate::Expr<evaluate::SubscriptInteger> AnalyzeKindSelector(
    SemanticsContext &, common::TypeCategory,
    const std::optional<parser::KindSelector> &);

// Semantic analysis of all expressions in a parse tree, which becomes
// decorated with typed representations for top-level expressions.
class ExprChecker {
public:
  explicit ExprChecker(SemanticsContext &);

  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}
  bool Walk(const parser::Program &);

  bool Pre(const parser::Expr &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::Variable &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::Selector &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::DataStmtValue &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::AllocateObject &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::PointerObject &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::DataStmtObject &);
  void Post(const parser::DataStmtObject &);
  bool Pre(const parser::DataImpliedDo &);

  bool Pre(const parser::CallStmt &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::AssignmentStmt &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  bool Pre(const parser::PointerAssignmentStmt &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }

  // Track whether we're in a WHERE statement or construct body
  bool Pre(const parser::WhereStmt &) {
    ++whereDepth_;
    exprAnalyzer_.set_inWhereBody(InWhereBody());
    return true;
  }
  void Post(const parser::WhereStmt &) {
    --whereDepth_;
    exprAnalyzer_.set_inWhereBody(InWhereBody());
  }
  bool Pre(const parser::WhereBodyConstruct &) {
    ++whereDepth_;
    exprAnalyzer_.set_inWhereBody(InWhereBody());
    return true;
  }
  void Post(const parser::WhereBodyConstruct &) {
    --whereDepth_;
    exprAnalyzer_.set_inWhereBody(InWhereBody());
  }

  bool Pre(const parser::ComponentDefStmt &) {
    // Already analyzed in name resolution and PDT instantiation;
    // do not attempt to re-analyze now without type parameters.
    return false;
  }

  template <typename A> bool Pre(const parser::Scalar<A> &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  template <typename A> bool Pre(const parser::Constant<A> &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  template <typename A> bool Pre(const parser::Integer<A> &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  template <typename A> bool Pre(const parser::Logical<A> &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }
  template <typename A> bool Pre(const parser::DefaultChar<A> &x) {
    exprAnalyzer_.Analyze(x);
    return false;
  }

private:
  bool InWhereBody() const { return whereDepth_ > 0; }

  SemanticsContext &context_;
  evaluate::ExpressionAnalyzer exprAnalyzer_{context_};
  int whereDepth_{0}; // nesting of WHERE statements & constructs
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_EXPRESSION_H_
