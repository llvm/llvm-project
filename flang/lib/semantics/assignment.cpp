//===-- lib/semantics/assignment.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "assignment.h"
#include "pointer-assignment.h"
#include "flang/common/idioms.h"
#include "flang/common/restorer.h"
#include "flang/evaluate/characteristics.h"
#include "flang/evaluate/expression.h"
#include "flang/evaluate/fold.h"
#include "flang/evaluate/tools.h"
#include "flang/parser/message.h"
#include "flang/parser/parse-tree-visitor.h"
#include "flang/parser/parse-tree.h"
#include "flang/semantics/expression.h"
#include "flang/semantics/symbol.h"
#include "flang/semantics/tools.h"
#include <optional>
#include <set>
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::semantics {

using ControlExpr = evaluate::Expr<evaluate::SubscriptInteger>;
using MaskExpr = evaluate::Expr<evaluate::LogicalResult>;

// The context tracks some number of active FORALL statements/constructs
// and some number of active WHERE statements/constructs.  WHERE can nest
// in FORALL but not vice versa.  Pointer assignments are allowed in
// FORALL but not in WHERE.  These constraints are manifest in the grammar
// and don't need to be rechecked here, since errors cannot appear in the
// parse tree.
struct Control {
  Symbol *name;
  ControlExpr lower, upper, step;
};

struct ForallContext {
  explicit ForallContext(const ForallContext *that) : outer{that} {}

  const ForallContext *outer{nullptr};
  std::optional<parser::CharBlock> constructName;
  std::vector<Control> control;
  std::optional<MaskExpr> maskExpr;
  std::set<parser::CharBlock> activeNames;
};

struct WhereContext {
  WhereContext(MaskExpr &&x, const WhereContext *o, const ForallContext *f)
    : outer{o}, forall{f}, thisMaskExpr{std::move(x)} {}
  const WhereContext *outer{nullptr};
  const ForallContext *forall{nullptr};  // innermost enclosing FORALL
  std::optional<parser::CharBlock> constructName;
  MaskExpr thisMaskExpr;  // independent of outer WHERE, if any
  MaskExpr cumulativeMaskExpr{thisMaskExpr};
};

class AssignmentContext {
public:
  explicit AssignmentContext(SemanticsContext &c) : context_{c} {}
  AssignmentContext(const AssignmentContext &c, WhereContext &w)
    : context_{c.context_}, where_{&w} {}
  AssignmentContext(const AssignmentContext &c, ForallContext &f)
    : context_{c.context_}, forall_{&f} {}

  bool operator==(const AssignmentContext &x) const { return this == &x; }

  void Analyze(const parser::AssignmentStmt &);
  void Analyze(const parser::PointerAssignmentStmt &);
  void Analyze(const parser::WhereStmt &);
  void Analyze(const parser::WhereConstruct &);
  void Analyze(const parser::ForallConstruct &);

  template<typename A> void Analyze(const parser::UnlabeledStatement<A> &stmt) {
    context_.set_location(stmt.source);
    Analyze(stmt.statement);
  }
  template<typename A> void Analyze(const common::Indirection<A> &x) {
    Analyze(x.value());
  }
  template<typename A> std::enable_if_t<UnionTrait<A>> Analyze(const A &x) {
    std::visit([&](const auto &y) { Analyze(y); }, x.u);
  }
  template<typename A> void Analyze(const std::list<A> &list) {
    for (const auto &elem : list) {
      Analyze(elem);
    }
  }
  template<typename A> void Analyze(const std::optional<A> &x) {
    if (x) {
      Analyze(*x);
    }
  }

private:
  void Analyze(const parser::WhereConstruct::MaskedElsewhere &);
  void Analyze(const parser::MaskedElsewhereStmt &);
  void Analyze(const parser::WhereConstruct::Elsewhere &);

  void CheckForPureContext(const SomeExpr &lhs, const SomeExpr &rhs,
      parser::CharBlock rhsSource, bool isPointerAssignment);

  MaskExpr GetMask(const parser::LogicalExpr &, bool defaultValue = true);

  template<typename... A>
  parser::Message *Say(parser::CharBlock at, A &&... args) {
    return &context_.Say(at, std::forward<A>(args)...);
  }

  SemanticsContext &context_;
  WhereContext *where_{nullptr};
  ForallContext *forall_{nullptr};
};

void AssignmentContext::Analyze(const parser::AssignmentStmt &stmt) {
  // Assignment statement analysis is in expression.cpp where user-defined
  // assignments can be recognized and replaced.
  if (const evaluate::Assignment * assignment{GetAssignment(stmt)}) {
    if (forall_) {
      // TODO: Warn if some name in forall_->activeNames or its outer
      // contexts does not appear on LHS
    }
    CheckForPureContext(assignment->lhs, assignment->rhs,
        std::get<parser::Expr>(stmt.t).source, false /* not => */);
  }
  // TODO: Fortran 2003 ALLOCATABLE assignment semantics (automatic
  // (re)allocation of LHS array when unallocated or nonconformable)
}

void AssignmentContext::Analyze(const parser::PointerAssignmentStmt &stmt) {
  CHECK(!where_);
  const evaluate::Assignment *assignment{GetAssignment(stmt)};
  if (!assignment) {
    return;
  }
  const SomeExpr &lhs{assignment->lhs};
  const SomeExpr &rhs{assignment->rhs};
  if (forall_) {
    // TODO: Warn if some name in forall_->activeNames or its outer
    // contexts does not appear on LHS
  }
  CheckForPureContext(lhs, rhs, std::get<parser::Expr>(stmt.t).source,
      true /* isPointerAssignment */);
  auto restorer{context_.foldingContext().messages().SetLocation(
      context_.location().value())};
  CheckPointerAssignment(context_.foldingContext(), *assignment);
}

void AssignmentContext::Analyze(const parser::WhereStmt &stmt) {
  WhereContext where{
      GetMask(std::get<parser::LogicalExpr>(stmt.t)), where_, forall_};
  AssignmentContext nested{*this, where};
  nested.Analyze(std::get<parser::AssignmentStmt>(stmt.t));
}

// N.B. Construct name matching is checked during label resolution.
void AssignmentContext::Analyze(const parser::WhereConstruct &construct) {
  const auto &whereStmt{
      std::get<parser::Statement<parser::WhereConstructStmt>>(construct.t)};
  WhereContext where{
      GetMask(std::get<parser::LogicalExpr>(whereStmt.statement.t)), where_,
      forall_};
  if (const auto &name{
          std::get<std::optional<parser::Name>>(whereStmt.statement.t)}) {
    where.constructName = name->source;
  }
  AssignmentContext nested{*this, where};
  nested.Analyze(std::get<std::list<parser::WhereBodyConstruct>>(construct.t));
  nested.Analyze(std::get<std::list<parser::WhereConstruct::MaskedElsewhere>>(
      construct.t));
  nested.Analyze(
      std::get<std::optional<parser::WhereConstruct::Elsewhere>>(construct.t));
}

void AssignmentContext::Analyze(
    const parser::WhereConstruct::MaskedElsewhere &elsewhere) {
  CHECK(where_);
  Analyze(
      std::get<parser::Statement<parser::MaskedElsewhereStmt>>(elsewhere.t));
  Analyze(std::get<std::list<parser::WhereBodyConstruct>>(elsewhere.t));
}

void AssignmentContext::Analyze(const parser::MaskedElsewhereStmt &elsewhere) {
  MaskExpr mask{GetMask(std::get<parser::LogicalExpr>(elsewhere.t))};
  MaskExpr copyCumulative{where_->cumulativeMaskExpr};
  MaskExpr notOldMask{evaluate::LogicalNegation(std::move(copyCumulative))};
  if (!evaluate::AreConformable(notOldMask, mask)) {
    context_.Say("mask of ELSEWHERE statement is not conformable with "
                 "the prior mask(s) in its WHERE construct"_err_en_US);
  }
  MaskExpr copyMask{mask};
  where_->cumulativeMaskExpr =
      evaluate::BinaryLogicalOperation(evaluate::LogicalOperator::Or,
          std::move(where_->cumulativeMaskExpr), std::move(copyMask));
  where_->thisMaskExpr = evaluate::BinaryLogicalOperation(
      evaluate::LogicalOperator::And, std::move(notOldMask), std::move(mask));
  if (where_->outer &&
      !evaluate::AreConformable(
          where_->outer->thisMaskExpr, where_->thisMaskExpr)) {
    context_.Say("effective mask of ELSEWHERE statement is not conformable "
                 "with the mask of the surrounding WHERE construct"_err_en_US);
  }
}

void AssignmentContext::Analyze(
    const parser::WhereConstruct::Elsewhere &elsewhere) {
  MaskExpr copyCumulative{DEREF(where_).cumulativeMaskExpr};
  where_->thisMaskExpr = evaluate::LogicalNegation(std::move(copyCumulative));
  Analyze(std::get<std::list<parser::WhereBodyConstruct>>(elsewhere.t));
}

// C1594 checks
static bool IsPointerDummyOfPureFunction(const Symbol &x) {
  return IsPointerDummy(x) && FindPureProcedureContaining(x.owner()) &&
      x.owner().symbol() && IsFunction(*x.owner().symbol());
}

static const char *WhyBaseObjectIsSuspicious(
    const Symbol &x, const Scope &scope) {
  // See C1594, first paragraph.  These conditions enable checks on both
  // left-hand and right-hand sides in various circumstances.
  if (IsHostAssociated(x, scope)) {
    return "host-associated";
  } else if (IsUseAssociated(x, scope)) {
    return "USE-associated";
  } else if (IsPointerDummyOfPureFunction(x)) {
    return "a POINTER dummy argument of a pure function";
  } else if (IsIntentIn(x)) {
    return "an INTENT(IN) dummy argument";
  } else if (FindCommonBlockContaining(x)) {
    return "in a COMMON block";
  } else {
    return nullptr;
  }
}

// Checks C1594(1,2)
void CheckDefinabilityInPureScope(parser::ContextualMessages &messages,
    const Symbol &lhs, const Scope &context, const Scope &pure) {
  if (pure.symbol()) {
    if (const char *why{WhyBaseObjectIsSuspicious(lhs, context)}) {
      evaluate::SayWithDeclaration(messages, lhs,
          "Pure subprogram '%s' may not define '%s' because it is %s"_err_en_US,
          pure.symbol()->name(), lhs.name(), why);
    }
  }
}

static std::optional<std::string> GetPointerComponentDesignatorName(
    const SomeExpr &expr) {
  if (const auto *derived{
          evaluate::GetDerivedTypeSpec(evaluate::DynamicType::From(expr))}) {
    UltimateComponentIterator ultimates{*derived};
    if (auto pointer{
            std::find_if(ultimates.begin(), ultimates.end(), IsPointer)}) {
      return pointer.BuildResultDesignatorName();
    }
  }
  return std::nullopt;
}

// Checks C1594(5,6)
void CheckCopyabilityInPureScope(parser::ContextualMessages &messages,
    const SomeExpr &expr, const Scope &scope) {
  if (const Symbol * base{GetFirstSymbol(expr)}) {
    if (const char *why{WhyBaseObjectIsSuspicious(*base, scope)}) {
      if (auto pointer{GetPointerComponentDesignatorName(expr)}) {
        evaluate::SayWithDeclaration(messages, *base,
            "A pure subprogram may not copy the value of '%s' because it is %s and has the POINTER component '%s'"_err_en_US,
            base->name(), why, *pointer);
      }
    }
  }
}

void AssignmentContext::CheckForPureContext(const SomeExpr &lhs,
    const SomeExpr &rhs, parser::CharBlock source, bool isPointerAssignment) {
  const Scope &scope{context_.FindScope(source)};
  if (const Scope * pure{FindPureProcedureContaining(scope)}) {
    parser::ContextualMessages messages{
        context_.location().value(), &context_.messages()};
    if (evaluate::ExtractCoarrayRef(lhs)) {
      messages.Say(
          "A pure subprogram may not define a coindexed object"_err_en_US);
    } else if (const Symbol * base{GetFirstSymbol(lhs)}) {
      if (const auto *assoc{base->detailsIf<AssocEntityDetails>()}) {
        if (auto dataRef{ExtractDataRef(assoc->expr())}) {
          // ASSOCIATE(a=>x) -- check x, not a, for "a=..."
          CheckDefinabilityInPureScope(
              messages, dataRef->GetFirstSymbol(), scope, *pure);
        }
      } else {
        CheckDefinabilityInPureScope(messages, *base, scope, *pure);
      }
    }
    if (isPointerAssignment) {
      if (const Symbol * base{GetFirstSymbol(rhs)}) {
        if (const char *why{
                WhyBaseObjectIsSuspicious(*base, scope)}) {  // C1594(3)
          evaluate::SayWithDeclaration(messages, *base,
              "A pure subprogram may not use '%s' as the target of pointer assignment because it is %s"_err_en_US,
              base->name(), why);
        }
      }
    } else {
      if (auto type{evaluate::DynamicType::From(lhs)}) {
        // C1596 checks for polymorphic deallocation in a pure subprogram
        // due to automatic reallocation on assignment
        if (type->IsPolymorphic()) {
          context_.Say(
              "Deallocation of polymorphic object is not permitted in a pure subprogram"_err_en_US);
        }
        if (const DerivedTypeSpec * derived{GetDerivedTypeSpec(type)}) {
          if (auto bad{FindPolymorphicAllocatableNonCoarrayUltimateComponent(
                  *derived)}) {
            evaluate::SayWithDeclaration(messages, *bad,
                "Deallocation of polymorphic non-coarray component '%s' is not permitted in a pure subprogram"_err_en_US,
                bad.BuildResultDesignatorName());
          } else {
            CheckCopyabilityInPureScope(messages, rhs, scope);
          }
        }
      }
    }
  }
}

MaskExpr AssignmentContext::GetMask(
    const parser::LogicalExpr &logicalExpr, bool defaultValue) {
  MaskExpr mask{defaultValue};
  if (const SomeExpr * expr{GetExpr(logicalExpr)}) {
    auto *logical{std::get_if<evaluate::Expr<evaluate::SomeLogical>>(&expr->u)};
    mask = evaluate::ConvertTo(mask, common::Clone(DEREF(logical)));
  }
  return mask;
}

AssignmentChecker::~AssignmentChecker() {}

AssignmentChecker::AssignmentChecker(SemanticsContext &context)
  : context_{new AssignmentContext{context}} {}
void AssignmentChecker::Enter(const parser::AssignmentStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::PointerAssignmentStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::WhereStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::WhereConstruct &x) {
  context_.value().Analyze(x);
}

}
template class Fortran::common::Indirection<
    Fortran::semantics::AssignmentContext>;
