//===-- lib/Semantics/assignment.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "assignment.h"
#include "pointer-assignment.h"
#include "flang/Common/idioms.h"
#include "flang/Common/restorer.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <optional>
#include <set>
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::semantics {

class AssignmentContext {
public:
  explicit AssignmentContext(SemanticsContext &context) : context_{context} {}
  AssignmentContext(AssignmentContext &&) = default;
  AssignmentContext(const AssignmentContext &) = delete;
  bool operator==(const AssignmentContext &x) const { return this == &x; }

  template <typename A> void PushWhereContext(const A &);
  void PopWhereContext();
  void Analyze(const parser::AssignmentStmt &);
  void Analyze(const parser::PointerAssignmentStmt &);
  void Analyze(const parser::ConcurrentControl &);

private:
  bool CheckForPureContext(const SomeExpr &lhs, const SomeExpr &rhs,
      parser::CharBlock rhsSource, bool isPointerAssignment);
  void CheckShape(parser::CharBlock, const SomeExpr *);
  template <typename... A>
  parser::Message *Say(parser::CharBlock at, A &&...args) {
    return &context_.Say(at, std::forward<A>(args)...);
  }
  evaluate::FoldingContext &foldingContext() {
    return context_.foldingContext();
  }

  SemanticsContext &context_;
  int whereDepth_{0}; // number of WHEREs currently nested in
  // shape of masks in LHS of assignments in current WHERE:
  std::vector<std::optional<std::int64_t>> whereExtents_;
};

void AssignmentContext::Analyze(const parser::AssignmentStmt &stmt) {
  if (const evaluate::Assignment * assignment{GetAssignment(stmt)}) {
    const SomeExpr &lhs{assignment->lhs};
    const SomeExpr &rhs{assignment->rhs};
    auto lhsLoc{std::get<parser::Variable>(stmt.t).GetSource()};
    auto rhsLoc{std::get<parser::Expr>(stmt.t).source};
    if (CheckForPureContext(lhs, rhs, rhsLoc, false)) {
      const Scope &scope{context_.FindScope(lhsLoc)};
      if (auto whyNot{WhyNotModifiable(lhsLoc, lhs, scope, true)}) {
        if (auto *msg{Say(lhsLoc,
                "Left-hand side of assignment is not modifiable"_err_en_US)}) { // C1158
          msg->Attach(*whyNot);
        }
      }
    }
    if (whereDepth_ > 0) {
      CheckShape(lhsLoc, &lhs);
    }
  }
}

void AssignmentContext::Analyze(const parser::PointerAssignmentStmt &stmt) {
  CHECK(whereDepth_ == 0);
  if (const evaluate::Assignment * assignment{GetAssignment(stmt)}) {
    const SomeExpr &lhs{assignment->lhs};
    const SomeExpr &rhs{assignment->rhs};
    CheckForPureContext(lhs, rhs, std::get<parser::Expr>(stmt.t).source, true);
    auto restorer{
        foldingContext().messages().SetLocation(context_.location().value())};
    CheckPointerAssignment(foldingContext(), *assignment);
  }
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

// Checks C1594(1,2); false if check fails
bool CheckDefinabilityInPureScope(parser::ContextualMessages &messages,
    const Symbol &lhs, const Scope &context, const Scope &pure) {
  if (pure.symbol()) {
    if (const char *why{WhyBaseObjectIsSuspicious(lhs, context)}) {
      evaluate::SayWithDeclaration(messages, lhs,
          "Pure subprogram '%s' may not define '%s' because it is %s"_err_en_US,
          pure.symbol()->name(), lhs.name(), why);
      return false;
    }
  }
  return true;
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

// Checks C1594(5,6); false if check fails
bool CheckCopyabilityInPureScope(parser::ContextualMessages &messages,
    const SomeExpr &expr, const Scope &scope) {
  if (const Symbol * base{GetFirstSymbol(expr)}) {
    if (const char *why{WhyBaseObjectIsSuspicious(*base, scope)}) {
      if (auto pointer{GetPointerComponentDesignatorName(expr)}) {
        evaluate::SayWithDeclaration(messages, *base,
            "A pure subprogram may not copy the value of '%s' because it is %s"
            " and has the POINTER component '%s'"_err_en_US,
            base->name(), why, *pointer);
        return false;
      }
    }
  }
  return true;
}

bool AssignmentContext::CheckForPureContext(const SomeExpr &lhs,
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
        auto dataRef{ExtractDataRef(assoc->expr(), true)};
        // ASSOCIATE(a=>x) -- check x, not a, for "a=..."
        base = dataRef ? &dataRef->GetFirstSymbol() : nullptr;
      }
      if (base &&
          !CheckDefinabilityInPureScope(messages, *base, scope, *pure)) {
        return false;
      }
    }
    if (isPointerAssignment) {
      if (const Symbol * base{GetFirstSymbol(rhs)}) {
        if (const char *why{
                WhyBaseObjectIsSuspicious(*base, scope)}) { // C1594(3)
          evaluate::SayWithDeclaration(messages, *base,
              "A pure subprogram may not use '%s' as the target of pointer assignment because it is %s"_err_en_US,
              base->name(), why);
          return false;
        }
      }
    } else if (auto type{evaluate::DynamicType::From(lhs)}) {
      // C1596 checks for polymorphic deallocation in a pure subprogram
      // due to automatic reallocation on assignment
      if (type->IsPolymorphic()) {
        context_.Say(
            "Deallocation of polymorphic object is not permitted in a pure subprogram"_err_en_US);
        return false;
      }
      if (const DerivedTypeSpec * derived{GetDerivedTypeSpec(type)}) {
        if (auto bad{FindPolymorphicAllocatableNonCoarrayUltimateComponent(
                *derived)}) {
          evaluate::SayWithDeclaration(messages, *bad,
              "Deallocation of polymorphic non-coarray component '%s' is not permitted in a pure subprogram"_err_en_US,
              bad.BuildResultDesignatorName());
          return false;
        } else {
          return CheckCopyabilityInPureScope(messages, rhs, scope);
        }
      }
    }
  }
  return true;
}

// 10.2.3.1(2) The masks and LHS of assignments must all have the same shape
void AssignmentContext::CheckShape(parser::CharBlock at, const SomeExpr *expr) {
  if (auto shape{evaluate::GetShape(foldingContext(), expr)}) {
    std::size_t size{shape->size()};
    if (whereDepth_ == 0) {
      whereExtents_.resize(size);
    } else if (whereExtents_.size() != size) {
      Say(at,
          "Must have rank %zd to match prior mask or assignment of"
          " WHERE construct"_err_en_US,
          whereExtents_.size());
      return;
    }
    for (std::size_t i{0}; i < size; ++i) {
      if (std::optional<std::int64_t> extent{evaluate::ToInt64((*shape)[i])}) {
        if (!whereExtents_[i]) {
          whereExtents_[i] = *extent;
        } else if (*whereExtents_[i] != *extent) {
          Say(at,
              "Dimension %d must have extent %jd to match prior mask or"
              " assignment of WHERE construct"_err_en_US,
              i + 1, *whereExtents_[i]);
        }
      }
    }
  }
}

template <typename A> void AssignmentContext::PushWhereContext(const A &x) {
  const auto &expr{std::get<parser::LogicalExpr>(x.t)};
  CheckShape(expr.thing.value().source, GetExpr(expr));
  ++whereDepth_;
}

void AssignmentContext::PopWhereContext() {
  --whereDepth_;
  if (whereDepth_ == 0) {
    whereExtents_.clear();
  }
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
  context_.value().PushWhereContext(x);
}
void AssignmentChecker::Leave(const parser::WhereStmt &) {
  context_.value().PopWhereContext();
}
void AssignmentChecker::Enter(const parser::WhereConstructStmt &x) {
  context_.value().PushWhereContext(x);
}
void AssignmentChecker::Leave(const parser::EndWhereStmt &) {
  context_.value().PopWhereContext();
}
void AssignmentChecker::Enter(const parser::MaskedElsewhereStmt &x) {
  context_.value().PushWhereContext(x);
}
void AssignmentChecker::Leave(const parser::MaskedElsewhereStmt &) {
  context_.value().PopWhereContext();
}

} // namespace Fortran::semantics
template class Fortran::common::Indirection<
    Fortran::semantics::AssignmentContext>;
