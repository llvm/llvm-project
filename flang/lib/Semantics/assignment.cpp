//===-- lib/Semantics/assignment.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "assignment.h"
#include "definable.h"
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
  bool CheckForPureContext(const SomeExpr &rhs, parser::CharBlock rhsSource,
      bool isPointerAssignment, bool isDefinedAssignment);
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
    const Scope &scope{context_.FindScope(lhsLoc)};
    if (auto whyNot{WhyNotDefinable(lhsLoc, scope,
            DefinabilityFlags{DefinabilityFlag::VectorSubscriptIsOk}, lhs)}) {
      if (auto *msg{Say(lhsLoc,
              "Left-hand side of assignment is not definable"_err_en_US)}) {
        msg->Attach(std::move(*whyNot));
      }
    }
    auto rhsLoc{std::get<parser::Expr>(stmt.t).source};
    CheckForPureContext(rhs, rhsLoc, false /*not a pointer assignment*/,
        std::holds_alternative<evaluate::ProcedureRef>(assignment->u));
    if (whereDepth_ > 0) {
      CheckShape(lhsLoc, &lhs);
    }
  }
}

void AssignmentContext::Analyze(const parser::PointerAssignmentStmt &stmt) {
  CHECK(whereDepth_ == 0);
  if (const evaluate::Assignment * assignment{GetAssignment(stmt)}) {
    const SomeExpr &rhs{assignment->rhs};
    CheckForPureContext(rhs, std::get<parser::Expr>(stmt.t).source,
        true /*this is a pointer assignment*/,
        false /*not a defined assignment*/);
    parser::CharBlock at{context_.location().value()};
    auto restorer{foldingContext().messages().SetLocation(at)};
    const Scope &scope{context_.FindScope(at)};
    CheckPointerAssignment(foldingContext(), *assignment, scope);
  }
}

static std::optional<std::string> GetPointerComponentDesignatorName(
    const SomeExpr &expr) {
  if (const auto *derived{
          evaluate::GetDerivedTypeSpec(evaluate::DynamicType::From(expr))}) {
    PotentialAndPointerComponentIterator potentials{*derived};
    if (auto pointer{
            std::find_if(potentials.begin(), potentials.end(), IsPointer)}) {
      return pointer.BuildResultDesignatorName();
    }
  }
  return std::nullopt;
}

// Checks C1594(5,6); false if check fails
bool CheckCopyabilityInPureScope(parser::ContextualMessages &messages,
    const SomeExpr &expr, const Scope &scope) {
  if (const Symbol * base{GetFirstSymbol(expr)}) {
    if (const char *why{
            WhyBaseObjectIsSuspicious(base->GetUltimate(), scope)}) {
      if (auto pointer{GetPointerComponentDesignatorName(expr)}) {
        evaluate::SayWithDeclaration(messages, *base,
            "A pure subprogram may not copy the value of '%s' because it is %s"
            " and has the POINTER potential subobject component '%s'"_err_en_US,
            base->name(), why, *pointer);
        return false;
      }
    }
  }
  return true;
}

bool AssignmentContext::CheckForPureContext(const SomeExpr &rhs,
    parser::CharBlock rhsSource, bool isPointerAssignment,
    bool isDefinedAssignment) {
  const Scope &scope{context_.FindScope(rhsSource)};
  if (!FindPureProcedureContaining(scope)) {
    return true;
  }
  parser::ContextualMessages messages{
      context_.location().value(), &context_.messages()};
  if (isPointerAssignment) {
    if (const Symbol * base{GetFirstSymbol(rhs)}) {
      if (const char *why{WhyBaseObjectIsSuspicious(
              base->GetUltimate(), scope)}) { // C1594(3)
        evaluate::SayWithDeclaration(messages, *base,
            "A pure subprogram may not use '%s' as the target of pointer assignment because it is %s"_err_en_US,
            base->name(), why);
        return false;
      }
    }
  } else if (!isDefinedAssignment) {
    return CheckCopyabilityInPureScope(messages, rhs, scope);
  }
  return true;
}

// 10.2.3.1(2) The masks and LHS of assignments must be arrays of the same shape
void AssignmentContext::CheckShape(parser::CharBlock at, const SomeExpr *expr) {
  if (auto shape{evaluate::GetShape(foldingContext(), expr)}) {
    std::size_t size{shape->size()};
    if (size == 0) {
      Say(at, "The mask or variable must not be scalar"_err_en_US);
    }
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
  CheckShape(expr.thing.value().source, GetExpr(context_, expr));
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
