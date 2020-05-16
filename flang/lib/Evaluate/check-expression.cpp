//===-- lib/Evaluate/check-expression.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <set>
#include <string>

namespace Fortran::evaluate {

// Constant expression predicate IsConstantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.
class IsConstantExprHelper : public AllTraverse<IsConstantExprHelper, true> {
public:
  using Base = AllTraverse<IsConstantExprHelper, true>;
  IsConstantExprHelper() : Base{*this} {}
  using Base::operator();

  template <int KIND> bool operator()(const TypeParamInquiry<KIND> &inq) const {
    return IsKindTypeParameter(inq.parameter());
  }
  bool operator()(const semantics::Symbol &symbol) const {
    return IsNamedConstant(symbol) || IsImpliedDoIndex(symbol);
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const semantics::ParamValue &param) const {
    return param.isExplicit() && (*this)(param.GetExplicit());
  }
  template <typename T> bool operator()(const FunctionRef<T> &call) const {
    if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
      return intrinsic->name == "kind";
      // TODO: other inquiry intrinsics
    } else {
      return false;
    }
  }
  bool operator()(const StructureConstructor &constructor) const {
    for (const auto &[symRef, expr] : constructor) {
      if (IsAllocatable(*symRef)) {
        return IsNullPointer(expr.value());
      } else if (IsPointer(*symRef)) {
        return IsNullPointer(expr.value()) || IsInitialDataTarget(expr.value());
      } else if (!(*this)(expr.value())) {
        return false;
      }
    }
    return true;
  }

  // Forbid integer division by zero in constants.
  template <int KIND>
  bool operator()(
      const Divide<Type<TypeCategory::Integer, KIND>> &division) const {
    using T = Type<TypeCategory::Integer, KIND>;
    if (const auto divisor{GetScalarConstantValue<T>(division.right())}) {
      return !divisor->IsZero();
    } else {
      return false;
    }
  }
};

template <typename A> bool IsConstantExpr(const A &x) {
  return IsConstantExprHelper{}(x);
}
template bool IsConstantExpr(const Expr<SomeType> &);
template bool IsConstantExpr(const Expr<SomeInteger> &);
template bool IsConstantExpr(const Expr<SubscriptInteger> &);

// Object pointer initialization checking predicate IsInitialDataTarget().
// This code determines whether an expression is allowable as the static
// data address used to initialize a pointer with "=> x".  See C765.
// If messages are requested, errors may be generated without returning
// a false result.
class IsInitialDataTargetHelper
    : public AllTraverse<IsInitialDataTargetHelper, true> {
public:
  using Base = AllTraverse<IsInitialDataTargetHelper, true>;
  using Base::operator();
  explicit IsInitialDataTargetHelper(parser::ContextualMessages *m)
      : Base{*this}, messages_{m} {}

  bool operator()(const BOZLiteralConstant &) const { return false; }
  bool operator()(const NullPointer &) const { return true; }
  template <typename T> bool operator()(const Constant<T> &) const {
    return false;
  }
  bool operator()(const semantics::Symbol &symbol) const {
    const Symbol &ultimate{symbol.GetUltimate()};
    if (IsAllocatable(ultimate)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an ALLOCATABLE '%s'"_err_en_US,
            ultimate.name());
      } else {
        return false;
      }
    } else if (ultimate.Corank() > 0) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to a coarray '%s'"_err_en_US,
            ultimate.name());
      } else {
        return false;
      }
    } else if (!ultimate.attrs().test(semantics::Attr::TARGET)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an object '%s' that lacks the TARGET attribute"_err_en_US,
            ultimate.name());
      } else {
        return false;
      }
    } else if (!IsSaved(ultimate)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an object '%s' that lacks the SAVE attribute"_err_en_US,
            ultimate.name());
      } else {
        return false;
      }
    }
    return true;
  }
  bool operator()(const StaticDataObject &) const { return false; }
  template <int KIND> bool operator()(const TypeParamInquiry<KIND> &) const {
    return false;
  }
  bool operator()(const Triplet &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        IsConstantExpr(x.stride());
  }
  bool operator()(const Subscript &x) const {
    return std::visit(common::visitors{
                          [&](const Triplet &t) { return (*this)(t); },
                          [&](const auto &y) {
                            return y.value().Rank() == 0 &&
                                IsConstantExpr(y.value());
                          },
                      },
        x.u);
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const Substring &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        (*this)(x.parent());
  }
  bool operator()(const DescriptorInquiry &) const { return false; }
  template <typename T> bool operator()(const ArrayConstructor<T> &) const {
    return false;
  }
  bool operator()(const StructureConstructor &) const { return false; }
  template <typename T> bool operator()(const FunctionRef<T> &) {
    return false;
  }
  template <typename D, typename R, typename... O>
  bool operator()(const Operation<D, R, O...> &) const {
    return false;
  }
  template <typename T> bool operator()(const Parentheses<T> &x) const {
    return (*this)(x.left());
  }
  bool operator()(const Relational<SomeType> &) const { return false; }

private:
  parser::ContextualMessages *messages_;
};

bool IsInitialDataTarget(
    const Expr<SomeType> &x, parser::ContextualMessages *messages) {
  return IsInitialDataTargetHelper{messages}(x);
}

// Specification expression validation (10.1.11(2), C1010)
class CheckSpecificationExprHelper
    : public AnyTraverse<CheckSpecificationExprHelper,
          std::optional<std::string>> {
public:
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<CheckSpecificationExprHelper, Result>;
  explicit CheckSpecificationExprHelper(
      const semantics::Scope &s, const IntrinsicProcTable &table)
      : Base{*this}, scope_{s}, table_{table} {}
  using Base::operator();

  Result operator()(const ProcedureDesignator &) const {
    return "dummy procedure argument";
  }
  Result operator()(const CoarrayRef &) const { return "coindexed reference"; }

  Result operator()(const semantics::Symbol &symbol) const {
    if (semantics::IsNamedConstant(symbol)) {
      return std::nullopt;
    } else if (scope_.IsDerivedType() && IsVariableName(symbol)) { // C750, C754
      return "derived type component or type parameter value not allowed to "
             "reference variable '"s +
          symbol.name().ToString() + "'";
    } else if (symbol.IsDummy()) {
      if (symbol.attrs().test(semantics::Attr::OPTIONAL)) {
        return "reference to OPTIONAL dummy argument '"s +
            symbol.name().ToString() + "'";
      } else if (symbol.attrs().test(semantics::Attr::INTENT_OUT)) {
        return "reference to INTENT(OUT) dummy argument '"s +
            symbol.name().ToString() + "'";
      } else if (symbol.has<semantics::ObjectEntityDetails>()) {
        return std::nullopt;
      } else {
        return "dummy procedure argument";
      }
    } else if (symbol.has<semantics::UseDetails>() ||
        symbol.has<semantics::HostAssocDetails>() ||
        symbol.owner().kind() == semantics::Scope::Kind::Module) {
      return std::nullopt;
    } else if (const auto *object{
                   symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      // TODO: what about EQUIVALENCE with data in COMMON?
      // TODO: does this work for blank COMMON?
      if (object->commonBlock()) {
        return std::nullopt;
      }
    }
    for (const semantics::Scope *s{&scope_}; !s->IsGlobal();) {
      s = &s->parent();
      if (s == &symbol.owner()) {
        return std::nullopt;
      }
    }
    return "reference to local entity '"s + symbol.name().ToString() + "'";
  }

  Result operator()(const Component &x) const {
    // Don't look at the component symbol.
    return (*this)(x.base());
  }
  Result operator()(const DescriptorInquiry &) const {
    // Subtle: Uses of SIZE(), LBOUND(), &c. that are valid in specification
    // expressions will have been converted to expressions over descriptor
    // inquiries by Fold().
    return std::nullopt;
  }

  template <int KIND>
  Result operator()(const TypeParamInquiry<KIND> &inq) const {
    if (scope_.IsDerivedType() && !IsConstantExpr(inq) &&
        inq.parameter().owner() != scope_) { // C750, C754
      return "non-constant reference to a type parameter inquiry not "
             "allowed for derived type components or type parameter values";
    }
    return std::nullopt;
  }

  template <typename T> Result operator()(const FunctionRef<T> &x) const {
    if (const auto *symbol{x.proc().GetSymbol()}) {
      if (!semantics::IsPureProcedure(*symbol)) {
        return "reference to impure function '"s + symbol->name().ToString() +
            "'";
      }
      if (semantics::IsStmtFunction(*symbol)) {
        return "reference to statement function '"s +
            symbol->name().ToString() + "'";
      }
      if (scope_.IsDerivedType()) { // C750, C754
        return "reference to function '"s + symbol->name().ToString() +
            "' not allowed for derived type components or type parameter"
            " values";
      }
      // TODO: other checks for standard module procedures
    } else {
      const SpecificIntrinsic &intrin{DEREF(x.proc().GetSpecificIntrinsic())};
      if (scope_.IsDerivedType()) { // C750, C754
        if ((table_.IsIntrinsic(intrin.name) &&
                badIntrinsicsForComponents_.find(intrin.name) !=
                    badIntrinsicsForComponents_.end()) ||
            IsProhibitedFunction(intrin.name)) {
          return "reference to intrinsic '"s + intrin.name +
              "' not allowed for derived type components or type parameter"
              " values";
        }
        if (table_.GetIntrinsicClass(intrin.name) ==
                IntrinsicClass::inquiryFunction &&
            !IsConstantExpr(x)) {
          return "non-constant reference to inquiry intrinsic '"s +
              intrin.name +
              "' not allowed for derived type components or type"
              " parameter values";
        }
      } else if (intrin.name == "present") {
        return std::nullopt; // no need to check argument(s)
      }
      if (IsConstantExpr(x)) {
        // inquiry functions may not need to check argument(s)
        return std::nullopt;
      }
    }
    return (*this)(x.arguments());
  }

private:
  const semantics::Scope &scope_;
  const IntrinsicProcTable &table_;
  const std::set<std::string> badIntrinsicsForComponents_{
      "allocated", "associated", "extends_type_of", "present", "same_type_as"};
  static bool IsProhibitedFunction(std::string name) { return false; }
};

template <typename A>
void CheckSpecificationExpr(const A &x, parser::ContextualMessages &messages,
    const semantics::Scope &scope, const IntrinsicProcTable &table) {
  if (auto why{CheckSpecificationExprHelper{scope, table}(x)}) {
    messages.Say("Invalid specification expression: %s"_err_en_US, *why);
  }
}

template void CheckSpecificationExpr(const Expr<SomeType> &,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
template void CheckSpecificationExpr(const Expr<SomeInteger> &,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
template void CheckSpecificationExpr(const Expr<SubscriptInteger> &,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
template void CheckSpecificationExpr(const std::optional<Expr<SomeType>> &,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
template void CheckSpecificationExpr(const std::optional<Expr<SomeInteger>> &,
    parser::ContextualMessages &, const semantics::Scope &,
    const IntrinsicProcTable &);
template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &, parser::ContextualMessages &,
    const semantics::Scope &, const IntrinsicProcTable &);

// IsSimplyContiguous() -- 9.5.4
class IsSimplyContiguousHelper
    : public AnyTraverse<IsSimplyContiguousHelper, std::optional<bool>> {
public:
  using Result = std::optional<bool>; // tri-state
  using Base = AnyTraverse<IsSimplyContiguousHelper, Result>;
  explicit IsSimplyContiguousHelper(const IntrinsicProcTable &t)
      : Base{*this}, table_{t} {}
  using Base::operator();

  Result operator()(const semantics::Symbol &symbol) const {
    if (symbol.attrs().test(semantics::Attr::CONTIGUOUS) ||
        symbol.Rank() == 0) {
      return true;
    } else if (semantics::IsPointer(symbol)) {
      return false;
    } else if (const auto *details{
                   symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      // N.B. ALLOCATABLEs are deferred shape, not assumed, and
      // are obviously contiguous.
      return !details->IsAssumedShape() && !details->IsAssumedRank();
    } else {
      return false;
    }
  }

  Result operator()(const ArrayRef &x) const {
    const auto &symbol{x.GetLastSymbol()};
    if (!(*this)(symbol)) {
      return false;
    } else if (auto rank{CheckSubscripts(x.subscript())}) {
      // a(:)%b(1,1) is not contiguous; a(1)%b(:,:) is
      return *rank > 0 || x.Rank() == 0;
    } else {
      return false;
    }
  }
  Result operator()(const CoarrayRef &x) const {
    return CheckSubscripts(x.subscript()).has_value();
  }
  Result operator()(const Component &x) const {
    return x.base().Rank() == 0 && (*this)(x.GetLastSymbol());
  }
  Result operator()(const ComplexPart &) const { return false; }
  Result operator()(const Substring &) const { return false; }

  template <typename T> Result operator()(const FunctionRef<T> &x) const {
    if (auto chars{
            characteristics::Procedure::Characterize(x.proc(), table_)}) {
      if (chars->functionResult) {
        const auto &result{*chars->functionResult};
        return !result.IsProcedurePointer() &&
            result.attrs.test(characteristics::FunctionResult::Attr::Pointer) &&
            result.attrs.test(
                characteristics::FunctionResult::Attr::Contiguous);
      }
    }
    return false;
  }

private:
  // If the subscripts can possibly be on a simply-contiguous array reference,
  // return the rank.
  static std::optional<int> CheckSubscripts(
      const std::vector<Subscript> &subscript) {
    bool anyTriplet{false};
    int rank{0};
    for (auto j{subscript.size()}; j-- > 0;) {
      if (const auto *triplet{std::get_if<Triplet>(&subscript[j].u)}) {
        if (!triplet->IsStrideOne()) {
          return std::nullopt;
        } else if (anyTriplet) {
          if (triplet->lower() || triplet->upper()) {
            // all triplets before the last one must be just ":"
            return std::nullopt;
          }
        } else {
          anyTriplet = true;
        }
        ++rank;
      } else if (anyTriplet || subscript[j].Rank() > 0) {
        return std::nullopt;
      }
    }
    return rank;
  }

  const IntrinsicProcTable &table_;
};

template <typename A>
bool IsSimplyContiguous(const A &x, const IntrinsicProcTable &table) {
  if (IsVariable(x)) {
    auto known{IsSimplyContiguousHelper{table}(x)};
    return known && *known;
  } else {
    return true; // not a variable
  }
}

template bool IsSimplyContiguous(
    const Expr<SomeType> &, const IntrinsicProcTable &);

} // namespace Fortran::evaluate
