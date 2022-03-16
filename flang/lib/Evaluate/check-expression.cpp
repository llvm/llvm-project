//===-- lib/Evaluate/check-expression.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <set>
#include <string>

namespace Fortran::evaluate {

// Constant expression predicates IsConstantExpr() & IsScopeInvariantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.
//
// The variant form (IsScopeInvariantExpr()) also accepts symbols that are
// INTENT(IN) dummy arguments without the VALUE attribute.
template <bool INVARIANT>
class IsConstantExprHelper
    : public AllTraverse<IsConstantExprHelper<INVARIANT>, true> {
public:
  using Base = AllTraverse<IsConstantExprHelper, true>;
  IsConstantExprHelper() : Base{*this} {}
  using Base::operator();

  // A missing expression is not considered to be constant.
  template <typename A> bool operator()(const std::optional<A> &x) const {
    return x && (*this)(*x);
  }

  bool operator()(const TypeParamInquiry &inq) const {
    return INVARIANT || semantics::IsKindTypeParameter(inq.parameter());
  }
  bool operator()(const semantics::Symbol &symbol) const {
    const auto &ultimate{GetAssociationRoot(symbol)};
    return IsNamedConstant(ultimate) || IsImpliedDoIndex(ultimate) ||
        IsInitialProcedureTarget(ultimate) ||
        ultimate.has<semantics::TypeParamDetails>() ||
        (INVARIANT && IsIntentIn(symbol) &&
            !symbol.attrs().test(semantics::Attr::VALUE));
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const semantics::ParamValue &param) const {
    return param.isExplicit() && (*this)(param.GetExplicit());
  }
  bool operator()(const ProcedureRef &) const;
  bool operator()(const StructureConstructor &constructor) const {
    for (const auto &[symRef, expr] : constructor) {
      if (!IsConstantStructureConstructorComponent(*symRef, expr.value())) {
        return false;
      }
    }
    return true;
  }
  bool operator()(const Component &component) const {
    return (*this)(component.base());
  }
  // Forbid integer division by zero in constants.
  template <int KIND>
  bool operator()(
      const Divide<Type<TypeCategory::Integer, KIND>> &division) const {
    using T = Type<TypeCategory::Integer, KIND>;
    if (const auto divisor{GetScalarConstantValue<T>(division.right())}) {
      return !divisor->IsZero() && (*this)(division.left());
    } else {
      return false;
    }
  }

  bool operator()(const Constant<SomeDerived> &) const { return true; }
  bool operator()(const DescriptorInquiry &x) const {
    const Symbol &sym{x.base().GetLastSymbol()};
    return INVARIANT && !IsAllocatable(sym) &&
        (!IsDummy(sym) ||
            (IsIntentIn(sym) && !sym.attrs().test(semantics::Attr::VALUE)));
  }

private:
  bool IsConstantStructureConstructorComponent(
      const Symbol &, const Expr<SomeType> &) const;
  bool IsConstantExprShape(const Shape &) const;
};

template <bool INVARIANT>
bool IsConstantExprHelper<INVARIANT>::IsConstantStructureConstructorComponent(
    const Symbol &component, const Expr<SomeType> &expr) const {
  if (IsAllocatable(component)) {
    return IsNullPointer(expr);
  } else if (IsPointer(component)) {
    return IsNullPointer(expr) || IsInitialDataTarget(expr) ||
        IsInitialProcedureTarget(expr);
  } else {
    return (*this)(expr);
  }
}

template <bool INVARIANT>
bool IsConstantExprHelper<INVARIANT>::operator()(
    const ProcedureRef &call) const {
  // LBOUND, UBOUND, and SIZE with DIM= arguments will have been rewritten
  // into DescriptorInquiry operations.
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
    if (intrinsic->name == "kind" ||
        intrinsic->name == IntrinsicProcTable::InvalidName) {
      // kind is always a constant, and we avoid cascading errors by considering
      // invalid calls to intrinsics to be constant
      return true;
    } else if (intrinsic->name == "lbound" && call.arguments().size() == 1) {
      // LBOUND(x) without DIM=
      auto base{ExtractNamedEntity(call.arguments()[0]->UnwrapExpr())};
      return base && IsConstantExprShape(GetLBOUNDs(*base));
    } else if (intrinsic->name == "ubound" && call.arguments().size() == 1) {
      // UBOUND(x) without DIM=
      auto base{ExtractNamedEntity(call.arguments()[0]->UnwrapExpr())};
      return base && IsConstantExprShape(GetUpperBounds(*base));
    } else if (intrinsic->name == "shape") {
      auto shape{GetShape(call.arguments()[0]->UnwrapExpr())};
      return shape && IsConstantExprShape(*shape);
    } else if (intrinsic->name == "size" && call.arguments().size() == 1) {
      // SIZE(x) without DIM
      auto shape{GetShape(call.arguments()[0]->UnwrapExpr())};
      return shape && IsConstantExprShape(*shape);
    }
    // TODO: STORAGE_SIZE
  }
  return false;
}

template <bool INVARIANT>
bool IsConstantExprHelper<INVARIANT>::IsConstantExprShape(
    const Shape &shape) const {
  for (const auto &extent : shape) {
    if (!(*this)(extent)) {
      return false;
    }
  }
  return true;
}

template <typename A> bool IsConstantExpr(const A &x) {
  return IsConstantExprHelper<false>{}(x);
}
template bool IsConstantExpr(const Expr<SomeType> &);
template bool IsConstantExpr(const Expr<SomeInteger> &);
template bool IsConstantExpr(const Expr<SubscriptInteger> &);
template bool IsConstantExpr(const StructureConstructor &);

// IsScopeInvariantExpr()
template <typename A> bool IsScopeInvariantExpr(const A &x) {
  return IsConstantExprHelper<true>{}(x);
}
template bool IsScopeInvariantExpr(const Expr<SomeType> &);
template bool IsScopeInvariantExpr(const Expr<SomeInteger> &);
template bool IsScopeInvariantExpr(const Expr<SubscriptInteger> &);

// IsActuallyConstant()
struct IsActuallyConstantHelper {
  template <typename A> bool operator()(const A &) { return false; }
  template <typename T> bool operator()(const Constant<T> &) { return true; }
  template <typename T> bool operator()(const Parentheses<T> &x) {
    return (*this)(x.left());
  }
  template <typename T> bool operator()(const Expr<T> &x) {
    return std::visit([=](const auto &y) { return (*this)(y); }, x.u);
  }
  bool operator()(const Expr<SomeType> &x) {
    if (IsNullPointer(x)) {
      return true;
    }
    return std::visit([this](const auto &y) { return (*this)(y); }, x.u);
  }
  template <typename A> bool operator()(const A *x) { return x && (*this)(*x); }
  template <typename A> bool operator()(const std::optional<A> &x) {
    return x && (*this)(*x);
  }
};

template <typename A> bool IsActuallyConstant(const A &x) {
  return IsActuallyConstantHelper{}(x);
}

template bool IsActuallyConstant(const Expr<SomeType> &);

// Object pointer initialization checking predicate IsInitialDataTarget().
// This code determines whether an expression is allowable as the static
// data address used to initialize a pointer with "=> x".  See C765.
class IsInitialDataTargetHelper
    : public AllTraverse<IsInitialDataTargetHelper, true> {
public:
  using Base = AllTraverse<IsInitialDataTargetHelper, true>;
  using Base::operator();
  explicit IsInitialDataTargetHelper(parser::ContextualMessages *m)
      : Base{*this}, messages_{m} {}

  bool emittedMessage() const { return emittedMessage_; }

  bool operator()(const BOZLiteralConstant &) const { return false; }
  bool operator()(const NullPointer &) const { return true; }
  template <typename T> bool operator()(const Constant<T> &) const {
    return false;
  }
  bool operator()(const semantics::Symbol &symbol) {
    // This function checks only base symbols, not components.
    const Symbol &ultimate{symbol.GetUltimate()};
    if (const auto *assoc{
            ultimate.detailsIf<semantics::AssocEntityDetails>()}) {
      if (const auto &expr{assoc->expr()}) {
        if (IsVariable(*expr)) {
          return (*this)(*expr);
        } else if (messages_) {
          messages_->Say(
              "An initial data target may not be an associated expression ('%s')"_err_en_US,
              ultimate.name());
          emittedMessage_ = true;
        }
      }
      return false;
    } else if (!ultimate.attrs().test(semantics::Attr::TARGET)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an object '%s' that lacks the TARGET attribute"_err_en_US,
            ultimate.name());
        emittedMessage_ = true;
      }
      return false;
    } else if (!IsSaved(ultimate)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an object '%s' that lacks the SAVE attribute"_err_en_US,
            ultimate.name());
        emittedMessage_ = true;
      }
      return false;
    } else {
      return CheckVarOrComponent(ultimate);
    }
  }
  bool operator()(const StaticDataObject &) const { return false; }
  bool operator()(const TypeParamInquiry &) const { return false; }
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
  bool operator()(const Component &x) {
    return CheckVarOrComponent(x.GetLastSymbol()) && (*this)(x.base());
  }
  bool operator()(const Substring &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        (*this)(x.parent());
  }
  bool operator()(const DescriptorInquiry &) const { return false; }
  template <typename T> bool operator()(const ArrayConstructor<T> &) const {
    return false;
  }
  bool operator()(const StructureConstructor &) const { return false; }
  template <typename D, typename R, typename... O>
  bool operator()(const Operation<D, R, O...> &) const {
    return false;
  }
  template <typename T> bool operator()(const Parentheses<T> &x) const {
    return (*this)(x.left());
  }
  bool operator()(const ProcedureRef &x) const {
    if (const SpecificIntrinsic * intrinsic{x.proc().GetSpecificIntrinsic()}) {
      return intrinsic->characteristics.value().attrs.test(
          characteristics::Procedure::Attr::NullPointer);
    }
    return false;
  }
  bool operator()(const Relational<SomeType> &) const { return false; }

private:
  bool CheckVarOrComponent(const semantics::Symbol &symbol) {
    const Symbol &ultimate{symbol.GetUltimate()};
    if (IsAllocatable(ultimate)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an ALLOCATABLE '%s'"_err_en_US,
            ultimate.name());
        emittedMessage_ = true;
      }
      return false;
    } else if (ultimate.Corank() > 0) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to a coarray '%s'"_err_en_US,
            ultimate.name());
        emittedMessage_ = true;
      }
      return false;
    }
    return true;
  }

  parser::ContextualMessages *messages_;
  bool emittedMessage_{false};
};

bool IsInitialDataTarget(
    const Expr<SomeType> &x, parser::ContextualMessages *messages) {
  IsInitialDataTargetHelper helper{messages};
  bool result{helper(x)};
  if (!result && messages && !helper.emittedMessage()) {
    messages->Say(
        "An initial data target must be a designator with constant subscripts"_err_en_US);
  }
  return result;
}

bool IsInitialProcedureTarget(const semantics::Symbol &symbol) {
  const auto &ultimate{symbol.GetUltimate()};
  return std::visit(
      common::visitors{
          [](const semantics::SubprogramDetails &subp) {
            return !subp.isDummy();
          },
          [](const semantics::SubprogramNameDetails &) { return true; },
          [&](const semantics::ProcEntityDetails &proc) {
            return !semantics::IsPointer(ultimate) && !proc.isDummy();
          },
          [](const auto &) { return false; },
      },
      ultimate.details());
}

bool IsInitialProcedureTarget(const ProcedureDesignator &proc) {
  if (const auto *intrin{proc.GetSpecificIntrinsic()}) {
    return !intrin->isRestrictedSpecific;
  } else if (proc.GetComponent()) {
    return false;
  } else {
    return IsInitialProcedureTarget(DEREF(proc.GetSymbol()));
  }
}

bool IsInitialProcedureTarget(const Expr<SomeType> &expr) {
  if (const auto *proc{std::get_if<ProcedureDesignator>(&expr.u)}) {
    return IsInitialProcedureTarget(*proc);
  } else {
    return IsNullPointer(expr);
  }
}

class ArrayConstantBoundChanger {
public:
  ArrayConstantBoundChanger(ConstantSubscripts &&lbounds)
      : lbounds_{std::move(lbounds)} {}

  template <typename A> A ChangeLbounds(A &&x) const {
    return std::move(x); // default case
  }
  template <typename T> Constant<T> ChangeLbounds(Constant<T> &&x) {
    x.set_lbounds(std::move(lbounds_));
    return std::move(x);
  }
  template <typename T> Expr<T> ChangeLbounds(Parentheses<T> &&x) {
    return ChangeLbounds(
        std::move(x.left())); // Constant<> can be parenthesized
  }
  template <typename T> Expr<T> ChangeLbounds(Expr<T> &&x) {
    return std::visit(
        [&](auto &&x) { return Expr<T>{ChangeLbounds(std::move(x))}; },
        std::move(x.u)); // recurse until we hit a constant
  }

private:
  ConstantSubscripts &&lbounds_;
};

// Converts, folds, and then checks type, rank, and shape of an
// initialization expression for a named constant, a non-pointer
// variable static initialization, a component default initializer,
// a type parameter default value, or instantiated type parameter value.
std::optional<Expr<SomeType>> NonPointerInitializationExpr(const Symbol &symbol,
    Expr<SomeType> &&x, FoldingContext &context,
    const semantics::Scope *instantiation) {
  CHECK(!IsPointer(symbol));
  if (auto symTS{
          characteristics::TypeAndShape::Characterize(symbol, context)}) {
    auto xType{x.GetType()};
    auto converted{ConvertToType(symTS->type(), Expr<SomeType>{x})};
    if (!converted &&
        symbol.owner().context().IsEnabled(
            common::LanguageFeature::LogicalIntegerAssignment)) {
      converted = DataConstantConversionExtension(context, symTS->type(), x);
      if (converted &&
          symbol.owner().context().ShouldWarn(
              common::LanguageFeature::LogicalIntegerAssignment)) {
        context.messages().Say(
            "nonstandard usage: initialization of %s with %s"_port_en_US,
            symTS->type().AsFortran(), x.GetType().value().AsFortran());
      }
    }
    if (converted) {
      auto folded{Fold(context, std::move(*converted))};
      if (IsActuallyConstant(folded)) {
        int symRank{GetRank(symTS->shape())};
        if (IsImpliedShape(symbol)) {
          if (folded.Rank() == symRank) {
            return {std::move(folded)};
          } else {
            context.messages().Say(
                "Implied-shape parameter '%s' has rank %d but its initializer has rank %d"_err_en_US,
                symbol.name(), symRank, folded.Rank());
          }
        } else if (auto extents{AsConstantExtents(context, symTS->shape())}) {
          if (folded.Rank() == 0 && symRank == 0) {
            // symbol and constant are both scalars
            return {std::move(folded)};
          } else if (folded.Rank() == 0 && symRank > 0) {
            // expand the scalar constant to an array
            return ScalarConstantExpander{std::move(*extents),
                AsConstantExtents(
                    context, GetRawLowerBounds(context, NamedEntity{symbol}))}
                .Expand(std::move(folded));
          } else if (auto resultShape{GetShape(context, folded)}) {
            if (CheckConformance(context.messages(), symTS->shape(),
                    *resultShape, CheckConformanceFlags::None,
                    "initialized object", "initialization expression")
                    .value_or(false /*fail if not known now to conform*/)) {
              // make a constant array with adjusted lower bounds
              return ArrayConstantBoundChanger{
                  std::move(*AsConstantExtents(context,
                      GetRawLowerBounds(context, NamedEntity{symbol})))}
                  .ChangeLbounds(std::move(folded));
            }
          }
        } else if (IsNamedConstant(symbol)) {
          if (IsExplicitShape(symbol)) {
            context.messages().Say(
                "Named constant '%s' array must have constant shape"_err_en_US,
                symbol.name());
          } else {
            // Declaration checking handles other cases
          }
        } else {
          context.messages().Say(
              "Shape of initialized object '%s' must be constant"_err_en_US,
              symbol.name());
        }
      } else if (IsErrorExpr(folded)) {
      } else if (IsLenTypeParameter(symbol)) {
        return {std::move(folded)};
      } else if (IsKindTypeParameter(symbol)) {
        if (instantiation) {
          context.messages().Say(
              "Value of kind type parameter '%s' (%s) must be a scalar INTEGER constant"_err_en_US,
              symbol.name(), folded.AsFortran());
        } else {
          return {std::move(folded)};
        }
      } else if (IsNamedConstant(symbol)) {
        context.messages().Say(
            "Value of named constant '%s' (%s) cannot be computed as a constant value"_err_en_US,
            symbol.name(), folded.AsFortran());
      } else {
        context.messages().Say(
            "Initialization expression for '%s' (%s) cannot be computed as a constant value"_err_en_US,
            symbol.name(), folded.AsFortran());
      }
    } else if (xType) {
      context.messages().Say(
          "Initialization expression cannot be converted to declared type of '%s' from %s"_err_en_US,
          symbol.name(), xType->AsFortran());
    } else {
      context.messages().Say(
          "Initialization expression cannot be converted to declared type of '%s'"_err_en_US,
          symbol.name());
    }
  }
  return std::nullopt;
}

// Specification expression validation (10.1.11(2), C1010)
class CheckSpecificationExprHelper
    : public AnyTraverse<CheckSpecificationExprHelper,
          std::optional<std::string>> {
public:
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<CheckSpecificationExprHelper, Result>;
  explicit CheckSpecificationExprHelper(
      const semantics::Scope &s, FoldingContext &context)
      : Base{*this}, scope_{s}, context_{context} {}
  using Base::operator();

  Result operator()(const CoarrayRef &) const { return "coindexed reference"; }

  Result operator()(const semantics::Symbol &symbol) const {
    const auto &ultimate{symbol.GetUltimate()};
    if (const auto *assoc{
            ultimate.detailsIf<semantics::AssocEntityDetails>()}) {
      return (*this)(assoc->expr());
    } else if (semantics::IsNamedConstant(ultimate) ||
        ultimate.owner().IsModule() || ultimate.owner().IsSubmodule()) {
      return std::nullopt;
    } else if (scope_.IsDerivedType() &&
        IsVariableName(ultimate)) { // C750, C754
      return "derived type component or type parameter value not allowed to "
             "reference variable '"s +
          ultimate.name().ToString() + "'";
    } else if (IsDummy(ultimate)) {
      if (ultimate.attrs().test(semantics::Attr::OPTIONAL)) {
        return "reference to OPTIONAL dummy argument '"s +
            ultimate.name().ToString() + "'";
      } else if (ultimate.attrs().test(semantics::Attr::INTENT_OUT)) {
        return "reference to INTENT(OUT) dummy argument '"s +
            ultimate.name().ToString() + "'";
      } else if (ultimate.has<semantics::ObjectEntityDetails>()) {
        return std::nullopt;
      } else {
        return "dummy procedure argument";
      }
    } else if (&symbol.owner() != &scope_ || &ultimate.owner() != &scope_) {
      return std::nullopt; // host association is in play
    } else if (const auto *object{
                   ultimate.detailsIf<semantics::ObjectEntityDetails>()}) {
      if (object->commonBlock()) {
        return std::nullopt;
      }
    }
    return "reference to local entity '"s + ultimate.name().ToString() + "'";
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

  Result operator()(const TypeParamInquiry &inq) const {
    if (scope_.IsDerivedType() && !IsConstantExpr(inq) &&
        inq.base() /* X%T, not local T */) { // C750, C754
      return "non-constant reference to a type parameter inquiry not "
             "allowed for derived type components or type parameter values";
    }
    return std::nullopt;
  }

  Result operator()(const ProcedureRef &x) const {
    if (const auto *symbol{x.proc().GetSymbol()}) {
      const Symbol &ultimate{symbol->GetUltimate()};
      if (!semantics::IsPureProcedure(ultimate)) {
        return "reference to impure function '"s + ultimate.name().ToString() +
            "'";
      }
      if (semantics::IsStmtFunction(ultimate)) {
        return "reference to statement function '"s +
            ultimate.name().ToString() + "'";
      }
      if (scope_.IsDerivedType()) { // C750, C754
        return "reference to function '"s + ultimate.name().ToString() +
            "' not allowed for derived type components or type parameter"
            " values";
      }
      if (auto procChars{
              characteristics::Procedure::Characterize(x.proc(), context_)}) {
        const auto iter{std::find_if(procChars->dummyArguments.begin(),
            procChars->dummyArguments.end(),
            [](const characteristics::DummyArgument &dummy) {
              return std::holds_alternative<characteristics::DummyProcedure>(
                  dummy.u);
            })};
        if (iter != procChars->dummyArguments.end()) {
          return "reference to function '"s + ultimate.name().ToString() +
              "' with dummy procedure argument '" + iter->name + '\'';
        }
      }
      // References to internal functions are caught in expression semantics.
      // TODO: other checks for standard module procedures
    } else {
      const SpecificIntrinsic &intrin{DEREF(x.proc().GetSpecificIntrinsic())};
      if (scope_.IsDerivedType()) { // C750, C754
        if ((context_.intrinsics().IsIntrinsic(intrin.name) &&
                badIntrinsicsForComponents_.find(intrin.name) !=
                    badIntrinsicsForComponents_.end()) ||
            IsProhibitedFunction(intrin.name)) {
          return "reference to intrinsic '"s + intrin.name +
              "' not allowed for derived type components or type parameter"
              " values";
        }
        if (context_.intrinsics().GetIntrinsicClass(intrin.name) ==
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
  FoldingContext &context_;
  const std::set<std::string> badIntrinsicsForComponents_{
      "allocated", "associated", "extends_type_of", "present", "same_type_as"};
  static bool IsProhibitedFunction(std::string name) { return false; }
};

template <typename A>
void CheckSpecificationExpr(
    const A &x, const semantics::Scope &scope, FoldingContext &context) {
  if (auto why{CheckSpecificationExprHelper{scope, context}(x)}) {
    context.messages().Say(
        "Invalid specification expression: %s"_err_en_US, *why);
  }
}

template void CheckSpecificationExpr(
    const Expr<SomeType> &, const semantics::Scope &, FoldingContext &);
template void CheckSpecificationExpr(
    const Expr<SomeInteger> &, const semantics::Scope &, FoldingContext &);
template void CheckSpecificationExpr(
    const Expr<SubscriptInteger> &, const semantics::Scope &, FoldingContext &);
template void CheckSpecificationExpr(const std::optional<Expr<SomeType>> &,
    const semantics::Scope &, FoldingContext &);
template void CheckSpecificationExpr(const std::optional<Expr<SomeInteger>> &,
    const semantics::Scope &, FoldingContext &);
template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &, const semantics::Scope &,
    FoldingContext &);

// IsSimplyContiguous() -- 9.5.4
class IsSimplyContiguousHelper
    : public AnyTraverse<IsSimplyContiguousHelper, std::optional<bool>> {
public:
  using Result = std::optional<bool>; // tri-state
  using Base = AnyTraverse<IsSimplyContiguousHelper, Result>;
  explicit IsSimplyContiguousHelper(FoldingContext &c)
      : Base{*this}, context_{c} {}
  using Base::operator();

  Result operator()(const semantics::Symbol &symbol) const {
    const auto &ultimate{symbol.GetUltimate()};
    if (ultimate.attrs().test(semantics::Attr::CONTIGUOUS)) {
      return true;
    } else if (ultimate.Rank() == 0) {
      // Extension: accept scalars as a degenerate case of
      // simple contiguity to allow their use in contexts like
      // data targets in pointer assignments with remapping.
      return true;
    } else if (semantics::IsPointer(ultimate) ||
        semantics::IsAssumedShape(ultimate)) {
      return false;
    } else if (const auto *details{
                   ultimate.detailsIf<semantics::ObjectEntityDetails>()}) {
      return !details->IsAssumedRank();
    } else if (auto assoc{Base::operator()(ultimate)}) {
      return assoc;
    } else {
      return false;
    }
  }

  Result operator()(const ArrayRef &x) const {
    const auto &symbol{x.GetLastSymbol()};
    if (!(*this)(symbol).has_value()) {
      return false;
    } else if (auto rank{CheckSubscripts(x.subscript())}) {
      if (x.Rank() == 0) {
        return true;
      } else if (*rank > 0) {
        // a(1)%b(:,:) is contiguous if an only if a(1)%b is contiguous.
        return (*this)(x.base());
      } else {
        // a(:)%b(1,1) is not contiguous.
        return false;
      }
    } else {
      return false;
    }
  }
  Result operator()(const CoarrayRef &x) const {
    return CheckSubscripts(x.subscript()).has_value();
  }
  Result operator()(const Component &x) const {
    return x.base().Rank() == 0 && (*this)(x.GetLastSymbol()).value_or(false);
  }
  Result operator()(const ComplexPart &) const { return false; }
  Result operator()(const Substring &) const { return false; }

  Result operator()(const ProcedureRef &x) const {
    if (auto chars{
            characteristics::Procedure::Characterize(x.proc(), context_)}) {
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

  FoldingContext &context_;
};

template <typename A>
bool IsSimplyContiguous(const A &x, FoldingContext &context) {
  if (IsVariable(x)) {
    auto known{IsSimplyContiguousHelper{context}(x)};
    return known && *known;
  } else {
    return true; // not a variable
  }
}

template bool IsSimplyContiguous(const Expr<SomeType> &, FoldingContext &);

// IsErrorExpr()
struct IsErrorExprHelper : public AnyTraverse<IsErrorExprHelper, bool> {
  using Result = bool;
  using Base = AnyTraverse<IsErrorExprHelper, Result>;
  IsErrorExprHelper() : Base{*this} {}
  using Base::operator();

  bool operator()(const SpecificIntrinsic &x) {
    return x.name == IntrinsicProcTable::InvalidName;
  }
};

template <typename A> bool IsErrorExpr(const A &x) {
  return IsErrorExprHelper{}(x);
}

template bool IsErrorExpr(const Expr<SomeType> &);

} // namespace Fortran::evaluate
