//===-- lib/Semantics/type.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/type.h"
#include "flang/Evaluate/fold.h"
#include "flang/Parser/characters.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <ostream>
#include <sstream>

namespace Fortran::semantics {

DerivedTypeSpec::DerivedTypeSpec(SourceName name, const Symbol &typeSymbol)
  : name_{name}, typeSymbol_{typeSymbol} {
  CHECK(typeSymbol.has<DerivedTypeDetails>());
}
DerivedTypeSpec::DerivedTypeSpec(const DerivedTypeSpec &that) = default;
DerivedTypeSpec::DerivedTypeSpec(DerivedTypeSpec &&that) = default;

void DerivedTypeSpec::set_scope(const Scope &scope) {
  CHECK(!scope_);
  ReplaceScope(scope);
}
void DerivedTypeSpec::ReplaceScope(const Scope &scope) {
  CHECK(scope.IsDerivedType());
  scope_ = &scope;
}

void DerivedTypeSpec::AddRawParamValue(
    const std::optional<parser::Keyword> &keyword, ParamValue &&value) {
  CHECK(parameters_.empty());
  rawParameters_.emplace_back(keyword ? &*keyword : nullptr, std::move(value));
}

void DerivedTypeSpec::CookParameters(evaluate::FoldingContext &foldingContext) {
  if (cooked_) {
    return;
  }
  cooked_ = true;
  auto &messages{foldingContext.messages()};
  if (IsForwardReferenced()) {
    messages.Say(typeSymbol_.name(),
        "Derived type '%s' was used but never defined"_err_en_US,
        typeSymbol_.name());
    return;
  }

  // Parameters of the most deeply nested "base class" come first when the
  // derived type is an extension.
  auto parameterNames{OrderParameterNames(typeSymbol_)};
  auto parameterDecls{OrderParameterDeclarations(typeSymbol_)};
  auto nextNameIter{parameterNames.begin()};
  RawParameters raw{std::move(rawParameters_)};
  for (auto &[maybeKeyword, value] : raw) {
    SourceName name;
    common::TypeParamAttr attr{common::TypeParamAttr::Kind};
    if (maybeKeyword) {
      name = maybeKeyword->v.source;
      auto it{std::find_if(parameterDecls.begin(), parameterDecls.end(),
          [&](const Symbol &symbol) { return symbol.name() == name; })};
      if (it == parameterDecls.end()) {
        messages.Say(name,
            "'%s' is not the name of a parameter for derived type '%s'"_err_en_US,
            name, typeSymbol_.name());
      } else {
        // Resolve the keyword's symbol
        maybeKeyword->v.symbol = const_cast<Symbol *>(&it->get());
        attr = it->get().get<TypeParamDetails>().attr();
      }
    } else if (nextNameIter != parameterNames.end()) {
      name = *nextNameIter++;
      auto it{std::find_if(parameterDecls.begin(), parameterDecls.end(),
          [&](const Symbol &symbol) { return symbol.name() == name; })};
      CHECK(it != parameterDecls.end());
      attr = it->get().get<TypeParamDetails>().attr();
    } else {
      messages.Say(name_,
          "Too many type parameters given for derived type '%s'"_err_en_US,
          typeSymbol_.name());
      break;
    }
    if (FindParameter(name)) {
      messages.Say(name_,
          "Multiple values given for type parameter '%s'"_err_en_US, name);
    } else {
      value.set_attr(attr);
      AddParamValue(name, std::move(value));
    }
  }
}

void DerivedTypeSpec::EvaluateParameters(
    evaluate::FoldingContext &foldingContext) {
  CookParameters(foldingContext);
  if (evaluated_) {
    return;
  }
  evaluated_ = true;
  auto &messages{foldingContext.messages()};

  // Fold the explicit type parameter value expressions first.  Do not
  // fold them within the scope of the derived type being instantiated;
  // these expressions cannot use its type parameters.  Convert the values
  // of the expressions to the declared types of the type parameters.
  auto parameterDecls{OrderParameterDeclarations(typeSymbol_)};
  for (const Symbol &symbol : parameterDecls) {
    const SourceName &name{symbol.name()};
    if (ParamValue * paramValue{FindParameter(name)}) {
      if (const MaybeIntExpr & expr{paramValue->GetExplicit()}) {
        if (auto converted{evaluate::ConvertToType(symbol, SomeExpr{*expr})}) {
          SomeExpr folded{
              evaluate::Fold(foldingContext, std::move(*converted))};
          if (auto *intExpr{std::get_if<SomeIntExpr>(&folded.u)}) {
            paramValue->SetExplicit(std::move(*intExpr));
            continue;
          }
        }
        evaluate::SayWithDeclaration(messages, symbol,
            "Value of type parameter '%s' (%s) is not convertible to its type"_err_en_US,
            name, expr->AsFortran());
      }
    }
  }

  // Default initialization expressions for the derived type's parameters
  // may reference other parameters so long as the declaration precedes the
  // use in the expression (10.1.12).  This is not necessarily the same
  // order as "type parameter order" (7.5.3.2).
  // Type parameter default value expressions are folded in declaration order
  // within the scope of the derived type so that the values of earlier type
  // parameters are available for use in the default initialization
  // expressions of later parameters.
  auto restorer{foldingContext.WithPDTInstance(*this)};
  for (const Symbol &symbol : parameterDecls) {
    const SourceName &name{symbol.name()};
    if (!FindParameter(name)) {
      const TypeParamDetails &details{symbol.get<TypeParamDetails>()};
      if (details.init()) {
        auto expr{
            evaluate::Fold(foldingContext, common::Clone(details.init()))};
        AddParamValue(name, ParamValue{std::move(*expr), details.attr()});
      } else {
        messages.Say(name_,
            "Type parameter '%s' lacks a value and has no default"_err_en_US,
            name);
      }
    }
  }
}

void DerivedTypeSpec::AddParamValue(SourceName name, ParamValue &&value) {
  CHECK(cooked_);
  auto pair{parameters_.insert(std::make_pair(name, std::move(value)))};
  CHECK(pair.second);  // name was not already present
}

bool DerivedTypeSpec::MightBeParameterized() const {
  return !cooked_ || !parameters_.empty();
}

bool DerivedTypeSpec::IsForwardReferenced() const {
  return typeSymbol_.get<DerivedTypeDetails>().isForwardReferenced();
}

bool DerivedTypeSpec::HasDefaultInitialization() const {
  for (const Scope *scope{scope_}; scope;
       scope = scope->GetDerivedTypeParent()) {
    for (const auto &pair : *scope) {
      const Symbol &symbol{*pair.second};
      if (IsAllocatable(symbol) || IsInitialized(symbol)) {
        return true;
      }
    }
  }
  return false;
}

ParamValue *DerivedTypeSpec::FindParameter(SourceName target) {
  return const_cast<ParamValue *>(
      const_cast<const DerivedTypeSpec *>(this)->FindParameter(target));
}

void DerivedTypeSpec::Instantiate(
    Scope &containingScope, SemanticsContext &context) {
  if (instantiated_) {
    return;
  }
  instantiated_ = true;
  auto &foldingContext{context.foldingContext()};
  if (IsForwardReferenced()) {
    foldingContext.messages().Say(typeSymbol_.name(),
        "The derived type '%s' was forward-referenced but not defined"_err_en_US,
        typeSymbol_.name());
    return;
  }
  CookParameters(foldingContext);
  EvaluateParameters(foldingContext);
  const Scope &typeScope{DEREF(typeSymbol_.scope())};
  if (!MightBeParameterized()) {
    scope_ = &typeScope;
    for (const auto &pair : typeScope) {
      const Symbol &symbol{*pair.second};
      if (const DeclTypeSpec * type{symbol.GetType()}) {
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          if (!(derived->IsForwardReferenced() &&
                  IsAllocatableOrPointer(symbol))) {
            auto &instantiatable{*const_cast<DerivedTypeSpec *>(derived)};
            instantiatable.Instantiate(containingScope, context);
          }
        }
      }
    }
    return;
  }
  Scope &newScope{containingScope.MakeScope(Scope::Kind::DerivedType)};
  newScope.set_derivedTypeSpec(*this);
  ReplaceScope(newScope);
  for (const Symbol &symbol : OrderParameterDeclarations(typeSymbol_)) {
    const SourceName &name{symbol.name()};
    if (typeScope.find(symbol.name()) != typeScope.end()) {
      // This type parameter belongs to the derived type itself, not to
      // one of its ancestors.  Put the type parameter expression value
      // into the new scope as the initialization value for the parameter.
      if (ParamValue * paramValue{FindParameter(name)}) {
        const TypeParamDetails &details{symbol.get<TypeParamDetails>()};
        paramValue->set_attr(details.attr());
        if (MaybeIntExpr expr{paramValue->GetExplicit()}) {
          // Ensure that any kind type parameters with values are
          // constant by now.
          if (details.attr() == common::TypeParamAttr::Kind) {
            // Any errors in rank and type will have already elicited
            // messages, so don't pile on by complaining further here.
            if (auto maybeDynamicType{expr->GetType()}) {
              if (expr->Rank() == 0 &&
                  maybeDynamicType->category() == TypeCategory::Integer) {
                if (!evaluate::ToInt64(*expr)) {
                  if (auto *msg{foldingContext.messages().Say(
                          "Value of kind type parameter '%s' (%s) is not "
                          "a scalar INTEGER constant"_err_en_US,
                          name, expr->AsFortran())}) {
                    msg->Attach(name, "declared here"_en_US);
                  }
                }
              }
            }
          }
          TypeParamDetails instanceDetails{details.attr()};
          if (const DeclTypeSpec * type{details.type()}) {
            instanceDetails.set_type(*type);
          }
          instanceDetails.set_init(std::move(*expr));
          newScope.try_emplace(name, std::move(instanceDetails));
        }
      }
    }
  }
  // Instantiate every non-parameter symbol from the original derived
  // type's scope into the new instance.
  auto restorer{foldingContext.WithPDTInstance(*this)};
  newScope.AddSourceRange(typeScope.sourceRange());
  for (const auto &pair : typeScope) {
    const Symbol &symbol{*pair.second};
    symbol.InstantiateComponent(newScope, context);
  }
}

std::string DerivedTypeSpec::AsFortran() const {
  std::stringstream ss;
  ss << name_;
  if (!rawParameters_.empty()) {
    CHECK(parameters_.empty());
    ss << '(';
    bool first = true;
    for (const auto &[maybeKeyword, value] : rawParameters_) {
      if (first) {
        first = false;
      } else {
        ss << ',';
      }
      if (maybeKeyword) {
        ss << maybeKeyword->v.source.ToString() << '=';
      }
      ss << value.AsFortran();
    }
    ss << ')';
  } else if (!parameters_.empty()) {
    ss << '(';
    bool first = true;
    for (const auto &[name, value] : parameters_) {
      if (first) {
        first = false;
      } else {
        ss << ',';
      }
      ss << name.ToString() << '=' << value.AsFortran();
    }
    ss << ')';
  }
  return ss.str();
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  return o << x.AsFortran();
}

Bound::Bound(int bound) : expr_{bound} {}

std::ostream &operator<<(std::ostream &o, const Bound &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else if (x.expr_) {
    x.expr_->AsFortran(o);
  } else {
    o << "<no-expr>";
  }
  return o;
}

std::ostream &operator<<(std::ostream &o, const ShapeSpec &x) {
  if (x.lb_.isAssumed()) {
    CHECK(x.ub_.isAssumed());
    o << "..";
  } else {
    if (!x.lb_.isDeferred()) {
      o << x.lb_;
    }
    o << ':';
    if (!x.ub_.isDeferred()) {
      o << x.ub_;
    }
  }
  return o;
}

bool ArraySpec::IsExplicitShape() const {
  return CheckAll([](const ShapeSpec &x) { return x.ubound().isExplicit(); });
}
bool ArraySpec::IsAssumedShape() const {
  return CheckAll([](const ShapeSpec &x) { return x.ubound().isDeferred(); });
}
bool ArraySpec::IsDeferredShape() const {
  return CheckAll([](const ShapeSpec &x) {
    return x.lbound().isDeferred() && x.ubound().isDeferred();
  });
}
bool ArraySpec::IsImpliedShape() const {
  return !IsAssumedRank() &&
      CheckAll([](const ShapeSpec &x) { return x.ubound().isAssumed(); });
}
bool ArraySpec::IsAssumedSize() const {
  return !empty() && !IsAssumedRank() && back().ubound().isAssumed() &&
      std::all_of(begin(), end() - 1,
          [](const ShapeSpec &x) { return x.ubound().isExplicit(); });
}
bool ArraySpec::IsAssumedRank() const {
  return Rank() == 1 && front().lbound().isAssumed();
}

std::ostream &operator<<(std::ostream &os, const ArraySpec &arraySpec) {
  char sep{'('};
  for (auto &shape : arraySpec) {
    os << sep << shape;
    sep = ',';
  }
  if (sep == ',') {
    os << ')';
  }
  return os;
}

ParamValue::ParamValue(MaybeIntExpr &&expr, common::TypeParamAttr attr)
  : attr_{attr}, expr_{std::move(expr)} {}
ParamValue::ParamValue(SomeIntExpr &&expr, common::TypeParamAttr attr)
  : attr_{attr}, expr_{std::move(expr)} {}
ParamValue::ParamValue(
    common::ConstantSubscript value, common::TypeParamAttr attr)
  : ParamValue(
        SomeIntExpr{evaluate::Expr<evaluate::SubscriptInteger>{value}}, attr) {}

void ParamValue::SetExplicit(SomeIntExpr &&x) {
  category_ = Category::Explicit;
  expr_ = std::move(x);
}

std::string ParamValue::AsFortran() const {
  switch (category_) {
    SWITCH_COVERS_ALL_CASES
  case Category::Assumed: return "*";
  case Category::Deferred: return ":";
  case Category::Explicit:
    if (expr_) {
      std::stringstream ss;
      expr_->AsFortran(ss);
      return ss.str();
    } else {
      return "";
    }
  }
}

std::ostream &operator<<(std::ostream &o, const ParamValue &x) {
  return o << x.AsFortran();
}

IntrinsicTypeSpec::IntrinsicTypeSpec(TypeCategory category, KindExpr &&kind)
  : category_{category}, kind_{std::move(kind)} {
  CHECK(category != TypeCategory::Derived);
}

static std::string KindAsFortran(const KindExpr &kind) {
  std::stringstream ss;
  if (auto k{evaluate::ToInt64(kind)}) {
    ss << *k;  // emit unsuffixed kind code
  } else {
    kind.AsFortran(ss);
  }
  return ss.str();
}

std::string IntrinsicTypeSpec::AsFortran() const {
  return parser::ToUpperCaseLetters(common::EnumToString(category_)) + '(' +
      KindAsFortran(kind_) + ')';
}

std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x) {
  return os << x.AsFortran();
}

std::string CharacterTypeSpec::AsFortran() const {
  return "CHARACTER(" + length_.AsFortran() + ',' + KindAsFortran(kind()) + ')';
}

std::ostream &operator<<(std::ostream &os, const CharacterTypeSpec &x) {
  return os << x.AsFortran();
}

DeclTypeSpec::DeclTypeSpec(NumericTypeSpec &&typeSpec)
  : category_{Numeric}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(LogicalTypeSpec &&typeSpec)
  : category_{Logical}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(const CharacterTypeSpec &typeSpec)
  : category_{Character}, typeSpec_{typeSpec} {}
DeclTypeSpec::DeclTypeSpec(CharacterTypeSpec &&typeSpec)
  : category_{Character}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(Category category, const DerivedTypeSpec &typeSpec)
  : category_{category}, typeSpec_{typeSpec} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category, DerivedTypeSpec &&typeSpec)
  : category_{category}, typeSpec_{std::move(typeSpec)} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category) : category_{category} {
  CHECK(category == TypeStar || category == ClassStar);
}
bool DeclTypeSpec::IsNumeric(TypeCategory tc) const {
  return category_ == Numeric && numericTypeSpec().category() == tc;
}
IntrinsicTypeSpec *DeclTypeSpec::AsIntrinsic() {
  return const_cast<IntrinsicTypeSpec *>(
      const_cast<const DeclTypeSpec *>(this)->AsIntrinsic());
}
const NumericTypeSpec &DeclTypeSpec::numericTypeSpec() const {
  CHECK(category_ == Numeric);
  return std::get<NumericTypeSpec>(typeSpec_);
}
const LogicalTypeSpec &DeclTypeSpec::logicalTypeSpec() const {
  CHECK(category_ == Logical);
  return std::get<LogicalTypeSpec>(typeSpec_);
}
bool DeclTypeSpec::operator==(const DeclTypeSpec &that) const {
  return category_ == that.category_ && typeSpec_ == that.typeSpec_;
}

std::string DeclTypeSpec::AsFortran() const {
  switch (category_) {
    SWITCH_COVERS_ALL_CASES
  case Numeric: return numericTypeSpec().AsFortran();
  case Logical: return logicalTypeSpec().AsFortran();
  case Character: return characterTypeSpec().AsFortran();
  case TypeDerived: return "TYPE(" + derivedTypeSpec().AsFortran() + ')';
  case ClassDerived: return "CLASS(" + derivedTypeSpec().AsFortran() + ')';
  case TypeStar: return "TYPE(*)";
  case ClassStar: return "CLASS(*)";
  }
}

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  return o << x.AsFortran();
}

void ProcInterface::set_symbol(const Symbol &symbol) {
  CHECK(!type_);
  symbol_ = &symbol;
}
void ProcInterface::set_type(const DeclTypeSpec &type) {
  CHECK(!symbol_);
  type_ = &type;
}
}
