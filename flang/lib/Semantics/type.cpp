//===-- lib/Semantics/type.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/type.h"
#include "check-declarations.h"
#include "compute-offsets.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/raw_ostream.h"

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

const Scope *DerivedTypeSpec::GetScope() const {
  return scope_ ? scope_ : typeSymbol_.scope();
}

void DerivedTypeSpec::AddRawParamValue(
    const parser::Keyword *keyword, ParamValue &&value) {
  CHECK(parameters_.empty());
  rawParameters_.emplace_back(keyword, std::move(value));
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
  auto nextNameIter{parameterNames.begin()};
  RawParameters raw{std::move(rawParameters_)};
  for (auto &[maybeKeyword, value] : raw) {
    SourceName name;
    common::TypeParamAttr attr{common::TypeParamAttr::Kind};
    if (maybeKeyword) {
      name = maybeKeyword->v.source;
      auto it{std::find_if(parameterNames.begin(), parameterNames.end(),
          [&](const Symbol &symbol) { return symbol.name() == name; })};
      if (it == parameterNames.end()) {
        messages.Say(name,
            "'%s' is not the name of a parameter for derived type '%s'"_err_en_US,
            name, typeSymbol_.name());
      } else {
        // Resolve the keyword's symbol
        maybeKeyword->v.symbol = const_cast<Symbol *>(&it->get());
        if (const auto *tpd{it->get().detailsIf<TypeParamDetails>()}) {
          attr = tpd->attr().value_or(attr);
        }
      }
    } else if (nextNameIter != parameterNames.end()) {
      name = nextNameIter->get().name();
      if (const auto *tpd{nextNameIter->get().detailsIf<TypeParamDetails>()}) {
        attr = tpd->attr().value_or(attr);
      }
      ++nextNameIter;
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

void DerivedTypeSpec::EvaluateParameters(SemanticsContext &context) {
  evaluate::FoldingContext &foldingContext{context.foldingContext()};
  CookParameters(foldingContext);
  if (evaluated_) {
    return;
  }
  evaluated_ = true;
  auto &messages{foldingContext.messages()};
  for (const Symbol &symbol : OrderParameterDeclarations(typeSymbol_)) {
    SourceName name{symbol.name()};
    int parameterKind{evaluate::TypeParamInquiry::Result::kind};
    // Compute the integer kind value of the type parameter,
    // which may depend on the values of earlier ones.
    if (const auto *typeSpec{symbol.GetType()}) {
      if (const IntrinsicTypeSpec * intrinType{typeSpec->AsIntrinsic()};
          intrinType && intrinType->category() == TypeCategory::Integer) {
        auto restorer{foldingContext.WithPDTInstance(*this)};
        auto folded{Fold(foldingContext, KindExpr{intrinType->kind()})};
        if (auto k{evaluate::ToInt64(folded)}; k &&
            evaluate::IsValidKindOfIntrinsicType(TypeCategory::Integer, *k)) {
          parameterKind = static_cast<int>(*k);
        } else {
          messages.Say(
              "Type of type parameter '%s' (%s) is not a valid kind of INTEGER"_err_en_US,
              name, intrinType->kind().AsFortran());
        }
      }
    }
    bool ok{
        symbol.get<TypeParamDetails>().attr() == common::TypeParamAttr::Len};
    if (ParamValue * paramValue{FindParameter(name)}) {
      // Explicit type parameter value expressions are not folded within
      // the scope of the derived type being instantiated, as the expressions
      // themselves are not in that scope and cannot reference its type
      // parameters.
      if (const MaybeIntExpr & expr{paramValue->GetExplicit()}) {
        evaluate::DynamicType dyType{TypeCategory::Integer, parameterKind};
        if (auto converted{evaluate::ConvertToType(dyType, SomeExpr{*expr})}) {
          SomeExpr folded{
              evaluate::Fold(foldingContext, std::move(*converted))};
          if (auto *intExpr{std::get_if<SomeIntExpr>(&folded.u)}) {
            ok = ok || evaluate::IsActuallyConstant(*intExpr);
            paramValue->SetExplicit(std::move(*intExpr));
          }
        } else if (!context.HasError(symbol)) {
          evaluate::SayWithDeclaration(messages, symbol,
              "Value of type parameter '%s' (%s) is not convertible to its type (%s)"_err_en_US,
              name, expr->AsFortran(), dyType.AsFortran());
        }
      }
    } else {
      // Default type parameter value expressions are folded within
      // the scope of the derived type being instantiated.
      const TypeParamDetails &details{symbol.get<TypeParamDetails>()};
      if (details.init() && details.attr()) {
        evaluate::DynamicType dyType{TypeCategory::Integer, parameterKind};
        if (auto converted{
                evaluate::ConvertToType(dyType, SomeExpr{*details.init()})}) {
          auto restorer{foldingContext.WithPDTInstance(*this)};
          SomeExpr folded{
              evaluate::Fold(foldingContext, std::move(*converted))};
          ok = ok || evaluate::IsActuallyConstant(folded);
          AddParamValue(name,
              ParamValue{std::move(std::get<SomeIntExpr>(folded.u)),
                  details.attr().value()});
        } else {
          if (!context.HasError(symbol)) {
            evaluate::SayWithDeclaration(messages, symbol,
                "Default value of type parameter '%s' (%s) is not convertible to its type (%s)"_err_en_US,
                name, details.init()->AsFortran(), dyType.AsFortran());
          }
        }
      } else if (!context.HasError(symbol)) {
        messages.Say(name_,
            "Type parameter '%s' lacks a value and has no default"_err_en_US,
            name);
      }
    }
    if (!ok && !context.HasError(symbol)) {
      messages.Say(
          "Value of KIND type parameter '%s' must be constant"_err_en_US, name);
    }
  }
}

void DerivedTypeSpec::AddParamValue(SourceName name, ParamValue &&value) {
  CHECK(cooked_);
  auto pair{parameters_.insert(std::make_pair(name, std::move(value)))};
  CHECK(pair.second); // name was not already present
}

bool DerivedTypeSpec::MightBeParameterized() const {
  return !cooked_ || !parameters_.empty();
}

bool DerivedTypeSpec::IsForwardReferenced() const {
  return typeSymbol_.get<DerivedTypeDetails>().isForwardReferenced();
}

bool DerivedTypeSpec::HasDefaultInitialization(
    bool ignoreAllocatable, bool ignorePointer) const {
  DirectComponentIterator components{*this};
  return bool{std::find_if(
      components.begin(), components.end(), [&](const Symbol &component) {
        return IsInitialized(component, /*ignoreDataStatements=*/true,
            ignoreAllocatable, ignorePointer);
      })};
}

bool DerivedTypeSpec::HasDestruction() const {
  if (!FinalsForDerivedTypeInstantiation(*this).empty()) {
    return true;
  }
  DirectComponentIterator components{*this};
  return bool{std::find_if(
      components.begin(), components.end(), [&](const Symbol &component) {
        return IsDestructible(component, &typeSymbol());
      })};
}

ParamValue *DerivedTypeSpec::FindParameter(SourceName target) {
  return const_cast<ParamValue *>(
      const_cast<const DerivedTypeSpec *>(this)->FindParameter(target));
}

static bool MatchKindParams(const Symbol &typeSymbol,
    const DerivedTypeSpec &thisSpec, const DerivedTypeSpec &thatSpec) {
  for (auto ref : typeSymbol.get<DerivedTypeDetails>().paramNameOrder()) {
    if (ref->get<TypeParamDetails>().attr() == common::TypeParamAttr::Kind) {
      const auto *thisValue{thisSpec.FindParameter(ref->name())};
      const auto *thatValue{thatSpec.FindParameter(ref->name())};
      if (!thisValue || !thatValue || *thisValue != *thatValue) {
        return false;
      }
    }
  }
  if (const DerivedTypeSpec *
      parent{typeSymbol.GetParentTypeSpec(typeSymbol.scope())}) {
    return MatchKindParams(parent->typeSymbol(), thisSpec, thatSpec);
  } else {
    return true;
  }
}

bool DerivedTypeSpec::MatchesOrExtends(const DerivedTypeSpec &that) const {
  const Symbol *typeSymbol{&typeSymbol_};
  while (typeSymbol != &that.typeSymbol_) {
    if (const DerivedTypeSpec *
        parent{typeSymbol->GetParentTypeSpec(typeSymbol->scope())}) {
      typeSymbol = &parent->typeSymbol_;
    } else {
      return false;
    }
  }
  return MatchKindParams(*typeSymbol, *this, that);
}

class InstantiateHelper {
public:
  InstantiateHelper(Scope &scope) : scope_{scope} {}
  // Instantiate components from fromScope into scope_
  void InstantiateComponents(const Scope &);

private:
  SemanticsContext &context() const { return scope_.context(); }
  evaluate::FoldingContext &foldingContext() {
    return context().foldingContext();
  }
  template <typename A> A Fold(A &&expr) {
    return evaluate::Fold(foldingContext(), std::move(expr));
  }
  void InstantiateComponent(const Symbol &);
  const DeclTypeSpec *InstantiateType(const Symbol &);
  const DeclTypeSpec &InstantiateIntrinsicType(
      SourceName, const DeclTypeSpec &);
  DerivedTypeSpec CreateDerivedTypeSpec(const DerivedTypeSpec &, bool);

  Scope &scope_;
};

static int PlumbPDTInstantiationDepth(const Scope *scope) {
  int depth{0};
  while (scope->IsParameterizedDerivedTypeInstantiation()) {
    ++depth;
    scope = &scope->parent();
  }
  return depth;
}

// Completes component derived type instantiation and initializer folding
// for a non-parameterized derived type Scope.
static void InstantiateNonPDTScope(Scope &typeScope, Scope &containingScope) {
  auto &context{containingScope.context()};
  auto &foldingContext{context.foldingContext()};
  for (auto &pair : typeScope) {
    Symbol &symbol{*pair.second};
    if (DeclTypeSpec * type{symbol.GetType()}) {
      if (DerivedTypeSpec * derived{type->AsDerived()}) {
        if (!(derived->IsForwardReferenced() &&
                IsAllocatableOrPointer(symbol))) {
          derived->Instantiate(containingScope);
        }
      }
    }
    if (!IsPointer(symbol)) {
      if (auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
        if (MaybeExpr & init{object->init()}) {
          auto restorer{foldingContext.messages().SetLocation(symbol.name())};
          init = evaluate::NonPointerInitializationExpr(
              symbol, std::move(*init), foldingContext);
        }
      }
    }
  }
  ComputeOffsets(context, typeScope);
}

void DerivedTypeSpec::Instantiate(Scope &containingScope) {
  if (instantiated_) {
    return;
  }
  instantiated_ = true;
  auto &context{containingScope.context()};
  auto &foldingContext{context.foldingContext()};
  if (IsForwardReferenced()) {
    foldingContext.messages().Say(typeSymbol_.name(),
        "The derived type '%s' was forward-referenced but not defined"_err_en_US,
        typeSymbol_.name());
    context.SetError(typeSymbol_);
    return;
  }
  EvaluateParameters(context);
  const Scope &typeScope{DEREF(typeSymbol_.scope())};
  if (!MightBeParameterized()) {
    scope_ = &typeScope;
    if (typeScope.derivedTypeSpec()) {
      CHECK(*this == *typeScope.derivedTypeSpec());
    } else {
      Scope &mutableTypeScope{const_cast<Scope &>(typeScope)};
      mutableTypeScope.set_derivedTypeSpec(*this);
      InstantiateNonPDTScope(mutableTypeScope, containingScope);
    }
    return;
  }
  // New PDT instantiation.  Create a new scope and populate it
  // with components that have been specialized for this set of
  // parameters.
  Scope &newScope{containingScope.MakeScope(Scope::Kind::DerivedType)};
  newScope.set_derivedTypeSpec(*this);
  ReplaceScope(newScope);
  auto restorer{foldingContext.WithPDTInstance(*this)};
  std::string desc{typeSymbol_.name().ToString()};
  char sep{'('};
  for (const Symbol &symbol : OrderParameterDeclarations(typeSymbol_)) {
    const SourceName &name{symbol.name()};
    if (typeScope.find(symbol.name()) != typeScope.end()) {
      // This type parameter belongs to the derived type itself, not to
      // one of its ancestors.  Put the type parameter expression value,
      // when there is one, into the new scope as the initialization value
      // for the parameter.  And when there is no explicit value, add an
      // uninitialized type parameter to forestall use of any default.
      if (ParamValue * paramValue{FindParameter(name)}) {
        const TypeParamDetails &details{symbol.get<TypeParamDetails>()};
        TypeParamDetails instanceDetails{};
        if (details.attr()) {
          paramValue->set_attr(*details.attr());
          instanceDetails.set_attr(*details.attr());
        }
        desc += sep;
        desc += name.ToString();
        desc += '=';
        sep = ',';
        if (MaybeIntExpr expr{paramValue->GetExplicit()}) {
          desc += expr->AsFortran();
          instanceDetails.set_init(
              std::move(DEREF(evaluate::UnwrapExpr<SomeIntExpr>(*expr))));
          if (auto dyType{expr->GetType()}) {
            instanceDetails.set_type(newScope.MakeNumericType(
                TypeCategory::Integer, KindExpr{dyType->kind()}));
          }
        }
        if (!instanceDetails.type()) {
          if (const DeclTypeSpec * type{details.type()}) {
            instanceDetails.set_type(*type);
          }
        }
        if (!instanceDetails.init()) {
          desc += '*';
        }
        newScope.try_emplace(name, std::move(instanceDetails));
      }
    }
  }
  parser::Message *contextMessage{nullptr};
  if (sep != '(') {
    desc += ')';
    contextMessage = new parser::Message{foldingContext.messages().at(),
        "instantiation of parameterized derived type '%s'"_en_US, desc};
    if (auto outer{containingScope.instantiationContext()}) {
      contextMessage->SetContext(outer.get());
    }
    newScope.set_instantiationContext(contextMessage);
  }
  // Instantiate nearly every non-parameter symbol from the original derived
  // type's scope into the new instance.
  auto restorer2{foldingContext.messages().SetContext(contextMessage)};
  if (PlumbPDTInstantiationDepth(&containingScope) > 100) {
    foldingContext.messages().Say(
        "Too many recursive parameterized derived type instantiations"_err_en_US);
  } else {
    InstantiateHelper{newScope}.InstantiateComponents(typeScope);
  }
}

void InstantiateHelper::InstantiateComponents(const Scope &fromScope) {
  // Instantiate symbols in declaration order; this ensures that
  // parent components and type parameters of ancestor types exist
  // by the time that they're needed.
  for (SymbolRef ref : fromScope.GetSymbols()) {
    InstantiateComponent(*ref);
  }
  ComputeOffsets(context(), scope_);
}

// Walks a parsed expression to prepare it for (re)analysis;
// clears out the typedExpr analysis results and re-resolves
// symbol table pointers of type parameters.
class ComponentInitResetHelper {
public:
  explicit ComponentInitResetHelper(Scope &scope) : scope_{scope} {}

  template <typename A> bool Pre(const A &) { return true; }

  template <typename A> void Post(const A &x) {
    if constexpr (parser::HasTypedExpr<A>()) {
      x.typedExpr.Reset();
    }
  }

  void Post(const parser::Name &name) {
    if (name.symbol && name.symbol->has<TypeParamDetails>()) {
      name.symbol = scope_.FindComponent(name.source);
    }
  }

private:
  Scope &scope_;
};

void InstantiateHelper::InstantiateComponent(const Symbol &oldSymbol) {
  auto pair{scope_.try_emplace(
      oldSymbol.name(), oldSymbol.attrs(), common::Clone(oldSymbol.details()))};
  Symbol &newSymbol{*pair.first->second};
  if (!pair.second) {
    // Symbol was already present in the scope, which can only happen
    // in the case of type parameters.
    CHECK(oldSymbol.has<TypeParamDetails>());
    return;
  }
  newSymbol.flags() = oldSymbol.flags();
  if (auto *details{newSymbol.detailsIf<ObjectEntityDetails>()}) {
    if (const DeclTypeSpec * newType{InstantiateType(newSymbol)}) {
      details->ReplaceType(*newType);
    }
    for (ShapeSpec &dim : details->shape()) {
      if (dim.lbound().isExplicit()) {
        dim.lbound().SetExplicit(Fold(std::move(dim.lbound().GetExplicit())));
      }
      if (dim.ubound().isExplicit()) {
        dim.ubound().SetExplicit(Fold(std::move(dim.ubound().GetExplicit())));
      }
    }
    for (ShapeSpec &dim : details->coshape()) {
      if (dim.lbound().isExplicit()) {
        dim.lbound().SetExplicit(Fold(std::move(dim.lbound().GetExplicit())));
      }
      if (dim.ubound().isExplicit()) {
        dim.ubound().SetExplicit(Fold(std::move(dim.ubound().GetExplicit())));
      }
    }
    if (const auto *parsedExpr{details->unanalyzedPDTComponentInit()}) {
      // Analyze the parsed expression in this PDT instantiation context.
      ComponentInitResetHelper resetter{scope_};
      parser::Walk(*parsedExpr, resetter);
      auto restorer{foldingContext().messages().SetLocation(newSymbol.name())};
      details->set_init(evaluate::Fold(
          foldingContext(), AnalyzeExpr(context(), *parsedExpr)));
      details->set_unanalyzedPDTComponentInit(nullptr);
      // Remove analysis results to prevent unparsing or other use of
      // instantiation-specific expressions.
      parser::Walk(*parsedExpr, resetter);
    }
    if (MaybeExpr & init{details->init()}) {
      // Non-pointer components with default initializers are
      // processed now so that those default initializers can be used
      // in PARAMETER structure constructors.
      auto restorer{foldingContext().messages().SetLocation(newSymbol.name())};
      init = IsPointer(newSymbol)
          ? Fold(std::move(*init))
          : evaluate::NonPointerInitializationExpr(
                newSymbol, std::move(*init), foldingContext());
    }
  } else if (auto *procDetails{newSymbol.detailsIf<ProcEntityDetails>()}) {
    // We have a procedure pointer.  Instantiate its return type
    if (const DeclTypeSpec * returnType{InstantiateType(newSymbol)}) {
      if (!procDetails->procInterface()) {
        procDetails->ReplaceType(*returnType);
      }
    }
  }
}

const DeclTypeSpec *InstantiateHelper::InstantiateType(const Symbol &symbol) {
  const DeclTypeSpec *type{symbol.GetType()};
  if (!type) {
    return nullptr; // error has occurred
  } else if (const DerivedTypeSpec * spec{type->AsDerived()}) {
    return &FindOrInstantiateDerivedType(scope_,
        CreateDerivedTypeSpec(*spec, symbol.test(Symbol::Flag::ParentComp)),
        type->category());
  } else if (type->AsIntrinsic()) {
    return &InstantiateIntrinsicType(symbol.name(), *type);
  } else if (type->category() == DeclTypeSpec::ClassStar) {
    return type;
  } else {
    common::die("InstantiateType: %s", type->AsFortran().c_str());
  }
}

/// Fold explicit length parameters of character components when the explicit
/// expression is a constant expression (if it only depends on KIND parameters).
/// Do not fold `character(len=pdt_length)`, even if the length parameter is
/// constant in the pdt instantiation, in order to avoid losing the information
/// that the character component is automatic (and must be a descriptor).
static ParamValue FoldCharacterLength(evaluate::FoldingContext &foldingContext,
    const CharacterTypeSpec &characterSpec) {
  if (const auto &len{characterSpec.length().GetExplicit()}) {
    if (evaluate::IsConstantExpr(*len)) {
      return ParamValue{evaluate::Fold(foldingContext, common::Clone(*len)),
          common::TypeParamAttr::Len};
    }
  }
  return characterSpec.length();
}

// Apply type parameter values to an intrinsic type spec.
const DeclTypeSpec &InstantiateHelper::InstantiateIntrinsicType(
    SourceName symbolName, const DeclTypeSpec &spec) {
  const IntrinsicTypeSpec &intrinsic{DEREF(spec.AsIntrinsic())};
  if (spec.category() != DeclTypeSpec::Character &&
      evaluate::IsActuallyConstant(intrinsic.kind())) {
    return spec; // KIND is already a known constant
  }
  // The expression was not originally constant, but now it must be so
  // in the context of a parameterized derived type instantiation.
  KindExpr copy{Fold(common::Clone(intrinsic.kind()))};
  int kind{context().GetDefaultKind(intrinsic.category())};
  if (auto value{evaluate::ToInt64(copy)}) {
    if (foldingContext().targetCharacteristics().IsTypeEnabled(
            intrinsic.category(), *value)) {
      kind = *value;
    } else {
      foldingContext().messages().Say(symbolName,
          "KIND parameter value (%jd) of intrinsic type %s "
          "did not resolve to a supported value"_err_en_US,
          *value,
          parser::ToUpperCaseLetters(EnumToString(intrinsic.category())));
    }
  }
  switch (spec.category()) {
  case DeclTypeSpec::Numeric:
    return scope_.MakeNumericType(intrinsic.category(), KindExpr{kind});
  case DeclTypeSpec::Logical:
    return scope_.MakeLogicalType(KindExpr{kind});
  case DeclTypeSpec::Character:
    return scope_.MakeCharacterType(
        FoldCharacterLength(foldingContext(), spec.characterTypeSpec()),
        KindExpr{kind});
  default:
    CRASH_NO_CASE;
  }
}

DerivedTypeSpec InstantiateHelper::CreateDerivedTypeSpec(
    const DerivedTypeSpec &spec, bool isParentComp) {
  DerivedTypeSpec result{spec};
  result.CookParameters(foldingContext()); // enables AddParamValue()
  if (isParentComp) {
    // Forward any explicit type parameter values from the
    // derived type spec under instantiation that define type parameters
    // of the parent component to the derived type spec of the
    // parent component.
    const DerivedTypeSpec &instanceSpec{DEREF(foldingContext().pdtInstance())};
    for (const auto &[name, value] : instanceSpec.parameters()) {
      if (scope_.find(name) == scope_.end()) {
        result.AddParamValue(name, ParamValue{value});
      }
    }
  }
  return result;
}

std::string DerivedTypeSpec::VectorTypeAsFortran() const {
  std::string buf;
  llvm::raw_string_ostream ss{buf};

  switch (category()) {
    SWITCH_COVERS_ALL_CASES
  case (Fortran::semantics::DerivedTypeSpec::Category::IntrinsicVector): {
    int64_t vecElemKind;
    int64_t vecElemCategory;

    for (const auto &pair : parameters()) {
      if (pair.first == "element_category") {
        vecElemCategory =
            Fortran::evaluate::ToInt64(pair.second.GetExplicit()).value_or(-1);
      } else if (pair.first == "element_kind") {
        vecElemKind =
            Fortran::evaluate::ToInt64(pair.second.GetExplicit()).value_or(0);
      }
    }

    assert((vecElemCategory >= 0 &&
               static_cast<size_t>(vecElemCategory) <
                   Fortran::common::VectorElementCategory_enumSize) &&
        "Vector element type is not specified");
    assert(vecElemKind && "Vector element kind is not specified");

    ss << "vector(";
    switch (static_cast<common::VectorElementCategory>(vecElemCategory)) {
      SWITCH_COVERS_ALL_CASES
    case common::VectorElementCategory::Integer:
      ss << "integer(" << vecElemKind << ")";
      break;
    case common::VectorElementCategory::Unsigned:
      ss << "unsigned(" << vecElemKind << ")";
      break;
    case common::VectorElementCategory::Real:
      ss << "real(" << vecElemKind << ")";
      break;
    }
    ss << ")";
    break;
  }
  case (Fortran::semantics::DerivedTypeSpec::Category::PairVector):
    ss << "__vector_pair";
    break;
  case (Fortran::semantics::DerivedTypeSpec::Category::QuadVector):
    ss << "__vector_quad";
    break;
  case (Fortran::semantics::DerivedTypeSpec::Category::DerivedType):
    Fortran::common::die("Vector element type not implemented");
  }
  return buf;
}

std::string DerivedTypeSpec::AsFortran() const {
  std::string buf;
  llvm::raw_string_ostream ss{buf};
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
  return buf;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const DerivedTypeSpec &x) {
  return o << x.AsFortran();
}

Bound::Bound(common::ConstantSubscript bound) : expr_{bound} {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const Bound &x) {
  if (x.isStar()) {
    o << '*';
  } else if (x.isColon()) {
    o << ':';
  } else if (x.expr_) {
    x.expr_->AsFortran(o);
  } else {
    o << "<no-expr>";
  }
  return o;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const ShapeSpec &x) {
  if (x.lb_.isStar()) {
    CHECK(x.ub_.isStar());
    o << "..";
  } else {
    if (!x.lb_.isColon()) {
      o << x.lb_;
    }
    o << ':';
    if (!x.ub_.isColon()) {
      o << x.ub_;
    }
  }
  return o;
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const ArraySpec &arraySpec) {
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
    : ParamValue(SomeIntExpr{evaluate::Expr<evaluate::SubscriptInteger>{value}},
          attr) {}

void ParamValue::SetExplicit(SomeIntExpr &&x) {
  category_ = Category::Explicit;
  expr_ = std::move(x);
}

std::string ParamValue::AsFortran() const {
  switch (category_) {
    SWITCH_COVERS_ALL_CASES
  case Category::Assumed:
    return "*";
  case Category::Deferred:
    return ":";
  case Category::Explicit:
    if (expr_) {
      std::string buf;
      llvm::raw_string_ostream ss{buf};
      expr_->AsFortran(ss);
      return buf;
    } else {
      return "";
    }
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const ParamValue &x) {
  return o << x.AsFortran();
}

IntrinsicTypeSpec::IntrinsicTypeSpec(TypeCategory category, KindExpr &&kind)
    : category_{category}, kind_{std::move(kind)} {
  CHECK(category != TypeCategory::Derived);
}

static std::string KindAsFortran(const KindExpr &kind) {
  std::string buf;
  llvm::raw_string_ostream ss{buf};
  if (auto k{evaluate::ToInt64(kind)}) {
    ss << *k; // emit unsuffixed kind code
  } else {
    kind.AsFortran(ss);
  }
  return buf;
}

std::string IntrinsicTypeSpec::AsFortran() const {
  return parser::ToUpperCaseLetters(common::EnumToString(category_)) + '(' +
      KindAsFortran(kind_) + ')';
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const IntrinsicTypeSpec &x) {
  return os << x.AsFortran();
}

std::string CharacterTypeSpec::AsFortran() const {
  return "CHARACTER(" + length_.AsFortran() + ',' + KindAsFortran(kind()) + ')';
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const CharacterTypeSpec &x) {
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
bool DeclTypeSpec::IsSequenceType() const {
  if (const DerivedTypeSpec * derivedType{AsDerived()}) {
    const auto *typeDetails{
        derivedType->typeSymbol().detailsIf<DerivedTypeDetails>()};
    return typeDetails && typeDetails->sequence();
  }
  return false;
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
  case Numeric:
    return numericTypeSpec().AsFortran();
  case Logical:
    return logicalTypeSpec().AsFortran();
  case Character:
    return characterTypeSpec().AsFortran();
  case TypeDerived:
    if (derivedTypeSpec()
            .typeSymbol()
            .get<DerivedTypeDetails>()
            .isDECStructure()) {
      return "RECORD" + derivedTypeSpec().typeSymbol().name().ToString();
    } else if (derivedTypeSpec().IsVectorType()) {
      return derivedTypeSpec().VectorTypeAsFortran();
    } else {
      return "TYPE(" + derivedTypeSpec().AsFortran() + ')';
    }
  case ClassDerived:
    return "CLASS(" + derivedTypeSpec().AsFortran() + ')';
  case TypeStar:
    return "TYPE(*)";
  case ClassStar:
    return "CLASS(*)";
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const DeclTypeSpec &x) {
  return o << x.AsFortran();
}

} // namespace Fortran::semantics
