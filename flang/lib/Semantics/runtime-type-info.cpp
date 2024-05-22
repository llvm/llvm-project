//===-- lib/Semantics/runtime-type-info.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/runtime-type-info.h"
#include "mod-file.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/tools.h"
#include <functional>
#include <list>
#include <map>
#include <string>

// The symbols added by this code to various scopes in the program include:
//   .b.TYPE.NAME  - Bounds values for an array component
//   .c.TYPE       - TYPE(Component) descriptions for TYPE
//   .di.TYPE.NAME - Data initialization for a component
//   .dp.TYPE.NAME - Data pointer initialization for a component
//   .dt.TYPE      - TYPE(DerivedType) description for TYPE
//   .kp.TYPE      - KIND type parameter values for TYPE
//   .lpk.TYPE     - Integer kinds of LEN type parameter values
//   .lv.TYPE.NAME - LEN type parameter values for a component's type
//   .n.NAME       - Character representation of a name
//   .p.TYPE       - TYPE(ProcPtrComponent) descriptions for TYPE
//   .s.TYPE       - TYPE(SpecialBinding) bindings for TYPE
//   .v.TYPE       - TYPE(Binding) bindings for TYPE

namespace Fortran::semantics {

static int FindLenParameterIndex(
    const SymbolVector &parameters, const Symbol &symbol) {
  int lenIndex{0};
  for (SymbolRef ref : parameters) {
    if (&*ref == &symbol) {
      return lenIndex;
    }
    if (ref->get<TypeParamDetails>().attr() == common::TypeParamAttr::Len) {
      ++lenIndex;
    }
  }
  DIE("Length type parameter not found in parameter order");
  return -1;
}

class RuntimeTableBuilder {
public:
  RuntimeTableBuilder(SemanticsContext &, RuntimeDerivedTypeTables &);
  void DescribeTypes(Scope &scope, bool inSchemata);

private:
  const Symbol *DescribeType(Scope &);
  const Symbol &GetSchemaSymbol(const char *) const;
  const DeclTypeSpec &GetSchema(const char *) const;
  SomeExpr GetEnumValue(const char *) const;
  Symbol &CreateObject(const std::string &, const DeclTypeSpec &, Scope &);
  // The names of created symbols are saved in and owned by the
  // RuntimeDerivedTypeTables instance returned by
  // BuildRuntimeDerivedTypeTables() so that references to those names remain
  // valid for lowering.
  SourceName SaveObjectName(const std::string &);
  SomeExpr SaveNameAsPointerTarget(Scope &, const std::string &);
  const SymbolVector *GetTypeParameters(const Symbol &);
  evaluate::StructureConstructor DescribeComponent(const Symbol &,
      const ObjectEntityDetails &, Scope &, Scope &,
      const std::string &distinctName, const SymbolVector *parameters);
  evaluate::StructureConstructor DescribeComponent(
      const Symbol &, const ProcEntityDetails &, Scope &);
  bool InitializeDataPointer(evaluate::StructureConstructorValues &,
      const Symbol &symbol, const ObjectEntityDetails &object, Scope &scope,
      Scope &dtScope, const std::string &distinctName);
  evaluate::StructureConstructor PackageIntValue(
      const SomeExpr &genre, std::int64_t = 0) const;
  SomeExpr PackageIntValueExpr(const SomeExpr &genre, std::int64_t = 0) const;
  std::vector<evaluate::StructureConstructor> DescribeBindings(
      const Scope &dtScope, Scope &);
  std::map<int, evaluate::StructureConstructor> DescribeSpecialGenerics(
      const Scope &dtScope, const Scope &thisScope,
      const DerivedTypeSpec *) const;
  void DescribeSpecialGeneric(const GenericDetails &,
      std::map<int, evaluate::StructureConstructor> &, const Scope &,
      const DerivedTypeSpec *) const;
  void DescribeSpecialProc(std::map<int, evaluate::StructureConstructor> &,
      const Symbol &specificOrBinding, bool isAssignment, bool isFinal,
      std::optional<common::DefinedIo>, const Scope *, const DerivedTypeSpec *,
      bool isTypeBound) const;
  void IncorporateDefinedIoGenericInterfaces(
      std::map<int, evaluate::StructureConstructor> &, common::DefinedIo,
      const Scope *, const DerivedTypeSpec *);

  // Instantiated for ParamValue and Bound
  template <typename A>
  evaluate::StructureConstructor GetValue(
      const A &x, const SymbolVector *parameters) {
    if (x.isExplicit()) {
      return GetValue(x.GetExplicit(), parameters);
    } else {
      return PackageIntValue(deferredEnum_);
    }
  }

  // Specialization for optional<Expr<SomeInteger and SubscriptInteger>>
  template <typename T>
  evaluate::StructureConstructor GetValue(
      const std::optional<evaluate::Expr<T>> &expr,
      const SymbolVector *parameters) {
    if (auto constValue{evaluate::ToInt64(expr)}) {
      return PackageIntValue(explicitEnum_, *constValue);
    }
    if (expr) {
      if (parameters) {
        if (const Symbol * lenParam{evaluate::ExtractBareLenParameter(*expr)}) {
          return PackageIntValue(
              lenParameterEnum_, FindLenParameterIndex(*parameters, *lenParam));
        }
      }
      // TODO: Replace a specification expression requiring actual operations
      // with a reference to a new anonymous LEN type parameter whose default
      // value captures the expression.  This replacement must take place when
      // the type is declared so that the new LEN type parameters appear in
      // all instantiations and structure constructors.
      context_.Say(location_,
          "derived type specification expression '%s' that is neither constant nor a length type parameter"_todo_en_US,
          expr->AsFortran());
    }
    return PackageIntValue(deferredEnum_);
  }

  SemanticsContext &context_;
  RuntimeDerivedTypeTables &tables_;
  std::map<const Symbol *, SymbolVector> orderedTypeParameters_;

  const DeclTypeSpec &derivedTypeSchema_; // TYPE(DerivedType)
  const DeclTypeSpec &componentSchema_; // TYPE(Component)
  const DeclTypeSpec &procPtrSchema_; // TYPE(ProcPtrComponent)
  const DeclTypeSpec &valueSchema_; // TYPE(Value)
  const DeclTypeSpec &bindingSchema_; // TYPE(Binding)
  const DeclTypeSpec &specialSchema_; // TYPE(SpecialBinding)
  SomeExpr deferredEnum_; // Value::Genre::Deferred
  SomeExpr explicitEnum_; // Value::Genre::Explicit
  SomeExpr lenParameterEnum_; // Value::Genre::LenParameter
  SomeExpr scalarAssignmentEnum_; // SpecialBinding::Which::ScalarAssignment
  SomeExpr
      elementalAssignmentEnum_; // SpecialBinding::Which::ElementalAssignment
  SomeExpr readFormattedEnum_; // SpecialBinding::Which::ReadFormatted
  SomeExpr readUnformattedEnum_; // SpecialBinding::Which::ReadUnformatted
  SomeExpr writeFormattedEnum_; // SpecialBinding::Which::WriteFormatted
  SomeExpr writeUnformattedEnum_; // SpecialBinding::Which::WriteUnformatted
  SomeExpr elementalFinalEnum_; // SpecialBinding::Which::ElementalFinal
  SomeExpr assumedRankFinalEnum_; // SpecialBinding::Which::AssumedRankFinal
  SomeExpr scalarFinalEnum_; // SpecialBinding::Which::ScalarFinal
  parser::CharBlock location_;
  std::set<const Scope *> ignoreScopes_;
};

RuntimeTableBuilder::RuntimeTableBuilder(
    SemanticsContext &c, RuntimeDerivedTypeTables &t)
    : context_{c}, tables_{t}, derivedTypeSchema_{GetSchema("derivedtype")},
      componentSchema_{GetSchema("component")},
      procPtrSchema_{GetSchema("procptrcomponent")},
      valueSchema_{GetSchema("value")},
      bindingSchema_{GetSchema(bindingDescCompName)},
      specialSchema_{GetSchema("specialbinding")},
      deferredEnum_{GetEnumValue("deferred")},
      explicitEnum_{GetEnumValue("explicit")},
      lenParameterEnum_{GetEnumValue("lenparameter")},
      scalarAssignmentEnum_{GetEnumValue("scalarassignment")},
      elementalAssignmentEnum_{GetEnumValue("elementalassignment")},
      readFormattedEnum_{GetEnumValue("readformatted")},
      readUnformattedEnum_{GetEnumValue("readunformatted")},
      writeFormattedEnum_{GetEnumValue("writeformatted")},
      writeUnformattedEnum_{GetEnumValue("writeunformatted")},
      elementalFinalEnum_{GetEnumValue("elementalfinal")},
      assumedRankFinalEnum_{GetEnumValue("assumedrankfinal")},
      scalarFinalEnum_{GetEnumValue("scalarfinal")} {
  ignoreScopes_.insert(tables_.schemata);
}

static void SetReadOnlyCompilerCreatedFlags(Symbol &symbol) {
  symbol.set(Symbol::Flag::CompilerCreated);
  // Runtime type info symbols may have types that are incompatible with the
  // PARAMETER attribute (the main issue is that they may be TARGET, and normal
  // Fortran parameters cannot be TARGETs).
  if (symbol.has<semantics::ObjectEntityDetails>() ||
      symbol.has<semantics::ProcEntityDetails>()) {
    symbol.set(Symbol::Flag::ReadOnly);
  }
}

// Save an arbitrarily shaped array constant of some derived type
// as an initialized data object in a scope.
static SomeExpr SaveDerivedPointerTarget(Scope &scope, SourceName name,
    std::vector<evaluate::StructureConstructor> &&x,
    evaluate::ConstantSubscripts &&shape) {
  if (x.empty()) {
    return SomeExpr{evaluate::NullPointer{}};
  } else {
    auto dyType{x.front().GetType()};
    const auto &derivedType{dyType.GetDerivedTypeSpec()};
    ObjectEntityDetails object;
    DeclTypeSpec typeSpec{DeclTypeSpec::TypeDerived, derivedType};
    if (const DeclTypeSpec * spec{scope.FindType(typeSpec)}) {
      object.set_type(*spec);
    } else {
      object.set_type(scope.MakeDerivedType(
          DeclTypeSpec::TypeDerived, common::Clone(derivedType)));
    }
    if (!shape.empty()) {
      ArraySpec arraySpec;
      for (auto n : shape) {
        arraySpec.push_back(ShapeSpec::MakeExplicit(Bound{0}, Bound{n - 1}));
      }
      object.set_shape(arraySpec);
    }
    object.set_init(
        evaluate::AsGenericExpr(evaluate::Constant<evaluate::SomeDerived>{
            derivedType, std::move(x), std::move(shape)}));
    Symbol &symbol{*scope
                        .try_emplace(name, Attrs{Attr::TARGET, Attr::SAVE},
                            std::move(object))
                        .first->second};
    SetReadOnlyCompilerCreatedFlags(symbol);
    return evaluate::AsGenericExpr(
        evaluate::Designator<evaluate::SomeDerived>{symbol});
  }
}

void RuntimeTableBuilder::DescribeTypes(Scope &scope, bool inSchemata) {
  inSchemata |= ignoreScopes_.find(&scope) != ignoreScopes_.end();
  if (scope.IsDerivedType()) {
    if (!inSchemata) { // don't loop trying to describe a schema
      DescribeType(scope);
    }
  } else {
    scope.InstantiateDerivedTypes();
  }
  for (Scope &child : scope.children()) {
    DescribeTypes(child, inSchemata);
  }
}

// Returns derived type instantiation's parameters in declaration order
const SymbolVector *RuntimeTableBuilder::GetTypeParameters(
    const Symbol &symbol) {
  auto iter{orderedTypeParameters_.find(&symbol)};
  if (iter != orderedTypeParameters_.end()) {
    return &iter->second;
  } else {
    return &orderedTypeParameters_
                .emplace(&symbol, OrderParameterDeclarations(symbol))
                .first->second;
  }
}

static Scope &GetContainingNonDerivedScope(Scope &scope) {
  Scope *p{&scope};
  while (p->IsDerivedType()) {
    p = &p->parent();
  }
  return *p;
}

static const Symbol &GetSchemaField(
    const DerivedTypeSpec &derived, const std::string &name) {
  const Scope &scope{
      DEREF(derived.scope() ? derived.scope() : derived.typeSymbol().scope())};
  auto iter{scope.find(SourceName(name))};
  CHECK(iter != scope.end());
  return *iter->second;
}

static const Symbol &GetSchemaField(
    const DeclTypeSpec &derived, const std::string &name) {
  return GetSchemaField(DEREF(derived.AsDerived()), name);
}

static evaluate::StructureConstructorValues &AddValue(
    evaluate::StructureConstructorValues &values, const DeclTypeSpec &spec,
    const std::string &name, SomeExpr &&x) {
  values.emplace(GetSchemaField(spec, name), std::move(x));
  return values;
}

static evaluate::StructureConstructorValues &AddValue(
    evaluate::StructureConstructorValues &values, const DeclTypeSpec &spec,
    const std::string &name, const SomeExpr &x) {
  values.emplace(GetSchemaField(spec, name), x);
  return values;
}

static SomeExpr IntToExpr(std::int64_t n) {
  return evaluate::AsGenericExpr(evaluate::ExtentExpr{n});
}

static evaluate::StructureConstructor Structure(
    const DeclTypeSpec &spec, evaluate::StructureConstructorValues &&values) {
  return {DEREF(spec.AsDerived()), std::move(values)};
}

static SomeExpr StructureExpr(evaluate::StructureConstructor &&x) {
  return SomeExpr{evaluate::Expr<evaluate::SomeDerived>{std::move(x)}};
}

static int GetIntegerKind(const Symbol &symbol) {
  auto dyType{evaluate::DynamicType::From(symbol)};
  CHECK((dyType && dyType->category() == TypeCategory::Integer) ||
      symbol.owner().context().HasError(symbol));
  return dyType && dyType->category() == TypeCategory::Integer
      ? dyType->kind()
      : symbol.owner().context().GetDefaultKind(TypeCategory::Integer);
}

// Save a rank-1 array constant of some numeric type as an
// initialized data object in a scope.
template <typename T>
static SomeExpr SaveNumericPointerTarget(
    Scope &scope, SourceName name, std::vector<typename T::Scalar> &&x) {
  if (x.empty()) {
    return SomeExpr{evaluate::NullPointer{}};
  } else {
    ObjectEntityDetails object;
    if (const auto *spec{scope.FindType(
            DeclTypeSpec{NumericTypeSpec{T::category, KindExpr{T::kind}}})}) {
      object.set_type(*spec);
    } else {
      object.set_type(scope.MakeNumericType(T::category, KindExpr{T::kind}));
    }
    auto elements{static_cast<evaluate::ConstantSubscript>(x.size())};
    ArraySpec arraySpec;
    arraySpec.push_back(ShapeSpec::MakeExplicit(Bound{0}, Bound{elements - 1}));
    object.set_shape(arraySpec);
    object.set_init(evaluate::AsGenericExpr(evaluate::Constant<T>{
        std::move(x), evaluate::ConstantSubscripts{elements}}));
    Symbol &symbol{*scope
                        .try_emplace(name, Attrs{Attr::TARGET, Attr::SAVE},
                            std::move(object))
                        .first->second};
    SetReadOnlyCompilerCreatedFlags(symbol);
    return evaluate::AsGenericExpr(
        evaluate::Expr<T>{evaluate::Designator<T>{symbol}});
  }
}

static SomeExpr SaveObjectInit(
    Scope &scope, SourceName name, const ObjectEntityDetails &object) {
  Symbol &symbol{*scope
                      .try_emplace(name, Attrs{Attr::TARGET, Attr::SAVE},
                          ObjectEntityDetails{object})
                      .first->second};
  CHECK(symbol.get<ObjectEntityDetails>().init().has_value());
  SetReadOnlyCompilerCreatedFlags(symbol);
  return evaluate::AsGenericExpr(
      evaluate::Designator<evaluate::SomeDerived>{symbol});
}

template <int KIND> static SomeExpr IntExpr(std::int64_t n) {
  return evaluate::AsGenericExpr(
      evaluate::Constant<evaluate::Type<TypeCategory::Integer, KIND>>{n});
}

static std::optional<std::string> GetSuffixIfTypeKindParameters(
    const DerivedTypeSpec &derivedTypeSpec, const SymbolVector *parameters) {
  if (parameters) {
    std::optional<std::string> suffix;
    for (SymbolRef ref : *parameters) {
      const auto &tpd{ref->get<TypeParamDetails>()};
      if (tpd.attr() == common::TypeParamAttr::Kind) {
        if (const auto *pv{derivedTypeSpec.FindParameter(ref->name())}) {
          if (pv->GetExplicit()) {
            if (auto instantiatedValue{evaluate::ToInt64(*pv->GetExplicit())}) {
              if (suffix.has_value()) {
                *suffix += "."s + std::to_string(*instantiatedValue);
              } else {
                suffix = "."s + std::to_string(*instantiatedValue);
              }
            }
          }
        }
      }
    }
    return suffix;
  }
  return std::nullopt;
}

const Symbol *RuntimeTableBuilder::DescribeType(Scope &dtScope) {
  if (const Symbol * info{dtScope.runtimeDerivedTypeDescription()}) {
    return info;
  }
  const DerivedTypeSpec *derivedTypeSpec{dtScope.derivedTypeSpec()};
  if (!derivedTypeSpec && !dtScope.IsDerivedTypeWithKindParameter() &&
      dtScope.symbol()) {
    // This derived type was declared (obviously, there's a Scope) but never
    // used in this compilation (no instantiated DerivedTypeSpec points here).
    // Create a DerivedTypeSpec now for it so that ComponentIterator
    // will work. This covers the case of a derived type that's declared in
    // a module but used only by clients and submodules, enabling the
    // run-time "no initialization needed here" flag to work.
    DerivedTypeSpec derived{dtScope.symbol()->name(), *dtScope.symbol()};
    if (const SymbolVector *
        lenParameters{GetTypeParameters(*dtScope.symbol())}) {
      // Create dummy deferred values for the length parameters so that the
      // DerivedTypeSpec is complete and can be used in helpers.
      for (SymbolRef lenParam : *lenParameters) {
        (void)lenParam;
        derived.AddRawParamValue(
            nullptr, ParamValue::Deferred(common::TypeParamAttr::Len));
      }
      derived.CookParameters(context_.foldingContext());
    }
    DeclTypeSpec &decl{
        dtScope.MakeDerivedType(DeclTypeSpec::TypeDerived, std::move(derived))};
    derivedTypeSpec = &decl.derivedTypeSpec();
  }
  const Symbol *dtSymbol{
      derivedTypeSpec ? &derivedTypeSpec->typeSymbol() : dtScope.symbol()};
  if (!dtSymbol) {
    return nullptr;
  }
  auto locationRestorer{common::ScopedSet(location_, dtSymbol->name())};
  // Check for an existing description that can be imported from a USE'd module
  std::string typeName{dtSymbol->name().ToString()};
  if (typeName.empty() ||
      (typeName.front() == '.' && !context_.IsTempName(typeName))) {
    return nullptr;
  }
  bool isPDTDefinitionWithKindParameters{
      !derivedTypeSpec && dtScope.IsDerivedTypeWithKindParameter()};
  bool isPDTInstantiation{derivedTypeSpec && &dtScope != dtSymbol->scope()};
  const SymbolVector *parameters{GetTypeParameters(*dtSymbol)};
  std::string distinctName{typeName};
  if (isPDTInstantiation) {
    // Only create new type descriptions for different kind parameter values.
    // Type with different length parameters/same kind parameters can all
    // share the same type description available in the current scope.
    if (auto suffix{
            GetSuffixIfTypeKindParameters(*derivedTypeSpec, parameters)}) {
      distinctName += *suffix;
    }
  } else if (isPDTDefinitionWithKindParameters) {
    return nullptr;
  }
  std::string dtDescName{".dt."s + distinctName};
  Scope *dtSymbolScope{const_cast<Scope *>(dtSymbol->scope())};
  Scope &scope{
      GetContainingNonDerivedScope(dtSymbolScope ? *dtSymbolScope : dtScope)};
  if (const auto it{scope.find(SourceName{dtDescName})}; it != scope.end()) {
    dtScope.set_runtimeDerivedTypeDescription(*it->second);
    return &*it->second;
  }

  // Create a new description object before populating it so that mutual
  // references will work as pointer targets.
  Symbol &dtObject{CreateObject(dtDescName, derivedTypeSchema_, scope)};
  dtScope.set_runtimeDerivedTypeDescription(dtObject);
  evaluate::StructureConstructorValues dtValues;
  AddValue(dtValues, derivedTypeSchema_, "name"s,
      SaveNameAsPointerTarget(scope, typeName));
  if (!isPDTDefinitionWithKindParameters) {
    auto sizeInBytes{static_cast<common::ConstantSubscript>(dtScope.size())};
    if (auto alignment{dtScope.alignment().value_or(0)}) {
      sizeInBytes += alignment - 1;
      sizeInBytes /= alignment;
      sizeInBytes *= alignment;
    }
    AddValue(
        dtValues, derivedTypeSchema_, "sizeinbytes"s, IntToExpr(sizeInBytes));
  }
  if (const Symbol *
      uninstDescObject{isPDTInstantiation
              ? DescribeType(DEREF(const_cast<Scope *>(dtSymbol->scope())))
              : nullptr}) {
    AddValue(dtValues, derivedTypeSchema_, "uninstantiated"s,
        evaluate::AsGenericExpr(evaluate::Expr<evaluate::SomeDerived>{
            evaluate::Designator<evaluate::SomeDerived>{
                DEREF(uninstDescObject)}}));
  } else {
    AddValue(dtValues, derivedTypeSchema_, "uninstantiated"s,
        SomeExpr{evaluate::NullPointer{}});
  }
  using Int8 = evaluate::Type<TypeCategory::Integer, 8>;
  using Int1 = evaluate::Type<TypeCategory::Integer, 1>;
  std::vector<Int8::Scalar> kinds;
  std::vector<Int1::Scalar> lenKinds;
  if (parameters) {
    // Package the derived type's parameters in declaration order for
    // each category of parameter.  KIND= type parameters are described
    // by their instantiated (or default) values, while LEN= type
    // parameters are described by their INTEGER kinds.
    for (SymbolRef ref : *parameters) {
      if (const auto *inst{dtScope.FindComponent(ref->name())}) {
        const auto &tpd{inst->get<TypeParamDetails>()};
        if (tpd.attr() == common::TypeParamAttr::Kind) {
          auto value{evaluate::ToInt64(tpd.init()).value_or(0)};
          if (derivedTypeSpec) {
            if (const auto *pv{derivedTypeSpec->FindParameter(inst->name())}) {
              if (pv->GetExplicit()) {
                if (auto instantiatedValue{
                        evaluate::ToInt64(*pv->GetExplicit())}) {
                  value = *instantiatedValue;
                }
              }
            }
          }
          kinds.emplace_back(value);
        } else { // LEN= parameter
          lenKinds.emplace_back(GetIntegerKind(*inst));
        }
      }
    }
  }
  AddValue(dtValues, derivedTypeSchema_, "kindparameter"s,
      SaveNumericPointerTarget<Int8>(
          scope, SaveObjectName(".kp."s + distinctName), std::move(kinds)));
  AddValue(dtValues, derivedTypeSchema_, "lenparameterkind"s,
      SaveNumericPointerTarget<Int1>(
          scope, SaveObjectName(".lpk."s + distinctName), std::move(lenKinds)));
  // Traverse the components of the derived type
  if (!isPDTDefinitionWithKindParameters) {
    std::vector<const Symbol *> dataComponentSymbols;
    std::vector<evaluate::StructureConstructor> procPtrComponents;
    for (const auto &pair : dtScope) {
      const Symbol &symbol{*pair.second};
      auto locationRestorer{common::ScopedSet(location_, symbol.name())};
      common::visit(
          common::visitors{
              [&](const TypeParamDetails &) {
                // already handled above in declaration order
              },
              [&](const ObjectEntityDetails &) {
                dataComponentSymbols.push_back(&symbol);
              },
              [&](const ProcEntityDetails &proc) {
                if (IsProcedurePointer(symbol)) {
                  procPtrComponents.emplace_back(
                      DescribeComponent(symbol, proc, scope));
                }
              },
              [&](const ProcBindingDetails &) { // handled in a later pass
              },
              [&](const GenericDetails &) { // ditto
              },
              [&](const auto &) {
                common::die(
                    "unexpected details on symbol '%s' in derived type scope",
                    symbol.name().ToString().c_str());
              },
          },
          symbol.details());
    }
    // Sort the data component symbols by offset before emitting them
    std::sort(dataComponentSymbols.begin(), dataComponentSymbols.end(),
        [](const Symbol *x, const Symbol *y) {
          return x->offset() < y->offset();
        });
    std::vector<evaluate::StructureConstructor> dataComponents;
    for (const Symbol *symbol : dataComponentSymbols) {
      auto locationRestorer{common::ScopedSet(location_, symbol->name())};
      dataComponents.emplace_back(
          DescribeComponent(*symbol, symbol->get<ObjectEntityDetails>(), scope,
              dtScope, distinctName, parameters));
    }
    AddValue(dtValues, derivedTypeSchema_, "component"s,
        SaveDerivedPointerTarget(scope, SaveObjectName(".c."s + distinctName),
            std::move(dataComponents),
            evaluate::ConstantSubscripts{
                static_cast<evaluate::ConstantSubscript>(
                    dataComponents.size())}));
    AddValue(dtValues, derivedTypeSchema_, "procptr"s,
        SaveDerivedPointerTarget(scope, SaveObjectName(".p."s + distinctName),
            std::move(procPtrComponents),
            evaluate::ConstantSubscripts{
                static_cast<evaluate::ConstantSubscript>(
                    procPtrComponents.size())}));
    // Compile the "vtable" of type-bound procedure bindings
    std::uint32_t specialBitSet{0};
    if (!dtSymbol->attrs().test(Attr::ABSTRACT)) {
      std::vector<evaluate::StructureConstructor> bindings{
          DescribeBindings(dtScope, scope)};
      AddValue(dtValues, derivedTypeSchema_, bindingDescCompName,
          SaveDerivedPointerTarget(scope, SaveObjectName(".v."s + distinctName),
              std::move(bindings),
              evaluate::ConstantSubscripts{
                  static_cast<evaluate::ConstantSubscript>(bindings.size())}));
      // Describe "special" bindings to defined assignments, FINAL subroutines,
      // and defined derived type I/O subroutines.  Defined assignments and I/O
      // subroutines override any parent bindings, but FINAL subroutines do not
      // (the runtime will call all of them).
      std::map<int, evaluate::StructureConstructor> specials{
          DescribeSpecialGenerics(dtScope, dtScope, derivedTypeSpec)};
      if (derivedTypeSpec) {
        for (auto &ref : FinalsForDerivedTypeInstantiation(*derivedTypeSpec)) {
          DescribeSpecialProc(specials, *ref, /*isAssignment-*/ false,
              /*isFinal=*/true, std::nullopt, nullptr, derivedTypeSpec,
              /*isTypeBound=*/true);
        }
        IncorporateDefinedIoGenericInterfaces(specials,
            common::DefinedIo::ReadFormatted, &scope, derivedTypeSpec);
        IncorporateDefinedIoGenericInterfaces(specials,
            common::DefinedIo::ReadUnformatted, &scope, derivedTypeSpec);
        IncorporateDefinedIoGenericInterfaces(specials,
            common::DefinedIo::WriteFormatted, &scope, derivedTypeSpec);
        IncorporateDefinedIoGenericInterfaces(specials,
            common::DefinedIo::WriteUnformatted, &scope, derivedTypeSpec);
      }
      // Pack the special procedure bindings in ascending order of their "which"
      // code values, and compile a little-endian bit-set of those codes for
      // use in O(1) look-up at run time.
      std::vector<evaluate::StructureConstructor> sortedSpecials;
      for (auto &pair : specials) {
        auto bit{std::uint32_t{1} << pair.first};
        CHECK(!(specialBitSet & bit));
        specialBitSet |= bit;
        sortedSpecials.emplace_back(std::move(pair.second));
      }
      AddValue(dtValues, derivedTypeSchema_, "special"s,
          SaveDerivedPointerTarget(scope, SaveObjectName(".s."s + distinctName),
              std::move(sortedSpecials),
              evaluate::ConstantSubscripts{
                  static_cast<evaluate::ConstantSubscript>(specials.size())}));
    }
    AddValue(dtValues, derivedTypeSchema_, "specialbitset"s,
        IntExpr<4>(specialBitSet));
    // Note the presence/absence of a parent component
    AddValue(dtValues, derivedTypeSchema_, "hasparent"s,
        IntExpr<1>(dtScope.GetDerivedTypeParent() != nullptr));
    // To avoid wasting run time attempting to initialize derived type
    // instances without any initialized components, analyze the type
    // and set a flag if there's nothing to do for it at run time.
    AddValue(dtValues, derivedTypeSchema_, "noinitializationneeded"s,
        IntExpr<1>(derivedTypeSpec &&
            !derivedTypeSpec->HasDefaultInitialization(false, false)));
    // Similarly, a flag to short-circuit destruction when not needed.
    AddValue(dtValues, derivedTypeSchema_, "nodestructionneeded"s,
        IntExpr<1>(derivedTypeSpec && !derivedTypeSpec->HasDestruction()));
    // Similarly, a flag to short-circuit finalization when not needed.
    AddValue(dtValues, derivedTypeSchema_, "nofinalizationneeded"s,
        IntExpr<1>(
            derivedTypeSpec && !MayRequireFinalization(*derivedTypeSpec)));
  }
  dtObject.get<ObjectEntityDetails>().set_init(MaybeExpr{
      StructureExpr(Structure(derivedTypeSchema_, std::move(dtValues)))});
  return &dtObject;
}

static const Symbol &GetSymbol(const Scope &schemata, SourceName name) {
  auto iter{schemata.find(name)};
  CHECK(iter != schemata.end());
  const Symbol &symbol{*iter->second};
  return symbol;
}

const Symbol &RuntimeTableBuilder::GetSchemaSymbol(const char *name) const {
  return GetSymbol(
      DEREF(tables_.schemata), SourceName{name, std::strlen(name)});
}

const DeclTypeSpec &RuntimeTableBuilder::GetSchema(
    const char *schemaName) const {
  Scope &schemata{DEREF(tables_.schemata)};
  SourceName name{schemaName, std::strlen(schemaName)};
  const Symbol &symbol{GetSymbol(schemata, name)};
  CHECK(symbol.has<DerivedTypeDetails>());
  CHECK(symbol.scope());
  CHECK(symbol.scope()->IsDerivedType());
  const DeclTypeSpec *spec{nullptr};
  if (symbol.scope()->derivedTypeSpec()) {
    DeclTypeSpec typeSpec{
        DeclTypeSpec::TypeDerived, *symbol.scope()->derivedTypeSpec()};
    spec = schemata.FindType(typeSpec);
  }
  if (!spec) {
    DeclTypeSpec typeSpec{
        DeclTypeSpec::TypeDerived, DerivedTypeSpec{name, symbol}};
    spec = schemata.FindType(typeSpec);
  }
  if (!spec) {
    spec = &schemata.MakeDerivedType(
        DeclTypeSpec::TypeDerived, DerivedTypeSpec{name, symbol});
  }
  CHECK(spec->AsDerived());
  return *spec;
}

SomeExpr RuntimeTableBuilder::GetEnumValue(const char *name) const {
  const Symbol &symbol{GetSchemaSymbol(name)};
  auto value{evaluate::ToInt64(symbol.get<ObjectEntityDetails>().init())};
  CHECK(value.has_value());
  return IntExpr<1>(*value);
}

Symbol &RuntimeTableBuilder::CreateObject(
    const std::string &name, const DeclTypeSpec &type, Scope &scope) {
  ObjectEntityDetails object;
  object.set_type(type);
  auto pair{scope.try_emplace(SaveObjectName(name),
      Attrs{Attr::TARGET, Attr::SAVE}, std::move(object))};
  CHECK(pair.second);
  Symbol &result{*pair.first->second};
  SetReadOnlyCompilerCreatedFlags(result);
  return result;
}

SourceName RuntimeTableBuilder::SaveObjectName(const std::string &name) {
  return *tables_.names.insert(name).first;
}

SomeExpr RuntimeTableBuilder::SaveNameAsPointerTarget(
    Scope &scope, const std::string &name) {
  CHECK(!name.empty());
  CHECK(name.front() != '.' || context_.IsTempName(name));
  ObjectEntityDetails object;
  auto len{static_cast<common::ConstantSubscript>(name.size())};
  if (const auto *spec{scope.FindType(DeclTypeSpec{CharacterTypeSpec{
          ParamValue{len, common::TypeParamAttr::Len}, KindExpr{1}}})}) {
    object.set_type(*spec);
  } else {
    object.set_type(scope.MakeCharacterType(
        ParamValue{len, common::TypeParamAttr::Len}, KindExpr{1}));
  }
  using evaluate::Ascii;
  using AsciiExpr = evaluate::Expr<Ascii>;
  object.set_init(evaluate::AsGenericExpr(AsciiExpr{name}));
  Symbol &symbol{*scope
                      .try_emplace(SaveObjectName(".n."s + name),
                          Attrs{Attr::TARGET, Attr::SAVE}, std::move(object))
                      .first->second};
  SetReadOnlyCompilerCreatedFlags(symbol);
  return evaluate::AsGenericExpr(
      AsciiExpr{evaluate::Designator<Ascii>{symbol}});
}

evaluate::StructureConstructor RuntimeTableBuilder::DescribeComponent(
    const Symbol &symbol, const ObjectEntityDetails &object, Scope &scope,
    Scope &dtScope, const std::string &distinctName,
    const SymbolVector *parameters) {
  evaluate::StructureConstructorValues values;
  auto &foldingContext{context_.foldingContext()};
  auto typeAndShape{evaluate::characteristics::TypeAndShape::Characterize(
      symbol, foldingContext)};
  CHECK(typeAndShape.has_value());
  auto dyType{typeAndShape->type()};
  const auto &shape{typeAndShape->shape()};
  AddValue(values, componentSchema_, "name"s,
      SaveNameAsPointerTarget(scope, symbol.name().ToString()));
  AddValue(values, componentSchema_, "category"s,
      IntExpr<1>(static_cast<int>(dyType.category())));
  if (dyType.IsUnlimitedPolymorphic() ||
      dyType.category() == TypeCategory::Derived) {
    AddValue(values, componentSchema_, "kind"s, IntExpr<1>(0));
  } else {
    AddValue(values, componentSchema_, "kind"s, IntExpr<1>(dyType.kind()));
  }
  AddValue(values, componentSchema_, "offset"s, IntExpr<8>(symbol.offset()));
  // CHARACTER length
  auto len{typeAndShape->LEN()};
  if (const semantics::DerivedTypeSpec *
      pdtInstance{dtScope.derivedTypeSpec()}) {
    auto restorer{foldingContext.WithPDTInstance(*pdtInstance)};
    len = Fold(foldingContext, std::move(len));
  }
  if (dyType.category() == TypeCategory::Character && len) {
    // Ignore IDIM(x) (represented as MAX(0, x))
    if (const auto *clamped{evaluate::UnwrapExpr<
            evaluate::Extremum<evaluate::SubscriptInteger>>(*len)}) {
      if (clamped->ordering == evaluate::Ordering::Greater &&
          clamped->left() == evaluate::Expr<evaluate::SubscriptInteger>{0}) {
        len = common::Clone(clamped->right());
      }
    }
    AddValue(values, componentSchema_, "characterlen"s,
        evaluate::AsGenericExpr(GetValue(len, parameters)));
  } else {
    AddValue(values, componentSchema_, "characterlen"s,
        PackageIntValueExpr(deferredEnum_));
  }
  // Describe component's derived type
  std::vector<evaluate::StructureConstructor> lenParams;
  if (dyType.category() == TypeCategory::Derived &&
      !dyType.IsUnlimitedPolymorphic()) {
    const DerivedTypeSpec &spec{dyType.GetDerivedTypeSpec()};
    Scope *derivedScope{const_cast<Scope *>(
        spec.scope() ? spec.scope() : spec.typeSymbol().scope())};
    const Symbol *derivedDescription{DescribeType(DEREF(derivedScope))};
    AddValue(values, componentSchema_, "derived"s,
        evaluate::AsGenericExpr(evaluate::Expr<evaluate::SomeDerived>{
            evaluate::Designator<evaluate::SomeDerived>{
                DEREF(derivedDescription)}}));
    // Package values of LEN parameters, if any
    if (const SymbolVector * specParams{GetTypeParameters(spec.typeSymbol())}) {
      for (SymbolRef ref : *specParams) {
        const auto &tpd{ref->get<TypeParamDetails>()};
        if (tpd.attr() == common::TypeParamAttr::Len) {
          if (const ParamValue * paramValue{spec.FindParameter(ref->name())}) {
            lenParams.emplace_back(GetValue(*paramValue, parameters));
          } else {
            lenParams.emplace_back(GetValue(tpd.init(), parameters));
          }
        }
      }
    }
  } else {
    // Subtle: a category of Derived with a null derived type pointer
    // signifies CLASS(*)
    AddValue(values, componentSchema_, "derived"s,
        SomeExpr{evaluate::NullPointer{}});
  }
  // LEN type parameter values for the component's type
  if (!lenParams.empty()) {
    AddValue(values, componentSchema_, "lenvalue"s,
        SaveDerivedPointerTarget(scope,
            SaveObjectName(
                ".lv."s + distinctName + "."s + symbol.name().ToString()),
            std::move(lenParams),
            evaluate::ConstantSubscripts{
                static_cast<evaluate::ConstantSubscript>(lenParams.size())}));
  } else {
    AddValue(values, componentSchema_, "lenvalue"s,
        SomeExpr{evaluate::NullPointer{}});
  }
  // Shape information
  int rank{evaluate::GetRank(shape)};
  AddValue(values, componentSchema_, "rank"s, IntExpr<1>(rank));
  if (rank > 0 && !IsAllocatable(symbol) && !IsPointer(symbol)) {
    std::vector<evaluate::StructureConstructor> bounds;
    evaluate::NamedEntity entity{symbol};
    for (int j{0}; j < rank; ++j) {
      bounds.emplace_back(
          GetValue(std::make_optional(
                       evaluate::GetRawLowerBound(foldingContext, entity, j)),
              parameters));
      bounds.emplace_back(GetValue(
          evaluate::GetRawUpperBound(foldingContext, entity, j), parameters));
    }
    AddValue(values, componentSchema_, "bounds"s,
        SaveDerivedPointerTarget(scope,
            SaveObjectName(
                ".b."s + distinctName + "."s + symbol.name().ToString()),
            std::move(bounds), evaluate::ConstantSubscripts{2, rank}));
  } else {
    AddValue(
        values, componentSchema_, "bounds"s, SomeExpr{evaluate::NullPointer{}});
  }
  // Default component initialization
  bool hasDataInit{false};
  if (IsAllocatable(symbol)) {
    AddValue(values, componentSchema_, "genre"s, GetEnumValue("allocatable"));
  } else if (IsPointer(symbol)) {
    AddValue(values, componentSchema_, "genre"s, GetEnumValue("pointer"));
    hasDataInit = InitializeDataPointer(
        values, symbol, object, scope, dtScope, distinctName);
  } else if (IsAutomatic(symbol)) {
    AddValue(values, componentSchema_, "genre"s, GetEnumValue("automatic"));
  } else {
    AddValue(values, componentSchema_, "genre"s, GetEnumValue("data"));
    hasDataInit = object.init().has_value();
    if (hasDataInit) {
      AddValue(values, componentSchema_, "initialization"s,
          SaveObjectInit(scope,
              SaveObjectName(
                  ".di."s + distinctName + "."s + symbol.name().ToString()),
              object));
    }
  }
  if (!hasDataInit) {
    AddValue(values, componentSchema_, "initialization"s,
        SomeExpr{evaluate::NullPointer{}});
  }
  return {DEREF(componentSchema_.AsDerived()), std::move(values)};
}

evaluate::StructureConstructor RuntimeTableBuilder::DescribeComponent(
    const Symbol &symbol, const ProcEntityDetails &proc, Scope &scope) {
  evaluate::StructureConstructorValues values;
  AddValue(values, procPtrSchema_, "name"s,
      SaveNameAsPointerTarget(scope, symbol.name().ToString()));
  AddValue(values, procPtrSchema_, "offset"s, IntExpr<8>(symbol.offset()));
  if (auto init{proc.init()}; init && *init) {
    AddValue(values, procPtrSchema_, "initialization"s,
        SomeExpr{evaluate::ProcedureDesignator{**init}});
  } else {
    AddValue(values, procPtrSchema_, "initialization"s,
        SomeExpr{evaluate::NullPointer{}});
  }
  return {DEREF(procPtrSchema_.AsDerived()), std::move(values)};
}

// Create a static pointer object with the same initialization
// from whence the runtime can memcpy() the data pointer
// component initialization.
// Creates and interconnects the symbols, scopes, and types for
//   TYPE :: ptrDt
//     type, POINTER :: name
//   END TYPE
//   TYPE(ptrDt), TARGET, SAVE :: ptrInit = ptrDt(designator)
// and then initializes the original component by setting
//   initialization = ptrInit
// which takes the address of ptrInit because the type is C_PTR.
// This technique of wrapping the data pointer component into
// a derived type instance disables any reason for lowering to
// attempt to dereference the RHS of an initializer, thereby
// allowing the runtime to actually perform the initialization
// by means of a simple memcpy() of the wrapped descriptor in
// ptrInit to the data pointer component being initialized.
bool RuntimeTableBuilder::InitializeDataPointer(
    evaluate::StructureConstructorValues &values, const Symbol &symbol,
    const ObjectEntityDetails &object, Scope &scope, Scope &dtScope,
    const std::string &distinctName) {
  if (object.init().has_value()) {
    SourceName ptrDtName{SaveObjectName(
        ".dp."s + distinctName + "."s + symbol.name().ToString())};
    Symbol &ptrDtSym{
        *scope.try_emplace(ptrDtName, Attrs{}, UnknownDetails{}).first->second};
    SetReadOnlyCompilerCreatedFlags(ptrDtSym);
    Scope &ptrDtScope{scope.MakeScope(Scope::Kind::DerivedType, &ptrDtSym)};
    ignoreScopes_.insert(&ptrDtScope);
    ObjectEntityDetails ptrDtObj;
    ptrDtObj.set_type(DEREF(object.type()));
    ptrDtObj.set_shape(object.shape());
    Symbol &ptrDtComp{*ptrDtScope
                           .try_emplace(symbol.name(), Attrs{Attr::POINTER},
                               std::move(ptrDtObj))
                           .first->second};
    DerivedTypeDetails ptrDtDetails;
    ptrDtDetails.add_component(ptrDtComp);
    ptrDtSym.set_details(std::move(ptrDtDetails));
    ptrDtSym.set_scope(&ptrDtScope);
    DeclTypeSpec &ptrDtDeclType{
        scope.MakeDerivedType(DeclTypeSpec::Category::TypeDerived,
            DerivedTypeSpec{ptrDtName, ptrDtSym})};
    DerivedTypeSpec &ptrDtDerived{DEREF(ptrDtDeclType.AsDerived())};
    ptrDtDerived.set_scope(ptrDtScope);
    ptrDtDerived.CookParameters(context_.foldingContext());
    ptrDtDerived.Instantiate(scope);
    ObjectEntityDetails ptrInitObj;
    ptrInitObj.set_type(ptrDtDeclType);
    evaluate::StructureConstructorValues ptrInitValues;
    AddValue(
        ptrInitValues, ptrDtDeclType, symbol.name().ToString(), *object.init());
    ptrInitObj.set_init(evaluate::AsGenericExpr(
        Structure(ptrDtDeclType, std::move(ptrInitValues))));
    AddValue(values, componentSchema_, "initialization"s,
        SaveObjectInit(scope,
            SaveObjectName(
                ".di."s + distinctName + "."s + symbol.name().ToString()),
            ptrInitObj));
    return true;
  } else {
    return false;
  }
}

evaluate::StructureConstructor RuntimeTableBuilder::PackageIntValue(
    const SomeExpr &genre, std::int64_t n) const {
  evaluate::StructureConstructorValues xs;
  AddValue(xs, valueSchema_, "genre"s, genre);
  AddValue(xs, valueSchema_, "value"s, IntToExpr(n));
  return Structure(valueSchema_, std::move(xs));
}

SomeExpr RuntimeTableBuilder::PackageIntValueExpr(
    const SomeExpr &genre, std::int64_t n) const {
  return StructureExpr(PackageIntValue(genre, n));
}

SymbolVector CollectBindings(const Scope &dtScope) {
  SymbolVector result;
  std::map<SourceName, Symbol *> localBindings;
  // Collect local bindings
  for (auto pair : dtScope) {
    Symbol &symbol{const_cast<Symbol &>(*pair.second)};
    if (auto *binding{symbol.detailsIf<ProcBindingDetails>()}) {
      localBindings.emplace(symbol.name(), &symbol);
      binding->set_numPrivatesNotOverridden(0);
    }
  }
  if (const Scope * parentScope{dtScope.GetDerivedTypeParent()}) {
    result = CollectBindings(*parentScope);
    // Apply overrides from the local bindings of the extended type
    for (auto iter{result.begin()}; iter != result.end(); ++iter) {
      const Symbol &symbol{**iter};
      auto overriderIter{localBindings.find(symbol.name())};
      if (overriderIter != localBindings.end()) {
        Symbol &overrider{*overriderIter->second};
        if (symbol.attrs().test(Attr::PRIVATE) &&
            FindModuleContaining(symbol.owner()) !=
                FindModuleContaining(dtScope)) {
          // Don't override inaccessible PRIVATE bindings
          auto &binding{overrider.get<ProcBindingDetails>()};
          binding.set_numPrivatesNotOverridden(
              binding.numPrivatesNotOverridden() + 1);
        } else {
          *iter = overrider;
          localBindings.erase(overriderIter);
        }
      }
    }
  }
  // Add remaining (non-overriding) local bindings in name order to the result
  for (auto pair : localBindings) {
    result.push_back(*pair.second);
  }
  return result;
}

std::vector<evaluate::StructureConstructor>
RuntimeTableBuilder::DescribeBindings(const Scope &dtScope, Scope &scope) {
  std::vector<evaluate::StructureConstructor> result;
  for (const SymbolRef &ref : CollectBindings(dtScope)) {
    evaluate::StructureConstructorValues values;
    AddValue(values, bindingSchema_, procCompName,
        SomeExpr{evaluate::ProcedureDesignator{
            ref.get().get<ProcBindingDetails>().symbol()}});
    AddValue(values, bindingSchema_, "name"s,
        SaveNameAsPointerTarget(scope, ref.get().name().ToString()));
    result.emplace_back(DEREF(bindingSchema_.AsDerived()), std::move(values));
  }
  return result;
}

std::map<int, evaluate::StructureConstructor>
RuntimeTableBuilder::DescribeSpecialGenerics(const Scope &dtScope,
    const Scope &thisScope, const DerivedTypeSpec *derivedTypeSpec) const {
  std::map<int, evaluate::StructureConstructor> specials;
  if (const Scope * parentScope{dtScope.GetDerivedTypeParent()}) {
    specials =
        DescribeSpecialGenerics(*parentScope, thisScope, derivedTypeSpec);
  }
  for (auto pair : dtScope) {
    const Symbol &symbol{*pair.second};
    if (const auto *generic{symbol.detailsIf<GenericDetails>()}) {
      DescribeSpecialGeneric(*generic, specials, thisScope, derivedTypeSpec);
    }
  }
  return specials;
}

void RuntimeTableBuilder::DescribeSpecialGeneric(const GenericDetails &generic,
    std::map<int, evaluate::StructureConstructor> &specials,
    const Scope &dtScope, const DerivedTypeSpec *derivedTypeSpec) const {
  common::visit(
      common::visitors{
          [&](const GenericKind::OtherKind &k) {
            if (k == GenericKind::OtherKind::Assignment) {
              for (auto ref : generic.specificProcs()) {
                DescribeSpecialProc(specials, *ref, /*isAssignment=*/true,
                    /*isFinal=*/false, std::nullopt, &dtScope, derivedTypeSpec,
                    /*isTypeBound=*/true);
              }
            }
          },
          [&](const common::DefinedIo &io) {
            switch (io) {
            case common::DefinedIo::ReadFormatted:
            case common::DefinedIo::ReadUnformatted:
            case common::DefinedIo::WriteFormatted:
            case common::DefinedIo::WriteUnformatted:
              for (auto ref : generic.specificProcs()) {
                DescribeSpecialProc(specials, *ref, /*isAssignment=*/false,
                    /*isFinal=*/false, io, &dtScope, derivedTypeSpec,
                    /*isTypeBound=*/true);
              }
              break;
            }
          },
          [](const auto &) {},
      },
      generic.kind().u);
}

void RuntimeTableBuilder::DescribeSpecialProc(
    std::map<int, evaluate::StructureConstructor> &specials,
    const Symbol &specificOrBinding, bool isAssignment, bool isFinal,
    std::optional<common::DefinedIo> io, const Scope *dtScope,
    const DerivedTypeSpec *derivedTypeSpec, bool isTypeBound) const {
  const auto *binding{specificOrBinding.detailsIf<ProcBindingDetails>()};
  if (binding && dtScope) { // use most recent override
    binding = &DEREF(dtScope->FindComponent(specificOrBinding.name()))
                   .get<ProcBindingDetails>();
  }
  const Symbol &specific{*(binding ? &binding->symbol() : &specificOrBinding)};
  if (auto proc{evaluate::characteristics::Procedure::Characterize(
          specific, context_.foldingContext())}) {
    std::uint8_t isArgDescriptorSet{0};
    std::uint8_t isArgContiguousSet{0};
    int argThatMightBeDescriptor{0};
    MaybeExpr which;
    if (isAssignment) {
      // Only type-bound asst's with the same type on both dummy arguments
      // are germane to the runtime, which needs only these to implement
      // component assignment as part of intrinsic assignment.
      // Non-type-bound generic INTERFACEs and assignments from distinct
      // types must not be used for component intrinsic assignment.
      CHECK(proc->dummyArguments.size() == 2);
      const auto t1{
          DEREF(std::get_if<evaluate::characteristics::DummyDataObject>(
                    &proc->dummyArguments[0].u))
              .type.type()};
      const auto t2{
          DEREF(std::get_if<evaluate::characteristics::DummyDataObject>(
                    &proc->dummyArguments[1].u))
              .type.type()};
      if (!binding || t1.category() != TypeCategory::Derived ||
          t2.category() != TypeCategory::Derived ||
          t1.IsUnlimitedPolymorphic() || t2.IsUnlimitedPolymorphic() ||
          t1.GetDerivedTypeSpec() != t2.GetDerivedTypeSpec()) {
        return;
      }
      which = proc->IsElemental() ? elementalAssignmentEnum_
                                  : scalarAssignmentEnum_;
      if (binding && binding->passName() &&
          *binding->passName() == proc->dummyArguments[1].name) {
        argThatMightBeDescriptor = 1;
        isArgDescriptorSet |= 2;
      } else {
        argThatMightBeDescriptor = 2; // the non-passed-object argument
        isArgDescriptorSet |= 1;
      }
    } else if (isFinal) {
      CHECK(binding == nullptr); // FINALs are not bindings
      CHECK(proc->dummyArguments.size() == 1);
      if (proc->IsElemental()) {
        which = elementalFinalEnum_;
      } else {
        const auto &dummyData{
            std::get<evaluate::characteristics::DummyDataObject>(
                proc->dummyArguments.at(0).u)};
        const auto &typeAndShape{dummyData.type};
        if (typeAndShape.attrs().test(
                evaluate::characteristics::TypeAndShape::Attr::AssumedRank)) {
          which = assumedRankFinalEnum_;
          isArgDescriptorSet |= 1;
        } else {
          which = scalarFinalEnum_;
          if (int rank{evaluate::GetRank(typeAndShape.shape())}; rank > 0) {
            which = IntExpr<1>(ToInt64(which).value() + rank);
            if (!proc->dummyArguments[0].CanBePassedViaImplicitInterface()) {
              argThatMightBeDescriptor = 1;
            }
            if (!typeAndShape.attrs().test(evaluate::characteristics::
                        TypeAndShape::Attr::AssumedShape) ||
                dummyData.attrs.test(evaluate::characteristics::
                        DummyDataObject::Attr::Contiguous)) {
              isArgContiguousSet |= 1;
            }
          }
        }
      }
    } else { // defined derived type I/O
      CHECK(proc->dummyArguments.size() >= 4);
      const auto *ddo{std::get_if<evaluate::characteristics::DummyDataObject>(
          &proc->dummyArguments[0].u)};
      if (!ddo) {
        return;
      }
      if (derivedTypeSpec &&
          !ddo->type.type().IsTkCompatibleWith(
              evaluate::DynamicType{*derivedTypeSpec})) {
        // Defined I/O specific procedure is not for this derived type.
        return;
      }
      if (ddo->type.type().IsPolymorphic()) {
        isArgDescriptorSet |= 1;
      }
      switch (io.value()) {
      case common::DefinedIo::ReadFormatted:
        which = readFormattedEnum_;
        break;
      case common::DefinedIo::ReadUnformatted:
        which = readUnformattedEnum_;
        break;
      case common::DefinedIo::WriteFormatted:
        which = writeFormattedEnum_;
        break;
      case common::DefinedIo::WriteUnformatted:
        which = writeUnformattedEnum_;
        break;
      }
    }
    if (argThatMightBeDescriptor != 0 &&
        !proc->dummyArguments.at(argThatMightBeDescriptor - 1)
             .CanBePassedViaImplicitInterface()) {
      isArgDescriptorSet |= 1 << (argThatMightBeDescriptor - 1);
    }
    evaluate::StructureConstructorValues values;
    auto index{evaluate::ToInt64(which)};
    CHECK(index.has_value());
    AddValue(
        values, specialSchema_, "which"s, SomeExpr{std::move(which.value())});
    AddValue(values, specialSchema_, "isargdescriptorset"s,
        IntExpr<1>(isArgDescriptorSet));
    AddValue(values, specialSchema_, "istypebound"s,
        IntExpr<1>(isTypeBound ? 1 : 0));
    AddValue(values, specialSchema_, "isargcontiguousset"s,
        IntExpr<1>(isArgContiguousSet));
    AddValue(values, specialSchema_, procCompName,
        SomeExpr{evaluate::ProcedureDesignator{specific}});
    // index might already be present in the case of an override
    specials.emplace(*index,
        evaluate::StructureConstructor{
            DEREF(specialSchema_.AsDerived()), std::move(values)});
  }
}

void RuntimeTableBuilder::IncorporateDefinedIoGenericInterfaces(
    std::map<int, evaluate::StructureConstructor> &specials,
    common::DefinedIo definedIo, const Scope *scope,
    const DerivedTypeSpec *derivedTypeSpec) {
  SourceName name{GenericKind::AsFortran(definedIo)};
  for (; !scope->IsGlobal(); scope = &scope->parent()) {
    if (auto asst{scope->find(name)}; asst != scope->end()) {
      const Symbol &generic{asst->second->GetUltimate()};
      const auto &genericDetails{generic.get<GenericDetails>()};
      CHECK(std::holds_alternative<common::DefinedIo>(genericDetails.kind().u));
      CHECK(std::get<common::DefinedIo>(genericDetails.kind().u) == definedIo);
      for (auto ref : genericDetails.specificProcs()) {
        DescribeSpecialProc(specials, *ref, false, false, definedIo, nullptr,
            derivedTypeSpec, false);
      }
    }
  }
}

RuntimeDerivedTypeTables BuildRuntimeDerivedTypeTables(
    SemanticsContext &context) {
  RuntimeDerivedTypeTables result;
  result.schemata = context.GetBuiltinModule(typeInfoBuiltinModule);
  if (result.schemata) {
    RuntimeTableBuilder builder{context, result};
    builder.DescribeTypes(context.globalScope(), false);
  }
  return result;
}

// Find the type of a defined I/O procedure's interface's initial "dtv"
// dummy argument.  Returns a non-null DeclTypeSpec pointer only if that
// dtv argument exists and is a derived type.
static const DeclTypeSpec *GetDefinedIoSpecificArgType(const Symbol &specific) {
  const Symbol *interface{&specific.GetUltimate()};
  if (const auto *procEntity{specific.detailsIf<ProcEntityDetails>()}) {
    interface = procEntity->procInterface();
  }
  if (interface) {
    if (const SubprogramDetails *
            subprogram{interface->detailsIf<SubprogramDetails>()};
        subprogram && !subprogram->dummyArgs().empty()) {
      if (const Symbol * dtvArg{subprogram->dummyArgs().at(0)}) {
        if (const DeclTypeSpec * declType{dtvArg->GetType()}) {
          return declType->AsDerived() ? declType : nullptr;
        }
      }
    }
  }
  return nullptr;
}

// Locate a particular scope's generic interface for a specific kind of
// defined I/O.
static const Symbol *FindGenericDefinedIo(
    const Scope &scope, common::DefinedIo which) {
  if (const Symbol * symbol{scope.FindSymbol(GenericKind::AsFortran(which))}) {
    const Symbol &generic{symbol->GetUltimate()};
    const auto &genericDetails{generic.get<GenericDetails>()};
    CHECK(std::holds_alternative<common::DefinedIo>(genericDetails.kind().u));
    CHECK(std::get<common::DefinedIo>(genericDetails.kind().u) == which);
    return &generic;
  } else {
    return nullptr;
  }
}

std::multimap<const Symbol *, NonTbpDefinedIo>
CollectNonTbpDefinedIoGenericInterfaces(
    const Scope &scope, bool useRuntimeTypeInfoEntries) {
  std::multimap<const Symbol *, NonTbpDefinedIo> result;
  if (!scope.IsTopLevel() &&
      (scope.GetImportKind() == Scope::ImportKind::All ||
          scope.GetImportKind() == Scope::ImportKind::Default)) {
    result = CollectNonTbpDefinedIoGenericInterfaces(
        scope.parent(), useRuntimeTypeInfoEntries);
  }
  if (scope.kind() != Scope::Kind::DerivedType) {
    for (common::DefinedIo which :
        {common::DefinedIo::ReadFormatted, common::DefinedIo::ReadUnformatted,
            common::DefinedIo::WriteFormatted,
            common::DefinedIo::WriteUnformatted}) {
      if (const Symbol * generic{FindGenericDefinedIo(scope, which)}) {
        for (auto specific : generic->get<GenericDetails>().specificProcs()) {
          if (const DeclTypeSpec *
              declType{GetDefinedIoSpecificArgType(*specific)}) {
            const DerivedTypeSpec &derived{DEREF(declType->AsDerived())};
            if (const Symbol *
                dtDesc{derived.scope()
                        ? derived.scope()->runtimeDerivedTypeDescription()
                        : nullptr}) {
              if (useRuntimeTypeInfoEntries &&
                  &derived.scope()->parent() == &generic->owner()) {
                // This non-TBP defined I/O generic was defined in the
                // same scope as the derived type, and it will be
                // included in the derived type's special bindings
                // by IncorporateDefinedIoGenericInterfaces().
              } else {
                // Local scope's specific overrides host's for this type
                bool updated{false};
                for (auto [iter, end]{result.equal_range(dtDesc)}; iter != end;
                     ++iter) {
                  NonTbpDefinedIo &nonTbp{iter->second};
                  if (nonTbp.definedIo == which) {
                    nonTbp.subroutine = &*specific;
                    nonTbp.isDtvArgPolymorphic = declType->IsPolymorphic();
                    updated = true;
                  }
                }
                if (!updated) {
                  result.emplace(dtDesc,
                      NonTbpDefinedIo{
                          &*specific, which, declType->IsPolymorphic()});
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

// ShouldIgnoreRuntimeTypeInfoNonTbpGenericInterfaces()
//
// Returns a true result when a kind of defined I/O generic procedure
// has a type (from a symbol or a NAMELIST) such that
// (1) there is a specific procedure matching that type for a non-type-bound
//     generic defined in the scope of the type, and
// (2) that specific procedure is unavailable or overridden in a particular
//     local scope.
// Specific procedures of non-type-bound defined I/O generic interfaces
// declared in the scope of a derived type are identified as special bindings
// in the derived type's runtime type information, as if they had been
// type-bound.  This predicate is meant to determine local situations in
// which those special bindings are not to be used.  Its result is intended
// to be put into the "ignoreNonTbpEntries" flag of
// runtime::NonTbpDefinedIoTable and passed (negated) as the
// "useRuntimeTypeInfoEntries" argument of
// CollectNonTbpDefinedIoGenericInterfaces() above.

static const Symbol *FindSpecificDefinedIo(const Scope &scope,
    const evaluate::DynamicType &derived, common::DefinedIo which) {
  if (const Symbol * generic{FindGenericDefinedIo(scope, which)}) {
    for (auto ref : generic->get<GenericDetails>().specificProcs()) {
      const Symbol &specific{*ref};
      if (const DeclTypeSpec *
          thisType{GetDefinedIoSpecificArgType(specific)}) {
        if (evaluate::DynamicType{DEREF(thisType->AsDerived()), true}
                .IsTkCompatibleWith(derived)) {
          return &specific.GetUltimate();
        }
      }
    }
  }
  return nullptr;
}

bool ShouldIgnoreRuntimeTypeInfoNonTbpGenericInterfaces(
    const Scope &scope, const DerivedTypeSpec *derived) {
  if (!derived) {
    return false;
  }
  const Symbol &typeSymbol{derived->typeSymbol()};
  const Scope &typeScope{typeSymbol.GetUltimate().owner()};
  evaluate::DynamicType dyType{*derived};
  for (common::DefinedIo which :
      {common::DefinedIo::ReadFormatted, common::DefinedIo::ReadUnformatted,
          common::DefinedIo::WriteFormatted,
          common::DefinedIo::WriteUnformatted}) {
    if (const Symbol *
        specific{FindSpecificDefinedIo(typeScope, dyType, which)}) {
      // There's a non-TBP defined I/O procedure in the scope of the type's
      // definition that applies to this type.  It will appear in the type's
      // runtime information.  Determine whether it still applies in the
      // scope of interest.
      if (FindSpecificDefinedIo(scope, dyType, which) != specific) {
        return true;
      }
    }
  }
  return false;
}

bool ShouldIgnoreRuntimeTypeInfoNonTbpGenericInterfaces(
    const Scope &scope, const DeclTypeSpec *type) {
  return type &&
      ShouldIgnoreRuntimeTypeInfoNonTbpGenericInterfaces(
          scope, type->AsDerived());
}

bool ShouldIgnoreRuntimeTypeInfoNonTbpGenericInterfaces(
    const Scope &scope, const Symbol *symbol) {
  if (!symbol) {
    return false;
  }
  return common::visit(
      common::visitors{
          [&](const NamelistDetails &x) {
            for (auto ref : x.objects()) {
              if (ShouldIgnoreRuntimeTypeInfoNonTbpGenericInterfaces(
                      scope, &*ref)) {
                return true;
              }
            }
            return false;
          },
          [&](const auto &) {
            return ShouldIgnoreRuntimeTypeInfoNonTbpGenericInterfaces(
                scope, symbol->GetType());
          },
      },
      symbol->GetUltimate().details());
}

} // namespace Fortran::semantics
