//===-- lib/Parser/openmp-parsers.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Top-level grammar specification for OpenMP.
// See OpenMP-4.5-grammar.txt for documentation.

#include "basic-parsers.h"
#include "expr-parsers.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

// OpenMP Directives and Clauses
namespace Fortran::parser {

constexpr auto startOmpLine = skipStuffBeforeStatement >> "!$OMP "_sptok;
constexpr auto endOmpLine = space >> endOfLine;

/// Parse OpenMP directive name (this includes compound directives).
struct OmpDirectiveNameParser {
  using resultType = llvm::omp::Directive;
  using Token = TokenStringMatch<false, false>;

  std::optional<resultType> Parse(ParseState &state) const {
    for (const NameWithId &nid : directives()) {
      if (attempt(Token(nid.first.data())).Parse(state)) {
        return nid.second;
      }
    }
    return std::nullopt;
  }

private:
  using NameWithId = std::pair<std::string, llvm::omp::Directive>;

  llvm::iterator_range<const NameWithId *> directives() const;
  void initTokens(NameWithId *) const;
};

llvm::iterator_range<const OmpDirectiveNameParser::NameWithId *>
OmpDirectiveNameParser::directives() const {
  static NameWithId table[llvm::omp::Directive_enumSize];
  [[maybe_unused]] static bool init = (initTokens(table), true);
  return llvm::make_range(std::cbegin(table), std::cend(table));
}

void OmpDirectiveNameParser::initTokens(NameWithId *table) const {
  for (size_t i{0}, e{llvm::omp::Directive_enumSize}; i != e; ++i) {
    auto id{static_cast<llvm::omp::Directive>(i)};
    llvm::StringRef name{llvm::omp::getOpenMPDirectiveName(id)};
    table[i] = std::make_pair(name.str(), id);
  }
  // Sort the table with respect to the directive name length in a descending
  // order. This is to make sure that longer names are tried first, before
  // any potential prefix (e.g. "target update" before "target").
  std::sort(table, table + llvm::omp::Directive_enumSize,
      [](auto &a, auto &b) { return a.first.size() > b.first.size(); });
}

// --- Modifier helpers -----------------------------------------------

template <typename Clause, typename Separator> struct ModifierList {
  constexpr ModifierList(Separator sep) : sep_(sep) {}
  constexpr ModifierList(const ModifierList &) = default;
  constexpr ModifierList(ModifierList &&) = default;

  using resultType = std::list<typename Clause::Modifier>;

  std::optional<resultType> Parse(ParseState &state) const {
    auto listp{nonemptySeparated(Parser<typename Clause::Modifier>{}, sep_)};
    if (auto result{attempt(listp).Parse(state)}) {
      if (!attempt(":"_tok).Parse(state)) {
        return std::nullopt;
      }
      return std::move(result);
    }
    return resultType{};
  }

private:
  const Separator sep_;
};

// Use a function to create ModifierList because functions allow "partial"
// template argument deduction: "modifierList<Clause>(sep)" would be legal,
// while "ModifierList<Clause>(sep)" would complain about a missing template
// argument "Separator".
template <typename Clause, typename Separator>
constexpr ModifierList<Clause, Separator> modifierList(Separator sep) {
  return ModifierList<Clause, Separator>(sep);
}

// Parse the input as any modifier from ClauseTy, but only succeed if
// the result was the SpecificTy. It requires that SpecificTy is one
// of the alternatives in ClauseTy::Modifier.
// The reason to have this is that ClauseTy::Modifier has "source",
// while specific modifiers don't. This class allows to parse a specific
// modifier together with obtaining its location.
template <typename SpecificTy, typename ClauseTy>
struct SpecificModifierParser {
  using resultType = typename ClauseTy::Modifier;
  std::optional<resultType> Parse(ParseState &state) const {
    if (auto result{attempt(Parser<resultType>{}).Parse(state)}) {
      if (std::holds_alternative<SpecificTy>(result->u)) {
        return result;
      }
    }
    return std::nullopt;
  }
};

// --- Iterator helpers -----------------------------------------------

// [5.0:47:17-18] In an iterator-specifier, if the iterator-type is not
// specified then the type of that iterator is default integer.
// [5.0:49:14] The iterator-type must be an integer type.
static std::list<EntityDecl> makeEntityList(std::list<ObjectName> &&names) {
  std::list<EntityDecl> entities;

  for (auto iter = names.begin(), end = names.end(); iter != end; ++iter) {
    EntityDecl entityDecl(
        /*ObjectName=*/std::move(*iter), std::optional<ArraySpec>{},
        std::optional<CoarraySpec>{}, std::optional<CharLength>{},
        std::optional<Initialization>{});
    entities.push_back(std::move(entityDecl));
  }
  return entities;
}

static TypeDeclarationStmt makeIterSpecDecl(
    DeclarationTypeSpec &&spec, std::list<ObjectName> &&names) {
  return TypeDeclarationStmt(
      std::move(spec), std::list<AttrSpec>{}, makeEntityList(std::move(names)));
}

static TypeDeclarationStmt makeIterSpecDecl(std::list<ObjectName> &&names) {
  // Assume INTEGER without kind selector.
  DeclarationTypeSpec typeSpec(
      IntrinsicTypeSpec{IntegerTypeSpec{std::nullopt}});

  return TypeDeclarationStmt(std::move(typeSpec), std::list<AttrSpec>{},
      makeEntityList(std::move(names)));
}

// --- Parsers for arguments ------------------------------------------

// At the moment these are only directive arguments. This is needed for
// parsing directive-specification.

TYPE_PARSER( //
    construct<OmpLocator>(Parser<OmpObject>{}) ||
    construct<OmpLocator>(Parser<FunctionReference>{}))

TYPE_PARSER(sourced( //
    construct<OmpArgument>(Parser<OmpMapperSpecifier>{}) ||
    construct<OmpArgument>(Parser<OmpReductionSpecifier>{}) ||
    construct<OmpArgument>(Parser<OmpLocator>{})))

TYPE_PARSER(construct<OmpLocatorList>(nonemptyList(Parser<OmpLocator>{})))

TYPE_PARSER( //
    construct<OmpTypeSpecifier>(Parser<TypeSpec>{}) ||
    construct<OmpTypeSpecifier>(Parser<DeclarationTypeSpec>{}))

TYPE_PARSER(construct<OmpReductionSpecifier>( //
    Parser<OmpReductionIdentifier>{},
    ":"_tok >> nonemptyList(Parser<OmpTypeSpecifier>{}),
    maybe(":"_tok >> Parser<OmpReductionCombiner>{})))

// --- Parsers for context traits -------------------------------------

static std::string nameToString(Name &&name) { return name.ToString(); }

TYPE_PARSER(sourced(construct<OmpTraitPropertyName>( //
    construct<OmpTraitPropertyName>(space >> charLiteralConstantWithoutKind) ||
    construct<OmpTraitPropertyName>(
        applyFunction(nameToString, Parser<Name>{})))))

TYPE_PARSER(sourced(construct<OmpTraitScore>( //
    "SCORE"_id >> parenthesized(scalarIntExpr))))

TYPE_PARSER(sourced(construct<OmpTraitPropertyExtension::Complex>(
    Parser<OmpTraitPropertyName>{},
    parenthesized(nonemptySeparated(
        indirect(Parser<OmpTraitPropertyExtension>{}), ",")))))

TYPE_PARSER(sourced(construct<OmpTraitPropertyExtension>(
    construct<OmpTraitPropertyExtension>(
        Parser<OmpTraitPropertyExtension::Complex>{}) ||
    construct<OmpTraitPropertyExtension>(Parser<OmpTraitPropertyName>{}) ||
    construct<OmpTraitPropertyExtension>(scalarExpr))))

TYPE_PARSER(construct<OmpTraitSelectorName::Value>(
    "ARCH"_id >> pure(OmpTraitSelectorName::Value::Arch) ||
    "ATOMIC_DEFAULT_MEM_ORDER"_id >>
        pure(OmpTraitSelectorName::Value::Atomic_Default_Mem_Order) ||
    "CONDITION"_id >> pure(OmpTraitSelectorName::Value::Condition) ||
    "DEVICE_NUM"_id >> pure(OmpTraitSelectorName::Value::Device_Num) ||
    "EXTENSION"_id >> pure(OmpTraitSelectorName::Value::Extension) ||
    "ISA"_id >> pure(OmpTraitSelectorName::Value::Isa) ||
    "KIND"_id >> pure(OmpTraitSelectorName::Value::Kind) ||
    "REQUIRES"_id >> pure(OmpTraitSelectorName::Value::Requires) ||
    "SIMD"_id >> pure(OmpTraitSelectorName::Value::Simd) ||
    "UID"_id >> pure(OmpTraitSelectorName::Value::Uid) ||
    "VENDOR"_id >> pure(OmpTraitSelectorName::Value::Vendor)))

TYPE_PARSER(sourced(construct<OmpTraitSelectorName>(
    // Parse predefined names first (because of SIMD).
    construct<OmpTraitSelectorName>(Parser<OmpTraitSelectorName::Value>{}) ||
    construct<OmpTraitSelectorName>(OmpDirectiveNameParser{}) ||
    // identifier-or-string for extensions
    construct<OmpTraitSelectorName>(
        applyFunction(nameToString, Parser<Name>{})) ||
    construct<OmpTraitSelectorName>(space >> charLiteralConstantWithoutKind))))

// Parser for OmpTraitSelector::Properties
template <typename... PropParser>
static constexpr auto propertyListParser(PropParser... pp) {
  // Parse the property list "(score(expr): item1...)" in three steps:
  // 1. Parse the "("
  // 2. Parse the optional "score(expr):"
  // 3. Parse the "item1, ...)", together with the ")".
  // The reason for including the ")" in the 3rd step is to force parsing
  // the entire list in each of the alternative property parsers. Otherwise,
  // the name parser could stop after "foo" in "(foo, bar(1))", without
  // allowing the next parser to give the list a try.
  using P = OmpTraitProperty;
  return maybe("(" >> //
      construct<OmpTraitSelector::Properties>(
          maybe(Parser<OmpTraitScore>{} / ":"),
          (attempt(nonemptyList(sourced(construct<P>(pp))) / ")") || ...)));
}

// Parser for OmpTraitSelector
struct TraitSelectorParser {
  using resultType = OmpTraitSelector;

  constexpr TraitSelectorParser(Parser<OmpTraitSelectorName> p) : np(p) {}

  std::optional<resultType> Parse(ParseState &state) const {
    auto name{attempt(np).Parse(state)};
    if (!name.has_value()) {
      return std::nullopt;
    }

    // Default fallback parser for lists that cannot be parser using the
    // primary property parser.
    auto extParser{Parser<OmpTraitPropertyExtension>{}};

    if (auto *v{std::get_if<OmpTraitSelectorName::Value>(&name->u)}) {
      // (*) The comments below show the sections of the OpenMP spec that
      // describe given trait. The cases marked with a (*) are those where
      // the spec doesn't assign any list-type to these traits, but for
      // convenience they can be treated as if they were.
      switch (*v) {
      // name-list properties
      case OmpTraitSelectorName::Value::Arch: // [6.0:319:18]
      case OmpTraitSelectorName::Value::Extension: // [6.0:319:30]
      case OmpTraitSelectorName::Value::Isa: // [6.0:319:15]
      case OmpTraitSelectorName::Value::Kind: // [6.0:319:10]
      case OmpTraitSelectorName::Value::Uid: // [6.0:319:23](*)
      case OmpTraitSelectorName::Value::Vendor: { // [6.0:319:27]
        auto pp{propertyListParser(Parser<OmpTraitPropertyName>{}, extParser)};
        return OmpTraitSelector(std::move(*name), std::move(*pp.Parse(state)));
      }
      // clause-list
      case OmpTraitSelectorName::Value::Atomic_Default_Mem_Order:
        // [6.0:321:26-29](*)
      case OmpTraitSelectorName::Value::Requires: // [6.0:319:33]
      case OmpTraitSelectorName::Value::Simd: { // [6.0:318:31]
        auto pp{propertyListParser(indirect(Parser<OmpClause>{}), extParser)};
        return OmpTraitSelector(std::move(*name), std::move(*pp.Parse(state)));
      }
      // expr-list
      case OmpTraitSelectorName::Value::Condition: // [6.0:321:33](*)
      case OmpTraitSelectorName::Value::Device_Num: { // [6.0:321:23-24](*)
        auto pp{propertyListParser(scalarExpr, extParser)};
        return OmpTraitSelector(std::move(*name), std::move(*pp.Parse(state)));
      }
      } // switch
    } else {
      // The other alternatives are `llvm::omp::Directive`, and `std::string`.
      // The former doesn't take any properties[1], the latter is a name of an
      // extension[2].
      // [1] [6.0:319:1-2]
      // [2] [6.0:319:36-37]
      auto pp{propertyListParser(extParser)};
      return OmpTraitSelector(std::move(*name), std::move(*pp.Parse(state)));
    }

    llvm_unreachable("Unhandled trait name?");
  }

private:
  const Parser<OmpTraitSelectorName> np;
};

TYPE_PARSER(sourced(construct<OmpTraitSelector>(
    sourced(TraitSelectorParser(Parser<OmpTraitSelectorName>{})))))

TYPE_PARSER(construct<OmpTraitSetSelectorName::Value>(
    "CONSTRUCT"_id >> pure(OmpTraitSetSelectorName::Value::Construct) ||
    "DEVICE"_id >> pure(OmpTraitSetSelectorName::Value::Device) ||
    "IMPLEMENTATION"_id >>
        pure(OmpTraitSetSelectorName::Value::Implementation) ||
    "TARGET_DEVICE"_id >> pure(OmpTraitSetSelectorName::Value::Target_Device) ||
    "USER"_id >> pure(OmpTraitSetSelectorName::Value::User)))

TYPE_PARSER(sourced(construct<OmpTraitSetSelectorName>(
    Parser<OmpTraitSetSelectorName::Value>{})))

TYPE_PARSER(sourced(construct<OmpTraitSetSelector>( //
    Parser<OmpTraitSetSelectorName>{},
    "=" >> braced(nonemptySeparated(Parser<OmpTraitSelector>{}, ",")))))

TYPE_PARSER(sourced(construct<OmpContextSelectorSpecification>(
    nonemptySeparated(Parser<OmpTraitSetSelector>{}, ","))))

// Note: OmpContextSelector is a type alias.

// --- Parsers for clause modifiers -----------------------------------

TYPE_PARSER(construct<OmpAlignment>(scalarIntExpr))

TYPE_PARSER(construct<OmpAlignModifier>( //
    "ALIGN" >> parenthesized(scalarIntExpr)))

TYPE_PARSER(construct<OmpAllocatorComplexModifier>(
    "ALLOCATOR" >> parenthesized(scalarIntExpr)))

TYPE_PARSER(construct<OmpAllocatorSimpleModifier>(scalarIntExpr))

TYPE_PARSER(construct<OmpChunkModifier>( //
    "SIMD" >> pure(OmpChunkModifier::Value::Simd)))

TYPE_PARSER(construct<OmpDependenceType>(
    "SINK" >> pure(OmpDependenceType::Value::Sink) ||
    "SOURCE" >> pure(OmpDependenceType::Value::Source)))

TYPE_PARSER(construct<OmpDeviceModifier>(
    "ANCESTOR" >> pure(OmpDeviceModifier::Value::Ancestor) ||
    "DEVICE_NUM" >> pure(OmpDeviceModifier::Value::Device_Num)))

TYPE_PARSER(construct<OmpExpectation>( //
    "PRESENT" >> pure(OmpExpectation::Value::Present)))

TYPE_PARSER(construct<OmpIteratorSpecifier>(
    // Using Parser<TypeDeclarationStmt> or Parser<EntityDecl> has the problem
    // that they will attempt to treat what follows the '=' as initialization.
    // There are several issues with that,
    // 1. integer :: i = 0:10 will be parsed as "integer :: i = 0", followed
    // by triplet ":10".
    // 2. integer :: j = i:10 will be flagged as an error because the
    // initializer 'i' must be constant (in declarations). In an iterator
    // specifier the 'j' is not an initializer and can be a variable.
    (applyFunction<TypeDeclarationStmt>(makeIterSpecDecl,
         Parser<DeclarationTypeSpec>{} / maybe("::"_tok),
         nonemptyList(Parser<ObjectName>{}) / "="_tok) ||
        applyFunction<TypeDeclarationStmt>(
            makeIterSpecDecl, nonemptyList(Parser<ObjectName>{}) / "="_tok)),
    subscriptTriplet))

// [5.0] 2.1.6 iterator -> iterator-specifier-list
TYPE_PARSER(construct<OmpIterator>( //
    "ITERATOR" >>
    parenthesized(nonemptyList(sourced(Parser<OmpIteratorSpecifier>{})))))

TYPE_PARSER(construct<OmpLastprivateModifier>(
    "CONDITIONAL" >> pure(OmpLastprivateModifier::Value::Conditional)))

// 2.15.3.7 LINEAR (linear-list: linear-step)
//          linear-list -> list | modifier(list)
//          linear-modifier -> REF | VAL | UVAL
TYPE_PARSER(construct<OmpLinearModifier>( //
    "REF" >> pure(OmpLinearModifier::Value::Ref) ||
    "VAL" >> pure(OmpLinearModifier::Value::Val) ||
    "UVAL" >> pure(OmpLinearModifier::Value::Uval)))

TYPE_PARSER(construct<OmpMapper>( //
    "MAPPER"_tok >> parenthesized(Parser<ObjectName>{})))

// map-type -> ALLOC | DELETE | FROM | RELEASE | TO | TOFROM
TYPE_PARSER(construct<OmpMapType>( //
    "ALLOC" >> pure(OmpMapType::Value::Alloc) ||
    "DELETE" >> pure(OmpMapType::Value::Delete) ||
    "FROM" >> pure(OmpMapType::Value::From) ||
    "RELEASE" >> pure(OmpMapType::Value::Release) ||
    "TO"_id >> pure(OmpMapType::Value::To) ||
    "TOFROM" >> pure(OmpMapType::Value::Tofrom)))

// map-type-modifier -> ALWAYS | CLOSE | OMPX_HOLD | PRESENT
TYPE_PARSER(construct<OmpMapTypeModifier>(
    "ALWAYS" >> pure(OmpMapTypeModifier::Value::Always) ||
    "CLOSE" >> pure(OmpMapTypeModifier::Value::Close) ||
    "OMPX_HOLD" >> pure(OmpMapTypeModifier::Value::Ompx_Hold) ||
    "PRESENT" >> pure(OmpMapTypeModifier::Value::Present)))

// 2.15.3.6 REDUCTION (reduction-identifier: variable-name-list)
TYPE_PARSER(construct<OmpReductionIdentifier>(Parser<DefinedOperator>{}) ||
    construct<OmpReductionIdentifier>(Parser<ProcedureDesignator>{}))

TYPE_PARSER(construct<OmpOrderModifier>(
    "REPRODUCIBLE" >> pure(OmpOrderModifier::Value::Reproducible) ||
    "UNCONSTRAINED" >> pure(OmpOrderModifier::Value::Unconstrained)))

TYPE_PARSER(construct<OmpOrderingModifier>(
    "MONOTONIC" >> pure(OmpOrderingModifier::Value::Monotonic) ||
    "NONMONOTONIC" >> pure(OmpOrderingModifier::Value::Nonmonotonic) ||
    "SIMD" >> pure(OmpOrderingModifier::Value::Simd)))

TYPE_PARSER(construct<OmpPrescriptiveness>(
    "STRICT" >> pure(OmpPrescriptiveness::Value::Strict)))

TYPE_PARSER(construct<OmpReductionModifier>(
    "INSCAN" >> pure(OmpReductionModifier::Value::Inscan) ||
    "TASK" >> pure(OmpReductionModifier::Value::Task) ||
    "DEFAULT" >> pure(OmpReductionModifier::Value::Default)))

TYPE_PARSER(construct<OmpStepComplexModifier>( //
    "STEP" >> parenthesized(scalarIntExpr)))

TYPE_PARSER(construct<OmpStepSimpleModifier>(scalarIntExpr))

TYPE_PARSER(construct<OmpTaskDependenceType>(
    "DEPOBJ" >> pure(OmpTaskDependenceType::Value::Depobj) ||
    "IN"_id >> pure(OmpTaskDependenceType::Value::In) ||
    "INOUT"_id >> pure(OmpTaskDependenceType::Value::Inout) ||
    "INOUTSET"_id >> pure(OmpTaskDependenceType::Value::Inoutset) ||
    "MUTEXINOUTSET" >> pure(OmpTaskDependenceType::Value::Mutexinoutset) ||
    "OUT" >> pure(OmpTaskDependenceType::Value::Out)))

TYPE_PARSER(construct<OmpVariableCategory>(
    "AGGREGATE" >> pure(OmpVariableCategory::Value::Aggregate) ||
    "ALL"_id >> pure(OmpVariableCategory::Value::All) ||
    "ALLOCATABLE" >> pure(OmpVariableCategory::Value::Allocatable) ||
    "POINTER" >> pure(OmpVariableCategory::Value::Pointer) ||
    "SCALAR" >> pure(OmpVariableCategory::Value::Scalar)))

// This could be auto-generated.
TYPE_PARSER(
    sourced(construct<OmpAffinityClause::Modifier>(Parser<OmpIterator>{})))

TYPE_PARSER(
    sourced(construct<OmpAlignedClause::Modifier>(Parser<OmpAlignment>{})))

TYPE_PARSER(sourced(construct<OmpAllocateClause::Modifier>(sourced(
    construct<OmpAllocateClause::Modifier>(Parser<OmpAlignModifier>{}) ||
    construct<OmpAllocateClause::Modifier>(
        Parser<OmpAllocatorComplexModifier>{}) ||
    construct<OmpAllocateClause::Modifier>(
        Parser<OmpAllocatorSimpleModifier>{})))))

TYPE_PARSER(sourced(
    construct<OmpDefaultmapClause::Modifier>(Parser<OmpVariableCategory>{})))

TYPE_PARSER(sourced(construct<OmpDependClause::TaskDep::Modifier>(sourced(
    construct<OmpDependClause::TaskDep::Modifier>(Parser<OmpIterator>{}) ||
    construct<OmpDependClause::TaskDep::Modifier>(
        Parser<OmpTaskDependenceType>{})))))

TYPE_PARSER(
    sourced(construct<OmpDeviceClause::Modifier>(Parser<OmpDeviceModifier>{})))

TYPE_PARSER(sourced(construct<OmpFromClause::Modifier>(
    sourced(construct<OmpFromClause::Modifier>(Parser<OmpExpectation>{}) ||
        construct<OmpFromClause::Modifier>(Parser<OmpMapper>{}) ||
        construct<OmpFromClause::Modifier>(Parser<OmpIterator>{})))))

TYPE_PARSER(sourced(
    construct<OmpGrainsizeClause::Modifier>(Parser<OmpPrescriptiveness>{})))

TYPE_PARSER(sourced(construct<OmpIfClause::Modifier>(OmpDirectiveNameParser{})))

TYPE_PARSER(sourced(construct<OmpInReductionClause::Modifier>(
    Parser<OmpReductionIdentifier>{})))

TYPE_PARSER(sourced(construct<OmpLastprivateClause::Modifier>(
    Parser<OmpLastprivateModifier>{})))

TYPE_PARSER(sourced(
    construct<OmpLinearClause::Modifier>(Parser<OmpLinearModifier>{}) ||
    construct<OmpLinearClause::Modifier>(Parser<OmpStepComplexModifier>{}) ||
    construct<OmpLinearClause::Modifier>(Parser<OmpStepSimpleModifier>{})))

TYPE_PARSER(sourced(construct<OmpMapClause::Modifier>(
    sourced(construct<OmpMapClause::Modifier>(Parser<OmpMapTypeModifier>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpMapper>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpIterator>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpMapType>{})))))

TYPE_PARSER(
    sourced(construct<OmpOrderClause::Modifier>(Parser<OmpOrderModifier>{})))

TYPE_PARSER(sourced(
    construct<OmpNumTasksClause::Modifier>(Parser<OmpPrescriptiveness>{})))

TYPE_PARSER(sourced(construct<OmpReductionClause::Modifier>(sourced(
    construct<OmpReductionClause::Modifier>(Parser<OmpReductionModifier>{}) ||
    construct<OmpReductionClause::Modifier>(
        Parser<OmpReductionIdentifier>{})))))

TYPE_PARSER(sourced(construct<OmpScheduleClause::Modifier>(sourced(
    construct<OmpScheduleClause::Modifier>(Parser<OmpChunkModifier>{}) ||
    construct<OmpScheduleClause::Modifier>(Parser<OmpOrderingModifier>{})))))

TYPE_PARSER(sourced(construct<OmpTaskReductionClause::Modifier>(
    Parser<OmpReductionIdentifier>{})))

TYPE_PARSER(sourced(construct<OmpToClause::Modifier>(
    sourced(construct<OmpToClause::Modifier>(Parser<OmpExpectation>{}) ||
        construct<OmpToClause::Modifier>(Parser<OmpMapper>{}) ||
        construct<OmpToClause::Modifier>(Parser<OmpIterator>{})))))

TYPE_PARSER(sourced(construct<OmpWhenClause::Modifier>( //
    Parser<OmpContextSelector>{})))

// --- Parsers for clauses --------------------------------------------

/// `MOBClause` is a clause that has a
///   std::tuple<Modifiers, OmpObjectList, bool>.
/// Helper function to create a typical modifiers-objects clause, where the
/// commas separating individual modifiers are optional, and the clause
/// contains a bool member to indicate whether it was fully comma-separated
/// or not.
template <bool CommaSeparated, typename MOBClause>
static inline MOBClause makeMobClause(
    std::list<typename MOBClause::Modifier> &&mods, OmpObjectList &&objs) {
  if (!mods.empty()) {
    return MOBClause{std::move(mods), std::move(objs), CommaSeparated};
  } else {
    using ListTy = std::list<typename MOBClause::Modifier>;
    return MOBClause{std::optional<ListTy>{}, std::move(objs), CommaSeparated};
  }
}

// [5.0] 2.10.1 affinity([aff-modifier:] locator-list)
//              aff-modifier: interator-modifier
TYPE_PARSER(construct<OmpAffinityClause>(
    maybe(nonemptyList(Parser<OmpAffinityClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

// 2.15.3.1 DEFAULT (PRIVATE | FIRSTPRIVATE | SHARED | NONE)
TYPE_PARSER(construct<OmpDefaultClause::DataSharingAttribute>(
    "PRIVATE" >> pure(OmpDefaultClause::DataSharingAttribute::Private) ||
    "FIRSTPRIVATE" >>
        pure(OmpDefaultClause::DataSharingAttribute::Firstprivate) ||
    "SHARED" >> pure(OmpDefaultClause::DataSharingAttribute::Shared) ||
    "NONE" >> pure(OmpDefaultClause::DataSharingAttribute::None)))

TYPE_PARSER(construct<OmpDefaultClause>(
    construct<OmpDefaultClause>(
        Parser<OmpDefaultClause::DataSharingAttribute>{}) ||
    construct<OmpDefaultClause>(indirect(Parser<OmpDirectiveSpecification>{}))))

// 2.5 PROC_BIND (MASTER | CLOSE | PRIMARY | SPREAD)
TYPE_PARSER(construct<OmpProcBindClause>(
    "CLOSE" >> pure(OmpProcBindClause::AffinityPolicy::Close) ||
    "MASTER" >> pure(OmpProcBindClause::AffinityPolicy::Master) ||
    "PRIMARY" >> pure(OmpProcBindClause::AffinityPolicy::Primary) ||
    "SPREAD" >> pure(OmpProcBindClause::AffinityPolicy::Spread)))

TYPE_PARSER(construct<OmpMapClause>(
    applyFunction<OmpMapClause>(makeMobClause<true>,
        modifierList<OmpMapClause>(","_tok), Parser<OmpObjectList>{}) ||
    applyFunction<OmpMapClause>(makeMobClause<false>,
        modifierList<OmpMapClause>(maybe(","_tok)), Parser<OmpObjectList>{})))

// [OpenMP 5.0]
// 2.19.7.2 defaultmap(implicit-behavior[:variable-category])
//  implicit-behavior -> ALLOC | TO | FROM | TOFROM | FIRSRTPRIVATE | NONE |
//  DEFAULT
//  variable-category -> ALL | SCALAR | AGGREGATE | ALLOCATABLE | POINTER
TYPE_PARSER(construct<OmpDefaultmapClause>(
    construct<OmpDefaultmapClause::ImplicitBehavior>(
        "ALLOC" >> pure(OmpDefaultmapClause::ImplicitBehavior::Alloc) ||
        "TO"_id >> pure(OmpDefaultmapClause::ImplicitBehavior::To) ||
        "FROM" >> pure(OmpDefaultmapClause::ImplicitBehavior::From) ||
        "TOFROM" >> pure(OmpDefaultmapClause::ImplicitBehavior::Tofrom) ||
        "FIRSTPRIVATE" >>
            pure(OmpDefaultmapClause::ImplicitBehavior::Firstprivate) ||
        "NONE" >> pure(OmpDefaultmapClause::ImplicitBehavior::None) ||
        "DEFAULT" >> pure(OmpDefaultmapClause::ImplicitBehavior::Default)),
    maybe(":" >> nonemptyList(Parser<OmpDefaultmapClause::Modifier>{}))))

TYPE_PARSER(construct<OmpScheduleClause::Kind>(
    "STATIC" >> pure(OmpScheduleClause::Kind::Static) ||
    "DYNAMIC" >> pure(OmpScheduleClause::Kind::Dynamic) ||
    "GUIDED" >> pure(OmpScheduleClause::Kind::Guided) ||
    "AUTO" >> pure(OmpScheduleClause::Kind::Auto) ||
    "RUNTIME" >> pure(OmpScheduleClause::Kind::Runtime)))

TYPE_PARSER(construct<OmpScheduleClause>(
    maybe(nonemptyList(Parser<OmpScheduleClause::Modifier>{}) / ":"),
    Parser<OmpScheduleClause::Kind>{}, maybe("," >> scalarIntExpr)))

// device([ device-modifier :] scalar-integer-expression)
TYPE_PARSER(construct<OmpDeviceClause>(
    maybe(nonemptyList(Parser<OmpDeviceClause::Modifier>{}) / ":"),
    scalarIntExpr))

// device_type(any | host | nohost)
TYPE_PARSER(construct<OmpDeviceTypeClause>(
    "ANY" >> pure(OmpDeviceTypeClause::DeviceTypeDescription::Any) ||
    "HOST" >> pure(OmpDeviceTypeClause::DeviceTypeDescription::Host) ||
    "NOHOST" >> pure(OmpDeviceTypeClause::DeviceTypeDescription::Nohost)))

// 2.12 IF (directive-name-modifier: scalar-logical-expr)
TYPE_PARSER(construct<OmpIfClause>(
    maybe(nonemptyList(Parser<OmpIfClause::Modifier>{}) / ":"),
    scalarLogicalExpr))

TYPE_PARSER(construct<OmpReductionClause>(
    maybe(nonemptyList(Parser<OmpReductionClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

// OMP 5.0 2.19.5.6 IN_REDUCTION (reduction-identifier: variable-name-list)
TYPE_PARSER(construct<OmpInReductionClause>(
    maybe(nonemptyList(Parser<OmpInReductionClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

TYPE_PARSER(construct<OmpTaskReductionClause>(
    maybe(nonemptyList(Parser<OmpTaskReductionClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

// OMP 5.0 2.11.4 allocate-clause -> ALLOCATE ([allocator:] variable-name-list)
// OMP 5.2 2.13.4 allocate-clause -> ALLOCATE ([allocate-modifier
//                                   [, allocate-modifier] :]
//                                   variable-name-list)
//                allocate-modifier -> allocator | align
TYPE_PARSER(construct<OmpAllocateClause>(
    maybe(nonemptyList(Parser<OmpAllocateClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

// iteration-offset -> +/- non-negative-constant-expr
TYPE_PARSER(construct<OmpIterationOffset>(
    Parser<DefinedOperator>{}, scalarIntConstantExpr))

// iteration -> iteration-variable [+/- nonnegative-scalar-integer-constant]
TYPE_PARSER(construct<OmpIteration>(name, maybe(Parser<OmpIterationOffset>{})))

TYPE_PARSER(construct<OmpIterationVector>(nonemptyList(Parser<OmpIteration>{})))

TYPE_PARSER(construct<OmpDoacross>(
    construct<OmpDoacross>(construct<OmpDoacross::Sink>(
        "SINK"_tok >> ":"_tok >> Parser<OmpIterationVector>{})) ||
    construct<OmpDoacross>(construct<OmpDoacross::Source>("SOURCE"_tok))))

TYPE_CONTEXT_PARSER("Omp Depend clause"_en_US,
    construct<OmpDependClause>(
        // Try to parse OmpDoacross first, because TaskDep will succeed on
        // "sink: xxx", interpreting it to not have any modifiers, and "sink"
        // being an OmpObject. Parsing of the TaskDep variant will stop right
        // after the "sink", leaving the ": xxx" unvisited.
        construct<OmpDependClause>(Parser<OmpDoacross>{}) ||
        // Parse TaskDep after Doacross.
        construct<OmpDependClause>(construct<OmpDependClause::TaskDep>(
            maybe(nonemptyList(Parser<OmpDependClause::TaskDep::Modifier>{}) /
                ": "),
            Parser<OmpObjectList>{}))))

TYPE_CONTEXT_PARSER("Omp Doacross clause"_en_US,
    construct<OmpDoacrossClause>(Parser<OmpDoacross>{}))

TYPE_PARSER(construct<OmpFromClause>(
    applyFunction<OmpFromClause>(makeMobClause<true>,
        modifierList<OmpFromClause>(","_tok), Parser<OmpObjectList>{}) ||
    applyFunction<OmpFromClause>(makeMobClause<false>,
        modifierList<OmpFromClause>(maybe(","_tok)), Parser<OmpObjectList>{})))

TYPE_PARSER(construct<OmpToClause>(
    applyFunction<OmpToClause>(makeMobClause<true>,
        modifierList<OmpToClause>(","_tok), Parser<OmpObjectList>{}) ||
    applyFunction<OmpToClause>(makeMobClause<false>,
        modifierList<OmpToClause>(maybe(","_tok)), Parser<OmpObjectList>{})))

OmpLinearClause makeLinearFromOldSyntax(OmpLinearClause::Modifier &&lm,
    OmpObjectList &&objs, std::optional<OmpLinearClause::Modifier> &&ssm) {
  std::list<OmpLinearClause::Modifier> mods;
  mods.emplace_back(std::move(lm));
  if (ssm) {
    mods.emplace_back(std::move(*ssm));
  }
  return OmpLinearClause{std::move(objs),
      mods.empty() ? decltype(mods){} : std::move(mods),
      /*PostModified=*/false};
}

TYPE_PARSER(
    // Parse the "modifier(x)" first, because syntacticaly it will match
    // an array element (i.e. a list item).
    // LINEAR(linear-modifier(list) [: step-simple-modifier])
    construct<OmpLinearClause>( //
        applyFunction<OmpLinearClause>(makeLinearFromOldSyntax,
            SpecificModifierParser<OmpLinearModifier, OmpLinearClause>{},
            parenthesized(Parser<OmpObjectList>{}),
            maybe(":"_tok >> SpecificModifierParser<OmpStepSimpleModifier,
                                 OmpLinearClause>{}))) ||
    // LINEAR(list [: modifiers])
    construct<OmpLinearClause>( //
        Parser<OmpObjectList>{},
        maybe(":"_tok >> nonemptyList(Parser<OmpLinearClause::Modifier>{})),
        /*PostModified=*/pure(true)))

// OpenMPv5.2 12.5.2 detach-clause -> DETACH (event-handle)
TYPE_PARSER(construct<OmpDetachClause>(Parser<OmpObject>{}))

// 2.8.1 ALIGNED (list: alignment)
TYPE_PARSER(construct<OmpAlignedClause>(Parser<OmpObjectList>{},
    maybe(":" >> nonemptyList(Parser<OmpAlignedClause::Modifier>{}))))

TYPE_PARSER(construct<OmpUpdateClause>(
    construct<OmpUpdateClause>(Parser<OmpDependenceType>{}) ||
    construct<OmpUpdateClause>(Parser<OmpTaskDependenceType>{})))

TYPE_PARSER(construct<OmpOrderClause>(
    maybe(nonemptyList(Parser<OmpOrderClause::Modifier>{}) / ":"),
    "CONCURRENT" >> pure(OmpOrderClause::Ordering::Concurrent)))

TYPE_PARSER(construct<OmpMatchClause>(
    Parser<traits::OmpContextSelectorSpecification>{}))

TYPE_PARSER(construct<OmpOtherwiseClause>(
    maybe(indirect(sourced(Parser<OmpDirectiveSpecification>{})))))

TYPE_PARSER(construct<OmpWhenClause>(
    maybe(nonemptyList(Parser<OmpWhenClause::Modifier>{}) / ":"),
    maybe(indirect(sourced(Parser<OmpDirectiveSpecification>{})))))

// OMP 5.2 12.6.1 grainsize([ prescriptiveness :] scalar-integer-expression)
TYPE_PARSER(construct<OmpGrainsizeClause>(
    maybe(nonemptyList(Parser<OmpGrainsizeClause::Modifier>{}) / ":"),
    scalarIntExpr))

// OMP 5.2 12.6.2 num_tasks([ prescriptiveness :] scalar-integer-expression)
TYPE_PARSER(construct<OmpNumTasksClause>(
    maybe(nonemptyList(Parser<OmpNumTasksClause::Modifier>{}) / ":"),
    scalarIntExpr))

TYPE_PARSER(
    construct<OmpObject>(designator) || construct<OmpObject>("/" >> name / "/"))

// OMP 5.0 2.19.4.5 LASTPRIVATE ([lastprivate-modifier :] list)
TYPE_PARSER(construct<OmpLastprivateClause>(
    maybe(nonemptyList(Parser<OmpLastprivateClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

// OMP 5.2 11.7.1 BIND ( PARALLEL | TEAMS | THREAD )
TYPE_PARSER(construct<OmpBindClause>(
    "PARALLEL" >> pure(OmpBindClause::Binding::Parallel) ||
    "TEAMS" >> pure(OmpBindClause::Binding::Teams) ||
    "THREAD" >> pure(OmpBindClause::Binding::Thread)))

TYPE_PARSER(construct<OmpAlignClause>(scalarIntExpr))

TYPE_PARSER(construct<OmpAtClause>(
    "EXECUTION" >> pure(OmpAtClause::ActionTime::Execution) ||
    "COMPILATION" >> pure(OmpAtClause::ActionTime::Compilation)))

TYPE_PARSER(construct<OmpSeverityClause>(
    "FATAL" >> pure(OmpSeverityClause::Severity::Fatal) ||
    "WARNING" >> pure(OmpSeverityClause::Severity::Warning)))

TYPE_PARSER(construct<OmpMessageClause>(expr))

TYPE_PARSER(
    "ACQUIRE" >> construct<OmpClause>(construct<OmpClause::Acquire>()) ||
    "ACQ_REL" >> construct<OmpClause>(construct<OmpClause::AcqRel>()) ||
    "AFFINITY" >> construct<OmpClause>(construct<OmpClause::Affinity>(
                      parenthesized(Parser<OmpAffinityClause>{}))) ||
    "ALIGN" >> construct<OmpClause>(construct<OmpClause::Align>(
                   parenthesized(Parser<OmpAlignClause>{}))) ||
    "ALIGNED" >> construct<OmpClause>(construct<OmpClause::Aligned>(
                     parenthesized(Parser<OmpAlignedClause>{}))) ||
    "ALLOCATE" >> construct<OmpClause>(construct<OmpClause::Allocate>(
                      parenthesized(Parser<OmpAllocateClause>{}))) ||
    "ALLOCATOR" >> construct<OmpClause>(construct<OmpClause::Allocator>(
                       parenthesized(scalarIntExpr))) ||
    "AT" >> construct<OmpClause>(construct<OmpClause::At>(
                parenthesized(Parser<OmpAtClause>{}))) ||
    "ATOMIC_DEFAULT_MEM_ORDER" >>
        construct<OmpClause>(construct<OmpClause::AtomicDefaultMemOrder>(
            parenthesized(Parser<OmpAtomicDefaultMemOrderClause>{}))) ||
    "BIND" >> construct<OmpClause>(construct<OmpClause::Bind>(
                  parenthesized(Parser<OmpBindClause>{}))) ||
    "COLLAPSE" >> construct<OmpClause>(construct<OmpClause::Collapse>(
                      parenthesized(scalarIntConstantExpr))) ||
    "COPYIN" >> construct<OmpClause>(construct<OmpClause::Copyin>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "COPYPRIVATE" >> construct<OmpClause>(construct<OmpClause::Copyprivate>(
                         (parenthesized(Parser<OmpObjectList>{})))) ||
    "DEFAULT"_id >> construct<OmpClause>(construct<OmpClause::Default>(
                        parenthesized(Parser<OmpDefaultClause>{}))) ||
    "DEFAULTMAP" >> construct<OmpClause>(construct<OmpClause::Defaultmap>(
                        parenthesized(Parser<OmpDefaultmapClause>{}))) ||
    "DEPEND" >> construct<OmpClause>(construct<OmpClause::Depend>(
                    parenthesized(Parser<OmpDependClause>{}))) ||
    "DESTROY" >>
        construct<OmpClause>(construct<OmpClause::Destroy>(maybe(parenthesized(
            construct<OmpDestroyClause>(Parser<OmpObject>{}))))) ||
    "DEVICE" >> construct<OmpClause>(construct<OmpClause::Device>(
                    parenthesized(Parser<OmpDeviceClause>{}))) ||
    "DEVICE_TYPE" >> construct<OmpClause>(construct<OmpClause::DeviceType>(
                         parenthesized(Parser<OmpDeviceTypeClause>{}))) ||
    "DIST_SCHEDULE" >>
        construct<OmpClause>(construct<OmpClause::DistSchedule>(
            parenthesized("STATIC" >> maybe("," >> scalarIntExpr)))) ||
    "DOACROSS" >>
        construct<OmpClause>(parenthesized(Parser<OmpDoacrossClause>{})) ||
    "DYNAMIC_ALLOCATORS" >>
        construct<OmpClause>(construct<OmpClause::DynamicAllocators>()) ||
    "ENTER" >> construct<OmpClause>(construct<OmpClause::Enter>(
                   parenthesized(Parser<OmpObjectList>{}))) ||
    "EXCLUSIVE" >> construct<OmpClause>(construct<OmpClause::Exclusive>(
                       parenthesized(Parser<OmpObjectList>{}))) ||
    "FILTER" >> construct<OmpClause>(construct<OmpClause::Filter>(
                    parenthesized(scalarIntExpr))) ||
    "FINAL" >> construct<OmpClause>(construct<OmpClause::Final>(
                   parenthesized(scalarLogicalExpr))) ||
    "FIRSTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Firstprivate>(
                          parenthesized(Parser<OmpObjectList>{}))) ||
    "FROM" >> construct<OmpClause>(construct<OmpClause::From>(
                  parenthesized(Parser<OmpFromClause>{}))) ||
    "FULL" >> construct<OmpClause>(construct<OmpClause::Full>()) ||
    "GRAINSIZE" >> construct<OmpClause>(construct<OmpClause::Grainsize>(
                       parenthesized(Parser<OmpGrainsizeClause>{}))) ||
    "HAS_DEVICE_ADDR" >>
        construct<OmpClause>(construct<OmpClause::HasDeviceAddr>(
            parenthesized(Parser<OmpObjectList>{}))) ||
    "HINT" >> construct<OmpClause>(
                  construct<OmpClause::Hint>(parenthesized(constantExpr))) ||
    "IF" >> construct<OmpClause>(construct<OmpClause::If>(
                parenthesized(Parser<OmpIfClause>{}))) ||
    "INBRANCH" >> construct<OmpClause>(construct<OmpClause::Inbranch>()) ||
    "INCLUSIVE" >> construct<OmpClause>(construct<OmpClause::Inclusive>(
                       parenthesized(Parser<OmpObjectList>{}))) ||
    "IS_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::IsDevicePtr>(
                           parenthesized(Parser<OmpObjectList>{}))) ||
    "LASTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Lastprivate>(
                         parenthesized(Parser<OmpLastprivateClause>{}))) ||
    "LINEAR" >> construct<OmpClause>(construct<OmpClause::Linear>(
                    parenthesized(Parser<OmpLinearClause>{}))) ||
    "LINK" >> construct<OmpClause>(construct<OmpClause::Link>(
                  parenthesized(Parser<OmpObjectList>{}))) ||
    "MAP" >> construct<OmpClause>(construct<OmpClause::Map>(
                 parenthesized(Parser<OmpMapClause>{}))) ||
    "MATCH" >> construct<OmpClause>(construct<OmpClause::Match>(
                   parenthesized(Parser<OmpMatchClause>{}))) ||
    "MERGEABLE" >> construct<OmpClause>(construct<OmpClause::Mergeable>()) ||
    "MESSAGE" >> construct<OmpClause>(construct<OmpClause::Message>(
                     parenthesized(Parser<OmpMessageClause>{}))) ||
    "NOCONTEXT" >> construct<OmpClause>(construct<OmpClause::Nocontext>(
                       parenthesized(scalarLogicalExpr))) ||
    "NOGROUP" >> construct<OmpClause>(construct<OmpClause::Nogroup>()) ||
    "NONTEMPORAL" >> construct<OmpClause>(construct<OmpClause::Nontemporal>(
                         parenthesized(nonemptyList(name)))) ||
    "NOTINBRANCH" >>
        construct<OmpClause>(construct<OmpClause::Notinbranch>()) ||
    "NOVARIANTS" >> construct<OmpClause>(construct<OmpClause::Novariants>(
                        parenthesized(scalarLogicalExpr))) ||
    "NOWAIT" >> construct<OmpClause>(construct<OmpClause::Nowait>()) ||
    "NUM_TASKS" >> construct<OmpClause>(construct<OmpClause::NumTasks>(
                       parenthesized(Parser<OmpNumTasksClause>{}))) ||
    "NUM_TEAMS" >> construct<OmpClause>(construct<OmpClause::NumTeams>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_THREADS" >> construct<OmpClause>(construct<OmpClause::NumThreads>(
                         parenthesized(scalarIntExpr))) ||
    "OMPX_BARE" >> construct<OmpClause>(construct<OmpClause::OmpxBare>()) ||
    "ORDER" >> construct<OmpClause>(construct<OmpClause::Order>(
                   parenthesized(Parser<OmpOrderClause>{}))) ||
    "ORDERED" >> construct<OmpClause>(construct<OmpClause::Ordered>(
                     maybe(parenthesized(scalarIntConstantExpr)))) ||
    "OTHERWISE" >> construct<OmpClause>(construct<OmpClause::Otherwise>(
                       maybe(parenthesized(Parser<OmpOtherwiseClause>{})))) ||
    "PARTIAL" >> construct<OmpClause>(construct<OmpClause::Partial>(
                     maybe(parenthesized(scalarIntConstantExpr)))) ||
    "PRIORITY" >> construct<OmpClause>(construct<OmpClause::Priority>(
                      parenthesized(scalarIntExpr))) ||
    "PRIVATE" >> construct<OmpClause>(construct<OmpClause::Private>(
                     parenthesized(Parser<OmpObjectList>{}))) ||
    "PROC_BIND" >> construct<OmpClause>(construct<OmpClause::ProcBind>(
                       parenthesized(Parser<OmpProcBindClause>{}))) ||
    "REDUCTION"_id >> construct<OmpClause>(construct<OmpClause::Reduction>(
                          parenthesized(Parser<OmpReductionClause>{}))) ||
    "IN_REDUCTION" >> construct<OmpClause>(construct<OmpClause::InReduction>(
                          parenthesized(Parser<OmpInReductionClause>{}))) ||
    "DETACH" >> construct<OmpClause>(construct<OmpClause::Detach>(
                    parenthesized(Parser<OmpDetachClause>{}))) ||
    "TASK_REDUCTION" >>
        construct<OmpClause>(construct<OmpClause::TaskReduction>(
            parenthesized(Parser<OmpTaskReductionClause>{}))) ||
    "RELAXED" >> construct<OmpClause>(construct<OmpClause::Relaxed>()) ||
    "RELEASE" >> construct<OmpClause>(construct<OmpClause::Release>()) ||
    "REVERSE_OFFLOAD" >>
        construct<OmpClause>(construct<OmpClause::ReverseOffload>()) ||
    "SAFELEN" >> construct<OmpClause>(construct<OmpClause::Safelen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "SCHEDULE" >> construct<OmpClause>(construct<OmpClause::Schedule>(
                      parenthesized(Parser<OmpScheduleClause>{}))) ||
    "SEQ_CST" >> construct<OmpClause>(construct<OmpClause::SeqCst>()) ||
    "SEVERITY" >> construct<OmpClause>(construct<OmpClause::Severity>(
                      parenthesized(Parser<OmpSeverityClause>{}))) ||
    "SHARED" >> construct<OmpClause>(construct<OmpClause::Shared>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "SIMD"_id >> construct<OmpClause>(construct<OmpClause::Simd>()) ||
    "SIMDLEN" >> construct<OmpClause>(construct<OmpClause::Simdlen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "SIZES" >> construct<OmpClause>(construct<OmpClause::Sizes>(
                   parenthesized(nonemptyList(scalarIntExpr)))) ||
    "PERMUTATION" >> construct<OmpClause>(construct<OmpClause::Permutation>(
                         parenthesized(nonemptyList(scalarIntExpr)))) ||
    "THREADS" >> construct<OmpClause>(construct<OmpClause::Threads>()) ||
    "THREAD_LIMIT" >> construct<OmpClause>(construct<OmpClause::ThreadLimit>(
                          parenthesized(scalarIntExpr))) ||
    "TO" >> construct<OmpClause>(construct<OmpClause::To>(
                parenthesized(Parser<OmpToClause>{}))) ||
    "USE_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::UseDevicePtr>(
                            parenthesized(Parser<OmpObjectList>{}))) ||
    "USE_DEVICE_ADDR" >>
        construct<OmpClause>(construct<OmpClause::UseDeviceAddr>(
            parenthesized(Parser<OmpObjectList>{}))) ||
    "UNIFIED_ADDRESS" >>
        construct<OmpClause>(construct<OmpClause::UnifiedAddress>()) ||
    "UNIFIED_SHARED_MEMORY" >>
        construct<OmpClause>(construct<OmpClause::UnifiedSharedMemory>()) ||
    "UNIFORM" >> construct<OmpClause>(construct<OmpClause::Uniform>(
                     parenthesized(nonemptyList(name)))) ||
    "UNTIED" >> construct<OmpClause>(construct<OmpClause::Untied>()) ||
    "UPDATE" >> construct<OmpClause>(construct<OmpClause::Update>(
                    parenthesized(Parser<OmpUpdateClause>{}))) ||
    "WHEN" >> construct<OmpClause>(construct<OmpClause::When>(
                  parenthesized(Parser<OmpWhenClause>{}))))

// [Clause, [Clause], ...]
TYPE_PARSER(sourced(construct<OmpClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OmpClause>{})))))

// 2.1 (variable | /common-block/ | array-sections)
TYPE_PARSER(construct<OmpObjectList>(nonemptyList(Parser<OmpObject>{})))

TYPE_PARSER(sourced(construct<OmpErrorDirective>(
    verbatim("ERROR"_tok), Parser<OmpClauseList>{})))

// --- Parsers for directives and constructs --------------------------

TYPE_PARSER(sourced(construct<OmpDirectiveSpecification>( //
    OmpDirectiveNameParser{},
    maybe(parenthesized(nonemptyList(Parser<OmpArgument>{}))),
    maybe(Parser<OmpClauseList>{}))))

TYPE_PARSER(sourced(construct<OmpNothingDirective>("NOTHING" >> ok)))

TYPE_PARSER(sourced(construct<OpenMPUtilityConstruct>(
    sourced(construct<OpenMPUtilityConstruct>(
        sourced(Parser<OmpErrorDirective>{}))) ||
    sourced(construct<OpenMPUtilityConstruct>(
        sourced(Parser<OmpNothingDirective>{}))))))

TYPE_PARSER(sourced(construct<OmpMetadirectiveDirective>(
    "METADIRECTIVE" >> Parser<OmpClauseList>{})))

// Omp directives enclosing do loop
TYPE_PARSER(sourced(construct<OmpLoopDirective>(first(
    "DISTRIBUTE PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::OMPD_distribute_parallel_do_simd),
    "DISTRIBUTE PARALLEL DO" >>
        pure(llvm::omp::Directive::OMPD_distribute_parallel_do),
    "DISTRIBUTE SIMD" >> pure(llvm::omp::Directive::OMPD_distribute_simd),
    "DISTRIBUTE" >> pure(llvm::omp::Directive::OMPD_distribute),
    "DO SIMD" >> pure(llvm::omp::Directive::OMPD_do_simd),
    "DO" >> pure(llvm::omp::Directive::OMPD_do),
    "LOOP" >> pure(llvm::omp::Directive::OMPD_loop),
    "MASKED TASKLOOP SIMD" >>
        pure(llvm::omp::Directive::OMPD_masked_taskloop_simd),
    "MASKED TASKLOOP" >> pure(llvm::omp::Directive::OMPD_masked_taskloop),
    "MASTER TASKLOOP SIMD" >>
        pure(llvm::omp::Directive::OMPD_master_taskloop_simd),
    "MASTER TASKLOOP" >> pure(llvm::omp::Directive::OMPD_master_taskloop),
    "PARALLEL DO SIMD" >> pure(llvm::omp::Directive::OMPD_parallel_do_simd),
    "PARALLEL DO" >> pure(llvm::omp::Directive::OMPD_parallel_do),
    "PARALLEL MASKED TASKLOOP SIMD" >>
        pure(llvm::omp::Directive::OMPD_parallel_masked_taskloop_simd),
    "PARALLEL MASKED TASKLOOP" >>
        pure(llvm::omp::Directive::OMPD_parallel_masked_taskloop),
    "PARALLEL MASTER TASKLOOP SIMD" >>
        pure(llvm::omp::Directive::OMPD_parallel_master_taskloop_simd),
    "PARALLEL MASTER TASKLOOP" >>
        pure(llvm::omp::Directive::OMPD_parallel_master_taskloop),
    "SIMD" >> pure(llvm::omp::Directive::OMPD_simd),
    "TARGET LOOP" >> pure(llvm::omp::Directive::OMPD_target_loop),
    "TARGET PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::OMPD_target_parallel_do_simd),
    "TARGET PARALLEL DO" >> pure(llvm::omp::Directive::OMPD_target_parallel_do),
    "TARGET PARALLEL LOOP" >>
        pure(llvm::omp::Directive::OMPD_target_parallel_loop),
    "TARGET SIMD" >> pure(llvm::omp::Directive::OMPD_target_simd),
    "TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::
                OMPD_target_teams_distribute_parallel_do_simd),
    "TARGET TEAMS DISTRIBUTE PARALLEL DO" >>
        pure(llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do),
    "TARGET TEAMS DISTRIBUTE SIMD" >>
        pure(llvm::omp::Directive::OMPD_target_teams_distribute_simd),
    "TARGET TEAMS DISTRIBUTE" >>
        pure(llvm::omp::Directive::OMPD_target_teams_distribute),
    "TARGET TEAMS LOOP" >> pure(llvm::omp::Directive::OMPD_target_teams_loop),
    "TASKLOOP SIMD" >> pure(llvm::omp::Directive::OMPD_taskloop_simd),
    "TASKLOOP" >> pure(llvm::omp::Directive::OMPD_taskloop),
    "TEAMS DISTRIBUTE PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd),
    "TEAMS DISTRIBUTE PARALLEL DO" >>
        pure(llvm::omp::Directive::OMPD_teams_distribute_parallel_do),
    "TEAMS DISTRIBUTE SIMD" >>
        pure(llvm::omp::Directive::OMPD_teams_distribute_simd),
    "TEAMS DISTRIBUTE" >> pure(llvm::omp::Directive::OMPD_teams_distribute),
    "TEAMS LOOP" >> pure(llvm::omp::Directive::OMPD_teams_loop),
    "TILE" >> pure(llvm::omp::Directive::OMPD_tile),
    "UNROLL" >> pure(llvm::omp::Directive::OMPD_unroll)))))

TYPE_PARSER(sourced(construct<OmpBeginLoopDirective>(
    sourced(Parser<OmpLoopDirective>{}), Parser<OmpClauseList>{})))

// 2.14.1 construct-type-clause -> PARALLEL | SECTIONS | DO | TASKGROUP
TYPE_PARSER(sourced(construct<OmpCancelType>(
    first("PARALLEL" >> pure(OmpCancelType::Type::Parallel),
        "SECTIONS" >> pure(OmpCancelType::Type::Sections),
        "DO" >> pure(OmpCancelType::Type::Do),
        "TASKGROUP" >> pure(OmpCancelType::Type::Taskgroup)))))

// 2.14.2 Cancellation Point construct
TYPE_PARSER(sourced(construct<OpenMPCancellationPointConstruct>(
    verbatim("CANCELLATION POINT"_tok), Parser<OmpCancelType>{})))

// 2.14.1 Cancel construct
TYPE_PARSER(sourced(construct<OpenMPCancelConstruct>(verbatim("CANCEL"_tok),
    Parser<OmpCancelType>{}, maybe("IF" >> parenthesized(scalarLogicalExpr)))))

TYPE_PARSER(sourced(construct<OmpFailClause>(
    parenthesized(indirect(Parser<OmpMemoryOrderClause>{})))))

// 2.17.7 Atomic construct/2.17.8 Flush construct [OpenMP 5.0]
//        memory-order-clause ->
//                               seq_cst
//                               acq_rel
//                               release
//                               acquire
//                               relaxed
TYPE_PARSER(sourced(construct<OmpMemoryOrderClause>(
    sourced("SEQ_CST" >> construct<OmpClause>(construct<OmpClause::SeqCst>()) ||
        "ACQ_REL" >> construct<OmpClause>(construct<OmpClause::AcqRel>()) ||
        "RELEASE" >> construct<OmpClause>(construct<OmpClause::Release>()) ||
        "ACQUIRE" >> construct<OmpClause>(construct<OmpClause::Acquire>()) ||
        "RELAXED" >> construct<OmpClause>(construct<OmpClause::Relaxed>())))))

// 2.4 Requires construct [OpenMP 5.0]
//        atomic-default-mem-order-clause ->
//                               seq_cst
//                               acq_rel
//                               relaxed
TYPE_PARSER(construct<OmpAtomicDefaultMemOrderClause>(
    "SEQ_CST" >> pure(common::OmpAtomicDefaultMemOrderType::SeqCst) ||
    "ACQ_REL" >> pure(common::OmpAtomicDefaultMemOrderType::AcqRel) ||
    "RELAXED" >> pure(common::OmpAtomicDefaultMemOrderType::Relaxed)))

// 2.17.7 Atomic construct
//        atomic-clause -> memory-order-clause | HINT(hint-expression)
TYPE_PARSER(sourced(construct<OmpAtomicClause>(
    construct<OmpAtomicClause>(Parser<OmpMemoryOrderClause>{}) ||
    construct<OmpAtomicClause>("FAIL" >> Parser<OmpFailClause>{}) ||
    construct<OmpAtomicClause>("HINT" >>
        sourced(construct<OmpClause>(
            construct<OmpClause::Hint>(parenthesized(constantExpr))))))))

// atomic-clause-list -> [atomic-clause, [atomic-clause], ...]
TYPE_PARSER(sourced(construct<OmpAtomicClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OmpAtomicClause>{})))))

TYPE_PARSER(sourced(construct<OpenMPDepobjConstruct>(verbatim("DEPOBJ"_tok),
    parenthesized(Parser<OmpObject>{}), sourced(Parser<OmpClause>{}))))

TYPE_PARSER(sourced(construct<OpenMPFlushConstruct>(verbatim("FLUSH"_tok),
    many(maybe(","_tok) >> sourced(Parser<OmpMemoryOrderClause>{})),
    maybe(parenthesized(Parser<OmpObjectList>{})))))

// Simple Standalone Directives
TYPE_PARSER(sourced(construct<OmpSimpleStandaloneDirective>(first(
    "BARRIER" >> pure(llvm::omp::Directive::OMPD_barrier),
    "ORDERED" >> pure(llvm::omp::Directive::OMPD_ordered),
    "SCAN" >> pure(llvm::omp::Directive::OMPD_scan),
    "TARGET ENTER DATA" >> pure(llvm::omp::Directive::OMPD_target_enter_data),
    "TARGET EXIT DATA" >> pure(llvm::omp::Directive::OMPD_target_exit_data),
    "TARGET UPDATE" >> pure(llvm::omp::Directive::OMPD_target_update),
    "TASKWAIT" >> pure(llvm::omp::Directive::OMPD_taskwait),
    "TASKYIELD" >> pure(llvm::omp::Directive::OMPD_taskyield)))))

TYPE_PARSER(sourced(construct<OpenMPSimpleStandaloneConstruct>(
    Parser<OmpSimpleStandaloneDirective>{}, Parser<OmpClauseList>{})))

// Standalone Constructs
TYPE_PARSER(
    sourced(construct<OpenMPStandaloneConstruct>(
                Parser<OpenMPSimpleStandaloneConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPFlushConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPCancelConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(
            Parser<OpenMPCancellationPointConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(
            Parser<OmpMetadirectiveDirective>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPDepobjConstruct>{})) /
    endOfLine)

// Directives enclosing structured-block
TYPE_PARSER(construct<OmpBlockDirective>(first(
    "MASKED" >> pure(llvm::omp::Directive::OMPD_masked),
    "MASTER" >> pure(llvm::omp::Directive::OMPD_master),
    "ORDERED" >> pure(llvm::omp::Directive::OMPD_ordered),
    "PARALLEL MASKED" >> pure(llvm::omp::Directive::OMPD_parallel_masked),
    "PARALLEL MASTER" >> pure(llvm::omp::Directive::OMPD_parallel_master),
    "PARALLEL WORKSHARE" >> pure(llvm::omp::Directive::OMPD_parallel_workshare),
    "PARALLEL" >> pure(llvm::omp::Directive::OMPD_parallel),
    "SCOPE" >> pure(llvm::omp::Directive::OMPD_scope),
    "SINGLE" >> pure(llvm::omp::Directive::OMPD_single),
    "TARGET DATA" >> pure(llvm::omp::Directive::OMPD_target_data),
    "TARGET PARALLEL" >> pure(llvm::omp::Directive::OMPD_target_parallel),
    "TARGET TEAMS" >> pure(llvm::omp::Directive::OMPD_target_teams),
    "TARGET" >> pure(llvm::omp::Directive::OMPD_target),
    "TASK"_id >> pure(llvm::omp::Directive::OMPD_task),
    "TASKGROUP" >> pure(llvm::omp::Directive::OMPD_taskgroup),
    "TEAMS" >> pure(llvm::omp::Directive::OMPD_teams),
    "WORKSHARE" >> pure(llvm::omp::Directive::OMPD_workshare))))

TYPE_PARSER(sourced(construct<OmpBeginBlockDirective>(
    sourced(Parser<OmpBlockDirective>{}), Parser<OmpClauseList>{})))

TYPE_PARSER(construct<OmpReductionInitializerClause>(
    "INITIALIZER" >> parenthesized("OMP_PRIV =" >> expr)))

// 2.16 Declare Reduction Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareReductionConstruct>(
    verbatim("DECLARE REDUCTION"_tok),
    "(" >> Parser<OmpReductionIdentifier>{} / ":",
    nonemptyList(Parser<DeclarationTypeSpec>{}) / ":",
    Parser<OmpReductionCombiner>{} / ")",
    maybe(Parser<OmpReductionInitializerClause>{}))))

// declare-target with list
TYPE_PARSER(sourced(construct<OmpDeclareTargetWithList>(
    parenthesized(Parser<OmpObjectList>{}))))

// declare-target with clause
TYPE_PARSER(
    sourced(construct<OmpDeclareTargetWithClause>(Parser<OmpClauseList>{})))

// declare-target-specifier
TYPE_PARSER(
    construct<OmpDeclareTargetSpecifier>(Parser<OmpDeclareTargetWithList>{}) ||
    construct<OmpDeclareTargetSpecifier>(Parser<OmpDeclareTargetWithClause>{}))

// 2.10.6 Declare Target Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareTargetConstruct>(
    verbatim("DECLARE TARGET"_tok), Parser<OmpDeclareTargetSpecifier>{})))

// mapper-specifier
TYPE_PARSER(construct<OmpMapperSpecifier>(
    maybe(name / ":" / !":"_tok), typeSpec / "::", name))

// OpenMP 5.2: 5.8.8 Declare Mapper Construct
TYPE_PARSER(sourced(
    construct<OpenMPDeclareMapperConstruct>(verbatim("DECLARE MAPPER"_tok),
        parenthesized(Parser<OmpMapperSpecifier>{}), Parser<OmpClauseList>{})))

TYPE_PARSER(construct<OmpReductionCombiner>(Parser<AssignmentStmt>{}) ||
    construct<OmpReductionCombiner>(Parser<FunctionReference>{}))

// 2.17.7 atomic -> ATOMIC [clause [,]] atomic-clause [[,] clause] |
//                  ATOMIC [clause]
//       clause -> memory-order-clause | HINT(hint-expression)
//       memory-order-clause -> SEQ_CST | ACQ_REL | RELEASE | ACQUIRE | RELAXED
//       atomic-clause -> READ | WRITE | UPDATE | CAPTURE

// OMP END ATOMIC
TYPE_PARSER(construct<OmpEndAtomic>(startOmpLine >> "END ATOMIC"_tok))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] READ [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    sourced(construct<OmpAtomicRead>(
        Parser<OmpAtomicClauseList>{} / maybe(","_tok), verbatim("READ"_tok),
        Parser<OmpAtomicClauseList>{} / endOmpLine, statement(assignmentStmt),
        maybe(Parser<OmpEndAtomic>{} / endOmpLine))))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] CAPTURE [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    sourced(construct<OmpAtomicCapture>(
        Parser<OmpAtomicClauseList>{} / maybe(","_tok), verbatim("CAPTURE"_tok),
        Parser<OmpAtomicClauseList>{} / endOmpLine, statement(assignmentStmt),
        statement(assignmentStmt), Parser<OmpEndAtomic>{} / endOmpLine)))

TYPE_PARSER(construct<OmpAtomicCompareIfStmt>(indirect(Parser<IfStmt>{})) ||
    construct<OmpAtomicCompareIfStmt>(indirect(Parser<IfConstruct>{})))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] COMPARE [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    sourced(construct<OmpAtomicCompare>(
        Parser<OmpAtomicClauseList>{} / maybe(","_tok), verbatim("COMPARE"_tok),
        Parser<OmpAtomicClauseList>{} / endOmpLine,
        Parser<OmpAtomicCompareIfStmt>{},
        maybe(Parser<OmpEndAtomic>{} / endOmpLine))))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] UPDATE [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    sourced(construct<OmpAtomicUpdate>(
        Parser<OmpAtomicClauseList>{} / maybe(","_tok), verbatim("UPDATE"_tok),
        Parser<OmpAtomicClauseList>{} / endOmpLine, statement(assignmentStmt),
        maybe(Parser<OmpEndAtomic>{} / endOmpLine))))

// OMP ATOMIC [atomic-clause-list]
TYPE_PARSER(sourced(construct<OmpAtomic>(verbatim("ATOMIC"_tok),
    Parser<OmpAtomicClauseList>{} / endOmpLine, statement(assignmentStmt),
    maybe(Parser<OmpEndAtomic>{} / endOmpLine))))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] WRITE [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    sourced(construct<OmpAtomicWrite>(
        Parser<OmpAtomicClauseList>{} / maybe(","_tok), verbatim("WRITE"_tok),
        Parser<OmpAtomicClauseList>{} / endOmpLine, statement(assignmentStmt),
        maybe(Parser<OmpEndAtomic>{} / endOmpLine))))

// Atomic Construct
TYPE_PARSER(construct<OpenMPAtomicConstruct>(Parser<OmpAtomicRead>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomicCapture>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomicCompare>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomicWrite>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomicUpdate>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomic>{}))

// 2.13.2 OMP CRITICAL
TYPE_PARSER(startOmpLine >>
    sourced(construct<OmpEndCriticalDirective>(
        verbatim("END CRITICAL"_tok), maybe(parenthesized(name)))) /
        endOmpLine)
TYPE_PARSER(sourced(construct<OmpCriticalDirective>(verbatim("CRITICAL"_tok),
                maybe(parenthesized(name)), Parser<OmpClauseList>{})) /
    endOmpLine)

TYPE_PARSER(construct<OpenMPCriticalConstruct>(
    Parser<OmpCriticalDirective>{}, block, Parser<OmpEndCriticalDirective>{}))

TYPE_PARSER(sourced(construct<OmpDispatchDirective>(
    verbatim("DISPATCH"_tok), Parser<OmpClauseList>{})))

TYPE_PARSER(
    construct<OmpEndDispatchDirective>(startOmpLine >> "END DISPATCH"_tok))

TYPE_PARSER(sourced(construct<OpenMPDispatchConstruct>(
    Parser<OmpDispatchDirective>{} / endOmpLine, block,
    maybe(Parser<OmpEndDispatchDirective>{} / endOmpLine))))

// 2.11.3 Executable Allocate directive
TYPE_PARSER(
    sourced(construct<OpenMPExecutableAllocate>(verbatim("ALLOCATE"_tok),
        maybe(parenthesized(Parser<OmpObjectList>{})), Parser<OmpClauseList>{},
        maybe(nonemptyList(Parser<OpenMPDeclarativeAllocate>{})) / endOmpLine,
        statement(allocateStmt))))

// 6.7 Allocators construct [OpenMP 5.2]
//     allocators-construct -> ALLOCATORS [allocate-clause [,]]
//                                allocate-stmt
//                             [omp-end-allocators-construct]
TYPE_PARSER(sourced(construct<OpenMPAllocatorsConstruct>(
    verbatim("ALLOCATORS"_tok), Parser<OmpClauseList>{} / endOmpLine,
    statement(allocateStmt), maybe(Parser<OmpEndAllocators>{} / endOmpLine))))

TYPE_PARSER(construct<OmpEndAllocators>(startOmpLine >> "END ALLOCATORS"_tok))

// 2.8.2 Declare Simd construct
TYPE_PARSER(
    sourced(construct<OpenMPDeclareSimdConstruct>(verbatim("DECLARE SIMD"_tok),
        maybe(parenthesized(name)), Parser<OmpClauseList>{})))

// 2.4 Requires construct
TYPE_PARSER(sourced(construct<OpenMPRequiresConstruct>(
    verbatim("REQUIRES"_tok), Parser<OmpClauseList>{})))

// 2.15.2 Threadprivate directive
TYPE_PARSER(sourced(construct<OpenMPThreadprivate>(
    verbatim("THREADPRIVATE"_tok), parenthesized(Parser<OmpObjectList>{}))))

// 2.11.3 Declarative Allocate directive
TYPE_PARSER(
    sourced(construct<OpenMPDeclarativeAllocate>(verbatim("ALLOCATE"_tok),
        parenthesized(Parser<OmpObjectList>{}), Parser<OmpClauseList>{})) /
    lookAhead(endOmpLine / !statement(allocateStmt)))

// Declarative constructs
TYPE_PARSER(startOmpLine >>
    withMessage("expected OpenMP construct"_err_en_US,
        sourced(construct<OpenMPDeclarativeConstruct>(
                    Parser<OpenMPDeclareReductionConstruct>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPDeclareMapperConstruct>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPDeclareSimdConstruct>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPDeclareTargetConstruct>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPDeclarativeAllocate>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPRequiresConstruct>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPThreadprivate>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPUtilityConstruct>{}) ||
            construct<OpenMPDeclarativeConstruct>(
                Parser<OmpMetadirectiveDirective>{})) /
            endOmpLine))

// Block Construct
TYPE_PARSER(construct<OpenMPBlockConstruct>(
    Parser<OmpBeginBlockDirective>{} / endOmpLine, block,
    Parser<OmpEndBlockDirective>{} / endOmpLine))

// OMP SECTIONS Directive
TYPE_PARSER(construct<OmpSectionsDirective>(first(
    "SECTIONS" >> pure(llvm::omp::Directive::OMPD_sections),
    "PARALLEL SECTIONS" >> pure(llvm::omp::Directive::OMPD_parallel_sections))))

// OMP BEGIN and END SECTIONS Directive
TYPE_PARSER(sourced(construct<OmpBeginSectionsDirective>(
    sourced(Parser<OmpSectionsDirective>{}), Parser<OmpClauseList>{})))
TYPE_PARSER(
    startOmpLine >> sourced(construct<OmpEndSectionsDirective>(
                        sourced("END"_tok >> Parser<OmpSectionsDirective>{}),
                        Parser<OmpClauseList>{})))

// OMP SECTION-BLOCK

TYPE_PARSER(construct<OpenMPSectionConstruct>(block))

TYPE_PARSER(maybe(startOmpLine >> "SECTION"_tok / endOmpLine) >>
    construct<OmpSectionBlocks>(nonemptySeparated(
        construct<OpenMPConstruct>(sourced(Parser<OpenMPSectionConstruct>{})),
        startOmpLine >> "SECTION"_tok / endOmpLine)))

// OMP SECTIONS (OpenMP 5.0 - 2.8.1), PARALLEL SECTIONS (OpenMP 5.0 - 2.13.3)
TYPE_PARSER(construct<OpenMPSectionsConstruct>(
    Parser<OmpBeginSectionsDirective>{} / endOmpLine,
    Parser<OmpSectionBlocks>{}, Parser<OmpEndSectionsDirective>{} / endOmpLine))

TYPE_CONTEXT_PARSER("OpenMP construct"_en_US,
    startOmpLine >>
        withMessage("expected OpenMP construct"_err_en_US,
            first(construct<OpenMPConstruct>(Parser<OpenMPSectionsConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPLoopConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPBlockConstruct>{}),
                // OpenMPBlockConstruct is attempted before
                // OpenMPStandaloneConstruct to resolve !$OMP ORDERED
                construct<OpenMPConstruct>(Parser<OpenMPStandaloneConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPAtomicConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPUtilityConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPDispatchConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPExecutableAllocate>{}),
                construct<OpenMPConstruct>(Parser<OpenMPAllocatorsConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPDeclarativeAllocate>{}),
                construct<OpenMPConstruct>(Parser<OpenMPCriticalConstruct>{}))))

// END OMP Block directives
TYPE_PARSER(
    startOmpLine >> sourced(construct<OmpEndBlockDirective>(
                        sourced("END"_tok >> Parser<OmpBlockDirective>{}),
                        Parser<OmpClauseList>{})))

// END OMP Loop directives
TYPE_PARSER(
    startOmpLine >> sourced(construct<OmpEndLoopDirective>(
                        sourced("END"_tok >> Parser<OmpLoopDirective>{}),
                        Parser<OmpClauseList>{})))

TYPE_PARSER(construct<OpenMPLoopConstruct>(
    Parser<OmpBeginLoopDirective>{} / endOmpLine))
} // namespace Fortran::parser
