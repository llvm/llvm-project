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
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Frontend/OpenMP/OMP.h"

// OpenMP Directives and Clauses
namespace Fortran::parser {
using namespace Fortran::parser::omp;

// Helper function to print the buffer contents starting at the current point.
[[maybe_unused]] static std::string ahead(const ParseState &state) {
  return std::string(
      state.GetLocation(), std::min<size_t>(64, state.BytesRemaining()));
}

constexpr auto startOmpLine = skipStuffBeforeStatement >> "!$OMP "_sptok;
constexpr auto endOmpLine = space >> endOfLine;

// Given a parser for a single element, and a parser for a list of elements
// of the same type, create a parser that constructs the entire list by having
// the single element be the head of the list, and the rest be the tail.
template <typename ParserH, typename ParserT> struct ConsParser {
  static_assert(std::is_same_v<std::list<typename ParserH::resultType>,
      typename ParserT::resultType>);

  using resultType = typename ParserT::resultType;
  constexpr ConsParser(ParserH h, ParserT t) : head_(h), tail_(t) {}

  std::optional<resultType> Parse(ParseState &state) const {
    if (auto &&first{head_.Parse(state)}) {
      if (auto rest{tail_.Parse(state)}) {
        rest->push_front(std::move(*first));
        return std::move(*rest);
      }
    }
    return std::nullopt;
  }

private:
  const ParserH head_;
  const ParserT tail_;
};

template <typename ParserH, typename ParserT,
    typename ValueH = typename ParserH::resultType,
    typename ValueT = typename ParserT::resultType,
    typename = std::enable_if_t<std::is_same_v<std::list<ValueH>, ValueT>>>
constexpr auto cons(ParserH head, ParserT tail) {
  return ConsParser<ParserH, ParserT>(head, tail);
}

// Given a parser P for a wrapper class, invoke P, and if it succeeds return
// the wrapped object.
template <typename Parser> struct UnwrapParser {
  static_assert(
      Parser::resultType::WrapperTrait::value && "Wrapper class required");
  using resultType = decltype(Parser::resultType::v);
  constexpr UnwrapParser(Parser p) : parser_(p) {}

  std::optional<resultType> Parse(ParseState &state) const {
    if (auto result{parser_.Parse(state)}) {
      return result->v;
    }
    return std::nullopt;
  }

private:
  const Parser parser_;
};

template <typename Parser> constexpr auto unwrap(const Parser &p) {
  return UnwrapParser<Parser>(p);
}

// Check (without advancing the parsing location) if the next thing in the
// input would be accepted by the "checked" parser, and if so, run the "parser"
// parser.
// The intended use is with the "checker" parser being some token, followed
// by a more complex parser that consumes the token plus more things, e.g.
// "PARALLEL"_id >= Parser<OmpDirectiveSpecification>{}.
//
// The >= has a higher precedence than ||, so it can be used just like >>
// in an alternatives parser without parentheses.
template <typename PA, typename PB>
constexpr auto operator>=(PA checker, PB parser) {
  return lookAhead(checker) >> parser;
}

// This parser succeeds if the given parser succeeds, and the result
// satisfies the given condition. Specifically, it succeeds if:
// 1. The parser given as the argument succeeds, and
// 2. The condition function (called with PA::resultType) returns true
//    for the result.
template <typename PA, typename CF> struct PredicatedParser {
  using resultType = typename PA::resultType;

  constexpr PredicatedParser(PA parser, CF condition)
      : parser_(parser), condition_(condition) {}

  std::optional<resultType> Parse(ParseState &state) const {
    if (auto result{parser_.Parse(state)}; result && condition_(*result)) {
      return result;
    }
    return std::nullopt;
  }

private:
  const PA parser_;
  const CF condition_;
};

template <typename PA, typename CF>
constexpr auto predicated(PA parser, CF condition) {
  return PredicatedParser(parser, condition);
}

/// Parse OpenMP directive name (this includes compound directives).
struct OmpDirectiveNameParser {
  using resultType = OmpDirectiveName;
  using Token = TokenStringMatch<false, false>;

  std::optional<resultType> Parse(ParseState &state) const {
    if (state.BytesRemaining() == 0) {
      return std::nullopt;
    }
    auto begin{state.GetLocation()};
    char next{static_cast<char>(std::tolower(*begin))};

    for (const NameWithId &nid : directives_starting_with(next)) {
      if (attempt(Token(nid.first.data())).Parse(state)) {
        OmpDirectiveName n;
        n.v = nid.second;
        n.source = parser::CharBlock(begin, state.GetLocation());
        return n;
      }
    }
    return std::nullopt;
  }

private:
  using NameWithId = std::pair<std::string, llvm::omp::Directive>;
  using ConstIterator = std::vector<NameWithId>::const_iterator;

  llvm::iterator_range<ConstIterator> directives_starting_with(
      char initial) const;
  void initTokens(std::vector<NameWithId>[]) const;
};

llvm::iterator_range<OmpDirectiveNameParser::ConstIterator>
OmpDirectiveNameParser::directives_starting_with(char initial) const {
  static const std::vector<NameWithId> empty{};
  if (initial < 'a' || initial > 'z') {
    return llvm::make_range(std::cbegin(empty), std::cend(empty));
  }

  static std::vector<NameWithId> table['z' - 'a' + 1];
  [[maybe_unused]] static bool init = (initTokens(table), true);

  int index = initial - 'a';
  return llvm::make_range(std::cbegin(table[index]), std::cend(table[index]));
}

void OmpDirectiveNameParser::initTokens(std::vector<NameWithId> table[]) const {
  for (size_t i{0}, e{llvm::omp::Directive_enumSize}; i != e; ++i) {
    llvm::StringSet spellings;
    auto id{static_cast<llvm::omp::Directive>(i)};
    for (unsigned version : llvm::omp::getOpenMPVersions()) {
      spellings.insert(llvm::omp::getOpenMPDirectiveName(id, version));
    }
    for (auto &[name, _] : spellings) {
      char initial{static_cast<char>(std::tolower(name.front()))};
      table[initial - 'a'].emplace_back(name.str(), id);
    }
  }
  // Sort the table with respect to the directive name length in a descending
  // order. This is to make sure that longer names are tried first, before
  // any potential prefix (e.g. "target update" before "target").
  for (int initial{'a'}; initial != 'z' + 1; ++initial) {
    llvm::stable_sort(table[initial - 'a'],
        [](auto &a, auto &b) { return a.first.size() > b.first.size(); });
  }
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

TYPE_PARSER(sourced( //
    construct<OmpArgumentList>(nonemptyList(Parser<OmpArgument>{}))))

TYPE_PARSER( //
    construct<OmpTypeSpecifier>(Parser<DeclarationTypeSpec>{}) ||
    construct<OmpTypeSpecifier>(Parser<TypeSpec>{}))

// 2.15.3.6 REDUCTION (reduction-identifier: variable-name-list)
TYPE_PARSER(construct<OmpReductionIdentifier>(Parser<DefinedOperator>{}) ||
    construct<OmpReductionIdentifier>(Parser<ProcedureDesignator>{}))

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
    construct<OmpTraitSelectorName>(unwrap(OmpDirectiveNameParser{})) ||
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

TYPE_PARSER(construct<OmpAccessGroup>( //
    "CGROUP" >> pure(OmpAccessGroup::Value::Cgroup)))

TYPE_PARSER(construct<OmpAlignment>(scalarIntExpr))

TYPE_PARSER(construct<OmpAlignModifier>( //
    "ALIGN" >> parenthesized(scalarIntExpr)))

TYPE_PARSER(construct<OmpAllocatorComplexModifier>(
    "ALLOCATOR" >> parenthesized(scalarIntExpr)))

TYPE_PARSER(construct<OmpAllocatorSimpleModifier>(scalarIntExpr))

TYPE_PARSER(construct<OmpAlwaysModifier>( //
    "ALWAYS" >> pure(OmpAlwaysModifier::Value::Always)))

TYPE_PARSER(construct<OmpAutomapModifier>(
    "AUTOMAP" >> pure(OmpAutomapModifier::Value::Automap)))

TYPE_PARSER(construct<OmpChunkModifier>( //
    "SIMD" >> pure(OmpChunkModifier::Value::Simd)))

TYPE_PARSER(construct<OmpCloseModifier>( //
    "CLOSE" >> pure(OmpCloseModifier::Value::Close)))

TYPE_PARSER(construct<OmpDeleteModifier>( //
    "DELETE" >> pure(OmpDeleteModifier::Value::Delete)))

TYPE_PARSER(construct<OmpDependenceType>(
    "SINK" >> pure(OmpDependenceType::Value::Sink) ||
    "SOURCE" >> pure(OmpDependenceType::Value::Source)))

TYPE_PARSER(construct<OmpDeviceModifier>(
    "ANCESTOR" >> pure(OmpDeviceModifier::Value::Ancestor) ||
    "DEVICE_NUM" >> pure(OmpDeviceModifier::Value::Device_Num)))

TYPE_PARSER(construct<OmpDirectiveNameModifier>(OmpDirectiveNameParser{}))

TYPE_PARSER(construct<OmpExpectation>( //
    "PRESENT" >> pure(OmpExpectation::Value::Present)))

TYPE_PARSER(construct<OmpInteropRuntimeIdentifier>(
    construct<OmpInteropRuntimeIdentifier>(charLiteralConstant) ||
    construct<OmpInteropRuntimeIdentifier>(scalarIntConstantExpr)))

TYPE_PARSER(construct<OmpInteropPreference>(verbatim("PREFER_TYPE"_tok) >>
    parenthesized(nonemptyList(Parser<OmpInteropRuntimeIdentifier>{}))))

TYPE_PARSER(construct<OmpInteropType>(
    "TARGETSYNC" >> pure(OmpInteropType::Value::TargetSync) ||
    "TARGET" >> pure(OmpInteropType::Value::Target)))

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

// map-type -> ALLOC | DELETE | FROM | RELEASE | STORAGE | TO | TOFROM
TYPE_PARSER(construct<OmpMapType>( //
    "ALLOC" >> pure(OmpMapType::Value::Alloc) ||
    // Parse "DELETE" as OmpDeleteModifier
    "FROM" >> pure(OmpMapType::Value::From) ||
    "RELEASE" >> pure(OmpMapType::Value::Release) ||
    "STORAGE" >> pure(OmpMapType::Value::Storage) ||
    "TO"_id >> pure(OmpMapType::Value::To) ||
    "TOFROM" >> pure(OmpMapType::Value::Tofrom)))

TYPE_PARSER(construct<OmpOrderModifier>(
    "REPRODUCIBLE" >> pure(OmpOrderModifier::Value::Reproducible) ||
    "UNCONSTRAINED" >> pure(OmpOrderModifier::Value::Unconstrained)))

TYPE_PARSER(construct<OmpOrderingModifier>(
    "MONOTONIC" >> pure(OmpOrderingModifier::Value::Monotonic) ||
    "NONMONOTONIC" >> pure(OmpOrderingModifier::Value::Nonmonotonic) ||
    "SIMD" >> pure(OmpOrderingModifier::Value::Simd)))

TYPE_PARSER(construct<OmpPrescriptiveness>(
    "STRICT" >> pure(OmpPrescriptiveness::Value::Strict) ||
    "FALLBACK" >> pure(OmpPrescriptiveness::Value::Fallback)))

TYPE_PARSER(construct<OmpPresentModifier>( //
    "PRESENT" >> pure(OmpPresentModifier::Value::Present)))

TYPE_PARSER(construct<OmpReductionModifier>(
    "INSCAN" >> pure(OmpReductionModifier::Value::Inscan) ||
    "TASK" >> pure(OmpReductionModifier::Value::Task) ||
    "DEFAULT" >> pure(OmpReductionModifier::Value::Default)))

TYPE_PARSER(construct<OmpRefModifier>( //
    "REF_PTEE" >> pure(OmpRefModifier::Value::Ref_Ptee) ||
    "REF_PTR"_id >> pure(OmpRefModifier::Value::Ref_Ptr) ||
    "REF_PTR_PTEE" >> pure(OmpRefModifier::Value::Ref_Ptr_Ptee)))

TYPE_PARSER(construct<OmpSelfModifier>( //
    "SELF" >> pure(OmpSelfModifier::Value::Self)))

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

TYPE_PARSER(construct<OmpxHoldModifier>( //
    "OMPX_HOLD" >> pure(OmpxHoldModifier::Value::Ompx_Hold)))

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

TYPE_PARSER( //
    sourced(construct<OmpDynGroupprivateClause::Modifier>(
        Parser<OmpAccessGroup>{})) ||
    sourced(construct<OmpDynGroupprivateClause::Modifier>(
        Parser<OmpPrescriptiveness>{})))

TYPE_PARSER(
    sourced(construct<OmpDeviceClause::Modifier>(Parser<OmpDeviceModifier>{})))

TYPE_PARSER(
    sourced(construct<OmpEnterClause::Modifier>(Parser<OmpAutomapModifier>{})))

TYPE_PARSER(sourced(construct<OmpFromClause::Modifier>(
    sourced(construct<OmpFromClause::Modifier>(Parser<OmpExpectation>{}) ||
        construct<OmpFromClause::Modifier>(Parser<OmpMapper>{}) ||
        construct<OmpFromClause::Modifier>(Parser<OmpIterator>{})))))

TYPE_PARSER(sourced(
    construct<OmpGrainsizeClause::Modifier>(Parser<OmpPrescriptiveness>{})))

TYPE_PARSER(sourced(
    construct<OmpIfClause::Modifier>(Parser<OmpDirectiveNameModifier>{})))

TYPE_PARSER(sourced(
    construct<OmpInitClause::Modifier>(
        construct<OmpInitClause::Modifier>(Parser<OmpInteropPreference>{})) ||
    construct<OmpInitClause::Modifier>(Parser<OmpInteropType>{})))

TYPE_PARSER(sourced(construct<OmpInReductionClause::Modifier>(
    Parser<OmpReductionIdentifier>{})))

TYPE_PARSER(sourced(construct<OmpLastprivateClause::Modifier>(
    Parser<OmpLastprivateModifier>{})))

TYPE_PARSER(sourced(
    construct<OmpLinearClause::Modifier>(Parser<OmpLinearModifier>{}) ||
    construct<OmpLinearClause::Modifier>(Parser<OmpStepComplexModifier>{}) ||
    construct<OmpLinearClause::Modifier>(Parser<OmpStepSimpleModifier>{})))

TYPE_PARSER(sourced(construct<OmpMapClause::Modifier>(
    sourced(construct<OmpMapClause::Modifier>(Parser<OmpAlwaysModifier>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpCloseModifier>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpDeleteModifier>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpPresentModifier>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpRefModifier>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpSelfModifier>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpMapper>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpIterator>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpMapType>{}) ||
        construct<OmpMapClause::Modifier>(Parser<OmpxHoldModifier>{})))))

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

TYPE_PARSER(construct<OmpAppendArgsClause::OmpAppendOp>(
    "INTEROP" >> parenthesized(nonemptyList(Parser<OmpInteropType>{}))))

TYPE_PARSER(construct<OmpAdjustArgsClause::OmpAdjustOp>(
    "NOTHING" >> pure(OmpAdjustArgsClause::OmpAdjustOp::Value::Nothing) ||
    "NEED_DEVICE_PTR" >>
        pure(OmpAdjustArgsClause::OmpAdjustOp::Value::Need_Device_Ptr)))

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

TYPE_PARSER(construct<OmpAdjustArgsClause>(
    (Parser<OmpAdjustArgsClause::OmpAdjustOp>{} / ":"),
    Parser<OmpObjectList>{}))

// [5.0] 2.10.1 affinity([aff-modifier:] locator-list)
//              aff-modifier: interator-modifier
TYPE_PARSER(construct<OmpAffinityClause>(
    maybe(nonemptyList(Parser<OmpAffinityClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

// 2.4 Requires construct [OpenMP 5.0]
//        atomic-default-mem-order-clause ->
//                               acq_rel
//                               acquire
//                               relaxed
//                               release
//                               seq_cst
TYPE_PARSER(construct<OmpAtomicDefaultMemOrderClause>(
    "ACQ_REL" >> pure(common::OmpMemoryOrderType::Acq_Rel) ||
    "ACQUIRE" >> pure(common::OmpMemoryOrderType::Acquire) ||
    "RELAXED" >> pure(common::OmpMemoryOrderType::Relaxed) ||
    "RELEASE" >> pure(common::OmpMemoryOrderType::Release) ||
    "SEQ_CST" >> pure(common::OmpMemoryOrderType::Seq_Cst)))

TYPE_PARSER(construct<OmpCancellationConstructTypeClause>(
    OmpDirectiveNameParser{}, maybe(parenthesized(scalarLogicalExpr))))

TYPE_PARSER(construct<OmpAppendArgsClause>(
    nonemptyList(Parser<OmpAppendArgsClause::OmpAppendOp>{})))

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

TYPE_PARSER(construct<OmpDynGroupprivateClause>(
    maybe(nonemptyList(Parser<OmpDynGroupprivateClause::Modifier>{}) / ":"),
    scalarIntExpr))

TYPE_PARSER(construct<OmpEnterClause>(
    maybe(nonemptyList(Parser<OmpEnterClause::Modifier>{}) / ":"),
    Parser<OmpObjectList>{}))

TYPE_PARSER(construct<OmpFailClause>(
    "ACQ_REL" >> pure(common::OmpMemoryOrderType::Acq_Rel) ||
    "ACQUIRE" >> pure(common::OmpMemoryOrderType::Acquire) ||
    "RELAXED" >> pure(common::OmpMemoryOrderType::Relaxed) ||
    "RELEASE" >> pure(common::OmpMemoryOrderType::Release) ||
    "SEQ_CST" >> pure(common::OmpMemoryOrderType::Seq_Cst)))

TYPE_PARSER(construct<OmpGraphIdClause>(expr))

TYPE_PARSER(construct<OmpGraphResetClause>(expr))

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
//  DEFAULT | PRESENT
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
        "DEFAULT" >> pure(OmpDefaultmapClause::ImplicitBehavior::Default) ||
        "PRESENT" >> pure(OmpDefaultmapClause::ImplicitBehavior::Present)),
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

TYPE_PARSER(construct<OmpHintClause>(scalarIntConstantExpr))

// init clause
TYPE_PARSER(construct<OmpInitClause>(
    maybe(nonemptyList(Parser<OmpInitClause::Modifier>{}) / ":"),
    Parser<OmpObject>{}))

// 2.8.1 ALIGNED (list: alignment)
TYPE_PARSER(construct<OmpAlignedClause>(Parser<OmpObjectList>{},
    maybe(":" >> nonemptyList(Parser<OmpAlignedClause::Modifier>{}))))

TYPE_PARSER( //
    construct<OmpUpdateClause>(parenthesized(Parser<OmpDependenceType>{})) ||
    construct<OmpUpdateClause>(parenthesized(Parser<OmpTaskDependenceType>{})))

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
    construct<OmpObject>(designator) || "/" >> construct<OmpObject>(name) / "/")

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

TYPE_PARSER(construct<OmpHoldsClause>(indirect(expr)))
TYPE_PARSER(construct<OmpAbsentClause>(many(maybe(","_tok) >>
    construct<llvm::omp::Directive>(unwrap(OmpDirectiveNameParser{})))))
TYPE_PARSER(construct<OmpContainsClause>(many(maybe(","_tok) >>
    construct<llvm::omp::Directive>(unwrap(OmpDirectiveNameParser{})))))

TYPE_PARSER( //
    "ABSENT" >> construct<OmpClause>(construct<OmpClause::Absent>(
                    parenthesized(Parser<OmpAbsentClause>{}))) ||
    "ACQUIRE" >> construct<OmpClause>(construct<OmpClause::Acquire>()) ||
    "ACQ_REL" >> construct<OmpClause>(construct<OmpClause::AcqRel>()) ||
    "ADJUST_ARGS" >> construct<OmpClause>(construct<OmpClause::AdjustArgs>(
                         parenthesized(Parser<OmpAdjustArgsClause>{}))) ||
    "AFFINITY" >> construct<OmpClause>(construct<OmpClause::Affinity>(
                      parenthesized(Parser<OmpAffinityClause>{}))) ||
    "ALIGN" >> construct<OmpClause>(construct<OmpClause::Align>(
                   parenthesized(Parser<OmpAlignClause>{}))) ||
    "ALIGNED" >> construct<OmpClause>(construct<OmpClause::Aligned>(
                     parenthesized(Parser<OmpAlignedClause>{}))) ||
    "ALLOCATE" >> construct<OmpClause>(construct<OmpClause::Allocate>(
                      parenthesized(Parser<OmpAllocateClause>{}))) ||
    "APPEND_ARGS" >> construct<OmpClause>(construct<OmpClause::AppendArgs>(
                         parenthesized(Parser<OmpAppendArgsClause>{}))) ||
    "ALLOCATOR" >> construct<OmpClause>(construct<OmpClause::Allocator>(
                       parenthesized(scalarIntExpr))) ||
    "AT" >> construct<OmpClause>(construct<OmpClause::At>(
                parenthesized(Parser<OmpAtClause>{}))) ||
    "ATOMIC_DEFAULT_MEM_ORDER" >>
        construct<OmpClause>(construct<OmpClause::AtomicDefaultMemOrder>(
            parenthesized(Parser<OmpAtomicDefaultMemOrderClause>{}))) ||
    "BIND" >> construct<OmpClause>(construct<OmpClause::Bind>(
                  parenthesized(Parser<OmpBindClause>{}))) ||
    "CAPTURE" >> construct<OmpClause>(construct<OmpClause::Capture>()) ||
    "COLLAPSE" >> construct<OmpClause>(construct<OmpClause::Collapse>(
                      parenthesized(scalarIntConstantExpr))) ||
    "COMPARE" >> construct<OmpClause>(construct<OmpClause::Compare>()) ||
    "CONTAINS" >> construct<OmpClause>(construct<OmpClause::Contains>(
                      parenthesized(Parser<OmpContainsClause>{}))) ||
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
    "DYN_GROUPPRIVATE" >>
        construct<OmpClause>(construct<OmpClause::DynGroupprivate>(
            parenthesized(Parser<OmpDynGroupprivateClause>{}))) ||
    "ENTER" >> construct<OmpClause>(construct<OmpClause::Enter>(
                   parenthesized(Parser<OmpEnterClause>{}))) ||
    "EXCLUSIVE" >> construct<OmpClause>(construct<OmpClause::Exclusive>(
                       parenthesized(Parser<OmpObjectList>{}))) ||
    "FAIL" >> construct<OmpClause>(construct<OmpClause::Fail>(
                  parenthesized(Parser<OmpFailClause>{}))) ||
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
    "GRAPH_ID" >> construct<OmpClause>(construct<OmpClause::GraphId>(
                      parenthesized(Parser<OmpGraphIdClause>{}))) ||
    "GRAPH_RESET" >>
        construct<OmpClause>(construct<OmpClause::GraphReset>(
            maybe(parenthesized(Parser<OmpGraphResetClause>{})))) ||
    "HAS_DEVICE_ADDR" >>
        construct<OmpClause>(construct<OmpClause::HasDeviceAddr>(
            parenthesized(Parser<OmpObjectList>{}))) ||
    "HINT" >> construct<OmpClause>(construct<OmpClause::Hint>(
                  parenthesized(Parser<OmpHintClause>{}))) ||
    "HOLDS" >> construct<OmpClause>(construct<OmpClause::Holds>(
                   parenthesized(Parser<OmpHoldsClause>{}))) ||
    "IF" >> construct<OmpClause>(construct<OmpClause::If>(
                parenthesized(Parser<OmpIfClause>{}))) ||
    "INBRANCH" >> construct<OmpClause>(construct<OmpClause::Inbranch>()) ||
    "INDIRECT" >> construct<OmpClause>(construct<OmpClause::Indirect>(
                      maybe(parenthesized(scalarLogicalExpr)))) ||
    "INIT" >> construct<OmpClause>(construct<OmpClause::Init>(
                  parenthesized(Parser<OmpInitClause>{}))) ||
    "INCLUSIVE" >> construct<OmpClause>(construct<OmpClause::Inclusive>(
                       parenthesized(Parser<OmpObjectList>{}))) ||
    "INITIALIZER" >> construct<OmpClause>(construct<OmpClause::Initializer>(
                         parenthesized(Parser<OmpInitializerClause>{}))) ||
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
    "NO_OPENMP"_id >> construct<OmpClause>(construct<OmpClause::NoOpenmp>()) ||
    "NO_OPENMP_ROUTINES" >>
        construct<OmpClause>(construct<OmpClause::NoOpenmpRoutines>()) ||
    "NO_PARALLELISM" >>
        construct<OmpClause>(construct<OmpClause::NoParallelism>()) ||
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
    "READ" >> construct<OmpClause>(construct<OmpClause::Read>()) ||
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
    "USE" >> construct<OmpClause>(construct<OmpClause::Use>(
                 parenthesized(Parser<OmpObject>{}))) ||
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
                    maybe(Parser<OmpUpdateClause>{}))) ||
    "WHEN" >> construct<OmpClause>(construct<OmpClause::When>(
                  parenthesized(Parser<OmpWhenClause>{}))) ||
    "WRITE" >> construct<OmpClause>(construct<OmpClause::Write>()) ||
    // Cancellable constructs
    construct<OmpClause>(construct<OmpClause::CancellationConstructType>(
        Parser<OmpCancellationConstructTypeClause>{})))

// [Clause, [Clause], ...]
TYPE_PARSER(sourced(construct<OmpClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OmpClause>{})))))

// 2.1 (variable | /common-block/ | array-sections)
TYPE_PARSER(construct<OmpObjectList>(nonemptyList(Parser<OmpObject>{})))

TYPE_PARSER(sourced(construct<OmpErrorDirective>(
    verbatim("ERROR"_tok), Parser<OmpClauseList>{})))

// --- Parsers for directives and constructs --------------------------

TYPE_PARSER(sourced(construct<OmpDirectiveName>(OmpDirectiveNameParser{})))

OmpDirectiveSpecification static makeFlushFromOldSyntax(Verbatim &&text,
    std::optional<OmpClauseList> &&clauses,
    std::optional<OmpArgumentList> &&args,
    OmpDirectiveSpecification::Flags &&flags) {
  return OmpDirectiveSpecification{OmpDirectiveName(text), std::move(args),
      std::move(clauses), std::move(flags)};
}

TYPE_PARSER(sourced(
    // Parse the old syntax: FLUSH [clauses] [(objects)]
    construct<OmpDirectiveSpecification>(
        // Force this old-syntax parser to fail for FLUSH followed by '('.
        // Otherwise it could succeed on the new syntax but have one of
        // lists absent in the parsed result.
        // E.g. for FLUSH(x) SEQ_CST it would find no clauses following
        // the directive name, parse the argument list "(x)" and stop.
        applyFunction<OmpDirectiveSpecification>(makeFlushFromOldSyntax,
            verbatim("FLUSH"_tok) / !lookAhead("("_tok),
            maybe(Parser<OmpClauseList>{}),
            maybe(parenthesized(Parser<OmpArgumentList>{})),
            pure(OmpDirectiveSpecification::Flags::DeprecatedSyntax))) ||
    // Parse the standard syntax: directive [(arguments)] [clauses]
    construct<OmpDirectiveSpecification>( //
        sourced(OmpDirectiveNameParser{}),
        maybe(parenthesized(Parser<OmpArgumentList>{})),
        maybe(Parser<OmpClauseList>{}),
        pure(OmpDirectiveSpecification::Flags::None))))

static bool IsStandaloneOrdered(const OmpDirectiveSpecification &dirSpec) {
  // An ORDERED construct is standalone if it has DOACROSS or DEPEND clause.
  return dirSpec.DirId() == llvm::omp::Directive::OMPD_ordered &&
      llvm::any_of(dirSpec.Clauses().v, [](const OmpClause &clause) {
        llvm::omp::Clause id{clause.Id()};
        return id == llvm::omp::Clause::OMPC_depend ||
            id == llvm::omp::Clause::OMPC_doacross;
      });
}

struct StrictlyStructuredBlockParser {
  using resultType = Block;

  std::optional<resultType> Parse(ParseState &state) const {
    // Detect BLOCK construct without parsing the entire thing.
    if (lookAhead(skipStuffBeforeStatement >> "BLOCK"_tok).Parse(state)) {
      if (auto epc{Parser<ExecutionPartConstruct>{}.Parse(state)}) {
        if (GetFortranBlockConstruct(*epc) != nullptr) {
          Block body;
          body.emplace_back(std::move(*epc));
          return std::move(body);
        }
      }
    }
    return std::nullopt;
  }
};

struct LooselyStructuredBlockParser {
  using resultType = Block;

  std::optional<resultType> Parse(ParseState &state) const {
    // Detect BLOCK construct without parsing the entire thing.
    if (lookAhead(skipStuffBeforeStatement >> "BLOCK"_tok).Parse(state)) {
      return std::nullopt;
    }
    if (auto &&body{block.Parse(state)}) {
      // Empty body is ok.
      return std::move(body);
    }
    return std::nullopt;
  }
};

TYPE_PARSER(sourced(construct<OmpNothingDirective>("NOTHING" >> ok)))

TYPE_PARSER(sourced(construct<OpenMPUtilityConstruct>(
    sourced(construct<OpenMPUtilityConstruct>(
        sourced(Parser<OmpErrorDirective>{}))) ||
    sourced(construct<OpenMPUtilityConstruct>(
        sourced(Parser<OmpNothingDirective>{}))))))

TYPE_PARSER(sourced(construct<OmpMetadirectiveDirective>(
    verbatim("METADIRECTIVE"_tok), Parser<OmpClauseList>{})))

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

static inline constexpr auto IsDirective(llvm::omp::Directive dir) {
  return [dir](const OmpDirectiveName &name) -> bool { return dir == name.v; };
}

struct OmpBeginDirectiveParser {
  using resultType = OmpDirectiveSpecification;

  constexpr OmpBeginDirectiveParser(llvm::omp::Directive dir) : dir_(dir) {}

  std::optional<resultType> Parse(ParseState &state) const {
    auto &&p{predicated(Parser<OmpDirectiveName>{}, IsDirective(dir_)) >=
        Parser<OmpDirectiveSpecification>{}};
    return p.Parse(state);
  }

private:
  llvm::omp::Directive dir_;
};

struct OmpEndDirectiveParser {
  using resultType = OmpDirectiveSpecification;

  constexpr OmpEndDirectiveParser(llvm::omp::Directive dir) : dir_(dir) {}

  std::optional<resultType> Parse(ParseState &state) const {
    if (startOmpLine.Parse(state)) {
      if (auto endToken{verbatim("END"_sptok).Parse(state)}) {
        if (auto &&dirSpec{OmpBeginDirectiveParser(dir_).Parse(state)}) {
          // Extend the "source" on both the OmpDirectiveName and the
          // OmpDirectiveNameSpecification.
          CharBlock &nameSource{std::get<OmpDirectiveName>(dirSpec->t).source};
          nameSource.ExtendToCover(endToken->source);
          dirSpec->source.ExtendToCover(endToken->source);
          return std::move(*dirSpec);
        }
      }
    }
    return std::nullopt;
  }

private:
  llvm::omp::Directive dir_;
};

struct OmpStatementConstructParser {
  using resultType = OmpBlockConstruct;

  constexpr OmpStatementConstructParser(llvm::omp::Directive dir) : dir_(dir) {}

  std::optional<resultType> Parse(ParseState &state) const {
    if (auto begin{OmpBeginDirectiveParser(dir_).Parse(state)}) {
      Block body;
      if (auto stmt{attempt(Parser<ExecutionPartConstruct>{}).Parse(state)}) {
        body.emplace_back(std::move(*stmt));
      }
      // Allow empty block. Check for this in semantics.

      auto end{maybe(OmpEndDirectiveParser{dir_}).Parse(state)};
      return OmpBlockConstruct{OmpBeginDirective(std::move(*begin)),
          std::move(body),
          llvm::transformOptional(std::move(*end),
              [](auto &&s) { return OmpEndDirective(std::move(s)); })};
    }
    return std::nullopt;
  }

private:
  llvm::omp::Directive dir_;
};

struct OmpBlockConstructParser {
  using resultType = OmpBlockConstruct;

  constexpr OmpBlockConstructParser(llvm::omp::Directive dir) : dir_(dir) {}

  std::optional<resultType> Parse(ParseState &state) const {
    if (auto &&begin{OmpBeginDirectiveParser(dir_).Parse(state)}) {
      if (IsStandaloneOrdered(*begin)) {
        return std::nullopt;
      }
      if (auto &&body{attempt(StrictlyStructuredBlockParser{}).Parse(state)}) {
        // Try strictly-structured block with an optional end-directive
        auto end{maybe(OmpEndDirectiveParser{dir_}).Parse(state)};
        return OmpBlockConstruct{OmpBeginDirective(std::move(*begin)),
            std::move(*body),
            llvm::transformOptional(std::move(*end),
                [](auto &&s) { return OmpEndDirective(std::move(s)); })};
      } else if (auto &&body{
                     attempt(LooselyStructuredBlockParser{}).Parse(state)}) {
        // Try loosely-structured block with a mandatory end-directive.
        auto end{maybe(OmpEndDirectiveParser{dir_}).Parse(state)};
        // Delay the error for a missing end-directive until semantics so that
        // we have better control over the output.
        return OmpBlockConstruct{OmpBeginDirective(std::move(*begin)),
            std::move(*body),
            llvm::transformOptional(std::move(*end),
                [](auto &&s) { return OmpEndDirective(std::move(s)); })};
      }
    }
    return std::nullopt;
  }

private:
  llvm::omp::Directive dir_;
};

TYPE_PARSER(sourced(construct<OpenMPAllocatorsConstruct>(
    OmpStatementConstructParser{llvm::omp::Directive::OMPD_allocators})))

TYPE_PARSER(sourced(construct<OpenMPDispatchConstruct>(
    OmpStatementConstructParser{llvm::omp::Directive::OMPD_dispatch})))

// Parser for an arbitrary OpenMP ATOMIC construct.
//
// Depending on circumstances, an ATOMIC construct applies to one or more
// following statements. In certain cases when a single statement is
// expected, the end-directive is optional. The specifics depend on both
// the clauses used, and the form of the executable statement. To emit
// more meaningful messages in case of errors, the exact analysis of the
// structure of the construct will be delayed until semantic checks.
//
// The parser will first try the case when the end-directive is present,
// and will parse at most "BodyLimit" (and potentially zero) constructs
// while looking for the end-directive before it gives up.
// Then it will assume that no end-directive is present, and will try to
// parse a single executable construct as the body of the construct.
//
// The limit on the number of constructs is there to reduce the amount of
// unnecessary parsing when the end-directive is absent. It's higher than
// the maximum number of statements in any valid construct to accept cases
// when extra statements are present by mistake.
// A problem can occur when atomic constructs without end-directive follow
// each other closely, e.g.
//   !$omp atomic write
//     x = v
//   !$omp atomic update
//     x = x + 1
//   ...
// The speculative parsing will become "recursive", and has the potential
// to take a (practically) infinite amount of time given a sufficiently
// large number of such constructs in a row. Since atomic constructs cannot
// contain other OpenMP constructs, guarding against recursive calls to the
// atomic construct parser solves the problem.
struct OmpAtomicConstructParser {
  using resultType = OpenMPAtomicConstruct;

  static constexpr size_t BodyLimit{5};

  std::optional<resultType> Parse(ParseState &state) const {
    if (recursing_) {
      return std::nullopt;
    }
    recursing_ = true;

    auto dirSpec{Parser<OmpDirectiveSpecification>{}.Parse(state)};
    if (!dirSpec || dirSpec->DirId() != llvm::omp::Directive::OMPD_atomic) {
      recursing_ = false;
      return std::nullopt;
    }

    auto exec{Parser<ExecutionPartConstruct>{}};
    auto end{OmpEndDirectiveParser{llvm::omp::Directive::OMPD_atomic}};
    TailType tail;

    if (ParseOne(exec, end, tail, state)) {
      if (!tail.first.empty()) {
        if (auto &&rest{attempt(LimitedTailParser(BodyLimit)).Parse(state)}) {
          for (auto &&s : rest->first) {
            tail.first.emplace_back(std::move(s));
          }
          assert(!tail.second);
          tail.second = std::move(rest->second);
        }
      }
      recursing_ = false;
      return OpenMPAtomicConstruct{OmpBeginDirective(std::move(*dirSpec)),
          std::move(tail.first),
          llvm::transformOptional(std::move(tail.second),
              [](auto &&s) { return OmpEndDirective(std::move(s)); })};
    }

    recursing_ = false;
    return std::nullopt;
  }

private:
  // Begin-directive + TailType = entire construct.
  using TailType = std::pair<Block, std::optional<OmpDirectiveSpecification>>;

  // Parse either an ExecutionPartConstruct, or atomic end-directive. When
  // successful, record the result in the "tail" provided, otherwise fail.
  static std::optional<Success> ParseOne( //
      Parser<ExecutionPartConstruct> &exec, OmpEndDirectiveParser &end,
      TailType &tail, ParseState &state) {
    auto isRecovery{[](const ExecutionPartConstruct &e) {
      return std::holds_alternative<ErrorRecovery>(e.u);
    }};
    if (auto &&stmt{attempt(exec).Parse(state)}; stmt && !isRecovery(*stmt)) {
      tail.first.emplace_back(std::move(*stmt));
    } else if (auto &&dir{attempt(end).Parse(state)}) {
      tail.second = std::move(*dir);
    } else {
      return std::nullopt;
    }
    return Success{};
  }

  struct LimitedTailParser {
    using resultType = TailType;

    constexpr LimitedTailParser(size_t count) : count_(count) {}

    std::optional<resultType> Parse(ParseState &state) const {
      auto exec{Parser<ExecutionPartConstruct>{}};
      auto end{OmpEndDirectiveParser{llvm::omp::Directive::OMPD_atomic}};
      TailType tail;

      for (size_t i{0}; i != count_; ++i) {
        if (ParseOne(exec, end, tail, state)) {
          if (tail.second) {
            // Return when the end-directive was parsed.
            return std::move(tail);
          }
        } else {
          break;
        }
      }
      return std::nullopt;
    }

  private:
    const size_t count_;
  };

  // The recursion guard should become thread_local if parsing is ever
  // parallelized.
  static bool recursing_;
};

bool OmpAtomicConstructParser::recursing_{false};

TYPE_PARSER(sourced( //
    construct<OpenMPAtomicConstruct>(OmpAtomicConstructParser{})))

static bool IsSimpleStandalone(const OmpDirectiveName &name) {
  switch (name.v) {
  case llvm::omp::Directive::OMPD_barrier:
  case llvm::omp::Directive::OMPD_scan:
  case llvm::omp::Directive::OMPD_target_enter_data:
  case llvm::omp::Directive::OMPD_target_exit_data:
  case llvm::omp::Directive::OMPD_target_update:
  case llvm::omp::Directive::OMPD_taskwait:
  case llvm::omp::Directive::OMPD_taskyield:
    return true;
  default:
    return false;
  }
}

TYPE_PARSER(sourced( //
    construct<OpenMPSimpleStandaloneConstruct>(
        predicated(OmpDirectiveNameParser{}, IsSimpleStandalone) >=
        Parser<OmpDirectiveSpecification>{}) ||
    construct<OpenMPSimpleStandaloneConstruct>(
        predicated(Parser<OmpDirectiveSpecification>{}, IsStandaloneOrdered))))

TYPE_PARSER(sourced( //
    construct<OpenMPFlushConstruct>(
        predicated(OmpDirectiveNameParser{},
            IsDirective(llvm::omp::Directive::OMPD_flush)) >=
        Parser<OmpDirectiveSpecification>{})))

// 2.14.2 Cancellation Point construct
TYPE_PARSER(sourced( //
    construct<OpenMPCancellationPointConstruct>(
        predicated(OmpDirectiveNameParser{},
            IsDirective(llvm::omp::Directive::OMPD_cancellation_point)) >=
        Parser<OmpDirectiveSpecification>{})))

// 2.14.1 Cancel construct
TYPE_PARSER(sourced( //
    construct<OpenMPCancelConstruct>(
        predicated(OmpDirectiveNameParser{},
            IsDirective(llvm::omp::Directive::OMPD_cancel)) >=
        Parser<OmpDirectiveSpecification>{})))

TYPE_PARSER(sourced( //
    construct<OpenMPDepobjConstruct>(
        predicated(OmpDirectiveNameParser{},
            IsDirective(llvm::omp::Directive::OMPD_depobj)) >=
        Parser<OmpDirectiveSpecification>{})))

// OMP 5.2 14.1 Interop construct
TYPE_PARSER(sourced( //
    construct<OpenMPInteropConstruct>(
        predicated(OmpDirectiveNameParser{},
            IsDirective(llvm::omp::Directive::OMPD_interop)) >=
        Parser<OmpDirectiveSpecification>{})))

// Standalone Constructs
TYPE_PARSER(
    sourced( //
        construct<OpenMPStandaloneConstruct>(
            Parser<OpenMPSimpleStandaloneConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPFlushConstruct>{}) ||
        // Try CANCELLATION POINT before CANCEL.
        construct<OpenMPStandaloneConstruct>(
            Parser<OpenMPCancellationPointConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPCancelConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(
            Parser<OmpMetadirectiveDirective>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPDepobjConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(
            Parser<OpenMPInteropConstruct>{})) /
    endOfLine)

TYPE_PARSER(construct<OmpInitializerProc>(Parser<ProcedureDesignator>{},
    parenthesized(many(maybe(","_tok) >> Parser<ActualArgSpec>{}))))

TYPE_PARSER(construct<OmpInitializerClause>(
    construct<OmpInitializerClause>(assignmentStmt) ||
    construct<OmpInitializerClause>(Parser<OmpInitializerProc>{})))

// OpenMP 5.2: 7.5.4 Declare Variant directive
TYPE_PARSER(sourced(construct<OmpDeclareVariantDirective>(
    verbatim("DECLARE VARIANT"_tok) || verbatim("DECLARE_VARIANT"_tok),
    "(" >> maybe(name / ":"), name / ")", Parser<OmpClauseList>{})))

// 2.16 Declare Reduction Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareReductionConstruct>(
    verbatim("DECLARE REDUCTION"_tok) || verbatim("DECLARE_REDUCTION"_tok),
    "(" >> indirect(Parser<OmpReductionSpecifier>{}) / ")",
    maybe(Parser<OmpClauseList>{}))))

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
    verbatim("DECLARE TARGET"_tok) || verbatim("DECLARE_TARGET"_tok),
    Parser<OmpDeclareTargetSpecifier>{})))

static OmpMapperSpecifier ConstructOmpMapperSpecifier(
    std::optional<Name> &&mapperName, TypeSpec &&typeSpec, Name &&varName) {
  // If a name is present, parse: name ":" typeSpec "::" name
  // This matches the syntax: <mapper-name> : <type-spec> :: <variable-name>
  if (mapperName.has_value() && mapperName->ToString() != "default") {
    return OmpMapperSpecifier{
        mapperName->ToString(), std::move(typeSpec), std::move(varName)};
  }
  // If the name is missing, use the DerivedTypeSpec name to construct the
  // default mapper name.
  // This matches the syntax: <type-spec> :: <variable-name>
  if (DerivedTypeSpec * derived{std::get_if<DerivedTypeSpec>(&typeSpec.u)}) {
    return OmpMapperSpecifier{
        std::get<Name>(derived->t).ToString() + llvm::omp::OmpDefaultMapperName,
        std::move(typeSpec), std::move(varName)};
  }
  return OmpMapperSpecifier{std::string(llvm::omp::OmpDefaultMapperName),
      std::move(typeSpec), std::move(varName)};
}

// mapper-specifier
TYPE_PARSER(applyFunction<OmpMapperSpecifier>(ConstructOmpMapperSpecifier,
    maybe(name / ":" / !":"_tok), typeSpec / "::", name))

// OpenMP 5.2: 5.8.8 Declare Mapper Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareMapperConstruct>(
    verbatim("DECLARE MAPPER"_tok) || verbatim("DECLARE_MAPPER"_tok),
    parenthesized(Parser<OmpMapperSpecifier>{}), Parser<OmpClauseList>{})))

TYPE_PARSER(construct<OmpReductionCombiner>(Parser<AssignmentStmt>{}) ||
    construct<OmpReductionCombiner>(Parser<FunctionReference>{}))

TYPE_PARSER(construct<OpenMPCriticalConstruct>(
    OmpBlockConstructParser{llvm::omp::Directive::OMPD_critical}))

// 2.11.3 Executable Allocate directive
TYPE_PARSER(
    sourced(construct<OpenMPExecutableAllocate>(verbatim("ALLOCATE"_tok),
        maybe(parenthesized(Parser<OmpObjectList>{})), Parser<OmpClauseList>{},
        maybe(nonemptyList(Parser<OpenMPDeclarativeAllocate>{})) / endOmpLine,
        statement(allocateStmt))))

// 2.8.2 Declare Simd construct
TYPE_PARSER(sourced(construct<OpenMPDeclareSimdConstruct>(
    verbatim("DECLARE SIMD"_tok) || verbatim("DECLARE_SIMD"_tok),
    maybe(parenthesized(name)), Parser<OmpClauseList>{})))

TYPE_PARSER(sourced( //
    construct<OpenMPGroupprivate>(
        predicated(OmpDirectiveNameParser{},
            IsDirective(llvm::omp::Directive::OMPD_groupprivate)) >=
        Parser<OmpDirectiveSpecification>{})))

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

// Assumes Construct
TYPE_PARSER(sourced(construct<OpenMPDeclarativeAssumes>(
    verbatim("ASSUMES"_tok), Parser<OmpClauseList>{})))

// Declarative constructs
TYPE_PARSER(
    startOmpLine >> withMessage("expected OpenMP construct"_err_en_US,
                        sourced(construct<OpenMPDeclarativeConstruct>(
                                    Parser<OpenMPDeclarativeAssumes>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPDeclareReductionConstruct>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPDeclareMapperConstruct>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPDeclareSimdConstruct>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPDeclareTargetConstruct>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OmpDeclareVariantDirective>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPDeclarativeAllocate>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPGroupprivate>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPRequiresConstruct>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPThreadprivate>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OpenMPUtilityConstruct>{}) ||
                            construct<OpenMPDeclarativeConstruct>(
                                Parser<OmpMetadirectiveDirective>{})) /
                            endOmpLine))

TYPE_PARSER(construct<OpenMPAssumeConstruct>(
    sourced(OmpBlockConstructParser{llvm::omp::Directive::OMPD_assume})))

// Block Construct
#define MakeBlockConstruct(dir) \
  construct<OmpBlockConstruct>(OmpBlockConstructParser{dir})
TYPE_PARSER( //
    MakeBlockConstruct(llvm::omp::Directive::OMPD_masked) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_master) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_ordered) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_parallel_masked) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_parallel_master) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_parallel_workshare) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_parallel) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_scope) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_single) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_target_data) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_target_parallel) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_target_teams) ||
    MakeBlockConstruct(
        llvm::omp::Directive::OMPD_target_teams_workdistribute) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_target) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_task) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_taskgraph) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_taskgroup) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_teams) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_teams_workdistribute) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_workshare) ||
    MakeBlockConstruct(llvm::omp::Directive::OMPD_workdistribute))
#undef MakeBlockConstruct

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

static constexpr auto sectionDir{
    startOmpLine >> (predicated(OmpDirectiveNameParser{},
                         IsDirective(llvm::omp::Directive::OMPD_section)) >=
                        Parser<OmpDirectiveSpecification>{})};

// OMP SECTIONS (OpenMP 5.0 - 2.8.1), PARALLEL SECTIONS (OpenMP 5.0 - 2.13.3)
TYPE_PARSER(sourced(construct<OpenMPSectionsConstruct>(
    Parser<OmpBeginSectionsDirective>{} / endOmpLine,
    cons( //
        construct<OpenMPConstruct>(sourced(
            construct<OpenMPSectionConstruct>(maybe(sectionDir), block))),
        many(construct<OpenMPConstruct>(
            sourced(construct<OpenMPSectionConstruct>(sectionDir, block))))),
    maybe(Parser<OmpEndSectionsDirective>{} / endOmpLine))))

static bool IsExecutionPart(const OmpDirectiveName &name) {
  return name.IsExecutionPart();
}

TYPE_PARSER(construct<OpenMPExecDirective>(
    startOmpLine >> predicated(Parser<OmpDirectiveName>{}, IsExecutionPart)))

TYPE_CONTEXT_PARSER("OpenMP construct"_en_US,
    startOmpLine >>
        withMessage("expected OpenMP construct"_err_en_US,
            first(construct<OpenMPConstruct>(Parser<OpenMPSectionsConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPLoopConstruct>{}),
                construct<OpenMPConstruct>(Parser<OmpBlockConstruct>{}),
                // OmpBlockConstruct is attempted before
                // OpenMPStandaloneConstruct to resolve !$OMP ORDERED
                construct<OpenMPConstruct>(Parser<OpenMPStandaloneConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPAtomicConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPUtilityConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPDispatchConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPExecutableAllocate>{}),
                construct<OpenMPConstruct>(Parser<OpenMPAllocatorsConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPDeclarativeAllocate>{}),
                construct<OpenMPConstruct>(Parser<OpenMPAssumeConstruct>{}),
                construct<OpenMPConstruct>(Parser<OpenMPCriticalConstruct>{}))))

// END OMP Loop directives
TYPE_PARSER(
    startOmpLine >> sourced(construct<OmpEndLoopDirective>(
                        sourced("END"_tok >> Parser<OmpLoopDirective>{}),
                        Parser<OmpClauseList>{})))

TYPE_PARSER(construct<OpenMPLoopConstruct>(
    Parser<OmpBeginLoopDirective>{} / endOmpLine))
} // namespace Fortran::parser
