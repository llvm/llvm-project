//===-- lib/Semantics/check-omp-variant.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic checks for METADIRECTIVE, DECLARE VARIANT, and related constructs.
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"

#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Common/visit.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/omp-declare-variant.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

#include "llvm/Frontend/OpenMP/OMP.h"

#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

namespace Fortran::semantics {

using namespace Fortran::semantics::omp;

void OmpStructureChecker::Enter(const parser::OmpClause::When &x) {
  OmpVerifyModifiers(
      x.v, llvm::omp::OMPC_when, GetContext().clauseSource, context_);
  // Record this WHEN clause's context selector so the variant directive it
  // controls can be paired with it for static-applicability matching.
  if (const auto &modifiers{std::get<0>(x.v.t)};
      modifiers && modifiers->size() == 1) {
    currentWhenSelector_ =
        &std::get<parser::modifier::OmpContextSelector>(modifiers->front().u);
  }
}

void OmpStructureChecker::Leave(const parser::OmpClause::When &) {
  currentWhenSelector_ = nullptr;
}

void OmpStructureChecker::CheckContextSelectorSpecification(
    const parser::OmpContextSelector &ctx) {
  using SetName = parser::OmpTraitSetSelectorName;
  std::map<SetName::Value, const SetName *> visited;

  for (const parser::OmpTraitSetSelector &traitSet : ctx.v) {
    auto &name{std::get<SetName>(traitSet.t)};
    auto [prev, unique]{visited.insert(std::make_pair(name.v, &name))};
    if (!unique) {
      std::string showName{parser::ToUpperCaseLetters(name.ToString())};
      parser::MessageFormattedText txt(
          "Repeated trait set name %s in a context specifier"_err_en_US,
          showName);
      parser::Message message(name.source, txt);
      message.Attach(prev->second->source,
          "Previous trait set %s provided here"_en_US, showName);
      context_.Say(std::move(message));
    }
    CheckTraitSetSelector(traitSet);
  }
}

void OmpStructureChecker::Enter(const parser::OmpContextSelector &ctx) {
  EnterDirectiveNest(ContextSelectorNest);
  CheckContextSelectorSpecification(ctx);
}

void OmpStructureChecker::Leave(const parser::OmpContextSelector &) {
  ExitDirectiveNest(ContextSelectorNest);
}

const std::list<parser::OmpTraitProperty> &
OmpStructureChecker::GetTraitPropertyList(
    const parser::OmpTraitSelector &trait) {
  static const std::list<parser::OmpTraitProperty> empty{};
  auto &[_, maybeProps]{trait.t};
  if (maybeProps) {
    using PropertyList = std::list<parser::OmpTraitProperty>;
    return std::get<PropertyList>(maybeProps->t);
  } else {
    return empty;
  }
}

std::optional<llvm::omp::Clause> OmpStructureChecker::GetClauseFromProperty(
    const parser::OmpTraitProperty &property) {
  using MaybeClause = std::optional<llvm::omp::Clause>;

  // The parser for OmpClause will only succeed if the clause was
  // given with all required arguments.
  // If this is a string or complex extension with a clause name,
  // treat it as a clause and let the trait checker deal with it.

  auto getClauseFromString{[&](const std::string &s) -> MaybeClause {
    auto id{llvm::omp::getOpenMPClauseKind(parser::ToLowerCaseLetters(s))};
    if (id != llvm::omp::Clause::OMPC_unknown) {
      return id;
    } else {
      return std::nullopt;
    }
  }};

  return common::visit( //
      common::visitors{
          [&](const parser::OmpTraitPropertyName &x) -> MaybeClause {
            return getClauseFromString(x.v);
          },
          [&](const common::Indirection<parser::OmpClause> &x) -> MaybeClause {
            return x.value().Id();
          },
          [&](const parser::ScalarExpr &x) -> MaybeClause {
            return std::nullopt;
          },
          [&](const parser::OmpTraitPropertyExtension &x) -> MaybeClause {
            using ExtProperty = parser::OmpTraitPropertyExtension;
            if (auto *name{std::get_if<parser::OmpTraitPropertyName>(&x.u)}) {
              return getClauseFromString(name->v);
            } else if (auto *cpx{std::get_if<ExtProperty::Complex>(&x.u)}) {
              return getClauseFromString(
                  std::get<parser::OmpTraitPropertyName>(cpx->t).v);
            }
            return std::nullopt;
          },
      },
      property.u);
}

void OmpStructureChecker::CheckTraitSelectorList(
    const std::list<parser::OmpTraitSelector> &traits) {
  // [6.0:322:20]
  // Each trait-selector-name may only be specified once in a trait selector
  // set.

  // Cannot store OmpTraitSelectorName directly, because it's not copyable.
  using TraitName = parser::OmpTraitSelectorName;
  using BareName = decltype(TraitName::u);
  std::map<BareName, const TraitName *> visited;

  for (const parser::OmpTraitSelector &trait : traits) {
    auto &name{std::get<TraitName>(trait.t)};

    auto [prev, unique]{visited.insert(std::make_pair(name.u, &name))};
    if (!unique) {
      std::string showName{parser::ToUpperCaseLetters(name.ToString())};
      parser::MessageFormattedText txt(
          "Repeated trait name %s in a trait set"_err_en_US, showName);
      parser::Message message(name.source, txt);
      message.Attach(prev->second->source,
          "Previous trait %s provided here"_en_US, showName);
      context_.Say(std::move(message));
    }
  }
}

void OmpStructureChecker::CheckTraitSetSelector(
    const parser::OmpTraitSetSelector &traitSet) {

  // Trait Set      |           Allowed traits | D-traits | X-traits | Score |
  //
  // Construct      |     Simd, directive-name |      Yes |       No |    No |
  // Device         |          Arch, Isa, Kind |       No |      Yes |    No |
  // Implementation | Atomic_Default_Mem_Order |       No |      Yes |   Yes |
  //                |      Extension, Requires |          |          |       |
  //                |                   Vendor |          |          |       |
  // Target_Device  |    Arch, Device_Num, Isa |       No |      Yes |    No |
  //                |                Kind, Uid |          |          |       |
  // User           |                Condition |       No |       No |   Yes |

  struct TraitSetConfig {
    std::set<parser::OmpTraitSelectorName::Value> allowed;
    bool allowsDirectiveTraits;
    bool allowsExtensionTraits;
    bool allowsScore;
  };

  using SName = parser::OmpTraitSetSelectorName::Value;
  using TName = parser::OmpTraitSelectorName::Value;

  static const std::map<SName, TraitSetConfig> configs{
      {SName::Construct, //
          {{TName::Simd}, true, false, false}},
      {SName::Device, //
          {{TName::Arch, TName::Isa, TName::Kind}, false, true, false}},
      {SName::Implementation, //
          {{TName::Atomic_Default_Mem_Order, TName::Extension, TName::Requires,
               TName::Vendor},
              false, true, true}},
      {SName::Target_Device, //
          {{TName::Arch, TName::Device_Num, TName::Isa, TName::Kind,
               TName::Uid},
              false, true, false}},
      {SName::User, //
          {{TName::Condition}, false, false, true}},
  };

  auto checkTraitSet{[&](const TraitSetConfig &config) {
    auto &[setName, traits]{traitSet.t};
    auto usn{parser::ToUpperCaseLetters(setName.ToString())};

    // Check if there are any duplicate traits.
    CheckTraitSelectorList(traits);

    for (const parser::OmpTraitSelector &trait : traits) {
      // Don't use structured bindings here, because they cannot be captured
      // before C++20.
      auto &traitName = std::get<parser::OmpTraitSelectorName>(trait.t);
      auto &maybeProps =
          std::get<std::optional<parser::OmpTraitSelector::Properties>>(
              trait.t);

      // Check allowed traits
      common::visit( //
          common::visitors{
              [&](parser::OmpTraitSelectorName::Value v) {
                if (!config.allowed.count(v)) {
                  context_.Say(traitName.source,
                      "%s is not a valid trait for %s trait set"_err_en_US,
                      parser::ToUpperCaseLetters(traitName.ToString()), usn);
                }
              },
              [&](llvm::omp::Directive) {
                if (!config.allowsDirectiveTraits) {
                  context_.Say(traitName.source,
                      "Directive name is not a valid trait for %s trait set"_err_en_US,
                      usn);
                }
              },
              [&](const std::string &) {
                if (!config.allowsExtensionTraits) {
                  context_.Say(traitName.source,
                      "Extension traits are not valid for %s trait set"_err_en_US,
                      usn);
                }
              },
          },
          traitName.u);

      // Check score
      if (maybeProps) {
        auto &[maybeScore, _]{maybeProps->t};
        if (maybeScore) {
          if (!config.allowsScore)
            context_.Say(maybeScore->source,
                "SCORE is not allowed for %s trait set"_err_en_US, usn);
          else
            CheckTraitScore(*maybeScore);
        }
      }

      // Check the properties of the individual traits
      CheckTraitSelector(traitSet, trait);
    }
  }};

  checkTraitSet(
      configs.at(std::get<parser::OmpTraitSetSelectorName>(traitSet.t).v));
}

void OmpStructureChecker::CheckTraitScore(const parser::OmpTraitScore &score) {
  // [6.0:322:23]
  // A score-expression must be a non-negative constant integer expression.
  if (auto value{GetIntValue(score)}; !value || value < 0) {
    context_.Say(score.source,
        "SCORE expression must be a non-negative constant integer expression"_err_en_US);
  }
}

bool OmpStructureChecker::VerifyTraitPropertyLists(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  using TraitName = parser::OmpTraitSelectorName;
  using PropertyList = std::list<parser::OmpTraitProperty>;
  auto &[traitName, maybeProps]{trait.t};

  auto checkPropertyList{[&](const PropertyList &properties, auto isValid,
                             const std::string &message) {
    bool foundInvalid{false};
    for (const parser::OmpTraitProperty &prop : properties) {
      if (!isValid(prop)) {
        if (foundInvalid) {
          context_.Say(
              prop.source, "More invalid properties are present"_err_en_US);
          break;
        }
        context_.Say(prop.source, "%s"_err_en_US, message);
        foundInvalid = true;
      }
    }
    return !foundInvalid;
  }};

  bool invalid{false};

  if (std::holds_alternative<llvm::omp::Directive>(traitName.u)) {
    // Directive-name traits don't have properties.
    if (maybeProps) {
      context_.Say(trait.source,
          "Directive-name traits cannot have properties"_err_en_US);
      invalid = true;
    }
  }
  // Ignore properties on extension traits.

  // See `TraitSelectorParser` in openmp-parser.cpp
  if (auto *v{std::get_if<TraitName::Value>(&traitName.u)}) {
    switch (*v) {
    // name-list properties
    case parser::OmpTraitSelectorName::Value::Arch:
    case parser::OmpTraitSelectorName::Value::Extension:
    case parser::OmpTraitSelectorName::Value::Isa:
    case parser::OmpTraitSelectorName::Value::Kind:
    case parser::OmpTraitSelectorName::Value::Uid:
    case parser::OmpTraitSelectorName::Value::Vendor:
      if (maybeProps) {
        auto isName{[](const parser::OmpTraitProperty &prop) {
          return std::holds_alternative<parser::OmpTraitPropertyName>(prop.u);
        }};
        invalid = !checkPropertyList(std::get<PropertyList>(maybeProps->t),
            isName, "Trait property should be a name");
      }
      break;
    // clause-list
    case parser::OmpTraitSelectorName::Value::Atomic_Default_Mem_Order:
    case parser::OmpTraitSelectorName::Value::Requires:
    case parser::OmpTraitSelectorName::Value::Simd:
      if (maybeProps) {
        auto isClause{[&](const parser::OmpTraitProperty &prop) {
          return GetClauseFromProperty(prop).has_value();
        }};
        invalid = !checkPropertyList(std::get<PropertyList>(maybeProps->t),
            isClause, "Trait property should be a clause");
      }
      break;
    // expr-list
    case parser::OmpTraitSelectorName::Value::Condition:
    case parser::OmpTraitSelectorName::Value::Device_Num:
      if (maybeProps) {
        auto isExpr{[](const parser::OmpTraitProperty &prop) {
          return std::holds_alternative<parser::ScalarExpr>(prop.u);
        }};
        invalid = !checkPropertyList(std::get<PropertyList>(maybeProps->t),
            isExpr, "Trait property should be a scalar expression");
      }
      break;
    } // switch
  }

  return !invalid;
}

void OmpStructureChecker::CheckTraitSelector(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  using TraitName = parser::OmpTraitSelectorName;
  auto &[traitName, maybeProps]{trait.t};

  // Only do the detailed checks if the property lists are valid.
  if (VerifyTraitPropertyLists(traitSet, trait)) {
    if (std::holds_alternative<llvm::omp::Directive>(traitName.u) ||
        std::holds_alternative<std::string>(traitName.u)) {
      // No properties here: directives don't have properties, and
      // we don't implement any extension traits now.
      return;
    }

    // Specific traits we want to check.
    // Limitations:
    // (1) The properties for these traits are defined in "Additional
    // Definitions for the OpenMP API Specification". It's not clear how
    // to define them in a portable way, and how to verify their validity,
    // especially if they get replaced by their integer values (in case
    // they are defined as enums).
    // (2) These are entirely implementation-defined, and at the moment
    // there is no known schema to validate these values.
    auto v{std::get<TraitName::Value>(traitName.u)};
    switch (v) {
    case TraitName::Value::Arch:
      // Unchecked, TBD(1)
      break;
    case TraitName::Value::Atomic_Default_Mem_Order:
      CheckTraitADMO(traitSet, trait);
      break;
    case TraitName::Value::Condition:
      CheckTraitCondition(traitSet, trait);
      break;
    case TraitName::Value::Device_Num:
      CheckTraitDeviceNum(traitSet, trait);
      break;
    case TraitName::Value::Extension:
      // Ignore
      break;
    case TraitName::Value::Isa:
      // Unchecked, TBD(1)
      break;
    case TraitName::Value::Kind:
      // Unchecked, TBD(1)
      break;
    case TraitName::Value::Requires:
      CheckTraitRequires(traitSet, trait);
      break;
    case TraitName::Value::Simd:
      CheckTraitSimd(traitSet, trait);
      break;
    case TraitName::Value::Uid:
      // Unchecked, TBD(2)
      break;
    case TraitName::Value::Vendor:
      // Unchecked, TBD(1)
      break;
    }
  }
}

void OmpStructureChecker::CheckTraitADMO(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  if (properties.size() != 1) {
    context_.Say(trait.source,
        "%s trait requires a single clause property"_err_en_US,
        parser::ToUpperCaseLetters(traitName.ToString()));
  } else {
    const parser::OmpTraitProperty &property{properties.front()};
    auto clauseId{*GetClauseFromProperty(property)};
    // Check that the clause belongs to the memory-order clause-set.
    // Clause sets will hopefully be autogenerated at some point.
    switch (clauseId) {
    case llvm::omp::Clause::OMPC_acq_rel:
    case llvm::omp::Clause::OMPC_acquire:
    case llvm::omp::Clause::OMPC_relaxed:
    case llvm::omp::Clause::OMPC_release:
    case llvm::omp::Clause::OMPC_seq_cst:
      break;
    default:
      context_.Say(property.source,
          "%s trait requires a clause from the memory-order clause set"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()));
    }

    using ClauseProperty = common::Indirection<parser::OmpClause>;
    if (!std::holds_alternative<ClauseProperty>(property.u)) {
      context_.Say(property.source,
          "Invalid clause specification for %s"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId)));
    }
  }
}

void OmpStructureChecker::CheckTraitCondition(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  if (properties.size() != 1) {
    context_.Say(trait.source,
        "%s trait requires a single expression property"_err_en_US,
        parser::ToUpperCaseLetters(traitName.ToString()));
  } else {
    const parser::OmpTraitProperty &property{properties.front()};
    auto &scalarExpr{std::get<parser::ScalarExpr>(property.u)};

    auto maybeType{GetDynamicType(scalarExpr.thing.value())};
    if (!maybeType || maybeType->category() != TypeCategory::Logical) {
      context_.Say(property.source,
          "%s trait requires a single LOGICAL expression"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()));
    }
  }
}

void OmpStructureChecker::CheckTraitDeviceNum(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  if (properties.size() != 1) {
    context_.Say(trait.source,
        "%s trait requires a single expression property"_err_en_US,
        parser::ToUpperCaseLetters(traitName.ToString()));
  }
  // No other checks at the moment.
}

void OmpStructureChecker::CheckTraitRequires(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  unsigned version{context_.langOptions().OpenMPVersion};
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  for (const parser::OmpTraitProperty &property : properties) {
    auto clauseId{*GetClauseFromProperty(property)};
    if (!llvm::omp::isAllowedClauseForDirective(
            llvm::omp::OMPD_requires, clauseId, version)) {
      context_.Say(property.source,
          "%s trait requires a clause from the requirement clause set"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()));
    }

    using ClauseProperty = common::Indirection<parser::OmpClause>;
    if (!std::holds_alternative<ClauseProperty>(property.u)) {
      context_.Say(property.source,
          "Invalid clause specification for %s"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId)));
    }
  }
}

void OmpStructureChecker::CheckTraitSimd(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  unsigned version{context_.langOptions().OpenMPVersion};
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  for (const parser::OmpTraitProperty &property : properties) {
    auto clauseId{*GetClauseFromProperty(property)};
    if (!llvm::omp::isAllowedClauseForDirective(
            llvm::omp::OMPD_declare_simd, clauseId, version)) {
      context_.Say(property.source,
          "%s trait requires a clause that is allowed on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()),
          parser::ToUpperCaseLetters(
              getDirectiveName(llvm::omp::OMPD_declare_simd)));
    }

    using ClauseProperty = common::Indirection<parser::OmpClause>;
    if (!std::holds_alternative<ClauseProperty>(property.u)) {
      context_.Say(property.source,
          "Invalid clause specification for %s"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId)));
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpDirectiveSpecification &x) {
  // OmpDirectiveSpecification exists on its own only in clauses on
  // METADIRECTIVE.
  // In other cases it's a part of other constructs that handle directive
  // context stack by themselves.
  if (!GetDirectiveNest(MetadirectiveNest) && !GetDirectiveNest(ApplyNest)) {
    return;
  }

  llvm::omp::Directive dirId{x.DirId()};
  if (const parser::OpenMPConstruct *meta{GetCurrentConstruct()}) {
    if (parser::Unwrap<parser::OmpDelimitedMetadirectiveDirective>(meta->u)) {
      unsigned version{context_.langOptions().OpenMPVersion};
      switch (llvm::omp::getDirectiveAssociation(dirId)) {
      case llvm::omp::Association::Block:
      case llvm::omp::Association::LoopNest:
      case llvm::omp::Association::LoopSeq:
        break;
      default:
        if (dirId != llvm::omp::Directive::OMPD_nothing) {
          context_.Say(x.DirName().source,
              "A directive in BEGIN %s should have a corresponding end-directive"_err_en_US,
              parser::omp::GetUpperName(
                  llvm::omp::Directive::OMPD_metadirective, version));
        }
      }
    }
  }

  PushContextAndClauseSets(
      std::get<parser::OmpDirectiveName>(x.t).source, dirId);

  // Record each variant directive. A loop-associated one is later validated
  // against the loop nest that follows the metadirective.
  if (dirId != llvm::omp::Directive::OMPD_metadirective) {
    metadirectiveLoopVariants_.push_back({currentWhenSelector_, &x});
  }
}

void OmpStructureChecker::Leave(const parser::OmpDirectiveSpecification &x) {
  if (GetDirectiveNest(MetadirectiveNest) || GetDirectiveNest(ApplyNest)) {
    dirContext_.pop_back();
  }
}

void OmpStructureChecker::Enter(const parser::OmpMetadirectiveDirective &x) {
  EnterDirectiveNest(MetadirectiveNest);
  metadirectiveLoopVariants_.clear();
}

void OmpStructureChecker::Leave(const parser::OmpMetadirectiveDirective &) {
  ExitDirectiveNest(MetadirectiveNest);
}

// Return true if the variant guarded by `selector` might be selected, so its
// associated loop must still be checked. Skip it only when it is provably
// unselectable.
static bool mayVariantBeSelected(
    const parser::traits::OmpContextSelectorSpecification *selector,
    SemanticsContext &context, OmpVariantMatchContext &matchContext) {
  if (!selector ||
      FindUnsupportedSelectorFeature(*selector, context) !=
          UnsupportedSelectorFeature::None) {
    return true;
  }
  llvm::omp::VariantMatchInfo vmi;
  (void)MakeVariantMatchInfo(vmi, *selector, context);
  const auto &required{vmi.RequiredTraits};
  using TP = llvm::omp::TraitProperty;
  // In the default "match all" mode, a required trait that can never match
  // (e.g. a compile-time-false condition or an invalid device/implementation
  // property) makes the variant unselectable. match_any and match_none tolerate
  // such traits, so only reject variants in the default mode.
  bool matchAll{
      !required.test(unsigned(TP::implementation_extension_match_any)) &&
      !required.test(unsigned(TP::implementation_extension_match_none))};
  if (matchAll &&
      (required.test(unsigned(TP::user_condition_false)) ||
          required.test(unsigned(TP::invalid)))) {
    return false;
  }
  // Matching a device or implementation selector requires a known target.
  if (context.targetTriple().empty()) {
    return true;
  }
  return llvm::omp::isVariantApplicableInContext(
      vmi, matchContext, /*DeviceOrImplementationSetOnly=*/true);
}

// Check a loop-associated metadirective's variants against the loop nest they
// apply to. The nest is not attached to the directive in the parse tree. It is
// the next executable construct, either a following sibling or the first
// execution-part construct for a declarative metadirective.
void OmpStructureChecker::Enter(const parser::ExecutionPartConstruct &x) {
  if (metadirectiveLoopVariants_.empty()) {
    return;
  }
  if (parser::Unwrap<parser::CompilerDirective>(x)) {
    return;
  }
  // This is the first construct after the metadirective. It consumes the
  // pending variants, whether or not it is a loop nest.
  if (!parser::Unwrap<parser::DoConstruct>(x)) {
    // A non-loop construct follows, so a loop-associated variant has no loop
    // nest to associate with.
    CheckMetadirectiveVariantsWithoutLoop();
    return;
  }

  // A loop nest follows. Take the pending variants off the worklist and
  // validate them against it.
  std::vector<MetadirectiveLoopVariant> variants;
  variants.swap(metadirectiveLoopVariants_);

  unsigned version{context_.langOptions().OpenMPVersion};
  LoopSequence sequence(x, version, /*allowAllLoops=*/true, &context_);
  const parser::DoConstruct &rootLoop{*parser::Unwrap<parser::DoConstruct>(x)};
  const auto &[haveSemantic, havePerfect]{sequence.depth()};

  const auto MsgRequiresCanonical{
      "This construct requires a canonical loop %s"_err_en_US};
  const auto MsgNotValidAffectedLoop{
      "%s is not a valid affected loop"_because_en_US};

  auto checkRootLoopCanonical =
      [&](const parser::OmpDirectiveSpecification &spec, bool isSequence) {
        parser::CharBlock source{*parser::GetSource(rootLoop)};
        Reason reason;
        if (rootLoop.IsDoWhile()) {
          reason.Say(source, MsgNotValidAffectedLoop, "DO WHILE loop");
        } else if (rootLoop.IsDoConcurrent() && !IsDoConcurrentLegal(version)) {
          reason.Say(source, MsgNotValidAffectedLoop, "DO CONCURRENT loop");
        } else if (!rootLoop.GetLoopControl()) {
          reason.Say(
              source, MsgNotValidAffectedLoop, "DO loop without loop control");
        }

        if (!reason) {
          return true;
        }

        auto &msg{context_.Say(spec.DirName().source, MsgRequiresCanonical,
            isSequence ? "sequence" : "nest")};
        reason.AttachTo(msg);
        return false;
      };

  // Build the matching context once for the static-applicability gate below.
  OmpVariantMatchContext matchContext{context_};

  for (const MetadirectiveLoopVariant &variant : variants) {
    const parser::OmpDirectiveSpecification *spec{variant.spec};
    // Skip variants that can never be selected on this compilation target so
    // that their associated loop is not diagnosed.
    if (!mayVariantBeSelected(variant.selector, context_, matchContext)) {
      continue;
    }
    auto assoc{llvm::omp::getDirectiveAssociation(spec->DirId())};
    if (assoc == llvm::omp::Association::LoopNest) {
      if (!checkRootLoopCanonical(*spec, /*isSequence=*/false)) {
        continue;
      }

      auto [needDepth, needPerfect]{
          GetAffectedNestDepthWithReason(*spec, version, &context_)};
      auto haveDepth{needPerfect ? havePerfect : haveSemantic};
      if (!needDepth || *needDepth.value <= 0 || !haveDepth ||
          *haveDepth.value <= 0) {
        continue;
      }
      if (*needDepth.value > *haveDepth.value) {
        std::string_view perfectTxt{needPerfect ? " perfect" : ""};
        auto &msg{context_.Say(spec->DirName().source,
            "This construct requires a%s nest of depth %" PRId64
            ", but the associated nest is a%s nest of depth %" PRId64
            ""_err_en_US,
            perfectTxt, *needDepth.value, perfectTxt, *haveDepth.value)};
        haveDepth.reason.AttachTo(msg);
        needDepth.reason.AttachTo(msg);
      } else {
        CheckRectangularNest(*spec, sequence);
      }
    } else if (assoc == llvm::omp::Association::LoopSeq) {
      (void)checkRootLoopCanonical(*spec, /*isSequence=*/true);
    }
  }
}

// Diagnose loop-associated metadirective variants that are not followed by a
// loop nest, either because the metadirective is the last construct in the
// execution part or because a non-loop construct follows it. Variants that
// cannot be selected on this target are skipped.
void OmpStructureChecker::CheckMetadirectiveVariantsWithoutLoop() {
  std::vector<MetadirectiveLoopVariant> variants;
  variants.swap(metadirectiveLoopVariants_);

  OmpVariantMatchContext matchContext{context_};
  const auto MsgShouldContainDoOr{
      "This construct should contain a DO-loop or a loop-%s-generating construct"_err_en_US};

  for (const MetadirectiveLoopVariant &variant : variants) {
    if (!mayVariantBeSelected(variant.selector, context_, matchContext)) {
      continue;
    }
    auto assoc{llvm::omp::getDirectiveAssociation(variant.spec->DirId())};
    if (assoc == llvm::omp::Association::LoopNest) {
      context_.Say(
          variant.spec->DirName().source, MsgShouldContainDoOr, "nest");
    } else if (assoc == llvm::omp::Association::LoopSeq) {
      context_.Say(
          variant.spec->DirName().source, MsgShouldContainDoOr, "sequence");
    }
  }
}

static const parser::traits::OmpContextSelectorSpecification *
getMatchClauseContextSelector(const parser::OmpDirectiveSpecification &spec) {
  for (const parser::OmpClause &clause : spec.Clauses().v) {
    if (clause.Id() == llvm::omp::Clause::OMPC_match)
      return &std::get<parser::OmpClause::Match>(clause.u).v.v;
  }
  return nullptr;
}

void OmpStructureChecker::CheckDeclareVariantUserConditions(
    const parser::OmpContextSelector &ctx) {
  using SetName = parser::OmpTraitSetSelectorName;
  using TraitName = parser::OmpTraitSelectorName;

  for (const parser::OmpTraitSetSelector &traitSet : ctx.v) {
    if (std::get<SetName>(traitSet.t).v != SetName::Value::User) {
      continue;
    }
    for (const parser::OmpTraitSelector &trait :
        std::get<std::list<parser::OmpTraitSelector>>(traitSet.t)) {
      const auto &traitName{std::get<TraitName>(trait.t)};
      if (!std::holds_alternative<TraitName::Value>(traitName.u) ||
          std::get<TraitName::Value>(traitName.u) !=
              TraitName::Value::Condition) {
        continue;
      }
      const auto &maybeProps{
          std::get<std::optional<parser::OmpTraitSelector::Properties>>(
              trait.t)};
      if (!maybeProps) {
        continue;
      }
      const auto &properties{
          std::get<std::list<parser::OmpTraitProperty>>(maybeProps->t)};
      if (properties.size() != 1) {
        continue;
      }
      const parser::OmpTraitProperty &property{properties.front()};
      const parser::ScalarExpr &scalarExpr{
          std::get<parser::ScalarExpr>(property.u)};
      auto maybeType{GetDynamicType(scalarExpr.thing.value())};
      if (!maybeType || maybeType->category() != TypeCategory::Logical) {
        continue;
      }
      if (const auto *expr{GetExpr(scalarExpr)}) {
        if (!IsConstantExpr(*expr, &context_.foldingContext())) {
          context_.Say(property.source,
              "Run-time USER condition in the MATCH clause is not yet implemented"_err_en_US);
        }
      }
    }
  }
}

static bool IsProcedureOrFunction(const Symbol &symbol) {
  return IsProcedure(symbol) || IsFunction(symbol);
}

// OpenMP 6.0, 9.6 (declare_variant), Fortran restriction: "The characteristic
// of the function variant must be compatible with the characteristic of the
// base function after the implementation defined transformation for its OpenMP
// context."
static void CheckDeclareVariantInterface(SemanticsContext &context,
    const Symbol &base, const Symbol &variant, parser::CharBlock source) {
  auto &foldingContext{context.foldingContext()};
  auto baseChars{
      evaluate::characteristics::Procedure::Characterize(base, foldingContext)};
  auto variantChars{evaluate::characteristics::Procedure::Characterize(
      variant, foldingContext)};
  // If either procedure cannot be characterized, Characterize has already
  // emitted diagnostics; do not add more.
  if (!baseChars || !variantChars) {
    return;
  }

  std::string whyNot;
  if (!baseChars->IsCompatibleWith(
          *variantChars, /*ignoreImplicitVsExplicit=*/false, &whyNot)) {
    context.Say(source,
        "The variant procedure '%s' is not compatible with the base procedure '%s': %s"_err_en_US,
        variant.name(), base.name(), whyNot);
  }
}

// A declare-variant call is rewritten to the variant at every reference to the
// base, so the variant must be accessible at each of those references. The name
// of an internal procedure is only visible within its host, so it may only be a
// variant of a procedure that is internal to the same host; otherwise a
// reference to the base from elsewhere could not name the variant. OpenMP does
// not clearly specify this; Flang restricts it to avoid resolving a call to an
// internal procedure that is not accessible there. Returns true if the pairing
// is allowed.
static bool CheckVariantAccessibility(SemanticsContext &context,
    const Symbol &base, const Symbol &variant, parser::CharBlock source) {
  if (ClassifyProcedure(variant) != ProcedureDefinitionClass::Internal) {
    return true;
  }
  if (ClassifyProcedure(base) == ProcedureDefinitionClass::Internal &&
      &base.owner() == &variant.owner()) {
    return true;
  }
  context.Say(source,
      "The variant procedure '%s' is an internal procedure and is not accessible at every reference to the base procedure '%s'"_err_en_US,
      variant.name(), base.name());
  return false;
}

void OmpStructureChecker::CheckOmpDeclareVariantDirective(
    const parser::OmpDeclareVariantDirective &x) {
  const parser::OmpDirectiveSpecification &spec{x.v};
  const parser::OmpArgumentList &args{spec.Arguments()};

  bool hasArgModifiers{false};
  for (const parser::OmpClause &clause : x.v.Clauses().v) {
    if (clause.Id() == llvm::omp::Clause::OMPC_adjust_args) {
      hasArgModifiers = true;
      context_.Say(clause.source,
          "ADJUST_ARGS clause on the DECLARE VARIANT directive is not yet implemented"_err_en_US);
    } else if (clause.Id() == llvm::omp::Clause::OMPC_append_args) {
      hasArgModifiers = true;
      context_.Say(clause.source,
          "APPEND_ARGS clause on the DECLARE VARIANT directive is not yet implemented"_err_en_US);
    }
  }

  if (args.v.size() != 1) {
    context_.Say(args.source,
        "DECLARE_VARIANT directive should have a single argument"_err_en_US);
    return;
  }

  auto InvalidArgument{[&](parser::CharBlock source) {
    context_.Say(source,
        "The argument to the DECLARE_VARIANT directive should be [base-name:]variant-name"_err_en_US);
  }};

  auto CheckProcedureSymbol{[&](const Symbol *sym, parser::CharBlock source) {
    if (sym) {
      if (!IsProcedureOrFunction(*sym)) {
        auto &msg{context_.Say(source,
            "The name '%s' should refer to a procedure"_err_en_US,
            sym->name())};
        if (sym->test(Symbol::Flag::Implicit)) {
          msg.Attach(source, "The name '%s' has been implicitly declared"_en_US,
              sym->name());
        }
      }
    } else {
      InvalidArgument(source);
    }
  }};

  const Symbol *base{nullptr};
  const Symbol *variant{nullptr};
  const parser::OmpArgument &arg{args.v.front()};
  common::visit( //
      common::visitors{
          [&](const parser::OmpBaseVariantNames &y) {
            base = GetObjectSymbol(std::get<0>(y.t));
            variant = GetObjectSymbol(std::get<1>(y.t));
            CheckProcedureSymbol(base, arg.source);
            CheckProcedureSymbol(variant, arg.source);
          },
          [&](const parser::OmpObject &y) {
            variant = GetArgumentSymbol(arg);
            CheckProcedureSymbol(variant, arg.source);
            const Scope &containingScope{context_.FindScope(x.source)};
            if (const Symbol *host{
                    GetProgramUnitContaining(containingScope).symbol()}) {
              base = host;
            }
          },
          [&](auto &&y) { InvalidArgument(arg.source); },
      },
      arg.u);

  const parser::traits::OmpContextSelectorSpecification *matchSelector{
      getMatchClauseContextSelector(spec)};
  if (!matchSelector) {
    context_.Say(x.source,
        "DECLARE_VARIANT directive requires a MATCH clause"_err_en_US);
    return;
  }

  // Only validate and record the pairing when both names resolve to procedures.
  // Otherwise a diagnostic was already issued (see CheckProcedureSymbol), and
  // proceeding would emit spurious follow-on errors.
  if (base && variant && IsProcedureOrFunction(*base) &&
      IsProcedureOrFunction(*variant)) {
    base = &base->GetUltimate();
    variant = &variant->GetUltimate();
    if (base == variant) {
      context_.Say(arg.source,
          "The variant procedure must differ from the base procedure"_err_en_US);
    } else if (!declareVariantPairs_.emplace(base, variant).second) {
      context_.Say(arg.source,
          "Variant '%s' was already specified for '%s' in another DECLARE VARIANT directive"_err_en_US,
          variant->name(), base->name());
    } else if (CheckVariantAccessibility(
                   context_, *base, *variant, arg.source)) {
      if (!hasArgModifiers) {
        // adjust_args/append_args perform the "transformation for its OpenMP
        // context", so the variant interface intentionally differs from the
        // base; skip the same-interface check until they are supported.
        CheckDeclareVariantInterface(context_, *base, *variant, arg.source);
      }
      // Record the variant on its base procedure.
      if (base->has<SubprogramDetails>()) {
        auto &details{const_cast<Symbol &>(*base).get<SubprogramDetails>()};
        details.addOmpDeclareVariant(
            OmpDeclareVariantEntry{*variant, matchSelector});
      }
    }
  }

  EnterDirectiveNest(ContextSelectorNest);
  CheckContextSelectorSpecification(*matchSelector);
  CheckDeclareVariantUserConditions(*matchSelector);
  ExitDirectiveNest(ContextSelectorNest);
}

void OmpStructureChecker::Enter(const parser::OmpDeclareVariantDirective &x) {
  CheckOmpDeclareVariantDirective(x);
}

} // namespace Fortran::semantics
