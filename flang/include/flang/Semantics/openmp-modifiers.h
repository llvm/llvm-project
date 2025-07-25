//===-- flang/lib/Semantics/openmp-modifiers.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_MODIFIERS_H_
#define FORTRAN_SEMANTICS_OPENMP_MODIFIERS_H_

#include "flang/Common/enum-set.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <variant>

namespace Fortran::semantics {

// Ref: [5.2:58]
//
// Syntactic properties for Clauses, Arguments and Modifiers
//
// Inverse properties:
//   not Required  -> Optional
//   not Unique    -> Repeatable
//   not Exclusive -> Compatible
//   not Ultimate  -> Free
//
// Clause defaults:   Optional, Repeatable, Compatible, Free
// Argument defaults: Required,     Unique, Compatible, Free
// Modifier defaults: Optional,     Unique, Compatible, Free
//
// ---
// Each modifier is used as either pre-modifier (i.e. modifier: item),
// or post-modifier (i.e. item: modifier). The default is pre-.
// Add an additional property that reflects the type of modifier.

ENUM_CLASS(OmpProperty, Required, Unique, Exclusive, Ultimate, Post)
using OmpProperties = common::EnumSet<OmpProperty, OmpProperty_enumSize>;
using OmpClauses =
    common::EnumSet<llvm::omp::Clause, llvm::omp::Clause_enumSize>;

struct OmpModifierDescriptor {
  // Modifier name for use in diagnostic messages.
  const OmpProperties &props(unsigned version) const;
  const OmpClauses &clauses(unsigned version) const;
  unsigned since(llvm::omp::Clause id) const;

  const llvm::StringRef name;
  // Version-dependent properties of the modifier.
  const std::map<unsigned, OmpProperties> props_;
  // Version-dependent set of clauses to which the modifier can apply.
  const std::map<unsigned, OmpClauses> clauses_;
};

template <typename SpecificTy> const OmpModifierDescriptor &OmpGetDescriptor();

#define DECLARE_DESCRIPTOR(name) \
  template <> const OmpModifierDescriptor &OmpGetDescriptor<name>()

DECLARE_DESCRIPTOR(parser::OmpAlignment);
DECLARE_DESCRIPTOR(parser::OmpAlignModifier);
DECLARE_DESCRIPTOR(parser::OmpAllocatorComplexModifier);
DECLARE_DESCRIPTOR(parser::OmpAllocatorSimpleModifier);
DECLARE_DESCRIPTOR(parser::OmpAlwaysModifier);
DECLARE_DESCRIPTOR(parser::OmpChunkModifier);
DECLARE_DESCRIPTOR(parser::OmpCloseModifier);
DECLARE_DESCRIPTOR(parser::OmpContextSelector);
DECLARE_DESCRIPTOR(parser::OmpDeleteModifier);
DECLARE_DESCRIPTOR(parser::OmpDependenceType);
DECLARE_DESCRIPTOR(parser::OmpDeviceModifier);
DECLARE_DESCRIPTOR(parser::OmpDirectiveNameModifier);
DECLARE_DESCRIPTOR(parser::OmpExpectation);
DECLARE_DESCRIPTOR(parser::OmpInteropPreference);
DECLARE_DESCRIPTOR(parser::OmpInteropType);
DECLARE_DESCRIPTOR(parser::OmpIterator);
DECLARE_DESCRIPTOR(parser::OmpLastprivateModifier);
DECLARE_DESCRIPTOR(parser::OmpLinearModifier);
DECLARE_DESCRIPTOR(parser::OmpMapper);
DECLARE_DESCRIPTOR(parser::OmpMapType);
DECLARE_DESCRIPTOR(parser::OmpMapTypeModifier);
DECLARE_DESCRIPTOR(parser::OmpOrderModifier);
DECLARE_DESCRIPTOR(parser::OmpOrderingModifier);
DECLARE_DESCRIPTOR(parser::OmpPrescriptiveness);
DECLARE_DESCRIPTOR(parser::OmpPresentModifier);
DECLARE_DESCRIPTOR(parser::OmpReductionIdentifier);
DECLARE_DESCRIPTOR(parser::OmpReductionModifier);
DECLARE_DESCRIPTOR(parser::OmpRefModifier);
DECLARE_DESCRIPTOR(parser::OmpSelfModifier);
DECLARE_DESCRIPTOR(parser::OmpStepComplexModifier);
DECLARE_DESCRIPTOR(parser::OmpStepSimpleModifier);
DECLARE_DESCRIPTOR(parser::OmpTaskDependenceType);
DECLARE_DESCRIPTOR(parser::OmpVariableCategory);
DECLARE_DESCRIPTOR(parser::OmpxHoldModifier);

#undef DECLARE_DESCRIPTOR

// Explanation of terminology:
//
// A typical clause with modifier[s] looks like this (with parts that are
// not relevant here removed):
//   struct OmpSomeClause {
//     struct Modifier {
//       using Variant = std::variant<Specific1, Specific2...>;
//       Variant u;
//     };
//     std::tuple<std::optional<std::list<Modifier>>, ...> t;
//   };
//
// The Specific1, etc. refer to parser classes that represent modifiers,
// e.g. OmpIterator or OmpTaskDependenceType. The Variant type contains
// all modifiers that are allowed for a given clause. The Modifier class
// is there to wrap the variant into the form that the parse tree visitor
// expects, i.e. with traits, member "u", etc.
//
// To avoid ambiguities with the word "modifier" (e.g. is it "any modifier",
// or "this specific modifier"?), the following code uses different terms:
//
// - UnionTy:    refers to the nested "Modifier" class, i.e.
//               "OmpSomeClause::Modifier" in the example above.
// - SpecificTy: refers to any of the alternatives, i.e. "Specific1" or
//               "Specific2".

template <typename UnionTy>
const OmpModifierDescriptor &OmpGetDescriptor(const UnionTy &modifier) {
  return common::visit(
      [](auto &&m) -> decltype(auto) {
        using SpecificTy = llvm::remove_cvref_t<decltype(m)>;
        return OmpGetDescriptor<SpecificTy>();
      },
      modifier.u);
}

/// Return the optional list of modifiers for a given `Omp[...]Clause`.
/// Specifically, the parameter type `ClauseTy` is the class that OmpClause::v
/// holds.
template <typename ClauseTy>
const std::optional<std::list<typename ClauseTy::Modifier>> &OmpGetModifiers(
    const ClauseTy &clause) {
  using UnionTy = typename ClauseTy::Modifier;
  return std::get<std::optional<std::list<UnionTy>>>(clause.t);
}

namespace detail {
/// Finds the first entry in the iterator range that holds the `SpecificTy`
/// alternative, or the end iterator if it does not exist.
/// The `SpecificTy` should be provided, the `UnionTy` is expected to be
/// auto-deduced, e.g.
///   const std::optional<std::list<X>> &modifiers = ...
///   ... = findInRange<OmpIterator>(modifiers->begin(), modifiers->end());
template <typename SpecificTy, typename UnionTy>
typename std::list<UnionTy>::const_iterator findInRange(
    typename std::list<UnionTy>::const_iterator begin,
    typename std::list<UnionTy>::const_iterator end) {
  for (auto it{begin}; it != end; ++it) {
    if (std::holds_alternative<SpecificTy>(it->u)) {
      return it;
    }
  }
  return end;
}
} // namespace detail

/// Finds the first entry in the list that holds the `SpecificTy` alternative,
/// and returns the pointer to that alternative. If such an entry does not
/// exist, it returns nullptr.
template <typename SpecificTy, typename UnionTy>
const SpecificTy *OmpGetUniqueModifier(
    const std::optional<std::list<UnionTy>> &modifiers) {
  const SpecificTy *found{nullptr};
  if (modifiers) {
    auto end{modifiers->cend()};
    auto at{detail::findInRange<SpecificTy, UnionTy>(modifiers->cbegin(), end)};
    if (at != end) {
      found = &std::get<SpecificTy>(at->u);
    }
  }
  return found;
}

template <typename SpecificTy> struct OmpSpecificModifierIterator {
  using VectorTy = std::vector<const SpecificTy *>;
  OmpSpecificModifierIterator(
      std::shared_ptr<VectorTy> list, typename VectorTy::const_iterator where)
      : specificList(list), at(where) {}

  OmpSpecificModifierIterator &operator++() {
    ++at;
    return *this;
  }
  // OmpSpecificModifierIterator &operator++(int);
  OmpSpecificModifierIterator &operator--() {
    --at;
    return *this;
  }
  // OmpSpecificModifierIterator &operator--(int);

  const SpecificTy *operator*() const { return *at; }
  bool operator==(const OmpSpecificModifierIterator &other) const {
    assert(specificList.get() == other.specificList.get() &&
        "comparing unrelated iterators");
    return at == other.at;
  }
  bool operator!=(const OmpSpecificModifierIterator &other) const {
    return !(*this == other);
  }

private:
  std::shared_ptr<VectorTy> specificList;
  typename VectorTy::const_iterator at;
};

template <typename SpecificTy, typename UnionTy>
llvm::iterator_range<OmpSpecificModifierIterator<SpecificTy>>
OmpGetRepeatableModifier(const std::optional<std::list<UnionTy>> &modifiers) {
  using VectorTy = std::vector<const SpecificTy *>;
  std::shared_ptr<VectorTy> items(new VectorTy);
  if (modifiers) {
    for (auto &m : *modifiers) {
      if (auto *s = std::get_if<SpecificTy>(&m.u)) {
        items->push_back(s);
      }
    }
  }
  return llvm::iterator_range(
      OmpSpecificModifierIterator(items, items->begin()),
      OmpSpecificModifierIterator(items, items->end()));
}

// Attempt to prevent creating a range based on an expiring modifier list.
template <typename SpecificTy, typename UnionTy>
llvm::iterator_range<OmpSpecificModifierIterator<SpecificTy>>
OmpGetRepeatableModifier(std::optional<std::list<UnionTy>> &&) = delete;

template <typename SpecificTy, typename UnionTy>
Fortran::parser::CharBlock OmpGetModifierSource(
    const std::optional<std::list<UnionTy>> &modifiers,
    const SpecificTy *specific) {
  if (!modifiers || !specific) {
    return Fortran::parser::CharBlock{};
  }
  for (auto &m : *modifiers) {
    if (std::get_if<SpecificTy>(&m.u) == specific) {
      return m.source;
    }
  }
  llvm_unreachable("`specific` must be a member of `modifiers`");
}

namespace detail {
template <typename T> constexpr const T *make_nullptr() {
  return static_cast<const T *>(nullptr);
}

/// Verify that all modifiers are allowed in the given OpenMP version.
template <typename UnionTy>
bool verifyVersions(const std::optional<std::list<UnionTy>> &modifiers,
    llvm::omp::Clause id, parser::CharBlock clauseSource,
    SemanticsContext &semaCtx) {
  if (!modifiers) {
    return true;
  }
  unsigned version{semaCtx.langOptions().OpenMPVersion};
  bool result{true};
  for (auto &m : *modifiers) {
    const OmpModifierDescriptor &desc{OmpGetDescriptor(m)};
    unsigned since{desc.since(id)};
    if (since == ~0u) {
      // This shouldn't really happen, but have it just in case.
      semaCtx.Say(m.source,
          "'%s' modifier is not supported on %s clause"_err_en_US,
          desc.name.str(),
          parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(id)));
    } else if (version < since) {
      semaCtx.Say(m.source,
          "'%s' modifier is not supported in OpenMP v%d.%d, try -fopenmp-version=%d"_warn_en_US,
          desc.name.str(), version / 10, version % 10, since);
      result = false;
    }
  }
  return result;
}

/// Helper function for verifying the Required property:
/// For a specific SpecificTy, if SpecificTy is has the Required property,
/// check if the list has an item that holds SpecificTy as an alternative.
/// If SpecificTy does not have the Required property, ignore it.
template <typename SpecificTy, typename UnionTy>
bool verifyIfRequired(const SpecificTy *,
    const std::optional<std::list<UnionTy>> &modifiers,
    parser::CharBlock clauseSource, SemanticsContext &semaCtx) {
  unsigned version{semaCtx.langOptions().OpenMPVersion};
  const OmpModifierDescriptor &desc{OmpGetDescriptor<SpecificTy>()};
  if (!desc.props(version).test(OmpProperty::Required)) {
    // If the modifier is not required, there is nothing to do.
    return true;
  }
  bool present{modifiers.has_value()};
  present = present && llvm::any_of(*modifiers, [](auto &&m) {
    return std::holds_alternative<SpecificTy>(m.u);
  });
  if (!present) {
    semaCtx.Say(
        clauseSource, "'%s' modifier is required"_err_en_US, desc.name.str());
  }
  return present;
}

/// Helper function for verifying the Required property:
/// Visit all specific types in UnionTy, and verify the Required property
/// for each one of them.
template <typename UnionTy, size_t... Idxs>
bool verifyRequiredPack(const std::optional<std::list<UnionTy>> &modifiers,
    parser::CharBlock clauseSource, SemanticsContext &semaCtx,
    std::integer_sequence<size_t, Idxs...>) {
  using VariantTy = typename UnionTy::Variant;
  return (verifyIfRequired(
              make_nullptr<std::variant_alternative_t<Idxs, VariantTy>>(),
              modifiers, clauseSource, semaCtx) &&
      ...);
}

/// Verify the Required property for the given list. Return true if the
/// list is valid, or false otherwise.
template <typename UnionTy>
bool verifyRequired(const std::optional<std::list<UnionTy>> &modifiers,
    llvm::omp::Clause id, parser::CharBlock clauseSource,
    SemanticsContext &semaCtx) {
  using VariantTy = typename UnionTy::Variant;
  return verifyRequiredPack(modifiers, clauseSource, semaCtx,
      std::make_index_sequence<std::variant_size_v<VariantTy>>{});
}

/// Helper function to verify the Unique property.
/// If SpecificTy has the Unique property, and an item is found holding
/// it as the alternative, verify that none of the elements that follow
/// hold SpecificTy as the alternative.
template <typename UnionTy, typename SpecificTy>
bool verifyIfUnique(const SpecificTy *,
    typename std::list<UnionTy>::const_iterator specific,
    typename std::list<UnionTy>::const_iterator end,
    SemanticsContext &semaCtx) {
  // `specific` is the location of the modifier of type SpecificTy.
  assert(specific != end && "`specific` must be a valid location");

  unsigned version{semaCtx.langOptions().OpenMPVersion};
  const OmpModifierDescriptor &desc{OmpGetDescriptor<SpecificTy>()};
  // Ultimate implies Unique.
  if (!desc.props(version).test(OmpProperty::Unique) &&
      !desc.props(version).test(OmpProperty::Ultimate)) {
    return true;
  }
  if (std::next(specific) != end) {
    auto next{
        detail::findInRange<SpecificTy, UnionTy>(std::next(specific), end)};
    if (next != end) {
      semaCtx.Say(next->source,
          "'%s' modifier cannot occur multiple times"_err_en_US,
          desc.name.str());
    }
  }
  return true;
}

/// Verify the Unique property for the given list. Return true if the
/// list is valid, or false otherwise.
template <typename UnionTy>
bool verifyUnique(const std::optional<std::list<UnionTy>> &modifiers,
    llvm::omp::Clause id, parser::CharBlock clauseSource,
    SemanticsContext &semaCtx) {
  if (!modifiers) {
    return true;
  }
  bool result{true};
  for (auto it{modifiers->cbegin()}, end{modifiers->cend()}; it != end; ++it) {
    result = common::visit(
                 [&](auto &&m) {
                   return verifyIfUnique<UnionTy>(&m, it, end, semaCtx);
                 },
                 it->u) &&
        result;
  }
  return result;
}

/// Verify the Ultimate property for the given list. Return true if the
/// list is valid, or false otherwise.
template <typename UnionTy>
bool verifyUltimate(const std::optional<std::list<UnionTy>> &modifiers,
    llvm::omp::Clause id, parser::CharBlock clauseSource,
    SemanticsContext &semaCtx) {
  if (!modifiers || modifiers->size() <= 1) {
    return true;
  }
  unsigned version{semaCtx.langOptions().OpenMPVersion};
  bool result{true};
  auto first{modifiers->cbegin()};
  auto last{std::prev(modifiers->cend())};

  // Any item that has the Ultimate property has to be either at the back
  // or at the front of the list (depending on whether it's a pre- or a post-
  // modifier).
  // Walk over the list, and if a given item has the Ultimate property but is
  // not at the right position, mark it as an error.
  for (auto it{first}, end{modifiers->cend()}; it != end; ++it) {
    result =
        common::visit(
            [&](auto &&m) {
              using SpecificTy = llvm::remove_cvref_t<decltype(m)>;
              const OmpModifierDescriptor &desc{OmpGetDescriptor<SpecificTy>()};
              auto &props{desc.props(version)};

              if (props.test(OmpProperty::Ultimate)) {
                bool isPre = !props.test(OmpProperty::Post);
                if (it == (isPre ? last : first)) {
                  // Skip, since this is the correct place for this modifier.
                  return true;
                }
                llvm::StringRef where{isPre ? "last" : "first"};
                semaCtx.Say(it->source,
                    "'%s' should be the %s modifier"_err_en_US, desc.name.str(),
                    where.str());
                return false;
              }
              return true;
            },
            it->u) &&
        result;
  }
  return result;
}

/// Verify the Exclusive property for the given list. Return true if the
/// list is valid, or false otherwise.
template <typename UnionTy>
bool verifyExclusive(const std::optional<std::list<UnionTy>> &modifiers,
    llvm::omp::Clause id, parser::CharBlock clauseSource,
    SemanticsContext &semaCtx) {
  if (!modifiers || modifiers->size() <= 1) {
    return true;
  }
  unsigned version{semaCtx.langOptions().OpenMPVersion};
  const UnionTy &front{modifiers->front()};
  const OmpModifierDescriptor &frontDesc{OmpGetDescriptor(front)};

  auto second{std::next(modifiers->cbegin())};
  auto end{modifiers->end()};

  auto emitErrorMessage{[&](const UnionTy &excl, const UnionTy &other) {
    const OmpModifierDescriptor &descExcl{OmpGetDescriptor(excl)};
    const OmpModifierDescriptor &descOther{OmpGetDescriptor(other)};
    parser::MessageFormattedText txt(
        "An exclusive '%s' modifier cannot be specified together with a modifier of a different type"_err_en_US,
        descExcl.name.str());
    parser::Message message(excl.source, txt);
    message.Attach(
        other.source, "'%s' provided here"_en_US, descOther.name.str());
    semaCtx.Say(std::move(message));
  }};

  if (frontDesc.props(version).test(OmpProperty::Exclusive)) {
    // If the first item has the Exclusive property, then check if there is
    // another item in the rest of the list with a different SpecificTy as
    // the alternative, and mark it as an error. This allows multiple Exclusive
    // items to coexist as long as they hold the same SpecificTy.
    bool result{true};
    size_t frontIndex{front.u.index()};
    for (auto it{second}; it != end; ++it) {
      if (it->u.index() != frontIndex) {
        emitErrorMessage(front, *it);
        result = false;
        break;
      }
    }
    return result;
  } else {
    // If the first item does not have the Exclusive property, then check
    // if there is an item in the rest of the list that is Exclusive, and
    // mark it as an error if so.
    bool result{true};
    for (auto it{second}; it != end; ++it) {
      const OmpModifierDescriptor &desc{OmpGetDescriptor(*it)};
      if (desc.props(version).test(OmpProperty::Exclusive)) {
        emitErrorMessage(*it, front);
        result = false;
        break;
      }
    }
    return result;
  }
}
} // namespace detail

template <typename ClauseTy>
bool OmpVerifyModifiers(const ClauseTy &clause, llvm::omp::Clause id,
    parser::CharBlock clauseSource, SemanticsContext &semaCtx) {
  auto &modifiers{OmpGetModifiers(clause)};
  bool results[]{//
      detail::verifyVersions(modifiers, id, clauseSource, semaCtx),
      detail::verifyRequired(modifiers, id, clauseSource, semaCtx),
      detail::verifyUnique(modifiers, id, clauseSource, semaCtx),
      detail::verifyUltimate(modifiers, id, clauseSource, semaCtx),
      detail::verifyExclusive(modifiers, id, clauseSource, semaCtx)};
  return llvm::all_of(results, [](bool x) { return x; });
}
} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_OPENMP_MODIFIERS_H_
