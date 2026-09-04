//===-- flang/lib/Semantics/openmp-modifiers.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_MODIFIERS_H_
#define FORTRAN_SEMANTICS_OPENMP_MODIFIERS_H_

#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/semantics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Frontend/OpenMP/OMPDescriptors.h"

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
const llvm::omp::descriptor::Modifier &OmpGetDescriptor(
    const UnionTy &modifier) {
  return common::visit(
      [](auto &&m) -> decltype(auto) {
        using SpecificTy = llvm::remove_cvref_t<decltype(m)>;
        return llvm::omp::getDescriptor(SpecificTy::Id);
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
} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_OPENMP_MODIFIERS_H_
