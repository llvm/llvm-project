//===-- lib/Semantics/check-omp-syntax.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"

#include "flang/Common/visit.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/openmp-utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/Directive/Spelling.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Frontend/OpenMP/OMPDescriptors.h"

#include <algorithm>
#include <limits>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

namespace Fortran::semantics {
using namespace Fortran::parser::omp;

template <typename ElemTy>
static llvm::omp::Properties GetProperties(
    ElemTy id, llvm::omp::Version version) {
  assert(version && "Expecting valid version");
  const auto &desc{llvm::omp::getDescriptor(id)};
  return desc.getProperties(version);
}

static llvm::omp::Version GetClosestVersion(
    llvm::directive::VersionRange range, llvm::omp::Version version) {
  if (range.isValid()) {
    int intVer{static_cast<int>(static_cast<unsigned>(version))};
    return llvm::omp::Version(
        intVer >= range.Min ? std::min(intVer, range.Max) : range.Min);
  }
  return llvm::omp::Version();
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ResultTy = llvm::DenseMap<ElemTy,
        std::pair<parser::CharBlock, llvm::directive::VersionRange>>>
static ResultTy VerifyVersions(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  using ElemSetTy = llvm::omp::EnumSet<ElemTy>;
  ResultTy result;

  auto &odesc{llvm::omp::getDescriptor(ownerId)};
  ElemSetTy allowed{descriptor::GetAllowedElements(odesc, version)};

  for (const AppliedElementTy &elem : info.elements) {
    if (!allowed.test(elem.id.value)) {
      result.insert({elem.id.value,
          {elem.id.source,
              descriptor::GetVersionRangeForElement(elem.id.value, ownerId)}});
    }
  }
  return result;
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ElemSetTy = llvm::omp::EnumSet<ElemTy>,
    typename ResultTy = std::pair<ElemSetTy, SetsSetTy>>
static ResultTy VerifyRequired(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  ResultTy required;
  auto &odesc{llvm::omp::getDescriptor(ownerId)};

  for (auto e : descriptor::GetElements(odesc, version)) {
    auto &edesc{llvm::omp::getDescriptor(e)};
    if (edesc.getProperties(version).test(llvm::omp::Property::Required)) {
      required.first.set(e);
    }
  }
  for (auto s : descriptor::GetSets(odesc, version)) {
    auto &sdesc{llvm::omp::getDescriptor(s)};
    if (sdesc.getProperties(version).test(llvm::omp::Property::Required)) {
      required.second.set(s);
    }
  }

  for (const AppliedElementTy &elem : info.elements) {
    required.first.reset(elem.id.value);
    required.second &= ~elem.sets;
  }

  return required;
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ResultTy =
        llvm::DenseMap<ElemTy, std::pair<parser::CharBlock, parser::CharBlock>>>
static ResultTy VerifyUnique(const AppliedElementInfo<ElemTy, SetsSetTy> &info,
    OwnerTy ownerId, llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;

  ResultTy repeated;
  llvm::DenseMap<ElemTy, parser::CharBlock> present;
  for (const AppliedElementTy &elem : info.elements) {
    if (!elem.version) {
      // Skip invalid elements.
      continue;
    }
    auto &edesc{llvm::omp::getDescriptor(elem.id.value)};
    if (edesc.getProperties(elem.version).test(llvm::omp::Property::Unique)) {
      auto [where, inserted]{present.insert({elem.id.value, elem.id.source})};
      if (!inserted) {
        repeated.insert({elem.id.value, {where->second, elem.id.source}});
      }
      continue;
    }
    for (auto s : elem.sets) {
      auto &sdesc{llvm::omp::getDescriptor(s)};
      if (sdesc.getProperties(elem.version).test(llvm::omp::Property::Unique)) {
        auto [where, inserted]{present.insert({elem.id.value, elem.id.source})};
        if (!inserted) {
          repeated.insert({elem.id.value, {where->second, elem.id.source}});
        }
        // One unique set is enough.
        break;
      }
    }
  }

  return repeated;
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ResultTy = llvm::DenseMap<ElemTy,
        std::tuple<ElemTy, parser::CharBlock, parser::CharBlock>>>
static ResultTy VerifyExclusive(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  ResultTy result;

  for (const AppliedElementTy &elem : info.elements) {
    if (!elem.version) {
      // Skip invalid elements.
      continue;
    }
    auto properties{GetProperties(elem.id.value, elem.version)};
    if (!properties.test(llvm::omp::Property::Exclusive)) {
      continue;
    }
    // Element is exclusive, it cannot coexist with any other element.
    for (const AppliedElementTy &other : info.elements) {
      if (other.version && other.id.value != elem.id.value) {
        result.insert(
            {elem.id.value, {other.id.value, elem.id.source, other.id.source}});
        break;
      }
    }
  }

  return result;
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ResultTy = llvm::DenseMap<ElemTy,
        std::tuple<ElemTy, parser::CharBlock, parser::CharBlock>>>
static ResultTy VerifyMutuallyExclusive(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  using SetTy = typename SetsSetTy::value_type;

  ResultTy result;

  // The are-mutually-exclusive relation is not symmetric here, since it
  // depends on version, and the applicable version may be different for
  // different elements.
  // For example, element A may be allowed in v1.0, element B may be
  // allowed in v2.0, and also the set {A, B} may be exclusive in v2.0.
  // If the current version is v1.0, the effective versions will be
  // 1.0 and 2.0 for A and B respectively.
  // When looking at A, there is no indication that it interacts with B
  // in any way, it's only when looking at B that the mutual-exclusivity
  // becomes evident.

  // First collect all exclusive sets that any specified element is a
  // member of in its applicable version.
  llvm::DenseMap<SetTy, llvm::SetVector<llvm::omp::Version>> exclusiveSets;
  for (const AppliedElementTy &elem : info.elements) {
    if (!elem.version) {
      // Skip invalid elements.
      continue;
    }
    for (auto s : elem.sets) {
      auto properties{GetProperties(s, elem.version)};
      if (properties.test(llvm::omp::Property::Exclusive)) {
        exclusiveSets[s].insert(elem.version);
      }
    }
  }

  // Then iterate over all specified elements and check is they are members
  // of some exclusive set for any applicable version for that set.
  llvm::DenseMap<SetTy, const AppliedElementTy *> exclusive;
  for (const AppliedElementTy &elem : info.elements) {
    if (!elem.version) {
      // Skip invalid elements.
      continue;
    }
    for (auto [s, versions] : exclusiveSets) {
      auto &sdesc{llvm::omp::getDescriptor(s)};
      for (llvm::omp::Version v : versions) {
        if (!descriptor::GetElements(sdesc, v).test(elem.id.value)) {
          continue;
        }
        auto [where, inserted]{exclusive.insert({s, &elem})};
        if (!inserted) {
          const AppliedElementTy *prev{where->second};
          if (prev->id.value != elem.id.value) {
            result.insert({elem.id.value,
                {prev->id.value, elem.id.source, prev->id.source}});
            // Stop version traversal.
            break;
          }
        }
      }
    }
  }

  return result;
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ResultTy = llvm::DenseMap<ElemTy, parser::CharBlock>>
static ResultTy VerifyUltimate(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version, bool last = true) {
  ResultTy result;
  if (info.elements.empty()) {
    return result;
  }

  // Check if there is an ultimate modifier that is in a wrong position.
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  auto rest{last
          ? llvm::ArrayRef<AppliedElementTy>(info.elements).drop_back(1)
          : llvm::ArrayRef<AppliedElementTy>(info.elements).drop_front(1)};

  for (const AppliedElementTy &elem : rest) {
    if (!elem.version) {
      // Skip invalid elements.
      continue;
    }
    auto properties{GetProperties(elem.id.value, elem.version)};
    if (properties.test(llvm::omp::Property::Ultimate)) {
      result.insert({elem.id.value, elem.id.source});
      continue;
    }
    for (auto s : elem.sets) {
      auto properties{GetProperties(s, elem.version)};
      if (properties.test(llvm::omp::Property::Ultimate)) {
        result.insert({elem.id.value, elem.id.source});
        break;
      }
    }
  }

  return result;
}

bool OmpStructureChecker::VerifyModifierVersion(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  // Verify that the specified modifiers are allowed in this version.
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};
  llvm::omp::Version maxVer{std::numeric_limits<int>::max()};

  auto result = VerifyVersions(info, clause.value, version);

  for (auto &[m, svr] : result) {
    std::string modName{llvm::omp::getDescriptor(m).getName().str()};
    std::string clauseName{GetUpperName(clause.value, version)};
    llvm::omp::Version since(svr.second.Min);
    llvm::omp::Version until(svr.second.Max);

    if (since == maxVer && until == 0u) {
      // This shouldn't really happen, but have it just in case.
      context_.Say(svr.first,
          "'%s' modifier is not supported on %s clause"_err_en_US, modName,
          clauseName);
    } else if (since != maxVer && version < since) {
      context_.Say(svr.first,
          "'%s' modifier is not supported on %s clause in %s, %s"_warn_en_US,
          modName, clauseName, omp::ThisVersion(version),
          omp::TryVersion(since));
    } else if (until != 0u && version > until) {
      context_.Say(svr.first,
          "'%s' modifier is no longer supported on %s clause in %s"_warn_en_US,
          modName, clauseName, omp::ThisVersion(version));
    }
  }

  return result.empty();
}

bool OmpStructureChecker::VerifyModifierRequired(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto result = VerifyRequired(info, clause.value, version);

  for (llvm::omp::Modifier m : result.first) {
    auto &mdesc{llvm::omp::getDescriptor(m)};
    context_.Say(
        clause.source, "'%s' modifier is required"_err_en_US, mdesc.getName());
  }
  for (llvm::omp::ModifierSet s : result.second) {
    auto &sdesc{llvm::omp::getDescriptor(s)};
    // If the group is required, at least one modifier from that group must
    // be present.
    if (llvm::omp::isModifierGroup(s)) {
      context_.Say(clause.source,
          "modifier from '%s' modifier group is required"_err_en_US,
          sdesc.getName());
    } else {
      context_.Say(clause.source,
          "modifier from the modifier set on %s clause is required"_err_en_US,
          GetUpperName(clause.value, version));
    }
  }

  return result.first.empty() && result.second.empty();
}

bool OmpStructureChecker::VerifyModifierUnique(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto result = VerifyUnique(info, clause.value, version);

  for (auto [id, where] : result) {
    auto &mdesc{llvm::omp::getDescriptor(id)};
    context_
        .Say(where.first, "'%s' modifier cannot occur multiple times"_err_en_US,
            mdesc.getName())
        .Attach(where.second, "previous occurrence of this modifier"_en_US);
  }

  return result.empty();
}

bool OmpStructureChecker::VerifyModifierExclusive(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto resultExcl = VerifyExclusive(info, clause.value, version);

  for (auto [id, wrong] : resultExcl) {
    auto [otherId, source, otherSource] = wrong;
    context_
        .Say(source,
            "An exclusive '%s' modifier cannot be specified together with a modifier of a different type"_err_en_US,
            llvm::omp::getDescriptor(id).getName())
        .Attach(otherSource, "'%s' provided here"_en_US,
            llvm::omp::getDescriptor(otherId).getName());
  }

  auto resultMut = VerifyMutuallyExclusive(info, clause.value, version);

  for (auto [id, wrong] : resultMut) {
    auto [otherId, source, otherSource] = wrong;
    auto thisName{llvm::omp::getDescriptor(id).getName().str()};
    context_
        .Say(otherSource,
            "The '%s' and '%s' modifiers are mutually exclusive"_err_en_US,
            llvm::omp::getDescriptor(otherId).getName(), thisName)
        .Attach(source, "'%s' modifier specified here"_en_US, thisName);
  }

  return resultExcl.empty() && resultMut.empty();
}

bool OmpStructureChecker::VerifyModifierUltimate(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};
  auto &cdesc{llvm::omp::getDescriptor(clause.value)};
  bool last{
      !cdesc.getProperties(version).test(llvm::omp::Property::PostModified)};
  std::string expected{last ? "last" : "first"};

  auto result = VerifyUltimate(info, clause.value, version, last);

  for (auto [id, where] : result) {
    context_.Say(where, "'%s' should be the %s modifier"_err_en_US,
        llvm::omp::getDescriptor(id).getName(), expected);
  }

  return result.empty();
}

// Collect the information about modifiers specified on the given clause.
// If a modifier is allowed on this clause in "version", store the list of
// modifier sets that the clause allows in "version" in AppliedModifier.
// If a modifier is not allowed on this clause in "version", but is allowed
// on it in another version v, store the list of modifier sets that the clause
// allows on v.
// In either case, store the applied version in AppliedModifier.
// If the modifier is not allowed in any version, the applied version will
// be the default (i.e. 0) and no sets will be stored.
template <typename UnionTy>
AppliedModifierInfo GetAppliedModifiers(llvm::omp::Clause clauseId,
    llvm::omp::Version version,
    const std::optional<std::list<UnionTy>> &modifiers) {
  AppliedModifierInfo info;
  if (modifiers) {
    auto cdesc{llvm::omp::getDescriptor(clauseId)};
    for (auto &m : *modifiers) {
      common::visit(
          [&](auto &&t) {
            auto &am{info.elements.emplace_back(AppliedModifier{})};
            am.id = WithSource{t.Id, m.source};
            am.version = GetClosestVersion(
                descriptor::GetVersionRangeForElement(am.id.value, clauseId),
                version);
            if (am.version) {
              for (auto s : cdesc.getModifierSets(am.version)) {
                auto &sdesc{llvm::omp::getDescriptor(s)};
                if (sdesc.getModifiers(am.version).test(am.id.value)) {
                  am.sets.set(s);
                }
              }
            }
          },
          m.u);
    }
  }
  return info;
}

static AppliedModifierInfo GetAppliedModifiersFromWrapper(
    llvm::omp::Clause clauseId, llvm::omp::Version version,
    const parser::OmpDependClause &depend) {
  using TaskDep = parser::OmpDependClause::TaskDep;
  if (auto *task{std::get_if<TaskDep>(&depend.u)}) {
    using Modifiers = std::optional<std::list<TaskDep::Modifier>>;
    return GetAppliedModifiers(
        llvm::omp::Clause::OMPC_depend, version, std::get<Modifiers>(task->t));
  } else if (auto *doa{std::get_if<parser::OmpDoacross>(&depend.u)}) {
    using Modifiers = std::optional<std::list<parser::OmpDoacross::Modifier>>;
    return GetAppliedModifiers(
        llvm::omp::Clause::OMPC_depend, version, std::get<Modifiers>(doa->t));
  }
  llvm_unreachable("Unexpected alternative in depend");
}

static AppliedModifierInfo GetAppliedModifiersFromWrapper(
    llvm::omp::Clause clauseId, llvm::omp::Version version,
    const parser::OmpDoacrossClause &doacross) {
  using Modifiers = std::optional<std::list<parser::OmpDoacross::Modifier>>;
  return GetAppliedModifiers(llvm::omp::Clause::OMPC_doacross, version,
      std::get<Modifiers>(doacross.v.t));
}

template <typename T>
static AppliedModifierInfo GetAppliedModifiersFromWrapper(
    llvm::omp::Clause clauseId, llvm::omp::Version version, const T &wrapper) {
  if constexpr (HasModifier<T>) {
    using Modifiers = std::optional<std::list<typename T::Modifier>>;
    return GetAppliedModifiers(
        clauseId, version, std::get<Modifiers>(wrapper.t));
  } else {
    return AppliedModifierInfo{};
  }
}

AppliedModifierInfo GetAppliedModifiers(
    const parser::OmpClause &clause, llvm::omp::Version version) {
  return common::visit(
      [&](auto &&s) {
        using TypeS = llvm::remove_cvref_t<decltype(s)>;
        if constexpr (WrapperTrait<TypeS>) {
          return GetAppliedModifiersFromWrapper(clause.Id(), version, s.v);
        } else {
          return AppliedModifierInfo{};
        }
      },
      clause.u);
}

bool OmpStructureChecker::VerifyModifierSyntax(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  // Run all checks without short-circuiting, return 'true' if all succeed.
  bool valid[]{
      VerifyModifierVersion(clause, info),
      VerifyModifierRequired(clause, info),
      VerifyModifierUnique(clause, info),
      VerifyModifierExclusive(clause, info),
      VerifyModifierUltimate(clause, info),
  };

  return llvm::all_of(valid, [](bool x) { return x; });
}

void OmpStructureChecker::VerifyModifierSyntax(const parser::OmpClause &x) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};
  llvm::omp::Clause id{x.Id()};
  auto clauseId{WithSource(id, x.source)};
  switch (id) {
  case llvm::omp::Clause::OMPC_uses_allocators: {
    // The traits of the deprecated syntax are stored as a traits-array
    // modifier, but they are not the 5.2 modifier, so they must not be
    // version-checked. A modifier that postdates the OpenMP version in effect
    // is only warned about, so the specification is accepted as an extension
    // and must still be checked, otherwise a malformed one would reach lowering
    // unvalidated.
    auto &uac{parser::UnwrapRef<parser::OmpUsesAllocatorsClause>(x)};
    for (auto &&as : uac.v) {
      bool legacy{std::get<bool>(as.t)};
      if (!legacy) {
        VerifyModifierSyntax(
            clauseId, GetAppliedModifiers(id, version, OmpGetModifiers(as)));
      }
    }
    break;
  }
  default:
    VerifyModifierSyntax(clauseId, GetAppliedModifiers(x, version));
    break;
  }
}
} // namespace Fortran::semantics
