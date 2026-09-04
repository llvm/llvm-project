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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/Directive/Spelling.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Frontend/OpenMP/OMPDescriptors.h"

#include <algorithm>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

namespace Fortran::semantics {
using namespace Fortran::parser::omp;

template <typename T> struct SetTypeFor {
  using type = llvm::omp::EnumSet<T,
      llvm::to_underlying(T::Last_) - llvm::to_underlying(T::First_) + 1>;
};

static llvm::omp::Modifiers getElements(
    const llvm::omp::descriptor::Clause &cdesc, llvm::omp::Version version) {
  return cdesc.getModifiers(version);
}

static llvm::omp::Modifiers getElements(
    const llvm::omp::descriptor::ModifierSet &sdesc,
    llvm::omp::Version version) {
  return sdesc.getModifiers(version);
}

static llvm::omp::ModifierSets getSets(
    const llvm::omp::descriptor::Clause &cdesc, llvm::omp::Version version) {
  return cdesc.getModifierSets(version);
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ResultTy = llvm::DenseMap<ElemTy,
        std::pair<parser::CharBlock, llvm::directive::VersionRange>>>
static ResultTy VerifyVersions(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  ResultTy result;

  auto &odesc{llvm::omp::getDescriptor(ownerId)};
  auto elements{getElements(odesc, version)};

  for (const AppliedElementTy &elem : info.elements) {
    if (elements.test(elem.id.value)) {
      continue;
    }
    llvm::omp::Version since{~0u}, until{0u};
    for (llvm::omp::Version v : odesc.getVersions()) {
      if (getElements(odesc, v).test(elem.id.value)) {
        if (v < version) {
          until = std::max(until, v);
        } else if (v > version) {
          since = std::min(since, v);
        }
      }
    }
    int minVer = static_cast<unsigned>(since);
    int maxVer = static_cast<unsigned>(until);
    result.insert({elem.id.value,
        {elem.id.source, llvm::directive::VersionRange{minVer, maxVer}}});
  }
  return result;
}

template < //
    typename ElemTy, typename SetsSetTy, typename OwnerTy,
    typename ElemSetTy = typename SetTypeFor<ElemTy>::type,
    typename ResultTy = std::pair<ElemSetTy, SetsSetTy>>
static ResultTy VerifyRequired(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  ResultTy required;
  auto &odesc{llvm::omp::getDescriptor(ownerId)};

  for (auto e : getElements(odesc, version)) {
    auto &edesc{llvm::omp::getDescriptor(e)};
    if (edesc.getProperties(version).test(llvm::omp::Property::Required)) {
      required.first.set(e);
    }
  }
  for (auto s : getSets(odesc, version)) {
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
  using ElemSetTy = typename SetTypeFor<ElemTy>::type;
  ElemSetTy unique;

  auto &odesc{llvm::omp::getDescriptor(ownerId)};
  auto elements{getElements(odesc, version)};

  for (auto e : elements) {
    auto &edesc{llvm::omp::getDescriptor(e)};
    // Exclusive modifiers should have the "unique" property present as well.
    if (edesc.getProperties(version).test(llvm::omp::Property::Unique)) {
      unique.set(e);
    }
  }
  for (auto s : getSets(odesc, version)) {
    auto &sdesc{llvm::omp::getDescriptor(s)};
    if (sdesc.getProperties(version).test(llvm::omp::Property::Unique)) {
      unique |= getElements(sdesc, version);
    }
  }

  ResultTy repeated;
  llvm::DenseMap<ElemTy, parser::CharBlock> present;
  for (const AppliedElementTy &elem : info.elements) {
    if (!elements.test(elem.id.value)) {
      // Skip invalid elements.
      continue;
    }
    if (unique.test(elem.id.value)) {
      auto [where, inserted]{present.insert({elem.id.value, elem.id.source})};
      if (!inserted) {
        repeated.insert({elem.id.value, {where->second, elem.id.source}});
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

  auto &odesc{llvm::omp::getDescriptor(ownerId)};
  auto elements{getElements(odesc, version)};

  llvm::DenseMap<ElemTy, parser::CharBlock> present;
  for (const AppliedElementTy &elem : info.elements) {
    if (!elements.test(elem.id.value)) {
      // Skip invalid elements.
      continue;
    }
    present.insert({elem.id.value, elem.id.source});
  }

  for (auto [id, source] : present) {
    auto &edesc{llvm::omp::getDescriptor(id)};
    if (!edesc.getProperties(version).test(llvm::omp::Property::Exclusive)) {
      continue;
    }
    // Element is exclusive, it cannot coexist with any other element.
    for (auto [otherId, otherSource] : present) {
      if (otherId != id) {
        result.insert({id, {otherId, source, otherSource}});
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

  auto &odesc{llvm::omp::getDescriptor(ownerId)};
  auto elements{getElements(odesc, version)};

  llvm::DenseMap<SetTy, const AppliedElementTy *> exclusive;
  for (const AppliedElementTy &elem : info.elements) {
    if (!elements.test(elem.id.value)) {
      // Skip invalid elements.
      continue;
    }
    for (auto s : elem.sets) {
      auto &sdesc{llvm::omp::getDescriptor(s)};
      if (!sdesc.getProperties(version).test(llvm::omp::Property::Exclusive)) {
        continue;
      }
      auto [where, inserted]{exclusive.insert({s, &elem})};
      if (!inserted) {
        const AppliedElementTy *prev{where->second};
        if (prev->id.value != elem.id.value) {
          result.insert({elem.id.value,
              {prev->id.value, elem.id.source, prev->id.source}});
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

  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;
  using ElemSetTy = typename SetTypeFor<ElemTy>::type;
  ElemSetTy ultimate;

  auto &odesc{llvm::omp::getDescriptor(ownerId)};
  auto elements{getElements(odesc, version)};

  for (auto e : elements) {
    auto &edesc{llvm::omp::getDescriptor(e)};
    if (edesc.getProperties(version).test(llvm::omp::Property::Ultimate)) {
      ultimate.set(e);
    }
  }
  for (auto s : getSets(odesc, version)) {
    auto &sdesc{llvm::omp::getDescriptor(s)};
    if (sdesc.getProperties(version).test(llvm::omp::Property::Ultimate)) {
      ultimate |= getElements(sdesc, version);
    }
  }

  // Check if there is an ultimate modifier that is in a wrong position.
  auto rest{last
          ? llvm::ArrayRef<AppliedElementTy>(info.elements).drop_back(1)
          : llvm::ArrayRef<AppliedElementTy>(info.elements).drop_front(1)};

  for (const AppliedElementTy &elem : rest) {
    if (!elements.test(elem.id.value)) {
      // Skip invalid elements.
      continue;
    }
    if (ultimate.test(elem.id.value)) {
      result.insert({elem.id.value, elem.id.source});
    }
  }

  return result;
}

bool OmpStructureChecker::VerifyModifierVersion(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  // Verify that the specified modifiers are allowed in this version.
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto result = VerifyVersions(info, clause.value, version);

  for (auto &[m, svr] : result) {
    std::string modName{llvm::omp::getDescriptor(m).getName().str()};
    std::string clauseName{GetUpperName(clause.value, version)};
    llvm::omp::Version since(svr.second.Min);
    llvm::omp::Version until(svr.second.Max);

    if (since == ~0u && until == 0u) {
      // This shouldn't really happen, but have it just in case.
      context_.Say(svr.first,
          "'%s' modifier is not supported on %s clause"_err_en_US, modName,
          clauseName);
    } else if (since != ~0u && version < since) {
      context_.Say(svr.first,
          "'%s' modifier is not supported in %s on %s clause, %s"_warn_en_US,
          modName, omp::ThisVersion(version), clauseName,
          omp::TryVersion(since));
    } else if (until != 0u && version > until) {
      context_.Say(svr.first,
          "'%s' modifier is no longer supported in %s on %s clause"_warn_en_US,
          modName, omp::ThisVersion(version), clauseName);
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
    context_.Say(clause.source, "'%s' modifier is required"_err_en_US,
        mdesc.getName().str());
  }
  for (llvm::omp::ModifierSet s : result.second) {
    auto &sdesc{llvm::omp::getDescriptor(s)};
    // If the group is required, at least one modifier from that group must
    // be present.
    if (llvm::omp::isModifierGroup(s)) {
      context_.Say(clause.source,
          "modifier from '%s' modifier group is required"_err_en_US,
          sdesc.getName().str());
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
            mdesc.getName().str())
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
            llvm::omp::getDescriptor(id).getName().str())
        .Attach(otherSource, "'%s' provided here"_en_US,
            llvm::omp::getDescriptor(otherId).getName().str());
  }

  auto resultMut = VerifyMutuallyExclusive(info, clause.value, version);

  for (auto [id, wrong] : resultMut) {
    auto [otherId, source, otherSource] = wrong;
    auto thisName{llvm::omp::getDescriptor(id).getName().str()};
    context_
        .Say(otherSource,
            "The '%s' and '%s' modifiers are mutually exclusive"_err_en_US,
            llvm::omp::getDescriptor(otherId).getName().str(), thisName)
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
        llvm::omp::getDescriptor(id).getName().str(), expected);
  }

  return result.empty();
}

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
            for (auto s : cdesc.getModifierSets(version)) {
              auto &sdesc{llvm::omp::getDescriptor(s)};
              if (sdesc.getModifiers(version).test(am.id.value)) {
                am.sets.set(s);
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

bool OmpStructureChecker::VerifyModifiers(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  // Run all checks without short-circuiting, return 'true' if all succeed.
  bool valid[]{
      VerifyModifierVersion(clause, info),
      VerifyModifierRequired(clause, info),
      VerifyModifierUnique(clause, info),
      VerifyModifierUltimate(clause, info),
      VerifyModifierExclusive(clause, info),
  };

  return llvm::all_of(valid, [](bool x) { return x; });
}

void OmpStructureChecker::VerifyModifiers(const parser::OmpClause &x) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};
  llvm::omp::Clause id{x.Id()};
  auto clauseId{WithSource(id, x.source)};
  switch (id) {
  case llvm::omp::Clause::OMPC_ompx_bare:
  case llvm::omp::Clause::OMPC_cancellation_construct_type:
    // Those are extensions/synthetic clauses and they don't have descriptors.
    break;
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
        VerifyModifiers(
            clauseId, GetAppliedModifiers(id, version, OmpGetModifiers(as)));
      }
    }
    break;
  }
  default:
    VerifyModifiers(clauseId, GetAppliedModifiers(x, version));
    break;
  }
}
} // namespace Fortran::semantics
