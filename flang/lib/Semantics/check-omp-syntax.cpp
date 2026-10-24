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
#include "llvm/ADT/StringExtras.h"
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

static llvm::omp::Version NextVersion(llvm::omp::Version version) {
  auto versions{llvm::omp::getOpenMPVersions()};
  for (auto [idx, ver] : llvm::enumerate(versions)) {
    if (ver == version && idx + 1 < versions.size()) {
      return versions[idx + 1];
    }
  }
  return llvm::omp::Version();
}

static std::string EnumSetToString(
    llvm::omp::Clauses set, llvm::omp::Version version) {
  llvm::SmallVector<std::string> names;
  for (llvm::omp::Clause c : set) {
    names.emplace_back(GetUpperName(c, version));
  }
  if (names.size() == 1) {
    return names.front();
  }
  return llvm::join(llvm::ArrayRef(names).drop_back(), ", ") + " or " +
      names.back();
}

static std::string EnumSetToString(
    llvm::omp::Modifiers set, llvm::omp::Version version) {
  llvm::SmallVector<std::string> names;
  for (llvm::omp::Modifier m : set) {
    names.emplace_back(llvm::omp::getDescriptor(m).getName().str());
  }
  if (names.size() == 1) {
    return names.front();
  }
  return llvm::join(llvm::ArrayRef(names).drop_back(), ", ") + " or " +
      names.back();
}

static std::string OneOfClauses(
    llvm::omp::ClauseSet set, llvm::omp::Version version) {
  auto &sdesc{llvm::omp::getDescriptor(set)};
  llvm::omp::Clauses members{sdesc.getClauses(version)};

  if (size_t count{members.size()}; count == 1) {
    return EnumSetToString(members, version) + " clause";
  } else if (count > 1) {
    return "One of " + EnumSetToString(members, version) + " clauses";
  }
  return "";
}

static std::string OneOfModifiers(
    llvm::omp::ModifierSet set, llvm::omp::Version version) {
  auto &sdesc{llvm::omp::getDescriptor(set)};
  llvm::omp::Modifiers members{sdesc.getModifiers(version)};

  if (size_t count{members.size()}; count == 1) {
    return EnumSetToString(members, version) + " modifier";
  } else if (count > 1) {
    return "One of " + EnumSetToString(members, version) + " modifiers";
  }
  return "";
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
    typename SetTy = typename SetsSetTy::value_type,
    typename ResultTy = llvm::DenseMap<ElemTy,
        std::tuple<ElemTy, SetTy, parser::CharBlock, parser::CharBlock>>>
static ResultTy VerifyMutuallyExclusive(
    const AppliedElementInfo<ElemTy, SetsSetTy> &info, OwnerTy ownerId,
    llvm::omp::Version version) {
  using AppliedElementTy = AppliedElement<ElemTy, SetsSetTy>;

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
                {prev->id.value, s, elem.id.source, prev->id.source}});
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

bool OmpStructureChecker::VerifyClauseVersion(
    parser::OmpDirectiveName dirName, const AppliedClauseInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};
  llvm::omp::Version maxVer{std::numeric_limits<int>::max()};
  bool isCancel{dirName.v == llvm::omp::Directive::OMPD_cancel ||
      dirName.v == llvm::omp::Directive::OMPD_cancellation_point};

  auto result = VerifyVersions(info, dirName.v, version);

  for (auto &[c, svr] : result) {
    std::string cname{GetUpperName(c, version)};
    std::string dname{GetUpperName(dirName.v, version)};
    llvm::omp::Version since(svr.second.Min);
    llvm::omp::Version until(svr.second.Max);

    // Cancellation construct type clauses are directive names. They are
    // only allowed on CANCEL and CANCELLATION_POINT directives. They may
    // appear as byproducts of parsing an invalid directive name,
    // e.g. SECTIONS PARALLEL, where SECTIONS will be the directive name,
    // and PARALLEL will be a cancellation-construct-type clause.
    // This may cause confusing error messages to be emitted, so deal with
    // these cases separately.
    if (!isCancel && c == llvm::omp::Clause::OMPC_cancellation_construct_type) {
      context_.Say(svr.first, "%s cannot follow %s"_err_en_US,
          parser::ToUpperCaseLetters(svr.first.ToString()), dname);
      continue;
    }

    if (since == maxVer && until == 0u) {
      context_.Say(svr.first,
          "%s clause is not allowed on %s directive"_err_en_US, cname, dname);
    } else if (since != maxVer && version < since) {
      context_.Warn(common::UsageWarning::OpenMPFuture, svr.first,
          "%s clause is not allowed on %s directive in %s, %s"_warn_en_US,
          cname, dname, omp::ThisVersion(version), omp::TryVersion(since));
      SetAllowedClauseOverride(c, dirName.v, since);
    } else if (until != 0u && version > until) {
      context_.Warn(common::UsageWarning::OpenMPDeprecated, svr.first,
          "%s clause is no longer allowed on %s directive since %s"_warn_en_US,
          cname, dname, omp::ThisVersion(NextVersion(until)));
      SetAllowedClauseOverride(c, dirName.v);
    }
  }

  return result.empty();
}

// In OpenMP 6.0+ the COMBINER clause is required on DECLARE_REDUCTION,
// even though the old syntax (with the combiner expression inside the
// directive argument) is still allowed.
static bool missingCombiner(llvm::omp::Directive d, llvm::omp::Clause c) {
  return d == llvm::omp::Directive::OMPD_declare_reduction &&
      c == llvm::omp::Clause::OMPC_combiner;
}

bool OmpStructureChecker::VerifyClauseRequired(
    parser::OmpDirectiveName dirName, const AppliedClauseInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto result = VerifyRequired(info, dirName.v, version);

  for (llvm::omp::Clause c : result.first) {
    // Exceptions:
    if (version >= 60 && missingCombiner(dirName.v, c)) {
      continue;
    }

    context_.Say(dirName.source,
        "%s clause is required on %s directive"_err_en_US,
        GetUpperName(c, version), GetUpperName(dirName.v, version));
  }

  for (llvm::omp::ClauseSet s : result.second) {
    // If the group is required, at least one clause from that group must
    // be present.
    // Note: The tricky part is that when a directive accepts a clause
    // group, it may still have restrictions that exclude some members of
    // that group. For example FLUSH accepts memory-order group, but not the
    // RELAXED clause (note that the memory-order group is not "required").
    // If such a restriction applied to a required group, we don't want to say
    //   Directive XYZ requires one of FOO, BAR or BAZ clauses
    // and then
    //   BAZ clause is not allowed on XYZ directive
    // This hasn't happened yet, but may happen in the future.
    if (s != llvm::omp::ClauseSet::CancelDirectiveName) {
      context_.Say(dirName.source, "%s is required on %s directive"_err_en_US,
          OneOfClauses(s, version), GetUpperName(dirName.v, version));
    } else {
      // cancel-directive-name is somewhat special: the ClauseSet doesn't
      // contain any actual clauses. Moreover, the clauses are cancellable
      // directive names and have no separate definitions. They are encoded
      // as directive ids inside OmpCancellationConstructTypeClause with the
      // id OMPC_cancellation_construct_type.
      context_.Say(dirName.source,
          "One of '%s' clauses is required on %s directive"_err_en_US,
          llvm::omp::getDescriptor(s).getName().str(),
          GetUpperName(dirName.v, version));
    }
  }

  return result.first.empty() && result.second.empty();
}

bool OmpStructureChecker::VerifyClauseUnique(
    parser::OmpDirectiveName dirName, const AppliedClauseInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto result = VerifyUnique(info, dirName.v, version);

  for (auto [id, where] : result) {
    context_
        .Say(where.second,
            "At most one %s clause can appear on %s directive"_err_en_US,
            GetUpperName(id, version), GetUpperName(dirName.v, version))
        .Attach(where.first, "previous occurrence of this clause"_en_US);
  }
  return result.empty();
}

bool OmpStructureChecker::VerifyClauseExclusive(
    parser::OmpDirectiveName dirName, const AppliedClauseInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto resultExcl = VerifyExclusive(info, dirName.v, version);

  for (auto [id, wrong] : resultExcl) {
    auto [otherId, source, otherSource] = wrong;
    context_
        .Say(source,
            "%s clause cannot be specified together with a clause of a different type"_err_en_US,
            GetUpperName(id, version))
        .Attach(otherSource, "%s provided here"_en_US,
            GetUpperName(otherId, version));
  }

  auto resultMut = VerifyMutuallyExclusive(info, dirName.v, version);

  for (auto [id, wrong] : resultMut) {
    auto [otherId, setId, source, otherSource] = wrong;
    auto thisName{GetUpperName(id, version)};
    std::string annot;
    if (llvm::omp::isClauseGroup(setId)) {
      auto &sdesc{llvm::omp::getDescriptor(setId)};
      annot = " as members of '" + sdesc.getName().str() + "' clause group";
    }
    context_
        .Say(otherSource,
            "%s and %s clauses are mutually exclusive%s"_err_en_US,
            GetUpperName(otherId, version), thisName, annot)
        .Attach(source, "%s clause specified here"_en_US, thisName);
  }

  return resultExcl.empty() && resultMut.empty();
}

bool OmpStructureChecker::VerifyClauseUltimate(
    parser::OmpDirectiveName dirName, const AppliedClauseInfo &info) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  auto result = VerifyUltimate(info, dirName.v, version, /*last=*/true);

  for (auto [id, where] : result) {
    context_.Say(where, "%s should be the last clause"_err_en_US,
        GetUpperName(id, version));
  }

  return result.empty();
}

// Collect the information about clauses specified on the given directive.
// If a clause is allowed on this directive in "version", store the list of
// clause sets that the directive allows in "version" in AppliedClause.
// If a clause is not allowed on this directive in "version", but is allowed
// on it in another version v, store the list of clause sets that the directive
// allows in v.
// In either case, store the applied version in AppliedClause.
// If the clause is not allowed in any version, the applied version will
// be the default (i.e. 0) and no sets will be stored.
AppliedClauseInfo GetAppliedClauses(
    const parser::OmpDirectiveSpecification *beginSpec,
    const parser::OmpDirectiveSpecification *endSpec,
    llvm::omp::Version version) {
  using AppliedClause = AppliedClauseInfo::ElementTy;
  AppliedClauseInfo info;
  llvm::omp::Directive dirId{beginSpec->DirId()};
  auto &ddesc{llvm::omp::getDescriptor(dirId)};

  auto addClauses = [&](const parser::OmpClauseList &clauses) {
    for (auto &clause : clauses.v) {
      auto &am{info.elements.emplace_back(AppliedClause{})};
      am.id = WithSource{clause.Id(), clause.source};
      am.version = GetClosestVersion(
          descriptor::GetVersionRangeForElement(am.id.value, dirId), version);
      if (am.version) {
        for (auto s : ddesc.getClauseSets(am.version)) {
          auto &sdesc{llvm::omp::getDescriptor(s)};
          if (sdesc.getClauses(am.version).test(am.id.value)) {
            am.sets.set(s);
          }
        }
      }
    }
  };

  addClauses(DEREF(beginSpec).Clauses());
  if (endSpec) {
    addClauses(endSpec->Clauses());
  }

  return info;
}

bool OmpStructureChecker::VerifyClauseSyntax(
    parser::OmpDirectiveName dirName, const AppliedClauseInfo &info) {
  bool valid[]{
      VerifyClauseVersion(dirName, info),
      VerifyClauseRequired(dirName, info),
      VerifyClauseUnique(dirName, info),
      VerifyClauseExclusive(dirName, info),
      VerifyClauseUltimate(dirName, info),
  };

  return llvm::all_of(valid, [](bool x) { return x; });
}

void OmpStructureChecker::VerifyClauseSyntax(
    const parser::OmpDirectiveSpecification *beginSpec,
    const parser::OmpDirectiveSpecification *endSpec) {
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};

  if (endSpec) {
    for (const parser::OmpClause &clause : endSpec->Clauses().v) {
      llvm::omp::Clause id{clause.Id()};
      auto &desc{llvm::omp::getDescriptor(id)};
      if (!desc.getProperties(version).test(llvm::omp::Property::EndClause)) {
        context_.Say(clause.source,
            "%s clause is not allowed on an end-directive"_err_en_US,
            GetUpperName(id, version));
      }
    }
  }
  VerifyClauseSyntax(
      beginSpec->DirName(), GetAppliedClauses(beginSpec, endSpec, version));
}

bool OmpStructureChecker::VerifyModifierVersion(
    WithSource<llvm::omp::Clause> clause, const AppliedModifierInfo &info) {
  // Verify that the specified modifiers are allowed in this version.
  llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};
  llvm::omp::Version maxVer{std::numeric_limits<int>::max()};

  auto result = VerifyVersions(info, clause.value, version);

  for (auto &[m, svr] : result) {
    std::string modName{llvm::omp::getDescriptor(m).getName()};
    std::string clauseName{GetUpperName(clause.value, version)};
    llvm::omp::Version since(svr.second.Min);
    llvm::omp::Version until(svr.second.Max);

    if (since == maxVer && until == 0u) {
      // This shouldn't really happen because the set of allowed modifiers
      // is specified in the AST node for the clause, but have this check
      // just to cover all bases.
      context_.Say(svr.first,
          "'%s' modifier is not allowed on %s clause"_err_en_US, modName,
          clauseName);
    } else if (since != maxVer && version < since) {
      context_.Warn(common::UsageWarning::OpenMPFuture, svr.first,
          "'%s' modifier is not allowed on %s clause in %s, %s"_warn_en_US,
          modName, clauseName, omp::ThisVersion(version),
          omp::TryVersion(since));
    } else if (until != 0u && version > until) {
      context_.Warn(common::UsageWarning::OpenMPDeprecated, svr.first,
          "'%s' modifier is no longer allowed on %s clause since %s"_warn_en_US,
          modName, clauseName, omp::ThisVersion(NextVersion(until)));
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
    context_.Say(clause.source, "%s is required on %s clause"_err_en_US,
        OneOfModifiers(s, version), GetUpperName(clause.value, version));
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
        .Say(where.second,
            "'%s' modifier cannot occur multiple times"_err_en_US,
            mdesc.getName())
        .Attach(where.first, "previous occurrence of this modifier"_en_US);
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
            "'%s' modifier cannot be specified together with a modifier of a different type"_err_en_US,
            llvm::omp::getDescriptor(id).getName())
        .Attach(otherSource, "'%s' provided here"_en_US,
            llvm::omp::getDescriptor(otherId).getName());
  }

  auto resultMut = VerifyMutuallyExclusive(info, clause.value, version);

  for (auto [id, wrong] : resultMut) {
    auto [otherId, setId, source, otherSource] = wrong;
    auto thisName{llvm::omp::getDescriptor(id).getName()};
    std::string annot;
    if (llvm::omp::isModifierGroup(setId)) {
      auto &sdesc{llvm::omp::getDescriptor(setId)};
      annot = " as members of '" + sdesc.getName().str() + "' modifier group";
    }
    context_
        .Say(otherSource,
            "'%s' and '%s' modifiers are mutually exclusive%s"_err_en_US,
            llvm::omp::getDescriptor(otherId).getName(), thisName, annot)
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
  using AppliedModifier = AppliedModifierInfo::ElementTy;
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

// Mark clauseId as allowed on dirId.
// * If dirId is a compound directive and "since" is a valid version,
//   identify all leafs that allow the clause in version "since" or later,
//   and mark the clause as allowed on these leafs as well.
//   This is intended for allowing a "future case" in the current version.
// * If dirId is a compound directive and "since" is not a valid version
//   (i.e. !since is true) then mark the clause as allowed on all leafs
//   that allow it in _some_ version. This is intended for allowing
//   "deprecated cases".
// * If dirId is not a compound directive the "since" parameter is ignored.
void OmpStructureChecker::SetAllowedClauseOverride(llvm::omp::Clause clauseId,
    llvm::omp::Directive dirId, llvm::omp::Version since) {
  omp::SemanticOverrides &overrides{context_.GetOmpSemanticOverrides()};
  overrides.allowedClauses[clauseId].set(dirId);

  auto leafs{llvm::omp::getLeafConstructsOrSelf(dirId)};
  if (leafs.size() > 1) {
    llvm::omp::Version version{context_.langOptions().getOpenMPVersion()};
    assert(
        (!since || since > version) && "\"since\" should be a future version");
    for (llvm::omp::Directive leaf : leafs) {
      auto range{descriptor::GetVersionRangeForElement(clauseId, leaf)};
      if (range.isValid() && (!since || since >= range.Min)) {
        overrides.allowedClauses[clauseId].set(leaf);
      }
    }
  }
}
} // namespace Fortran::semantics
