//===- Availability.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Availability information for Decls.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Availability.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallVectorWithFlags.h"

namespace {

/// Represents the availability of a symbol across platforms.
struct AvailabilitySet {
  bool getUnconditionallyDeprecated() const {
    return Availabilities.getFlag(0);
  }
  void setUnconditionallyDeprecated(bool Value) {
    Availabilities.setFlag(0, Value);
  }
  bool getUnconditionallyUnavailable() const {
    return Availabilities.getFlag(1);
  }
  void setUnconditionallyUnavailable(bool Value) {
    Availabilities.setFlag(1, Value);
  }

  void insert(clang::AvailabilityInfo &&Availability) {
    auto *Found = getForPlatform(Availability.Domain);
    if (Found)
      Found->mergeWith(std::move(Availability));
    else
      Availabilities.emplace_back(std::move(Availability));
  }

  clang::AvailabilityInfo *getForPlatform(llvm::StringRef Domain) {
    auto *It = llvm::find_if(Availabilities,
                             [Domain](const clang::AvailabilityInfo &Info) {
                               return Domain.compare(Info.Domain) == 0;
                             });
    return It == Availabilities.end() ? nullptr : It;
  }

private:
  llvm::SmallVectorWithFlags<clang::AvailabilityInfo,
                             llvm::CalculateSmallVectorDefaultInlinedElements<
                                 clang::AvailabilityInfo>::value,
                             2>
      Availabilities;
};

static void createInfoForDecl(const clang::Decl *Decl,
                              AvailabilitySet &Availabilities) {
  // Collect availability attributes from all redeclarations.
  for (const auto *RD : Decl->redecls()) {
    for (const auto *A : RD->specific_attrs<clang::AvailabilityAttr>()) {
      const auto *Eff = A->getEffectiveAttr();
      Availabilities.insert(clang::AvailabilityInfo(
          Eff->getPlatform()->getName(), Eff->getIntroduced(),
          Eff->getDeprecated(), Eff->getObsoleted(), Eff->getUnavailable(),
          false, false));
    }

    if (const auto *A = RD->getAttr<clang::UnavailableAttr>())
      if (!A->isImplicit())
        Availabilities.setUnconditionallyUnavailable(true);

    if (const auto *A = RD->getAttr<clang::DeprecatedAttr>())
      if (!A->isImplicit())
        Availabilities.setUnconditionallyDeprecated(true);
  }
}

} // namespace

namespace clang {

void AvailabilityInfo::mergeWith(AvailabilityInfo Other) {
  if (isDefault() && Other.isDefault())
    return;

  if (Domain.empty())
    Domain = Other.Domain;

  UnconditionallyUnavailable |= Other.UnconditionallyUnavailable;
  UnconditionallyDeprecated |= Other.UnconditionallyDeprecated;
  Unavailable |= Other.Unavailable;

  Introduced = std::max(Introduced, Other.Introduced);

  // Default VersionTuple is 0.0.0 so if both are non default let's pick the
  // smallest version number, otherwise select the one that is non-zero if there
  // is one.
  if (!Deprecated.empty() && !Other.Deprecated.empty())
    Deprecated = std::min(Deprecated, Other.Deprecated);
  else
    Deprecated = std::max(Deprecated, Other.Deprecated);

  if (!Obsoleted.empty() && !Other.Obsoleted.empty())
    Obsoleted = std::min(Obsoleted, Other.Obsoleted);
  else
    Obsoleted = std::max(Obsoleted, Other.Obsoleted);
}

AvailabilityInfo AvailabilityInfo::createFromDecl(const Decl *D) {
  AvailabilitySet Availabilities;
  // Walk DeclContexts upwards starting from D to find the combined availability
  // of the symbol.
  for (const auto *Ctx = D; Ctx;
       Ctx = llvm::cast_or_null<Decl>(Ctx->getDeclContext()))
    createInfoForDecl(Ctx, Availabilities);

  if (auto *Avail = Availabilities.getForPlatform(
          D->getASTContext().getTargetInfo().getPlatformName())) {
    Avail->UnconditionallyDeprecated =
        Availabilities.getUnconditionallyDeprecated();
    Avail->UnconditionallyUnavailable =
        Availabilities.getUnconditionallyUnavailable();
    return std::move(*Avail);
  }

  AvailabilityInfo Avail;
  Avail.UnconditionallyDeprecated =
      Availabilities.getUnconditionallyDeprecated();
  Avail.UnconditionallyUnavailable =
      Availabilities.getUnconditionallyUnavailable();
  return Avail;
}

} // namespace clang
