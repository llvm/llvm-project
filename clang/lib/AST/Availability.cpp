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

namespace {

/// Represents the availability of a symbol across platforms.
struct AvailabilitySet {
  bool UnconditionallyDeprecated = false;
  bool UnconditionallyUnavailable = false;

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
  llvm::SmallVector<clang::AvailabilityInfo> Availabilities;
};

static void createInfoForDecl(const clang::Decl *Decl,
                              AvailabilitySet &Availabilities) {
  // Collect availability attributes from all redeclarations.
  for (const auto *RD : Decl->redecls()) {
    for (const auto *A : RD->specific_attrs<clang::AvailabilityAttr>()) {
      Availabilities.insert(clang::AvailabilityInfo(
          A->getPlatform()->getName(), A->getIntroduced(), A->getDeprecated(),
          A->getObsoleted(), A->getUnavailable(), false, false));
    }

    if (const auto *A = RD->getAttr<clang::UnavailableAttr>())
      if (!A->isImplicit())
        Availabilities.UnconditionallyUnavailable = true;

    if (const auto *A = RD->getAttr<clang::DeprecatedAttr>())
      if (!A->isImplicit())
        Availabilities.UnconditionallyDeprecated = true;
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
    Avail->UnconditionallyDeprecated = Availabilities.UnconditionallyDeprecated;
    Avail->UnconditionallyUnavailable =
        Availabilities.UnconditionallyUnavailable;
    return std::move(*Avail);
  }

  AvailabilityInfo Avail;
  Avail.UnconditionallyDeprecated = Availabilities.UnconditionallyDeprecated;
  Avail.UnconditionallyUnavailable = Availabilities.UnconditionallyUnavailable;
  return Avail;
}

} // namespace clang
