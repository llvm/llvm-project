//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyAliases.h"
#include "../ClangTidyModule.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h" // IWYU pragma: keep
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include <cassert>

namespace clang::tidy {

/// Alias table. Kept sorted by alias name for readability.
static constexpr ClangTidyAliases::Entry AliasTable[] = {
    // Permanent aliases.
    {"cert-dcl03-c", "misc-static-assert", ClangTidyAliases::NoExpiration},
    {"cert-oop11-cpp", "performance-move-constructor-init",
     ClangTidyAliases::NoExpiration},
};

ArrayRef<ClangTidyAliases::Entry> ClangTidyAliases::getReference() {
  return AliasTable;
}

bool ClangTidyAliases::isActive(const Entry &E) {
  return E.ExpirationVersion <= NoExpiration ||
         LLVM_VERSION_MAJOR < E.ExpirationVersion;
}

ClangTidyAliases::ActiveRange ClangTidyAliases::activeEntries() {
  return llvm::make_filter_range(getReference(), isActive);
}

StringRef ClangTidyAliases::getCanonicalForAlias(StringRef Alias) {
  static const auto *Map = [] {
    auto *M = new llvm::StringMap<StringRef>();
    for (const auto &Entry : activeEntries()) {
      [[maybe_unused]] auto Result =
          M->try_emplace(Entry.Alias, Entry.Canonical);
      assert(Result.second && "Duplicate alias in ClangTidyAliases table");
    }
    return M;
  }();
  auto It = Map->find(Alias);
  if (It != Map->end())
    return It->second;
  return {};
}

const SmallVector<StringRef> &
ClangTidyAliases::getAliasesForCanonical(StringRef Canonical) {
  static const SmallVector<StringRef> Empty;
  static const auto *ReverseMap = [] {
    auto *M = new llvm::StringMap<SmallVector<StringRef>>();
    for (const auto &Entry : activeEntries())
      (*M)[Entry.Canonical].push_back(Entry.Alias);
    return M;
  }();
  auto It = ReverseMap->find(Canonical);
  if (It != ReverseMap->end())
    return It->second;
  return Empty;
}

namespace aliases {
namespace {

class AliasesModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    for (const auto &Entry : ClangTidyAliases::activeEntries())
      CheckFactories.registerCheckAlias(Entry.Alias, Entry.Canonical);
  }
};

} // namespace

static ClangTidyModuleRegistry::Add<AliasesModule>
    X("aliases-module", "Adds check aliases for backward compatibility.");

} // namespace aliases

// This anchor is used to force the linker to link in the generated object file
// and thus register the AliasesModule.
volatile int AliasesModuleAnchorSource = 0; // NOLINT(misc-use-internal-linkage)

} // namespace clang::tidy
