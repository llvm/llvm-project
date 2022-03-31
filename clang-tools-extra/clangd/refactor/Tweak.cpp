//===--- Tweak.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Tweak.h"
#include "FeatureModule.h"
#include "SourceCode.h"
#include "index/Index.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"
#include <functional>
#include <memory>
#include <utility>
#include <vector>

LLVM_INSTANTIATE_REGISTRY(llvm::Registry<clang::clangd::Tweak>)

namespace clang {
namespace clangd {

/// A handy typedef to save some typing.
typedef llvm::Registry<Tweak> TweakRegistry;

namespace {
/// Asserts invariants on TweakRegistry. No-op with assertion disabled.
void validateRegistry() {
#ifndef NDEBUG
  llvm::StringSet<> Seen;
  for (const auto &E : TweakRegistry::entries()) {
    // REGISTER_TWEAK ensures E.getName() is equal to the tweak class name.
    // We check that id() matches it.
    assert(E.instantiate()->id() == E.getName() &&
           "id should be equal to class name");
    assert(Seen.try_emplace(E.getName()).second && "duplicate check id");
  }
#endif
}

std::vector<std::unique_ptr<Tweak>>
getAllTweaks(const FeatureModuleSet *Modules) {
  std::vector<std::unique_ptr<Tweak>> All;
  for (const auto &E : TweakRegistry::entries())
    All.emplace_back(E.instantiate());
  if (Modules) {
    for (auto &M : *Modules)
      M.contributeTweaks(All);
  }
  return All;
}
} // namespace

Tweak::Selection::Selection(const SymbolIndex *Index, ParsedAST &AST,
                            unsigned RangeBegin, unsigned RangeEnd,
                            SelectionTree ASTSelection,
                            llvm::vfs::FileSystem *FS)
    : Index(Index), AST(&AST), SelectionBegin(RangeBegin),
      SelectionEnd(RangeEnd), ASTSelection(std::move(ASTSelection)), FS(FS) {
  auto &SM = AST.getSourceManager();
  Code = SM.getBufferData(SM.getMainFileID());
  Cursor = SM.getComposedLoc(SM.getMainFileID(), RangeBegin);
}

std::vector<std::unique_ptr<Tweak>>
prepareTweaks(const Tweak::Selection &S,
              llvm::function_ref<bool(const Tweak &)> Filter,
              const FeatureModuleSet *Modules) {
  validateRegistry();

  std::vector<std::unique_ptr<Tweak>> Available;
  for (auto &T : getAllTweaks(Modules)) {
    if (!Filter(*T) || !T->prepare(S))
      continue;
    Available.push_back(std::move(T));
  }
  // Ensure deterministic order of the results.
  llvm::sort(Available,
             [](const std::unique_ptr<Tweak> &L,
                const std::unique_ptr<Tweak> &R) { return L->id() < R->id(); });
  return Available;
}

llvm::Expected<std::unique_ptr<Tweak>>
prepareTweak(StringRef ID, const Tweak::Selection &S,
             const FeatureModuleSet *Modules) {
  for (auto &T : getAllTweaks(Modules)) {
    if (T->id() != ID)
      continue;
    if (!T->prepare(S))
      return error("failed to prepare() tweak {0}", ID);
    return std::move(T);
  }
  return error("tweak ID {0} is invalid", ID);
}

llvm::Expected<std::pair<Path, Edit>>
Tweak::Effect::fileEdit(const SourceManager &SM, FileID FID,
                        tooling::Replacements Replacements) {
  Edit Ed(SM.getBufferData(FID), std::move(Replacements));
  if (auto FilePath = getCanonicalPath(SM.getFileEntryForID(FID), SM))
    return std::make_pair(*FilePath, std::move(Ed));
  return error("Failed to get absolute path for edited file: {0}",
               SM.getFileEntryForID(FID)->getName());
}

llvm::Expected<Tweak::Effect>
Tweak::Effect::mainFileEdit(const SourceManager &SM,
                            tooling::Replacements Replacements) {
  auto PathAndEdit = fileEdit(SM, SM.getMainFileID(), std::move(Replacements));
  if (!PathAndEdit)
    return PathAndEdit.takeError();
  Tweak::Effect E;
  E.ApplyEdits.try_emplace(PathAndEdit->first, PathAndEdit->second);
  return E;
}

} // namespace clangd
} // namespace clang
