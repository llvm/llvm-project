//===--- Tweak.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Tweak.h"
#include "Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"
#include <functional>
#include <memory>

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
} // namespace

std::vector<std::unique_ptr<Tweak>> prepareTweaks(const Tweak::Selection &S) {
  validateRegistry();

  std::vector<std::unique_ptr<Tweak>> Available;
  for (const auto &E : TweakRegistry::entries()) {
    std::unique_ptr<Tweak> T = E.instantiate();
    if (!T->prepare(S))
      continue;
    Available.push_back(std::move(T));
  }
  // Ensure deterministic order of the results.
  llvm::sort(Available,
             [](const std::unique_ptr<Tweak> &L,
                const std::unique_ptr<Tweak> &R) { return L->id() < R->id(); });
  return Available;
}

llvm::Expected<std::unique_ptr<Tweak>> prepareTweak(StringRef ID,
                                                    const Tweak::Selection &S) {
  auto It = llvm::find_if(
      TweakRegistry::entries(),
      [ID](const TweakRegistry::entry &E) { return E.getName() == ID; });
  if (It == TweakRegistry::end())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "id of the tweak is invalid");
  std::unique_ptr<Tweak> T = It->instantiate();
  if (!T->prepare(S))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to prepare() a check");
  return std::move(T);
}

} // namespace clangd
} // namespace clang
