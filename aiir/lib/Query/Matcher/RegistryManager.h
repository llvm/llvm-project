//===--- RegistryManager.h - Matcher registry -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RegistryManager to manage registry of all known matchers.
//
// The registry provides a generic interface to construct any matcher by name.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRQUERY_MATCHER_REGISTRYMANAGER_H
#define AIIR_TOOLS_AIIRQUERY_MATCHER_REGISTRYMANAGER_H

#include "Diagnostics.h"
#include "aiir/Query/Matcher/Marshallers.h"
#include "aiir/Query/Matcher/Registry.h"
#include "aiir/Query/Matcher/VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace aiir::query::matcher {

using MatcherCtor = const internal::MatcherDescriptor *;

struct MatcherCompletion {
  MatcherCompletion() = default;
  MatcherCompletion(llvm::StringRef typedText, llvm::StringRef matcherDecl)
      : typedText(typedText.str()), matcherDecl(matcherDecl.str()) {}

  bool operator==(const MatcherCompletion &other) const {
    return typedText == other.typedText && matcherDecl == other.matcherDecl;
  }

  // The text to type to select this matcher.
  std::string typedText;

  // The "declaration" of the matcher, with type information.
  std::string matcherDecl;
};

class RegistryManager {
public:
  RegistryManager() = delete;

  static std::optional<MatcherCtor>
  lookupMatcherCtor(llvm::StringRef matcherName,
                    const Registry &matcherRegistry);

  static std::vector<ArgKind> getAcceptedCompletionTypes(
      llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context);

  static std::vector<MatcherCompletion>
  getMatcherCompletions(ArrayRef<ArgKind> acceptedTypes,
                        const Registry &matcherRegistry);

  static VariantMatcher constructMatcher(MatcherCtor ctor,
                                         internal::SourceRange nameRange,
                                         llvm::StringRef functionName,
                                         ArrayRef<ParserValue> args,
                                         internal::Diagnostics *error);
};

} // namespace aiir::query::matcher

#endif // AIIR_TOOLS_AIIRQUERY_MATCHER_REGISTRYMANAGER_H
