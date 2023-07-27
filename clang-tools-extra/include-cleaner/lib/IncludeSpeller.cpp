//===--- IncludeSpeller.cpp------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/IncludeSpeller.h"
#include "clang-include-cleaner/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Registry.h"
#include <memory>
#include <string>

LLVM_INSTANTIATE_REGISTRY(clang::include_cleaner::IncludeSpellingStrategy)

namespace clang::include_cleaner {
namespace {

// Fallback strategy to default spelling via header search.
class DefaultIncludeSpeller : public IncludeSpeller {
public:
  std::string operator()(const Input &Input) const override {
    switch (Input.H.kind()) {
    case Header::Standard:
      return Input.H.standard().name().str();
    case Header::Verbatim:
      return Input.H.verbatim().str();
    case Header::Physical:
      bool IsSystem = false;
      std::string FinalSpelling = Input.HS.suggestPathToFileForDiagnostics(
          Input.H.physical(), Input.Main->tryGetRealPathName(), &IsSystem);
      return IsSystem ? "<" + FinalSpelling + ">" : "\"" + FinalSpelling + "\"";
    }
    llvm_unreachable("Unknown clang::include_cleaner::Header::Kind enum");
  }
};

} // namespace

std::string spellHeader(const IncludeSpeller::Input &Input) {
  static auto *Spellers = [] {
    auto *Result =
        new llvm::SmallVector<std::unique_ptr<include_cleaner::IncludeSpeller>>;
    for (const auto &Strategy :
         include_cleaner::IncludeSpellingStrategy::entries())
      Result->push_back(Strategy.instantiate());
    Result->push_back(std::make_unique<DefaultIncludeSpeller>());
    return Result;
  }();

  std::string Spelling;
  for (const auto &Speller : *Spellers) {
    Spelling = (*Speller)(Input);
    if (!Spelling.empty())
      break;
  }
  return Spelling;
}

} // namespace clang::include_cleaner