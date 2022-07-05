//===--- CXX.cpp - Define public interfaces for C++ grammar ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/cxx/CXX.h"
#include "clang-pseudo/Language.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"
#include <utility>

namespace clang {
namespace pseudo {
namespace cxx {
namespace {
static const char *CXXBNF =
#include "CXXBNF.inc"
    ;
} // namespace

const Language &getLanguage() {
  static const auto &CXXLanguage = []() -> const Language & {
    std::vector<std::string> Diags;
    auto G = Grammar::parseBNF(CXXBNF, Diags);
    assert(Diags.empty());
    LRTable Table = LRTable::buildSLR(G);
    const Language *PL = new Language{
        std::move(G),
        std::move(Table),
    };
    return *PL;
  }();
  return CXXLanguage;
}

} // namespace cxx
} // namespace pseudo
} // namespace clang
