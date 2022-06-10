//===--- CXX.cpp - Define public interfaces for C++ grammar ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/cxx/CXX.h"
#include "clang-pseudo/grammar/LRTable.h"

namespace clang {
namespace pseudo {
namespace cxx {

static const char *CXXBNF =
#include "CXXBNF.inc"
    ;

const Grammar &getGrammar() {
  static std::vector<std::string> Diags;
  static Grammar *G = Grammar::parseBNF(CXXBNF, Diags).release();
  assert(Diags.empty());
  return *G;
}

const LRTable &getLRTable() {
  static LRTable *Table = new LRTable(LRTable::buildSLR(getGrammar()));
  return *Table;
}

} // namespace cxx
} // namespace pseudo
} // namespace clang
