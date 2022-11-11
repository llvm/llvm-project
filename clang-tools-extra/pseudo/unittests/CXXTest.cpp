//===--- CXXTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/cxx/CXX.h"
#include "gtest/gtest.h"

namespace clang {
namespace pseudo {
namespace cxx {
namespace {

TEST(CXX, GeneratedEnums) {
  const auto &Lang = clang::pseudo::cxx::getLanguage();
  EXPECT_EQ("iteration-statement",
            Lang.G.symbolName(Symbol::iteration_statement));
  EXPECT_EQ("iteration-statement := DO statement WHILE ( expression ) ;",
            Lang.G.dumpRule(
                rule::iteration_statement::
                    DO__statement__WHILE__L_PAREN__expression__R_PAREN__SEMI));
}

} // namespace
} // namespace cxx
} // namespace pseudo
} // namespace clang
