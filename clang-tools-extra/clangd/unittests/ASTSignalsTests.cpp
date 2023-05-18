//===-- ASTSignalsTests.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTSignals.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::_;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(ASTSignals, Derive) {
  TestTU TU = TestTU::withCode(R"cpp(
  namespace ns1 {
  namespace ns2 {
  namespace {
  int func() {
    tar::X a;
    a.Y = 1;
    return ADD(tar::kConst, a.Y, tar::foo()) + fooInNS2() + tar::foo();
  }
  } // namespace
  } // namespace ns2
  } // namespace ns1
  )cpp");

  TU.HeaderCode = R"cpp(
  #define ADD(x, y, z) (x + y + z)
  namespace tar {  // A related namespace.
  int kConst = 5;
  int foo();
  void bar();  // Unused symbols are not recorded.
  class X {
    public: int Y;
  };
  } // namespace tar
  namespace ns1::ns2 { int fooInNS2(); }}
  )cpp";
  ASTSignals Signals = ASTSignals::derive(TU.build());
  std::vector<std::pair<StringRef, int>> NS;
  for (const auto &P : Signals.RelatedNamespaces)
    NS.emplace_back(P.getKey(), P.getValue());
  EXPECT_THAT(NS, UnorderedElementsAre(Pair("ns1::", 1), Pair("ns1::ns2::", 1),
                                       Pair("tar::", /*foo, kConst, X*/ 3)));

  std::vector<std::pair<SymbolID, int>> Sym;
  for (const auto &P : Signals.ReferencedSymbols)
    Sym.emplace_back(P.getFirst(), P.getSecond());
  EXPECT_THAT(
      Sym,
      UnorderedElementsAre(
          Pair(ns("tar").ID, 4), Pair(ns("ns1").ID, 1),
          Pair(ns("ns1::ns2").ID, 1), Pair(_ /*int func();*/, 1),
          Pair(cls("tar::X").ID, 1), Pair(var("tar::kConst").ID, 1),
          Pair(func("tar::foo").ID, 2), Pair(func("ns1::ns2::fooInNS2").ID, 1),
          Pair(sym("Y", index::SymbolKind::Variable, "@N@tar@S@X@FI@\\0").ID,
               2),
          Pair(_ /*a*/, 3)));
  EXPECT_EQ(Signals.InsertionDirective, Symbol::IncludeDirective::Include);
}
} // namespace
} // namespace clangd
} // namespace clang
