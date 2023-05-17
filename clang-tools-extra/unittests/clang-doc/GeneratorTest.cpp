//===-- clang-doc/GeneratorTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangDocTest.h"
#include "Generators.h"
#include "Representation.h"
#include "Serialize.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

TEST(GeneratorTest, emitIndex) {
  Index Idx;
  auto InfoA = std::make_unique<Info>();
  InfoA->Name = "A";
  InfoA->USR = serialize::hashUSR("1");
  Generator::addInfoToIndex(Idx, InfoA.get());
  auto InfoC = std::make_unique<Info>();
  InfoC->Name = "C";
  InfoC->USR = serialize::hashUSR("3");
  Reference RefB = Reference(SymbolID(), "B");
  RefB.USR = serialize::hashUSR("2");
  InfoC->Namespace = {std::move(RefB)};
  Generator::addInfoToIndex(Idx, InfoC.get());
  auto InfoD = std::make_unique<Info>();
  InfoD->Name = "D";
  InfoD->USR = serialize::hashUSR("4");
  auto InfoF = std::make_unique<Info>();
  InfoF->Name = "F";
  InfoF->USR = serialize::hashUSR("6");
  Reference RefD = Reference(SymbolID(), "D");
  RefD.USR = serialize::hashUSR("4");
  Reference RefE = Reference(SymbolID(), "E");
  RefE.USR = serialize::hashUSR("5");
  InfoF->Namespace = {std::move(RefE), std::move(RefD)};
  Generator::addInfoToIndex(Idx, InfoF.get());
  auto InfoG = std::make_unique<Info>(InfoType::IT_namespace);
  Generator::addInfoToIndex(Idx, InfoG.get());

  Index ExpectedIdx;
  Index IndexA;
  IndexA.Name = "A";
  ExpectedIdx.Children.emplace_back(std::move(IndexA));
  Index IndexB;
  IndexB.Name = "B";
  Index IndexC;
  IndexC.Name = "C";
  IndexB.Children.emplace_back(std::move(IndexC));
  ExpectedIdx.Children.emplace_back(std::move(IndexB));
  Index IndexD;
  IndexD.Name = "D";
  Index IndexE;
  IndexE.Name = "E";
  Index IndexF;
  IndexF.Name = "F";
  IndexE.Children.emplace_back(std::move(IndexF));
  IndexD.Children.emplace_back(std::move(IndexE));
  ExpectedIdx.Children.emplace_back(std::move(IndexD));
  Index IndexG;
  IndexG.Name = "GlobalNamespace";
  IndexG.RefType = InfoType::IT_namespace;
  ExpectedIdx.Children.emplace_back(std::move(IndexG));

  CheckIndex(ExpectedIdx, Idx);
}

TEST(GeneratorTest, sortIndex) {
  Index Idx;
  Idx.Children.emplace_back("b");
  Idx.Children.emplace_back("aA");
  Idx.Children.emplace_back("aa");
  Idx.Children.emplace_back("A");
  Idx.Children.emplace_back("a");
  Idx.sort();

  Index ExpectedIdx;
  ExpectedIdx.Children.emplace_back("a");
  ExpectedIdx.Children.emplace_back("A");
  ExpectedIdx.Children.emplace_back("aa");
  ExpectedIdx.Children.emplace_back("aA");
  ExpectedIdx.Children.emplace_back("b");

  CheckIndex(ExpectedIdx, Idx);
}

} // namespace doc
} // namespace clang
