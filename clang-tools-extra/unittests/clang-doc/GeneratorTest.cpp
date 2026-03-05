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
  IndexA.USR = serialize::hashUSR("1");
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexA.USR),
                                   std::move(IndexA));
  Index IndexB;
  IndexB.Name = "B";
  IndexB.USR = serialize::hashUSR("2");
  Index IndexC;
  IndexC.Name = "C";
  IndexC.USR = serialize::hashUSR("3");
  IndexB.Children.try_emplace(llvm::toStringRef(IndexC.USR), std::move(IndexC));
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexB.USR),
                                   std::move(IndexB));
  Index IndexD;
  IndexD.Name = "D";
  IndexD.USR = serialize::hashUSR("4");
  Index IndexE;
  IndexE.Name = "E";
  IndexE.USR = serialize::hashUSR("5");
  Index IndexF;
  IndexF.Name = "F";
  IndexF.USR = serialize::hashUSR("6");
  IndexE.Children.try_emplace(llvm::toStringRef(IndexF.USR), std::move(IndexF));
  IndexD.Children.try_emplace(llvm::toStringRef(IndexE.USR), std::move(IndexE));
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexD.USR),
                                   std::move(IndexD));
  Index IndexG;
  IndexG.Name = "GlobalNamespace";
  IndexG.RefType = InfoType::IT_namespace;
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexG.USR),
                                   std::move(IndexG));

  CheckIndex(ExpectedIdx, Idx);
}

TEST(GeneratorTest, sortIndex) {
  Index Idx;
  Index IndexA;
  IndexA.Name = "a";
  IndexA.USR = serialize::hashUSR("1");
  Idx.Children.try_emplace(llvm::toStringRef(IndexA.USR), std::move(IndexA));

  Index IndexB;
  IndexB.Name = "A";
  IndexB.USR = serialize::hashUSR("2");
  Idx.Children.try_emplace(llvm::toStringRef(IndexB.USR), std::move(IndexB));

  Index IndexC;
  IndexC.Name = "aa";
  IndexC.USR = serialize::hashUSR("3");
  Idx.Children.try_emplace(llvm::toStringRef(IndexC.USR), std::move(IndexC));

  Index IndexD;
  IndexD.Name = "aA";
  IndexD.USR = serialize::hashUSR("4");
  Idx.Children.try_emplace(llvm::toStringRef(IndexD.USR), std::move(IndexD));

  Index IndexE;
  IndexE.Name = "b";
  IndexE.USR = serialize::hashUSR("5");
  Idx.Children.try_emplace(llvm::toStringRef(IndexE.USR), std::move(IndexE));

  Idx.sort();

  Index ExpectedIdx;
  Index IndexAExp;
  IndexAExp.Name = "a";
  IndexAExp.USR = serialize::hashUSR("1");
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexAExp.USR),
                                   std::move(IndexAExp));

  Index IndexBExp;
  IndexBExp.Name = "A";
  IndexBExp.USR = serialize::hashUSR("2");
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexBExp.USR),
                                   std::move(IndexBExp));

  Index IndexCExp;
  IndexCExp.Name = "aa";
  IndexCExp.USR = serialize::hashUSR("3");
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexCExp.USR),
                                   std::move(IndexCExp));

  Index IndexDExp;
  IndexDExp.Name = "aA";
  IndexDExp.USR = serialize::hashUSR("4");
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexDExp.USR),
                                   std::move(IndexDExp));

  Index IndexEExp;
  IndexEExp.Name = "b";
  IndexEExp.USR = serialize::hashUSR("5");
  ExpectedIdx.Children.try_emplace(llvm::toStringRef(IndexEExp.USR),
                                   std::move(IndexEExp));

  CheckIndex(ExpectedIdx, Idx);
}

} // namespace doc
} // namespace clang
