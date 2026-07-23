//===-- flang/unittests/Evaluate/designator-path.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/designator-path.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Support/Fortran-features.h"
#include "flang/Support/LangOptions.h"
#include "flang/Support/default-kinds.h"
#include "flang/Testing/testing.h"

using namespace Fortran::evaluate;

namespace {
namespace common = Fortran::common;
namespace parser = Fortran::parser;
namespace semantics = Fortran::semantics;
using IntExpr = Expr<SubscriptInteger>;

IntExpr Int(int n) { return IntExpr{n}; }

Subscript Scalar(int n) { return Subscript{Int(n)}; }

Triplet TripletSubscript(
    std::optional<int> lower, std::optional<int> upper, int stride = 1) {
  return Triplet{lower ? std::optional<IntExpr>{Int(*lower)} : std::nullopt,
      upper ? std::optional<IntExpr>{Int(*upper)} : std::nullopt, Int(stride)};
}

Subscript Section(
    std::optional<int> lower, std::optional<int> upper, int stride = 1) {
  return Subscript{TripletSubscript(lower, upper, stride)};
}

Subscript FullSection() {
  return Subscript{Triplet{std::nullopt, std::nullopt, Int(1)}};
}

DesignatorPath PathWithSubscripts(std::vector<Subscript> subscripts) {
  DesignatorPath path;
  path.AddSubscripts(std::move(subscripts));
  return path;
}

DesignatorPath PathWithComponent(const semantics::Symbol *symbol) {
  DesignatorPath path;
  path.AddComponent(*symbol);
  return path;
}

void CheckRelation(const DesignatorPath &x, const DesignatorPath &y,
    DesignatorRelation relation) {
  TEST(x.Compare(y) == relation);
}

class SymbolFixture {
public:
  const semantics::Symbol &MakeSymbol(const char *name) {
    return scope_.MakeSymbol(parser::CharBlock{name}, semantics::Attrs{},
        semantics::UnknownDetails{});
  }

private:
  parser::AllSources allSources_;
  parser::AllCookedSources allCookedSources_{allSources_};
  common::IntrinsicTypeDefaultKinds defaultKinds_;
  common::LanguageFeatureControl languageFeatures_;
  common::LangOptions langOptions_;
  semantics::SemanticsContext context_{
      defaultKinds_, languageFeatures_, langOptions_, allCookedSources_};
  semantics::Scope &scope_{
      context_.globalScope().MakeScope(semantics::Scope::Kind::MainProgram)};
};

void TestGetConstantSubscriptRange() {
  auto scalarRange{DesignatorPath::GetConstantSubscriptRange(Scalar(4))};
  TEST(scalarRange.has_value());
  TEST(scalarRange->lower == 4);
  TEST(scalarRange->upper == 4);

  auto sectionRange{DesignatorPath::GetConstantSubscriptRange(Section(2, 7))};
  TEST(sectionRange.has_value());
  TEST(sectionRange->lower == 2);
  TEST(sectionRange->upper == 7);

  TEST(!DesignatorPath::GetConstantSubscriptRange(Section(2, 7, 2)));
  TEST(!DesignatorPath::GetConstantSubscriptRange(FullSection()));
}

void TestFullTripletDetection() {
  TEST(DesignatorPath::IsFullTriplet(TripletSubscript({}, {})));
  TEST(!DesignatorPath::IsFullTriplet(TripletSubscript(1, {})));
  TEST(!DesignatorPath::IsFullTriplet(TripletSubscript({}, 10)));
  TEST(!DesignatorPath::IsFullTriplet(TripletSubscript({}, {}, 2)));
}

void TestCompareSubscripts() {
  TEST(DesignatorPath::CompareSubscripts(Scalar(3), Scalar(3)) ==
      DesignatorRelation::Equal);
  TEST(DesignatorPath::CompareSubscripts(FullSection(), Scalar(3)) ==
      DesignatorRelation::Contains);
  TEST(DesignatorPath::CompareSubscripts(FullSection(), FullSection()) ==
      DesignatorRelation::Equal);
  TEST(DesignatorPath::CompareSubscripts(Scalar(3), FullSection()) ==
      DesignatorRelation::ContainedBy);
  TEST(DesignatorPath::CompareSubscripts(Section(1, 5), Section(6, 10)) ==
      DesignatorRelation::Disjoint);
  TEST(DesignatorPath::CompareSubscripts(Section(1, 5), Section(1, 5)) ==
      DesignatorRelation::Equal);
  TEST(DesignatorPath::CompareSubscripts(Section(1, 10), Section(3, 5)) ==
      DesignatorRelation::Contains);
  TEST(DesignatorPath::CompareSubscripts(Section(3, 5), Section(1, 10)) ==
      DesignatorRelation::ContainedBy);
  TEST(DesignatorPath::CompareSubscripts(Section(1, 5), Section(5, 10)) ==
      DesignatorRelation::Overlaps);
  TEST(DesignatorPath::CompareSubscripts(Section(1, 5, 2), Section(1, 5)) ==
      DesignatorRelation::Disjoint);
}

void TestCompareSubscriptLists() {
  TEST(DesignatorPath::CompareSubscriptLists({}, {FullSection()}) ==
      DesignatorRelation::Equal);
  TEST(DesignatorPath::CompareSubscriptLists({FullSection()}, {}) ==
      DesignatorRelation::Equal);
  TEST(DesignatorPath::CompareSubscriptLists({FullSection()},
           {FullSection(), FullSection()}) == DesignatorRelation::Disjoint);
  TEST(DesignatorPath::CompareSubscriptLists({Scalar(1)},
           {Scalar(1), Scalar(2)}) == DesignatorRelation::Disjoint);
  TEST(DesignatorPath::CompareSubscriptLists({Scalar(1), Scalar(2)},
           {Scalar(1), Scalar(2)}) == DesignatorRelation::Equal);
  TEST(DesignatorPath::CompareSubscriptLists({Section(1, 10), Scalar(2)},
           {Section(3, 5), Scalar(2)}) == DesignatorRelation::Contains);
  TEST(DesignatorPath::CompareSubscriptLists({Section(3, 5), Scalar(2)},
           {Section(1, 10), Scalar(2)}) == DesignatorRelation::ContainedBy);
  TEST(DesignatorPath::CompareSubscriptLists({Section(1, 10), Scalar(2)},
           {Section(3, 5), FullSection()}) == DesignatorRelation::Overlaps);
  TEST(DesignatorPath::CompareSubscriptLists({Section(1, 5), Scalar(2)},
           {Section(6, 10), Scalar(2)}) == DesignatorRelation::Disjoint);
}

void TestCompareParts() {
  SymbolFixture symbols;
  const semantics::Symbol &symbol1{symbols.MakeSymbol("a")};
  const semantics::Symbol &symbol2{symbols.MakeSymbol("b")};
  DesignatorPath::Part component1{{}, &symbol1};
  DesignatorPath::Part component1Again{{}, &symbol1};
  DesignatorPath::Part component2{{}, &symbol2};
  DesignatorPath::Part subscripts{{Section(1, 5)}, nullptr};
  DesignatorPath::Part subscriptedComponent{{Scalar(3)}, &symbol1};

  TEST(DesignatorPath::CompareParts(component1, component1Again) ==
      DesignatorRelation::Equal);
  TEST(DesignatorPath::CompareParts(component1, component2) ==
      DesignatorRelation::Disjoint);
  TEST(DesignatorPath::CompareParts(component1, subscripts) ==
      DesignatorRelation::Overlaps);
  TEST(DesignatorPath::CompareParts(subscripts, subscriptedComponent) ==
      DesignatorRelation::Contains);
}

void TestCombineRelations() {
  TEST(DesignatorPath::CombineRelations(false, false, false) ==
      DesignatorRelation::Equal);
  TEST(DesignatorPath::CombineRelations(true, false, false) ==
      DesignatorRelation::Contains);
  TEST(DesignatorPath::CombineRelations(false, true, false) ==
      DesignatorRelation::ContainedBy);
  TEST(DesignatorPath::CombineRelations(false, false, true) ==
      DesignatorRelation::Overlaps);
  TEST(DesignatorPath::CombineRelations(true, true, false) ==
      DesignatorRelation::Overlaps);
}

void TestComparePaths() {
  DesignatorPath empty;
  CheckRelation(empty, empty, DesignatorRelation::Equal);
  CheckRelation(
      empty, PathWithSubscripts({Scalar(1)}), DesignatorRelation::Disjoint);

  CheckRelation(PathWithSubscripts({Scalar(1)}),
      PathWithSubscripts({Scalar(1)}), DesignatorRelation::Equal);
  CheckRelation(PathWithSubscripts({Section(1, 10)}),
      PathWithSubscripts({Scalar(5)}), DesignatorRelation::Contains);
  CheckRelation(PathWithSubscripts({Scalar(5)}),
      PathWithSubscripts({Section(1, 10)}), DesignatorRelation::ContainedBy);
  CheckRelation(PathWithSubscripts({Section(1, 5)}),
      PathWithSubscripts({Section(5, 10)}), DesignatorRelation::Overlaps);
  CheckRelation(PathWithSubscripts({Section(1, 5)}),
      PathWithSubscripts({Section(6, 10)}), DesignatorRelation::Disjoint);

  SymbolFixture symbols;
  const semantics::Symbol &symbol{symbols.MakeSymbol("c")};
  DesignatorPath parent{PathWithComponent(&symbol)};
  DesignatorPath child{PathWithComponent(&symbol)};
  child.AddSubscripts({Scalar(1)});
  CheckRelation(parent, child, DesignatorRelation::Contains);
  CheckRelation(child, parent, DesignatorRelation::ContainedBy);
}

void TestMayContainSubscripts() {
  TEST(DesignatorPath::SubscriptMayContain(Scalar(1), Scalar(1)));
  TEST(DesignatorPath::SubscriptMayContain(FullSection(), Scalar(7)));
  TEST(!DesignatorPath::SubscriptMayContain(Scalar(7), FullSection()));
  TEST(!DesignatorPath::SubscriptMayContain(Section(1, 5), FullSection()));
  TEST(DesignatorPath::SubscriptMayContain(Section(1, 10), Scalar(7)));
  TEST(!DesignatorPath::SubscriptMayContain(Section(1, 5), Scalar(7)));
  TEST(DesignatorPath::SubscriptMayContain(Section(1, 5, 2), Scalar(7)));
  TEST(!DesignatorPath::SubscriptListMayContain(
      {Scalar(1)}, {Scalar(1), Scalar(2)}));
  TEST(DesignatorPath::SubscriptListMayContain({FullSection()}, {}));
  TEST(!DesignatorPath::SubscriptListMayContain(
      {FullSection()}, {Scalar(1), Scalar(2)}));
  TEST(DesignatorPath::SubscriptListMayContain(
      {FullSection(), Section(1, 10)}, {Scalar(2), Scalar(5)}));
}

void TestMayContainPartsAndPaths() {
  SymbolFixture symbols;
  const semantics::Symbol &symbol1{symbols.MakeSymbol("d")};
  const semantics::Symbol &symbol2{symbols.MakeSymbol("e")};
  DesignatorPath::Part component1{{}, &symbol1};
  DesignatorPath::Part component2{{}, &symbol2};
  DesignatorPath::Part subscripts{{Section(1, 10)}, nullptr};
  DesignatorPath::Part scalarSubscript{{Scalar(5)}, nullptr};
  DesignatorPath::Part scalarComponent{{Scalar(5)}, &symbol1};

  TEST(DesignatorPath::PartMayContain(component1, component1));
  TEST(!DesignatorPath::PartMayContain(component1, component2));
  TEST(DesignatorPath::PartMayContain(subscripts, scalarComponent));
  TEST(DesignatorPath::PartMayContain(subscripts, scalarSubscript));

  DesignatorPath empty;
  DesignatorPath parent{PathWithComponent(&symbol1)};
  DesignatorPath child{PathWithComponent(&symbol1)};
  child.AddSubscripts({Scalar(1)});
  DesignatorPath sibling{PathWithComponent(&symbol2)};

  TEST(empty.MayContain(parent));
  TEST(parent.MayContain(parent));
  TEST(parent.MayContain(child));
  TEST(!parent.MayContain(empty));
  TEST(!child.MayContain(parent));
  TEST(!parent.MayContain(sibling));
}

void TestAddFunctionsAndMap() {
  SymbolFixture symbols;
  const semantics::Symbol &symbol{symbols.MakeSymbol("f")};
  DesignatorPath path;
  TEST(path.empty());
  path.AddComponent(symbol);
  path.AddSubscripts({Scalar(1), Scalar(2)});
  TEST(path.Parts().size() == 2);
  TEST(path.Parts()[0].subscripts.empty());
  TEST(path.Parts()[0].symbol == &symbol);
  TEST(path.Parts()[1].subscripts.size() == 2);
  TEST(path.Parts()[1].symbol == nullptr);

  DesignatorPathMap<int> map;
  TEST(map.empty());
  map.push_back(path, 42);
  TEST(!map.empty());
  TEST(map.begin()->value == 42);
  map.erase(map.begin());
  TEST(map.empty());
  map.push_back(DesignatorPath{}, 7);
  map.clear();
  TEST(map.empty());
}

void TestSubscriptsPrecedeComponentWithinPart() {
  SymbolFixture symbols;
  const semantics::Symbol &base{symbols.MakeSymbol("g")};
  const semantics::Symbol &y{symbols.MakeSymbol("h")};
  const semantics::Symbol &z{symbols.MakeSymbol("i")};

  DesignatorPath x;
  x.SetBase(NamedEntity{base});
  TEST(x.Base().has_value());
  TEST(x.Parts().empty());

  DesignatorPath differentBase;
  differentBase.SetBase(NamedEntity{y});

  DesignatorPath xFull;
  xFull.SetBase(NamedEntity{base});
  xFull.AddSubscripts({FullSection()});
  TEST(xFull.Parts().size() == 1);
  TEST(xFull.Parts()[0].subscripts.size() == 1);
  TEST(xFull.Parts()[0].subscripts[0] == FullSection());
  const auto *fullTriplet{
      std::get_if<Triplet>(&xFull.Parts()[0].subscripts[0].u)};
  TEST(fullTriplet != nullptr);
  if (fullTriplet) {
    TEST(!fullTriplet->GetLower());
    TEST(!fullTriplet->GetUpper());
    TEST(DesignatorPath::IsFullTriplet(*fullTriplet));
  }
  TEST(xFull.Parts()[0].symbol == nullptr);
  TEST(!(xFull == x));
  TEST(x.Compare(xFull) == DesignatorRelation::Equal);
  TEST(xFull.Compare(x) == DesignatorRelation::Equal);
  TEST(x.MayContain(xFull));
  TEST(xFull.MayContain(x));
  TEST(!xFull.MayContain(differentBase));

  DesignatorPath xSection;
  xSection.SetBase(NamedEntity{base});
  xSection.AddSubscripts({Section(1, 10)});
  TEST(xSection.Base().has_value());
  TEST(xSection.Parts().size() == 1);
  TEST(xSection.Parts()[0].subscripts.size() == 1);
  TEST(xSection.Parts()[0].symbol == nullptr);

  DesignatorPath xSectionY;
  xSectionY.SetBase(NamedEntity{base});
  xSectionY.AddSubscripts({Section(1, 10)});
  xSectionY.AddComponent(y);
  TEST(xSectionY.Parts().size() == 1);
  TEST(xSectionY.Parts()[0].subscripts.size() == 1);
  TEST(xSectionY.Parts()[0].symbol == &y);

  DesignatorPath xSectionYFull{xSectionY};
  xSectionYFull.AddSubscripts({FullSection()});
  TEST(xSectionYFull.Parts().size() == 2);
  TEST(xSectionYFull.Parts()[0].subscripts.size() == 1);
  TEST(xSectionYFull.Parts()[0].symbol == &y);
  TEST(xSectionYFull.Parts()[1].subscripts.size() == 1);
  TEST(xSectionYFull.Parts()[1].subscripts[0] == FullSection());
  TEST(xSectionYFull.Parts()[1].symbol == nullptr);
  TEST(!(xSectionYFull == xSectionY));
  TEST(xSectionYFull.Compare(xSectionY) == DesignatorRelation::Equal);
  TEST(xSectionY.Compare(xSectionYFull) == DesignatorRelation::Equal);
  TEST(xSectionYFull.MayContain(xSectionY));
  TEST(xSectionY.MayContain(xSectionYFull));

  DesignatorPath xSectionYFullZ;
  xSectionYFullZ.SetBase(NamedEntity{base});
  xSectionYFullZ.AddSubscripts({Section(1, 10)});
  xSectionYFullZ.AddComponent(y);
  xSectionYFullZ.AddSubscripts({FullSection()});
  xSectionYFullZ.AddComponent(z);
  TEST(xSectionYFullZ.Parts().size() == 2);
  TEST(xSectionYFullZ.Parts()[0].subscripts.size() == 1);
  TEST(xSectionYFullZ.Parts()[0].symbol == &y);
  TEST(xSectionYFullZ.Parts()[1].subscripts.size() == 1);
  TEST(xSectionYFullZ.Parts()[1].subscripts[0] == FullSection());
  TEST(xSectionYFullZ.Parts()[1].symbol == &z);
}

} // namespace

int main() {
  TestGetConstantSubscriptRange();
  TestFullTripletDetection();
  TestCompareSubscripts();
  TestCompareSubscriptLists();
  TestCompareParts();
  TestCombineRelations();
  TestComparePaths();
  TestMayContainSubscripts();
  TestMayContainPartsAndPaths();
  TestAddFunctionsAndMap();
  TestSubscriptsPrecedeComponentWithinPart();
  return testing::Complete();
}
