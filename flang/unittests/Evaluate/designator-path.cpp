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
#include <initializer_list>

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

// Compare the selection of a single subscript with that of another.
void CheckSubscriptRelation(
    const Subscript &x, const Subscript &y, DesignatorRelation relation) {
  CheckRelation(PathWithSubscripts({x}), PathWithSubscripts({y}), relation);
}

void CheckSubscriptMayContain(
    const Subscript &x, const Subscript &y, bool expected) {
  TEST(PathWithSubscripts({x}).MayContain(PathWithSubscripts({y})) == expected);
}

class SymbolFixture {
public:
  const semantics::Symbol &MakeSymbol(const char *name) {
    return scope_.MakeSymbol(parser::CharBlock{name}, semantics::Attrs{},
        semantics::UnknownDetails{});
  }

  semantics::Symbol &MakeCommonBlock(const char *name) {
    return scope_.MakeSymbol(parser::CharBlock{name}, semantics::Attrs{},
        semantics::CommonBlockDetails{parser::CharBlock{name}});
  }

  const semantics::Symbol &MakeCommonMember(
      const char *name, semantics::Symbol &block) {
    auto &symbol{scope_.MakeSymbol(parser::CharBlock{name}, semantics::Attrs{},
        semantics::ObjectEntityDetails{})};
    symbol.get<semantics::ObjectEntityDetails>().set_commonBlock(block);
    block.get<semantics::CommonBlockDetails>().add_object(symbol);
    return symbol;
  }

  const semantics::Symbol &MakeHostAssociated(
      const char *name, const semantics::Symbol &symbol) {
    return scope_.MakeSymbol(parser::CharBlock{name}, semantics::Attrs{},
        semantics::HostAssocDetails{symbol});
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

void TestCompareSubscripts() {
  CheckSubscriptRelation(Scalar(3), Scalar(3), DesignatorRelation::Equal);
  CheckSubscriptRelation(Scalar(3), Scalar(4), DesignatorRelation::Disjoint);
  CheckSubscriptRelation(
      FullSection(), Scalar(3), DesignatorRelation::Contains);
  CheckSubscriptRelation(
      FullSection(), FullSection(), DesignatorRelation::Equal);
  CheckSubscriptRelation(
      Scalar(3), FullSection(), DesignatorRelation::ContainedBy);
  CheckSubscriptRelation(
      Section(1, 5), Section(6, 10), DesignatorRelation::Disjoint);
  CheckSubscriptRelation(
      Section(1, 5), Section(1, 5), DesignatorRelation::Equal);
  CheckSubscriptRelation(
      Section(1, 10), Section(3, 5), DesignatorRelation::Contains);
  CheckSubscriptRelation(
      Section(3, 5), Section(1, 10), DesignatorRelation::ContainedBy);
  CheckSubscriptRelation(
      Section(1, 5), Section(5, 10), DesignatorRelation::Overlaps);
  // A scalar and a one-element section select the same element.
  CheckSubscriptRelation(Scalar(3), Section(3, 3), DesignatorRelation::Equal);
  // Strided sections are not compared precisely and are reported as Disjoint.
  CheckSubscriptRelation(
      Section(1, 5, 2), Section(1, 5), DesignatorRelation::Disjoint);
}

void TestCompareSubscriptLists() {
  CheckRelation(PathWithSubscripts({}), PathWithSubscripts({FullSection()}),
      DesignatorRelation::Equal);
  CheckRelation(PathWithSubscripts({FullSection()}), PathWithSubscripts({}),
      DesignatorRelation::Equal);
  CheckRelation(PathWithSubscripts({FullSection()}),
      PathWithSubscripts({FullSection(), FullSection()}),
      DesignatorRelation::Disjoint);
  CheckRelation(PathWithSubscripts({Scalar(1)}),
      PathWithSubscripts({Scalar(1), Scalar(2)}), DesignatorRelation::Disjoint);
  CheckRelation(PathWithSubscripts({Scalar(1), Scalar(2)}),
      PathWithSubscripts({Scalar(1), Scalar(2)}), DesignatorRelation::Equal);
  CheckRelation(PathWithSubscripts({Section(1, 10), Scalar(2)}),
      PathWithSubscripts({Section(3, 5), Scalar(2)}),
      DesignatorRelation::Contains);
  CheckRelation(PathWithSubscripts({Section(3, 5), Scalar(2)}),
      PathWithSubscripts({Section(1, 10), Scalar(2)}),
      DesignatorRelation::ContainedBy);
  // One dimension contains and the other is contained, so they only overlap.
  CheckRelation(PathWithSubscripts({Section(1, 10), Scalar(2)}),
      PathWithSubscripts({Section(3, 5), FullSection()}),
      DesignatorRelation::Overlaps);
  CheckRelation(PathWithSubscripts({Section(1, 5), Scalar(2)}),
      PathWithSubscripts({Section(6, 10), Scalar(2)}),
      DesignatorRelation::Disjoint);
}

void TestCompareParts() {
  SymbolFixture symbols;
  const semantics::Symbol &symbol1{symbols.MakeSymbol("a")};
  const semantics::Symbol &symbol2{symbols.MakeSymbol("b")};
  DesignatorPath component1{PathWithComponent(&symbol1)};
  DesignatorPath component1Again{PathWithComponent(&symbol1)};
  DesignatorPath component2{PathWithComponent(&symbol2)};
  DesignatorPath subscripts{PathWithSubscripts({Section(1, 5)})};
  DesignatorPath subscriptedComponent{PathWithSubscripts({Scalar(3)})};
  subscriptedComponent.AddComponent(symbol1);

  CheckRelation(component1, component1Again, DesignatorRelation::Equal);
  CheckRelation(component1, component2, DesignatorRelation::Disjoint);
  CheckRelation(component1, subscripts, DesignatorRelation::Overlaps);
  CheckRelation(subscripts, subscriptedComponent, DesignatorRelation::Contains);
}

void TestComparePaths() {
  DesignatorPath empty;
  CheckRelation(empty, empty, DesignatorRelation::Equal);
  CheckRelation(
      empty, PathWithSubscripts({Scalar(1)}), DesignatorRelation::Disjoint);

  SymbolFixture symbols;
  const semantics::Symbol &symbol{symbols.MakeSymbol("c")};
  DesignatorPath parent{PathWithComponent(&symbol)};
  DesignatorPath child{PathWithComponent(&symbol)};
  child.AddSubscripts({Scalar(1)});
  CheckRelation(parent, child, DesignatorRelation::Contains);
  CheckRelation(child, parent, DesignatorRelation::ContainedBy);
}

void TestMayContainSubscripts() {
  CheckSubscriptMayContain(Scalar(1), Scalar(1), true);
  CheckSubscriptMayContain(FullSection(), Scalar(7), true);
  CheckSubscriptMayContain(Scalar(7), FullSection(), false);
  CheckSubscriptMayContain(Section(1, 5), FullSection(), false);
  CheckSubscriptMayContain(Section(1, 10), Scalar(7), true);
  CheckSubscriptMayContain(Section(1, 5), Scalar(7), false);

  // A scalar covers a one-element section of the same element, and only that.
  CheckSubscriptMayContain(Scalar(1), Section(1, 1), true);
  CheckSubscriptMayContain(Section(1, 1), Scalar(1), true);
  CheckSubscriptMayContain(Scalar(1), Section(1, 2), false);
  CheckSubscriptMayContain(Scalar(2), Section(1, 1), false);

  // Constant strides decide containment exactly.
  CheckSubscriptMayContain(Section(1, 5, 2), Scalar(1), true);
  CheckSubscriptMayContain(Section(1, 5, 2), Scalar(3), true);
  CheckSubscriptMayContain(Section(1, 5, 2), Scalar(5), true);
  CheckSubscriptMayContain(Section(1, 5, 2), Scalar(4), false);
  CheckSubscriptMayContain(Section(1, 5, 2), Scalar(7), false);
  CheckSubscriptMayContain(Section(1, 9, 2), Section(3, 7, 2), true);
  CheckSubscriptMayContain(Section(1, 9, 2), Section(3, 7, 4), true);
  CheckSubscriptMayContain(Section(1, 9, 2), Section(3, 7, 3), false);
  CheckSubscriptMayContain(Section(1, 9, 4), Section(1, 9, 2), false);
  CheckSubscriptMayContain(Section(9, 1, -2), Scalar(5), true);
  CheckSubscriptMayContain(Section(9, 1, -2), Scalar(4), false);

  CheckSubscriptMayContain(Scalar(1), Scalar(2), false);

  TEST(!PathWithSubscripts({Scalar(1)})
          .MayContain(PathWithSubscripts({Scalar(1), Scalar(2)})));
  TEST(PathWithSubscripts({FullSection()}).MayContain(PathWithSubscripts({})));
  TEST(!PathWithSubscripts({FullSection()})
          .MayContain(PathWithSubscripts({Scalar(1), Scalar(2)})));
  TEST(PathWithSubscripts({FullSection(), Section(1, 10)})
          .MayContain(PathWithSubscripts({Scalar(2), Scalar(5)})));
}

void TestMayContainPartsAndPaths() {
  SymbolFixture symbols;
  const semantics::Symbol &symbol1{symbols.MakeSymbol("d")};
  const semantics::Symbol &symbol2{symbols.MakeSymbol("e")};
  DesignatorPath component1{PathWithComponent(&symbol1)};
  DesignatorPath component2{PathWithComponent(&symbol2)};
  DesignatorPath subscripts{PathWithSubscripts({Section(1, 10)})};
  DesignatorPath scalarSubscript{PathWithSubscripts({Scalar(5)})};
  DesignatorPath scalarComponent{PathWithSubscripts({Scalar(5)})};
  scalarComponent.AddComponent(symbol1);

  TEST(component1.MayContain(component1));
  TEST(!component1.MayContain(component2));
  TEST(subscripts.MayContain(scalarComponent));
  TEST(subscripts.MayContain(scalarSubscript));

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

// DEFAULT(NONE) relies on MayContain agreeing with Compare, and the conflict
// analysis relies on Compare being symmetric.
void TestCompareIsSymmetricAndAgreesWithMayContain() {
  SymbolFixture symbols;
  auto &block{symbols.MakeCommonBlock("blk")};
  const auto &member{symbols.MakeCommonMember("m", block)};
  const auto &sibling{symbols.MakeCommonMember("n", block)};
  const auto &component1{symbols.MakeSymbol("p")};
  const auto &component2{symbols.MakeSymbol("q")};

  std::vector<DesignatorPath> paths;
  paths.emplace_back();
  for (const Subscript &subscript :
      {Scalar(1), Scalar(5), Scalar(7), Section(1, 1), Section(1, 5),
          Section(1, 10), Section(5, 10), Section(6, 10), FullSection(),
          Section(1, 5, 2), Section(2, 10, 4), Section(10, 1, -1)}) {
    paths.push_back(PathWithSubscripts({subscript}));
    DesignatorPath withComponent{PathWithSubscripts({subscript})};
    withComponent.AddComponent(component1);
    paths.push_back(withComponent);
  }
  paths.push_back(PathWithComponent(&component1));
  paths.push_back(PathWithComponent(&component2));
  for (const semantics::Symbol *base :
      std::initializer_list<const semantics::Symbol *>{
          &block, &member, &sibling}) {
    DesignatorPath path;
    path.SetBase(NamedEntity{*base});
    paths.push_back(path);
    path.AddSubscripts({Section(1, 5)});
    paths.push_back(path);
  }

  for (const DesignatorPath &x : paths) {
    for (const DesignatorPath &y : paths) {
      DesignatorRelation relation{x.Compare(y)};
      DesignatorRelation reverse{y.Compare(x)};
      switch (relation) {
      case DesignatorRelation::Equal:
      case DesignatorRelation::Overlaps:
      case DesignatorRelation::Disjoint:
        TEST(reverse == relation);
        break;
      case DesignatorRelation::Contains:
        TEST(reverse == DesignatorRelation::ContainedBy);
        break;
      case DesignatorRelation::ContainedBy:
        TEST(reverse == DesignatorRelation::Contains);
        break;
      }
      if (relation == DesignatorRelation::Equal) {
        TEST(x.MayContain(y));
        TEST(y.MayContain(x));
      } else if (relation == DesignatorRelation::Contains) {
        TEST(x.MayContain(y));
      }
    }
  }
}

void TestPathConstruction() {
  SymbolFixture symbols;
  const semantics::Symbol &symbol{symbols.MakeSymbol("f")};
  DesignatorPath path;
  TEST(path.empty());
  path.AddComponent(symbol);
  path.AddSubscripts({Scalar(1), Scalar(2)});
  TEST(!path.empty());
  TEST(path.parts().size() == 2);
  TEST(path.parts()[0].subscripts.empty());
  TEST(path.parts()[0].symbol == &symbol);
  TEST(path.parts()[1].subscripts.size() == 2);
  TEST(path.parts()[1].symbol == nullptr);
}

void TestSubscriptsPrecedeComponentWithinPart() {
  SymbolFixture symbols;
  const semantics::Symbol &base{symbols.MakeSymbol("g")};
  const semantics::Symbol &y{symbols.MakeSymbol("h")};
  const semantics::Symbol &z{symbols.MakeSymbol("i")};

  DesignatorPath x;
  x.SetBase(NamedEntity{base});
  TEST(x.base().has_value());
  TEST(x.parts().empty());
  TEST(x.HasBaseOnly());

  DesignatorPath differentBase;
  differentBase.SetBase(NamedEntity{y});

  DesignatorPath xFull;
  xFull.SetBase(NamedEntity{base});
  xFull.AddSubscripts({FullSection()});
  TEST(xFull.parts().size() == 1);
  TEST(xFull.parts()[0].subscripts.size() == 1);
  TEST(xFull.parts()[0].subscripts[0] == FullSection());
  const auto *fullTriplet{
      std::get_if<Triplet>(&xFull.parts()[0].subscripts[0].u)};
  TEST(fullTriplet != nullptr);
  if (fullTriplet) {
    TEST(!fullTriplet->GetLower());
    TEST(!fullTriplet->GetUpper());
  }
  TEST(xFull.parts()[0].symbol == nullptr);
  TEST(!(xFull == x));
  TEST(x.Compare(xFull) == DesignatorRelation::Equal);
  TEST(xFull.Compare(x) == DesignatorRelation::Equal);
  TEST(x.MayContain(xFull));
  TEST(xFull.MayContain(x));
  TEST(!xFull.MayContain(differentBase));

  DesignatorPath xSection;
  xSection.SetBase(NamedEntity{base});
  xSection.AddSubscripts({Section(1, 10)});
  TEST(xSection.base().has_value());
  TEST(xSection.parts().size() == 1);
  TEST(xSection.parts()[0].subscripts.size() == 1);
  TEST(xSection.parts()[0].symbol == nullptr);

  DesignatorPath xSectionY;
  xSectionY.SetBase(NamedEntity{base});
  xSectionY.AddSubscripts({Section(1, 10)});
  xSectionY.AddComponent(y);
  TEST(xSectionY.parts().size() == 1);
  TEST(xSectionY.parts()[0].subscripts.size() == 1);
  TEST(xSectionY.parts()[0].symbol == &y);

  DesignatorPath xSectionYFull{xSectionY};
  xSectionYFull.AddSubscripts({FullSection()});
  TEST(xSectionYFull.parts().size() == 2);
  TEST(xSectionYFull.parts()[0].subscripts.size() == 1);
  TEST(xSectionYFull.parts()[0].symbol == &y);
  TEST(xSectionYFull.parts()[1].subscripts.size() == 1);
  TEST(xSectionYFull.parts()[1].subscripts[0] == FullSection());
  TEST(xSectionYFull.parts()[1].symbol == nullptr);
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
  TEST(xSectionYFullZ.parts().size() == 2);
  TEST(xSectionYFullZ.parts()[0].subscripts.size() == 1);
  TEST(xSectionYFullZ.parts()[0].symbol == &y);
  TEST(xSectionYFullZ.parts()[1].subscripts.size() == 1);
  TEST(xSectionYFullZ.parts()[1].subscripts[0] == FullSection());
  TEST(xSectionYFullZ.parts()[1].symbol == &z);
}

void TestAsFortran() {
  SymbolFixture symbols;
  const semantics::Symbol &a{symbols.MakeSymbol("a")};
  const semantics::Symbol &x{symbols.MakeSymbol("x")};
  const semantics::Symbol &y{symbols.MakeSymbol("y")};
  DesignatorPath path;
  path.SetBase(NamedEntity{a});
  TEST(path.AsFortran() == "a");
  path.AddSubscripts({Scalar(1), Section(2, 4)});
  TEST(path.AsFortran() == "a(1_8,2_8:4_8:1_8)");
  path.AddComponent(x);
  TEST(path.AsFortran() == "a(1_8,2_8:4_8:1_8)%x");
  path.AddSubscripts({FullSection()});
  path.AddComponent(y);
  TEST(path.AsFortran() == "a(1_8,2_8:4_8:1_8)%x(::1_8)%y");
}

void TestCommonBlockPaths() {
  SymbolFixture symbols;
  auto &block{symbols.MakeCommonBlock("blk")};
  auto &otherBlock{symbols.MakeCommonBlock("other")};
  const auto &a{symbols.MakeCommonMember("a", block)};
  const auto &b{symbols.MakeCommonMember("b", block)};
  const auto &c{symbols.MakeCommonMember("c", otherBlock)};
  const auto &component{symbols.MakeSymbol("component")};
  const auto &unrelated{symbols.MakeSymbol("unrelated")};
  const auto &alias{symbols.MakeHostAssociated("alias", a)};

  DesignatorPath wholeBlock, member, sibling, other, outside, hostAssociated;
  wholeBlock.SetBase(NamedEntity{block});
  member.SetBase(NamedEntity{a});
  sibling.SetBase(NamedEntity{b});
  other.SetBase(NamedEntity{c});
  outside.SetBase(NamedEntity{unrelated});
  hostAssociated.SetBase(NamedEntity{alias});
  TEST(wholeBlock.commonBlock() == &block);
  TEST(member.commonBlock() == &block);
  TEST(hostAssociated.commonBlock() == &block);
  TEST(!outside.commonBlock());

  CheckRelation(wholeBlock, member, DesignatorRelation::Contains);
  CheckRelation(member, wholeBlock, DesignatorRelation::ContainedBy);
  CheckRelation(wholeBlock, hostAssociated, DesignatorRelation::Contains);
  CheckRelation(member, sibling, DesignatorRelation::Disjoint);
  CheckRelation(wholeBlock, other, DesignatorRelation::Disjoint);
  CheckRelation(wholeBlock, outside, DesignatorRelation::Disjoint);
  TEST(wholeBlock.MayContain(member));
  TEST(!member.MayContain(wholeBlock));
  TEST(!wholeBlock.MayContain(other));
  TEST(!member.MayContain(sibling));

  auto section{member};
  section.AddSubscripts({Section(1, 5)});
  auto subobject{member};
  subobject.AddComponent(component);
  CheckRelation(wholeBlock, section, DesignatorRelation::Contains);
  CheckRelation(section, wholeBlock, DesignatorRelation::ContainedBy);
  CheckRelation(wholeBlock, subobject, DesignatorRelation::Contains);
  TEST(wholeBlock.MayContain(section));
  TEST(wholeBlock.MayContain(subobject));
  TEST(section.commonBlock() == &block);

  auto copied{wholeBlock};
  TEST(copied == wholeBlock);
  copied.SetBase(NamedEntity{unrelated});
  TEST(!copied.commonBlock());
  CheckRelation(copied, member, DesignatorRelation::Disjoint);
}

} // namespace

int main() {
  TestCompareSubscripts();
  TestCompareSubscriptLists();
  TestCompareParts();
  TestComparePaths();
  TestMayContainSubscripts();
  TestMayContainPartsAndPaths();
  TestCompareIsSymmetricAndAgreesWithMayContain();
  TestPathConstruction();
  TestSubscriptsPrecedeComponentWithinPart();
  TestAsFortran();
  TestCommonBlockPaths();
  return testing::Complete();
}
