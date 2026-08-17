//===- EntityLinkageTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using clang::ssaf::EntityBinding;
using clang::ssaf::EntityCoalescing;
using clang::ssaf::EntityDefinitionKind;
using clang::ssaf::EntityLinkage;
using clang::ssaf::EntityLinkageType;
using clang::ssaf::EntityVisibility;

namespace {

constexpr inline auto None = EntityLinkageType::None;
constexpr inline auto Internal = EntityLinkageType::Internal;
constexpr inline auto External = EntityLinkageType::External;

// Builds an EntityLinkage with the given linkage type and fixed defaults for
// the remaining properties (an ordinary strong external definition).
constexpr EntityLinkage mk(EntityLinkageType LT) {
  return EntityLinkage(LT, EntityBinding::Strong, EntityCoalescing::None,
                       EntityVisibility::Default,
                       EntityDefinitionKind::Definition);
}

TEST(EntityLinkageTest, Constructor) {
  EntityLinkage L(External, EntityBinding::Weak, EntityCoalescing::ODR,
                  EntityVisibility::Hidden, EntityDefinitionKind::Declaration);
  EXPECT_EQ(L.getLinkage(), External);
}

TEST(EntityLinkageTest, CopyConstructor) {
  EntityLinkage Original = mk(External);
  EntityLinkage Copy = Original;

  EXPECT_EQ(Copy, Original);
}

TEST(EntityLinkageTest, AssignmentOperator) {
  EntityLinkage Linkage1 = mk(None);
  EntityLinkage Linkage2 = mk(External);

  Linkage1 = Linkage2;

  EXPECT_EQ(Linkage1, Linkage2);
}

TEST(EntityLinkageTest, EqualityOperatorReflexive) {
  EXPECT_EQ(mk(None), mk(None));
  EXPECT_EQ(mk(Internal), mk(Internal));
  EXPECT_EQ(mk(External), mk(External));
}

TEST(EntityLinkageTest, EqualityOperatorDistinct) {
  EXPECT_NE(mk(None), mk(Internal));
  EXPECT_NE(mk(None), mk(External));
  EXPECT_NE(mk(Internal), mk(External));
}

TEST(EntityLinkageTest, EqualityConsidersAllProperties) {
  const EntityLinkage Base(External, EntityBinding::Strong,
                           EntityCoalescing::None, EntityVisibility::Default,
                           EntityDefinitionKind::Definition);
  EXPECT_NE(Base,
            EntityLinkage(External, EntityBinding::Weak, EntityCoalescing::None,
                          EntityVisibility::Default,
                          EntityDefinitionKind::Definition));
  EXPECT_NE(Base,
            EntityLinkage(External, EntityBinding::Strong,
                          EntityCoalescing::ODR, EntityVisibility::Default,
                          EntityDefinitionKind::Definition));
  EXPECT_NE(Base,
            EntityLinkage(External, EntityBinding::Strong,
                          EntityCoalescing::None, EntityVisibility::Hidden,
                          EntityDefinitionKind::Definition));
  EXPECT_NE(Base,
            EntityLinkage(External, EntityBinding::Strong,
                          EntityCoalescing::None, EntityVisibility::Default,
                          EntityDefinitionKind::Declaration));
}

TEST(EntityLinkageTypeTest, FormatProvider) {
  EXPECT_EQ(llvm::formatv("{0}", EntityLinkageType::None).str(), "None");
  EXPECT_EQ(llvm::formatv("{0}", EntityLinkageType::Internal).str(),
            "Internal");
  EXPECT_EQ(llvm::formatv("{0}", EntityLinkageType::External).str(),
            "External");
}

TEST(EntityLinkageTypeTest, StreamOutputNone) {
  std::string S;
  llvm::raw_string_ostream(S) << EntityLinkageType::None;
  EXPECT_EQ(S, "None");
}

TEST(EntityLinkageTypeTest, StreamOutputInternal) {
  std::string S;
  llvm::raw_string_ostream(S) << EntityLinkageType::Internal;
  EXPECT_EQ(S, "Internal");
}

TEST(EntityLinkageTypeTest, StreamOutputExternal) {
  std::string S;
  llvm::raw_string_ostream(S) << EntityLinkageType::External;
  EXPECT_EQ(S, "External");
}

TEST(EntityLinkageTest, FormatProvider) {
  EXPECT_EQ(llvm::formatv("{0}", mk(None)).str(),
            "EntityLinkage(None, Strong, None, Default, Definition)");
}

TEST(EntityLinkageTest, StreamOutput) {
  std::string S;
  llvm::raw_string_ostream(S) << EntityLinkage(
      External, EntityBinding::Weak, EntityCoalescing::ODR,
      EntityVisibility::Hidden, EntityDefinitionKind::Declaration);
  EXPECT_EQ(S, "EntityLinkage(External, Weak, ODR, Hidden, Declaration)");
}

} // namespace
