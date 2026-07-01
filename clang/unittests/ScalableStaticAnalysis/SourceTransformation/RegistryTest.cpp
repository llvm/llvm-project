//===- RegistryTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixture.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformation.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace ssaf;

namespace {

class StubEditEmitter : public SourceEditEmitter {
public:
  void addReplacement(clang::tooling::Replacement) override {}
};

class StubReportEmitter : public TransformationReportEmitter {
public:
  void addResult(StringRef, clang::SarifResultLevel, clang::CharSourceRange,
                 StringRef) override {}
};

class StubTransformation : public Transformation {
public:
  using Transformation::Transformation;
};

} // namespace

static TransformationRegistry::Add<StubTransformation>
    RegisterStubTransformation("stub-transformation",
                               "A transformation for testing");

namespace {

class TransformationRegistryTest : public TestFixture {};

TEST_F(TransformationRegistryTest, isTransformationRegistered) {
  EXPECT_FALSE(isTransformationRegistered("not-a-transformation"));
  EXPECT_TRUE(isTransformationRegistered("stub-transformation"));
}

TEST_F(TransformationRegistryTest, makeTransformation) {
  WPASuite Suite = makeWPASuite();
  StubEditEmitter Edits;
  StubReportEmitter Report;
  std::unique_ptr<Transformation> T =
      makeTransformation("stub-transformation", Suite, Edits, Report);
  EXPECT_NE(T, nullptr);
}

TEST_F(TransformationRegistryTest, EnumeratingRegistryEntries) {
  auto Entries = TransformationRegistry::entries();
  EXPECT_TRUE(llvm::any_of(Entries, [](const auto &Entry) {
    return StringRef(Entry.getName()) == "stub-transformation";
  }));
}

TEST_F(TransformationRegistryTest, PrintAvailableTransformations) {
  std::string Buffer;
  raw_string_ostream OS(Buffer);
  printAvailableTransformations(OS);
  EXPECT_NE(StringRef(Buffer).find("stub-transformation"), StringRef::npos);
  EXPECT_NE(StringRef(Buffer).find("A transformation for testing"),
            StringRef::npos);
}

} // namespace
