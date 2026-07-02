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
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReport.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportFormatRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

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

//===----------------------------------------------------------------------===//
// TransformationReportFormatRegistry tests.
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Test-local format stubs.
//===----------------------------------------------------------------------===//

class FakeTransformationReportFormat : public TransformationReportFormat {
public:
  virtual llvm::StringRef tag() const {
    return "FakeTransformationReportFormat";
  }

  llvm::Error write(const ReportDocument &, llvm::StringRef) override {
    return llvm::Error::success();
  }
};

class FakeTransformationReportFormatA : public FakeTransformationReportFormat {
public:
  llvm::StringRef tag() const override { return "A"; }
};
class FakeTransformationReportFormatB : public FakeTransformationReportFormat {
public:
  llvm::StringRef tag() const override { return "B"; }
};

} // namespace

//===----------------------------------------------------------------------===//
// File-scope registrations.
//===----------------------------------------------------------------------===//

static TransformationReportFormatRegistry::Add<FakeTransformationReportFormatA>
    RegisterFakeReportSarifA("report-sarif",
                             "Fake transformation-report format A");

// Duplicate report registration.
static TransformationReportFormatRegistry::Add<FakeTransformationReportFormatB>
    RegisterFakeReportSarifB("report-sarif",
                             "Fake transformation-report format B (duplicate)");

namespace {

TEST(TransformationReportFormatRegistryTest, MissReturnsFalse) {
  EXPECT_FALSE(isTransformationReportFormatRegistered("report-nonexistent"));
}

TEST(TransformationReportFormatRegistryTest, HitReturnsTrue) {
  EXPECT_TRUE(isTransformationReportFormatRegistered("report-sarif"));
}

TEST(TransformationReportFormatRegistryTest, MakeReturnsNullptrOnMiss) {
  EXPECT_EQ(makeTransformationReportFormat("report-nonexistent"), nullptr);
}

TEST(TransformationReportFormatRegistryTest, MakeReturnsNonNullOnHit) {
  auto Format = makeTransformationReportFormat("report-sarif");
  ASSERT_NE(Format, nullptr);
  auto *Fake = static_cast<FakeTransformationReportFormat *>(Format.get());
  EXPECT_EQ(Fake->tag(), "A");
}

TEST(TransformationReportFormatRegistryTest, LookupIsCaseSensitive) {
  EXPECT_FALSE(isTransformationReportFormatRegistered("REPORT-SARIF"));
  EXPECT_EQ(makeTransformationReportFormat("REPORT-SARIF"), nullptr);
}

TEST(TransformationReportFormatRegistryTest,
     MultiComponentExtensionDoesNotPartialMatch) {
  EXPECT_FALSE(isTransformationReportFormatRegistered("report-sarif.json"));
  EXPECT_EQ(makeTransformationReportFormat("report-sarif.json"), nullptr);

  EXPECT_TRUE(isTransformationReportFormatRegistered("report-sarif"));
  EXPECT_NE(makeTransformationReportFormat("report-sarif"), nullptr);
}

TEST(TransformationReportFormatRegistryTest,
     DuplicateRegistrationBothEntriesPresent) {
  unsigned Count = 0;
  for (const auto &Entry : TransformationReportFormatRegistry::entries())
    if (llvm::StringRef(Entry.getName()) == "report-sarif")
      ++Count;
  EXPECT_EQ(Count, 2u);
}

TEST(TransformationReportFormatRegistryTest,
     ExtensionResolutionOfRSarifDotJson) {
  llvm::StringRef Path = "r.sarif.json";
  llvm::StringRef Ext = llvm::sys::path::extension(Path);
  EXPECT_EQ(Ext, ".json");
  ASSERT_TRUE(Ext.consume_front("."));
  EXPECT_EQ(Ext, "json");

  EXPECT_FALSE(isTransformationReportFormatRegistered(Ext));
  EXPECT_EQ(makeTransformationReportFormat(Ext), nullptr);
}

TEST(TransformationReportFormatRegistryTest, PrintAvailableFullOutput) {
  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  printAvailableTransformationReportFormats(OS);

  llvm::SmallVector<llvm::StringRef> Lines;
  llvm::StringRef(Buf).split(Lines, '\n');

  // The header and blank separator are deterministic.
  ASSERT_GE(Lines.size(), 2u);
  EXPECT_EQ(Lines[0],
            "OVERVIEW: Available SSAF transformation-report formats:");
  EXPECT_EQ(Lines[1], "");

  // Entry order across the registry is unspecified (globals across TUs),
  // so match entries by set membership on complete lines rather than by
  // index. Also tolerates other formats being registered into the test
  // binary (e.g. the real "sarif" format if force-linker wiring lands).
  auto containsLine = [&](llvm::StringRef Expected) {
    return llvm::any_of(llvm::drop_begin(Lines, 2),
                        [&](llvm::StringRef L) { return L == Expected; });
  };

  EXPECT_TRUE(
      containsLine("  report-sarif - Fake transformation-report format A"));
  EXPECT_TRUE(containsLine(
      "  report-sarif - Fake transformation-report format B (duplicate)"));
}

} // namespace
