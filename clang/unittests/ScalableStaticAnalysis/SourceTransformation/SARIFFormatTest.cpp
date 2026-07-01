//===- SARIFFormatTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the SARIF transformation-report format.
//
// Each test builds a small ReportDocument, runs the SARIF writer
// (registered as "sarif"), and asserts on the emitted document:
//   - tool.driver.name / fullName / version
//   - result.level is currently hardcoded to "note"
//   - absent Range (std::nullopt) -> omitted `locations` key
//   - empty ruleId is forwarded verbatim
//   - no `fix` / `fixes` keys on any result
//   - artifactLocation.uri is an absolute file:// URI
//   - zero results produces a well-formed doc with empty runs[0].results
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/ScalableStaticAnalysis/SSAFForceLinker.h" // IWYU pragma: keep
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReport.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportFormatRegistry.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang;
using namespace clang::ssaf;

namespace {

//===----------------------------------------------------------------------===//
// Test fixture: creates a unique per-test directory (mirrors the pattern in
// clang/unittests/ScalableStaticAnalysis/Serialization/JSONFormatTest so
// output paths do not pre-exist and the writer's `file_exists` preflight
// passes), and a minimal SourceManager backed by an in-memory VFS so tests
// can construct real CharSourceRanges pointing into a fake source file.
//===----------------------------------------------------------------------===//
class SARIFFormatTest : public ::testing::Test {
public:
  using PathString = llvm::SmallString<128>;

protected:
  SARIFFormatTest()
      : InMemoryFileSystem(
            llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>()),
        FileMgr(FileSystemOptions(), InMemoryFileSystem),
        Diags(DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr) {}

  void SetUp() override {
    std::error_code EC =
        llvm::sys::fs::createUniqueDirectory("sarif-format-test", TestDir);

    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { llvm::sys::fs::remove_directories(TestDir); }

  PathString makePath(llvm::StringRef FileOrDirectoryName) const {
    PathString FullPath = TestDir;
    llvm::sys::path::append(FullPath, FileOrDirectoryName);
    return FullPath;
  }

  PathString TestDir;

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  FileManager FileMgr;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;

  FileID registerSource(llvm::StringRef Name, llvm::StringRef Text) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBufferCopy(Text);
    FileEntryRef File =
        FileMgr.getVirtualFileRef(Name, Buf->getBufferSize(), 0);
    SourceMgr.overrideFileContents(File, std::move(Buf));
    return SourceMgr.getOrCreateFileID(File, SrcMgr::C_User);
  }

  // Build a CharSourceRange spanning (BeginLine,BeginCol)..(EndLine,EndCol) in
  // FID. Guaranteed valid (character-granular, from an in-memory file we just
  // registered), so addLocations() will accept it.
  CharSourceRange makeRange(FileID FID, unsigned BeginLine, unsigned BeginCol,
                            unsigned EndLine, unsigned EndCol) {
    SourceLocation Begin = SourceMgr.translateLineCol(FID, BeginLine, BeginCol);
    SourceLocation End = SourceMgr.translateLineCol(FID, EndLine, EndCol);
    return CharSourceRange{SourceRange{Begin, End}, /*ITR=*/false};
  }

  // Read back a file and parse it as JSON.
  static llvm::Expected<llvm::json::Value> readJSON(llvm::StringRef Path) {
    auto BufOrErr = llvm::MemoryBuffer::getFile(Path);
    if (!BufOrErr)
      return llvm::errorCodeToError(BufOrErr.getError());
    return llvm::json::parse((*BufOrErr)->getBuffer());
  }

  // Recursively walk `V` and return true if any JSON object anywhere in the
  // tree has a key matching `Key`.
  static bool treeContainsKey(const llvm::json::Value &V, llvm::StringRef Key) {
    if (const auto *Obj = V.getAsObject()) {
      for (const auto &KV : *Obj) {
        if (llvm::StringRef(KV.first) == Key)
          return true;
        if (treeContainsKey(KV.second, Key))
          return true;
      }
      return false;
    }
    if (const auto *Arr = V.getAsArray()) {
      for (const auto &Item : *Arr)
        if (treeContainsKey(Item, Key))
          return true;
    }
    return false;
  }
};

TEST_F(SARIFFormatTest, ToolDriverNameAndVersion) {
  FileID FID = registerSource("/FakeFile.cpp", "int x;\nint y;\nint z;\n");
  CharSourceRange R = makeRange(FID, /*BL=*/1, /*BC=*/5, /*EL=*/1, /*EC=*/6);

  ReportDocument Doc{/*TransformationName=*/"MyTransform", SourceMgr, {}};

  Doc.Results.push_back({"rule", R, "msg"});

  PathString OutPath = makePath("report.sarif");
  auto Fmt = makeTransformationReportFormat("sarif");

  ASSERT_TRUE(Fmt);
  ASSERT_THAT_ERROR(Fmt->write(Doc, OutPath), llvm::Succeeded());

  auto ParsedOrErr = readJSON(OutPath);

  ASSERT_THAT_EXPECTED(ParsedOrErr, llvm::Succeeded());

  const llvm::json::Object *Root = ParsedOrErr->getAsObject();

  ASSERT_TRUE(Root);

  const llvm::json::Array *Runs = Root->getArray("runs");

  ASSERT_TRUE(Runs);
  ASSERT_EQ(Runs->size(), 1u);

  const llvm::json::Object *Run0 = (*Runs)[0].getAsObject();

  ASSERT_TRUE(Run0);

  const llvm::json::Object *Driver =
      Run0->getObject("tool")->getObject("driver");

  ASSERT_TRUE(Driver);

  auto Name = Driver->getString("name");

  ASSERT_TRUE(Name.has_value());
  EXPECT_EQ(*Name, "clang-ssaf");

  auto FullName = Driver->getString("fullName");

  ASSERT_TRUE(FullName);
  EXPECT_EQ(*FullName,
            "clang ScalableStaticAnalysisFramework source transformation "
            "(MyTransform)");

  auto Version = Driver->getString("version");

  ASSERT_TRUE(Version);
  EXPECT_EQ(*Version, CLANG_VERSION_STRING);
}

TEST_F(SARIFFormatTest, TestLevel) {
  FileID FID = registerSource("/FakeFile.cpp", "int a;\n");
  CharSourceRange R = makeRange(FID, /*BL=*/1, /*BC=*/5, /*EL=*/1, /*EC=*/6);

  ReportDocument Doc{"MyTransform", SourceMgr, {}};
  Doc.Results.push_back({"rule", R, "msg"});

  PathString OutPath = makePath("report.sarif");
  auto Fmt = makeTransformationReportFormat("sarif");

  ASSERT_TRUE(Fmt);
  ASSERT_THAT_ERROR(Fmt->write(Doc, OutPath), llvm::Succeeded());

  auto ParsedOrErr = readJSON(OutPath);

  ASSERT_THAT_EXPECTED(ParsedOrErr, llvm::Succeeded());
  const llvm::json::Array *Results = ParsedOrErr->getAsObject()
                                         ->getArray("runs")
                                         ->front()
                                         .getAsObject()
                                         ->getArray("results");

  ASSERT_TRUE(Results);
  ASSERT_EQ(Results->size(), 1u);
  EXPECT_EQ(*(*Results)[0].getAsObject()->getString("level"), "note");
}

//===----------------------------------------------------------------------===//
// Absent Range (std::nullopt) -> no `locations` key on that result object.
// Mix with a present range to assert the positive side too.
//===----------------------------------------------------------------------===//
TEST_F(SARIFFormatTest, InvalidRangeOmitsLocations) {
  FileID FID = registerSource("/FakeFile.cpp", "int alpha;\nint beta;\n");
  CharSourceRange Valid =
      makeRange(FID, /*BL=*/1, /*BC=*/5, /*EL=*/1, /*EC=*/10);

  ReportDocument Doc{"MyTransform", SourceMgr, {}};
  // Valid range entry first.
  Doc.Results.push_back({"rule-valid", Valid, "valid range"});
  // No-range entry.
  Doc.Results.push_back({"rule-invalid", std::nullopt, "no location"});

  PathString OutPath = makePath("report.sarif");
  auto Fmt = makeTransformationReportFormat("sarif");

  ASSERT_TRUE(Fmt);
  ASSERT_THAT_ERROR(Fmt->write(Doc, OutPath), llvm::Succeeded());

  auto ParsedOrErr = readJSON(OutPath);

  ASSERT_THAT_EXPECTED(ParsedOrErr, llvm::Succeeded());
  const llvm::json::Array *Results = ParsedOrErr->getAsObject()
                                         ->getArray("runs")
                                         ->front()
                                         .getAsObject()
                                         ->getArray("results");

  ASSERT_TRUE(Results);
  ASSERT_EQ(Results->size(), 2u);

  const llvm::json::Object *R0 = (*Results)[0].getAsObject();
  const llvm::json::Object *R1 = (*Results)[1].getAsObject();

  ASSERT_TRUE(R0);
  ASSERT_TRUE(R1);

  // Valid-range entry has a locations array.
  const llvm::json::Array *Locs0 = R0->getArray("locations");

  ASSERT_NE(Locs0, nullptr) << "valid-range result must have `locations`";
  EXPECT_FALSE(Locs0->empty());

  // Invalid-range entry has NO locations key.
  EXPECT_EQ(R1->get("locations"), nullptr)
      << "invalid-range result must not have `locations`";
}

//===----------------------------------------------------------------------===//
// Empty ruleId is forwarded verbatim. A matching rule entry is created.
//===----------------------------------------------------------------------===//
TEST_F(SARIFFormatTest, EmptyRuleIdAccepted) {
  FileID FID = registerSource("/FakeFile.cpp", "int q;\n");
  CharSourceRange R = makeRange(FID, /*BL=*/1, /*BC=*/5, /*EL=*/1, /*EC=*/6);

  ReportDocument Doc{"MyTransform", SourceMgr, {}};
  Doc.Results.push_back({/*RuleId=*/"", R, "run-wide finding"});

  PathString OutPath = makePath("report.sarif");
  auto Fmt = makeTransformationReportFormat("sarif");

  ASSERT_TRUE(Fmt);
  ASSERT_THAT_ERROR(Fmt->write(Doc, OutPath), llvm::Succeeded());

  auto ParsedOrErr = readJSON(OutPath);

  ASSERT_THAT_EXPECTED(ParsedOrErr, llvm::Succeeded());
  const llvm::json::Object *Run0 =
      ParsedOrErr->getAsObject()->getArray("runs")->front().getAsObject();

  ASSERT_TRUE(Run0);

  const llvm::json::Array *Results = Run0->getArray("results");

  ASSERT_TRUE(Results);
  ASSERT_EQ(Results->size(), 1u);
  auto RuleId = (*Results)[0].getAsObject()->getString("ruleId");

  ASSERT_TRUE(RuleId.has_value());
  EXPECT_EQ(*RuleId, "");

  // A rule entry with id "" exists in tool.driver.rules.
  const llvm::json::Array *Rules =
      Run0->getObject("tool")->getObject("driver")->getArray("rules");

  ASSERT_TRUE(Rules);
  ASSERT_EQ(Rules->size(), 1u);
  auto Id = (*Rules)[0].getAsObject()->getString("id");

  ASSERT_TRUE(Id.has_value());
  EXPECT_EQ(*Id, "");
}

//===----------------------------------------------------------------------===//
// No `fix` / `fixes` keys appear anywhere in the document. The framework
// intentionally keeps source edits out of the report, so even a regex walk
// across the file body should not find these substrings as JSON keys.
//===----------------------------------------------------------------------===//
TEST_F(SARIFFormatTest, NoFixKey) {
  FileID FID = registerSource("/FakeFile.cpp", "int p;\nint q;\n");

  ReportDocument Doc{"MyTransform", SourceMgr, {}};
  Doc.Results.push_back({"rule-1", std::nullopt, "m1"});
  Doc.Results.push_back({"rule-2", makeRange(FID, 1, 5, 1, 6), "m2"});
  Doc.Results.push_back({"rule-3", std::nullopt, "m3"});

  PathString OutPath = makePath("report.sarif");
  auto Fmt = makeTransformationReportFormat("sarif");

  ASSERT_TRUE(Fmt);
  ASSERT_THAT_ERROR(Fmt->write(Doc, OutPath), llvm::Succeeded());

  auto ParsedOrErr = readJSON(OutPath);

  ASSERT_THAT_EXPECTED(ParsedOrErr, llvm::Succeeded());

  EXPECT_FALSE(treeContainsKey(*ParsedOrErr, "fix"))
      << "SARIF report must not contain a `fix` key";

  EXPECT_FALSE(treeContainsKey(*ParsedOrErr, "fixes"))
      << "SARIF report must not contain a `fixes` key";
}

//===----------------------------------------------------------------------===//
// URIs are absolute file:// paths. Use a regex matcher on URI shape rather
// than hardcoding a path (tempdir varies across hosts / test runners).
//===----------------------------------------------------------------------===//
TEST_F(SARIFFormatTest, URIsAbsoluteFileScheme) {
  FileID FID = registerSource("/FakeFile.cpp", "int m;\nint n;\n");
  CharSourceRange Valid = makeRange(FID, 1, 5, 1, 6);

  ReportDocument Doc{"MyTransform", SourceMgr, {}};
  Doc.Results.push_back({"rule-valid", Valid, "finding"});

  PathString OutPath = makePath("report.sarif");
  auto Fmt = makeTransformationReportFormat("sarif");

  ASSERT_TRUE(Fmt);
  ASSERT_THAT_ERROR(Fmt->write(Doc, OutPath), llvm::Succeeded());

  auto ParsedOrErr = readJSON(OutPath);

  ASSERT_THAT_EXPECTED(ParsedOrErr, llvm::Succeeded());
  const llvm::json::Array *Results = ParsedOrErr->getAsObject()
                                         ->getArray("runs")
                                         ->front()
                                         .getAsObject()
                                         ->getArray("results");

  ASSERT_TRUE(Results);
  ASSERT_EQ(Results->size(), 1u);

  const llvm::json::Array *Locs =
      (*Results)[0].getAsObject()->getArray("locations");

  ASSERT_TRUE(Locs);
  ASSERT_EQ(Locs->size(), 1u);

  auto URI = (*Locs)[0]
                 .getAsObject()
                 ->getObject("physicalLocation")
                 ->getObject("artifactLocation")
                 ->getString("uri");

  ASSERT_TRUE(URI.has_value());

  llvm::Regex URIMatcher("^file://.*FakeFile\\.cpp$");
  std::string Err;

  ASSERT_TRUE(URIMatcher.isValid(Err)) << Err;
  EXPECT_TRUE(URIMatcher.match(*URI))
      << "URI did not match file://<path>FakeFile.cpp: " << *URI;
}

//===----------------------------------------------------------------------===//
// Zero results: still produces a well-formed SARIF doc with populated
// tool.driver and an empty runs[0].results array.
//===----------------------------------------------------------------------===//
TEST_F(SARIFFormatTest, EmptyResultsValidDocument) {
  registerSource("/FakeFile.cpp", "int k;\n");

  ReportDocument Doc{"MyTransform", SourceMgr, {}};
  // No results pushed.

  PathString OutPath = makePath("report.sarif");
  auto Fmt = makeTransformationReportFormat("sarif");

  ASSERT_TRUE(Fmt);
  ASSERT_THAT_ERROR(Fmt->write(Doc, OutPath), llvm::Succeeded());
  ASSERT_TRUE(llvm::sys::fs::exists(OutPath));

  auto ParsedOrErr = readJSON(OutPath);

  ASSERT_THAT_EXPECTED(ParsedOrErr, llvm::Succeeded());
  const llvm::json::Object *Root = ParsedOrErr->getAsObject();

  ASSERT_TRUE(Root);

  auto Schema = Root->getString("$schema");

  ASSERT_TRUE(Schema.has_value());
  EXPECT_NE(Schema->find("sarif-schema-2.1.0"), llvm::StringRef::npos);

  const llvm::json::Array *Runs = Root->getArray("runs");

  ASSERT_TRUE(Runs);
  ASSERT_EQ(Runs->size(), 1u);
  const llvm::json::Object *Run0 = (*Runs)[0].getAsObject();

  ASSERT_TRUE(Run0);

  // tool.driver still populated.
  const llvm::json::Object *Driver =
      Run0->getObject("tool")->getObject("driver");

  ASSERT_TRUE(Driver);
  ASSERT_TRUE(Driver->getString("name").has_value());
  EXPECT_EQ(*Driver->getString("name"), "clang-ssaf");

  // Results is present and empty.
  const llvm::json::Array *Results = Run0->getArray("results");

  ASSERT_TRUE(Results);
  EXPECT_TRUE(Results->empty());
}

} // namespace
