//===- TUSummaryTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for SSAF JSON serialization format reading and writing of
// TUSummary and TUSummaryEncoding.
//
//===----------------------------------------------------------------------===//

#include "JSONFormatTest.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"

#include <memory>

using namespace clang::ssaf;
using namespace llvm;
using ::testing::AllOf;
using ::testing::HasSubstr;

namespace {

// ============================================================================
// TUSummaryOps - Parameterization for TUSummary/TUSummaryEncoding tests
// ============================================================================

SummaryOps TUSummaryOps{
    "Resolved", "TUSummary",
    [](llvm::StringRef FilePath) -> llvm::Error {
      auto Result = JSONFormat().readTUSummary(FilePath);
      return Result ? llvm::Error::success() : Result.takeError();
    },
    [](llvm::StringRef FilePath) -> llvm::Error {
      BuildNamespace BN(BuildNamespaceKind::CompilationUnit, "test.cpp");
      TUSummary S(llvm::Triple("arm64-apple-macosx"), std::move(BN));
      return JSONFormat().writeTUSummary(S, FilePath);
    },
    [](llvm::StringRef InputFilePath,
       llvm::StringRef OutputFilePath) -> llvm::Error {
      auto ExpectedS = JSONFormat().readTUSummary(InputFilePath);
      if (!ExpectedS) {
        return ExpectedS.takeError();
      }
      return JSONFormat().writeTUSummary(*ExpectedS, OutputFilePath);
    }};

SummaryOps TUSummaryEncodingOps{
    "Encoding", "TUSummary",
    [](llvm::StringRef FilePath) -> llvm::Error {
      auto Result = JSONFormat().readTUSummaryEncoding(FilePath);
      return Result ? llvm::Error::success() : Result.takeError();
    },
    [](llvm::StringRef FilePath) -> llvm::Error {
      BuildNamespace BN(BuildNamespaceKind::CompilationUnit, "test.cpp");
      TUSummaryEncoding E(llvm::Triple("arm64-apple-macosx"), std::move(BN));
      return JSONFormat().writeTUSummaryEncoding(E, FilePath);
    },
    [](llvm::StringRef InputFilePath,
       llvm::StringRef OutputFilePath) -> llvm::Error {
      auto ExpectedE = JSONFormat().readTUSummaryEncoding(InputFilePath);
      if (!ExpectedE) {
        return ExpectedE.takeError();
      }
      return JSONFormat().writeTUSummaryEncoding(*ExpectedE, OutputFilePath);
    }};

// ============================================================================
// LUSummaryTest Test Fixture
// ============================================================================

class TUSummaryTest : public SummaryTest {};

INSTANTIATE_TEST_SUITE_P(JSONFormat, TUSummaryTest,
                         ::testing::Values(TUSummaryOps, TUSummaryEncodingOps),
                         [](const ::testing::TestParamInfo<SummaryOps> &Info) {
                           return Info.param.GTestInstantiationSuffix;
                         });

// ============================================================================
// JSONFormatTUSummaryTest Test Fixture
// ============================================================================

class JSONFormatTUSummaryTest : public JSONFormatTest {
protected:
  llvm::Expected<TUSummary> readTUSummaryFromFile(StringRef FileName) const {
    PathString FilePath = makePath(FileName);
    return JSONFormat().readTUSummary(FilePath);
  }

  llvm::Expected<TUSummary>
  readTUSummaryFromString(StringRef JSON,
                          StringRef FileName = "test.json") const {
    auto ExpectedFilePath = writeJSON(JSON, FileName);
    if (!ExpectedFilePath) {
      return ExpectedFilePath.takeError();
    }

    return readTUSummaryFromFile(FileName);
  }

  llvm::Error writeTUSummary(const TUSummary &Summary,
                             StringRef FileName) const {
    PathString FilePath = makePath(FileName);
    return JSONFormat().writeTUSummary(Summary, FilePath);
  }
};

// ============================================================================
// readJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, NonexistentFile) {
  auto Result = readFromFile("nonexistent.json");

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                                            HasSubstr("file does not exist"))));
}

TEST_P(TUSummaryTest, PathIsDirectory) {
  PathString DirName("test_directory.json");

  auto ExpectedDirPath = makeDirectory(DirName);
  ASSERT_THAT_EXPECTED(ExpectedDirPath, Succeeded());

  auto Result = readFromFile(DirName);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                              HasSubstr("path is a directory, not a file"))));
}

TEST_P(TUSummaryTest, NotJsonExtension) {
  PathString FileName("test.txt");

  auto ExpectedFilePath = writeJSON("{}", FileName);
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto Result = readFromFile(FileName);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("failed to read file"),
                HasSubstr("file does not end with '.json' extension"))));
}

TEST_P(TUSummaryTest, BrokenSymlink) {
#ifdef _WIN32
  GTEST_SKIP() << "Symlink model differs on Windows";
#endif

  const PathString SymlinkFileName("broken_symlink.json");

  // Create a symlink pointing to a non-existent file
  auto ExpectedSymlinkPath =
      makeSymlink("nonexistent_target.json", SymlinkFileName);
  ASSERT_THAT_EXPECTED(ExpectedSymlinkPath, Succeeded());

  auto Result = readFromFile(SymlinkFileName);

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                                            HasSubstr("failed to read file"))));
}

TEST_P(TUSummaryTest, NoReadPermission) {
  if (!permissionsAreEnforced()) {
    GTEST_SKIP() << "File permission checks are not enforced in this "
                    "environment";
  }

  PathString FileName("no-read-permission.json");

  auto ExpectedFilePath = writeJSON(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })",
                                    FileName);
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto PermError = setPermission(FileName, sys::fs::perms::owner_write |
                                               sys::fs::perms::owner_exe);
  ASSERT_THAT_ERROR(std::move(PermError), Succeeded());

  auto Result = readFromFile(FileName);

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                                            HasSubstr("failed to read file"))));

  // Restore permissions for cleanup
  auto RestoreError = setPermission(FileName, sys::fs::perms::all_all);
  EXPECT_THAT_ERROR(std::move(RestoreError), Succeeded());
}

// ============================================================================
// JSONFormat::entityDataMapEntryFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, ReadEntitySummaryMissingData) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "target_triple": "arm64-apple-macosx",
    "type": "TUSummary",
    "id_table": [],
    "linkage_table": [],
    "data": [
      {
        "summary_name": "NullEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {}
          }
        ]
      }
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to deserialize EntitySummary"),
          HasSubstr("null EntitySummary data for "
                    "'SummaryName(NullEntitySummaryForJSONFormatTest)'"))));
}

TEST_F(JSONFormatTUSummaryTest, ReadEntitySummaryMismatchedSummaryName) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "target_triple": "arm64-apple-macosx",
    "type": "TUSummary",
    "id_table": [],
    "linkage_table": [],
    "data": [
      {
        "summary_name": "MismatchedEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {}
          }
        ]
      }
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to deserialize EntitySummary"),
          HasSubstr("EntitySummary data for "
                    "'SummaryName(MismatchedEntitySummaryForJSONFormatTest)' "
                    "reports "
                    "mismatched "
                    "'SummaryName(MismatchedEntitySummaryForJSONFormatTest_"
                    "WrongName)"
                    "'"))));
}

// ============================================================================
// JSONFormat::entityDataMapEntryToJSON() Fatal Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, WriteEntitySummaryMissingData) {
  TUSummary Summary(
      llvm::Triple("arm64-apple-macosx"),
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));

  NestedBuildNamespace Namespace =
      NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EntityId EI = getIdTable(Summary).getId(
      EntityName{"c:@F@foo", "", std::move(Namespace)});

  SummaryName SN("NullEntitySummaryForJSONFormatTest");
  getData(Summary)[SN][EI] = nullptr;

  EXPECT_DEATH(
      { (void)writeTUSummary(Summary, "output.json"); },
      "JSONFormat - null EntitySummary data for "
      "'SummaryName\\(NullEntitySummaryForJSONFormatTest\\)'");
}

TEST_F(JSONFormatTUSummaryTest, WriteEntitySummaryMismatchedSummaryName) {
  TUSummary Summary(
      llvm::Triple("arm64-apple-macosx"),
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));

  NestedBuildNamespace Namespace =
      NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EntityId EI = getIdTable(Summary).getId(
      EntityName{"c:@F@foo", "", std::move(Namespace)});

  SummaryName SN("MismatchedEntitySummaryForJSONFormatTest");
  getData(Summary)[SN][EI] =
      std::make_unique<MismatchedEntitySummaryForJSONFormatTest>();

  EXPECT_DEATH(
      { (void)writeTUSummary(Summary, "output.json"); },
      "JSONFormat - EntitySummary data for "
      "'SummaryName\\(MismatchedEntitySummaryForJSONFormatTest\\)' reports "
      "mismatched "
      "'SummaryName\\(MismatchedEntitySummaryForJSONFormatTest_WrongName\\)"
      "'");
}

// ============================================================================
// JSONFormat::writeJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, WriteFileAlreadyExists) {
  PathString FileName("existing.json");

  auto ExpectedFilePath = writeJSON("{}", FileName);
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto Result = writeEmpty(FileName);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("file already exists"))));
}

TEST_P(TUSummaryTest, WriteParentDirectoryNotFound) {
  PathString FilePath = makePath("nonexistent-dir", "test.json");

  auto Result = GetParam().WriteEmpty(FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("parent directory does not exist"))));
}

TEST_P(TUSummaryTest, WriteNotJsonExtension) {
  auto Result = writeEmpty("test.txt");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("writing TUSummary to file"),
                HasSubstr("failed to write file"),
                HasSubstr("file does not end with '.json' extension"))));
}

TEST_P(TUSummaryTest, WriteStreamOpenFailure) {
  if (!permissionsAreEnforced()) {
    GTEST_SKIP() << "File permission checks are not enforced in this "
                    "environment";
  }

  const PathString DirName("write-protected-dir");

  auto ExpectedDirPath = makeDirectory(DirName);
  ASSERT_THAT_EXPECTED(ExpectedDirPath, Succeeded());

  auto PermError = setPermission(DirName, sys::fs::perms::owner_read |
                                              sys::fs::perms::owner_exe);
  ASSERT_THAT_ERROR(std::move(PermError), Succeeded());

  PathString FilePath = makePath(DirName, "test.json");

  auto Result = GetParam().WriteEmpty(FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"))));

  // Restore permissions for cleanup
  auto RestoreError = setPermission(DirName, sys::fs::perms::all_all);
  EXPECT_THAT_ERROR(std::move(RestoreError), Succeeded());
}

// ============================================================================
// JSONFormat::writeTUSummary() Error Tests (TUSummary-only)
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, WriteEntitySummaryNoFormatInfo) {
  TUSummary Summary(
      llvm::Triple("arm64-apple-macosx"),
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));

  NestedBuildNamespace Namespace =
      NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EntityId EI = getIdTable(Summary).getId(
      EntityName{"c:@F@foo", "", std::move(Namespace)});

  SummaryName UnknownSN("UnregisteredEntitySummaryForJSONFormatTest");
  getData(Summary)[UnknownSN][EI] =
      std::make_unique<UnregisteredEntitySummaryForJSONFormatTest>();

  auto Result = writeTUSummary(Summary, "output.json");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("writing TUSummary to file"),
          HasSubstr("writing SummaryData entry to index '0'"),
          HasSubstr("writing EntitySummary entries to field "
                    "'summary_data'"),
          HasSubstr("writing EntitySummary entry to index '0'"),
          HasSubstr("writing EntitySummary to field 'entity_summary'"),
          HasSubstr("failed to serialize EntitySummary"),
          HasSubstr(
              "no FormatInfo registered for "
              "'SummaryName(UnregisteredEntitySummaryForJSONFormatTest)'"))));
}

} // anonymous namespace
