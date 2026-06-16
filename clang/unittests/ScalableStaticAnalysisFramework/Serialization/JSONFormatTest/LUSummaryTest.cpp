//===- LUSummaryTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for SSAF JSON serialization format reading and writing of
// LUSummary and LUSummaryEncoding.
//
//===----------------------------------------------------------------------===//

#include "JSONFormatTest.h"

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"

#include <memory>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;
using ::testing::AllOf;
using ::testing::HasSubstr;

namespace {

// ============================================================================
// LUSummaryOps - Parameterization for LUSummary/LUSummaryEncoding tests
// ============================================================================

SummaryOps LUSummaryOps{
    "Resolved", "LUSummary",
    [](llvm::StringRef FilePath) -> llvm::Error {
      auto Result = JSONFormat().readLUSummary(FilePath);
      return Result ? llvm::Error::success() : Result.takeError();
    },
    [](llvm::StringRef FilePath) -> llvm::Error {
      BuildNamespace BN(BuildNamespaceKind::CompilationUnit, "test.cpp");
      NestedBuildNamespace NBN(std::move(BN));
      LUSummary S(std::move(NBN));
      return JSONFormat().writeLUSummary(S, FilePath);
    },
    [](llvm::StringRef InputFilePath,
       llvm::StringRef OutputFilePath) -> llvm::Error {
      auto ExpectedS = JSONFormat().readLUSummary(InputFilePath);
      if (!ExpectedS) {
        return ExpectedS.takeError();
      }
      return JSONFormat().writeLUSummary(*ExpectedS, OutputFilePath);
    }};

SummaryOps LUSummaryEncodingOps{
    "Encoding", "LUSummary",
    [](llvm::StringRef FilePath) -> llvm::Error {
      auto Result = JSONFormat().readLUSummaryEncoding(FilePath);
      return Result ? llvm::Error::success() : Result.takeError();
    },
    [](llvm::StringRef FilePath) -> llvm::Error {
      BuildNamespace BN(BuildNamespaceKind::CompilationUnit, "test.cpp");
      NestedBuildNamespace NBN(std::move(BN));
      LUSummaryEncoding E(std::move(NBN));
      return JSONFormat().writeLUSummaryEncoding(E, FilePath);
    },
    [](llvm::StringRef InputFilePath,
       llvm::StringRef OutputFilePath) -> llvm::Error {
      auto ExpectedE = JSONFormat().readLUSummaryEncoding(InputFilePath);
      if (!ExpectedE) {
        return ExpectedE.takeError();
      }
      return JSONFormat().writeLUSummaryEncoding(*ExpectedE, OutputFilePath);
    }};

// ============================================================================
// LUSummaryTest Test Fixture
// ============================================================================

class LUSummaryTest : public SummaryTest {};

INSTANTIATE_TEST_SUITE_P(JSONFormat, LUSummaryTest,
                         ::testing::Values(LUSummaryOps, LUSummaryEncodingOps),
                         [](const ::testing::TestParamInfo<SummaryOps> &Info) {
                           return Info.param.GTestInstantiationSuffix;
                         });

// ============================================================================
// JSONFormatLUSummaryTest Test Fixture
// ============================================================================

class JSONFormatLUSummaryTest : public JSONFormatTest {
protected:
  llvm::Expected<LUSummary> readLUSummaryFromFile(StringRef FileName) const {
    PathString FilePath = makePath(FileName);
    return JSONFormat().readLUSummary(FilePath);
  }

  llvm::Expected<LUSummary>
  readLUSummaryFromString(StringRef JSON,
                          StringRef FileName = "test.json") const {
    auto ExpectedFilePath = writeJSON(JSON, FileName);
    if (!ExpectedFilePath) {
      return ExpectedFilePath.takeError();
    }

    return readLUSummaryFromFile(FileName);
  }

  llvm::Error writeLUSummary(const LUSummary &Summary,
                             StringRef FileName) const {
    PathString FilePath = makePath(FileName);
    return JSONFormat().writeLUSummary(Summary, FilePath);
  }
};

// ============================================================================
// readJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, NonexistentFile) {
  auto Result = readFromFile("nonexistent.json");

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(HasSubstr("reading LUSummary from"),
                                            HasSubstr("file does not exist"))));
}

TEST_P(LUSummaryTest, PathIsDirectory) {
  PathString DirName("test_directory.json");

  auto ExpectedDirPath = makeDirectory(DirName);
  ASSERT_THAT_EXPECTED(ExpectedDirPath, Succeeded());

  auto Result = readFromFile(DirName);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("reading LUSummary from"),
                              HasSubstr("path is a directory, not a file"))));
}

TEST_P(LUSummaryTest, NotJsonExtension) {
  PathString FileName("test.txt");

  auto ExpectedFilePath = writeJSON("{}", FileName);
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto Result = readFromFile(FileName);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("failed to read file"),
                HasSubstr("file does not end with '.json' extension"))));
}

TEST_P(LUSummaryTest, BrokenSymlink) {
#ifdef _WIN32
  GTEST_SKIP() << "Symlink model differs on Windows";
#endif

  const PathString SymlinkFileName("broken_symlink.json");

  // Create a symlink pointing to a non-existent file
  auto ExpectedSymlinkPath =
      makeSymlink("nonexistent_target.json", "broken_symlink.json");
  ASSERT_THAT_EXPECTED(ExpectedSymlinkPath, Succeeded());

  auto Result = readFromFile(SymlinkFileName);

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(HasSubstr("reading LUSummary from"),
                                            HasSubstr("failed to read file"))));
}

TEST_P(LUSummaryTest, NoReadPermission) {
  if (!permissionsAreEnforced()) {
    GTEST_SKIP() << "File permission checks are not enforced in this "
                    "environment";
  }

  PathString FileName("no-read-permission.json");

  auto ExpectedFilePath = writeJSON(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
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
                    FailedWithMessage(AllOf(HasSubstr("reading LUSummary from"),
                                            HasSubstr("failed to read file"))));

  // Restore permissions for cleanup
  auto RestoreError = setPermission(FileName, sys::fs::perms::all_all);
  EXPECT_THAT_ERROR(std::move(RestoreError), Succeeded());
}

// ============================================================================
// JSONFormat::entityDataMapFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatLUSummaryTest, ReadEntitySummaryMissingData) {
  auto Result = readLUSummaryFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
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
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to deserialize EntitySummary"),
          HasSubstr("null EntitySummary data for "
                    "'SummaryName(NullEntitySummaryForJSONFormatTest)'"))));
}

TEST_F(JSONFormatLUSummaryTest, ReadEntitySummaryMismatchedSummaryName) {
  auto Result = readLUSummaryFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
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
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to deserialize EntitySummary"),
          HasSubstr("EntitySummary data for "
                    "'SummaryName(MismatchedEntitySummaryForJSONFormatTest)'"
                    " reports mismatched "
                    "'SummaryName(MismatchedEntitySummaryForJSONFormatTest_"
                    "WrongName)'"))));
}

// ============================================================================
// JSONFormat::entityDataMapEntryToJSON() Fatal Tests
// ============================================================================

TEST_F(JSONFormatLUSummaryTest, WriteEntitySummaryMissingData) {
  NestedBuildNamespace NBN(
      BuildNamespace(BuildNamespaceKind::LinkUnit, "test.exe"));
  LUSummary Summary(std::move(NBN));

  NestedBuildNamespace Namespace =
      NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EntityId EI = getIdTable(Summary).getId(
      EntityName{"c:@F@foo", "", std::move(Namespace)});

  SummaryName SN("NullEntitySummaryForJSONFormatTest");
  getData(Summary)[SN][EI] = nullptr;

  EXPECT_DEATH(
      { (void)writeLUSummary(Summary, "output.json"); },
      "JSONFormat - null EntitySummary data for "
      "'SummaryName\\(NullEntitySummaryForJSONFormatTest\\)'");
}

TEST_F(JSONFormatLUSummaryTest, WriteEntitySummaryMismatchedSummaryName) {
  NestedBuildNamespace NBN(
      BuildNamespace(BuildNamespaceKind::LinkUnit, "test.exe"));
  LUSummary Summary(std::move(NBN));

  NestedBuildNamespace Namespace =
      NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EntityId EI = getIdTable(Summary).getId(
      EntityName{"c:@F@foo", "", std::move(Namespace)});

  SummaryName SN("MismatchedEntitySummaryForJSONFormatTest");
  getData(Summary)[SN][EI] =
      std::make_unique<MismatchedEntitySummaryForJSONFormatTest>();

  EXPECT_DEATH(
      { (void)writeLUSummary(Summary, "output.json"); },
      "JSONFormat - EntitySummary data for "
      "'SummaryName\\(MismatchedEntitySummaryForJSONFormatTest\\)' "
      "reports "
      "mismatched "
      "'SummaryName\\(MismatchedEntitySummaryForJSONFormatTest_WrongName\\)'");
}

// ============================================================================
// JSONFormat::writeJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, WriteFileAlreadyExists) {
  PathString FileName("existing.json");

  auto ExpectedFilePath = writeJSON("{}", FileName);
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto Result = writeEmpty(FileName);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing LUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("file already exists"))));
}

TEST_P(LUSummaryTest, WriteParentDirectoryNotFound) {
  PathString FilePath = makePath("nonexistent-dir", "test.json");

  auto Result = GetParam().WriteEmpty(FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing LUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("parent directory does not exist"))));
}

TEST_P(LUSummaryTest, WriteNotJsonExtension) {
  auto Result = writeEmpty("test.txt");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("writing LUSummary to file"),
                HasSubstr("failed to write file"),
                HasSubstr("file does not end with '.json' extension"))));
}

TEST_P(LUSummaryTest, WriteStreamOpenFailure) {
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
      FailedWithMessage(AllOf(HasSubstr("writing LUSummary to file"),
                              HasSubstr("failed to write file"))));

  // Restore permissions for cleanup
  auto RestoreError = setPermission(DirName, sys::fs::perms::all_all);
  EXPECT_THAT_ERROR(std::move(RestoreError), Succeeded());
}

// ============================================================================
// JSONFormat::writeLUSummary() Error Tests (LUSummary-only)
// ============================================================================

TEST_F(JSONFormatLUSummaryTest, WriteEntitySummaryNoFormatInfo) {
  NestedBuildNamespace NBN(
      BuildNamespace(BuildNamespaceKind::LinkUnit, "test.exe"));
  LUSummary Summary(std::move(NBN));

  NestedBuildNamespace Namespace =
      NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EntityId EI = getIdTable(Summary).getId(
      EntityName{"c:@F@foo", "", std::move(Namespace)});

  SummaryName UnknownSN("UnregisteredEntitySummaryForJSONFormatTest");
  getData(Summary)[UnknownSN][EI] =
      std::make_unique<UnregisteredEntitySummaryForJSONFormatTest>();

  auto Result = writeLUSummary(Summary, "output.json");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("writing LUSummary to file"),
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
