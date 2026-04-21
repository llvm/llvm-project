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

TEST_P(LUSummaryTest, InvalidSyntax) {
  auto Result = readFromString("{ invalid json }");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("reading LUSummary from file"),
                              HasSubstr("Expected object key"))));
}

TEST_P(LUSummaryTest, NotObject) {
  auto Result = readFromString("[]");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("reading LUSummary from file"),
                              HasSubstr("failed to read LUSummary"),
                              HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::entityLinkageFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, LinkageTableEntryLinkageMissingType) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [
      {
        "id": 0,
        "linkage": {}
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("reading EntityLinkage from field 'linkage'"),
                HasSubstr("failed to read EntityLinkageType from field 'type'"),
                HasSubstr("expected JSON string"))));
}

TEST_P(LUSummaryTest, LinkageTableEntryLinkageInvalidType) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "invalid_type" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("reading EntityLinkage from field 'linkage'"),
                HasSubstr("invalid EntityLinkageType value 'invalid_type' for "
                          "field 'type'"))));
}

// ============================================================================
// JSONFormat::linkageTableEntryFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, LinkageTableEntryMissingId) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [
      {
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(LUSummaryTest, LinkageTableEntryIdNotUInt64) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [
      {
        "id": "not_a_number",
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(LUSummaryTest, LinkageTableEntryMissingLinkage) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [
      {
        "id": 0
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityLinkage from field 'linkage'"),
                HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::linkageTableFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, LinkageTableNotArray) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": {},
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("failed to read LinkageTable from field 'linkage_table'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(LUSummaryTest, LinkageTableElementNotObject) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": ["invalid"],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("failed to read LinkageTable entry from index '0'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, LinkageTableExtraId) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to deserialize LinkageTable"),
                HasSubstr("extra 'EntityId(0)' not present in IdTable"))));
}

TEST_P(LUSummaryTest, LinkageTableMissingId) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("failed to deserialize LinkageTable"),
                HasSubstr("missing 'EntityId(0)' present in IdTable"))));
}

TEST_P(LUSummaryTest, LinkageTableDuplicateId) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "External" }
      },
      {
        "id": 0,
        "linkage": { "type": "Internal" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("failed to insert LinkageTable entry at index '1'"),
                HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::nestedBuildNamespaceFromJSON() Error Tests (lu_namespace field)
// ============================================================================

TEST_P(LUSummaryTest, LUNamespaceNotArray) {
  auto Result = readFromString(R"({
    "lu_namespace": { "kind": "CompilationUnit", "name": "test.cpp" },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr(
              "failed to read NestedBuildNamespace from field 'lu_namespace'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(LUSummaryTest, LUNamespaceElementNotObject) {
  auto Result = readFromString(R"({
    "lu_namespace": ["invalid"],
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading NestedBuildNamespace from field 'lu_namespace'"),
          HasSubstr("failed to read BuildNamespace from index '0'"),
          HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, LUNamespaceElementMissingKind) {
  auto Result = readFromString(R"({
    "lu_namespace": [{ "name": "test.cpp" }],
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading NestedBuildNamespace from field 'lu_namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("failed to read BuildNamespaceKind from field 'kind'"),
          HasSubstr("expected JSON string"))));
}

TEST_P(LUSummaryTest, LUNamespaceElementInvalidKind) {
  auto Result = readFromString(R"({
    "lu_namespace": [{ "kind": "invalid_kind", "name": "test.cpp" }],
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading NestedBuildNamespace from field 'lu_namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("reading BuildNamespaceKind from field 'kind'"),
          HasSubstr("invalid BuildNamespaceKind value 'invalid_kind' for "
                    "field 'kind'"))));
}

TEST_P(LUSummaryTest, LUNamespaceElementMissingName) {
  auto Result = readFromString(R"({
    "lu_namespace": [{ "kind": "CompilationUnit" }],
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading NestedBuildNamespace from field 'lu_namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("failed to read BuildNamespaceName from field 'name'"),
          HasSubstr("expected JSON string"))));
}

// ============================================================================
// JSONFormat::nestedBuildNamespaceFromJSON() Error Tests (EntityName namespace)
// ============================================================================

TEST_P(LUSummaryTest, NamespaceElementNotObject) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": ["invalid"]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "None" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("failed to read BuildNamespace from index '0'"),
          HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, NamespaceElementMissingKind) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "Internal" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("failed to read BuildNamespaceKind from field 'kind'"),
          HasSubstr("expected JSON string"))));
}

TEST_P(LUSummaryTest, NamespaceElementInvalidKind) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "invalid_kind",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("reading BuildNamespaceKind from field 'kind'"),
          HasSubstr("invalid BuildNamespaceKind value 'invalid_kind' for "
                    "field 'kind'"))));
}

TEST_P(LUSummaryTest, NamespaceElementMissingName) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "None" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("failed to read BuildNamespaceName from field 'name'"),
          HasSubstr("expected JSON string"))));
}

// ============================================================================
// JSONFormat::entityNameFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, EntityNameMissingUSR) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "None" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(
                        HasSubstr("reading LUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("reading EntityName from field 'name'"),
                        HasSubstr("failed to read USR from field 'usr'"),
                        HasSubstr("expected JSON string"))));
}

TEST_P(LUSummaryTest, EntityNameMissingSuffix) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "namespace": [
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(
                        HasSubstr("reading LUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("reading EntityName from field 'name'"),
                        HasSubstr("failed to read Suffix from field 'suffix'"),
                        HasSubstr("expected JSON string"))));
}

TEST_P(LUSummaryTest, EntityNameMissingNamespace) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "None" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("reading EntityName from field 'name'"),
                HasSubstr("failed to read NestedBuildNamespace from field "
                          "'namespace'"),
                HasSubstr("expected JSON array"))));
}

// ============================================================================
// JSONFormat::entityIdTableEntryFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, IDTableEntryMissingID) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(LUSummaryTest, IDTableEntryMissingName) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityName from field 'name'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, IDTableEntryIDNotUInt64) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": "not_a_number",
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

// ============================================================================
// JSONFormat::entityIdTableFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, IDTableNotArray) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": {},
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("failed to read IdTable from field 'id_table'"),
                HasSubstr("expected JSON array"))));
}

TEST_P(LUSummaryTest, IDTableElementNotObject) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": ["invalid"],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("failed to read EntityIdTable entry from index '0'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, DuplicateEntity) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      { "id": 0, "linkage": { "type": "None" } },
      { "id": 1, "linkage": { "type": "None" } }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("failed to insert EntityIdTable entry at index '1'"),
                HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::readLUSummary() Error Tests (LUSummary-only)
// ============================================================================

TEST_F(JSONFormatLUSummaryTest, ReadEntitySummaryNoFormatInfo) {
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
        "summary_name": "UnregisteredEntitySummaryForJSONFormatTest",
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("failed to deserialize EntitySummary"),
          HasSubstr(
              "no FormatInfo registered for "
              "'SummaryName(UnregisteredEntitySummaryForJSONFormatTest)'"))));
}

// ============================================================================
// PairsEntitySummaryForJSONFormatTest Deserialization Error Tests
// ============================================================================

TEST_F(JSONFormatLUSummaryTest,
       PairsEntitySummaryForJSONFormatTestMissingPairsField) {
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid field 'pairs'"))));
}

TEST_F(JSONFormatLUSummaryTest,
       PairsEntitySummaryForJSONFormatTestInvalidPairsFieldType) {
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": "not_an_array"
            }
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid field 'pairs'"))));
}

TEST_F(JSONFormatLUSummaryTest,
       PairsEntitySummaryForJSONFormatTestPairsElementNotObject) {
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": ["not_an_object"]
            }
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("pairs element at index 0 is not a JSON object"))));
}

TEST_F(JSONFormatLUSummaryTest,
       PairsEntitySummaryForJSONFormatTestMissingFirstField) {
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": [
                {
                  "second": {"@": 1}
                }
              ]
            }
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'first' field at index '0'"))));
}

TEST_F(JSONFormatLUSummaryTest,
       PairsEntitySummaryForJSONFormatTestInvalidFirstField) {
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": [
                {
                  "first": "not_a_number",
                  "second": {"@": 1}
                }
              ]
            }
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'first' field at index '0'"))));
}

TEST_F(JSONFormatLUSummaryTest,
       PairsEntitySummaryForJSONFormatTestMissingSecondField) {
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": [
                {
                  "first": {"@": 0}
                }
              ]
            }
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'second' field at index '0'"))));
}

TEST_F(JSONFormatLUSummaryTest,
       PairsEntitySummaryForJSONFormatTestInvalidSecondField) {
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": [
                {
                  "first": {"@": 0},
                  "second": "not_a_number"
                }
              ]
            }
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
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'second' field at index '0'"))));
}

// ============================================================================
// JSONFormat::entityDataMapFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, EntityDataMissingEntityID) {
  auto Result = readFromString(R"({
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_summary": {}
          }
        ]
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to read EntityId from field 'entity_id'"),
          HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(LUSummaryTest, EntityDataMissingEntitySummary) {
  auto Result = readFromString(R"({
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0
          }
        ]
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to read EntitySummary from field 'entity_summary'"),
          HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, EntityIDNotUInt64) {
  auto Result = readFromString(R"({
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": "not_a_number",
            "entity_summary": {}
          }
        ]
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to read EntityId from field 'entity_id'"),
          HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

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
// JSONFormat::entityDataMapFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, EntityDataElementNotObject) {
  auto Result = readFromString(R"({
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": ["invalid"]
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("failed to read EntitySummary entry from index '0'"),
          HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, DuplicateEntityIdInDataMap) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            },
            {
              "kind": "LinkUnit",
              "name": "test.exe"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": {
          "type": "None"
        }
      }
    ],
    "data": [
      {
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": { "pairs": [] }
          },
          {
            "entity_id": 0,
            "entity_summary": { "pairs": [] }
          }
        ]
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("failed to insert EntitySummary entry at index '1'"),
          HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::summaryDataMapFromJSON() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, DataEntryMissingSummaryName) {
  auto Result = readFromString(R"({
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
        "summary_data": []
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("failed to read SummaryName from field 'summary_name'"),
          HasSubstr("expected JSON string"))));
}

TEST_P(LUSummaryTest, DataEntryMissingData) {
  auto Result = readFromString(R"({
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest"
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading SummaryData entries from field 'data'"),
                HasSubstr("reading SummaryData entry from index '0'"),
                HasSubstr("failed to read EntitySummary entries from field "
                          "'summary_data'"),
                HasSubstr("expected JSON array"))));
}

// ============================================================================
// JSONFormat::summaryDataMapFromJSON() / encodingSummaryDataMapFromJSON() Tests
// ============================================================================

TEST_P(LUSummaryTest, DataNotArray) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [],
    "data": {}
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("failed to read SummaryData entries from field 'data'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(LUSummaryTest, DataElementNotObject) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": [],
    "data": ["invalid"]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("reading SummaryData entries from field 'data'"),
                HasSubstr("failed to read SummaryData entry from index '0'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(LUSummaryTest, DuplicateSummaryName) {
  auto Result = readFromString(R"({
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": []
      },
      {
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": []
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("failed to insert SummaryData entry at index '1'"),
          HasSubstr("encountered duplicate "
                    "'SummaryName(PairsEntitySummaryForJSONFormatTest)'"))));
}

// ============================================================================
// JSONFormat::readLUSummary() / readLUSummaryEncoding() Error Tests
// ============================================================================

TEST_P(LUSummaryTest, MissingLUNamespace) {
  auto Result = readFromString(R"({
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr(
              "failed to read NestedBuildNamespace from field 'lu_namespace'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(LUSummaryTest, MissingIDTable) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading LUSummary from file"),
                HasSubstr("failed to read IdTable from field 'id_table'"),
                HasSubstr("expected JSON array"))));
}

TEST_P(LUSummaryTest, MissingLinkageTable) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("failed to read LinkageTable from field 'linkage_table'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(LUSummaryTest, MissingData) {
  auto Result = readFromString(R"({
    "lu_namespace": [
      {
        "kind": "LinkUnit",
        "name": "test.exe"
      }
    ],
    "id_table": [],
    "linkage_table": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading LUSummary from file"),
          HasSubstr("failed to read SummaryData entries from field 'data'"),
          HasSubstr("expected JSON array"))));
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

// ============================================================================
// Round-Trip Tests - Serialization Verification
// ============================================================================

TEST_P(LUSummaryTest, RoundTripEmptyNamespace) {
  readWriteCompare(R"({
    "lu_namespace": [
      {
        "kind": "CompilationUnit",
        "name": "test.cpp"
      }
    ],
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");
}

TEST_P(LUSummaryTest, RoundTripSingleNamespaceElement) {
  readWriteCompare(R"({
    "lu_namespace": [
      { "kind": "CompilationUnit", "name": "test.cpp" }
    ],
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");
}

TEST_P(LUSummaryTest, RoundTripMultipleNamespaceElements) {
  readWriteCompare(R"({
    "lu_namespace": [
      { "kind": "CompilationUnit", "name": "a.cpp" },
      { "kind": "CompilationUnit", "name": "b.cpp" },
      { "kind": "LinkUnit", "name": "libtest.so" }
    ],
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");
}

TEST_P(LUSummaryTest, RoundTripWithTwoSummaryTypes) {
  readWriteCompare(R"({
    "lu_namespace": [
      {
        "kind": "CompilationUnit",
        "name": "test.cpp"
      }
    ],
    "id_table": [
      {
        "id": 3,
        "name": {
          "usr": "c:@F@qux",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@bar",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 4,
        "name": {
          "usr": "c:@F@quux",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 2,
        "name": {
          "usr": "c:@F@baz",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 3,
        "linkage": { "type": "Internal" }
      },
      {
        "id": 1,
        "linkage": { "type": "None" }
      },
      {
        "id": 4,
        "linkage": { "type": "External" }
      },
      {
        "id": 0,
        "linkage": { "type": "None" }
      },
      {
        "id": 2,
        "linkage": { "type": "Internal" }
      }
    ],
    "data": [
      {
        "summary_name": "TagsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 4,
            "entity_summary": { "tags": ["exported", "hot"] }
          },
          {
            "entity_id": 1,
            "entity_summary": { "tags": ["internal-only"] }
          },
          {
            "entity_id": 3,
            "entity_summary": { "tags": ["internal-only"] }
          },
          {
            "entity_id": 0,
            "entity_summary": { "tags": ["entry-point"] }
          },
          {
            "entity_id": 2,
            "entity_summary": { "tags": [] }
          }
        ]
      },
      {
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": [
          {
            "entity_id": 1,
            "entity_summary": {
              "pairs": [
                { "first": {"@": 1}, "second": {"@": 3} }
              ]
            }
          },
          {
            "entity_id": 4,
            "entity_summary": {
              "pairs": [
                { "first": {"@": 4}, "second": {"@": 0} },
                { "first": {"@": 4}, "second": {"@": 2} }
              ]
            }
          },
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": []
            }
          },
          {
            "entity_id": 3,
            "entity_summary": {
              "pairs": [
                { "first": {"@": 3}, "second": {"@": 1} }
              ]
            }
          },
          {
            "entity_id": 2,
            "entity_summary": {
              "pairs": [
                { "first": {"@": 2}, "second": {"@": 4} },
                { "first": {"@": 2}, "second": {"@": 3} }
              ]
            }
          }
        ]
      }
    ]
  })");
}

TEST_P(LUSummaryTest, RoundTripWithEmptyDataEntry) {
  readWriteCompare(R"({
    "lu_namespace": [
      { "kind": "CompilationUnit", "name": "test.cpp" }
    ],
    "id_table": [],
    "linkage_table": [],
    "data": [
      {
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": []
      }
    ]
  })");
}

TEST_P(LUSummaryTest, RoundTripLinkageTableWithNoneLinkage) {
  readWriteCompare(R"({
    "lu_namespace": [
      { "kind": "CompilationUnit", "name": "test.cpp" }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "None" }
      }
    ],
    "data": []
  })");
}

TEST_P(LUSummaryTest, RoundTripLinkageTableWithInternalLinkage) {
  readWriteCompare(R"({
    "lu_namespace": [
      { "kind": "CompilationUnit", "name": "test.cpp" }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@bar",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "Internal" }
      }
    ],
    "data": []
  })");
}

TEST_P(LUSummaryTest, RoundTripLinkageTableWithExternalLinkage) {
  readWriteCompare(R"({
    "lu_namespace": [
      { "kind": "CompilationUnit", "name": "test.cpp" }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@baz",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");
}

TEST_P(LUSummaryTest, RoundTripLinkageTableWithMultipleEntries) {
  readWriteCompare(R"({
    "lu_namespace": [
      { "kind": "CompilationUnit", "name": "test.cpp" }
    ],
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@bar",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 2,
        "name": {
          "usr": "c:@F@baz",
          "suffix": "",
          "namespace": [
            {
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "None" }
      },
      {
        "id": 1,
        "linkage": { "type": "Internal" }
      },
      {
        "id": 2,
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");
}

} // anonymous namespace
