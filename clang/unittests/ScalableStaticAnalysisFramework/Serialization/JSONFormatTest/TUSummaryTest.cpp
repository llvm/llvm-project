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

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
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
      TUSummary S(std::move(BN));
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
      TUSummaryEncoding E(std::move(BN));
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

TEST_P(TUSummaryTest, InvalidSyntax) {
  auto Result = readFromString("{ invalid json }");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                              HasSubstr("Expected object key"))));
}

TEST_P(TUSummaryTest, NotObject) {
  auto Result = readFromString("[]");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                              HasSubstr("failed to read TUSummary"),
                              HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::entityLinkageFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, LinkageTableEntryLinkageMissingType) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("reading EntityLinkage from field 'linkage'"),
                HasSubstr("failed to read EntityLinkageType from field 'type'"),
                HasSubstr("expected JSON string"))));
}

TEST_P(TUSummaryTest, LinkageTableEntryLinkageInvalidType) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("reading EntityLinkage from field 'linkage'"),
                HasSubstr("invalid EntityLinkageType value 'invalid_type' for "
                          "field 'type'"))));
}

// ============================================================================
// JSONFormat::linkageTableEntryFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, LinkageTableEntryMissingId) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(TUSummaryTest, LinkageTableEntryIdNotUInt64) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(TUSummaryTest, LinkageTableEntryMissingLinkage) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityLinkage from field 'linkage'"),
                HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::linkageTableFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, LinkageTableNotArray) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": {},
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read LinkageTable from field 'linkage_table'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(TUSummaryTest, LinkageTableElementNotObject) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": ["invalid"],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("failed to read LinkageTable entry from index '0'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(TUSummaryTest, LinkageTableExtraId) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to deserialize LinkageTable"),
                HasSubstr("extra 'EntityId(0)' not present in IdTable"))));
}

TEST_P(TUSummaryTest, LinkageTableMissingId) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("failed to deserialize LinkageTable"),
                HasSubstr("missing 'EntityId(0)' present in IdTable"))));
}

TEST_P(TUSummaryTest, LinkageTableDuplicateId) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("failed to insert LinkageTable entry at index '1'"),
                HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::buildNamespaceKindFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, InvalidKind) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "invalid_kind",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading BuildNamespace from field 'tu_namespace'"),
                HasSubstr("reading BuildNamespaceKind from field 'kind'"),
                HasSubstr("invalid BuildNamespaceKind value 'invalid_kind' for "
                          "field 'kind'"))));
}

// ============================================================================
// JSONFormat::buildNamespaceFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, MissingKind) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading BuildNamespace from field 'tu_namespace'"),
          HasSubstr("failed to read BuildNamespaceKind from field 'kind'"),
          HasSubstr("expected JSON string"))));
}

TEST_P(TUSummaryTest, MissingName) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading BuildNamespace from field 'tu_namespace'"),
          HasSubstr("failed to read BuildNamespaceName from field 'name'"),
          HasSubstr("expected JSON string"))));
}

// ============================================================================
// JSONFormat::tuEntityNameFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, EntityNameMissingUSR) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "suffix": ""
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

  EXPECT_THAT_ERROR(std::move(Result),
                    FailedWithMessage(AllOf(
                        HasSubstr("reading TUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("reading EntityName from field 'name'"),
                        HasSubstr("failed to read USR from field 'usr'"),
                        HasSubstr("expected JSON string"))));
}

TEST_P(TUSummaryTest, EntityNameMissingSuffix) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo"
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
                        HasSubstr("reading TUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("reading EntityName from field 'name'"),
                        HasSubstr("failed to read Suffix from field 'suffix'"),
                        HasSubstr("expected JSON string"))));
}

// ============================================================================
// JSONFormat::tuEntityIdTableEntryFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, IDTableEntryMissingID) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(TUSummaryTest, IDTableEntryMissingName) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0
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
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityName from field 'name'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(TUSummaryTest, IDTableEntryIDNotUInt64) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": "not_a_number",
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

// ============================================================================
// JSONFormat::tuEntityIdTableFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, IDTableNotArray) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": {},
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("failed to read IdTable from field 'id_table'"),
                HasSubstr("expected JSON array"))));
}

TEST_P(TUSummaryTest, IDTableElementNotObject) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [123],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("failed to read EntityIdTable entry from index '0'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(TUSummaryTest, DuplicateEntity) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      }
    ],
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "Internal" }
      },
      {
        "id": 1,
        "linkage": { "type": "External" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("failed to insert EntityIdTable entry at index '1'"),
                HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::entitySummaryFromJSON() / encodingDataMapEntryFromJSON() Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, ReadEntitySummaryNoFormatInfo) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
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

TEST_F(JSONFormatTUSummaryTest,
       PairsEntitySummaryForJSONFormatTestMissingPairsField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid field 'pairs'"))));
}

TEST_F(JSONFormatTUSummaryTest,
       PairsEntitySummaryForJSONFormatTestInvalidPairsFieldType) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid field 'pairs'"))));
}

TEST_F(JSONFormatTUSummaryTest,
       PairsEntitySummaryForJSONFormatTestPairsElementNotObject) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("pairs element at index 0 is not a JSON object"))));
}

TEST_F(JSONFormatTUSummaryTest,
       PairsEntitySummaryForJSONFormatTestMissingFirstField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'first' field at index '0'"))));
}

TEST_F(JSONFormatTUSummaryTest,
       PairsEntitySummaryForJSONFormatTestInvalidFirstField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'first' field at index '0'"))));
}

TEST_F(JSONFormatTUSummaryTest,
       PairsEntitySummaryForJSONFormatTestMissingSecondField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'second' field at index '0'"))));
}

TEST_F(JSONFormatTUSummaryTest,
       PairsEntitySummaryForJSONFormatTestInvalidSecondField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("reading EntitySummary from field 'entity_summary'"),
          HasSubstr("missing or invalid 'second' field at index '0'"))));
}

// ============================================================================
// JSONFormat::entityDataMapEntryFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, EntityDataMissingEntityID) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to read EntityId from field 'entity_id'"),
          HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_P(TUSummaryTest, EntityDataMissingEntitySummary) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to read EntitySummary from field 'entity_summary'"),
          HasSubstr("expected JSON object"))));
}

TEST_P(TUSummaryTest, EntityIDNotUInt64) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("reading EntitySummary entry from index '0'"),
          HasSubstr("failed to read EntityId from field 'entity_id'"),
          HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_F(JSONFormatTUSummaryTest, ReadEntitySummaryMissingData) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr(
              "EntitySummary data for "
              "'SummaryName(MismatchedEntitySummaryForJSONFormatTest)' reports "
              "mismatched "
              "'SummaryName(MismatchedEntitySummaryForJSONFormatTest_WrongName)"
              "'"))));
}

// ============================================================================
// JSONFormat::entityDataMapEntryToJSON() Fatal Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, WriteEntitySummaryMissingData) {
  TUSummary Summary(
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
      "'SummaryName\\(MismatchedEntitySummaryForJSONFormatTest_WrongName\\)'");
}

// ============================================================================
// JSONFormat::entityDataMapFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, EntityDataElementNotObject) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("failed to read EntitySummary entry from index '0'"),
          HasSubstr("expected JSON object"))));
}

TEST_P(TUSummaryTest, DuplicateEntityIdInDataMap) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
            "entity_summary": {
              "pairs": []
            }
          },
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": []
            }
          }
        ]
      }
    ]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("failed to insert EntitySummary entry at index '1'"),
          HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::summaryDataMapEntryFromJSON() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, DataEntryMissingSummaryName) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("failed to read SummaryName from field 'summary_name'"),
          HasSubstr("expected JSON string"))));
}

TEST_P(TUSummaryTest, DataEntryMissingData) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr(
              "failed to read EntitySummary entries from field 'summary_data'"),
          HasSubstr("expected JSON array"))));
}

// ============================================================================
// JSONFormat::summaryDataMapFromJSON() / encodingSummaryDataMapFromJSON() Tests
// ============================================================================

TEST_P(TUSummaryTest, DataNotArray) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": {}
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read SummaryData entries from field 'data'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(TUSummaryTest, DataElementNotObject) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": ["invalid"]
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading SummaryData entries from field 'data'"),
                HasSubstr("failed to read SummaryData entry from index '0'"),
                HasSubstr("expected JSON object"))));
}

TEST_P(TUSummaryTest, DuplicateSummaryName) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("failed to insert SummaryData entry at index '1'"),
          HasSubstr("encountered duplicate "
                    "'SummaryName(PairsEntitySummaryForJSONFormatTest)'"))));
}

// ============================================================================
// JSONFormat::readTUSummary() / readTUSummaryEncoding() Error Tests
// ============================================================================

TEST_P(TUSummaryTest, MissingTUNamespace) {
  auto Result = readFromString(R"({
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read BuildNamespace from field 'tu_namespace'"),
          HasSubstr("expected JSON object"))));
}

TEST_P(TUSummaryTest, MissingIDTable) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("failed to read IdTable from field 'id_table'"),
                HasSubstr("expected JSON array"))));
}

TEST_P(TUSummaryTest, MissingLinkageTable) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read LinkageTable from field 'linkage_table'"),
          HasSubstr("expected JSON array"))));
}

TEST_P(TUSummaryTest, MissingData) {
  auto Result = readFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": []
  })");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read SummaryData entries from field 'data'"),
          HasSubstr("expected JSON array"))));
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

// ============================================================================
// Round-Trip Tests - Serialization Verification
// ============================================================================

TEST_P(TUSummaryTest, RoundTripEmpty) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");
}

TEST_P(TUSummaryTest, RoundTripWithTwoSummaryTypes) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 3,
        "name": {
          "usr": "c:@F@qux",
          "suffix": ""
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@bar",
          "suffix": ""
        }
      },
      {
        "id": 4,
        "name": {
          "usr": "c:@F@quux",
          "suffix": ""
        }
      },
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      },
      {
        "id": 2,
        "name": {
          "usr": "c:@F@baz",
          "suffix": ""
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

TEST_P(TUSummaryTest, RoundTripLinkUnit) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "LinkUnit",
      "name": "libtest.so"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");
}

TEST_P(TUSummaryTest, RoundTripWithEmptyDataEntry) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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

TEST_P(TUSummaryTest, RoundTripLinkageTableWithNoneLinkage) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
}

TEST_P(TUSummaryTest, RoundTripLinkageTableWithInternalLinkage) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@bar",
          "suffix": ""
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

TEST_P(TUSummaryTest, RoundTripLinkageTableWithExternalLinkage) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@baz",
          "suffix": ""
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

TEST_P(TUSummaryTest, RoundTripLinkageTableWithMultipleEntries) {
  readWriteCompare(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": ""
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@bar",
          "suffix": ""
        }
      },
      {
        "id": 2,
        "name": {
          "usr": "c:@F@baz",
          "suffix": ""
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
