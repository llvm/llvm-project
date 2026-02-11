//===- unittests/Analysis/Scalable/JSONFormatTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for SSAF JSON serialization format reading and writing.
//
//===----------------------------------------------------------------------===//


#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;
using ::testing::AllOf;
using ::testing::HasSubstr;

namespace {

// ============================================================================
// Test Analysis - Simple analysis for testing JSON serialization
// ============================================================================

struct TestAnalysis : EntitySummary {
  TestAnalysis() : EntitySummary(SummaryName("test_summary")) {}
  std::vector<std::pair<EntityId, EntityId>> Pairs;
};

static json::Object
serializeTestAnalysis(const EntitySummary &Summary,
                      const JSONFormat::EntityIdConverter &Converter) {
  const auto &TA = static_cast<const TestAnalysis &>(Summary);
  json::Array PairsArray;
  for (const auto &[First, Second] : TA.Pairs) {
    PairsArray.push_back(json::Object{
        {"first", Converter.toJSON(First)},
        {"second", Converter.toJSON(Second)},
    });
  }
  return json::Object{{"pairs", std::move(PairsArray)}};
}

static Expected<std::unique_ptr<EntitySummary>>
deserializeTestAnalysis(const json::Object &Obj, EntityIdTable &IdTable,
                        const JSONFormat::EntityIdConverter &Converter) {
  auto Result = std::make_unique<TestAnalysis>();
  const json::Array *PairsArray = Obj.getArray("pairs");
  if (!PairsArray)
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'pairs'");
  for (const auto &[Index, Value] : llvm::enumerate(*PairsArray)) {
    const json::Object *Pair = Value.getAsObject();
    if (!Pair)
      return createStringError(
          inconvertibleErrorCode(),
          "pairs element at index %zu is not a JSON object", Index);
    auto FirstOpt = Pair->getInteger("first");
    if (!FirstOpt)
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'first' field at index '%zu'", Index);
    auto SecondOpt = Pair->getInteger("second");
    if (!SecondOpt)
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'second' field at index '%zu'", Index);
    Result->Pairs.emplace_back(Converter.fromJSON(*FirstOpt),
                               Converter.fromJSON(*SecondOpt));
  }
  return std::move(Result);
}

struct TestAnalysisFormatInfo : JSONFormat::FormatInfo {
  TestAnalysisFormatInfo()
      : JSONFormat::FormatInfo(SummaryName("test_summary"),
                               serializeTestAnalysis, deserializeTestAnalysis) {
  }
};

static llvm::Registry<JSONFormat::FormatInfo>::Add<TestAnalysisFormatInfo>
    RegisterTestAnalysis("TestAnalysis", "Format info for test analysis data");

// ============================================================================
// Test Fixture
// ============================================================================

class JSONFormatTest : public ::testing::Test {
protected:
  SmallString<128> TestDir;

  void SetUp() override {
    std::error_code EC =
        sys::fs::createUniqueDirectory("json-format-test", TestDir);
    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

  llvm::Expected<TUSummary> readJSON(StringRef JSON,
                                     StringRef Filename = "test.json") {
    SmallString<128> FilePath = TestDir;
    sys::path::append(FilePath, Filename);

    std::error_code EC;
    raw_fd_ostream OS(FilePath, EC);
    EXPECT_FALSE(EC) << "Failed to create file: " << EC.message();
    OS << JSON;
    OS.close();

    auto Result = JSONFormat(vfs::getRealFileSystem()).readTUSummary(FilePath);
    return Result;
  }

  void readWriteJSON(StringRef InputJSON) {
    // Read the input JSON
    auto Summary1 = readJSON(InputJSON, "input.json");
    ASSERT_THAT_EXPECTED(Summary1, Succeeded());

    // Write to first output file
    SmallString<128> Output1Path = TestDir;
    sys::path::append(Output1Path, "output1.json");

    JSONFormat Format(vfs::getRealFileSystem());
    auto WriteErr1 = Format.writeTUSummary(*Summary1, Output1Path);
    ASSERT_THAT_ERROR(std::move(WriteErr1), Succeeded());

    // Read back from first output
    auto Summary2 = Format.readTUSummary(Output1Path);
    ASSERT_THAT_EXPECTED(Summary2, Succeeded());

    // Write to second output file
    SmallString<128> Output2Path = TestDir;
    sys::path::append(Output2Path, "output2.json");

    auto WriteErr2 = Format.writeTUSummary(*Summary2, Output2Path);
    ASSERT_THAT_ERROR(std::move(WriteErr2), Succeeded());

    // Compare the two output files byte-by-byte
    auto Buffer1 = MemoryBuffer::getFile(Output1Path);
    ASSERT_TRUE(Buffer1) << "Failed to read output1.json";

    auto Buffer2 = MemoryBuffer::getFile(Output2Path);
    ASSERT_TRUE(Buffer2) << "Failed to read output2.json";

    EXPECT_EQ(Buffer1.get()->getBuffer(), Buffer2.get()->getBuffer())
        << "Serialization is not stable: first write differs from second write";
  }
};

// ============================================================================
// readJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, NonexistentFile) {
  SmallString<128> NonexistentPath = TestDir;
  sys::path::append(NonexistentPath, "nonexistent.json");

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(NonexistentPath);

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                                      HasSubstr("file does not exist"))));
}

TEST_F(JSONFormatTest, PathIsDirectory) {
  SmallString<128> DirPath = TestDir;
  sys::path::append(DirPath, "test_directory.json");

  std::error_code EC = sys::fs::create_directory(DirPath);
  ASSERT_FALSE(EC) << "Failed to create directory: " << EC.message();

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(DirPath);

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                              HasSubstr("path is a directory, not a file"))));
}

TEST_F(JSONFormatTest, NotJsonExtension) {
  auto Result = readJSON("{}", "test.txt");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("failed to read file"),
                  HasSubstr("file does not end with '.json' extension"))));
}

TEST_F(JSONFormatTest, BrokenSymlink) {
  SmallString<128> TargetPath = TestDir;
  sys::path::append(TargetPath, "nonexistent_target.json");

  SmallString<128> SymlinkPath = TestDir;
  sys::path::append(SymlinkPath, "broken_symlink.json");

  // Create a symlink pointing to a non-existent file
  std::error_code EC = sys::fs::create_link(TargetPath, SymlinkPath);
  ASSERT_FALSE(EC) << "Failed to create symlink: " << EC.message();

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(SymlinkPath);

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("failed to read file"))));
}

TEST_F(JSONFormatTest, NoReadPermission) {
#ifdef _WIN32
  GTEST_SKIP() << "Permission model differs on Windows";
#endif

  SmallString<128> FilePath = TestDir;
  sys::path::append(FilePath, "no_read_permission.json");

  // Create file with valid JSON
  std::error_code EC;
  raw_fd_ostream OS(FilePath, EC);
  ASSERT_FALSE(EC) << "Failed to create file: " << EC.message();
  OS << R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })";
  OS.close();

  // Remove read permissions
  sys::fs::perms Perms =
      sys::fs::perms::owner_write | sys::fs::perms::owner_exe;
  EC = sys::fs::setPermissions(FilePath, Perms);
  ASSERT_FALSE(EC) << "Failed to set permissions: " << EC.message();

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(FilePath);

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("failed to read file"))));

  // Restore permissions for cleanup
  sys::fs::setPermissions(FilePath, sys::fs::perms::all_all);
}

TEST_F(JSONFormatTest, InvalidSyntax) {
  auto Result = readJSON("{ invalid json }");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("Expected object key"))));
}

TEST_F(JSONFormatTest, NotObject) {
  auto Result = readJSON("[]");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("failed to read TUSummary"),
                                      HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::buildNamespaceKindFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, InvalidKind) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "invalid_kind",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading BuildNamespace from field 'tu_namespace'"),
          HasSubstr("reading BuildNamespaceKind from field 'kind'"),
          HasSubstr(
              "invalid 'kind' BuildNamespaceKind value 'invalid_kind'"))));
}

// ============================================================================
// JSONFormat::buildNamespaceFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, MissingKind) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading BuildNamespace from field 'tu_namespace'"),
          HasSubstr("failed to read BuildNamespaceKind from field 'kind'"),
          HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatTest, MissingName) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading BuildNamespace from field 'tu_namespace'"),
          HasSubstr("failed to read BuildNamespaceName from field 'name'"),
          HasSubstr("expected JSON string"))));
}

// ============================================================================
// JSONFormat::nestedBuildNamespaceFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, NamespaceElementNotObject) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
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
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("reading EntityName from field 'name'"),
                HasSubstr("reading NesteBuildNamespace from field 'namespace'"),
                HasSubstr("failed to read BuildNamespace from index '0'"),
                HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::entityNameFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, EntityNameMissingUSR) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "suffix": "",
          "namespace": []
        }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(
                  AllOf(HasSubstr("reading TUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("reading EntityName from field 'name'"),
                        HasSubstr("failed to read USR from field 'usr'"),
                        HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatTest, EntityNameMissingSuffix) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "namespace": []
        }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(
                  AllOf(HasSubstr("reading TUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("reading EntityName from field 'name'"),
                        HasSubstr("failed to read Suffix from field 'suffix'"),
                        HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatTest, EntityNameMissingNamespace) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
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
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr(
              "failed to read NestedBuildNamespace from field 'namespace'"),
          HasSubstr("expected JSON array"))));
}

// ============================================================================
// JSONFormat::entityIdTableEntryFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, IDTableEntryMissingID) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": []
        }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(
                  AllOf(HasSubstr("reading TUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("failed to read EntityId from field 'id'"),
                        HasSubstr("expected JSON (unsigned 64-bit)"))));
}

TEST_F(JSONFormatTest, IDTableEntryMissingName) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0
      }
    ],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading IdTable from field 'id_table'"),
                  HasSubstr("reading EntityIdTable entry from index '0'"),
                  HasSubstr("failed to read EntityName from field 'name'"),
                  HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTest, IDTableEntryIDNotUInt64) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": "not_a_number",
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": []
        }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(
                  AllOf(HasSubstr("reading TUSummary from file"),
                        HasSubstr("reading IdTable from field 'id_table'"),
                        HasSubstr("reading EntityIdTable entry from index '0'"),
                        HasSubstr("failed to read EntityId from field 'id'"),
                        HasSubstr("expected JSON integer (unsigned 64-bit)"))));
}

// ============================================================================
// JSONFormat::entityIdTableFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, IDTableNotArray) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": {},
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("failed to read IdTable from field 'id_table'"),
                  HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatTest, IDTableElementNotObject) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [123],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("failed to read EntityIdTable entry from index '0'"),
                HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTest, DuplicateEntity) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "compilation_unit",
              "name": "test.cpp"
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
              "kind": "compilation_unit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("failed to insert EntityIdTable entry at index '1'"),
                HasSubstr("encountered duplicate EntityId '0'"))));
}

// ============================================================================
// JSONFormat::entitySummaryFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, EntitySummaryNoFormatInfo) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "unknown_summary_type",
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
              "no FormatInfo registered for summary 'unknown_summary_type'"))));
}

TEST_F(JSONFormatTest, TestAnalysisMissingField) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
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

TEST_F(JSONFormatTest, TestAnalysisInvalidPair) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": [
                {
                  "first": 0,
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

TEST_F(JSONFormatTest, EntityDataMissingEntityID) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": [
          {
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
          HasSubstr("failed to read EntityId from field 'entity_id'"),
          HasSubstr("expected JSON integer"))));
}

TEST_F(JSONFormatTest, EntityDataMissingEntitySummary) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": [
          {
            "entity_id": 0
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
          HasSubstr("failed to read EntitySummary from field 'entity_summary'"),
          HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTest, EntityIDNotUInt64) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": [
          {
            "entity_id": "not_a_number",
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
          HasSubstr("failed to read EntityId from field 'entity_id'"),
          HasSubstr("expected JSON integer"))));
}

// ============================================================================
// JSONFormat::entityDataMapFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, EntityDataElementNotObject) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": ["invalid"]
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
          HasSubstr("failed to read EntitySummary entry from index '0'"),
          HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTest, DuplicateEntityIdInDataMap) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": []
        }
      }
    ],
    "data": [
      {
        "summary_name": "test_summary",
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("reading EntitySummary entries from field 'summary_data'"),
          HasSubstr("failed to insert EntitySummary entry at index '1'"),
          HasSubstr("encountered duplicate EntityId '0'"))));
}

// ============================================================================
// JSONFormat::summaryDataMapEntryFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, DataEntryMissingSummaryName) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_data": []
      }
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("failed to read SummaryName from field 'summary_name'"),
          HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatTest, DataEntryMissingData) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary"
      }
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr(
              "failed to read EntitySummary entries from field 'summary_data'"),
          HasSubstr("expected JSON array"))));
}

// ============================================================================
// JSONFormat::summaryDataMapFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, DataNotArray) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": {}
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read SummaryData entries from field 'data'"),
          HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatTest, DataElementNotObject) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": ["invalid"]
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading SummaryData entries from field 'data'"),
                  HasSubstr("failed to read SummaryData entry from index '0'"),
                  HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTest, DuplicateSummaryName) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": []
      },
      {
        "summary_name": "test_summary",
        "summary_data": []
      }
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("failed to insert SummaryData entry at index '1'"),
          HasSubstr("encountered duplicate SummaryName 'test_summary'"))));
}

// ============================================================================
// JSONFormat::readTUSummary() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, MissingTUNamespace) {
  auto Result = readJSON(R"({
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read BuildNamespace from field 'tu_namespace'"),
          HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTest, MissingIDTable) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("failed to read IdTable from field 'id_table'"),
                  HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatTest, MissingData) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read SummaryData entries from field 'data'"),
          HasSubstr("expected JSON array"))));
}

// ============================================================================
// JSONFormat::writeJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, WriteFileAlreadyExists) {
  SmallString<128> FilePath = TestDir;
  sys::path::append(FilePath, "existing.json");

  // Create an existing file
  std::error_code EC;
  raw_fd_ostream OS(FilePath, EC);
  ASSERT_FALSE(EC) << "Failed to create file: " << EC.message();
  OS << "{}";
  OS.close();

  // Try to write to the same path
  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.writeTUSummary(Summary, FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("file already exists"))));
}

TEST_F(JSONFormatTest, WriteParentDirectoryNotFound) {
  SmallString<128> FilePath = TestDir;
  sys::path::append(FilePath, "nonexistent_dir", "test.json");

  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.writeTUSummary(Summary, FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("parent directory does not exist"))));
}

TEST_F(JSONFormatTest, WriteNotJsonExtension) {
  SmallString<128> FilePath = TestDir;
  sys::path::append(FilePath, "test.txt");

  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.writeTUSummary(Summary, FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("writing TUSummary to file"),
                HasSubstr("failed to write file"),
                HasSubstr("file does not end with '.json' extension"))));
}

TEST_F(JSONFormatTest, WriteStreamOpenFailure) {
#ifdef _WIN32
  GTEST_SKIP() << "Permission model differs on Windows";
#endif

  SmallString<128> DirPath = TestDir;
  sys::path::append(DirPath, "write_protected_dir");

  // Create a directory without write permissions
  std::error_code EC = sys::fs::create_directory(DirPath);
  ASSERT_FALSE(EC) << "Failed to create directory: " << EC.message();

  sys::fs::perms Perms = sys::fs::perms::owner_read | sys::fs::perms::owner_exe;
  EC = sys::fs::setPermissions(DirPath, Perms);
  ASSERT_FALSE(EC) << "Failed to set permissions: " << EC.message();

  SmallString<128> FilePath = DirPath;
  sys::path::append(FilePath, "test.json");

  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.writeTUSummary(Summary, FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"))));

  // Restore permissions for cleanup
  sys::fs::setPermissions(DirPath, sys::fs::perms::all_all);
}

// ============================================================================
// Round-Trip Tests - Serialization Verification
// ============================================================================

TEST_F(JSONFormatTest, Empty) {
  readWriteJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatTest, LinkUnit) {
  readWriteJSON(R"({
    "tu_namespace": {
      "kind": "link_unit",
      "name": "libtest.so"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatTest, WithIDTable) {
  readWriteJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "compilation_unit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@bar",
          "suffix": "1",
          "namespace": [
            {
              "kind": "compilation_unit",
              "name": "test.cpp"
            },
            {
              "kind": "link_unit",
              "name": "libtest.so"
            }
          ]
        }
      }
    ],
    "data": []
  })");
}

TEST_F(JSONFormatTest, WithEmptyDataEntry) {
  readWriteJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": []
      }
    ]
  })");
}

TEST_F(JSONFormatTest, RoundTripWithIDTable) {
  readWriteJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {
              "kind": "compilation_unit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "data": []
  })");
}

TEST_F(JSONFormatTest, RoundTripTestAnalysis) {
  readWriteJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@main",
          "suffix": "",
          "namespace": [
            {
              "kind": "compilation_unit",
              "name": "test.cpp"
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
              "kind": "compilation_unit",
              "name": "test.cpp"
            }
          ]
        }
      },
      {
        "id": 2,
        "name": {
          "usr": "c:@F@bar",
          "suffix": "",
          "namespace": [
            {
              "kind": "compilation_unit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "data": [
      {
        "summary_name": "test_summary",
        "summary_data": [
          {
            "entity_id": 0,
            "entity_summary": {
              "pairs": [
                {
                  "first": 0,
                  "second": 1
                },
                {
                  "first": 1,
                  "second": 2
                }
              ]
            }
          }
        ]
      }
    ]
  })");
}

} // anonymous namespace
