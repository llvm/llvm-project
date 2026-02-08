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
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;

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
                             "missing required field 'pairs'");
  for (size_t I = 0; I < PairsArray->size(); ++I) {
    const json::Object *Pair = (*PairsArray)[I].getAsObject();
    if (!Pair)
      return createStringError(
          inconvertibleErrorCode(),
          "pairs element at index %zu is not a JSON object", I);
    auto FirstOpt = Pair->getInteger("first");
    if (!FirstOpt)
      return createStringError(inconvertibleErrorCode(),
                               "missing or invalid 'first' field at index %zu",
                               I);
    auto SecondOpt = Pair->getInteger("second");
    if (!SecondOpt)
      return createStringError(inconvertibleErrorCode(),
                               "missing or invalid 'second' field at index %zu",
                               I);
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
// Base Fixture with Common Utilities
// ============================================================================

class JSONFormatTestBase : public ::testing::Test {
protected:
  SmallString<128> TestDir;

  void SetUp() override {
    std::error_code EC =
        sys::fs::createUniqueDirectory("json-format-test", TestDir);
    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

  // Helper to create a temporary JSON file and read it using JSONFormat
  std::pair<JSONFormat, SmallString<128>>
  readJSON(StringRef JSON, StringRef Filename = "test.json") {
    SmallString<128> FilePath = TestDir;
    sys::path::append(FilePath, Filename);

    std::error_code EC;
    raw_fd_ostream OS(FilePath, EC);
    EXPECT_FALSE(EC) << "Failed to create file: " << EC.message();
    OS << JSON;
    OS.close();

    return {JSONFormat(vfs::getRealFileSystem()), FilePath};
  }

  // Helper to check if error message contains expected substrings
  bool errorContains(Error &Err, ArrayRef<StringRef> Parts) {
    std::string ErrorMsg = toString(std::move(Err));
    for (StringRef Part : Parts) {
      if (ErrorMsg.find(Part.str()) == std::string::npos)
        return false;
    }
    return true;
  }
};

// ============================================================================
// Fixture for Error Tests
// ============================================================================

class JSONFormatErrorTest : public JSONFormatTestBase {
protected:
  void expectError(StringRef JSON, ArrayRef<StringRef> ErrorParts,
                   StringRef Filename = "test.json") {
    auto [Format, FilePath] = createFormat(JSON, Filename);
    auto Result = Format.readTUSummary(FilePath);
    ASSERT_FALSE(Result) << "Expected error but read succeeded";

    Error Err = Result.takeError();
    std::string ErrorMsg = toString(std::move(Err));

    bool allFound = true;
    for (StringRef Part : ErrorParts) {
      if (ErrorMsg.find(Part.str()) == std::string::npos) {
        allFound = false;
        break;
      }
    }

    EXPECT_TRUE(allFound) << "Error message didn't contain expected parts.\n"
                          << "Actual error: " << ErrorMsg << "\n"
                          << "Expected parts: [" <<
        [&]() {
          std::string result;
          for (size_t i = 0; i < ErrorParts.size(); ++i) {
            if (i > 0)
              result += ", ";
            result += "\"" + ErrorParts[i].str() + "\"";
          }
          return result;
        }() << "]";
  }
};

// ============================================================================
// Fixture for Valid Configuration Tests
// ============================================================================

class JSONFormatValidTest : public JSONFormatTestBase {
protected:
  void expectSuccess(StringRef JSON, StringRef Filename = "test.json") {
    auto [Format, FilePath] = createFormat(JSON, Filename);
    auto Result = Format.readTUSummary(FilePath);
    if (!Result) {
      FAIL() << "Read failed: " << toString(Result.takeError());
    }
  }
};

// ============================================================================
// Fixture for Round-Trip Tests
// ============================================================================

class JSONFormatRoundTripTest : public JSONFormatTestBase {
protected:
  void testRoundTrip(StringRef InputJSON) {
    // Read the input
    auto [InputFormat, InputPath] = createFormat(InputJSON, "input.json");
    auto Summary = InputFormat.readTUSummary(InputPath);
    if (!Summary) {
      FAIL() << "Failed to read input: " << toString(Summary.takeError());
    }

    // Write to output file
    SmallString<128> OutputPath = TestDir;
    sys::path::append(OutputPath, "output.json");

    JSONFormat OutputFormat(vfs::getRealFileSystem());
    auto WriteErr = OutputFormat.writeTUSummary(*Summary, OutputPath);
    if (WriteErr) {
      FAIL() << "Failed to write output: " << toString(std::move(WriteErr));
    }

    // Read back the written file
    auto RoundTrip = OutputFormat.readTUSummary(OutputPath);
    if (!RoundTrip) {
      FAIL() << "Failed to read round-trip output: "
             << toString(RoundTrip.takeError());
    }
  }
};

// ============================================================================
// File Access Error Tests
// ============================================================================

TEST_F(JSONFormatErrorTest, NonexistentFile) {
  SmallString<128> NonexistentPath = TestDir;
  sys::path::append(NonexistentPath, "nonexistent.json");

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(NonexistentPath);
  ASSERT_FALSE(Result);

  Error Err = Result.takeError();
  EXPECT_TRUE(
      errorContains(Err, {"reading TUSummary from", "file does not exist"}));
}

TEST_F(JSONFormatErrorTest, NotJsonExtension) {
  expectError("{}", {"reading TUSummary from", "not a JSON file"}, "test.txt");
}

// ============================================================================
// JSON Syntax Error Tests
// ============================================================================

TEST_F(JSONFormatErrorTest, InvalidSyntax) {
  expectError("{ invalid json }", {"reading TUSummary from",
                                   "failed to read JSON object from file"});
}

TEST_F(JSONFormatErrorTest, NotObject) {
  expectError(
      "[]", {"reading TUSummary from", "failed to read JSON object from file"});
}

// ============================================================================
// Root Structure Error Tests
// ============================================================================

TEST_F(JSONFormatErrorTest, MissingTUNamespace) {
  expectError(
      R"({
    "id_table": [],
    "data": []
  })",
      {"reading TUSummary from", "missing or invalid field 'tu_namespace'"});
}

TEST_F(JSONFormatErrorTest, MissingKind) {
  expectError(R"({
    "tu_namespace": {
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })",
              {"reading TUSummary from", "failed to deserialize BuildNamespace",
               "missing required field 'kind' (expected BuildNamespaceKind)"});
}

TEST_F(JSONFormatErrorTest, MissingName) {
  expectError(R"({
    "tu_namespace": {
      "kind": "compilation_unit"
    },
    "id_table": [],
    "data": []
  })",
              {"reading TUSummary from", "failed to deserialize BuildNamespace",
               "missing required field 'name'"});
}

TEST_F(JSONFormatErrorTest, InvalidKind) {
  expectError(R"({
    "tu_namespace": {
      "kind": "invalid_kind",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })",
              {"reading TUSummary from", "failed to deserialize BuildNamespace",
               "invalid 'kind' BuildNamespaceKind value"});
}

TEST_F(JSONFormatErrorTest, MissingIDTable) {
  expectError(
      R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "data": []
  })",
      {"reading TUSummary from", "missing or invalid field 'id_table'"});
}

TEST_F(JSONFormatErrorTest, MissingData) {
  expectError(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": []
  })",
              {"reading TUSummary from", "missing or invalid field 'data'"});
}

// ============================================================================
// ID Table Error Tests
// ============================================================================

TEST_F(JSONFormatErrorTest, IDTableNotArray) {
  expectError(
      R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": {},
    "data": []
  })",
      {"reading TUSummary from", "missing or invalid field 'id_table'"});
}

TEST_F(JSONFormatErrorTest, IDTableElementNotObject) {
  expectError(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [123],
    "data": []
  })",
              {"reading TUSummary from", "failed to deserialize EntityIdTable",
               "element at index 0 is not a JSON object",
               "(expected EntityIdTable entry with 'id' and 'name' fields)"});
}

TEST_F(JSONFormatErrorTest, IDTableEntryMissingID) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize EntityIdTable at index 0",
       "failed to deserialize EntityIdTable entry",
       "missing required field 'id' (expected unsigned integer EntityId)"});
}

TEST_F(JSONFormatErrorTest, IDTableEntryMissingName) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize EntityIdTable at index 0",
       "failed to deserialize EntityIdTable entry",
       "missing or invalid field 'name' (expected EntityName JSON object)"});
}

TEST_F(JSONFormatErrorTest, IDTableEntryIDNotUInt64) {
  expectError(R"({
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
  })",
              {"reading TUSummary from",
               "failed to deserialize EntityIdTable at index 0",
               "failed to deserialize EntityIdTable entry",
               "field 'id' is not a valid unsigned 64-bit integer",
               "(expected non-negative EntityId value)"});
}

TEST_F(JSONFormatErrorTest, DuplicateEntity) {
  expectError(R"({
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
  })",
              {"reading TUSummary from", "failed to deserialize EntityIdTable",
               "duplicate EntityName found at index",
               "(EntityId=0 already exists in table)"});
}

// ============================================================================
// Entity Name Error Tests
// ============================================================================

TEST_F(JSONFormatErrorTest, EntityNameMissingUSR) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize EntityIdTable at index 0",
       "failed to deserialize EntityIdTable entry",
       "failed to deserialize EntityName",
       "missing required field 'usr' (Unified Symbol Resolution string)"});
}

TEST_F(JSONFormatErrorTest, EntityNameMissingSuffix) {
  expectError(R"({
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
  })",
              {"reading TUSummary from",
               "failed to deserialize EntityIdTable at index 0",
               "failed to deserialize EntityIdTable entry",
               "failed to deserialize EntityName",
               "missing required field 'suffix'"});
}

TEST_F(JSONFormatErrorTest, EntityNameMissingNamespace) {
  expectError(R"({
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
  })",
              {"reading TUSummary from",
               "failed to deserialize EntityIdTable at index 0",
               "failed to deserialize EntityIdTable entry",
               "failed to deserialize EntityName",
               "missing or invalid field 'namespace'",
               "(expected JSON array of BuildNamespace objects)"});
}

TEST_F(JSONFormatErrorTest, NamespaceElementNotObject) {
  expectError(R"({
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
  })",
              {"reading TUSummary from",
               "failed to deserialize EntityIdTable at index 0",
               "failed to deserialize EntityIdTable entry",
               "failed to deserialize EntityName",
               "failed to deserialize NestedBuildNamespace",
               "element at index 0 is not a JSON object"});
}

// ============================================================================
// Data Array Error Tests
// ============================================================================

TEST_F(JSONFormatErrorTest, DataNotArray) {
  expectError(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": {}
  })",
              {"reading TUSummary from", "missing or invalid field 'data'"});
}

TEST_F(JSONFormatErrorTest, DataElementNotObject) {
  expectError(
      R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": ["invalid"]
  })",
      {"reading TUSummary from", "failed to deserialize SummaryDataMap",
       "element at index 0 is not a JSON object",
       "(expected SummaryDataMap entry with 'summary_name' and 'summary_data'",
       "fields)"});
}

TEST_F(JSONFormatErrorTest, DataEntryMissingSummaryName) {
  expectError(R"({
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
  })",
              {"reading TUSummary from",
               "failed to deserialize SummaryDataMap at index 0",
               "failed to deserialize SummaryDataMap entry",
               "missing required field 'summary_name'"});
}

TEST_F(JSONFormatErrorTest, DataEntryMissingData) {
  expectError(R"({
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
  })",
              {"reading TUSummary from",
               "failed to deserialize SummaryDataMap at index 0",
               "failed to deserialize SummaryDataMap entry",
               "missing or invalid field 'summary_data'"});
}

TEST_F(JSONFormatErrorTest, DuplicateSummaryName) {
  expectError(R"({
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
  })",
              {"reading TUSummary from", "failed to deserialize SummaryDataMap",
               "duplicate SummaryName 'test_summary' found at index"});
}

// ============================================================================
// Entity Data Error Tests
// ============================================================================

TEST_F(JSONFormatErrorTest, EntityDataElementNotObject) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize SummaryDataMap at index 0",
       "failed to deserialize SummaryDataMap entry for summary 'test_summary'",
       "failed to deserialize EntityDataMap",
       "element at index 0 is not a JSON object",
       "(expected EntityDataMap entry with 'entity_id' and 'entity_summary'",
       "fields)"});
}

TEST_F(JSONFormatErrorTest, EntityDataMissingEntityID) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize SummaryDataMap at index 0",
       "failed to deserialize SummaryDataMap entry for summary 'test_summary'",
       "failed to deserialize EntityDataMap at index 0",
       "failed to deserialize EntityDataMap entry",
       "missing required field 'entity_id' (expected unsigned integer "
       "EntityId)"});
}

TEST_F(JSONFormatErrorTest, EntityDataMissingEntitySummary) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize SummaryDataMap at index 0",
       "failed to deserialize SummaryDataMap entry for summary 'test_summary'",
       "failed to deserialize EntityDataMap at index 0",
       "failed to deserialize EntityDataMap entry",
       "missing or invalid field 'entity_summary'",
       "(expected EntitySummary JSON object)"});
}

TEST_F(JSONFormatErrorTest, EntityIDNotUInt64) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize SummaryDataMap at index 0",
       "failed to deserialize SummaryDataMap entry for summary 'test_summary'",
       "failed to deserialize EntityDataMap at index 0",
       "failed to deserialize EntityDataMap entry",
       "field 'entity_id' is not a valid unsigned 64-bit integer",
       "(expected non-negative EntityId value)"});
}

TEST_F(JSONFormatErrorTest, EntitySummaryNoFormatInfo) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize SummaryDataMap at index 0",
       "failed to deserialize SummaryDataMap entry for summary "
       "'unknown_summary_type'",
       "failed to deserialize EntityDataMap at index 0",
       "failed to deserialize EntityDataMap entry",
       "failed to deserialize EntitySummary",
       "no FormatInfo was registered for summary name: unknown_summary_type"});
}

// ============================================================================
// Analysis-Specific Error Tests - TestAnalysis
// ============================================================================

TEST_F(JSONFormatErrorTest, TestAnalysisMissingField) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize SummaryDataMap at index 0",
       "failed to deserialize SummaryDataMap entry for summary 'test_summary'",
       "failed to deserialize EntityDataMap at index 0",
       "failed to deserialize EntityDataMap entry",
       "missing required field 'pairs'"});
}

TEST_F(JSONFormatErrorTest, TestAnalysisInvalidPair) {
  expectError(
      R"({
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
  })",
      {"reading TUSummary from",
       "failed to deserialize SummaryDataMap at index 0",
       "failed to deserialize SummaryDataMap entry for summary 'test_summary'",
       "failed to deserialize EntityDataMap at index 0",
       "failed to deserialize EntityDataMap entry",
       "missing or invalid 'second' field at index 0"});
}

// ============================================================================
// Valid Configuration Tests
// ============================================================================

TEST_F(JSONFormatValidTest, Empty) {
  expectSuccess(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatValidTest, LinkUnit) {
  expectSuccess(R"({
    "tu_namespace": {
      "kind": "link_unit",
      "name": "libtest.so"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatValidTest, WithIDTable) {
  expectSuccess(R"({
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

TEST_F(JSONFormatValidTest, WithEmptyDataEntry) {
  expectSuccess(R"({
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

// ============================================================================
// Round-Trip Tests
// ============================================================================

TEST_F(JSONFormatRoundTripTest, Empty) {
  testRoundTrip(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatRoundTripTest, WithIDTable) {
  testRoundTrip(R"({
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

TEST_F(JSONFormatRoundTripTest, LinkUnit) {
  testRoundTrip(R"({
    "tu_namespace": {
      "kind": "link_unit",
      "name": "libtest.so"
    },
    "id_table": [],
    "data": []
  })");
}

// ============================================================================
// Analysis-Specific Round-Trip Tests
// ============================================================================

TEST_F(JSONFormatRoundTripTest, TestAnalysis) {
  testRoundTrip(R"({
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
