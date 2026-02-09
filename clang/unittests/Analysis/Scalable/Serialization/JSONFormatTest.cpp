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
// Custom Matchers
// ============================================================================

// Helper to check if an Error or Expected succeeded
template <typename T> struct SuccessChecker {
  static bool isSuccess(T &val) {
    // For Expected<U>
    return static_cast<bool>(val);
  }
  static std::string getError(T &val) { return toString(val.takeError()); }
};

// Specialization for Error type
template <> struct SuccessChecker<Error> {
  static bool isSuccess(Error &val) {
    // For Error, success means no error (false/empty)
    return !static_cast<bool>(val);
  }
  static std::string getError(Error &val) { return toString(std::move(val)); }
};

// Matcher for Expected<T> or Error success
MATCHER(Succeeded, "") {
  // Cast away constness to get mutable access
  auto &mutable_arg =
      const_cast<std::remove_const_t<std::remove_reference_t<decltype(arg)>> &>(
          arg);

  using ArgType = std::remove_const_t<std::remove_reference_t<decltype(arg)>>;

  if (!SuccessChecker<ArgType>::isSuccess(mutable_arg)) {
    *result_listener << "Operation failed with error: "
                     << SuccessChecker<ArgType>::getError(mutable_arg);
    return false;
  }
  return true;
}

// Matcher for Expected<T> or Error failure with specific error message
MATCHER_P(FailedWith, SubstrMatcher, "") {
  // Cast away constness to get mutable access
  auto &mutable_arg =
      const_cast<std::remove_const_t<std::remove_reference_t<decltype(arg)>> &>(
          arg);

  using ArgType = std::remove_const_t<std::remove_reference_t<decltype(arg)>>;

  if (SuccessChecker<ArgType>::isSuccess(mutable_arg)) {
    *result_listener << "Expected operation to fail, but it succeeded";
    return false;
  }

  std::string ErrorMsg = SuccessChecker<ArgType>::getError(mutable_arg);

  if (!::testing::Matches(SubstrMatcher)(ErrorMsg)) {
    *result_listener << "Error message was: " << ErrorMsg;
    return false;
  }
  return true;
}

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

  auto readJSON(StringRef JSON, StringRef Filename = "test.json") {
    SmallString<128> FilePath = TestDir;
    sys::path::append(FilePath, Filename);

    std::error_code EC;
    raw_fd_ostream OS(FilePath, EC);
    EXPECT_FALSE(EC) << "Failed to create file: " << EC.message();
    OS << JSON;
    OS.close();

    return JSONFormat(vfs::getRealFileSystem()).readTUSummary(FilePath);
  }

  void testRoundTrip(StringRef InputJSON) {
    auto Summary = readJSON(InputJSON, "input.json");
    ASSERT_THAT(Summary, Succeeded());

    SmallString<128> OutputPath = TestDir;
    sys::path::append(OutputPath, "output.json");

    JSONFormat OutputFormat(vfs::getRealFileSystem());
    auto WriteErr = OutputFormat.writeTUSummary(*Summary, OutputPath);
    ASSERT_THAT(WriteErr, Succeeded());

    auto RoundTrip = OutputFormat.readTUSummary(OutputPath);
    ASSERT_THAT(RoundTrip, Succeeded());
  }
};

// ============================================================================
// File Access Error Tests
// ============================================================================

TEST_F(JSONFormatTest, NonexistentFile) {
  SmallString<128> NonexistentPath = TestDir;
  sys::path::append(NonexistentPath, "nonexistent.json");

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(NonexistentPath);

  EXPECT_THAT(Result, FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                                       HasSubstr("file does not exist"))));
}

TEST_F(JSONFormatTest, PathIsDirectory) {
  SmallString<128> DirPath = TestDir;
  sys::path::append(DirPath, "test_directory.json");

  std::error_code EC = sys::fs::create_directory(DirPath);
  ASSERT_FALSE(EC) << "Failed to create directory: " << EC.message();

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(DirPath);

  EXPECT_THAT(Result,
              FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                               HasSubstr("path is a directory, not a file"))));
}

TEST_F(JSONFormatTest, BrokenSymlink) {
#ifdef _WIN32
  GTEST_SKIP() << "Symlink test skipped on Windows";
#else
  SmallString<128> SymlinkPath = TestDir;
  sys::path::append(SymlinkPath, "symlink.json");

  // Create a symlink to a non-existent file
  SmallString<128> NonexistentTarget = TestDir;
  sys::path::append(NonexistentTarget, "does_not_exist.json");

  std::error_code EC = sys::fs::create_link(NonexistentTarget, SymlinkPath);
  if (EC) {
    GTEST_SKIP() << "Failed to create symlink (may need elevated privileges): "
                 << EC.message();
  }

  // Verify the symlink points to a non-existent file by checking file status
  sys::fs::file_status Status;
  EC = sys::fs::status(SymlinkPath, Status);
  if (!EC) {
    // If status succeeds, the target exists - skip test
    GTEST_SKIP() << "Symlink unexpectedly points to existing file";
  }

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(SymlinkPath);

  EXPECT_THAT(Result, FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                                       HasSubstr("file does not exist"))));
#endif
}

TEST_F(JSONFormatTest, FileWithoutReadPermission) {
#ifdef _WIN32
  GTEST_SKIP() << "Permission test skipped on Windows (uses different ACL "
                  "model)";
#else
  SmallString<128> FilePath = TestDir;
  sys::path::append(FilePath, "no_read.json");

  // Create a file with valid JSON content
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

  // Remove read permissions (chmod 000)
  auto Perms = sys::fs::perms::all_all;
  EC = sys::fs::setPermissions(FilePath, Perms);
  ASSERT_FALSE(EC) << "Failed to set permissions: " << EC.message();

  // Now remove all permissions
  EC = sys::fs::setPermissions(FilePath, static_cast<sys::fs::perms>(0));
  if (EC) {
    GTEST_SKIP() << "Failed to remove permissions (may be running as root): "
                 << EC.message();
  }

  JSONFormat Format(vfs::getRealFileSystem());
  auto Result = Format.readTUSummary(FilePath);

  // Restore permissions for cleanup
  sys::fs::setPermissions(FilePath, sys::fs::perms::all_all);

  EXPECT_THAT(Result, FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                                       HasSubstr("failed to read file"))));
#endif
}

TEST_F(JSONFormatTest, NotJsonExtension) {
  auto Result = readJSON("{}", "test.txt");

  EXPECT_THAT(Result, FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                                       HasSubstr("not a JSON file"))));
}

// ============================================================================
// JSON Syntax Error Tests
// ============================================================================

TEST_F(JSONFormatTest, InvalidSyntax) {
  auto Result = readJSON("{ invalid json }");

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to read JSON object from file"))));
}

TEST_F(JSONFormatTest, NotObject) {
  auto Result = readJSON("[]");

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to read JSON object from file"))));
}

// ============================================================================
// Root Structure Error Tests
// ============================================================================

TEST_F(JSONFormatTest, MissingTUNamespace) {
  auto Result = readJSON(R"({
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                       HasSubstr("missing or invalid field 'tu_namespace'"))));
}

TEST_F(JSONFormatTest, MissingKind) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to deserialize BuildNamespace"),
                          HasSubstr("missing or invalid field 'kind' "
                                    "(expected BuildNamespaceKind)"))));
}

TEST_F(JSONFormatTest, MissingName) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to deserialize BuildNamespace"),
                          HasSubstr("missing or invalid field 'name' "
                                    "(expected string)"))));
}

TEST_F(JSONFormatTest, InvalidKind) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "invalid_kind",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to deserialize BuildNamespace"),
                          HasSubstr("while parsing field 'kind'"),
                          HasSubstr("invalid 'kind' BuildNamespaceKind "
                                    "value"))));
}

TEST_F(JSONFormatTest, MissingIDTable) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "data": []
  })");

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("missing or invalid field 'id_table'"))));
}

TEST_F(JSONFormatTest, MissingData) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": []
  })");

  EXPECT_THAT(Result,
              FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                               HasSubstr("missing or invalid field 'data'"))));
}

// ============================================================================
// ID Table Error Tests
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

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("missing or invalid field 'id_table'"))));
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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to deserialize EntityIdTable"),
          HasSubstr("element at index 0 is not a JSON object"),
          HasSubstr(
              "(expected EntityIdTable entry with 'id' and 'name' fields)"))));
}

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

  EXPECT_THAT(Result,
              FailedWith(AllOf(
                  HasSubstr("reading TUSummary from"),
                  HasSubstr("failed to read JSON array from field 'id_table'"),
                  HasSubstr("failed to deserialize EntityIdTable at index 0"),
                  HasSubstr("failed to deserialize EntityIdTable entry"),
                  HasSubstr("missing or invalid field 'id' "
                            "(expected unsigned integer EntityId)"))));
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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'id_table'"),
          HasSubstr("failed to deserialize EntityIdTable at index 0"),
          HasSubstr("failed to deserialize EntityIdTable entry"),
          HasSubstr("failed to read JSON object from field 'name'"),
          HasSubstr("missing or invalid field 'name' (expected EntityName JSON "
                    "object)"))));
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

  EXPECT_THAT(
      Result,
      FailedWith(
          AllOf(HasSubstr("reading TUSummary from"),
                HasSubstr("failed to read JSON array from field 'id_table'"),
                HasSubstr("failed to deserialize EntityIdTable at index 0"),
                HasSubstr("failed to deserialize EntityIdTable entry"),
                HasSubstr("field 'id' is not a valid unsigned 64-bit integer"),
                HasSubstr("(expected non-negative EntityId value)"))));
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

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to deserialize EntityIdTable"),
                          HasSubstr("duplicate EntityName found at index"),
                          HasSubstr("(EntityId=0 already exists in table)"))));
}

// ============================================================================
// Entity Name Error Tests
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

  EXPECT_THAT(Result,
              FailedWith(AllOf(
                  HasSubstr("reading TUSummary from"),
                  HasSubstr("failed to read JSON array from field 'id_table'"),
                  HasSubstr("failed to deserialize EntityIdTable at index 0"),
                  HasSubstr("failed to deserialize EntityIdTable entry"),
                  HasSubstr("failed to read JSON object from field 'name'"),
                  HasSubstr("failed to deserialize EntityName"),
                  HasSubstr("missing or invalid field 'usr' "
                            "(expected string (Unified Symbol Resolution))"))));
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

  EXPECT_THAT(Result,
              FailedWith(AllOf(
                  HasSubstr("reading TUSummary from"),
                  HasSubstr("failed to read JSON array from field 'id_table'"),
                  HasSubstr("failed to deserialize EntityIdTable at "
                            "index 0"),
                  HasSubstr("failed to deserialize EntityIdTable entry"),
                  HasSubstr("failed to read JSON object from field 'name'"),
                  HasSubstr("failed to deserialize EntityName"),
                  HasSubstr("missing or invalid field 'suffix' "
                            "(expected string)"))));
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

  EXPECT_THAT(
      Result,
      FailedWith(
          AllOf(HasSubstr("reading TUSummary from"),
                HasSubstr("failed to read JSON array from field 'id_table'"),
                HasSubstr("failed to deserialize EntityIdTable at index 0"),
                HasSubstr("failed to deserialize EntityIdTable entry"),
                HasSubstr("failed to read JSON object from field 'name'"),
                HasSubstr("failed to deserialize EntityName"),
                HasSubstr("failed to read JSON array from field 'namespace'"),
                HasSubstr("missing or invalid field 'namespace'"),
                HasSubstr("(expected JSON array of BuildNamespace objects)"))));
}

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

  EXPECT_THAT(Result,
              FailedWith(AllOf(
                  HasSubstr("reading TUSummary from"),
                  HasSubstr("failed to read JSON array from field 'id_table'"),
                  HasSubstr("failed to deserialize EntityIdTable at index 0"),
                  HasSubstr("failed to deserialize EntityIdTable entry"),
                  HasSubstr("failed to read JSON object from field 'name'"),
                  HasSubstr("failed to deserialize EntityName"),
                  HasSubstr("failed to read JSON array from field 'namespace'"),
                  HasSubstr("failed to deserialize NestedBuildNamespace"),
                  HasSubstr("element at index 0 is not a JSON object"))));
}

// ============================================================================
// Data Array Error Tests
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

  EXPECT_THAT(Result,
              FailedWith(AllOf(HasSubstr("reading TUSummary from"),
                               HasSubstr("missing or invalid field 'data'"))));
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

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to deserialize SummaryDataMap"),
                          HasSubstr("element at index 0 is not a JSON object"),
                          HasSubstr("(expected SummaryDataMap entry with "
                                    "'summary_name' and 'summary_data'"),
                          HasSubstr("fields)"))));
}

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

  EXPECT_THAT(
      Result,
      FailedWith(
          AllOf(HasSubstr("reading TUSummary from"),
                HasSubstr("failed to read JSON array from field 'data'"),
                HasSubstr("failed to deserialize SummaryDataMap at "
                          "index 0"),
                HasSubstr("failed to deserialize SummaryDataMap entry"),
                HasSubstr("missing or invalid field 'summary_name' "
                          "(expected string (analysis summary identifier))"))));
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

  EXPECT_THAT(Result,
              FailedWith(AllOf(
                  HasSubstr("reading TUSummary from"),
                  HasSubstr("failed to read JSON array from field 'data'"),
                  HasSubstr("failed to deserialize SummaryDataMap at index 0"),
                  HasSubstr("failed to deserialize SummaryDataMap entry"),
                  HasSubstr("missing or invalid field 'summary_data'"))));
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

  EXPECT_THAT(Result, FailedWith(AllOf(
                          HasSubstr("reading TUSummary from"),
                          HasSubstr("failed to deserialize SummaryDataMap"),
                          HasSubstr("duplicate SummaryName 'test_summary' "
                                    "found at index"))));
}

// ============================================================================
// Entity Data Error Tests
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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'data'"),
          HasSubstr("failed to deserialize SummaryDataMap at index 0"),
          HasSubstr("failed to deserialize SummaryDataMap entry"),
          HasSubstr("for summary 'test_summary'"),
          HasSubstr("failed to read JSON array from field 'summary_data'"),
          HasSubstr("failed to deserialize EntityDataMap"),
          HasSubstr("element at index 0 is not a JSON object"),
          HasSubstr("(expected EntityDataMap entry with 'entity_id' and "
                    "'entity_summary'"),
          HasSubstr("fields)"))));
}

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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'data'"),
          HasSubstr("failed to deserialize SummaryDataMap at index 0"),
          HasSubstr("failed to deserialize SummaryDataMap entry"),
          HasSubstr("for summary 'test_summary'"),
          HasSubstr("failed to read JSON array from field 'summary_data'"),
          HasSubstr("failed to deserialize EntityDataMap at index 0"),
          HasSubstr("failed to deserialize EntityDataMap entry"),
          HasSubstr("missing or invalid field 'entity_id' "
                    "(expected unsigned integer EntityId)"))));
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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'data'"),
          HasSubstr("failed to deserialize SummaryDataMap at index 0"),
          HasSubstr("failed to deserialize SummaryDataMap entry"),
          HasSubstr("for summary 'test_summary'"),
          HasSubstr("failed to read JSON array from field 'summary_data'"),
          HasSubstr("failed to deserialize EntityDataMap at index 0"),
          HasSubstr("failed to deserialize EntityDataMap entry"),
          HasSubstr("failed to read JSON object from field 'entity_summary'"),
          HasSubstr("missing or invalid field 'entity_summary'"),
          HasSubstr("(expected EntitySummary JSON object)"))));
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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'data'"),
          HasSubstr("failed to deserialize SummaryDataMap at index 0"),
          HasSubstr("failed to deserialize SummaryDataMap entry"),
          HasSubstr("for summary 'test_summary'"),
          HasSubstr("failed to read JSON array from field 'summary_data'"),
          HasSubstr("failed to deserialize EntityDataMap at index 0"),
          HasSubstr("failed to deserialize EntityDataMap entry"),
          HasSubstr("field 'entity_id' is not a valid unsigned 64-bit integer"),
          HasSubstr("(expected non-negative EntityId value)"))));
}

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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'data'"),
          HasSubstr("failed to deserialize SummaryDataMap at index 0"),
          HasSubstr("failed to deserialize SummaryDataMap entry"),
          HasSubstr("for summary 'unknown_summary_type'"),
          HasSubstr("failed to read JSON array from field 'summary_data'"),
          HasSubstr("failed to deserialize EntityDataMap at index 0"),
          HasSubstr("failed to deserialize EntityDataMap entry"),
          HasSubstr("failed to read JSON object from field 'entity_summary'"),
          HasSubstr("failed to deserialize EntitySummary"),
          HasSubstr("no FormatInfo was registered for summary name: "
                    "unknown_summary_type"))));
}

// ============================================================================
// Analysis-Specific Error Tests - TestAnalysis
// ============================================================================

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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'data'"),
          HasSubstr("failed to deserialize SummaryDataMap at "
                    "index 0"),
          HasSubstr("failed to deserialize SummaryDataMap entry"),
          HasSubstr("for summary 'test_summary'"),
          HasSubstr("failed to read JSON array from field 'summary_data'"),
          HasSubstr("failed to deserialize EntityDataMap at "
                    "index 0"),
          HasSubstr("failed to deserialize EntityDataMap entry"),
          HasSubstr("failed to read JSON object from field 'entity_summary'"),
          HasSubstr("missing required field 'pairs'"))));
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

  EXPECT_THAT(
      Result,
      FailedWith(AllOf(
          HasSubstr("reading TUSummary from"),
          HasSubstr("failed to read JSON array from field 'data'"),
          HasSubstr("failed to deserialize SummaryDataMap at "
                    "index 0"),
          HasSubstr("failed to deserialize SummaryDataMap entry"),
          HasSubstr("for summary 'test_summary'"),
          HasSubstr("failed to read JSON array from field 'summary_data'"),
          HasSubstr("failed to deserialize EntityDataMap at "
                    "index 0"),
          HasSubstr("failed to deserialize EntityDataMap entry"),
          HasSubstr("failed to read JSON object from field 'entity_summary'"),
          HasSubstr("missing or invalid 'second' field at "
                    "index 0"))));
}

// ============================================================================
// Valid Configuration Tests
// ============================================================================

TEST_F(JSONFormatTest, Empty) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT(Result, Succeeded());
}

TEST_F(JSONFormatTest, LinkUnit) {
  auto Result = readJSON(R"({
    "tu_namespace": {
      "kind": "link_unit",
      "name": "libtest.so"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT(Result, Succeeded());
}

TEST_F(JSONFormatTest, WithIDTable) {
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

  EXPECT_THAT(Result, Succeeded());
}

TEST_F(JSONFormatTest, WithEmptyDataEntry) {
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
      }
    ]
  })");

  EXPECT_THAT(Result, Succeeded());
}

// ============================================================================
// Round-Trip Tests
// ============================================================================

TEST_F(JSONFormatTest, RoundTripEmpty) {
  testRoundTrip(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatTest, RoundTripWithIDTable) {
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

TEST_F(JSONFormatTest, RoundTripLinkUnit) {
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

TEST_F(JSONFormatTest, RoundTripTestAnalysis) {
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
