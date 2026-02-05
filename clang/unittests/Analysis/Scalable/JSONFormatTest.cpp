//===- unittests/Analysis/Scalable/JSONFormatTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <fstream>

using namespace clang::ssaf;
using llvm::Failed;
using llvm::Succeeded;
using ::testing::AllOf;
using ::testing::HasSubstr;

namespace {

// Helper function to check that an error message contains all specified
// substrings
::testing::Matcher<std::string>
ContainsAllSubstrings(std::initializer_list<const char *> substrings) {
  std::vector<::testing::Matcher<std::string>> matchers;
  for (const char *substr : substrings) {
    matchers.push_back(HasSubstr(substr));
  }
  return ::testing::AllOfArray(matchers);
}

//===----------------------------------------------------------------------===//
// Test Fixtures and Helpers
//===----------------------------------------------------------------------===//

// Helper class to manage temporary directories for testing
class TempDir {
  llvm::SmallString<128> Path;
  std::error_code EC;

public:
  TempDir() {
    EC = llvm::sys::fs::createUniqueDirectory("JSONFormatTest", Path);
  }

  ~TempDir() {
    if (!EC && llvm::sys::fs::exists(Path)) {
      llvm::sys::fs::remove_directories(Path);
    }
  }

  llvm::StringRef path() const { return Path; }
  bool isValid() const { return !EC; }
};

// Helper function to write a file with content
void writeFile(llvm::StringRef Path, llvm::StringRef Content) {
  std::ofstream File(Path.str());
  File << Content.str();
  File.close();
}

// Helper function to create a directory
void createDir(llvm::StringRef Path) {
  llvm::sys::fs::create_directories(Path);
}

// Base test fixture for JSONFormat tests
class JSONFormatTestBase : public ::testing::Test {
protected:
  JSONFormat Format{llvm::vfs::getRealFileSystem()};
};

//===----------------------------------------------------------------------===//
// TUSummary Read Tests - TUNamespace
//===----------------------------------------------------------------------===//

class JSONFormatReadTUNamespaceTest : public JSONFormatTestBase {};

TEST_F(JSONFormatReadTUNamespaceTest, NonexistentDirectory) {
  auto Result = Format.readTUSummary("/nonexistent/path");
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from '/nonexistent/path'",
                   "failed to read JSON from file", "tu_namespace.json",
                   "failed to validate JSON file", "file does not exist"}));
}

TEST_F(JSONFormatReadTUNamespaceTest, MissingTUNamespaceFile) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg, ContainsAllSubstrings({"failed to read TUSummary from",
                                               "failed to read JSON from file",
                                               "tu_namespace.json",
                                               "file does not exist"}));
}

TEST_F(JSONFormatReadTUNamespaceTest, InvalidJSONSyntax) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath, "{ invalid json }");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg, ContainsAllSubstrings({"failed to read TUSummary from",
                                               "failed to read JSON from file",
                                               "tu_namespace.json"}));
}

TEST_F(JSONFormatReadTUNamespaceTest, NotJSONObject) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath, "[]");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings({"failed to read TUSummary from",
                                     "failed to read JSON object from file",
                                     "tu_namespace.json"}));
}

TEST_F(JSONFormatReadTUNamespaceTest, MissingKindField) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath, R"({"name": "test.cpp"})");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize BuildNamespace from file",
           "tu_namespace.json",
           "missing required field 'kind' (expected BuildNamespaceKind)"}));
}

TEST_F(JSONFormatReadTUNamespaceTest, InvalidKindValue) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath, R"({"kind": "InvalidKind", "name": "test.cpp"})");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize BuildNamespace from file",
                   "tu_namespace.json",
                   "invalid 'kind' BuildNamespaceKind value 'InvalidKind'"}));
}

TEST_F(JSONFormatReadTUNamespaceTest, MissingNameField) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath, R"({"kind": "compilation_unit"})");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize BuildNamespace from file",
                   "tu_namespace.json", "missing required field 'name'"}));
}

//===----------------------------------------------------------------------===//
// TUSummary Read Tests - IdTable
//===----------------------------------------------------------------------===//

class JSONFormatReadIdTableTest : public JSONFormatTestBase {
protected:
  void SetUpValidTUNamespace(TempDir &Dir) {
    llvm::SmallString<128> TUNamespacePath(Dir.path());
    llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
    writeFile(TUNamespacePath,
              R"({"kind": "compilation_unit", "name": "test.cpp"})");
  }
};

TEST_F(JSONFormatReadIdTableTest, MissingIdTableFile) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespace(Dir);

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings({"failed to read TUSummary from",
                                     "failed to read JSON from file",
                                     "id_table.json", "file does not exist"}));
}

TEST_F(JSONFormatReadIdTableTest, NotJSONArray) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespace(Dir);

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, "{}");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings({"failed to read TUSummary from",
                                     "failed to read JSON array from file",
                                     "id_table.json"}));
}

TEST_F(JSONFormatReadIdTableTest, ElementNotObject) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespace(Dir);

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, "[\"not an object\"]");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize EntityIdTable from file", "id_table.json",
           "element at index 0 is not a JSON object",
           "expected EntityIdTable entry with 'id' and 'name' fields"}));
}

TEST_F(JSONFormatReadIdTableTest, EntryMissingName) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespace(Dir);

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, R"([{"id": 0}])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize EntityIdTable entry from file",
                   "id_table.json",
                   "missing or invalid field 'name' (expected EntityName JSON "
                   "object)"}));
}

TEST_F(JSONFormatReadIdTableTest, EntryMissingId) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespace(Dir);

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, R"([{
    "name": {
      "usr": "c:@F@foo",
      "suffix": "",
      "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
    }
  }])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize EntityIdTable entry from file",
                   "id_table.json",
                   "missing required field 'id' (expected unsigned integer "
                   "EntityId)"}));
}

TEST_F(JSONFormatReadIdTableTest, EntryIdNotInteger) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespace(Dir);

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, R"([{
    "id": "not a number",
    "name": {
      "usr": "c:@F@foo",
      "suffix": "",
      "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
    }
  }])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize EntityIdTable entry from file",
           "id_table.json", "field 'id' is not a valid unsigned 64-bit integer",
           "expected non-negative EntityId value"}));
}

TEST_F(JSONFormatReadIdTableTest, DuplicateEntityName) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespace(Dir);

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, R"([
    {
      "id": 0,
      "name": {
        "usr": "c:@F@foo",
        "suffix": "",
        "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
      }
    },
    {
      "id": 1,
      "name": {
        "usr": "c:@F@foo",
        "suffix": "",
        "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
      }
    }
  ])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize EntityIdTable from file",
                   "id_table.json", "duplicate EntityName found at index 1",
                   "EntityId=0 already exists in table"}));
}

//===----------------------------------------------------------------------===//
// TUSummary Read Tests - EntityName
//===----------------------------------------------------------------------===//

class JSONFormatReadEntityNameTest : public JSONFormatTestBase {
protected:
  void SetUpValidTUNamespaceAndPartialIdTable(TempDir &Dir,
                                              llvm::StringRef EntityNameJson) {
    llvm::SmallString<128> TUNamespacePath(Dir.path());
    llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
    writeFile(TUNamespacePath,
              R"({"kind": "compilation_unit", "name": "test.cpp"})");

    llvm::SmallString<128> IdTablePath(Dir.path());
    llvm::sys::path::append(IdTablePath, "id_table.json");
    std::string JsonContent = "[{\"id\": 0, \"name\": ";
    JsonContent += EntityNameJson.str();
    JsonContent += "}]";
    writeFile(IdTablePath, JsonContent);
  }
};

TEST_F(JSONFormatReadEntityNameTest, MissingUSR) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "suffix": "",
    "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize EntityName from file", "id_table.json",
           "missing required field 'usr' (Unified Symbol Resolution string)"}));
}

TEST_F(JSONFormatReadEntityNameTest, MissingSuffix) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "usr": "c:@F@foo",
    "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize EntityName from file",
                   "id_table.json", "missing required field 'suffix'"}));
}

TEST_F(JSONFormatReadEntityNameTest, MissingNamespace) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "usr": "c:@F@foo",
    "suffix": ""
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize EntityName from file",
                   "id_table.json", "missing or invalid field 'namespace'",
                   "expected JSON array of BuildNamespace objects"}));
}

TEST_F(JSONFormatReadEntityNameTest, NamespaceNotArray) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "usr": "c:@F@foo",
    "suffix": "",
    "namespace": "not an array"
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize EntityName from file",
                   "id_table.json", "missing or invalid field 'namespace'",
                   "expected JSON array of BuildNamespace objects"}));
}

TEST_F(JSONFormatReadEntityNameTest, NamespaceElementNotObject) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "usr": "c:@F@foo",
    "suffix": "",
    "namespace": ["not an object"]
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize NestedBuildNamespace from file",
                   "id_table.json", "element at index 0 is not a JSON object",
                   "expected BuildNamespace object"}));
}

TEST_F(JSONFormatReadEntityNameTest, NamespaceMissingKind) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "usr": "c:@F@foo",
    "suffix": "",
    "namespace": [{"name": "test.cpp"}]
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize NestedBuildNamespace from file",
           "id_table.json", "at index 0",
           "failed to deserialize BuildNamespace from file",
           "missing required field 'kind' (expected BuildNamespaceKind)"}));
}

TEST_F(JSONFormatReadEntityNameTest, NamespaceInvalidKind) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "usr": "c:@F@foo",
    "suffix": "",
    "namespace": [{"kind": "InvalidKind", "name": "test.cpp"}]
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize NestedBuildNamespace from file",
                   "id_table.json", "at index 0",
                   "failed to deserialize BuildNamespace from file",
                   "invalid 'kind' BuildNamespaceKind value 'InvalidKind'"}));
}

TEST_F(JSONFormatReadEntityNameTest, NamespaceMissingName) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndPartialIdTable(Dir, R"({
    "usr": "c:@F@foo",
    "suffix": "",
    "namespace": [{"kind": "compilation_unit"}]
  })");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize NestedBuildNamespace from file",
                   "id_table.json", "at index 0",
                   "failed to deserialize BuildNamespace from file",
                   "missing required field 'name'"}));
}

//===----------------------------------------------------------------------===//
// TUSummary Read Tests - Data Directory and Files
//===----------------------------------------------------------------------===//

class JSONFormatReadDataTest : public JSONFormatTestBase {
protected:
  void SetUpValidTUNamespaceAndIdTable(TempDir &Dir) {
    llvm::SmallString<128> TUNamespacePath(Dir.path());
    llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
    writeFile(TUNamespacePath,
              R"({"kind": "compilation_unit", "name": "test.cpp"})");

    llvm::SmallString<128> IdTablePath(Dir.path());
    llvm::sys::path::append(IdTablePath, "id_table.json");
    writeFile(IdTablePath, "[]");
  }
};

TEST_F(JSONFormatReadDataTest, MissingDataDirectory) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings({"failed to read TUSummary from",
                                     "data directory does not exist"}));
}

TEST_F(JSONFormatReadDataTest, DataPathIsFile) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  writeFile(DataPath, "content");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings({"failed to read TUSummary from",
                                     "data path is not a directory"}));
}

TEST_F(JSONFormatReadDataTest, NonJSONFileInDataDirectory) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  llvm::SmallString<128> SummaryPath(DataPath);
  llvm::sys::path::append(SummaryPath, "summary.txt");
  writeFile(SummaryPath, "{}");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to read TUSummary data from file", "summary.txt",
                   "failed to validate JSON file", "not a JSON file"}));
}

TEST_F(JSONFormatReadDataTest, FileNotJSONObject) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  llvm::SmallString<128> SummaryPath(DataPath);
  llvm::sys::path::append(SummaryPath, "summary.json");
  writeFile(SummaryPath, "[]");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings({"failed to read TUSummary from",
                                     "failed to read TUSummary data from file",
                                     "summary.json",
                                     "failed to read JSON object from file"}));
}

TEST_F(JSONFormatReadDataTest, MissingSummaryName) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  llvm::SmallString<128> SummaryPath(DataPath);
  llvm::sys::path::append(SummaryPath, "summary.json");
  writeFile(SummaryPath, R"({"summary_data": []})");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize summary data from file",
                   "summary.json", "missing required field 'summary_name'",
                   "expected string identifier for the analysis summary"}));
}

TEST_F(JSONFormatReadDataTest, MissingSummaryData) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  llvm::SmallString<128> SummaryPath(DataPath);
  llvm::sys::path::append(SummaryPath, "summary.json");
  writeFile(SummaryPath, R"({"summary_name": "test"})");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg, ContainsAllSubstrings(
                            {"failed to read TUSummary from",
                             "failed to deserialize summary data from file",
                             "summary.json", "for summary 'test'",
                             "missing or invalid field 'summary_data'",
                             "expected JSON array of entity summaries"}));
}

TEST_F(JSONFormatReadDataTest, SummaryDataNotArray) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  llvm::SmallString<128> SummaryPath(DataPath);
  llvm::sys::path::append(SummaryPath, "summary.json");
  writeFile(SummaryPath,
            R"({"summary_name": "test", "summary_data": "not an array"})");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg, ContainsAllSubstrings(
                            {"failed to read TUSummary from",
                             "failed to deserialize summary data from file",
                             "summary.json", "for summary 'test'",
                             "missing or invalid field 'summary_data'",
                             "expected JSON array of entity summaries"}));
}

TEST_F(JSONFormatReadDataTest, DuplicateSummaryName) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceAndIdTable(Dir);

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  llvm::SmallString<128> SummaryPath1(DataPath);
  llvm::sys::path::append(SummaryPath1, "summary1.json");
  writeFile(SummaryPath1, R"({"summary_name": "test", "summary_data": []})");

  llvm::SmallString<128> SummaryPath2(DataPath);
  llvm::sys::path::append(SummaryPath2, "summary2.json");
  writeFile(SummaryPath2, R"({"summary_name": "test", "summary_data": []})");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from", "failed to read TUSummary data",
           "duplicate SummaryName 'test' encountered in file"}));
}

//===----------------------------------------------------------------------===//
// TUSummary Read Tests - Entity Data
//===----------------------------------------------------------------------===//

class JSONFormatReadEntityDataTest : public JSONFormatTestBase {
protected:
  void SetUpValidTUNamespaceIdTableAndDataDir(TempDir &Dir,
                                              llvm::StringRef SummaryDataJson) {
    llvm::SmallString<128> TUNamespacePath(Dir.path());
    llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
    writeFile(TUNamespacePath,
              R"({"kind": "compilation_unit", "name": "test.cpp"})");

    llvm::SmallString<128> IdTablePath(Dir.path());
    llvm::sys::path::append(IdTablePath, "id_table.json");
    writeFile(IdTablePath, "[]");

    llvm::SmallString<128> DataPath(Dir.path());
    llvm::sys::path::append(DataPath, "data");
    createDir(DataPath);

    llvm::SmallString<128> SummaryPath(DataPath);
    llvm::sys::path::append(SummaryPath, "summary.json");
    std::string JsonContent = R"({"summary_name": "test", "summary_data": )";
    JsonContent += SummaryDataJson.str();
    JsonContent += "}";
    writeFile(SummaryPath, JsonContent);
  }
};

TEST_F(JSONFormatReadEntityDataTest, ElementNotObject) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceIdTableAndDataDir(Dir, "[\"not an object\"]");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize entity data map from file", "summary.json",
           "element at index 0 is not a JSON object",
           "expected object with 'entity_id' and 'entity_summary' fields"}));
}

TEST_F(JSONFormatReadEntityDataTest, MissingEntityId) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceIdTableAndDataDir(Dir, R"([{"entity_summary": {}}])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize entity data map entry from file",
           "summary.json", "at index 0", "missing required field 'entity_id'",
           "expected unsigned integer EntityId"}));
}

TEST_F(JSONFormatReadEntityDataTest, EntityIdNotInteger) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceIdTableAndDataDir(
      Dir, R"([{"entity_id": "not a number", "entity_summary": {}}])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(
      ErrorMsg,
      ContainsAllSubstrings(
          {"failed to read TUSummary from",
           "failed to deserialize entity data map entry from file",
           "summary.json", "at index 0",
           "field 'entity_id' is not a valid unsigned 64-bit integer"}));
}

TEST_F(JSONFormatReadEntityDataTest, MissingEntitySummary) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceIdTableAndDataDir(Dir, R"([{"entity_id": 0}])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize entity data map entry from file",
                   "summary.json", "at index 0",
                   "missing or invalid field 'entity_summary'",
                   "expected EntitySummary JSON object"}));
}

TEST_F(JSONFormatReadEntityDataTest, EntitySummaryNotObject) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceIdTableAndDataDir(
      Dir, R"([{"entity_id": 0, "entity_summary": "not an object"}])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize entity data map entry from file",
                   "summary.json", "at index 0",
                   "missing or invalid field 'entity_summary'",
                   "expected EntitySummary JSON object"}));
}

TEST_F(JSONFormatReadEntityDataTest,
       EntitySummaryDeserializationNotImplemented) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());
  SetUpValidTUNamespaceIdTableAndDataDir(
      Dir, R"([{"entity_id": 0, "entity_summary": {}}])");

  auto Result = Format.readTUSummary(Dir.path());
  ASSERT_FALSE(Result);
  std::string ErrorMsg = llvm::toString(Result.takeError());
  EXPECT_THAT(ErrorMsg,
              ContainsAllSubstrings(
                  {"failed to read TUSummary from",
                   "failed to deserialize entity data map entry from file",
                   "summary.json", "at index 0",
                   "EntitySummary deserialization from file",
                   "is not yet implemented"}));
}

// Note: DuplicateEntityId test cannot be implemented without EntitySummary
// deserialization support, as the error occurs during EntitySummary parsing
// before the duplicate check is reached.

//===----------------------------------------------------------------------===//
// TUSummary Write Tests
//===----------------------------------------------------------------------===//

class JSONFormatWriteTest : public JSONFormatTestBase {};

TEST_F(JSONFormatWriteTest, Success) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  BuildNamespace TUNamespace = BuildNamespace::makeCompilationUnit("test.cpp");
  TUSummary Summary(TUNamespace);

  auto Error = Format.writeTUSummary(Summary, Dir.path());
  EXPECT_THAT_ERROR(std::move(Error), Succeeded());

  // Verify files were created
  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  EXPECT_TRUE(llvm::sys::fs::exists(TUNamespacePath));

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  EXPECT_TRUE(llvm::sys::fs::exists(IdTablePath));

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  EXPECT_TRUE(llvm::sys::fs::exists(DataPath));
  EXPECT_TRUE(llvm::sys::fs::is_directory(DataPath));
}

TEST_F(JSONFormatWriteTest, DataDirectoryExistsAsFile) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  // Create 'data' as a file instead of directory
  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  writeFile(DataPath, "content");

  BuildNamespace TUNamespace = BuildNamespace::makeCompilationUnit("test.cpp");
  TUSummary Summary(TUNamespace);

  auto Error = Format.writeTUSummary(Summary, Dir.path());
  ASSERT_TRUE(!!Error);
  std::string ErrorMsg = llvm::toString(std::move(Error));
  EXPECT_THAT(ErrorMsg, ContainsAllSubstrings(
                            {"failed to write TUSummary to",
                             "data path exists but is not a directory"}));
}

//===----------------------------------------------------------------------===//
// TUSummary Success Cases
//===----------------------------------------------------------------------===//

class JSONFormatSuccessTest : public JSONFormatTestBase {};

TEST_F(JSONFormatSuccessTest, ReadWithEmptyIdTable) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath,
            R"({"kind": "compilation_unit", "name": "test.cpp"})");

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, "[]");

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  auto Result = Format.readTUSummary(Dir.path());
  EXPECT_THAT_ERROR(Result.takeError(), Succeeded());
}

TEST_F(JSONFormatSuccessTest, ReadWithNonEmptyIdTable) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath,
            R"({"kind": "compilation_unit", "name": "test.cpp"})");

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, R"([
    {
      "id": 0,
      "name": {
        "usr": "c:@F@foo",
        "suffix": "",
        "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
      }
    },
    {
      "id": 1,
      "name": {
        "usr": "c:@F@bar",
        "suffix": "1",
        "namespace": [
          {"kind": "compilation_unit", "name": "test.cpp"},
          {"kind": "link_unit", "name": "libtest.so"}
        ]
      }
    }
  ])");

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  auto Result = Format.readTUSummary(Dir.path());
  EXPECT_THAT_ERROR(Result.takeError(), Succeeded());
}

TEST_F(JSONFormatSuccessTest, ReadWithEmptyData) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath,
            R"({"kind": "compilation_unit", "name": "test.cpp"})");

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, "[]");

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  // Add an empty summary data file
  llvm::SmallString<128> SummaryPath(DataPath);
  llvm::sys::path::append(SummaryPath, "summary.json");
  writeFile(SummaryPath, R"({"summary_name": "test", "summary_data": []})");

  auto Result = Format.readTUSummary(Dir.path());
  EXPECT_THAT_ERROR(Result.takeError(), Succeeded());
}

TEST_F(JSONFormatSuccessTest, ReadWithLinkUnitNamespace) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath, R"({"kind": "link_unit", "name": "libtest.so"})");

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, "[]");

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  auto Result = Format.readTUSummary(Dir.path());
  EXPECT_THAT_ERROR(Result.takeError(), Succeeded());
}

//===----------------------------------------------------------------------===//
// Round-Trip Tests
//===----------------------------------------------------------------------===//

class JSONFormatRoundTripTest : public JSONFormatTestBase {};

TEST_F(JSONFormatRoundTripTest, EmptyIdTable) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  BuildNamespace TUNamespace = BuildNamespace::makeCompilationUnit("test.cpp");
  TUSummary Summary(TUNamespace);

  auto WriteError = Format.writeTUSummary(Summary, Dir.path());
  EXPECT_THAT_ERROR(std::move(WriteError), Succeeded());

  auto ReadResult = Format.readTUSummary(Dir.path());
  EXPECT_THAT_ERROR(ReadResult.takeError(), Succeeded());
}

TEST_F(JSONFormatRoundTripTest, NonEmptyIdTable) {
  TempDir Dir;
  ASSERT_TRUE(Dir.isValid());

  // Manually create the files to test roundtrip
  llvm::SmallString<128> TUNamespacePath(Dir.path());
  llvm::sys::path::append(TUNamespacePath, "tu_namespace.json");
  writeFile(TUNamespacePath,
            R"({"kind": "compilation_unit", "name": "test.cpp"})");

  llvm::SmallString<128> IdTablePath(Dir.path());
  llvm::sys::path::append(IdTablePath, "id_table.json");
  writeFile(IdTablePath, R"([
    {
      "id": 0,
      "name": {
        "usr": "c:@F@foo",
        "suffix": "",
        "namespace": [{"kind": "compilation_unit", "name": "test.cpp"}]
      }
    }
  ])");

  llvm::SmallString<128> DataPath(Dir.path());
  llvm::sys::path::append(DataPath, "data");
  createDir(DataPath);

  auto ReadResult = Format.readTUSummary(Dir.path());
  ASSERT_THAT_EXPECTED(ReadResult, Succeeded());

  TempDir Dir2;
  ASSERT_TRUE(Dir2.isValid());

  auto WriteError = Format.writeTUSummary(*ReadResult, Dir2.path());
  EXPECT_THAT_ERROR(std::move(WriteError), Succeeded());

  // Verify the written files
  llvm::SmallString<128> TUNamespacePath2(Dir2.path());
  llvm::sys::path::append(TUNamespacePath2, "tu_namespace.json");
  EXPECT_TRUE(llvm::sys::fs::exists(TUNamespacePath2));
}

} // namespace
