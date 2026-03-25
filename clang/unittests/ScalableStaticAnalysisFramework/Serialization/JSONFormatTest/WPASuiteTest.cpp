//===- WPASuiteTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for SSAF JSON serialization format reading and writing of
// WPASuite.
//
//===----------------------------------------------------------------------===//

#include "JSONFormatTest.h"

#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
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
// First Test AnalysisResult - Tags (no entity ID references)
// ============================================================================

struct TagsAnalysisResultForJSONFormatTest final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("TagsAnalysisResultForJSONFormatTest");
  }

  std::vector<std::string> Tags;
};

json::Object serializeTagsAnalysisResult(const AnalysisResult &Result,
                                         JSONFormat::EntityIdToJSONFn) {
  const auto &R =
      static_cast<const TagsAnalysisResultForJSONFormatTest &>(Result);
  json::Array TagsArray;
  for (const auto &Tag : R.Tags)
    TagsArray.push_back(Tag);
  return json::Object{{"tags", std::move(TagsArray)}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeTagsAnalysisResult(const json::Object &Obj,
                              JSONFormat::EntityIdFromJSONFn) {
  const json::Array *TagsArray = Obj.getArray("tags");
  if (!TagsArray)
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'tags'");

  auto R = std::make_unique<TagsAnalysisResultForJSONFormatTest>();
  for (const auto &[Index, Val] : llvm::enumerate(*TagsArray)) {
    auto S = Val.getAsString();
    if (!S)
      return createStringError(inconvertibleErrorCode(),
                               "tags element at index %zu is not a string",
                               Index);
    R->Tags.push_back(S->str());
  }
  return std::move(R);
}

JSONFormat::AnalysisResultRegistryGenerator::Add<
    TagsAnalysisResultForJSONFormatTest>
    RegisterTagsAnalysisFormatInfo(serializeTagsAnalysisResult,
                                   deserializeTagsAnalysisResult);

// ============================================================================
// Second Test AnalysisResult - Counts (with entity ID references)
// ============================================================================

struct CountsAnalysisResultForJSONFormatTest final : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("CountsAnalysisResultForJSONFormatTest");
  }

  std::vector<std::pair<EntityId, int>> Counts;
};

json::Object
serializeCountsAnalysisResult(const AnalysisResult &Result,
                              JSONFormat::EntityIdToJSONFn ToJSON) {
  const auto &R =
      static_cast<const CountsAnalysisResultForJSONFormatTest &>(Result);
  json::Array CountsArray;
  for (const auto &[EI, Count] : R.Counts) {
    CountsArray.push_back(
        json::Object{{"entity_id", ToJSON(EI)}, {"count", Count}});
  }
  return json::Object{{"counts", std::move(CountsArray)}};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeCountsAnalysisResult(const json::Object &Obj,
                                JSONFormat::EntityIdFromJSONFn FromJSON) {
  const json::Array *CountsArray = Obj.getArray("counts");
  if (!CountsArray)
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'counts'");

  auto R = std::make_unique<CountsAnalysisResultForJSONFormatTest>();
  for (const auto &[Index, Val] : llvm::enumerate(*CountsArray)) {
    const json::Object *Entry = Val.getAsObject();
    if (!Entry)
      return createStringError(inconvertibleErrorCode(),
                               "counts element at index %zu is not an object",
                               Index);
    const json::Object *EIObj = Entry->getObject("entity_id");
    if (!EIObj)
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'entity_id' field at index %zu", Index);
    auto ExpectedEI = FromJSON(*EIObj);
    if (!ExpectedEI)
      return ExpectedEI.takeError();

    auto CountVal = Entry->getInteger("count");
    if (!CountVal)
      return createStringError(inconvertibleErrorCode(),
                               "missing or invalid 'count' field at index %zu",
                               Index);
    R->Counts.emplace_back(*ExpectedEI, static_cast<int>(*CountVal));
  }
  return std::move(R);
}

JSONFormat::AnalysisResultRegistryGenerator::Add<
    CountsAnalysisResultForJSONFormatTest>
    RegisterCountsAnalysisFormatInfo(serializeCountsAnalysisResult,
                                     deserializeCountsAnalysisResult);

// ============================================================================
// FailingDeserializerAnalysisResult - always returns an error on deserialize
// ============================================================================

struct FailingDeserializerAnalysisResultForJSONFormatTest final
    : AnalysisResult {
  static AnalysisName analysisName() {
    return AnalysisName("FailingDeserializerAnalysisResultForJSONFormatTest");
  }
};

json::Object
serializeFailingDeserializerAnalysisResult(const AnalysisResult &,
                                           JSONFormat::EntityIdToJSONFn) {
  return json::Object{};
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeFailingDeserializerAnalysisResult(const json::Object &,
                                             JSONFormat::EntityIdFromJSONFn) {
  return createStringError(inconvertibleErrorCode(),
                           "intentional deserializer failure");
}

JSONFormat::AnalysisResultRegistryGenerator::Add<
    FailingDeserializerAnalysisResultForJSONFormatTest>
    RegisterFailingDeserializerAnalysisFormatInfo(
        serializeFailingDeserializerAnalysisResult,
        deserializeFailingDeserializerAnalysisResult);

// ============================================================================
// JSONFormatWPASuiteTest Fixture
// ============================================================================

class JSONFormatWPASuiteTest : public JSONFormatTest {
protected:
  llvm::Expected<WPASuite>
  readWPASuiteFromString(StringRef JSON,
                         StringRef FileName = "test.json") const {
    auto ExpectedFilePath = writeJSON(JSON, FileName);
    if (!ExpectedFilePath)
      return ExpectedFilePath.takeError();
    return JSONFormat().readWPASuite(makePath(FileName));
  }

  llvm::Expected<WPASuite> readWPASuiteFromFile(StringRef FileName) const {
    return JSONFormat().readWPASuite(makePath(FileName));
  }

  llvm::Error writeWPASuite(const WPASuite &Suite, StringRef FileName) const {
    return JSONFormat().writeWPASuite(Suite, makePath(FileName));
  }

  // Builds an empty WPASuite (no id_table entries, no results) and writes it.
  llvm::Error writeEmptyWPASuite(StringRef FileName) const {
    WPASuite Suite = makeWPASuite();
    return writeWPASuite(Suite, FileName);
  }

  void readWriteCompare(StringRef JSON) const {
    const PathString InputFileName("input.json");
    const PathString OutputFileName("output.json");

    auto ExpectedInputFilePath = writeJSON(JSON, InputFileName);
    ASSERT_THAT_EXPECTED(ExpectedInputFilePath, Succeeded());

    auto ExpectedSuite = readWPASuiteFromFile(InputFileName);
    ASSERT_THAT_EXPECTED(ExpectedSuite, Succeeded());

    ASSERT_THAT_ERROR(writeWPASuite(*ExpectedSuite, OutputFileName),
                      Succeeded());

    auto ExpectedInputJSON = readJSONFromFile(InputFileName);
    ASSERT_THAT_EXPECTED(ExpectedInputJSON, Succeeded());

    auto ExpectedOutputJSON = readJSONFromFile(OutputFileName);
    ASSERT_THAT_EXPECTED(ExpectedOutputJSON, Succeeded());

    auto ExpectedNormalizedInput =
        normalizeWPASuiteJSON(std::move(*ExpectedInputJSON));
    ASSERT_THAT_EXPECTED(ExpectedNormalizedInput, Succeeded());

    auto ExpectedNormalizedOutput =
        normalizeWPASuiteJSON(std::move(*ExpectedOutputJSON));
    ASSERT_THAT_EXPECTED(ExpectedNormalizedOutput, Succeeded());

    ASSERT_EQ(*ExpectedNormalizedInput, *ExpectedNormalizedOutput)
        << "Serialization is broken: input is different from output\n"
        << "Input:  " << llvm::formatv("{0:2}", *ExpectedNormalizedInput).str()
        << "\n"
        << "Output: "
        << llvm::formatv("{0:2}", *ExpectedNormalizedOutput).str();
  }

private:
  static llvm::Expected<json::Value> normalizeWPASuiteJSON(json::Value Val) {
    auto *Obj = Val.getAsObject();
    if (!Obj)
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize WPASuite JSON: expected object");

    auto *IdTable = Obj->getArray("id_table");
    if (!IdTable)
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize WPASuite JSON: missing 'id_table' array");

    llvm::sort(*IdTable, [](const json::Value &A, const json::Value &B) {
      const auto *OA = A.getAsObject();
      const auto *OB = B.getAsObject();
      if (!OA || !OB)
        return false;
      auto IA = OA->getInteger("id");
      auto IB = OB->getInteger("id");
      return IA && IB && *IA < *IB;
    });

    auto *Results = Obj->getArray("results");
    if (!Results)
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize WPASuite JSON: missing 'results' array");

    llvm::sort(*Results, [](const json::Value &A, const json::Value &B) {
      const auto *OA = A.getAsObject();
      const auto *OB = B.getAsObject();
      if (!OA || !OB)
        return false;
      auto NA = OA->getString("analysis_name");
      auto NB = OB->getString("analysis_name");
      return NA && NB && *NA < *NB;
    });

    return Val;
  }
};

// ============================================================================
// readJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatWPASuiteTest, NonexistentFile) {
  auto Result = readWPASuiteFromFile("nonexistent.json");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading WPASuite from"),
                                      HasSubstr("file does not exist"))));
}

TEST_F(JSONFormatWPASuiteTest, PathIsDirectory) {
  auto ExpectedDirPath = makeDirectory("test_directory.json");
  ASSERT_THAT_EXPECTED(ExpectedDirPath, Succeeded());

  auto Result = readWPASuiteFromFile("test_directory.json");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from"),
                              HasSubstr("path is a directory, not a file"))));
}

TEST_F(JSONFormatWPASuiteTest, NotJsonExtension) {
  auto ExpectedFilePath = writeJSON("{}", "test.txt");
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto Result = JSONFormat().readWPASuite(makePath("test.txt"));

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("failed to read file"),
                              HasSubstr("file does not end with '.json'"))));
}

TEST_F(JSONFormatWPASuiteTest, InvalidSyntax) {
  auto Result = readWPASuiteFromString("{ invalid json }");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                                      HasSubstr("Expected object key"))));
}

TEST_F(JSONFormatWPASuiteTest, NotObject) {
  auto Result = readWPASuiteFromString("[]");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                                      HasSubstr("failed to read WPASuite"),
                                      HasSubstr("expected JSON object"))));
}

// ============================================================================
// Structural Error Tests - id_table
// ============================================================================

TEST_F(JSONFormatWPASuiteTest, MissingIdTable) {
  auto Result = readWPASuiteFromString(R"({"results": []})");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("failed to read IdTable from field "
                                        "'id_table'"),
                              HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatWPASuiteTest, IdTableNotArray) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": {},
    "results": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("failed to read IdTable from field "
                                        "'id_table'"),
                              HasSubstr("expected JSON array"))));
}

// ============================================================================
// Structural Error Tests - results
// ============================================================================

TEST_F(JSONFormatWPASuiteTest, MissingResults) {
  auto Result = readWPASuiteFromString(R"({"id_table": []})");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("failed to read WPA results from field "
                                        "'results'"),
                              HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultsNotArray) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": {}
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("failed to read WPA results from field "
                                        "'results'"),
                              HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultEntryNotObject) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": ["invalid"]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("reading WPA results from field "
                                        "'results'"),
                              HasSubstr("failed to read WPA result entry from "
                                        "index '0'"),
                              HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultEntryMissingAnalysisName) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [{"result": {}}]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("reading WPA results from field "
                                        "'results'"),
                              HasSubstr("reading WPA result entry from index "
                                        "'0'"),
                              HasSubstr("failed to read AnalysisName from "
                                        "field 'analysis_name'"),
                              HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultEntryAnalysisNameNotString) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [{"analysis_name": 42, "result": {}}]
  })");

  EXPECT_THAT_EXPECTED(Result, FailedWithMessage(AllOf(
                                   HasSubstr("reading WPASuite from file"),
                                   HasSubstr("failed to read AnalysisName from "
                                             "field 'analysis_name'"),
                                   HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultEntryNoFormatInfo) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [
      {"analysis_name": "UnregisteredAnalysis", "result": {}}
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(
                  AllOf(HasSubstr("reading WPASuite from file"),
                        HasSubstr("reading WPA results from field 'results'"),
                        HasSubstr("reading WPA result entry from index '0'"),
                        HasSubstr("no support registered for analysis: "
                                  "AnalysisName(UnregisteredAnalysis)"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultEntryMissingResultField) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [
      {"analysis_name": "TagsAnalysisResultForJSONFormatTest"}
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("reading WPA results from field "
                                        "'results'"),
                              HasSubstr("reading WPA result entry from index "
                                        "'0'"),
                              HasSubstr("failed to read AnalysisResult from "
                                        "field 'result'"),
                              HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultEntryResultNotObject) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [
      {"analysis_name": "TagsAnalysisResultForJSONFormatTest",
       "result": "not_an_object"}
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("failed to read AnalysisResult from "
                                        "field 'result'"),
                              HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatWPASuiteTest, ResultEntryDeserializerError) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [
      {"analysis_name": "FailingDeserializerAnalysisResultForJSONFormatTest",
       "result": {}}
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading WPASuite from file"),
                              HasSubstr("reading WPA results from field "
                                        "'results'"),
                              HasSubstr("reading WPA result entry from index "
                                        "'0'"),
                              HasSubstr("intentional deserializer failure"))));
}

TEST_F(JSONFormatWPASuiteTest, DuplicateAnalysisName) {
  auto Result = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [
      {"analysis_name": "TagsAnalysisResultForJSONFormatTest",
       "result": {"tags": []}},
      {"analysis_name": "TagsAnalysisResultForJSONFormatTest",
       "result": {"tags": []}}
    ]
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading WPASuite from file"),
          HasSubstr("reading WPA results from field 'results'"),
          HasSubstr("failed to insert WPA result at index '1'"),
          HasSubstr("encountered duplicate "
                    "'AnalysisName(TagsAnalysisResultForJSONFormatTest)'"))));
}

// ============================================================================
// Write Error Tests
// ============================================================================

TEST_F(JSONFormatWPASuiteTest, WriteFileAlreadyExists) {
  auto ExpectedFilePath = writeJSON("{}", "existing.json");
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto Result = writeEmptyWPASuite("existing.json");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing WPASuite to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("file already exists"))));
}

TEST_F(JSONFormatWPASuiteTest, WriteParentDirectoryNotFound) {
  PathString FilePath = makePath("nonexistent-dir", "test.json");

  auto Result = JSONFormat().writeWPASuite(makeWPASuite(), FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing WPASuite to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("parent directory does not exist"))));
}

TEST_F(JSONFormatWPASuiteTest, WriteNotJsonExtension) {
  auto Result =
      JSONFormat().writeWPASuite(makeWPASuite(), makePath("test.txt"));

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing WPASuite to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("file does not end with '.json'"))));
}

TEST_F(JSONFormatWPASuiteTest, WriteNoFormatInfo) {
  WPASuite Suite = makeWPASuite();
  getData(Suite).emplace(
      AnalysisName("UnregisteredAnalysisForJSONFormatTest"),
      std::make_unique<TagsAnalysisResultForJSONFormatTest>());

  auto Result = writeWPASuite(Suite, "output.json");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(
          HasSubstr("writing WPASuite to file"),
          HasSubstr("no support registered for analysis: "
                    "AnalysisName(UnregisteredAnalysisForJSONFormatTest)"))));
}

// ============================================================================
// Round-Trip Tests
// ============================================================================

TEST_F(JSONFormatWPASuiteTest, RoundTripEmpty) {
  readWriteCompare(R"({
    "id_table": [],
    "results": []
  })");
}

TEST_F(JSONFormatWPASuiteTest, RoundTripSingleResultNoEntities) {
  readWriteCompare(R"({
    "id_table": [],
    "results": [
      {
        "analysis_name": "TagsAnalysisResultForJSONFormatTest",
        "result": {"tags": ["alpha", "beta", "gamma"]}
      }
    ]
  })");
}

TEST_F(JSONFormatWPASuiteTest, RoundTripSingleResultWithEntityRefs) {
  readWriteCompare(R"({
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {"kind": "CompilationUnit", "name": "a.cpp"},
            {"kind": "LinkUnit", "name": "a.exe"}
          ]
        }
      },
      {
        "id": 1,
        "name": {
          "usr": "c:@F@bar",
          "suffix": "",
          "namespace": [
            {"kind": "CompilationUnit", "name": "a.cpp"},
            {"kind": "LinkUnit", "name": "a.exe"}
          ]
        }
      }
    ],
    "results": [
      {
        "analysis_name": "CountsAnalysisResultForJSONFormatTest",
        "result": {
          "counts": [
            {"entity_id": {"@": 0}, "count": 42},
            {"entity_id": {"@": 1}, "count": 7}
          ]
        }
      }
    ]
  })");
}

TEST_F(JSONFormatWPASuiteTest, RoundTripMultipleResults) {
  readWriteCompare(R"({
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {"kind": "LinkUnit", "name": "test.exe"}
          ]
        }
      }
    ],
    "results": [
      {
        "analysis_name": "CountsAnalysisResultForJSONFormatTest",
        "result": {
          "counts": [
            {"entity_id": {"@": 0}, "count": 100}
          ]
        }
      },
      {
        "analysis_name": "TagsAnalysisResultForJSONFormatTest",
        "result": {"tags": ["important"]}
      }
    ]
  })");
}

TEST_F(JSONFormatWPASuiteTest, RoundTripEmptyResultPayload) {
  readWriteCompare(R"({
    "id_table": [],
    "results": [
      {
        "analysis_name": "TagsAnalysisResultForJSONFormatTest",
        "result": {"tags": []}
      }
    ]
  })");
}

// ============================================================================
// Content Verification Tests
// ============================================================================

TEST_F(JSONFormatWPASuiteTest, ReadVerifyTagsResult) {
  auto ExpectedSuite = readWPASuiteFromString(R"({
    "id_table": [],
    "results": [
      {
        "analysis_name": "TagsAnalysisResultForJSONFormatTest",
        "result": {"tags": ["foo", "bar"]}
      }
    ]
  })");

  ASSERT_THAT_EXPECTED(ExpectedSuite, Succeeded());
  const WPASuite &Suite = *ExpectedSuite;

  ASSERT_TRUE(
      Suite.contains(TagsAnalysisResultForJSONFormatTest::analysisName()));
  auto ExpectedResult = Suite.get<TagsAnalysisResultForJSONFormatTest>();
  ASSERT_THAT_EXPECTED(ExpectedResult, Succeeded());

  const auto &R = *ExpectedResult;
  ASSERT_EQ(R.Tags.size(), 2u);
  EXPECT_EQ(R.Tags[0], "foo");
  EXPECT_EQ(R.Tags[1], "bar");
}

TEST_F(JSONFormatWPASuiteTest, ReadVerifyCountsResultWithEntityId) {
  auto ExpectedSuite = readWPASuiteFromString(R"({
    "id_table": [
      {
        "id": 0,
        "name": {
          "usr": "c:@F@foo",
          "suffix": "",
          "namespace": [
            {"kind": "CompilationUnit", "name": "test.cpp"},
            {"kind": "LinkUnit", "name": "test.exe"}
          ]
        }
      }
    ],
    "results": [
      {
        "analysis_name": "CountsAnalysisResultForJSONFormatTest",
        "result": {
          "counts": [
            {"entity_id": {"@": 0}, "count": 99}
          ]
        }
      }
    ]
  })");

  ASSERT_THAT_EXPECTED(ExpectedSuite, Succeeded());
  const WPASuite &Suite = *ExpectedSuite;

  ASSERT_TRUE(
      Suite.contains(CountsAnalysisResultForJSONFormatTest::analysisName()));
  auto ExpectedResult = Suite.get<CountsAnalysisResultForJSONFormatTest>();
  ASSERT_THAT_EXPECTED(ExpectedResult, Succeeded());

  const auto &R = *ExpectedResult;
  ASSERT_EQ(R.Counts.size(), 1u);
  EXPECT_EQ(R.Counts[0].second, 99);
}

} // namespace
