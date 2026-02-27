//===- TUSummaryTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for SSAF JSON serialization format reading and writing of
// TUSummary.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "JSONFormatTest.h"
#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "llvm/Support/Registry.h"
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
// First Test Analysis - Simple analysis for testing JSON serialization.
// ============================================================================

struct PairsEntitySummaryForJSONFormatTest final : EntitySummary {

  SummaryName getSummaryName() const override {
    return SummaryName("PairsEntitySummaryForJSONFormatTest");
  }

  std::vector<std::pair<EntityId, EntityId>> Pairs;
};

static json::Object serializePairsEntitySummaryForJSONFormatTest(
    const EntitySummary &Summary,
    const JSONFormat::EntityIdConverter &Converter) {
  const auto &TA =
      static_cast<const PairsEntitySummaryForJSONFormatTest &>(Summary);
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
deserializePairsEntitySummaryForJSONFormatTest(
    const json::Object &Obj, EntityIdTable &IdTable,
    const JSONFormat::EntityIdConverter &Converter) {
  auto Result = std::make_unique<PairsEntitySummaryForJSONFormatTest>();
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

struct PairsEntitySummaryForJSONFormatTestFormatInfo final
    : JSONFormat::FormatInfo {
  PairsEntitySummaryForJSONFormatTestFormatInfo()
      : JSONFormat::FormatInfo(
            SummaryName("PairsEntitySummaryForJSONFormatTest"),
            serializePairsEntitySummaryForJSONFormatTest,
            deserializePairsEntitySummaryForJSONFormatTest) {}
};

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    PairsEntitySummaryForJSONFormatTestFormatInfo>
    RegisterPairsEntitySummaryForJSONFormatTest(
        "PairsEntitySummaryForJSONFormatTest",
        "Format info for PairsArrayEntitySummary");

// ============================================================================
// Second Test Analysis - Simple analysis for multi-summary round-trip tests.
// ============================================================================

struct TagsEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("TagsEntitySummaryForJSONFormatTest");
  }

  std::vector<std::string> Tags;
};

static json::Object serializeTagsEntitySummaryForJSONFormatTest(
    const EntitySummary &Summary, const JSONFormat::EntityIdConverter &) {
  const auto &TA =
      static_cast<const TagsEntitySummaryForJSONFormatTest &>(Summary);
  json::Array TagsArray;
  for (const auto &Tag : TA.Tags) {
    TagsArray.push_back(Tag);
  }
  return json::Object{{"tags", std::move(TagsArray)}};
}

static Expected<std::unique_ptr<EntitySummary>>
deserializeTagsEntitySummaryForJSONFormatTest(
    const json::Object &Obj, EntityIdTable &,
    const JSONFormat::EntityIdConverter &) {
  auto Result = std::make_unique<TagsEntitySummaryForJSONFormatTest>();
  const json::Array *TagsArray = Obj.getArray("tags");
  if (!TagsArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'tags'");
  }
  for (const auto &[Index, Value] : llvm::enumerate(*TagsArray)) {
    auto Tag = Value.getAsString();
    if (!Tag) {
      return createStringError(inconvertibleErrorCode(),
                               "tags element at index %zu is not a string",
                               Index);
    }
    Result->Tags.push_back(Tag->str());
  }
  return std::move(Result);
}

struct TagsEntitySummaryForJSONFormatTestFormatInfo final
    : JSONFormat::FormatInfo {
  TagsEntitySummaryForJSONFormatTestFormatInfo()
      : JSONFormat::FormatInfo(
            SummaryName("TagsEntitySummaryForJSONFormatTest"),
            serializeTagsEntitySummaryForJSONFormatTest,
            deserializeTagsEntitySummaryForJSONFormatTest) {}
};

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    TagsEntitySummaryForJSONFormatTestFormatInfo>
    RegisterTagsEntitySummaryForJSONFormatTest(
        "TagsEntitySummaryForJSONFormatTest",
        "Format info for TagsEntitySummary");

// ============================================================================
// NullEntitySummaryForJSONFormatTest - For null data checks
// ============================================================================

struct NullEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("NullEntitySummaryForJSONFormatTest");
  }
};

struct NullEntitySummaryForJSONFormatTestFormatInfo final
    : JSONFormat::FormatInfo {
  NullEntitySummaryForJSONFormatTestFormatInfo()
      : JSONFormat::FormatInfo(
            SummaryName("NullEntitySummaryForJSONFormatTest"),
            [](const EntitySummary &, const JSONFormat::EntityIdConverter &)
                -> json::Object { return json::Object{}; },
            [](const json::Object &, EntityIdTable &,
               const JSONFormat::EntityIdConverter &)
                -> llvm::Expected<std::unique_ptr<EntitySummary>> {
              return nullptr;
            }) {}
};

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    NullEntitySummaryForJSONFormatTestFormatInfo>
    RegisterNullEntitySummaryForJSONFormatTest(
        "NullEntitySummaryForJSONFormatTest",
        "Format info for NullEntitySummary");

// ============================================================================
// UnregisteredEntitySummaryForJSONFormatTest - For missing FormatInfo checks
// ============================================================================

struct UnregisteredEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("UnregisteredEntitySummaryForJSONFormatTest");
  }
};

// ============================================================================
// MismatchedEntitySummaryForJSONFormatTest - For mismatched SummaryName checks
// ============================================================================

struct MismatchedEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("MismatchedEntitySummaryForJSONFormatTest_WrongName");
  }
};

struct MismatchedEntitySummaryForJSONFormatTestFormatInfo final
    : JSONFormat::FormatInfo {
  MismatchedEntitySummaryForJSONFormatTestFormatInfo()
      : JSONFormat::FormatInfo(
            SummaryName("MismatchedEntitySummaryForJSONFormatTest"),
            [](const EntitySummary &, const JSONFormat::EntityIdConverter &)
                -> json::Object { return json::Object{}; },
            [](const json::Object &, EntityIdTable &,
               const JSONFormat::EntityIdConverter &)
                -> llvm::Expected<std::unique_ptr<EntitySummary>> {
              return std::make_unique<
                  MismatchedEntitySummaryForJSONFormatTest>();
            }) {}
};

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    MismatchedEntitySummaryForJSONFormatTestFormatInfo>
    RegisterMismatchedEntitySummaryForJSONFormatTest(
        "MismatchedEntitySummaryForJSONFormatTest",
        "Format info for MismatchedEntitySummary");

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
    if (!ExpectedFilePath)
      return ExpectedFilePath.takeError();

    return readTUSummaryFromFile(FileName);
  }

  llvm::Error writeTUSummary(const TUSummary &Summary,
                             StringRef FileName) const {
    PathString FilePath = makePath(FileName);

    return JSONFormat().writeTUSummary(Summary, FilePath);
  }

  static llvm::Error normalizeIDTable(json::Array &IDTable) {
    for (const auto &[Index, Entry] : llvm::enumerate(IDTable)) {
      const auto *EntryObj = Entry.getAsObject();
      if (!EntryObj) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: id_table entry at index %zu "
            "is not an object",
            Index);
      }

      const auto *IDValue = EntryObj->get("id");
      if (!IDValue) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: id_table entry at index %zu "
            "does not contain an 'id' field",
            Index);
      }

      if (!IDValue->getAsUINT64()) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: id_table entry at index %zu "
            "does not contain a valid 'id' uint64_t field",
            Index);
      }
    }

    // Safe to dereference: all entries were validated above.
    llvm::sort(IDTable, [](const json::Value &A, const json::Value &B) {
      return *A.getAsObject()->get("id")->getAsUINT64() <
             *B.getAsObject()->get("id")->getAsUINT64();
    });

    return llvm::Error::success();
  }

  static llvm::Error normalizeLinkageTable(json::Array &LinkageTable) {
    for (const auto &[Index, Entry] : llvm::enumerate(LinkageTable)) {
      const auto *EntryObj = Entry.getAsObject();
      if (!EntryObj) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: linkage_table entry at index "
            "%zu is not an object",
            Index);
      }

      const auto *IDValue = EntryObj->get("id");
      if (!IDValue) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: linkage_table entry at index "
            "%zu does not contain an 'id' field",
            Index);
      }

      if (!IDValue->getAsUINT64()) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: linkage_table entry at index "
            "%zu does not contain a valid 'id' uint64_t field",
            Index);
      }
    }

    // Safe to dereference: all entries were validated above.
    llvm::sort(LinkageTable, [](const json::Value &A, const json::Value &B) {
      return *A.getAsObject()->get("id")->getAsUINT64() <
             *B.getAsObject()->get("id")->getAsUINT64();
    });

    return llvm::Error::success();
  }

  static llvm::Error normalizeSummaryData(json::Array &SummaryData,
                                          size_t DataIndex) {
    for (const auto &[SummaryIndex, SummaryEntry] :
         llvm::enumerate(SummaryData)) {
      const auto *SummaryEntryObj = SummaryEntry.getAsObject();
      if (!SummaryEntryObj) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: data entry at index %zu, "
            "summary_data entry at index %zu is not an object",
            DataIndex, SummaryIndex);
      }

      const auto *EntityIDValue = SummaryEntryObj->get("entity_id");
      if (!EntityIDValue) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: data entry at index %zu, "
            "summary_data entry at index %zu does not contain an "
            "'entity_id' field",
            DataIndex, SummaryIndex);
      }

      if (!EntityIDValue->getAsUINT64()) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: data entry at index %zu, "
            "summary_data entry at index %zu does not contain a valid "
            "'entity_id' uint64_t field",
            DataIndex, SummaryIndex);
      }
    }

    // Safe to dereference: all entries were validated above.
    llvm::sort(SummaryData, [](const json::Value &A, const json::Value &B) {
      return *A.getAsObject()->get("entity_id")->getAsUINT64() <
             *B.getAsObject()->get("entity_id")->getAsUINT64();
    });

    return llvm::Error::success();
  }

  static llvm::Error normalizeData(json::Array &Data) {
    for (const auto &[DataIndex, DataEntry] : llvm::enumerate(Data)) {
      auto *DataEntryObj = DataEntry.getAsObject();
      if (!DataEntryObj) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: data entry at index %zu "
            "is not an object",
            DataIndex);
      }

      if (!DataEntryObj->getString("summary_name")) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: data entry at index %zu "
            "does not contain a 'summary_name' string field",
            DataIndex);
      }

      auto *SummaryData = DataEntryObj->getArray("summary_data");
      if (!SummaryData) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: data entry at index %zu "
            "does not contain a 'summary_data' array field",
            DataIndex);
      }

      if (auto Err = normalizeSummaryData(*SummaryData, DataIndex)) {
        return Err;
      }
    }

    // Safe to dereference: all entries were validated above.
    llvm::sort(Data, [](const json::Value &A, const json::Value &B) {
      return *A.getAsObject()->getString("summary_name") <
             *B.getAsObject()->getString("summary_name");
    });

    return llvm::Error::success();
  }

  static Expected<json::Value> normalizeTUSummaryJSON(json::Value Val) {
    auto *Obj = Val.getAsObject();
    if (!Obj) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize TUSummary JSON: expected an object");
    }

    auto *IDTable = Obj->getArray("id_table");
    if (!IDTable) {
      return createStringError(inconvertibleErrorCode(),
                               "Cannot normalize TUSummary JSON: 'id_table' "
                               "field is either missing or has the wrong type");
    }
    if (auto Err = normalizeIDTable(*IDTable)) {
      return std::move(Err);
    }

    auto *LinkageTable = Obj->getArray("linkage_table");
    if (!LinkageTable) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize TUSummary JSON: 'linkage_table' "
          "field is either missing or has the wrong type");
    }
    if (auto Err = normalizeLinkageTable(*LinkageTable)) {
      return std::move(Err);
    }

    auto *Data = Obj->getArray("data");
    if (!Data) {
      return createStringError(inconvertibleErrorCode(),
                               "Cannot normalize TUSummary JSON: 'data' "
                               "field is either missing or has the wrong type");
    }
    if (auto Err = normalizeData(*Data)) {
      return std::move(Err);
    }

    return Val;
  }

  // Compare two TUSummary JSON values with normalization.
  static Expected<bool> compareTUSummaryJSON(json::Value A, json::Value B) {
    auto ExpectedNormalizedA = normalizeTUSummaryJSON(std::move(A));
    if (!ExpectedNormalizedA)
      return ExpectedNormalizedA.takeError();

    auto ExpectedNormalizedB = normalizeTUSummaryJSON(std::move(B));
    if (!ExpectedNormalizedB)
      return ExpectedNormalizedB.takeError();

    return *ExpectedNormalizedA == *ExpectedNormalizedB;
  }

  void readWriteCompareTUSummary(StringRef JSON) const {
    const PathString InputFileName("input.json");
    const PathString OutputFileName("output.json");

    auto ExpectedInputFilePath = writeJSON(JSON, InputFileName);
    ASSERT_THAT_EXPECTED(ExpectedInputFilePath, Succeeded());

    auto ExpectedTUSummary = readTUSummaryFromFile(InputFileName);
    ASSERT_THAT_EXPECTED(ExpectedTUSummary, Succeeded());

    auto WriteError = writeTUSummary(*ExpectedTUSummary, OutputFileName);
    ASSERT_THAT_ERROR(std::move(WriteError), Succeeded())
        << "Failed to write to file: " << OutputFileName;

    auto ExpectedInputJSON = readJSONFromFile(InputFileName);
    ASSERT_THAT_EXPECTED(ExpectedInputJSON, Succeeded());
    auto ExpectedOutputJSON = readJSONFromFile(OutputFileName);
    ASSERT_THAT_EXPECTED(ExpectedOutputJSON, Succeeded());

    auto ExpectedComparisonResult =
        compareTUSummaryJSON(*ExpectedInputJSON, *ExpectedOutputJSON);
    ASSERT_THAT_EXPECTED(ExpectedComparisonResult, Succeeded())
        << "Failed to normalize JSON for comparison";

    if (!*ExpectedComparisonResult) {
      auto ExpectedNormalizedInput = normalizeTUSummaryJSON(*ExpectedInputJSON);
      auto ExpectedNormalizedOutput =
          normalizeTUSummaryJSON(*ExpectedOutputJSON);
      FAIL() << "Serialization is broken: input JSON is different from output "
                "json\n"
             << "Input:  "
             << (ExpectedNormalizedInput
                     ? llvm::formatv("{0:2}", *ExpectedNormalizedInput).str()
                     : "normalization failed")
             << "\n"
             << "Output: "
             << (ExpectedNormalizedOutput
                     ? llvm::formatv("{0:2}", *ExpectedNormalizedOutput).str()
                     : "normalization failed");
    }
  }
};

// ============================================================================
// readJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, NonexistentFile) {
  auto Result = readTUSummaryFromFile("nonexistent.json");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                                      HasSubstr("file does not exist"))));
}

TEST_F(JSONFormatTUSummaryTest, PathIsDirectory) {
  PathString DirName("test_directory.json");

  auto ExpectedDirPath = makeDirectory(DirName);
  ASSERT_THAT_EXPECTED(ExpectedDirPath, Succeeded());

  auto Result = readTUSummaryFromFile(DirName);

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                              HasSubstr("path is a directory, not a file"))));
}

TEST_F(JSONFormatTUSummaryTest, NotJsonExtension) {
  PathString FileName("test.txt");

  auto ExpectedFilePath = writeJSON("{}", FileName);
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  auto Result = readTUSummaryFromFile(FileName);

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("failed to read file"),
                  HasSubstr("file does not end with '.json' extension"))));
}

TEST_F(JSONFormatTUSummaryTest, BrokenSymlink) {
#ifdef _WIN32
  GTEST_SKIP() << "Symlink model differs on Windows";
#endif

  // Create a symlink pointing to a non-existent file
  auto ExpectedSymlinkPath =
      makeSymlink("nonexistent_target.json", "broken_symlink.json");
  ASSERT_THAT_EXPECTED(ExpectedSymlinkPath, Succeeded());

  auto Result = readTUSummaryFromFile(*ExpectedSymlinkPath);

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("failed to read file"))));
}

TEST_F(JSONFormatTUSummaryTest, NoReadPermission) {
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

  auto Result = readTUSummaryFromFile(FileName);

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("failed to read file"))));

  // Restore permissions for cleanup
  auto RestoreError = setPermission(FileName, sys::fs::perms::all_all);
  EXPECT_THAT_ERROR(std::move(RestoreError), Succeeded());
}

TEST_F(JSONFormatTUSummaryTest, InvalidSyntax) {
  auto Result = readTUSummaryFromString("{ invalid json }");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("Expected object key"))));
}

TEST_F(JSONFormatTUSummaryTest, NotObject) {
  auto Result = readTUSummaryFromString("[]");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("failed to read TUSummary"),
                                      HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::entityLinkageFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, LinkageTableEntryLinkageMissingType) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("reading EntityLinkage from field 'linkage'"),
                HasSubstr("failed to read EntityLinkageType from field 'type'"),
                HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatTUSummaryTest, LinkageTableEntryLinkageInvalidType) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading LinkageTable from field 'linkage_table'"),
          HasSubstr("reading LinkageTable entry from index '0'"),
          HasSubstr("reading EntityLinkage from field 'linkage'"),
          HasSubstr("invalid 'type' EntityLinkageType value 'invalid_type'"))));
}

// ============================================================================
// JSONFormat::linkageTableEntryFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, LinkageTableEntryMissingId) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_F(JSONFormatTUSummaryTest, LinkageTableEntryIdNotUInt64) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading LinkageTable from field 'linkage_table'"),
                HasSubstr("reading LinkageTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_F(JSONFormatTUSummaryTest, LinkageTableEntryMissingLinkage) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result,
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

TEST_F(JSONFormatTUSummaryTest, LinkageTableNotArray) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": {},
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read LinkageTable from field 'linkage_table'"),
          HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatTUSummaryTest, LinkageTableElementNotObject) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": ["invalid"],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading LinkageTable from field 'linkage_table'"),
                  HasSubstr("failed to read LinkageTable entry from index '0'"),
                  HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTUSummaryTest, LinkageTableExtraId) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading LinkageTable from field 'linkage_table'"),
                  HasSubstr("reading LinkageTable entry from index '0'"),
                  HasSubstr("failed to deserialize LinkageTable"),
                  HasSubstr("extra 'EntityId(0)' not present in IdTable"))));
}

TEST_F(JSONFormatTUSummaryTest, LinkageTableMissingId) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
              "kind": "CompilationUnit",
              "name": "test.cpp"
            }
          ]
        }
      }
    ],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading LinkageTable from field 'linkage_table'"),
                  HasSubstr("failed to deserialize LinkageTable"),
                  HasSubstr("missing 'EntityId(0)' present in IdTable"))));
}

TEST_F(JSONFormatTUSummaryTest, LinkageTableDuplicateId) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
      },
      {
        "id": 0,
        "linkage": { "type": "Internal" }
      }
    ],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading LinkageTable from field 'linkage_table'"),
                  HasSubstr("failed to insert LinkageTable entry at index '1'"),
                  HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::buildNamespaceKindFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, InvalidKind) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "invalid_kind",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
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

TEST_F(JSONFormatTUSummaryTest, MissingKind) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
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

TEST_F(JSONFormatTUSummaryTest, MissingName) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit"
    },
    "id_table": [],
    "linkage_table": [],
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

TEST_F(JSONFormatTUSummaryTest, NamespaceElementNotObject) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "None" }
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
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("failed to read BuildNamespace from index '0'"),
          HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTUSummaryTest, NamespaceElementMissingKind) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("failed to read BuildNamespaceKind from field 'kind'"),
          HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatTUSummaryTest, NamespaceElementInvalidKind) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("reading BuildNamespaceKind from field 'kind'"),
          HasSubstr(
              "invalid 'kind' BuildNamespaceKind value 'invalid_kind'"))));
}

TEST_F(JSONFormatTUSummaryTest, NamespaceElementMissingName) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
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

TEST_F(JSONFormatTUSummaryTest, EntityNameMissingUSR) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "Internal" }
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

TEST_F(JSONFormatTUSummaryTest, EntityNameMissingSuffix) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
    "linkage_table": [
      {
        "id": 0,
        "linkage": { "type": "External" }
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

TEST_F(JSONFormatTUSummaryTest, EntityNameMissingNamespace) {
  auto Result = readTUSummaryFromString(R"({
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

TEST_F(JSONFormatTUSummaryTest, IDTableEntryMissingID) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_F(JSONFormatTUSummaryTest, IDTableEntryMissingName) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading IdTable from field 'id_table'"),
                  HasSubstr("reading EntityIdTable entry from index '0'"),
                  HasSubstr("failed to read EntityName from field 'name'"),
                  HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTUSummaryTest, IDTableEntryIDNotUInt64) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

// ============================================================================
// JSONFormat::entityIdTableFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, IDTableNotArray) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": {},
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("failed to read IdTable from field 'id_table'"),
                  HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatTUSummaryTest, IDTableElementNotObject) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [123],
    "linkage_table": [],
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

TEST_F(JSONFormatTUSummaryTest, DuplicateEntity) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
              "kind": "CompilationUnit",
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
      },
      {
        "id": 1,
        "linkage": { "type": "External" }
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
                HasSubstr("encountered duplicate 'EntityId(0)'"))));
}

// ============================================================================
// JSONFormat::entitySummaryFromJSON() Error Tests
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
                  "second": 1
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
                  "second": 1
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
                  "first": 0
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

TEST_F(JSONFormatTUSummaryTest, EntityDataMissingEntityID) {
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
          HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_F(JSONFormatTUSummaryTest, EntityDataMissingEntitySummary) {
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

TEST_F(JSONFormatTUSummaryTest, EntityIDNotUInt64) {
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

TEST_F(JSONFormatTUSummaryTest, EntityDataElementNotObject) {
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

TEST_F(JSONFormatTUSummaryTest, DuplicateEntityIdInDataMap) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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

  EXPECT_THAT_EXPECTED(
      Result,
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

TEST_F(JSONFormatTUSummaryTest, DataEntryMissingSummaryName) {
  auto Result = readTUSummaryFromString(R"({
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

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading SummaryData entries from field 'data'"),
          HasSubstr("reading SummaryData entry from index '0'"),
          HasSubstr("failed to read SummaryName from field 'summary_name'"),
          HasSubstr("expected JSON string"))));
}

TEST_F(JSONFormatTUSummaryTest, DataEntryMissingData) {
  auto Result = readTUSummaryFromString(R"({
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

TEST_F(JSONFormatTUSummaryTest, DataNotArray) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": {}
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read SummaryData entries from field 'data'"),
          HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatTUSummaryTest, DataElementNotObject) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": ["invalid"]
  })");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading SummaryData entries from field 'data'"),
                  HasSubstr("failed to read SummaryData entry from index '0'"),
                  HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTUSummaryTest, DuplicateSummaryName) {
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
        "summary_data": []
      },
      {
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
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
          HasSubstr("encountered duplicate "
                    "'SummaryName(PairsEntitySummaryForJSONFormatTest)'"))));
}

// ============================================================================
// JSONFormat::readTUSummary() Error Tests
// ============================================================================

TEST_F(JSONFormatTUSummaryTest, MissingTUNamespace) {
  auto Result = readTUSummaryFromString(R"({
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read BuildNamespace from field 'tu_namespace'"),
          HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTUSummaryTest, MissingIDTable) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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

TEST_F(JSONFormatTUSummaryTest, MissingLinkageTable) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("failed to read LinkageTable from field 'linkage_table'"),
          HasSubstr("expected JSON array"))));
}

TEST_F(JSONFormatTUSummaryTest, MissingData) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": []
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

TEST_F(JSONFormatTUSummaryTest, WriteFileAlreadyExists) {
  PathString FileName("existing.json");

  auto ExpectedFilePath = writeJSON("{}", FileName);
  ASSERT_THAT_EXPECTED(ExpectedFilePath, Succeeded());

  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));

  // Try to write to the same path
  auto Result = writeTUSummary(Summary, FileName);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("file already exists"))));
}

TEST_F(JSONFormatTUSummaryTest, WriteParentDirectoryNotFound) {
  PathString FilePath = makePath("nonexistent-dir", "test.json");

  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));

  auto Result = JSONFormat().writeTUSummary(Summary, FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"),
                              HasSubstr("parent directory does not exist"))));
}

TEST_F(JSONFormatTUSummaryTest, WriteNotJsonExtension) {
  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));

  auto Result = writeTUSummary(Summary, "test.txt");

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(
          AllOf(HasSubstr("writing TUSummary to file"),
                HasSubstr("failed to write file"),
                HasSubstr("file does not end with '.json' extension"))));
}

TEST_F(JSONFormatTUSummaryTest, WriteStreamOpenFailure) {
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

  TUSummary Summary(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));

  auto Result = JSONFormat().writeTUSummary(Summary, FilePath);

  EXPECT_THAT_ERROR(
      std::move(Result),
      FailedWithMessage(AllOf(HasSubstr("writing TUSummary to file"),
                              HasSubstr("failed to write file"))));

  // Restore permissions for cleanup
  auto RestoreError = setPermission(DirName, sys::fs::perms::all_all);
  EXPECT_THAT_ERROR(std::move(RestoreError), Succeeded());
}

// ============================================================================
// JSONFormat::writeTUSummary() Error Tests
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

TEST_F(JSONFormatTUSummaryTest, RoundTripEmpty) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatTUSummaryTest, RoundTripWithTwoSummaryTypes) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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
                { "first": 1, "second": 3 }
              ]
            }
          },
          {
            "entity_id": 4,
            "entity_summary": {
              "pairs": [
                { "first": 4, "second": 0 },
                { "first": 4, "second": 2 }
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
                { "first": 3, "second": 1 }
              ]
            }
          },
          {
            "entity_id": 2,
            "entity_summary": {
              "pairs": [
                { "first": 2, "second": 4 },
                { "first": 2, "second": 3 }
              ]
            }
          }
        ]
      }
    ]
  })");
}

TEST_F(JSONFormatTUSummaryTest, RoundTripLinkUnit) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "LinkUnit",
      "name": "libtest.so"
    },
    "id_table": [],
    "linkage_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatTUSummaryTest, RoundTripWithEmptyDataEntry) {
  readWriteCompareTUSummary(R"({
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

TEST_F(JSONFormatTUSummaryTest, RoundTripLinkageTableWithNoneLinkage) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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

TEST_F(JSONFormatTUSummaryTest, RoundTripLinkageTableWithInternalLinkage) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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

TEST_F(JSONFormatTUSummaryTest, RoundTripLinkageTableWithExternalLinkage) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
      "name": "test.cpp"
    },
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

TEST_F(JSONFormatTUSummaryTest, RoundTripLinkageTableWithMultipleEntries) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "CompilationUnit",
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
