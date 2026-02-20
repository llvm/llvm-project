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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace clang::ssaf;
using namespace llvm;
using ::testing::AllOf;
using ::testing::HasSubstr;

namespace {

// ============================================================================
// Test Analysis - Simple analysis for testing JSON serialization.
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
// Test Fixture
// ============================================================================

class JSONFormatTest : public ::testing::Test {
public:
  using PathString = SmallString<128>;

protected:
  SmallString<128> TestDir;

  void SetUp() override {
    std::error_code EC =
        sys::fs::createUniqueDirectory("json-format-test", TestDir);
    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

  PathString makePath(StringRef FileOrDirectoryName) const {
    PathString FullPath = TestDir;
    sys::path::append(FullPath, FileOrDirectoryName);

    return FullPath;
  }

  PathString makePath(StringRef Dir, StringRef FileName) const {
    PathString FullPath = TestDir;
    sys::path::append(FullPath, Dir, FileName);

    return FullPath;
  }

  Expected<PathString> makeDirectory(StringRef DirectoryName) const {
    PathString DirPath = makePath(DirectoryName);

    std::error_code EC = sys::fs::create_directory(DirPath);
    if (EC) {
      return createStringError(EC, "Failed to create directory '%s': %s",
                               DirPath.c_str(), EC.message().c_str());
    }

    return DirPath;
  }

  Expected<PathString> makeSymlink(StringRef TargetFileName,
                                   StringRef SymlinkFileName) const {
    PathString TargetPath = makePath(TargetFileName);
    PathString SymlinkPath = makePath(SymlinkFileName);

    std::error_code EC = sys::fs::create_link(TargetPath, SymlinkPath);
    if (EC) {
      return createStringError(EC, "Failed to create symlink '%s' -> '%s': %s",
                               SymlinkPath.c_str(), TargetPath.c_str(),
                               EC.message().c_str());
    }

    return SymlinkPath;
  }

  llvm::Error setPermission(StringRef FileName,
                            const sys::fs::perms Perms) const {
    PathString Path = makePath(FileName);

    std::error_code EC = sys::fs::setPermissions(Path, Perms);
    if (EC) {
      return createStringError(EC, "Failed to set permissions on '%s': %s",
                               Path.c_str(), EC.message().c_str());
    }

    return llvm::Error::success();
  }

  Expected<json::Value> readJSONFromFile(StringRef FileName) const {
    PathString FilePath = makePath(FileName);

    auto BufferOrError = MemoryBuffer::getFile(FilePath);
    if (!BufferOrError) {
      return createStringError(BufferOrError.getError(),
                               "Failed to read file: %s", FilePath.c_str());
    }

    Expected<json::Value> ExpectedValue =
        json::parse(BufferOrError.get()->getBuffer());
    if (!ExpectedValue)
      return ExpectedValue.takeError();

    return *ExpectedValue;
  }

  Expected<PathString> writeJSON(StringRef JSON, StringRef FileName) const {
    PathString FilePath = makePath(FileName);

    std::error_code EC;
    llvm::raw_fd_ostream OS(FilePath, EC);
    if (EC) {
      return createStringError(EC, "Failed to create file '%s': %s",
                               FilePath.c_str(), EC.message().c_str());
    }

    OS << JSON;
    OS.close();

    if (OS.has_error()) {
      return createStringError(OS.error(), "Failed to write to file '%s': %s",
                               FilePath.c_str(), OS.error().message().c_str());
    }

    return FilePath;
  }

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

  // Normalize TUSummary JSON by sorting id_table by id field.
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

    // Validate all id_table entries before sorting.
    for (const auto &[Index, Entry] : llvm::enumerate(*IDTable)) {
      const auto *EntryObj = Entry.getAsObject();
      if (!EntryObj) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: id_table entry at index %zu is "
            "not an object",
            Index);
      }

      const auto *IDValue = EntryObj->get("id");
      if (!IDValue) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: id_table entry at index %zu does "
            "not contain an 'id' field",
            Index);
      }

      auto EntryID = IDValue->getAsUINT64();
      if (!EntryID) {
        return createStringError(
            inconvertibleErrorCode(),
            "Cannot normalize TUSummary JSON: id_table entry at index %zu does "
            "not contain a valid 'id' uint64_t field",
            Index);
      }
    }

    // Sort id_table entries by the "id" field to ensure deterministic ordering
    // for comparison. Use projection-based comparison for strict-weak-ordering.
    llvm::sort(*IDTable, [](const json::Value &A, const json::Value &B) {
      // Safe to assume these succeed because we validated above.
      const auto *AObj = A.getAsObject();
      const auto *BObj = B.getAsObject();
      uint64_t AID = *AObj->get("id")->getAsUINT64();
      uint64_t BID = *BObj->get("id")->getAsUINT64();
      return AID < BID;
    });

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

TEST_F(JSONFormatTest, NonexistentFile) {
  auto Result = readTUSummaryFromFile("nonexistent.json");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                                      HasSubstr("file does not exist"))));
}

TEST_F(JSONFormatTest, PathIsDirectory) {
  PathString DirName("test_directory.json");

  auto ExpectedDirPath = makeDirectory(DirName);
  ASSERT_THAT_EXPECTED(ExpectedDirPath, Succeeded());

  auto Result = readTUSummaryFromFile(DirName);

  EXPECT_THAT_EXPECTED(
      Result,
      FailedWithMessage(AllOf(HasSubstr("reading TUSummary from"),
                              HasSubstr("path is a directory, not a file"))));
}

TEST_F(JSONFormatTest, NotJsonExtension) {
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

TEST_F(JSONFormatTest, BrokenSymlink) {
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

TEST_F(JSONFormatTest, NoReadPermission) {
#ifdef _WIN32
  GTEST_SKIP() << "Permission model differs on Windows";
#endif

  PathString FileName("no-read-permission.json");

  auto ExpectedFilePath = writeJSON(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, InvalidSyntax) {
  auto Result = readTUSummaryFromString("{ invalid json }");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("Expected object key"))));
}

TEST_F(JSONFormatTest, NotObject) {
  auto Result = readTUSummaryFromString("[]");

  EXPECT_THAT_EXPECTED(
      Result, FailedWithMessage(AllOf(HasSubstr("reading TUSummary from file"),
                                      HasSubstr("failed to read TUSummary"),
                                      HasSubstr("expected JSON object"))));
}

// ============================================================================
// JSONFormat::buildNamespaceKindFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, InvalidKind) {
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
      FailedWithMessage(AllOf(
          HasSubstr("reading TUSummary from file"),
          HasSubstr("reading IdTable from field 'id_table'"),
          HasSubstr("reading EntityIdTable entry from index '0'"),
          HasSubstr("reading EntityName from field 'name'"),
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("failed to read BuildNamespace from index '0'"),
          HasSubstr("expected JSON object"))));
}

TEST_F(JSONFormatTest, NamespaceElementMissingKind) {
  auto Result = readTUSummaryFromString(R"({
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

TEST_F(JSONFormatTest, NamespaceElementInvalidKind) {
  auto Result = readTUSummaryFromString(R"({
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
              "kind": "invalid_kind",
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

TEST_F(JSONFormatTest, NamespaceElementMissingName) {
  auto Result = readTUSummaryFromString(R"({
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
              "kind": "compilation_unit"
            }
          ]
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
          HasSubstr("reading NestedBuildNamespace from field 'namespace'"),
          HasSubstr("reading BuildNamespace from index '0'"),
          HasSubstr("failed to read BuildNamespaceName from field 'name'"),
          HasSubstr("expected JSON string"))));
}

// ============================================================================
// JSONFormat::entityNameFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, EntityNameMissingUSR) {
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
      Result,
      FailedWithMessage(
          AllOf(HasSubstr("reading TUSummary from file"),
                HasSubstr("reading IdTable from field 'id_table'"),
                HasSubstr("reading EntityIdTable entry from index '0'"),
                HasSubstr("failed to read EntityId from field 'id'"),
                HasSubstr("expected JSON number (unsigned 64-bit integer)"))));
}

TEST_F(JSONFormatTest, IDTableEntryMissingName) {
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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

TEST_F(JSONFormatTest, IDTableNotArray) {
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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

// ============================================================================
// PairsEntitySummaryForJSONFormatTest Deserialization Error Tests
// ============================================================================

TEST_F(JSONFormatTest, PairsEntitySummaryForJSONFormatTestMissingPairsField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest,
       PairsEntitySummaryForJSONFormatTestInvalidPairsFieldType) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest,
       PairsEntitySummaryForJSONFormatTestPairsElementNotObject) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, PairsEntitySummaryForJSONFormatTestMissingFirstField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, PairsEntitySummaryForJSONFormatTestInvalidFirstField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, PairsEntitySummaryForJSONFormatTestMissingSecondField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, PairsEntitySummaryForJSONFormatTestInvalidSecondField) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, EntityDataMissingEntityID) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, EntityDataMissingEntitySummary) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, EntityIDNotUInt64) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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
          HasSubstr("expected JSON integer"))));
}

// ============================================================================
// JSONFormat::entityDataMapFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, EntityDataElementNotObject) {
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, DuplicateEntityIdInDataMap) {
  auto Result = readTUSummaryFromString(R"({
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
          HasSubstr("encountered duplicate EntityId '0'"))));
}

// ============================================================================
// JSONFormat::summaryDataMapEntryFromJSON() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, DataEntryMissingSummaryName) {
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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

TEST_F(JSONFormatTest, DataNotArray) {
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
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
      Result, FailedWithMessage(AllOf(
                  HasSubstr("reading TUSummary from file"),
                  HasSubstr("reading SummaryData entries from field 'data'"),
                  HasSubstr("failed to insert SummaryData entry at index '1'"),
                  HasSubstr("encountered duplicate SummaryName "
                            "'PairsEntitySummaryForJSONFormatTest'"))));
}

// ============================================================================
// JSONFormat::readTUSummary() Error Tests
// ============================================================================

TEST_F(JSONFormatTest, MissingTUNamespace) {
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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
  auto Result = readTUSummaryFromString(R"({
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

TEST_F(JSONFormatTest, WriteParentDirectoryNotFound) {
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

TEST_F(JSONFormatTest, WriteNotJsonExtension) {
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

TEST_F(JSONFormatTest, WriteStreamOpenFailure) {
#ifdef _WIN32
  GTEST_SKIP() << "Permission model differs on Windows";
#endif

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
// Round-Trip Tests - Serialization Verification
// ============================================================================

TEST_F(JSONFormatTest, Empty) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatTest, LinkUnit) {
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "link_unit",
      "name": "libtest.so"
    },
    "id_table": [],
    "data": []
  })");
}

TEST_F(JSONFormatTest, WithIDTable) {
  readWriteCompareTUSummary(R"({
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
  readWriteCompareTUSummary(R"({
    "tu_namespace": {
      "kind": "compilation_unit",
      "name": "test.cpp"
    },
    "id_table": [],
    "data": [
      {
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
        "summary_data": []
      }
    ]
  })");
}

TEST_F(JSONFormatTest, RoundTripWithIDTable) {
  readWriteCompareTUSummary(R"({
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

TEST_F(JSONFormatTest, RoundTripPairsEntitySummaryForJSONFormatTest) {
  readWriteCompareTUSummary(R"({
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
        "summary_name": "PairsEntitySummaryForJSONFormatTest",
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
