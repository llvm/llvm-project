//===- JSONFormatTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared normalization helpers for SSAF JSON serialization format unit tests.
//
//===----------------------------------------------------------------------===//

#include "JSONFormatTest.h"

#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Registry.h"
#include "llvm/Testing/Support/Error.h"
#ifndef _WIN32
#include <unistd.h>
#endif

using namespace clang::ssaf;
using namespace llvm;
using PathString = JSONFormatTest::PathString;

// ============================================================================
// Test Fixture
// ============================================================================

void JSONFormatTest::SetUp() {
  std::error_code EC =
      llvm::sys::fs::createUniqueDirectory("json-format-test", TestDir);
  ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
}

void JSONFormatTest::TearDown() { llvm::sys::fs::remove_directories(TestDir); }

JSONFormatTest::PathString
JSONFormatTest::makePath(llvm::StringRef FileOrDirectoryName) const {
  PathString FullPath = TestDir;
  llvm::sys::path::append(FullPath, FileOrDirectoryName);

  return FullPath;
}

PathString JSONFormatTest::makePath(llvm::StringRef Dir,
                                    llvm::StringRef FileName) const {
  PathString FullPath = TestDir;
  llvm::sys::path::append(FullPath, Dir, FileName);

  return FullPath;
}

llvm::Expected<PathString>
JSONFormatTest::makeDirectory(llvm::StringRef DirectoryName) const {
  PathString DirPath = makePath(DirectoryName);

  std::error_code EC = llvm::sys::fs::create_directory(DirPath);
  if (EC) {
    return llvm::createStringError(EC, "Failed to create directory '%s': %s",
                                   DirPath.c_str(), EC.message().c_str());
  }

  return DirPath;
}

llvm::Expected<PathString>
JSONFormatTest::makeSymlink(llvm::StringRef TargetFileName,
                            llvm::StringRef SymlinkFileName) const {
  PathString TargetPath = makePath(TargetFileName);
  PathString SymlinkPath = makePath(SymlinkFileName);

  std::error_code EC = llvm::sys::fs::create_link(TargetPath, SymlinkPath);
  if (EC) {
    return llvm::createStringError(
        EC, "Failed to create symlink '%s' -> '%s': %s", SymlinkPath.c_str(),
        TargetPath.c_str(), EC.message().c_str());
  }

  return SymlinkPath;
}

llvm::Error JSONFormatTest::setPermission(llvm::StringRef FileName,
                                          llvm::sys::fs::perms Perms) const {
  PathString Path = makePath(FileName);

  std::error_code EC = llvm::sys::fs::setPermissions(Path, Perms);
  if (EC) {
    return llvm::createStringError(EC, "Failed to set permissions on '%s': %s",
                                   Path.c_str(), EC.message().c_str());
  }

  return llvm::Error::success();
}

bool JSONFormatTest::permissionsAreEnforced() const {
#ifdef _WIN32
  return false;
#else
  if (getuid() == 0) {
    return false;
  }

  // Write a probe file, remove read permission, and try to open it.
  PathString ProbePath = makePath("perm-probe.json");

  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(ProbePath, EC);
    if (EC) {
      return true; // Probe setup failed; assume enforced to avoid
                   // silently suppressing the test.
    }
    OS << "{}";
  }

  std::error_code PermEC = llvm::sys::fs::setPermissions(
      ProbePath, llvm::sys::fs::perms::owner_write);
  if (PermEC) {
    return true; // Probe setup failed; assume enforced to avoid
                 // silently suppressing the test.
  }

  auto Buffer = llvm::MemoryBuffer::getFile(ProbePath);
  bool Enforced = !Buffer; // If open failed, permissions are enforced.

  // Restore permissions so TearDown can clean up the temp directory.
  llvm::sys::fs::setPermissions(ProbePath, llvm::sys::fs::perms::all_all);

  return Enforced;
#endif
}

llvm::Expected<llvm::json::Value>
JSONFormatTest::readJSONFromFile(llvm::StringRef FileName) const {
  PathString FilePath = makePath(FileName);

  auto BufferOrError = llvm::MemoryBuffer::getFile(FilePath);
  if (!BufferOrError) {
    return llvm::createStringError(BufferOrError.getError(),
                                   "Failed to read file: %s", FilePath.c_str());
  }

  llvm::Expected<llvm::json::Value> ExpectedValue =
      llvm::json::parse(BufferOrError.get()->getBuffer());
  if (!ExpectedValue) {
    return ExpectedValue.takeError();
  }

  return *ExpectedValue;
}

llvm::Expected<PathString>
JSONFormatTest::writeJSON(llvm::StringRef JSON,
                          llvm::StringRef FileName) const {
  PathString FilePath = makePath(FileName);

  std::error_code EC;
  llvm::raw_fd_ostream OS(FilePath, EC);
  if (EC) {
    return llvm::createStringError(EC, "Failed to create file '%s': %s",
                                   FilePath.c_str(), EC.message().c_str());
  }

  OS << JSON;
  OS.close();

  if (OS.has_error()) {
    return llvm::createStringError(
        OS.error(), "Failed to write to file '%s': %s", FilePath.c_str(),
        OS.error().message().c_str());
  }

  return FilePath;
}

// ============================================================================
// Summary JSON Normalization Helpers
// ============================================================================

namespace {

llvm::Error normalizeIDTable(json::Array &IDTable,
                             llvm::StringRef SummaryClassName) {
  for (const auto &[Index, Entry] : llvm::enumerate(IDTable)) {
    const auto *EntryObj = Entry.getAsObject();
    if (!EntryObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: id_table entry at index %zu "
          "is not an object",
          SummaryClassName.data(), Index);
    }

    const auto *IDValue = EntryObj->get("id");
    if (!IDValue) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: id_table entry at index %zu "
          "does not contain an 'id' field",
          SummaryClassName.data(), Index);
    }

    if (!IDValue->getAsUINT64()) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: id_table entry at index %zu "
          "does not contain a valid 'id' uint64_t field",
          SummaryClassName.data(), Index);
    }
  }

  // Safe to dereference: all entries were validated above.
  llvm::sort(IDTable, [](const json::Value &A, const json::Value &B) {
    return *A.getAsObject()->get("id")->getAsUINT64() <
           *B.getAsObject()->get("id")->getAsUINT64();
  });

  return llvm::Error::success();
}

llvm::Error normalizeLinkageTable(json::Array &LinkageTable,
                                  llvm::StringRef SummaryClassName) {
  for (const auto &[Index, Entry] : llvm::enumerate(LinkageTable)) {
    const auto *EntryObj = Entry.getAsObject();
    if (!EntryObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: linkage_table entry at index "
          "%zu is not an object",
          SummaryClassName.data(), Index);
    }

    const auto *IDValue = EntryObj->get("id");
    if (!IDValue) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: linkage_table entry at index "
          "%zu does not contain an 'id' field",
          SummaryClassName.data(), Index);
    }

    if (!IDValue->getAsUINT64()) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: linkage_table entry at index "
          "%zu does not contain a valid 'id' uint64_t field",
          SummaryClassName.data(), Index);
    }
  }

  // Safe to dereference: all entries were validated above.
  llvm::sort(LinkageTable, [](const json::Value &A, const json::Value &B) {
    return *A.getAsObject()->get("id")->getAsUINT64() <
           *B.getAsObject()->get("id")->getAsUINT64();
  });

  return llvm::Error::success();
}

llvm::Error normalizeSummaryData(json::Array &SummaryData, size_t DataIndex,
                                 llvm::StringRef SummaryClassName) {
  for (const auto &[SummaryIndex, SummaryEntry] :
       llvm::enumerate(SummaryData)) {
    const auto *SummaryEntryObj = SummaryEntry.getAsObject();
    if (!SummaryEntryObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: data entry at index %zu, "
          "summary_data entry at index %zu is not an object",
          SummaryClassName.data(), DataIndex, SummaryIndex);
    }

    const auto *EntityIDValue = SummaryEntryObj->get("entity_id");
    if (!EntityIDValue) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: data entry at index %zu, "
          "summary_data entry at index %zu does not contain an "
          "'entity_id' field",
          SummaryClassName.data(), DataIndex, SummaryIndex);
    }

    if (!EntityIDValue->getAsUINT64()) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: data entry at index %zu, "
          "summary_data entry at index %zu does not contain a valid "
          "'entity_id' uint64_t field",
          SummaryClassName.data(), DataIndex, SummaryIndex);
    }
  }

  // Safe to dereference: all entries were validated above.
  llvm::sort(SummaryData, [](const json::Value &A, const json::Value &B) {
    return *A.getAsObject()->get("entity_id")->getAsUINT64() <
           *B.getAsObject()->get("entity_id")->getAsUINT64();
  });

  return llvm::Error::success();
}

llvm::Error normalizeData(json::Array &Data, llvm::StringRef SummaryClassName) {
  for (const auto &[DataIndex, DataEntry] : llvm::enumerate(Data)) {
    auto *DataEntryObj = DataEntry.getAsObject();
    if (!DataEntryObj) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: data entry at index %zu "
          "is not an object",
          SummaryClassName.data(), DataIndex);
    }

    if (!DataEntryObj->getString("summary_name")) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: data entry at index %zu "
          "does not contain a 'summary_name' string field",
          SummaryClassName.data(), DataIndex);
    }

    auto *SummaryData = DataEntryObj->getArray("summary_data");
    if (!SummaryData) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot normalize %s JSON: data entry at index %zu "
          "does not contain a 'summary_data' array field",
          SummaryClassName.data(), DataIndex);
    }

    if (auto Err =
            normalizeSummaryData(*SummaryData, DataIndex, SummaryClassName)) {
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

Expected<json::Value> normalizeSummaryJSON(json::Value Val,
                                           llvm::StringRef SummaryClassName) {
  auto *Obj = Val.getAsObject();
  if (!Obj) {
    return createStringError(inconvertibleErrorCode(),
                             "Cannot normalize %s JSON: expected an object",
                             SummaryClassName.data());
  }

  auto *IDTable = Obj->getArray("id_table");
  if (!IDTable) {
    return createStringError(inconvertibleErrorCode(),
                             "Cannot normalize %s JSON: 'id_table' "
                             "field is either missing or has the wrong type",
                             SummaryClassName.data());
  }
  if (auto Err = normalizeIDTable(*IDTable, SummaryClassName)) {
    return std::move(Err);
  }

  auto *LinkageTable = Obj->getArray("linkage_table");
  if (!LinkageTable) {
    return createStringError(inconvertibleErrorCode(),
                             "Cannot normalize %s JSON: 'linkage_table' "
                             "field is either missing or has the wrong type",
                             SummaryClassName.data());
  }
  if (auto Err = normalizeLinkageTable(*LinkageTable, SummaryClassName)) {
    return std::move(Err);
  }

  auto *Data = Obj->getArray("data");
  if (!Data) {
    return createStringError(inconvertibleErrorCode(),
                             "Cannot normalize %s JSON: 'data' "
                             "field is either missing or has the wrong type",
                             SummaryClassName.data());
  }
  if (auto Err = normalizeData(*Data, SummaryClassName)) {
    return std::move(Err);
  }

  return Val;
}

} // namespace

// ============================================================================
// SummaryTest Fixture Implementation
// ============================================================================

llvm::Error SummaryTest::readFromString(StringRef JSON,
                                        StringRef FileName) const {
  auto ExpectedFilePath = writeJSON(JSON, FileName);
  if (!ExpectedFilePath) {
    return ExpectedFilePath.takeError();
  }
  return GetParam().ReadFromFile(*ExpectedFilePath);
}

llvm::Error SummaryTest::readFromFile(StringRef FileName) const {
  return GetParam().ReadFromFile(makePath(FileName));
}

llvm::Error SummaryTest::writeEmpty(StringRef FileName) const {
  return GetParam().WriteEmpty(makePath(FileName));
}

llvm::Error SummaryTest::readWriteRoundTrip(StringRef InputFileName,
                                            StringRef OutputFileName) const {
  return GetParam().ReadWriteRoundTrip(makePath(InputFileName),
                                       makePath(OutputFileName));
}

void SummaryTest::readWriteCompare(StringRef JSON) const {
  const PathString InputFileName("input.json");
  const PathString OutputFileName("output.json");

  auto ExpectedInputFilePath = writeJSON(JSON, InputFileName);
  ASSERT_THAT_EXPECTED(ExpectedInputFilePath, Succeeded());

  ASSERT_THAT_ERROR(readWriteRoundTrip(InputFileName, OutputFileName),
                    Succeeded());

  auto ExpectedInputJSON = readJSONFromFile(InputFileName);
  ASSERT_THAT_EXPECTED(ExpectedInputJSON, Succeeded());

  auto ExpectedOutputJSON = readJSONFromFile(OutputFileName);
  ASSERT_THAT_EXPECTED(ExpectedOutputJSON, Succeeded());

  auto ExpectedNormalizedInputJSON =
      normalizeSummaryJSON(*ExpectedInputJSON, GetParam().SummaryClassName);
  ASSERT_THAT_EXPECTED(ExpectedNormalizedInputJSON, Succeeded());

  auto ExpectedNormalizedOutputJSON =
      normalizeSummaryJSON(*ExpectedOutputJSON, GetParam().SummaryClassName);
  ASSERT_THAT_EXPECTED(ExpectedNormalizedOutputJSON, Succeeded());

  ASSERT_EQ(*ExpectedNormalizedInputJSON, *ExpectedNormalizedOutputJSON)
      << "Serialization is broken: input is different from output\n"
      << "Input:  "
      << llvm::formatv("{0:2}", *ExpectedNormalizedInputJSON).str() << "\n"
      << "Output: "
      << llvm::formatv("{0:2}", *ExpectedNormalizedOutputJSON).str();
}

namespace {

// ============================================================================
// First Test Analysis - Simple analysis for testing JSON serialization.
// ============================================================================

json::Object serializePairsEntitySummaryForJSONFormatTest(
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

Expected<std::unique_ptr<EntitySummary>>
deserializePairsEntitySummaryForJSONFormatTest(
    const json::Object &Obj, EntityIdTable &IdTable,
    const JSONFormat::EntityIdConverter &Converter) {
  auto Result = std::make_unique<PairsEntitySummaryForJSONFormatTest>();
  const json::Array *PairsArray = Obj.getArray("pairs");
  if (!PairsArray) {
    return createStringError(inconvertibleErrorCode(),
                             "missing or invalid field 'pairs'");
  }
  for (const auto &[Index, Value] : llvm::enumerate(*PairsArray)) {
    const json::Object *Pair = Value.getAsObject();
    if (!Pair) {
      return createStringError(
          inconvertibleErrorCode(),
          "pairs element at index %zu is not a JSON object", Index);
    }
    auto FirstOpt = Pair->getInteger("first");
    if (!FirstOpt) {
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'first' field at index '%zu'", Index);
    }
    auto SecondOpt = Pair->getInteger("second");
    if (!SecondOpt) {
      return createStringError(
          inconvertibleErrorCode(),
          "missing or invalid 'second' field at index '%zu'", Index);
    }
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

llvm::Registry<JSONFormat::FormatInfo>::Add<
    PairsEntitySummaryForJSONFormatTestFormatInfo>
    RegisterPairsEntitySummaryForJSONFormatTest(
        "PairsEntitySummaryForJSONFormatTest",
        "Format info for PairsArrayEntitySummary");

// ============================================================================
// Second Test Analysis - Simple analysis for multi-summary round-trip tests.
// ============================================================================

json::Object serializeTagsEntitySummaryForJSONFormatTest(
    const EntitySummary &Summary, const JSONFormat::EntityIdConverter &) {
  const auto &TA =
      static_cast<const TagsEntitySummaryForJSONFormatTest &>(Summary);
  json::Array TagsArray;
  for (const auto &Tag : TA.Tags) {
    TagsArray.push_back(Tag);
  }
  return json::Object{{"tags", std::move(TagsArray)}};
}

Expected<std::unique_ptr<EntitySummary>>
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

llvm::Registry<JSONFormat::FormatInfo>::Add<
    TagsEntitySummaryForJSONFormatTestFormatInfo>
    RegisterTagsEntitySummaryForJSONFormatTest(
        "TagsEntitySummaryForJSONFormatTest",
        "Format info for TagsEntitySummary");

// ============================================================================
// NullEntitySummaryForJSONFormatTest - For null data checks
// ============================================================================

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

llvm::Registry<JSONFormat::FormatInfo>::Add<
    NullEntitySummaryForJSONFormatTestFormatInfo>
    RegisterNullEntitySummaryForJSONFormatTest(
        "NullEntitySummaryForJSONFormatTest",
        "Format info for NullEntitySummary");

// ============================================================================
// MismatchedEntitySummaryForJSONFormatTest - For mismatched SummaryName checks
// ============================================================================

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

llvm::Registry<JSONFormat::FormatInfo>::Add<
    MismatchedEntitySummaryForJSONFormatTestFormatInfo>
    RegisterMismatchedEntitySummaryForJSONFormatTest(
        "MismatchedEntitySummaryForJSONFormatTest",
        "Format info for MismatchedEntitySummary");

} // namespace
