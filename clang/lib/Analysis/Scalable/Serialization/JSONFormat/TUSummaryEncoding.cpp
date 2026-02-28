//===- TUSummaryEncoding.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/Analysis/Scalable/EntityLinker/EntitySummaryEncoding.h"
#include "clang/Analysis/Scalable/EntityLinker/TUSummaryEncoding.h"

namespace clang::ssaf {

//----------------------------------------------------------------------------
// JSONEntitySummaryEncoding
//----------------------------------------------------------------------------

namespace {

class JSONEntitySummaryEncoding final : public EntitySummaryEncoding {
  friend JSONFormat;

public:
  void
  patch(const std::map<EntityId, EntityId> &EntityResolutionTable) override {
    ErrorBuilder::fatal("will be implemented in the future");
  }

private:
  explicit JSONEntitySummaryEncoding(llvm::json::Value Data)
      : Data(std::move(Data)) {}

  llvm::json::Value Data;
};

} // namespace

//----------------------------------------------------------------------------
// EncodingDataMapEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
JSONFormat::encodingDataMapEntryFromJSON(
    const Object &EntityDataMapEntryObject) const {
  const Value *EntityIdIntValue = EntityDataMapEntryObject.get("entity_id");
  if (!EntityIdIntValue) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityId", "entity_id",
                                "number (unsigned 64-bit integer)")
        .build();
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityId", "entity_id",
                                "number (unsigned 64-bit integer)")
        .build();
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  const Object *OptEntitySummaryObject =
      EntityDataMapEntryObject.getObject("entity_summary");
  if (!OptEntitySummaryObject) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntitySummary", "entity_summary", "object")
        .build();
  }

  std::unique_ptr<EntitySummaryEncoding> Encoding(
      new JSONEntitySummaryEncoding(Value(Object(*OptEntitySummaryObject))));

  return std::make_pair(std::move(EI), std::move(Encoding));
}

Object JSONFormat::encodingDataMapEntryToJSON(
    EntityId EI, const std::unique_ptr<EntitySummaryEncoding> &Encoding) const {
  Object Entry;
  Entry["entity_id"] = entityIdToJSON(EI);

  // All EntitySummaryEncoding objects stored in a TUSummaryEncoding read by
  // JSONFormat are JSONEntitySummaryEncoding instances, since
  // encodingDataMapEntryFromJSON is the only place that creates them.
  auto *JSONEncoding = static_cast<JSONEntitySummaryEncoding *>(Encoding.get());
  Entry["entity_summary"] = JSONEncoding->Data;

  return Entry;
}

//----------------------------------------------------------------------------
// EncodingDataMap
//----------------------------------------------------------------------------

llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
JSONFormat::encodingDataMapFromJSON(const Array &EntityDataArray) const {
  std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>> EncodingDataMap;

  for (const auto &[Index, EntityDataMapEntryValue] :
       llvm::enumerate(EntityDataArray)) {

    const Object *OptEntityDataMapEntryObject =
        EntityDataMapEntryValue.getAsObject();
    if (!OptEntityDataMapEntryObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "EntitySummary entry", Index, "object")
          .build();
    }

    auto ExpectedEntry =
        encodingDataMapEntryFromJSON(*OptEntityDataMapEntryObject);
    if (!ExpectedEntry) {
      return ErrorBuilder::wrap(ExpectedEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "EntitySummary entry",
                   Index)
          .build();
    }

    auto [DataIt, DataInserted] =
        EncodingDataMap.insert(std::move(*ExpectedEntry));
    if (!DataInserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "EntitySummary entry", Index, DataIt->first)
          .build();
    }
  }

  return std::move(EncodingDataMap);
}

Array JSONFormat::encodingDataMapToJSON(
    const std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>
        &EncodingDataMap) const {
  Array Result;
  Result.reserve(EncodingDataMap.size());

  for (const auto &[EI, Encoding] : EncodingDataMap) {
    Result.push_back(encodingDataMapEntryToJSON(EI, Encoding));
  }

  return Result;
}

//----------------------------------------------------------------------------
// EncodingSummaryDataMapEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<
    SummaryName, std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>>
JSONFormat::encodingSummaryDataMapEntryFromJSON(
    const Object &SummaryDataMapEntryObject) const {
  std::optional<llvm::StringRef> OptSummaryNameStr =
      SummaryDataMapEntryObject.getString("summary_name");
  if (!OptSummaryNameStr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "SummaryName", "summary_name", "string")
        .build();
  }

  SummaryName SN = summaryNameFromJSON(*OptSummaryNameStr);

  const Array *OptEntityDataArray =
      SummaryDataMapEntryObject.getArray("summary_data");
  if (!OptEntityDataArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntitySummary entries", "summary_data",
                                "array")
        .build();
  }

  auto ExpectedEncodingDataMap = encodingDataMapFromJSON(*OptEntityDataArray);
  if (!ExpectedEncodingDataMap)
    return ErrorBuilder::wrap(ExpectedEncodingDataMap.takeError())
        .context(ErrorMessages::ReadingFromField, "EntitySummary entries",
                 "summary_data")
        .build();

  return std::make_pair(std::move(SN), std::move(*ExpectedEncodingDataMap));
}

Object JSONFormat::encodingSummaryDataMapEntryToJSON(
    const SummaryName &SN,
    const std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>
        &EncodingMap) const {
  Object Result;

  Result["summary_name"] = summaryNameToJSON(SN);
  Result["summary_data"] = encodingDataMapToJSON(EncodingMap);

  return Result;
}

//----------------------------------------------------------------------------
// EncodingSummaryDataMap
//----------------------------------------------------------------------------

llvm::Expected<std::map<
    SummaryName, std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>>
JSONFormat::encodingSummaryDataMapFromJSON(
    const Array &SummaryDataArray) const {
  std::map<SummaryName,
           std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
      EncodingSummaryDataMap;

  for (const auto &[Index, SummaryDataMapEntryValue] :
       llvm::enumerate(SummaryDataArray)) {

    const Object *OptSummaryDataMapEntryObject =
        SummaryDataMapEntryValue.getAsObject();
    if (!OptSummaryDataMapEntryObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "SummaryData entry", Index, "object")
          .build();
    }

    auto ExpectedEntry =
        encodingSummaryDataMapEntryFromJSON(*OptSummaryDataMapEntryObject);
    if (!ExpectedEntry) {
      return ErrorBuilder::wrap(ExpectedEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "SummaryData entry", Index)
          .build();
    }

    auto [SummaryIt, SummaryInserted] =
        EncodingSummaryDataMap.emplace(std::move(*ExpectedEntry));
    if (!SummaryInserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "SummaryData entry", Index, SummaryIt->first)
          .build();
    }
  }

  return std::move(EncodingSummaryDataMap);
}

Array JSONFormat::encodingSummaryDataMapToJSON(
    const std::map<SummaryName,
                   std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
        &EncodingSummaryDataMap) const {
  Array Result;
  Result.reserve(EncodingSummaryDataMap.size());

  for (const auto &[SN, EncodingMap] : EncodingSummaryDataMap) {
    Result.push_back(encodingSummaryDataMapEntryToJSON(SN, EncodingMap));
  }

  return Result;
}

//----------------------------------------------------------------------------
// TUSummaryEncoding
//----------------------------------------------------------------------------

llvm::Expected<TUSummaryEncoding>
JSONFormat::readTUSummaryEncoding(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject, "TUSummary",
                                "object")
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  const Object &RootObject = *RootObjectPtr;

  const Object *TUNamespaceObject = RootObject.getObject("tu_namespace");
  if (!TUNamespaceObject) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "BuildNamespace", "tu_namespace", "object")
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  auto ExpectedTUNamespace = buildNamespaceFromJSON(*TUNamespaceObject);
  if (!ExpectedTUNamespace) {
    return ErrorBuilder::wrap(ExpectedTUNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "BuildNamespace",
                 "tu_namespace")
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  TUSummaryEncoding Encoding(std::move(*ExpectedTUNamespace));

  {
    const Array *IdTableArray = RootObject.getArray("id_table");
    if (!IdTableArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "IdTable", "id_table", "array")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    auto ExpectedIdTable = entityIdTableFromJSON(*IdTableArray);
    if (!ExpectedIdTable) {
      return ErrorBuilder::wrap(ExpectedIdTable.takeError())
          .context(ErrorMessages::ReadingFromField, "IdTable", "id_table")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    getIdTable(Encoding) = std::move(*ExpectedIdTable);
  }

  {
    const Array *LinkageTableArray = RootObject.getArray("linkage_table");
    if (!LinkageTableArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "LinkageTable", "linkage_table", "array")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    auto ExpectedIdRange =
        llvm::make_second_range(getEntities(getIdTable(Encoding)));
    std::set<EntityId> ExpectedIds(ExpectedIdRange.begin(),
                                   ExpectedIdRange.end());

    // Move ExpectedIds in since linkageTableFromJSON consumes it to verify
    // that the linkage table contains exactly the ids present in the IdTable.
    auto ExpectedLinkageTable =
        linkageTableFromJSON(*LinkageTableArray, std::move(ExpectedIds));
    if (!ExpectedLinkageTable) {
      return ErrorBuilder::wrap(ExpectedLinkageTable.takeError())
          .context(ErrorMessages::ReadingFromField, "LinkageTable",
                   "linkage_table")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    getLinkageTable(Encoding) = std::move(*ExpectedLinkageTable);
  }

  {
    const Array *SummaryDataArray = RootObject.getArray("data");
    if (!SummaryDataArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "SummaryData entries", "data", "array")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    auto ExpectedEncodingSummaryDataMap =
        encodingSummaryDataMapFromJSON(*SummaryDataArray);
    if (!ExpectedEncodingSummaryDataMap) {
      return ErrorBuilder::wrap(ExpectedEncodingSummaryDataMap.takeError())
          .context(ErrorMessages::ReadingFromField, "SummaryData entries",
                   "data")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    getData(Encoding) = std::move(*ExpectedEncodingSummaryDataMap);
  }

  return std::move(Encoding);
}

llvm::Error
JSONFormat::writeTUSummaryEncoding(const TUSummaryEncoding &SummaryEncoding,
                                   llvm::StringRef Path) {
  Object RootObject;

  RootObject["tu_namespace"] =
      buildNamespaceToJSON(getTUNamespace(SummaryEncoding));

  RootObject["id_table"] = entityIdTableToJSON(getIdTable(SummaryEncoding));

  RootObject["linkage_table"] =
      linkageTableToJSON(getLinkageTable(SummaryEncoding));

  RootObject["data"] = encodingSummaryDataMapToJSON(getData(SummaryEncoding));

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "TUSummary", Path)
        .build();
  }

  return llvm::Error::success();
}

} // namespace clang::ssaf
