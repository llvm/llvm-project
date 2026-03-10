//===- JSONFormatImpl.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/Analysis/Scalable/Serialization/SerializationFormatRegistry.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/Support/Registry.h"

LLVM_INSTANTIATE_REGISTRY(llvm::Registry<clang::ssaf::JSONFormat::FormatInfo>)

static clang::ssaf::SerializationFormatRegistry::Add<clang::ssaf::JSONFormat>
    RegisterJSONFormat("JSON", "JSON serialization format");

namespace clang::ssaf {

void initializeJSONFormat() {}

//----------------------------------------------------------------------------
// JSON Reader and Writer
//----------------------------------------------------------------------------

llvm::Expected<Value> readJSON(llvm::StringRef Path) {
  if (!llvm::sys::fs::exists(Path)) {
    return ErrorBuilder::create(std::errc::no_such_file_or_directory,
                                ErrorMessages::FailedToReadFile, Path,
                                ErrorMessages::FileNotFound)
        .build();
  }

  if (llvm::sys::fs::is_directory(Path)) {
    return ErrorBuilder::create(std::errc::is_a_directory,
                                ErrorMessages::FailedToReadFile, Path,
                                ErrorMessages::FileIsDirectory)
        .build();
  }

  if (!Path.ends_with(JSONFormatFileExtension)) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadFile, Path,
                                llvm::formatv(ErrorMessages::FileIsNotJSON,
                                              JSONFormatFileExtension))
        .build();
  }

  auto BufferOrError = llvm::MemoryBuffer::getFile(Path);
  if (!BufferOrError) {
    const std::error_code EC = BufferOrError.getError();
    return ErrorBuilder::create(EC, ErrorMessages::FailedToReadFile, Path,
                                EC.message())
        .build();
  }

  return llvm::json::parse(BufferOrError.get()->getBuffer());
}

llvm::Error writeJSON(Value &&V, llvm::StringRef Path) {
  if (llvm::sys::fs::exists(Path)) {
    return ErrorBuilder::create(std::errc::file_exists,
                                ErrorMessages::FailedToWriteFile, Path,
                                ErrorMessages::FileExists)
        .build();
  }

  llvm::StringRef Dir = llvm::sys::path::parent_path(Path);
  if (!Dir.empty() && !llvm::sys::fs::is_directory(Dir)) {
    return ErrorBuilder::create(std::errc::no_such_file_or_directory,
                                ErrorMessages::FailedToWriteFile, Path,
                                ErrorMessages::ParentDirectoryNotFound)
        .build();
  }

  if (!Path.ends_with(JSONFormatFileExtension)) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToWriteFile, Path,
                                llvm::formatv(ErrorMessages::FileIsNotJSON,
                                              JSONFormatFileExtension))
        .build();
  }

  std::error_code EC;
  llvm::raw_fd_ostream OutStream(Path, EC, llvm::sys::fs::OF_Text);

  if (EC) {
    return ErrorBuilder::create(EC, ErrorMessages::FailedToWriteFile, Path,
                                EC.message())
        .build();
  }

  OutStream << llvm::formatv("{0:2}\n", V);
  OutStream.flush();

  // This path handles post-write stream errors (e.g. ENOSPC after buffered
  // writes). It is difficult to exercise in unit tests so it is intentionally
  // left without test coverage.
  if (OutStream.has_error()) {
    return ErrorBuilder::create(OutStream.error(),
                                ErrorMessages::FailedToWriteFile, Path,
                                OutStream.error().message())
        .build();
  }

  return llvm::Error::success();
}

//----------------------------------------------------------------------------
// JSONFormat Static Methods
//----------------------------------------------------------------------------

std::map<SummaryName, JSONFormat::FormatInfo> JSONFormat::initFormatInfos() {
  std::map<SummaryName, FormatInfo> FormatInfos;
  for (const auto &FormatInfoEntry : llvm::Registry<FormatInfo>::entries()) {
    std::unique_ptr<FormatInfo> Info = FormatInfoEntry.instantiate();
    bool Inserted = FormatInfos.try_emplace(Info->ForSummary, *Info).second;
    if (!Inserted) {
      llvm::report_fatal_error(
          "FormatInfo is already registered for summary: " +
          Info->ForSummary.str());
    }
  }
  return FormatInfos;
}

//----------------------------------------------------------------------------
// SummaryName
//----------------------------------------------------------------------------

SummaryName summaryNameFromJSON(llvm::StringRef SummaryNameStr) {
  return SummaryName(SummaryNameStr.str());
}

llvm::StringRef summaryNameToJSON(const SummaryName &SN) { return SN.str(); }

//----------------------------------------------------------------------------
// EntityId
//----------------------------------------------------------------------------

EntityId JSONFormat::entityIdFromJSON(const uint64_t EntityIdIndex) const {
  return makeEntityId(static_cast<size_t>(EntityIdIndex));
}

uint64_t JSONFormat::entityIdToJSON(EntityId EI) const {
  return static_cast<uint64_t>(getIndex(EI));
}

llvm::Expected<EntityId>
JSONFormat::entityIdFromJSONObject(const Object &EntityIdObject) {
  if (EntityIdObject.size() != 1) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadEntityIdObject,
                                JSONEntityIdKey)
        .build();
  }

  const llvm::json::Value *AtVal = EntityIdObject.get(JSONEntityIdKey);
  if (!AtVal) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadEntityIdObject,
                                JSONEntityIdKey)
        .build();
  }

  std::optional<uint64_t> OptEntityIdIndex = AtVal->getAsUINT64();
  if (!OptEntityIdIndex) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadEntityIdObject,
                                JSONEntityIdKey)
        .build();
  }

  return makeEntityId(static_cast<size_t>(*OptEntityIdIndex));
}

Object JSONFormat::entityIdToJSONObject(EntityId EI) {
  Object Result;
  Result[JSONEntityIdKey] = static_cast<uint64_t>(getIndex(EI));
  return Result;
}

//----------------------------------------------------------------------------
// BuildNamespaceKind
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespaceKind>
buildNamespaceKindFromJSON(llvm::StringRef BuildNamespaceKindStr) {
  auto OptBuildNamespaceKind =
      buildNamespaceKindFromString(BuildNamespaceKindStr);
  if (!OptBuildNamespaceKind) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::InvalidBuildNamespaceKind,
                                BuildNamespaceKindStr)
        .build();
  }
  return *OptBuildNamespaceKind;
}

// Provided for consistency with respect to rest of the codebase.
llvm::StringRef buildNamespaceKindToJSON(BuildNamespaceKind BNK) {
  return buildNamespaceKindToString(BNK);
}

//----------------------------------------------------------------------------
// BuildNamespace
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespace>
JSONFormat::buildNamespaceFromJSON(const Object &BuildNamespaceObject) const {
  auto OptBuildNamespaceKindStr = BuildNamespaceObject.getString("kind");
  if (!OptBuildNamespaceKindStr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "BuildNamespaceKind", "kind", "string")
        .build();
  }

  auto ExpectedKind = buildNamespaceKindFromJSON(*OptBuildNamespaceKindStr);
  if (!ExpectedKind) {
    return ErrorBuilder::wrap(ExpectedKind.takeError())
        .context(ErrorMessages::ReadingFromField, "BuildNamespaceKind", "kind")
        .build();
  }

  auto OptNameStr = BuildNamespaceObject.getString("name");
  if (!OptNameStr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "BuildNamespaceName", "name", "string")
        .build();
  }

  return {BuildNamespace(*ExpectedKind, *OptNameStr)};
}

Object JSONFormat::buildNamespaceToJSON(const BuildNamespace &BN) const {
  Object Result;
  Result["kind"] = buildNamespaceKindToJSON(getKind(BN));
  Result["name"] = getName(BN);
  return Result;
}

//----------------------------------------------------------------------------
// NestedBuildNamespace
//----------------------------------------------------------------------------

llvm::Expected<NestedBuildNamespace> JSONFormat::nestedBuildNamespaceFromJSON(
    const Array &NestedBuildNamespaceArray) const {
  std::vector<BuildNamespace> Namespaces;

  size_t NamespaceCount = NestedBuildNamespaceArray.size();
  Namespaces.reserve(NamespaceCount);

  for (const auto &[Index, BuildNamespaceValue] :
       llvm::enumerate(NestedBuildNamespaceArray)) {
    const Object *BuildNamespaceObject = BuildNamespaceValue.getAsObject();
    if (!BuildNamespaceObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "BuildNamespace", Index, "object")
          .build();
    }

    auto ExpectedBuildNamespace = buildNamespaceFromJSON(*BuildNamespaceObject);
    if (!ExpectedBuildNamespace) {
      return ErrorBuilder::wrap(ExpectedBuildNamespace.takeError())
          .context(ErrorMessages::ReadingFromIndex, "BuildNamespace", Index)
          .build();
    }

    Namespaces.push_back(std::move(*ExpectedBuildNamespace));
  }

  return NestedBuildNamespace(std::move(Namespaces));
}

Array JSONFormat::nestedBuildNamespaceToJSON(
    const NestedBuildNamespace &NBN) const {
  Array Result;
  const auto &Namespaces = getNamespaces(NBN);
  Result.reserve(Namespaces.size());

  for (const auto &BN : Namespaces) {
    Result.push_back(buildNamespaceToJSON(BN));
  }

  return Result;
}

//----------------------------------------------------------------------------
// EntityName
//----------------------------------------------------------------------------

llvm::Expected<EntityName>
JSONFormat::entityNameFromJSON(const Object &EntityNameObject) const {
  const auto OptUSR = EntityNameObject.getString("usr");
  if (!OptUSR) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField, "USR",
                                "usr", "string")
        .build();
  }

  const auto OptSuffix = EntityNameObject.getString("suffix");
  if (!OptSuffix) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "Suffix", "suffix", "string")
        .build();
  }

  const Array *OptNamespaceArray = EntityNameObject.getArray("namespace");
  if (!OptNamespaceArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "NestedBuildNamespace", "namespace", "array")
        .build();
  }

  auto ExpectedNamespace = nestedBuildNamespaceFromJSON(*OptNamespaceArray);
  if (!ExpectedNamespace) {
    return ErrorBuilder::wrap(ExpectedNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "NestedBuildNamespace",
                 "namespace")
        .build();
  }

  return EntityName{*OptUSR, *OptSuffix, std::move(*ExpectedNamespace)};
}

Object JSONFormat::entityNameToJSON(const EntityName &EN) const {
  Object Result;
  Result["usr"] = getUSR(EN);
  Result["suffix"] = getSuffix(EN);
  Result["namespace"] = nestedBuildNamespaceToJSON(getNamespace(EN));
  return Result;
}

//----------------------------------------------------------------------------
// EntityLinkageType
//----------------------------------------------------------------------------

llvm::Expected<EntityLinkageType>
entityLinkageTypeFromJSON(llvm::StringRef EntityLinkageTypeStr) {
  auto OptEntityLinkageType = entityLinkageTypeFromString(EntityLinkageTypeStr);
  if (!OptEntityLinkageType) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::InvalidEntityLinkageType,
                                EntityLinkageTypeStr)
        .build();
  }
  return *OptEntityLinkageType;
}

// Provided for consistency with respect to rest of the codebase.
llvm::StringRef entityLinkageTypeToJSON(EntityLinkageType LT) {
  return entityLinkageTypeToString(LT);
}

//----------------------------------------------------------------------------
// EntityLinkage
//----------------------------------------------------------------------------

llvm::Expected<EntityLinkage>
JSONFormat::entityLinkageFromJSON(const Object &EntityLinkageObject) const {
  auto OptLinkageStr = EntityLinkageObject.getString("type");
  if (!OptLinkageStr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityLinkageType", "type", "string")
        .build();
  }

  auto ExpectedLinkageType = entityLinkageTypeFromJSON(*OptLinkageStr);
  if (!ExpectedLinkageType) {
    return ErrorBuilder::wrap(ExpectedLinkageType.takeError())
        .context(ErrorMessages::ReadingFromField, "EntityLinkageType", "type")
        .build();
  }

  return EntityLinkage(*ExpectedLinkageType);
}

Object JSONFormat::entityLinkageToJSON(const EntityLinkage &EL) const {
  Object Result;
  Result["type"] = entityLinkageTypeToJSON(getLinkage(EL));
  return Result;
}

//----------------------------------------------------------------------------
// EntityIdTableEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityName, EntityId>>
JSONFormat::entityIdTableEntryFromJSON(
    const Object &EntityIdTableEntryObject) const {

  const Object *OptEntityNameObject =
      EntityIdTableEntryObject.getObject("name");
  if (!OptEntityNameObject) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityName", "name", "object")
        .build();
  }

  auto ExpectedEntityName = entityNameFromJSON(*OptEntityNameObject);
  if (!ExpectedEntityName) {
    return ErrorBuilder::wrap(ExpectedEntityName.takeError())
        .context(ErrorMessages::ReadingFromField, "EntityName", "name")
        .build();
  }

  const Value *EntityIdIntValue = EntityIdTableEntryObject.get("id");
  if (!EntityIdIntValue) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityId", "id",
                                "number (unsigned 64-bit integer)")
        .build();
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityId", "id",
                                "number (unsigned 64-bit integer)")
        .build();
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  return std::make_pair(std::move(*ExpectedEntityName), std::move(EI));
}

Object JSONFormat::entityIdTableEntryToJSON(const EntityName &EN,
                                            EntityId EI) const {
  Object Entry;
  Entry["id"] = entityIdToJSON(EI);
  Entry["name"] = entityNameToJSON(EN);
  return Entry;
}

//----------------------------------------------------------------------------
// EntityIdTable
//----------------------------------------------------------------------------

llvm::Expected<EntityIdTable>
JSONFormat::entityIdTableFromJSON(const Array &EntityIdTableArray) const {
  EntityIdTable IdTable;
  std::map<EntityName, EntityId> &Entities = getEntities(IdTable);

  for (const auto &[Index, EntityIdTableEntryValue] :
       llvm::enumerate(EntityIdTableArray)) {
    const Object *OptEntityIdTableEntryObject =
        EntityIdTableEntryValue.getAsObject();
    if (!OptEntityIdTableEntryObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "EntityIdTable entry", Index, "object")
          .build();
    }

    auto ExpectedEntityIdTableEntry =
        entityIdTableEntryFromJSON(*OptEntityIdTableEntryObject);
    if (!ExpectedEntityIdTableEntry) {
      return ErrorBuilder::wrap(ExpectedEntityIdTableEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "EntityIdTable entry",
                   Index)
          .build();
    }

    auto [EntityIt, EntityInserted] =
        Entities.emplace(std::move(*ExpectedEntityIdTableEntry));
    if (!EntityInserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "EntityIdTable entry", Index,
                                  EntityIt->second)
          .build();
    }
  }

  return IdTable;
}

Array JSONFormat::entityIdTableToJSON(const EntityIdTable &IdTable) const {
  Array EntityIdTableArray;
  const auto &Entities = getEntities(IdTable);
  EntityIdTableArray.reserve(Entities.size());

  for (const auto &[EntityName, EntityId] : Entities) {
    EntityIdTableArray.push_back(
        entityIdTableEntryToJSON(EntityName, EntityId));
  }

  return EntityIdTableArray;
}

//----------------------------------------------------------------------------
// LinkageTableEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityId, EntityLinkage>>
JSONFormat::linkageTableEntryFromJSON(
    const Object &LinkageTableEntryObject) const {
  const Value *EntityIdIntValue = LinkageTableEntryObject.get("id");
  if (!EntityIdIntValue) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityId", "id",
                                "number (unsigned 64-bit integer)")
        .build();
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityId", "id",
                                "number (unsigned 64-bit integer)")
        .build();
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  const Object *OptEntityLinkageObject =
      LinkageTableEntryObject.getObject("linkage");
  if (!OptEntityLinkageObject) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "EntityLinkage", "linkage", "object")
        .build();
  }

  auto ExpectedEntityLinkage = entityLinkageFromJSON(*OptEntityLinkageObject);
  if (!ExpectedEntityLinkage) {
    return ErrorBuilder::wrap(ExpectedEntityLinkage.takeError())
        .context(ErrorMessages::ReadingFromField, "EntityLinkage", "linkage")
        .build();
  }

  return std::make_pair(std::move(EI), std::move(*ExpectedEntityLinkage));
}

Object JSONFormat::linkageTableEntryToJSON(EntityId EI,
                                           const EntityLinkage &EL) const {
  Object Entry;
  Entry["id"] = entityIdToJSON(EI);
  Entry["linkage"] = entityLinkageToJSON(EL);
  return Entry;
}

//----------------------------------------------------------------------------
// LinkageTable
//----------------------------------------------------------------------------

// ExpectedIds is the set of EntityIds from the IdTable that must appear in the
// linkage table—no more, no fewer. It is taken by value because it is consumed
// during parsing: each successfully matched id is erased from the set, and any
// ids remaining at the end are reported as missing.
llvm::Expected<std::map<EntityId, EntityLinkage>>
JSONFormat::linkageTableFromJSON(const Array &LinkageTableArray,
                                 std::set<EntityId> ExpectedIds) const {
  std::map<EntityId, EntityLinkage> LinkageTable;

  for (const auto &[Index, LinkageTableEntryValue] :
       llvm::enumerate(LinkageTableArray)) {
    const Object *OptLinkageTableEntryObject =
        LinkageTableEntryValue.getAsObject();
    if (!OptLinkageTableEntryObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "LinkageTable entry", Index, "object")
          .build();
    }

    auto ExpectedLinkageTableEntry =
        linkageTableEntryFromJSON(*OptLinkageTableEntryObject);
    if (!ExpectedLinkageTableEntry) {
      return ErrorBuilder::wrap(ExpectedLinkageTableEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "LinkageTable entry", Index)
          .build();
    }

    const EntityId EI = ExpectedLinkageTableEntry->first;

    auto [It, Inserted] =
        LinkageTable.insert(std::move(*ExpectedLinkageTableEntry));
    if (!Inserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "LinkageTable entry", Index, It->first)
          .build();
    }

    if (ExpectedIds.erase(EI) == 0) {
      return ErrorBuilder::create(
                 std::errc::invalid_argument,
                 ErrorMessages::FailedToDeserializeLinkageTableExtraId, EI)
          .context(ErrorMessages::ReadingFromIndex, "LinkageTable entry", Index)
          .build();
    }
  }

  if (!ExpectedIds.empty()) {
    return ErrorBuilder::create(
               std::errc::invalid_argument,
               ErrorMessages::FailedToDeserializeLinkageTableMissingId,
               *ExpectedIds.begin())
        .build();
  }

  return LinkageTable;
}

Array JSONFormat::linkageTableToJSON(
    const std::map<EntityId, EntityLinkage> &LinkageTable) const {
  Array Result;
  Result.reserve(LinkageTable.size());

  for (const auto &[EI, EL] : LinkageTable) {
    Result.push_back(linkageTableEntryToJSON(EI, EL));
  }

  return Result;
}

//----------------------------------------------------------------------------
// EntitySummary
//----------------------------------------------------------------------------

llvm::Expected<std::unique_ptr<EntitySummary>>
JSONFormat::entitySummaryFromJSON(const SummaryName &SN,
                                  const Object &EntitySummaryObject,
                                  EntityIdTable &IdTable) const {
  auto InfoIt = FormatInfos.find(SN);
  if (InfoIt == FormatInfos.end()) {
    return ErrorBuilder::create(
               std::errc::invalid_argument,
               ErrorMessages::FailedToDeserializeEntitySummaryNoFormatInfo, SN)
        .build();
  }

  const auto &InfoEntry = InfoIt->second;
  assert(InfoEntry.ForSummary == SN);

  return InfoEntry.Deserialize(EntitySummaryObject, IdTable,
                               entityIdFromJSONObject);
}

llvm::Expected<Object>
JSONFormat::entitySummaryToJSON(const SummaryName &SN,
                                const EntitySummary &ES) const {
  auto InfoIt = FormatInfos.find(SN);
  if (InfoIt == FormatInfos.end()) {
    return ErrorBuilder::create(
               std::errc::invalid_argument,
               ErrorMessages::FailedToSerializeEntitySummaryNoFormatInfo, SN)
        .build();
  }

  const auto &InfoEntry = InfoIt->second;
  assert(InfoEntry.ForSummary == SN);

  return InfoEntry.Serialize(ES, entityIdToJSONObject);
}

//----------------------------------------------------------------------------
// EntityDataMapEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityId, std::unique_ptr<EntitySummary>>>
JSONFormat::entityDataMapEntryFromJSON(const Object &EntityDataMapEntryObject,
                                       const SummaryName &SN,
                                       EntityIdTable &IdTable) const {

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

  auto ExpectedEntitySummary =
      entitySummaryFromJSON(SN, *OptEntitySummaryObject, IdTable);
  if (!ExpectedEntitySummary) {
    return ErrorBuilder::wrap(ExpectedEntitySummary.takeError())
        .context(ErrorMessages::ReadingFromField, "EntitySummary",
                 "entity_summary")
        .build();
  }

  if (*ExpectedEntitySummary == nullptr) {
    return ErrorBuilder::create(
               std::errc::invalid_argument,
               ErrorMessages::FailedToDeserializeEntitySummaryMissingData, SN)
        .build();
  }

  auto ActualSN = (*ExpectedEntitySummary)->getSummaryName();
  if (SN != ActualSN) {
    return ErrorBuilder::create(
               std::errc::invalid_argument,
               ErrorMessages::
                   FailedToDeserializeEntitySummaryMismatchedSummaryName,
               SN, ActualSN)
        .build();
  }

  return std::make_pair(std::move(EI), std::move(*ExpectedEntitySummary));
}

llvm::Expected<Object> JSONFormat::entityDataMapEntryToJSON(
    const EntityId EI, const std::unique_ptr<EntitySummary> &EntitySummary,
    const SummaryName &SN) const {
  Object Entry;

  Entry["entity_id"] = entityIdToJSON(EI);

  if (!EntitySummary) {
    ErrorBuilder::fatal(
        ErrorMessages::FailedToSerializeEntitySummaryMissingData, SN);
  }

  const auto ActualSN = EntitySummary->getSummaryName();
  if (SN != ActualSN) {
    ErrorBuilder::fatal(
        ErrorMessages::FailedToSerializeEntitySummaryMismatchedSummaryName, SN,
        ActualSN);
  }

  auto ExpectedEntitySummaryObject = entitySummaryToJSON(SN, *EntitySummary);
  if (!ExpectedEntitySummaryObject) {
    return ErrorBuilder::wrap(ExpectedEntitySummaryObject.takeError())
        .context(ErrorMessages::WritingToField, "EntitySummary",
                 "entity_summary")
        .build();
  }

  Entry["entity_summary"] = std::move(*ExpectedEntitySummaryObject);

  return Entry;
}

//----------------------------------------------------------------------------
// EntityDataMap
//----------------------------------------------------------------------------

llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummary>>>
JSONFormat::entityDataMapFromJSON(const SummaryName &SN,
                                  const Array &EntityDataArray,
                                  EntityIdTable &IdTable) const {
  std::map<EntityId, std::unique_ptr<EntitySummary>> EntityDataMap;

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

    auto ExpectedEntityDataMapEntry =
        entityDataMapEntryFromJSON(*OptEntityDataMapEntryObject, SN, IdTable);
    if (!ExpectedEntityDataMapEntry) {
      return ErrorBuilder::wrap(ExpectedEntityDataMapEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "EntitySummary entry",
                   Index)
          .build();
    }

    auto [DataIt, DataInserted] =
        EntityDataMap.insert(std::move(*ExpectedEntityDataMapEntry));
    if (!DataInserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "EntitySummary entry", Index, DataIt->first)
          .build();
    }
  }

  return std::move(EntityDataMap);
}

llvm::Expected<Array> JSONFormat::entityDataMapToJSON(
    const SummaryName &SN,
    const std::map<EntityId, std::unique_ptr<EntitySummary>> &EntityDataMap)
    const {
  Array Result;
  Result.reserve(EntityDataMap.size());

  for (const auto &[Index, EntityDataMapEntry] :
       llvm::enumerate(EntityDataMap)) {
    const auto &[EntityId, EntitySummary] = EntityDataMapEntry;

    auto ExpectedEntityDataMapEntryObject =
        entityDataMapEntryToJSON(EntityId, EntitySummary, SN);

    if (!ExpectedEntityDataMapEntryObject) {
      return ErrorBuilder::wrap(ExpectedEntityDataMapEntryObject.takeError())
          .context(ErrorMessages::WritingToIndex, "EntitySummary entry", Index)
          .build();
    }

    Result.push_back(std::move(*ExpectedEntityDataMapEntryObject));
  }

  return Result;
}

//----------------------------------------------------------------------------
// SummaryDataMapEntry
//----------------------------------------------------------------------------

llvm::Expected<
    std::pair<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
JSONFormat::summaryDataMapEntryFromJSON(const Object &SummaryDataMapEntryObject,
                                        EntityIdTable &IdTable) const {

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

  auto ExpectedEntityDataMap =
      entityDataMapFromJSON(SN, *OptEntityDataArray, IdTable);
  if (!ExpectedEntityDataMap) {
    return ErrorBuilder::wrap(ExpectedEntityDataMap.takeError())
        .context(ErrorMessages::ReadingFromField, "EntitySummary entries",
                 "summary_data")
        .build();
  }

  return std::make_pair(std::move(SN), std::move(*ExpectedEntityDataMap));
}

llvm::Expected<Object> JSONFormat::summaryDataMapEntryToJSON(
    const SummaryName &SN,
    const std::map<EntityId, std::unique_ptr<EntitySummary>> &SD) const {
  Object Result;

  Result["summary_name"] = summaryNameToJSON(SN);

  auto ExpectedSummaryDataArray = entityDataMapToJSON(SN, SD);
  if (!ExpectedSummaryDataArray) {
    return ErrorBuilder::wrap(ExpectedSummaryDataArray.takeError())
        .context(ErrorMessages::WritingToField, "EntitySummary entries",
                 "summary_data")
        .build();
  }

  Result["summary_data"] = std::move(*ExpectedSummaryDataArray);

  return Result;
}

//----------------------------------------------------------------------------
// SummaryDataMap
//----------------------------------------------------------------------------

llvm::Expected<
    std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
JSONFormat::summaryDataMapFromJSON(const Array &SummaryDataArray,
                                   EntityIdTable &IdTable) const {
  std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>
      SummaryDataMap;

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

    auto ExpectedSummaryDataMapEntry =
        summaryDataMapEntryFromJSON(*OptSummaryDataMapEntryObject, IdTable);
    if (!ExpectedSummaryDataMapEntry) {
      return ErrorBuilder::wrap(ExpectedSummaryDataMapEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "SummaryData entry", Index)
          .build();
    }

    auto [SummaryIt, SummaryInserted] =
        SummaryDataMap.emplace(std::move(*ExpectedSummaryDataMapEntry));
    if (!SummaryInserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "SummaryData entry", Index, SummaryIt->first)
          .build();
    }
  }

  return std::move(SummaryDataMap);
}

llvm::Expected<Array> JSONFormat::summaryDataMapToJSON(
    const std::map<SummaryName,
                   std::map<EntityId, std::unique_ptr<EntitySummary>>>
        &SummaryDataMap) const {
  Array Result;
  Result.reserve(SummaryDataMap.size());

  for (const auto &[Index, SummaryDataMapEntry] :
       llvm::enumerate(SummaryDataMap)) {
    const auto &[SummaryName, DataMap] = SummaryDataMapEntry;

    auto ExpectedSummaryDataMapObject =
        summaryDataMapEntryToJSON(SummaryName, DataMap);
    if (!ExpectedSummaryDataMapObject) {
      return ErrorBuilder::wrap(ExpectedSummaryDataMapObject.takeError())
          .context(ErrorMessages::WritingToIndex, "SummaryData entry", Index)
          .build();
    }

    Result.push_back(std::move(*ExpectedSummaryDataMapObject));
  }

  return std::move(Result);
}

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

  // All EntitySummaryEncoding objects stored in a TUSummaryEncoding or
  // LUSummaryEncoding read by JSONFormat are JSONEntitySummaryEncoding
  // instances, since encodingDataMapEntryFromJSON is the only place that
  // creates them.
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
  if (!ExpectedEncodingDataMap) {
    return ErrorBuilder::wrap(ExpectedEncodingDataMap.takeError())
        .context(ErrorMessages::ReadingFromField, "EntitySummary entries",
                 "summary_data")
        .build();
  }

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

} // namespace clang::ssaf
