#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Registry.h"

using namespace clang::ssaf;

namespace {
// Helper to wrap an error with additional context
template <typename... Args>
llvm::Error wrapError(llvm::Error E, std::errc ErrorCode, const char *Fmt,
                      Args &&...Vals) {
  return llvm::joinErrors(
      llvm::createStringError(ErrorCode, Fmt, std::forward<Args>(Vals)...),
      std::move(E));
}

// Convenience version that defaults to invalid_argument
template <typename... Args>
llvm::Error wrapError(llvm::Error E, const char *Fmt, Args &&...Vals) {
  return wrapError(std::move(E), std::errc::invalid_argument, Fmt,
                   std::forward<Args>(Vals)...);
}

} // namespace

//----------------------------------------------------------------------------
// JSON Reader and Writer
//----------------------------------------------------------------------------

llvm::Error isJSONFile(llvm::StringRef Path) {
  if (!llvm::sys::fs::exists(Path))
    return llvm::createStringError(std::errc::no_such_file_or_directory,
                                   "file does not exist: '%s'",
                                   Path.str().c_str());

  if (!Path.ends_with_insensitive(".json"))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "not a JSON file: '%s'", Path.str().c_str());

  return llvm::Error::success();
}

llvm::Expected<llvm::json::Value> readJSON(llvm::StringRef Path) {
  if (llvm::Error Err = isJSONFile(Path))
    return wrapError(std::move(Err), "failed to validate JSON file '%s'",
                     Path.str().c_str());

  auto BufferOrError = llvm::MemoryBuffer::getFile(Path);
  if (!BufferOrError) {
    return llvm::createStringError(BufferOrError.getError(),
                                   "failed to read file '%s'",
                                   Path.str().c_str());
  }

  return llvm::json::parse(BufferOrError.get()->getBuffer());
}

llvm::Expected<llvm::json::Object> readJSONObject(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON)
    return wrapError(ExpectedJSON.takeError(),
                     "failed to read JSON object from file '%s'",
                     Path.str().c_str());

  llvm::json::Object *Object = ExpectedJSON->getAsObject();
  if (!Object) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to read JSON object from file '%s'",
                                   Path.str().c_str());
  }
  return std::move(*Object);
}

llvm::Error writeJSON(llvm::json::Value &&Value, llvm::StringRef Path) {
  std::error_code EC;
  llvm::raw_fd_ostream OutStream(Path, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    return llvm::createStringError(EC, "failed to open '%s'",
                                   Path.str().c_str());
  }

  OutStream << llvm::formatv("{0:2}\n", Value);
  OutStream.flush();

  if (OutStream.has_error()) {
    return llvm::createStringError(OutStream.error(), "write failed");
  }

  return llvm::Error::success();
}

//----------------------------------------------------------------------------
// JSONFormat Constructor
//----------------------------------------------------------------------------

JSONFormat::JSONFormat(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : SerializationFormat(FS) {
  for (const auto &FormatInfoEntry : llvm::Registry<FormatInfo>::entries()) {
    std::unique_ptr<FormatInfo> Info = FormatInfoEntry.instantiate();
    bool Inserted = FormatInfos.try_emplace(Info->ForSummary, *Info).second;
    if (!Inserted) {
      llvm::report_fatal_error(
          "Format info was already registered for summary name: " +
          Info->ForSummary.str());
    }
  }
}

//----------------------------------------------------------------------------
// EntityId
//----------------------------------------------------------------------------

EntityId JSONFormat::entityIdFromJSON(const uint64_t EntityIdIndex) const {
  return SerializationFormat::makeEntityId(static_cast<size_t>(EntityIdIndex));
}

uint64_t JSONFormat::entityIdToJSON(EntityId EI) const {
  return static_cast<uint64_t>(SerializationFormat::getEntityIdIndex(EI));
}

//----------------------------------------------------------------------------
// BuildNamespaceKind
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespaceKind>
JSONFormat::buildNamespaceKindFromJSON(llvm::StringRef BuildNamespaceKindStr) {
  auto OptBuildNamespaceKind = parseBuildNamespaceKind(BuildNamespaceKindStr);
  if (!OptBuildNamespaceKind) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "invalid 'kind' BuildNamespaceKind value '%s'",
        BuildNamespaceKindStr.str().c_str());
  }

  return *OptBuildNamespaceKind;
}

llvm::StringRef buildNamespaceKindToJSON(BuildNamespaceKind BNK) {
  return toString(BNK);
}

//----------------------------------------------------------------------------
// BuildNamespace
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespace> JSONFormat::buildNamespaceFromJSON(
    const llvm::json::Object &BuildNamespaceObject) {
  auto OptBuildNamespaceKindStr = BuildNamespaceObject.getString("kind");
  if (!OptBuildNamespaceKindStr) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize BuildNamespace: "
        "missing required field 'kind' (expected BuildNamespaceKind)");
  }

  auto ExpectedKind = buildNamespaceKindFromJSON(*OptBuildNamespaceKindStr);
  if (!ExpectedKind)
    return wrapError(ExpectedKind.takeError(),
                     "failed to deserialize BuildNamespace");

  auto OptNameStr = BuildNamespaceObject.getString("name");
  if (!OptNameStr) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to deserialize BuildNamespace: "
                                   "missing required field 'name'");
  }

  return {BuildNamespace(*ExpectedKind, *OptNameStr)};
}

llvm::json::Object
JSONFormat::buildNamespaceToJSON(const BuildNamespace &BN) const {
  llvm::json::Object Result;
  Result["kind"] = buildNamespaceKindToJSON(getBuildNamespaceKind(BN));
  Result["name"] = getBuildNamespaceName(BN);
  return Result;
}

//----------------------------------------------------------------------------
// NestedBuildNamespace
//----------------------------------------------------------------------------

llvm::Expected<NestedBuildNamespace> JSONFormat::nestedBuildNamespaceFromJSON(
    const llvm::json::Array &NestedBuildNamespaceArray) {
  std::vector<BuildNamespace> Namespaces;

  size_t NamespaceCount = NestedBuildNamespaceArray.size();
  Namespaces.reserve(NamespaceCount);
  for (size_t Index = 0; Index < NamespaceCount; ++Index) {
    const llvm::json::Value &BuildNamespaceValue =
        NestedBuildNamespaceArray[Index];

    const llvm::json::Object *BuildNamespaceObject =
        BuildNamespaceValue.getAsObject();
    if (!BuildNamespaceObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize NestedBuildNamespace: "
          "element at index %zu is not a JSON object "
          "(expected BuildNamespace object)",
          Index);
    }

    auto ExpectedBuildNamespace = buildNamespaceFromJSON(*BuildNamespaceObject);
    if (!ExpectedBuildNamespace)
      return wrapError(
          ExpectedBuildNamespace.takeError(),
          "failed to deserialize NestedBuildNamespace at index %zu", Index);

    Namespaces.push_back(std::move(*ExpectedBuildNamespace));
  }

  return NestedBuildNamespace(std::move(Namespaces));
}

llvm::json::Array
JSONFormat::nestedBuildNamespaceToJSON(const NestedBuildNamespace &NBN) const {
  llvm::json::Array Result;
  const auto &Namespaces = getNestedBuildNamespaces(NBN);
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
JSONFormat::entityNameFromJSON(const llvm::json::Object &EntityNameObject) {
  const auto OptUSR = EntityNameObject.getString("usr");
  if (!OptUSR) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityName: "
        "missing required field 'usr' (Unified Symbol Resolution string)");
  }

  const auto OptSuffix = EntityNameObject.getString("suffix");
  if (!OptSuffix) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to deserialize EntityName: "
                                   "missing required field 'suffix'");
  }

  const llvm::json::Array *OptNamespaceArray =
      EntityNameObject.getArray("namespace");
  if (!OptNamespaceArray) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityName: "
        "missing or invalid field 'namespace' "
        "(expected JSON array of BuildNamespace objects)");
  }

  auto ExpectedNamespace = nestedBuildNamespaceFromJSON(*OptNamespaceArray);
  if (!ExpectedNamespace)
    return wrapError(ExpectedNamespace.takeError(),
                     "failed to deserialize EntityName");

  return EntityName{*OptUSR, *OptSuffix, std::move(*ExpectedNamespace)};
}

llvm::json::Object JSONFormat::entityNameToJSON(const EntityName &EN) const {
  llvm::json::Object Result;
  Result["usr"] = getEntityNameUSR(EN);
  Result["suffix"] = getEntityNameSuffix(EN);
  Result["namespace"] = nestedBuildNamespaceToJSON(getEntityNameNamespace(EN));
  return Result;
}

//----------------------------------------------------------------------------
// SummaryName
//----------------------------------------------------------------------------

SummaryName summaryNameFromJSON(llvm::StringRef SummaryNameStr) {
  return SummaryName(SummaryNameStr.str());
}

llvm::StringRef summaryNameToJSON(const SummaryName &SN) { return SN.str(); }

//----------------------------------------------------------------------------
// EntityIdTable
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityName, EntityId>>
JSONFormat::entityIdTableEntryFromJSON(
    const llvm::json::Object &EntityIdTableEntryObject) {

  const llvm::json::Object *OptEntityNameObject =
      EntityIdTableEntryObject.getObject("name");
  if (!OptEntityNameObject) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityIdTable entry: "
        "missing or invalid field 'name' (expected EntityName JSON object)");
  }

  auto ExpectedEntityName = entityNameFromJSON(*OptEntityNameObject);
  if (!ExpectedEntityName)
    return wrapError(ExpectedEntityName.takeError(),
                     "failed to deserialize EntityIdTable entry");

  const llvm::json::Value *EntityIdIntValue =
      EntityIdTableEntryObject.get("id");
  if (!EntityIdIntValue) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityIdTable entry: "
        "missing required field 'id' (expected unsigned integer EntityId)");
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityIdTable entry: "
        "field 'id' is not a valid unsigned 64-bit integer "
        "(expected non-negative EntityId value)");
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  return std::make_pair(std::move(*ExpectedEntityName), std::move(EI));
}

llvm::Expected<EntityIdTable>
JSONFormat::entityIdTableFromJSON(const llvm::json::Array &EntityIdTableArray) {
  const size_t EntityCount = EntityIdTableArray.size();

  EntityIdTable IdTable;
  std::map<EntityName, EntityId> &Entities = getEntities(IdTable);

  for (size_t Index = 0; Index < EntityCount; ++Index) {
    const llvm::json::Value &EntityIdTableEntryValue =
        EntityIdTableArray[Index];

    const llvm::json::Object *OptEntityIdTableEntryObject =
        EntityIdTableEntryValue.getAsObject();

    if (!OptEntityIdTableEntryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize EntityIdTable: "
          "element at index %zu is not a JSON object "
          "(expected EntityIdTable entry with 'id' and 'name' fields)",
          Index);
    }

    auto ExpectedEntityIdTableEntry =
        entityIdTableEntryFromJSON(*OptEntityIdTableEntryObject);
    if (!ExpectedEntityIdTableEntry)
      return wrapError(ExpectedEntityIdTableEntry.takeError(),
                       "failed to deserialize EntityIdTable at index %zu",
                       Index);

    auto [EntityIt, EntityInserted] =
        Entities.emplace(std::move(*ExpectedEntityIdTableEntry));
    if (!EntityInserted) {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "failed to deserialize EntityIdTable: "
                                     "duplicate EntityName found at index %zu "
                                     "(EntityId=%zu already exists in table)",
                                     Index, getEntityIdIndex(EntityIt->second));
    }
  }

  return IdTable;
}

llvm::json::Array
JSONFormat::entityIdTableToJSON(const EntityIdTable &IdTable) const {
  llvm::json::Array EntityIdTableArray;
  const auto &Entities = getEntities(IdTable);
  EntityIdTableArray.reserve(Entities.size());

  for (const auto &[EntityName, EntityId] : Entities) {
    llvm::json::Object Entry;
    Entry["id"] = entityIdToJSON(EntityId);
    Entry["name"] = entityNameToJSON(EntityName);
    EntityIdTableArray.push_back(std::move(Entry));
  }

  return EntityIdTableArray;
}

//----------------------------------------------------------------------------
// EntitySummary
//----------------------------------------------------------------------------

llvm::Expected<std::unique_ptr<EntitySummary>>
JSONFormat::entitySummaryFromJSON(const SummaryName &SN,
                                  const llvm::json::Object &EntitySummaryObject,
                                  EntityIdTable &IdTable) {
  auto InfoIt = FormatInfos.find(SN);
  if (InfoIt == FormatInfos.end()) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntitySummary: "
        "no FormatInfo was registered for summary name: %s",
        SN.str().data());
  }
  const auto &InfoEntry = InfoIt->second;
  assert(InfoEntry.ForSummary == SN);

  EntityIdConverter Converter(*this);
  return InfoEntry.Deserialize(EntitySummaryObject, IdTable, Converter);
}

llvm::json::Object
JSONFormat::entitySummaryToJSON(const SummaryName &SN,
                                const EntitySummary &ES) const {
  auto InfoIt = FormatInfos.find(SN);
  if (InfoIt == FormatInfos.end()) {
    llvm::report_fatal_error(
        "Failed to serialize EntitySummary: no FormatInfo was registered for "
        "summary name: " +
        SN.str());
  }
  const auto &InfoEntry = InfoIt->second;
  assert(InfoEntry.ForSummary == SN);

  EntityIdConverter Converter(*this);
  return InfoEntry.Serialize(ES, Converter);
}

//----------------------------------------------------------------------------
// SummaryData
//----------------------------------------------------------------------------

llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummary>>>
JSONFormat::entityDataMapFromJSON(const SummaryName &SN,
                                  const llvm::json::Array &EntityDataArray,
                                  EntityIdTable &IdTable) {
  std::map<EntityId, std::unique_ptr<EntitySummary>> EntityDataMap;

  size_t Index = 0;
  for (const llvm::json::Value &EntityDataEntryValue : EntityDataArray) {
    const llvm::json::Object *OptEntityDataEntryObject =
        EntityDataEntryValue.getAsObject();
    if (!OptEntityDataEntryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map: "
          "element at index %zu is not a JSON object "
          "(expected object with 'entity_id' and 'entity_summary' fields)",
          Index);
    }

    const llvm::json::Value *EntityIdIntValue =
        OptEntityDataEntryObject->get("entity_id");
    if (!EntityIdIntValue) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map entry "
          "at index %zu: missing required field 'entity_id' "
          "(expected unsigned integer EntityId)",
          Index);
    }

    const std::optional<uint64_t> OptEntityIdInt =
        EntityIdIntValue->getAsUINT64();
    if (!OptEntityIdInt) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map entry "
          "at index %zu: field 'entity_id' is not a valid unsigned 64-bit "
          "integer",
          Index);
    }

    EntityId EI = entityIdFromJSON(*OptEntityIdInt);

    const llvm::json::Object *OptEntitySummaryObject =
        OptEntityDataEntryObject->getObject("entity_summary");
    if (!OptEntitySummaryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map entry "
          "at index %zu: missing or invalid field 'entity_summary' "
          "(expected EntitySummary JSON object)",
          Index);
    }

    auto ExpectedEntitySummary =
        entitySummaryFromJSON(SN, *OptEntitySummaryObject, IdTable);

    if (!ExpectedEntitySummary) {
      return wrapError(
          ExpectedEntitySummary.takeError(),
          "failed to deserialize entity data map entry at index %zu", Index);
    }

    auto [DataIt, DataInserted] = EntityDataMap.insert(
        {std::move(EI), std::move(*ExpectedEntitySummary)});
    if (!DataInserted) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map: "
          "duplicate EntityId (%zu) found at index %zu",
          getEntityIdIndex(DataIt->first), Index);
    }

    ++Index;
  }

  return EntityDataMap;
}

llvm::json::Array JSONFormat::entityDataMapToJSON(
    const SummaryName &SN,
    const std::map<EntityId, std::unique_ptr<EntitySummary>> &EntityDataMap)
    const {
  llvm::json::Array Result;
  Result.reserve(EntityDataMap.size());
  for (const auto &[EntityId, EntitySummary] : EntityDataMap) {
    llvm::json::Object Entry;
    Entry["entity_id"] = entityIdToJSON(EntityId);
    Entry["entity_summary"] = entitySummaryToJSON(SN, *EntitySummary);
    Result.push_back(std::move(Entry));
  }
  return Result;
}

llvm::Expected<
    std::pair<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
JSONFormat::summaryDataMapEntryFromJSON(
    const llvm::json::Object &SummaryDataObject, EntityIdTable &IdTable) {

  std::optional<llvm::StringRef> OptSummaryNameStr =
      SummaryDataObject.getString("summary_name");

  if (!OptSummaryNameStr) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize summary data: "
        "missing required field 'summary_name' "
        "(expected string identifier for the analysis summary)");
  }

  SummaryName SN = summaryNameFromJSON(*OptSummaryNameStr);

  const llvm::json::Array *OptEntityDataArray =
      SummaryDataObject.getArray("data");
  if (!OptEntityDataArray) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize summary data for summary '%s': "
        "missing or invalid field 'data' "
        "(expected JSON array of entity summaries)",
        SN.str().data());
  }

  auto ExpectedEntityDataMap =
      entityDataMapFromJSON(SN, *OptEntityDataArray, IdTable);
  if (!ExpectedEntityDataMap)
    return wrapError(ExpectedEntityDataMap.takeError(),
                     "failed to deserialize summary data for summary '%s'",
                     SN.str().data());

  return std::make_pair(std::move(SN), std::move(*ExpectedEntityDataMap));
}

llvm::json::Object JSONFormat::summaryDataMapEntryToJSON(
    const SummaryName &SN,
    const std::map<EntityId, std::unique_ptr<EntitySummary>> &SD) const {
  llvm::json::Object Result;
  Result["summary_name"] = summaryNameToJSON(SN);
  Result["data"] = entityDataMapToJSON(SN, SD);
  return Result;
}

//----------------------------------------------------------------------------
// TUSummary
//----------------------------------------------------------------------------

llvm::Expected<TUSummary> JSONFormat::readTUSummary(llvm::StringRef Path) {
  // Read the JSON object from the file
  auto ExpectedRootObject = readJSONObject(Path);
  if (!ExpectedRootObject)
    return wrapError(ExpectedRootObject.takeError(),
                     "failed to read TUSummary from '%s'", Path.str().c_str());

  const llvm::json::Object &RootObject = *ExpectedRootObject;

  // Parse TUNamespace field
  const llvm::json::Object *TUNamespaceObject =
      RootObject.getObject("tu_namespace");
  if (!TUNamespaceObject) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to read TUSummary from '%s': "
        "missing or invalid field 'tu_namespace' (expected JSON object)",
        Path.str().c_str());
  }

  auto ExpectedTUNamespace = buildNamespaceFromJSON(*TUNamespaceObject);
  if (!ExpectedTUNamespace)
    return wrapError(ExpectedTUNamespace.takeError(),
                     "failed to read TUSummary from '%s'", Path.str().c_str());

  TUSummary Summary(std::move(*ExpectedTUNamespace));

  // Parse IdTable field
  {
    const llvm::json::Array *IdTableArray = RootObject.getArray("id_table");
    if (!IdTableArray) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to read TUSummary from '%s': "
          "missing or invalid field 'id_table' (expected JSON array)",
          Path.str().c_str());
    }

    auto ExpectedIdTable = entityIdTableFromJSON(*IdTableArray);
    if (!ExpectedIdTable)
      return wrapError(ExpectedIdTable.takeError(),
                       "failed to read TUSummary from '%s'",
                       Path.str().c_str());

    getIdTable(Summary) = std::move(*ExpectedIdTable);
  }

  // Parse data field
  {
    const llvm::json::Array *SummaryDataArray = RootObject.getArray("data");
    if (!SummaryDataArray) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to read TUSummary from '%s': "
          "missing or invalid field 'data' (expected JSON array)",
          Path.str().c_str());
    }

    // Parse each summary data entry
    std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>
        Data;
    for (const llvm::json::Value &SummaryDataValue : *SummaryDataArray) {
      const llvm::json::Object *SummaryDataObject =
          SummaryDataValue.getAsObject();
      if (!SummaryDataObject) {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "failed to read TUSummary from '%s': "
                                       "data array contains non-object element",
                                       Path.str().c_str());
      }

      auto ExpectedSummaryDataEntry =
          summaryDataMapEntryFromJSON(*SummaryDataObject, getIdTable(Summary));
      if (!ExpectedSummaryDataEntry)
        return wrapError(ExpectedSummaryDataEntry.takeError(),
                         "failed to read TUSummary from '%s'",
                         Path.str().c_str());

      auto [SummaryIt, SummaryInserted] =
          Data.emplace(std::move(*ExpectedSummaryDataEntry));
      if (!SummaryInserted) {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "failed to read TUSummary from '%s': "
                                       "duplicate SummaryName '%s' found",
                                       Path.str().c_str(),
                                       SummaryIt->first.str().data());
      }
    }

    getData(Summary) = std::move(Data);
  }

  return Summary;
}

llvm::Error JSONFormat::writeTUSummary(const TUSummary &S,
                                       llvm::StringRef Path) {
  llvm::json::Object RootObject;

  RootObject["tu_namespace"] = buildNamespaceToJSON(getTUNamespace(S));

  RootObject["id_table"] = entityIdTableToJSON(getIdTable(S));

  {
    llvm::json::Array SummaryDataArray;
    for (const auto &[SummaryName, DataMap] : getData(S)) {
      SummaryDataArray.push_back(
          summaryDataMapEntryToJSON(SummaryName, DataMap));
    }
    RootObject["data"] = std::move(SummaryDataArray);
  }

  // Write the JSON to file
  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return wrapError(std::move(Error), std::errc::io_error,
                     "failed to write TUSummary to '%s'", Path.str().c_str());
  }

  return llvm::Error::success();
}
