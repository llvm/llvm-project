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

//----------------------------------------------------------------------------
// JSON Reader and Writer
//----------------------------------------------------------------------------

llvm::Error isJSONFile(llvm::StringRef Path) {
  if (!llvm::sys::fs::exists(Path))
    return llvm::createStringError(std::errc::no_such_file_or_directory,
                                   "file does not exist: '%s'",
                                   Path.str().c_str());

  if (llvm::sys::fs::is_directory(Path))
    return llvm::createStringError(std::errc::is_a_directory,
                                   "path is a directory, not a file: '%s'",
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
  return *Object;
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

llvm::StringRef buildNamespaceKindToJSON(BuildNamespaceKind BNK) {
  return toString(BNK);
}

SummaryName summaryNameFromJSON(llvm::StringRef SummaryNameStr) {
  return SummaryName(SummaryNameStr.str());
}

llvm::StringRef summaryNameToJSON(const SummaryName &SN) { return SN.str(); }

} // namespace

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

llvm::Expected<BuildNamespaceKind> JSONFormat::buildNamespaceKindFromJSON(
    llvm::StringRef BuildNamespaceKindStr) const {
  auto OptBuildNamespaceKind = parseBuildNamespaceKind(BuildNamespaceKindStr);
  if (!OptBuildNamespaceKind) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "invalid 'kind' BuildNamespaceKind value '%s'",
        BuildNamespaceKindStr.str().c_str());
  }

  return *OptBuildNamespaceKind;
}

//----------------------------------------------------------------------------
// BuildNamespace
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespace> JSONFormat::buildNamespaceFromJSON(
    const llvm::json::Object &BuildNamespaceObject) const {
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
    const llvm::json::Array &NestedBuildNamespaceArray) const {
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

llvm::Expected<EntityName> JSONFormat::entityNameFromJSON(
    const llvm::json::Object &EntityNameObject) const {
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
// EntityIdTable
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityName, EntityId>>
JSONFormat::entityIdTableEntryFromJSON(
    const llvm::json::Object &EntityIdTableEntryObject) const {

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

llvm::Expected<EntityIdTable> JSONFormat::entityIdTableFromJSON(
    const llvm::json::Array &EntityIdTableArray) const {
  EntityIdTable IdTable;
  std::map<EntityName, EntityId> &Entities = getEntities(IdTable);

  for (size_t Index = 0; Index < EntityIdTableArray.size(); ++Index) {
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
                                  EntityIdTable &IdTable) const {
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
// EntityDataMapEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityId, std::unique_ptr<EntitySummary>>>
JSONFormat::entityDataMapEntryFromJSON(
    const llvm::json::Object &EntityDataMapEntryObject, const SummaryName &SN,
    EntityIdTable &IdTable) const {

  const llvm::json::Value *EntityIdIntValue =
      EntityDataMapEntryObject.get("entity_id");
  if (!EntityIdIntValue) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to deserialize EntityDataMap entry: "
                                   "missing required field 'entity_id' "
                                   "(expected unsigned integer EntityId)");
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityDataMap entry: "
        "field 'entity_id' is not a valid unsigned 64-bit integer "
        "(expected non-negative EntityId value)");
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  const llvm::json::Object *OptEntitySummaryObject =
      EntityDataMapEntryObject.getObject("entity_summary");
  if (!OptEntitySummaryObject) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to deserialize EntityDataMap entry: "
                                   "missing or invalid field 'entity_summary' "
                                   "(expected EntitySummary JSON object)");
  }

  auto ExpectedEntitySummary =
      entitySummaryFromJSON(SN, *OptEntitySummaryObject, IdTable);
  if (!ExpectedEntitySummary)
    return wrapError(ExpectedEntitySummary.takeError(),
                     "failed to deserialize EntityDataMap entry");

  return std::make_pair(std::move(EI), std::move(*ExpectedEntitySummary));
}

//----------------------------------------------------------------------------
// EntityDataMap
//----------------------------------------------------------------------------

llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummary>>>
JSONFormat::entityDataMapFromJSON(const SummaryName &SN,
                                  const llvm::json::Array &EntityDataArray,
                                  EntityIdTable &IdTable) const {
  std::map<EntityId, std::unique_ptr<EntitySummary>> EntityDataMap;

  for (size_t Index = 0; Index < EntityDataArray.size(); ++Index) {
    const llvm::json::Value &EntityDataMapEntryValue = EntityDataArray[Index];

    const llvm::json::Object *OptEntityDataMapEntryObject =
        EntityDataMapEntryValue.getAsObject();
    if (!OptEntityDataMapEntryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize EntityDataMap: "
          "element at index %zu is not a JSON object "
          "(expected EntityDataMap entry with 'entity_id' and 'entity_summary' "
          "fields)",
          Index);
    }

    auto ExpectedEntityDataMapEntry =
        entityDataMapEntryFromJSON(*OptEntityDataMapEntryObject, SN, IdTable);
    if (!ExpectedEntityDataMapEntry)
      return wrapError(ExpectedEntityDataMapEntry.takeError(),
                       "failed to deserialize EntityDataMap at index %zu",
                       Index);

    auto [DataIt, DataInserted] =
        EntityDataMap.insert(std::move(*ExpectedEntityDataMapEntry));
    if (!DataInserted) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize EntityDataMap: "
          "duplicate EntityId (%zu) found at index %zu",
          getEntityIdIndex(DataIt->first), Index);
    }
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

//----------------------------------------------------------------------------
// SummaryDataMapEntry
//----------------------------------------------------------------------------

llvm::Expected<
    std::pair<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
JSONFormat::summaryDataMapEntryFromJSON(
    const llvm::json::Object &SummaryDataMapEntryObject,
    EntityIdTable &IdTable) const {

  std::optional<llvm::StringRef> OptSummaryNameStr =
      SummaryDataMapEntryObject.getString("summary_name");

  if (!OptSummaryNameStr) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize SummaryDataMap entry: "
        "missing required field 'summary_name' "
        "(expected string identifier for the analysis summary)");
  }

  SummaryName SN = summaryNameFromJSON(*OptSummaryNameStr);

  const llvm::json::Array *OptEntityDataArray =
      SummaryDataMapEntryObject.getArray("data");
  if (!OptEntityDataArray) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize SummaryDataMap entry: "
        "missing or invalid field 'data' "
        "(expected JSON array of entity data entries)");
  }

  auto ExpectedEntityDataMap =
      entityDataMapFromJSON(SN, *OptEntityDataArray, IdTable);
  if (!ExpectedEntityDataMap)
    return wrapError(
        ExpectedEntityDataMap.takeError(),
        "failed to deserialize SummaryDataMap entry for summary '%s'",
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
// SummaryDataMap
//----------------------------------------------------------------------------

llvm::Expected<
    std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
JSONFormat::summaryDataMapFromJSON(const llvm::json::Array &SummaryDataArray,
                                   EntityIdTable &IdTable) const {
  std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>
      SummaryDataMap;

  for (size_t Index = 0; Index < SummaryDataArray.size(); ++Index) {
    const llvm::json::Value &SummaryDataMapEntryValue = SummaryDataArray[Index];

    const llvm::json::Object *OptSummaryDataMapEntryObject =
        SummaryDataMapEntryValue.getAsObject();
    if (!OptSummaryDataMapEntryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize SummaryDataMap: "
          "element at index %zu is not a JSON object "
          "(expected SummaryDataMap entry with 'summary_name' and 'data' "
          "fields)",
          Index);
    }

    auto ExpectedSummaryDataMapEntry =
        summaryDataMapEntryFromJSON(*OptSummaryDataMapEntryObject, IdTable);
    if (!ExpectedSummaryDataMapEntry)
      return wrapError(ExpectedSummaryDataMapEntry.takeError(),
                       "failed to deserialize SummaryDataMap at index %zu",
                       Index);

    auto [SummaryIt, SummaryInserted] =
        SummaryDataMap.emplace(std::move(*ExpectedSummaryDataMapEntry));
    if (!SummaryInserted) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize SummaryDataMap: "
          "duplicate SummaryName '%s' found at index %zu",
          SummaryIt->first.str().data(), Index);
    }
  }

  return SummaryDataMap;
}

llvm::json::Array JSONFormat::summaryDataMapToJSON(
    const std::map<SummaryName,
                   std::map<EntityId, std::unique_ptr<EntitySummary>>>
        &SummaryDataMap) const {
  llvm::json::Array Result;
  Result.reserve(SummaryDataMap.size());
  for (const auto &[SummaryName, DataMap] : SummaryDataMap) {
    Result.push_back(summaryDataMapEntryToJSON(SummaryName, DataMap));
  }
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

    auto ExpectedSummaryDataMap =
        summaryDataMapFromJSON(*SummaryDataArray, getIdTable(Summary));
    if (!ExpectedSummaryDataMap)
      return wrapError(ExpectedSummaryDataMap.takeError(),
                       "failed to read TUSummary from '%s'",
                       Path.str().c_str());

    getData(Summary) = std::move(*ExpectedSummaryDataMap);
  }

  return Summary;
}

llvm::Error JSONFormat::writeTUSummary(const TUSummary &S,
                                       llvm::StringRef Path) {
  llvm::json::Object RootObject;

  RootObject["tu_namespace"] = buildNamespaceToJSON(getTUNamespace(S));

  RootObject["id_table"] = entityIdTableToJSON(getIdTable(S));

  RootObject["data"] = summaryDataMapToJSON(getData(S));

  // Write the JSON to file
  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return wrapError(std::move(Error), std::errc::io_error,
                     "failed to write TUSummary to '%s'", Path.str().c_str());
  }

  return llvm::Error::success();
}
