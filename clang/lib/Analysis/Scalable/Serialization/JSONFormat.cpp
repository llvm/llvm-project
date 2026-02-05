#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace clang::ssaf;

namespace {
constexpr size_t kPathBufferSize = 128;
constexpr const char *TUSummaryTUNamespaceFilename = "tu_namespace.json";
constexpr const char *TUSummaryIdTableFilename = "id_table.json";
constexpr const char *TUSummaryDataDirname = "data";

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
                     "failed to read JSON from file '%s'", Path.str().c_str());

  llvm::json::Object *Object = ExpectedJSON->getAsObject();
  if (!Object) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to read JSON object from file '%s'",
                                   Path.str().c_str());
  }
  return std::move(*Object);
}

llvm::Expected<llvm::json::Array> readJSONArray(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON)
    return wrapError(ExpectedJSON.takeError(),
                     "failed to read JSON from file '%s'", Path.str().c_str());

  llvm::json::Array *Array = ExpectedJSON->getAsArray();
  if (!Array) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to read JSON array from file '%s'",
                                   Path.str().c_str());
  }
  return std::move(*Array);
}

llvm::Error writeJSON(llvm::json::Value &&Value, llvm::StringRef Path) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC);
  if (EC) {
    return llvm::createStringError(EC, "failed to open '%s'",
                                   Path.str().c_str());
  }

  OS << llvm::formatv("{0:2}\n", Value);

  OS.flush();
  if (OS.has_error()) {
    return llvm::createStringError(OS.error(), "write failed");
  }

  return llvm::Error::success();
}

//----------------------------------------------------------------------------
// EntityId
//----------------------------------------------------------------------------

EntityId JSONFormat::entityIdFromJSON(const uint64_t EntityIdIndex) {
  return makeEntityId(static_cast<size_t>(EntityIdIndex));
}

uint64_t JSONFormat::entityIdToJSON(EntityId EI) const {
  return static_cast<uint64_t>(getEntityIdIndex(EI));
}

//----------------------------------------------------------------------------
// BuildNamespaceKind
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespaceKind>
buildNamespaceKindFromJSON(llvm::StringRef BuildNamespaceKindStr,
                           llvm::StringRef Path) {
  auto OptBuildNamespaceKind = parseBuildNamespaceKind(BuildNamespaceKindStr);
  if (!OptBuildNamespaceKind) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "invalid 'kind' BuildNamespaceKind value '%s' in file '%s'",
        BuildNamespaceKindStr.str().c_str(), Path.str().c_str());
  }

  return *OptBuildNamespaceKind;
}

llvm::StringRef buildNamespaceKindToJSON(BuildNamespaceKind BNK) {
  return toString(BNK);
}

//----------------------------------------------------------------------------
// BuildNamespace
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespace>
buildNamespaceFromJSON(const llvm::json::Object &BuildNamespaceObject,
                       llvm::StringRef Path) {
  auto OptBuildNamespaceKindStr = BuildNamespaceObject.getString("kind");
  if (!OptBuildNamespaceKindStr) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize BuildNamespace from file '%s': "
        "missing required field 'kind' (expected BuildNamespaceKind)",
        Path.str().c_str());
  }

  auto ExpectedKind =
      buildNamespaceKindFromJSON(*OptBuildNamespaceKindStr, Path);
  if (!ExpectedKind)
    return wrapError(ExpectedKind.takeError(),
                     "failed to deserialize BuildNamespace from file '%s'",
                     Path.str().c_str());

  auto OptNameStr = BuildNamespaceObject.getString("name");
  if (!OptNameStr) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize BuildNamespace from file '%s': "
        "missing required field 'name'",
        Path.str().c_str());
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

llvm::Expected<NestedBuildNamespace>
nestedBuildNamespaceFromJSON(const llvm::json::Array &NestedBuildNamespaceArray,
                             llvm::StringRef Path) {
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
          "failed to deserialize NestedBuildNamespace from file '%s': "
          "element at index %zu is not a JSON object "
          "(expected BuildNamespace object)",
          Path.str().c_str(), Index);
    }

    auto ExpectedBuildNamespace =
        buildNamespaceFromJSON(*BuildNamespaceObject, Path);
    if (!ExpectedBuildNamespace)
      return wrapError(
          ExpectedBuildNamespace.takeError(),
          "failed to deserialize NestedBuildNamespace from file '%s' "
          "at index %zu",
          Path.str().c_str(), Index);

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
entityNameFromJSON(const llvm::json::Object &EntityNameObject,
                   llvm::StringRef Path) {
  const auto OptUSR = EntityNameObject.getString("usr");
  if (!OptUSR) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityName from file '%s': "
        "missing required field 'usr' (Unified Symbol Resolution string)",
        Path.str().c_str());
  }

  const auto OptSuffix = EntityNameObject.getString("suffix");
  if (!OptSuffix) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityName from file '%s': "
        "missing required field 'suffix'",
        Path.str().c_str());
  }

  const llvm::json::Array *OptNamespaceArray =
      EntityNameObject.getArray("namespace");
  if (!OptNamespaceArray) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityName from file '%s': "
        "missing or invalid field 'namespace' "
        "(expected JSON array of BuildNamespace objects)",
        Path.str().c_str());
  }

  auto ExpectedNamespace =
      nestedBuildNamespaceFromJSON(*OptNamespaceArray, Path);
  if (!ExpectedNamespace)
    return wrapError(ExpectedNamespace.takeError(),
                     "failed to deserialize EntityName from file '%s'",
                     Path.str().c_str());

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
    const llvm::json::Object &EntityIdTableEntryObject, llvm::StringRef Path) {

  const llvm::json::Object *OptEntityNameObject =
      EntityIdTableEntryObject.getObject("name");
  if (!OptEntityNameObject) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityIdTable entry from file '%s': "
        "missing or invalid field 'name' (expected EntityName JSON object)",
        Path.str().c_str());
  }

  auto ExpectedEntityName = entityNameFromJSON(*OptEntityNameObject, Path);
  if (!ExpectedEntityName)
    return wrapError(ExpectedEntityName.takeError(),
                     "failed to deserialize EntityIdTable entry from file '%s'",
                     Path.str().c_str());

  const llvm::json::Value *EntityIdIntValue =
      EntityIdTableEntryObject.get("id");
  if (!EntityIdIntValue) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityIdTable entry from file '%s': "
        "missing required field 'id' (expected unsigned integer EntityId)",
        Path.str().c_str());
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize EntityIdTable entry from file '%s': "
        "field 'id' is not a valid unsigned 64-bit integer "
        "(expected non-negative EntityId value)",
        Path.str().c_str());
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  return std::make_pair(std::move(*ExpectedEntityName), std::move(EI));
}

llvm::Expected<EntityIdTable>
JSONFormat::entityIdTableFromJSON(const llvm::json::Array &EntityIdTableArray,
                                  llvm::StringRef Path) {
  const size_t EntityCount = EntityIdTableArray.size();

  EntityIdTable IdTable;
  std::map<EntityName, EntityId> &Entities =
      getEntitiesForDeserialization(IdTable);

  for (size_t Index = 0; Index < EntityCount; ++Index) {
    const llvm::json::Value &EntityIdTableEntryValue =
        EntityIdTableArray[Index];

    const llvm::json::Object *OptEntityIdTableEntryObject =
        EntityIdTableEntryValue.getAsObject();

    if (!OptEntityIdTableEntryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize EntityIdTable from file '%s': "
          "element at index %zu is not a JSON object "
          "(expected EntityIdTable entry with 'id' and 'name' fields)",
          Path.str().c_str(), Index);
    }

    auto ExpectedEntityIdTableEntry =
        entityIdTableEntryFromJSON(*OptEntityIdTableEntryObject, Path);
    if (!ExpectedEntityIdTableEntry)
      return wrapError(
          ExpectedEntityIdTableEntry.takeError(),
          "failed to deserialize EntityIdTable from file '%s' at index %zu",
          Path.str().c_str(), Index);

    auto [EntityIt, EntityInserted] =
        Entities.emplace(std::move(*ExpectedEntityIdTableEntry));
    if (!EntityInserted) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize EntityIdTable from file '%s': "
          "duplicate EntityName found at index %zu "
          "(EntityId=%zu already exists in table)",
          Path.str().c_str(), Index, getEntityIdIndex(EntityIt->second));
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
entitySummaryFromJSON(const llvm::json::Object &EntitySummaryObject,
                      llvm::StringRef Path) {
  return llvm::createStringError(
      std::errc::function_not_supported,
      "EntitySummary deserialization from file '%s' is not yet implemented",
      Path.str().c_str());
}

llvm::json::Object entitySummaryToJSON(const EntitySummary &ES) {
  // TODO
  llvm::json::Object Result;
  return Result;
}

//----------------------------------------------------------------------------
// SummaryData
//----------------------------------------------------------------------------

llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummary>>>
JSONFormat::entityDataMapFromJSON(const llvm::json::Array &EntityDataArray,
                                  llvm::StringRef Path) {
  std::map<EntityId, std::unique_ptr<EntitySummary>> EntityDataMap;

  size_t Index = 0;
  for (const llvm::json::Value &EntityDataEntryValue : EntityDataArray) {
    const llvm::json::Object *OptEntityDataEntryObject =
        EntityDataEntryValue.getAsObject();
    if (!OptEntityDataEntryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map from file '%s': "
          "element at index %zu is not a JSON object "
          "(expected object with 'entity_id' and 'entity_summary' fields)",
          Path.str().c_str(), Index);
    }

    const llvm::json::Value *EntityIdIntValue =
        OptEntityDataEntryObject->get("entity_id");
    if (!EntityIdIntValue) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map entry from file '%s' "
          "at index %zu: missing required field 'entity_id' "
          "(expected unsigned integer EntityId)",
          Path.str().c_str(), Index);
    }

    const std::optional<uint64_t> OptEntityIdInt =
        EntityIdIntValue->getAsUINT64();
    if (!OptEntityIdInt) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map entry from file '%s' "
          "at index %zu: field 'entity_id' is not a valid unsigned 64-bit "
          "integer",
          Path.str().c_str(), Index);
    }

    EntityId EI = entityIdFromJSON(*OptEntityIdInt);

    const llvm::json::Object *OptEntitySummaryObject =
        OptEntityDataEntryObject->getObject("entity_summary");
    if (!OptEntitySummaryObject) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map entry from file '%s' "
          "at index %zu: missing or invalid field 'entity_summary' "
          "(expected EntitySummary JSON object)",
          Path.str().c_str(), Index);
    }

    auto ExpectedEntitySummary =
        entitySummaryFromJSON(*OptEntitySummaryObject, Path);

    if (!ExpectedEntitySummary) {
      return wrapError(
          ExpectedEntitySummary.takeError(),
          "failed to deserialize entity data map entry from file '%s' "
          "at index %zu",
          Path.str().c_str(), Index);
    }

    auto [DataIt, DataInserted] = EntityDataMap.insert(
        {std::move(EI), std::move(*ExpectedEntitySummary)});
    if (!DataInserted) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to deserialize entity data map from file '%s': "
          "duplicate EntityId (%zu) found at index %zu",
          Path.str().c_str(), getEntityIdIndex(DataIt->first), Index);
    }

    ++Index;
  }

  return EntityDataMap;
}

llvm::json::Array JSONFormat::entityDataMapToJSON(
    const std::map<EntityId, std::unique_ptr<EntitySummary>> &EntityDataMap)
    const {
  llvm::json::Array Result;
  Result.reserve(EntityDataMap.size());
  for (const auto &[EntityId, EntitySummary] : EntityDataMap) {
    llvm::json::Object Entry;
    Entry["entity_id"] = entityIdToJSON(EntityId);
    Entry["entity_summary"] = entitySummaryToJSON(*EntitySummary);
    Result.push_back(std::move(Entry));
  }
  return Result;
}

llvm::Expected<
    std::pair<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
JSONFormat::summaryDataMapEntryFromJSON(
    const llvm::json::Object &SummaryDataObject, llvm::StringRef Path) {

  std::optional<llvm::StringRef> OptSummaryNameStr =
      SummaryDataObject.getString("summary_name");

  if (!OptSummaryNameStr) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize summary data from file '%s': "
        "missing required field 'summary_name' "
        "(expected string identifier for the analysis summary)",
        Path.str().c_str());
  }

  SummaryName SN = summaryNameFromJSON(*OptSummaryNameStr);

  const llvm::json::Array *OptEntityDataArray =
      SummaryDataObject.getArray("summary_data");
  if (!OptEntityDataArray) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to deserialize summary data from file '%s' for summary '%s': "
        "missing or invalid field 'summary_data' "
        "(expected JSON array of entity summaries)",
        Path.str().c_str(), SN.str().data());
  }

  auto ExpectedEntityDataMap = entityDataMapFromJSON(*OptEntityDataArray, Path);
  if (!ExpectedEntityDataMap)
    return wrapError(
        ExpectedEntityDataMap.takeError(),
        "failed to deserialize summary data from file '%s' for summary '%s'",
        Path.str().c_str(), SN.str().data());

  return std::make_pair(std::move(SN), std::move(*ExpectedEntityDataMap));
}

llvm::json::Object JSONFormat::summaryDataMapEntryToJSON(
    const SummaryName &SN,
    const std::map<EntityId, std::unique_ptr<EntitySummary>> &SD) const {
  llvm::json::Object Result;
  Result["summary_name"] = summaryNameToJSON(SN);
  Result["summary_data"] = entityDataMapToJSON(SD);
  return Result;
}

//----------------------------------------------------------------------------
// SummaryDataMap
//----------------------------------------------------------------------------

llvm::Expected<
    std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
JSONFormat::readTUSummaryData(llvm::StringRef Path) {
  if (!llvm::sys::fs::exists(Path)) {
    return llvm::createStringError(
        std::errc::no_such_file_or_directory,
        "failed to read TUSummary data: directory does not exist: '%s'",
        Path.str().c_str());
  }

  if (!llvm::sys::fs::is_directory(Path)) {
    return llvm::createStringError(
        std::errc::not_a_directory,
        "failed to read TUSummary data: path is not a directory: '%s'",
        Path.str().c_str());
  }

  std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>
      Data;
  std::error_code EC;

  llvm::sys::fs::directory_iterator Dir(Path, EC);
  if (EC) {
    return llvm::createStringError(
        EC, "failed to read TUSummary data: cannot iterate directory '%s'",
        Path.str().c_str());
  }

  for (llvm::sys::fs::directory_iterator End; Dir != End && !EC;
       Dir.increment(EC)) {
    std::string SummaryPath = Dir->path();

    auto ExpectedObject = readJSONObject(SummaryPath);
    if (!ExpectedObject)
      return wrapError(ExpectedObject.takeError(),
                       "failed to read TUSummary data from file '%s'",
                       SummaryPath.c_str());

    auto ExpectedSummaryDataMap =
        summaryDataMapEntryFromJSON(*ExpectedObject, SummaryPath);
    if (!ExpectedSummaryDataMap)
      return wrapError(ExpectedSummaryDataMap.takeError(),
                       "failed to read TUSummary data from file '%s'",
                       SummaryPath.c_str());

    auto [SummaryIt, SummaryInserted] =
        Data.emplace(std::move(*ExpectedSummaryDataMap));
    if (!SummaryInserted) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to read TUSummary data from directory '%s': "
          "duplicate SummaryName '%s' encountered in file '%s'",
          Path.str().c_str(), SummaryIt->first.str().data(),
          SummaryPath.c_str());
    }
  }

  if (EC) {
    return llvm::createStringError(EC,
                                   "failed to read TUSummary data: "
                                   "error during directory iteration of '%s'",
                                   Path.str().c_str());
  }

  return Data;
}

std::string makeValidFilename(llvm::StringRef Name, size_t Prefix,
                              char Replacement) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << llvm::format("%02d-%s", Prefix, Name.str().c_str());

  for (size_t Index = 0; Index < Result.size(); ++Index) {
    char &Actual = Result[Index];
    if (llvm::isAlnum(Actual) || Actual == '-' || Actual == '_')
      continue;
    Actual = Replacement;
  }

  return Result;
}

llvm::Error JSONFormat::writeTUSummaryData(
    const std::map<SummaryName,
                   std::map<EntityId, std::unique_ptr<EntitySummary>>> &Data,
    llvm::StringRef Path) {
  if (!llvm::sys::fs::exists(Path)) {
    return llvm::createStringError(
        std::errc::no_such_file_or_directory,
        "failed to write TUSummary data: directory does not exist: '%s'",
        Path.str().c_str());
  }

  if (!llvm::sys::fs::is_directory(Path)) {
    return llvm::createStringError(
        std::errc::not_a_directory,
        "failed to write TUSummary data: path is not a directory: '%s'",
        Path.str().c_str());
  }

  size_t Index = 0;
  for (const auto &[SummaryName, DataMap] : Data) {
    llvm::SmallString<kPathBufferSize> SummaryPath(Path);
    llvm::sys::path::append(SummaryPath,
                            makeValidFilename(SummaryName.str(), Index, '_'));
    llvm::sys::path::replace_extension(SummaryPath, ".json");

    llvm::json::Object Result = summaryDataMapEntryToJSON(SummaryName, DataMap);
    if (auto Error = writeJSON(std::move(Result), SummaryPath)) {
      return wrapError(
          std::move(Error), std::errc::io_error,
          "failed to write TUSummary data to directory '%s': cannot write "
          "summary '%s' to file '%s'",
          Path.str().c_str(), SummaryName.str().data(),
          SummaryPath.str().data());
    }

    ++Index;
  }

  return llvm::Error::success();
}

//----------------------------------------------------------------------------
// TUSummary
//----------------------------------------------------------------------------

llvm::Expected<TUSummary> JSONFormat::readTUSummary(llvm::StringRef Path) {

  // Populate TUNamespace field.
  llvm::SmallString<kPathBufferSize> TUNamespacePath(Path);
  llvm::sys::path::append(TUNamespacePath, TUSummaryTUNamespaceFilename);

  auto ExpectedObject = readJSONObject(TUNamespacePath);
  if (!ExpectedObject)
    return wrapError(ExpectedObject.takeError(),
                     "failed to read TUSummary from '%s'", Path.str().c_str());

  auto ExpectedTUNamespace =
      buildNamespaceFromJSON(*ExpectedObject, TUNamespacePath);
  if (!ExpectedTUNamespace)
    return wrapError(ExpectedTUNamespace.takeError(),
                     "failed to read TUSummary from '%s'", Path.str().c_str());

  TUSummary Summary(std::move(*ExpectedTUNamespace));

  // Populate IdTable field.
  {
    llvm::SmallString<kPathBufferSize> IdTablePath(Path);
    llvm::sys::path::append(IdTablePath, TUSummaryIdTableFilename);

    auto ExpectedArray = readJSONArray(IdTablePath);
    if (!ExpectedArray)
      return wrapError(ExpectedArray.takeError(),
                       "failed to read TUSummary from '%s'",
                       Path.str().c_str());

    auto ExpectedIdTable = entityIdTableFromJSON(*ExpectedArray, IdTablePath);
    if (!ExpectedIdTable)
      return wrapError(ExpectedIdTable.takeError(),
                       "failed to read TUSummary from '%s'",
                       Path.str().c_str());

    getIdTableForDeserialization(Summary) = std::move(*ExpectedIdTable);
  }

  // Populate Data field.
  {
    llvm::SmallString<kPathBufferSize> DataPath(Path);
    llvm::sys::path::append(DataPath, TUSummaryDataDirname);

    if (!llvm::sys::fs::exists(DataPath)) {
      return llvm::createStringError(std::errc::no_such_file_or_directory,
                                     "failed to read TUSummary from '%s': "
                                     "data directory does not exist: '%s'",
                                     Path.str().c_str(), DataPath.str().data());
    }

    if (!llvm::sys::fs::is_directory(DataPath)) {
      return llvm::createStringError(std::errc::not_a_directory,
                                     "failed to read TUSummary from '%s': "
                                     "data path is not a directory: '%s'",
                                     Path.str().c_str(), DataPath.str().data());
    }

    auto ExpectedData = readTUSummaryData(DataPath);
    if (!ExpectedData)
      return wrapError(ExpectedData.takeError(),
                       "failed to read TUSummary from '%s'",
                       Path.str().c_str());

    getDataForDeserialization(Summary) = std::move(*ExpectedData);
  }

  return Summary;
}

llvm::Error JSONFormat::writeTUSummary(const TUSummary &S,
                                       llvm::StringRef Dir) {
  // Serialize TUNamespace field.
  {
    llvm::SmallString<kPathBufferSize> TUNamespacePath(Dir);
    llvm::sys::path::append(TUNamespacePath, TUSummaryTUNamespaceFilename);

    llvm::json::Object BuildNamespaceObj =
        buildNamespaceToJSON(getTUNamespace(S));
    if (auto Error = writeJSON(std::move(BuildNamespaceObj), TUNamespacePath)) {
      return wrapError(std::move(Error), std::errc::io_error,
                       "failed to write TUSummary to '%s': "
                       "cannot write TUNamespace file '%s'",
                       Dir.str().c_str(), TUNamespacePath.str().data());
    }
  }

  // Serialize IdTable field.
  {
    llvm::SmallString<kPathBufferSize> IdTablePath(Dir);
    llvm::sys::path::append(IdTablePath, TUSummaryIdTableFilename);

    llvm::json::Array IdTableObj = entityIdTableToJSON(getIdTable(S));
    if (auto Error = writeJSON(std::move(IdTableObj), IdTablePath)) {
      return wrapError(std::move(Error), std::errc::io_error,
                       "failed to write TUSummary to '%s': "
                       "cannot write IdTable file '%s'",
                       Dir.str().c_str(), IdTablePath.str().data());
    }
  }

  // Serialize Data field.
  {
    llvm::SmallString<kPathBufferSize> DataPath(Dir);
    llvm::sys::path::append(DataPath, TUSummaryDataDirname);

    // Create the data directory if it doesn't exist
    if (std::error_code EC = llvm::sys::fs::create_directory(DataPath)) {
      // If error is not "already exists", return error
      if (EC != std::errc::file_exists) {
        return llvm::createStringError(EC,
                                       "failed to write TUSummary to '%s': "
                                       "cannot create data directory '%s'",
                                       Dir.str().c_str(),
                                       DataPath.str().data());
      }
    }

    // Verify it's a directory (could be a file with the same name)
    if (llvm::sys::fs::exists(DataPath) &&
        !llvm::sys::fs::is_directory(DataPath)) {
      return llvm::createStringError(
          std::errc::not_a_directory,
          "failed to write TUSummary to '%s': data path exists but is not a "
          "directory: '%s'",
          Dir.str().c_str(), DataPath.str().data());
    }

    if (auto Error = writeTUSummaryData(getData(S), DataPath)) {
      return wrapError(std::move(Error), std::errc::io_error,
                       "failed to write TUSummary to '%s': cannot write data",
                       Dir.str().c_str());
    }
  }

  return llvm::Error::success();
}
