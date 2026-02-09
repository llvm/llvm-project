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
//----------------------------------------------------------------------------
// ErrorBuilder - Fluent API for constructing contextual errors
//----------------------------------------------------------------------------

class ErrorBuilder {
private:
  std::error_code Code;
  std::vector<std::string> ContextStack;
  llvm::Error WrappedError = llvm::Error::success();

public:
  explicit ErrorBuilder(std::errc EC) : Code(std::make_error_code(EC)) {}
  explicit ErrorBuilder(std::error_code EC) : Code(EC) {}

  // Add context message without formatting (for plain strings)
  ErrorBuilder &context(const char *Msg) {
    ContextStack.push_back(Msg);
    return *this;
  }

  // Add context message with formatting
  template <typename... Args>
  ErrorBuilder &context(const char *Fmt, Args &&...ArgVals) {
    ContextStack.push_back(
        llvm::formatv(Fmt, std::forward<Args>(ArgVals)...).str());
    return *this;
  }

  // Wrap an existing error as the cause
  ErrorBuilder &cause(llvm::Error E) {
    // Consume the old WrappedError before assigning (LLVM Error requires
    // checking)
    llvm::consumeError(std::move(WrappedError));
    WrappedError = std::move(E);
    return *this;
  }

  // Build the final error
  llvm::Error build() {
    if (ContextStack.empty() && !WrappedError)
      return llvm::Error::success();

    if (ContextStack.empty())
      return std::move(WrappedError);

    std::string FinalMessage = llvm::join(ContextStack, ": ");
    auto E = llvm::createStringError(Code, "%s", FinalMessage.c_str());

    if (WrappedError)
      return llvm::joinErrors(std::move(E), std::move(WrappedError));

    return E;
  }
};

//----------------------------------------------------------------------------
// Error Message Constants
//----------------------------------------------------------------------------

namespace ErrorMessages {
// File validation errors
constexpr const char *FileNotFound = "file does not exist: '{0}'";
constexpr const char *IsDirectory = "path is a directory, not a file: '{0}'";
constexpr const char *NotJSONFile = "not a JSON file: '{0}'";
constexpr const char *FailedToValidateJSONFile =
    "failed to validate JSON file '{0}'";
constexpr const char *FailedToReadFile = "failed to read file '{0}'";
constexpr const char *FailedToReadJSONObject =
    "failed to read JSON object from file '{0}'";
constexpr const char *FailedToOpenFile = "failed to open '{0}'";
constexpr const char *WriteFailed = "write failed";

// BuildNamespace errors
constexpr const char *InvalidBuildNamespaceKind =
    "invalid 'kind' BuildNamespaceKind value '{0}'";
constexpr const char *MissingBuildNamespaceKind =
    "failed to deserialize BuildNamespace: "
    "missing required field 'kind' (expected BuildNamespaceKind)";
constexpr const char *MissingBuildNamespaceName =
    "failed to deserialize BuildNamespace: "
    "missing required field 'name'";
constexpr const char *FailedToDeserializeBuildNamespace =
    "failed to deserialize BuildNamespace";

// NestedBuildNamespace errors
constexpr const char *NestedBuildNamespaceElementNotObject =
    "failed to deserialize NestedBuildNamespace: "
    "element at index {0} is not a JSON object "
    "(expected BuildNamespace object)";
constexpr const char *FailedToDeserializeNestedBuildNamespace =
    "failed to deserialize NestedBuildNamespace at index {0}";

// EntityName errors
constexpr const char *MissingEntityNameUSR =
    "failed to deserialize EntityName: "
    "missing required field 'usr' (Unified Symbol Resolution string)";
constexpr const char *MissingEntityNameSuffix =
    "failed to deserialize EntityName: "
    "missing required field 'suffix'";
constexpr const char *MissingEntityNameNamespace =
    "failed to deserialize EntityName: "
    "missing or invalid field 'namespace' "
    "(expected JSON array of BuildNamespace objects)";
constexpr const char *FailedToDeserializeEntityName =
    "failed to deserialize EntityName";

// EntityIdTable entry errors
constexpr const char *MissingEntityIdTableEntryName =
    "failed to deserialize EntityIdTable entry: "
    "missing or invalid field 'name' (expected EntityName JSON object)";
constexpr const char *MissingEntityIdTableEntryId =
    "failed to deserialize EntityIdTable entry: "
    "missing required field 'id' (expected unsigned integer EntityId)";
constexpr const char *InvalidEntityIdTableEntryId =
    "failed to deserialize EntityIdTable entry: "
    "field 'id' is not a valid unsigned 64-bit integer "
    "(expected non-negative EntityId value)";
constexpr const char *FailedToDeserializeEntityIdTableEntry =
    "failed to deserialize EntityIdTable entry";

// EntityIdTable errors
constexpr const char *EntityIdTableElementNotObject =
    "failed to deserialize EntityIdTable: "
    "element at index {0} is not a JSON object "
    "(expected EntityIdTable entry with 'id' and 'name' fields)";
constexpr const char *FailedToDeserializeEntityIdTable =
    "failed to deserialize EntityIdTable at index {0}";
constexpr const char *DuplicateEntityName =
    "failed to deserialize EntityIdTable: "
    "duplicate EntityName found at index {0} "
    "(EntityId={1} already exists in table)";

// EntitySummary errors
constexpr const char *NoFormatInfoForSummary =
    "failed to deserialize EntitySummary: "
    "no FormatInfo was registered for summary name: {0}";

// EntityDataMap entry errors
constexpr const char *MissingEntityDataMapEntryEntityId =
    "failed to deserialize EntityDataMap entry: "
    "missing required field 'entity_id' (expected unsigned integer EntityId)";
constexpr const char *InvalidEntityDataMapEntryEntityId =
    "failed to deserialize EntityDataMap entry: "
    "field 'entity_id' is not a valid unsigned 64-bit integer "
    "(expected non-negative EntityId value)";
constexpr const char *MissingEntityDataMapEntryEntitySummary =
    "failed to deserialize EntityDataMap entry: "
    "missing or invalid field 'entity_summary' "
    "(expected EntitySummary JSON object)";
constexpr const char *FailedToDeserializeEntityDataMapEntry =
    "failed to deserialize EntityDataMap entry";

// EntityDataMap errors
constexpr const char *EntityDataMapElementNotObject =
    "failed to deserialize EntityDataMap: "
    "element at index {0} is not a JSON object "
    "(expected EntityDataMap entry with 'entity_id' and 'entity_summary' "
    "fields)";
constexpr const char *FailedToDeserializeEntityDataMap =
    "failed to deserialize EntityDataMap at index {0}";
constexpr const char *DuplicateEntityId =
    "failed to deserialize EntityDataMap: "
    "duplicate EntityId ({0}) found at index {1}";

// SummaryDataMap entry errors
constexpr const char *MissingSummaryDataMapEntrySummaryName =
    "failed to deserialize SummaryDataMap entry: "
    "missing required field 'summary_name' "
    "(expected string identifier for the analysis summary)";
constexpr const char *MissingSummaryDataMapEntrySummaryData =
    "failed to deserialize SummaryDataMap entry: "
    "missing or invalid field 'summary_data' "
    "(expected JSON array of entity data entries)";
constexpr const char *FailedToDeserializeSummaryDataMapEntry =
    "failed to deserialize SummaryDataMap entry for summary '{0}'";

// SummaryDataMap errors
constexpr const char *SummaryDataMapElementNotObject =
    "failed to deserialize SummaryDataMap: "
    "element at index {0} is not a JSON object "
    "(expected SummaryDataMap entry with 'summary_name' and 'summary_data' "
    "fields)";
constexpr const char *FailedToDeserializeSummaryDataMap =
    "failed to deserialize SummaryDataMap at index {0}";
constexpr const char *DuplicateSummaryName =
    "failed to deserialize SummaryDataMap: "
    "duplicate SummaryName '{0}' found at index {1}";

// TUSummary errors
constexpr const char *ReadingTUSummaryFrom = "reading TUSummary from '{0}'";
constexpr const char *MissingTUNamespace =
    "missing or invalid field 'tu_namespace' (expected JSON object)";
constexpr const char *MissingIdTable =
    "missing or invalid field 'id_table' (expected JSON array)";
constexpr const char *MissingData =
    "missing or invalid field 'data' (expected JSON array)";
constexpr const char *WritingTUSummaryTo = "writing TUSummary to '{0}'";
} // namespace ErrorMessages

} // namespace

//----------------------------------------------------------------------------
// JSON Reader and Writer
//----------------------------------------------------------------------------

namespace {

llvm::Error isJSONFile(llvm::StringRef Path) {
  if (!llvm::sys::fs::exists(Path))
    return ErrorBuilder(std::errc::no_such_file_or_directory)
        .context(ErrorMessages::FileNotFound, Path.str().c_str())
        .build();

  if (llvm::sys::fs::is_directory(Path))
    return ErrorBuilder(std::errc::is_a_directory)
        .context(ErrorMessages::IsDirectory, Path.str().c_str())
        .build();

  if (!Path.ends_with_insensitive(".json"))
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::NotJSONFile, Path.str().c_str())
        .build();

  return llvm::Error::success();
}

llvm::Expected<llvm::json::Value> readJSON(llvm::StringRef Path) {
  if (llvm::Error Err = isJSONFile(Path))
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToValidateJSONFile, Path.str().c_str())
        .cause(std::move(Err))
        .build();

  auto BufferOrError = llvm::MemoryBuffer::getFile(Path);
  if (!BufferOrError) {
    return ErrorBuilder(BufferOrError.getError())
        .context(ErrorMessages::FailedToReadFile, Path.str().c_str())
        .build();
  }

  return llvm::json::parse(BufferOrError.get()->getBuffer());
}

llvm::Expected<llvm::json::Object> readJSONObject(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToReadJSONObject, Path.str().c_str())
        .cause(ExpectedJSON.takeError())
        .build();

  llvm::json::Object *Object = ExpectedJSON->getAsObject();
  if (!Object) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToReadJSONObject, Path.str().c_str())
        .build();
  }
  return *Object;
}

llvm::Error writeJSON(llvm::json::Value &&Value, llvm::StringRef Path) {
  std::error_code EC;
  llvm::raw_fd_ostream OutStream(Path, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    return ErrorBuilder(EC)
        .context(ErrorMessages::FailedToOpenFile, Path.str().c_str())
        .build();
  }

  OutStream << llvm::formatv("{0:2}\n", Value);
  OutStream.flush();

  if (OutStream.has_error()) {
    return ErrorBuilder(OutStream.error())
        .context(ErrorMessages::WriteFailed)
        .build();
  }

  return llvm::Error::success();
}

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
// SummaryName
//----------------------------------------------------------------------------

namespace {

SummaryName summaryNameFromJSON(llvm::StringRef SummaryNameStr) {
  return SummaryName(SummaryNameStr.str());
}

llvm::StringRef summaryNameToJSON(const SummaryName &SN) { return SN.str(); }

} // namespace

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
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::InvalidBuildNamespaceKind,
                 BuildNamespaceKindStr.str().c_str())
        .build();
  }

  return *OptBuildNamespaceKind;
}

namespace {

llvm::StringRef buildNamespaceKindToJSON(BuildNamespaceKind BNK) {
  return toString(BNK);
}

} // namespace

//----------------------------------------------------------------------------
// BuildNamespace
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespace> JSONFormat::buildNamespaceFromJSON(
    const llvm::json::Object &BuildNamespaceObject) const {
  auto OptBuildNamespaceKindStr = BuildNamespaceObject.getString("kind");
  if (!OptBuildNamespaceKindStr) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingBuildNamespaceKind)
        .build();
  }

  auto ExpectedKind = buildNamespaceKindFromJSON(*OptBuildNamespaceKindStr);
  if (!ExpectedKind)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToDeserializeBuildNamespace)
        .cause(ExpectedKind.takeError())
        .build();

  auto OptNameStr = BuildNamespaceObject.getString("name");
  if (!OptNameStr) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingBuildNamespaceName)
        .build();
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
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::NestedBuildNamespaceElementNotObject, Index)
          .build();
    }

    auto ExpectedBuildNamespace = buildNamespaceFromJSON(*BuildNamespaceObject);
    if (!ExpectedBuildNamespace)
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::FailedToDeserializeNestedBuildNamespace,
                   Index)
          .cause(ExpectedBuildNamespace.takeError())
          .build();

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
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingEntityNameUSR)
        .build();
  }

  const auto OptSuffix = EntityNameObject.getString("suffix");
  if (!OptSuffix) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingEntityNameSuffix)
        .build();
  }

  const llvm::json::Array *OptNamespaceArray =
      EntityNameObject.getArray("namespace");
  if (!OptNamespaceArray) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingEntityNameNamespace)
        .build();
  }

  auto ExpectedNamespace = nestedBuildNamespaceFromJSON(*OptNamespaceArray);
  if (!ExpectedNamespace)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToDeserializeEntityName)
        .cause(ExpectedNamespace.takeError())
        .build();

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
// EntityIdTableEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<EntityName, EntityId>>
JSONFormat::entityIdTableEntryFromJSON(
    const llvm::json::Object &EntityIdTableEntryObject) const {

  const llvm::json::Object *OptEntityNameObject =
      EntityIdTableEntryObject.getObject("name");
  if (!OptEntityNameObject) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingEntityIdTableEntryName)
        .build();
  }

  auto ExpectedEntityName = entityNameFromJSON(*OptEntityNameObject);
  if (!ExpectedEntityName)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToDeserializeEntityIdTableEntry)
        .cause(ExpectedEntityName.takeError())
        .build();

  const llvm::json::Value *EntityIdIntValue =
      EntityIdTableEntryObject.get("id");
  if (!EntityIdIntValue) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingEntityIdTableEntryId)
        .build();
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::InvalidEntityIdTableEntryId)
        .build();
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  return std::make_pair(std::move(*ExpectedEntityName), std::move(EI));
}

llvm::json::Object JSONFormat::entityIdTableEntryToJSON(const EntityName &EN,
                                                        EntityId EI) const {
  llvm::json::Object Entry;
  Entry["id"] = entityIdToJSON(EI);
  Entry["name"] = entityNameToJSON(EN);
  return Entry;
}

//----------------------------------------------------------------------------
// EntityIdTable
//----------------------------------------------------------------------------

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
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::EntityIdTableElementNotObject, Index)
          .build();
    }

    auto ExpectedEntityIdTableEntry =
        entityIdTableEntryFromJSON(*OptEntityIdTableEntryObject);
    if (!ExpectedEntityIdTableEntry)
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::FailedToDeserializeEntityIdTable, Index)
          .cause(ExpectedEntityIdTableEntry.takeError())
          .build();

    auto [EntityIt, EntityInserted] =
        Entities.emplace(std::move(*ExpectedEntityIdTableEntry));
    if (!EntityInserted) {
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::DuplicateEntityName, Index,
                   getEntityIdIndex(EntityIt->second))
          .build();
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
    EntityIdTableArray.push_back(
        entityIdTableEntryToJSON(EntityName, EntityId));
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
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::NoFormatInfoForSummary, SN.str().data())
        .build();
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
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingEntityDataMapEntryEntityId)
        .build();
  }

  const std::optional<uint64_t> OptEntityIdInt =
      EntityIdIntValue->getAsUINT64();
  if (!OptEntityIdInt) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::InvalidEntityDataMapEntryEntityId)
        .build();
  }

  EntityId EI = entityIdFromJSON(*OptEntityIdInt);

  const llvm::json::Object *OptEntitySummaryObject =
      EntityDataMapEntryObject.getObject("entity_summary");
  if (!OptEntitySummaryObject) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingEntityDataMapEntryEntitySummary)
        .build();
  }

  auto ExpectedEntitySummary =
      entitySummaryFromJSON(SN, *OptEntitySummaryObject, IdTable);
  if (!ExpectedEntitySummary)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToDeserializeEntityDataMapEntry)
        .cause(ExpectedEntitySummary.takeError())
        .build();

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
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::EntityDataMapElementNotObject, Index)
          .build();
    }

    auto ExpectedEntityDataMapEntry =
        entityDataMapEntryFromJSON(*OptEntityDataMapEntryObject, SN, IdTable);
    if (!ExpectedEntityDataMapEntry)
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::FailedToDeserializeEntityDataMap, Index)
          .cause(ExpectedEntityDataMapEntry.takeError())
          .build();

    auto [DataIt, DataInserted] =
        EntityDataMap.insert(std::move(*ExpectedEntityDataMapEntry));
    if (!DataInserted) {
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::DuplicateEntityId,
                   getEntityIdIndex(DataIt->first), Index)
          .build();
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
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingSummaryDataMapEntrySummaryName)
        .build();
  }

  SummaryName SN = summaryNameFromJSON(*OptSummaryNameStr);

  const llvm::json::Array *OptEntityDataArray =
      SummaryDataMapEntryObject.getArray("summary_data");
  if (!OptEntityDataArray) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::MissingSummaryDataMapEntrySummaryData)
        .build();
  }

  auto ExpectedEntityDataMap =
      entityDataMapFromJSON(SN, *OptEntityDataArray, IdTable);
  if (!ExpectedEntityDataMap)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::FailedToDeserializeSummaryDataMapEntry,
                 SN.str().data())
        .cause(ExpectedEntityDataMap.takeError())
        .build();

  return std::make_pair(std::move(SN), std::move(*ExpectedEntityDataMap));
}

llvm::json::Object JSONFormat::summaryDataMapEntryToJSON(
    const SummaryName &SN,
    const std::map<EntityId, std::unique_ptr<EntitySummary>> &SD) const {
  llvm::json::Object Result;
  Result["summary_name"] = summaryNameToJSON(SN);
  Result["summary_data"] = entityDataMapToJSON(SN, SD);
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
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::SummaryDataMapElementNotObject, Index)
          .build();
    }

    auto ExpectedSummaryDataMapEntry =
        summaryDataMapEntryFromJSON(*OptSummaryDataMapEntryObject, IdTable);
    if (!ExpectedSummaryDataMapEntry)
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::FailedToDeserializeSummaryDataMap, Index)
          .cause(ExpectedSummaryDataMapEntry.takeError())
          .build();

    auto [SummaryIt, SummaryInserted] =
        SummaryDataMap.emplace(std::move(*ExpectedSummaryDataMapEntry));
    if (!SummaryInserted) {
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::DuplicateSummaryName,
                   SummaryIt->first.str().data(), Index)
          .build();
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
  auto ExpectedRootObject = readJSONObject(Path);
  if (!ExpectedRootObject)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::ReadingTUSummaryFrom, Path.str().c_str())
        .cause(ExpectedRootObject.takeError())
        .build();

  const llvm::json::Object &RootObject = *ExpectedRootObject;

  // Parse TUNamespace field
  const llvm::json::Object *TUNamespaceObject =
      RootObject.getObject("tu_namespace");
  if (!TUNamespaceObject) {
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::ReadingTUSummaryFrom, Path.str().c_str())
        .context(ErrorMessages::MissingTUNamespace)
        .build();
  }

  auto ExpectedTUNamespace = buildNamespaceFromJSON(*TUNamespaceObject);
  if (!ExpectedTUNamespace)
    return ErrorBuilder(std::errc::invalid_argument)
        .context(ErrorMessages::ReadingTUSummaryFrom, Path.str().c_str())
        .cause(ExpectedTUNamespace.takeError())
        .build();

  TUSummary Summary(std::move(*ExpectedTUNamespace));

  // Parse IdTable field
  {
    const llvm::json::Array *IdTableArray = RootObject.getArray("id_table");
    if (!IdTableArray) {
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::ReadingTUSummaryFrom, Path.str().c_str())
          .context(ErrorMessages::MissingIdTable)
          .build();
    }

    auto ExpectedIdTable = entityIdTableFromJSON(*IdTableArray);
    if (!ExpectedIdTable)
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::ReadingTUSummaryFrom, Path.str().c_str())
          .cause(ExpectedIdTable.takeError())
          .build();

    getIdTable(Summary) = std::move(*ExpectedIdTable);
  }

  // Parse Data field
  {
    const llvm::json::Array *SummaryDataArray = RootObject.getArray("data");
    if (!SummaryDataArray) {
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::ReadingTUSummaryFrom, Path.str().c_str())
          .context(ErrorMessages::MissingData)
          .build();
    }

    auto ExpectedSummaryDataMap =
        summaryDataMapFromJSON(*SummaryDataArray, getIdTable(Summary));
    if (!ExpectedSummaryDataMap)
      return ErrorBuilder(std::errc::invalid_argument)
          .context(ErrorMessages::ReadingTUSummaryFrom, Path.str().c_str())
          .cause(ExpectedSummaryDataMap.takeError())
          .build();

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

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder(std::errc::io_error)
        .context(ErrorMessages::WritingTUSummaryTo, Path.str().c_str())
        .cause(std::move(Error))
        .build();
  }

  return llvm::Error::success();
}
