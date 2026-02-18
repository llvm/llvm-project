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

using Array = llvm::json::Array;
using Object = llvm::json::Object;
using Value = llvm::json::Value;

LLVM_INSTANTIATE_REGISTRY(llvm::Registry<JSONFormat::FormatInfo>)

//----------------------------------------------------------------------------
// ErrorBuilder - Fluent API for constructing contextual errors.
//----------------------------------------------------------------------------

namespace {

class ErrorBuilder {
private:
  std::error_code Code;
  std::vector<std::string> ContextStack;

  // Private constructor - only accessible via static factories.
  explicit ErrorBuilder(std::error_code EC) : Code(EC) {}

  // Helper: Format message and add to context stack.
  template <typename... Args>
  void addFormattedContext(const char *Fmt, Args &&...ArgVals) {
    std::string Message =
        llvm::formatv(Fmt, std::forward<Args>(ArgVals)...).str();
    ContextStack.push_back(std::move(Message));
  }

public:
  // Static factory: Create new error from error code and formatted message.
  template <typename... Args>
  static ErrorBuilder create(std::error_code EC, const char *Fmt,
                             Args &&...ArgVals) {
    ErrorBuilder Builder(EC);
    Builder.addFormattedContext(Fmt, std::forward<Args>(ArgVals)...);
    return Builder;
  }

  // Convenience overload for std::errc.
  template <typename... Args>
  static ErrorBuilder create(std::errc EC, const char *Fmt, Args &&...ArgVals) {
    return create(std::make_error_code(EC), Fmt,
                  std::forward<Args>(ArgVals)...);
  }

  // Static factory: Wrap existing error and optionally add context.
  static ErrorBuilder wrap(llvm::Error E) {
    if (!E) {
      llvm::consumeError(std::move(E));
      // Return builder with generic error code for success case.
      return ErrorBuilder(std::make_error_code(std::errc::invalid_argument));
    }

    std::error_code EC;
    bool FirstError = true;
    ErrorBuilder Builder(std::make_error_code(std::errc::invalid_argument));

    llvm::handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EI) {
      // Capture error code from the first error only.
      if (FirstError) {
        EC = EI.convertToErrorCode();
        Builder.Code = EC;
        FirstError = false;
      }

      // Collect messages from all errors.
      std::string ErrorMsg = EI.message();
      if (!ErrorMsg.empty()) {
        Builder.ContextStack.push_back(std::move(ErrorMsg));
      }
    });

    return Builder;
  }

  // Add context (plain string).
  ErrorBuilder &context(const char *Msg) {
    ContextStack.push_back(Msg);
    return *this;
  }

  // Add context (formatted string).
  template <typename... Args>
  ErrorBuilder &context(const char *Fmt, Args &&...ArgVals) {
    addFormattedContext(Fmt, std::forward<Args>(ArgVals)...);
    return *this;
  }

  // Build the final error.
  llvm::Error build() {
    if (ContextStack.empty())
      return llvm::Error::success();

    // Reverse the context stack so that the most recent context appears first
    // and the wrapped error (if any) appears last.
    return llvm::createStringError(
        llvm::join(llvm::reverse(ContextStack), "\n"), Code);
  }
};

} // namespace

//----------------------------------------------------------------------------
// File Format Constant
//----------------------------------------------------------------------------

namespace {

constexpr const char *JSONFormatFileExtension = ".json";

}

//----------------------------------------------------------------------------
// Error Message Constants
//----------------------------------------------------------------------------

namespace {

namespace ErrorMessages {

constexpr const char *FailedToReadFile = "failed to read file '{0}': {1}";
constexpr const char *FailedToWriteFile = "failed to write file '{0}': {1}";
constexpr const char *FileNotFound = "file does not exist";
constexpr const char *FileIsDirectory = "path is a directory, not a file";
constexpr const char *FileIsNotJSON = "file does not end with '{0}' extension";
constexpr const char *FileExists = "file already exists";
constexpr const char *ParentDirectoryNotFound =
    "parent directory does not exist";

constexpr const char *ReadingFromField = "reading {0} from field '{1}'";
constexpr const char *WritingToField = "writing {0} to field '{1}'";
constexpr const char *ReadingFromIndex = "reading {0} from index '{1}'";
constexpr const char *WritingToIndex = "writing {0} to index '{1}'";
constexpr const char *ReadingFromFile = "reading {0} from file '{1}'";
constexpr const char *WritingToFile = "writing {0} to file '{1}'";

constexpr const char *FailedInsertionOnDuplication =
    "failed to insert {0} at index '{1}': encountered duplicate {2} '{3}'";

constexpr const char *FailedToReadObject =
    "failed to read {0}: expected JSON {1}";
constexpr const char *FailedToReadObjectAtField =
    "failed to read {0} from field '{1}': expected JSON {2}";
constexpr const char *FailedToReadObjectAtIndex =
    "failed to read {0} from index '{1}': expected JSON {2}";

constexpr const char *FailedToDeserializeEntitySummary =
    "failed to deserialize EntitySummary: no FormatInfo registered for summary "
    "'{0}'";
constexpr const char *FailedToSerializeEntitySummary =
    "failed to serialize EntitySummary: no FormatInfo registered for summary "
    "'{0}'";

constexpr const char *InvalidBuildNamespaceKind =
    "invalid 'kind' BuildNamespaceKind value '{0}'";

} // namespace ErrorMessages

} // namespace

//----------------------------------------------------------------------------
// JSON Reader and Writer
//----------------------------------------------------------------------------

namespace {

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

  if (!Path.ends_with_insensitive(JSONFormatFileExtension)) {
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

llvm::Error writeJSON(Value &&Value, llvm::StringRef Path) {
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

  if (!Path.ends_with_insensitive(JSONFormatFileExtension)) {
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

  OutStream << llvm::formatv("{0:2}\n", Value);
  OutStream.flush();

  if (OutStream.has_error()) {
    return ErrorBuilder::create(OutStream.error(),
                                ErrorMessages::FailedToWriteFile, Path,
                                OutStream.error().message())
        .build();
  }

  return llvm::Error::success();
}

} // namespace

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
  return makeEntityId(static_cast<size_t>(EntityIdIndex));
}

uint64_t JSONFormat::entityIdToJSON(EntityId EI) const {
  return static_cast<uint64_t>(getIndex(EI));
}

//----------------------------------------------------------------------------
// BuildNamespaceKind
//----------------------------------------------------------------------------

llvm::Expected<BuildNamespaceKind> JSONFormat::buildNamespaceKindFromJSON(
    llvm::StringRef BuildNamespaceKindStr) const {
  auto OptBuildNamespaceKind = parseBuildNamespaceKind(BuildNamespaceKindStr);
  if (!OptBuildNamespaceKind) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::InvalidBuildNamespaceKind,
                                BuildNamespaceKindStr)
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
  if (!ExpectedKind)
    return ErrorBuilder::wrap(ExpectedKind.takeError())
        .context(ErrorMessages::ReadingFromField, "BuildNamespaceKind", "kind")
        .build();

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
    if (!ExpectedEntityIdTableEntry)
      return ErrorBuilder::wrap(ExpectedEntityIdTableEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "EntityIdTable entry",
                   Index)
          .build();

    auto [EntityIt, EntityInserted] =
        Entities.emplace(std::move(*ExpectedEntityIdTableEntry));
    if (!EntityInserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "EntityIdTable entry", Index, "EntityId",
                                  getIndex(EntityIt->second))
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
// EntitySummary
//----------------------------------------------------------------------------

llvm::Expected<std::unique_ptr<EntitySummary>>
JSONFormat::entitySummaryFromJSON(const SummaryName &SN,
                                  const Object &EntitySummaryObject,
                                  EntityIdTable &IdTable) const {
  auto InfoIt = FormatInfos.find(SN);
  if (InfoIt == FormatInfos.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToDeserializeEntitySummary,
                                SN.str())
        .build();
  }

  const auto &InfoEntry = InfoIt->second;
  assert(InfoEntry.ForSummary == SN);

  EntityIdConverter Converter(*this);
  return InfoEntry.Deserialize(EntitySummaryObject, IdTable, Converter);
}

llvm::Expected<Object>
JSONFormat::entitySummaryToJSON(const SummaryName &SN,
                                const EntitySummary &ES) const {
  auto InfoIt = FormatInfos.find(SN);
  if (InfoIt == FormatInfos.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToSerializeEntitySummary,
                                SN.str())
        .build();
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
                                "EntityId", "entity_id", "integer")
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

  return std::make_pair(std::move(EI), std::move(*ExpectedEntitySummary));
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
    if (!ExpectedEntityDataMapEntry)
      return ErrorBuilder::wrap(ExpectedEntityDataMapEntry.takeError())
          .context(ErrorMessages::ReadingFromIndex, "EntitySummary entry",
                   Index)
          .build();

    auto [DataIt, DataInserted] =
        EntityDataMap.insert(std::move(*ExpectedEntityDataMapEntry));
    if (!DataInserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "EntitySummary entry", Index, "EntityId",
                                  getIndex(DataIt->first))
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

    Object Entry;

    Entry["entity_id"] = entityIdToJSON(EntityId);

    auto ExpectedEntitySummaryObject = entitySummaryToJSON(SN, *EntitySummary);
    if (!ExpectedEntitySummaryObject) {
      return ErrorBuilder::wrap(ExpectedEntitySummaryObject.takeError())
          .context(ErrorMessages::WritingToIndex, "EntitySummary entry", Index)
          .build();
    }

    Entry["entity_summary"] = std::move(*ExpectedEntitySummaryObject);

    Result.push_back(std::move(Entry));
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
  if (!ExpectedEntityDataMap)
    return ErrorBuilder::wrap(ExpectedEntityDataMap.takeError())
        .context(ErrorMessages::ReadingFromField, "EntitySummary entries",
                 "summary_data")
        .build();

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
                                  "SummaryData entry", Index, "SummaryName",
                                  SummaryIt->first.str())
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
// TUSummary
//----------------------------------------------------------------------------

llvm::Expected<TUSummary> JSONFormat::readTUSummary(llvm::StringRef Path) {
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

  TUSummary Summary(std::move(*ExpectedTUNamespace));

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

    getIdTable(Summary) = std::move(*ExpectedIdTable);
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

    auto ExpectedSummaryDataMap =
        summaryDataMapFromJSON(*SummaryDataArray, getIdTable(Summary));
    if (!ExpectedSummaryDataMap) {
      return ErrorBuilder::wrap(ExpectedSummaryDataMap.takeError())
          .context(ErrorMessages::ReadingFromField, "SummaryData entries",
                   "data")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    getData(Summary) = std::move(*ExpectedSummaryDataMap);
  }

  return std::move(Summary);
}

llvm::Error JSONFormat::writeTUSummary(const TUSummary &S,
                                       llvm::StringRef Path) {
  Object RootObject;

  RootObject["tu_namespace"] = buildNamespaceToJSON(getTUNamespace(S));

  RootObject["id_table"] = entityIdTableToJSON(getIdTable(S));

  auto ExpectedDataObject = summaryDataMapToJSON(getData(S));
  if (!ExpectedDataObject) {
    return ErrorBuilder::wrap(ExpectedDataObject.takeError())
        .context(ErrorMessages::WritingToFile, "TUSummary", Path)
        .build();
  }

  RootObject["data"] = std::move(*ExpectedDataObject);

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "TUSummary", Path)
        .build();
  }

  return llvm::Error::success();
}
