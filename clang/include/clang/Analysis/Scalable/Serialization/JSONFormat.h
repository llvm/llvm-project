//===- JSONFormat.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JSON serialization format implementation
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_H
#define CLANG_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_H

#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/JSON.h"

namespace clang::ssaf {

class EntitySummary;
class EntityIdTable;
class SummaryName;

class JSONFormat : public SerializationFormat {
public:
  // Helper class to provide limited access to EntityId conversion methods
  // Only exposes EntityId serialization/deserialization to format handlers
  class EntityIdConverter {
  public:
    EntityId fromJSON(uint64_t EntityIdIndex) const {
      return Format.entityIdFromJSON(EntityIdIndex);
    }

    uint64_t toJSON(EntityId EI) const { return Format.entityIdToJSON(EI); }

  private:
    friend class JSONFormat;
    EntityIdConverter(const JSONFormat &Format) : Format(Format) {}
    const JSONFormat &Format;
  };

  explicit JSONFormat(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  ~JSONFormat() = default;

  llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) override;

  llvm::Error writeTUSummary(const TUSummary &Summary,
                             llvm::StringRef Path) override;

  using SerializerFn = llvm::function_ref<llvm::json::Object(
      const EntitySummary &, const EntityIdConverter &)>;
  using DeserializerFn =
      llvm::function_ref<llvm::Expected<std::unique_ptr<EntitySummary>>(
          const llvm::json::Object &, EntityIdTable &,
          const EntityIdConverter &)>;

  using FormatInfo = FormatInfoEntry<SerializerFn, DeserializerFn>;

private:
  std::map<SummaryName, FormatInfo> FormatInfos;

  EntityId entityIdFromJSON(const uint64_t EntityIdIndex) const;
  uint64_t entityIdToJSON(EntityId EI) const;

  llvm::Expected<BuildNamespaceKind>
  buildNamespaceKindFromJSON(llvm::StringRef BuildNamespaceKindStr) const;

  llvm::Expected<BuildNamespace>
  buildNamespaceFromJSON(const llvm::json::Object &BuildNamespaceObject) const;
  llvm::json::Object buildNamespaceToJSON(const BuildNamespace &BN) const;

  llvm::Expected<NestedBuildNamespace> nestedBuildNamespaceFromJSON(
      const llvm::json::Array &NestedBuildNamespaceArray) const;
  llvm::json::Array
  nestedBuildNamespaceToJSON(const NestedBuildNamespace &NBN) const;

  llvm::Expected<EntityName>
  entityNameFromJSON(const llvm::json::Object &EntityNameObject) const;
  llvm::json::Object entityNameToJSON(const EntityName &EN) const;

  llvm::Expected<std::pair<EntityName, EntityId>> entityIdTableEntryFromJSON(
      const llvm::json::Object &EntityIdTableEntryObject) const;
  llvm::Expected<EntityIdTable>
  entityIdTableFromJSON(const llvm::json::Array &EntityIdTableArray) const;
  llvm::json::Array entityIdTableToJSON(const EntityIdTable &IdTable) const;

  llvm::Expected<std::unique_ptr<EntitySummary>>
  entitySummaryFromJSON(const SummaryName &SN,
                        const llvm::json::Object &EntitySummaryObject,
                        EntityIdTable &IdTable) const;
  llvm::json::Object entitySummaryToJSON(const SummaryName &SN,
                                         const EntitySummary &ES) const;

  llvm::Expected<std::pair<EntityId, std::unique_ptr<EntitySummary>>>
  entityDataMapEntryFromJSON(const llvm::json::Object &EntityDataMapEntryObject,
                             const SummaryName &SN,
                             EntityIdTable &IdTable) const;
  llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummary>>>
  entityDataMapFromJSON(const SummaryName &SN,
                        const llvm::json::Array &EntityDataArray,
                        EntityIdTable &IdTable) const;
  llvm::json::Array
  entityDataMapToJSON(const SummaryName &SN,
                      const std::map<EntityId, std::unique_ptr<EntitySummary>>
                          &EntityDataMap) const;

  llvm::Expected<std::pair<SummaryName,
                           std::map<EntityId, std::unique_ptr<EntitySummary>>>>
  summaryDataMapEntryFromJSON(const llvm::json::Object &SummaryDataObject,
                              EntityIdTable &IdTable) const;
  llvm::json::Object summaryDataMapEntryToJSON(
      const SummaryName &SN,
      const std::map<EntityId, std::unique_ptr<EntitySummary>> &SD) const;

  llvm::Expected<
      std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
  summaryDataMapFromJSON(const llvm::json::Array &SummaryDataArray,
                         EntityIdTable &IdTable) const;
  llvm::json::Array summaryDataMapToJSON(
      const std::map<SummaryName,
                     std::map<EntityId, std::unique_ptr<EntitySummary>>>
          &SummaryDataMap) const;
};

} // namespace clang::ssaf

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_H
