//===- JSONFormat.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JSON serialization format implementation.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_H
#define CLANG_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_H

#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"

namespace clang::ssaf {

class EntityIdTable;
class EntitySummary;
class SummaryName;

class JSONFormat final : public SerializationFormat {
  using Array = llvm::json::Array;
  using Object = llvm::json::Object;

public:
  // Helper class to provide limited access to EntityId conversion methods.
  // Only exposes EntityId serialization/deserialization to format handlers.
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

  llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) override;

  llvm::Error writeTUSummary(const TUSummary &Summary,
                             llvm::StringRef Path) override;

  using SerializerFn = llvm::function_ref<Object(const EntitySummary &,
                                                 const EntityIdConverter &)>;
  using DeserializerFn =
      llvm::function_ref<llvm::Expected<std::unique_ptr<EntitySummary>>(
          const Object &, EntityIdTable &, const EntityIdConverter &)>;

  using FormatInfo = FormatInfoEntry<SerializerFn, DeserializerFn>;

private:
  static std::map<SummaryName, FormatInfo> initFormatInfos();
  const std::map<SummaryName, FormatInfo> FormatInfos = initFormatInfos();

  EntityId entityIdFromJSON(const uint64_t EntityIdIndex) const;
  uint64_t entityIdToJSON(EntityId EI) const;

  llvm::Expected<BuildNamespaceKind>
  buildNamespaceKindFromJSON(llvm::StringRef BuildNamespaceKindStr) const;

  llvm::Expected<BuildNamespace>
  buildNamespaceFromJSON(const Object &BuildNamespaceObject) const;
  Object buildNamespaceToJSON(const BuildNamespace &BN) const;

  llvm::Expected<NestedBuildNamespace>
  nestedBuildNamespaceFromJSON(const Array &NestedBuildNamespaceArray) const;
  Array nestedBuildNamespaceToJSON(const NestedBuildNamespace &NBN) const;

  llvm::Expected<EntityName>
  entityNameFromJSON(const Object &EntityNameObject) const;
  Object entityNameToJSON(const EntityName &EN) const;

  llvm::Expected<std::pair<EntityName, EntityId>>
  entityIdTableEntryFromJSON(const Object &EntityIdTableEntryObject) const;
  llvm::Expected<EntityIdTable>
  entityIdTableFromJSON(const Array &EntityIdTableArray) const;
  Object entityIdTableEntryToJSON(const EntityName &EN, EntityId EI) const;
  Array entityIdTableToJSON(const EntityIdTable &IdTable) const;

  llvm::Expected<std::unique_ptr<EntitySummary>>
  entitySummaryFromJSON(const SummaryName &SN,
                        const Object &EntitySummaryObject,
                        EntityIdTable &IdTable) const;
  llvm::Expected<Object> entitySummaryToJSON(const SummaryName &SN,
                                             const EntitySummary &ES) const;

  llvm::Expected<std::pair<EntityId, std::unique_ptr<EntitySummary>>>
  entityDataMapEntryFromJSON(const Object &EntityDataMapEntryObject,
                             const SummaryName &SN,
                             EntityIdTable &IdTable) const;
  llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummary>>>
  entityDataMapFromJSON(const SummaryName &SN, const Array &EntityDataArray,
                        EntityIdTable &IdTable) const;
  llvm::Expected<Array>
  entityDataMapToJSON(const SummaryName &SN,
                      const std::map<EntityId, std::unique_ptr<EntitySummary>>
                          &EntityDataMap) const;

  llvm::Expected<std::pair<SummaryName,
                           std::map<EntityId, std::unique_ptr<EntitySummary>>>>
  summaryDataMapEntryFromJSON(const Object &SummaryDataObject,
                              EntityIdTable &IdTable) const;
  llvm::Expected<Object> summaryDataMapEntryToJSON(
      const SummaryName &SN,
      const std::map<EntityId, std::unique_ptr<EntitySummary>> &SD) const;

  llvm::Expected<
      std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
  summaryDataMapFromJSON(const Array &SummaryDataArray,
                         EntityIdTable &IdTable) const;
  llvm::Expected<Array> summaryDataMapToJSON(
      const std::map<SummaryName,
                     std::map<EntityId, std::unique_ptr<EntitySummary>>>
          &SummaryDataMap) const;
};

} // namespace clang::ssaf

namespace llvm {
extern template class CLANG_TEMPLATE_ABI
    Registry<clang::ssaf::JSONFormat::FormatInfo>;
} // namespace llvm

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_H
