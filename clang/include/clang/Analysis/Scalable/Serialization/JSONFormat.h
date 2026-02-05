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
#include "llvm/Support/JSON.h"

namespace clang::ssaf {

class EntitySummary;
class SummaryName;

class JSONFormat : public SerializationFormat {
public:
  explicit JSONFormat(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
      : SerializationFormat(FS) {}

  ~JSONFormat() = default;

  llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) override;

  llvm::Error writeTUSummary(const TUSummary &Summary,
                             llvm::StringRef OutputDir) override;

private:
  EntityId entityIdFromJSON(const uint64_t EntityIdIndex);
  uint64_t entityIdToJSON(EntityId EI) const;

  llvm::json::Object buildNamespaceToJSON(const BuildNamespace &BN) const;
  llvm::json::Array
  nestedBuildNamespaceToJSON(const NestedBuildNamespace &NBN) const;
  llvm::json::Object entityNameToJSON(const EntityName &EN) const;
  llvm::Expected<std::pair<EntityName, EntityId>>
  entityIdTableEntryFromJSON(const llvm::json::Object &EntityIdTableEntryObject,
                             llvm::StringRef Path);
  llvm::Expected<EntityIdTable>
  entityIdTableFromJSON(const llvm::json::Array &EntityIdTableArray,
                        llvm::StringRef Path);
  llvm::json::Array entityIdTableToJSON(const EntityIdTable &IdTable) const;

  llvm::Expected<std::map<EntityId, std::unique_ptr<EntitySummary>>>
  entityDataMapFromJSON(const llvm::json::Array &EntityDataArray,
                        llvm::StringRef Path);
  llvm::json::Array entityDataMapToJSON(
      const std::map<EntityId, std::unique_ptr<EntitySummary>> &EntityDataMap)
      const;

  llvm::Expected<std::pair<SummaryName,
                           std::map<EntityId, std::unique_ptr<EntitySummary>>>>
  summaryDataMapEntryFromJSON(const llvm::json::Object &SummaryDataObject,
                              llvm::StringRef Path);
  llvm::json::Object summaryDataMapEntryToJSON(
      const SummaryName &SN,
      const std::map<EntityId, std::unique_ptr<EntitySummary>> &SD) const;

  llvm::Expected<
      std::map<SummaryName, std::map<EntityId, std::unique_ptr<EntitySummary>>>>
  readTUSummaryData(llvm::StringRef Path);
  llvm::Error writeTUSummaryData(
      const std::map<SummaryName,
                     std::map<EntityId, std::unique_ptr<EntitySummary>>> &Data,
      llvm::StringRef Path);
};

} // namespace clang::ssaf

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMAT_H
