//===- SerializationFormat.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract SerializationFormat interface for reading and writing
// TUSummary and LinkUnitResolution data.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_H
#define CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_H

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <map>
#include <vector>

namespace clang::ssaf {

class EntityId;
class EntityIdTable;
class EntityName;
class TUSummary;
class TUSummaryData;
class SummaryName;
class EntitySummary;

/// Abstract base class for serialization formats.
class SerializationFormat {
protected:
  // Helpers providing access to implementation details of basic data structures
  // for efficient serialization/deserialization.

  static size_t getEntityIdIndex(const EntityId &EI);
  static EntityId makeEntityId(const size_t Index);

  static const std::map<EntityName, EntityId> &
  getEntities(const EntityIdTable &EIT);
  static std::map<EntityName, EntityId> &
  getEntitiesForDeserialization(EntityIdTable &EIT);

  static EntityIdTable &getIdTableForDeserialization(TUSummary &S);
  static BuildNamespace &getTUNamespaceForDeserialization(TUSummary &S);
  static std::map<SummaryName,
                  std::map<EntityId, std::unique_ptr<EntitySummary>>> &
  getDataForDeserialization(TUSummary &S);
  static const EntityIdTable &getIdTable(const TUSummary &S);
  static const BuildNamespace &getTUNamespace(const TUSummary &S);
  static const std::map<SummaryName,
                        std::map<EntityId, std::unique_ptr<EntitySummary>>> &
  getData(const TUSummary &S);

  static BuildNamespaceKind getBuildNamespaceKind(const BuildNamespace &BN);
  static llvm::StringRef getBuildNamespaceName(const BuildNamespace &BN);
  static const std::vector<BuildNamespace> &
  getNestedBuildNamespaces(const NestedBuildNamespace &NBN);

  static llvm::StringRef getEntityNameUSR(const EntityName &EN);
  static const llvm::SmallString<16> &getEntityNameSuffix(const EntityName &EN);
  static const NestedBuildNamespace &
  getEntityNameNamespace(const EntityName &EN);

public:
  virtual ~SerializationFormat() = default;

  virtual llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) = 0;

  virtual llvm::Error writeTUSummary(const TUSummary &Summary,
                                     llvm::StringRef OutputDir) = 0;
};

} // namespace clang::ssaf

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_H
