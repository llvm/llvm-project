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
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <vector>

namespace clang::ssaf {

class EntityId;
class EntityIdTable;
class EntityName;
class TUSummary;
class TUSummaryData;

/// Abstract base class for serialization formats.
class SerializationFormat {
protected:
  // Helpers providing access to implementation details of basic data structures
  // for efficient serialization/deserialization.
  static EntityIdTable &getIdTableForDeserialization(TUSummary &S);
  static BuildNamespace &getTUNamespaceForDeserialization(TUSummary &S);
  static const EntityIdTable &getIdTable(const TUSummary &S);
  static const BuildNamespace &getTUNamespace(const TUSummary &S);

  static BuildNamespaceKind getBuildNamespaceKind(const BuildNamespace &BN);
  static llvm::StringRef getBuildNamespaceName(const BuildNamespace &BN);
  static const std::vector<BuildNamespace> &
  getNestedBuildNamespaces(const NestedBuildNamespace &NBN);

  static llvm::StringRef getEntityNameUSR(const EntityName &EN);
  static const llvm::SmallString<16> &getEntityNameSuffix(const EntityName &EN);
  static const NestedBuildNamespace &
  getEntityNameNamespace(const EntityName &EN);

public:
  explicit SerializationFormat(
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);
  virtual ~SerializationFormat() = default;

  virtual TUSummary readTUSummary(llvm::StringRef Path) = 0;

  virtual void writeTUSummary(const TUSummary &Summary,
                              llvm::StringRef OutputDir) = 0;

protected:
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
};

} // namespace clang::ssaf

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_H
