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

  // Accessors for TUSummary:
  template <class T> static auto &IdTableOf(T &X) { return X.IdTable; }
  template <class T> static auto &TUNamespaceOf(T &X) { return X.TUNamespace; }

  // Accessors for BuildNamespace:
  template <class T> static auto &KindOf(T &X) { return X.Kind; }
  template <class T> static auto &NameOf(T &X) { return X.Name; }

  // Accessors for NestedBuildNamespace:
  template <class T> static auto &NamespacesOf(T &X) { return X.Namespaces; }

  // Accessors for EntityName:
  template <class T> static auto &USROf(T &X) { return X.USR; }
  template <class T> static auto &SuffixOf(T &X) { return X.Suffix; }
  template <class T> static auto &NamespaceOf(T &X) { return X.Namespace; }

public:
  virtual ~SerializationFormat() = default;

  virtual TUSummary readTUSummary(llvm::StringRef Path) = 0;

  virtual void writeTUSummary(const TUSummary &Summary,
                              llvm::StringRef OutputDir) = 0;
};

} // namespace clang::ssaf

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_H
