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
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ExtensibleRTTI.h"

namespace clang::ssaf {

class EntityId;
class EntityIdTable;
class EntityName;
class EntitySummary;

/// Abstract base class for serialization formats.
class SerializationFormat
    : public llvm::RTTIExtends<SerializationFormat, llvm::RTTIRoot> {
public:
  virtual ~SerializationFormat() = default;

  virtual TUSummary readTUSummary(llvm::StringRef Path) = 0;

  virtual void writeTUSummary(const TUSummary &Summary,
                              llvm::StringRef OutputDir) = 0;

  static char ID; // For RTTIExtends.

protected:
  // Helpers providing access to implementation details of basic data structures
  // for efficient serialization/deserialization.
#define FIELD(CLASS, FIELD_NAME)                                               \
  static const auto &get##FIELD_NAME(const CLASS &X) { return X.FIELD_NAME; }  \
  static auto &get##FIELD_NAME(CLASS &X) { return X.FIELD_NAME; }
#include "clang/Analysis/Scalable/Model/PrivateFieldNames.def"
};

template <class SerializerFn, class DeserializerFn> struct FormatInfoEntry {
  FormatInfoEntry(SummaryName ForSummary, SerializerFn Serialize,
                  DeserializerFn Deserialize)
      : ForSummary(ForSummary), Serialize(Serialize), Deserialize(Deserialize) {
  }

  SummaryName ForSummary;
  SerializerFn Serialize;
  DeserializerFn Deserialize;
};

} // namespace clang::ssaf

#endif // CLANG_ANALYSIS_SCALABLE_SERIALIZATION_SERIALIZATION_FORMAT_H
