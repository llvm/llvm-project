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

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace clang::ssaf {

/// Abstract base class for serialization formats.
class SerializationFormat {
public:
  virtual ~SerializationFormat() = default;

  virtual llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) = 0;

  virtual llvm::Error writeTUSummary(const TUSummary &Summary,
                                     llvm::StringRef Path) = 0;

  virtual llvm::Expected<TUSummaryEncoding>
  readTUSummaryEncoding(llvm::StringRef Path) = 0;

  virtual llvm::Error
  writeTUSummaryEncoding(const TUSummaryEncoding &SummaryEncoding,
                         llvm::StringRef Path) = 0;

  virtual llvm::Expected<LUSummary> readLUSummary(llvm::StringRef Path) = 0;

  virtual llvm::Error writeLUSummary(const LUSummary &Summary,
                                     llvm::StringRef Path) = 0;

  virtual llvm::Expected<LUSummaryEncoding>
  readLUSummaryEncoding(llvm::StringRef Path) = 0;

  virtual llvm::Error
  writeLUSummaryEncoding(const LUSummaryEncoding &SummaryEncoding,
                         llvm::StringRef Path) = 0;

  /// Invokes \p Callback once for each analysis that has registered
  /// serialization support for this format.
  virtual void forEachRegisteredAnalysis(
      llvm::function_ref<void(llvm::StringRef Name, llvm::StringRef Desc)>
          Callback) const = 0;

protected:
  // Helpers providing access to implementation details of basic data structures
  // for efficient serialization/deserialization.

  static EntityId makeEntityId(const size_t Index) { return EntityId(Index); }

#define FIELD(CLASS, FIELD_NAME)                                               \
  static const auto &get##FIELD_NAME(const CLASS &X) { return X.FIELD_NAME; }  \
  static auto &get##FIELD_NAME(CLASS &X) { return X.FIELD_NAME; }
#include "clang/ScalableStaticAnalysisFramework/Core/Model/PrivateFieldNames.def"
};

template <class SerializerFn, class DeserializerFn> struct FormatInfoEntry {
  FormatInfoEntry(SummaryName ForSummary, SerializerFn Serialize,
                  DeserializerFn Deserialize)
      : ForSummary(ForSummary), Serialize(Serialize), Deserialize(Deserialize) {
  }
  virtual ~FormatInfoEntry() = default;

  SummaryName ForSummary;
  SerializerFn Serialize;
  DeserializerFn Deserialize;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SERIALIZATION_SERIALIZATIONFORMAT_H
