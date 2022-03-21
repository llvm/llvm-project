//===- ExtractAPI/Serialization/SymbolGraphSerializer.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the SymbolGraphSerializer class.
///
/// Implement an APISerializer for the Symbol Graph format for ExtractAPI.
/// See https://github.com/apple/swift-docc-symbolkit.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SYMBOLGRAPHSERIALIZER_H
#define LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SYMBOLGRAPHSERIALIZER_H

#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/Serialization/SerializerBase.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace extractapi {

using namespace llvm::json;

/// The serializer that organizes API information in the Symbol Graph format.
///
/// The Symbol Graph format (https://github.com/apple/swift-docc-symbolkit)
/// models an API set as a directed graph, where nodes are symbol declarations,
/// and edges are relationships between the connected symbols.
class SymbolGraphSerializer : public APISerializer {
  virtual void anchor();

  /// A JSON array of formatted symbols in \c APISet.
  Array Symbols;

  /// A JSON array of formatted symbol relationships in \c APISet.
  Array Relationships;

  /// The Symbol Graph format version used by this serializer.
  static const VersionTuple FormatVersion;

public:
  /// Serialize the APIs in \c APISet in the Symbol Graph format.
  ///
  /// \returns a JSON object that contains the root of the formatted
  /// Symbol Graph.
  Object serialize();

  /// Implement the APISerializer::serialize interface. Wrap serialize(void) and
  /// write out the serialized JSON object to \p os.
  void serialize(raw_ostream &os) override;

private:
  /// Synthesize the metadata section of the Symbol Graph format.
  ///
  /// The metadata section describes information about the Symbol Graph itself,
  /// including the format version and the generator information.
  Object serializeMetadata() const;

  /// Synthesize the module section of the Symbol Graph format.
  ///
  /// The module section contains information about the product that is defined
  /// by the given API set.
  /// Note that "module" here is not to be confused with the Clang/C++ module
  /// concept.
  Object serializeModule() const;

  /// Determine if the given \p Record should be skipped during serialization.
  bool shouldSkip(const APIRecord &Record) const;

  /// Format the common API information for \p Record.
  ///
  /// This handles the shared information of all kinds of API records,
  /// for example identifier and source location. The resulting object is then
  /// augmented with kind-specific symbol information by the caller.
  /// This method also checks if the given \p Record should be skipped during
  /// serialization.
  ///
  /// \returns \c None if this \p Record should be skipped, or a JSON object
  /// containing common symbol information of \p Record.
  Optional<Object> serializeAPIRecord(const APIRecord &Record) const;

  /// Serialize a global record.
  void serializeGlobalRecord(const GlobalRecord &Record);

public:
  SymbolGraphSerializer(const APISet &API, APISerializerOption Options = {})
      : APISerializer(API, Options) {}
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SYMBOLGRAPHSERIALIZER_H
