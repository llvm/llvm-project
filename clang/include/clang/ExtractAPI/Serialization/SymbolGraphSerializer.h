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
#include "clang/ExtractAPI/APIIgnoresList.h"
#include "clang/ExtractAPI/Serialization/SerializerBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

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

  /// Indicates whether child symbols should be serialized. This is mainly
  /// useful for \c serializeSingleSymbolSGF.
  bool ShouldRecurse;

public:
  /// Serialize the APIs in \c APISet in the Symbol Graph format.
  ///
  /// \returns a JSON object that contains the root of the formatted
  /// Symbol Graph.
  Object serialize();

  /// Implement the APISerializer::serialize interface. Wrap serialize(void) and
  /// write out the serialized JSON object to \p os.
  void serialize(raw_ostream &os) override;

  /// Serialize a single symbol SGF. This is primarily used for libclang.
  ///
  /// \returns an optional JSON Object representing the payload that libclang
  /// expects for providing symbol information for a single symbol. If this is
  /// not a known symbol returns \c None.
  static std::optional<Object> serializeSingleSymbolSGF(StringRef USR,
                                                        const APISet &API);

  /// The kind of a relationship between two symbols.
  enum RelationshipKind {
    /// The source symbol is a member of the target symbol.
    /// For example enum constants are members of the enum, class/instance
    /// methods are members of the class, etc.
    MemberOf,

    /// The source symbol is inherited from the target symbol.
    InheritsFrom,

    /// The source symbol conforms to the target symbol.
    /// For example Objective-C protocol conformances.
    ConformsTo,
  };

  /// Get the string representation of the relationship kind.
  static StringRef getRelationshipString(RelationshipKind Kind);

private:
  /// Just serialize the currently recorded objects in Symbol Graph format.
  Object serializeCurrentGraph();

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
  /// \returns \c std::nullopt if this \p Record should be skipped, or a JSON
  /// object containing common symbol information of \p Record.
  template <typename RecordTy>
  std::optional<Object> serializeAPIRecord(const RecordTy &Record) const;

  /// Helper method to serialize second-level member records of \p Record and
  /// the member-of relationships.
  template <typename MemberTy>
  void serializeMembers(const APIRecord &Record,
                        const SmallVector<std::unique_ptr<MemberTy>> &Members);

  /// Serialize the \p Kind relationship between \p Source and \p Target.
  ///
  /// Record the relationship between the two symbols in
  /// SymbolGraphSerializer::Relationships.
  void serializeRelationship(RelationshipKind Kind, SymbolReference Source,
                             SymbolReference Target);

  /// Serialize a global function record.
  void serializeGlobalFunctionRecord(const GlobalFunctionRecord &Record);

  /// Serialize a global variable record.
  void serializeGlobalVariableRecord(const GlobalVariableRecord &Record);

  /// Serialize an enum record.
  void serializeEnumRecord(const EnumRecord &Record);

  /// Serialize a struct record.
  void serializeStructRecord(const StructRecord &Record);

  /// Serialize an Objective-C container record.
  void serializeObjCContainerRecord(const ObjCContainerRecord &Record);

  /// Serialize a macro definition record.
  void serializeMacroDefinitionRecord(const MacroDefinitionRecord &Record);

  /// Serialize a typedef record.
  void serializeTypedefRecord(const TypedefRecord &Record);

  void serializeSingleRecord(const APIRecord *Record);

public:
  SymbolGraphSerializer(const APISet &API, const APIIgnoresList &IgnoresList,
                        APISerializerOption Options = {},
                        bool ShouldRecurse = true)
      : APISerializer(API, IgnoresList, Options), ShouldRecurse(ShouldRecurse) {
  }
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SYMBOLGRAPHSERIALIZER_H
