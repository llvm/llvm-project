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
/// Implement an APISetVisitor to serialize the APISet into the Symbol Graph
/// format for ExtractAPI. See https://github.com/apple/swift-docc-symbolkit.
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

/// Common options to customize the visitor output.
struct SymbolGraphSerializerOption {
  /// Do not include unnecessary whitespaces to save space.
  bool Compact;
};

/// The visitor that organizes API information in the Symbol Graph format.
///
/// The Symbol Graph format (https://github.com/apple/swift-docc-symbolkit)
/// models an API set as a directed graph, where nodes are symbol declarations,
/// and edges are relationships between the connected symbols.
class SymbolGraphSerializer : public APISetVisitor<SymbolGraphSerializer> {
  /// A JSON array of formatted symbols in \c APISet.
  Array Symbols;

  /// A JSON array of formatted symbol relationships in \c APISet.
  Array Relationships;

  /// The Symbol Graph format version used by this serializer.
  static const VersionTuple FormatVersion;

  /// Indicates whether child symbols should be visited. This is mainly
  /// useful for \c serializeSingleSymbolSGF.
  bool ShouldRecurse;

public:
  /// Serialize the APIs in \c APISet in the Symbol Graph format.
  ///
  /// \returns a JSON object that contains the root of the formatted
  /// Symbol Graph.
  Object serialize();

  ///  Wrap serialize(void) and write out the serialized JSON object to \p os.
  void serialize(raw_ostream &os);

  /// Serialize a single symbol SGF. This is primarily used for libclang.
  ///
  /// \returns an optional JSON Object representing the payload that libclang
  /// expects for providing symbol information for a single symbol. If this is
  /// not a known symbol returns \c std::nullopt.
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

protected:
  /// The list of symbols to ignore.
  ///
  /// Note: This should be consulted before emitting a symbol.
  const APIIgnoresList &IgnoresList;

  SymbolGraphSerializerOption Options;

public:
  /// Visit a global function record.
  void visitGlobalFunctionRecord(const GlobalFunctionRecord &Record);

  /// Visit a global variable record.
  void visitGlobalVariableRecord(const GlobalVariableRecord &Record);

  /// Visit an enum record.
  void visitEnumRecord(const EnumRecord &Record);

  /// Visit a struct record.
  void visitStructRecord(const StructRecord &Record);

  /// Visit an Objective-C container record.
  void visitObjCContainerRecord(const ObjCContainerRecord &Record);

  /// Visit a macro definition record.
  void visitMacroDefinitionRecord(const MacroDefinitionRecord &Record);

  /// Visit a typedef record.
  void visitTypedefRecord(const TypedefRecord &Record);

  /// Serialize a single record.
  void serializeSingleRecord(const APIRecord *Record);

  SymbolGraphSerializer(const APISet &API, const APIIgnoresList &IgnoresList,
                        SymbolGraphSerializerOption Options = {},
                        bool ShouldRecurse = true)
      : APISetVisitor(API), ShouldRecurse(ShouldRecurse),
        IgnoresList(IgnoresList), Options(Options) {}
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SYMBOLGRAPHSERIALIZER_H
