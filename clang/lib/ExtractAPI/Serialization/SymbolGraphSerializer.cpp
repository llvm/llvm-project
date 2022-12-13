//===- ExtractAPI/Serialization/SymbolGraphSerializer.cpp -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the SymbolGraphSerializer.
///
//===----------------------------------------------------------------------===//

#include "clang/ExtractAPI/Serialization/SymbolGraphSerializer.h"
#include "clang/Basic/Version.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/APIIgnoresList.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VersionTuple.h"
#include <type_traits>

using namespace clang;
using namespace clang::extractapi;
using namespace llvm;
using namespace llvm::json;

namespace {

/// Helper function to inject a JSON object \p Obj into another object \p Paren
/// at position \p Key.
void serializeObject(Object &Paren, StringRef Key, Optional<Object> Obj) {
  if (Obj)
    Paren[Key] = std::move(Obj.value());
}

/// Helper function to inject a JSON array \p Array into object \p Paren at
/// position \p Key.
void serializeArray(Object &Paren, StringRef Key, Optional<Array> Array) {
  if (Array)
    Paren[Key] = std::move(Array.value());
}

/// Serialize a \c VersionTuple \p V with the Symbol Graph semantic version
/// format.
///
/// A semantic version object contains three numeric fields, representing the
/// \c major, \c minor, and \c patch parts of the version tuple.
/// For example version tuple 1.0.3 is serialized as:
/// \code
///   {
///     "major" : 1,
///     "minor" : 0,
///     "patch" : 3
///   }
/// \endcode
///
/// \returns \c std::nullopt if the version \p V is empty, or an \c Object
/// containing the semantic version representation of \p V.
Optional<Object> serializeSemanticVersion(const VersionTuple &V) {
  if (V.empty())
    return std::nullopt;

  Object Version;
  Version["major"] = V.getMajor();
  Version["minor"] = V.getMinor().value_or(0);
  Version["patch"] = V.getSubminor().value_or(0);
  return Version;
}

/// Serialize the OS information in the Symbol Graph platform property.
///
/// The OS information in Symbol Graph contains the \c name of the OS, and an
/// optional \c minimumVersion semantic version field.
Object serializeOperatingSystem(const Triple &T) {
  Object OS;
  OS["name"] = T.getOSTypeName(T.getOS());
  serializeObject(OS, "minimumVersion",
                  serializeSemanticVersion(T.getMinimumSupportedOSVersion()));
  return OS;
}

/// Serialize the platform information in the Symbol Graph module section.
///
/// The platform object describes a target platform triple in corresponding
/// three fields: \c architecture, \c vendor, and \c operatingSystem.
Object serializePlatform(const Triple &T) {
  Object Platform;
  Platform["architecture"] = T.getArchName();
  Platform["vendor"] = T.getVendorName();
  Platform["operatingSystem"] = serializeOperatingSystem(T);
  return Platform;
}

/// Serialize a source position.
Object serializeSourcePosition(const PresumedLoc &Loc) {
  assert(Loc.isValid() && "invalid source position");

  Object SourcePosition;
  SourcePosition["line"] = Loc.getLine();
  SourcePosition["character"] = Loc.getColumn();

  return SourcePosition;
}

/// Serialize a source location in file.
///
/// \param Loc The presumed location to serialize.
/// \param IncludeFileURI If true, include the file path of \p Loc as a URI.
/// Defaults to false.
Object serializeSourceLocation(const PresumedLoc &Loc,
                               bool IncludeFileURI = false) {
  Object SourceLocation;
  serializeObject(SourceLocation, "position", serializeSourcePosition(Loc));

  if (IncludeFileURI) {
    std::string FileURI = "file://";
    // Normalize file path to use forward slashes for the URI.
    FileURI += sys::path::convert_to_slash(Loc.getFilename());
    SourceLocation["uri"] = FileURI;
  }

  return SourceLocation;
}

/// Serialize a source range with begin and end locations.
Object serializeSourceRange(const PresumedLoc &BeginLoc,
                            const PresumedLoc &EndLoc) {
  Object SourceRange;
  serializeObject(SourceRange, "start", serializeSourcePosition(BeginLoc));
  serializeObject(SourceRange, "end", serializeSourcePosition(EndLoc));
  return SourceRange;
}

/// Serialize the availability attributes of a symbol.
///
/// Availability information contains the introduced, deprecated, and obsoleted
/// versions of the symbol for a given domain (roughly corresponds to a
/// platform) as semantic versions, if not default.  Availability information
/// also contains flags to indicate if the symbol is unconditionally unavailable
/// or deprecated, i.e. \c __attribute__((unavailable)) and \c
/// __attribute__((deprecated)).
///
/// \returns \c std::nullopt if the symbol has default availability attributes,
/// or an \c Array containing the formatted availability information.
Optional<Array> serializeAvailability(const AvailabilitySet &Availabilities) {
  if (Availabilities.isDefault())
    return std::nullopt;

  Array AvailabilityArray;

  if (Availabilities.isUnconditionallyDeprecated()) {
    Object UnconditionallyDeprecated;
    UnconditionallyDeprecated["domain"] = "*";
    UnconditionallyDeprecated["isUnconditionallyDeprecated"] = true;
    AvailabilityArray.emplace_back(std::move(UnconditionallyDeprecated));
  }

  // Note unconditionally unavailable records are skipped.

  for (const auto &AvailInfo : Availabilities) {
    Object Availability;
    Availability["domain"] = AvailInfo.Domain;
    serializeObject(Availability, "introducedVersion",
                    serializeSemanticVersion(AvailInfo.Introduced));
    serializeObject(Availability, "deprecatedVersion",
                    serializeSemanticVersion(AvailInfo.Deprecated));
    serializeObject(Availability, "obsoletedVersion",
                    serializeSemanticVersion(AvailInfo.Obsoleted));
    AvailabilityArray.emplace_back(std::move(Availability));
  }

  return AvailabilityArray;
}

/// Get the language name string for interface language references.
StringRef getLanguageName(Language Lang) {
  switch (Lang) {
  case Language::C:
    return "c";
  case Language::ObjC:
    return "objective-c";

  // Unsupported language currently
  case Language::CXX:
  case Language::ObjCXX:
  case Language::OpenCL:
  case Language::OpenCLCXX:
  case Language::CUDA:
  case Language::RenderScript:
  case Language::HIP:
  case Language::HLSL:

  // Languages that the frontend cannot parse and compile
  case Language::Unknown:
  case Language::Asm:
  case Language::LLVM_IR:
    llvm_unreachable("Unsupported language kind");
  }

  llvm_unreachable("Unhandled language kind");
}

/// Serialize the identifier object as specified by the Symbol Graph format.
///
/// The identifier property of a symbol contains the USR for precise and unique
/// references, and the interface language name.
Object serializeIdentifier(const APIRecord &Record, Language Lang) {
  Object Identifier;
  Identifier["precise"] = Record.USR;
  Identifier["interfaceLanguage"] = getLanguageName(Lang);

  return Identifier;
}

/// Serialize the documentation comments attached to a symbol, as specified by
/// the Symbol Graph format.
///
/// The Symbol Graph \c docComment object contains an array of lines. Each line
/// represents one line of striped documentation comment, with source range
/// information.
/// e.g.
/// \code
///   /// This is a documentation comment
///       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'  First line.
///   ///     with multiple lines.
///       ^~~~~~~~~~~~~~~~~~~~~~~'         Second line.
/// \endcode
///
/// \returns \c std::nullopt if \p Comment is empty, or an \c Object containing
/// the formatted lines.
Optional<Object> serializeDocComment(const DocComment &Comment) {
  if (Comment.empty())
    return std::nullopt;

  Object DocComment;
  Array LinesArray;
  for (const auto &CommentLine : Comment) {
    Object Line;
    Line["text"] = CommentLine.Text;
    serializeObject(Line, "range",
                    serializeSourceRange(CommentLine.Begin, CommentLine.End));
    LinesArray.emplace_back(std::move(Line));
  }
  serializeArray(DocComment, "lines", LinesArray);

  return DocComment;
}

/// Serialize the declaration fragments of a symbol.
///
/// The Symbol Graph declaration fragments is an array of tagged important
/// parts of a symbol's declaration. The fragments sequence can be joined to
/// form spans of declaration text, with attached information useful for
/// purposes like syntax-highlighting etc. For example:
/// \code
///   const int pi; -> "declarationFragments" : [
///                      {
///                        "kind" : "keyword",
///                        "spelling" : "const"
///                      },
///                      {
///                        "kind" : "text",
///                        "spelling" : " "
///                      },
///                      {
///                        "kind" : "typeIdentifier",
///                        "preciseIdentifier" : "c:I",
///                        "spelling" : "int"
///                      },
///                      {
///                        "kind" : "text",
///                        "spelling" : " "
///                      },
///                      {
///                        "kind" : "identifier",
///                        "spelling" : "pi"
///                      }
///                    ]
/// \endcode
///
/// \returns \c std::nullopt if \p DF is empty, or an \c Array containing the
/// formatted declaration fragments array.
Optional<Array> serializeDeclarationFragments(const DeclarationFragments &DF) {
  if (DF.getFragments().empty())
    return std::nullopt;

  Array Fragments;
  for (const auto &F : DF.getFragments()) {
    Object Fragment;
    Fragment["spelling"] = F.Spelling;
    Fragment["kind"] = DeclarationFragments::getFragmentKindString(F.Kind);
    if (!F.PreciseIdentifier.empty())
      Fragment["preciseIdentifier"] = F.PreciseIdentifier;
    Fragments.emplace_back(std::move(Fragment));
  }

  return Fragments;
}

/// Serialize the \c names field of a symbol as specified by the Symbol Graph
/// format.
///
/// The Symbol Graph names field contains multiple representations of a symbol
/// that can be used for different applications:
///   - \c title : The simple declared name of the symbol;
///   - \c subHeading : An array of declaration fragments that provides tags,
///     and potentially more tokens (for example the \c +/- symbol for
///     Objective-C methods). Can be used as sub-headings for documentation.
Object serializeNames(const APIRecord &Record) {
  Object Names;
  Names["title"] = Record.Name;
  serializeArray(Names, "subHeading",
                 serializeDeclarationFragments(Record.SubHeading));
  DeclarationFragments NavigatorFragments;
  NavigatorFragments.append(Record.Name,
                            DeclarationFragments::FragmentKind::Identifier,
                            /*PreciseIdentifier*/ "");
  serializeArray(Names, "navigator",
                 serializeDeclarationFragments(NavigatorFragments));

  return Names;
}

/// Serialize the symbol kind information.
///
/// The Symbol Graph symbol kind property contains a shorthand \c identifier
/// which is prefixed by the source language name, useful for tooling to parse
/// the kind, and a \c displayName for rendering human-readable names.
Object serializeSymbolKind(const APIRecord &Record, Language Lang) {
  auto AddLangPrefix = [&Lang](StringRef S) -> std::string {
    return (getLanguageName(Lang) + "." + S).str();
  };

  Object Kind;
  switch (Record.getKind()) {
  case APIRecord::RK_GlobalFunction:
    Kind["identifier"] = AddLangPrefix("func");
    Kind["displayName"] = "Function";
    break;
  case APIRecord::RK_GlobalVariable:
    Kind["identifier"] = AddLangPrefix("var");
    Kind["displayName"] = "Global Variable";
    break;
  case APIRecord::RK_EnumConstant:
    Kind["identifier"] = AddLangPrefix("enum.case");
    Kind["displayName"] = "Enumeration Case";
    break;
  case APIRecord::RK_Enum:
    Kind["identifier"] = AddLangPrefix("enum");
    Kind["displayName"] = "Enumeration";
    break;
  case APIRecord::RK_StructField:
    Kind["identifier"] = AddLangPrefix("property");
    Kind["displayName"] = "Instance Property";
    break;
  case APIRecord::RK_Struct:
    Kind["identifier"] = AddLangPrefix("struct");
    Kind["displayName"] = "Structure";
    break;
  case APIRecord::RK_ObjCIvar:
    Kind["identifier"] = AddLangPrefix("ivar");
    Kind["displayName"] = "Instance Variable";
    break;
  case APIRecord::RK_ObjCMethod:
    if (cast<ObjCMethodRecord>(&Record)->IsInstanceMethod) {
      Kind["identifier"] = AddLangPrefix("method");
      Kind["displayName"] = "Instance Method";
    } else {
      Kind["identifier"] = AddLangPrefix("type.method");
      Kind["displayName"] = "Type Method";
    }
    break;
  case APIRecord::RK_ObjCProperty:
    if (cast<ObjCPropertyRecord>(&Record)->isClassProperty()) {
      Kind["identifier"] = AddLangPrefix("type.property");
      Kind["displayName"] = "Type Property";
    } else {
      Kind["identifier"] = AddLangPrefix("property");
      Kind["displayName"] = "Instance Property";
    }
    break;
  case APIRecord::RK_ObjCInterface:
    Kind["identifier"] = AddLangPrefix("class");
    Kind["displayName"] = "Class";
    break;
  case APIRecord::RK_ObjCCategory:
    // We don't serialize out standalone Objective-C category symbols yet.
    llvm_unreachable("Serializing standalone Objective-C category symbols is "
                     "not supported.");
    break;
  case APIRecord::RK_ObjCProtocol:
    Kind["identifier"] = AddLangPrefix("protocol");
    Kind["displayName"] = "Protocol";
    break;
  case APIRecord::RK_MacroDefinition:
    Kind["identifier"] = AddLangPrefix("macro");
    Kind["displayName"] = "Macro";
    break;
  case APIRecord::RK_Typedef:
    Kind["identifier"] = AddLangPrefix("typealias");
    Kind["displayName"] = "Type Alias";
    break;
  }

  return Kind;
}

template <typename RecordTy>
Optional<Object> serializeFunctionSignatureMixinImpl(const RecordTy &Record,
                                                     std::true_type) {
  const auto &FS = Record.Signature;
  if (FS.empty())
    return std::nullopt;

  Object Signature;
  serializeArray(Signature, "returns",
                 serializeDeclarationFragments(FS.getReturnType()));

  Array Parameters;
  for (const auto &P : FS.getParameters()) {
    Object Parameter;
    Parameter["name"] = P.Name;
    serializeArray(Parameter, "declarationFragments",
                   serializeDeclarationFragments(P.Fragments));
    Parameters.emplace_back(std::move(Parameter));
  }

  if (!Parameters.empty())
    Signature["parameters"] = std::move(Parameters);

  return Signature;
}

template <typename RecordTy>
Optional<Object> serializeFunctionSignatureMixinImpl(const RecordTy &Record,
                                                     std::false_type) {
  return std::nullopt;
}

/// Serialize the function signature field, as specified by the
/// Symbol Graph format.
///
/// The Symbol Graph function signature property contains two arrays.
///   - The \c returns array is the declaration fragments of the return type;
///   - The \c parameters array contains names and declaration fragments of the
///     parameters.
///
/// \returns \c std::nullopt if \p FS is empty, or an \c Object containing the
/// formatted function signature.
template <typename RecordTy>
void serializeFunctionSignatureMixin(Object &Paren, const RecordTy &Record) {
  serializeObject(Paren, "functionSignature",
                  serializeFunctionSignatureMixinImpl(
                      Record, has_function_signature<RecordTy>()));
}

} // namespace

void SymbolGraphSerializer::anchor() {}

/// Defines the format version emitted by SymbolGraphSerializer.
const VersionTuple SymbolGraphSerializer::FormatVersion{0, 5, 3};

Object SymbolGraphSerializer::serializeMetadata() const {
  Object Metadata;
  serializeObject(Metadata, "formatVersion",
                  serializeSemanticVersion(FormatVersion));
  Metadata["generator"] = clang::getClangFullVersion();
  return Metadata;
}

Object SymbolGraphSerializer::serializeModule() const {
  Object Module;
  // The user is expected to always pass `--product-name=` on the command line
  // to populate this field.
  Module["name"] = ProductName;
  serializeObject(Module, "platform", serializePlatform(API.getTarget()));
  return Module;
}

bool SymbolGraphSerializer::shouldSkip(const APIRecord &Record) const {
  // Skip explicitly ignored symbols.
  if (IgnoresList.shouldIgnore(Record.Name))
    return true;

  // Skip unconditionally unavailable symbols
  if (Record.Availabilities.isUnconditionallyUnavailable())
    return true;

  // Filter out symbols prefixed with an underscored as they are understood to
  // be symbols clients should not use.
  if (Record.Name.startswith("_"))
    return true;

  return false;
}

template <typename RecordTy>
Optional<Object>
SymbolGraphSerializer::serializeAPIRecord(const RecordTy &Record) const {
  if (shouldSkip(Record))
    return std::nullopt;

  Object Obj;
  serializeObject(Obj, "identifier",
                  serializeIdentifier(Record, API.getLanguage()));
  serializeObject(Obj, "kind", serializeSymbolKind(Record, API.getLanguage()));
  serializeObject(Obj, "names", serializeNames(Record));
  serializeObject(
      Obj, "location",
      serializeSourceLocation(Record.Location, /*IncludeFileURI=*/true));
  serializeArray(Obj, "availability",
                 serializeAvailability(Record.Availabilities));
  serializeObject(Obj, "docComment", serializeDocComment(Record.Comment));
  serializeArray(Obj, "declarationFragments",
                 serializeDeclarationFragments(Record.Declaration));
  // TODO: Once we keep track of symbol access information serialize it
  // correctly here.
  Obj["accessLevel"] = "public";
  serializeArray(Obj, "pathComponents", Array(PathComponents));

  serializeFunctionSignatureMixin(Obj, Record);

  return Obj;
}

template <typename MemberTy>
void SymbolGraphSerializer::serializeMembers(
    const APIRecord &Record,
    const SmallVector<std::unique_ptr<MemberTy>> &Members) {
  for (const auto &Member : Members) {
    auto MemberPathComponentGuard = makePathComponentGuard(Member->Name);
    auto MemberRecord = serializeAPIRecord(*Member);
    if (!MemberRecord)
      continue;

    Symbols.emplace_back(std::move(*MemberRecord));
    serializeRelationship(RelationshipKind::MemberOf, *Member, Record);
  }
}

StringRef SymbolGraphSerializer::getRelationshipString(RelationshipKind Kind) {
  switch (Kind) {
  case RelationshipKind::MemberOf:
    return "memberOf";
  case RelationshipKind::InheritsFrom:
    return "inheritsFrom";
  case RelationshipKind::ConformsTo:
    return "conformsTo";
  }
  llvm_unreachable("Unhandled relationship kind");
}

void SymbolGraphSerializer::serializeRelationship(RelationshipKind Kind,
                                                  SymbolReference Source,
                                                  SymbolReference Target) {
  Object Relationship;
  Relationship["source"] = Source.USR;
  Relationship["target"] = Target.USR;
  Relationship["targetFallback"] = Target.Name;
  Relationship["kind"] = getRelationshipString(Kind);

  Relationships.emplace_back(std::move(Relationship));
}

void SymbolGraphSerializer::serializeGlobalFunctionRecord(
    const GlobalFunctionRecord &Record) {
  auto GlobalPathComponentGuard = makePathComponentGuard(Record.Name);

  auto Obj = serializeAPIRecord(Record);
  if (!Obj)
    return;

  Symbols.emplace_back(std::move(*Obj));
}

void SymbolGraphSerializer::serializeGlobalVariableRecord(
    const GlobalVariableRecord &Record) {
  auto GlobalPathComponentGuard = makePathComponentGuard(Record.Name);

  auto Obj = serializeAPIRecord(Record);
  if (!Obj)
    return;

  Symbols.emplace_back(std::move(*Obj));
}

void SymbolGraphSerializer::serializeEnumRecord(const EnumRecord &Record) {
  auto EnumPathComponentGuard = makePathComponentGuard(Record.Name);
  auto Enum = serializeAPIRecord(Record);
  if (!Enum)
    return;

  Symbols.emplace_back(std::move(*Enum));
  serializeMembers(Record, Record.Constants);
}

void SymbolGraphSerializer::serializeStructRecord(const StructRecord &Record) {
  auto StructPathComponentGuard = makePathComponentGuard(Record.Name);
  auto Struct = serializeAPIRecord(Record);
  if (!Struct)
    return;

  Symbols.emplace_back(std::move(*Struct));
  serializeMembers(Record, Record.Fields);
}

void SymbolGraphSerializer::serializeObjCContainerRecord(
    const ObjCContainerRecord &Record) {
  auto ObjCContainerPathComponentGuard = makePathComponentGuard(Record.Name);
  auto ObjCContainer = serializeAPIRecord(Record);
  if (!ObjCContainer)
    return;

  Symbols.emplace_back(std::move(*ObjCContainer));

  serializeMembers(Record, Record.Ivars);
  serializeMembers(Record, Record.Methods);
  serializeMembers(Record, Record.Properties);

  for (const auto &Protocol : Record.Protocols)
    // Record that Record conforms to Protocol.
    serializeRelationship(RelationshipKind::ConformsTo, Record, Protocol);

  if (auto *ObjCInterface = dyn_cast<ObjCInterfaceRecord>(&Record)) {
    if (!ObjCInterface->SuperClass.empty())
      // If Record is an Objective-C interface record and it has a super class,
      // record that Record is inherited from SuperClass.
      serializeRelationship(RelationshipKind::InheritsFrom, Record,
                            ObjCInterface->SuperClass);

    // Members of categories extending an interface are serialized as members of
    // the interface.
    for (const auto *Category : ObjCInterface->Categories) {
      serializeMembers(Record, Category->Ivars);
      serializeMembers(Record, Category->Methods);
      serializeMembers(Record, Category->Properties);

      // Surface the protocols of the category to the interface.
      for (const auto &Protocol : Category->Protocols)
        serializeRelationship(RelationshipKind::ConformsTo, Record, Protocol);
    }
  }
}

void SymbolGraphSerializer::serializeMacroDefinitionRecord(
    const MacroDefinitionRecord &Record) {
  auto MacroPathComponentGuard = makePathComponentGuard(Record.Name);
  auto Macro = serializeAPIRecord(Record);

  if (!Macro)
    return;

  Symbols.emplace_back(std::move(*Macro));
}

void SymbolGraphSerializer::serializeTypedefRecord(
    const TypedefRecord &Record) {
  // Typedefs of anonymous types have their entries unified with the underlying
  // type.
  bool ShouldDrop = Record.UnderlyingType.Name.empty();
  // enums declared with `NS_OPTION` have a named enum and a named typedef, with
  // the same name
  ShouldDrop |= (Record.UnderlyingType.Name == Record.Name);
  if (ShouldDrop)
    return;

  auto TypedefPathComponentGuard = makePathComponentGuard(Record.Name);
  auto Typedef = serializeAPIRecord(Record);
  if (!Typedef)
    return;

  (*Typedef)["type"] = Record.UnderlyingType.USR;

  Symbols.emplace_back(std::move(*Typedef));
}

SymbolGraphSerializer::PathComponentGuard
SymbolGraphSerializer::makePathComponentGuard(StringRef Component) {
  return PathComponentGuard(PathComponents, Component);
}

Object SymbolGraphSerializer::serialize() {
  Object Root;
  serializeObject(Root, "metadata", serializeMetadata());
  serializeObject(Root, "module", serializeModule());

  // Serialize global variables in the API set.
  for (const auto &GlobalVar : API.getGlobalVariables())
    serializeGlobalVariableRecord(*GlobalVar.second);

  for (const auto &GlobalFunction : API.getGlobalFunctions())
    serializeGlobalFunctionRecord(*GlobalFunction.second);

  // Serialize enum records in the API set.
  for (const auto &Enum : API.getEnums())
    serializeEnumRecord(*Enum.second);

  // Serialize struct records in the API set.
  for (const auto &Struct : API.getStructs())
    serializeStructRecord(*Struct.second);

  // Serialize Objective-C interface records in the API set.
  for (const auto &ObjCInterface : API.getObjCInterfaces())
    serializeObjCContainerRecord(*ObjCInterface.second);

  // Serialize Objective-C protocol records in the API set.
  for (const auto &ObjCProtocol : API.getObjCProtocols())
    serializeObjCContainerRecord(*ObjCProtocol.second);

  for (const auto &Macro : API.getMacros())
    serializeMacroDefinitionRecord(*Macro.second);

  for (const auto &Typedef : API.getTypedefs())
    serializeTypedefRecord(*Typedef.second);

  Root["symbols"] = std::move(Symbols);
  Root["relationships"] = std::move(Relationships);

  return Root;
}

void SymbolGraphSerializer::serialize(raw_ostream &os) {
  Object root = serialize();
  if (Options.Compact)
    os << formatv("{0}", Value(std::move(root))) << "\n";
  else
    os << formatv("{0:2}", Value(std::move(root))) << "\n";
}
