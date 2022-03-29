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
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VersionTuple.h"

using namespace clang;
using namespace clang::extractapi;
using namespace llvm;
using namespace llvm::json;

namespace {

/// Helper function to inject a JSON object \p Obj into another object \p Paren
/// at position \p Key.
void serializeObject(Object &Paren, StringRef Key, Optional<Object> Obj) {
  if (Obj)
    Paren[Key] = std::move(Obj.getValue());
}

/// Helper function to inject a JSON array \p Array into object \p Paren at
/// position \p Key.
void serializeArray(Object &Paren, StringRef Key, Optional<Array> Array) {
  if (Array)
    Paren[Key] = std::move(Array.getValue());
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
/// \returns \c None if the version \p V is empty, or an \c Object containing
/// the semantic version representation of \p V.
Optional<Object> serializeSemanticVersion(const VersionTuple &V) {
  if (V.empty())
    return None;

  Object Version;
  Version["major"] = V.getMajor();
  Version["minor"] = V.getMinor().getValueOr(0);
  Version["patch"] = V.getSubminor().getValueOr(0);
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

/// Serialize a source location in file.
///
/// \param Loc The presumed location to serialize.
/// \param IncludeFileURI If true, include the file path of \p Loc as a URI.
/// Defaults to false.
Object serializeSourcePosition(const PresumedLoc &Loc,
                               bool IncludeFileURI = false) {
  assert(Loc.isValid() && "invalid source position");

  Object SourcePosition;
  SourcePosition["line"] = Loc.getLine();
  SourcePosition["character"] = Loc.getColumn();

  if (IncludeFileURI) {
    std::string FileURI = "file://";
    // Normalize file path to use forward slashes for the URI.
    FileURI += sys::path::convert_to_slash(Loc.getFilename());
    SourcePosition["uri"] = FileURI;
  }

  return SourcePosition;
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
/// versions of the symbol as semantic versions, if not default.
/// Availability information also contains flags to indicate if the symbol is
/// unconditionally unavailable or deprecated,
/// i.e. \c __attribute__((unavailable)) and \c __attribute__((deprecated)).
///
/// \returns \c None if the symbol has default availability attributes, or
/// an \c Object containing the formatted availability information.
Optional<Object> serializeAvailability(const AvailabilityInfo &Avail) {
  if (Avail.isDefault())
    return None;

  Object Availbility;
  serializeObject(Availbility, "introducedVersion",
                  serializeSemanticVersion(Avail.Introduced));
  serializeObject(Availbility, "deprecatedVersion",
                  serializeSemanticVersion(Avail.Deprecated));
  serializeObject(Availbility, "obsoletedVersion",
                  serializeSemanticVersion(Avail.Obsoleted));
  if (Avail.isUnavailable())
    Availbility["isUnconditionallyUnavailable"] = true;
  if (Avail.isUnconditionallyDeprecated())
    Availbility["isUnconditionallyDeprecated"] = true;

  return Availbility;
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
/// \returns \c None if \p Comment is empty, or an \c Object containing the
/// formatted lines.
Optional<Object> serializeDocComment(const DocComment &Comment) {
  if (Comment.empty())
    return None;

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
/// \returns \c None if \p DF is empty, or an \c Array containing the formatted
/// declaration fragments array.
Optional<Array> serializeDeclarationFragments(const DeclarationFragments &DF) {
  if (DF.getFragments().empty())
    return None;

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

/// Serialize the function signature field of a function, as specified by the
/// Symbol Graph format.
///
/// The Symbol Graph function signature property contains two arrays.
///   - The \c returns array is the declaration fragments of the return type;
///   - The \c parameters array contains names and declaration fragments of the
///     parameters.
///
/// \returns \c None if \p FS is empty, or an \c Object containing the
/// formatted function signature.
Optional<Object> serializeFunctionSignature(const FunctionSignature &FS) {
  if (FS.empty())
    return None;

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
  case APIRecord::RK_Global: {
    auto *GR = dyn_cast<GlobalRecord>(&Record);
    switch (GR->GlobalKind) {
    case GVKind::Function:
      Kind["identifier"] = AddLangPrefix("func");
      Kind["displayName"] = "Function";
      break;
    case GVKind::Variable:
      Kind["identifier"] = AddLangPrefix("var");
      Kind["displayName"] = "Global Variable";
      break;
    case GVKind::Unknown:
      // Unknown global kind
      break;
    }
    break;
  }
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
    if (dyn_cast<ObjCMethodRecord>(&Record)->IsInstanceMethod) {
      Kind["identifier"] = AddLangPrefix("method");
      Kind["displayName"] = "Instance Method";
    } else {
      Kind["identifier"] = AddLangPrefix("type.method");
      Kind["displayName"] = "Type Method";
    }
    break;
  case APIRecord::RK_ObjCProperty:
    Kind["identifier"] = AddLangPrefix("property");
    Kind["displayName"] = "Instance Property";
    break;
  case APIRecord::RK_ObjCInterface:
    Kind["identifier"] = AddLangPrefix("class");
    Kind["displayName"] = "Class";
    break;
  case APIRecord::RK_ObjCProtocol:
    Kind["identifier"] = AddLangPrefix("protocol");
    Kind["displayName"] = "Protocol";
    break;
  }

  return Kind;
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
  // Skip unconditionally unavailable symbols
  if (Record.Availability.isUnconditionallyUnavailable())
    return true;

  return false;
}

Optional<Object>
SymbolGraphSerializer::serializeAPIRecord(const APIRecord &Record) const {
  if (shouldSkip(Record))
    return None;

  Object Obj;
  serializeObject(Obj, "identifier",
                  serializeIdentifier(Record, API.getLanguage()));
  serializeObject(Obj, "kind", serializeSymbolKind(Record, API.getLanguage()));
  serializeObject(Obj, "names", serializeNames(Record));
  serializeObject(
      Obj, "location",
      serializeSourcePosition(Record.Location, /*IncludeFileURI=*/true));
  serializeObject(Obj, "availbility",
                  serializeAvailability(Record.Availability));
  serializeObject(Obj, "docComment", serializeDocComment(Record.Comment));
  serializeArray(Obj, "declarationFragments",
                 serializeDeclarationFragments(Record.Declaration));

  return Obj;
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
  Relationship["kind"] = getRelationshipString(Kind);

  Relationships.emplace_back(std::move(Relationship));
}

void SymbolGraphSerializer::serializeGlobalRecord(const GlobalRecord &Record) {
  auto Obj = serializeAPIRecord(Record);
  if (!Obj)
    return;

  if (Record.GlobalKind == GVKind::Function)
    serializeObject(*Obj, "parameters",
                    serializeFunctionSignature(Record.Signature));

  Symbols.emplace_back(std::move(*Obj));
}

void SymbolGraphSerializer::serializeEnumRecord(const EnumRecord &Record) {
  auto Enum = serializeAPIRecord(Record);
  if (!Enum)
    return;

  Symbols.emplace_back(std::move(*Enum));

  for (const auto &Constant : Record.Constants) {
    auto EnumConstant = serializeAPIRecord(*Constant);
    if (!EnumConstant)
      continue;

    Symbols.emplace_back(std::move(*EnumConstant));
    serializeRelationship(RelationshipKind::MemberOf, *Constant, Record);
  }
}

void SymbolGraphSerializer::serializeStructRecord(const StructRecord &Record) {
  auto Struct = serializeAPIRecord(Record);
  if (!Struct)
    return;

  Symbols.emplace_back(std::move(*Struct));

  for (const auto &Field : Record.Fields) {
    auto StructField = serializeAPIRecord(*Field);
    if (!StructField)
      continue;

    Symbols.emplace_back(std::move(*StructField));
    serializeRelationship(RelationshipKind::MemberOf, *Field, Record);
  }
}

void SymbolGraphSerializer::serializeObjCContainerRecord(
    const ObjCContainerRecord &Record) {
  auto ObjCContainer = serializeAPIRecord(Record);
  if (!ObjCContainer)
    return;

  Symbols.emplace_back(std::move(*ObjCContainer));

  // Record instance variables and that the instance variables are members of
  // the container.
  for (const auto &Ivar : Record.Ivars) {
    auto ObjCIvar = serializeAPIRecord(*Ivar);
    if (!ObjCIvar)
      continue;

    Symbols.emplace_back(std::move(*ObjCIvar));
    serializeRelationship(RelationshipKind::MemberOf, *Ivar, Record);
  }

  // Record methods and that the methods are members of the container.
  for (const auto &Method : Record.Methods) {
    auto ObjCMethod = serializeAPIRecord(*Method);
    if (!ObjCMethod)
      continue;

    Symbols.emplace_back(std::move(*ObjCMethod));
    serializeRelationship(RelationshipKind::MemberOf, *Method, Record);
  }

  // Record properties and that the properties are members of the container.
  for (const auto &Property : Record.Properties) {
    auto ObjCProperty = serializeAPIRecord(*Property);
    if (!ObjCProperty)
      continue;

    Symbols.emplace_back(std::move(*ObjCProperty));
    serializeRelationship(RelationshipKind::MemberOf, *Property, Record);
  }

  for (const auto &Protocol : Record.Protocols)
    // Record that Record conforms to Protocol.
    serializeRelationship(RelationshipKind::ConformsTo, Record, Protocol);

  if (auto *ObjCInterface = dyn_cast<ObjCInterfaceRecord>(&Record))
    if (!ObjCInterface->SuperClass.empty())
      // If Record is an Objective-C interface record and it has a super class,
      // record that Record is inherited from SuperClass.
      serializeRelationship(RelationshipKind::InheritsFrom, Record,
                            ObjCInterface->SuperClass);
}

Object SymbolGraphSerializer::serialize() {
  Object Root;
  serializeObject(Root, "metadata", serializeMetadata());
  serializeObject(Root, "module", serializeModule());

  // Serialize global records in the API set.
  for (const auto &Global : API.getGlobals())
    serializeGlobalRecord(*Global.second);

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

  Root["symbols"] = std::move(Symbols);
  Root["relationhips"] = std::move(Relationships);

  return Root;
}

void SymbolGraphSerializer::serialize(raw_ostream &os) {
  Object root = serialize();
  if (Options.Compact)
    os << formatv("{0}", Value(std::move(root))) << "\n";
  else
    os << formatv("{0:2}", Value(std::move(root))) << "\n";
}
