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
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Version.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VersionTuple.h"
#include <optional>
#include <type_traits>

using namespace clang;
using namespace clang::extractapi;
using namespace llvm;
using namespace llvm::json;

namespace {

/// Helper function to inject a JSON object \p Obj into another object \p Paren
/// at position \p Key.
void serializeObject(Object &Paren, StringRef Key, std::optional<Object> Obj) {
  if (Obj)
    Paren[Key] = std::move(*Obj);
}

/// Helper function to inject a StringRef \p String into an object \p Paren at
/// position \p Key
void serializeString(Object &Paren, StringRef Key,
                     std::optional<std::string> String) {
  if (String)
    Paren[Key] = std::move(*String);
}

/// Helper function to inject a JSON array \p Array into object \p Paren at
/// position \p Key.
void serializeArray(Object &Paren, StringRef Key, std::optional<Array> Array) {
  if (Array)
    Paren[Key] = std::move(*Array);
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
std::optional<Object> serializeSemanticVersion(const VersionTuple &V) {
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
std::optional<Array>
serializeAvailability(const AvailabilitySet &Availabilities) {
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
    if (AvailInfo.Unavailable)
      Availability["isUnconditionallyUnavailable"] = true;
    else {
      serializeObject(Availability, "introducedVersion",
                      serializeSemanticVersion(AvailInfo.Introduced));
      serializeObject(Availability, "deprecatedVersion",
                      serializeSemanticVersion(AvailInfo.Deprecated));
      serializeObject(Availability, "obsoletedVersion",
                      serializeSemanticVersion(AvailInfo.Obsoleted));
    }
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
  case Language::CXX:
    return "c++";

  // Unsupported language currently
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
std::optional<Object> serializeDocComment(const DocComment &Comment) {
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
std::optional<Array>
serializeDeclarationFragments(const DeclarationFragments &DF) {
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
  if (auto *CategoryRecord =
          dyn_cast_or_null<const ObjCCategoryRecord>(&Record))
    Names["title"] =
        (CategoryRecord->Interface.Name + " (" + Record.Name + ")").str();
  else
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

Object serializeSymbolKind(APIRecord::RecordKind RK, Language Lang) {
  auto AddLangPrefix = [&Lang](StringRef S) -> std::string {
    return (getLanguageName(Lang) + "." + S).str();
  };

  Object Kind;
  switch (RK) {
  case APIRecord::RK_Unknown:
    llvm_unreachable("Records should have an explicit kind");
    break;
  case APIRecord::RK_Namespace:
    Kind["identifier"] = AddLangPrefix("namespace");
    Kind["displayName"] = "Namespace";
    break;
  case APIRecord::RK_GlobalFunction:
    Kind["identifier"] = AddLangPrefix("func");
    Kind["displayName"] = "Function";
    break;
  case APIRecord::RK_GlobalFunctionTemplate:
    Kind["identifier"] = AddLangPrefix("func");
    Kind["displayName"] = "Function Template";
    break;
  case APIRecord::RK_GlobalFunctionTemplateSpecialization:
    Kind["identifier"] = AddLangPrefix("func");
    Kind["displayName"] = "Function Template Specialization";
    break;
  case APIRecord::RK_GlobalVariableTemplate:
    Kind["identifier"] = AddLangPrefix("var");
    Kind["displayName"] = "Global Variable Template";
    break;
  case APIRecord::RK_GlobalVariableTemplateSpecialization:
    Kind["identifier"] = AddLangPrefix("var");
    Kind["displayName"] = "Global Variable Template Specialization";
    break;
  case APIRecord::RK_GlobalVariableTemplatePartialSpecialization:
    Kind["identifier"] = AddLangPrefix("var");
    Kind["displayName"] = "Global Variable Template Partial Specialization";
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
  case APIRecord::RK_CXXField:
    Kind["identifier"] = AddLangPrefix("property");
    Kind["displayName"] = "Instance Property";
    break;
  case APIRecord::RK_Union:
    Kind["identifier"] = AddLangPrefix("union");
    Kind["displayName"] = "Union";
    break;
  case APIRecord::RK_StaticField:
    Kind["identifier"] = AddLangPrefix("type.property");
    Kind["displayName"] = "Type Property";
    break;
  case APIRecord::RK_ClassTemplate:
  case APIRecord::RK_ClassTemplateSpecialization:
  case APIRecord::RK_ClassTemplatePartialSpecialization:
  case APIRecord::RK_CXXClass:
    Kind["identifier"] = AddLangPrefix("class");
    Kind["displayName"] = "Class";
    break;
  case APIRecord::RK_CXXMethodTemplate:
    Kind["identifier"] = AddLangPrefix("method");
    Kind["displayName"] = "Method Template";
    break;
  case APIRecord::RK_CXXMethodTemplateSpecialization:
    Kind["identifier"] = AddLangPrefix("method");
    Kind["displayName"] = "Method Template Specialization";
    break;
  case APIRecord::RK_CXXFieldTemplate:
    Kind["identifier"] = AddLangPrefix("property");
    Kind["displayName"] = "Template Property";
    break;
  case APIRecord::RK_Concept:
    Kind["identifier"] = AddLangPrefix("concept");
    Kind["displayName"] = "Concept";
    break;
  case APIRecord::RK_CXXStaticMethod:
    Kind["identifier"] = AddLangPrefix("type.method");
    Kind["displayName"] = "Static Method";
    break;
  case APIRecord::RK_CXXInstanceMethod:
    Kind["identifier"] = AddLangPrefix("method");
    Kind["displayName"] = "Instance Method";
    break;
  case APIRecord::RK_CXXConstructorMethod:
    Kind["identifier"] = AddLangPrefix("method");
    Kind["displayName"] = "Constructor";
    break;
  case APIRecord::RK_CXXDestructorMethod:
    Kind["identifier"] = AddLangPrefix("method");
    Kind["displayName"] = "Destructor";
    break;
  case APIRecord::RK_ObjCIvar:
    Kind["identifier"] = AddLangPrefix("ivar");
    Kind["displayName"] = "Instance Variable";
    break;
  case APIRecord::RK_ObjCInstanceMethod:
    Kind["identifier"] = AddLangPrefix("method");
    Kind["displayName"] = "Instance Method";
    break;
  case APIRecord::RK_ObjCClassMethod:
    Kind["identifier"] = AddLangPrefix("type.method");
    Kind["displayName"] = "Type Method";
    break;
  case APIRecord::RK_ObjCInstanceProperty:
    Kind["identifier"] = AddLangPrefix("property");
    Kind["displayName"] = "Instance Property";
    break;
  case APIRecord::RK_ObjCClassProperty:
    Kind["identifier"] = AddLangPrefix("type.property");
    Kind["displayName"] = "Type Property";
    break;
  case APIRecord::RK_ObjCInterface:
    Kind["identifier"] = AddLangPrefix("class");
    Kind["displayName"] = "Class";
    break;
  case APIRecord::RK_ObjCCategory:
    Kind["identifier"] = AddLangPrefix("class.extension");
    Kind["displayName"] = "Class Extension";
    break;
  case APIRecord::RK_ObjCCategoryModule:
    Kind["identifier"] = AddLangPrefix("module.extension");
    Kind["displayName"] = "Module Extension";
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

/// Serialize the symbol kind information.
///
/// The Symbol Graph symbol kind property contains a shorthand \c identifier
/// which is prefixed by the source language name, useful for tooling to parse
/// the kind, and a \c displayName for rendering human-readable names.
Object serializeSymbolKind(const APIRecord &Record, Language Lang) {
  return serializeSymbolKind(Record.getKind(), Lang);
}

template <typename RecordTy>
std::optional<Object>
serializeFunctionSignatureMixinImpl(const RecordTy &Record, std::true_type) {
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
std::optional<Object>
serializeFunctionSignatureMixinImpl(const RecordTy &Record, std::false_type) {
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

template <typename RecordTy>
std::optional<std::string> serializeAccessMixinImpl(const RecordTy &Record,
                                                    std::true_type) {
  const auto &AccessControl = Record.Access;
  std::string Access;
  if (AccessControl.empty())
    return std::nullopt;
  Access = AccessControl.getAccess();
  return Access;
}

template <typename RecordTy>
std::optional<std::string> serializeAccessMixinImpl(const RecordTy &Record,
                                                    std::false_type) {
  return std::nullopt;
}

template <typename RecordTy>
void serializeAccessMixin(Object &Paren, const RecordTy &Record) {
  auto accessLevel = serializeAccessMixinImpl(Record, has_access<RecordTy>());
  if (!accessLevel.has_value())
    accessLevel = "public";
  serializeString(Paren, "accessLevel", accessLevel);
}

template <typename RecordTy>
std::optional<Object> serializeTemplateMixinImpl(const RecordTy &Record,
                                                 std::true_type) {
  const auto &Template = Record.Templ;
  if (Template.empty())
    return std::nullopt;

  Object Generics;
  Array GenericParameters;
  for (const auto &Param : Template.getParameters()) {
    Object Parameter;
    Parameter["name"] = Param.Name;
    Parameter["index"] = Param.Index;
    Parameter["depth"] = Param.Depth;
    GenericParameters.emplace_back(std::move(Parameter));
  }
  if (!GenericParameters.empty())
    Generics["parameters"] = std::move(GenericParameters);

  Array GenericConstraints;
  for (const auto &Constr : Template.getConstraints()) {
    Object Constraint;
    Constraint["kind"] = Constr.Kind;
    Constraint["lhs"] = Constr.LHS;
    Constraint["rhs"] = Constr.RHS;
    GenericConstraints.emplace_back(std::move(Constraint));
  }

  if (!GenericConstraints.empty())
    Generics["constraints"] = std::move(GenericConstraints);

  return Generics;
}

template <typename RecordTy>
std::optional<Object> serializeTemplateMixinImpl(const RecordTy &Record,
                                                 std::false_type) {
  return std::nullopt;
}

template <typename RecordTy>
void serializeTemplateMixin(Object &Paren, const RecordTy &Record) {
  serializeObject(Paren, "swiftGenerics",
                  serializeTemplateMixinImpl(Record, has_template<RecordTy>()));
}

struct PathComponent {
  StringRef USR;
  StringRef Name;
  APIRecord::RecordKind Kind;

  PathComponent(StringRef USR, StringRef Name, APIRecord::RecordKind Kind)
      : USR(USR), Name(Name), Kind(Kind) {}
};

template <typename RecordTy>
bool generatePathComponents(
    const RecordTy &Record, const APISet &API,
    function_ref<void(const PathComponent &)> ComponentTransformer) {
  SmallVector<PathComponent, 4> ReverseComponenents;
  ReverseComponenents.emplace_back(Record.USR, Record.Name, Record.getKind());
  const auto *CurrentParent = &Record.ParentInformation;
  bool FailedToFindParent = false;
  while (CurrentParent && !CurrentParent->empty()) {
    PathComponent CurrentParentComponent(CurrentParent->ParentUSR,
                                         CurrentParent->ParentName,
                                         CurrentParent->ParentKind);

    auto *ParentRecord = CurrentParent->ParentRecord;
    // Slow path if we don't have a direct reference to the ParentRecord
    if (!ParentRecord)
      ParentRecord = API.findRecordForUSR(CurrentParent->ParentUSR);

    // If the parent is a category extended from internal module then we need to
    // pretend this belongs to the associated interface.
    if (auto *CategoryRecord =
            dyn_cast_or_null<ObjCCategoryRecord>(ParentRecord)) {
      if (!CategoryRecord->IsFromExternalModule) {
        ParentRecord = API.findRecordForUSR(CategoryRecord->Interface.USR);
        CurrentParentComponent = PathComponent(CategoryRecord->Interface.USR,
                                               CategoryRecord->Interface.Name,
                                               APIRecord::RK_ObjCInterface);
      }
    }

    // The parent record doesn't exist which means the symbol shouldn't be
    // treated as part of the current product.
    if (!ParentRecord) {
      FailedToFindParent = true;
      break;
    }

    ReverseComponenents.push_back(std::move(CurrentParentComponent));
    CurrentParent = &ParentRecord->ParentInformation;
  }

  for (const auto &PC : reverse(ReverseComponenents))
    ComponentTransformer(PC);

  return FailedToFindParent;
}

Object serializeParentContext(const PathComponent &PC, Language Lang) {
  Object ParentContextElem;
  ParentContextElem["usr"] = PC.USR;
  ParentContextElem["name"] = PC.Name;
  ParentContextElem["kind"] = serializeSymbolKind(PC.Kind, Lang)["identifier"];
  return ParentContextElem;
}

template <typename RecordTy>
Array generateParentContexts(const RecordTy &Record, const APISet &API,
                             Language Lang) {
  Array ParentContexts;
  generatePathComponents(
      Record, API, [Lang, &ParentContexts](const PathComponent &PC) {
        ParentContexts.push_back(serializeParentContext(PC, Lang));
      });

  return ParentContexts;
}
} // namespace

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
  Module["name"] = API.ProductName;
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
std::optional<Object>
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
  SmallVector<StringRef, 4> PathComponentsNames;
  // If this returns true it indicates that we couldn't find a symbol in the
  // hierarchy.
  if (generatePathComponents(Record, API,
                             [&PathComponentsNames](const PathComponent &PC) {
                               PathComponentsNames.push_back(PC.Name);
                             }))
    return {};

  serializeArray(Obj, "pathComponents", Array(PathComponentsNames));

  serializeFunctionSignatureMixin(Obj, Record);
  serializeAccessMixin(Obj, Record);
  serializeTemplateMixin(Obj, Record);

  return Obj;
}

template <typename MemberTy>
void SymbolGraphSerializer::serializeMembers(
    const APIRecord &Record,
    const SmallVector<std::unique_ptr<MemberTy>> &Members) {
  // Members should not be serialized if we aren't recursing.
  if (!ShouldRecurse)
    return;
  for (const auto &Member : Members) {
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
  case RelationshipKind::ExtensionTo:
    return "extensionTo";
  }
  llvm_unreachable("Unhandled relationship kind");
}

StringRef SymbolGraphSerializer::getConstraintString(ConstraintKind Kind) {
  switch (Kind) {
  case ConstraintKind::Conformance:
    return "conformance";
  case ConstraintKind::ConditionalConformance:
    return "conditionalConformance";
  }
  llvm_unreachable("Unhandled constraint kind");
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

void SymbolGraphSerializer::visitNamespaceRecord(
    const NamespaceRecord &Record) {
  auto Namespace = serializeAPIRecord(Record);
  if (!Namespace)
    return;
  Symbols.emplace_back(std::move(*Namespace));
  if (!Record.ParentInformation.empty())
    serializeRelationship(RelationshipKind::MemberOf, Record,
                          Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitGlobalFunctionRecord(
    const GlobalFunctionRecord &Record) {
  auto Obj = serializeAPIRecord(Record);
  if (!Obj)
    return;

  Symbols.emplace_back(std::move(*Obj));
}

void SymbolGraphSerializer::visitGlobalVariableRecord(
    const GlobalVariableRecord &Record) {
  auto Obj = serializeAPIRecord(Record);
  if (!Obj)
    return;

  Symbols.emplace_back(std::move(*Obj));
}

void SymbolGraphSerializer::visitEnumRecord(const EnumRecord &Record) {
  auto Enum = serializeAPIRecord(Record);
  if (!Enum)
    return;

  Symbols.emplace_back(std::move(*Enum));
  serializeMembers(Record, Record.Constants);
}

void SymbolGraphSerializer::visitStructRecord(const StructRecord &Record) {
  auto Struct = serializeAPIRecord(Record);
  if (!Struct)
    return;

  Symbols.emplace_back(std::move(*Struct));
  serializeMembers(Record, Record.Fields);
}

void SymbolGraphSerializer::visitStaticFieldRecord(
    const StaticFieldRecord &Record) {
  auto StaticField = serializeAPIRecord(Record);
  if (!StaticField)
    return;
  Symbols.emplace_back(std::move(*StaticField));
  serializeRelationship(RelationshipKind::MemberOf, Record, Record.Context);
}

void SymbolGraphSerializer::visitCXXClassRecord(const CXXClassRecord &Record) {
  auto Class = serializeAPIRecord(Record);
  if (!Class)
    return;

  Symbols.emplace_back(std::move(*Class));
  for (const auto Base : Record.Bases)
    serializeRelationship(RelationshipKind::InheritsFrom, Record, Base);
  if (!Record.ParentInformation.empty())
    serializeRelationship(RelationshipKind::MemberOf, Record,
                          Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitClassTemplateRecord(
    const ClassTemplateRecord &Record) {
  auto Class = serializeAPIRecord(Record);
  if (!Class)
    return;

  Symbols.emplace_back(std::move(*Class));
  for (const auto Base : Record.Bases)
    serializeRelationship(RelationshipKind::InheritsFrom, Record, Base);
  if (!Record.ParentInformation.empty())
    serializeRelationship(RelationshipKind::MemberOf, Record,
                          Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitClassTemplateSpecializationRecord(
    const ClassTemplateSpecializationRecord &Record) {
  auto Class = serializeAPIRecord(Record);
  if (!Class)
    return;

  Symbols.emplace_back(std::move(*Class));

  for (const auto Base : Record.Bases)
    serializeRelationship(RelationshipKind::InheritsFrom, Record, Base);
  if (!Record.ParentInformation.empty())
    serializeRelationship(RelationshipKind::MemberOf, Record,
                          Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitClassTemplatePartialSpecializationRecord(
    const ClassTemplatePartialSpecializationRecord &Record) {
  auto Class = serializeAPIRecord(Record);
  if (!Class)
    return;

  Symbols.emplace_back(std::move(*Class));

  for (const auto Base : Record.Bases)
    serializeRelationship(RelationshipKind::InheritsFrom, Record, Base);
  if (!Record.ParentInformation.empty())
    serializeRelationship(RelationshipKind::MemberOf, Record,
                          Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitCXXInstanceMethodRecord(
    const CXXInstanceMethodRecord &Record) {
  auto InstanceMethod = serializeAPIRecord(Record);
  if (!InstanceMethod)
    return;

  Symbols.emplace_back(std::move(*InstanceMethod));
  serializeRelationship(RelationshipKind::MemberOf, Record,
                        Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitCXXStaticMethodRecord(
    const CXXStaticMethodRecord &Record) {
  auto StaticMethod = serializeAPIRecord(Record);
  if (!StaticMethod)
    return;

  Symbols.emplace_back(std::move(*StaticMethod));
  serializeRelationship(RelationshipKind::MemberOf, Record,
                        Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitMethodTemplateRecord(
    const CXXMethodTemplateRecord &Record) {
  if (!ShouldRecurse)
    // Ignore child symbols
    return;
  auto MethodTemplate = serializeAPIRecord(Record);
  if (!MethodTemplate)
    return;
  Symbols.emplace_back(std::move(*MethodTemplate));
  serializeRelationship(RelationshipKind::MemberOf, Record,
                        Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitMethodTemplateSpecializationRecord(
    const CXXMethodTemplateSpecializationRecord &Record) {
  if (!ShouldRecurse)
    // Ignore child symbols
    return;
  auto MethodTemplateSpecialization = serializeAPIRecord(Record);
  if (!MethodTemplateSpecialization)
    return;
  Symbols.emplace_back(std::move(*MethodTemplateSpecialization));
  serializeRelationship(RelationshipKind::MemberOf, Record,
                        Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitCXXFieldRecord(const CXXFieldRecord &Record) {
  if (!ShouldRecurse)
    return;
  auto CXXField = serializeAPIRecord(Record);
  if (!CXXField)
    return;
  Symbols.emplace_back(std::move(*CXXField));
  serializeRelationship(RelationshipKind::MemberOf, Record,
                        Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitCXXFieldTemplateRecord(
    const CXXFieldTemplateRecord &Record) {
  if (!ShouldRecurse)
    // Ignore child symbols
    return;
  auto CXXFieldTemplate = serializeAPIRecord(Record);
  if (!CXXFieldTemplate)
    return;
  Symbols.emplace_back(std::move(*CXXFieldTemplate));
  serializeRelationship(RelationshipKind::MemberOf, Record,
                        Record.ParentInformation.ParentRecord);
}

void SymbolGraphSerializer::visitConceptRecord(const ConceptRecord &Record) {
  auto Concept = serializeAPIRecord(Record);
  if (!Concept)
    return;

  Symbols.emplace_back(std::move(*Concept));
}

void SymbolGraphSerializer::visitGlobalVariableTemplateRecord(
    const GlobalVariableTemplateRecord &Record) {
  auto GlobalVariableTemplate = serializeAPIRecord(Record);
  if (!GlobalVariableTemplate)
    return;
  Symbols.emplace_back(std::move(*GlobalVariableTemplate));
}

void SymbolGraphSerializer::visitGlobalVariableTemplateSpecializationRecord(
    const GlobalVariableTemplateSpecializationRecord &Record) {
  auto GlobalVariableTemplateSpecialization = serializeAPIRecord(Record);
  if (!GlobalVariableTemplateSpecialization)
    return;
  Symbols.emplace_back(std::move(*GlobalVariableTemplateSpecialization));
}

void SymbolGraphSerializer::
    visitGlobalVariableTemplatePartialSpecializationRecord(
        const GlobalVariableTemplatePartialSpecializationRecord &Record) {
  auto GlobalVariableTemplatePartialSpecialization = serializeAPIRecord(Record);
  if (!GlobalVariableTemplatePartialSpecialization)
    return;
  Symbols.emplace_back(std::move(*GlobalVariableTemplatePartialSpecialization));
}

void SymbolGraphSerializer::visitGlobalFunctionTemplateRecord(
    const GlobalFunctionTemplateRecord &Record) {
  auto GlobalFunctionTemplate = serializeAPIRecord(Record);
  if (!GlobalFunctionTemplate)
    return;
  Symbols.emplace_back(std::move(*GlobalFunctionTemplate));
}

void SymbolGraphSerializer::visitGlobalFunctionTemplateSpecializationRecord(
    const GlobalFunctionTemplateSpecializationRecord &Record) {
  auto GlobalFunctionTemplateSpecialization = serializeAPIRecord(Record);
  if (!GlobalFunctionTemplateSpecialization)
    return;
  Symbols.emplace_back(std::move(*GlobalFunctionTemplateSpecialization));
}

void SymbolGraphSerializer::visitObjCContainerRecord(
    const ObjCContainerRecord &Record) {
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

void SymbolGraphSerializer::visitObjCCategoryRecord(
    const ObjCCategoryRecord &Record) {
  if (!Record.IsFromExternalModule)
    return;

  // Check if the current Category' parent has been visited before, if so skip.
  if (!visitedCategories.contains(Record.Interface.Name)) {
    visitedCategories.insert(Record.Interface.Name);
    Object Obj;
    serializeObject(Obj, "identifier",
                    serializeIdentifier(Record, API.getLanguage()));
    serializeObject(Obj, "kind",
                    serializeSymbolKind(APIRecord::RK_ObjCCategoryModule,
                                        API.getLanguage()));
    Obj["accessLevel"] = "public";
    Symbols.emplace_back(std::move(Obj));
  }

  Object Relationship;
  Relationship["source"] = Record.USR;
  Relationship["target"] = Record.Interface.USR;
  Relationship["targetFallback"] = Record.Interface.Name;
  Relationship["kind"] = getRelationshipString(RelationshipKind::ExtensionTo);
  Relationships.emplace_back(std::move(Relationship));

  auto ObjCCategory = serializeAPIRecord(Record);

  if (!ObjCCategory)
    return;

  Symbols.emplace_back(std::move(*ObjCCategory));
  serializeMembers(Record, Record.Methods);
  serializeMembers(Record, Record.Properties);

  // Surface the protocols of the category to the interface.
  for (const auto &Protocol : Record.Protocols)
    serializeRelationship(RelationshipKind::ConformsTo, Record, Protocol);
}

void SymbolGraphSerializer::visitMacroDefinitionRecord(
    const MacroDefinitionRecord &Record) {
  auto Macro = serializeAPIRecord(Record);

  if (!Macro)
    return;

  Symbols.emplace_back(std::move(*Macro));
}

void SymbolGraphSerializer::serializeSingleRecord(const APIRecord *Record) {
  switch (Record->getKind()) {
  case APIRecord::RK_Unknown:
    llvm_unreachable("Records should have a known kind!");
  case APIRecord::RK_GlobalFunction:
    visitGlobalFunctionRecord(*cast<GlobalFunctionRecord>(Record));
    break;
  case APIRecord::RK_GlobalVariable:
    visitGlobalVariableRecord(*cast<GlobalVariableRecord>(Record));
    break;
  case APIRecord::RK_Enum:
    visitEnumRecord(*cast<EnumRecord>(Record));
    break;
  case APIRecord::RK_Struct:
    visitStructRecord(*cast<StructRecord>(Record));
    break;
  case APIRecord::RK_StaticField:
    visitStaticFieldRecord(*cast<StaticFieldRecord>(Record));
    break;
  case APIRecord::RK_CXXClass:
    visitCXXClassRecord(*cast<CXXClassRecord>(Record));
    break;
  case APIRecord::RK_ObjCInterface:
    visitObjCContainerRecord(*cast<ObjCInterfaceRecord>(Record));
    break;
  case APIRecord::RK_ObjCProtocol:
    visitObjCContainerRecord(*cast<ObjCProtocolRecord>(Record));
    break;
  case APIRecord::RK_ObjCCategory:
    visitObjCCategoryRecord(*cast<ObjCCategoryRecord>(Record));
    break;
  case APIRecord::RK_MacroDefinition:
    visitMacroDefinitionRecord(*cast<MacroDefinitionRecord>(Record));
    break;
  case APIRecord::RK_Typedef:
    visitTypedefRecord(*cast<TypedefRecord>(Record));
    break;
  default:
    if (auto Obj = serializeAPIRecord(*Record)) {
      Symbols.emplace_back(std::move(*Obj));
      auto &ParentInformation = Record->ParentInformation;
      if (!ParentInformation.empty())
        serializeRelationship(RelationshipKind::MemberOf, *Record,
                              *ParentInformation.ParentRecord);
    }
    break;
  }
}

void SymbolGraphSerializer::visitTypedefRecord(const TypedefRecord &Record) {
  // Typedefs of anonymous types have their entries unified with the underlying
  // type.
  bool ShouldDrop = Record.UnderlyingType.Name.empty();
  // enums declared with `NS_OPTION` have a named enum and a named typedef, with
  // the same name
  ShouldDrop |= (Record.UnderlyingType.Name == Record.Name);
  if (ShouldDrop)
    return;

  auto Typedef = serializeAPIRecord(Record);
  if (!Typedef)
    return;

  (*Typedef)["type"] = Record.UnderlyingType.USR;

  Symbols.emplace_back(std::move(*Typedef));
}

Object SymbolGraphSerializer::serialize() {
  traverseAPISet();
  return serializeCurrentGraph();
}

Object SymbolGraphSerializer::serializeCurrentGraph() {
  Object Root;
  serializeObject(Root, "metadata", serializeMetadata());
  serializeObject(Root, "module", serializeModule());

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

std::optional<Object>
SymbolGraphSerializer::serializeSingleSymbolSGF(StringRef USR,
                                                const APISet &API) {
  APIRecord *Record = API.findRecordForUSR(USR);
  if (!Record)
    return {};

  Object Root;
  APIIgnoresList EmptyIgnores;
  SymbolGraphSerializer Serializer(API, EmptyIgnores,
                                   /*Options.Compact*/ {true},
                                   /*ShouldRecurse*/ false);
  Serializer.serializeSingleRecord(Record);
  serializeObject(Root, "symbolGraph", Serializer.serializeCurrentGraph());

  Language Lang = API.getLanguage();
  serializeArray(Root, "parentContexts",
                 generateParentContexts(*Record, API, Lang));

  Array RelatedSymbols;

  for (const auto &Fragment : Record->Declaration.getFragments()) {
    // If we don't have a USR there isn't much we can do.
    if (Fragment.PreciseIdentifier.empty())
      continue;

    APIRecord *RelatedRecord = API.findRecordForUSR(Fragment.PreciseIdentifier);

    // If we can't find the record let's skip.
    if (!RelatedRecord)
      continue;

    Object RelatedSymbol;
    RelatedSymbol["usr"] = RelatedRecord->USR;
    RelatedSymbol["declarationLanguage"] = getLanguageName(Lang);
    // TODO: once we record this properly let's serialize it right.
    RelatedSymbol["accessLevel"] = "public";
    RelatedSymbol["filePath"] = RelatedRecord->Location.getFilename();
    RelatedSymbol["moduleName"] = API.ProductName;
    RelatedSymbol["isSystem"] = RelatedRecord->IsFromSystemHeader;

    serializeArray(RelatedSymbol, "parentContexts",
                   generateParentContexts(*RelatedRecord, API, Lang));
    RelatedSymbols.push_back(std::move(RelatedSymbol));
  }

  serializeArray(Root, "relatedSymbols", RelatedSymbols);
  return Root;
}
