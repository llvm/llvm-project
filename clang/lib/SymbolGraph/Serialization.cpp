//===- SymbolGraph/Serialization.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the SymbolGraph serializer and parser.
///
//===----------------------------------------------------------------------===//

#include "clang/SymbolGraph/Serialization.h"
#include "clang/Basic/Version.h"
#include "clang/SymbolGraph/API.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::symbolgraph;
using namespace llvm;
using namespace llvm::json;

namespace {

static void serializeObject(Object &Paren, StringRef Key,
                            Optional<Object> Obj) {
  if (Obj)
    Paren[Key] = std::move(Obj.getValue());
}

static void serializeArray(Object &Paren, StringRef Key,
                           Optional<Array> Array) {
  if (Array)
    Paren[Key] = std::move(Array.getValue());
}

// SymbolGraph: SemanticVersion
static Optional<Object> serializeSemanticVersion(const VersionTuple &V) {
  if (V.empty())
    return None;

  Object Version;
  Version["major"] = V.getMajor();
  Version["minor"] = V.getMinor().getValueOr(0);
  Version["patch"] = V.getSubminor().getValueOr(0);
  return Version;
}

static Object serializeOperatingSystem(const Triple &T) {
  Object OS;
  OS["name"] = T.getOSTypeName(T.getOS());
  serializeObject(OS, "minimumVersion",
                  serializeSemanticVersion(T.getMinimumSupportedOSVersion()));
  return OS;
}

// SymbolGraph: Platform
static Object serializePlatform(const Triple &T) {
  Object Platform;
  Platform["architecture"] = T.getArchName();
  Platform["vendor"] = T.getVendorName();
  Platform["operatingSystem"] = serializeOperatingSystem(T);
  return Platform;
}

// SymbolGraph: SourcePosition
static Object serializeSourcePosition(const PresumedLoc &Loc,
                                      bool IncludeFileURI = false) {
  assert(Loc.isValid() && "invalid source position");

  Object SourcePosition;
  SourcePosition["line"] = Loc.getLine();
  SourcePosition["character"] = Loc.getColumn();

  if (IncludeFileURI) {
    std::string FileURI = "file://";
    FileURI += sys::path::convert_to_slash(Loc.getFilename());
    SourcePosition["uri"] = FileURI;
  }

  return SourcePosition;
}

// SymbolGraph: SourceRange
static Object serializeSourceRange(const PresumedLoc &BeginLoc,
                                   const PresumedLoc &EndLoc) {
  Object SourceRange;
  serializeObject(SourceRange, "start", serializeSourcePosition(BeginLoc));
  serializeObject(SourceRange, "end", serializeSourcePosition(EndLoc));
  return SourceRange;
}

// SymbolGraph: AvailabilityItem
static Optional<Object> serializeAvailability(const AvailabilityInfo &Avail) {
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

static StringRef getLanguageName(const LangOptions &LangOpts) {
  auto Language =
      LangStandard::getLangStandardForKind(LangOpts.LangStd).getLanguage();
  switch (Language) {
  case Language::C:
    return "c";
  case Language::ObjC:
    return "objc";

  // Unsupported language currently
  case Language::CXX:
  case Language::ObjCXX:
  case Language::OpenCL:
  case Language::OpenCLCXX:
  case Language::CUDA:
  case Language::RenderScript:
  case Language::HIP:

  // Languages that the frontend cannot parse and compile
  case Language::Unknown:
  case Language::Asm:
  case Language::LLVM_IR:
    llvm_unreachable("Unsupported language kind");
  }

  llvm_unreachable("Unhandled language kind");
}

// SymbolGraph: Symbol::identifier
static Object serializeIdentifier(const APIRecord &Record,
                                  const LangOptions &LangOpts) {
  Object Identifier;
  Identifier["precise"] = Record.USR;
  Identifier["interfaceLanguage"] = getLanguageName(LangOpts);

  return Identifier;
}

// SymbolGraph: DocComment
static Optional<Object> serializeDocComment(const DocComment &Comment) {
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

static Optional<Array>
serializeDeclarationFragments(const DeclarationFragments &DF) {
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

static Optional<Object>
serializeFunctionSignature(const FunctionSignature &FS) {
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

static Object serializeNames(const APIRecord &Record) {
  Object Names;
  Names["title"] = Record.Name;
  serializeArray(Names, "subHeading",
                 serializeDeclarationFragments(Record.SubHeading));

  return Names;
}

// SymbolGraph: Symbol::kind
static Object serializeSymbolKind(const APIRecord &Record,
                                  const LangOptions &LangOpts) {
  Object Kind;
  switch (Record.getKind()) {
  case APIRecord::RK_Global:
    auto *GR = dyn_cast<GlobalRecord>(&Record);
    switch (GR->GlobalKind) {
    case GVKind::Function:
      Kind["identifier"] = (getLanguageName(LangOpts) + ".func").str();
      Kind["displayName"] = "Function";
      break;
    case GVKind::Variable:
      Kind["identifier"] = (getLanguageName(LangOpts) + ".var").str();
      Kind["displayName"] = "Global Variable";
      break;
    case GVKind::Unknown:
      // Unknown global kind
      break;
    }
    break;
  }

  return Kind;
}

} // namespace

const VersionTuple Serializer::FormatVersion{0, 5, 3};

Object Serializer::serializeMetadata() const {
  Object Metadata;
  serializeObject(Metadata, "formatVersion",
                  serializeSemanticVersion(FormatVersion));
  Metadata["generator"] = clang::getClangFullVersion();
  return Metadata;
}

Object Serializer::serializeModule() const {
  Object Module;
  // FIXME: What to put in here?
  Module["name"] = "";
  serializeObject(Module, "platform", serializePlatform(API.getTarget()));
  return Module;
}

bool Serializer::shouldSkip(const APIRecord &Record) const {
  // Skip unconditionally unavailable symbols
  if (Record.Availability.isUnconditionallyUnavailable())
    return true;

  return false;
}

Optional<Object> Serializer::serializeAPIRecord(const APIRecord &Record) const {
  if (shouldSkip(Record))
    return None;

  Object Obj;
  serializeObject(Obj, "identifier",
                  serializeIdentifier(Record, API.getLangOpts()));
  serializeObject(Obj, "kind", serializeSymbolKind(Record, API.getLangOpts()));
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

void Serializer::serializeGlobalRecord(const GlobalRecord &Record) {
  auto Obj = serializeAPIRecord(Record);
  if (!Obj)
    return;

  if (Record.GlobalKind == GVKind::Function)
    serializeObject(*Obj, "parameters",
                    serializeFunctionSignature(Record.Signature));

  Symbols.emplace_back(std::move(*Obj));
}

Object Serializer::serialize() {
  Object Root;
  serializeObject(Root, "metadata", serializeMetadata());
  serializeObject(Root, "module", serializeModule());

  for (const auto &Global : API.getGlobals())
    serializeGlobalRecord(*Global.second);

  Root["symbols"] = std::move(Symbols);
  Root["relationhips"] = std::move(Relationships);

  return Root;
}

void Serializer::serialize(raw_ostream &os) {
  Object root = serialize();
  if (Options.Compact)
    os << formatv("{0}", Value(std::move(root))) << "\n";
  else
    os << formatv("{0:2}", Value(std::move(root))) << "\n";
}
