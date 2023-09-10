#include "clang-symbolgraph-merger/SymbolGraph.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/VersionTuple.h"
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

using namespace sgmerger;
using namespace llvm;
using namespace llvm::json;
using namespace clang::extractapi;

namespace {

APIRecord::RecordKind getSymbolKind(const Object &Kind) {

  if (auto Identifier = Kind.getString("identifier")) {
    // Remove danguage prefix
    auto Id = Identifier->split('.').second;
    if (Id.equals("func"))
      return APIRecord::RK_GlobalFunction;
    if (Id.equals("var"))
      return APIRecord::RK_GlobalVariable;
    if (Id.equals("enum.case"))
      return APIRecord::RK_EnumConstant;
    if (Id.equals("enum"))
      return APIRecord::RK_Enum;
    if (Id.equals("property"))
      return APIRecord::RK_StructField;
    if (Id.equals("struct"))
      return APIRecord::RK_Struct;
    if (Id.equals("ivar"))
      return APIRecord::RK_ObjCIvar;
    if (Id.equals("method"))
      return APIRecord::RK_ObjCInstanceMethod;
    if (Id.equals("type.method"))
      return APIRecord::RK_ObjCClassMethod;
    if (Id.equals("property"))
      return APIRecord::RK_ObjCInstanceProperty;
    if (Id.equals("type.property"))
      return APIRecord::RK_ObjCClassProperty;
    if (Id.equals("class"))
      return APIRecord::RK_ObjCInterface;
    if (Id.equals("protocod"))
      return APIRecord::RK_ObjCProtocol;
    if (Id.equals("macro"))
      return APIRecord::RK_MacroDefinition;
    if (Id.equals("typealias"))
      return APIRecord::RK_Typedef;
  }
  return APIRecord::RK_Unknown;
}

VersionTuple parseVersionTupleFromJSON(const Object *VTObj) {
  auto Major = VTObj->getInteger("major").value_or(0);
  auto Minor = VTObj->getInteger("minor").value_or(0);
  auto Patch = VTObj->getInteger("patch").value_or(0);
  return VersionTuple(Major, Minor, Patch);
}

RecordLocation parseSourcePositionFromJSON(const Object *PosObj,
                                           std::string Filename = "") {
  assert(PosObj);
  unsigned Line = PosObj->getInteger("line").value_or(0);
  unsigned Col = PosObj->getInteger("character").value_or(0);
  return RecordLocation(Line, Col, Filename);
}

RecordLocation parseRecordLocationFromJSON(const Object *LocObj) {
  assert(LocObj);

  std::string Filename(LocObj->getString("uri").value_or(""));
  // extract file name from URI
  std::string URIScheme = "file://";
  if (Filename.find(URIScheme) == 0)
    Filename.erase(0, URIScheme.length());

  const auto *PosObj = LocObj->getObject("position");

  return parseSourcePositionFromJSON(PosObj, Filename);
}

DocComment parseCommentsFromJSON(const Object *CommentsObj) {
  assert(CommentsObj);
  const auto *LinesArray = CommentsObj->getArray("lines");
  DocComment Comments;
  if (LinesArray) {
    for (auto &LineValue : *LinesArray) {
      const auto *LineObj = LineValue.getAsObject();
      auto Text = LineObj->getString("text").value_or("");

      // parse range
      const auto *BeginLocObj = LineObj->getObject("start");
      RecordLocation BeginLoc = parseSourcePositionFromJSON(BeginLocObj);
      const auto *EndLocObj = LineObj->getObject("end");
      RecordLocation EndLoc = parseSourcePositionFromJSON(EndLocObj);
      Comments.push_back(CommentLine(Text, BeginLoc, EndLoc));
    }
  }
  return Comments;
}

AvailabilitySet parseAvailabilitiesFromJSON(const Array *AvailablityArray) {
  if (AvailablityArray) {
    SmallVector<AvailabilityInfo, 4> AList;
    for (auto &AvailablityValue : *AvailablityArray) {
      const auto *AvailablityObj = AvailablityValue.getAsObject();
      auto Domain = AvailablityObj->getString("domain").value_or("");
      auto IntroducedVersion = parseVersionTupleFromJSON(
          AvailablityObj->getObject("introducedVersion"));
      auto ObsoletedVersion = parseVersionTupleFromJSON(
          AvailablityObj->getObject("obsoletedVersion"));
      auto DeprecatedVersion = parseVersionTupleFromJSON(
          AvailablityObj->getObject("deprecatedVersion"));
      AList.emplace_back(AvailabilityInfo(Domain, IntroducedVersion,
                                          DeprecatedVersion, ObsoletedVersion,
                                          false));
    }
    return AvailabilitySet(AList);
  }
  return nullptr;
}

DeclarationFragments parseDeclFragmentsFromJSON(const Array *FragmentsArray) {
  DeclarationFragments Fragments;
  if (FragmentsArray) {
    for (auto &FragmentValue : *FragmentsArray) {
      Object FragmentObj = *(FragmentValue.getAsObject());
      auto Spelling = FragmentObj.getString("spelling").value_or("");
      auto FragmentKind = DeclarationFragments::parseFragmentKindFromString(
          FragmentObj.getString("kind").value_or(""));
      StringRef PreciseIdentifier =
          FragmentObj.getString("preciseIdentifier").value_or("");
      Fragments.append(Spelling, FragmentKind, PreciseIdentifier);
    }
  }
  return Fragments;
}

FunctionSignature parseFunctionSignaturesFromJSON(const Object *SignaturesObj) {
  FunctionSignature ParsedSignatures;
  if (SignaturesObj) {
    // parse return type
    const auto *RT = SignaturesObj->getArray("returns");
    ParsedSignatures.setReturnType(parseDeclFragmentsFromJSON(RT));

    // parse function parameters
    if (const auto *ParamArray = SignaturesObj->getArray("parameters")) {
      for (auto &Param : *ParamArray) {
        auto ParamObj = *(Param.getAsObject());
        auto Name = ParamObj.getString("name").value_or("");
        auto Fragments = parseDeclFragmentsFromJSON(
            ParamObj.getArray("declarationFragments"));
        ParsedSignatures.addParameter(Name, Fragments);
      }
    }
  }
  return ParsedSignatures;
}

std::vector<SymbolGraph::Symbol>
parseSymbolsFromJSON(const Array *SymbolsArray) {
  std::vector<SymbolGraph::Symbol> SymbolsVector;
  if (SymbolsArray) {
    for (const auto &S : *SymbolsArray)
      if (const auto *Symbol = S.getAsObject())
        SymbolsVector.push_back(SymbolGraph::Symbol(*Symbol));
  }
  return SymbolsVector;
}

} // namespace

SymbolGraph::Symbol::Symbol(const Object &SymbolObject)
    : SymbolObj(SymbolObject) {

  AccessLevel = SymbolObj.getString("accessLevel").value_or("unknown");
  Kind = getSymbolKind(*(SymbolObject.getObject("kind")));

  // parse Doc comments
  if (const auto *CommentsArray = SymbolObject.getObject("docComment"))
    Comments = parseCommentsFromJSON(CommentsArray);

  // parse Availabilityinfo
  if (const auto *AvailabilityArray = SymbolObj.getArray("availability"))
    Availabilities = parseAvailabilitiesFromJSON(AvailabilityArray);

  // parse declaration fragments
  if (const auto *FragmentsArray = SymbolObj.getArray("declarationFragments"))
    DeclFragments = parseDeclFragmentsFromJSON(FragmentsArray);

  // parse function signatures if any
  if (const auto *FunctionSignObj = SymbolObj.getObject("functionSignature"))
    FunctionSign = parseFunctionSignaturesFromJSON(FunctionSignObj);

  // parse identifier
  if (const auto *IDObj = SymbolObj.getObject("identifier"))
    USR = IDObj->getString("precise").value_or("");

  // parse Location
  if (const auto *LocObj = SymbolObject.getObject("location"))
    Location = parseRecordLocationFromJSON(LocObj);

  // parse name and subheadings.
  if (const auto *NamesObj = SymbolObj.getObject("names")) {
    Name = NamesObj->getString("title").value_or("");
    if (const auto *SubHObj = NamesObj->getArray("subHeading"))
      SubHeadings = parseDeclFragmentsFromJSON(SubHObj);
  }

  // parse underlying type in case of Typedef
  auto UType = SymbolObject.getString("type");
  if (UType.has_value()) {
    auto UTypeUSR = UType.value();
    // FIXME: this is a hacky way for Underlying type to be
    // serialized into the final graph. Get someway to extract the
    // actual name of the underlying type from USR
    UnderLyingType = SymbolReference(" ", UTypeUSR);
  }
}

SymbolGraph::SymbolGraph(const llvm::StringRef JSON) {
  Expected<llvm::json::Value> SGValue = llvm::json::parse(JSON);
  if (SGValue) {
    assert(SGValue && SGValue->kind() == llvm::json::Value::Object);
    if (const auto *SGObject = SGValue->getAsObject()) {
      SymbolGraphObject = *SGObject;
      if (const auto *MetadataObj = SGObject->getObject("metadata"))
        Metadata = *MetadataObj;
      if (const auto *ModuleObj = SGObject->getObject("module"))
        Module = *ModuleObj;
      if (const auto *RelArray = SGObject->getArray("relationships"))
        Relationships = *RelArray;

      Symbols = parseSymbolsFromJSON(SGObject->getArray("symbols"));
    }
  }
}
