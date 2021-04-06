//===--- SarifDiagnostics.cpp - Sarif Diagnostics for Paths -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SarifDiagnostics object.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/MacroExpansionContext.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace clang;
using namespace ento;

namespace {
class SarifDiagnostics : public PathDiagnosticConsumer {
  std::string OutputFile;
  const LangOptions &LO;

public:
  SarifDiagnostics(const std::string &Output, const LangOptions &LO)
      : OutputFile(Output), LO(LO) {}
  ~SarifDiagnostics() override = default;

  void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                            FilesMade *FM) override;

  StringRef getName() const override { return "SarifDiagnostics"; }
  PathGenerationScheme getGenerationScheme() const override { return Minimal; }
  bool supportsLogicalOpControlFlow() const override { return true; }
  bool supportsCrossFileDiagnostics() const override { return true; }
};
} // end anonymous namespace

void ento::createSarifDiagnosticConsumer(
    PathDiagnosticConsumerOptions DiagOpts, PathDiagnosticConsumers &C,
    const std::string &Output, const Preprocessor &PP,
    const cross_tu::CrossTranslationUnitContext &CTU,
    const MacroExpansionContext &MacroExpansions) {

  // TODO: Emit an error here.
  if (Output.empty())
    return;

  C.push_back(new SarifDiagnostics(Output, PP.getLangOpts()));
  createTextMinimalPathDiagnosticConsumer(std::move(DiagOpts), C, Output, PP,
                                          CTU, MacroExpansions);
}

static StringRef getFileName(const FileEntry &FE) {
  StringRef Filename = FE.tryGetRealPathName();
  if (Filename.empty())
    Filename = FE.getName();
  return Filename;
}

static std::string percentEncodeURICharacter(char C) {
  // RFC 3986 claims alpha, numeric, and this handful of
  // characters are not reserved for the path component and
  // should be written out directly. Otherwise, percent
  // encode the character and write that out instead of the
  // reserved character.
  if (llvm::isAlnum(C) ||
      StringRef::npos != StringRef("-._~:@!$&'()*+,;=").find(C))
    return std::string(&C, 1);
  return "%" + llvm::toHex(StringRef(&C, 1));
}

static std::string fileNameToURI(StringRef Filename) {
  llvm::SmallString<32> Ret = StringRef("file://");

  // Get the root name to see if it has a URI authority.
  StringRef Root = sys::path::root_name(Filename);
  if (Root.startswith("//")) {
    // There is an authority, so add it to the URI.
    Ret += Root.drop_front(2).str();
  } else if (!Root.empty()) {
    // There is no authority, so end the component and add the root to the URI.
    Ret += Twine("/" + Root).str();
  }

  auto Iter = sys::path::begin(Filename), End = sys::path::end(Filename);
  assert(Iter != End && "Expected there to be a non-root path component.");
  // Add the rest of the path components, encoding any reserved characters;
  // we skip past the first path component, as it was handled it above.
  std::for_each(++Iter, End, [&Ret](StringRef Component) {
    // For reasons unknown to me, we may get a backslash with Windows native
    // paths for the initial backslash following the drive component, which
    // we need to ignore as a URI path part.
    if (Component == "\\")
      return;

    // Add the separator between the previous path part and the one being
    // currently processed.
    Ret += "/";

    // URI encode the part.
    for (char C : Component) {
      Ret += percentEncodeURICharacter(C);
    }
  });

  return std::string(Ret);
}

static json::Object createArtifactLocation(const FileEntry &FE) {
  return json::Object{{"uri", fileNameToURI(getFileName(FE))}};
}

static json::Object createArtifact(const FileEntry &FE) {
  return json::Object{{"location", createArtifactLocation(FE)},
                      {"roles", json::Array{"resultFile"}},
                      {"length", FE.getSize()},
                      {"mimeType", "text/plain"}};
}

static json::Object createArtifactLocation(const FileEntry &FE,
                                           json::Array &Artifacts) {
  std::string FileURI = fileNameToURI(getFileName(FE));

  // See if the Artifacts array contains this URI already. If it does not,
  // create a new artifact object to add to the array.
  auto I = llvm::find_if(Artifacts, [&](const json::Value &File) {
    if (const json::Object *Obj = File.getAsObject()) {
      if (const json::Object *FileLoc = Obj->getObject("location")) {
        Optional<StringRef> URI = FileLoc->getString("uri");
        return URI && URI->equals(FileURI);
      }
    }
    return false;
  });

  // Calculate the index within the artifact array so it can be stored in
  // the JSON object.
  auto Index = static_cast<unsigned>(std::distance(Artifacts.begin(), I));
  if (I == Artifacts.end())
    Artifacts.push_back(createArtifact(FE));

  return json::Object{{"uri", FileURI}, {"index", Index}};
}

static unsigned int adjustColumnPos(const SourceManager &SM, SourceLocation Loc,
                                    unsigned int TokenLen = 0) {
  assert(!Loc.isInvalid() && "invalid Loc when adjusting column position");

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedExpansionLoc(Loc);
  assert(LocInfo.second > SM.getExpansionColumnNumber(Loc) &&
         "position in file is before column number?");

  Optional<MemoryBufferRef> Buf = SM.getBufferOrNone(LocInfo.first);
  assert(Buf && "got an invalid buffer for the location's file");
  assert(Buf->getBufferSize() >= (LocInfo.second + TokenLen) &&
         "token extends past end of buffer?");

  // Adjust the offset to be the start of the line, since we'll be counting
  // Unicode characters from there until our column offset.
  unsigned int Off = LocInfo.second - (SM.getExpansionColumnNumber(Loc) - 1);
  unsigned int Ret = 1;
  while (Off < (LocInfo.second + TokenLen)) {
    Off += getNumBytesForUTF8(Buf->getBuffer()[Off]);
    Ret++;
  }

  return Ret;
}

static json::Object createTextRegion(const LangOptions &LO, SourceRange R,
                                     const SourceManager &SM) {
  json::Object Region{
      {"startLine", SM.getExpansionLineNumber(R.getBegin())},
      {"startColumn", adjustColumnPos(SM, R.getBegin())},
  };
  if (R.getBegin() == R.getEnd()) {
    Region["endColumn"] = adjustColumnPos(SM, R.getBegin());
  } else {
    Region["endLine"] = SM.getExpansionLineNumber(R.getEnd());
    Region["endColumn"] = adjustColumnPos(
        SM, R.getEnd(),
        Lexer::MeasureTokenLength(R.getEnd(), SM, LO));
  }
  return Region;
}

static json::Object createPhysicalLocation(const LangOptions &LO,
                                           SourceRange R, const FileEntry &FE,
                                           const SourceManager &SMgr,
                                           json::Array &Artifacts) {
  return json::Object{
      {{"artifactLocation", createArtifactLocation(FE, Artifacts)},
       {"region", createTextRegion(LO, R, SMgr)}}};
}

enum class Importance { Important, Essential, Unimportant };

static StringRef importanceToStr(Importance I) {
  switch (I) {
  case Importance::Important:
    return "important";
  case Importance::Essential:
    return "essential";
  case Importance::Unimportant:
    return "unimportant";
  }
  llvm_unreachable("Fully covered switch is not so fully covered");
}

static json::Object createThreadFlowLocation(json::Object &&Location,
                                             Importance I) {
  return json::Object{{"location", std::move(Location)},
                      {"importance", importanceToStr(I)}};
}

static json::Object createMessage(StringRef Text) {
  return json::Object{{"text", Text.str()}};
}

static json::Object createLocation(json::Object &&PhysicalLocation,
                                   StringRef Message = "") {
  json::Object Ret{{"physicalLocation", std::move(PhysicalLocation)}};
  if (!Message.empty())
    Ret.insert({"message", createMessage(Message)});
  return Ret;
}

static Importance calculateImportance(const PathDiagnosticPiece &Piece) {
  switch (Piece.getKind()) {
  case PathDiagnosticPiece::Call:
  case PathDiagnosticPiece::Macro:
  case PathDiagnosticPiece::Note:
  case PathDiagnosticPiece::PopUp:
    // FIXME: What should be reported here?
    break;
  case PathDiagnosticPiece::Event:
    return Piece.getTagStr() == "ConditionBRVisitor" ? Importance::Important
                                                     : Importance::Essential;
  case PathDiagnosticPiece::ControlFlow:
    return Importance::Unimportant;
  }
  return Importance::Unimportant;
}

static json::Object createThreadFlow(const LangOptions &LO,
                                     const PathPieces &Pieces,
                                     json::Array &Artifacts) {
  const SourceManager &SMgr = Pieces.front()->getLocation().getManager();
  json::Array Locations;
  for (const auto &Piece : Pieces) {
    const PathDiagnosticLocation &P = Piece->getLocation();
    Locations.push_back(createThreadFlowLocation(
        createLocation(createPhysicalLocation(
                           LO, P.asRange(),
                           *P.asLocation().getExpansionLoc().getFileEntry(),
                           SMgr, Artifacts),
                       Piece->getString()),
        calculateImportance(*Piece)));
  }
  return json::Object{{"locations", std::move(Locations)}};
}

static json::Object createCodeFlow(const LangOptions &LO,
                                   const PathPieces &Pieces,
                                   json::Array &Artifacts) {
  return json::Object{
      {"threadFlows", json::Array{createThreadFlow(LO, Pieces, Artifacts)}}};
}

static json::Object createResult(const LangOptions &LO,
                                 const PathDiagnostic &Diag,
                                 json::Array &Artifacts,
                                 const StringMap<unsigned> &RuleMapping) {
  const PathPieces &Path = Diag.path.flatten(false);
  const SourceManager &SMgr = Path.front()->getLocation().getManager();

  auto Iter = RuleMapping.find(Diag.getCheckerName());
  assert(Iter != RuleMapping.end() && "Rule ID is not in the array index map?");

  return json::Object{
      {"message", createMessage(Diag.getVerboseDescription())},
      {"codeFlows", json::Array{createCodeFlow(LO, Path, Artifacts)}},
      {"locations",
       json::Array{createLocation(createPhysicalLocation(
           LO, Diag.getLocation().asRange(),
           *Diag.getLocation().asLocation().getExpansionLoc().getFileEntry(),
           SMgr, Artifacts))}},
      {"ruleIndex", Iter->getValue()},
      {"ruleId", Diag.getCheckerName()}};
}

static StringRef getRuleDescription(StringRef CheckName) {
  return llvm::StringSwitch<StringRef>(CheckName)
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, HELPTEXT, DOC_URI, IS_HIDDEN)                 \
  .Case(FULLNAME, HELPTEXT)
#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS
      ;
}

static StringRef getRuleHelpURIStr(StringRef CheckName) {
  return llvm::StringSwitch<StringRef>(CheckName)
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, HELPTEXT, DOC_URI, IS_HIDDEN)                 \
  .Case(FULLNAME, DOC_URI)
#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS
      ;
}

static json::Object createRule(const PathDiagnostic &Diag) {
  StringRef CheckName = Diag.getCheckerName();
  json::Object Ret{
      {"fullDescription", createMessage(getRuleDescription(CheckName))},
      {"name", CheckName},
      {"id", CheckName}};

  std::string RuleURI = std::string(getRuleHelpURIStr(CheckName));
  if (!RuleURI.empty())
    Ret["helpUri"] = RuleURI;

  return Ret;
}

static json::Array createRules(std::vector<const PathDiagnostic *> &Diags,
                               StringMap<unsigned> &RuleMapping) {
  json::Array Rules;
  llvm::StringSet<> Seen;

  llvm::for_each(Diags, [&](const PathDiagnostic *D) {
    StringRef RuleID = D->getCheckerName();
    std::pair<llvm::StringSet<>::iterator, bool> P = Seen.insert(RuleID);
    if (P.second) {
      RuleMapping[RuleID] = Rules.size(); // Maps RuleID to an Array Index.
      Rules.push_back(createRule(*D));
    }
  });

  return Rules;
}

static json::Object createTool(std::vector<const PathDiagnostic *> &Diags,
                               StringMap<unsigned> &RuleMapping) {
  return json::Object{
      {"driver", json::Object{{"name", "clang"},
                              {"fullName", "clang static analyzer"},
                              {"language", "en-US"},
                              {"version", getClangFullVersion()},
                              {"rules", createRules(Diags, RuleMapping)}}}};
}

static json::Object createRun(const LangOptions &LO,
                              std::vector<const PathDiagnostic *> &Diags) {
  json::Array Results, Artifacts;
  StringMap<unsigned> RuleMapping;
  json::Object Tool = createTool(Diags, RuleMapping);
  
  llvm::for_each(Diags, [&](const PathDiagnostic *D) {
    Results.push_back(createResult(LO, *D, Artifacts, RuleMapping));
  });

  return json::Object{{"tool", std::move(Tool)},
                      {"results", std::move(Results)},
                      {"artifacts", std::move(Artifacts)},
                      {"columnKind", "unicodeCodePoints"}};
}

void SarifDiagnostics::FlushDiagnosticsImpl(
    std::vector<const PathDiagnostic *> &Diags, FilesMade *) {
  // We currently overwrite the file if it already exists. However, it may be
  // useful to add a feature someday that allows the user to append a run to an
  // existing SARIF file. One danger from that approach is that the size of the
  // file can become large very quickly, so decoding into JSON to append a run
  // may be an expensive operation.
  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFile, EC, llvm::sys::fs::OF_TextWithCRLF);
  if (EC) {
    llvm::errs() << "warning: could not create file: " << EC.message() << '\n';
    return;
  }
  json::Object Sarif{
      {"$schema",
       "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"},
      {"version", "2.1.0"},
      {"runs", json::Array{createRun(LO, Diags)}}};
  OS << llvm::formatv("{0:2}\n", json::Value(std::move(Sarif)));
}
