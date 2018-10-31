//===--- SarifDiagnostics.cpp - Sarif Diagnostics for Paths -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the SarifDiagnostics object.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace clang;
using namespace ento;

namespace {
class SarifDiagnostics : public PathDiagnosticConsumer {
  std::string OutputFile;

public:
  SarifDiagnostics(AnalyzerOptions &, const std::string &Output)
      : OutputFile(Output) {}
  ~SarifDiagnostics() override = default;

  void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                            FilesMade *FM) override;

  StringRef getName() const override { return "SarifDiagnostics"; }
  PathGenerationScheme getGenerationScheme() const override { return Minimal; }
  bool supportsLogicalOpControlFlow() const override { return true; }
  bool supportsCrossFileDiagnostics() const override { return true; }
};
} // end anonymous namespace

void ento::createSarifDiagnosticConsumer(AnalyzerOptions &AnalyzerOpts,
                                         PathDiagnosticConsumers &C,
                                         const std::string &Output,
                                         const Preprocessor &) {
  C.push_back(new SarifDiagnostics(AnalyzerOpts, Output));
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
  } else {
    // There is no authority, so end the component and add the root to the URI.
    Ret += Twine("/" + Root).str();
  }

  // Add the rest of the path components, encoding any reserved characters.
  std::for_each(std::next(sys::path::begin(Filename)), sys::path::end(Filename),
                [&Ret](StringRef Component) {
                  // For reasons unknown to me, we may get a backslash with
                  // Windows native paths for the initial backslash following
                  // the drive component, which we need to ignore as a URI path
                  // part.
                  if (Component == "\\")
                    return;

                  // Add the separator between the previous path part and the
                  // one being currently processed.
                  Ret += "/";

                  // URI encode the part.
                  for (char C : Component) {
                    Ret += percentEncodeURICharacter(C);
                  }
                });

  return Ret.str().str();
}

static json::Object createFileLocation(const FileEntry &FE) {
  return json::Object{{"uri", fileNameToURI(getFileName(FE))}};
}

static json::Object createFile(const FileEntry &FE) {
  return json::Object{{"fileLocation", createFileLocation(FE)},
                      {"roles", json::Array{"resultFile"}},
                      {"length", FE.getSize()},
                      {"mimeType", "text/plain"}};
}

static json::Object createFileLocation(const FileEntry &FE,
                                       json::Object &Files) {
  std::string FileURI = fileNameToURI(getFileName(FE));
  if (!Files.get(FileURI))
    Files[FileURI] = createFile(FE);

  return json::Object{{"uri", FileURI}};
}

static json::Object createTextRegion(SourceRange R, const SourceManager &SM) {
  return json::Object{
      {"startLine", SM.getExpansionLineNumber(R.getBegin())},
      {"endLine", SM.getExpansionLineNumber(R.getEnd())},
      {"startColumn", SM.getExpansionColumnNumber(R.getBegin())},
      {"endColumn", SM.getExpansionColumnNumber(R.getEnd())}};
}

static json::Object createPhysicalLocation(SourceRange R, const FileEntry &FE,
                                           const SourceManager &SMgr,
                                           json::Object &Files) {
  return json::Object{{{"fileLocation", createFileLocation(FE, Files)},
                       {"region", createTextRegion(R, SMgr)}}};
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

static json::Object createThreadFlowLocation(int Step, json::Object &&Location,
                                             Importance I) {
  return json::Object{{"step", Step},
                      {"location", std::move(Location)},
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
  case PathDiagnosticPiece::Kind::Call:
  case PathDiagnosticPiece::Kind::Macro:
  case PathDiagnosticPiece::Kind::Note:
    // FIXME: What should be reported here?
    break;
  case PathDiagnosticPiece::Kind::Event:
    return Piece.getTagStr() == "ConditionBRVisitor" ? Importance::Important
                                                     : Importance::Essential;
  case PathDiagnosticPiece::Kind::ControlFlow:
    return Importance::Unimportant;
  }
  return Importance::Unimportant;
}

static json::Object createThreadFlow(const PathPieces &Pieces,
                                     json::Object &Files) {
  const SourceManager &SMgr = Pieces.front()->getLocation().getManager();
  int Step = 1;
  json::Array Locations;
  for (const auto &Piece : Pieces) {
    const PathDiagnosticLocation &P = Piece->getLocation();
    Locations.push_back(createThreadFlowLocation(
        Step++,
        createLocation(createPhysicalLocation(P.asRange(),
                                              *P.asLocation().getFileEntry(),
                                              SMgr, Files),
                       Piece->getString()),
        calculateImportance(*Piece)));
  }
  return json::Object{{"locations", std::move(Locations)}};
}

static json::Object createCodeFlow(const PathPieces &Pieces,
                                   json::Object &Files) {
  return json::Object{
      {"threadFlows", json::Array{createThreadFlow(Pieces, Files)}}};
}

static json::Object createTool() {
  return json::Object{{"name", "clang"},
                      {"fullName", "clang static analyzer"},
                      {"language", "en-US"},
                      {"version", getClangFullVersion()}};
}

static json::Object createResult(const PathDiagnostic &Diag,
                                 json::Object &Files) {
  const PathPieces &Path = Diag.path.flatten(false);
  const SourceManager &SMgr = Path.front()->getLocation().getManager();

  return json::Object{
      {"message", createMessage(Diag.getVerboseDescription())},
      {"codeFlows", json::Array{createCodeFlow(Path, Files)}},
      {"locations",
       json::Array{createLocation(createPhysicalLocation(
           Diag.getLocation().asRange(),
           *Diag.getLocation().asLocation().getFileEntry(), SMgr, Files))}},
      {"ruleId", Diag.getCheckName()}};
}

static json::Object createRun(std::vector<const PathDiagnostic *> &Diags) {
  json::Array Results;
  json::Object Files;

  llvm::for_each(Diags, [&](const PathDiagnostic *D) {
    Results.push_back(createResult(*D, Files));
  });

  return json::Object{{"tool", createTool()},
                      {"results", std::move(Results)},
                      {"files", std::move(Files)}};
}

void SarifDiagnostics::FlushDiagnosticsImpl(
    std::vector<const PathDiagnostic *> &Diags, FilesMade *) {
  // We currently overwrite the file if it already exists. However, it may be
  // useful to add a feature someday that allows the user to append a run to an
  // existing SARIF file. One danger from that approach is that the size of the
  // file can become large very quickly, so decoding into JSON to append a run
  // may be an expensive operation.
  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFile, EC, llvm::sys::fs::F_Text);
  if (EC) {
    llvm::errs() << "warning: could not create file: " << EC.message() << '\n';
    return;
  }
  json::Object Sarif{{"$schema", "http://json.schemastore.org/sarif-2.0.0"},
                     {"version", "2.0.0-beta.2018-09-26"},
                     {"runs", json::Array{createRun(Diags)}}};
  OS << llvm::formatv("{0:2}", json::Value(std::move(Sarif)));
}
