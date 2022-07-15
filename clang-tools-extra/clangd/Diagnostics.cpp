//===--- Diagnostics.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//

#include "Diagnostics.h"
#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "Compiler.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "clang/Basic/AllDiagnostics.h" // IWYU pragma: keep
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace clang {
namespace clangd {
namespace {

const char *getDiagnosticCode(unsigned ID) {
  switch (ID) {
#define DIAG(ENUM, CLASS, DEFAULT_MAPPING, DESC, GROPU, SFINAE, NOWERROR,      \
             SHOWINSYSHEADER, SHOWINSYSMACRO, DEFERRABLE, CATEGORY)            \
  case clang::diag::ENUM:                                                      \
    return #ENUM;
#include "clang/Basic/DiagnosticASTKinds.inc"
#include "clang/Basic/DiagnosticAnalysisKinds.inc"
#include "clang/Basic/DiagnosticCommentKinds.inc"
#include "clang/Basic/DiagnosticCommonKinds.inc"
#include "clang/Basic/DiagnosticDriverKinds.inc"
#include "clang/Basic/DiagnosticFrontendKinds.inc"
#include "clang/Basic/DiagnosticLexKinds.inc"
#include "clang/Basic/DiagnosticParseKinds.inc"
#include "clang/Basic/DiagnosticRefactoringKinds.inc"
#include "clang/Basic/DiagnosticSemaKinds.inc"
#include "clang/Basic/DiagnosticSerializationKinds.inc"
#undef DIAG
  default:
    return nullptr;
  }
}

bool mentionsMainFile(const Diag &D) {
  if (D.InsideMainFile)
    return true;
  // Fixes are always in the main file.
  if (!D.Fixes.empty())
    return true;
  for (auto &N : D.Notes) {
    if (N.InsideMainFile)
      return true;
  }
  return false;
}

bool isExcluded(unsigned DiagID) {
  // clang will always fail parsing MS ASM, we don't link in desc + asm parser.
  if (DiagID == clang::diag::err_msasm_unable_to_create_target ||
      DiagID == clang::diag::err_msasm_unsupported_arch)
    return true;
  return false;
}

// Checks whether a location is within a half-open range.
// Note that clang also uses closed source ranges, which this can't handle!
bool locationInRange(SourceLocation L, CharSourceRange R,
                     const SourceManager &M) {
  assert(R.isCharRange());
  if (!R.isValid() || M.getFileID(R.getBegin()) != M.getFileID(R.getEnd()) ||
      M.getFileID(R.getBegin()) != M.getFileID(L))
    return false;
  return L != R.getEnd() && M.isPointWithin(L, R.getBegin(), R.getEnd());
}

// Clang diags have a location (shown as ^) and 0 or more ranges (~~~~).
// LSP needs a single range.
Range diagnosticRange(const clang::Diagnostic &D, const LangOptions &L) {
  auto &M = D.getSourceManager();
  auto Loc = M.getFileLoc(D.getLocation());
  for (const auto &CR : D.getRanges()) {
    auto R = Lexer::makeFileCharRange(CR, M, L);
    if (locationInRange(Loc, R, M))
      return halfOpenToRange(M, R);
  }
  // The range may be given as a fixit hint instead.
  for (const auto &F : D.getFixItHints()) {
    auto R = Lexer::makeFileCharRange(F.RemoveRange, M, L);
    if (locationInRange(Loc, R, M))
      return halfOpenToRange(M, R);
  }
  // If the token at the location is not a comment, we use the token.
  // If we can't get the token at the location, fall back to using the location
  auto R = CharSourceRange::getCharRange(Loc);
  Token Tok;
  if (!Lexer::getRawToken(Loc, Tok, M, L, true) && Tok.isNot(tok::comment)) {
    R = CharSourceRange::getTokenRange(Tok.getLocation(), Tok.getEndLoc());
  }
  return halfOpenToRange(M, R);
}

// Try to find a location in the main-file to report the diagnostic D.
// Returns a description like "in included file", or nullptr on failure.
const char *getMainFileRange(const Diag &D, const SourceManager &SM,
                             SourceLocation DiagLoc, Range &R) {
  // Look for a note in the main file indicating template instantiation.
  for (const auto &N : D.Notes) {
    if (N.InsideMainFile) {
      switch (N.ID) {
      case diag::note_template_class_instantiation_was_here:
      case diag::note_template_class_explicit_specialization_was_here:
      case diag::note_template_class_instantiation_here:
      case diag::note_template_member_class_here:
      case diag::note_template_member_function_here:
      case diag::note_function_template_spec_here:
      case diag::note_template_static_data_member_def_here:
      case diag::note_template_variable_def_here:
      case diag::note_template_enum_def_here:
      case diag::note_template_nsdmi_here:
      case diag::note_template_type_alias_instantiation_here:
      case diag::note_template_exception_spec_instantiation_here:
      case diag::note_template_requirement_instantiation_here:
      case diag::note_evaluating_exception_spec_here:
      case diag::note_default_arg_instantiation_here:
      case diag::note_default_function_arg_instantiation_here:
      case diag::note_explicit_template_arg_substitution_here:
      case diag::note_function_template_deduction_instantiation_here:
      case diag::note_deduced_template_arg_substitution_here:
      case diag::note_prior_template_arg_substitution:
      case diag::note_template_default_arg_checking:
      case diag::note_concept_specialization_here:
      case diag::note_nested_requirement_here:
      case diag::note_checking_constraints_for_template_id_here:
      case diag::note_checking_constraints_for_var_spec_id_here:
      case diag::note_checking_constraints_for_class_spec_id_here:
      case diag::note_checking_constraints_for_function_here:
      case diag::note_constraint_substitution_here:
      case diag::note_constraint_normalization_here:
      case diag::note_parameter_mapping_substitution_here:
        R = N.Range;
        return "in template";
      default:
        break;
      }
    }
  }
  // Look for where the file with the error was #included.
  auto GetIncludeLoc = [&SM](SourceLocation SLoc) {
    return SM.getIncludeLoc(SM.getFileID(SLoc));
  };
  for (auto IncludeLocation = GetIncludeLoc(SM.getExpansionLoc(DiagLoc));
       IncludeLocation.isValid();
       IncludeLocation = GetIncludeLoc(IncludeLocation)) {
    if (clangd::isInsideMainFile(IncludeLocation, SM)) {
      R.start = sourceLocToPosition(SM, IncludeLocation);
      R.end = sourceLocToPosition(
          SM,
          Lexer::getLocForEndOfToken(IncludeLocation, 0, SM, LangOptions()));
      return "in included file";
    }
  }
  return nullptr;
}

// Place the diagnostic the main file, rather than the header, if possible:
//   - for errors in included files, use the #include location
//   - for errors in template instantiation, use the instantiation location
// In both cases, add the original header location as a note.
bool tryMoveToMainFile(Diag &D, FullSourceLoc DiagLoc) {
  const SourceManager &SM = DiagLoc.getManager();
  DiagLoc = DiagLoc.getExpansionLoc();
  Range R;
  const char *Prefix = getMainFileRange(D, SM, DiagLoc, R);
  if (!Prefix)
    return false;

  // Add a note that will point to real diagnostic.
  auto FE = *SM.getFileEntryRefForID(SM.getFileID(DiagLoc));
  D.Notes.emplace(D.Notes.begin());
  Note &N = D.Notes.front();
  N.AbsFile = std::string(FE.getFileEntry().tryGetRealPathName());
  N.File = std::string(FE.getName());
  N.Message = "error occurred here";
  N.Range = D.Range;

  // Update diag to point at include inside main file.
  D.File = SM.getFileEntryRefForID(SM.getMainFileID())->getName().str();
  D.Range = std::move(R);
  D.InsideMainFile = true;
  // Update message to mention original file.
  D.Message = llvm::formatv("{0}: {1}", Prefix, D.Message);
  return true;
}

bool isInsideMainFile(const clang::Diagnostic &D) {
  if (!D.hasSourceManager())
    return false;

  return clangd::isInsideMainFile(D.getLocation(), D.getSourceManager());
}

bool isNote(DiagnosticsEngine::Level L) {
  return L == DiagnosticsEngine::Note || L == DiagnosticsEngine::Remark;
}

llvm::StringRef diagLeveltoString(DiagnosticsEngine::Level Lvl) {
  switch (Lvl) {
  case DiagnosticsEngine::Ignored:
    return "ignored";
  case DiagnosticsEngine::Note:
    return "note";
  case DiagnosticsEngine::Remark:
    return "remark";
  case DiagnosticsEngine::Warning:
    return "warning";
  case DiagnosticsEngine::Error:
    return "error";
  case DiagnosticsEngine::Fatal:
    return "fatal error";
  }
  llvm_unreachable("unhandled DiagnosticsEngine::Level");
}

/// Prints a single diagnostic in a clang-like manner, the output includes
/// location, severity and error message. An example of the output message is:
///
///     main.cpp:12:23: error: undeclared identifier
///
/// For main file we only print the basename and for all other files we print
/// the filename on a separate line to provide a slightly more readable output
/// in the editors:
///
///     dir1/dir2/dir3/../../dir4/header.h:12:23
///     error: undeclared identifier
void printDiag(llvm::raw_string_ostream &OS, const DiagBase &D) {
  if (D.InsideMainFile) {
    // Paths to main files are often taken from compile_command.json, where they
    // are typically absolute. To reduce noise we print only basename for them,
    // it should not be confusing and saves space.
    OS << llvm::sys::path::filename(D.File) << ":";
  } else {
    OS << D.File << ":";
  }
  // Note +1 to line and character. clangd::Range is zero-based, but when
  // printing for users we want one-based indexes.
  auto Pos = D.Range.start;
  OS << (Pos.line + 1) << ":" << (Pos.character + 1) << ":";
  // The non-main-file paths are often too long, putting them on a separate
  // line improves readability.
  if (D.InsideMainFile)
    OS << " ";
  else
    OS << "\n";
  OS << diagLeveltoString(D.Severity) << ": " << D.Message;
}

/// Capitalizes the first word in the diagnostic's message.
std::string capitalize(std::string Message) {
  if (!Message.empty())
    Message[0] = llvm::toUpper(Message[0]);
  return Message;
}

/// Returns a message sent to LSP for the main diagnostic in \p D.
/// This message may include notes, if they're not emitted in some other way.
/// Example output:
///
///     no matching function for call to 'foo'
///
///     main.cpp:3:5: note: candidate function not viable: requires 2 arguments
///
///     dir1/dir2/dir3/../../dir4/header.h:12:23
///     note: candidate function not viable: requires 3 arguments
std::string mainMessage(const Diag &D, const ClangdDiagnosticOptions &Opts) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << D.Message;
  if (Opts.DisplayFixesCount && !D.Fixes.empty())
    OS << " (" << (D.Fixes.size() > 1 ? "fixes" : "fix") << " available)";
  // If notes aren't emitted as structured info, add them to the message.
  if (!Opts.EmitRelatedLocations)
    for (auto &Note : D.Notes) {
      OS << "\n\n";
      printDiag(OS, Note);
    }
  OS.flush();
  return capitalize(std::move(Result));
}

/// Returns a message sent to LSP for the note of the main diagnostic.
std::string noteMessage(const Diag &Main, const DiagBase &Note,
                        const ClangdDiagnosticOptions &Opts) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << Note.Message;
  // If the client doesn't support structured links between the note and the
  // original diagnostic, then emit the main diagnostic to give context.
  if (!Opts.EmitRelatedLocations) {
    OS << "\n\n";
    printDiag(OS, Main);
  }
  OS.flush();
  return capitalize(std::move(Result));
}
} // namespace

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const DiagBase &D) {
  OS << "[";
  if (!D.InsideMainFile)
    OS << D.File << ":";
  OS << D.Range.start << "-" << D.Range.end << "] ";

  return OS << D.Message;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Fix &F) {
  OS << F.Message << " {";
  const char *Sep = "";
  for (const auto &Edit : F.Edits) {
    OS << Sep << Edit;
    Sep = ", ";
  }
  return OS << "}";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Diag &D) {
  OS << static_cast<const DiagBase &>(D);
  if (!D.Notes.empty()) {
    OS << ", notes: {";
    const char *Sep = "";
    for (auto &Note : D.Notes) {
      OS << Sep << Note;
      Sep = ", ";
    }
    OS << "}";
  }
  if (!D.Fixes.empty()) {
    OS << ", fixes: {";
    const char *Sep = "";
    for (auto &Fix : D.Fixes) {
      OS << Sep << Fix;
      Sep = ", ";
    }
    OS << "}";
  }
  return OS;
}

CodeAction toCodeAction(const Fix &F, const URIForFile &File) {
  CodeAction Action;
  Action.title = F.Message;
  Action.kind = std::string(CodeAction::QUICKFIX_KIND);
  Action.edit.emplace();
  Action.edit->changes[File.uri()] = {F.Edits.begin(), F.Edits.end()};
  return Action;
}

Diag toDiag(const llvm::SMDiagnostic &D, Diag::DiagSource Source) {
  Diag Result;
  Result.Message = D.getMessage().str();
  switch (D.getKind()) {
  case llvm::SourceMgr::DK_Error:
    Result.Severity = DiagnosticsEngine::Error;
    break;
  case llvm::SourceMgr::DK_Warning:
    Result.Severity = DiagnosticsEngine::Warning;
    break;
  default:
    break;
  }
  Result.Source = Source;
  Result.AbsFile = D.getFilename().str();
  Result.InsideMainFile = D.getSourceMgr()->FindBufferContainingLoc(
                              D.getLoc()) == D.getSourceMgr()->getMainFileID();
  if (D.getRanges().empty())
    Result.Range = {{D.getLineNo() - 1, D.getColumnNo()},
                    {D.getLineNo() - 1, D.getColumnNo()}};
  else
    Result.Range = {{D.getLineNo() - 1, (int)D.getRanges().front().first},
                    {D.getLineNo() - 1, (int)D.getRanges().front().second}};
  return Result;
}

void toLSPDiags(
    const Diag &D, const URIForFile &File, const ClangdDiagnosticOptions &Opts,
    llvm::function_ref<void(clangd::Diagnostic, llvm::ArrayRef<Fix>)> OutFn) {
  clangd::Diagnostic Main;
  Main.severity = getSeverity(D.Severity);

  // Main diagnostic should always refer to a range inside main file. If a
  // diagnostic made it so for, it means either itself or one of its notes is
  // inside main file. It's also possible that there's a fix in the main file,
  // but we preserve fixes iff primary diagnostic is in the main file.
  if (D.InsideMainFile) {
    Main.range = D.Range;
  } else {
    auto It =
        llvm::find_if(D.Notes, [](const Note &N) { return N.InsideMainFile; });
    assert(It != D.Notes.end() &&
           "neither the main diagnostic nor notes are inside main file");
    Main.range = It->Range;
  }

  Main.code = D.Name;
  if (auto URI = getDiagnosticDocURI(D.Source, D.ID, D.Name)) {
    Main.codeDescription.emplace();
    Main.codeDescription->href = std::move(*URI);
  }
  switch (D.Source) {
  case Diag::Clang:
    Main.source = "clang";
    break;
  case Diag::ClangTidy:
    Main.source = "clang-tidy";
    break;
  case Diag::Clangd:
    Main.source = "clangd";
    break;
  case Diag::ClangdConfig:
    Main.source = "clangd-config";
    break;
  case Diag::Unknown:
    break;
  }
  if (Opts.EmbedFixesInDiagnostics) {
    Main.codeActions.emplace();
    for (const auto &Fix : D.Fixes)
      Main.codeActions->push_back(toCodeAction(Fix, File));
    if (Main.codeActions->size() == 1)
      Main.codeActions->front().isPreferred = true;
  }
  if (Opts.SendDiagnosticCategory && !D.Category.empty())
    Main.category = D.Category;

  Main.message = mainMessage(D, Opts);
  if (Opts.EmitRelatedLocations) {
    Main.relatedInformation.emplace();
    for (auto &Note : D.Notes) {
      if (!Note.AbsFile) {
        vlog("Dropping note from unknown file: {0}", Note);
        continue;
      }
      DiagnosticRelatedInformation RelInfo;
      RelInfo.location.range = Note.Range;
      RelInfo.location.uri =
          URIForFile::canonicalize(*Note.AbsFile, File.file());
      RelInfo.message = noteMessage(D, Note, Opts);
      Main.relatedInformation->push_back(std::move(RelInfo));
    }
  }
  OutFn(std::move(Main), D.Fixes);

  // If we didn't emit the notes as relatedLocations, emit separate diagnostics
  // so the user can find the locations easily.
  if (!Opts.EmitRelatedLocations)
    for (auto &Note : D.Notes) {
      if (!Note.InsideMainFile)
        continue;
      clangd::Diagnostic Res;
      Res.severity = getSeverity(Note.Severity);
      Res.range = Note.Range;
      Res.message = noteMessage(D, Note, Opts);
      OutFn(std::move(Res), llvm::ArrayRef<Fix>());
    }

  // FIXME: Get rid of the copies here by taking in a mutable clangd::Diag.
  for (auto &Entry : D.OpaqueData)
    Main.data.insert({Entry.first, Entry.second});
}

int getSeverity(DiagnosticsEngine::Level L) {
  switch (L) {
  case DiagnosticsEngine::Remark:
    return 4;
  case DiagnosticsEngine::Note:
    return 3;
  case DiagnosticsEngine::Warning:
    return 2;
  case DiagnosticsEngine::Fatal:
  case DiagnosticsEngine::Error:
    return 1;
  case DiagnosticsEngine::Ignored:
    return 0;
  }
  llvm_unreachable("Unknown diagnostic level!");
}

std::vector<Diag> StoreDiags::take(const clang::tidy::ClangTidyContext *Tidy) {
  // Do not forget to emit a pending diagnostic if there is one.
  flushLastDiag();

  // Fill in name/source now that we have all the context needed to map them.
  for (auto &Diag : Output) {
    if (const char *ClangDiag = getDiagnosticCode(Diag.ID)) {
      // Warnings controlled by -Wfoo are better recognized by that name.
      StringRef Warning = DiagnosticIDs::getWarningOptionForDiag(Diag.ID);
      if (!Warning.empty()) {
        Diag.Name = ("-W" + Warning).str();
      } else {
        StringRef Name(ClangDiag);
        // Almost always an error, with a name like err_enum_class_reference.
        // Drop the err_ prefix for brevity.
        Name.consume_front("err_");
        Diag.Name = std::string(Name);
      }
      Diag.Source = Diag::Clang;
      continue;
    }
    if (Tidy != nullptr) {
      std::string TidyDiag = Tidy->getCheckName(Diag.ID);
      if (!TidyDiag.empty()) {
        Diag.Name = std::move(TidyDiag);
        Diag.Source = Diag::ClangTidy;
        // clang-tidy bakes the name into diagnostic messages. Strip it out.
        // It would be much nicer to make clang-tidy not do this.
        auto CleanMessage = [&](std::string &Msg) {
          StringRef Rest(Msg);
          if (Rest.consume_back("]") && Rest.consume_back(Diag.Name) &&
              Rest.consume_back(" ["))
            Msg.resize(Rest.size());
        };
        CleanMessage(Diag.Message);
        for (auto &Note : Diag.Notes)
          CleanMessage(Note.Message);
        for (auto &Fix : Diag.Fixes)
          CleanMessage(Fix.Message);
        continue;
      }
    }
  }
  // Deduplicate clang-tidy diagnostics -- some clang-tidy checks may emit
  // duplicated messages due to various reasons (e.g. the check doesn't handle
  // template instantiations well; clang-tidy alias checks).
  std::set<std::pair<Range, std::string>> SeenDiags;
  llvm::erase_if(Output, [&](const Diag& D) {
    return !SeenDiags.emplace(D.Range, D.Message).second;
  });
  return std::move(Output);
}

void StoreDiags::BeginSourceFile(const LangOptions &Opts,
                                 const Preprocessor *PP) {
  LangOpts = Opts;
  if (PP) {
    OrigSrcMgr = &PP->getSourceManager();
  }
}

void StoreDiags::EndSourceFile() {
  flushLastDiag();
  LangOpts = None;
  OrigSrcMgr = nullptr;
}

/// Sanitizes a piece for presenting it in a synthesized fix message. Ensures
/// the result is not too large and does not contain newlines.
static void writeCodeToFixMessage(llvm::raw_ostream &OS, llvm::StringRef Code) {
  constexpr unsigned MaxLen = 50;
  if (Code == "\n") {
    OS << "\\n";
    return;
  }
  // Only show the first line if there are many.
  llvm::StringRef R = Code.split('\n').first;
  // Shorten the message if it's too long.
  R = R.take_front(MaxLen);

  OS << R;
  if (R.size() != Code.size())
    OS << "…";
}

/// Fills \p D with all information, except the location-related bits.
/// Also note that ID and Name are not part of clangd::DiagBase and should be
/// set elsewhere.
static void fillNonLocationData(DiagnosticsEngine::Level DiagLevel,
                                const clang::Diagnostic &Info,
                                clangd::DiagBase &D) {
  llvm::SmallString<64> Message;
  Info.FormatDiagnostic(Message);

  D.Message = std::string(Message.str());
  D.Severity = DiagLevel;
  D.Category = DiagnosticIDs::getCategoryNameFromID(
                   DiagnosticIDs::getCategoryNumberForDiag(Info.getID()))
                   .str();
}

void StoreDiags::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                  const clang::Diagnostic &Info) {
  // If the diagnostic was generated for a different SourceManager, skip it.
  // This happens when a module is imported and needs to be implicitly built.
  // The compilation of that module will use the same StoreDiags, but different
  // SourceManager.
  if (OrigSrcMgr && Info.hasSourceManager() &&
      OrigSrcMgr != &Info.getSourceManager()) {
    IgnoreDiagnostics::log(DiagLevel, Info);
    return;
  }

  DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);
  bool OriginallyError =
      Info.getDiags()->getDiagnosticIDs()->isDefaultMappingAsError(
          Info.getID());

  if (Info.getLocation().isInvalid()) {
    // Handle diagnostics coming from command-line arguments. The source manager
    // is *not* available at this point, so we cannot use it.
    if (!OriginallyError) {
      IgnoreDiagnostics::log(DiagLevel, Info);
      return; // non-errors add too much noise, do not show them.
    }

    flushLastDiag();

    LastDiag = Diag();
    LastDiagLoc.reset();
    LastDiagOriginallyError = OriginallyError;
    LastDiag->ID = Info.getID();
    fillNonLocationData(DiagLevel, Info, *LastDiag);
    LastDiag->InsideMainFile = true;
    // Put it at the start of the main file, for a lack of a better place.
    LastDiag->Range.start = Position{0, 0};
    LastDiag->Range.end = Position{0, 0};
    return;
  }

  if (!LangOpts || !Info.hasSourceManager()) {
    IgnoreDiagnostics::log(DiagLevel, Info);
    return;
  }

  SourceManager &SM = Info.getSourceManager();

  auto FillDiagBase = [&](DiagBase &D) {
    fillNonLocationData(DiagLevel, Info, D);

    D.InsideMainFile = isInsideMainFile(Info);
    D.Range = diagnosticRange(Info, *LangOpts);
    D.File = std::string(SM.getFilename(Info.getLocation()));
    D.AbsFile = getCanonicalPath(
        SM.getFileEntryForID(SM.getFileID(Info.getLocation())), SM);
    D.ID = Info.getID();
    return D;
  };

  auto AddFix = [&](bool SyntheticMessage) -> bool {
    assert(!Info.getFixItHints().empty() &&
           "diagnostic does not have attached fix-its");
    // No point in generating fixes, if the diagnostic is for a different file.
    if (!LastDiag->InsideMainFile)
      return false;
    // Copy as we may modify the ranges.
    auto FixIts = Info.getFixItHints().vec();
    llvm::SmallVector<TextEdit, 1> Edits;
    for (auto &FixIt : FixIts) {
      // Allow fixits within a single macro-arg expansion to be applied.
      // This can be incorrect if the argument is expanded multiple times in
      // different contexts. Hopefully this is rare!
      if (FixIt.RemoveRange.getBegin().isMacroID() &&
          FixIt.RemoveRange.getEnd().isMacroID() &&
          SM.getFileID(FixIt.RemoveRange.getBegin()) ==
              SM.getFileID(FixIt.RemoveRange.getEnd())) {
        FixIt.RemoveRange = CharSourceRange(
            {SM.getTopMacroCallerLoc(FixIt.RemoveRange.getBegin()),
             SM.getTopMacroCallerLoc(FixIt.RemoveRange.getEnd())},
            FixIt.RemoveRange.isTokenRange());
      }
      // Otherwise, follow clang's behavior: no fixits in macros.
      if (FixIt.RemoveRange.getBegin().isMacroID() ||
          FixIt.RemoveRange.getEnd().isMacroID())
        return false;
      if (!isInsideMainFile(FixIt.RemoveRange.getBegin(), SM))
        return false;
      Edits.push_back(toTextEdit(FixIt, SM, *LangOpts));
    }

    llvm::SmallString<64> Message;
    // If requested and possible, create a message like "change 'foo' to 'bar'".
    if (SyntheticMessage && FixIts.size() == 1) {
      const auto &FixIt = FixIts.front();
      bool Invalid = false;
      llvm::StringRef Remove =
          Lexer::getSourceText(FixIt.RemoveRange, SM, *LangOpts, &Invalid);
      llvm::StringRef Insert = FixIt.CodeToInsert;
      if (!Invalid) {
        llvm::raw_svector_ostream M(Message);
        if (!Remove.empty() && !Insert.empty()) {
          M << "change '";
          writeCodeToFixMessage(M, Remove);
          M << "' to '";
          writeCodeToFixMessage(M, Insert);
          M << "'";
        } else if (!Remove.empty()) {
          M << "remove '";
          writeCodeToFixMessage(M, Remove);
          M << "'";
        } else if (!Insert.empty()) {
          M << "insert '";
          writeCodeToFixMessage(M, Insert);
          M << "'";
        }
        // Don't allow source code to inject newlines into diagnostics.
        std::replace(Message.begin(), Message.end(), '\n', ' ');
      }
    }
    if (Message.empty()) // either !SyntheticMessage, or we failed to make one.
      Info.FormatDiagnostic(Message);
    LastDiag->Fixes.push_back(
        Fix{std::string(Message.str()), std::move(Edits)});
    return true;
  };

  if (!isNote(DiagLevel)) {
    // Handle the new main diagnostic.
    flushLastDiag();

    LastDiag = Diag();
    // FIXME: Merge with feature modules.
    if (Adjuster)
      DiagLevel = Adjuster(DiagLevel, Info);

    FillDiagBase(*LastDiag);
    if (isExcluded(LastDiag->ID))
      LastDiag->Severity = DiagnosticsEngine::Ignored;
    if (DiagCB)
      DiagCB(Info, *LastDiag);
    // Don't bother filling in the rest if diag is going to be dropped.
    if (LastDiag->Severity == DiagnosticsEngine::Ignored)
      return;

    LastDiagLoc.emplace(Info.getLocation(), Info.getSourceManager());
    LastDiagOriginallyError = OriginallyError;
    if (!Info.getFixItHints().empty())
      AddFix(true /* try to invent a message instead of repeating the diag */);
    if (Fixer) {
      auto ExtraFixes = Fixer(LastDiag->Severity, Info);
      LastDiag->Fixes.insert(LastDiag->Fixes.end(), ExtraFixes.begin(),
                             ExtraFixes.end());
    }
  } else {
    // Handle a note to an existing diagnostic.
    if (!LastDiag) {
      assert(false && "Adding a note without main diagnostic");
      IgnoreDiagnostics::log(DiagLevel, Info);
      return;
    }

    // If a diagnostic was suppressed due to the suppression filter,
    // also suppress notes associated with it.
    if (LastDiag->Severity == DiagnosticsEngine::Ignored)
      return;

    // Give include-fixer a chance to replace a note with a fix.
    if (Fixer) {
      auto ReplacementFixes = Fixer(LastDiag->Severity, Info);
      if (!ReplacementFixes.empty()) {
        assert(Info.getNumFixItHints() == 0 &&
               "Include-fixer replaced a note with clang fix-its attached!");
        LastDiag->Fixes.insert(LastDiag->Fixes.end(), ReplacementFixes.begin(),
                               ReplacementFixes.end());
        return;
      }
    }

    if (!Info.getFixItHints().empty()) {
      // A clang note with fix-it is not a separate diagnostic in clangd. We
      // attach it as a Fix to the main diagnostic instead.
      if (!AddFix(false /* use the note as the message */))
        IgnoreDiagnostics::log(DiagLevel, Info);
    } else {
      // A clang note without fix-its corresponds to clangd::Note.
      Note N;
      FillDiagBase(N);

      LastDiag->Notes.push_back(std::move(N));
    }
  }
}

void StoreDiags::flushLastDiag() {
  if (!LastDiag)
    return;
  auto Finish = llvm::make_scope_exit([&, NDiags(Output.size())] {
    if (Output.size() == NDiags) // No new diag emitted.
      vlog("Dropped diagnostic: {0}: {1}", LastDiag->File, LastDiag->Message);
    LastDiag.reset();
  });

  if (LastDiag->Severity == DiagnosticsEngine::Ignored)
    return;
  // Move errors that occur from headers into main file.
  if (!LastDiag->InsideMainFile && LastDiagLoc && LastDiagOriginallyError) {
    if (tryMoveToMainFile(*LastDiag, *LastDiagLoc)) {
      // Suppress multiple errors from the same inclusion.
      if (!IncludedErrorLocations
               .insert({LastDiag->Range.start.line,
                        LastDiag->Range.start.character})
               .second)
        return;
    }
  }
  if (!mentionsMainFile(*LastDiag))
    return;
  Output.push_back(std::move(*LastDiag));
}

bool isBuiltinDiagnosticSuppressed(unsigned ID,
                                   const llvm::StringSet<> &Suppress,
                                   const LangOptions &LangOpts) {
  // Don't complain about header-only stuff in mainfiles if it's a header.
  // FIXME: would be cleaner to suppress in clang, once we decide whether the
  //        behavior should be to silently-ignore or respect the pragma.
  if (ID == diag::pp_pragma_sysheader_in_main_file && LangOpts.IsHeaderFile)
    return true;

  if (const char *CodePtr = getDiagnosticCode(ID)) {
    if (Suppress.contains(normalizeSuppressedCode(CodePtr)))
      return true;
  }
  StringRef Warning = DiagnosticIDs::getWarningOptionForDiag(ID);
  if (!Warning.empty() && Suppress.contains(Warning))
    return true;
  return false;
}

llvm::StringRef normalizeSuppressedCode(llvm::StringRef Code) {
  Code.consume_front("err_");
  Code.consume_front("-W");
  return Code;
}

llvm::Optional<std::string> getDiagnosticDocURI(Diag::DiagSource Source,
                                                unsigned ID,
                                                llvm::StringRef Name) {
  switch (Source) {
  case Diag::Unknown:
    break;
  case Diag::Clang:
    // There is a page listing many warning flags, but it provides too little
    // information to be worth linking.
    // https://clang.llvm.org/docs/DiagnosticsReference.html
    break;
  case Diag::ClangTidy:
    return {("https://clang.llvm.org/extra/clang-tidy/checks/" + Name + ".html")
                .str()};
  case Diag::Clangd:
    if (Name == "unused-includes")
      return {"https://clangd.llvm.org/guides/include-cleaner"};
    break;
  case Diag::ClangdConfig:
    // FIXME: we should link to https://clangd.llvm.org/config
    // However we have no diagnostic codes, which the link should describe!
    break;
  }
  return llvm::None;
}

} // namespace clangd
} // namespace clang
