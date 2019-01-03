//===--- ClangRefactorTest.cpp - ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-refactor-test tool that is used to test the
//  refactoring library in Clang.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Refactor.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Tooling/Refactor/SymbolName.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace clang;

namespace opts {

static cl::OptionCategory
    ClangRefactorTestOptions("clang-refactor-test common options");

cl::SubCommand RenameInitiateSubcommand(
    "rename-initiate", "Initiate renaming in an initial translation unit");

cl::SubCommand RenameInitiateUSRSubcommand(
    "rename-initiate-usr",
    "Initiate renaming in an translation unit on a specific declaration");

cl::SubCommand RenameIndexedFileSubcommand(
    "rename-indexed-file",
    "Initiate renaming and find occurrences in an indexed file");

cl::SubCommand ListRefactoringActionsSubcommand("list-actions",
                                                "Print the list of the "
                                                "refactoring actions that can "
                                                "be performed at the specified "
                                                "location");

cl::SubCommand InitiateActionSubcommand("initiate",
                                        "Initiate a refactoring action");

cl::SubCommand
    PerformActionSubcommand("perform",
                            "Initiate and perform a refactoring action");

const cl::desc
    AtOptionDescription("The location at which the refactoring should be "
                        "initiated (<file>:<line>:<column>)");

const cl::desc InRangeOptionDescription(
    "The location(s) at which the refactoring should be "
    "initiated (<file>:<line>:<column>-<last-column>)");

const cl::desc SelectedRangeOptionDescription(
    "The selected source range in which the refactoring should be "
    "initiated (<file>:<line>:<column>-<line>:<column>)");

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

namespace rename {
static cl::list<std::string> AtLocation("at", AtOptionDescription, cl::Required,
                                        cl::cat(ClangRefactorTestOptions),
                                        cl::sub(RenameInitiateSubcommand),
                                        cl::OneOrMore);

static cl::opt<std::string>
    USR("usr", cl::desc("The USR of the declaration that should be renamed"),
        cl::cat(ClangRefactorTestOptions), cl::sub(RenameInitiateUSRSubcommand),
        cl::Required);

static cl::opt<std::string>
    NewName("new-name", cl::desc("The new name to change the symbol to."),
            cl::Required, cl::cat(ClangRefactorTestOptions),
            cl::sub(RenameInitiateSubcommand),
            cl::sub(RenameInitiateUSRSubcommand));

static cl::list<std::string>
    IndexedNames("name", cl::desc("The names of the renamed symbols"),
                 cl::Required, cl::OneOrMore, cl::cat(ClangRefactorTestOptions),
                 cl::sub(RenameIndexedFileSubcommand));

static cl::list<std::string> IndexedNewNames(
    "new-name", cl::desc("The new name to change the symbol to."), cl::Required,
    cl::OneOrMore, cl::cat(ClangRefactorTestOptions),
    cl::sub(RenameIndexedFileSubcommand));

static cl::opt<std::string>
    IndexedSymbolKind("indexed-symbol-kind",
                      cl::desc("The kind of the indexed symbol."), cl::Optional,
                      cl::cat(ClangRefactorTestOptions),
                      cl::sub(RenameIndexedFileSubcommand));

static cl::opt<std::string>
    IndexedFileName("indexed-file", cl::desc("The name of the indexed file"),
                    cl::Required, cl::cat(ClangRefactorTestOptions),
                    cl::sub(RenameIndexedFileSubcommand));

static cl::list<std::string>
    IndexedLocations("indexed-at",
                     cl::desc("The location of an indexed occurrence "
                              "([<kind>|<symbol-index>:]<line>:<column>)"),
                     cl::ZeroOrMore, cl::cat(ClangRefactorTestOptions),
                     cl::sub(RenameIndexedFileSubcommand));

static cl::opt<bool> AvoidTextual(
    "no-textual-matches", cl::desc("Avoid searching for textual matches"),
    cl::cat(ClangRefactorTestOptions), cl::sub(RenameIndexedFileSubcommand));

static cl::opt<bool> DumpSymbols(
    "dump-symbols", cl::desc("Dump the information about the renamed symbols"),
    cl::cat(ClangRefactorTestOptions), cl::sub(RenameInitiateSubcommand),
    cl::sub(RenameInitiateUSRSubcommand));
}

namespace listActions {
cl::opt<std::string> AtLocation("at", AtOptionDescription, cl::Required,
                                cl::cat(ClangRefactorTestOptions),
                                cl::sub(ListRefactoringActionsSubcommand));

cl::opt<std::string> SelectedRange("selected", SelectedRangeOptionDescription,
                                   cl::cat(ClangRefactorTestOptions),
                                   cl::sub(ListRefactoringActionsSubcommand));

cl::opt<bool> DumpRawActionType(
    "dump-raw-action-type",
    cl::desc("Prints the action type integer value for each listed action"),
    cl::cat(ClangRefactorTestOptions),
    cl::sub(ListRefactoringActionsSubcommand));
}

namespace initiateAndPerform {
cl::list<std::string> InLocationRanges("in", cl::ZeroOrMore,
                                       InRangeOptionDescription,
                                       cl::cat(ClangRefactorTestOptions),
                                       cl::sub(InitiateActionSubcommand));

cl::list<std::string> AtLocations("at", cl::ZeroOrMore, AtOptionDescription,
                                  cl::cat(ClangRefactorTestOptions),
                                  cl::sub(InitiateActionSubcommand),
                                  cl::sub(PerformActionSubcommand));

cl::list<std::string> SelectedRanges("selected", cl::ZeroOrMore,
                                     SelectedRangeOptionDescription,
                                     cl::cat(ClangRefactorTestOptions),
                                     cl::sub(InitiateActionSubcommand),
                                     cl::sub(PerformActionSubcommand));

cl::opt<std::string> ActionName("action", cl::Required,
                                cl::desc("The name of the refactoring action"),
                                cl::cat(ClangRefactorTestOptions),
                                cl::sub(InitiateActionSubcommand),
                                cl::sub(PerformActionSubcommand));

cl::opt<bool> LocationAgnostic(
    "location-agnostic",
    cl::desc(
        "Ignore the location of initiation when verifying result consistency"),
    cl::cat(ClangRefactorTestOptions), cl::sub(InitiateActionSubcommand));

cl::opt<unsigned> CandidateIndex(
    "candidate",
    cl::desc(
        "The index of the refactoring candidate which should be performed"),
    cl::cat(ClangRefactorTestOptions), cl::sub(PerformActionSubcommand));

cl::opt<std::string> ContinuationFile(
    "continuation-file",
    cl::desc("The source file in which the continuation should run"),
    cl::cat(ClangRefactorTestOptions), cl::sub(PerformActionSubcommand));

cl::opt<std::string> QueryResults(
    "query-results", cl::desc("The indexer query results that should be passed "
                              "into the continuation"),
    cl::cat(ClangRefactorTestOptions), cl::sub(PerformActionSubcommand));

cl::opt<bool> EmitAssociatedInfo(
    "emit-associated", cl::desc("Dump additional associated information"),
    cl::cat(ClangRefactorTestOptions), cl::sub(PerformActionSubcommand));
}

cl::opt<bool> Apply(
    "apply",
    cl::desc(
        "Apply the changes and print the modified file to standard output"),
    cl::cat(ClangRefactorTestOptions), cl::sub(PerformActionSubcommand),
    cl::sub(RenameInitiateSubcommand), cl::sub(RenameIndexedFileSubcommand));

cl::opt<bool>
    Diff("diff",
         cl::desc("Display the replaced text in red when -apply is specified"),
         cl::cat(ClangRefactorTestOptions), cl::sub(PerformActionSubcommand),
         cl::sub(RenameInitiateSubcommand),
         cl::sub(RenameIndexedFileSubcommand));

cl::opt<int> Context("context", cl::desc("How many lines of context should be "
                                         "displayed when -apply is specified"),
                     cl::cat(ClangRefactorTestOptions),
                     cl::sub(PerformActionSubcommand),
                     cl::sub(RenameInitiateSubcommand),
                     cl::sub(RenameIndexedFileSubcommand));

static cl::opt<std::string> FileName(
    cl::Positional, cl::desc("<filename>"), cl::Required,
    cl::cat(ClangRefactorTestOptions), cl::sub(RenameInitiateSubcommand),
    cl::sub(RenameInitiateUSRSubcommand), cl::sub(RenameIndexedFileSubcommand),
    cl::sub(ListRefactoringActionsSubcommand),
    cl::sub(InitiateActionSubcommand), cl::sub(PerformActionSubcommand));

static cl::opt<bool> IgnoreFilenameForInitiationTU(
    "ignore-filename-for-initiation-tu", cl::Optional,
    cl::cat(ClangRefactorTestOptions), cl::sub(RenameIndexedFileSubcommand));

static cl::list<std::string> CompilerArguments(
    cl::ConsumeAfter, cl::desc("<arguments to be passed to the compiler>"),
    cl::cat(ClangRefactorTestOptions), cl::sub(RenameInitiateSubcommand),
    cl::sub(RenameInitiateUSRSubcommand), cl::sub(RenameIndexedFileSubcommand),
    cl::sub(ListRefactoringActionsSubcommand),
    cl::sub(InitiateActionSubcommand), cl::sub(PerformActionSubcommand));

static cl::opt<std::string> ImplementationTU(
    "implementation-tu", cl::desc("The name of the implementation TU"),
    cl::cat(ClangRefactorTestOptions), cl::sub(RenameInitiateSubcommand));
}

static const char *renameOccurrenceKindString(CXSymbolOccurrenceKind Kind,
                                              bool IsLocal,
                                              bool IsMacroExpansion) {
  switch (Kind) {
  case CXSymbolOccurrence_MatchingSymbol:
    return IsMacroExpansion ? "macro" : IsLocal ? "rename local" : "rename";
  case CXSymbolOccurrence_MatchingSelector:
    assert(!IsLocal && "Objective-C selector renames must be global");
    return IsMacroExpansion ? "selector in macro" : "selector";
  case CXSymbolOccurrence_MatchingImplicitProperty:
    assert(!IsLocal);
    return IsMacroExpansion ? "implicit-property in macro"
                            : "implicit-property";
  case CXSymbolOccurrence_MatchingCommentString:
    return "comment";
  case CXSymbolOccurrence_MatchingDocCommentString:
    return "documentation";
  case CXSymbolOccurrence_MatchingFilename:
    return "filename";
  case CXSymbolOccurrence_MatchingStringLiteral:
    return "string-literal";
  case CXSymbolOccurrence_ExtractedDeclaration:
    return "extracted-decl";
  case CXSymbolOccurrence_ExtractedDeclaration_Reference:
    return "extracted-decl-ref";
  }
  llvm_unreachable("unexpected CXSymbolOccurrenceKind value");
}

static int apply(ArrayRef<CXRefactoringReplacement> Replacements,
                 StringRef Filename) {
  // Assume that the replacements are sorted.
  auto Result = MemoryBuffer::getFile(Filename);
  if (!Result) {
    errs() << "Failed to open " << Filename << "\n";
    return 1;
  }

  raw_ostream &OS = outs();

  int Context = opts::Context;

  MemoryBuffer &Buffer = **Result;
  std::vector<std::pair<StringRef, std::vector<CXRefactoringReplacement>>>
      Lines;
  for (auto I = line_iterator(Buffer, /*SkipBlanks=*/false),
            E = line_iterator();
       I != E; ++I)
    Lines.push_back(
        std::make_pair(*I, std::vector<CXRefactoringReplacement>()));
  unsigned FlushedLine = 1;
  auto FlushUntil = [&](unsigned Line) {
    // Adjust the first flushed line if needed when printing in context mode.
    if (FlushedLine == 1 && Context)
      FlushedLine = std::max(int(Line) - Context, 1);
    for (; FlushedLine < Line; ++FlushedLine) {
      const auto &Line = Lines[FlushedLine - 1];
      if (Line.second.empty()) {
        OS << Line.first << "\n";
        continue;
      }

      unsigned I = 0;
      for (const CXRefactoringReplacement &Replacement : Line.second) {
        OS << Line.first.substr(I, Replacement.Range.Begin.Column - 1 - I);
        if (opts::Diff) {
          OS.changeColor(raw_ostream::RED, false, true);
          OS << Line.first.substr(Replacement.Range.Begin.Column - 1,
                                  Replacement.Range.End.Column - 1 -
                                      (Replacement.Range.Begin.Column - 1));
        }
        OS.changeColor(raw_ostream::GREEN);
        OS << clang_getCString(Replacement.ReplacementString);
        OS.resetColor();
        I = Replacement.Range.End.Column - 1;
      }
      OS << Line.first.substr(I);
      if (I < Line.first.size() || opts::Diff)
        OS << "\n";
    }
  };

  int EndLineMax = 0;
  for (const CXRefactoringReplacement &Replacement : Replacements) {
    EndLineMax = std::max(int(Replacement.Range.End.Line), EndLineMax);
    unsigned StartingLine = Replacement.Range.Begin.Line;
    FlushUntil(StartingLine);
    if (Replacement.Range.End.Line == StartingLine) {
      Lines[StartingLine - 1].second.push_back(Replacement);
      continue;
    }
    // Multi-line replacements have to be split
    for (unsigned I = StartingLine; I <= Replacement.Range.End.Line; ++I) {
      CXRefactoringReplacement NewReplacement;
      if (I == Replacement.Range.End.Line)
        NewReplacement.ReplacementString = Replacement.ReplacementString;
      else
        // FIXME: This is a hack to workaround the fact that the API doesn't
        // provide a way to create a null string. This should be fixed when
        // upstreaming.
        NewReplacement.ReplacementString = {0, 0};
      NewReplacement.Range.Begin.Line = I;
      NewReplacement.Range.Begin.Column =
          I == StartingLine ? Replacement.Range.Begin.Column : 1;
      NewReplacement.Range.End.Line = I;
      NewReplacement.Range.End.Column = I == Replacement.Range.End.Line
                                            ? Replacement.Range.End.Column
                                            : Lines[I - 1].first.size() + 1;
      NewReplacement.AssociatedData = nullptr;
      Lines[I - 1].second.push_back(NewReplacement);
    }
  }
  FlushUntil(Context ? std::min(int(Lines.size()), EndLineMax + Context) + 1
                     : Lines.size() + 2);
  // Print out a dividor when printing in the context mode.
  if (Context) {
    for (int I = 0; I < 80; ++I)
      OS << '-';
    OS << "\n";
  }
  return 0;
}

/// Converts the given renamed \p Occurrence into a string value that represents
/// this occurrence.
static std::string
occurrenceToString(const CXSymbolOccurrence &Occurrence, bool IsLocal,
                   const tooling::OldSymbolName &NewName,
                   const tooling::OldSymbolName &ExpectedReplacementStrings,
                   StringRef Filename) {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << renameOccurrenceKindString(Occurrence.Kind, IsLocal,
                                   Occurrence.IsMacroExpansion)
     << ' ';
  if (!Filename.empty())
    OS << '"' << Filename << "\" ";

  bool FirstRange = true;
  assert(NewName.size() >= Occurrence.NumNamePieces &&
         "new name doesn't match the number of pieces");
  for (unsigned J = 0; J != Occurrence.NumNamePieces; ++J) {
    if (!FirstRange) // TODO
      OS << ", ";

    // Print the replacement string if it doesn't match the expected string.
    if (NewName[J] != ExpectedReplacementStrings[J])
      OS << '"' << NewName[J] << "\" ";

    CXFileRange Range = Occurrence.NamePieces[J];
    OS << Range.Begin.Line << ":" << Range.Begin.Column << " -> "
       << Range.End.Line << ":" << Range.End.Column;
    FirstRange = false;
  }
  return OS.str();
}

static CXCursorKind
renameIndexedOccurrenceKindStringToKind(StringRef Str, CXCursorKind Default) {
  return llvm::StringSwitch<CXCursorKind>(Str)
      .Case("objc-im", CXCursor_ObjCInstanceMethodDecl)
      .Case("objc-cm", CXCursor_ObjCClassMethodDecl)
      .Case("objc-message", CXCursor_ObjCMessageExpr)
      .Case("include", CXCursor_InclusionDirective)
      .Case("objc-class", CXCursor_ObjCInterfaceDecl)
      .Default(Default);
}

/// Parses the string passed as the -indexed-at argument.
std::pair<CXRenamedIndexedSymbolLocation, unsigned>
parseIndexedOccurrence(StringRef IndexedOccurrence,
                       CXCursorKind DefaultCursorKind) {
  StringRef LineColumnLoc = IndexedOccurrence;
  CXCursorKind Kind = DefaultCursorKind;
  unsigned SymbolIndex = 0;
  if (LineColumnLoc.count(':') > 1) {
    std::pair<StringRef, StringRef> Split = LineColumnLoc.split(':');
    // The first value is either the kind or the symbol index.
    if (Split.first.getAsInteger(10, SymbolIndex)) {
      if (Split.second.count(':') > 1) {
        std::pair<StringRef, StringRef> SecondSplit = Split.second.split(':');
        if (SecondSplit.first.getAsInteger(10, SymbolIndex))
          assert(false && "expected symbol index");
        Split.second = SecondSplit.second;
      }
      Kind = renameIndexedOccurrenceKindStringToKind(Split.first, Kind);
    }
    LineColumnLoc = Split.second;
  }
  auto Loc = std::string("-:") + LineColumnLoc.str();
  auto Location = ParsedSourceLocation::FromString(Loc);
  return std::make_pair(
      CXRenamedIndexedSymbolLocation{{Location.Line, Location.Column}, Kind},
      SymbolIndex);
}

/// Compare the produced occurrences to the expected occurrences that were
/// gathered at the first location. Return true if the occurrences are
/// different.
static bool compareOccurrences(ArrayRef<std::string> ExpectedReplacements,
                               CXSymbolOccurrencesResult Occurrences,
                               bool IsLocal,
                               const tooling::OldSymbolName &NewSymbolName,
                               bool PrintFilenames) {
  unsigned NumFiles = clang_SymbolOccurrences_getNumFiles(Occurrences);
  size_t ExpectedReplacementIndex = 0;
  for (unsigned FileIndex = 0; FileIndex < NumFiles; ++FileIndex) {
    CXSymbolOccurrencesInFile FileResult;
    clang_SymbolOccurrences_getOccurrencesForFile(Occurrences, FileIndex,
                                                  &FileResult);
    StringRef Filename =
        PrintFilenames ? clang_getCString(FileResult.Filename) : "";

    for (unsigned I = 0; I != FileResult.NumOccurrences; ++I) {
      std::string Replacement =
          occurrenceToString(FileResult.Occurrences[I], IsLocal, NewSymbolName,
                             NewSymbolName, Filename);
      if (ExpectedReplacementIndex >= ExpectedReplacements.size() ||
          Replacement != ExpectedReplacements[ExpectedReplacementIndex])
        return true;
      ++ExpectedReplacementIndex;
    }
  }
  // Verify that all of the expected replacements were checked.
  return ExpectedReplacementIndex != ExpectedReplacements.size();
}

struct ImplementationTUWrapper {
  CXTranslationUnit TU = nullptr;

  ImplementationTUWrapper() {}
  ~ImplementationTUWrapper() { clang_disposeTranslationUnit(TU); }

  ImplementationTUWrapper(const ImplementationTUWrapper &) = delete;
  ImplementationTUWrapper &operator=(const ImplementationTUWrapper &) = delete;

  bool load(CXRefactoringAction Action, CXIndex CIdx,
            ArrayRef<const char *> Args);
};

bool ImplementationTUWrapper::load(CXRefactoringAction Action, CXIndex CIdx,
                                   ArrayRef<const char *> Args) {
  if (!clang_RefactoringAction_requiresImplementationTU(Action))
    return false;
  CXString USR =
      clang_RefactoringAction_getUSRThatRequiresImplementationTU(Action);
  outs() << "Implementation TU USR: '" << clang_getCString(USR) << "'\n";
  clang_disposeString(USR);
  if (!TU) {
    CXErrorCode Err = clang_parseTranslationUnit2(
        CIdx, opts::ImplementationTU.c_str(), Args.data(), Args.size(), 0, 0,
        CXTranslationUnit_KeepGoing, &TU);
    if (Err != CXError_Success) {
      errs() << "error: failed to load implementation TU '"
             << opts::ImplementationTU << "'\n";
      return true;
    }
  }
  CXErrorCode Err = clang_RefactoringAction_addImplementationTU(Action, TU);
  if (Err != CXError_Success) {
    errs() << "error: failed to add implementation TU '"
           << opts::ImplementationTU << "'\n";
    return true;
  }
  return false;
}

static bool reportNewNameError(CXErrorCode Err) {
  std::string NewName = opts::RenameIndexedFileSubcommand
                            ? opts::rename::IndexedNewNames[0]
                            : opts::rename::NewName;
  if (Err == CXError_RefactoringNameSizeMismatch)
    errs() << "error: the number of strings in the new name '" << NewName
           << "' doesn't match the the number of strings in the old name\n";
  else if (Err == CXError_RefactoringNameInvalid)
    errs() << "error: invalid new name '" << NewName << "'\n";
  else
    return true;
  return false;
}

int rename(CXTranslationUnit TU, CXIndex CIdx, ArrayRef<const char *> Args) {
  assert(!opts::RenameIndexedFileSubcommand);
  // Contains the renamed source replacements for the first location. It is
  // compared to replacements from follow-up renames to ensure that all renames
  // give the same result.
  std::vector<std::string> ExpectedReplacements;
  // Should we print out the filenames. False by default, but true when multiple
  // files are modified.
  bool PrintFilenames = false;
  ImplementationTUWrapper ImplementationTU;

  auto RenameAt = [&](const ParsedSourceLocation &Location,
                      const std::string &USR) -> int {
    CXRefactoringAction RenamingAction;
    CXErrorCode Err;
    CXDiagnosticSet Diags = nullptr;
    if (USR.empty()) {
      CXSourceLocation Loc =
          clang_getLocation(TU, clang_getFile(TU, Location.FileName.c_str()),
                            Location.Line, Location.Column);
      Err = clang_Refactoring_initiateAction(
          TU, Loc, clang_getNullRange(), CXRefactor_Rename,
          /*Options=*/nullptr, &RenamingAction, &Diags);
    } else {
      Err = clang_Refactoring_initiateActionOnDecl(
          TU, USR.c_str(), CXRefactor_Rename, /*Options=*/nullptr,
          &RenamingAction, nullptr);
    }
    if (Err != CXError_Success) {
      errs() << "error: could not rename symbol "
             << (USR.empty() ? "at the given location\n"
                             : "with the given USR\n");
      if (USR.empty()) {
        unsigned NumDiags = clang_getNumDiagnosticsInSet(Diags);
        for (unsigned DiagID = 0; DiagID < NumDiags; ++DiagID) {
          CXDiagnostic Diag = clang_getDiagnosticInSet(Diags, DiagID);
          CXString Spelling = clang_getDiagnosticSpelling(Diag);
          errs() << clang_getCString(Spelling) << "\n";
          clang_disposeString(Spelling);
        }
      }
      clang_disposeDiagnosticSet(Diags);
      return 1;
    }
    clang_disposeDiagnosticSet(Diags);

    if (ImplementationTU.load(RenamingAction, CIdx, Args))
      return 1;

    Err = clang_Refactoring_initiateRenamingOperation(RenamingAction);
    if (Err != CXError_Success) {
      errs() << "error: failed to initiate the renaming operation!\n";
      return 1;
    }

    bool IsLocal = clang_RefactoringAction_getInitiatedActionType(
                       RenamingAction) == CXRefactor_Rename_Local;

    unsigned NumSymbols = clang_RenamingOperation_getNumSymbols(RenamingAction);
    if (opts::rename::DumpSymbols) {
      outs() << "Renaming " << NumSymbols << " symbols\n";
      for (unsigned I = 0; I < NumSymbols; ++I) {
        CXString USR =
            clang_RenamingOperation_getUSRForSymbol(RenamingAction, I);
        outs() << "'" << clang_getCString(USR) << "'\n";
        clang_disposeString(USR);
      }
    }

    CXSymbolOccurrencesResult Occurrences;
    Occurrences = clang_Refactoring_findSymbolOccurrencesInInitiationTU(
        RenamingAction, Args.data(), Args.size(), 0, 0);

    clang_RefactoringAction_dispose(RenamingAction);

    // FIXME: This is a hack
    LangOptions LangOpts;
    LangOpts.ObjC = true;
    tooling::OldSymbolName NewSymbolName(opts::rename::NewName, LangOpts);

    if (ExpectedReplacements.empty()) {
      if (opts::Apply) {
        // FIXME: support --apply.
      }

      unsigned NumFiles = clang_SymbolOccurrences_getNumFiles(Occurrences);
      if (NumFiles > 1)
        PrintFilenames = true;
      // Convert the occurrences to strings
      for (unsigned FileIndex = 0; FileIndex < NumFiles; ++FileIndex) {
        CXSymbolOccurrencesInFile FileResult;
        clang_SymbolOccurrences_getOccurrencesForFile(Occurrences, FileIndex,
                                                      &FileResult);
        StringRef Filename =
            PrintFilenames ? clang_getCString(FileResult.Filename) : "";
        for (unsigned I = 0; I != FileResult.NumOccurrences; ++I)
          ExpectedReplacements.push_back(
              occurrenceToString(FileResult.Occurrences[I], IsLocal,
                                 NewSymbolName, NewSymbolName, Filename));
      }
      clang_SymbolOccurrences_dispose(Occurrences);
      return 0;
    }
    // Compare the produced occurrences to the expected occurrences that were
    // gathered at the first location.
    bool AreOccurrencesDifferent =
        compareOccurrences(ExpectedReplacements, Occurrences, IsLocal,
                           NewSymbolName, PrintFilenames);
    clang_SymbolOccurrences_dispose(Occurrences);
    if (!AreOccurrencesDifferent)
      return 0;
    errs() << "error: occurrences for a rename at " << Location.FileName << ":"
           << Location.Line << ":" << Location.Column
           << " differ to occurrences from the rename at the first location!\n";
    return 1;
  };

  std::vector<ParsedSourceLocation> ParsedLocations;
  for (const auto &I : enumerate(opts::rename::AtLocation)) {
    auto Location = ParsedSourceLocation::FromString(I.value());
    if (Location.FileName.empty()) {
      errs()
          << "error: The -at option must use the <file:line:column> format\n";
      return 1;
    }
    ParsedLocations.push_back(Location);
  }

  if (opts::RenameInitiateUSRSubcommand) {
    if (RenameAt(ParsedSourceLocation(), opts::rename::USR))
      return 1;
  } else {
    assert(!ParsedLocations.empty() && "No -at locations");

    for (const auto &Location : ParsedLocations) {
      if (RenameAt(Location, ""))
        return 1;
    }
  }

  // Print the produced renamed replacements
  if (opts::Apply)
    return 0;
  for (const auto &Replacement : ExpectedReplacements)
    outs() << Replacement << "\n";
  if (ExpectedReplacements.empty())
    outs() << "no replacements found\n";
  return 0;
}

int renameIndexedFile(CXIndex CIdx, ArrayRef<const char *> Args) {
  assert(opts::RenameIndexedFileSubcommand);

  // Compute the number of symbols.
  unsigned NumSymbols = opts::rename::IndexedNames.size();

  // Get the occurrences of a symbol.
  CXCursorKind DefaultCursorKind = renameIndexedOccurrenceKindStringToKind(
      opts::rename::IndexedSymbolKind, CXCursor_NotImplemented);
  std::vector<std::vector<CXIndexedSymbolLocation>> IndexedOccurrences(
      NumSymbols, std::vector<CXIndexedSymbolLocation>());
  for (const auto &IndexedOccurrence : opts::rename::IndexedLocations) {
    auto Occurrence =
        parseIndexedOccurrence(IndexedOccurrence, DefaultCursorKind);
    unsigned SymbolIndex = Occurrence.second;
    assert(SymbolIndex < IndexedOccurrences.size() && "Invalid symbol index");
    IndexedOccurrences[SymbolIndex].push_back(CXIndexedSymbolLocation{
        Occurrence.first.Location, Occurrence.first.CursorKind});
  }

  // Create the indexed symbols.
  std::vector<CXIndexedSymbol> IndexedSymbols;
  for (const auto &I : llvm::enumerate(IndexedOccurrences)) {
    const auto &Occurrences = I.value();
    const char *Name =
        opts::rename::IndexedNames[opts::rename::IndexedNames.size() < 2
                                       ? 0
                                       : I.index()]
            .c_str();
    IndexedSymbols.push_back({Occurrences.data(), (unsigned)Occurrences.size(),
                              DefaultCursorKind, Name});
  }

  CXRefactoringOptionSet Options = nullptr;
  if (opts::rename::AvoidTextual) {
    Options = clang_RefactoringOptionSet_create();
    clang_RefactoringOptionSet_add(Options,
                                   CXRefactorOption_AvoidTextualMatches);
  }

  CXSymbolOccurrencesResult Occurrences;
  CXErrorCode Err = clang_Refactoring_findSymbolOccurrencesInIndexedFile(
      IndexedSymbols.data(), IndexedSymbols.size(), CIdx,
      opts::rename::IndexedFileName.c_str(), Args.data(), Args.size(), 0, 0,
      Options, &Occurrences);
  if (Err != CXError_Success) {
    if (reportNewNameError(Err))
      errs() << "error: failed to perform indexed file rename\n";
    return 1;
  }

  if (Options)
    clang_RefactoringOptionSet_dispose(Options);

  // Should we print out the filenames. False by default, but true when multiple
  // files are modified.
  bool PrintFilenames = false;
  unsigned NumFiles = clang_SymbolOccurrences_getNumFiles(Occurrences);
  if (NumFiles > 1)
    PrintFilenames = true;

  LangOptions LangOpts;
  LangOpts.ObjC = true;
  tooling::OldSymbolName ExpectedReplacementStrings(
      opts::rename::IndexedNewNames[0], LangOpts);

  // Print the occurrences.
  bool HasReplacements = false;
  for (unsigned FileIndex = 0; FileIndex < NumFiles; ++FileIndex) {
    CXSymbolOccurrencesInFile FileResult;
    clang_SymbolOccurrences_getOccurrencesForFile(Occurrences, FileIndex,
                                                  &FileResult);
    StringRef Filename =
        PrintFilenames ? clang_getCString(FileResult.Filename) : "";
    HasReplacements = FileResult.NumOccurrences;
    for (unsigned I = 0; I != FileResult.NumOccurrences; ++I) {
      unsigned SymbolIndex = FileResult.Occurrences[I].SymbolIndex;
      const char *NewName =
          opts::rename::IndexedNewNames[opts::rename::IndexedNewNames.size() < 2
                                            ? 0
                                            : SymbolIndex]
              .c_str();
      LangOptions LangOpts;
      LangOpts.ObjC = true;
      tooling::OldSymbolName NewSymbolName(NewName, LangOpts);

      outs() << occurrenceToString(FileResult.Occurrences[I], /*IsLocal*/ false,
                                   NewSymbolName, ExpectedReplacementStrings,
                                   Filename)
             << "\n";
    }
  }
  if (!HasReplacements)
    outs() << "no replacements found\n";
  clang_SymbolOccurrences_dispose(Occurrences);
  return 0;
}

/// Returns the last column number of a line in a file.
static unsigned lastColumnForFile(StringRef Filename, unsigned LineNo) {
  auto Buf = llvm::MemoryBuffer::getFile(Filename);
  if (!Buf)
    return 0;
  unsigned LineCount = 1;
  for (llvm::line_iterator Lines(**Buf, /*SkipBlanks=*/false);
       !Lines.is_at_end(); ++Lines, ++LineCount) {
    if (LineNo == LineCount)
      return Lines->size() + 1;
  }
  return 0;
}

struct ParsedSourceLineRange : ParsedSourceLocation {
  unsigned MaxColumn;

  ParsedSourceLineRange() {}
  ParsedSourceLineRange(const ParsedSourceLocation &Loc)
      : ParsedSourceLocation(Loc), MaxColumn(Loc.Column) {}

  static Optional<ParsedSourceLineRange> FromString(StringRef Str) {
    std::pair<StringRef, StringRef> RangeSplit = Str.rsplit('-');
    auto PSL = ParsedSourceLocation::FromString(RangeSplit.first);
    ParsedSourceLineRange Result;
    Result.FileName = std::move(PSL.FileName);
    Result.Line = PSL.Line;
    Result.Column = PSL.Column;
    if (Result.FileName.empty())
      return None;
    if (RangeSplit.second == "end")
      Result.MaxColumn = lastColumnForFile(Result.FileName, Result.Line);
    else if (RangeSplit.second.getAsInteger(10, Result.MaxColumn))
      return None;
    if (Result.MaxColumn < Result.Column)
      return None;
    return Result;
  }
};

struct OldParsedSourceRange {
  ParsedSourceLocation Begin, End;

  OldParsedSourceRange(const ParsedSourceLocation &Begin,
                    const ParsedSourceLocation &End)
      : Begin(Begin), End(End) {}

  static Optional<OldParsedSourceRange> FromString(StringRef Str) {
    std::pair<StringRef, StringRef> RangeSplit = Str.rsplit('-');
    auto Begin = ParsedSourceLocation::FromString(RangeSplit.first);
    if (Begin.FileName.empty())
      return None;
    std::string EndString = Begin.FileName + ":" + RangeSplit.second.str();
    auto End = ParsedSourceLocation::FromString(EndString);
    if (End.FileName.empty())
      return None;
    return OldParsedSourceRange(Begin, End);
  }
};

int listRefactoringActions(CXTranslationUnit TU) {
  auto Location =
      ParsedSourceLocation::FromString(opts::listActions::AtLocation);
  if (Location.FileName.empty()) {
    errs() << "error: The -at option must use the <file:line:column> format\n";
    return 1;
  }
  CXSourceRange Range;
  if (!opts::listActions::SelectedRange.empty()) {
    auto SelectionRange =
        OldParsedSourceRange::FromString(opts::listActions::SelectedRange);
    if (!SelectionRange) {
      errs() << "error: The -selected option must use the "
                "<file:line:column-line:column> format\n";
      return 1;
    }
    auto Begin = SelectionRange.getValue().Begin;
    auto End = SelectionRange.getValue().End;
    CXFile File = clang_getFile(TU, Begin.FileName.c_str());
    Range =
        clang_getRange(clang_getLocation(TU, File, Begin.Line, Begin.Column),
                       clang_getLocation(TU, File, End.Line, End.Column));
  } else
    Range = clang_getNullRange();
  CXSourceLocation Loc =
      clang_getLocation(TU, clang_getFile(TU, Location.FileName.c_str()),
                        Location.Line, Location.Column);
  CXRefactoringActionSet ActionSet;
  CXRefactoringActionSetWithDiagnostics FailedActionSet;
  CXErrorCode Err =
      clang_Refactoring_findActionsWithInitiationFailureDiagnosicsAt(
          TU, Loc, Range, /*Options=*/nullptr, &ActionSet, &FailedActionSet);
  if (FailedActionSet.NumActions) {
    errs() << "Failed to initiate " << FailedActionSet.NumActions
           << " actions because:\n";
    for (unsigned I = 0; I < FailedActionSet.NumActions; ++I) {
      errs() << clang_getCString(clang_RefactoringActionType_getName(
                    FailedActionSet.Actions[I].Action))
             << ":";
      CXDiagnosticSet Diags = FailedActionSet.Actions[I].Diagnostics;
      unsigned NumDiags = clang_getNumDiagnosticsInSet(Diags);
      for (unsigned DiagID = 0; DiagID < NumDiags; ++DiagID) {
        CXDiagnostic Diag = clang_getDiagnosticInSet(Diags, DiagID);
        CXString Spelling = clang_getDiagnosticSpelling(Diag);
        errs() << ' ' << clang_getCString(Spelling);
        clang_disposeString(Spelling);
      }
      errs() << "\n";
    }
  }
  if (Err == CXError_RefactoringActionUnavailable)
    errs() << "No refactoring actions are available at the given location\n";
  if (Err != CXError_Success)
    return 1;
  // Print the list of refactoring actions.
  outs() << "Found " << ActionSet.NumActions << " actions:\n";
  for (unsigned I = 0; I < ActionSet.NumActions; ++I) {
    outs() << clang_getCString(
        clang_RefactoringActionType_getName(ActionSet.Actions[I]));
    if (opts::listActions::DumpRawActionType)
      outs() << "(" << ActionSet.Actions[I] << ")";
    outs() << "\n";
  }
  clang_RefactoringActionSet_dispose(&ActionSet);
  clang_RefactoringActionSetWithDiagnostics_dispose(&FailedActionSet);
  return 0;
}

static std::string locationToString(CXSourceLocation Loc) {
  unsigned Line, Column;
  clang_getFileLocation(Loc, nullptr, &Line, &Column, nullptr);
  std::string S;
  llvm::raw_string_ostream OS(S);
  OS << Line << ':' << Column;
  return OS.str();
}

static std::string rangeToString(CXSourceRange Range) {
  return locationToString(clang_getRangeStart(Range)) + " -> " +
         locationToString(clang_getRangeEnd(Range));
}

static std::string
refactoringCandidatesToString(CXRefactoringCandidateSet Candidates) {
  std::string Results = "with multiple candidates:";
  for (unsigned I = 0; I < Candidates.NumCandidates; ++I) {
    Results += "\n";
    Results += clang_getCString(Candidates.Candidates[I].Description);
  }
  return Results;
}

static void printEscaped(StringRef Str, raw_ostream &OS) {
  size_t Pos = Str.find('\n');
  OS << Str.substr(0, Pos);
  if (Pos == StringRef::npos)
    return;
  OS << "\\n";
  printEscaped(Str.substr(Pos + 1), OS);
}

bool printRefactoringReplacements(
    CXRefactoringResult Result, CXRefactoringContinuation Continuation,
    CXRefactoringContinuation CurrentContinuation) {
  CXRefactoringReplacements Replacements =
      clang_RefactoringResult_getSourceReplacements(Result);
  if (Replacements.NumFileReplacementSets == 0) {
    if (CurrentContinuation)
      return false;
    errs() << "error: no replacements produced!\n";
    return true;
  }
  // Print out the produced results.
  for (unsigned FileIndex = 0; FileIndex < Replacements.NumFileReplacementSets;
       ++FileIndex) {
    const CXRefactoringFileReplacementSet &FileSet =
        Replacements.FileReplacementSets[FileIndex];
    if (opts::Apply) {
      apply(llvm::makeArrayRef(FileSet.Replacements, FileSet.NumReplacements),
            clang_getCString(FileSet.Filename));
      continue;
    }
    for (unsigned I = 0; I < FileSet.NumReplacements; ++I) {
      const CXRefactoringReplacement &Replacement = FileSet.Replacements[I];

      if (Continuation) {
        // Always print the filenames in with continuations.
        outs() << '"' << clang_getCString(FileSet.Filename) << "\" ";
      }
      outs() << '"';
      printEscaped(clang_getCString(Replacement.ReplacementString), outs());
      outs() << "\" ";
      CXFileRange Range = Replacement.Range;
      outs() << Range.Begin.Line << ":" << Range.Begin.Column << " -> "
             << Range.End.Line << ":" << Range.End.Column;
      if (opts::initiateAndPerform::EmitAssociatedInfo) {
        CXRefactoringReplacementAssociatedSymbolOccurrences Info =
            clang_RefactoringReplacement_getAssociatedSymbolOccurrences(
                Replacement);
        for (const CXSymbolOccurrence &SymbolOccurrence :
             llvm::makeArrayRef(Info.AssociatedSymbolOccurrences,
                                Info.NumAssociatedSymbolOccurrences)) {
          outs() << " [Symbol " << renameOccurrenceKindString(
                                       SymbolOccurrence.Kind, /*IsLocal*/ false,
                                       SymbolOccurrence.IsMacroExpansion)
                 << ' ' << SymbolOccurrence.SymbolIndex;
          for (const auto &Piece :
               llvm::makeArrayRef(SymbolOccurrence.NamePieces,
                                  SymbolOccurrence.NumNamePieces)) {
            outs() << ' ' << Piece.Begin.Line << ":" << Piece.Begin.Column
                   << " -> " << Piece.End.Line << ":" << Piece.End.Column;
          }
          outs() << ']';
        }
      }
      outs() << "\n";
    }
  }
  return false;
}

/// Returns the last column number of a line in a file.
static std::string queryResultsForFile(StringRef Filename, StringRef Name,
                                       StringRef FileSubstitution) {
  auto Buf = llvm::MemoryBuffer::getFile(Filename);
  if (!Buf)
    return "<invalid>";
  StringRef Buffer = (*Buf)->getBuffer();
  std::string Label = Name.str() + ":";
  size_t I = Buffer.find(Label);
  if (I == StringRef::npos)
    return "<invalid>";
  I = I + Label.size();
  auto Result = Buffer.substr(I, Buffer.find('\n', I) - I);
  std::string Sub1 = llvm::Regex("%s").sub(FileSubstitution, Result);
  return llvm::Regex("%S").sub(llvm::sys::path::parent_path(FileSubstitution),
                               Sub1);
}

static Optional<std::pair<unsigned, unsigned>>
findSelectionLocInSource(StringRef Buffer, StringRef Label) {
  size_t I = Buffer.find(Label);
  if (I == StringRef::npos)
    return None;
  I = I + Label.size();
  auto LocParts =
      Buffer.substr(I, Buffer.find_first_of("\n/", I) - I).trim().split(":");
  unsigned CurrentLine = Buffer.take_front(I).count('\n') + 1;
  if (LocParts.second.empty())
    return None;
  StringRef LineString = LocParts.first;
  unsigned Line, Column;
  enum ExprKind { Literal, Add, Sub };
  ExprKind Expr = LineString.startswith("+")
                      ? Add
                      : LineString.startswith("-") ? Sub : Literal;
  if (LineString.drop_front(Expr != Literal ? 1 : 0).getAsInteger(10, Line))
    return None;
  if (Expr == Add)
    Line += CurrentLine;
  else if (Expr == Sub)
    Line = CurrentLine - Line;
  if (LocParts.second.getAsInteger(10, Column))
    return None;
  return std::make_pair(Line, Column);
}

static Optional<ParsedSourceLocation> selectionLocForFile(StringRef Filename,
                                                          StringRef Name) {
  auto Buf = llvm::MemoryBuffer::getFile(Filename);
  if (!Buf)
    return None;

  StringRef Buffer = (*Buf)->getBuffer();
  std::string Label = Name.str() + ":";
  auto Start = findSelectionLocInSource(Buffer, Label);
  if (!Start)
    return None;
  // Create the resulting source location.
  // FIXME: Parse can be avoided.
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << Filename << ":" << Start->first << ":" << Start->second;
  return ParsedSourceLocation::FromString(OS.str());
}

static Optional<OldParsedSourceRange> selectionRangeForFile(StringRef Filename,
                                                         StringRef Name) {
  auto Buf = llvm::MemoryBuffer::getFile(Filename);
  if (!Buf)
    return None;

  StringRef Buffer = (*Buf)->getBuffer();
  std::string BeginLabel = Name.str() + "-begin:";
  std::string EndLabel = Name.str() + "-end:";
  auto Start = findSelectionLocInSource(Buffer, BeginLabel);
  auto End = findSelectionLocInSource(Buffer, EndLabel);
  if (!Start || !End)
    return None;
  // Create the resulting source range.
  // FIXME: Parse can be avoided.
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << Filename << ":" << Start->first << ":" << Start->second << "-"
     << End->first << ":" << End->second;
  return OldParsedSourceRange::FromString(OS.str());
}

bool performOperation(CXRefactoringAction Action, ArrayRef<const char *> Args,
                      CXIndex CIdx) {
  if (opts::initiateAndPerform::CandidateIndex.getNumOccurrences()) {
    if (clang_RefactoringAction_selectRefactoringCandidate(
            Action, opts::initiateAndPerform::CandidateIndex)) {
      errs() << "error: failed to select the refactoring candidate!\n";
      return true;
    }
  }
  CXRefactoringOptionSet Options = nullptr;
  CXString FailureReason;
  CXRefactoringResult Result = clang_Refactoring_performOperation(
      Action, Args.data(), Args.size(), nullptr, 0, Options, &FailureReason);
  if (!Result) {
    errs() << "error: failed to perform the refactoring operation";
    if (const char *Reason = clang_getCString(FailureReason))
      errs() << " (" << Reason << ')';
    errs() << "!\n";
    clang_disposeString(FailureReason);
    return true;
  }
  CXRefactoringContinuation Continuation =
      clang_RefactoringResult_getContinuation(Result);
  bool AreReplacementsInvalid =
      printRefactoringReplacements(Result, Continuation, Continuation);
  clang_RefactoringResult_dispose(Result);
  if (AreReplacementsInvalid) {
    clang_RefactoringContinuation_dispose(Continuation);
    return true;
  }
  if (!Continuation)
    return false;
  assert(clang_RefactoringContinuation_getNumIndexerQueries(Continuation) !=
             0 &&
         "Missing indexer queries?");
  std::string QueryResults = queryResultsForFile(
      opts::FileName, opts::initiateAndPerform::QueryResults,
      /*FileSubstitution=*/opts::initiateAndPerform::ContinuationFile);
  clang_RefactoringContinuation_loadSerializedIndexerQueryResults(
      Continuation, /*Source=*/QueryResults.c_str());
  CXDiagnosticSet Diags =
      clang_RefactoringContinuation_verifyBeforeFinalizing(Continuation);
  if (Diags) {
    llvm::errs() << "error: continuation failed: ";
    for (unsigned I = 0, E = clang_getNumDiagnosticsInSet(Diags); I != E; ++I) {
      CXDiagnostic Diag = clang_getDiagnosticInSet(Diags, I);
      CXString Spelling = clang_getDiagnosticSpelling(Diag);
      errs() << clang_getCString(Spelling) << "\n";
      clang_disposeString(Spelling);
      clang_disposeDiagnostic(Diag);
    }
    clang_RefactoringContinuation_dispose(Continuation);
    clang_disposeDiagnosticSet(Diags);
    return true;
  }
  clang_RefactoringContinuation_finalizeEvaluationInInitationTU(Continuation);
  // Load the continuation TU.
  CXTranslationUnit ContinuationTU;
  CXErrorCode Err = clang_parseTranslationUnit2(
      CIdx, opts::initiateAndPerform::ContinuationFile.c_str(), Args.data(),
      Args.size(), 0, 0, CXTranslationUnit_KeepGoing, &ContinuationTU);
  if (Err != CXError_Success) {
    errs() << "error: failed to load '"
           << opts::initiateAndPerform::ContinuationFile.c_str() << "'\n";
    clang_RefactoringContinuation_dispose(Continuation);
    return true;
  }
  Result = clang_RefactoringContinuation_continueOperationInTU(
      Continuation, ContinuationTU, &FailureReason);
  if (!Result) {
    errs() << "error: failed to perform the refactoring continuation";
    if (const char *Reason = clang_getCString(FailureReason))
      errs() << " (" << Reason << ')';
    errs() << "!\n";
    clang_disposeString(FailureReason);
    clang_disposeTranslationUnit(ContinuationTU);
    clang_RefactoringContinuation_dispose(Continuation);
    return true;
  }
  // FIXME: Continuations can be chained in the future.
  AreReplacementsInvalid =
      printRefactoringReplacements(Result, Continuation, nullptr);
  clang_RefactoringResult_dispose(Result);
  clang_disposeTranslationUnit(ContinuationTU);
  clang_RefactoringContinuation_dispose(Continuation);
  return AreReplacementsInvalid;
}

int initiateAndPerformAction(CXTranslationUnit TU, ArrayRef<const char *> Args,
                             CXIndex CIdx) {
  std::vector<ParsedSourceLineRange> Ranges;
  std::vector<OldParsedSourceRange> SelectionRanges;
  for (const auto &Range : opts::initiateAndPerform::InLocationRanges) {
    auto ParsedLineRange = ParsedSourceLineRange::FromString(Range);
    if (!ParsedLineRange) {
      errs()
          << "error: The -in option must use the <file:line:column[-column]> "
             "format\n";
      return 1;
    }
    Ranges.push_back(ParsedLineRange.getValue());
  }
  for (const auto &Range : opts::initiateAndPerform::AtLocations) {
    if (!StringRef(Range).contains(':')) {
      auto ParsedLocation = selectionLocForFile(opts::FileName, Range);
      if (!ParsedLocation) {
        errs() << "error: The -at option must use the <file:line:column> "
                  "format\n";
        return 1;
      }
      Ranges.push_back(*ParsedLocation);
      continue;
    }
    // TODO: Remove old location in arguments in favour of new testing
    // locations.
    auto ParsedLocation = ParsedSourceLocation::FromString(Range);
    if (ParsedLocation.FileName.empty()) {
      errs() << "error: The -at option must use the <file:line:column> "
                "format\n";
      return 1;
    }
    Ranges.push_back(ParsedLocation);
  }
  for (const auto &Range : opts::initiateAndPerform::SelectedRanges) {
    auto ParsedRange = StringRef(Range).contains(':')
                           ? OldParsedSourceRange::FromString(Range)
                           : selectionRangeForFile(opts::FileName, Range);
    if (!ParsedRange) {
      errs() << "error: The -selected option must use the "
                "<file:line:column-line:column> format or refer to the name of "
                "the selection specifier in the source\n";
      return 1;
    }
    SelectionRanges.push_back(ParsedRange.getValue());
  }
  if (Ranges.empty() && SelectionRanges.empty()) {
    errs() << "error: -in or -at options must be specified at least once!";
    return 1;
  }
  if (!Ranges.empty() && !SelectionRanges.empty()) {
    errs() << "error: -in or -at options can't be used with -selected!";
    return 1;
  }

  auto ActionTypeOrNone = StringSwitch<Optional<CXRefactoringActionType>>(
                              opts::initiateAndPerform::ActionName)
#define REFACTORING_OPERATION_ACTION(Name, Spelling, Command)                  \
  .Case(Command, CXRefactor_##Name)
#define REFACTORING_OPERATION_SUB_ACTION(Name, Parent, Spelling, Command)      \
  .Case(Command, CXRefactor_##Parent##_##Name)
#include "clang/Tooling/Refactor/RefactoringActions.def"
                              .Default(None);
  if (!ActionTypeOrNone) {
    errs() << "error: invalid action '" << opts::initiateAndPerform::ActionName
           << "'\n";
    return 1;
  }
  CXRefactoringActionType ActionType = *ActionTypeOrNone;

  Optional<bool> Initiated;
  Optional<std::string> InitiationFailureReason;
  Optional<std::string> LocationCandidateInformation;
  auto InitiateAndPerform =
      [&](const ParsedSourceLocation &Location, unsigned Column,
          Optional<OldParsedSourceRange> SelectionRange = None) -> bool {
    CXSourceLocation Loc =
        clang_getLocation(TU, clang_getFile(TU, Location.FileName.c_str()),
                          Location.Line, Column);
    CXSourceRange Range;
    if (SelectionRange) {
      auto Begin = SelectionRange.getValue().Begin;
      auto End = SelectionRange.getValue().End;
      CXFile File = clang_getFile(TU, Begin.FileName.c_str());
      Range =
          clang_getRange(clang_getLocation(TU, File, Begin.Line, Begin.Column),
                         clang_getLocation(TU, File, End.Line, End.Column));
    } else
      Range = clang_getNullRange();
    CXRefactoringAction Action;
    CXString FailureReason;
    CXErrorCode Err = clang_Refactoring_initiateActionAt(
        TU, Loc, Range, ActionType, /*Options=*/nullptr, &Action,
        &FailureReason);
    std::string ReasonString;
    if (const char *Reason = clang_getCString(FailureReason))
      ReasonString = Reason;
    clang_disposeString(FailureReason);
    if (InitiationFailureReason.hasValue() &&
        InitiationFailureReason.getValue() != ReasonString) {
      errs() << "error: inconsistent results in a single action range!\n";
      return true;
    }
    InitiationFailureReason = std::move(ReasonString);
    if (Err == CXError_RefactoringActionUnavailable) {
      if (Initiated.hasValue() && Initiated.getValue()) {
        errs() << "error: inconsistent results in a single action range!\n";
        return true;
      }
      Initiated = false;
    } else if (Err != CXError_Success)
      return true;
    else if (Initiated.hasValue() && !Initiated.getValue()) {
      errs() << "error: inconsistent results in a single action range!\n";
      return true;
    } else
      Initiated = true;

    CXRefactoringCandidateSet Candidates;
    if (clang_RefactoringAction_getRefactoringCandidates(Action, &Candidates) ==
            CXError_Success &&
        Candidates.NumCandidates > 1) {
      std::string CandidateString = refactoringCandidatesToString(Candidates);
      if (LocationCandidateInformation) {
        if (*LocationCandidateInformation != CandidateString) {
          errs() << "error: inconsistent results in a single action range!\n";
          return true;
        }
      } else
        LocationCandidateInformation = CandidateString;
    } else if (opts::InitiateActionSubcommand &&
               !opts::initiateAndPerform::LocationAgnostic) {
      CXSourceRange Range =
          clang_RefactoringAction_getSourceRangeOfInterest(Action);
      std::string LocationString =
          std::string("at ") +
          (!clang_Range_isNull(Range)
               ? SelectionRange ? rangeToString(Range)
                                : locationToString(clang_getRangeStart(Range))
               : "<unknown>");
      if (!LocationCandidateInformation.hasValue())
        LocationCandidateInformation = LocationString;
      else if (LocationCandidateInformation.getValue() != LocationString) {
        errs() << "error: inconsistent results in a single action range!\n";
        return true;
      }
    }

    if (!*Initiated)
      return false;

    bool Failed = opts::PerformActionSubcommand
                      ? performOperation(Action, Args, CIdx)
                      : false;
    clang_RefactoringAction_dispose(Action);
    return Failed;
  };

  // Iterate over all of the possible locations and perform the initiation
  // at each range.
  for (const ParsedSourceLineRange &LineRange : Ranges) {
    for (unsigned Column = LineRange.Column; Column <= LineRange.MaxColumn;
         ++Column) {
      if (InitiateAndPerform(LineRange, Column))
        return 1;
    }
  }

  for (const OldParsedSourceRange &SelectionRange : SelectionRanges) {
    if (InitiateAndPerform(SelectionRange.Begin, SelectionRange.Begin.Column,
                           SelectionRange))
      return 1;
  }

  if (!Initiated.getValue()) {
    errs() << "Failed to initiate the refactoring action";
    if (InitiationFailureReason.hasValue() &&
        !InitiationFailureReason.getValue().empty())
      errs() << " (" << InitiationFailureReason.getValue() << ')';
    errs() << "!\n";
    return 1;
  }
  if (opts::InitiateActionSubcommand) {
    outs() << "Initiated the '" << opts::initiateAndPerform::ActionName
           << "' action";
    if (!opts::initiateAndPerform::LocationAgnostic)
      outs() << ' ' << LocationCandidateInformation.getValue();
    outs() << "\n";
  }
  return 0;
}

struct MainTU {
  CXIndex CIdx;
  CXTranslationUnit TU = nullptr;

  MainTU() {
    CIdx = clang_createIndex(0, 0);
  }
  ~MainTU() {
    if (TU)
      clang_disposeTranslationUnit(TU);
    clang_disposeIndex(CIdx);
  }
};

int main(int argc, const char **argv) {
  cl::HideUnrelatedOptions(opts::ClangRefactorTestOptions);

  cl::ParseCommandLineOptions(argc, argv, "Clang refactoring test tool\n");
  cl::PrintOptionValues();

  MainTU MainTranslationUnit;

  std::vector<const char *> Args;
  for (const auto &Arg : opts::CompilerArguments) {
    Args.push_back(Arg.c_str());
  }
  CXErrorCode Err = clang_parseTranslationUnit2(
      MainTranslationUnit.CIdx,
      opts::IgnoreFilenameForInitiationTU ? nullptr : opts::FileName.c_str(),
      Args.data(), Args.size(), 0, 0, CXTranslationUnit_KeepGoing,
      &MainTranslationUnit.TU);
  if (Err != CXError_Success) {
    errs() << "error: failed to load '" << opts::FileName << "'\n";
    return 1;
  }

  if (opts::RenameInitiateSubcommand || opts::RenameInitiateUSRSubcommand)
    return rename(MainTranslationUnit.TU, MainTranslationUnit.CIdx, Args);
  else if (opts::RenameIndexedFileSubcommand)
    return renameIndexedFile(MainTranslationUnit.CIdx, Args);
  else if (opts::ListRefactoringActionsSubcommand)
    return listRefactoringActions(MainTranslationUnit.TU);
  else if (opts::InitiateActionSubcommand || opts::PerformActionSubcommand)
    return initiateAndPerformAction(MainTranslationUnit.TU, Args,
                                    MainTranslationUnit.CIdx);

  return 0;
}
