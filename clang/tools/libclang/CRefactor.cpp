//===- CRefactor.cpp - Refactoring API hooks ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Clang-C refactoring library.
//
//===----------------------------------------------------------------------===//

#include "CIndexDiagnostic.h"
#include "CIndexer.h"
#include "CLog.h"
#include "CXCursor.h"
#include "CXSourceLocation.h"
#include "CXString.h"
#include "CXTranslationUnit.h"
#include "clang-c/Refactor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/DiagnosticCategories.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Refactor/IndexerQuery.h"
#include "clang/Tooling/Refactor/RefactoringActionFinder.h"
#include "clang/Tooling/Refactor/RefactoringActions.h"
#include "clang/Tooling/Refactor/RefactoringOperation.h"
#include "clang/Tooling/Refactor/RefactoringOptions.h"
#include "clang/Tooling/Refactor/RenameIndexedFile.h"
#include "clang/Tooling/Refactor/RenamingOperation.h"
#include "clang/Tooling/Refactor/SymbolOccurrenceFinder.h"
#include "clang/Tooling/Refactor/USRFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include <set>
#include <vector>

using namespace clang;
using namespace clang::tooling;

static RefactoringActionType
translateRefactoringActionType(CXRefactoringActionType Action) {
  switch (Action) {
#define REFACTORING_ACTION(Name, Spelling)                                     \
  case CXRefactor_##Name:                                                      \
    return RefactoringActionType::Name;
#include "clang/Tooling/Refactor/RefactoringActions.def"
  }
  llvm_unreachable("unknown CXRefactoringActionType value");
}

static CXRefactoringActionType
translateRefactoringActionType(RefactoringActionType Action) {
  switch (Action) {
#define REFACTORING_ACTION(Name, Spelling)                                     \
  case RefactoringActionType::Name:                                            \
    return CXRefactor_##Name;
#include "clang/Tooling/Refactor/RefactoringActions.def"
  }
  llvm_unreachable("unknown RefactoringActionType value");
}

static CXSymbolOccurrenceKind
translateOccurrenceKind(rename::OldSymbolOccurrence::OccurrenceKind Kind) {
  switch (Kind) {
  case rename::OldSymbolOccurrence::MatchingSymbol:
    return CXSymbolOccurrence_MatchingSymbol;
  case rename::OldSymbolOccurrence::MatchingSelector:
    return CXSymbolOccurrence_MatchingSelector;
  case rename::OldSymbolOccurrence::MatchingImplicitProperty:
    return CXSymbolOccurrence_MatchingImplicitProperty;
  case rename::OldSymbolOccurrence::MatchingComment:
    return CXSymbolOccurrence_MatchingCommentString;
  case rename::OldSymbolOccurrence::MatchingDocComment:
    return CXSymbolOccurrence_MatchingDocCommentString;
  case rename::OldSymbolOccurrence::MatchingFilename:
    return CXSymbolOccurrence_MatchingFilename;
  case rename::OldSymbolOccurrence::MatchingStringLiteral:
    return CXSymbolOccurrence_MatchingStringLiteral;
  }
  llvm_unreachable("unknown OccurrenceKind value");
}

namespace {

// TODO: Remove
class RenamingResult {
  struct RenamedNameString {
    CXString NewString;
    unsigned OldLength;
  };
  typedef SmallVector<RenamedNameString, 4> SymbolNameInfo;
  std::vector<SymbolNameInfo> NameInfo;

  /// The set of files that have to be modified.
  llvm::SmallVector<CXString, 2> Filenames;
  llvm::SpecificBumpPtrAllocator<CXRefactoringReplacement_Old> Replacements;
  std::vector<std::vector<CXRenamedSymbolOccurrence>> Occurrences;

  void addOccurrence(const rename::OldSymbolOccurrence &RenamedOccurrence,
                     const SourceManager &SM, const LangOptions &LangOpts) {
    CXRefactoringReplacement_Old *OccurrenceReplacements =
        Replacements.Allocate(RenamedOccurrence.locations().size());

    unsigned I = 0;
    const auto &SymbolNameInfo = NameInfo[RenamedOccurrence.SymbolIndex];
    if (!RenamedOccurrence.IsMacroExpansion &&
        RenamedOccurrence.Kind !=
            rename::OldSymbolOccurrence::MatchingComment &&
        RenamedOccurrence.Kind !=
            rename::OldSymbolOccurrence::MatchingDocComment)
      assert(RenamedOccurrence.locations().size() == SymbolNameInfo.size());
    for (const auto &Location : RenamedOccurrence.locations()) {
      CXSourceRange Range = cxloc::translateSourceRange(
          SM, LangOpts,
          CharSourceRange::getCharRange(RenamedOccurrence.getLocationRange(
              Location, SymbolNameInfo[I].OldLength)));
      CXFileLocation Begin, End;
      clang_getFileLocation(clang_getRangeStart(Range), nullptr, &Begin.Line,
                            &Begin.Column, nullptr);
      clang_getFileLocation(clang_getRangeEnd(Range), nullptr, &End.Line,
                            &End.Column, nullptr);

      OccurrenceReplacements[I] = CXRefactoringReplacement_Old{
          {Begin, End},
          RenamedOccurrence.IsMacroExpansion ? cxstring::createNull()
                                             : SymbolNameInfo[I].NewString};
      ++I;
    }

    Occurrences.back().push_back(CXRenamedSymbolOccurrence{
        OccurrenceReplacements, I,
        translateOccurrenceKind(RenamedOccurrence.Kind),
        RenamedOccurrence.IsMacroExpansion});
  }

public:
  RenamingResult(ArrayRef<OldSymbolName> NewNames,
                 ArrayRef<rename::Symbol> Symbols) {
    assert(NewNames.size() == Symbols.size());
    for (size_t I = 0, E = NewNames.size(); I != E; ++I) {
      const auto &NewName = NewNames[I];
      const auto &OldName = Symbols[I].Name;

      assert(NewName.size() == OldName.size());
      SymbolNameInfo Info;
      for (size_t I = 0, E = NewName.size(); I != E; ++I)
        Info.push_back(RenamedNameString{cxstring::createDup(NewName[I]),
                                         (unsigned)OldName[I].size()});
      NameInfo.push_back(std::move(Info));
    }
  }

  // FIXME: Don't duplicate code, Use just one constructor.
  RenamingResult(ArrayRef<OldSymbolName> NewNames,
                 ArrayRef<OldSymbolName> OldNames) {
    assert(NewNames.size() == OldNames.size());
    for (size_t I = 0, E = NewNames.size(); I != E; ++I) {
      const auto &NewName = NewNames[I];
      const auto &OldName = OldNames[I];

      assert(NewName.size() == OldName.size());
      SymbolNameInfo Info;
      for (size_t I = 0, E = NewName.size(); I != E; ++I)
        Info.push_back(RenamedNameString{cxstring::createDup(NewName[I]),
                                         (unsigned)OldName[I].size()});
      NameInfo.push_back(std::move(Info));
    }
  }

  ~RenamingResult() {
    for (const auto &SymbolInfo : NameInfo)
      for (const auto &NameString : SymbolInfo)
        clang_disposeString(NameString.NewString);
    for (const auto &Filename : Filenames)
      clang_disposeString(Filename);
  }

  void
  handleTUResults(CXTranslationUnit TU,
                  llvm::MutableArrayRef<rename::OldSymbolOccurrence> Results) {
    ASTUnit *Unit = cxtu::getASTUnit(TU);
    assert(Unit && "Invalid TU");
    auto &Ctx = Unit->getASTContext();

    // Find the set of files that have to be modified and gather the indices of
    // the occurrences for each file.
    const SourceManager &SM = Ctx.getSourceManager();
    typedef std::set<rename::OldSymbolOccurrence> OccurrenceSet;
    llvm::StringMap<OccurrenceSet> FilenamesToSymbolOccurrences;
    for (auto &Occurrence : Results) {
      const std::pair<FileID, unsigned> DecomposedLocation =
          SM.getDecomposedLoc(Occurrence.locations()[0]);
      const FileEntry *Entry = SM.getFileEntryForID(DecomposedLocation.first);
      assert(Entry && "Invalid file entry");
      auto &FileOccurrences =
          FilenamesToSymbolOccurrences
              .try_emplace(Entry->getName(), OccurrenceSet())
              .first->getValue();
      FileOccurrences.insert(std::move(Occurrence));
    }

    // Create the filenames
    for (const auto &FilenameCount : FilenamesToSymbolOccurrences)
      Filenames.push_back(cxstring::createDup(FilenameCount.getKey()));

    unsigned FileIndex = 0;
    for (const auto &RenamedOccurrences : FilenamesToSymbolOccurrences) {
      assert(clang_getCString(Filenames[FileIndex]) ==
                 RenamedOccurrences.getKey() &&
             "Unstable iteration order");
      Occurrences.push_back(std::vector<CXRenamedSymbolOccurrence>());
      for (const auto &Occurrence : RenamedOccurrences.getValue())
        addOccurrence(Occurrence, SM, Ctx.getLangOpts());
      ++FileIndex;
    }
  }

  void addMainFilename(const SourceManager &SM) {
    assert(Filenames.empty() && "Main filename should be added only once");
    Filenames.push_back(cxstring::createDup(
        SM.getFileEntryForID(SM.getMainFileID())->getName()));
    Occurrences.push_back(std::vector<CXRenamedSymbolOccurrence>());
  }

  void
  handleSingleFileTUResults(const ASTContext &Ctx,
                            ArrayRef<rename::OldSymbolOccurrence> Occurrences) {
    addMainFilename(Ctx.getSourceManager());
    for (const auto &Occurrence : Occurrences)
      addOccurrence(Occurrence, Ctx.getSourceManager(), Ctx.getLangOpts());
  }

  void
  handleIndexedFileOccurrence(const rename::OldSymbolOccurrence &Occurrence,
                              const SourceManager &SM,
                              const LangOptions &LangOpts) {
    if (Filenames.empty()) {
      addMainFilename(SM);
    }
    addOccurrence(Occurrence, SM, LangOpts);
  }

  ArrayRef<CXRenamedSymbolOccurrence> getOccurrences(unsigned FileIndex) const {
    return Occurrences[FileIndex];
  }

  ArrayRef<CXString> getFilenames() const { return Filenames; }
};

class SymbolOccurrencesResult {
  struct SymbolNamePiece {
    unsigned OldLength;
  };
  typedef SmallVector<SymbolNamePiece, 4> SymbolNameInfo;
  std::vector<SymbolNameInfo> NameInfo;

  /// The set of files that have to be modified.
  llvm::SmallVector<CXString, 2> Filenames;
  llvm::SpecificBumpPtrAllocator<CXFileRange> Ranges;
  std::vector<std::vector<CXSymbolOccurrence>> SymbolOccurrences;

  void addOccurrence(const rename::OldSymbolOccurrence &RenamedOccurrence,
                     const SourceManager &SM, const LangOptions &LangOpts) {
    ArrayRef<SourceLocation> Locations = RenamedOccurrence.locations();
    CXFileRange *OccurrenceRanges = Ranges.Allocate(Locations.size());

    unsigned I = 0;
    const auto &SymbolNameInfo = NameInfo[RenamedOccurrence.SymbolIndex];
    if (!RenamedOccurrence.IsMacroExpansion &&
        RenamedOccurrence.Kind !=
            rename::OldSymbolOccurrence::MatchingComment &&
        RenamedOccurrence.Kind !=
            rename::OldSymbolOccurrence::MatchingDocComment)
      assert(Locations.size() == SymbolNameInfo.size());
    for (const auto &Location : Locations) {
      CXSourceRange Range = cxloc::translateSourceRange(
          SM, LangOpts,
          CharSourceRange::getCharRange(RenamedOccurrence.getLocationRange(
              Location, SymbolNameInfo[I].OldLength)));
      CXFileLocation Begin, End;
      clang_getFileLocation(clang_getRangeStart(Range), nullptr, &Begin.Line,
                            &Begin.Column, nullptr);
      clang_getFileLocation(clang_getRangeEnd(Range), nullptr, &End.Line,
                            &End.Column, nullptr);
      OccurrenceRanges[I] = CXFileRange{Begin, End};
      ++I;
    }

    SymbolOccurrences.back().push_back(CXSymbolOccurrence{
        OccurrenceRanges, /*NumNamePieces=*/I,
        translateOccurrenceKind(RenamedOccurrence.Kind),
        RenamedOccurrence.IsMacroExpansion, RenamedOccurrence.SymbolIndex});
  }

public:
  SymbolOccurrencesResult(ArrayRef<rename::Symbol> Symbols) {
    for (const auto &Symbol : Symbols) {
      const OldSymbolName &Name = Symbol.Name;
      SymbolNameInfo Info;
      for (size_t I = 0, E = Name.size(); I != E; ++I)
        Info.push_back(SymbolNamePiece{(unsigned)Name[I].size()});
      NameInfo.push_back(std::move(Info));
    }
  }

  SymbolOccurrencesResult(ArrayRef<OldSymbolName> Names) {
    for (const OldSymbolName &Name : Names) {
      SymbolNameInfo Info;
      for (size_t I = 0, E = Name.size(); I != E; ++I)
        Info.push_back(SymbolNamePiece{(unsigned)Name[I].size()});
      NameInfo.push_back(std::move(Info));
    }
  }

  ~SymbolOccurrencesResult() {
    for (const auto &Filename : Filenames)
      clang_disposeString(Filename);
  }

  void
  handleTUResults(CXTranslationUnit TU,
                  llvm::MutableArrayRef<rename::OldSymbolOccurrence> Results) {
    ASTUnit *Unit = cxtu::getASTUnit(TU);
    assert(Unit && "Invalid TU");
    auto &Ctx = Unit->getASTContext();

    // Find the set of files that have to be modified and gather the indices of
    // the occurrences for each file.
    const SourceManager &SM = Ctx.getSourceManager();
    typedef std::set<rename::OldSymbolOccurrence> OccurrenceSet;
    llvm::StringMap<OccurrenceSet> FilenamesToSymbolOccurrences;
    for (auto &Occurrence : Results) {
      const std::pair<FileID, unsigned> DecomposedLocation =
          SM.getDecomposedLoc(Occurrence.locations()[0]);
      const FileEntry *Entry = SM.getFileEntryForID(DecomposedLocation.first);
      assert(Entry && "Invalid file entry");
      auto &FileOccurrences =
          FilenamesToSymbolOccurrences
              .try_emplace(Entry->getName(), OccurrenceSet())
              .first->getValue();
      FileOccurrences.insert(std::move(Occurrence));
    }

    // Create the filenames
    for (const auto &FilenameCount : FilenamesToSymbolOccurrences)
      Filenames.push_back(cxstring::createDup(FilenameCount.getKey()));

    unsigned FileIndex = 0;
    for (const auto &RenamedOccurrences : FilenamesToSymbolOccurrences) {
      assert(clang_getCString(Filenames[FileIndex]) ==
                 RenamedOccurrences.getKey() &&
             "Unstable iteration order");
      SymbolOccurrences.push_back(std::vector<CXSymbolOccurrence>());
      for (const auto &Occurrence : RenamedOccurrences.getValue())
        addOccurrence(Occurrence, SM, Ctx.getLangOpts());
      ++FileIndex;
    }
  }

  void addMainFilename(const SourceManager &SM) {
    assert(Filenames.empty() && "Main filename should be added only once");
    Filenames.push_back(cxstring::createDup(
        SM.getFileEntryForID(SM.getMainFileID())->getName()));
    SymbolOccurrences.push_back(std::vector<CXSymbolOccurrence>());
  }

  void
  handleIndexedFileOccurrence(const rename::OldSymbolOccurrence &Occurrence,
                              const SourceManager &SM,
                              const LangOptions &LangOpts) {
    if (Filenames.empty()) {
      addMainFilename(SM);
    }
    addOccurrence(Occurrence, SM, LangOpts);
  }

  ArrayRef<CXSymbolOccurrence> getOccurrences(unsigned FileIndex) const {
    return SymbolOccurrences[FileIndex];
  }

  ArrayRef<CXString> getFilenames() const { return Filenames; }
};

class RenamingAction {
public:
  LangOptions LangOpts;
  IdentifierTable IDs;
  // TODO: Remove
  SmallVector<OldSymbolName, 4> NewNames;
  SymbolOperation Operation;

  RenamingAction(const LangOptions &LangOpts, SymbolOperation Operation)
      : LangOpts(LangOpts), IDs(LangOpts), Operation(std::move(Operation)) {}

  /// \brief Sets the new renaming name and returns CXError_Success on success.
  // TODO: Remove
  CXErrorCode setNewName(StringRef Name) {
    OldSymbolName NewSymbolName(Name, LangOpts);
    if (NewSymbolName.size() != Operation.symbols()[0].Name.size())
      return CXError_RefactoringNameSizeMismatch;
    if (!rename::isNewNameValid(NewSymbolName, Operation, IDs, LangOpts))
      return CXError_RefactoringNameInvalid;
    rename::determineNewNames(std::move(NewSymbolName), Operation, NewNames,
                              LangOpts);
    return CXError_Success;
  }

  // TODO: Remove
  CXString usrForSymbolAt(unsigned Index) {
    llvm::SmallVector<char, 128> Buff;
    if (index::generateUSRForDecl(Operation.symbols()[Index].FoundDecl, Buff))
      return cxstring::createNull();
    return cxstring::createDup(StringRef(Buff.begin(), Buff.size()));
  }

  // TODO: Remove
  CXString getUSRThatRequiresImplementationTU() {
    llvm::SmallVector<char, 128> Buff;
    if (!Operation.requiresImplementationTU() ||
        index::generateUSRForDecl(Operation.declThatRequiresImplementationTU(),
                                  Buff))
      return cxstring::createNull();
    return cxstring::createDup(StringRef(Buff.begin(), Buff.size()));
  }

  // TODO: Remove
  RenamingResult *handlePrimaryTU(CXTranslationUnit TU, ASTUnit &Unit) {
    // Perform the renaming.
    if (NewNames.empty())
      return nullptr;

    const ASTContext &Context = Unit.getASTContext();
    auto Occurrences = rename::findSymbolOccurrences(
        Operation, Context.getTranslationUnitDecl());
    auto *Result = new RenamingResult(NewNames, Operation.symbols());
    Result->handleTUResults(TU, Occurrences);
    return Result;
  }

  SymbolOccurrencesResult *findSymbolsInInitiationTU(CXTranslationUnit TU,
                                                     ASTUnit &Unit) {
    const ASTContext &Context = Unit.getASTContext();
    auto Occurrences = rename::findSymbolOccurrences(
        Operation, Context.getTranslationUnitDecl());
    auto *Result = new SymbolOccurrencesResult(Operation.symbols());
    Result->handleTUResults(TU, Occurrences);
    return Result;
  }
};

static bool isObjCSelectorKind(CXCursorKind Kind) {
  return Kind == CXCursor_ObjCInstanceMethodDecl ||
         Kind == CXCursor_ObjCClassMethodDecl ||
         Kind == CXCursor_ObjCMessageExpr;
}

// TODO: Remove
static bool isObjCSelector(const CXRenamedIndexedSymbol &Symbol) {
  if (isObjCSelectorKind(Symbol.CursorKind))
    return true;
  for (const auto &Occurrence : llvm::makeArrayRef(
           Symbol.IndexedLocations, Symbol.IndexedLocationCount)) {
    if (isObjCSelectorKind(Occurrence.CursorKind))
      return true;
  }
  return false;
}

static bool isObjCSelector(const CXIndexedSymbol &Symbol) {
  if (isObjCSelectorKind(Symbol.CursorKind))
    return true;
  for (const auto &Occurrence : llvm::makeArrayRef(
           Symbol.IndexedLocations, Symbol.IndexedLocationCount)) {
    if (isObjCSelectorKind(Occurrence.CursorKind))
      return true;
  }
  return false;
}

// New names are initialized and verified after the LangOptions are created.
CXErrorCode computeNewNames(ArrayRef<CXRenamedIndexedSymbol> Symbols,
                            ArrayRef<OldSymbolName> SymbolNames,
                            const LangOptions &LangOpts,
                            SmallVectorImpl<OldSymbolName> &NewNames) {
  IdentifierTable IDs(LangOpts);
  for (const auto &Symbol : Symbols) {
    OldSymbolName NewSymbolName(Symbol.NewName, LangOpts);
    if (NewSymbolName.size() != SymbolNames[0].size())
      return CXError_RefactoringNameSizeMismatch;
    if (!rename::isNewNameValid(NewSymbolName, isObjCSelector(Symbol), IDs,
                                LangOpts))
      return CXError_RefactoringNameInvalid;
    NewNames.push_back(std::move(NewSymbolName));
  }
  return CXError_Success;
}

static rename::IndexedOccurrence::OccurrenceKind
translateIndexedOccurrenceKind(CXCursorKind Kind) {
  switch (Kind) {
  case CXCursor_ObjCMessageExpr:
    return rename::IndexedOccurrence::IndexedObjCMessageSend;
  case CXCursor_InclusionDirective:
    return rename::IndexedOccurrence::InclusionDirective;
  default:
    return rename::IndexedOccurrence::IndexedSymbol;
  }
}

/// ClangTool::run is not thread-safe, so we have to guard it.
static llvm::ManagedStatic<llvm::sys::Mutex> ClangToolConstructionMutex;

// TODO: Remove
CXErrorCode performIndexedFileRename(
    ArrayRef<CXRenamedIndexedSymbol> Symbols, StringRef Filename,
    ArrayRef<const char *> Arguments, CXIndex CIdx,
    MutableArrayRef<CXUnsavedFile> UnsavedFiles,
    const RefactoringOptionSet *Options, CXRenamingResult &Result) {
  Result = nullptr;

  // Adjust the given command line arguments to ensure that any positional
  // arguments in them are stripped.
  std::vector<const char *> ClangToolArguments;
  ClangToolArguments.push_back("--");
  for (const auto &Arg : Arguments) {
    // Remove the '-gmodules' option, as the -fmodules-format=obj isn't
    // supported without the linked object reader.
    if (StringRef(Arg) == "-gmodules")
      continue;
    ClangToolArguments.push_back(Arg);
  }
  int Argc = ClangToolArguments.size();
  std::string ErrorMessage;
  std::unique_ptr<CompilationDatabase> Compilations =
      FixedCompilationDatabase::loadFromCommandLine(
          Argc, ClangToolArguments.data(), ErrorMessage);
  if (!Compilations) {
    llvm::errs() << "CRefactor: Failed to load command line: " << ErrorMessage
                 << "\n";
    return CXError_Failure;
  }

  // Translate the symbols.
  llvm::SmallVector<rename::IndexedSymbol, 4> IndexedSymbols;
  for (const auto &Symbol : Symbols) {

    // Parse the symbol name.
    bool IsObjCSelector = false;
    // Selectors have to be parsed.
    if (isObjCSelector(Symbol))
      IsObjCSelector = true;
    // Ensure that we don't get selectors with incorrect symbol kind.
    else if (StringRef(Symbol.Name).contains(':'))
      return CXError_InvalidArguments;

    std::vector<rename::IndexedOccurrence> IndexedOccurrences;
    for (const auto &Loc : llvm::makeArrayRef(Symbol.IndexedLocations,
                                              Symbol.IndexedLocationCount)) {
      rename::IndexedOccurrence Result;
      Result.Line = Loc.Location.Line;
      Result.Column = Loc.Location.Column;
      Result.Kind = translateIndexedOccurrenceKind(Loc.CursorKind);
      IndexedOccurrences.push_back(Result);
    }

    IndexedSymbols.emplace_back(OldSymbolName(Symbol.Name, IsObjCSelector),
                                IndexedOccurrences,
                                /*IsObjCSelector=*/IsObjCSelector);
  }

  class ToolRunner final : public FrontendActionFactory,
                           public rename::IndexedFileOccurrenceConsumer {
    ArrayRef<CXRenamedIndexedSymbol> Symbols;
    ArrayRef<rename::IndexedSymbol> IndexedSymbols;
    rename::IndexedFileRenamerLock &Lock;
    const RefactoringOptionSet *Options;

  public:
    RenamingResult *Result;
    CXErrorCode Err;

    ToolRunner(ArrayRef<CXRenamedIndexedSymbol> Symbols,
               ArrayRef<rename::IndexedSymbol> IndexedSymbols,
               rename::IndexedFileRenamerLock &Lock,
               const RefactoringOptionSet *Options)
        : Symbols(Symbols), IndexedSymbols(IndexedSymbols), Lock(Lock),
          Options(Options), Result(nullptr), Err(CXError_Success) {}

    clang::FrontendAction *create() override {
      return new rename::IndexedFileOccurrenceProducer(IndexedSymbols, *this,
                                                       Lock, Options);
    }

    void handleOccurrence(const rename::OldSymbolOccurrence &Occurrence,
                          SourceManager &SM,
                          const LangOptions &LangOpts) override {
      if (Err != CXError_Success)
        return;
      if (!Result) {
        SmallVector<OldSymbolName, 4> SymbolNames;
        for (const auto &Symbol : IndexedSymbols)
          SymbolNames.push_back(Symbol.Name);
        SmallVector<OldSymbolName, 4> NewNames;
        Err = computeNewNames(Symbols, SymbolNames, LangOpts, NewNames);
        if (Err != CXError_Success)
          return;
        Result = new RenamingResult(NewNames, SymbolNames);
      }
      Result->handleIndexedFileOccurrence(Occurrence, SM, LangOpts);
    }
  };

  rename::IndexedFileRenamerLock Lock(*ClangToolConstructionMutex);
  auto Runner =
      llvm::make_unique<ToolRunner>(Symbols, IndexedSymbols, Lock, Options);

  // Run a clang tool on the input file.
  std::string Name = Filename.str();
  ClangTool Tool(*Compilations, Name);
  Tool.run(Runner.get());
  if (Runner->Err != CXError_Success)
    return Runner->Err;
  Result = Runner->Result;
  return CXError_Success;
}

CXErrorCode performIndexedSymbolSearch(
    ArrayRef<CXIndexedSymbol> Symbols, StringRef Filename,
    ArrayRef<const char *> Arguments, CXIndex CIdx,
    MutableArrayRef<CXUnsavedFile> UnsavedFiles,
    const RefactoringOptionSet *Options, CXSymbolOccurrencesResult &Result) {
  Result = nullptr;

  // Adjust the given command line arguments to ensure that any positional
  // arguments in them are stripped.
  std::vector<const char *> ClangToolArguments;
  ClangToolArguments.push_back("--");
  for (const auto &Arg : Arguments) {
    // Remove the '-gmodules' option, as the -fmodules-format=obj isn't
    // supported without the linked object reader.
    if (StringRef(Arg) == "-gmodules")
      continue;
    ClangToolArguments.push_back(Arg);
  }
  int Argc = ClangToolArguments.size();
  std::string ErrorMessage;
  std::unique_ptr<CompilationDatabase> Compilations =
      FixedCompilationDatabase::loadFromCommandLine(
          Argc, ClangToolArguments.data(), ErrorMessage);
  if (!Compilations) {
    llvm::errs() << "CRefactor: Failed to load command line: " << ErrorMessage
                 << "\n";
    return CXError_Failure;
  }

  // Translate the symbols.
  llvm::SmallVector<rename::IndexedSymbol, 4> IndexedSymbols;
  for (const auto &Symbol : Symbols) {

    // Parse the symbol name.
    bool IsObjCSelector = false;
    // Selectors have to be parsed.
    if (isObjCSelector(Symbol))
      IsObjCSelector = true;
    // Ensure that we don't get selectors with incorrect symbol kind.
    else if (StringRef(Symbol.Name).contains(':'))
      return CXError_InvalidArguments;

    std::vector<rename::IndexedOccurrence> IndexedOccurrences;
    for (const auto &Loc : llvm::makeArrayRef(Symbol.IndexedLocations,
                                              Symbol.IndexedLocationCount)) {
      rename::IndexedOccurrence Result;
      Result.Line = Loc.Location.Line;
      Result.Column = Loc.Location.Column;
      Result.Kind = translateIndexedOccurrenceKind(Loc.CursorKind);
      IndexedOccurrences.push_back(Result);
    }

    IndexedSymbols.emplace_back(
        OldSymbolName(Symbol.Name, IsObjCSelector), IndexedOccurrences,
        /*IsObjCSelector=*/IsObjCSelector,
        /*SearchForStringLiteralOccurrences=*/
        Symbol.CursorKind == CXCursor_ObjCInterfaceDecl);
  }

  class ToolRunner final : public FrontendActionFactory,
                           public rename::IndexedFileOccurrenceConsumer {
    ArrayRef<rename::IndexedSymbol> IndexedSymbols;
    rename::IndexedFileRenamerLock &Lock;
    const RefactoringOptionSet *Options;

  public:
    SymbolOccurrencesResult *Result;

    ToolRunner(ArrayRef<rename::IndexedSymbol> IndexedSymbols,
               rename::IndexedFileRenamerLock &Lock,
               const RefactoringOptionSet *Options)
        : IndexedSymbols(IndexedSymbols), Lock(Lock), Options(Options),
          Result(nullptr) {}

    clang::FrontendAction *create() override {
      return new rename::IndexedFileOccurrenceProducer(IndexedSymbols, *this,
                                                       Lock, Options);
    }

    void handleOccurrence(const rename::OldSymbolOccurrence &Occurrence,
                          SourceManager &SM,
                          const LangOptions &LangOpts) override {
      if (!Result) {
        SmallVector<OldSymbolName, 4> SymbolNames;
        for (const auto &Symbol : IndexedSymbols)
          SymbolNames.push_back(Symbol.Name);
        Result = new SymbolOccurrencesResult(SymbolNames);
      }
      Result->handleIndexedFileOccurrence(Occurrence, SM, LangOpts);
    }
  };

  rename::IndexedFileRenamerLock Lock(*ClangToolConstructionMutex);
  auto Runner = llvm::make_unique<ToolRunner>(IndexedSymbols, Lock, Options);

  // Run a clang tool on the input file.
  std::string Name = Filename.str();
  ClangTool Tool(*Compilations, Name);
  for (const CXUnsavedFile &File : UnsavedFiles)
    Tool.mapVirtualFile(File.Filename, StringRef(File.Contents, File.Length));
  if (Tool.run(Runner.get()))
    return CXError_Failure;
  Result = Runner->Result;
  return CXError_Success;
}

class RefactoringAction {
  std::unique_ptr<RefactoringOperation> Operation;
  std::unique_ptr<RenamingAction> Rename;

  SmallVector<CXRefactoringCandidate, 2> RefactoringCandidates;
  CXRefactoringCandidateSet CandidateSet = {nullptr, 0};
  bool HasCandidateSet = false;

public:
  CXRefactoringActionType Type;
  unsigned SelectedCandidate = 0;
  CXTranslationUnit InitiationTU;
  // TODO: Remove (no longer needed due to continuations).
  CXTranslationUnit ImplementationTU;

  RefactoringAction(std::unique_ptr<RefactoringOperation> Operation,
                    CXRefactoringActionType Type,
                    CXTranslationUnit InitiationTU)
      : Operation(std::move(Operation)), Type(Type), InitiationTU(InitiationTU),
        ImplementationTU(nullptr) {}

  RefactoringAction(std::unique_ptr<RenamingAction> Rename,
                    CXTranslationUnit InitiationTU)
      : Rename(std::move(Rename)),
        Type(this->Rename->Operation.isLocal() ? CXRefactor_Rename_Local
                                               : CXRefactor_Rename),
        InitiationTU(InitiationTU), ImplementationTU(nullptr) {}

  ~RefactoringAction() {
    for (const auto &Candidate : RefactoringCandidates)
      clang_disposeString(Candidate.Description);
  }

  RefactoringOperation *getOperation() const { return Operation.get(); }

  RenamingAction *getRenamingAction() const { return Rename.get(); }

  CXRefactoringCandidateSet getRefactoringCandidates() {
    if (HasCandidateSet)
      return CandidateSet;
    HasCandidateSet = true;
    RefactoringOperation *Operation = getOperation();
    if (!Operation)
      return CandidateSet;
    auto Candidates = Operation->getRefactoringCandidates();
    if (Candidates.empty())
      return CandidateSet;
    for (const auto &Candidate : Candidates)
      RefactoringCandidates.push_back({cxstring::createDup(Candidate)});
    CandidateSet = {RefactoringCandidates.data(),
                    (unsigned)RefactoringCandidates.size()};
    return CandidateSet;
  }

  CXErrorCode selectCandidate(unsigned Index) {
    RefactoringOperation *Operation = getOperation();
    if (!Operation)
      return CXError_InvalidArguments;
    if (Index != 0 && Index >= getRefactoringCandidates().NumCandidates)
      return CXError_InvalidArguments;
    SelectedCandidate = Index;
    return CXError_Success;
  }
};

static bool operator==(const CXFileLocation &LHS, const CXFileLocation &RHS) {
  return LHS.Line == RHS.Line && LHS.Column == RHS.Column;
}

static CXFileRange translateOffsetToRelativeRange(unsigned Offset,
                                                  unsigned Size,
                                                  StringRef Source) {
  assert(Source.drop_front(Offset).take_front(Size).count('\n') == 0 &&
         "Newlines in translated range?");
  StringRef Prefix = Source.take_front(Offset);
  unsigned StartLines = Prefix.count('\n') + 1;
  if (StartLines > 1)
    Offset -= Prefix.rfind('\n') + 1;
  return CXFileRange{{StartLines, Offset + 1}, {StartLines, Offset + 1 + Size}};
}

class RefactoringResultWrapper {
public:
  CXRefactoringReplacements_Old Replacements; // TODO: Remove.
  CXRefactoringReplacements SourceReplacements;
  std::unique_ptr<RefactoringContinuation> Continuation;
  llvm::BumpPtrAllocator Allocator;
  CXTranslationUnit TU;

  struct AssociatedReplacementInfo {
    CXSymbolOccurrence *AssociatedSymbolOccurrences;
    unsigned NumAssociatedSymbolOccurrences;
  };

  ~RefactoringResultWrapper() {
    // TODO: Remove.
    for (unsigned I = 0; I < Replacements.NumFileReplacementSets; ++I) {
      const CXRefactoringFileReplacementSet_Old &FileSet =
          Replacements.FileReplacementSets[I];
      clang_disposeString(FileSet.Filename);
      for (unsigned J = 0; J < FileSet.NumReplacements; ++J)
        clang_disposeString(FileSet.Replacements[J].ReplacementString);
      delete[] FileSet.Replacements;
    }
    delete[] Replacements.FileReplacementSets;

    for (unsigned I = 0; I < SourceReplacements.NumFileReplacementSets; ++I) {
      const CXRefactoringFileReplacementSet &FileSet =
          SourceReplacements.FileReplacementSets[I];
      clang_disposeString(FileSet.Filename);
      for (unsigned J = 0; J < FileSet.NumReplacements; ++J)
        clang_disposeString(FileSet.Replacements[J].ReplacementString);
    }
  }

  RefactoringResultWrapper(
      ArrayRef<RefactoringReplacement> Replacements,
      ArrayRef<std::unique_ptr<RefactoringResultAssociatedSymbol>>
          AssociatedSymbols,
      std::unique_ptr<RefactoringContinuation> Continuation,
      ASTContext &Context, CXTranslationUnit TU)
      : Continuation(std::move(Continuation)), TU(TU) {
    SourceManager &SM = Context.getSourceManager();

    if (Replacements.empty()) {
      assert(AssociatedSymbols.empty() && "Symbols without replacements??");
      // TODO: Remove begin
      this->Replacements.NumFileReplacementSets = 0;
      this->Replacements.FileReplacementSets = nullptr;
      // Remove end
      this->SourceReplacements.NumFileReplacementSets = 0;
      this->SourceReplacements.FileReplacementSets = nullptr;
      return;
    }
    llvm::SmallDenseMap<const RefactoringResultAssociatedSymbol *, unsigned>
        AssociatedSymbolToIndex;
    for (const auto &Symbol : llvm::enumerate(AssociatedSymbols))
      AssociatedSymbolToIndex[Symbol.value().get()] = Symbol.index();

    // Find the set of files that have to be modified and gather the indices of
    // the occurrences for each file.
    llvm::DenseMap<const FileEntry *, std::vector<unsigned>>
        FilesToReplacements;
    for (const auto &Replacement : llvm::enumerate(Replacements)) {
      SourceLocation Loc = Replacement.value().Range.getBegin();
      const std::pair<FileID, unsigned> DecomposedLocation =
          SM.getDecomposedLoc(Loc);
      assert(DecomposedLocation.first.isValid() && "Invalid file!");
      const FileEntry *Entry = SM.getFileEntryForID(DecomposedLocation.first);
      FilesToReplacements.try_emplace(Entry, std::vector<unsigned>())
          .first->second.push_back(Replacement.index());
    }

    // TODO: Remove
    unsigned NumFiles = FilesToReplacements.size();
    auto *FileReplacementSets =
        new CXRefactoringFileReplacementSet_Old[NumFiles];

    unsigned FileIndex = 0;
    for (const auto &Entry : FilesToReplacements) {
      CXRefactoringFileReplacementSet_Old &FileSet =
          FileReplacementSets[FileIndex];
      ++FileIndex;
      ArrayRef<unsigned> ReplacementIndices = Entry.second;
      FileSet.Filename = cxstring::createDup(Entry.first->getName());
      FileSet.NumReplacements = ReplacementIndices.size();
      auto *FileReplacements =
          new CXRefactoringReplacement_Old[ReplacementIndices.size()];
      FileSet.Replacements = FileReplacements;

      unsigned NumRemoved = 0;
      for (unsigned I = 0; I < FileSet.NumReplacements; ++I) {
        const RefactoringReplacement &RefReplacement =
            Replacements[ReplacementIndices[I]];
        CXSourceRange Range = cxloc::translateSourceRange(
            SM, Context.getLangOpts(),
            CharSourceRange::getCharRange(RefReplacement.Range.getBegin(),
                                          RefReplacement.Range.getEnd()));
        CXFileLocation Begin, End;
        clang_getFileLocation(clang_getRangeStart(Range), nullptr, &Begin.Line,
                              &Begin.Column, nullptr);
        clang_getFileLocation(clang_getRangeEnd(Range), nullptr, &End.Line,
                              &End.Column, nullptr);

        if (I && FileReplacements[I - NumRemoved - 1].Range.End == Begin) {
          // Merge the previous and the current replacement.
          FileReplacements[I - NumRemoved - 1].Range.End = End;
          std::string Replacement =
              std::string(clang_getCString(
                  FileReplacements[I - NumRemoved - 1].ReplacementString)) +
              RefReplacement.ReplacementString;
          clang_disposeString(
              FileReplacements[I - NumRemoved - 1].ReplacementString);
          FileReplacements[I - NumRemoved - 1].ReplacementString =
              cxstring::createDup(Replacement);
          NumRemoved++;
          continue;
        }

        CXRefactoringReplacement_Old &Replacement =
            FileReplacements[I - NumRemoved];
        Replacement.ReplacementString =
            cxstring::createDup(RefReplacement.ReplacementString);
        Replacement.Range.Begin = Begin;
        Replacement.Range.End = End;
      }
      FileSet.NumReplacements -= NumRemoved;
    }

    this->Replacements.FileReplacementSets = FileReplacementSets;
    this->Replacements.NumFileReplacementSets = NumFiles;

    // TODO: Outdent.
    {
      unsigned NumFiles = FilesToReplacements.size();
      auto *FileReplacementSets =
          Allocator.Allocate<CXRefactoringFileReplacementSet>(NumFiles);
      SourceReplacements.FileReplacementSets = FileReplacementSets;
      SourceReplacements.NumFileReplacementSets = NumFiles;
      unsigned FileIndex = 0;
      for (const auto &Entry : FilesToReplacements) {
        CXRefactoringFileReplacementSet &FileSet =
            FileReplacementSets[FileIndex];
        ++FileIndex;
        ArrayRef<unsigned> ReplacementIndices = Entry.second;
        FileSet.Filename = cxstring::createDup(Entry.first->getName());
        FileSet.NumReplacements = ReplacementIndices.size();
        auto *FileReplacements = Allocator.Allocate<CXRefactoringReplacement>(
            ReplacementIndices.size());
        FileSet.Replacements = FileReplacements;

        unsigned NumRemoved = 0;
        for (unsigned I = 0; I < FileSet.NumReplacements; ++I) {
          const RefactoringReplacement &RefReplacement =
              Replacements[ReplacementIndices[I]];
          CXSourceRange Range = cxloc::translateSourceRange(
              SM, Context.getLangOpts(),
              CharSourceRange::getCharRange(RefReplacement.Range.getBegin(),
                                            RefReplacement.Range.getEnd()));
          CXFileLocation Begin, End;
          clang_getFileLocation(clang_getRangeStart(Range), nullptr,
                                &Begin.Line, &Begin.Column, nullptr);
          clang_getFileLocation(clang_getRangeEnd(Range), nullptr, &End.Line,
                                &End.Column, nullptr);

          if (I && FileReplacements[I - NumRemoved - 1].Range.End == Begin) {
            // Merge the previous and the current replacement.
            FileReplacements[I - NumRemoved - 1].Range.End = End;
            std::string Replacement =
                std::string(clang_getCString(
                    FileReplacements[I - NumRemoved - 1].ReplacementString)) +
                RefReplacement.ReplacementString;
            clang_disposeString(
                FileReplacements[I - NumRemoved - 1].ReplacementString);
            FileReplacements[I - NumRemoved - 1].ReplacementString =
                cxstring::createDup(Replacement);
            NumRemoved++;
            continue;
          }

          CXRefactoringReplacement &Replacement =
              FileReplacements[I - NumRemoved];
          Replacement.ReplacementString =
              cxstring::createDup(RefReplacement.ReplacementString);
          Replacement.Range.Begin = Begin;
          Replacement.Range.End = End;
          unsigned NumAssociatedSymbols = RefReplacement.SymbolLocations.size();
          if (!NumAssociatedSymbols) {
            Replacement.AssociatedData = nullptr;
            continue;
          }
          AssociatedReplacementInfo *AssociatedData =
              Allocator.Allocate<AssociatedReplacementInfo>();
          Replacement.AssociatedData = AssociatedData;
          AssociatedData->AssociatedSymbolOccurrences =
              Allocator.Allocate<CXSymbolOccurrence>(NumAssociatedSymbols);
          AssociatedData->NumAssociatedSymbolOccurrences = NumAssociatedSymbols;
          unsigned SymbolIndex = 0;
          for (const auto &AssociatedSymbol : RefReplacement.SymbolLocations) {
            unsigned Index = AssociatedSymbolToIndex[AssociatedSymbol.first];
            const RefactoringReplacement::AssociatedSymbolLocation &Loc =
                AssociatedSymbol.second;
            CXFileRange *NamePieces =
                Allocator.Allocate<CXFileRange>(Loc.Offsets.size());
            assert(AssociatedSymbol.first->getName().size() ==
                       Loc.Offsets.size() &&
                   "mismatching symbol name and offsets");
            for (const auto &Offset : llvm::enumerate(Loc.Offsets)) {
              StringRef NamePiece =
                  AssociatedSymbol.first->getName()[Offset.index()];
              NamePieces[Offset.index()] = translateOffsetToRelativeRange(
                  Offset.value(), NamePiece.size(),
                  RefReplacement.ReplacementString);
            }
            AssociatedData->AssociatedSymbolOccurrences[SymbolIndex] =
                CXSymbolOccurrence{
                    NamePieces, (unsigned)Loc.Offsets.size(),
                    Loc.IsDeclaration
                        ? CXSymbolOccurrence_ExtractedDeclaration
                        : CXSymbolOccurrence_ExtractedDeclaration_Reference,
                    /*IsMacroExpansion=*/0, Index};
            ++SymbolIndex;
          }
        }
        FileSet.NumReplacements -= NumRemoved;
      }
    }
  }
};

class RefactoringContinuationWrapper {
public:
  std::unique_ptr<RefactoringContinuation> Continuation;
  struct QueryWrapper {
    indexer::IndexerQuery *Query;
    CXTranslationUnit TU;
    std::vector<indexer::Indexed<PersistentDeclRef<Decl>>> DeclResults;
    unsigned ConsumedResults = 0;

    QueryWrapper(indexer::IndexerQuery *Query, CXTranslationUnit TU)
        : Query(Query), TU(TU) {}
  };
  SmallVector<QueryWrapper, 4> Queries;
  bool IsInitiationTUAbandoned = false;

  RefactoringContinuationWrapper(
      std::unique_ptr<RefactoringContinuation> Continuation,
      CXTranslationUnit TU)
      : Continuation(std::move(Continuation)) {
    Queries.emplace_back(this->Continuation->getASTUnitIndexerQuery(), TU);
    assert(Queries.back().Query && "Invalid ast query");
    std::vector<indexer::IndexerQuery *> AdditionalQueries =
        this->Continuation->getAdditionalIndexerQueries();
    for (indexer::IndexerQuery *IQ : AdditionalQueries)
      Queries.emplace_back(IQ, TU);
  }
};

class RefactoringDiagnosticConsumer : public DiagnosticConsumer {
  const ASTContext &Context;
  DiagnosticConsumer *PreviousClient;
  std::unique_ptr<DiagnosticConsumer> PreviousClientPtr;
  llvm::SmallVector<StoredDiagnostic, 2> RenameDiagnostics;
  llvm::SmallVector<StoredDiagnostic, 1> ContinuationDiagnostics;

public:
  RefactoringDiagnosticConsumer(ASTContext &Context) : Context(Context) {
    PreviousClient = Context.getDiagnostics().getClient();
    PreviousClientPtr = Context.getDiagnostics().takeClient();
    Context.getDiagnostics().setClient(this, /*ShouldOwnClient=*/false);
  }

  ~RefactoringDiagnosticConsumer() {
    if (PreviousClientPtr)
      Context.getDiagnostics().setClient(PreviousClientPtr.release());
    else
      Context.getDiagnostics().setClient(PreviousClient,
                                         /*ShouldOwnClient=*/false);
  }

  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override {
    unsigned Cat = DiagnosticIDs::getCategoryNumberForDiag(Info.getID());
    if (Cat == diag::DiagCat_Rename_Issue)
      RenameDiagnostics.push_back(StoredDiagnostic(Level, Info));
    else if (Cat == diag::DiagCat_Refactoring_Continuation_Issue)
      ContinuationDiagnostics.push_back(StoredDiagnostic(Level, Info));
    else
      assert(false && "Unhandled refactoring category");
  }

  CXDiagnosticSetImpl *createDiags() const {
    if (RenameDiagnostics.empty() && ContinuationDiagnostics.empty())
      return nullptr;
    llvm::SmallVector<StoredDiagnostic, 2> AllDiagnostics;
    for (const auto &D : RenameDiagnostics)
      AllDiagnostics.push_back(D);
    for (const auto &D : ContinuationDiagnostics)
      AllDiagnostics.push_back(D);
    return cxdiag::createStoredDiags(AllDiagnostics, Context.getLangOpts());
  }

  CXRefactoringActionSetWithDiagnostics createActionSet() const {
    if (RenameDiagnostics.empty())
      return {nullptr, 0};
    CXRefactoringActionWithDiagnostics *Actions =
        new CXRefactoringActionWithDiagnostics[1];
    Actions[0].Action = CXRefactor_Rename;
    Actions[0].Diagnostics =
        cxdiag::createStoredDiags(RenameDiagnostics, Context.getLangOpts());
    return {Actions, 1};
  }
};

} // end anonymous namespace

template <typename T>
static T withRenamingAction(CXRefactoringAction Action, T DefaultValue,
                            llvm::function_ref<T(RenamingAction &)> Callback) {
  if (!Action)
    return DefaultValue;
  RenamingAction *Rename =
      static_cast<RefactoringAction *>(Action)->getRenamingAction();
  if (!Rename)
    return DefaultValue;
  return Callback(*Rename);
}

static enum CXIndexerQueryKind
translateDeclPredicate(const indexer::DeclPredicate &Predicate) {
  indexer::DeclEntity Entity;
  if (Predicate == Entity.isDefined().Predicate)
    return CXIndexerQuery_Decl_IsDefined;
  return CXIndexerQuery_Unknown;
}

extern "C" {

CXString
clang_RefactoringActionType_getName(enum CXRefactoringActionType Action) {
  return cxstring::createRef(
      getRefactoringActionTypeName(translateRefactoringActionType(Action)));
}

void clang_RefactoringActionSet_dispose(CXRefactoringActionSet *Set) {
  if (Set && Set->Actions)
    delete[] Set->Actions;
}

void clang_RefactoringActionSetWithDiagnostics_dispose(
    CXRefactoringActionSetWithDiagnostics *Set) {
  if (Set && Set->Actions) {
    for (auto &S : llvm::makeArrayRef(Set->Actions, Set->NumActions))
      clang_disposeDiagnosticSet(S.Diagnostics);
    delete[] Set->Actions;
  }
}

CXRefactoringOptionSet clang_RefactoringOptionSet_create() {
  return new RefactoringOptionSet;
}

CXRefactoringOptionSet
clang_RefactoringOptionSet_createFromString(const char *String) {
  RefactoringOptionSet *Result = new RefactoringOptionSet;
  auto Options = RefactoringOptionSet::parse(String);
  if (Options) {
    *Result = std::move(*Options);
    return Result;
  }
  llvm::handleAllErrors(Options.takeError(),
                        [](const llvm::StringError &Error) {});
  return clang_RefactoringOptionSet_create();
}

void clang_RefactoringOptionSet_add(CXRefactoringOptionSet Set,
                                    enum CXRefactoringOption Option) {
  if (!Set)
    return;
  switch (Option) {
  case CXRefactorOption_AvoidTextualMatches:
    static_cast<RefactoringOptionSet *>(Set)->add(
        option::AvoidTextualMatches::getTrue());
    break;
  }
}

CXString clang_RefactoringOptionSet_toString(CXRefactoringOptionSet Set) {
  if (!Set)
    return cxstring::createNull();
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  static_cast<RefactoringOptionSet *>(Set)->print(OS);
  return cxstring::createDup(OS.str());
}

void clang_RefactoringOptionSet_dispose(CXRefactoringOptionSet Set) {
  if (Set)
    delete static_cast<RefactoringOptionSet *>(Set);
}

enum CXErrorCode
clang_Refactoring_findActionsAt(CXTranslationUnit TU, CXSourceLocation Location,
                                CXSourceRange SelectionRange,
                                CXRefactoringOptionSet Options,
                                CXRefactoringActionSet *OutSet) {
  return clang_Refactoring_findActionsWithInitiationFailureDiagnosicsAt(
      TU, Location, SelectionRange, Options, OutSet, /*OutFailureSet=*/nullptr);
}

enum CXErrorCode clang_Refactoring_findActionsWithInitiationFailureDiagnosicsAt(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, CXRefactoringOptionSet Options,
    CXRefactoringActionSet *OutSet,
    CXRefactoringActionSetWithDiagnostics *OutFailureSet) {
  LOG_FUNC_SECTION { *Log << TU << ' '; }
  if (OutFailureSet) {
    OutFailureSet->Actions = nullptr;
    OutFailureSet->NumActions = 0;
  }

  if (!OutSet)
    return CXError_InvalidArguments;

  OutSet->Actions = nullptr;
  OutSet->NumActions = 0;

  if (cxtu::isNotUsableTU(TU)) {
    LOG_BAD_TU(TU);
    return CXError_InvalidArguments;
  }

  ASTUnit *CXXUnit = cxtu::getASTUnit(TU);
  if (!CXXUnit)
    return CXError_InvalidArguments;

  SourceLocation Loc = cxloc::translateSourceLocation(Location);
  if (Loc.isInvalid())
    return CXError_InvalidArguments;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);
  (void)Options; // FIXME: handle options
  ASTContext &Context = CXXUnit->getASTContext();
  RefactoringDiagnosticConsumer DiagConsumer(Context);
  RefactoringActionSet ActionSet = findActionSetAt(
      Loc, cxloc::translateCXSourceRange(SelectionRange), Context);
  if (OutFailureSet)
    *OutFailureSet = DiagConsumer.createActionSet();
  if (ActionSet.Actions.empty())
    return CXError_RefactoringActionUnavailable;

  CXRefactoringActionType *Actions =
      new CXRefactoringActionType[ActionSet.Actions.size()];
  OutSet->Actions = Actions;
  OutSet->NumActions = ActionSet.Actions.size();
  for (const auto &Action : llvm::enumerate(ActionSet.Actions))
    Actions[Action.index()] = translateRefactoringActionType(Action.value());
  return CXError_Success;
}

void clang_RefactoringAction_dispose(CXRefactoringAction Action) {
  if (Action)
    delete static_cast<RefactoringAction *>(Action);
}

CXSourceRange
clang_RefactoringAction_getSourceRangeOfInterest(CXRefactoringAction Action) {
  if (Action) {
    RefactoringOperation *Operation =
        static_cast<RefactoringAction *>(Action)->getOperation();
    if (Operation) {
      ASTUnit *CXXUnit = cxtu::getASTUnit(
          static_cast<RefactoringAction *>(Action)->InitiationTU);
      if (const Stmt *S = Operation->getTransformedStmt()) {
        SourceRange Range = S->getSourceRange();
        if (const Stmt *Last = Operation->getLastTransformedStmt())
          Range.setEnd(Last->getEndLoc());
        return cxloc::translateSourceRange(CXXUnit->getASTContext(), Range);
      } else if (const Decl *D = Operation->getTransformedDecl()) {
        SourceRange Range = D->getSourceRange();
        if (const Decl *Last = Operation->getLastTransformedDecl())
          Range.setEnd(Last->getEndLoc());
        return cxloc::translateSourceRange(CXXUnit->getASTContext(), Range);
      }
    }
  }
  return clang_getNullRange();
}

int clang_RefactoringAction_requiresImplementationTU(
    CXRefactoringAction Action) {
  return withRenamingAction<int>(Action, 0, [](RenamingAction &Action) {
    return Action.Operation.requiresImplementationTU();
  });
}

CXString clang_RefactoringAction_getUSRThatRequiresImplementationTU(
    CXRefactoringAction Action) {
  return withRenamingAction<CXString>(
      Action, cxstring::createNull(), [](RenamingAction &Action) {
        return Action.getUSRThatRequiresImplementationTU();
      });
}

enum CXErrorCode
clang_RefactoringAction_addImplementationTU(CXRefactoringAction Action,
                                            CXTranslationUnit TU) {
  if (!Action || !TU)
    return CXError_InvalidArguments;
  // Prohibit multiple additions of implementation TU.
  if (static_cast<RefactoringAction *>(Action)->ImplementationTU)
    return CXError_Failure;
  static_cast<RefactoringAction *>(Action)->ImplementationTU = TU;
  return CXError_Success;
}

enum CXErrorCode clang_RefactoringAction_getRefactoringCandidates(
    CXRefactoringAction Action,
    CXRefactoringCandidateSet *OutRefactoringCandidateSet) {
  if (!Action || !OutRefactoringCandidateSet)
    return CXError_InvalidArguments;
  *OutRefactoringCandidateSet =
      static_cast<RefactoringAction *>(Action)->getRefactoringCandidates();
  return CXError_Success;
}

enum CXErrorCode
clang_RefactoringAction_selectRefactoringCandidate(CXRefactoringAction Action,
                                                   unsigned Index) {
  if (!Action)
    return CXError_InvalidArguments;
  return static_cast<RefactoringAction *>(Action)->selectCandidate(Index);
}

// TODO: Remove.
enum CXErrorCode clang_Refactoring_initiateActionAt(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, enum CXRefactoringActionType ActionType,
    CXRefactoringOptionSet Options, CXRefactoringAction *OutAction,
    CXString *OutFailureReason) {
  CXDiagnosticSet Diags;
  CXErrorCode Result = clang_Refactoring_initiateAction(
      TU, Location, SelectionRange, ActionType, Options, OutAction, &Diags);
  if (OutFailureReason && Diags && clang_getNumDiagnosticsInSet(Diags) == 1) {
    CXString Spelling =
        clang_getDiagnosticSpelling(clang_getDiagnosticInSet(Diags, 0));
    *OutFailureReason = cxstring::createDup(clang_getCString(Spelling));
    clang_disposeString(Spelling);
  } else if (OutFailureReason)
    *OutFailureReason = cxstring::createEmpty();
  clang_disposeDiagnosticSet(Diags);
  return Result;
}

enum CXErrorCode clang_Refactoring_initiateAction(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, enum CXRefactoringActionType ActionType,
    CXRefactoringOptionSet Options, CXRefactoringAction *OutAction,
    CXDiagnosticSet *OutDiagnostics) {
  if (!OutAction)
    return CXError_InvalidArguments;
  *OutAction = nullptr;
  if (OutDiagnostics)
    *OutDiagnostics = nullptr;

  if (cxtu::isNotUsableTU(TU)) {
    LOG_BAD_TU(TU);
    return CXError_InvalidArguments;
  }

  ASTUnit *CXXUnit = cxtu::getASTUnit(TU);
  if (!CXXUnit)
    return CXError_InvalidArguments;

  SourceLocation Loc = cxloc::translateSourceLocation(Location);
  if (Loc.isInvalid())
    return CXError_InvalidArguments;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  (void)Options; // FIXME: handle options
  ASTContext &Context = CXXUnit->getASTContext();
  RefactoringDiagnosticConsumer DiagConsumer(Context);
  auto Operation = initiateRefactoringOperationAt(
      Loc, cxloc::translateCXSourceRange(SelectionRange), Context,
      translateRefactoringActionType(ActionType));
  if (!Operation.Initiated) {
    if (OutDiagnostics) {
      if (!Operation.FailureReason.empty()) {
        // TODO: Remove when other actions migrate to diagnostics.
        StoredDiagnostic Diag(DiagnosticsEngine::Error, /*ID=*/0,
                              Operation.FailureReason);
        *OutDiagnostics =
            cxdiag::createStoredDiags(Diag, Context.getLangOpts());
      } else
        *OutDiagnostics = DiagConsumer.createDiags();
    }
    return CXError_RefactoringActionUnavailable;
  }
  if (Operation.RefactoringOp)
    *OutAction = new RefactoringAction(std::move(Operation.RefactoringOp),
                                       ActionType, TU);
  else
    *OutAction = new RefactoringAction(
        llvm::make_unique<RenamingAction>(CXXUnit->getLangOpts(),
                                          std::move(*Operation.SymbolOp)),
        TU);
  return CXError_Success;
}

enum CXErrorCode clang_Refactoring_initiateActionOnDecl(
    CXTranslationUnit TU, const char *DeclUSR,
    enum CXRefactoringActionType ActionType, CXRefactoringOptionSet Options,
    CXRefactoringAction *OutAction, CXString *OutFailureReason) {
  if (!OutAction)
    return CXError_InvalidArguments;
  *OutAction = nullptr;
  if (OutFailureReason)
    *OutFailureReason = cxstring::createNull();

  if (cxtu::isNotUsableTU(TU)) {
    LOG_BAD_TU(TU);
    return CXError_InvalidArguments;
  }

  ASTUnit *CXXUnit = cxtu::getASTUnit(TU);
  if (!CXXUnit)
    return CXError_InvalidArguments;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  (void)Options; // FIXME: handle options
  auto Operation = initiateRefactoringOperationOnDecl(
      DeclUSR, CXXUnit->getASTContext(),
      translateRefactoringActionType(ActionType));
  if (!Operation.Initiated)
    return CXError_RefactoringActionUnavailable;
  // FIXME: Don't dupe with above
  if (Operation.RefactoringOp)
    *OutAction = new RefactoringAction(std::move(Operation.RefactoringOp),
                                       ActionType, TU);
  else
    *OutAction = new RefactoringAction(
        llvm::make_unique<RenamingAction>(CXXUnit->getLangOpts(),
                                          std::move(*Operation.SymbolOp)),
        TU);
  return CXError_Success;
}

enum CXErrorCode
clang_Refactoring_initiateRenamingOperation(CXRefactoringAction Action) {
  if (!Action)
    return CXError_InvalidArguments;
  RefactoringAction *RefAction = static_cast<RefactoringAction *>(Action);
  RenamingAction *Rename = RefAction->getRenamingAction();
  if (!Rename)
    return CXError_InvalidArguments;
  // TODO
  return CXError_Success;
}

CINDEX_LINKAGE
enum CXErrorCode clang_Refactoring_findRenamedCursor(
    CXTranslationUnit TU, CXSourceLocation Location,
    CXSourceRange SelectionRange, CXCursor *OutCursor) {
  if (!OutCursor)
    return CXError_InvalidArguments;

  if (cxtu::isNotUsableTU(TU)) {
    LOG_BAD_TU(TU);
    return CXError_InvalidArguments;
  }

  ASTUnit *CXXUnit = cxtu::getASTUnit(TU);
  if (!CXXUnit)
    return CXError_InvalidArguments;
  SourceLocation Loc = cxloc::translateSourceLocation(Location);
  if (Loc.isInvalid())
    return CXError_InvalidArguments;

  const NamedDecl *ND = rename::getNamedDeclAt(CXXUnit->getASTContext(), Loc);
  if (!ND) {
    *OutCursor = cxcursor::MakeCXCursorInvalid(CXCursor_NoDeclFound, TU);
    return CXError_RefactoringActionUnavailable;
  }

  *OutCursor = cxcursor::MakeCXCursor(ND, TU);
  return CXError_Success;
}

enum CXErrorCode clang_RenamingOperation_setNewName(CXRefactoringAction Action,
                                                    const char *NewName) {
  return withRenamingAction<CXErrorCode>(
      Action, CXError_InvalidArguments,
      [=](RenamingAction &Action) -> CXErrorCode {
        if (!NewName)
          return CXError_InvalidArguments;
        StringRef Name = NewName;
        if (Name.empty())
          return CXError_InvalidArguments;
        return Action.setNewName(Name);
      });
}

enum CXRefactoringActionType
clang_RefactoringAction_getInitiatedActionType(CXRefactoringAction Action) {
  return static_cast<RefactoringAction *>(Action)->Type;
}

unsigned clang_RenamingOperation_getNumSymbols(CXRefactoringAction Action) {
  return withRenamingAction<unsigned>(Action, 0, [](RenamingAction &Action) {
    return Action.Operation.symbols().size();
  });
}

CXString clang_RenamingOperation_getUSRForSymbol(CXRefactoringAction Action,
                                                 unsigned Index) {
  return withRenamingAction<CXString>(
      Action, cxstring::createNull(),
      [=](RenamingAction &Action) { return Action.usrForSymbolAt(Index); });
}

CXRenamingResult clang_Refactoring_findRenamedOccurrencesInPrimaryTUs(
    CXRefactoringAction Action, const char *const *CommandLineArgs,
    int NumCommandLineArgs, CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles) {
  if (!Action)
    return nullptr;
  RefactoringAction *RefAction = static_cast<RefactoringAction *>(Action);
  RenamingAction *Rename = RefAction->getRenamingAction();
  if (!Rename)
    return nullptr;

  // TODO: Handle implementation TU
  if (cxtu::isNotUsableTU(RefAction->InitiationTU)) {
    LOG_BAD_TU(RefAction->InitiationTU);
    return nullptr;
  }

  ASTUnit *CXXUnit = cxtu::getASTUnit(RefAction->InitiationTU);
  if (!CXXUnit)
    return nullptr;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  return Rename->handlePrimaryTU(RefAction->InitiationTU, *CXXUnit);
}

CXSymbolOccurrencesResult clang_Refactoring_findSymbolOccurrencesInInitiationTU(
    CXRefactoringAction Action, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles) {
  if (!Action)
    return nullptr;
  RefactoringAction *RefAction = static_cast<RefactoringAction *>(Action);
  RenamingAction *Rename = RefAction->getRenamingAction();
  if (!Rename)
    return nullptr;

  if (cxtu::isNotUsableTU(RefAction->InitiationTU)) {
    LOG_BAD_TU(RefAction->InitiationTU);
    return nullptr;
  }

  ASTUnit *CXXUnit = cxtu::getASTUnit(RefAction->InitiationTU);
  if (!CXXUnit)
    return nullptr;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  return Rename->findSymbolsInInitiationTU(RefAction->InitiationTU, *CXXUnit);
}

CXErrorCode clang_Refactoring_findRenamedOccurrencesInIndexedFile(
    const CXRenamedIndexedSymbol *Symbols, unsigned NumSymbols, CXIndex CIdx,
    const char *Filename, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles, CXRefactoringOptionSet Options,
    CXRenamingResult *OutResult) {
  if (!OutResult)
    return CXError_InvalidArguments;
  if (!Symbols || !NumSymbols || !Filename)
    return CXError_InvalidArguments;
  return performIndexedFileRename(
      llvm::makeArrayRef(Symbols, NumSymbols), StringRef(Filename),
      llvm::makeArrayRef(CommandLineArgs, NumCommandLineArgs), CIdx,
      MutableArrayRef<CXUnsavedFile>(UnsavedFiles, NumUnsavedFiles),
      Options ? static_cast<RefactoringOptionSet *>(Options) : nullptr,
      *OutResult);
}

CXErrorCode clang_Refactoring_findSymbolOccurrencesInIndexedFile(
    const CXIndexedSymbol *Symbols, unsigned NumSymbols, CXIndex CIdx,
    const char *Filename, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles, CXRefactoringOptionSet Options,
    CXSymbolOccurrencesResult *OutResult) {
  if (!OutResult)
    return CXError_InvalidArguments;
  if (!Symbols || !NumSymbols || !Filename)
    return CXError_InvalidArguments;
  return performIndexedSymbolSearch(
      llvm::makeArrayRef(Symbols, NumSymbols), StringRef(Filename),
      llvm::makeArrayRef(CommandLineArgs, NumCommandLineArgs), CIdx,
      MutableArrayRef<CXUnsavedFile>(UnsavedFiles, NumUnsavedFiles),
      Options ? static_cast<RefactoringOptionSet *>(Options) : nullptr,
      *OutResult);
}

unsigned clang_RenamingResult_getNumModifiedFiles(CXRenamingResult Result) {
  if (Result)
    return static_cast<RenamingResult *>(Result)->getFilenames().size();
  return 0;
}

void clang_RenamingResult_getResultForFile(CXRenamingResult Result,
                                           unsigned FileIndex,
                                           CXFileRenamingResult *OutResult) {
  if (!Result ||
      FileIndex >=
          static_cast<RenamingResult *>(Result)->getFilenames().size()) {
    OutResult->Filename = cxstring::createNull();
    OutResult->NumOccurrences = 0;
    OutResult->Occurrences = nullptr;
    return;
  }
  auto &RenameResult = *static_cast<RenamingResult *>(Result);
  OutResult->Filename = RenameResult.getFilenames()[FileIndex];
  OutResult->NumOccurrences = RenameResult.getOccurrences(FileIndex).size();
  OutResult->Occurrences = RenameResult.getOccurrences(FileIndex).data();
}

void clang_RenamingResult_dispose(CXRenamingResult Result) {
  if (Result)
    delete static_cast<RenamingResult *>(Result);
}

unsigned clang_SymbolOccurrences_getNumFiles(CXSymbolOccurrencesResult Result) {
  if (Result)
    return static_cast<SymbolOccurrencesResult *>(Result)
        ->getFilenames()
        .size();
  return 0;
}

void clang_SymbolOccurrences_getOccurrencesForFile(
    CXSymbolOccurrencesResult Result, unsigned FileIndex,
    CXSymbolOccurrencesInFile *OutResult) {
  if (!Result ||
      FileIndex >= static_cast<SymbolOccurrencesResult *>(Result)
                       ->getFilenames()
                       .size()) {
    OutResult->Filename = cxstring::createNull();
    OutResult->NumOccurrences = 0;
    OutResult->Occurrences = nullptr;
    return;
  }
  auto &RenameResult = *static_cast<SymbolOccurrencesResult *>(Result);
  OutResult->Filename = RenameResult.getFilenames()[FileIndex];
  OutResult->NumOccurrences = RenameResult.getOccurrences(FileIndex).size();
  OutResult->Occurrences = RenameResult.getOccurrences(FileIndex).data();
}

void clang_SymbolOccurrences_dispose(CXSymbolOccurrencesResult Result) {
  if (Result)
    delete static_cast<SymbolOccurrencesResult *>(Result);
}

CXRefactoringResult clang_Refactoring_performOperation(
    CXRefactoringAction Action, const char *const *CommandLineArgs,
    int NumCommandLineArgs, struct CXUnsavedFile *UnsavedFiles,
    unsigned NumUnsavedFiles, CXRefactoringOptionSet Options,
    CXString *OutFailureReason) {
  if (OutFailureReason)
    *OutFailureReason = cxstring::createNull();
  if (!Action)
    return nullptr;
  RefactoringAction *RefAction = static_cast<RefactoringAction *>(Action);
  if (!RefAction->getOperation())
    return nullptr;

  ASTUnit *CXXUnit = cxtu::getASTUnit(RefAction->InitiationTU);
  if (!CXXUnit)
    return nullptr;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  RefactoringOptionSet EmptyOptionSet;
  const RefactoringOptionSet &OptionSet =
      Options ? *static_cast<RefactoringOptionSet *>(Options) : EmptyOptionSet;
  llvm::Expected<RefactoringResult> Result = RefAction->getOperation()->perform(
      CXXUnit->getASTContext(), CXXUnit->getPreprocessor(), OptionSet,
      RefAction->SelectedCandidate);
  if (!Result) {
    if (OutFailureReason) {
      (void)!llvm::handleErrors(
          Result.takeError(), [&](const RefactoringOperationError &Error) {
            *OutFailureReason = cxstring::createDup(Error.FailureReason);
          });
    }
    return nullptr;
  }
  return new RefactoringResultWrapper(
      Result.get().Replacements, Result.get().AssociatedSymbols,
      std::move(Result.get().Continuation), CXXUnit->getASTContext(),
      RefAction->InitiationTU);
}

void clang_RefactoringResult_getReplacements(
    CXRefactoringResult Result,
    CXRefactoringReplacements_Old *OutReplacements) {
  if (!OutReplacements)
    return;
  if (!Result) {
    OutReplacements->FileReplacementSets = nullptr;
    OutReplacements->NumFileReplacementSets = 0;
    return;
  }
  *OutReplacements = static_cast<RefactoringResultWrapper *>(Result)->Replacements;
}

CXRefactoringReplacements
clang_RefactoringResult_getSourceReplacements(CXRefactoringResult Result) {
  if (!Result)
    return CXRefactoringReplacements{nullptr, 0};
  return static_cast<RefactoringResultWrapper *>(Result)->SourceReplacements;
}

CXRefactoringReplacementAssociatedSymbolOccurrences
clang_RefactoringReplacement_getAssociatedSymbolOccurrences(
    CXRefactoringReplacement Replacement) {
  if (!Replacement.AssociatedData)
    return CXRefactoringReplacementAssociatedSymbolOccurrences{nullptr, 0};
  auto *Data =
      static_cast<RefactoringResultWrapper::AssociatedReplacementInfo *>(
          Replacement.AssociatedData);
  return CXRefactoringReplacementAssociatedSymbolOccurrences{
      Data->AssociatedSymbolOccurrences, Data->NumAssociatedSymbolOccurrences};
}

void clang_RefactoringResult_dispose(CXRefactoringResult Result) {
  if (Result)
    delete static_cast<RefactoringResultWrapper *>(Result);
}

CXRefactoringContinuation
clang_RefactoringResult_getContinuation(CXRefactoringResult Result) {
  if (!Result)
    return nullptr;
  auto *Wrapper = static_cast<RefactoringResultWrapper *>(Result);
  if (!Wrapper->Continuation)
    return nullptr;
  return new RefactoringContinuationWrapper(std::move(Wrapper->Continuation),
                                            Wrapper->TU);
}

enum CXErrorCode
clang_RefactoringContinuation_loadSerializedIndexerQueryResults(
    CXRefactoringContinuation Continuation, const char *Source) {
  if (!Continuation)
    return CXError_InvalidArguments;
  auto *Wrapper = static_cast<RefactoringContinuationWrapper *>(Continuation);
  llvm::SmallVector<indexer::IndexerQuery *, 4> Queries;
  for (const auto &Query : Wrapper->Queries)
    Queries.push_back(Query.Query);
  auto Err = indexer::IndexerQuery::loadResultsFromYAML(Source, Queries);
  if (Err) {
    consumeError(std::move(Err));
    return CXError_Failure;
  }
  return CXError_Success;
}

unsigned clang_RefactoringContinuation_getNumIndexerQueries(
    CXRefactoringContinuation Continuation) {
  if (Continuation)
    return static_cast<RefactoringContinuationWrapper *>(Continuation)
        ->Queries.size();
  return 0;
}

CXIndexerQuery clang_RefactoringContinuation_getIndexerQuery(
    CXRefactoringContinuation Continuation, unsigned Index) {
  if (!Continuation)
    return nullptr;
  auto *Wrapper = static_cast<RefactoringContinuationWrapper *>(Continuation);
  if (Index >= Wrapper->Queries.size())
    return nullptr;
  return &Wrapper->Queries[Index];
}

CXDiagnosticSet clang_RefactoringContinuation_verifyBeforeFinalizing(
    CXRefactoringContinuation Continuation) {
  if (!Continuation)
    return nullptr;
  auto *Wrapper = static_cast<RefactoringContinuationWrapper *>(Continuation);
  CXTranslationUnit TU = Wrapper->Queries[0].TU;
  ASTUnit *CXXUnit = cxtu::getASTUnit(TU);
  if (!CXXUnit)
    return nullptr;
  ASTContext &Context = CXXUnit->getASTContext();
  RefactoringDiagnosticConsumer DiagConsumer(Context);
  for (const auto &Query : Wrapper->Queries) {
    if (Query.Query->verify(Context))
      break;
  }
  return DiagConsumer.createDiags();
}

void clang_RefactoringContinuation_finalizeEvaluationInInitationTU(
    CXRefactoringContinuation Continuation) {
  if (!Continuation)
    return;
  auto *Wrapper = static_cast<RefactoringContinuationWrapper *>(Continuation);
  Wrapper->Queries.clear();
  Wrapper->Continuation->persistTUSpecificState();
  Wrapper->IsInitiationTUAbandoned = true;
}

CXRefactoringResult clang_RefactoringContinuation_continueOperationInTU(
    CXRefactoringContinuation Continuation, CXTranslationUnit TU,
    CXString *OutFailureReason) {
  if (!Continuation || !TU)
    return nullptr;
  ASTUnit *CXXUnit = cxtu::getASTUnit(TU);
  if (!CXXUnit)
    return nullptr;
  ASTUnit::ConcurrencyCheck Check(*CXXUnit);
  const auto *Wrapper =
      static_cast<RefactoringContinuationWrapper *>(Continuation);
  if (!Wrapper->IsInitiationTUAbandoned) {
    // FIXME: We can avoid conversions of TU-specific state if the given TU is
    // the same as the initiation TU.
    clang_RefactoringContinuation_finalizeEvaluationInInitationTU(Continuation);
  }
  auto Result =
      Wrapper->Continuation->runInExternalASTUnit(CXXUnit->getASTContext());
  if (!Result) {
    if (OutFailureReason) {
      (void)!llvm::handleErrors(
          Result.takeError(), [&](const RefactoringOperationError &Error) {
            *OutFailureReason = cxstring::createDup(Error.FailureReason);
          });
    }
    return nullptr;
  }
  return new RefactoringResultWrapper(
      Result.get().Replacements, Result.get().AssociatedSymbols,
      std::move(Result.get().Continuation), CXXUnit->getASTContext(), TU);
}

void clang_RefactoringContinuation_dispose(
    CXRefactoringContinuation Continuation) {
  if (Continuation)
    delete static_cast<RefactoringContinuationWrapper *>(Continuation);
}

enum CXIndexerQueryKind clang_IndexerQuery_getKind(CXIndexerQuery Query) {
  if (!Query)
    return CXIndexerQuery_Unknown;
  const auto *IQ =
      static_cast<RefactoringContinuationWrapper::QueryWrapper *>(Query)->Query;
  if (const auto *DQ = dyn_cast<indexer::DeclarationsQuery>(IQ)) {
    const indexer::detail::DeclPredicateNode &Node = DQ->getPredicateNode();
    if (const auto *NP =
            dyn_cast<indexer::detail::DeclPredicateNotPredicate>(&Node))
      return translateDeclPredicate(
          cast<indexer::detail::DeclPredicateNodePredicate>(NP->getChild())
              .getPredicate());
    return translateDeclPredicate(
        cast<indexer::detail::DeclPredicateNodePredicate>(Node).getPredicate());
  } else if (isa<indexer::ASTUnitForImplementationOfDeclarationQuery>(IQ))
    return CXIndexerQuery_Decl_FileThatShouldImplement;
  return CXIndexerQuery_Unknown;
}

unsigned clang_IndexerQuery_getNumCursors(CXIndexerQuery Query) {
  if (!Query)
    return 0;
  const auto *IQ =
      static_cast<RefactoringContinuationWrapper::QueryWrapper *>(Query)->Query;
  if (const auto *DQ = dyn_cast<indexer::DeclarationsQuery>(IQ))
    return DQ->getInputs().size();
  else if (isa<indexer::ASTUnitForImplementationOfDeclarationQuery>(IQ))
    return 1;
  return 0;
}

CXCursor clang_IndexerQuery_getCursor(CXIndexerQuery Query,
                                      unsigned CursorIndex) {
  if (Query) {
    const auto *Wrapper =
        static_cast<RefactoringContinuationWrapper::QueryWrapper *>(Query);
    const indexer::IndexerQuery *IQ = Wrapper->Query;
    CXTranslationUnit TU = Wrapper->TU;
    if (const auto *DQ = dyn_cast<indexer::DeclarationsQuery>(IQ)) {
      if (CursorIndex < DQ->getInputs().size())
        return cxcursor::MakeCXCursor(DQ->getInputs()[CursorIndex], TU);
    } else if (const auto *ASTQuery = dyn_cast<
                   indexer::ASTUnitForImplementationOfDeclarationQuery>(IQ)) {
      if (CursorIndex == 0)
        return cxcursor::MakeCXCursor(ASTQuery->getDecl(), TU);
    }
  }
  return cxcursor::MakeCXCursorInvalid(CXCursor_InvalidCode);
}

enum CXIndexerQueryAction
clang_IndexerQuery_consumeIntResult(CXIndexerQuery Query, unsigned CursorIndex,
                                    int Value) {
  if (!Query)
    return CXIndexerQueryAction_None;
  auto *Wrapper =
      static_cast<RefactoringContinuationWrapper::QueryWrapper *>(Query);
  auto *DQ = dyn_cast<indexer::DeclarationsQuery>(Wrapper->Query);
  if (!DQ)
    return CXIndexerQueryAction_None;
  if (CursorIndex >= DQ->getInputs().size() ||
      Wrapper->ConsumedResults == DQ->getInputs().size())
    return CXIndexerQueryAction_None;
  if (Wrapper->DeclResults.empty())
    Wrapper->DeclResults.resize(DQ->getInputs().size(),
                                indexer::Indexed<PersistentDeclRef<Decl>>(
                                    PersistentDeclRef<Decl>::create(nullptr)));
  // Filter the declarations!
  bool IsNot = false;
  if (isa<indexer::detail::DeclPredicateNotPredicate>(DQ->getPredicateNode()))
    IsNot = true;
  bool Result = IsNot ? !Value : !!Value;
  Wrapper->DeclResults[CursorIndex] = indexer::Indexed<PersistentDeclRef<Decl>>(
      PersistentDeclRef<Decl>::create(Result ? DQ->getInputs()[CursorIndex]
                                             : nullptr),
      Result ? indexer::QueryBoolResult::Yes : indexer::QueryBoolResult::No);
  Wrapper->ConsumedResults++;
  if (Wrapper->ConsumedResults == Wrapper->DeclResults.size()) {
    // We've received all the results, pass them back to the query.
    DQ->setOutput(std::move(Wrapper->DeclResults));
  }
  return CXIndexerQueryAction_None;
}

enum CXIndexerQueryAction
clang_IndexerQuery_consumeFileResult(CXIndexerQuery Query, unsigned CursorIndex,
                                     const char *Filename) {
  if (!Query || !Filename)
    return CXIndexerQueryAction_None;
  auto *IQ =
      static_cast<RefactoringContinuationWrapper::QueryWrapper *>(Query)->Query;
  if (auto *ASTQuery =
          dyn_cast<indexer::ASTUnitForImplementationOfDeclarationQuery>(IQ)) {
    if (CursorIndex != 0)
      return CXIndexerQueryAction_None;
    ASTQuery->setResult(PersistentFileID(Filename));
    return CXIndexerQueryAction_RunContinuationInTUThatHasThisFile;
  }
  return CXIndexerQueryAction_None;
}
}
