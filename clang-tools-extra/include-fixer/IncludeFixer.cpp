//===-- IncludeFixer.cpp - Include inserter based on sema callbacks -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IncludeFixer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "include-fixer"

using namespace clang;

namespace clang {
namespace include_fixer {
namespace {

class Action;

class PreprocessorHooks : public clang::PPCallbacks {
public:
  explicit PreprocessorHooks(Action *EnclosingPass)
      : EnclosingPass(EnclosingPass), TrackedFile(nullptr) {}

  void FileChanged(clang::SourceLocation loc,
                   clang::PPCallbacks::FileChangeReason reason,
                   clang::SrcMgr::CharacteristicKind file_type,
                   clang::FileID prev_fid) override;

  void InclusionDirective(clang::SourceLocation HashLocation,
                          const clang::Token &IncludeToken,
                          llvm::StringRef FileName, bool IsAngled,
                          clang::CharSourceRange FileNameRange,
                          const clang::FileEntry *IncludeFile,
                          llvm::StringRef SearchPath,
                          llvm::StringRef relative_path,
                          const clang::Module *imported) override;

private:
  /// The current Action.
  Action *EnclosingPass;

  /// The current FileEntry.
  const clang::FileEntry *TrackedFile;
};

/// Manages the parse, gathers include suggestions.
class Action : public clang::ASTFrontendAction,
               public clang::ExternalSemaSource {
public:
  explicit Action(SymbolIndexManager &SymbolIndexMgr, bool MinimizeIncludePaths)
      : SymbolIndexMgr(SymbolIndexMgr),
        MinimizeIncludePaths(MinimizeIncludePaths) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    StringRef InFile) override {
    Filename = InFile;
    Compiler.getPreprocessor().addPPCallbacks(
        llvm::make_unique<PreprocessorHooks>(this));
    return llvm::make_unique<clang::ASTConsumer>();
  }

  void ExecuteAction() override {
    clang::CompilerInstance *Compiler = &getCompilerInstance();
    assert(!Compiler->hasSema() && "CI already has Sema");

    // Set up our hooks into sema and parse the AST.
    if (hasCodeCompletionSupport() &&
        !Compiler->getFrontendOpts().CodeCompletionAt.FileName.empty())
      Compiler->createCodeCompletionConsumer();

    clang::CodeCompleteConsumer *CompletionConsumer = nullptr;
    if (Compiler->hasCodeCompletionConsumer())
      CompletionConsumer = &Compiler->getCodeCompletionConsumer();

    Compiler->createSema(getTranslationUnitKind(), CompletionConsumer);
    Compiler->getSema().addExternalSource(this);

    clang::ParseAST(Compiler->getSema(), Compiler->getFrontendOpts().ShowStats,
                    Compiler->getFrontendOpts().SkipFunctionBodies);
  }

  /// Callback for incomplete types. If we encounter a forward declaration we
  /// have the fully qualified name ready. Just query that.
  bool MaybeDiagnoseMissingCompleteType(clang::SourceLocation Loc,
                                        clang::QualType T) override {
    // Ignore spurious callbacks from SFINAE contexts.
    if (getCompilerInstance().getSema().isSFINAEContext())
      return false;

    clang::ASTContext &context = getCompilerInstance().getASTContext();
    query(T.getUnqualifiedType().getAsString(context.getPrintingPolicy()), Loc);
    return false;
  }

  /// Callback for unknown identifiers. Try to piece together as much
  /// qualification as we can get and do a query.
  clang::TypoCorrection CorrectTypo(const DeclarationNameInfo &Typo,
                                    int LookupKind, Scope *S, CXXScopeSpec *SS,
                                    CorrectionCandidateCallback &CCC,
                                    DeclContext *MemberContext,
                                    bool EnteringContext,
                                    const ObjCObjectPointerType *OPT) override {
    // Ignore spurious callbacks from SFINAE contexts.
    if (getCompilerInstance().getSema().isSFINAEContext())
      return clang::TypoCorrection();

    std::string TypoScopeString;
    if (S) {
      // FIXME: Currently we only use namespace contexts. Use other context
      // types for query.
      for (const auto *Context = S->getEntity(); Context;
           Context = Context->getParent()) {
        if (const auto *ND = dyn_cast<NamespaceDecl>(Context)) {
          if (!ND->getName().empty())
            TypoScopeString = ND->getNameAsString() + "::" + TypoScopeString;
        }
      }
    }

    /// If we have a scope specification, use that to get more precise results.
    std::string QueryString;
    if (SS && SS->getRange().isValid()) {
      auto Range = CharSourceRange::getTokenRange(SS->getRange().getBegin(),
                                                  Typo.getLoc());
      StringRef Source =
          Lexer::getSourceText(Range, getCompilerInstance().getSourceManager(),
                               getCompilerInstance().getLangOpts());

      // Skip forward until we find a character that's neither identifier nor
      // colon. This is a bit of a hack around the fact that we will only get a
      // single callback for a long nested name if a part of the beginning is
      // unknown. For example:
      //
      // llvm::sys::path::parent_path(...)
      // ^~~~  ^~~
      //    known
      //            ^~~~
      //      unknown, last callback
      //                  ^~~~~~~~~~~
      //                  no callback
      //
      // With the extension we get the full nested name specifier including
      // parent_path.
      // FIXME: Don't rely on source text.
      const char *End = Source.end();
      while (isIdentifierBody(*End) || *End == ':')
        ++End;

      QueryString = std::string(Source.begin(), End);
    } else {
      QueryString = Typo.getAsString();
    }

    // Follow C++ Lookup rules. Firstly, lookup the identifier with scoped
    // namespace contexts. If fails, falls back to identifier.
    // For example:
    //
    // namespace a {
    // b::foo f;
    // }
    //
    // 1. lookup a::b::foo.
    // 2. lookup b::foo.
    if (!query(TypoScopeString + QueryString, Typo.getLoc()))
      query(QueryString, Typo.getLoc());

    // FIXME: We should just return the name we got as input here and prevent
    // clang from trying to correct the typo by itself. That may change the
    // identifier to something that's not wanted by the user.
    return clang::TypoCorrection();
  }

  StringRef filename() const { return Filename; }

  /// Called for each include file we discover is in the file.
  /// \param SourceManager the active SourceManager
  /// \param canonical_path the canonical path to the include file
  /// \param uttered_path the path as it appeared in the program
  /// \param IsAngled whether angle brackets were used
  /// \param HashLocation the source location of the include's \#
  /// \param EndLocation the source location following the include
  void NextInclude(clang::SourceManager *SourceManager,
                   llvm::StringRef canonical_path, llvm::StringRef uttered_path,
                   bool IsAngled, clang::SourceLocation HashLocation,
                   clang::SourceLocation EndLocation) {
    unsigned Offset = SourceManager->getFileOffset(HashLocation);
    if (FirstIncludeOffset == -1U)
      FirstIncludeOffset = Offset;
  }

  /// Get the minimal include for a given path.
  std::string minimizeInclude(StringRef Include,
                              clang::SourceManager &SourceManager,
                              clang::HeaderSearch &HeaderSearch) {
    if (!MinimizeIncludePaths)
      return Include;

    // Get the FileEntry for the include.
    StringRef StrippedInclude = Include.trim("\"<>");
    const FileEntry *Entry =
        SourceManager.getFileManager().getFile(StrippedInclude);

    // If the file doesn't exist return the path from the database.
    // FIXME: This should never happen.
    if (!Entry)
      return Include;

    bool IsSystem;
    std::string Suggestion =
        HeaderSearch.suggestPathToFileForDiagnostics(Entry, &IsSystem);

    return IsSystem ? '<' + Suggestion + '>' : '"' + Suggestion + '"';
  }

  /// Generate replacements for the suggested includes.
  /// \return true if changes will be made, false otherwise.
  bool Rewrite(clang::SourceManager &SourceManager,
               clang::HeaderSearch &HeaderSearch,
               std::vector<clang::tooling::Replacement> &replacements) {
    if (Untried.empty())
      return false;

    const auto &ToTry = UntriedList.front();
    std::string ToAdd = "#include " +
                        minimizeInclude(ToTry, SourceManager, HeaderSearch) +
                        "\n";
    DEBUG(llvm::dbgs() << "Adding " << ToAdd << "\n");

    if (FirstIncludeOffset == -1U)
      FirstIncludeOffset = 0;

    replacements.push_back(clang::tooling::Replacement(
        SourceManager, FileBegin.getLocWithOffset(FirstIncludeOffset), 0,
        ToAdd));

    // We currently abort after the first inserted include. The more
    // includes we have the less safe this becomes due to error recovery
    // changing the results.
    // FIXME: Handle multiple includes at once.
    return true;
  }

  /// Sets the location at the very top of the file.
  void setFileBegin(clang::SourceLocation Location) { FileBegin = Location; }

  /// Add an include to the set of includes to try.
  /// \param include_path The include path to try.
  void TryInclude(const std::string &query, const std::string &include_path) {
    if (Untried.insert(include_path).second)
      UntriedList.push_back(include_path);
  }

private:
  /// Query the database for a given identifier.
  bool query(StringRef Query, SourceLocation Loc) {
    assert(!Query.empty() && "Empty query!");

    // Save database lookups by not looking up identifiers multiple times.
    if (!SeenQueries.insert(Query).second)
      return true;

    DEBUG(llvm::dbgs() << "Looking up '" << Query << "' at ");
    DEBUG(Loc.print(llvm::dbgs(), getCompilerInstance().getSourceManager()));
    DEBUG(llvm::dbgs() << " ...");

    std::string error_text;
    auto SearchReply = SymbolIndexMgr.search(Query);
    DEBUG(llvm::dbgs() << SearchReply.size() << " replies\n");
    if (SearchReply.empty())
      return false;

    // Add those files to the set of includes to try out.
    // FIXME: Rank the results and pick the best one instead of the first one.
    TryInclude(Query, SearchReply[0]);

    return true;
  }

  /// The client to use to find cross-references.
  SymbolIndexManager &SymbolIndexMgr;

  // Remeber things we looked up to avoid querying things twice.
  llvm::StringSet<> SeenQueries;

  /// The absolute path to the file being processed.
  std::string Filename;

  /// The location of the beginning of the tracked file.
  clang::SourceLocation FileBegin;

  /// The offset of the last include in the original source file. This will
  /// be used as the insertion point for new include directives.
  unsigned FirstIncludeOffset = -1U;

  /// Includes we have left to try. A set to unique them and a list to keep
  /// track of the order. We prefer includes that were discovered early to avoid
  /// getting caught in results from error recovery.
  std::set<std::string> Untried;
  std::vector<std::string> UntriedList;

  /// Whether we should use the smallest possible include path.
  bool MinimizeIncludePaths = true;
};

void PreprocessorHooks::FileChanged(clang::SourceLocation Loc,
                                    clang::PPCallbacks::FileChangeReason Reason,
                                    clang::SrcMgr::CharacteristicKind FileType,
                                    clang::FileID PrevFID) {
  // Remember where the main file starts.
  if (Reason == clang::PPCallbacks::EnterFile) {
    clang::SourceManager *SourceManager =
        &EnclosingPass->getCompilerInstance().getSourceManager();
    clang::FileID loc_id = SourceManager->getFileID(Loc);
    if (const clang::FileEntry *FileEntry =
            SourceManager->getFileEntryForID(loc_id)) {
      if (FileEntry->getName() == EnclosingPass->filename()) {
        EnclosingPass->setFileBegin(Loc);
        TrackedFile = FileEntry;
      }
    }
  }
}

void PreprocessorHooks::InclusionDirective(
    clang::SourceLocation HashLocation, const clang::Token &IncludeToken,
    llvm::StringRef FileName, bool IsAngled,
    clang::CharSourceRange FileNameRange, const clang::FileEntry *IncludeFile,
    llvm::StringRef SearchPath, llvm::StringRef relative_path,
    const clang::Module *imported) {
  // Remember include locations so we can insert our new include at the end of
  // the include block.
  clang::SourceManager *SourceManager =
      &EnclosingPass->getCompilerInstance().getSourceManager();
  auto IDPosition = SourceManager->getDecomposedExpansionLoc(HashLocation);
  const FileEntry *SourceFile =
      SourceManager->getFileEntryForID(IDPosition.first);
  if (!IncludeFile || TrackedFile != SourceFile)
    return;
  EnclosingPass->NextInclude(SourceManager, IncludeFile->getName(), FileName,
                             IsAngled, HashLocation, FileNameRange.getEnd());
}

} // namespace

IncludeFixerActionFactory::IncludeFixerActionFactory(
    SymbolIndexManager &SymbolIndexMgr,
    std::vector<clang::tooling::Replacement> &Replacements,
    bool MinimizeIncludePaths)
    : SymbolIndexMgr(SymbolIndexMgr), Replacements(Replacements),
      MinimizeIncludePaths(MinimizeIncludePaths) {}

IncludeFixerActionFactory::~IncludeFixerActionFactory() = default;

bool IncludeFixerActionFactory::runInvocation(
    clang::CompilerInvocation *Invocation, clang::FileManager *Files,
    std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
    clang::DiagnosticConsumer *Diagnostics) {
  assert(Invocation->getFrontendOpts().Inputs.size() == 1);

  // Set up Clang.
  clang::CompilerInstance Compiler(PCHContainerOps);
  Compiler.setInvocation(Invocation);
  Compiler.setFileManager(Files);

  // Create the compiler's actual diagnostics engine. We want to drop all
  // diagnostics here.
  Compiler.createDiagnostics(new clang::IgnoringDiagConsumer,
                             /*ShouldOwnClient=*/true);
  Compiler.createSourceManager(*Files);

  // Run the parser, gather missing includes.
  auto ScopedToolAction =
      llvm::make_unique<Action>(SymbolIndexMgr, MinimizeIncludePaths);
  Compiler.ExecuteAction(*ScopedToolAction);

  // Generate replacements.
  ScopedToolAction->Rewrite(Compiler.getSourceManager(),
                            Compiler.getPreprocessor().getHeaderSearchInfo(),
                            Replacements);

  // Technically this should only return true if we're sure that we have a
  // parseable file. We don't know that though.
  return true;
}

} // namespace include_fixer
} // namespace clang
