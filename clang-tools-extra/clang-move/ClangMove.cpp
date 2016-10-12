//===-- ClangMove.cpp - Implement ClangMove functationalities ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangMove.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/Path.h"

using namespace clang::ast_matchers;

namespace clang {
namespace move {
namespace {

// Make the Path absolute using the CurrentDir if the Path is not an absolute
// path. An empty Path will result in an empty string.
std::string MakeAbsolutePath(StringRef CurrentDir, StringRef Path) {
  if (Path.empty())
    return "";
  llvm::SmallString<128> InitialDirectory(CurrentDir);
  llvm::SmallString<128> AbsolutePath(Path);
  if (std::error_code EC =
          llvm::sys::fs::make_absolute(InitialDirectory, AbsolutePath))
    llvm::errs() << "Warning: could not make absolute file: '" <<  EC.message()
                 << '\n';
  llvm::sys::path::remove_dots(AbsolutePath, /*remove_dot_dot=*/true);
  llvm::sys::path::native(AbsolutePath);
  return AbsolutePath.str();
}

// Make the Path absolute using the current working directory of the given
// SourceManager if the Path is not an absolute path.
//
// The Path can be a path relative to the build directory, or retrieved from
// the SourceManager.
std::string MakeAbsolutePath(const SourceManager& SM, StringRef Path) {
  llvm::SmallString<128> AbsolutePath(Path);
  if (std::error_code EC =
       SM.getFileManager().getVirtualFileSystem()->makeAbsolute(AbsolutePath))
    llvm::errs() << "Warning: could not make absolute file: '" <<  EC.message()
                 << '\n';
  // Handle symbolic link path cases.
  // We are trying to get the real file path of the symlink.
  const DirectoryEntry *Dir = SM.getFileManager().getDirectory(
       llvm::sys::path::parent_path(AbsolutePath.str()));
  if (Dir) {
    StringRef DirName = SM.getFileManager().getCanonicalName(Dir);
    SmallVector<char, 128> AbsoluteFilename;
    llvm::sys::path::append(AbsoluteFilename, DirName,
                            llvm::sys::path::filename(AbsolutePath.str()));
    return llvm::StringRef(AbsoluteFilename.data(), AbsoluteFilename.size())
        .str();
  }
  return AbsolutePath.str();
}

// Matches AST nodes that are expanded within the given AbsoluteFilePath.
AST_POLYMORPHIC_MATCHER_P(isExpansionInFile,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt, TypeLoc),
                          std::string, AbsoluteFilePath) {
  auto &SourceManager = Finder->getASTContext().getSourceManager();
  auto ExpansionLoc = SourceManager.getExpansionLoc(Node.getLocStart());
  if (ExpansionLoc.isInvalid())
    return false;
  auto FileEntry =
      SourceManager.getFileEntryForID(SourceManager.getFileID(ExpansionLoc));
  if (!FileEntry)
    return false;
  return MakeAbsolutePath(SourceManager, FileEntry->getName()) ==
         AbsoluteFilePath;
}

class FindAllIncludes : public clang::PPCallbacks {
public:
  explicit FindAllIncludes(SourceManager *SM, ClangMoveTool *const MoveTool)
      : SM(*SM), MoveTool(MoveTool) {}

  void InclusionDirective(clang::SourceLocation HashLoc,
                          const clang::Token & /*IncludeTok*/,
                          StringRef FileName, bool IsAngled,
                          clang::CharSourceRange /*FilenameRange*/,
                          const clang::FileEntry * /*File*/,
                          StringRef SearchPath, StringRef /*RelativePath*/,
                          const clang::Module * /*Imported*/) override {
    if (const auto *FileEntry = SM.getFileEntryForID(SM.getFileID(HashLoc)))
      MoveTool->addIncludes(FileName, IsAngled, SearchPath,
                            FileEntry->getName(), SM);
  }

private:
  const SourceManager &SM;
  ClangMoveTool *const MoveTool;
};

// Expand to get the end location of the line where the EndLoc of the given
// Decl.
SourceLocation
getLocForEndOfDecl(const clang::Decl *D, const SourceManager *SM,
                   const LangOptions &LangOpts = clang::LangOptions()) {
  std::pair<FileID, unsigned> LocInfo = SM->getDecomposedLoc(D->getLocEnd());
  // Try to load the file buffer.
  bool InvalidTemp = false;
  llvm::StringRef File = SM->getBufferData(LocInfo.first, &InvalidTemp);
  if (InvalidTemp)
    return SourceLocation();

  const char *TokBegin = File.data() + LocInfo.second;
  // Lex from the start of the given location.
  Lexer Lex(SM->getLocForStartOfFile(LocInfo.first), LangOpts, File.begin(),
            TokBegin, File.end());

  llvm::SmallVector<char, 16> Line;
  // FIXME: this is a bit hacky to get ReadToEndOfLine work.
  Lex.setParsingPreprocessorDirective(true);
  Lex.ReadToEndOfLine(&Line);
  SourceLocation EndLoc  = D->getLocEnd().getLocWithOffset(Line.size());
  // If we already reach EOF, just return the EOF SourceLocation;
  // otherwise, move 1 offset ahead to include the trailing newline character
  // '\n'.
  return SM->getLocForEndOfFile(LocInfo.first) == EndLoc
             ? EndLoc
             : EndLoc.getLocWithOffset(1);
}

// Get full range of a Decl including the comments associated with it.
clang::CharSourceRange
GetFullRange(const clang::SourceManager *SM, const clang::Decl *D,
             const clang::LangOptions &options = clang::LangOptions()) {
  clang::SourceRange Full = D->getSourceRange();
  Full.setEnd(getLocForEndOfDecl(D, SM));
  // Expand to comments that are associated with the Decl.
  if (const auto* Comment =
          D->getASTContext().getRawCommentForDeclNoCache(D)) {
    if (SM->isBeforeInTranslationUnit(Full.getEnd(), Comment->getLocEnd()))
      Full.setEnd(Comment->getLocEnd());
    // FIXME: Don't delete a preceding comment, if there are no other entities
    // it could refer to.
    if (SM->isBeforeInTranslationUnit(Comment->getLocStart(),
                                      Full.getBegin()))
      Full.setBegin(Comment->getLocStart());
  }

  return clang::CharSourceRange::getCharRange(Full);
}

std::string getDeclarationSourceText(const clang::Decl *D,
                                     const clang::SourceManager *SM) {
  llvm::StringRef SourceText = clang::Lexer::getSourceText(
      GetFullRange(SM, D), *SM, clang::LangOptions());
  return SourceText.str();
}

clang::tooling::Replacement
getReplacementInChangedCode(const clang::tooling::Replacements &Replacements,
                            const clang::tooling::Replacement &Replacement) {
  unsigned Start = Replacements.getShiftedCodePosition(Replacement.getOffset());
  unsigned End = Replacements.getShiftedCodePosition(Replacement.getOffset() +
                                                     Replacement.getLength());
  return clang::tooling::Replacement(Replacement.getFilePath(), Start,
                                     End - Start,
                                     Replacement.getReplacementText());
}

void addOrMergeReplacement(const clang::tooling::Replacement &Replacement,
                           clang::tooling::Replacements *Replacements) {
  auto Err = Replacements->add(Replacement);
  if (Err) {
    llvm::consumeError(std::move(Err));
    auto Replace = getReplacementInChangedCode(*Replacements, Replacement);
    *Replacements = Replacements->merge(clang::tooling::Replacements(Replace));
  }
}

bool isInHeaderFile(const clang::SourceManager &SM, const clang::Decl *D,
                    llvm::StringRef OriginalRunningDirectory,
                    llvm::StringRef OldHeader) {
  if (OldHeader.empty())
    return false;
  auto ExpansionLoc = SM.getExpansionLoc(D->getLocStart());
  if (ExpansionLoc.isInvalid())
    return false;

  if (const auto *FE = SM.getFileEntryForID(SM.getFileID(ExpansionLoc))) {
    return MakeAbsolutePath(SM, FE->getName()) ==
           MakeAbsolutePath(OriginalRunningDirectory, OldHeader);
  }

  return false;
}

std::vector<std::string> GetNamespaces(const clang::Decl *D) {
  std::vector<std::string> Namespaces;
  for (const auto *Context = D->getDeclContext(); Context;
       Context = Context->getParent()) {
    if (llvm::isa<clang::TranslationUnitDecl>(Context) ||
        llvm::isa<clang::LinkageSpecDecl>(Context))
      break;

    if (const auto *ND = llvm::dyn_cast<clang::NamespaceDecl>(Context))
      Namespaces.push_back(ND->getName().str());
  }
  std::reverse(Namespaces.begin(), Namespaces.end());
  return Namespaces;
}

clang::tooling::Replacements
createInsertedReplacements(const std::vector<std::string> &Includes,
                           const std::vector<ClangMoveTool::MovedDecl> &Decls,
                           llvm::StringRef FileName) {
  clang::tooling::Replacements InsertedReplacements;

  // Add #Includes.
  std::string AllIncludesString;
  // FIXME: Add header guard.
  for (const auto &Include : Includes)
    AllIncludesString += Include;

  if (!AllIncludesString.empty()) {
    clang::tooling::Replacement InsertInclude(FileName, 0, 0,
                                              AllIncludesString + "\n");
    addOrMergeReplacement(InsertInclude, &InsertedReplacements);
  }

  // Add moved class definition and its related declarations. All declarations
  // in same namespace are grouped together.
  std::vector<std::string> CurrentNamespaces;
  for (const auto &MovedDecl : Decls) {
    std::vector<std::string> DeclNamespaces = GetNamespaces(MovedDecl.Decl);
    auto CurrentIt = CurrentNamespaces.begin();
    auto DeclIt = DeclNamespaces.begin();
    while (CurrentIt != CurrentNamespaces.end() &&
           DeclIt != DeclNamespaces.end()) {
      if (*CurrentIt != *DeclIt)
        break;
      ++CurrentIt;
      ++DeclIt;
    }
    std::vector<std::string> NextNamespaces(CurrentNamespaces.begin(),
                                            CurrentIt);
    NextNamespaces.insert(NextNamespaces.end(), DeclIt, DeclNamespaces.end());
    auto RemainingSize = CurrentNamespaces.end() - CurrentIt;
    for (auto It = CurrentNamespaces.rbegin(); RemainingSize > 0;
         --RemainingSize, ++It) {
      assert(It < CurrentNamespaces.rend());
      auto code = "} // namespace " + *It + "\n";
      clang::tooling::Replacement InsertedReplacement(FileName, 0, 0, code);
      addOrMergeReplacement(InsertedReplacement, &InsertedReplacements);
    }
    while (DeclIt != DeclNamespaces.end()) {
      clang::tooling::Replacement InsertedReplacement(
          FileName, 0, 0, "namespace " + *DeclIt + " {\n");
      addOrMergeReplacement(InsertedReplacement, &InsertedReplacements);
      ++DeclIt;
    }

    clang::tooling::Replacement InsertedReplacement(
        FileName, 0, 0, getDeclarationSourceText(MovedDecl.Decl, MovedDecl.SM));
    addOrMergeReplacement(InsertedReplacement, &InsertedReplacements);

    CurrentNamespaces = std::move(NextNamespaces);
  }
  std::reverse(CurrentNamespaces.begin(), CurrentNamespaces.end());
  for (const auto &NS : CurrentNamespaces) {
    clang::tooling::Replacement InsertedReplacement(
        FileName, 0, 0, "} // namespace " + NS + "\n");
    addOrMergeReplacement(InsertedReplacement, &InsertedReplacements);
  }
  return InsertedReplacements;
}

} // namespace

std::unique_ptr<clang::ASTConsumer>
ClangMoveAction::CreateASTConsumer(clang::CompilerInstance &Compiler,
                                   StringRef /*InFile*/) {
  Compiler.getPreprocessor().addPPCallbacks(llvm::make_unique<FindAllIncludes>(
      &Compiler.getSourceManager(), &MoveTool));
  return MatchFinder.newASTConsumer();
}

ClangMoveTool::ClangMoveTool(
    const MoveDefinitionSpec &MoveSpec,
    std::map<std::string, tooling::Replacements> &FileToReplacements,
    llvm::StringRef OriginalRunningDirectory, llvm::StringRef FallbackStyle)
    : Spec(MoveSpec), FileToReplacements(FileToReplacements),
      OriginalRunningDirectory(OriginalRunningDirectory),
      FallbackStyle(FallbackStyle) {
  Spec.Name = llvm::StringRef(Spec.Name).ltrim(':');
  if (!Spec.NewHeader.empty())
    CCIncludes.push_back("#include \"" + Spec.NewHeader + "\"\n");
}

void ClangMoveTool::registerMatchers(ast_matchers::MatchFinder *Finder) {
  std::string FullyQualifiedName = "::" + Spec.Name;
  auto InOldHeader = isExpansionInFile(
      MakeAbsolutePath(OriginalRunningDirectory, Spec.OldHeader));
  auto InOldCC = isExpansionInFile(
      MakeAbsolutePath(OriginalRunningDirectory, Spec.OldCC));
  auto InOldFiles = anyOf(InOldHeader, InOldCC);
  auto InMovedClass =
      hasDeclContext(cxxRecordDecl(hasName(FullyQualifiedName)));

  // Match moved class declarations.
  auto MovedClass = cxxRecordDecl(
      InOldFiles, hasName(FullyQualifiedName), isDefinition(),
      hasDeclContext(anyOf(namespaceDecl(), translationUnitDecl())));
  Finder->addMatcher(MovedClass.bind("moved_class"), this);

  // Match moved class methods (static methods included) which are defined
  // outside moved class declaration.
  Finder->addMatcher(cxxMethodDecl(InOldFiles,
                                   ofClass(hasName(FullyQualifiedName)),
                                   isDefinition())
                         .bind("class_method"),
                     this);

  // Match static member variable definition of the moved class.
  Finder->addMatcher(varDecl(InMovedClass, InOldCC, isDefinition())
                         .bind("class_static_var_decl"),
                     this);

  auto inAnonymousNamespace = hasParent(namespaceDecl(isAnonymous()));
  // Match functions/variables definitions which are defined in anonymous
  // namespace in old cc.
  Finder->addMatcher(
      namedDecl(anyOf(functionDecl(isDefinition()), varDecl(isDefinition())),
                inAnonymousNamespace)
          .bind("decls_in_anonymous_ns"),
      this);

  // Match static functions/variabale definitions in old cc.
  Finder->addMatcher(
      namedDecl(anyOf(functionDecl(isDefinition(), unless(InMovedClass),
                                   isStaticStorageClass(), InOldCC),
                      varDecl(isDefinition(), unless(InMovedClass),
                              isStaticStorageClass(), InOldCC)))
          .bind("static_decls"),
      this);

  // Match forward declarations in old header.
  Finder->addMatcher(
      cxxRecordDecl(unless(anyOf(isImplicit(), isDefinition())), InOldHeader)
          .bind("fwd_decl"),
      this);
}

void ClangMoveTool::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *CMD =
          Result.Nodes.getNodeAs<clang::CXXMethodDecl>("class_method")) {
    // Skip inline class methods. isInline() ast matcher doesn't ignore this
    // case.
    if (!CMD->isInlined()) {
      MovedDecls.emplace_back(CMD, &Result.Context->getSourceManager());
      RemovedDecls.push_back(MovedDecls.back());
    }
  } else if (const auto *VD = Result.Nodes.getNodeAs<clang::VarDecl>(
                 "class_static_var_decl")) {
    MovedDecls.emplace_back(VD, &Result.Context->getSourceManager());
    RemovedDecls.push_back(MovedDecls.back());
  } else if (const auto *class_decl =
                 Result.Nodes.getNodeAs<clang::CXXRecordDecl>("moved_class")) {
    MovedDecls.emplace_back(class_decl, &Result.Context->getSourceManager());
    RemovedDecls.push_back(MovedDecls.back());
  } else if (const auto *FWD =
                 Result.Nodes.getNodeAs<clang::CXXRecordDecl>("fwd_decl")) {
    // Skip all forwad declarations which appear after moved class declaration.
    if (RemovedDecls.empty())
      MovedDecls.emplace_back(FWD, &Result.Context->getSourceManager());
  } else if (const auto *FD = Result.Nodes.getNodeAs<clang::NamedDecl>(
                 "decls_in_anonymous_ns")) {
    MovedDecls.emplace_back(FD, &Result.Context->getSourceManager());
  } else if (const auto *ND =
                 Result.Nodes.getNodeAs<clang::NamedDecl>("static_decls")) {
    MovedDecls.emplace_back(ND, &Result.Context->getSourceManager());
  }
}

void ClangMoveTool::addIncludes(llvm::StringRef IncludeHeader,
                                bool IsAngled,
                                llvm::StringRef SearchPath,
                                llvm::StringRef FileName,
                                const SourceManager& SM) {
  SmallVector<char, 128> HeaderWithSearchPath;
  llvm::sys::path::append(HeaderWithSearchPath, SearchPath, IncludeHeader);
  std::string AbsoluteOldHeader =
      MakeAbsolutePath(OriginalRunningDirectory, Spec.OldHeader);
  // FIXME: Add old.h to the new.cc/h when the new target has dependencies on
  // old.h/c. For instance, when moved class uses another class defined in
  // old.h, the old.h should be added in new.h.
  if (AbsoluteOldHeader ==
      MakeAbsolutePath(SM, llvm::StringRef(HeaderWithSearchPath.data(),
                                           HeaderWithSearchPath.size())))
    return;

  std::string IncludeLine =
      IsAngled ? ("#include <" + IncludeHeader + ">\n").str()
               : ("#include \"" + IncludeHeader + "\"\n").str();

  std::string AbsoluteCurrentFile = MakeAbsolutePath(SM, FileName);
  if (AbsoluteOldHeader == AbsoluteCurrentFile) {
    HeaderIncludes.push_back(IncludeLine);
  } else if (MakeAbsolutePath(OriginalRunningDirectory, Spec.OldCC) ==
             AbsoluteCurrentFile) {
    CCIncludes.push_back(IncludeLine);
  }
}

void ClangMoveTool::removeClassDefinitionInOldFiles() {
  for (const auto &MovedDecl : RemovedDecls) {
    const auto &SM = *MovedDecl.SM;
    auto Range = GetFullRange(&SM, MovedDecl.Decl);
    clang::tooling::Replacement RemoveReplacement(
        *MovedDecl.SM, clang::CharSourceRange::getCharRange(
                           Range.getBegin(), Range.getEnd()),
        "");
    std::string FilePath = RemoveReplacement.getFilePath().str();
    addOrMergeReplacement(RemoveReplacement, &FileToReplacements[FilePath]);

    llvm::StringRef Code =
        SM.getBufferData(SM.getFileID(MovedDecl.Decl->getLocation()));
    format::FormatStyle Style =
        format::getStyle("file", FilePath, FallbackStyle);
    auto CleanReplacements = format::cleanupAroundReplacements(
        Code, FileToReplacements[FilePath], Style);

    if (!CleanReplacements) {
      llvm::errs() << llvm::toString(CleanReplacements.takeError()) << "\n";
      continue;
    }
    FileToReplacements[FilePath] = *CleanReplacements;
  }
}

void ClangMoveTool::moveClassDefinitionToNewFiles() {
  std::vector<MovedDecl> NewHeaderDecls;
  std::vector<MovedDecl> NewCCDecls;
  for (const auto &MovedDecl : MovedDecls) {
    if (isInHeaderFile(*MovedDecl.SM, MovedDecl.Decl, OriginalRunningDirectory,
                       Spec.OldHeader))
      NewHeaderDecls.push_back(MovedDecl);
    else
      NewCCDecls.push_back(MovedDecl);
  }

  if (!Spec.NewHeader.empty())
    FileToReplacements[Spec.NewHeader] = createInsertedReplacements(
        HeaderIncludes, NewHeaderDecls, Spec.NewHeader);
  if (!Spec.NewCC.empty())
    FileToReplacements[Spec.NewCC] =
        createInsertedReplacements(CCIncludes, NewCCDecls, Spec.NewCC);
}

void ClangMoveTool::onEndOfTranslationUnit() {
  if (RemovedDecls.empty())
    return;
  removeClassDefinitionInOldFiles();
  moveClassDefinitionToNewFiles();
}

} // namespace move
} // namespace clang
