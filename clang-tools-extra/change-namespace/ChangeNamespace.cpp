//===-- ChangeNamespace.cpp - Change namespace implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "ChangeNamespace.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace change_namespace {

namespace {

inline std::string
joinNamespaces(const llvm::SmallVectorImpl<StringRef> &Namespaces) {
  if (Namespaces.empty())
    return "";
  std::string Result = Namespaces.front();
  for (auto I = Namespaces.begin() + 1, E = Namespaces.end(); I != E; ++I)
    Result += ("::" + *I).str();
  return Result;
}

SourceLocation startLocationForType(TypeLoc TLoc) {
  // For elaborated types (e.g. `struct a::A`) we want the portion after the
  // `struct` but including the namespace qualifier, `a::`.
  if (TLoc.getTypeLocClass() == TypeLoc::Elaborated) {
    NestedNameSpecifierLoc NestedNameSpecifier =
        TLoc.castAs<ElaboratedTypeLoc>().getQualifierLoc();
    if (NestedNameSpecifier.getNestedNameSpecifier())
      return NestedNameSpecifier.getBeginLoc();
    TLoc = TLoc.getNextTypeLoc();
  }
  return TLoc.getLocStart();
}

SourceLocation EndLocationForType(TypeLoc TLoc) {
  // Dig past any namespace or keyword qualifications.
  while (TLoc.getTypeLocClass() == TypeLoc::Elaborated ||
         TLoc.getTypeLocClass() == TypeLoc::Qualified)
    TLoc = TLoc.getNextTypeLoc();

  // The location for template specializations (e.g. Foo<int>) includes the
  // templated types in its location range.  We want to restrict this to just
  // before the `<` character.
  if (TLoc.getTypeLocClass() == TypeLoc::TemplateSpecialization)
    return TLoc.castAs<TemplateSpecializationTypeLoc>()
        .getLAngleLoc()
        .getLocWithOffset(-1);
  return TLoc.getEndLoc();
}

// Returns the containing namespace of `InnerNs` by skipping `PartialNsName`.
// If the `InnerNs` does not have `PartialNsName` as suffix, nullptr is
// returned.
// For example, if `InnerNs` is "a::b::c" and `PartialNsName` is "b::c", then
// the NamespaceDecl of namespace "a" will be returned.
const NamespaceDecl *getOuterNamespace(const NamespaceDecl *InnerNs,
                                       llvm::StringRef PartialNsName) {
  const auto *CurrentContext = llvm::cast<DeclContext>(InnerNs);
  const auto *CurrentNs = InnerNs;
  llvm::SmallVector<llvm::StringRef, 4> PartialNsNameSplitted;
  PartialNsName.split(PartialNsNameSplitted, "::");
  while (!PartialNsNameSplitted.empty()) {
    // Get the inner-most namespace in CurrentContext.
    while (CurrentContext && !llvm::isa<NamespaceDecl>(CurrentContext))
      CurrentContext = CurrentContext->getParent();
    if (!CurrentContext)
      return nullptr;
    CurrentNs = llvm::cast<NamespaceDecl>(CurrentContext);
    if (PartialNsNameSplitted.back() != CurrentNs->getNameAsString())
      return nullptr;
    PartialNsNameSplitted.pop_back();
    CurrentContext = CurrentContext->getParent();
  }
  return CurrentNs;
}

// FIXME: get rid of this helper function if this is supported in clang-refactor
// library.
SourceLocation getStartOfNextLine(SourceLocation Loc, const SourceManager &SM,
                                  const LangOptions &LangOpts) {
  if (Loc.isMacroID() &&
      !Lexer::isAtEndOfMacroExpansion(Loc, SM, LangOpts, &Loc))
    return SourceLocation();
  // Break down the source location.
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
  // Try to load the file buffer.
  bool InvalidTemp = false;
  llvm::StringRef File = SM.getBufferData(LocInfo.first, &InvalidTemp);
  if (InvalidTemp)
    return SourceLocation();

  const char *TokBegin = File.data() + LocInfo.second;
  // Lex from the start of the given location.
  Lexer Lex(SM.getLocForStartOfFile(LocInfo.first), LangOpts, File.begin(),
            TokBegin, File.end());

  llvm::SmallVector<char, 16> Line;
  // FIXME: this is a bit hacky to get ReadToEndOfLine work.
  Lex.setParsingPreprocessorDirective(true);
  Lex.ReadToEndOfLine(&Line);
  // FIXME: should not +1 at EOF.
  return Loc.getLocWithOffset(Line.size() + 1);
}

// Returns `R` with new range that refers to code after `Replaces` being
// applied.
tooling::Replacement
getReplacementInChangedCode(const tooling::Replacements &Replaces,
                            const tooling::Replacement &R) {
  unsigned NewStart = Replaces.getShiftedCodePosition(R.getOffset());
  unsigned NewEnd =
      Replaces.getShiftedCodePosition(R.getOffset() + R.getLength());
  return tooling::Replacement(R.getFilePath(), NewStart, NewEnd - NewStart,
                              R.getReplacementText());
}

// Adds a replacement `R` into `Replaces` or merges it into `Replaces` by
// applying all existing Replaces first if there is conflict.
void addOrMergeReplacement(const tooling::Replacement &R,
                           tooling::Replacements *Replaces) {
  auto Err = Replaces->add(R);
  if (Err) {
    llvm::consumeError(std::move(Err));
    auto Replace = getReplacementInChangedCode(*Replaces, R);
    *Replaces = Replaces->merge(tooling::Replacements(Replace));
  }
}

tooling::Replacement createReplacement(SourceLocation Start, SourceLocation End,
                                       llvm::StringRef ReplacementText,
                                       const SourceManager &SM) {
  if (!Start.isValid() || !End.isValid()) {
    llvm::errs() << "start or end location were invalid\n";
    return tooling::Replacement();
  }
  if (SM.getDecomposedLoc(Start).first != SM.getDecomposedLoc(End).first) {
    llvm::errs()
        << "start or end location were in different macro expansions\n";
    return tooling::Replacement();
  }
  Start = SM.getSpellingLoc(Start);
  End = SM.getSpellingLoc(End);
  if (SM.getFileID(Start) != SM.getFileID(End)) {
    llvm::errs() << "start or end location were in different files\n";
    return tooling::Replacement();
  }
  return tooling::Replacement(
      SM, CharSourceRange::getTokenRange(SM.getSpellingLoc(Start),
                                         SM.getSpellingLoc(End)),
      ReplacementText);
}

tooling::Replacement createInsertion(SourceLocation Loc,
                                     llvm::StringRef InsertText,
                                     const SourceManager &SM) {
  if (Loc.isInvalid()) {
    llvm::errs() << "insert Location is invalid.\n";
    return tooling::Replacement();
  }
  Loc = SM.getSpellingLoc(Loc);
  return tooling::Replacement(SM, Loc, 0, InsertText);
}

// Returns the shortest qualified name for declaration `DeclName` in the
// namespace `NsName`. For example, if `DeclName` is "a::b::X" and `NsName`
// is "a::c::d", then "b::X" will be returned.
std::string getShortestQualifiedNameInNamespace(llvm::StringRef DeclName,
                                                llvm::StringRef NsName) {
  llvm::SmallVector<llvm::StringRef, 4> DeclNameSplitted;
  DeclName.split(DeclNameSplitted, "::");
  if (DeclNameSplitted.size() == 1)
    return DeclName;
  const auto UnqualifiedName = DeclNameSplitted.back();
  while (true) {
    const auto Pos = NsName.find_last_of(':');
    if (Pos == llvm::StringRef::npos)
      return DeclName;
    const auto Prefix = NsName.substr(0, Pos - 1);
    if (DeclName.startswith(Prefix))
      return (Prefix + "::" + UnqualifiedName).str();
    NsName = Prefix;
  }
  return DeclName;
}

std::string wrapCodeInNamespace(StringRef NestedNs, std::string Code) {
  if (Code.back() != '\n')
    Code += "\n";
  llvm::SmallVector<StringRef, 4> NsSplitted;
  NestedNs.split(NsSplitted, "::");
  while (!NsSplitted.empty()) {
    // FIXME: consider code style for comments.
    Code = ("namespace " + NsSplitted.back() + " {\n" + Code +
            "} // namespace " + NsSplitted.back() + "\n")
               .str();
    NsSplitted.pop_back();
  }
  return Code;
}

} // anonymous namespace

ChangeNamespaceTool::ChangeNamespaceTool(
    llvm::StringRef OldNs, llvm::StringRef NewNs, llvm::StringRef FilePattern,
    std::map<std::string, tooling::Replacements> *FileToReplacements,
    llvm::StringRef FallbackStyle)
    : FallbackStyle(FallbackStyle), FileToReplacements(*FileToReplacements),
      OldNamespace(OldNs.ltrim(':')), NewNamespace(NewNs.ltrim(':')),
      FilePattern(FilePattern) {
  FileToReplacements->clear();
  llvm::SmallVector<llvm::StringRef, 4> OldNsSplitted;
  llvm::SmallVector<llvm::StringRef, 4> NewNsSplitted;
  llvm::StringRef(OldNamespace).split(OldNsSplitted, "::");
  llvm::StringRef(NewNamespace).split(NewNsSplitted, "::");
  // Calculates `DiffOldNamespace` and `DiffNewNamespace`.
  while (!OldNsSplitted.empty() && !NewNsSplitted.empty() &&
         OldNsSplitted.front() == NewNsSplitted.front()) {
    OldNsSplitted.erase(OldNsSplitted.begin());
    NewNsSplitted.erase(NewNsSplitted.begin());
  }
  DiffOldNamespace = joinNamespaces(OldNsSplitted);
  DiffNewNamespace = joinNamespaces(NewNsSplitted);
}

// FIXME: handle the following symbols:
//   - Function references.
//   - Variable references.
void ChangeNamespaceTool::registerMatchers(ast_matchers::MatchFinder *Finder) {
  // Match old namespace blocks.
  std::string FullOldNs = "::" + OldNamespace;
  Finder->addMatcher(
      namespaceDecl(hasName(FullOldNs), isExpansionInFileMatching(FilePattern))
          .bind("old_ns"),
      this);

  auto IsInMovedNs =
      allOf(hasAncestor(namespaceDecl(hasName(FullOldNs)).bind("ns_decl")),
            isExpansionInFileMatching(FilePattern));

  // Match forward-declarations in the old namespace.
  Finder->addMatcher(
      cxxRecordDecl(unless(anyOf(isImplicit(), isDefinition())), IsInMovedNs)
          .bind("fwd_decl"),
      this);

  // Match references to types that are not defined in the old namespace.
  // Forward-declarations in the old namespace are also matched since they will
  // be moved back to the old namespace.
  auto DeclMatcher = namedDecl(
      hasAncestor(namespaceDecl()),
      unless(anyOf(
          hasAncestor(namespaceDecl(isAnonymous())),
          hasAncestor(cxxRecordDecl()),
          allOf(IsInMovedNs, unless(cxxRecordDecl(unless(isDefinition())))))));
  // Match TypeLocs on the declaration. Carefully match only the outermost
  // TypeLoc that's directly linked to the old class and don't handle nested
  // name specifier locs.
  Finder->addMatcher(
      typeLoc(IsInMovedNs,
              loc(qualType(hasDeclaration(DeclMatcher.bind("from_decl")))),
              unless(anyOf(hasParent(typeLoc(
                               loc(qualType(hasDeclaration(DeclMatcher))))),
                           hasParent(nestedNameSpecifierLoc()))),
              hasAncestor(decl().bind("dc")))
          .bind("type"),
      this);
  // Types in `UsingShadowDecl` is not matched by `typeLoc` above, so we need to
  // special case it.
  Finder->addMatcher(
      usingDecl(hasAnyUsingShadowDecl(IsInMovedNs)).bind("using_decl"), this);
  // Handle types in nested name specifier.
  Finder->addMatcher(nestedNameSpecifierLoc(
                         hasAncestor(decl(IsInMovedNs).bind("dc")),
                         loc(nestedNameSpecifier(specifiesType(
                             hasDeclaration(DeclMatcher.bind("from_decl"))))))
                         .bind("nested_specifier_loc"),
                     this);
}

void ChangeNamespaceTool::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *NsDecl = Result.Nodes.getNodeAs<NamespaceDecl>("old_ns")) {
    moveOldNamespace(Result, NsDecl);
  } else if (const auto *FwdDecl =
                 Result.Nodes.getNodeAs<CXXRecordDecl>("fwd_decl")) {
    moveClassForwardDeclaration(Result, FwdDecl);
  } else if (const auto *UsingDeclaration =
                 Result.Nodes.getNodeAs<UsingDecl>("using_decl")) {
    fixUsingShadowDecl(Result, UsingDeclaration);
  } else if (const auto *Specifier =
                 Result.Nodes.getNodeAs<NestedNameSpecifierLoc>(
                     "nested_specifier_loc")) {
    SourceLocation Start = Specifier->getBeginLoc();
    SourceLocation End = EndLocationForType(Specifier->getTypeLoc());
    fixTypeLoc(Result, Start, End, Specifier->getTypeLoc());
  } else {
    const auto *TLoc = Result.Nodes.getNodeAs<TypeLoc>("type");
    assert(TLoc != nullptr && "Expecting callback for TypeLoc");
    fixTypeLoc(Result, startLocationForType(*TLoc), EndLocationForType(*TLoc),
               *TLoc);
  }
}

// Stores information about a moved namespace in `MoveNamespaces` and leaves
// the actual movement to `onEndOfTranslationUnit()`.
void ChangeNamespaceTool::moveOldNamespace(
    const ast_matchers::MatchFinder::MatchResult &Result,
    const NamespaceDecl *NsDecl) {
  // If the namespace is empty, do nothing.
  if (Decl::castToDeclContext(NsDecl)->decls_empty())
    return;

  // Get the range of the code in the old namespace.
  SourceLocation Start = NsDecl->decls_begin()->getLocStart();
  SourceLocation End = NsDecl->getRBraceLoc().getLocWithOffset(-1);
  // Create a replacement that deletes the code in the old namespace merely for
  // retrieving offset and length from it.
  const auto R = createReplacement(Start, End, "", *Result.SourceManager);
  MoveNamespace MoveNs;
  MoveNs.Offset = R.getOffset();
  MoveNs.Length = R.getLength();

  // Insert the new namespace after `DiffOldNamespace`. For example, if
  // `OldNamespace` is "a::b::c" and `NewNamespace` is `a::x::y`, then
  // "x::y" will be inserted inside the existing namespace "a" and after "a::b".
  // `OuterNs` is the first namespace in `DiffOldNamespace`, e.g. "namespace b"
  // in the above example.
  // FIXME: consider the case where DiffOldNamespace is empty.
  const NamespaceDecl *OuterNs = getOuterNamespace(NsDecl, DiffOldNamespace);
  SourceLocation LocAfterNs =
      getStartOfNextLine(OuterNs->getRBraceLoc(), *Result.SourceManager,
                         Result.Context->getLangOpts());
  assert(LocAfterNs.isValid() &&
         "Failed to get location after DiffOldNamespace");
  MoveNs.InsertionOffset = Result.SourceManager->getFileOffset(
      Result.SourceManager->getSpellingLoc(LocAfterNs));

  MoveNs.FID = Result.SourceManager->getFileID(Start);
  MoveNs.SourceMgr = Result.SourceManager;
  MoveNamespaces[R.getFilePath()].push_back(MoveNs);
}

// Removes a class forward declaration from the code in the moved namespace and
// creates an `InsertForwardDeclaration` to insert the forward declaration back
// into the old namespace after moving code from the old namespace to the new
// namespace.
// For example, changing "a" to "x":
// Old code:
//   namespace a {
//   class FWD;
//   class A { FWD *fwd; }
//   }  // a
// New code:
//   namespace a {
//   class FWD;
//   }  // a
//   namespace x {
//   class A { a::FWD *fwd; }
//   }  // x
void ChangeNamespaceTool::moveClassForwardDeclaration(
    const ast_matchers::MatchFinder::MatchResult &Result,
    const CXXRecordDecl *FwdDecl) {
  SourceLocation Start = FwdDecl->getLocStart();
  SourceLocation End = FwdDecl->getLocEnd();
  SourceLocation AfterSemi = Lexer::findLocationAfterToken(
      End, tok::semi, *Result.SourceManager, Result.Context->getLangOpts(),
      /*SkipTrailingWhitespaceAndNewLine=*/true);
  if (AfterSemi.isValid())
    End = AfterSemi.getLocWithOffset(-1);
  // Delete the forward declaration from the code to be moved.
  const auto Deletion =
      createReplacement(Start, End, "", *Result.SourceManager);
  addOrMergeReplacement(Deletion, &FileToReplacements[Deletion.getFilePath()]);
  llvm::StringRef Code = Lexer::getSourceText(
      CharSourceRange::getTokenRange(
          Result.SourceManager->getSpellingLoc(Start),
          Result.SourceManager->getSpellingLoc(End)),
      *Result.SourceManager, Result.Context->getLangOpts());
  // Insert the forward declaration back into the old namespace after moving the
  // code from old namespace to new namespace.
  // Insertion information is stored in `InsertFwdDecls` and actual
  // insertion will be performed in `onEndOfTranslationUnit`.
  // Get the (old) namespace that contains the forward declaration.
  const auto *NsDecl = Result.Nodes.getNodeAs<NamespaceDecl>("ns_decl");
  // The namespace contains the forward declaration, so it must not be empty.
  assert(!NsDecl->decls_empty());
  const auto Insertion = createInsertion(NsDecl->decls_begin()->getLocStart(),
                                         Code, *Result.SourceManager);
  InsertForwardDeclaration InsertFwd;
  InsertFwd.InsertionOffset = Insertion.getOffset();
  InsertFwd.ForwardDeclText = Insertion.getReplacementText().str();
  InsertFwdDecls[Insertion.getFilePath()].push_back(InsertFwd);
}

// Replaces a qualified symbol that refers to a declaration `DeclName` with the
// shortest qualified name possible when the reference is in `NewNamespace`.
void ChangeNamespaceTool::replaceQualifiedSymbolInDeclContext(
    const ast_matchers::MatchFinder::MatchResult &Result, const Decl *DeclCtx,
    SourceLocation Start, SourceLocation End, llvm::StringRef DeclName) {
  const auto *NsDeclContext =
      DeclCtx->getDeclContext()->getEnclosingNamespaceContext();
  const auto *NsDecl = llvm::dyn_cast<NamespaceDecl>(NsDeclContext);
  // Calculate the name of the `NsDecl` after it is moved to new namespace.
  std::string OldNs = NsDecl->getQualifiedNameAsString();
  llvm::StringRef Postfix = OldNs;
  bool Consumed = Postfix.consume_front(OldNamespace);
  assert(Consumed && "Expect OldNS to start with OldNamespace.");
  (void)Consumed;
  const std::string NewNs = (NewNamespace + Postfix).str();

  llvm::StringRef NestedName = Lexer::getSourceText(
      CharSourceRange::getTokenRange(
          Result.SourceManager->getSpellingLoc(Start),
          Result.SourceManager->getSpellingLoc(End)),
      *Result.SourceManager, Result.Context->getLangOpts());
  // If the symbol is already fully qualified, no change needs to be make.
  if (NestedName.startswith("::"))
    return;
  std::string ReplaceName =
      getShortestQualifiedNameInNamespace(DeclName, NewNs);
  // If the new nested name in the new namespace is the same as it was in the
  // old namespace, we don't create replacement.
  if (NestedName == ReplaceName)
    return;
  auto R = createReplacement(Start, End, ReplaceName, *Result.SourceManager);
  addOrMergeReplacement(R, &FileToReplacements[R.getFilePath()]);
}

// Replace the [Start, End] of `Type` with the shortest qualified name when the
// `Type` is in `NewNamespace`.
void ChangeNamespaceTool::fixTypeLoc(
    const ast_matchers::MatchFinder::MatchResult &Result, SourceLocation Start,
    SourceLocation End, TypeLoc Type) {
  // FIXME: do not rename template parameter.
  if (Start.isInvalid() || End.isInvalid())
    return;
  // The declaration which this TypeLoc refers to.
  const auto *FromDecl = Result.Nodes.getNodeAs<NamedDecl>("from_decl");
  // `hasDeclaration` gives underlying declaration, but if the type is
  // a typedef type, we need to use the typedef type instead.
  if (auto *Typedef = Type.getType()->getAs<TypedefType>())
    FromDecl = Typedef->getDecl();

  const Decl *DeclCtx = Result.Nodes.getNodeAs<Decl>("dc");
  assert(DeclCtx && "Empty decl context.");
  replaceQualifiedSymbolInDeclContext(Result, DeclCtx, Start, End,
                                      FromDecl->getQualifiedNameAsString());
}

void ChangeNamespaceTool::fixUsingShadowDecl(
    const ast_matchers::MatchFinder::MatchResult &Result,
    const UsingDecl *UsingDeclaration) {
  SourceLocation Start = UsingDeclaration->getLocStart();
  SourceLocation End = UsingDeclaration->getLocEnd();
  if (Start.isInvalid() || End.isInvalid()) return;

  assert(UsingDeclaration->shadow_size() > 0);
  // FIXME: it might not be always accurate to use the first using-decl.
  const NamedDecl *TargetDecl =
      UsingDeclaration->shadow_begin()->getTargetDecl();
  std::string TargetDeclName = TargetDecl->getQualifiedNameAsString();
  // FIXME: check if target_decl_name is in moved ns, which doesn't make much
  // sense. If this happens, we need to use name with the new namespace.
  // Use fully qualified name in UsingDecl for now.
  auto R = createReplacement(Start, End, "using ::" + TargetDeclName,
                             *Result.SourceManager);
  addOrMergeReplacement(R, &FileToReplacements[R.getFilePath()]);
}

void ChangeNamespaceTool::onEndOfTranslationUnit() {
  // Move namespace blocks and insert forward declaration to old namespace.
  for (const auto &FileAndNsMoves : MoveNamespaces) {
    auto &NsMoves = FileAndNsMoves.second;
    if (NsMoves.empty())
      continue;
    const std::string &FilePath = FileAndNsMoves.first;
    auto &Replaces = FileToReplacements[FilePath];
    auto &SM = *NsMoves.begin()->SourceMgr;
    llvm::StringRef Code = SM.getBufferData(NsMoves.begin()->FID);
    auto ChangedCode = tooling::applyAllReplacements(Code, Replaces);
    if (!ChangedCode) {
      llvm::errs() << llvm::toString(ChangedCode.takeError()) << "\n";
      continue;
    }
    // Replacements on the changed code for moving namespaces and inserting
    // forward declarations to old namespaces.
    tooling::Replacements NewReplacements;
    // Cut the changed code from the old namespace and paste the code in the new
    // namespace.
    for (const auto &NsMove : NsMoves) {
      // Calculate the range of the old namespace block in the changed
      // code.
      const unsigned NewOffset = Replaces.getShiftedCodePosition(NsMove.Offset);
      const unsigned NewLength =
          Replaces.getShiftedCodePosition(NsMove.Offset + NsMove.Length) -
          NewOffset;
      tooling::Replacement Deletion(FilePath, NewOffset, NewLength, "");
      std::string MovedCode = ChangedCode->substr(NewOffset, NewLength);
      std::string MovedCodeWrappedInNewNs =
          wrapCodeInNamespace(DiffNewNamespace, MovedCode);
      // Calculate the new offset at which the code will be inserted in the
      // changed code.
      unsigned NewInsertionOffset =
          Replaces.getShiftedCodePosition(NsMove.InsertionOffset);
      tooling::Replacement Insertion(FilePath, NewInsertionOffset, 0,
                                     MovedCodeWrappedInNewNs);
      addOrMergeReplacement(Deletion, &NewReplacements);
      addOrMergeReplacement(Insertion, &NewReplacements);
    }
    // After moving namespaces, insert forward declarations back to old
    // namespaces.
    const auto &FwdDeclInsertions = InsertFwdDecls[FilePath];
    for (const auto &FwdDeclInsertion : FwdDeclInsertions) {
      unsigned NewInsertionOffset =
          Replaces.getShiftedCodePosition(FwdDeclInsertion.InsertionOffset);
      tooling::Replacement Insertion(FilePath, NewInsertionOffset, 0,
                                     FwdDeclInsertion.ForwardDeclText);
      addOrMergeReplacement(Insertion, &NewReplacements);
    }
    // Add replacements referring to the changed code to existing replacements,
    // which refers to the original code.
    Replaces = Replaces.merge(NewReplacements);
    format::FormatStyle Style =
        format::getStyle("file", FilePath, FallbackStyle);
    // Clean up old namespaces if there is nothing in it after moving.
    auto CleanReplacements =
        format::cleanupAroundReplacements(Code, Replaces, Style);
    if (!CleanReplacements) {
      llvm::errs() << llvm::toString(CleanReplacements.takeError()) << "\n";
      continue;
    }
    FileToReplacements[FilePath] = *CleanReplacements;
  }
}

} // namespace change_namespace
} // namespace clang
