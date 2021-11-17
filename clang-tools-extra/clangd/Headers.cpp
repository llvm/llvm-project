//===--- Headers.cpp - Include headers ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "Compiler.h"
#include "Preamble.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {

const char IWYUPragmaKeep[] = "// IWYU pragma: keep";

class IncludeStructure::RecordHeaders : public PPCallbacks,
                                        public CommentHandler {
public:
  RecordHeaders(const CompilerInstance &CI, IncludeStructure *Out)
      : SM(CI.getSourceManager()),
        HeaderInfo(CI.getPreprocessor().getHeaderSearchInfo()), Out(Out) {}

  // Record existing #includes - both written and resolved paths. Only #includes
  // in the main file are collected.
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          llvm::StringRef FileName, bool IsAngled,
                          CharSourceRange /*FilenameRange*/,
                          const FileEntry *File, llvm::StringRef /*SearchPath*/,
                          llvm::StringRef /*RelativePath*/,
                          const clang::Module * /*Imported*/,
                          SrcMgr::CharacteristicKind FileKind) override {
    auto MainFID = SM.getMainFileID();
    // If an include is part of the preamble patch, translate #line directives.
    if (InBuiltinFile)
      HashLoc = translatePreamblePatchLocation(HashLoc, SM);

    // Record main-file inclusions (including those mapped from the preamble
    // patch).
    if (isInsideMainFile(HashLoc, SM)) {
      Out->MainFileIncludes.emplace_back();
      auto &Inc = Out->MainFileIncludes.back();
      Inc.Written =
          (IsAngled ? "<" + FileName + ">" : "\"" + FileName + "\"").str();
      Inc.Resolved = std::string(File ? File->tryGetRealPathName() : "");
      Inc.HashOffset = SM.getFileOffset(HashLoc);
      Inc.HashLine =
          SM.getLineNumber(SM.getFileID(HashLoc), Inc.HashOffset) - 1;
      Inc.FileKind = FileKind;
      Inc.Directive = IncludeTok.getIdentifierInfo()->getPPKeywordID();
      if (LastPragmaKeepInMainFileLine == Inc.HashLine)
        Inc.BehindPragmaKeep = true;
      if (File) {
        IncludeStructure::HeaderID HID = Out->getOrCreateID(File);
        Inc.HeaderID = static_cast<unsigned>(HID);
        if (IsAngled)
          if (auto StdlibHeader = stdlib::Header::named(Inc.Written)) {
            auto &IDs = Out->StdlibHeaders[*StdlibHeader];
            // Few physical files for one stdlib header name, linear scan is ok.
            if (!llvm::is_contained(IDs, HID))
              IDs.push_back(HID);
          }
      }
    }

    // Record include graph (not just for main-file includes)
    if (File) {
      auto *IncludingFileEntry = SM.getFileEntryForID(SM.getFileID(HashLoc));
      if (!IncludingFileEntry) {
        assert(SM.getBufferName(HashLoc).startswith("<") &&
               "Expected #include location to be a file or <built-in>");
        // Treat as if included from the main file.
        IncludingFileEntry = SM.getFileEntryForID(MainFID);
      }
      auto IncludingID = Out->getOrCreateID(IncludingFileEntry),
           IncludedID = Out->getOrCreateID(File);
      Out->IncludeChildren[IncludingID].push_back(IncludedID);
    }
  }

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    switch (Reason) {
    case PPCallbacks::EnterFile:
      ++Level;
      if (BuiltinFile.isInvalid() && SM.isWrittenInBuiltinFile(Loc)) {
        BuiltinFile = SM.getFileID(Loc);
        InBuiltinFile = true;
      }
      break;
    case PPCallbacks::ExitFile: {
      --Level;
      if (PrevFID == BuiltinFile)
        InBuiltinFile = false;
      // At file exit time HeaderSearchInfo is valid and can be used to
      // determine whether the file was a self-contained header or not.
      if (const FileEntry *FE = SM.getFileEntryForID(PrevFID))
        if (!isSelfContainedHeader(FE, PrevFID, SM, HeaderInfo))
          Out->NonSelfContained.insert(
              *Out->getID(SM.getFileEntryForID(PrevFID)));
      break;
    }
    case PPCallbacks::RenameFile:
    case PPCallbacks::SystemHeaderPragma:
      break;
    }
  }

  // Given:
  //
  // #include "foo.h"
  // #include "bar.h" // IWYU pragma: keep
  //
  // The order in which the callbacks will be triggered:
  //
  // 1. InclusionDirective("foo.h")
  // 2. HandleComment("// IWYU pragma: keep")
  // 3. InclusionDirective("bar.h")
  //
  // HandleComment will store the last location of "IWYU pragma: keep" comment
  // in the main file, so that when InclusionDirective is called, it will know
  // that the next inclusion is behind the IWYU pragma.
  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    if (!inMainFile() || Range.getBegin().isMacroID())
      return false;
    bool Err = false;
    llvm::StringRef Text = SM.getCharacterData(Range.getBegin(), &Err);
    if (Err || !Text.consume_front(IWYUPragmaKeep))
      return false;
    unsigned Offset = SM.getFileOffset(Range.getBegin());
    LastPragmaKeepInMainFileLine =
        SM.getLineNumber(SM.getFileID(Range.getBegin()), Offset) - 1;
    return false;
  }

private:
  // Keeps track of include depth for the current file. It's 1 for main file.
  int Level = 0;
  bool inMainFile() const { return Level == 1; }

  const SourceManager &SM;
  HeaderSearch &HeaderInfo;
  // Set after entering the <built-in> file.
  FileID BuiltinFile;
  // Indicates whether <built-in> file is part of include stack.
  bool InBuiltinFile = false;

  IncludeStructure *Out;

  // The last line "IWYU pragma: keep" was seen in the main file, 0-indexed.
  int LastPragmaKeepInMainFileLine = -1;
};

bool isLiteralInclude(llvm::StringRef Include) {
  return Include.startswith("<") || Include.startswith("\"");
}

bool HeaderFile::valid() const {
  return (Verbatim && isLiteralInclude(File)) ||
         (!Verbatim && llvm::sys::path::is_absolute(File));
}

llvm::Expected<HeaderFile> toHeaderFile(llvm::StringRef Header,
                                        llvm::StringRef HintPath) {
  if (isLiteralInclude(Header))
    return HeaderFile{Header.str(), /*Verbatim=*/true};
  auto U = URI::parse(Header);
  if (!U)
    return U.takeError();

  auto IncludePath = URI::includeSpelling(*U);
  if (!IncludePath)
    return IncludePath.takeError();
  if (!IncludePath->empty())
    return HeaderFile{std::move(*IncludePath), /*Verbatim=*/true};

  auto Resolved = URI::resolve(*U, HintPath);
  if (!Resolved)
    return Resolved.takeError();
  return HeaderFile{std::move(*Resolved), /*Verbatim=*/false};
}

llvm::SmallVector<llvm::StringRef, 1> getRankedIncludes(const Symbol &Sym) {
  auto Includes = Sym.IncludeHeaders;
  // Sort in descending order by reference count and header length.
  llvm::sort(Includes, [](const Symbol::IncludeHeaderWithReferences &LHS,
                          const Symbol::IncludeHeaderWithReferences &RHS) {
    if (LHS.References == RHS.References)
      return LHS.IncludeHeader.size() < RHS.IncludeHeader.size();
    return LHS.References > RHS.References;
  });
  llvm::SmallVector<llvm::StringRef, 1> Headers;
  for (const auto &Include : Includes)
    Headers.push_back(Include.IncludeHeader);
  return Headers;
}

void IncludeStructure::collect(const CompilerInstance &CI) {
  auto &SM = CI.getSourceManager();
  MainFileEntry = SM.getFileEntryForID(SM.getMainFileID());
  auto Collector = std::make_unique<RecordHeaders>(CI, this);
  CI.getPreprocessor().addCommentHandler(Collector.get());
  CI.getPreprocessor().addPPCallbacks(std::move(Collector));
}

llvm::Optional<IncludeStructure::HeaderID>
IncludeStructure::getID(const FileEntry *Entry) const {
  // HeaderID of the main file is always 0;
  if (Entry == MainFileEntry) {
    return static_cast<IncludeStructure::HeaderID>(0u);
  }
  auto It = UIDToIndex.find(Entry->getUniqueID());
  if (It == UIDToIndex.end())
    return llvm::None;
  return It->second;
}

IncludeStructure::HeaderID
IncludeStructure::getOrCreateID(const FileEntry *Entry) {
  // Main file's FileEntry was not known at IncludeStructure creation time.
  if (Entry == MainFileEntry) {
    if (RealPathNames.front().empty())
      RealPathNames.front() = MainFileEntry->tryGetRealPathName().str();
    return MainFileID;
  }
  auto R = UIDToIndex.try_emplace(
      Entry->getUniqueID(),
      static_cast<IncludeStructure::HeaderID>(RealPathNames.size()));
  if (R.second)
    RealPathNames.emplace_back();
  IncludeStructure::HeaderID Result = R.first->getSecond();
  std::string &RealPathName = RealPathNames[static_cast<unsigned>(Result)];
  if (RealPathName.empty())
    RealPathName = Entry->tryGetRealPathName().str();
  return Result;
}

llvm::DenseMap<IncludeStructure::HeaderID, unsigned>
IncludeStructure::includeDepth(HeaderID Root) const {
  // Include depth 0 is the main file only.
  llvm::DenseMap<HeaderID, unsigned> Result;
  assert(static_cast<unsigned>(Root) < RealPathNames.size());
  Result[Root] = 0;
  std::vector<IncludeStructure::HeaderID> CurrentLevel;
  CurrentLevel.push_back(Root);
  llvm::DenseSet<IncludeStructure::HeaderID> Seen;
  Seen.insert(Root);

  // Each round of BFS traversal finds the next depth level.
  std::vector<IncludeStructure::HeaderID> PreviousLevel;
  for (unsigned Level = 1; !CurrentLevel.empty(); ++Level) {
    PreviousLevel.clear();
    PreviousLevel.swap(CurrentLevel);
    for (const auto &Parent : PreviousLevel) {
      for (const auto &Child : IncludeChildren.lookup(Parent)) {
        if (Seen.insert(Child).second) {
          CurrentLevel.push_back(Child);
          Result[Child] = Level;
        }
      }
    }
  }
  return Result;
}

void IncludeInserter::addExisting(const Inclusion &Inc) {
  IncludedHeaders.insert(Inc.Written);
  if (!Inc.Resolved.empty())
    IncludedHeaders.insert(Inc.Resolved);
}

/// FIXME(ioeric): we might not want to insert an absolute include path if the
/// path is not shortened.
bool IncludeInserter::shouldInsertInclude(
    PathRef DeclaringHeader, const HeaderFile &InsertedHeader) const {
  assert(InsertedHeader.valid());
  if (!HeaderSearchInfo && !InsertedHeader.Verbatim)
    return false;
  if (FileName == DeclaringHeader || FileName == InsertedHeader.File)
    return false;
  auto Included = [&](llvm::StringRef Header) {
    return IncludedHeaders.find(Header) != IncludedHeaders.end();
  };
  return !Included(DeclaringHeader) && !Included(InsertedHeader.File);
}

llvm::Optional<std::string>
IncludeInserter::calculateIncludePath(const HeaderFile &InsertedHeader,
                                      llvm::StringRef IncludingFile) const {
  assert(InsertedHeader.valid());
  if (InsertedHeader.Verbatim)
    return InsertedHeader.File;
  bool IsSystem = false;
  std::string Suggested;
  if (HeaderSearchInfo) {
    Suggested = HeaderSearchInfo->suggestPathToFileForDiagnostics(
        InsertedHeader.File, BuildDir, IncludingFile, &IsSystem);
  } else {
    // Calculate include relative to including file only.
    StringRef IncludingDir = llvm::sys::path::parent_path(IncludingFile);
    SmallString<256> RelFile(InsertedHeader.File);
    // Replacing with "" leaves "/RelFile" if IncludingDir doesn't end in "/".
    llvm::sys::path::replace_path_prefix(RelFile, IncludingDir, "./");
    Suggested = llvm::sys::path::convert_to_slash(
        llvm::sys::path::remove_leading_dotslash(RelFile));
  }
  // FIXME: should we allow (some limited number of) "../header.h"?
  if (llvm::sys::path::is_absolute(Suggested))
    return None;
  if (IsSystem)
    Suggested = "<" + Suggested + ">";
  else
    Suggested = "\"" + Suggested + "\"";
  return Suggested;
}

llvm::Optional<TextEdit>
IncludeInserter::insert(llvm::StringRef VerbatimHeader) const {
  llvm::Optional<TextEdit> Edit = None;
  if (auto Insertion = Inserter.insert(VerbatimHeader.trim("\"<>"),
                                       VerbatimHeader.startswith("<")))
    Edit = replacementToEdit(Code, *Insertion);
  return Edit;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Inclusion &Inc) {
  return OS << Inc.Written << " = "
            << (!Inc.Resolved.empty() ? Inc.Resolved : "[unresolved]")
            << " at line" << Inc.HashLine;
}

bool operator==(const Inclusion &LHS, const Inclusion &RHS) {
  return std::tie(LHS.Directive, LHS.FileKind, LHS.HashOffset, LHS.HashLine,
                  LHS.Resolved, LHS.Written) ==
         std::tie(RHS.Directive, RHS.FileKind, RHS.HashOffset, RHS.HashLine,
                  RHS.Resolved, RHS.Written);
}

namespace stdlib {
static llvm::StringRef *HeaderNames;
static std::pair<llvm::StringRef, llvm::StringRef> *SymbolNames;
static unsigned *SymbolHeaderIDs;
static llvm::DenseMap<llvm::StringRef, unsigned> *HeaderIDs;
// Maps symbol name -> Symbol::ID, within a namespace.
using NSSymbolMap = llvm::DenseMap<llvm::StringRef, unsigned>;
static llvm::DenseMap<llvm::StringRef, NSSymbolMap *> *NamespaceSymbols;

static int initialize() {
  unsigned SymCount = 0;
#define SYMBOL(Name, NS, Header) ++SymCount;
#include "CSymbolMap.inc"
#include "StdSymbolMap.inc"
#undef SYMBOL
  SymbolNames = new std::remove_reference_t<decltype(*SymbolNames)>[SymCount];
  SymbolHeaderIDs =
      new std::remove_reference_t<decltype(*SymbolHeaderIDs)>[SymCount];
  NamespaceSymbols = new std::remove_reference_t<decltype(*NamespaceSymbols)>;
  HeaderIDs = new std::remove_reference_t<decltype(*HeaderIDs)>;

  auto AddNS = [&](llvm::StringRef NS) -> NSSymbolMap & {
    auto R = NamespaceSymbols->try_emplace(NS, nullptr);
    if (R.second)
      R.first->second = new NSSymbolMap();
    return *R.first->second;
  };

  auto AddHeader = [&](llvm::StringRef Header) -> unsigned {
    return HeaderIDs->try_emplace(Header, HeaderIDs->size()).first->second;
  };

  auto Add = [&, SymIndex(0)](llvm::StringRef Name, llvm::StringRef NS,
                              llvm::StringRef HeaderName) mutable {
    if (NS == "None")
      NS = "";

    SymbolNames[SymIndex] = {NS, Name};
    SymbolHeaderIDs[SymIndex] = AddHeader(HeaderName);

    NSSymbolMap &NSSymbols = AddNS(NS);
    NSSymbols.try_emplace(Name, SymIndex);

    ++SymIndex;
  };
#define SYMBOL(Name, NS, Header) Add(#Name, #NS, #Header);
#include "CSymbolMap.inc"
#include "StdSymbolMap.inc"
#undef SYMBOL

  HeaderNames = new llvm::StringRef[HeaderIDs->size()];
  for (const auto &E : *HeaderIDs)
    HeaderNames[E.second] = E.first;

  return 0;
}

static void ensureInitialized() {
  static int Dummy = initialize();
  (void)Dummy;
}

llvm::Optional<Header> Header::named(llvm::StringRef Name) {
  ensureInitialized();
  auto It = HeaderIDs->find(Name);
  if (It == HeaderIDs->end())
    return llvm::None;
  return Header(It->second);
}
llvm::StringRef Header::name() const { return HeaderNames[ID]; }
llvm::StringRef Symbol::scope() const { return SymbolNames[ID].first; }
llvm::StringRef Symbol::name() const { return SymbolNames[ID].second; }
llvm::Optional<Symbol> Symbol::named(llvm::StringRef Scope,
                                     llvm::StringRef Name) {
  ensureInitialized();
  if (NSSymbolMap *NSSymbols = NamespaceSymbols->lookup(Scope)) {
    auto It = NSSymbols->find(Name);
    if (It != NSSymbols->end())
      return Symbol(It->second);
  }
  return llvm::None;
}
Header Symbol::header() const { return Header(SymbolHeaderIDs[ID]); }
llvm::SmallVector<Header> Symbol::headers() const {
  return {header()}; // FIXME: multiple in case of ambiguity
}

Recognizer::Recognizer() { ensureInitialized(); }

NSSymbolMap *Recognizer::namespaceSymbols(const NamespaceDecl *D) {
  auto It = NamespaceCache.find(D);
  if (It != NamespaceCache.end())
    return It->second;

  NSSymbolMap *Result = [&]() -> NSSymbolMap * {
    if (!D) // Nullptr means the global namespace
      return NamespaceSymbols->lookup("");
    if (D->isAnonymousNamespace())
      return nullptr;
    if (D->isInlineNamespace()) {
      if (auto *Parent = llvm::dyn_cast_or_null<NamespaceDecl>(D->getParent()))
        return namespaceSymbols(Parent);
      return nullptr;
    }
    return NamespaceSymbols->lookup(printNamespaceScope(*D));
  }();
  NamespaceCache.try_emplace(D, Result);
  return Result;
}

llvm::Optional<Symbol> Recognizer::operator()(const Decl *D) {
  // If D is std::vector::iterator, `vector` is the outer symbol to look up.
  // We keep all the candidate DCs as some may turn out to be anon enums.
  // Do this resolution lazily as we may turn out not to have a std namespace.
  llvm::SmallVector<const DeclContext *> IntermediateDecl;
  const DeclContext *DC = D->getDeclContext();
  while (DC && !DC->isNamespace()) {
    if (NamedDecl::classofKind(DC->getDeclKind()))
      IntermediateDecl.push_back(DC);
    DC = DC->getParent();
  }
  NSSymbolMap *Symbols = namespaceSymbols(cast_or_null<NamespaceDecl>(DC));
  if (!Symbols)
    return llvm::None;

  llvm::StringRef Name = [&]() -> llvm::StringRef {
    for (const auto *SymDC : llvm::reverse(IntermediateDecl)) {
      DeclarationName N = cast<NamedDecl>(SymDC)->getDeclName();
      if (const auto *II = N.getAsIdentifierInfo())
        return II->getName();
      if (!N.isEmpty())
        return ""; // e.g. operator<: give up
    }
    if (const auto *ND = llvm::dyn_cast<NamedDecl>(D))
      if (const auto *II = ND->getIdentifier())
        return II->getName();
    return "";
  }();
  if (Name.empty())
    return llvm::None;

  auto It = Symbols->find(Name);
  if (It == Symbols->end())
    return llvm::None;
  return Symbol(It->second);
}

} // namespace stdlib

} // namespace clangd
} // namespace clang
