//===--- SourceCode.h - Manipulating source code as strings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SourceCode.h"

#include "Context.h"
#include "FuzzyMatch.h"
#include "Logger.h"
#include "Protocol.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include <algorithm>

namespace clang {
namespace clangd {

// Here be dragons. LSP positions use columns measured in *UTF-16 code units*!
// Clangd uses UTF-8 and byte-offsets internally, so conversion is nontrivial.

// Iterates over unicode codepoints in the (UTF-8) string. For each,
// invokes CB(UTF-8 length, UTF-16 length), and breaks if it returns true.
// Returns true if CB returned true, false if we hit the end of string.
template <typename Callback>
static bool iterateCodepoints(llvm::StringRef U8, const Callback &CB) {
  // A codepoint takes two UTF-16 code unit if it's astral (outside BMP).
  // Astral codepoints are encoded as 4 bytes in UTF-8, starting with 11110xxx.
  for (size_t I = 0; I < U8.size();) {
    unsigned char C = static_cast<unsigned char>(U8[I]);
    if (LLVM_LIKELY(!(C & 0x80))) { // ASCII character.
      if (CB(1, 1))
        return true;
      ++I;
      continue;
    }
    // This convenient property of UTF-8 holds for all non-ASCII characters.
    size_t UTF8Length = llvm::countLeadingOnes(C);
    // 0xxx is ASCII, handled above. 10xxx is a trailing byte, invalid here.
    // 11111xxx is not valid UTF-8 at all. Assert because it's probably our bug.
    assert((UTF8Length >= 2 && UTF8Length <= 4) &&
           "Invalid UTF-8, or transcoding bug?");
    I += UTF8Length; // Skip over all trailing bytes.
    // A codepoint takes two UTF-16 code unit if it's astral (outside BMP).
    // Astral codepoints are encoded as 4 bytes in UTF-8 (11110xxx ...)
    if (CB(UTF8Length, UTF8Length == 4 ? 2 : 1))
      return true;
  }
  return false;
}

// Returns the byte offset into the string that is an offset of \p Units in
// the specified encoding.
// Conceptually, this converts to the encoding, truncates to CodeUnits,
// converts back to UTF-8, and returns the length in bytes.
static size_t measureUnits(llvm::StringRef U8, int Units, OffsetEncoding Enc,
                           bool &Valid) {
  Valid = Units >= 0;
  if (Units <= 0)
    return 0;
  size_t Result = 0;
  switch (Enc) {
  case OffsetEncoding::UTF8:
    Result = Units;
    break;
  case OffsetEncoding::UTF16:
    Valid = iterateCodepoints(U8, [&](int U8Len, int U16Len) {
      Result += U8Len;
      Units -= U16Len;
      return Units <= 0;
    });
    if (Units < 0) // Offset in the middle of a surrogate pair.
      Valid = false;
    break;
  case OffsetEncoding::UTF32:
    Valid = iterateCodepoints(U8, [&](int U8Len, int U16Len) {
      Result += U8Len;
      Units--;
      return Units <= 0;
    });
    break;
  case OffsetEncoding::UnsupportedEncoding:
    llvm_unreachable("unsupported encoding");
  }
  // Don't return an out-of-range index if we overran.
  if (Result > U8.size()) {
    Valid = false;
    return U8.size();
  }
  return Result;
}

Key<OffsetEncoding> kCurrentOffsetEncoding;
static OffsetEncoding lspEncoding() {
  auto *Enc = Context::current().get(kCurrentOffsetEncoding);
  return Enc ? *Enc : OffsetEncoding::UTF16;
}

// Like most strings in clangd, the input is UTF-8 encoded.
size_t lspLength(llvm::StringRef Code) {
  size_t Count = 0;
  switch (lspEncoding()) {
  case OffsetEncoding::UTF8:
    Count = Code.size();
    break;
  case OffsetEncoding::UTF16:
    iterateCodepoints(Code, [&](int U8Len, int U16Len) {
      Count += U16Len;
      return false;
    });
    break;
  case OffsetEncoding::UTF32:
    iterateCodepoints(Code, [&](int U8Len, int U16Len) {
      ++Count;
      return false;
    });
    break;
  case OffsetEncoding::UnsupportedEncoding:
    llvm_unreachable("unsupported encoding");
  }
  return Count;
}

llvm::Expected<size_t> positionToOffset(llvm::StringRef Code, Position P,
                                        bool AllowColumnsBeyondLineLength) {
  if (P.line < 0)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Line value can't be negative ({0})", P.line),
        llvm::errc::invalid_argument);
  if (P.character < 0)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Character value can't be negative ({0})", P.character),
        llvm::errc::invalid_argument);
  size_t StartOfLine = 0;
  for (int I = 0; I != P.line; ++I) {
    size_t NextNL = Code.find('\n', StartOfLine);
    if (NextNL == llvm::StringRef::npos)
      return llvm::make_error<llvm::StringError>(
          llvm::formatv("Line value is out of range ({0})", P.line),
          llvm::errc::invalid_argument);
    StartOfLine = NextNL + 1;
  }
  StringRef Line =
      Code.substr(StartOfLine).take_until([](char C) { return C == '\n'; });

  // P.character may be in UTF-16, transcode if necessary.
  bool Valid;
  size_t ByteInLine = measureUnits(Line, P.character, lspEncoding(), Valid);
  if (!Valid && !AllowColumnsBeyondLineLength)
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("{0} offset {1} is invalid for line {2}", lspEncoding(),
                      P.character, P.line),
        llvm::errc::invalid_argument);
  return StartOfLine + ByteInLine;
}

Position offsetToPosition(llvm::StringRef Code, size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  llvm::StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == llvm::StringRef::npos) ? 0 : (PrevNL + 1);
  Position Pos;
  Pos.line = Lines;
  Pos.character = lspLength(Before.substr(StartOfLine));
  return Pos;
}

Position sourceLocToPosition(const SourceManager &SM, SourceLocation Loc) {
  // We use the SourceManager's line tables, but its column number is in bytes.
  FileID FID;
  unsigned Offset;
  std::tie(FID, Offset) = SM.getDecomposedSpellingLoc(Loc);
  Position P;
  P.line = static_cast<int>(SM.getLineNumber(FID, Offset)) - 1;
  bool Invalid = false;
  llvm::StringRef Code = SM.getBufferData(FID, &Invalid);
  if (!Invalid) {
    auto ColumnInBytes = SM.getColumnNumber(FID, Offset) - 1;
    auto LineSoFar = Code.substr(Offset - ColumnInBytes, ColumnInBytes);
    P.character = lspLength(LineSoFar);
  }
  return P;
}

bool isValidFileRange(const SourceManager &Mgr, SourceRange R) {
  if (!R.getBegin().isValid() || !R.getEnd().isValid())
    return false;

  FileID BeginFID;
  size_t BeginOffset = 0;
  std::tie(BeginFID, BeginOffset) = Mgr.getDecomposedLoc(R.getBegin());

  FileID EndFID;
  size_t EndOffset = 0;
  std::tie(EndFID, EndOffset) = Mgr.getDecomposedLoc(R.getEnd());

  return BeginFID.isValid() && BeginFID == EndFID && BeginOffset <= EndOffset;
}

bool halfOpenRangeContains(const SourceManager &Mgr, SourceRange R,
                           SourceLocation L) {
  assert(isValidFileRange(Mgr, R));

  FileID BeginFID;
  size_t BeginOffset = 0;
  std::tie(BeginFID, BeginOffset) = Mgr.getDecomposedLoc(R.getBegin());
  size_t EndOffset = Mgr.getFileOffset(R.getEnd());

  FileID LFid;
  size_t LOffset;
  std::tie(LFid, LOffset) = Mgr.getDecomposedLoc(L);
  return BeginFID == LFid && BeginOffset <= LOffset && LOffset < EndOffset;
}

bool halfOpenRangeTouches(const SourceManager &Mgr, SourceRange R,
                          SourceLocation L) {
  return L == R.getEnd() || halfOpenRangeContains(Mgr, R, L);
}

llvm::Optional<SourceRange> toHalfOpenFileRange(const SourceManager &Mgr,
                                                const LangOptions &LangOpts,
                                                SourceRange R) {
  auto Begin = Mgr.getFileLoc(R.getBegin());
  if (Begin.isInvalid())
    return llvm::None;
  auto End = Mgr.getFileLoc(R.getEnd());
  if (End.isInvalid())
    return llvm::None;
  End = Lexer::getLocForEndOfToken(End, 0, Mgr, LangOpts);

  SourceRange Result(Begin, End);
  if (!isValidFileRange(Mgr, Result))
    return llvm::None;
  return Result;
}

llvm::StringRef toSourceCode(const SourceManager &SM, SourceRange R) {
  assert(isValidFileRange(SM, R));
  bool Invalid = false;
  auto *Buf = SM.getBuffer(SM.getFileID(R.getBegin()), &Invalid);
  assert(!Invalid);

  size_t BeginOffset = SM.getFileOffset(R.getBegin());
  size_t EndOffset = SM.getFileOffset(R.getEnd());
  return Buf->getBuffer().substr(BeginOffset, EndOffset - BeginOffset);
}

llvm::Expected<SourceLocation> sourceLocationInMainFile(const SourceManager &SM,
                                                        Position P) {
  llvm::StringRef Code = SM.getBuffer(SM.getMainFileID())->getBuffer();
  auto Offset =
      positionToOffset(Code, P, /*AllowColumnBeyondLineLength=*/false);
  if (!Offset)
    return Offset.takeError();
  return SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(*Offset);
}

Range halfOpenToRange(const SourceManager &SM, CharSourceRange R) {
  // Clang is 1-based, LSP uses 0-based indexes.
  Position Begin = sourceLocToPosition(SM, R.getBegin());
  Position End = sourceLocToPosition(SM, R.getEnd());

  return {Begin, End};
}

std::pair<size_t, size_t> offsetToClangLineColumn(llvm::StringRef Code,
                                                  size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  llvm::StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == llvm::StringRef::npos) ? 0 : (PrevNL + 1);
  return {Lines + 1, Offset - StartOfLine + 1};
}

std::pair<StringRef, StringRef> splitQualifiedName(StringRef QName) {
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos)
    return {llvm::StringRef(), QName};
  return {QName.substr(0, Pos + 2), QName.substr(Pos + 2)};
}

TextEdit replacementToEdit(llvm::StringRef Code,
                           const tooling::Replacement &R) {
  Range ReplacementRange = {
      offsetToPosition(Code, R.getOffset()),
      offsetToPosition(Code, R.getOffset() + R.getLength())};
  return {ReplacementRange, R.getReplacementText()};
}

std::vector<TextEdit> replacementsToEdits(llvm::StringRef Code,
                                          const tooling::Replacements &Repls) {
  std::vector<TextEdit> Edits;
  for (const auto &R : Repls)
    Edits.push_back(replacementToEdit(Code, R));
  return Edits;
}

llvm::Optional<std::string> getCanonicalPath(const FileEntry *F,
                                             const SourceManager &SourceMgr) {
  if (!F)
    return None;

  llvm::SmallString<128> FilePath = F->getName();
  if (!llvm::sys::path::is_absolute(FilePath)) {
    if (auto EC =
            SourceMgr.getFileManager().getVirtualFileSystem().makeAbsolute(
                FilePath)) {
      elog("Could not turn relative path '{0}' to absolute: {1}", FilePath,
           EC.message());
      return None;
    }
  }

  // Handle the symbolic link path case where the current working directory
  // (getCurrentWorkingDirectory) is a symlink./ We always want to the real
  // file path (instead of the symlink path) for the  C++ symbols.
  //
  // Consider the following example:
  //
  //   src dir: /project/src/foo.h
  //   current working directory (symlink): /tmp/build -> /project/src/
  //
  //  The file path of Symbol is "/project/src/foo.h" instead of
  //  "/tmp/build/foo.h"
  if (const DirectoryEntry *Dir = SourceMgr.getFileManager().getDirectory(
          llvm::sys::path::parent_path(FilePath))) {
    llvm::SmallString<128> RealPath;
    llvm::StringRef DirName = SourceMgr.getFileManager().getCanonicalName(Dir);
    llvm::sys::path::append(RealPath, DirName,
                            llvm::sys::path::filename(FilePath));
    return RealPath.str().str();
  }

  return FilePath.str().str();
}

TextEdit toTextEdit(const FixItHint &FixIt, const SourceManager &M,
                    const LangOptions &L) {
  TextEdit Result;
  Result.range =
      halfOpenToRange(M, Lexer::makeFileCharRange(FixIt.RemoveRange, M, L));
  Result.newText = FixIt.CodeToInsert;
  return Result;
}

bool isRangeConsecutive(const Range &Left, const Range &Right) {
  return Left.end.line == Right.start.line &&
         Left.end.character == Right.start.character;
}

FileDigest digest(llvm::StringRef Content) {
  return llvm::SHA1::hash({(const uint8_t *)Content.data(), Content.size()});
}

llvm::Optional<FileDigest> digestFile(const SourceManager &SM, FileID FID) {
  bool Invalid = false;
  llvm::StringRef Content = SM.getBufferData(FID, &Invalid);
  if (Invalid)
    return None;
  return digest(Content);
}

format::FormatStyle getFormatStyleForFile(llvm::StringRef File,
                                          llvm::StringRef Content,
                                          llvm::vfs::FileSystem *FS) {
  auto Style = format::getStyle(format::DefaultFormatStyle, File,
                                format::DefaultFallbackStyle, Content, FS);
  if (!Style) {
    log("getStyle() failed for file {0}: {1}. Fallback is LLVM style.", File,
        Style.takeError());
    Style = format::getLLVMStyle();
  }
  return *Style;
}

llvm::Expected<tooling::Replacements>
cleanupAndFormat(StringRef Code, const tooling::Replacements &Replaces,
                 const format::FormatStyle &Style) {
  auto CleanReplaces = cleanupAroundReplacements(Code, Replaces, Style);
  if (!CleanReplaces)
    return CleanReplaces;
  return formatReplacements(Code, std::move(*CleanReplaces), Style);
}

template <typename Action>
static void lex(llvm::StringRef Code, const format::FormatStyle &Style,
                Action A) {
  // FIXME: InMemoryFileAdapter crashes unless the buffer is null terminated!
  std::string NullTerminatedCode = Code.str();
  SourceManagerForFile FileSM("dummy.cpp", NullTerminatedCode);
  auto &SM = FileSM.get();
  auto FID = SM.getMainFileID();
  Lexer Lex(FID, SM.getBuffer(FID), SM, format::getFormattingLangOpts(Style));
  Token Tok;

  while (!Lex.LexFromRawLexer(Tok))
    A(Tok);
}

llvm::StringMap<unsigned> collectIdentifiers(llvm::StringRef Content,
                                             const format::FormatStyle &Style) {
  llvm::StringMap<unsigned> Identifiers;
  lex(Content, Style, [&](const clang::Token &Tok) {
    switch (Tok.getKind()) {
    case tok::identifier:
      ++Identifiers[Tok.getIdentifierInfo()->getName()];
      break;
    case tok::raw_identifier:
      ++Identifiers[Tok.getRawIdentifier()];
      break;
    default:
      break;
    }
  });
  return Identifiers;
}

namespace {
enum NamespaceEvent {
  BeginNamespace, // namespace <ns> {.     Payload is resolved <ns>.
  EndNamespace,   // } // namespace <ns>.  Payload is resolved *outer* namespace.
  UsingDirective  // using namespace <ns>. Payload is unresolved <ns>.
};
// Scans C++ source code for constructs that change the visible namespaces.
void parseNamespaceEvents(
    llvm::StringRef Code, const format::FormatStyle &Style,
    llvm::function_ref<void(NamespaceEvent, llvm::StringRef)> Callback) {

  // Stack of enclosing namespaces, e.g. {"clang", "clangd"}
  std::vector<std::string> Enclosing; // Contains e.g. "clang", "clangd"
  // Stack counts open braces. true if the brace opened a namespace.
  std::vector<bool> BraceStack;

  enum {
    Default,
    Namespace,          // just saw 'namespace'
    NamespaceName,      // just saw 'namespace' NSName
    Using,              // just saw 'using'
    UsingNamespace,     // just saw 'using namespace'
    UsingNamespaceName, // just saw 'using namespace' NSName
  } State = Default;
  std::string NSName;

  lex(Code, Style, [&](const clang::Token &Tok) {
    switch(Tok.getKind()) {
    case tok::raw_identifier:
      // In raw mode, this could be a keyword or a name.
      switch (State) {
      case UsingNamespace:
      case UsingNamespaceName:
        NSName.append(Tok.getRawIdentifier());
        State = UsingNamespaceName;
        break;
      case Namespace:
      case NamespaceName:
        NSName.append(Tok.getRawIdentifier());
        State = NamespaceName;
        break;
      case Using:
        State =
            (Tok.getRawIdentifier() == "namespace") ? UsingNamespace : Default;
        break;
      case Default:
        NSName.clear();
        if (Tok.getRawIdentifier() == "namespace")
          State = Namespace;
        else if (Tok.getRawIdentifier() == "using")
          State = Using;
        break;
      }
      break;
    case tok::coloncolon:
      // This can come at the beginning or in the middle of a namespace name.
      switch (State) {
      case UsingNamespace:
      case UsingNamespaceName:
        NSName.append("::");
        State = UsingNamespaceName;
        break;
      case NamespaceName:
        NSName.append("::");
        State = NamespaceName;
        break;
      case Namespace: // Not legal here.
      case Using:
      case Default:
        State = Default;
        break;
      }
      break;
    case tok::l_brace:
      // Record which { started a namespace, so we know when } ends one.
      if (State == NamespaceName) {
        // Parsed: namespace <name> {
        BraceStack.push_back(true);
        Enclosing.push_back(NSName);
        Callback(BeginNamespace, llvm::join(Enclosing, "::"));
      } else {
        // This case includes anonymous namespaces (State = Namespace).
        // For our purposes, they're not namespaces and we ignore them.
        BraceStack.push_back(false);
      }
      State = Default;
      break;
    case tok::r_brace:
      // If braces are unmatched, we're going to be confused, but don't crash.
      if (!BraceStack.empty()) {
        if (BraceStack.back()) {
          // Parsed: } // namespace
          Enclosing.pop_back();
          Callback(EndNamespace, llvm::join(Enclosing, "::"));
        }
        BraceStack.pop_back();
      }
      break;
    case tok::semi:
      if (State == UsingNamespaceName)
        // Parsed: using namespace <name> ;
        Callback(UsingDirective, llvm::StringRef(NSName));
      State = Default;
      break;
    default:
      State = Default;
      break;
    }
  });
}

// Returns the prefix namespaces of NS: {"" ... NS}.
llvm::SmallVector<llvm::StringRef, 8> ancestorNamespaces(llvm::StringRef NS) {
  llvm::SmallVector<llvm::StringRef, 8> Results;
  Results.push_back(NS.take_front(0));
  NS.split(Results, "::", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef &R : Results)
    R = NS.take_front(R.end() - NS.begin());
  return Results;
}

} // namespace

std::vector<std::string> visibleNamespaces(llvm::StringRef Code,
                                           const format::FormatStyle &Style) {
  std::string Current;
  // Map from namespace to (resolved) namespaces introduced via using directive.
  llvm::StringMap<llvm::StringSet<>> UsingDirectives;

  parseNamespaceEvents(Code, Style,
                       [&](NamespaceEvent Event, llvm::StringRef NS) {
                         switch (Event) {
                         case BeginNamespace:
                         case EndNamespace:
                           Current = NS;
                           break;
                         case UsingDirective:
                           if (NS.consume_front("::"))
                             UsingDirectives[Current].insert(NS);
                           else {
                             for (llvm::StringRef Enclosing :
                                  ancestorNamespaces(Current)) {
                               if (Enclosing.empty())
                                 UsingDirectives[Current].insert(NS);
                               else
                                 UsingDirectives[Current].insert(
                                     (Enclosing + "::" + NS).str());
                             }
                           }
                           break;
                         }
                       });

  std::vector<std::string> Found;
  for (llvm::StringRef Enclosing : ancestorNamespaces(Current)) {
    Found.push_back(Enclosing);
    auto It = UsingDirectives.find(Enclosing);
    if (It != UsingDirectives.end())
      for (const auto& Used : It->second)
        Found.push_back(Used.getKey());
  }


  llvm::sort(Found, [&](const std::string &LHS, const std::string &RHS) {
    if (Current == RHS)
      return false;
    if (Current == LHS)
      return true;
    return LHS < RHS;
  });
  Found.erase(std::unique(Found.begin(), Found.end()), Found.end());
  return Found;
}

llvm::StringSet<> collectWords(llvm::StringRef Content) {
  // We assume short words are not significant.
  // We may want to consider other stopwords, e.g. language keywords.
  // (A very naive implementation showed no benefit, but lexing might do better)
  static constexpr int MinWordLength = 4;

  std::vector<CharRole> Roles(Content.size());
  calculateRoles(Content, Roles);

  llvm::StringSet<> Result;
  llvm::SmallString<256> Word;
  auto Flush = [&] {
    if (Word.size() >= MinWordLength) {
      for (char &C : Word)
        C = llvm::toLower(C);
      Result.insert(Word);
    }
    Word.clear();
  };
  for (unsigned I = 0; I < Content.size(); ++I) {
    switch (Roles[I]) {
    case Head:
      Flush();
      LLVM_FALLTHROUGH;
    case Tail:
      Word.push_back(Content[I]);
      break;
    case Unknown:
    case Separator:
      Flush();
      break;
    }
  }
  Flush();

  return Result;
}

} // namespace clangd
} // namespace clang
