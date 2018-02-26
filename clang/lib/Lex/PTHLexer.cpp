//===- PTHLexer.cpp - Lex from a token stream -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTHLexer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/PTHLexer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/PTHManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/OnDiskHashTable.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <utility>

using namespace clang;

static const unsigned StoredTokenSize = 1 + 1 + 2 + 4 + 4;

//===----------------------------------------------------------------------===//
// PTHLexer methods.
//===----------------------------------------------------------------------===//

PTHLexer::PTHLexer(Preprocessor &PP, FileID FID, const unsigned char *D,
                   const unsigned char *ppcond, PTHManager &PM)
    : PreprocessorLexer(&PP, FID), TokBuf(D), CurPtr(D), PPCond(ppcond),
      CurPPCondPtr(ppcond), PTHMgr(PM) {
  FileStartLoc = PP.getSourceManager().getLocForStartOfFile(FID);
}

bool PTHLexer::Lex(Token& Tok) {
  //===--------------------------------------==//
  // Read the raw token data.
  //===--------------------------------------==//
  using namespace llvm::support;

  // Shadow CurPtr into an automatic variable.
  const unsigned char *CurPtrShadow = CurPtr;

  // Read in the data for the token.
  unsigned Word0 = endian::readNext<uint32_t, little, aligned>(CurPtrShadow);
  uint32_t IdentifierID =
      endian::readNext<uint32_t, little, aligned>(CurPtrShadow);
  uint32_t FileOffset =
      endian::readNext<uint32_t, little, aligned>(CurPtrShadow);

  tok::TokenKind TKind = (tok::TokenKind) (Word0 & 0xFF);
  Token::TokenFlags TFlags = (Token::TokenFlags) ((Word0 >> 8) & 0xFF);
  uint32_t Len = Word0 >> 16;

  CurPtr = CurPtrShadow;

  //===--------------------------------------==//
  // Construct the token itself.
  //===--------------------------------------==//

  Tok.startToken();
  Tok.setKind(TKind);
  Tok.setFlag(TFlags);
  assert(!LexingRawMode);
  Tok.setLocation(FileStartLoc.getLocWithOffset(FileOffset));
  Tok.setLength(Len);

  // Handle identifiers.
  if (Tok.isLiteral()) {
    Tok.setLiteralData((const char*) (PTHMgr.SpellingBase + IdentifierID));
  }
  else if (IdentifierID) {
    MIOpt.ReadToken();
    IdentifierInfo *II = PTHMgr.GetIdentifierInfo(IdentifierID-1);

    Tok.setIdentifierInfo(II);

    // Change the kind of this identifier to the appropriate token kind, e.g.
    // turning "for" into a keyword.
    Tok.setKind(II->getTokenID());

    if (II->isHandleIdentifierCase())
      return PP->HandleIdentifier(Tok);

    return true;
  }

  //===--------------------------------------==//
  // Process the token.
  //===--------------------------------------==//
  if (TKind == tok::eof) {
    // Save the end-of-file token.
    EofToken = Tok;

    assert(!ParsingPreprocessorDirective);
    assert(!LexingRawMode);

    return LexEndOfFile(Tok);
  }

  if (TKind == tok::hash && Tok.isAtStartOfLine()) {
    LastHashTokPtr = CurPtr - StoredTokenSize;
    assert(!LexingRawMode);
    PP->HandleDirective(Tok);

    return false;
  }

  if (TKind == tok::eod) {
    assert(ParsingPreprocessorDirective);
    ParsingPreprocessorDirective = false;
    return true;
  }

  MIOpt.ReadToken();
  return true;
}

bool PTHLexer::LexEndOfFile(Token &Result) {
  // If we hit the end of the file while parsing a preprocessor directive,
  // end the preprocessor directive first.  The next token returned will
  // then be the end of file.
  if (ParsingPreprocessorDirective) {
    ParsingPreprocessorDirective = false; // Done parsing the "line".
    return true;  // Have a token.
  }
  
  assert(!LexingRawMode);

  // If we are in a #if directive, emit an error.
  while (!ConditionalStack.empty()) {
    if (PP->getCodeCompletionFileLoc() != FileStartLoc)
      PP->Diag(ConditionalStack.back().IfLoc,
               diag::err_pp_unterminated_conditional);
    ConditionalStack.pop_back();
  }

  // Finally, let the preprocessor handle this.
  return PP->HandleEndOfFile(Result);
}

// FIXME: We can just grab the last token instead of storing a copy
// into EofToken.
void PTHLexer::getEOF(Token& Tok) {
  assert(EofToken.is(tok::eof));
  Tok = EofToken;
}

void PTHLexer::DiscardToEndOfLine() {
  assert(ParsingPreprocessorDirective && ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // We assume that if the preprocessor wishes to discard to the end of
  // the line that it also means to end the current preprocessor directive.
  ParsingPreprocessorDirective = false;

  // Skip tokens by only peeking at their token kind and the flags.
  // We don't need to actually reconstruct full tokens from the token buffer.
  // This saves some copies and it also reduces IdentifierInfo* lookup.
  const unsigned char* p = CurPtr;
  while (true) {
    // Read the token kind.  Are we at the end of the file?
    tok::TokenKind x = (tok::TokenKind) (uint8_t) *p;
    if (x == tok::eof) break;

    // Read the token flags.  Are we at the start of the next line?
    Token::TokenFlags y = (Token::TokenFlags) (uint8_t) p[1];
    if (y & Token::StartOfLine) break;

    // Skip to the next token.
    p += StoredTokenSize;
  }

  CurPtr = p;
}

/// SkipBlock - Used by Preprocessor to skip the current conditional block.
bool PTHLexer::SkipBlock() {
  using namespace llvm::support;

  assert(CurPPCondPtr && "No cached PP conditional information.");
  assert(LastHashTokPtr && "No known '#' token.");

  const unsigned char *HashEntryI = nullptr;
  uint32_t TableIdx;

  do {
    // Read the token offset from the side-table.
    uint32_t Offset = endian::readNext<uint32_t, little, aligned>(CurPPCondPtr);

    // Read the target table index from the side-table.
    TableIdx = endian::readNext<uint32_t, little, aligned>(CurPPCondPtr);

    // Compute the actual memory address of the '#' token data for this entry.
    HashEntryI = TokBuf + Offset;

    // Optmization: "Sibling jumping".  #if...#else...#endif blocks can
    //  contain nested blocks.  In the side-table we can jump over these
    //  nested blocks instead of doing a linear search if the next "sibling"
    //  entry is not at a location greater than LastHashTokPtr.
    if (HashEntryI < LastHashTokPtr && TableIdx) {
      // In the side-table we are still at an entry for a '#' token that
      // is earlier than the last one we saw.  Check if the location we would
      // stride gets us closer.
      const unsigned char* NextPPCondPtr =
        PPCond + TableIdx*(sizeof(uint32_t)*2);
      assert(NextPPCondPtr >= CurPPCondPtr);
      // Read where we should jump to.
      const unsigned char *HashEntryJ =
          TokBuf + endian::readNext<uint32_t, little, aligned>(NextPPCondPtr);

      if (HashEntryJ <= LastHashTokPtr) {
        // Jump directly to the next entry in the side table.
        HashEntryI = HashEntryJ;
        TableIdx = endian::readNext<uint32_t, little, aligned>(NextPPCondPtr);
        CurPPCondPtr = NextPPCondPtr;
      }
    }
  }
  while (HashEntryI < LastHashTokPtr);
  assert(HashEntryI == LastHashTokPtr && "No PP-cond entry found for '#'");
  assert(TableIdx && "No jumping from #endifs.");

  // Update our side-table iterator.
  const unsigned char* NextPPCondPtr = PPCond + TableIdx*(sizeof(uint32_t)*2);
  assert(NextPPCondPtr >= CurPPCondPtr);
  CurPPCondPtr = NextPPCondPtr;

  // Read where we should jump to.
  HashEntryI =
      TokBuf + endian::readNext<uint32_t, little, aligned>(NextPPCondPtr);
  uint32_t NextIdx = endian::readNext<uint32_t, little, aligned>(NextPPCondPtr);

  // By construction NextIdx will be zero if this is a #endif.  This is useful
  // to know to obviate lexing another token.
  bool isEndif = NextIdx == 0;

  // This case can occur when we see something like this:
  //
  //  #if ...
  //   /* a comment or nothing */
  //  #elif
  //
  // If we are skipping the first #if block it will be the case that CurPtr
  // already points 'elif'.  Just return.

  if (CurPtr > HashEntryI) {
    assert(CurPtr == HashEntryI + StoredTokenSize);
    // Did we reach a #endif?  If so, go ahead and consume that token as well.
    if (isEndif)
      CurPtr += StoredTokenSize * 2;
    else
      LastHashTokPtr = HashEntryI;

    return isEndif;
  }

  // Otherwise, we need to advance.  Update CurPtr to point to the '#' token.
  CurPtr = HashEntryI;

  // Update the location of the last observed '#'.  This is useful if we
  // are skipping multiple blocks.
  LastHashTokPtr = CurPtr;

  // Skip the '#' token.
  assert(((tok::TokenKind)*CurPtr) == tok::hash);
  CurPtr += StoredTokenSize;

  // Did we reach a #endif?  If so, go ahead and consume that token as well.
  if (isEndif) {
    CurPtr += StoredTokenSize * 2;
  }

  return isEndif;
}

SourceLocation PTHLexer::getSourceLocation() {
  // getSourceLocation is not on the hot path.  It is used to get the location
  // of the next token when transitioning back to this lexer when done
  // handling a #included file.  Just read the necessary data from the token
  // data buffer to construct the SourceLocation object.
  // NOTE: This is a virtual function; hence it is defined out-of-line.
  using namespace llvm::support;

  const unsigned char *OffsetPtr = CurPtr + (StoredTokenSize - 4);
  uint32_t Offset = endian::readNext<uint32_t, little, aligned>(OffsetPtr);
  return FileStartLoc.getLocWithOffset(Offset);
}

//===----------------------------------------------------------------------===//
// PTH file lookup: map from strings to file data.
//===----------------------------------------------------------------------===//

/// PTHFileLookup - This internal data structure is used by the PTHManager
///  to map from FileEntry objects managed by FileManager to offsets within
///  the PTH file.
namespace {

class PTHFileData {
  const uint32_t TokenOff;
  const uint32_t PPCondOff;

public:
  PTHFileData(uint32_t tokenOff, uint32_t ppCondOff)
      : TokenOff(tokenOff), PPCondOff(ppCondOff) {}

  uint32_t getTokenOffset() const { return TokenOff; }
  uint32_t getPPCondOffset() const { return PPCondOff; }
};

class PTHFileLookupCommonTrait {
public:
  using internal_key_type = std::pair<unsigned char, StringRef>;
  using hash_value_type = unsigned;
  using offset_type = unsigned;

  static hash_value_type ComputeHash(internal_key_type x) {
    return llvm::HashString(x.second);
  }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace llvm::support;

    unsigned keyLen =
        (unsigned)endian::readNext<uint16_t, little, unaligned>(d);
    unsigned dataLen = (unsigned) *(d++);
    return std::make_pair(keyLen, dataLen);
  }

  static internal_key_type ReadKey(const unsigned char* d, unsigned) {
    unsigned char k = *(d++); // Read the entry kind.
    return std::make_pair(k, (const char*) d);
  }
};

} // namespace

class PTHManager::PTHFileLookupTrait : public PTHFileLookupCommonTrait {
public:
  using external_key_type = const FileEntry *;
  using data_type = PTHFileData;

  static internal_key_type GetInternalKey(const FileEntry* FE) {
    return std::make_pair((unsigned char) 0x1, FE->getName());
  }

  static bool EqualKey(internal_key_type a, internal_key_type b) {
    return a.first == b.first && a.second == b.second;
  }

  static PTHFileData ReadData(const internal_key_type& k,
                              const unsigned char* d, unsigned) {
    using namespace llvm::support;

    assert(k.first == 0x1 && "Only file lookups can match!");
    uint32_t x = endian::readNext<uint32_t, little, unaligned>(d);
    uint32_t y = endian::readNext<uint32_t, little, unaligned>(d);
    return PTHFileData(x, y);
  }
};

class PTHManager::PTHStringLookupTrait {
public:
  using data_type = uint32_t;
  using external_key_type = const std::pair<const char *, unsigned>;
  using internal_key_type = external_key_type;
  using hash_value_type = uint32_t;
  using offset_type = unsigned;

  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return (a.second == b.second) ? memcmp(a.first, b.first, a.second) == 0
                                  : false;
  }

  static hash_value_type ComputeHash(const internal_key_type& a) {
    return llvm::HashString(StringRef(a.first, a.second));
  }

  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace llvm::support;

    return std::make_pair(
        (unsigned)endian::readNext<uint16_t, little, unaligned>(d),
        sizeof(uint32_t));
  }

  static std::pair<const char*, unsigned>
  ReadKey(const unsigned char* d, unsigned n) {
      assert(n >= 2 && d[n-1] == '\0');
      return std::make_pair((const char*) d, n-1);
    }

  static uint32_t ReadData(const internal_key_type& k, const unsigned char* d,
                           unsigned) {
    using namespace llvm::support;

    return endian::readNext<uint32_t, little, unaligned>(d);
  }
};

//===----------------------------------------------------------------------===//
// PTHManager methods.
//===----------------------------------------------------------------------===//

PTHManager::PTHManager(
    std::unique_ptr<const llvm::MemoryBuffer> buf,
    std::unique_ptr<PTHFileLookup> fileLookup, const unsigned char *idDataTable,
    std::unique_ptr<IdentifierInfo *[], llvm::FreeDeleter> perIDCache,
    std::unique_ptr<PTHStringIdLookup> stringIdLookup, unsigned numIds,
    const unsigned char *spellingBase, const char *originalSourceFile)
    : Buf(std::move(buf)), PerIDCache(std::move(perIDCache)),
      FileLookup(std::move(fileLookup)), IdDataTable(idDataTable),
      StringIdLookup(std::move(stringIdLookup)), NumIds(numIds),
      SpellingBase(spellingBase), OriginalSourceFile(originalSourceFile) {}

PTHManager::~PTHManager() = default;

static void InvalidPTH(DiagnosticsEngine &Diags, const char *Msg) {
  Diags.Report(Diags.getCustomDiagID(DiagnosticsEngine::Error, "%0")) << Msg;
}

PTHManager *PTHManager::Create(StringRef file, DiagnosticsEngine &Diags) {
  // Memory map the PTH file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(file);

  if (!FileOrErr) {
    // FIXME: Add ec.message() to this diag.
    Diags.Report(diag::err_invalid_pth_file) << file;
    return nullptr;
  }
  std::unique_ptr<llvm::MemoryBuffer> File = std::move(FileOrErr.get());

  using namespace llvm::support;

  // Get the buffer ranges and check if there are at least three 32-bit
  // words at the end of the file.
  const unsigned char *BufBeg = (const unsigned char*)File->getBufferStart();
  const unsigned char *BufEnd = (const unsigned char*)File->getBufferEnd();

  // Check the prologue of the file.
  if ((BufEnd - BufBeg) < (signed)(sizeof("cfe-pth") + 4 + 4) ||
      memcmp(BufBeg, "cfe-pth", sizeof("cfe-pth")) != 0) {
    Diags.Report(diag::err_invalid_pth_file) << file;
    return nullptr;
  }

  // Read the PTH version.
  const unsigned char *p = BufBeg + (sizeof("cfe-pth"));
  unsigned Version = endian::readNext<uint32_t, little, aligned>(p);

  if (Version < PTHManager::Version) {
    InvalidPTH(Diags,
        Version < PTHManager::Version
        ? "PTH file uses an older PTH format that is no longer supported"
        : "PTH file uses a newer PTH format that cannot be read");
    return nullptr;
  }

  // Compute the address of the index table at the end of the PTH file.
  const unsigned char *PrologueOffset = p;

  if (PrologueOffset >= BufEnd) {
    Diags.Report(diag::err_invalid_pth_file) << file;
    return nullptr;
  }

  // Construct the file lookup table.  This will be used for mapping from
  // FileEntry*'s to cached tokens.
  const unsigned char* FileTableOffset = PrologueOffset + sizeof(uint32_t)*2;
  const unsigned char *FileTable =
      BufBeg + endian::readNext<uint32_t, little, aligned>(FileTableOffset);

  if (!(FileTable > BufBeg && FileTable < BufEnd)) {
    Diags.Report(diag::err_invalid_pth_file) << file;
    return nullptr; // FIXME: Proper error diagnostic?
  }

  std::unique_ptr<PTHFileLookup> FL(PTHFileLookup::Create(FileTable, BufBeg));

  // Warn if the PTH file is empty.  We still want to create a PTHManager
  // as the PTH could be used with -include-pth.
  if (FL->isEmpty())
    InvalidPTH(Diags, "PTH file contains no cached source data");

  // Get the location of the table mapping from persistent ids to the
  // data needed to reconstruct identifiers.
  const unsigned char* IDTableOffset = PrologueOffset + sizeof(uint32_t)*0;
  const unsigned char *IData =
      BufBeg + endian::readNext<uint32_t, little, aligned>(IDTableOffset);

  if (!(IData >= BufBeg && IData < BufEnd)) {
    Diags.Report(diag::err_invalid_pth_file) << file;
    return nullptr;
  }

  // Get the location of the hashtable mapping between strings and
  // persistent IDs.
  const unsigned char* StringIdTableOffset = PrologueOffset + sizeof(uint32_t)*1;
  const unsigned char *StringIdTable =
      BufBeg + endian::readNext<uint32_t, little, aligned>(StringIdTableOffset);
  if (!(StringIdTable >= BufBeg && StringIdTable < BufEnd)) {
    Diags.Report(diag::err_invalid_pth_file) << file;
    return nullptr;
  }

  std::unique_ptr<PTHStringIdLookup> SL(
      PTHStringIdLookup::Create(StringIdTable, BufBeg));

  // Get the location of the spelling cache.
  const unsigned char* spellingBaseOffset = PrologueOffset + sizeof(uint32_t)*3;
  const unsigned char *spellingBase =
      BufBeg + endian::readNext<uint32_t, little, aligned>(spellingBaseOffset);
  if (!(spellingBase >= BufBeg && spellingBase < BufEnd)) {
    Diags.Report(diag::err_invalid_pth_file) << file;
    return nullptr;
  }

  // Get the number of IdentifierInfos and pre-allocate the identifier cache.
  uint32_t NumIds = endian::readNext<uint32_t, little, aligned>(IData);

  // Pre-allocate the persistent ID -> IdentifierInfo* cache.  We use calloc()
  // so that we in the best case only zero out memory once when the OS returns
  // us new pages.
  std::unique_ptr<IdentifierInfo *[], llvm::FreeDeleter> PerIDCache;

  if (NumIds) {
    PerIDCache.reset((IdentifierInfo **)calloc(NumIds, sizeof(PerIDCache[0])));
    if (!PerIDCache) {
      InvalidPTH(Diags, "Could not allocate memory for processing PTH file");
      return nullptr;
    }
  }

  // Compute the address of the original source file.
  const unsigned char* originalSourceBase = PrologueOffset + sizeof(uint32_t)*4;
  unsigned len =
      endian::readNext<uint16_t, little, unaligned>(originalSourceBase);
  if (!len) originalSourceBase = nullptr;

  // Create the new PTHManager.
  return new PTHManager(std::move(File), std::move(FL), IData,
                        std::move(PerIDCache), std::move(SL), NumIds,
                        spellingBase, (const char *)originalSourceBase);
}

IdentifierInfo* PTHManager::LazilyCreateIdentifierInfo(unsigned PersistentID) {
  using namespace llvm::support;

  // Look in the PTH file for the string data for the IdentifierInfo object.
  const unsigned char* TableEntry = IdDataTable + sizeof(uint32_t)*PersistentID;
  const unsigned char *IDData =
      (const unsigned char *)Buf->getBufferStart() +
      endian::readNext<uint32_t, little, aligned>(TableEntry);
  assert(IDData < (const unsigned char*)Buf->getBufferEnd());

  // Allocate the object.
  std::pair<IdentifierInfo,const unsigned char*> *Mem =
      Alloc.Allocate<std::pair<IdentifierInfo, const unsigned char *>>();

  Mem->second = IDData;
  assert(IDData[0] != '\0');
  IdentifierInfo *II = new ((void*) Mem) IdentifierInfo();

  // Store the new IdentifierInfo in the cache.
  PerIDCache[PersistentID] = II;
  assert(II->getNameStart() && II->getNameStart()[0] != '\0');
  return II;
}

IdentifierInfo* PTHManager::get(StringRef Name) {
  // Double check our assumption that the last character isn't '\0'.
  assert(Name.empty() || Name.back() != '\0');
  PTHStringIdLookup::iterator I =
      StringIdLookup->find(std::make_pair(Name.data(), Name.size()));
  if (I == StringIdLookup->end()) // No identifier found?
    return nullptr;

  // Match found.  Return the identifier!
  assert(*I > 0);
  return GetIdentifierInfo(*I-1);
}

PTHLexer *PTHManager::CreateLexer(FileID FID) {
  const FileEntry *FE = PP->getSourceManager().getFileEntryForID(FID);
  if (!FE)
    return nullptr;

  using namespace llvm::support;

  // Lookup the FileEntry object in our file lookup data structure.  It will
  // return a variant that indicates whether or not there is an offset within
  // the PTH file that contains cached tokens.
  PTHFileLookup::iterator I = FileLookup->find(FE);

  if (I == FileLookup->end()) // No tokens available?
    return nullptr;

  const PTHFileData& FileData = *I;

  const unsigned char *BufStart = (const unsigned char *)Buf->getBufferStart();
  // Compute the offset of the token data within the buffer.
  const unsigned char* data = BufStart + FileData.getTokenOffset();

  // Get the location of pp-conditional table.
  const unsigned char* ppcond = BufStart + FileData.getPPCondOffset();
  uint32_t Len = endian::readNext<uint32_t, little, aligned>(ppcond);
  if (Len == 0) ppcond = nullptr;

  assert(PP && "No preprocessor set yet!");
  return new PTHLexer(*PP, FID, data, ppcond, *this);
}

//===----------------------------------------------------------------------===//
// 'stat' caching.
//===----------------------------------------------------------------------===//

namespace {

class PTHStatData {
public:
  uint64_t Size;
  time_t ModTime;
  llvm::sys::fs::UniqueID UniqueID;
  const bool HasData = false;
  bool IsDirectory;

  PTHStatData() = default;
  PTHStatData(uint64_t Size, time_t ModTime, llvm::sys::fs::UniqueID UniqueID,
              bool IsDirectory)
      : Size(Size), ModTime(ModTime), UniqueID(UniqueID), HasData(true),
        IsDirectory(IsDirectory) {}
};

class PTHStatLookupTrait : public PTHFileLookupCommonTrait {
public:
  using external_key_type = StringRef; // const char*
  using data_type = PTHStatData;

  static internal_key_type GetInternalKey(StringRef path) {
    // The key 'kind' doesn't matter here because it is ignored in EqualKey.
    return std::make_pair((unsigned char) 0x0, path);
  }

  static bool EqualKey(internal_key_type a, internal_key_type b) {
    // When doing 'stat' lookups we don't care about the kind of 'a' and 'b',
    // just the paths.
    return a.second == b.second;
  }

  static data_type ReadData(const internal_key_type& k, const unsigned char* d,
                            unsigned) {
    if (k.first /* File or Directory */) {
      bool IsDirectory = true;
      if (k.first == 0x1 /* File */) {
        IsDirectory = false;
        d += 4 * 2; // Skip the first 2 words.
      }

      using namespace llvm::support;

      uint64_t File = endian::readNext<uint64_t, little, unaligned>(d);
      uint64_t Device = endian::readNext<uint64_t, little, unaligned>(d);
      llvm::sys::fs::UniqueID UniqueID(Device, File);
      time_t ModTime = endian::readNext<uint64_t, little, unaligned>(d);
      uint64_t Size = endian::readNext<uint64_t, little, unaligned>(d);
      return data_type(Size, ModTime, UniqueID, IsDirectory);
    }

    // Negative stat.  Don't read anything.
    return data_type();
  }
};

} // namespace

namespace clang {

class PTHStatCache : public FileSystemStatCache {
  using CacheTy = llvm::OnDiskChainedHashTable<PTHStatLookupTrait>;

  CacheTy Cache;

public:
  PTHStatCache(PTHManager::PTHFileLookup &FL)
      : Cache(FL.getNumBuckets(), FL.getNumEntries(), FL.getBuckets(),
              FL.getBase()) {}

  LookupResult getStat(StringRef Path, FileData &Data, bool isFile,
                       std::unique_ptr<vfs::File> *F,
                       vfs::FileSystem &FS) override {
    // Do the lookup for the file's data in the PTH file.
    CacheTy::iterator I = Cache.find(Path);

    // If we don't get a hit in the PTH file just forward to 'stat'.
    if (I == Cache.end())
      return statChained(Path, Data, isFile, F, FS);

    const PTHStatData &D = *I;

    if (!D.HasData)
      return CacheMissing;

    Data.Name = Path;
    Data.Size = D.Size;
    Data.ModTime = D.ModTime;
    Data.UniqueID = D.UniqueID;
    Data.IsDirectory = D.IsDirectory;
    Data.IsNamedPipe = false;
    Data.InPCH = true;

    return CacheExists;
  }
};

} // namespace clang

std::unique_ptr<FileSystemStatCache> PTHManager::createStatCache() {
  return llvm::make_unique<PTHStatCache>(*FileLookup);
}
