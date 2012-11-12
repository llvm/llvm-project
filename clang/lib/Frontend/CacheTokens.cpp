//===--- CacheTokens.cpp - Caching of lexer tokens for PTH support --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a possible implementation of PTH support for Clang that is
// based on caching lexed tokens and identifiers.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"

// FIXME: put this somewhere else?
#ifndef S_ISDIR
#define S_ISDIR(x) (((x)&_S_IFDIR)!=0)
#endif

using namespace clang;
using namespace clang::io;

//===----------------------------------------------------------------------===//
// PTH-specific stuff.
//===----------------------------------------------------------------------===//

namespace {
class PTHEntry {
  Offset TokenData, PPCondData;

public:
  PTHEntry() {}

  PTHEntry(Offset td, Offset ppcd)
    : TokenData(td), PPCondData(ppcd) {}

  Offset getTokenOffset() const { return TokenData; }
  Offset getPPCondTableOffset() const { return PPCondData; }
};


class PTHEntryKeyVariant {
  union { const FileEntry* FE; const char* Path; };
  enum { IsFE = 0x1, IsDE = 0x2, IsNoExist = 0x0 } Kind;
  struct stat *StatBuf;
public:
  PTHEntryKeyVariant(const FileEntry *fe)
    : FE(fe), Kind(IsFE), StatBuf(0) {}

  PTHEntryKeyVariant(struct stat* statbuf, const char* path)
    : Path(path), Kind(IsDE), StatBuf(new struct stat(*statbuf)) {}

  explicit PTHEntryKeyVariant(const char* path)
    : Path(path), Kind(IsNoExist), StatBuf(0) {}

  bool isFile() const { return Kind == IsFE; }

  StringRef getString() const {
    return Kind == IsFE ? FE->getName() : Path;
  }

  unsigned getKind() const { return (unsigned) Kind; }

  void EmitData(raw_ostream& Out) {
    switch (Kind) {
    case IsFE:
      // Emit stat information.
      ::Emit32(Out, FE->getInode());
      ::Emit32(Out, FE->getDevice());
      ::Emit16(Out, FE->getFileMode());
      ::Emit64(Out, FE->getModificationTime());
      ::Emit64(Out, FE->getSize());
      break;
    case IsDE:
      // Emit stat information.
      ::Emit32(Out, (uint32_t) StatBuf->st_ino);
      ::Emit32(Out, (uint32_t) StatBuf->st_dev);
      ::Emit16(Out, (uint16_t) StatBuf->st_mode);
      ::Emit64(Out, (uint64_t) StatBuf->st_mtime);
      ::Emit64(Out, (uint64_t) StatBuf->st_size);
      delete StatBuf;
      break;
    default:
      break;
    }
  }

  unsigned getRepresentationLength() const {
    return Kind == IsNoExist ? 0 : 4 + 4 + 2 + 8 + 8;
  }
};

class FileEntryPTHEntryInfo {
public:
  typedef PTHEntryKeyVariant key_type;
  typedef key_type key_type_ref;

  typedef PTHEntry data_type;
  typedef const PTHEntry& data_type_ref;

  static unsigned ComputeHash(PTHEntryKeyVariant V) {
    return llvm::HashString(V.getString());
  }

  static std::pair<unsigned,unsigned>
  EmitKeyDataLength(raw_ostream& Out, PTHEntryKeyVariant V,
                    const PTHEntry& E) {

    unsigned n = V.getString().size() + 1 + 1;
    ::Emit16(Out, n);

    unsigned m = V.getRepresentationLength() + (V.isFile() ? 4 + 4 : 0);
    ::Emit8(Out, m);

    return std::make_pair(n, m);
  }

  static void EmitKey(raw_ostream& Out, PTHEntryKeyVariant V, unsigned n){
    // Emit the entry kind.
    ::Emit8(Out, (unsigned) V.getKind());
    // Emit the string.
    Out.write(V.getString().data(), n - 1);
  }

  static void EmitData(raw_ostream& Out, PTHEntryKeyVariant V,
                       const PTHEntry& E, unsigned) {


    // For file entries emit the offsets into the PTH file for token data
    // and the preprocessor blocks table.
    if (V.isFile()) {
      ::Emit32(Out, E.getTokenOffset());
      ::Emit32(Out, E.getPPCondTableOffset());
    }

    // Emit any other data associated with the key (i.e., stat information).
    V.EmitData(Out);
  }
};

class OffsetOpt {
  bool valid;
  Offset off;
public:
  OffsetOpt() : valid(false) {}
  bool hasOffset() const { return valid; }
  Offset getOffset() const { assert(valid); return off; }
  void setOffset(Offset o) { off = o; valid = true; }
};
} // end anonymous namespace

typedef OnDiskChainedHashTableGenerator<FileEntryPTHEntryInfo> PTHMap;

namespace {
class PTHWriter {
  typedef llvm::DenseMap<const IdentifierInfo*,uint32_t> IDMap;
  typedef llvm::StringMap<OffsetOpt, llvm::BumpPtrAllocator> CachedStrsTy;

  IDMap IM;
  llvm::raw_fd_ostream& Out;
  Preprocessor& PP;
  uint32_t idcount;
  PTHMap PM;
  CachedStrsTy CachedStrs;
  Offset CurStrOffset;
  std::vector<llvm::StringMapEntry<OffsetOpt>*> StrEntries;

  //// Get the persistent id for the given IdentifierInfo*.
  uint32_t ResolveID(const IdentifierInfo* II);

  /// Emit a token to the PTH file.
  void EmitToken(const Token& T);

  void Emit8(uint32_t V) { ::Emit8(Out, V); }

  void Emit16(uint32_t V) { ::Emit16(Out, V); }

  void Emit32(uint32_t V) { ::Emit32(Out, V); }

  void EmitBuf(const char *Ptr, unsigned NumBytes) {
    Out.write(Ptr, NumBytes);
  }

  void EmitString(StringRef V) {
    ::Emit16(Out, V.size());
    EmitBuf(V.data(), V.size());
  }

  /// EmitIdentifierTable - Emits two tables to the PTH file.  The first is
  ///  a hashtable mapping from identifier strings to persistent IDs.
  ///  The second is a straight table mapping from persistent IDs to string data
  ///  (the keys of the first table).
  std::pair<Offset, Offset> EmitIdentifierTable();

  /// EmitFileTable - Emit a table mapping from file name strings to PTH
  /// token data.
  Offset EmitFileTable() { return PM.Emit(Out); }

  PTHEntry LexTokens(Lexer& L);
  Offset EmitCachedSpellings();

public:
  PTHWriter(llvm::raw_fd_ostream& out, Preprocessor& pp)
    : Out(out), PP(pp), idcount(0), CurStrOffset(0) {}

  PTHMap &getPM() { return PM; }
  void GeneratePTH(const std::string &MainFile);
};
} // end anonymous namespace

uint32_t PTHWriter::ResolveID(const IdentifierInfo* II) {
  // Null IdentifierInfo's map to the persistent ID 0.
  if (!II)
    return 0;

  IDMap::iterator I = IM.find(II);
  if (I != IM.end())
    return I->second; // We've already added 1.

  IM[II] = ++idcount; // Pre-increment since '0' is reserved for NULL.
  return idcount;
}

void PTHWriter::EmitToken(const Token& T) {
  // Emit the token kind, flags, and length.
  Emit32(((uint32_t) T.getKind()) | ((((uint32_t) T.getFlags())) << 8)|
         (((uint32_t) T.getLength()) << 16));

  if (!T.isLiteral()) {
    Emit32(ResolveID(T.getIdentifierInfo()));
  } else {
    // We cache *un-cleaned* spellings. This gives us 100% fidelity with the
    // source code.
    StringRef s(T.getLiteralData(), T.getLength());

    // Get the string entry.
    llvm::StringMapEntry<OffsetOpt> *E = &CachedStrs.GetOrCreateValue(s);

    // If this is a new string entry, bump the PTH offset.
    if (!E->getValue().hasOffset()) {
      E->getValue().setOffset(CurStrOffset);
      StrEntries.push_back(E);
      CurStrOffset += s.size() + 1;
    }

    // Emit the relative offset into the PTH file for the spelling string.
    Emit32(E->getValue().getOffset());
  }

  // Emit the offset into the original source file of this token so that we
  // can reconstruct its SourceLocation.
  Emit32(PP.getSourceManager().getFileOffset(T.getLocation()));
}

PTHEntry PTHWriter::LexTokens(Lexer& L) {
  // Pad 0's so that we emit tokens to a 4-byte alignment.
  // This speed up reading them back in.
  Pad(Out, 4);
  Offset TokenOff = (Offset) Out.tell();

  // Keep track of matching '#if' ... '#endif'.
  typedef std::vector<std::pair<Offset, unsigned> > PPCondTable;
  PPCondTable PPCond;
  std::vector<unsigned> PPStartCond;
  bool ParsingPreprocessorDirective = false;
  Token Tok;

  do {
    L.LexFromRawLexer(Tok);
  NextToken:

    if ((Tok.isAtStartOfLine() || Tok.is(tok::eof)) &&
        ParsingPreprocessorDirective) {
      // Insert an eod token into the token cache.  It has the same
      // position as the next token that is not on the same line as the
      // preprocessor directive.  Observe that we continue processing
      // 'Tok' when we exit this branch.
      Token Tmp = Tok;
      Tmp.setKind(tok::eod);
      Tmp.clearFlag(Token::StartOfLine);
      Tmp.setIdentifierInfo(0);
      EmitToken(Tmp);
      ParsingPreprocessorDirective = false;
    }

    if (Tok.is(tok::raw_identifier)) {
      PP.LookUpIdentifierInfo(Tok);
      EmitToken(Tok);
      continue;
    }

    if (Tok.is(tok::hash) && Tok.isAtStartOfLine()) {
      // Special processing for #include.  Store the '#' token and lex
      // the next token.
      assert(!ParsingPreprocessorDirective);
      Offset HashOff = (Offset) Out.tell();

      // Get the next token.
      Token NextTok;
      L.LexFromRawLexer(NextTok);

      // If we see the start of line, then we had a null directive "#".  In
      // this case, discard both tokens.
      if (NextTok.isAtStartOfLine())
        goto NextToken;

      // The token is the start of a directive.  Emit it.
      EmitToken(Tok);
      Tok = NextTok;

      // Did we see 'include'/'import'/'include_next'?
      if (Tok.isNot(tok::raw_identifier)) {
        EmitToken(Tok);
        continue;
      }

      IdentifierInfo* II = PP.LookUpIdentifierInfo(Tok);
      tok::PPKeywordKind K = II->getPPKeywordID();

      ParsingPreprocessorDirective = true;

      switch (K) {
      case tok::pp_not_keyword:
        // Invalid directives "#foo" can occur in #if 0 blocks etc, just pass
        // them through.
      default:
        break;

      case tok::pp_include:
      case tok::pp_import:
      case tok::pp_include_next: {
        // Save the 'include' token.
        EmitToken(Tok);
        // Lex the next token as an include string.
        L.setParsingPreprocessorDirective(true);
        L.LexIncludeFilename(Tok);
        L.setParsingPreprocessorDirective(false);
        assert(!Tok.isAtStartOfLine());
        if (Tok.is(tok::raw_identifier))
          PP.LookUpIdentifierInfo(Tok);

        break;
      }
      case tok::pp_if:
      case tok::pp_ifdef:
      case tok::pp_ifndef: {
        // Add an entry for '#if' and friends.  We initially set the target
        // index to 0.  This will get backpatched when we hit #endif.
        PPStartCond.push_back(PPCond.size());
        PPCond.push_back(std::make_pair(HashOff, 0U));
        break;
      }
      case tok::pp_endif: {
        // Add an entry for '#endif'.  We set the target table index to itself.
        // This will later be set to zero when emitting to the PTH file.  We
        // use 0 for uninitialized indices because that is easier to debug.
        unsigned index = PPCond.size();
        // Backpatch the opening '#if' entry.
        assert(!PPStartCond.empty());
        assert(PPCond.size() > PPStartCond.back());
        assert(PPCond[PPStartCond.back()].second == 0);
        PPCond[PPStartCond.back()].second = index;
        PPStartCond.pop_back();
        // Add the new entry to PPCond.
        PPCond.push_back(std::make_pair(HashOff, index));
        EmitToken(Tok);

        // Some files have gibberish on the same line as '#endif'.
        // Discard these tokens.
        do
          L.LexFromRawLexer(Tok);
        while (Tok.isNot(tok::eof) && !Tok.isAtStartOfLine());
        // We have the next token in hand.
        // Don't immediately lex the next one.
        goto NextToken;
      }
      case tok::pp_elif:
      case tok::pp_else: {
        // Add an entry for #elif or #else.
        // This serves as both a closing and opening of a conditional block.
        // This means that its entry will get backpatched later.
        unsigned index = PPCond.size();
        // Backpatch the previous '#if' entry.
        assert(!PPStartCond.empty());
        assert(PPCond.size() > PPStartCond.back());
        assert(PPCond[PPStartCond.back()].second == 0);
        PPCond[PPStartCond.back()].second = index;
        PPStartCond.pop_back();
        // Now add '#elif' as a new block opening.
        PPCond.push_back(std::make_pair(HashOff, 0U));
        PPStartCond.push_back(index);
        break;
      }
      }
    }

    EmitToken(Tok);
  }
  while (Tok.isNot(tok::eof));

  assert(PPStartCond.empty() && "Error: imblanced preprocessor conditionals.");

  // Next write out PPCond.
  Offset PPCondOff = (Offset) Out.tell();

  // Write out the size of PPCond so that clients can identifer empty tables.
  Emit32(PPCond.size());

  for (unsigned i = 0, e = PPCond.size(); i!=e; ++i) {
    Emit32(PPCond[i].first - TokenOff);
    uint32_t x = PPCond[i].second;
    assert(x != 0 && "PPCond entry not backpatched.");
    // Emit zero for #endifs.  This allows us to do checking when
    // we read the PTH file back in.
    Emit32(x == i ? 0 : x);
  }

  return PTHEntry(TokenOff, PPCondOff);
}

Offset PTHWriter::EmitCachedSpellings() {
  // Write each cached strings to the PTH file.
  Offset SpellingsOff = Out.tell();

  for (std::vector<llvm::StringMapEntry<OffsetOpt>*>::iterator
       I = StrEntries.begin(), E = StrEntries.end(); I!=E; ++I)
    EmitBuf((*I)->getKeyData(), (*I)->getKeyLength()+1 /*nul included*/);

  return SpellingsOff;
}

void PTHWriter::GeneratePTH(const std::string &MainFile) {
  // Generate the prologue.
  Out << "cfe-pth" << '\0';
  Emit32(PTHManager::Version);

  // Leave 4 words for the prologue.
  Offset PrologueOffset = Out.tell();
  for (unsigned i = 0; i < 4; ++i)
    Emit32(0);

  // Write the name of the MainFile.
  if (!MainFile.empty()) {
    EmitString(MainFile);
  } else {
    // String with 0 bytes.
    Emit16(0);
  }
  Emit8(0);

  // Iterate over all the files in SourceManager.  Create a lexer
  // for each file and cache the tokens.
  SourceManager &SM = PP.getSourceManager();
  const LangOptions &LOpts = PP.getLangOpts();

  for (SourceManager::fileinfo_iterator I = SM.fileinfo_begin(),
       E = SM.fileinfo_end(); I != E; ++I) {
    const SrcMgr::ContentCache &C = *I->second;
    const FileEntry *FE = C.OrigEntry;

    // FIXME: Handle files with non-absolute paths.
    if (llvm::sys::path::is_relative(FE->getName()))
      continue;

    const llvm::MemoryBuffer *B = C.getBuffer(PP.getDiagnostics(), SM);
    if (!B) continue;

    FileID FID = SM.createFileID(FE, SourceLocation(), SrcMgr::C_User);
    const llvm::MemoryBuffer *FromFile = SM.getBuffer(FID);
    Lexer L(FID, FromFile, SM, LOpts);
    PM.insert(FE, LexTokens(L));
  }

  // Write out the identifier table.
  const std::pair<Offset,Offset> &IdTableOff = EmitIdentifierTable();

  // Write out the cached strings table.
  Offset SpellingOff = EmitCachedSpellings();

  // Write out the file table.
  Offset FileTableOff = EmitFileTable();

  // Finally, write the prologue.
  Out.seek(PrologueOffset);
  Emit32(IdTableOff.first);
  Emit32(IdTableOff.second);
  Emit32(FileTableOff);
  Emit32(SpellingOff);
}

namespace {
/// StatListener - A simple "interpose" object used to monitor stat calls
/// invoked by FileManager while processing the original sources used
/// as input to PTH generation.  StatListener populates the PTHWriter's
/// file map with stat information for directories as well as negative stats.
/// Stat information for files are populated elsewhere.
class StatListener : public FileSystemStatCache {
  PTHMap &PM;
public:
  StatListener(PTHMap &pm) : PM(pm) {}
  ~StatListener() {}

  LookupResult getStat(const char *Path, struct stat &StatBuf,
                       int *FileDescriptor) {
    LookupResult Result = statChained(Path, StatBuf, FileDescriptor);

    if (Result == CacheMissing) // Failed 'stat'.
      PM.insert(PTHEntryKeyVariant(Path), PTHEntry());
    else if (S_ISDIR(StatBuf.st_mode)) {
      // Only cache directories with absolute paths.
      if (llvm::sys::path::is_relative(Path))
        return Result;

      PM.insert(PTHEntryKeyVariant(&StatBuf, Path), PTHEntry());
    }

    return Result;
  }
};
} // end anonymous namespace


void clang::CacheTokens(Preprocessor &PP, llvm::raw_fd_ostream* OS) {
  // Get the name of the main file.
  const SourceManager &SrcMgr = PP.getSourceManager();
  const FileEntry *MainFile = SrcMgr.getFileEntryForID(SrcMgr.getMainFileID());
  SmallString<128> MainFilePath(MainFile->getName());

  llvm::sys::fs::make_absolute(MainFilePath);

  // Create the PTHWriter.
  PTHWriter PW(*OS, PP);

  // Install the 'stat' system call listener in the FileManager.
  StatListener *StatCache = new StatListener(PW.getPM());
  PP.getFileManager().addStatCache(StatCache, /*AtBeginning=*/true);

  // Lex through the entire file.  This will populate SourceManager with
  // all of the header information.
  Token Tok;
  PP.EnterMainSourceFile();
  do { PP.Lex(Tok); } while (Tok.isNot(tok::eof));

  // Generate the PTH file.
  PP.getFileManager().removeStatCache(StatCache);
  PW.GeneratePTH(MainFilePath.str());
}

//===----------------------------------------------------------------------===//

namespace {
class PTHIdKey {
public:
  const IdentifierInfo* II;
  uint32_t FileOffset;
};

class PTHIdentifierTableTrait {
public:
  typedef PTHIdKey* key_type;
  typedef key_type  key_type_ref;

  typedef uint32_t  data_type;
  typedef data_type data_type_ref;

  static unsigned ComputeHash(PTHIdKey* key) {
    return llvm::HashString(key->II->getName());
  }

  static std::pair<unsigned,unsigned>
  EmitKeyDataLength(raw_ostream& Out, const PTHIdKey* key, uint32_t) {
    unsigned n = key->II->getLength() + 1;
    ::Emit16(Out, n);
    return std::make_pair(n, sizeof(uint32_t));
  }

  static void EmitKey(raw_ostream& Out, PTHIdKey* key, unsigned n) {
    // Record the location of the key data.  This is used when generating
    // the mapping from persistent IDs to strings.
    key->FileOffset = Out.tell();
    Out.write(key->II->getNameStart(), n);
  }

  static void EmitData(raw_ostream& Out, PTHIdKey*, uint32_t pID,
                       unsigned) {
    ::Emit32(Out, pID);
  }
};
} // end anonymous namespace

/// EmitIdentifierTable - Emits two tables to the PTH file.  The first is
///  a hashtable mapping from identifier strings to persistent IDs.  The second
///  is a straight table mapping from persistent IDs to string data (the
///  keys of the first table).
///
std::pair<Offset,Offset> PTHWriter::EmitIdentifierTable() {
  // Build two maps:
  //  (1) an inverse map from persistent IDs -> (IdentifierInfo*,Offset)
  //  (2) a map from (IdentifierInfo*, Offset)* -> persistent IDs

  // Note that we use 'calloc', so all the bytes are 0.
  PTHIdKey *IIDMap = (PTHIdKey*)calloc(idcount, sizeof(PTHIdKey));

  // Create the hashtable.
  OnDiskChainedHashTableGenerator<PTHIdentifierTableTrait> IIOffMap;

  // Generate mapping from persistent IDs -> IdentifierInfo*.
  for (IDMap::iterator I = IM.begin(), E = IM.end(); I != E; ++I) {
    // Decrement by 1 because we are using a vector for the lookup and
    // 0 is reserved for NULL.
    assert(I->second > 0);
    assert(I->second-1 < idcount);
    unsigned idx = I->second-1;

    // Store the mapping from persistent ID to IdentifierInfo*
    IIDMap[idx].II = I->first;

    // Store the reverse mapping in a hashtable.
    IIOffMap.insert(&IIDMap[idx], I->second);
  }

  // Write out the inverse map first.  This causes the PCIDKey entries to
  // record PTH file offsets for the string data.  This is used to write
  // the second table.
  Offset StringTableOffset = IIOffMap.Emit(Out);

  // Now emit the table mapping from persistent IDs to PTH file offsets.
  Offset IDOff = Out.tell();
  Emit32(idcount);  // Emit the number of identifiers.
  for (unsigned i = 0 ; i < idcount; ++i)
    Emit32(IIDMap[i].FileOffset);

  // Finally, release the inverse map.
  free(IIDMap);

  return std::make_pair(IDOff, StringTableOffset);
}
