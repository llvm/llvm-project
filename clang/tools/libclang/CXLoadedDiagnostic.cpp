/*===-- CXLoadedDiagnostic.cpp - Handling of persisent diags -*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Implements handling of persisent diagnostics.                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "CXLoadedDiagnostic.h"
#include "CXString.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/Optional.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include <assert.h>

using namespace clang;
using namespace clang::cxstring;

//===----------------------------------------------------------------------===//
// Extend CXDiagnosticSetImpl which contains strings for diagnostics.
//===----------------------------------------------------------------------===//

typedef llvm::DenseMap<unsigned, llvm::StringRef> Strings;

namespace {
class CXLoadedDiagnosticSetImpl : public CXDiagnosticSetImpl {
public:
  CXLoadedDiagnosticSetImpl() : CXDiagnosticSetImpl(true), FakeFiles(FO) {}
  virtual ~CXLoadedDiagnosticSetImpl() {}  

  llvm::StringRef makeString(const char *blob, unsigned blobLen);
  
  llvm::BumpPtrAllocator Alloc;
  Strings Categories;
  Strings WarningFlags;
  Strings FileNames;
  
  FileSystemOptions FO;
  FileManager FakeFiles;
  llvm::DenseMap<unsigned, const FileEntry *> Files;
};
}

llvm::StringRef CXLoadedDiagnosticSetImpl::makeString(const char *blob,
                                                      unsigned bloblen) {
  char *mem = Alloc.Allocate<char>(bloblen + 1);
  memcpy(mem, blob, bloblen);
  // Add a null terminator for those clients accessing the buffer
  // like a c-string.
  mem[bloblen] = '\0';
  return llvm::StringRef(mem, bloblen);
}

//===----------------------------------------------------------------------===//
// Cleanup.
//===----------------------------------------------------------------------===//

CXLoadedDiagnostic::~CXLoadedDiagnostic() {}

//===----------------------------------------------------------------------===//
// Public CXLoadedDiagnostic methods.
//===----------------------------------------------------------------------===//

CXDiagnosticSeverity CXLoadedDiagnostic::getSeverity() const {
  // FIXME: possibly refactor with logic in CXStoredDiagnostic.
  switch (severity) {
    case DiagnosticsEngine::Ignored: return CXDiagnostic_Ignored;
    case DiagnosticsEngine::Note:    return CXDiagnostic_Note;
    case DiagnosticsEngine::Warning: return CXDiagnostic_Warning;
    case DiagnosticsEngine::Error:   return CXDiagnostic_Error;
    case DiagnosticsEngine::Fatal:   return CXDiagnostic_Fatal;
  }
  
  llvm_unreachable("Invalid diagnostic level");
}

static CXSourceLocation makeLocation(const CXLoadedDiagnostic::Location *DLoc) {
  // The lowest bit of ptr_data[0] is always set to 1 to indicate this
  // is a persistent diagnostic.
  uintptr_t V = (uintptr_t) DLoc;
  V |= 0x1;
  CXSourceLocation Loc = { {  (void*) V, 0 }, 0 };
  return Loc;
}  

CXSourceLocation CXLoadedDiagnostic::getLocation() const {
  // The lowest bit of ptr_data[0] is always set to 1 to indicate this
  // is a persistent diagnostic.
  return makeLocation(&DiagLoc);
}

CXString CXLoadedDiagnostic::getSpelling() const {
  return cxstring::createCXString(Spelling, false);
}

CXString CXLoadedDiagnostic::getDiagnosticOption(CXString *Disable) const {
  if (DiagOption.empty())
    return createCXString("");

  // FIXME: possibly refactor with logic in CXStoredDiagnostic.
  if (Disable)
    *Disable = createCXString((Twine("-Wno-") + DiagOption).str());
  return createCXString((Twine("-W") + DiagOption).str());
}

unsigned CXLoadedDiagnostic::getCategory() const {
  return category;
}

CXString CXLoadedDiagnostic::getCategoryText() const {
  return cxstring::createCXString(CategoryText);
}

unsigned CXLoadedDiagnostic::getNumRanges() const {
  return Ranges.size();
}

CXSourceRange CXLoadedDiagnostic::getRange(unsigned Range) const {
  assert(Range < Ranges.size());
  return Ranges[Range];
}

unsigned CXLoadedDiagnostic::getNumFixIts() const {
  return FixIts.size();
}

CXString CXLoadedDiagnostic::getFixIt(unsigned FixIt,
                                      CXSourceRange *ReplacementRange) const {
  assert(FixIt < FixIts.size());
  if (ReplacementRange)
    *ReplacementRange = FixIts[FixIt].first;
  return FixIts[FixIt].second;
}

void CXLoadedDiagnostic::decodeLocation(CXSourceLocation location,
                                        CXFile *file,
                                        unsigned int *line,
                                        unsigned int *column,
                                        unsigned int *offset) {
  
  
  // CXSourceLocation consists of the following fields:
  //
  //   void *ptr_data[2];
  //   unsigned int_data;
  //
  // The lowest bit of ptr_data[0] is always set to 1 to indicate this
  // is a persistent diagnostic.
  //
  // For now, do the unoptimized approach and store the data in a side
  // data structure.  We can optimize this case later.
  
  uintptr_t V = (uintptr_t) location.ptr_data[0];
  assert((V & 0x1) == 1);
  V &= ~(uintptr_t)1;
  
  const Location &Loc = *((Location*)V);
  
  if (file)
    *file = Loc.file;  
  if (line)
    *line = Loc.line;
  if (column)
    *column = Loc.column;
  if (offset)
    *offset = Loc.offset;
}

//===----------------------------------------------------------------------===//
// Deserialize diagnostics.
//===----------------------------------------------------------------------===//

enum { MaxSupportedVersion = 1 };
typedef SmallVector<uint64_t, 64> RecordData;
enum LoadResult { Failure = 1, Success = 0 };
enum StreamResult { Read_EndOfStream,
                    Read_BlockBegin,
                    Read_Failure,
                    Read_Record,
                    Read_BlockEnd };

namespace {
class DiagLoader {
  enum CXLoadDiag_Error *error;
  CXString *errorString;
  
  void reportBad(enum CXLoadDiag_Error code, llvm::StringRef err) {
    if (error)
      *error = code;
    if (errorString)
      *errorString = createCXString(err);
  }
  
  void reportInvalidFile(llvm::StringRef err) {
    return reportBad(CXLoadDiag_InvalidFile, err);
  }

  LoadResult readMetaBlock(llvm::BitstreamCursor &Stream);
  
  LoadResult readDiagnosticBlock(llvm::BitstreamCursor &Stream,
                                 CXDiagnosticSetImpl &Diags,
                                 CXLoadedDiagnosticSetImpl &TopDiags);

  StreamResult readToNextRecordOrBlock(llvm::BitstreamCursor &Stream,
                                       llvm::StringRef errorContext,
                                       unsigned &BlockOrRecordID,
                                       const bool atTopLevel = false);
  
  
  LoadResult readString(CXLoadedDiagnosticSetImpl &TopDiags,
                        Strings &strings, llvm::StringRef errorContext,
                        RecordData &Record,
                        const char *BlobStart,
                        unsigned BlobLen,
                        bool allowEmptyString = false);

  LoadResult readString(CXLoadedDiagnosticSetImpl &TopDiags,
                        llvm::StringRef &RetStr,
                        llvm::StringRef errorContext,
                        RecordData &Record,
                        const char *BlobStart,
                        unsigned BlobLen,
                        bool allowEmptyString = false);

  LoadResult readRange(CXLoadedDiagnosticSetImpl &TopDiags,
                       RecordData &Record, unsigned RecStartIdx,
                       CXSourceRange &SR);
  
  LoadResult readLocation(CXLoadedDiagnosticSetImpl &TopDiags,
                          RecordData &Record, unsigned &offset,
                          CXLoadedDiagnostic::Location &Loc);
                       
public:
  DiagLoader(enum CXLoadDiag_Error *e, CXString *es)
    : error(e), errorString(es) {
      if (error)
        *error = CXLoadDiag_None;
      if (errorString)
        *errorString = createCXString("");
    }

  CXDiagnosticSet load(const char *file);
};
}

CXDiagnosticSet DiagLoader::load(const char *file) {
  // Open the diagnostics file.
  std::string ErrStr;
  FileSystemOptions FO;
  FileManager FileMgr(FO);

  OwningPtr<llvm::MemoryBuffer> Buffer;
  Buffer.reset(FileMgr.getBufferForFile(file));

  if (!Buffer) {
    reportBad(CXLoadDiag_CannotLoad, ErrStr);
    return 0;
  }

  llvm::BitstreamReader StreamFile;
  StreamFile.init((const unsigned char *)Buffer->getBufferStart(),
                  (const unsigned char *)Buffer->getBufferEnd());

  llvm::BitstreamCursor Stream;
  Stream.init(StreamFile);

  // Sniff for the signature.
  if (Stream.Read(8) != 'D' ||
      Stream.Read(8) != 'I' ||
      Stream.Read(8) != 'A' ||
      Stream.Read(8) != 'G') {
    reportBad(CXLoadDiag_InvalidFile,
              "Bad header in diagnostics file");
    return 0;
  }

  OwningPtr<CXLoadedDiagnosticSetImpl>
    Diags(new CXLoadedDiagnosticSetImpl());

  while (true) {
    unsigned BlockID = 0;
    StreamResult Res = readToNextRecordOrBlock(Stream, "Top-level", 
                                               BlockID, true);
    switch (Res) {
      case Read_EndOfStream:
        return (CXDiagnosticSet) Diags.take();
      case Read_Failure:
        return 0;
      case Read_Record:
        llvm_unreachable("Top-level does not have records");
      case Read_BlockEnd:
        continue;
      case Read_BlockBegin:
        break;
    }
    
    switch (BlockID) {
      case serialized_diags::BLOCK_META:
        if (readMetaBlock(Stream))
          return 0;
        break;
      case serialized_diags::BLOCK_DIAG:
        if (readDiagnosticBlock(Stream, *Diags.get(), *Diags.get()))
          return 0;
        break;
      default:
        if (!Stream.SkipBlock()) {
          reportInvalidFile("Malformed block at top-level of diagnostics file");
          return 0;
        }
        break;
    }
  }
}

StreamResult DiagLoader::readToNextRecordOrBlock(llvm::BitstreamCursor &Stream,
                                                 llvm::StringRef errorContext,
                                                 unsigned &blockOrRecordID,
                                                 const bool atTopLevel) {
  
  blockOrRecordID = 0;

  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();

    // Handle the top-level specially.
    if (atTopLevel) {
      if (Code == llvm::bitc::ENTER_SUBBLOCK) {
        unsigned BlockID = Stream.ReadSubBlockID();
        if (BlockID == llvm::bitc::BLOCKINFO_BLOCK_ID) {
          if (Stream.ReadBlockInfoBlock()) {
            reportInvalidFile("Malformed BlockInfoBlock in diagnostics file");
            return Read_Failure;
          }
          continue;
        }
        blockOrRecordID = BlockID;
        return Read_BlockBegin;
      }
      reportInvalidFile("Only blocks can appear at the top of a "
                        "diagnostic file");
      return Read_Failure;
    }
    
    switch ((llvm::bitc::FixedAbbrevIDs)Code) {
      case llvm::bitc::ENTER_SUBBLOCK:
        blockOrRecordID = Stream.ReadSubBlockID();
        return Read_BlockBegin;
      
      case llvm::bitc::END_BLOCK:
        if (Stream.ReadBlockEnd()) {
          reportInvalidFile("Cannot read end of block");
          return Read_Failure;
        }
        return Read_BlockEnd;
        
      case llvm::bitc::DEFINE_ABBREV:
        Stream.ReadAbbrevRecord();
        continue;
        
      case llvm::bitc::UNABBREV_RECORD:
        reportInvalidFile("Diagnostics file should have no unabbreviated "
                          "records");
        return Read_Failure;
      
      default:
        // We found a record.
        blockOrRecordID = Code;
        return Read_Record;
    }
  }
  
  if (atTopLevel)
    return Read_EndOfStream;
  
  reportInvalidFile(Twine("Premature end of diagnostics file within ").str() + 
                    errorContext.str());
  return Read_Failure;
}

LoadResult DiagLoader::readMetaBlock(llvm::BitstreamCursor &Stream) {
  if (Stream.EnterSubBlock(clang::serialized_diags::BLOCK_META)) {
    reportInvalidFile("Malformed metadata block");
    return Failure;
  }

  bool versionChecked = false;
  
  while (true) {
    unsigned blockOrCode = 0;
    StreamResult Res = readToNextRecordOrBlock(Stream, "Metadata Block",
                                               blockOrCode);
    
    switch(Res) {
      case Read_EndOfStream:
        llvm_unreachable("EndOfStream handled by readToNextRecordOrBlock");
      case Read_Failure:
        return Failure;
      case Read_Record:
        break;
      case Read_BlockBegin:
        if (Stream.SkipBlock()) {
          reportInvalidFile("Malformed metadata block");
          return Failure;
        }
      case Read_BlockEnd:
        if (!versionChecked) {
          reportInvalidFile("Diagnostics file does not contain version"
                            " information");
          return Failure;
        }
        return Success;
    }
    
    RecordData Record;
    const char *Blob;
    unsigned BlobLen;
    unsigned recordID = Stream.ReadRecord(blockOrCode, Record, &Blob, &BlobLen);
    
    if (recordID == serialized_diags::RECORD_VERSION) {
      if (Record.size() < 1) {
        reportInvalidFile("malformed VERSION identifier in diagnostics file");
        return Failure;
      }
      if (Record[0] > MaxSupportedVersion) {
        reportInvalidFile("diagnosics file is a newer version than the one "
                          "supported");
        return Failure;
      }
      versionChecked = true;
    }
  }
}

LoadResult DiagLoader::readString(CXLoadedDiagnosticSetImpl &TopDiags,
                                  llvm::StringRef &RetStr,
                                  llvm::StringRef errorContext,
                                  RecordData &Record,
                                  const char *BlobStart,
                                  unsigned BlobLen,
                                  bool allowEmptyString) {
  
  // Basic buffer overflow check.
  if (BlobLen > 65536) {
    reportInvalidFile(std::string("Out-of-bounds string in ") +
                      std::string(errorContext));
    return Failure;
  }

  if (allowEmptyString && Record.size() >= 1 && BlobLen == 0) {
    RetStr = "";
    return Success;
  }
  
  if (Record.size() < 1 || BlobLen == 0) {
    reportInvalidFile(std::string("Corrupted ") + std::string(errorContext)
                      + std::string(" entry"));
    return Failure;
  }
  
  RetStr = TopDiags.makeString(BlobStart, BlobLen);
  return Success;
}

LoadResult DiagLoader::readString(CXLoadedDiagnosticSetImpl &TopDiags,
                                  Strings &strings,
                                  llvm::StringRef errorContext,
                                  RecordData &Record,
                                  const char *BlobStart,
                                  unsigned BlobLen,
                                  bool allowEmptyString) {
  llvm::StringRef RetStr;
  if (readString(TopDiags, RetStr, errorContext, Record, BlobStart, BlobLen,
                 allowEmptyString))
    return Failure;
  strings[Record[0]] = RetStr;
  return Success;
}

LoadResult DiagLoader::readLocation(CXLoadedDiagnosticSetImpl &TopDiags,
                                    RecordData &Record, unsigned &offset,
                                    CXLoadedDiagnostic::Location &Loc) {
  if (Record.size() < offset + 3) {
    reportInvalidFile("Corrupted source location");
    return Failure;
  }
  
  unsigned fileID = Record[offset++];
  if (fileID == 0) {
    // Sentinel value.
    Loc.file = 0;
    Loc.line = 0;
    Loc.column = 0;
    Loc.offset = 0;
    return Success;
  }

  const FileEntry *FE = TopDiags.Files[fileID];
  if (!FE) {
    reportInvalidFile("Corrupted file entry in source location");
    return Failure;
  }
  Loc.file = (void*) FE;
  Loc.line = Record[offset++];
  Loc.column = Record[offset++];
  Loc.offset = Record[offset++];
  return Success;
}

LoadResult DiagLoader::readRange(CXLoadedDiagnosticSetImpl &TopDiags,
                                 RecordData &Record,
                                 unsigned int RecStartIdx,
                                 CXSourceRange &SR) {
  CXLoadedDiagnostic::Location *Start, *End;
  Start = TopDiags.Alloc.Allocate<CXLoadedDiagnostic::Location>();
  End = TopDiags.Alloc.Allocate<CXLoadedDiagnostic::Location>();
  
  if (readLocation(TopDiags, Record, RecStartIdx, *Start))
    return Failure;
  if (readLocation(TopDiags, Record, RecStartIdx, *End))
    return Failure;
  
  CXSourceLocation startLoc = makeLocation(Start);
  CXSourceLocation endLoc = makeLocation(End);
  SR = clang_getRange(startLoc, endLoc);
  return Success;  
}

LoadResult DiagLoader::readDiagnosticBlock(llvm::BitstreamCursor &Stream,
                                           CXDiagnosticSetImpl &Diags,
                                           CXLoadedDiagnosticSetImpl &TopDiags){

  if (Stream.EnterSubBlock(clang::serialized_diags::BLOCK_DIAG)) {
    reportInvalidFile("malformed diagnostic block");
    return Failure;
  }
  
  OwningPtr<CXLoadedDiagnostic> D(new CXLoadedDiagnostic());
  RecordData Record;
  
  while (true) {
    unsigned blockOrCode = 0;
    StreamResult Res = readToNextRecordOrBlock(Stream, "Diagnostic Block",
                                               blockOrCode);
    switch (Res) {
      case Read_EndOfStream:
        llvm_unreachable("EndOfStream handled in readToNextRecordOrBlock");
      case Read_Failure:
        return Failure;
      case Read_BlockBegin: {
        // The only blocks we care about are subdiagnostics.
        if (blockOrCode != serialized_diags::BLOCK_DIAG) {
          if (!Stream.SkipBlock()) {
            reportInvalidFile("Invalid subblock in Diagnostics block");
            return Failure;
          }
        } else if (readDiagnosticBlock(Stream, D->getChildDiagnostics(),
                                       TopDiags)) {
          return Failure;
        }

        continue;
      }
      case Read_BlockEnd:
        Diags.appendDiagnostic(D.take());        
        return Success;
      case Read_Record:
        break;
    }
    
    // Read the record.
    Record.clear();
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    unsigned recID = Stream.ReadRecord(blockOrCode, Record,
                                       BlobStart, BlobLen);
    
    if (recID < serialized_diags::RECORD_FIRST ||
        recID > serialized_diags::RECORD_LAST)
      continue;
    
    switch ((serialized_diags::RecordIDs)recID) {  
      case serialized_diags::RECORD_VERSION:
        continue;
      case serialized_diags::RECORD_CATEGORY:
        if (readString(TopDiags, TopDiags.Categories, "category", Record,
                       BlobStart, BlobLen,
                       /* allowEmptyString */ true))
          return Failure;
        continue;
      
      case serialized_diags::RECORD_DIAG_FLAG:
        if (readString(TopDiags, TopDiags.WarningFlags, "warning flag", Record,
                       BlobStart, BlobLen))
          return Failure;
        continue;
        
      case serialized_diags::RECORD_FILENAME: {
        if (readString(TopDiags, TopDiags.FileNames, "filename", Record,
                       BlobStart, BlobLen))
          return Failure;

        if (Record.size() < 3) {
          reportInvalidFile("Invalid file entry");
          return Failure;
        }
        
        const FileEntry *FE =
          TopDiags.FakeFiles.getVirtualFile(TopDiags.FileNames[Record[0]],
                                            /* size */ Record[1],
                                            /* time */ Record[2]);
        
        TopDiags.Files[Record[0]] = FE;
        continue;
      }

      case serialized_diags::RECORD_SOURCE_RANGE: {
        CXSourceRange SR;
        if (readRange(TopDiags, Record, 0, SR))
          return Failure;
        D->Ranges.push_back(SR);
        continue;
      }
      
      case serialized_diags::RECORD_FIXIT: {
        CXSourceRange SR;
        if (readRange(TopDiags, Record, 0, SR))
          return Failure;
        llvm::StringRef RetStr;
        if (readString(TopDiags, RetStr, "FIXIT", Record, BlobStart, BlobLen,
                       /* allowEmptyString */ true))
          return Failure;
        D->FixIts.push_back(std::make_pair(SR, createCXString(RetStr, false)));
        continue;
      }
        
      case serialized_diags::RECORD_DIAG: {
        D->severity = Record[0];
        unsigned offset = 1;
        if (readLocation(TopDiags, Record, offset, D->DiagLoc))
          return Failure;
        D->category = Record[offset++];
        unsigned diagFlag = Record[offset++];
        D->DiagOption = diagFlag ? TopDiags.WarningFlags[diagFlag] : "";
        D->CategoryText = D->category ? TopDiags.Categories[D->category] : "";
        D->Spelling = TopDiags.makeString(BlobStart, BlobLen);
        continue;
      }
    }
  }
}

extern "C" {
CXDiagnosticSet clang_loadDiagnostics(const char *file,
                                      enum CXLoadDiag_Error *error,
                                      CXString *errorString) {
  DiagLoader L(error, errorString);
  return L.load(file);
}
} // end extern 'C'.
