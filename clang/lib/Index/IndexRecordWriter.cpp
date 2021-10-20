//===--- IndexRecordWriter.cpp - Index record serialization ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexRecordWriter.h"
#include "IndexDataStoreUtils.h"
#include "indexstore/indexstore.h"
#include "clang/Index/IndexDataStoreSymbolUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::index;
using namespace clang::index::store;
using namespace llvm;

using writer::OpaqueDecl;

namespace {
struct DeclInfo {
  OpaqueDecl D;
  SymbolRoleSet Roles;
  SymbolRoleSet RelatedRoles;
};

struct OccurrenceInfo {
  unsigned DeclID;
  OpaqueDecl D;
  SymbolRoleSet Roles;
  unsigned Line;
  unsigned Column;
  SmallVector<std::pair<writer::SymbolRelation, unsigned>, 4> Related;
};

struct RecordState {
  std::string RecordPath;
  SmallString<512> Buffer;
  BitstreamWriter Stream;

  DenseMap<OpaqueDecl, unsigned> IndexForDecl;
  std::vector<DeclInfo> Decls;
  std::vector<OccurrenceInfo> Occurrences;

  RecordState(std::string &&RecordPath)
      : RecordPath(std::move(RecordPath)), Stream(Buffer) {}
};
} // end anonymous namespace

static void writeBlockInfo(BitstreamWriter &Stream) {
  RecordData Record;

  Stream.EnterBlockInfoBlock();
#define BLOCK(X) emitBlockID(X ## _ID, #X, Stream, Record)
#define RECORD(X) emitRecordID(X, #X, Stream, Record)

  BLOCK(REC_VERSION_BLOCK);
  RECORD(REC_VERSION);

  BLOCK(REC_DECLS_BLOCK);
  RECORD(REC_DECLINFO);

  BLOCK(REC_DECLOFFSETS_BLOCK);
  RECORD(REC_DECLOFFSETS);

  BLOCK(REC_DECLOCCURRENCES_BLOCK);
  RECORD(REC_DECLOCCURRENCE);

#undef RECORD
#undef BLOCK
  Stream.ExitBlock();
}

static void writeVersionInfo(BitstreamWriter &Stream) {
  using namespace llvm::sys;

  Stream.EnterSubblock(REC_VERSION_BLOCK_ID, 3);

  auto Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(REC_VERSION));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Store format version
  unsigned AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  RecordData Record;
  Record.push_back(REC_VERSION);
  Record.push_back(STORE_FORMAT_VERSION);
  Stream.EmitRecordWithAbbrev(AbbrevCode, Record);

  Stream.ExitBlock();
}

template <typename T, typename Allocator>
static StringRef data(const std::vector<T, Allocator> &v) {
  if (v.empty())
    return StringRef();
  return StringRef(reinterpret_cast<const char *>(&v[0]), sizeof(T) * v.size());
}

template <typename T> static StringRef data(const SmallVectorImpl<T> &v) {
  return StringRef(reinterpret_cast<const char *>(v.data()),
                   sizeof(T) * v.size());
}

static void writeDecls(BitstreamWriter &Stream, ArrayRef<DeclInfo> Decls,
                       ArrayRef<OccurrenceInfo> Occurrences,
                       writer::SymbolWriterCallback GetSymbolForDecl) {
  SmallVector<uint32_t, 32> DeclOffsets;
  DeclOffsets.reserve(Decls.size());

  //===--------------------------------------------------------------------===//
  // DECLS_BLOCK_ID
  //===--------------------------------------------------------------------===//

  Stream.EnterSubblock(REC_DECLS_BLOCK_ID, 3);

  auto Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(REC_DECLINFO));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 5)); // Kind
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 5)); // SubKind
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 5)); // Language
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 9)); // Properties
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, SymbolRoleBitNum)); // Roles
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, SymbolRoleBitNum)); // Related Roles
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Length of name in block
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Length of USR in block
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name + USR + CodeGen symbol name
  unsigned AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

#ifndef NDEBUG
  StringSet<> USRSet;
  bool enableValidation = getenv("CLANG_INDEX_VALIDATION_CHECKS") != nullptr;
#endif

  RecordData Record;
  llvm::SmallString<256> Blob;
  llvm::SmallString<256> Scratch;
  for (auto &Info : Decls) {
    DeclOffsets.push_back(Stream.GetCurrentBitNo());
    Blob.clear();
    Scratch.clear();

    writer::Symbol SymInfo = GetSymbolForDecl(Info.D, Scratch);
    assert(SymInfo.SymInfo.Kind != SymbolKind::Unknown);
    assert(!SymInfo.USR.empty() && "Recorded decl without USR!");

    Blob += SymInfo.Name;
    Blob += SymInfo.USR;
    Blob += SymInfo.CodeGenName;

#ifndef NDEBUG
    if (enableValidation) {
      bool IsNew = USRSet.insert(SymInfo.USR).second;
      if (!IsNew) {
        llvm::errs() << "Index: Duplicate USR! " << SymInfo.USR << "\n";
        // FIXME: print more information so it's easier to find the declaration.
      }
    }
#endif

    Record.clear();
    Record.push_back(REC_DECLINFO);
    Record.push_back(getIndexStoreKind(SymInfo.SymInfo.Kind));
    Record.push_back(getIndexStoreSubKind(SymInfo.SymInfo.SubKind));
    Record.push_back(getIndexStoreLang(SymInfo.SymInfo.Lang));
    Record.push_back(getIndexStoreProperties(SymInfo.SymInfo.Properties));
    Record.push_back(getIndexStoreRoles(Info.Roles));
    Record.push_back(getIndexStoreRoles(Info.RelatedRoles));
    Record.push_back(SymInfo.Name.size());
    Record.push_back(SymInfo.USR.size());
    Stream.EmitRecordWithBlob(AbbrevCode, Record, Blob);
  }

  Stream.ExitBlock();

  //===--------------------------------------------------------------------===//
  // DECLOFFSETS_BLOCK_ID
  //===--------------------------------------------------------------------===//

  Stream.EnterSubblock(REC_DECLOFFSETS_BLOCK_ID, 3);

  Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(REC_DECLOFFSETS));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // Number of Decls
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Offsets array
  AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  Record.clear();
  Record.push_back(REC_DECLOFFSETS);
  Record.push_back(DeclOffsets.size());
  Stream.EmitRecordWithBlob(AbbrevCode, Record, data(DeclOffsets));

  Stream.ExitBlock();

  //===--------------------------------------------------------------------===//
  // DECLOCCURRENCES_BLOCK_ID
  //===--------------------------------------------------------------------===//

  Stream.EnterSubblock(REC_DECLOCCURRENCES_BLOCK_ID, 3);

  Abbrev = std::make_shared<BitCodeAbbrev>();
  Abbrev->Add(BitCodeAbbrevOp(REC_DECLOCCURRENCE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Decl ID
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, SymbolRoleBitNum)); // Roles
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 12)); // Line
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Column
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 4)); // Num related
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array)); // Related Roles/IDs
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 16)); // Roles or ID
  AbbrevCode = Stream.EmitAbbrev(std::move(Abbrev));

  for (auto &Occur : Occurrences) {
    Record.clear();
    Record.push_back(REC_DECLOCCURRENCE);
    Record.push_back(Occur.DeclID);
    Record.push_back(getIndexStoreRoles(Occur.Roles));
    Record.push_back(Occur.Line);
    Record.push_back(Occur.Column);
    Record.push_back(Occur.Related.size());
    for (auto &Rel : Occur.Related) {
      Record.push_back(getIndexStoreRoles(Rel.first.Roles));
      Record.push_back(Rel.second);
    }
    Stream.EmitRecordWithAbbrev(AbbrevCode, Record);
  }
  Stream.ExitBlock();
}

IndexRecordWriter::IndexRecordWriter(StringRef IndexPath)
    : RecordsPath(IndexPath) {
  store::appendRecordSubDir(RecordsPath);
}

IndexRecordWriter::Result
IndexRecordWriter::beginRecord(StringRef Filename, hash_code RecordHash,
                               std::string &Error, std::string *OutRecordFile) {
  using namespace llvm::sys;
  assert(!Record && "called beginRecord before calling endRecord on previous");

  std::string RecordName;
  {
    llvm::raw_string_ostream RN(RecordName);
    RN << path::filename(Filename);
    RN << "-" << toString(APInt(64, RecordHash), 36, /*Signed=*/false);
  }
  SmallString<256> RecordPath = RecordsPath.str();
  appendInteriorRecordPath(RecordName, RecordPath);

  if (OutRecordFile)
    *OutRecordFile = RecordName;

  if (std::error_code EC =
          fs::access(RecordPath.c_str(), fs::AccessMode::Exist)) {
    if (EC != errc::no_such_file_or_directory) {
      llvm::raw_string_ostream Err(Error);
      Err << "could not access record '" << RecordPath
          << "': " << EC.message();
      return Result::Failure;
    }
  } else {
    return Result::AlreadyExists;
  }

  // Write the record header.
  auto *State = new RecordState(std::string(RecordPath.str()));
  Record = State;
  llvm::BitstreamWriter &Stream = State->Stream;
  Stream.Emit('I', 8);
  Stream.Emit('D', 8);
  Stream.Emit('X', 8);
  Stream.Emit('R', 8);

  writeBlockInfo(Stream);
  writeVersionInfo(Stream);

  return Result::Success;
}

IndexRecordWriter::Result
IndexRecordWriter::endRecord(std::string &Error,
                             writer::SymbolWriterCallback GetSymbolForDecl) {
  assert(Record && "called endRecord without calling beginRecord");
  auto &State = *static_cast<RecordState *>(Record);
  Record = nullptr;
  struct ScopedDelete {
    RecordState *S;
    ScopedDelete(RecordState *S) : S(S) {}
    ~ScopedDelete() { delete S; }
  } Deleter(&State);

  if (!State.Decls.empty()) {
    writeDecls(State.Stream, State.Decls, State.Occurrences, GetSymbolForDecl);
  }

  if (std::error_code EC = sys::fs::create_directory(sys::path::parent_path(State.RecordPath))) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to create directory '" << sys::path::parent_path(State.RecordPath) << "': " << EC.message();
    return Result::Failure;
  }

  // Create a unique file to write to so that we can move the result into place
  // atomically. If this process crashes we don't want to interfere with any
  // other concurrent processes.
  SmallString<128> TempPath(State.RecordPath);
  TempPath += "-temp-%%%%%%%%";
  int TempFD;
  if (sys::fs::createUniqueFile(TempPath.str(), TempFD, TempPath)) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to create temporary file: " << TempPath;
    return Result::Failure;
  }

  raw_fd_ostream OS(TempFD, /*shouldClose=*/true);
  OS.write(State.Buffer.data(), State.Buffer.size());
  OS.close();

  if (OS.has_error()) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to write '" << TempPath << "': " << OS.error().message();
    OS.clear_error();
    return Result::Failure;
  }

  // Atomically move the unique file into place.
  if (std::error_code EC =
          sys::fs::rename(TempPath.c_str(), State.RecordPath.c_str())) {
    llvm::raw_string_ostream Err(Error);
    Err << "failed to rename '" << TempPath << "' to '" << State.RecordPath << "': " << EC.message();
    return Result::Failure;
  }

  return Result::Success;
}

void IndexRecordWriter::addOccurrence(
    OpaqueDecl D, SymbolRoleSet Roles, unsigned Line, unsigned Column,
    ArrayRef<writer::SymbolRelation> Related) {
  assert(Record && "called addOccurrence without calling beginRecord");
  auto &State = *static_cast<RecordState *>(Record);

  auto insertDecl = [&](OpaqueDecl D, SymbolRoleSet Roles,
                        SymbolRoleSet RelatedRoles) -> unsigned {
    auto Insert =
        State.IndexForDecl.insert(std::make_pair(D, State.Decls.size()));
    unsigned Index = Insert.first->second;

    if (Insert.second) {
      State.Decls.push_back(DeclInfo{D, Roles, RelatedRoles});
    } else {
      State.Decls[Index].Roles |= Roles;
      State.Decls[Index].RelatedRoles |= RelatedRoles;
    }
    return Index + 1;
  };

  unsigned DeclID = insertDecl(D, Roles, SymbolRoleSet());

  decltype(OccurrenceInfo::Related) RelatedDecls;
  RelatedDecls.reserve(Related.size());
  for (auto &Rel : Related) {
    unsigned ID = insertDecl(Rel.RelatedSymbol, SymbolRoleSet(), Rel.Roles);
    RelatedDecls.emplace_back(Rel, ID);
  }

  State.Occurrences.push_back(
      OccurrenceInfo{DeclID, D, Roles, Line, Column, std::move(RelatedDecls)});
}
