//===--- IndexDataStoreUtils.h - Functions/constants for the data store ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INDEX_INDEXDATASTOREUTILS_H
#define LLVM_CLANG_LIB_INDEX_INDEXDATASTOREUTILS_H

#include "llvm/Bitcode/BitCodes.h"
#include "clang/Basic/LLVM.h"

namespace llvm {
  class BitstreamWriter;
}

namespace clang {
namespace index {
namespace store {

static const unsigned STORE_FORMAT_VERSION = 5;

void appendUnitSubDir(SmallVectorImpl<char> &StorePathBuf);
void appendInteriorUnitPath(StringRef UnitName,
                            SmallVectorImpl<char> &PathBuf);
void appendRecordSubDir(SmallVectorImpl<char> &StorePathBuf);
void appendInteriorRecordPath(StringRef RecordName,
                              SmallVectorImpl<char> &PathBuf);

enum RecordBitRecord {
  REC_VERSION         = 0,
  REC_DECLINFO        = 1,
  REC_DECLOFFSETS     = 2,
  REC_DECLOCCURRENCE  = 3,
};

enum RecordBitBlock {
  REC_VERSION_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,
  REC_DECLS_BLOCK_ID,
  REC_DECLOFFSETS_BLOCK_ID,
  REC_DECLOCCURRENCES_BLOCK_ID,
};

enum UnitBitRecord {
  UNIT_VERSION        = 0,
  UNIT_INFO           = 1,
  UNIT_DEPENDENCY     = 2,
  UNIT_INCLUDE        = 3,
  UNIT_PATH           = 4,
  UNIT_PATH_BUFFER    = 5,
  UNIT_MODULE         = 6,
  UNIT_MODULE_BUFFER  = 7,
};

enum UnitBitBlock {
  UNIT_VERSION_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,
  UNIT_INFO_BLOCK_ID,
  UNIT_DEPENDENCIES_BLOCK_ID,
  UNIT_INCLUDES_BLOCK_ID,
  UNIT_PATHS_BLOCK_ID,
  UNIT_MODULES_BLOCK_ID,
};

enum UnitDependencyKind {
  UNIT_DEPEND_KIND_FILE = 0,
  UNIT_DEPEND_KIND_RECORD = 1,
  UNIT_DEPEND_KIND_UNIT = 2,
};
static const unsigned UnitDependencyKindBitNum = 2;

enum UnitFilePathPrefixKind {
  UNIT_PATH_PREFIX_NONE = 0,
  UNIT_PATH_PREFIX_WORKDIR = 1,
  UNIT_PATH_PREFIX_SYSROOT = 2,
};
static const unsigned UnitFilePathPrefixKindBitNum = 2;

typedef SmallVector<uint64_t, 64> RecordData;
typedef SmallVectorImpl<uint64_t> RecordDataImpl;

struct BitPathComponent {
  size_t Offset = 0;
  size_t Size = 0;
  BitPathComponent(size_t Offset, size_t Size) : Offset(Offset), Size(Size) {}
  BitPathComponent() = default;
};

struct DirBitPath {
  UnitFilePathPrefixKind PrefixKind = UNIT_PATH_PREFIX_NONE;
  BitPathComponent Dir;
  DirBitPath(UnitFilePathPrefixKind Kind,
             BitPathComponent Dir) : PrefixKind(Kind), Dir(Dir) {}
  DirBitPath() = default;
};

struct FileBitPath : DirBitPath {
  BitPathComponent Filename;
  FileBitPath(UnitFilePathPrefixKind Kind, BitPathComponent Dir,
              BitPathComponent Filename) : DirBitPath(Kind, Dir), Filename(Filename) {}
  FileBitPath() = default;
};

void emitBlockID(unsigned ID, const char *Name,
                 llvm::BitstreamWriter &Stream, RecordDataImpl &Record);

void emitRecordID(unsigned ID, const char *Name,
                  llvm::BitstreamWriter &Stream, RecordDataImpl &Record);

} // end namespace store
} // end namespace index
} // end namespace clang

#endif
