//===--  BitcodeWriter.h - ClangDoc Bitcode Writer --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a writer for serializing the clang-doc internal
// representation to LLVM bitcode. The writer takes in a stream and emits the
// generated bitcode to that stream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_BITCODEWRITER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_BITCODEWRITER_H

#include "Representation.h"
#include "clang/AST/AST.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include <initializer_list>
#include <vector>

namespace clang {
namespace doc {

// Current version number of clang-doc bitcode.
// Should be bumped when removing or changing BlockIds, RecordIds, or
// BitCodeConstants, though they can be added without breaking it.
static const unsigned VersionNumber = 2;

struct BitCodeConstants {
  static constexpr unsigned RecordSize = 32U;
  static constexpr unsigned SignatureBitSize = 8U;
  static constexpr unsigned SubblockIDSize = 4U;
  static constexpr unsigned BoolSize = 1U;
  static constexpr unsigned IntSize = 16U;
  static constexpr unsigned StringLengthSize = 16U;
  static constexpr unsigned FilenameLengthSize = 16U;
  static constexpr unsigned LineNumberSize = 16U;
  static constexpr unsigned ReferenceTypeSize = 8U;
  static constexpr unsigned USRLengthSize = 6U;
  static constexpr unsigned USRBitLengthSize = 8U;
  static constexpr char Signature[4] = {'D', 'O', 'C', 'S'};
  static constexpr int USRHashSize = 20;
};

// New Ids need to be added to both the enum here and the relevant IdNameMap in
// the implementation file.
enum BlockId {
  BI_VERSION_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,
  BI_NAMESPACE_BLOCK_ID,
  BI_ENUM_BLOCK_ID,
  BI_TYPE_BLOCK_ID,
  BI_FIELD_TYPE_BLOCK_ID,
  BI_MEMBER_TYPE_BLOCK_ID,
  BI_RECORD_BLOCK_ID,
  BI_FUNCTION_BLOCK_ID,
  BI_COMMENT_BLOCK_ID,
  BI_REFERENCE_BLOCK_ID,
  BI_LAST,
  BI_FIRST = BI_VERSION_BLOCK_ID
};

// New Ids need to be added to the enum here, and to the relevant IdNameMap and
// initialization list in the implementation file.
#define INFORECORDS(X) X##_USR, X##_NAME

enum RecordId {
  VERSION = 1,
  INFORECORDS(FUNCTION),
  FUNCTION_DEFLOCATION,
  FUNCTION_LOCATION,
  FUNCTION_ACCESS,
  FUNCTION_IS_METHOD,
  COMMENT_KIND,
  COMMENT_TEXT,
  COMMENT_NAME,
  COMMENT_DIRECTION,
  COMMENT_PARAMNAME,
  COMMENT_CLOSENAME,
  COMMENT_SELFCLOSING,
  COMMENT_EXPLICIT,
  COMMENT_ATTRKEY,
  COMMENT_ATTRVAL,
  COMMENT_ARG,
  FIELD_TYPE_NAME,
  MEMBER_TYPE_NAME,
  MEMBER_TYPE_ACCESS,
  INFORECORDS(NAMESPACE),
  INFORECORDS(ENUM),
  ENUM_DEFLOCATION,
  ENUM_LOCATION,
  ENUM_MEMBER,
  ENUM_SCOPED,
  INFORECORDS(RECORD),
  RECORD_DEFLOCATION,
  RECORD_LOCATION,
  RECORD_TAG_TYPE,
  REFERENCE_USR,
  REFERENCE_NAME,
  REFERENCE_TYPE,
  REFERENCE_FIELD,
  RI_LAST,
  RI_FIRST = VERSION
};

static constexpr unsigned BlockIdCount = BI_LAST - BI_FIRST;
static constexpr unsigned RecordIdCount = RI_LAST - RI_FIRST;

#undef INFORECORDS

// Identifiers for differentiating between subblocks
enum class FieldId { F_default, F_namespace, F_parent, F_vparent, F_type };

class ClangDocBitcodeWriter {
public:
  ClangDocBitcodeWriter(llvm::BitstreamWriter &Stream) : Stream(Stream) {
    emitHeader();
    emitBlockInfoBlock();
    emitVersionBlock();
  }

  // Write a specific info to a bitcode stream.
  bool dispatchInfoForWrite(Info *I);

  // Block emission of different info types.
  void emitBlock(const NamespaceInfo &I);
  void emitBlock(const RecordInfo &I);
  void emitBlock(const FunctionInfo &I);
  void emitBlock(const EnumInfo &I);
  void emitBlock(const TypeInfo &B);
  void emitBlock(const FieldTypeInfo &B);
  void emitBlock(const MemberTypeInfo &B);
  void emitBlock(const CommentInfo &B);
  void emitBlock(const Reference &B, FieldId F);

private:
  class AbbreviationMap {
    llvm::DenseMap<unsigned, unsigned> Abbrevs;

  public:
    AbbreviationMap() : Abbrevs(RecordIdCount) {}

    void add(RecordId RID, unsigned AbbrevID);
    unsigned get(RecordId RID) const;
  };

  class StreamSubBlockGuard {
    llvm::BitstreamWriter &Stream;

  public:
    StreamSubBlockGuard(llvm::BitstreamWriter &Stream_, BlockId ID)
        : Stream(Stream_) {
      // NOTE: SubBlockIDSize could theoretically be calculated on the fly,
      // based on the initialization list of records in each block.
      Stream.EnterSubblock(ID, BitCodeConstants::SubblockIDSize);
    }

    StreamSubBlockGuard() = default;
    StreamSubBlockGuard(const StreamSubBlockGuard &) = delete;
    StreamSubBlockGuard &operator=(const StreamSubBlockGuard &) = delete;

    ~StreamSubBlockGuard() { Stream.ExitBlock(); }
  };

  // Emission of validation and overview blocks.
  void emitHeader();
  void emitVersionBlock();
  void emitRecordID(RecordId ID);
  void emitBlockID(BlockId ID);
  void emitBlockInfoBlock();
  void emitBlockInfo(BlockId BID, const std::vector<RecordId> &RIDs);

  // Emission of individual record types.
  void emitRecord(StringRef Str, RecordId ID);
  void emitRecord(const SymbolID &Str, RecordId ID);
  void emitRecord(const Location &Loc, RecordId ID);
  void emitRecord(const Reference &Ref, RecordId ID);
  void emitRecord(bool Value, RecordId ID);
  void emitRecord(int Value, RecordId ID);
  void emitRecord(unsigned Value, RecordId ID);
  bool prepRecordData(RecordId ID, bool ShouldEmit = true);

  // Emission of appropriate abbreviation type.
  void emitAbbrev(RecordId ID, BlockId Block);

  // Static size is the maximum length of the block/record names we're pushing
  // to this + 1. Longest is currently `MemberTypeBlock` at 15 chars.
  SmallVector<uint32_t, BitCodeConstants::RecordSize> Record;
  llvm::BitstreamWriter &Stream;
  AbbreviationMap Abbrevs;
};

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_BITCODEWRITER_H
