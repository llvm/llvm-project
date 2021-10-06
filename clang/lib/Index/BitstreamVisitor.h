//===--- BitstreamVisitor.h - Helper for reading a bitstream --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INDEX_BITSTREAMVISITOR_H
#define LLVM_CLANG_LIB_INDEX_BITSTREAMVISITOR_H

#include "llvm/Bitstream/BitstreamReader.h"
#include "clang/Basic/LLVM.h"
#include "clang/Serialization/ASTReader.h"
#include <string>

namespace clang {
namespace index {
namespace store {

/// Helper class that saves the current stream position and
/// then restores it when destroyed.
struct SavedStreamPosition {
  explicit SavedStreamPosition(llvm::BitstreamCursor &Cursor)
    : Cursor(Cursor), Offset(Cursor.GetCurrentBitNo()) { }

  ~SavedStreamPosition() {
    if (llvm::Error Err = Cursor.JumpToBit(Offset))
      llvm::report_fatal_error(Twine("SavedStreamPosition failed jumping: ") +
                               toString(std::move(Err)));
  }

private:
  llvm::BitstreamCursor &Cursor;
  uint64_t Offset;
};

enum class StreamVisit {
  Continue,
  Skip,
  Abort
};

template <typename ImplClass>
class BitstreamVisitor {
  SmallVector<unsigned, 4> BlockStack;

protected:
  llvm::BitstreamCursor &Stream;
  Optional<llvm::BitstreamBlockInfo> BlockInfo;
  std::string *Error;

public:
  BitstreamVisitor(llvm::BitstreamCursor &Stream)
    : Stream(Stream) {}

  StreamVisit visitBlock(unsigned ID) {
    return StreamVisit::Continue;
  }

  bool visit(std::string &Error) {
    this->Error = &Error;

    ASTReader::RecordData Record;
    while (1) {
      Expected<llvm::BitstreamEntry> MaybeEntry = Stream.advance(llvm::BitstreamCursor::AF_DontPopBlockAtEnd);
      if (!MaybeEntry) {
        Error = toString(MaybeEntry.takeError());
        return false;
      }
      llvm::BitstreamEntry Entry = MaybeEntry.get();

      switch (Entry.Kind) {
      case llvm::BitstreamEntry::Error:
        Error = "malformed serialization";
        return false;

      case llvm::BitstreamEntry::EndBlock:
        if (BlockStack.empty())
          return true;
        BlockStack.pop_back();
        if (Stream.ReadBlockEnd()) {
          Error = "malformed serialization";
          return false;
        }
        if (Stream.AtEndOfStream())
          return true;
        break;

      case llvm::BitstreamEntry::SubBlock: {
        if (Entry.ID == llvm::bitc::BLOCKINFO_BLOCK_ID) {
          Expected<Optional<llvm::BitstreamBlockInfo>> MaybeBlockInfo = Stream.ReadBlockInfoBlock();
          if (!MaybeBlockInfo) {
            Error = toString(MaybeBlockInfo.takeError());
            return false;
          }
          BlockInfo = MaybeBlockInfo.get();
          if (!BlockInfo) {
            Error = "malformed BlockInfoBlock";
            return false;
          }
          Stream.setBlockInfo(&*BlockInfo);
          break;
        }

        StreamVisit Ret = static_cast<ImplClass*>(this)->visitBlock(Entry.ID);
        switch (Ret) {
        case StreamVisit::Continue:
          if (Stream.EnterSubBlock(Entry.ID)) {
            Error = "malformed block record";
            return false;
          }
          if (llvm::Error Err = readBlockAbbrevs(Stream)) {
            Error = toString(std::move(Err));
            return false;
          }
          BlockStack.push_back(Entry.ID);
          break;

        case StreamVisit::Skip: 
          if (Stream.SkipBlock()) {
            Error = "malformed serialization";
            return false;
          }
          if (Stream.AtEndOfStream())
            return true;
          break;

        case StreamVisit::Abort:
          return false;
        }
        break;
      }

      case llvm::BitstreamEntry::Record: {
        Record.clear();
        StringRef Blob;
        Expected<unsigned> MaybeRecID = Stream.readRecord(Entry.ID, Record, &Blob);
        if (!MaybeRecID) {
          Error = toString(MaybeRecID.takeError());
          return false;
        }
        unsigned RecID = MaybeRecID.get();
        unsigned BlockID = BlockStack.empty() ? 0 : BlockStack.back();
        StreamVisit Ret = static_cast<ImplClass*>(this)->visitRecord(BlockID, RecID, Record, Blob);
        switch (Ret) {
        case StreamVisit::Continue:
          break;

        case StreamVisit::Skip: 
          if (Expected<unsigned> Skipped = Stream.skipRecord(Entry.ID)) {
            Error = toString(Skipped.takeError());
            return false;
          }
          break;

        case StreamVisit::Abort:
          return false;
        }
        break;
      }
      }
    }
  }

  static llvm::Error readBlockAbbrevs(llvm::BitstreamCursor &Cursor) {
    while (true) {
      uint64_t Offset = Cursor.GetCurrentBitNo();
      Expected<unsigned> MaybeCode = Cursor.ReadCode();
      if (!MaybeCode)
        return MaybeCode.takeError();
      unsigned Code = MaybeCode.get();

      // We expect all abbrevs to be at the start of the block.
      if (Code != llvm::bitc::DEFINE_ABBREV) {
        if (llvm::Error Err = Cursor.JumpToBit(Offset))
          return Err;
        return llvm::Error::success();
      }
      if (llvm::Error Err = Cursor.ReadAbbrevRecord())
        return Err;
    }
  }
};

} // end namespace store
} // end namespace index
} // end namespace clang

#endif
