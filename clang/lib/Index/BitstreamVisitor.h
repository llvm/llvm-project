//===--- BitstreamVisitor.h - Helper for reading a bitstream --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INDEX_BITSTREAMVISITOR_H
#define LLVM_CLANG_LIB_INDEX_BITSTREAMVISITOR_H

#include "llvm/Bitcode/BitstreamReader.h"
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
    Cursor.JumpToBit(Offset);
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
      llvm::BitstreamEntry Entry = Stream.advance(llvm::BitstreamCursor::AF_DontPopBlockAtEnd);

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
          BlockInfo = Stream.ReadBlockInfoBlock();
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
          readBlockAbbrevs(Stream);
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
        unsigned RecID = Stream.readRecord(Entry.ID, Record, &Blob);
        unsigned BlockID = BlockStack.empty() ? 0 : BlockStack.back();
        StreamVisit Ret = static_cast<ImplClass*>(this)->visitRecord(BlockID, RecID, Record, Blob);
        switch (Ret) {
        case StreamVisit::Continue:
          break;

        case StreamVisit::Skip: 
          Stream.skipRecord(Entry.ID);
          break;

        case StreamVisit::Abort:
          return false;
        }
        break;
      }
      }
    }
  }

  static void readBlockAbbrevs(llvm::BitstreamCursor &Cursor) {
    while (true) {
      uint64_t Offset = Cursor.GetCurrentBitNo();
      unsigned Code = Cursor.ReadCode();

      // We expect all abbrevs to be at the start of the block.
      if (Code != llvm::bitc::DEFINE_ABBREV) {
        Cursor.JumpToBit(Offset);
        return;
      }
      Cursor.ReadAbbrevRecord();
    }
  }
};

} // end namespace store
} // end namespace index
} // end namespace clang

#endif
