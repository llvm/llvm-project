//===--  BitcodeReader.h - ClangDoc Bitcode Reader --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a reader for parsing the clang-doc internal
// representation from LLVM bitcode. The reader takes in a stream of bits and
// generates the set of infos that it represents.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_BITCODEREADER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_BITCODEREADER_H

#include "BitcodeWriter.h"
#include "Representation.h"
#include "clang/AST/AST.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitstreamReader.h"

namespace clang {
namespace doc {

// Class to read bitstream into an InfoSet collection
class ClangDocBitcodeReader {
public:
  ClangDocBitcodeReader(llvm::BitstreamCursor &Stream) : Stream(Stream) {}

  // Main entry point, calls readBlock to read each block in the given stream.
  llvm::Expected<std::vector<std::unique_ptr<Info>>> readBitcode();

private:
  enum class Cursor { BadBlock = 1, Record, BlockEnd, BlockBegin };

  // Top level parsing
  bool validateStream();
  bool readVersion();
  bool readBlockInfoBlock();

  // Read a block of records into a single Info struct, calls readRecord on each
  // record found.
  template <typename T> bool readBlock(unsigned ID, T I);

  // Step through a block of records to find the next data field.
  template <typename T> bool readSubBlock(unsigned ID, T I);

  // Read record data into the given Info data field, calling the appropriate
  // parseRecord functions to parse and store the data.
  template <typename T> bool readRecord(unsigned ID, T I);

  // Allocate the relevant type of info and add read data to the object.
  template <typename T> std::unique_ptr<Info> createInfo(unsigned ID);

  // Helper function to step through blocks to find and dispatch the next record
  // or block to be read.
  Cursor skipUntilRecordOrBlock(unsigned &BlockOrRecordID);

  // Helper function to set up the approriate type of Info.
  std::unique_ptr<Info> readBlockToInfo(unsigned ID);

  llvm::BitstreamCursor &Stream;
  Optional<llvm::BitstreamBlockInfo> BlockInfo;
  FieldId CurrentReferenceField;
};

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_BITCODEREADER_H
