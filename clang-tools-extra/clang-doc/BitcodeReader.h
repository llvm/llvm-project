//===--  BitcodeReader.h - ClangDoc Bitcode Reader --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Support/Error.h"
#include <optional>

namespace clang {
namespace doc {

// Class to read bitstream into an InfoSet collection
class ClangDocBitcodeReader {
public:
  ClangDocBitcodeReader(llvm::BitstreamCursor &Stream, DiagnosticsEngine &Diags)
      : Stream(Stream), Diags(Diags) {}

  // Main entry point, calls readBlock to read each block in the given stream.
  llvm::Expected<OwningPtrArray<Info>> readBitcode();

private:
  enum class Cursor { BadBlock = 1, Record, BlockEnd, BlockBegin };

  // Top level parsing
  llvm::Error validateStream();
  llvm::Error readVersion();
  llvm::Error readBlockInfoBlock();

  // Read a block of records into a single Info struct, calls readRecord on each
  // record found.
  template <typename T> llvm::Error readBlock(unsigned ID, T I);
  template <typename T> llvm::Error readBlockWithNamespace(unsigned ID, T I);

  template <typename T, typename BlockBeginHandler, typename BlockEndHandler,
            typename RecordHandler>
  llvm::Error parseBlock(unsigned ID, T I, BlockBeginHandler &&BBH,
                         BlockEndHandler &&BEH, RecordHandler &&RH);

  template <typename T, typename BlockBeginHandler, typename BlockEndHandler>
  llvm::Error parseBlock(unsigned ID, T I, BlockBeginHandler &&BBH,
                         BlockEndHandler &&BEH);

  template <typename ChildType>
  llvm::Expected<bool> readSubBlockIfMatch(unsigned ID, unsigned TargetID,
                                           llvm::SmallVectorImpl<ChildType> &V);

  struct ReferenceMap {
    FieldId Field;
    llvm::SmallVectorImpl<Reference> *Vec;
  };

  template <typename InfoT>
  llvm::Expected<bool>
  routeReferenceBlock(unsigned ID, llvm::SmallVectorImpl<Reference> &Namespaces,
                      InfoT *I,
                      std::initializer_list<ReferenceMap> Mappings = {});

  // Step through a block of records to find the next data field.
  template <typename T> llvm::Error readSubBlock(unsigned ID, T I);

  // Read record data into the given Info data field, calling the appropriate
  // parseRecord functions to parse and store the data.
  template <typename T> llvm::Error readRecord(unsigned ID, T I);

  // Allocate the relevant type of info and add read data to the object.
  template <typename T> llvm::Expected<OwnedPtr<Info>> createInfo(unsigned ID);

  // Helper function to step through blocks to find and dispatch the next record
  // or block to be read.
  llvm::Expected<Cursor> skipUntilRecordOrBlock(unsigned &BlockOrRecordID);

  // Helper function to set up the appropriate type of Info.
  llvm::Expected<OwnedPtr<Info>> readBlockToInfo(unsigned ID);

  template <typename InfoType, typename T, typename CallbackFunction>
  llvm::Error handleSubBlock(unsigned ID, T Parent, CallbackFunction Function);

  template <typename InfoType, typename T>
  llvm::Error handleSubBlock(unsigned ID, T Parent);

  template <typename InfoType, typename T, typename CallbackFunction>
  llvm::Error handleTypeSubBlock(unsigned ID, T Parent,
                                 CallbackFunction Function);

  llvm::BitstreamCursor &Stream;
  std::optional<llvm::BitstreamBlockInfo> BlockInfo;
  FieldId CurrentReferenceField = FieldId::F_default;
  DiagnosticsEngine &Diags;
};

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_BITCODEREADER_H
