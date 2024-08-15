//===--- IndexRecordHasher.h - Index record hashing -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INDEX_INDEXRECORDHASHER_H
#define LLVM_CLANG_LIB_INDEX_INDEXRECORDHASHER_H

#include <array>

namespace clang {
class ASTContext;

namespace index {
  class FileIndexRecord;

  std::array<uint8_t, 8> hashRecord(const FileIndexRecord &record,
                                    ASTContext &context);
} // end namespace index
} // end namespace clang

#endif
