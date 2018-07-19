//===- CXString.h - Routines for manipulating CXStrings -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXStrings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_LIBCLANG_INDEX_INTERNAL_H
#define LLVM_CLANG_TOOLS_LIBCLANG_INDEX_INTERNAL_H

#include "clang-c/Index.h"

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(blocks)

#define INVOKE_BLOCK2(block, arg1, arg2) block(arg1, arg2)

#else
// If we are compiled with a compiler that doesn't have native blocks support,
// define and call the block manually. 

#define INVOKE_BLOCK2(block, arg1, arg2) block->invoke(block, arg1, arg2)

typedef struct _CXCursorAndRangeVisitorBlock {
  void *isa;
  int flags;
  int reserved;
  enum CXVisitorResult (*invoke)(_CXCursorAndRangeVisitorBlock *,
                                 CXCursor, CXSourceRange);
} *CXCursorAndRangeVisitorBlock;

#endif // !__has_feature(blocks)

/// \brief The result of comparing two source ranges.
enum RangeComparisonResult {
  /// \brief Either the ranges overlap or one of the ranges is invalid.
  RangeOverlap,

  /// \brief The first range ends before the second range starts.
  RangeBefore,

  /// \brief The first range starts after the second range ends.
  RangeAfter
};

#endif
