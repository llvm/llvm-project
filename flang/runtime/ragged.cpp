//===-- runtime/ragged.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/ragged.h"
#include <cstdlib>

namespace Fortran::runtime {

inline bool isIndirection(const RaggedArrayHeader *const header) {
  return header->flags & 1;
}

inline std::size_t rank(const RaggedArrayHeader *const header) {
  return header->flags >> 1;
}

RaggedArrayHeader *RaggedArrayAllocate(RaggedArrayHeader *header, bool isHeader,
    std::int64_t rank, std::int64_t elementSize, std::int64_t *extentVector) {
  if (header && rank) {
    std::int64_t size{1};
    for (std::int64_t counter{0}; counter < rank; ++counter) {
      size *= extentVector[counter];
      if (size <= 0) {
        return nullptr;
      }
    }
    header->flags = (rank << 1) | isHeader;
    header->extentPointer = extentVector;
    if (isHeader) {
      header->bufferPointer = std::calloc(sizeof(RaggedArrayHeader), size);
    } else {
      header->bufferPointer =
          static_cast<void *>(std::calloc(elementSize, size));
    }
    return header;
  } else {
    return nullptr;
  }
}

// Deallocate a ragged array from the heap.
void RaggedArrayDeallocate(RaggedArrayHeader *raggedArrayHeader) {
  if (raggedArrayHeader) {
    if (std::size_t end{rank(raggedArrayHeader)}) {
      if (isIndirection(raggedArrayHeader)) {
        std::size_t linearExtent{1u};
        for (std::size_t counter{0u}; counter < end && linearExtent > 0;
             ++counter) {
          linearExtent *= raggedArrayHeader->extentPointer[counter];
        }
        for (std::size_t counter{0u}; counter < linearExtent; ++counter) {
          RaggedArrayDeallocate(&static_cast<RaggedArrayHeader *>(
              raggedArrayHeader->bufferPointer)[counter]);
        }
      }
      std::free(raggedArrayHeader->bufferPointer);
      std::free(raggedArrayHeader->extentPointer);
      raggedArrayHeader->flags = 0u;
    }
  }
}

extern "C" {
void *RTNAME(RaggedArrayAllocate)(void *header, bool isHeader,
    std::int64_t rank, std::int64_t elementSize, std::int64_t *extentVector) {
  auto *result = RaggedArrayAllocate(static_cast<RaggedArrayHeader *>(header),
      isHeader, rank, elementSize, extentVector);
  return static_cast<void *>(result);
}

void RTNAME(RaggedArrayDeallocate)(void *raggedArrayHeader) {
  RaggedArrayDeallocate(static_cast<RaggedArrayHeader *>(raggedArrayHeader));
}
} // extern "C"
} // namespace Fortran::runtime
