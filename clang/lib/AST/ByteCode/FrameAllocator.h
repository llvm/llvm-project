//===-------------------- FrameAllocator.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_INTERP_FRAME_ALLOCATOR_H
#define LLVM_CLANG_AST_INTERP_FRAME_ALLOCATOR_H

#include "llvm/Support/Compiler.h"
#ifndef NDEBUG
#include "llvm/ADT/SmallVector.h"
#endif
#include <algorithm>
#include <cassert>
#include <new>

namespace clang {
namespace interp {

// Set this to 1 to collect some light statistics.
// Print via printStats().
#define COLLECT_STATS 0

/// Allocator for function frames.
///
/// The function frame size includes the size reserved for local variables.
/// Function frames are allocated strictly in a LIFO manner, i.e. the last
/// created frame is the first frame that is destroyed.
///
/// Since the address a function frame is allocated in needs to stay stable
/// during the lifetime of the frame, we allocate them here in chunks.
///
/// A chunk is of (at least) MinChunkSize size and only gets deallocated once it
/// is empty AND the previous chunk is also empty.
///
class FrameAllocator final {
private:
  struct Chunk {
    Chunk *Prev = nullptr;
    unsigned Size;
    unsigned Used = 0;
    alignas(sizeof(void *)) char Memory[1];

    Chunk(unsigned Size) : Size(Size) {}
    unsigned bytesUnused() const { return Size - Used; }
  };
  static constexpr unsigned MinChunkSize = (4u * 1024u) - sizeof(Chunk);

  Chunk *Tail = nullptr;
#if COLLECT_STATS
  size_t MaxSize = 0;
  unsigned LargestFrame = 0;
  unsigned NumFrames = 0;
  unsigned NumAllocs = 0;
#endif

#ifndef NDEBUG
  llvm::SmallVector<unsigned> FrameSizes;
#endif

public:
  FrameAllocator() = default;
  FrameAllocator(FrameAllocator &) = delete;
  FrameAllocator(FrameAllocator &&) = delete;
  ~FrameAllocator() {
    while (Tail)
      deallocTail();
  }

  char *reserve(unsigned Size) {
    if (LLVM_UNLIKELY(!Tail))
      allocateNewChunk(std::max(Size, MinChunkSize));
    assert(Tail);

#ifndef NDEBUG
    FrameSizes.push_back(Size);
#endif

    char *Mem;
    if (Chunk *C = getChunkToUse(Size); C->bytesUnused() >= Size) {
      Mem = &C->Memory[C->Used];
      C->Used += Size;
    } else {
      // We need to allocate a new chunk. If the requested size is larger than
      // the minimum, use that.
      allocateNewChunk(std::max(Size, MinChunkSize));
      Tail->Used += Size;
      Mem = Tail->Memory;
    }

#if COLLECT_STATS
    LargestFrame = std::max(Size, LargestFrame);
    MaxSize = std::max(MaxSize, countAllBytes());
    ++NumFrames;
#endif

    return Mem;
  }

  /// Pop the memory of the last function frame that was added.
  /// The passed \c FrameSize needs to match the latest size passed to
  /// reserve(). If it doesn't, bad things will happen.
  void pop(unsigned FrameSize) {
#ifndef NDEBUG
    assert(FrameSize == FrameSizes.back());
#endif
    // Frame destructor must've already been called.
    assert(Tail);
    Chunk *C = Tail->Used == 0 ? Tail->Prev : Tail;
    assert(C);
    assert(FrameSize <= C->Used);
    C->Used -= FrameSize;

    // Deallocate the tail chunk *if* it is empty _and_ the previous chunk is
    // also empty.
    // Since we create chunks specicially for large frames, we need to loop
    // here.
    while (Tail->Used == 0 && Tail->Prev && Tail->Prev->Used == 0)
      deallocTail();

#ifndef NDEBUG
    FrameSizes.pop_back();
#endif
  }

private:
  /// Return the chunk to use to allocate a new frame into.
  /// This is not always this->Tail, since Tail might be empty AND have a
  /// previous chunk. In that case, we use the previous chunk, if it does have
  /// \p Size bytes left.
  Chunk *getChunkToUse(unsigned Size) {
    assert(Tail);
    if (Tail->Used == 0 && Tail->Prev && Tail->Prev->bytesUnused() >= Size)
      return Tail->Prev;
    return Tail;
  }

  void allocateNewChunk(unsigned Size) {
    char *Mem = new char[sizeof(Chunk) + Size];
    auto *C = new (Mem) Chunk(Size);
    C->Prev = Tail;
    Tail = C;

    assert(Tail);

#if COLLECT_STATS
    ++NumAllocs;
#endif
  }

  void deallocTail() {
    assert(Tail);
    Chunk *C = Tail;
    Tail = Tail->Prev;
    delete[] reinterpret_cast<char *>(C);
  }

#if COLLECT_STATS
  size_t countAllBytes() const {
    size_t Result = 0;
    Chunk *C = Tail;
    while (C) {
      Result += C->Size + sizeof(Chunk);
      C = C->Prev;
    }
    return Result;
  }
  void printStats() const {
    llvm::errs() << "*** FrameAllocator stats ***\n";
    if (!Tail) {
      llvm::errs() << "empty\n";
      return;
    }

    Chunk *C = Tail;
    unsigned N = 0;
    while (C) {
      llvm::errs() << "Chunk " << N << ": " << C->Used << " / " << C->Size
                   << " (";
      double Percentage =
          (static_cast<double>(C->Used) / static_cast<double>(C->Size)) * 100;
      llvm::errs() << llvm::formatv("{0:2}", Percentage) << "%)\n";
      ++N;
      C = C->Prev;
    }
    llvm::errs() << "Max allocated bytes: " << MaxSize << '\n';
    llvm::errs() << "Largest frame: " << LargestFrame << '\n';
    llvm::errs() << "Frames created: " << NumFrames << '\n';
    llvm::errs() << "Allocations: " << NumAllocs << '\n';

    llvm::errs() << "Occupancy: ";
    size_t AllUsed = 0;
    size_t AllSize = 0;
    for (Chunk *C = Tail; C; C = C->Prev) {
      AllUsed += C->Used;
      AllSize += C->Size;
    }
    double Occupancy =
        (static_cast<double>(AllUsed) / static_cast<double>(AllSize)) * 100;
    llvm::errs() << llvm::formatv("{0:2}", Occupancy) << "%\n";
  }
#endif
};
} // namespace interp
} // namespace clang

#endif
