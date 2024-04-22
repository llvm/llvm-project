/*===- CtxInstrProfiling.h- Contextual instrumentation-based PGO  ---------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#ifndef CTX_PROFILE_CTXINSTRPROFILING_H_
#define CTX_PROFILE_CTXINSTRPROFILING_H_

#include <sanitizer/common_interface_defs.h>

namespace __ctx_profile {

/// Arena (bump allocator) forming a linked list. Intentionally not thread safe.
/// Allocation and de-allocation happen using sanitizer APIs. We make that
/// explicit.
class Arena final {
public:
  // When allocating a new Arena, optionally specify an existing one to append
  // to, assumed to be the last in the Arena list. We only need to support
  // appending to the arena list.
  static Arena *allocateNewArena(size_t Size, Arena *Prev = nullptr);
  static void freeArenaList(Arena *&A);

  uint64_t size() const { return Size; }

  // Allocate S bytes or return nullptr if we don't have that many available.
  char *tryBumpAllocate(size_t S) {
    if (Pos + S > Size)
      return nullptr;
    Pos += S;
    return start() + (Pos - S);
  }

  Arena *next() const { return Next; }

  // the beginning of allocatable memory.
  const char *start() const { return const_cast<Arena *>(this)->start(); }
  const char *pos() const { return start() + Pos; }

private:
  explicit Arena(uint32_t Size) : Size(Size) {}
  ~Arena() = delete;

  char *start() { return reinterpret_cast<char *>(&this[1]); }

  Arena *Next = nullptr;
  uint64_t Pos = 0;
  const uint64_t Size;
};

} // namespace __ctx_profile
#endif // CTX_PROFILE_CTXINSTRPROFILING_H_
