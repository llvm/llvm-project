/*===- CtxInstrProfiling.h- Contextual instrumentation-based PGO  ---------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#ifndef CTX_PROFILE_CTXINSTRPROFILING_H_
#define CTX_PROFILE_CTXINSTRPROFILING_H_

#include "sanitizer_common/sanitizer_mutex.h"
#include <sanitizer/common_interface_defs.h>

namespace __ctx_profile {
using GUID = uint64_t;

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

class ContextNode final {
  const GUID Guid;
  ContextNode *const Next;
  const uint32_t NrCounters;
  const uint32_t NrCallsites;

public:
  ContextNode(GUID Guid, uint32_t NrCounters, uint32_t NrCallsites,
              ContextNode *Next = nullptr)
      : Guid(Guid), Next(Next), NrCounters(NrCounters),
        NrCallsites(NrCallsites) {}
  static inline ContextNode *alloc(char *Place, GUID Guid, uint32_t NrCounters,
                                   uint32_t NrCallsites,
                                   ContextNode *Next = nullptr);

  static inline size_t getAllocSize(uint32_t NrCounters, uint32_t NrCallsites) {
    return sizeof(ContextNode) + sizeof(uint64_t) * NrCounters +
           sizeof(ContextNode *) * NrCallsites;
  }

  uint64_t *counters() {
    ContextNode *addr_after = &(this[1]);
    return reinterpret_cast<uint64_t *>(reinterpret_cast<char *>(addr_after));
  }

  uint32_t counters_size() const { return NrCounters; }
  uint32_t callsites_size() const { return NrCallsites; }

  const uint64_t *counters() const {
    return const_cast<ContextNode *>(this)->counters();
  }

  ContextNode **subContexts() {
    return reinterpret_cast<ContextNode **>(&(counters()[NrCounters]));
  }

  ContextNode *const *subContexts() const {
    return const_cast<ContextNode *>(this)->subContexts();
  }

  GUID guid() const { return Guid; }
  ContextNode *next() { return Next; }

  size_t size() const { return getAllocSize(NrCounters, NrCallsites); }

  void reset();

  void onEntry() { ++counters()[0]; }

  uint64_t entrycount() const { return counters()[0]; }
};

/// ContextRoots are allocated by LLVM for entrypoints. The main concern is
/// the total size, LLVM doesn't actually dereference members.
struct ContextRoot {
  ContextNode *FirstNode = nullptr;
  Arena *FirstMemBlock = nullptr;
  Arena *CurrentMem = nullptr;
  // This is init-ed by the static zero initializer in LLVM.
  ::__sanitizer::StaticSpinMutex Taken;

  // Avoid surprises due to (unlikely) StaticSpinMutex changes.
  static_assert(sizeof(Taken) == 1);
};

/// This API is exposed for testing.
inline bool isScratch(const ContextNode *Ctx) {
  return (reinterpret_cast<uint64_t>(Ctx) & 1);
}

} // namespace __ctx_profile

extern "C" {

// LLVM fills these in when lowering a llvm.instrprof.callsite intrinsic.
// position 0 is used when the current context isn't scratch, 1 when it is.
extern __thread void *volatile __llvm_ctx_profile_expected_callee[2];
extern __thread __ctx_profile::ContextNode *
    *volatile __llvm_ctx_profile_callsite[2];

// __llvm_ctx_profile_current_context_root is exposed for unit testing,
// othwerise it's only used internally.
extern __thread __ctx_profile::ContextRoot
    *volatile __llvm_ctx_profile_current_context_root;

/// called by LLVM in the entry BB of a "entry point" function. The returned
/// pointer may be "tainted" - its LSB set to 1 - to indicate it's scratch.
__ctx_profile::ContextNode *
__llvm_ctx_profile_start_context(__ctx_profile::ContextRoot *Root,
                                 __ctx_profile::GUID Guid, uint32_t Counters,
                                 uint32_t Callsites);

/// paired with __llvm_ctx_profile_start_context, and called at the exit of the
/// entry point function.
void __llvm_ctx_profile_release_context(__ctx_profile::ContextRoot *Root);

/// called for any other function than entry points, in the entry BB of such
/// function. Same consideration about LSB of returned value as .._start_context
__ctx_profile::ContextNode *
__llvm_ctx_profile_get_context(void *Callee, __ctx_profile::GUID Guid,
                               uint32_t NrCounters, uint32_t NrCallsites);

/// Prepares for collection. Currently this resets counter values but preserves
/// internal structure.
void __llvm_ctx_profile_start_collection();

/// Completely free allocated memory.
void __llvm_ctx_profile_free();

/// Used to obtain the profile. The Writer is called for each root ContextNode,
/// with the ContextRoot::Taken taken. The Writer is responsible for traversing
/// the structure underneath.
bool __llvm_ctx_profile_fetch(
    void *Data, bool (*Writer)(void *, const __ctx_profile::ContextNode &));
}
#endif // CTX_PROFILE_CTXINSTRPROFILING_H_
