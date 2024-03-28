/*===- InstrProfilingArena.h- Simple arena  -------------------------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILE_INSTRPROFILINGARENA_H_
#define PROFILE_INSTRPROFILINGARENA_H_

#include "InstrProfiling.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_vector.h"

namespace __profile {
using GUID = uint64_t;

/// Arena forming a linked list, if more space is needed. Intentionally not
/// thread safe.
class Arena final {
public:
  static Arena *allocate(size_t Size, Arena *Prev = nullptr);
  uint64_t size() const { return Size; }
  char *tryAllocate(size_t S) {
    if (Pos + S > Size)
      return nullptr;
    Pos += S;
    return start() + (Pos - S);
  }
  Arena *next() const { return Next; }
  const char *start() const { return const_cast<Arena*>(this)->start(); }
  const char *pos() const { return start() + Pos; }

private:
  explicit Arena(uint32_t Size) : Size(Size) {}
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

  void onEntry() {
    ++counters()[0];
  }

  uint64_t entrycount() const {
    return counters()[0];
  }
};


// Exposed for test. Constructed and zero-initialized by LLVM. Implicitly,
// LLVM must know the shape of this.
struct ContextRoot {
  ContextNode *FirstNode = nullptr;
  Arena *FirstMemBlock = nullptr;
  Arena *CurrentMem = nullptr;
  // This is init-ed by the static zero initializer in LLVM.
  ::__sanitizer::StaticSpinMutex Taken;
};

inline bool isScratch(const ContextNode* Ctx) {
  return (reinterpret_cast<uint64_t>(Ctx) & 1);
}

} // namespace __profile

extern "C" {

// position 0 is used when the current context isn't scratch, 1 when it is.
extern __thread void *volatile __llvm_instrprof_expected_callee[2];
extern __thread __profile::ContextNode **volatile __llvm_instrprof_callsite[2];

extern __thread __profile::ContextRoot
    *volatile __llvm_instrprof_current_context_root;

COMPILER_RT_VISIBILITY __profile::ContextNode *
__llvm_instrprof_start_context(__profile::ContextRoot *Root,
                              __profile::GUID Guid, uint32_t Counters,
                              uint32_t Callsites);
COMPILER_RT_VISIBILITY void
__llvm_instrprof_release_context(__profile::ContextRoot *Root);

COMPILER_RT_VISIBILITY __profile::ContextNode *
__llvm_instrprof_get_context(void *Callee, __profile::GUID Guid,
                            uint32_t NrCounters, uint32_t NrCallsites);
}
#endif