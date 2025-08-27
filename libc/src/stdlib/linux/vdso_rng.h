//===-- vDSO based RNG ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STDLIB_LINUX_VDSO_RNG_H
#define LIBC_SRC_STDLIB_LINUX_VDSO_RNG_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/OSUtil/linux/vdso.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/raw_mutex.h"
// TODO: this is public entrypoint, we should remove it later on.
#include "src/sys/auxv/getauxval.h"

#include <linux/mman.h> //MREMAP_MAYMOVE

namespace LIBC_NAMESPACE_DECL {
namespace vdso_rng {
extern "C" {
using Destructor = void(void *);
[[gnu::weak]] extern void *__dso_handle;
int __cxa_thread_atexit_impl(Destructor *, void *, void *);
}

//===----------------------------------------------------------------------===//
// MMap Vector
//===----------------------------------------------------------------------===//
//
// We don't want to use malloc inside the RNG implementation which complicates
// the clean up on exiting. We use a raw mmap vector to maintain available
// states. This vector grows linearly. As the pool is cached globally, we don't
// clean it up on exit.
//
//===----------------------------------------------------------------------===//

class PtrStore {
  void **data;
  void **usage;
  void **end;

  LIBC_INLINE size_t remaining_capacity() const { return end - usage; }
  LIBC_INLINE size_t capacity() const { return end - data; }
  LIBC_INLINE size_t size() const { return usage - data; }
  LIBC_INLINE bool reserve(size_t incoming, size_t page_size) {
    if (remaining_capacity() >= incoming)
      return incoming;
    size_t old_size = size();
    size_t size_in_bytes = (old_size + incoming) * sizeof(void *);
    size_t roundup = ((size_in_bytes + page_size - 1) / page_size) * page_size;
    long sysret;
    if (data == nullptr)
      sysret = LIBC_NAMESPACE::syscall_impl<long>(
          SYS_mmap, /*addr=*/nullptr,
          /*length=*/roundup,
          /*prot=*/PROT_READ | PROT_WRITE,
          /*flags=*/MAP_PRIVATE | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    else
      sysret = LIBC_NAMESPACE::syscall_impl<long>(
          SYS_mremap, /*old_address=*/data,
          /*old_size=*/capacity() * sizeof(void *),
          /*new_size=*/roundup, MREMAP_MAYMOVE, /*new_address=*/nullptr);
    if (sysret == -1 /* MAP_FAILED */)
      return false;
    data = reinterpret_cast<void **>(sysret);
    usage = data + old_size;
    end = data + (roundup / sizeof(void *));
    return true;
  }

public:
  LIBC_INLINE constexpr PtrStore()
      : data(nullptr), usage(nullptr), end(nullptr) {}
  LIBC_INLINE void push_no_grow(void *ptr) {
    LIBC_ASSERT(remaining_capacity() > 0);
    *usage++ = ptr;
  }
  LIBC_INLINE void **bump(size_t count, size_t page_size) {
    if (!reserve(count, page_size))
      return nullptr;
    void **result = usage;
    usage += count;
    return result;
  }
  LIBC_INLINE void *pop() {
    if (size() == 0)
      return nullptr;
    return *--usage;
  }
};

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

struct VGetrandomOpaqueParams {
  unsigned int size_of_opaque_states;
  unsigned int mmap_prot;
  unsigned int mmap_flags;
  unsigned int reserved[13];
};

struct Config {
  size_t page_size;
  size_t pages_per_alloc;
  size_t states_per_page;
  vdso::VDSOSymType<vdso::VDSOSym::GetRandom> getrandom;
  VGetrandomOpaqueParams params;

  LIBC_INLINE bool is_valid() const { return page_size != 0; }
};

LIBC_INLINE_VAR constexpr size_t MINIMAL_STATE_NUMBER = 4;

//===----------------------------------------------------------------------===//
// Estimate the need of states based on CPU count. The global pool will grow
// linearly at the rate of cpu_count().
//===----------------------------------------------------------------------===//
LIBC_INLINE size_t cpu_count() {
  char cpu_set[128] = {0};
  int res = LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_getaffinity, 0,
                                              sizeof(cpu_set), cpu_set);
  if (res <= 0)
    return 1;

  size_t count = 0;
  for (auto byte : cpu_set)
    count += cpp::popcount(static_cast<unsigned char>(byte));

  return count > 0 ? count : 1;
}

//===----------------------------------------------------------------------===//
// Global State Pool
//===----------------------------------------------------------------------===//
class GlobalState {
public:
  struct Result {
    void *state;
    vdso::VDSOSymType<vdso::VDSOSym::GetRandom> vgetrandom;
    size_t size_of_opaque_states;

    LIBC_INLINE static Result invalid() { return {nullptr, nullptr, 0}; }
    LIBC_INLINE bool is_valid() const { return state != nullptr; }
  };

private:
  // A mutex protecting the global state.
  RawMutex lock;

  // Free list
  PtrStore free_list;

  // configuration.
  Config config;

  // Grow available states. This function can fail if the system is out of
  // memory.
  // - This routine assumes that the global config is valid.
  // - On success, this routine returns one opaque state for direct use.
  LIBC_INLINE void *grow();

  LIBC_INLINE void ensure_config();

public:
  LIBC_INLINE constexpr GlobalState() : lock(), free_list(), config() {}
  // Try acquire a state for local usage.
  LIBC_INLINE Result get();
  // Return a state back to the global pool.
  LIBC_INLINE void recycle(void *state);
};

LIBC_INLINE_VAR GlobalState global_state{};

class LocalState {
  // Whether there is an in-flight getrandom call. If so, we simply fallback to
  // syscall.
  bool in_flight = false;
  // Whether previous attempt to get a state has failed. If so, we won't try
  // again.
  bool failed = false;
  void *state = nullptr;
  // Cache the function pointer locally
  vdso::VDSOSymType<vdso::VDSOSym::GetRandom> vgetrandom = nullptr;
  size_t size_of_opaque_states = 0;

public:
  struct Guard {
    LocalState *tls;
    LIBC_INLINE Guard(LocalState *tls) : tls(tls) {
      tls->in_flight = true;
      cpp::atomic_thread_fence(cpp::MemoryOrder::SEQ_CST);
    }
    LIBC_INLINE Guard(Guard &&other) : tls(other.tls) { other.tls = nullptr; }
    LIBC_INLINE ~Guard() {
      cpp::atomic_thread_fence(cpp::MemoryOrder::SEQ_CST);
      if (tls)
        tls->in_flight = false;
    }
    LIBC_INLINE void fill(void *buf, size_t size) const;
  };
  LIBC_INLINE constexpr LocalState() {}
  LIBC_INLINE cpp::optional<Guard> get();
};

LIBC_INLINE_VAR LIBC_THREAD_LOCAL LocalState local_state{};

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

LIBC_INLINE void GlobalState::ensure_config() {
  // Already initialized.
  if (config.is_valid())
    return;

  config.getrandom =
      LIBC_NAMESPACE::vdso::TypedSymbol<vdso::VDSOSym::GetRandom>{};
  if (!config.getrandom)
    return;

  // Call with special flag to get the desired configuration.
  int res = config.getrandom(
      /*buf=*/nullptr, /*count=*/0, /*flags=*/0,
      /*opaque_states=*/&config.params,
      /*size_of_opaque_states=*/~0);
  if (res != 0)
    return;

  auto page_size = static_cast<size_t>(LIBC_NAMESPACE::getauxval(AT_PAGESZ));
  if (!page_size)
    return;

  size_t count = cpp::max(cpu_count(), MINIMAL_STATE_NUMBER);

  config.states_per_page = page_size / config.params.size_of_opaque_states;

  config.pages_per_alloc =
      count / config.states_per_page + (count % config.states_per_page != 0);

  // Finally set the page size to mark the config as valid.
  config.page_size = page_size;
}

LIBC_INLINE void *GlobalState::grow() {
  LIBC_ASSERT(config.is_valid());
  size_t total_states = config.pages_per_alloc * config.states_per_page;
  void **buffer = free_list.bump(total_states, config.page_size);

  // Failed to allocate space for pointers. Exit early.
  if (!buffer)
    return nullptr;

  long mmap_res = LIBC_NAMESPACE::syscall_impl<long>(
      SYS_mmap, /*addr=*/nullptr,
      /*length=*/config.page_size * config.pages_per_alloc,
      /*prot=*/config.params.mmap_prot,
      /*flags=*/config.params.mmap_flags,
      /*fd=*/-1, /*offset=*/0);

  // Failed to allocate memory. Exit early.
  if (mmap_res == -1 /* MAP_FAILED */)
    return nullptr;

  // Initialize the page.
  // Notice that states shall not go across page boundaries.
  char *pages = reinterpret_cast<char *>(mmap_res);
  for (size_t p = 0; p < config.pages_per_alloc; ++p) {
    char *page = &pages[p * config.page_size];
    for (size_t s = 0; s < config.states_per_page; ++s) {
      void *state = &page[s * config.params.size_of_opaque_states];
      *buffer++ = state;
    }
  }

  // Return the last one. It should always be valid at this stage.
  return free_list.pop();
}

LIBC_INLINE GlobalState::Result GlobalState::get() {
  cpp::lock_guard guard(lock);
  ensure_config();
  // If page size is not set, the global config is invalid. Early return.
  if (!config.is_valid())
    return Result::invalid();

  if (void *state = free_list.pop())
    return {state, config.getrandom, config.params.size_of_opaque_states};

  // At this stage, we know that the config is valid.
  return {grow(), config.getrandom, config.params.size_of_opaque_states};
}

LIBC_INLINE void GlobalState::recycle(void *state) {
  cpp::lock_guard guard(lock);
  LIBC_ASSERT(state != nullptr);
  free_list.push_no_grow(state);
}

//===----------------------------------------------------------------------===//
// LocalState
//===----------------------------------------------------------------------===//

LIBC_INLINE void LocalState::Guard::fill(void *buf, size_t size) const {
  LIBC_ASSERT(tls->state != nullptr);
  char *cursor = reinterpret_cast<char *>(buf);
  size_t remaining = size;
  while (remaining > 0) {
    int res = tls->vgetrandom(cursor, remaining, /* default random flag */ 0,
                              tls->state, tls->size_of_opaque_states);
    if (res < 0)
      continue;
    remaining -= static_cast<size_t>(res);
    cursor += res;
  }
}

LIBC_INLINE cpp::optional<LocalState::Guard> LocalState::get() {
  if (in_flight)
    return cpp::nullopt;

  Guard guard(this);

  // If uninitialized, try to get a state.
  if (!failed && !state) {
    int register_res = __cxa_thread_atexit_impl(
        [](void *self) {
          auto *tls = static_cast<LocalState *>(self);
          // Reject all future attempts to get a state.
          void *state = tls->state;
          tls->in_flight = true;
          tls->failed = true;
          tls->state = nullptr;
          cpp::atomic_thread_fence(cpp::MemoryOrder::SEQ_CST);
          if (state)
            global_state.recycle(state);
        },
        this, __dso_handle);
    if (register_res == 0) {
      GlobalState::Result result = global_state.get();
      state = result.state;
      vgetrandom = result.vgetrandom;
      size_of_opaque_states = result.size_of_opaque_states;
    }
    if (!state)
      failed = true;
  }

  if (!state)
    return cpp::nullopt;

  return cpp::move(guard);
}

} // namespace vdso_rng
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_SRC_STDLIB_LINUX_VDSO_RNG_H
