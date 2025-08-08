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
#include "src/__support/mpmc_stack.h"
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/linux/raw_mutex.h"
// TODO: this is public entrypoint, we should remove it later on.
#include "src/sys/auxv/getauxval.h"

namespace LIBC_NAMESPACE_DECL {
namespace vdso_rng {
extern "C" {
using Destructor = void(void *);
[[gnu::weak]] extern void *__dso_handle;
int __cxa_thread_atexit_impl(Destructor *, void *, void *);
}
class GlobalState {
public:
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
  };

private:
  // A lock-free stack of free opaque states.
  MPMCStack<void *> free_list{};
  // A mutex protecting the allocation of new pages.
  RawMutex allocation_mutex{};

  // Shared global configuration.
  static CallOnceFlag config_flag;
  static Config config;

  // We grow the states by the number of CPUs. This function uses
  // SYS_sched_getaffinity to get the number of CPUs.
  LIBC_INLINE static size_t cpu_count();

  // Grow available states. This function can fail if the system is out of
  // memory.
  // - This routine assumes that the global config is valid.
  // - On success, this routine returns one opaque state for direct use.
  LIBC_INLINE void *grow();

public:
  LIBC_INLINE constexpr GlobalState() {}
  LIBC_INLINE static const Config &get_config();
  LIBC_INLINE static const Config &get_config_unchecked() { return config; }
  LIBC_INLINE void *get();
  LIBC_INLINE void recycle(void *state);
};

LIBC_INLINE_VAR GlobalState global_state{};

class LocalState {
  bool in_flight = false;
  bool failed = false;
  void *state = nullptr;

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
  LIBC_INLINE cpp::optional<Guard> get() {
    if (in_flight)
      return cpp::nullopt;

    Guard guard(this);

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
              LIBC_NAMESPACE::vdso_rng::global_state.recycle(state);
          },
          this, __dso_handle);
      if (register_res == 0)
        state = LIBC_NAMESPACE::vdso_rng::global_state.get();
      if (!state)
        failed = true;
    }

    if (!state)
      return cpp::nullopt;

    return cpp::move(guard);
  }
};

LIBC_INLINE_VAR LIBC_THREAD_LOCAL LocalState local_state{};

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

LIBC_INLINE_VAR GlobalState::Config GlobalState::config{};
LIBC_INLINE_VAR CallOnceFlag GlobalState::config_flag = 0;

LIBC_INLINE size_t GlobalState::cpu_count() {
  char cpu_set[128] = {0};
  int res = LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_getaffinity, 0,
                                              sizeof(cpu_set), cpu_set);
  if (res <= 0)
    return 1;

  size_t count = 0;
  for (size_t i = 0; i < sizeof(cpu_set) / sizeof(unsigned long); ++i) {
    unsigned long *mask_ptr = reinterpret_cast<unsigned long *>(cpu_set);
    count += LIBC_NAMESPACE::cpp::popcount(mask_ptr[i]);
  }

  return count > 0 ? count : 1;
}

LIBC_INLINE const GlobalState::Config &GlobalState::get_config() {
  callonce(&config_flag, []() {
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

    config.page_size = LIBC_NAMESPACE::getauxval(AT_PAGESZ);
    if (!config.page_size)
      return;

    size_t count = cpp::max(cpu_count(), size_t{4});

    config.states_per_page =
        config.page_size / config.params.size_of_opaque_states;

    config.pages_per_alloc =
        count / config.states_per_page + (count % config.states_per_page != 0);
  });
  return config;
}

LIBC_INLINE void *GlobalState::grow() {
  cpp::lock_guard guard(allocation_mutex);

  // It is possible that when we finally grab the lock, other threads have
  // successfully finished the allocation already. Hence, we first try if we
  // can pop anything from the free list.
  if (cpp::optional<void *> state = free_list.pop())
    return *state;

  long mmap_res = LIBC_NAMESPACE::syscall_impl<long>(
      SYS_mmap, /*addr=*/nullptr,
      /*length=*/config.page_size * config.pages_per_alloc,
      /*prot=*/config.params.mmap_prot,
      /*flags=*/config.params.mmap_flags,
      /*fd=*/-1, /*offset=*/0);
  if (mmap_res == -1 /* MAP_FAILED */)
    return nullptr;

  char *pages = reinterpret_cast<char *>(mmap_res);

  // Initialize the page.
  size_t total_states = config.pages_per_alloc * config.states_per_page;
  size_t free_states = total_states - 1; // reserve one for direct use.
  __extension__ void *opaque_states[total_states];
  size_t index = 0;
  for (size_t p = 0; p < config.pages_per_alloc; ++p) {
    char *page = &pages[p * config.page_size];
    for (size_t s = 0; s < config.states_per_page; ++s) {
      void *state = &page[s * config.params.size_of_opaque_states];
      opaque_states[index++] = state;
    }
  }

  constexpr size_t RETRY_COUNT = 64;
  for (size_t i = 0; i < RETRY_COUNT; ++i) {
    if (free_list.push_all(opaque_states, free_states))
      break;
    // Abort if we are still short in memory after all these retries.
    if (i + 1 == RETRY_COUNT) {
      LIBC_NAMESPACE::syscall_impl<long>(
          SYS_munmap, pages, config.page_size * config.pages_per_alloc);
      return nullptr;
    }
  }

  return opaque_states[free_states];
}

LIBC_INLINE void *GlobalState::get() {
  const Config &config = get_config();
  // If page size is not set, the global config is invalid. Early return.
  if (!config.page_size)
    return nullptr;

  if (cpp::optional<void *> state = free_list.pop())
    return *state;

  // At this stage, we know that the config is valid.
  return grow();
}

LIBC_INLINE void GlobalState::recycle(void *state) {
  LIBC_ASSERT(state != nullptr);
  constexpr size_t RETRY_COUNT = 64;
  for (size_t i = 0; i < RETRY_COUNT; ++i)
    if (free_list.push(state))
      return;
  // Otherwise, we just let it leak. It won't be too bad not to reuse the state
  // since the OS can free the page if memory is tight.
}

//===----------------------------------------------------------------------===//
// LocalState
//===----------------------------------------------------------------------===//

LIBC_INLINE void LocalState::Guard::fill(void *buf, size_t size) const {
  LIBC_ASSERT(tls->state != nullptr);
  char *cursor = reinterpret_cast<char *>(buf);
  size_t remaining = size;
  const auto &config = GlobalState::get_config_unchecked();
  while (remaining > 0) {
    int res = config.getrandom(cursor, remaining, /* default random flag */ 0,
                               tls->state, config.params.size_of_opaque_states);
    if (res < 0)
      continue;
    remaining -= static_cast<size_t>(res);
    cursor += res;
  }
}

//===----------------------------------------------------------------------===//
// Fallback Fill
//===----------------------------------------------------------------------===//
LIBC_INLINE void fallback_rng_fill(void *buf, size_t size) {
  size_t remaining = size;
  char *cursor = reinterpret_cast<char *>(buf);
  while (remaining > 0) {
    int res =
        LIBC_NAMESPACE::syscall_impl<int>(SYS_getrandom, cursor, remaining, 0);
    if (res < 0)
      continue;
    remaining -= static_cast<size_t>(res);
    cursor += res;
  }
}

} // namespace vdso_rng
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_SRC_STDLIB_LINUX_VDSO_RNG_H
