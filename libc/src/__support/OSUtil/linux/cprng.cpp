//===- Linux implementation of secure random buffer generation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/OSUtil/linux/cprng.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/OSUtil/linux/syscall.h"
#include "src/__support/OSUtil/linux/vdso.h"
#include "src/__support/block.h"
#include "src/__support/libc_assert.h"
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/linux/raw_mutex.h"
#include "src/sched/sched_getaffinity.h"
#include "src/sched/sched_getcpucount.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/sysconf.h"

extern "C" int __cxa_thread_atexit_impl(void (*)(void *), void *, void *);
extern "C" int __cxa_atexit(void (*)(void *), void *, void *);
extern "C" [[gnu::weak, gnu::visibility("hidden")]] void *__dso_handle =
    nullptr;

namespace LIBC_NAMESPACE_DECL {
namespace cprng {
namespace {

using namespace vdso;
// A block of random state together with enough space to hold all its freelist.
struct StateBlock {
  StateBlock *prev;
  StateBlock *next;
  void *appendix[0];
  void *&pages() { return appendix[0]; }
  void *&freelist(size_t index) { return appendix[index + 1]; }
};

struct vgetrandom_opaque_params {
  unsigned size_of_opaque_state = 0;
  unsigned mmap_prot = 0;
  unsigned mmap_flags = 0;
  unsigned reserved[13];
};

class GlobalConfig {
public:
  const size_t page_size = 0;
  const size_t pages_per_block = 0;
  const size_t states_per_page = 0;
  const vgetrandom_opaque_params params = {};

private:
  static size_t guess_cpu_count() {
    cpu_set_t cpuset{};
    if (LIBC_NAMESPACE::sched_getaffinity(0, sizeof(cpuset), &cpuset))
      return 1u;
    int count = LIBC_NAMESPACE::__sched_getcpucount(sizeof(cpu_set_t), &cpuset);
    return static_cast<size_t>(count > 1 ? count : 1);
  }

private:
  constexpr GlobalConfig(size_t page_size, size_t pages_per_block,
                         size_t states_per_page,
                         vgetrandom_opaque_params params)
      : page_size(page_size), pages_per_block(pages_per_block),
        states_per_page(states_per_page), params(params) {}

public:
  static cpp::optional<GlobalConfig> get() {
    size_t page_size = 0;
    size_t states_per_page = 0;
    size_t pages_per_block = 0;
    vgetrandom_opaque_params params = {};

    // check symbol availability
    TypedSymbol<VDSOSym::GetRandom> vgetrandom;
    if (!vgetrandom)
      return cpp::nullopt;

    // get valid page size
    long page_size_res = sysconf(_SC_PAGESIZE);
    if (page_size_res <= 0)
      return cpp::nullopt;
    page_size = static_cast<size_t>(page_size_res);

    // get parameters for state allocation
    if (vgetrandom(nullptr, 0, 0, &params, ~0U))
      return cpp::nullopt;

    // Compute the number of states per page. On a valid human-constructable
    // computer as year 2024, the following operations shall not overflow given
    // the above operations are correctly returned.
    size_t guessed_bytes = guess_cpu_count() * params.size_of_opaque_state;
    size_t aligned_bytes = align_up(guessed_bytes, page_size);
    states_per_page = page_size / params.size_of_opaque_state;
    pages_per_block = aligned_bytes / page_size;
    return GlobalConfig(page_size, pages_per_block, states_per_page, params);
  }
};

// Utilities to allocate memory and hold it temporarily.
template <typename T> struct Allocation {
  T *ptr;
  size_t raw_size;
  Allocation(size_t raw_size) : ptr(nullptr), raw_size(raw_size) {
    AllocChecker ac{};
    ptr = static_cast<T *>(
        /* NOLINT(llvmlibc-callee-namespace) */ ::operator new(raw_size, ac));
    if (!ac)
      ptr = nullptr;
  }
  T *operator->() { return ptr; }
  operator bool() const { return ptr != nullptr; }
  ~Allocation() {
    if (ptr)
      /* NOLINT(llvmlibc-callee-namespace )*/ ::operator delete(ptr, raw_size);
  }
  void release() { ptr = nullptr; }
};

// A monotonic state pool.
class MonotonicStatePool {
  RawMutex mutex;
  GlobalConfig config;
  StateBlock sentinel;
  StateBlock *free_cursor;
  // invariant: at sentinel, free_count is always locally full such that we
  // don't need to specially check the condition for sentinel when recycling.
  size_t free_count;

  constexpr MonotonicStatePool(GlobalConfig config)
      : mutex(), config(config), sentinel{&sentinel, &sentinel, {}},
        free_cursor(&sentinel),
        free_count(config.pages_per_block * config.states_per_page) {}
  MonotonicStatePool(const MonotonicStatePool &) = delete;
  MonotonicStatePool(MonotonicStatePool &&) = delete;

  bool is_empty() const { return free_cursor == &sentinel; }
  bool is_locally_full() const {
    return free_count == config.pages_per_block * config.states_per_page;
  }
  size_t state_block_raw_size() const {
    return sizeof(StateBlock) +
           (config.states_per_page * config.pages_per_block + 1) *
               sizeof(void *);
  }
  bool allocate_new_block() {
    LIBC_ASSERT(is_empty());
    // allocate a new block
    size_t raw_size = state_block_raw_size();
    Allocation<StateBlock> block(raw_size);
    if (!block)
      return false;
    // allocate associated pages
    auto pages = static_cast<char *>(
        mmap(nullptr, config.pages_per_block * config.page_size,
             config.params.mmap_prot, config.params.mmap_flags, -1, 0));
    if (pages == MAP_FAILED)
      return false;
    // populate the block
    block->pages() = pages;
    size_t state_idx = 0;
    for (size_t p = 0; p < config.pages_per_block; ++p) {
      char *page = pages + p * config.page_size;
      for (size_t s = 0; s < config.states_per_page; ++s) {
        block->freelist(state_idx++) =
            page + s * config.params.size_of_opaque_state;
      }
    }
    // link the block to the sentinel
    block->next = sentinel.next;
    block->prev = &sentinel;
    block->next->prev = block.ptr;
    block->prev->next = block.ptr;
    // update the cursor
    free_cursor = block.ptr;
    free_count = state_idx;
    // release the ownership of allocation
    block.release();
    return true;
  }

public:
  void *get() {
    cpp::lock_guard guard{this->mutex};
    // if freelist is empty, allocate a new block
    if (is_empty() && !allocate_new_block())
      return nullptr;
    LIBC_ASSERT(free_count != 0);
    void *page = free_cursor->freelist(--free_count);
    // move cursor to the previous block if free_count is exhausted
    if (free_count == 0) {
      free_cursor = free_cursor->prev;
      free_count = config.states_per_page * config.pages_per_block;
    }
    return page;
  }
  size_t state_size() const { return config.params.size_of_opaque_state; }
  void recycle(void *state) {
    cpp::lock_guard guard{this->mutex};
    if (is_locally_full()) {
      free_cursor = free_cursor->next;
      free_count = 0;
    }
    free_cursor->freelist(free_count++) = static_cast<char *>(state);
  }
  ~MonotonicStatePool() {
    StateBlock *block = sentinel.next;
    size_t raw_size = state_block_raw_size();
    while (block != &sentinel) {
      munmap(block->pages(), config.pages_per_block * config.page_size);
      StateBlock *next = block->next;
      /* NOLINT(llvmlibc-callee-namespace) */ ::operator delete(
          block, raw_size, std::align_val_t{alignof(StateBlock)});
      block = next;
    }
  }
  static MonotonicStatePool *instance() {
    alignas(MonotonicStatePool) static char pool[sizeof(MonotonicStatePool)];
    static bool is_valid = false;
    static CallOnceFlag once_flag = callonce_impl::NOT_CALLED;
    callonce(&once_flag, []() {
      cpp::optional<GlobalConfig> config = GlobalConfig::get();
      if (!config)
        return;
      new (pool) MonotonicStatePool(*config);
      is_valid = /* NOLINT(llvmlibc-callee-namespace) */ !__cxa_atexit(
          [](void *) {
            reinterpret_cast<MonotonicStatePool *>(pool)->~MonotonicStatePool();
          },
          nullptr, __dso_handle);
    });
    if (!is_valid)
      return nullptr;
    return reinterpret_cast<MonotonicStatePool *>(pool);
  }
};

// We do not guarantee the correctness of calling the cprng functions in signal
// frames. However, we do want to make sure that an mistaken (maliciously
// induced) call to the cprng will never cause a state to be used twice due to
// reentrancy.
class ThreadLocalState {
  void *local_state;
  bool in_use;
  size_t remaining_trials;

  void destroy() {
    in_use = true;
    cpp::atomic_signal_fence(cpp::MemoryOrder::SEQ_CST);
    if (local_state)
      MonotonicStatePool::instance()->recycle(local_state);
  }

public:
  constexpr ThreadLocalState()
      : local_state(nullptr), in_use(false), remaining_trials(256) {}
  void *acquire() {
    if (in_use)
      return nullptr;
    in_use = true;
    cpp::atomic_signal_fence(cpp::MemoryOrder::SEQ_CST);
    if (local_state)
      return local_state;

    MonotonicStatePool *pool = MonotonicStatePool::instance();
    if (pool && remaining_trials) {
      remaining_trials--;
      local_state = pool->get();
      if (local_state) {
        if (/* NOLINT(llvmlibc-callee-namespace) */ __cxa_thread_atexit_impl(
                [](void *opaque) {
                  auto *state = static_cast<ThreadLocalState *>(opaque);
                  state->destroy();
                },
                this, __dso_handle) != 0) {
          pool->recycle(local_state);
          local_state = nullptr;
        }
      }
    }
    return local_state;
  }
  void release() {
    in_use = false;
    cpp::atomic_signal_fence(cpp::MemoryOrder::SEQ_CST);
  }
};
thread_local ThreadLocalState local_state{};

template <typename G>
size_t fill_buffer_impl(G generator, char *buffer, size_t length) {
  size_t filled = 0;
  while (filled < length) {
    int n = generator(&buffer[filled], length - filled);
    if (n < 0) {
      if (n == -EAGAIN || n == -EINTR)
        continue;
      break;
    }
    filled += static_cast<size_t>(n);
  }
  return filled;
}

} // namespace
size_t fill_buffer(char *buffer, size_t length) {
  void *state = local_state.acquire();
  if (state) {
    // Given state is valid, state_size shall be valid.
    size_t state_size = MonotonicStatePool::instance()->state_size();
    auto impl = [state, state_size](char *cursor, size_t len) {
      TypedSymbol<VDSOSym::GetRandom> vgetrandom;
      return vgetrandom(cursor, len, 0, state, state_size);
    };
    return fill_buffer_impl(impl, buffer, length);
  } else {
    auto impl = [](char *cursor, size_t len) {
      // use syscall to avoid errno handling
      return syscall_impl<int>(SYS_getrandom, cursor, len, 0);
    };
    return fill_buffer_impl(impl, buffer, length);
  }
  local_state.release();
}
} // namespace cprng
} // namespace LIBC_NAMESPACE_DECL
