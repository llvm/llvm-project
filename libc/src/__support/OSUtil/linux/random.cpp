//===- Linux implementation of secure random buffer generation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/OSUtil/linux/random.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/OSUtil/linux/syscall.h"
#include "src/__support/OSUtil/linux/vdso.h"
#include "src/__support/libc_assert.h"
#include "src/__support/memory_size.h"
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/linux/callonce.h"
#include "src/__support/threads/linux/raw_mutex.h"
#include "src/errno/libc_errno.h"
#include "src/sched/sched_getaffinity.h"
#include "src/sched/sched_getcpucount.h"
#include "src/stdlib/atexit.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/random/getrandom.h"
#include "src/unistd/sysconf.h"
#include <asm/param.h>

namespace LIBC_NAMESPACE_DECL {
namespace {
// errno protection
struct ErrnoProtect {
  int backup;
  ErrnoProtect() : backup(libc_errno) { libc_errno = 0; }
  ~ErrnoProtect() { libc_errno = backup; }
};

// parameters for allocating per-thread random state
struct Params {
  unsigned size_of_opaque_state;
  unsigned mmap_prot;
  unsigned mmap_flags;
  unsigned reserved[13];
};

// for registering thread-specific atexit callbacks
using Destructor = void(void *);
extern "C" int __cxa_thread_atexit_impl(Destructor *, void *, void *);
extern "C" [[gnu::weak, gnu::visibility("hidden")]] void *__dso_handle =
    nullptr;

class MMapContainer {
  void **ptr = nullptr;
  void **usage = nullptr;
  void **boundary = nullptr;

  internal::SafeMemSize capacity() const {
    return internal::SafeMemSize{
        static_cast<size_t>(reinterpret_cast<ptrdiff_t>(boundary) -
                            reinterpret_cast<ptrdiff_t>(ptr))};
  }

  internal::SafeMemSize bytes() const {
    return capacity() * internal::SafeMemSize{sizeof(void *)};
  }

  bool initialize() {
    internal::SafeMemSize page_size{static_cast<size_t>(sysconf(_SC_PAGESIZE))};
    if (!page_size.valid())
      return false;
    ptr = reinterpret_cast<void **>(mmap(nullptr, page_size,
                                         PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (ptr == MAP_FAILED)
      return false;
    usage = ptr;
    boundary = ptr + page_size / sizeof(void *);
    return true;
  }

  bool grow(size_t additional) {
    if (ptr == nullptr)
      return initialize();

    size_t old_capacity = capacity();

    internal::SafeMemSize target_bytes{additional};
    internal::SafeMemSize new_bytes = bytes();
    target_bytes = target_bytes + size();
    target_bytes = target_bytes * internal::SafeMemSize{sizeof(void *)};

    if (!target_bytes.valid())
      return false;
    while (new_bytes < target_bytes) {
      new_bytes = new_bytes * internal::SafeMemSize{static_cast<size_t>(2)};
      if (!new_bytes.valid())
        return false;
    }

    // TODO: migrate to syscall wrapper once it's available
    auto result = syscall_impl<intptr_t>(
        SYS_mremap, bytes(), static_cast<size_t>(new_bytes), MREMAP_MAYMOVE);

    if (result < 0 && result > -EXEC_PAGESIZE)
      return false;
    ptr = reinterpret_cast<void **>(result);
    usage = ptr + old_capacity;
    boundary = ptr + new_bytes / sizeof(void *);
    return true;
  }

public:
  MMapContainer() = default;
  ~MMapContainer() {
    if (!ptr)
      return;
    munmap(ptr, bytes());
  }

  bool ensure_space(size_t additional) {
    if (usage + additional >= boundary && !grow(additional))
      return false;
    return true;
  }

  void push_unchecked(void *value) {
    LIBC_ASSERT(usage != boundary && "pushing into full container");
    *usage++ = value;
  }

  using iterator = void **;
  using value_type = void *;
  iterator begin() const { return ptr; }
  iterator end() const { return usage; }

  bool empty() const { return begin() == end(); }
  void *pop() {
    LIBC_ASSERT(!empty() && "popping from empty container");
    return *--usage;
  }
  internal::SafeMemSize size() const {
    return internal::SafeMemSize{static_cast<size_t>(
        reinterpret_cast<ptrdiff_t>(usage) - reinterpret_cast<ptrdiff_t>(ptr))};
  }
};

class StateFactory {
  RawMutex mutex{};
  MMapContainer allocations{};
  MMapContainer freelist{};
  Params params{};
  size_t states_per_page = 0;
  size_t pages_per_allocation = 0;
  size_t page_size = 0;

  bool prepare() {
    vdso::TypedSymbol<vdso::VDSOSym::GetRandom> vgetrandom;

    if (!vgetrandom)
      return false;

    // get the allocation configuration suggested by the kernel
    if (vgetrandom(nullptr, 0, 0, &params, ~0UL))
      return false;

    cpu_set_t cs{};

    if (LIBC_NAMESPACE::sched_getaffinity(0, sizeof(cs), &cs))
      return false;

    internal::SafeMemSize count{static_cast<size_t>(
        LIBC_NAMESPACE::__sched_getcpucount(sizeof(cs), &cs))};

    internal::SafeMemSize allocation_size =
        internal::SafeMemSize{
            static_cast<size_t>(params.size_of_opaque_state)} *
        count;

    page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    allocation_size = allocation_size.align_up(page_size);
    if (!allocation_size.valid())
      return false;

    states_per_page = page_size / params.size_of_opaque_state;
    pages_per_allocation = allocation_size / page_size;

    return true;
  }

  bool allocate_new_states() {
    if (!allocations.ensure_space(1))
      return false;

    // we always ensure the freelist can contain all the allocated states
    internal::SafeMemSize total_size =
        internal::SafeMemSize{page_size} *
        internal::SafeMemSize{pages_per_allocation} *
        (internal::SafeMemSize{static_cast<size_t>(1)} + allocations.size());

    if (!total_size.valid() ||
        !freelist.ensure_space(total_size - freelist.size()))
      return false;

    auto *new_allocation =
        static_cast<char *>(mmap(nullptr, page_size * pages_per_allocation,
                                 params.mmap_prot, params.mmap_flags, -1, 0));
    if (new_allocation == MAP_FAILED)
      return false;

    for (size_t i = 0; i < pages_per_allocation; ++i) {
      auto *page = new_allocation + i * page_size;
      for (size_t j = 0; j < states_per_page; ++j)
        freelist.push_unchecked(page + j * params.size_of_opaque_state);
    }
    return true;
  }

  static StateFactory *instance();

  void *acquire() {
    cpp::lock_guard guard{mutex};
    if (freelist.empty() && !allocate_new_states())
      return nullptr;
    return freelist.pop();
  }
  void release(void *state) {
    cpp::lock_guard guard{mutex};
    // there should be no need to check this pushing
    freelist.push_unchecked(state);
  }
  ~StateFactory() {
    for (auto *allocation : allocations)
      munmap(allocation, page_size * pages_per_allocation);
  }

public:
  static void *acquire_global() {
    auto *factory = instance();
    if (!factory)
      return nullptr;
    return factory->acquire();
  }
  static void release_global(void *state) {
    auto *factory = instance();
    if (!factory)
      return;
    factory->release(state);
  }
  static size_t size_of_opaque_state() {
    return instance()->params.size_of_opaque_state;
  }
  static void postfork_cleanup();
};

thread_local bool fork_inflight = false;
thread_local void *tls_state = nullptr;
alignas(StateFactory) static char factory_storage[sizeof(StateFactory)]{};
static CallOnceFlag factory_onceflag = callonce_impl::NOT_CALLED;
static bool factory_valid = false;

StateFactory *StateFactory::instance() {
  callonce(&factory_onceflag, []() {
    auto *factory = new (factory_storage) StateFactory();
    factory_valid = factory->prepare();
    if (factory_valid)
      atexit([]() {
        auto factory = reinterpret_cast<StateFactory *>(factory_storage);
        factory->~StateFactory();
        factory_valid = false;
      });
  });
  return factory_valid ? reinterpret_cast<StateFactory *>(factory_storage)
                       : nullptr;
}

void StateFactory::postfork_cleanup() {
  if (factory_valid)
    reinterpret_cast<StateFactory *>(factory_storage)->~StateFactory();
  factory_onceflag = callonce_impl::NOT_CALLED;
  factory_valid = false;
}

void *acquire_tls() {
  if (fork_inflight)
    return nullptr;
  // previous acquire failed, do not try again
  if (tls_state == MAP_FAILED)
    return nullptr;
  // first acquirement
  if (tls_state == nullptr) {
    tls_state = StateFactory::acquire_global();
    // if still fails, remember the failure
    if (tls_state == nullptr) {
      tls_state = MAP_FAILED;
      return nullptr;
    } else {
      // register the release callback.
      if (__cxa_thread_atexit_impl(
              [](void *s) { StateFactory::release_global(s); }, tls_state,
              __dso_handle)) {
        StateFactory::release_global(tls_state);
        tls_state = MAP_FAILED;
        return nullptr;
      }
    }
  }
  return tls_state;
}

template <class F> void random_fill_impl(F gen, void *buf, size_t size) {
  auto *buffer = reinterpret_cast<uint8_t *>(buf);
  while (size > 0) {
    ssize_t len = gen(buffer, size);
    if (len == -1) {
      if (libc_errno == EINTR)
        continue;
      break;
    }
    size -= len;
    buffer += len;
  }
}
} // namespace

void random_fill(void *buf, size_t size) {
  ErrnoProtect protect;
  void *state = acquire_tls();
  if (state) {
    random_fill_impl(
        [state](void *buf, size_t size) {
          vdso::TypedSymbol<vdso::VDSOSym::GetRandom> vgetrandom;
          int res = vgetrandom(buf, size, 0, state,
                               StateFactory::size_of_opaque_state());
          if (res < 0) {
            libc_errno = -res;
            return -1;
          }
          return res;
        },
        buf, size);
  } else {
    random_fill_impl(
        [](void *buf, size_t size) {
          return LIBC_NAMESPACE::getrandom(buf, size, 0);
        },
        buf, size);
  }
}

void random_prefork() { fork_inflight = true; }
void random_postfork_parent() { fork_inflight = false; }
void random_postfork_child() {
  tls_state = nullptr;
  StateFactory::postfork_cleanup();
  fork_inflight = false;
}

} // namespace LIBC_NAMESPACE_DECL
