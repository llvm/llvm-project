//===-- vDSO based RNG ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STDLIB_LINUX_VSDO_RNG_H
#define LIBC_SRC_STDLIB_LINUX_VSDO_RNG_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/OSUtil/linux/vdso.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/blockstore.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/mpmc_stack.h"
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/linux/raw_mutex.h"
#include "src/sys/auxv/getauxval.h"

namespace LIBC_NAMESPACE_DECL {
namespace vsdo_rng {
class GlobalState {
public:
  struct VGetrandomOpaqueParams {
    unsigned int size_of_opaque_states;
    unsigned int mmap_prot;
    unsigned int mmap_flags;
    unsigned int reserved[13];
  };

private:
  struct Config {
    size_t page_size;
    size_t pages_per_alloc;
    size_t states_per_page;
    vdso::VDSOSymType<vdso::VDSOSym::GetRandom> getrandom;
    VGetrandomOpaqueParams params;
  };

  // A lock-free stack of free opaque states.
  MPMCStack<void *> free_list{};
  // A mutex protecting the allocation of new pages.
  RawMutex allocation_mutex{};
  // A block store of allocated pages.
  BlockStore<void *, 16> allocations{};

  // Shared global configuration.
  static CallOnceFlag config_flag;
  static Config config;

  // We grow the states by the number of CPUs. This function uses
  // SYS_sched_getaffinity to get the number of CPUs.
  LIBC_INLINE static size_t cpu_count();

  // Grow available states. This function can fail if the system is out of
  // memory.
  LIBC_INLINE bool grow();

public:
  LIBC_INLINE constexpr GlobalState() {}
  LIBC_INLINE static Config &get_config();
  LIBC_INLINE ~GlobalState() {}
};

class LocalState {};

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

LIBC_INLINE GlobalState::Config &GlobalState::get_config() {
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

    size_t count = cpu_count();

    config.states_per_page =
        config.page_size / config.params.size_of_opaque_states;

    config.pages_per_alloc =
        count / config.states_per_page + (count % config.states_per_page != 0);
  });
  return config;
}

LIBC_INLINE bool GlobalState::grow() {
  // reserve a slot for the new page.
  if (!allocations.push_back(nullptr))
    return false;
}

} // namespace vsdo_rng
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_SRC_STDLIB_LINUX_VSDO_RNG_H
