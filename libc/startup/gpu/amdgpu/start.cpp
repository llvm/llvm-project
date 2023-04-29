//===-- Implementation of crt for amdgpu ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"

extern "C" int main(int argc, char **argv, char **envp);

namespace __llvm_libc {

static cpp::Atomic<uint32_t> lock = 0;

static cpp::Atomic<uint32_t> count = 0;

extern "C" uintptr_t __init_array_start[];
extern "C" uintptr_t __init_array_end[];
extern "C" uintptr_t __fini_array_start[];
extern "C" uintptr_t __fini_array_end[];

using InitCallback = void(int, char **, char **);
using FiniCallback = void(void);

static uint64_t get_grid_size() {
  return gpu::get_num_threads() * gpu::get_num_blocks();
}

static void call_init_array_callbacks(int argc, char **argv, char **env) {
  size_t init_array_size = __init_array_end - __init_array_start;
  for (size_t i = 0; i < init_array_size; ++i)
    reinterpret_cast<InitCallback *>(__init_array_start[i])(argc, argv, env);
}

static void call_fini_array_callbacks() {
  size_t fini_array_size = __fini_array_end - __fini_array_start;
  for (size_t i = 0; i < fini_array_size; ++i)
    reinterpret_cast<FiniCallback *>(__fini_array_start[i])();
}

void initialize(int argc, char **argv, char **env, void *in, void *out,
                void *buffer) {
  // We need a single GPU thread to perform the initialization of the global
  // constructors and data. We simply mask off all but a single thread and
  // execute.
  count.fetch_add(1, cpp::MemoryOrder::RELAXED);
  if (gpu::get_thread_id() == 0 && gpu::get_block_id() == 0) {
    // We need to set up the RPC client first in case any of the constructors
    // require it.
    rpc::client.reset(&lock, in, out, buffer);

    // We want the fini array callbacks to be run after other atexit
    // callbacks are run. So, we register them before running the init
    // array callbacks as they can potentially register their own atexit
    // callbacks.
    atexit(&call_fini_array_callbacks);
    call_init_array_callbacks(argc, argv, env);
  }

  // We wait until every single thread launched on the GPU has seen the
  // initialization code. This will get very, very slow for high thread counts,
  // but for testing purposes it is unlikely to matter.
  while (count.load(cpp::MemoryOrder::RELAXED) != get_grid_size())
    rpc::sleep_briefly();
  gpu::sync_threads();
}

void finalize(int retval) {
  // We wait until every single thread launched on the GPU has finished
  // executing and reached the finalize region.
  count.fetch_sub(1, cpp::MemoryOrder::RELAXED);
  while (count.load(cpp::MemoryOrder::RELAXED) != 0)
    rpc::sleep_briefly();
  gpu::sync_threads();
  if (gpu::get_thread_id() == 0 && gpu::get_block_id() == 0) {
    // Only a single thread should call `exit` here, the rest should gracefully
    // return from the kernel. This is so only one thread calls the destructors
    // registred with 'atexit' above.
    __llvm_libc::exit(retval);
  }
}

} // namespace __llvm_libc

extern "C" [[gnu::visibility("protected"), clang::amdgpu_kernel]] void
_start(int argc, char **argv, char **envp, int *ret, void *in, void *out,
       void *buffer) {
  __llvm_libc::initialize(argc, argv, envp, in, out, buffer);

  __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);

  __llvm_libc::finalize(*ret);
}
