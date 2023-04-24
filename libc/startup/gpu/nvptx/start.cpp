//===-- Implementation of crt for nvptx -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"

extern "C" int main(int argc, char **argv, char **envp);

namespace __llvm_libc {

static cpp::Atomic<uint32_t> lock = 0;

static cpp::Atomic<uint32_t> init = 0;

void init_rpc(void *in, void *out, void *buffer) {
  // Only a single thread should update the RPC data.
  if (gpu::get_thread_id() == 0 && gpu::get_block_id() == 0) {
    rpc::client.reset(&lock, in, out, buffer);
    init.store(1, cpp::MemoryOrder::RELAXED);
  }

  // Wait until the previous thread signals that the data has been written.
  while (!init.load(cpp::MemoryOrder::RELAXED))
    rpc::sleep_briefly();

  // Wait for the threads in the block to converge and fence the write.
  gpu::sync_threads();
}

} // namespace __llvm_libc

extern "C" [[gnu::visibility("protected"), clang::nvptx_kernel]] void
_start(int argc, char **argv, char **envp, int *ret, void *in, void *out,
       void *buffer) {
  __llvm_libc::init_rpc(in, out, buffer);

  __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);
}
