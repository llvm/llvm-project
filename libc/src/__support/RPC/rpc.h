//===-- Shared memory RPC client / server interface -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_RPC_RPC_H
#define LLVM_LIBC_SRC_SUPPORT_RPC_RPC_H

#include "src/__support/CPP/atomic.h"

#include <stdint.h>

namespace __llvm_libc {
namespace rpc {

/// A list of opcodes that we use to invoke certain actions on the server. We
/// reserve the first 255 values for internal libc usage.
enum Opcode : uint64_t {
  NOOP = 0,
  PRINT_TO_STDERR = 1,
  EXIT = 2,
  LIBC_LAST = (1UL << 8) - 1,
};

/// A fixed size channel used to communicate between the RPC client and server.
struct Buffer {
  uint64_t data[8];
};

/// A common process used to synchronize communication between a client and a
/// server. The process contains an inbox and an outbox used for signaling
/// ownership of the shared buffer.
struct Process {
  cpp::Atomic<uint32_t> *inbox;
  cpp::Atomic<uint32_t> *outbox;
  Buffer *buffer;

  /// Initialize the communication channels.
  void reset(void *inbox, void *outbox, void *buffer) {
    *this = {
        reinterpret_cast<cpp::Atomic<uint32_t> *>(inbox),
        reinterpret_cast<cpp::Atomic<uint32_t> *>(outbox),
        reinterpret_cast<Buffer *>(buffer),
    };
  }
};

/// The RPC client used to make requests to the server.
struct Client : public Process {
  template <typename F, typename U> void run(F fill, U use);
};

/// The RPC server used to respond to the client.
struct Server : public Process {
  template <typename W, typename C> bool run(W work, C clean);
};

/// Run the RPC client protocol to communicate with the server. We perform the
/// following high level actions to complete a communication:
///   - Apply \p fill to the shared buffer and write 1 to the outbox.
///   - Wait until the inbox is 1.
///   - Apply \p use to the shared buffer and write 0 to the outbox.
///   - Wait until the inbox is 0.
template <typename F, typename U> void Client::run(F fill, U use) {
  bool in = inbox->load(cpp::MemoryOrder::RELAXED);
  bool out = outbox->load(cpp::MemoryOrder::RELAXED);
  atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  // Write to buffer then to the outbox.
  if (!in & !out) {
    fill(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(1, cpp::MemoryOrder::RELEASE);
    out = 1;
  }
  // Wait for the result from the server.
  if (!in & out) {
    while (!in)
      in = inbox->load(cpp::MemoryOrder::RELAXED);
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  }
  // Read from the buffer and then write to outbox.
  if (in & out) {
    use(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(0, cpp::MemoryOrder::RELEASE);
    out = 0;
  }
  // Wait for server to complete the communication.
  if (in & !out) {
    while (in)
      in = inbox->load(cpp::MemoryOrder::RELAXED);
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  }
}

/// Run the RPC server protocol to communicate with the client. This is
/// non-blocking and only checks the server a single time. We perform the
/// following high level actions to complete a communication:
///   - Query if the inbox is 1 and exit if there is no work to do.
///   - Apply \p work to the shared buffer and write 1 to the outbox.
///   - Wait until the inbox is 0.
///   - Apply \p clean to the shared buffer and write 0 to the outbox.
template <typename W, typename C> bool Server::run(W work, C clean) {
  bool in = inbox->load(cpp::MemoryOrder::RELAXED);
  bool out = outbox->load(cpp::MemoryOrder::RELAXED);
  atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  // No work to do, exit.
  if (!in & !out)
    return false;
  // Do work then write to the outbox.
  if (in & !out) {
    work(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(1, cpp::MemoryOrder::RELEASE);
    out = 1;
  }
  // Wait for the client to read the result.
  if (in & out) {
    while (in)
      in = inbox->load(cpp::MemoryOrder::RELAXED);
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  }
  // Clean up the buffer and signal the client.
  if (!in & out) {
    clean(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(0, cpp::MemoryOrder::RELEASE);
    out = 0;
  }

  return true;
}

} // namespace rpc
} // namespace __llvm_libc

#endif
