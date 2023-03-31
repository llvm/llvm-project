//===-- Shared memory RPC client / server interface -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a remote procedure call mechanism to communicate between
// heterogeneous devices that can share an address space atomically. We provide
// a client and a server to facilitate the remote call. The client makes request
// to the server using a shared communication channel. We use separate atomic
// signals to indicate which side, the client or the server is in ownership of
// the buffer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_RPC_RPC_H
#define LLVM_LIBC_SRC_SUPPORT_RPC_RPC_H

#include "rpc_util.h"
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
  LIBC_INLINE Process() = default;
  LIBC_INLINE Process(const Process &) = default;
  LIBC_INLINE Process &operator=(const Process &) = default;
  LIBC_INLINE ~Process() = default;

  cpp::Atomic<uint32_t> *inbox;
  cpp::Atomic<uint32_t> *outbox;
  Buffer *buffer;

  /// Initialize the communication channels.
  LIBC_INLINE void reset(void *inbox, void *outbox, void *buffer) {
    *this = {
        reinterpret_cast<cpp::Atomic<uint32_t> *>(inbox),
        reinterpret_cast<cpp::Atomic<uint32_t> *>(outbox),
        reinterpret_cast<Buffer *>(buffer),
    };
  }
};

/// The RPC client used to make requests to the server.
struct Client : public Process {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = default;
  LIBC_INLINE Client &operator=(const Client &) = default;
  LIBC_INLINE ~Client() = default;

  template <typename F, typename U> LIBC_INLINE void run(F fill, U use);
};

/// The RPC server used to respond to the client.
struct Server : public Process {
  LIBC_INLINE Server() = default;
  LIBC_INLINE Server(const Server &) = default;
  LIBC_INLINE Server &operator=(const Server &) = default;
  LIBC_INLINE ~Server() = default;

  template <typename W, typename C> LIBC_INLINE bool handle(W work, C clean);
};

/// Run the RPC client protocol to communicate with the server. We perform the
/// following high level actions to complete a communication:
///   - Apply \p fill to the shared buffer and write 1 to the outbox.
///   - Wait until the inbox is 1.
///   - Apply \p use to the shared buffer and write 0 to the outbox.
///   - Wait until the inbox is 0.
template <typename F, typename U> LIBC_INLINE void Client::run(F fill, U use) {
  bool in = inbox->load(cpp::MemoryOrder::RELAXED);
  bool out = outbox->load(cpp::MemoryOrder::RELAXED);
  atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  // Apply the \p fill to the buffer and signal the server.
  if (!in & !out) {
    fill(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(1, cpp::MemoryOrder::RELAXED);
    out = 1;
  }
  // Wait for the server to work on the buffer and respond.
  if (!in & out) {
    while (!in) {
      sleep_briefly();
      in = inbox->load(cpp::MemoryOrder::RELAXED);
    }
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  }
  // Apply \p use to the buffer and signal the server.
  if (in & out) {
    use(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(0, cpp::MemoryOrder::RELAXED);
    out = 0;
  }
  // Wait for the server to signal the end of the protocol.
  if (in & !out) {
    while (in) {
      sleep_briefly();
      in = inbox->load(cpp::MemoryOrder::RELAXED);
    }
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
template <typename W, typename C>
LIBC_INLINE bool Server::handle(W work, C clean) {
  bool in = inbox->load(cpp::MemoryOrder::RELAXED);
  bool out = outbox->load(cpp::MemoryOrder::RELAXED);
  atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  // There is no work to do, exit early.
  if (!in & !out)
    return false;
  // Apply \p work to the buffer and signal the client.
  if (in & !out) {
    work(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(1, cpp::MemoryOrder::RELAXED);
    out = 1;
  }
  // Wait for the client to use the buffer and respond.
  if (in & out) {
    while (in)
      in = inbox->load(cpp::MemoryOrder::RELAXED);
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  }
  // Clean up the buffer and signal the end of the protocol.
  if (!in & out) {
    clean(buffer);
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox->store(0, cpp::MemoryOrder::RELAXED);
    out = 0;
  }

  return true;
}

} // namespace rpc
} // namespace __llvm_libc

#endif
