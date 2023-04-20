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
#include "src/__support/CPP/optional.h"
#include "src/__support/GPU/utils.h"
#include "src/string/memory_utils/memcpy_implementations.h"

#include <stdint.h>

namespace __llvm_libc {
namespace rpc {

/// A list of opcodes that we use to invoke certain actions on the server.
enum Opcode : uint16_t {
  NOOP = 0,
  PRINT_TO_STDERR = 1,
  EXIT = 2,
  TEST_INCREMENT = 3,
};

/// A fixed size channel used to communicate between the RPC client and server.
struct alignas(64) Buffer {
  uint8_t data[62];
  uint16_t opcode;
};
static_assert(sizeof(Buffer) == 64, "Buffer size mismatch");

/// A common process used to synchronize communication between a client and a
/// server. The process contains an inbox and an outbox used for signaling
/// ownership of the shared buffer between both sides.
///
/// This process is designed to support mostly arbitrary combinations of 'send'
/// and 'recv' operations on the shared buffer as long as these operations are
/// mirrored by the other process. These operations exchange ownership of the
/// fixed-size buffer between the users of the protocol. The assumptions when
/// using this process are as follows:
///   - The client will always start with a 'send' operation
///   - The server will always start with a 'recv' operation
///   - For every 'send' / 'recv' call on one side of the process there is a
///     mirrored 'recv' / 'send' call.
///
/// The communication protocol is organized as a pair of two-state state
/// machines. One state machine tracks outgoing sends and the other tracks
/// incoming receives. For example, a 'send' operation uses its input 'Ack' bit
/// and its output 'Data' bit. If these bits are equal the sender owns the
/// buffer, otherwise the receiver owns the buffer and we wait. Similarly, a
/// 'recv' operation uses its output 'Ack' bit and input 'Data' bit. If these
/// bits are not equal the receiver owns the buffer, otherwise the sender owns
/// the buffer.
struct Process {
  LIBC_INLINE Process() = default;
  LIBC_INLINE Process(const Process &) = default;
  LIBC_INLINE Process &operator=(const Process &) = default;
  LIBC_INLINE ~Process() = default;

  static constexpr uint32_t Data = 0b01;
  static constexpr uint32_t Ack = 0b10;

  cpp::Atomic<uint32_t> *lock;
  cpp::Atomic<uint32_t> *inbox;
  cpp::Atomic<uint32_t> *outbox;
  Buffer *buffer;

  /// Initialize the communication channels.
  LIBC_INLINE void reset(void *lock, void *inbox, void *outbox, void *buffer) {
    *this = {
        reinterpret_cast<cpp::Atomic<uint32_t> *>(lock),
        reinterpret_cast<cpp::Atomic<uint32_t> *>(inbox),
        reinterpret_cast<cpp::Atomic<uint32_t> *>(outbox),
        reinterpret_cast<Buffer *>(buffer),
    };
  }

  /// Determines if this process owns the buffer for a send. We can send data if
  /// the output data bit matches the input acknowledge bit.
  LIBC_INLINE static bool can_send_data(uint32_t in, uint32_t out) {
    return bool(in & Process::Ack) == bool(out & Process::Data);
  }

  /// Determines if this process owns the buffer for a receive. We can send data
  /// if the output acknowledge bit does not match the input data bit.
  LIBC_INLINE static bool can_recv_data(uint32_t in, uint32_t out) {
    return bool(in & Process::Data) != bool(out & Process::Ack);
  }
};

/// The port provides the interface to communicate between the multiple
/// processes. A port is conceptually an index into the memory provided by the
/// underlying process that is guarded by a lock bit.
struct Port {
  // TODO: This should be move-only.
  LIBC_INLINE Port(Process &process, uint64_t index, uint32_t out)
      : process(process), index(index), out(out) {}
  LIBC_INLINE Port(const Port &) = default;
  LIBC_INLINE Port &operator=(const Port &) = delete;
  LIBC_INLINE ~Port() = default;

  template <typename U> LIBC_INLINE void recv(U use);
  template <typename F> LIBC_INLINE void send(F fill);
  template <typename F, typename U>
  LIBC_INLINE void send_and_recv(F fill, U use);
  template <typename W> LIBC_INLINE void recv_and_send(W work);
  LIBC_INLINE void send_n(const void *src, uint64_t size);
  template <typename A> LIBC_INLINE void recv_n(A alloc);

  LIBC_INLINE uint16_t get_opcode() const {
    return process.buffer[index].opcode;
  }

  LIBC_INLINE void close() {
    process.lock[index].store(0, cpp::MemoryOrder::RELAXED);
  }

private:
  Process &process;
  uint64_t index;
  uint32_t out;
};

/// The RPC client used to make requests to the server.
struct Client : public Process {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = default;
  LIBC_INLINE Client &operator=(const Client &) = default;
  LIBC_INLINE ~Client() = default;

  LIBC_INLINE cpp::optional<Port> try_open(uint16_t opcode);
  LIBC_INLINE Port open(uint16_t opcode);
};

/// The RPC server used to respond to the client.
struct Server : public Process {
  LIBC_INLINE Server() = default;
  LIBC_INLINE Server(const Server &) = default;
  LIBC_INLINE Server &operator=(const Server &) = default;
  LIBC_INLINE ~Server() = default;

  LIBC_INLINE cpp::optional<Port> try_open();
  LIBC_INLINE Port open();
};

/// Applies \p fill to the shared buffer and initiates a send operation.
template <typename F> LIBC_INLINE void Port::send(F fill) {
  uint32_t in = process.inbox[index].load(cpp::MemoryOrder::RELAXED);

  // We need to wait until we own the buffer before sending.
  while (!Process::can_send_data(in, out)) {
    sleep_briefly();
    in = process.inbox[index].load(cpp::MemoryOrder::RELAXED);
  }

  // Apply the \p fill function to initialize the buffer and release the memory.
  fill(&process.buffer[index]);
  out = out ^ Process::Data;
  atomic_thread_fence(cpp::MemoryOrder::RELEASE);
  process.outbox[index].store(out, cpp::MemoryOrder::RELAXED);
}

/// Applies \p use to the shared buffer and acknowledges the send.
template <typename U> LIBC_INLINE void Port::recv(U use) {
  uint32_t in = process.inbox[index].load(cpp::MemoryOrder::RELAXED);

  // We need to wait until we own the buffer before receiving.
  while (!Process::can_recv_data(in, out)) {
    sleep_briefly();
    in = process.inbox[index].load(cpp::MemoryOrder::RELAXED);
  }
  atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);

  // Apply the \p use function to read the memory out of the buffer.
  use(&process.buffer[index]);
  out = out ^ Process::Ack;
  process.outbox[index].store(out, cpp::MemoryOrder::RELAXED);
}

/// Combines a send and receive into a single function.
template <typename F, typename U>
LIBC_INLINE void Port::send_and_recv(F fill, U use) {
  send(fill);
  recv(use);
}

/// Combines a receive and send operation into a single function. The \p work
/// function modifies the buffer in-place and the send is only used to initiate
/// the copy back.
template <typename W> LIBC_INLINE void Port::recv_and_send(W work) {
  recv(work);
  send([](Buffer *) { /* no-op */ });
}

/// Sends an arbitrarily sized data buffer \p src across the shared channel in
/// multiples of the packet length.
LIBC_INLINE void Port::send_n(const void *src, uint64_t size) {
  // TODO: We could send the first bytes in this call and potentially save an
  // extra send operation.
  send([=](Buffer *buffer) { buffer->data[0] = size; });
  const uint8_t *ptr = reinterpret_cast<const uint8_t *>(src);
  for (uint64_t idx = 0; idx < size; idx += sizeof(Buffer::data)) {
    send([=](Buffer *buffer) {
      const uint64_t len =
          size - idx > sizeof(Buffer::data) ? sizeof(Buffer::data) : size - idx;
      inline_memcpy(buffer->data, ptr + idx, len);
    });
  }
}

/// Receives an arbitrarily sized data buffer across the shared channel in
/// multiples of the packet length. The \p alloc function is called with the
/// size of the data so that we can initialize the size of the \p dst buffer.
template <typename A> LIBC_INLINE void Port::recv_n(A alloc) {
  uint64_t size = 0;
  recv([&](Buffer *buffer) { size = buffer->data[0]; });
  uint8_t *dst = reinterpret_cast<uint8_t *>(alloc(size));
  for (uint64_t idx = 0; idx < size; idx += sizeof(Buffer::data)) {
    recv([=](Buffer *buffer) {
      uint64_t len =
          size - idx > sizeof(Buffer::data) ? sizeof(Buffer::data) : size - idx;
      inline_memcpy(dst + idx, buffer->data, len);
    });
  }
}

/// Attempts to open a port to use as the client. The client can only open a
/// port if we find an index that is in a valid sending state. That is, there
/// are send operations pending that haven't been serviced on this port. Each
/// port instance uses an associated \p opcode to tell the server what to do.
LIBC_INLINE cpp::optional<Port> Client::try_open(uint16_t opcode) {
  // Attempt to acquire the lock on this index.
  if (lock->fetch_or(1, cpp::MemoryOrder::RELAXED))
    return cpp::nullopt;

  uint32_t in = inbox->load(cpp::MemoryOrder::RELAXED);
  uint32_t out = outbox->load(cpp::MemoryOrder::RELAXED);

  // Once we acquire the index we need to check if we are in a valid sending
  // state.
  if (!can_send_data(in, out)) {
    lock->store(0, cpp::MemoryOrder::RELAXED);
    return cpp::nullopt;
  }

  buffer->opcode = opcode;
  return Port(*this, 0, out);
}

LIBC_INLINE Port Client::open(uint16_t opcode) {
  for (;;) {
    if (cpp::optional<Port> p = try_open(opcode))
      return p.value();
    sleep_briefly();
  }
}

/// Attempts to open a port to use as the server. The server can only open a
/// port if it has a pending receive operation
LIBC_INLINE cpp::optional<Port> Server::try_open() {
  uint32_t in = inbox->load(cpp::MemoryOrder::RELAXED);
  uint32_t out = outbox->load(cpp::MemoryOrder::RELAXED);

  // The server is passive, if there is no work pending don't bother
  // opening a port.
  if (!can_recv_data(in, out))
    return cpp::nullopt;

  // Attempt to acquire the lock on this index.
  if (lock->fetch_or(1, cpp::MemoryOrder::RELAXED))
    return cpp::nullopt;

  in = inbox->load(cpp::MemoryOrder::RELAXED);
  out = outbox->load(cpp::MemoryOrder::RELAXED);

  if (!can_recv_data(in, out)) {
    lock->store(0, cpp::MemoryOrder::RELAXED);
    return cpp::nullopt;
  }

  return Port(*this, 0, out);
}

LIBC_INLINE Port Server::open() {
  for (;;) {
    if (cpp::optional<Port> p = try_open())
      return p.value();
    sleep_briefly();
  }
}

} // namespace rpc
} // namespace __llvm_libc

#endif
