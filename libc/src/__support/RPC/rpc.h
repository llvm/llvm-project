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
#include "src/__support/CPP/functional.h"
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
  TEST_INTERFACE = 4,
  TEST_STREAM = 5,
};

/// A fixed size channel used to communicate between the RPC client and server.
struct Buffer {
  uint64_t data[8];
};
static_assert(sizeof(Buffer) == 64, "Buffer size mismatch");

/// The information associated with a packet. This indicates which operations to
/// perform and which threads are active in the slots.
struct Header {
  uint64_t mask;
  uint16_t opcode;
};

/// The data payload for the associated packet. We provide enough space for each
/// thread in the cooperating lane to have a buffer.
struct Payload {
#if defined(LIBC_TARGET_ARCH_IS_GPU)
  Buffer slot[gpu::LANE_SIZE];
#else
  // Flexible array size allocated at runtime to the appropriate size.
  Buffer slot[];
#endif
};

/// A packet used to share data between the client and server across an entire
/// lane. We use a lane as the minimum granularity for execution.
struct alignas(64) Packet {
  Header header;
  Payload payload;
};

// TODO: This should be configured by the server and passed in. The general rule
//       of thumb is that you should have at least as many ports as possible
//       concurrent work items on the GPU to mitigate the lack offorward
//       progress guarantees on the GPU.
constexpr uint64_t DEFAULT_PORT_COUNT = 64;

/// A common process used to synchronize communication between a client and a
/// server. The process contains an inbox and an outbox used for signaling
/// ownership of the shared buffer between both sides.
///
/// No process writes to its inbox. Each toggles the bit in the outbox to pass
/// ownership to the other process.
/// When inbox == outbox, the current state machine owns the buffer.
/// Initially the client is able to open any port as it will load 0 from both.
/// The server inbox read is inverted, so it loads inbox==1, outbox==0 until
/// the client has written to its outbox.
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
template <bool InvertInbox> struct Process {
  LIBC_INLINE Process() = default;
  LIBC_INLINE Process(const Process &) = delete;
  LIBC_INLINE Process &operator=(const Process &) = delete;
  LIBC_INLINE Process(Process &&) = default;
  LIBC_INLINE Process &operator=(Process &&) = default;
  LIBC_INLINE ~Process() = default;

  uint64_t port_count;
  uint32_t lane_size;
  cpp::Atomic<uint32_t> *inbox;
  cpp::Atomic<uint32_t> *outbox;
  Packet *packet;

  cpp::Atomic<uint32_t> lock[DEFAULT_PORT_COUNT] = {0};

  /// Initialize the communication channels.
  LIBC_INLINE void reset(uint64_t port_count, uint32_t lane_size, void *state) {
    uint64_t p = memory_offset_primary_mailbox(port_count);
    uint64_t s = memory_offset_secondary_mailbox(port_count);
    this->port_count = port_count;
    this->lane_size = lane_size;
    this->inbox = reinterpret_cast<cpp::Atomic<uint32_t> *>(
        static_cast<char *>(state) + (InvertInbox ? s : p));
    this->outbox = reinterpret_cast<cpp::Atomic<uint32_t> *>(
        static_cast<char *>(state) + (InvertInbox ? p : s));
    this->packet = reinterpret_cast<Packet *>(static_cast<char *>(state) +
                                              memory_offset_buffer(port_count));
  }

  /// Allocate a single block of memory for use by client and server
  /// template<size_t N>, N is generally a runtime value
  /// struct equivalent {
  ///   atomic<uint32_t> primary[N];
  ///   atomic<uint32_t> secondary[N];
  ///   Packet buffer[N];
  /// };
  LIBC_INLINE static uint64_t allocation_size(uint64_t port_count,
                                              uint32_t lane_size) {
    return memory_offset_buffer(port_count) +
           memory_allocated_buffer(port_count, lane_size);
  }

  /// The length of the packet is flexible because the server needs to look up
  /// the lane size at runtime. This helper indexes at the proper offset.
  LIBC_INLINE Packet &get_packet(uint64_t index) {
    return *reinterpret_cast<Packet *>(
        reinterpret_cast<uint8_t *>(packet) +
        index * align_up(sizeof(Header) + lane_size * sizeof(Buffer),
                         alignof(Packet)));
  }

  /// Inverting the bits loaded from the inbox in exactly one of the pair of
  /// processes means that each can use the same state transitions.
  /// Whichever process has InvertInbox==false is the initial owner.
  /// Inbox equal Outbox => current process owns the buffer
  /// Inbox difer Outbox => current process does not own the buffer
  /// At startup, memory is zero initialised and raw loads of either mailbox
  /// would return zero. Thus both would succeed in opening a port and data
  /// races result. If either inbox or outbox is inverted for one process, that
  /// process interprets memory as Inbox!=Outbox and thus waits for the other.
  /// It is simpler to invert reads from the inbox than writes to the outbox.
  LIBC_INLINE uint32_t load_inbox(uint64_t index) {
    uint32_t i = inbox[index].load(cpp::MemoryOrder::RELAXED);
    return InvertInbox ? !i : i;
  }

  /// Retrieve the outbox state from memory shared between processes.
  /// Never needs to invert the associated read.
  LIBC_INLINE uint32_t load_outbox(uint64_t index) {
    return outbox[index].load(cpp::MemoryOrder::RELAXED);
  }

  /// Signal to the other process that this one is finished with the buffer.
  /// Equivalent to loading outbox followed by store of the inverted value
  /// The outbox is write only by this warp and tracking the value locally is
  /// cheaper than calling load_outbox to get the value to store.
  LIBC_INLINE uint32_t invert_outbox(uint64_t index, uint32_t current_outbox) {
    uint32_t inverted_outbox = !current_outbox;
    outbox[index].store(inverted_outbox, cpp::MemoryOrder::RELAXED);
    return inverted_outbox;
  }

  /// Determines if this process needs to wait for ownership of the buffer.
  LIBC_INLINE static bool buffer_unavailable(uint32_t in, uint32_t out) {
    return in != out;
  }

  /// Attempt to claim the lock at index. Return true on lock taken.
  /// lane_mask is a bitmap of the threads in the warp that would hold the
  /// single lock on success, e.g. the result of gpu::get_lane_mask()
  /// The lock is held when the zeroth bit of the uint32_t at lock[index]
  /// is set, and available when that bit is clear. Bits [1, 32) are zero.
  /// Or with one is a no-op when the lock is already held.
  [[clang::convergent]] LIBC_INLINE bool try_lock(uint64_t lane_mask,
                                                  uint64_t index) {
    // On amdgpu, test and set to lock[index] and a sync_lane would suffice
    // On volta, need to handle differences between the threads running and
    // the threads that were detected in the previous call to get_lane_mask()
    //
    // All threads in lane_mask try to claim the lock. At most one can succeed.
    // There may be threads active which are not in lane mask which must not
    // succeed in taking the lock, as otherwise it will leak. This is handled
    // by making threads which are not in lane_mask or with 0, a no-op.
    uint32_t id = gpu::get_lane_id();
    bool id_in_lane_mask = lane_mask & (1ul << id);

    // All threads in the warp call fetch_or. Possibly at the same time.
    bool before =
        lock[index].fetch_or(id_in_lane_mask, cpp::MemoryOrder::RELAXED);
    uint64_t packed = gpu::ballot(lane_mask, before);

    // If every bit set in lane_mask is also set in packed, every single thread
    // in the warp failed to get the lock. Ballot returns unset for threads not
    // in the lane mask.
    //
    // Cases, per thread:
    // mask==0 -> unspecified before, discarded by ballot -> 0
    // mask==1 and before==0 (success), set zero by ballot -> 0
    // mask==1 and before==1 (failure), set one by ballot -> 1
    //
    // mask != packed implies at least one of the threads got the lock
    // atomic semantics of fetch_or mean at most one of the threads for the lock
    return lane_mask != packed;
  }

  /// Unlock the lock at index. We need a lane sync to keep this function
  /// convergent, otherwise the compiler will sink the store and deadlock.
  [[clang::convergent]] LIBC_INLINE void unlock(uint64_t lane_mask,
                                                uint64_t index) {
    // Wait for other threads in the warp to finish using the lock
    gpu::sync_lane(lane_mask);

    // Use exactly one thread to clear the bit at position 0 in lock[index]
    // Must restrict to a single thread to avoid one thread dropping the lock,
    // then an unrelated warp claiming the lock, then a second thread in this
    // warp dropping the lock again.
    uint32_t and_mask = ~(rpc::is_first_lane(lane_mask) ? 1 : 0);
    lock[index].fetch_and(and_mask, cpp::MemoryOrder::RELAXED);
    gpu::sync_lane(lane_mask);
  }

  /// Invokes a function accross every active buffer across the total lane size.
  LIBC_INLINE void invoke_rpc(cpp::function<void(Buffer *)> fn,
                              Packet &packet) {
    if constexpr (is_process_gpu()) {
      fn(&packet.payload.slot[gpu::get_lane_id()]);
    } else {
      for (uint32_t i = 0; i < lane_size; i += gpu::get_lane_size())
        if (packet.header.mask & 1ul << i)
          fn(&packet.payload.slot[i]);
    }
  }

  /// Alternate version that also provides the index of the current lane.
  LIBC_INLINE void invoke_rpc(cpp::function<void(Buffer *, uint32_t)> fn,
                              Packet &packet) {
    if constexpr (is_process_gpu()) {
      fn(&packet.payload.slot[gpu::get_lane_id()], gpu::get_lane_id());
    } else {
      for (uint32_t i = 0; i < lane_size; i += gpu::get_lane_size())
        if (packet.header.mask & 1ul << i)
          fn(&packet.payload.slot[i], i);
    }
  }

  /// Number of bytes allocated for mailbox or buffer
  LIBC_INLINE static uint64_t memory_allocated_mailbox(uint64_t port_count) {
    return port_count * sizeof(cpp::Atomic<uint32_t>);
  }

  LIBC_INLINE static uint64_t memory_allocated_buffer(uint64_t port_count,
                                                      uint32_t lane_size) {
#if defined(LIBC_TARGET_ARCH_IS_GPU)
    (void)lane_size;
    return port_count * sizeof(Packet);
#else
    return port_count * (sizeof(Packet) + sizeof(Buffer) * lane_size);
#endif
  }

  /// Offset of mailbox/buffer in single allocation
  LIBC_INLINE static uint64_t
  memory_offset_primary_mailbox(uint64_t /*port_count*/) {
    return 0;
  }
  LIBC_INLINE static uint64_t
  memory_offset_secondary_mailbox(uint64_t port_count) {
    return memory_allocated_mailbox(port_count);
  }
  LIBC_INLINE static uint64_t memory_offset_buffer(uint64_t port_count) {
    return align_up(2 * memory_allocated_mailbox(port_count), alignof(Packet));
  }
};

/// The port provides the interface to communicate between the multiple
/// processes. A port is conceptually an index into the memory provided by the
/// underlying process that is guarded by a lock bit.
template <bool T> struct Port {
  LIBC_INLINE Port(Process<T> &process, uint64_t lane_mask, uint64_t index,
                   uint32_t out)
      : process(process), lane_mask(lane_mask), index(index), out(out),
        receive(false) {}
  LIBC_INLINE ~Port() = default;

private:
  LIBC_INLINE Port(const Port &) = delete;
  LIBC_INLINE Port &operator=(const Port &) = delete;
  LIBC_INLINE Port(Port &&) = default;
  LIBC_INLINE Port &operator=(Port &&) = default;

  friend struct Client;
  friend struct Server;
  friend class cpp::optional<Port<T>>;

public:
  template <typename U> LIBC_INLINE void recv(U use);
  template <typename F> LIBC_INLINE void send(F fill);
  template <typename F, typename U>
  LIBC_INLINE void send_and_recv(F fill, U use);
  template <typename W> LIBC_INLINE void recv_and_send(W work);
  LIBC_INLINE void send_n(const void *const *src, uint64_t *size);
  LIBC_INLINE void send_n(const void *src, uint64_t size);
  template <typename A>
  LIBC_INLINE void recv_n(void **dst, uint64_t *size, A &&alloc);

  LIBC_INLINE uint16_t get_opcode() const {
    return process.get_packet(index).header.opcode;
  }

  LIBC_INLINE void close() {
    // If the server last did a receive it needs to exchange ownership before
    // closing the port.
    if (receive && T)
      out = process.invert_outbox(index, out);
    process.unlock(lane_mask, index);
  }

private:
  Process<T> &process;
  uint64_t lane_mask;
  uint64_t index;
  uint32_t out;
  bool receive;
};

/// The RPC client used to make requests to the server.
struct Client : public Process<false> {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = delete;
  LIBC_INLINE Client &operator=(const Client &) = delete;
  LIBC_INLINE ~Client() = default;

  using Port = rpc::Port<false>;
  template <uint16_t opcode> LIBC_INLINE cpp::optional<Port> try_open();
  template <uint16_t opcode> LIBC_INLINE Port open();
};

/// The RPC server used to respond to the client.
struct Server : public Process<true> {
  LIBC_INLINE Server() = default;
  LIBC_INLINE Server(const Server &) = delete;
  LIBC_INLINE Server &operator=(const Server &) = delete;
  LIBC_INLINE ~Server() = default;

  using Port = rpc::Port<true>;
  LIBC_INLINE cpp::optional<Port> try_open();
  LIBC_INLINE Port open();
};

/// Applies \p fill to the shared buffer and initiates a send operation.
template <bool T> template <typename F> LIBC_INLINE void Port<T>::send(F fill) {
  uint32_t in = process.load_inbox(index);

  // We need to wait until we own the buffer before sending.
  while (Process<T>::buffer_unavailable(in, out)) {
    sleep_briefly();
    in = process.load_inbox(index);
  }

  // Apply the \p fill function to initialize the buffer and release the memory.
  process.invoke_rpc(fill, process.get_packet(index));
  atomic_thread_fence(cpp::MemoryOrder::RELEASE);
  out = process.invert_outbox(index, out);
  receive = false;
}

/// Applies \p use to the shared buffer and acknowledges the send.
template <bool T> template <typename U> LIBC_INLINE void Port<T>::recv(U use) {
  // We only exchange ownership of the buffer during a receive if we are waiting
  // for a previous receive to finish.
  if (receive)
    out = process.invert_outbox(index, out);

  uint32_t in = process.load_inbox(index);

  // We need to wait until we own the buffer before receiving.
  while (Process<T>::buffer_unavailable(in, out)) {
    sleep_briefly();
    in = process.load_inbox(index);
  }
  atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);

  // Apply the \p use function to read the memory out of the buffer.
  process.invoke_rpc(use, process.get_packet(index));
  receive = true;
}

/// Combines a send and receive into a single function.
template <bool T>
template <typename F, typename U>
LIBC_INLINE void Port<T>::send_and_recv(F fill, U use) {
  send(fill);
  recv(use);
}

/// Combines a receive and send operation into a single function. The \p work
/// function modifies the buffer in-place and the send is only used to initiate
/// the copy back.
template <bool T>
template <typename W>
LIBC_INLINE void Port<T>::recv_and_send(W work) {
  recv(work);
  send([](Buffer *) { /* no-op */ });
}

/// Sends an arbitrarily sized data buffer \p src across the shared channel in
/// multiples of the packet length.
template <bool T>
LIBC_INLINE void Port<T>::send_n(const void *const *src, uint64_t *size) {
  // TODO: We could send the first bytes in this call and potentially save an
  // extra send operation.
  // TODO: We may need a way for the CPU to send different strings per thread.
  uint64_t num_sends = 0;
  send([&](Buffer *buffer, uint32_t id) {
    reinterpret_cast<uint64_t *>(buffer->data)[0] = lane_value(size, id);
    num_sends = is_process_gpu() ? lane_value(size, id)
                                 : max(lane_value(size, id), num_sends);
  });
  for (uint64_t idx = 0; idx < num_sends; idx += sizeof(Buffer::data)) {
    send([=](Buffer *buffer, uint32_t id) {
      const uint64_t len = lane_value(size, id) - idx > sizeof(Buffer::data)
                               ? sizeof(Buffer::data)
                               : lane_value(size, id) - idx;
      if (idx < lane_value(size, id))
        inline_memcpy(
            buffer->data,
            reinterpret_cast<const uint8_t *>(lane_value(src, id)) + idx, len);
    });
  }
  gpu::sync_lane(process.get_packet(index).header.mask);
}

/// Helper routine to simplify the interface when sending from the GPU using
/// thread private pointers to the underlying value.
template <bool T>
LIBC_INLINE void Port<T>::send_n(const void *src, uint64_t size) {
  static_assert(is_process_gpu(), "Only valid when running on the GPU");
  const void **src_ptr = &src;
  uint64_t *size_ptr = &size;
  send_n(src_ptr, size_ptr);
}

/// Receives an arbitrarily sized data buffer across the shared channel in
/// multiples of the packet length. The \p alloc function is called with the
/// size of the data so that we can initialize the size of the \p dst buffer.
template <bool T>
template <typename A>
LIBC_INLINE void Port<T>::recv_n(void **dst, uint64_t *size, A &&alloc) {
  uint64_t num_recvs = 0;
  recv([&](Buffer *buffer, uint32_t id) {
    lane_value(size, id) = reinterpret_cast<uint64_t *>(buffer->data)[0];
    lane_value(dst, id) =
        reinterpret_cast<uint8_t *>(alloc(lane_value(size, id)));
    num_recvs = is_process_gpu() ? lane_value(size, id)
                                 : max(lane_value(size, id), num_recvs);
  });
  for (uint64_t idx = 0; idx < num_recvs; idx += sizeof(Buffer::data)) {
    recv([=](Buffer *buffer, uint32_t id) {
      uint64_t len = lane_value(size, id) - idx > sizeof(Buffer::data)
                         ? sizeof(Buffer::data)
                         : lane_value(size, id) - idx;
      if (idx < lane_value(size, id))
        inline_memcpy(reinterpret_cast<uint8_t *>(lane_value(dst, id)) + idx,
                      buffer->data, len);
    });
  }
  return;
}

/// Attempts to open a port to use as the client. The client can only open a
/// port if we find an index that is in a valid sending state. That is, there
/// are send operations pending that haven't been serviced on this port. Each
/// port instance uses an associated \p opcode to tell the server what to do.
template <uint16_t opcode>
[[clang::convergent]] LIBC_INLINE cpp::optional<Client::Port>
Client::try_open() {
  // Perform a naive linear scan for a port that can be opened to send data.
  for (uint64_t index = 0; index < port_count; ++index) {
    // Attempt to acquire the lock on this index.
    uint64_t lane_mask = gpu::get_lane_mask();
    if (!try_lock(lane_mask, index))
      continue;

    // The mailbox state must be read with the lock held.
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);

    uint32_t in = load_inbox(index);
    uint32_t out = load_outbox(index);

    // Once we acquire the index we need to check if we are in a valid sending
    // state.
    if (buffer_unavailable(in, out)) {
      unlock(lane_mask, index);
      continue;
    }

    if (is_first_lane(lane_mask)) {
      get_packet(index).header.opcode = opcode;
      get_packet(index).header.mask = lane_mask;
    }
    gpu::sync_lane(lane_mask);
    return Port(*this, lane_mask, index, out);
  }
  return cpp::nullopt;
}

template <uint16_t opcode> LIBC_INLINE Client::Port Client::open() {
  for (;;) {
    if (cpp::optional<Client::Port> p = try_open<opcode>())
      return cpp::move(p.value());
    sleep_briefly();
  }
}

/// Attempts to open a port to use as the server. The server can only open a
/// port if it has a pending receive operation
[[clang::convergent]] LIBC_INLINE cpp::optional<Server::Port>
Server::try_open() {
  // Perform a naive linear scan for a port that has a pending request.
  for (uint64_t index = 0; index < port_count; ++index) {
    uint32_t in = load_inbox(index);
    uint32_t out = load_outbox(index);

    // The server is passive, if there is no work pending don't bother
    // opening a port.
    if (buffer_unavailable(in, out))
      continue;

    // Attempt to acquire the lock on this index.
    uint64_t lane_mask = gpu::get_lane_mask();
    // Attempt to acquire the lock on this index.
    if (!try_lock(lane_mask, index))
      continue;

    // The mailbox state must be read with the lock held.
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);

    in = load_inbox(index);
    out = load_outbox(index);

    if (buffer_unavailable(in, out)) {
      unlock(lane_mask, index);
      continue;
    }

    return Port(*this, lane_mask, index, out);
  }
  return cpp::nullopt;
}

LIBC_INLINE Server::Port Server::open() {
  for (;;) {
    if (cpp::optional<Server::Port> p = try_open())
      return cpp::move(p.value());
    sleep_briefly();
  }
}

} // namespace rpc
} // namespace __llvm_libc

#endif
