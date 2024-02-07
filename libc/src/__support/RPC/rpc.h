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

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_RPC_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_RPC_H

#include "rpc_util.h"
#include "src/__support/CPP/algorithm.h" // max
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/GPU/utils.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace rpc {

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

/// The maximum number of parallel ports that the RPC interface can support.
constexpr uint64_t MAX_PORT_COUNT = 4096;

/// A common process used to synchronize communication between a client and a
/// server. The process contains a read-only inbox and a write-only outbox used
/// for signaling ownership of the shared buffer between both sides. We assign
/// ownership of the buffer to the client if the inbox and outbox bits match,
/// otherwise it is owned by the server.
///
/// This process is designed to allow the client and the server to exchange data
/// using a fixed size packet in a mostly arbitrary order using the 'send' and
/// 'recv' operations. The following restrictions to this scheme apply:
///   - The client will always start with a 'send' operation.
///   - The server will always start with a 'recv' operation.
///   - Every 'send' or 'recv' call is mirrored by the other process.
template <bool Invert> struct Process {
  LIBC_INLINE Process() = default;
  LIBC_INLINE Process(const Process &) = delete;
  LIBC_INLINE Process &operator=(const Process &) = delete;
  LIBC_INLINE Process(Process &&) = default;
  LIBC_INLINE Process &operator=(Process &&) = default;
  LIBC_INLINE ~Process() = default;

  uint32_t port_count = 0;
  cpp::Atomic<uint32_t> *inbox = nullptr;
  cpp::Atomic<uint32_t> *outbox = nullptr;
  Header *header = nullptr;
  Buffer *packet = nullptr;

  static constexpr uint64_t NUM_BITS_IN_WORD = sizeof(uint32_t) * 8;
  cpp::Atomic<uint32_t> lock[MAX_PORT_COUNT / NUM_BITS_IN_WORD] = {0};

  LIBC_INLINE Process(uint32_t port_count, void *buffer)
      : port_count(port_count), inbox(reinterpret_cast<cpp::Atomic<uint32_t> *>(
                                    advance(buffer, inbox_offset(port_count)))),
        outbox(reinterpret_cast<cpp::Atomic<uint32_t> *>(
            advance(buffer, outbox_offset(port_count)))),
        header(reinterpret_cast<Header *>(
            advance(buffer, header_offset(port_count)))),
        packet(reinterpret_cast<Buffer *>(
            advance(buffer, buffer_offset(port_count)))) {}

  /// Allocate a memory buffer sufficient to store the following equivalent
  /// representation in memory.
  ///
  /// struct Equivalent {
  ///   Atomic<uint32_t> primary[port_count];
  ///   Atomic<uint32_t> secondary[port_count];
  ///   Header header[port_count];
  ///   Buffer packet[port_count][lane_size];
  /// };
  LIBC_INLINE static constexpr uint64_t allocation_size(uint32_t port_count,
                                                        uint32_t lane_size) {
    return buffer_offset(port_count) + buffer_bytes(port_count, lane_size);
  }

  /// Retrieve the inbox state from memory shared between processes.
  LIBC_INLINE uint32_t load_inbox(uint64_t lane_mask, uint32_t index) const {
    return gpu::broadcast_value(
        lane_mask,
        inbox[index].load(cpp::MemoryOrder::RELAXED, cpp::MemoryScope::SYSTEM));
  }

  /// Retrieve the outbox state from memory shared between processes.
  LIBC_INLINE uint32_t load_outbox(uint64_t lane_mask, uint32_t index) const {
    return gpu::broadcast_value(lane_mask,
                                outbox[index].load(cpp::MemoryOrder::RELAXED,
                                                   cpp::MemoryScope::SYSTEM));
  }

  /// Signal to the other process that this one is finished with the buffer.
  /// Equivalent to loading outbox followed by store of the inverted value
  /// The outbox is write only by this warp and tracking the value locally is
  /// cheaper than calling load_outbox to get the value to store.
  LIBC_INLINE uint32_t invert_outbox(uint32_t index, uint32_t current_outbox) {
    uint32_t inverted_outbox = !current_outbox;
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    outbox[index].store(inverted_outbox, cpp::MemoryOrder::RELAXED,
                        cpp::MemoryScope::SYSTEM);
    return inverted_outbox;
  }

  // Given the current outbox and inbox values, wait until the inbox changes
  // to indicate that this thread owns the buffer element.
  LIBC_INLINE void wait_for_ownership(uint64_t lane_mask, uint32_t index,
                                      uint32_t outbox, uint32_t in) {
    while (buffer_unavailable(in, outbox)) {
      sleep_briefly();
      in = load_inbox(lane_mask, index);
    }
    atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
  }

  /// The packet is a linearly allocated array of buffers used to communicate
  /// with the other process. This function returns the appropriate slot in this
  /// array such that the process can operate on an entire warp or wavefront.
  LIBC_INLINE Buffer *get_packet(uint32_t index, uint32_t lane_size) {
    return &packet[index * lane_size];
  }

  /// Determines if this process needs to wait for ownership of the buffer. We
  /// invert the condition on one of the processes to indicate that if one
  /// process owns the buffer then the other does not.
  LIBC_INLINE static bool buffer_unavailable(uint32_t in, uint32_t out) {
    bool cond = in != out;
    return Invert ? !cond : cond;
  }

  /// Attempt to claim the lock at index. Return true on lock taken.
  /// lane_mask is a bitmap of the threads in the warp that would hold the
  /// single lock on success, e.g. the result of gpu::get_lane_mask()
  /// The lock is held when the n-th bit of the lock bitfield is set.
  [[clang::convergent]] LIBC_INLINE bool try_lock(uint64_t lane_mask,
                                                  uint32_t index) {
    // On amdgpu, test and set to the nth lock bit and a sync_lane would suffice
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
    bool before = set_nth(lock, index, id_in_lane_mask);
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

    // If holding the lock then the caller can load values knowing said loads
    // won't move past the lock. No such guarantee is needed if the lock acquire
    // failed. This conditional branch is expected to fold in the caller after
    // inlining the current function.
    bool holding_lock = lane_mask != packed;
    if (holding_lock)
      atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
    return holding_lock;
  }

  /// Unlock the lock at index. We need a lane sync to keep this function
  /// convergent, otherwise the compiler will sink the store and deadlock.
  [[clang::convergent]] LIBC_INLINE void unlock(uint64_t lane_mask,
                                                uint32_t index) {
    // Do not move any writes past the unlock
    atomic_thread_fence(cpp::MemoryOrder::RELEASE);

    // Wait for other threads in the warp to finish using the lock
    gpu::sync_lane(lane_mask);

    // Use exactly one thread to clear the nth bit in the lock array Must
    // restrict to a single thread to avoid one thread dropping the lock, then
    // an unrelated warp claiming the lock, then a second thread in this warp
    // dropping the lock again.
    clear_nth(lock, index, gpu::is_first_lane(lane_mask));
    gpu::sync_lane(lane_mask);
  }

  /// Number of bytes to allocate for an inbox or outbox.
  LIBC_INLINE static constexpr uint64_t mailbox_bytes(uint32_t port_count) {
    return port_count * sizeof(cpp::Atomic<uint32_t>);
  }

  /// Number of bytes to allocate for the buffer containing the packets.
  LIBC_INLINE static constexpr uint64_t buffer_bytes(uint32_t port_count,
                                                     uint32_t lane_size) {
    return port_count * lane_size * sizeof(Buffer);
  }

  /// Offset of the inbox in memory. This is the same as the outbox if inverted.
  LIBC_INLINE static constexpr uint64_t inbox_offset(uint32_t port_count) {
    return Invert ? mailbox_bytes(port_count) : 0;
  }

  /// Offset of the outbox in memory. This is the same as the inbox if inverted.
  LIBC_INLINE static constexpr uint64_t outbox_offset(uint32_t port_count) {
    return Invert ? 0 : mailbox_bytes(port_count);
  }

  /// Offset of the buffer containing the packets after the inbox and outbox.
  LIBC_INLINE static constexpr uint64_t header_offset(uint32_t port_count) {
    return align_up(2 * mailbox_bytes(port_count), alignof(Header));
  }

  /// Offset of the buffer containing the packets after the inbox and outbox.
  LIBC_INLINE static constexpr uint64_t buffer_offset(uint32_t port_count) {
    return align_up(header_offset(port_count) + port_count * sizeof(Header),
                    alignof(Buffer));
  }

  /// Conditionally set the n-th bit in the atomic bitfield.
  LIBC_INLINE static constexpr uint32_t set_nth(cpp::Atomic<uint32_t> *bits,
                                                uint32_t index, bool cond) {
    uint32_t slot = index / NUM_BITS_IN_WORD;
    uint32_t bit = index % NUM_BITS_IN_WORD;
    return bits[slot].fetch_or(static_cast<uint32_t>(cond) << bit,
                               cpp::MemoryOrder::RELAXED,
                               cpp::MemoryScope::DEVICE) &
           (1u << bit);
  }

  /// Conditionally clear the n-th bit in the atomic bitfield.
  LIBC_INLINE static constexpr uint32_t clear_nth(cpp::Atomic<uint32_t> *bits,
                                                  uint32_t index, bool cond) {
    uint32_t slot = index / NUM_BITS_IN_WORD;
    uint32_t bit = index % NUM_BITS_IN_WORD;
    return bits[slot].fetch_and(~0u ^ (static_cast<uint32_t>(cond) << bit),
                                cpp::MemoryOrder::RELAXED,
                                cpp::MemoryScope::DEVICE) &
           (1u << bit);
  }
};

/// Invokes a function accross every active buffer across the total lane size.
static LIBC_INLINE void invoke_rpc(cpp::function<void(Buffer *)> fn,
                                   uint32_t lane_size, uint64_t lane_mask,
                                   Buffer *slot) {
  if constexpr (is_process_gpu()) {
    fn(&slot[gpu::get_lane_id()]);
  } else {
    for (uint32_t i = 0; i < lane_size; i += gpu::get_lane_size())
      if (lane_mask & (1ul << i))
        fn(&slot[i]);
  }
}

/// Alternate version that also provides the index of the current lane.
static LIBC_INLINE void invoke_rpc(cpp::function<void(Buffer *, uint32_t)> fn,
                                   uint32_t lane_size, uint64_t lane_mask,
                                   Buffer *slot) {
  if constexpr (is_process_gpu()) {
    fn(&slot[gpu::get_lane_id()], gpu::get_lane_id());
  } else {
    for (uint32_t i = 0; i < lane_size; i += gpu::get_lane_size())
      if (lane_mask & (1ul << i))
        fn(&slot[i], i);
  }
}

/// The port provides the interface to communicate between the multiple
/// processes. A port is conceptually an index into the memory provided by the
/// underlying process that is guarded by a lock bit.
template <bool T> struct Port {
  LIBC_INLINE Port(Process<T> &process, uint64_t lane_mask, uint32_t lane_size,
                   uint32_t index, uint32_t out)
      : process(process), lane_mask(lane_mask), lane_size(lane_size),
        index(index), out(out), receive(false), owns_buffer(true) {}
  LIBC_INLINE ~Port() = default;

private:
  LIBC_INLINE Port(const Port &) = delete;
  LIBC_INLINE Port &operator=(const Port &) = delete;
  LIBC_INLINE Port(Port &&) = default;
  LIBC_INLINE Port &operator=(Port &&) = default;

  friend struct Client;
  template <uint32_t U> friend struct Server;
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
    return process.header[index].opcode;
  }

  LIBC_INLINE uint16_t get_index() const { return index; }

  LIBC_INLINE void close() {
    // The server is passive, if it own the buffer when it closes we need to
    // give ownership back to the client.
    if (owns_buffer && T)
      out = process.invert_outbox(index, out);
    process.unlock(lane_mask, index);
  }

private:
  Process<T> &process;
  uint64_t lane_mask;
  uint32_t lane_size;
  uint32_t index;
  uint32_t out;
  bool receive;
  bool owns_buffer;
};

/// The RPC client used to make requests to the server.
struct Client {
  LIBC_INLINE Client() = default;
  LIBC_INLINE Client(const Client &) = delete;
  LIBC_INLINE Client &operator=(const Client &) = delete;
  LIBC_INLINE ~Client() = default;

  LIBC_INLINE Client(uint32_t port_count, void *buffer)
      : process(port_count, buffer) {}

  using Port = rpc::Port<false>;
  template <uint16_t opcode> LIBC_INLINE Port open();

private:
  Process<false> process;
};
static_assert(cpp::is_trivially_copyable<Client>::value &&
                  sizeof(Process<true>) == sizeof(Process<false>),
              "The client is not trivially copyable from the server");

/// The RPC server used to respond to the client.
template <uint32_t lane_size> struct Server {
  LIBC_INLINE Server() = default;
  LIBC_INLINE Server(const Server &) = delete;
  LIBC_INLINE Server &operator=(const Server &) = delete;
  LIBC_INLINE ~Server() = default;

  LIBC_INLINE Server(uint32_t port_count, void *buffer)
      : process(port_count, buffer) {}

  using Port = rpc::Port<true>;
  LIBC_INLINE cpp::optional<Port> try_open(uint32_t start = 0);
  LIBC_INLINE Port open();

  LIBC_INLINE static uint64_t allocation_size(uint32_t port_count) {
    return Process<true>::allocation_size(port_count, lane_size);
  }

private:
  Process<true> process;
};

/// Applies \p fill to the shared buffer and initiates a send operation.
template <bool T> template <typename F> LIBC_INLINE void Port<T>::send(F fill) {
  uint32_t in = owns_buffer ? out ^ T : process.load_inbox(lane_mask, index);

  // We need to wait until we own the buffer before sending.
  process.wait_for_ownership(lane_mask, index, out, in);

  // Apply the \p fill function to initialize the buffer and release the memory.
  invoke_rpc(fill, lane_size, process.header[index].mask,
             process.get_packet(index, lane_size));
  out = process.invert_outbox(index, out);
  owns_buffer = false;
  receive = false;
}

/// Applies \p use to the shared buffer and acknowledges the send.
template <bool T> template <typename U> LIBC_INLINE void Port<T>::recv(U use) {
  // We only exchange ownership of the buffer during a receive if we are waiting
  // for a previous receive to finish.
  if (receive) {
    out = process.invert_outbox(index, out);
    owns_buffer = false;
  }

  uint32_t in = owns_buffer ? out ^ T : process.load_inbox(lane_mask, index);

  // We need to wait until we own the buffer before receiving.
  process.wait_for_ownership(lane_mask, index, out, in);

  // Apply the \p use function to read the memory out of the buffer.
  invoke_rpc(use, lane_size, process.header[index].mask,
             process.get_packet(index, lane_size));
  receive = true;
  owns_buffer = true;
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

/// Helper routine to simplify the interface when sending from the GPU using
/// thread private pointers to the underlying value.
template <bool T>
LIBC_INLINE void Port<T>::send_n(const void *src, uint64_t size) {
  const void **src_ptr = &src;
  uint64_t *size_ptr = &size;
  send_n(src_ptr, size_ptr);
}

/// Sends an arbitrarily sized data buffer \p src across the shared channel in
/// multiples of the packet length.
template <bool T>
LIBC_INLINE void Port<T>::send_n(const void *const *src, uint64_t *size) {
  uint64_t num_sends = 0;
  send([&](Buffer *buffer, uint32_t id) {
    reinterpret_cast<uint64_t *>(buffer->data)[0] = lane_value(size, id);
    num_sends = is_process_gpu() ? lane_value(size, id)
                                 : cpp::max(lane_value(size, id), num_sends);
    uint64_t len =
        lane_value(size, id) > sizeof(Buffer::data) - sizeof(uint64_t)
            ? sizeof(Buffer::data) - sizeof(uint64_t)
            : lane_value(size, id);
    rpc_memcpy(&buffer->data[1], lane_value(src, id), len);
  });
  uint64_t idx = sizeof(Buffer::data) - sizeof(uint64_t);
  uint64_t mask = process.header[index].mask;
  while (gpu::ballot(mask, idx < num_sends)) {
    send([=](Buffer *buffer, uint32_t id) {
      uint64_t len = lane_value(size, id) - idx > sizeof(Buffer::data)
                         ? sizeof(Buffer::data)
                         : lane_value(size, id) - idx;
      if (idx < lane_value(size, id))
        rpc_memcpy(buffer->data, advance(lane_value(src, id), idx), len);
    });
    idx += sizeof(Buffer::data);
  }
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
                                 : cpp::max(lane_value(size, id), num_recvs);
    uint64_t len =
        lane_value(size, id) > sizeof(Buffer::data) - sizeof(uint64_t)
            ? sizeof(Buffer::data) - sizeof(uint64_t)
            : lane_value(size, id);
    rpc_memcpy(lane_value(dst, id), &buffer->data[1], len);
  });
  uint64_t idx = sizeof(Buffer::data) - sizeof(uint64_t);
  uint64_t mask = process.header[index].mask;
  while (gpu::ballot(mask, idx < num_recvs)) {
    recv([=](Buffer *buffer, uint32_t id) {
      uint64_t len = lane_value(size, id) - idx > sizeof(Buffer::data)
                         ? sizeof(Buffer::data)
                         : lane_value(size, id) - idx;
      if (idx < lane_value(size, id))
        rpc_memcpy(advance(lane_value(dst, id), idx), buffer->data, len);
    });
    idx += sizeof(Buffer::data);
  }
}

/// Continually attempts to open a port to use as the client. The client can
/// only open a port if we find an index that is in a valid sending state. That
/// is, there are send operations pending that haven't been serviced on this
/// port. Each port instance uses an associated \p opcode to tell the server
/// what to do. The Client interface provides the appropriate lane size to the
/// port using the platform's returned value.
template <uint16_t opcode>
[[clang::convergent]] LIBC_INLINE Client::Port Client::open() {
  // Repeatedly perform a naive linear scan for a port that can be opened to
  // send data.
  for (uint32_t index = gpu::get_cluster_id();; ++index) {
    // Start from the beginning if we run out of ports to check.
    if (index >= process.port_count)
      index = 0;

    // Attempt to acquire the lock on this index.
    uint64_t lane_mask = gpu::get_lane_mask();
    if (!process.try_lock(lane_mask, index))
      continue;

    uint32_t in = process.load_inbox(lane_mask, index);
    uint32_t out = process.load_outbox(lane_mask, index);

    // Once we acquire the index we need to check if we are in a valid sending
    // state.
    if (process.buffer_unavailable(in, out)) {
      process.unlock(lane_mask, index);
      continue;
    }

    if (gpu::is_first_lane(lane_mask)) {
      process.header[index].opcode = opcode;
      process.header[index].mask = lane_mask;
    }
    gpu::sync_lane(lane_mask);
    return Port(process, lane_mask, gpu::get_lane_size(), index, out);
  }
}

/// Attempts to open a port to use as the server. The server can only open a
/// port if it has a pending receive operation
template <uint32_t lane_size>
[[clang::convergent]] LIBC_INLINE
    cpp::optional<typename Server<lane_size>::Port>
    Server<lane_size>::try_open(uint32_t start) {
  // Perform a naive linear scan for a port that has a pending request.
  for (uint32_t index = start; index < process.port_count; ++index) {
    uint64_t lane_mask = gpu::get_lane_mask();
    uint32_t in = process.load_inbox(lane_mask, index);
    uint32_t out = process.load_outbox(lane_mask, index);

    // The server is passive, if there is no work pending don't bother
    // opening a port.
    if (process.buffer_unavailable(in, out))
      continue;

    // Attempt to acquire the lock on this index.
    if (!process.try_lock(lane_mask, index))
      continue;

    in = process.load_inbox(lane_mask, index);
    out = process.load_outbox(lane_mask, index);

    if (process.buffer_unavailable(in, out)) {
      process.unlock(lane_mask, index);
      continue;
    }

    return Port(process, lane_mask, lane_size, index, out);
  }
  return cpp::nullopt;
}

template <uint32_t lane_size>
LIBC_INLINE typename Server<lane_size>::Port Server<lane_size>::open() {
  for (;;) {
    if (cpp::optional<Server::Port> p = try_open())
      return cpp::move(p.value());
    sleep_briefly();
  }
}

} // namespace rpc
} // namespace LIBC_NAMESPACE

#endif
