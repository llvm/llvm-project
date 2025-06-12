//===-- GPU memory allocator implementation ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a parallel allocator intended for use on a GPU device.
// The core algorithm is slab allocator using a random walk over a bitfield for
// maximum parallel progress. Slab handling is done by a wait-free reference
// counted guard. The first use of a slab will create it from system memory for
// re-use. The last use will invalidate it and free the memory.
//
//===----------------------------------------------------------------------===//

#include "allocator.h"

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/new.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/threads/sleep.h"

namespace LIBC_NAMESPACE_DECL {

constexpr static uint64_t MAX_SIZE = /* 64 GiB */ 64ull * 1024 * 1024 * 1024;
constexpr static uint64_t SLAB_SIZE = /* 2 MiB */ 2ull * 1024 * 1024;
constexpr static uint64_t ARRAY_SIZE = MAX_SIZE / SLAB_SIZE;
constexpr static uint64_t SLAB_ALIGNMENT = SLAB_SIZE - 1;
constexpr static uint32_t BITS_IN_WORD = sizeof(uint32_t) * 8;
constexpr static uint32_t MIN_SIZE = 16;
constexpr static uint32_t MIN_ALIGNMENT = MIN_SIZE - 1;

// A sentinel used to indicate an invalid but non-null pointer value.
constexpr static uint64_t SENTINEL = cpp::numeric_limits<uint64_t>::max();

// The number of times we will try starting on a single index before skipping
// past it.
constexpr static uint32_t MAX_TRIES = 512;

static_assert(!(ARRAY_SIZE & (ARRAY_SIZE - 1)), "Must be a power of two");

namespace impl {
// Allocates more memory from the system through the RPC interface. All
// allocations from the system MUST be aligned on a 2MiB barrier. The default
// HSA allocator has this behavior for any allocation >= 2MiB and the CUDA
// driver provides an alignment field for virtual memory allocations.
static void *rpc_allocate(uint64_t size) {
  void *ptr = nullptr;
  rpc::Client::Port port = rpc::client.open<LIBC_MALLOC>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = size; },
      [&](rpc::Buffer *buffer, uint32_t) {
        ptr = reinterpret_cast<void *>(buffer->data[0]);
      });
  port.close();
  return ptr;
}

// Deallocates the associated system memory.
static void rpc_free(void *ptr) {
  rpc::Client::Port port = rpc::client.open<LIBC_FREE>();
  port.send([=](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = reinterpret_cast<uintptr_t>(ptr);
  });
  port.close();
}

// Convert a potentially disjoint bitmask into an increasing integer per-lane
// for use with indexing between gpu lanes.
static inline uint32_t lane_count(uint64_t lane_mask) {
  return cpp::popcount(lane_mask & ((uint64_t(1) << gpu::get_lane_id()) - 1));
}

// Obtain an initial value to seed a random number generator. We use the rounded
// multiples of the golden ratio from xorshift* as additional spreading.
static inline uint32_t entropy() {
  return (static_cast<uint32_t>(gpu::processor_clock()) ^
          (gpu::get_thread_id_x() * 0x632be59b) ^
          (gpu::get_block_id_x() * 0x85157af5)) *
         0x9e3779bb;
}

// Generate a random number and update the state using the xorshift32* PRNG.
static inline uint32_t xorshift32(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state * 0x9e3779bb;
}

// Final stage of murmurhash used to get a unique index for the global array
static inline uint32_t hash(uint32_t x) {
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  x *= 0xc2b2ae35;
  x ^= x >> 16;
  return x;
}

// Rounds the input value to the closest permitted chunk size. Here we accept
// the sum of the closest three powers of two. For a 2MiB slab size this is 48
// different chunk sizes. This gives us average internal fragmentation of 87.5%.
static inline uint32_t get_chunk_size(uint32_t x) {
  uint32_t y = x < MIN_SIZE ? MIN_SIZE : x;
  uint32_t pow2 = BITS_IN_WORD - cpp::countl_zero(y - 1);

  uint32_t s0 = 0b0100 << (pow2 - 3);
  uint32_t s1 = 0b0110 << (pow2 - 3);
  uint32_t s2 = 0b0111 << (pow2 - 3);
  uint32_t s3 = 0b1000 << (pow2 - 3);

  if (s0 > y)
    return (s0 + MIN_ALIGNMENT) & ~MIN_ALIGNMENT;
  if (s1 > y)
    return (s1 + MIN_ALIGNMENT) & ~MIN_ALIGNMENT;
  if (s2 > y)
    return (s2 + MIN_ALIGNMENT) & ~MIN_ALIGNMENT;
  return (s3 + MIN_ALIGNMENT) & ~MIN_ALIGNMENT;
}

// Rounds to the nearest power of two.
template <uint32_t N, typename T>
static inline constexpr T round_up(const T x) {
  static_assert(((N - 1) & N) == 0, "N must be a power of two");
  return (x + N) & ~(N - 1);
}

// Perform a lane parallel memset on a uint32_t pointer.
void uniform_memset(uint32_t *s, uint32_t c, uint32_t n, uint64_t uniform) {
  uint64_t mask = gpu::get_lane_mask();
  uint32_t workers = cpp::popcount(uniform);
  for (uint32_t i = impl::lane_count(mask & uniform); i < n; i += workers)
    s[i] = c;
}

} // namespace impl

/// A slab allocator used to hand out identically sized slabs of memory.
/// Allocation is done through random walks of a bitfield until a free bit is
/// encountered. This reduces contention and is highly parallel on a GPU.
///
/// 0       4           8       16                 ...                     2 MiB
/// ┌────────┬──────────┬────────┬──────────────────┬──────────────────────────┐
/// │ chunk  │  index   │  pad   │    bitfield[]    │         memory[]         │
/// └────────┴──────────┴────────┴──────────────────┴──────────────────────────┘
///
/// The size of the bitfield is the slab size divided by the chunk size divided
/// by the number of bits per word. We pad the interface to ensure 16 byte
/// alignment and to indicate that if the pointer is not aligned by 2MiB it
/// belongs to a slab rather than the global allocator.
struct Slab {
  // Header metadata for the slab, aligned to the minimum alignment.
  struct alignas(MIN_SIZE) Header {
    uint32_t chunk_size;
    uint32_t global_index;
  };

  // Initialize the slab with its chunk size and index in the global table for
  // use when freeing.
  Slab(uint32_t chunk_size, uint32_t global_index) {
    Header *header = reinterpret_cast<Header *>(memory);
    header->chunk_size = chunk_size;
    header->global_index = global_index;
  }

  // Set the necessary bitfield bytes to zero in parallel using many lanes. This
  // must be called before the bitfield can be accessed safely, memory is not
  // guaranteed to be zero initialized in the current implementation.
  void initialize(uint64_t uniform) {
    uint32_t size = (bitfield_bytes(get_chunk_size()) + sizeof(uint32_t) - 1) /
                    sizeof(uint32_t);
    impl::uniform_memset(get_bitfield(), 0, size, uniform);
  }

  // Get the number of chunks that can theoretically fit inside this slab.
  constexpr static uint32_t num_chunks(uint32_t chunk_size) {
    return SLAB_SIZE / chunk_size;
  }

  // Get the number of bytes needed to contain the bitfield bits.
  constexpr static uint32_t bitfield_bytes(uint32_t chunk_size) {
    return ((num_chunks(chunk_size) + BITS_IN_WORD - 1) / BITS_IN_WORD) * 8;
  }

  // The actual amount of memory available excluding the bitfield and metadata.
  constexpr static uint32_t available_bytes(uint32_t chunk_size) {
    return SLAB_SIZE - bitfield_bytes(chunk_size) - sizeof(Header);
  }

  // The number of chunks that can be stored in this slab.
  constexpr static uint32_t available_chunks(uint32_t chunk_size) {
    return available_bytes(chunk_size) / chunk_size;
  }

  // The length in bits of the bitfield.
  constexpr static uint32_t usable_bits(uint32_t chunk_size) {
    return available_bytes(chunk_size) / chunk_size;
  }

  // Get the location in the memory where we will store the chunk size.
  uint32_t get_chunk_size() const {
    return reinterpret_cast<const Header *>(memory)->chunk_size;
  }

  // Get the location in the memory where we will store the global index.
  uint32_t get_global_index() const {
    return reinterpret_cast<const Header *>(memory)->global_index;
  }

  // Get a pointer to where the bitfield is located in the memory.
  uint32_t *get_bitfield() {
    return reinterpret_cast<uint32_t *>(memory + sizeof(Header));
  }

  // Get a pointer to where the actual memory to be allocated lives.
  uint8_t *get_memory(uint32_t chunk_size) {
    return reinterpret_cast<uint8_t *>(get_bitfield()) +
           bitfield_bytes(chunk_size);
  }

  // Get a pointer to the actual memory given an index into the bitfield.
  void *ptr_from_index(uint32_t index, uint32_t chunk_size) {
    return get_memory(chunk_size) + index * chunk_size;
  }

  // Convert a pointer back into its bitfield index using its offset.
  uint32_t index_from_ptr(void *ptr, uint32_t chunk_size) {
    return static_cast<uint32_t>(reinterpret_cast<uint8_t *>(ptr) -
                                 get_memory(chunk_size)) /
           chunk_size;
  }

  // Randomly walks the bitfield until it finds a free bit. Allocations attempt
  // to put lanes right next to each other for better caching and convergence.
  void *allocate(uint64_t lane_mask, uint64_t uniform) {
    uint32_t chunk_size = get_chunk_size();
    uint32_t state = impl::entropy();

    // The uniform mask represents which lanes contain a uniform target pointer.
    // We attempt to place these next to each other.
    void *result = nullptr;
    for (uint64_t mask = lane_mask; mask;
         mask = gpu::ballot(lane_mask, !result)) {
      if (result)
        continue;

      uint32_t start = gpu::broadcast_value(lane_mask, impl::xorshift32(state));

      uint32_t id = impl::lane_count(uniform & mask);
      uint32_t index = (start + id) % usable_bits(chunk_size);
      uint32_t slot = index / BITS_IN_WORD;
      uint32_t bit = index % BITS_IN_WORD;

      // Get the mask of bits destined for the same slot and coalesce it.
      uint64_t match = uniform & gpu::match_any(mask, slot);
      uint32_t length = cpp::popcount(match);
      uint32_t bitmask = static_cast<uint32_t>((uint64_t(1) << length) - 1)
                         << bit;

      uint32_t before = 0;
      if (gpu::get_lane_id() == static_cast<uint32_t>(cpp::countr_zero(match)))
        before = cpp::AtomicRef(get_bitfield()[slot])
                     .fetch_or(bitmask, cpp::MemoryOrder::RELAXED);
      before = gpu::shuffle(mask, cpp::countr_zero(match), before);
      if (~before & (1 << bit))
        result = ptr_from_index(index, chunk_size);
      else
        sleep_briefly();
    }

    cpp::atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
    return result;
  }

  // Deallocates memory by resetting its corresponding bit in the bitfield.
  void deallocate(void *ptr) {
    uint32_t chunk_size = get_chunk_size();
    uint32_t index = index_from_ptr(ptr, chunk_size);
    uint32_t slot = index / BITS_IN_WORD;
    uint32_t bit = index % BITS_IN_WORD;

    cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    cpp::AtomicRef(get_bitfield()[slot])
        .fetch_and(~(1u << bit), cpp::MemoryOrder::RELAXED);
  }

  // The actual memory the slab will manage. All offsets are calculated at
  // runtime with the chunk size to keep the interface convergent when a warp or
  // wavefront is handling multiple sizes at once.
  uint8_t memory[SLAB_SIZE];
};

/// A wait-free guard around a pointer resource to be created dynamically if
/// space is available and freed once there are no more users.
struct GuardPtr {
private:
  struct RefCounter {
    // Indicates that the object is in its deallocation phase and thus invalid.
    static constexpr uint64_t INVALID = uint64_t(1) << 63;

    // If a read preempts an unlock call we indicate this so the following
    // unlock call can swap out the helped bit and maintain exclusive ownership.
    static constexpr uint64_t HELPED = uint64_t(1) << 62;

    // Resets the reference counter, cannot be reset to zero safely.
    void reset(uint32_t n, uint64_t &count) {
      counter.store(n, cpp::MemoryOrder::RELAXED);
      count = n;
    }

    // Acquire a slot in the reference counter if it is not invalid.
    bool acquire(uint32_t n, uint64_t &count) {
      count = counter.fetch_add(n, cpp::MemoryOrder::RELAXED) + n;
      return (count & INVALID) == 0;
    }

    // Release a slot in the reference counter. This function should only be
    // called following a valid acquire call.
    bool release(uint32_t n) {
      // If this thread caused the counter to reach zero we try to invalidate it
      // and obtain exclusive rights to deconstruct it. If the CAS failed either
      // another thread resurrected the counter and we quit, or a parallel read
      // helped us invalidating it. For the latter, claim that flag and return.
      if (counter.fetch_sub(n, cpp::MemoryOrder::RELAXED) == n) {
        uint64_t expected = 0;
        if (counter.compare_exchange_strong(expected, INVALID,
                                            cpp::MemoryOrder::RELAXED,
                                            cpp::MemoryOrder::RELAXED))
          return true;
        else if ((expected & HELPED) &&
                 (counter.exchange(INVALID, cpp::MemoryOrder::RELAXED) &
                  HELPED))
          return true;
      }
      return false;
    }

    // Returns the current reference count, potentially helping a releasing
    // thread.
    uint64_t read() {
      auto val = counter.load(cpp::MemoryOrder::RELAXED);
      if (val == 0 && counter.compare_exchange_strong(
                          val, INVALID | HELPED, cpp::MemoryOrder::RELAXED))
        return 0;
      return (val & INVALID) ? 0 : val;
    }

    cpp::Atomic<uint64_t> counter{0};
  };

  cpp::Atomic<Slab *> ptr{nullptr};
  RefCounter ref{};

  // Should be called be a single lane for each different pointer.
  template <typename... Args>
  Slab *try_lock_impl(uint32_t n, uint64_t &count, Args &&...args) {
    Slab *expected = ptr.load(cpp::MemoryOrder::RELAXED);
    if (!expected &&
        ptr.compare_exchange_strong(
            expected, reinterpret_cast<Slab *>(SENTINEL),
            cpp::MemoryOrder::RELAXED, cpp::MemoryOrder::RELAXED)) {
      count = cpp::numeric_limits<uint64_t>::max();
      void *raw = impl::rpc_allocate(sizeof(Slab));
      if (!raw)
        return nullptr;
      return new (raw) Slab(cpp::forward<Args>(args)...);
    }

    if (!expected || expected == reinterpret_cast<Slab *>(SENTINEL))
      return nullptr;

    if (!ref.acquire(n, count))
      return nullptr;

    cpp::atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
    return ptr.load(cpp::MemoryOrder::RELAXED);
  }

  // Finalize the associated memory and signal that it is ready to use by
  // resetting the counter.
  void finalize(Slab *mem, uint32_t n, uint64_t &count) {
    cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    ptr.store(mem, cpp::MemoryOrder::RELAXED);
    cpp::atomic_thread_fence(cpp::MemoryOrder::ACQUIRE);
    if (!ref.acquire(n, count))
      ref.reset(n, count);
  }

public:
  // Attempt to lock access to the pointer, potentially creating it if empty.
  // The uniform mask represents which lanes share the same pointer. For each
  // uniform value we elect a leader to handle it on behalf of the other lanes.
  template <typename... Args>
  Slab *try_lock(uint64_t lane_mask, uint64_t uniform, uint64_t &count,
                 Args &&...args) {
    count = 0;
    Slab *result = nullptr;
    if (gpu::get_lane_id() == uint32_t(cpp::countr_zero(uniform)))
      result = try_lock_impl(cpp::popcount(uniform), count,
                             cpp::forward<Args>(args)...);
    result = gpu::shuffle(lane_mask, cpp::countr_zero(uniform), result);
    count = gpu::shuffle(lane_mask, cpp::countr_zero(uniform), count);

    if (!result)
      return nullptr;

    // We defer storing the newly allocated slab until now so that we can use
    // multiple lanes to initialize it and release it for use.
    if (count == cpp::numeric_limits<uint64_t>::max()) {
      result->initialize(uniform);
      if (gpu::get_lane_id() == uint32_t(cpp::countr_zero(uniform)))
        finalize(result, cpp::popcount(uniform), count);
    }

    if (count != cpp::numeric_limits<uint64_t>::max())
      count = count - cpp::popcount(uniform) + impl::lane_count(uniform) + 1;

    return result;
  }

  // Release the associated lock on the pointer, potentially destroying it.
  void unlock(uint64_t lane_mask, uint64_t mask) {
    cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
    if (gpu::get_lane_id() == uint32_t(cpp::countr_zero(mask)) &&
        ref.release(cpp::popcount(mask))) {
      Slab *p = ptr.load(cpp::MemoryOrder::RELAXED);
      p->~Slab();
      impl::rpc_free(p);
      cpp::atomic_thread_fence(cpp::MemoryOrder::RELEASE);
      ptr.store(nullptr, cpp::MemoryOrder::RELAXED);
    }
    gpu::sync_lane(lane_mask);
  }

  // Get the current value of the reference counter.
  uint64_t use_count() { return ref.read(); }
};

// The global array used to search for a valid slab to allocate from.
static GuardPtr slots[ARRAY_SIZE] = {};

// Tries to find a slab in the table that can support the given chunk size.
static Slab *find_slab(uint32_t chunk_size) {
  // We start at a hashed value to spread out different chunk sizes.
  uint32_t start = impl::hash(chunk_size);
  uint64_t lane_mask = gpu::get_lane_mask();
  uint64_t uniform = gpu::match_any(lane_mask, chunk_size);

  Slab *result = nullptr;
  uint32_t nudge = 0;
  for (uint64_t mask = lane_mask; mask;
       mask = gpu::ballot(lane_mask, !result), ++nudge) {
    uint32_t index = cpp::numeric_limits<uint32_t>::max();
    for (uint32_t offset = nudge / MAX_TRIES;
         gpu::ballot(lane_mask, index == cpp::numeric_limits<uint32_t>::max());
         offset += cpp::popcount(uniform & lane_mask)) {
      uint32_t candidate =
          (start + offset + impl::lane_count(uniform & lane_mask)) % ARRAY_SIZE;
      uint64_t available =
          gpu::ballot(lane_mask, slots[candidate].use_count() <
                                     Slab::available_chunks(chunk_size));
      uint32_t new_index = gpu::shuffle(
          lane_mask, cpp::countr_zero(available & uniform), candidate);

      // Each uniform group will use the first empty slot they find.
      if ((index == cpp::numeric_limits<uint32_t>::max() &&
           (available & uniform)))
        index = new_index;

      // Guaruntees that this loop will eventuall exit if there is no space.
      if (offset >= ARRAY_SIZE) {
        result = reinterpret_cast<Slab *>(SENTINEL);
        index = 0;
      }
    }

    // Try to claim a slot for the found slot.
    if (!result) {
      uint64_t reserved = 0;
      Slab *slab = slots[index].try_lock(lane_mask & mask, uniform & mask,
                                         reserved, chunk_size, index);
      // If we find a slab with a matching chunk size then we store the result.
      // Otherwise, we need to free the claimed lock and continue. In the case
      // of out-of-memory we return a sentinel value.
      if (slab && reserved <= Slab::available_chunks(chunk_size) &&
          slab->get_chunk_size() == chunk_size) {
        result = slab;
      } else if (slab && (reserved > Slab::available_chunks(chunk_size) ||
                          slab->get_chunk_size() != chunk_size)) {
        if (slab->get_chunk_size() != chunk_size)
          start = index + 1;
        slots[index].unlock(gpu::get_lane_mask(),
                            gpu::get_lane_mask() & uniform);
      } else if (!slab && reserved == cpp::numeric_limits<uint64_t>::max()) {
        result = reinterpret_cast<Slab *>(SENTINEL);
      } else {
        sleep_briefly();
      }
    }
  }
  return result;
}

// Release the lock associated with a given slab.
static void release_slab(Slab *slab) {
  uint32_t index = slab->get_global_index();
  uint64_t lane_mask = gpu::get_lane_mask();
  uint64_t uniform = gpu::match_any(lane_mask, index);
  slots[index].unlock(lane_mask, uniform);
}

namespace gpu {

void *allocate(uint64_t size) {
  if (!size)
    return nullptr;

  // Allocations requiring a full slab or more go directly to memory.
  if (size >= SLAB_SIZE / 2)
    return impl::rpc_allocate(impl::round_up<SLAB_SIZE>(size));

  // Try to find a slab for the rounded up chunk size and allocate from it.
  uint32_t chunk_size = impl::get_chunk_size(static_cast<uint32_t>(size));
  Slab *slab = find_slab(chunk_size);
  if (!slab || slab == reinterpret_cast<Slab *>(SENTINEL))
    return nullptr;

  uint64_t lane_mask = gpu::get_lane_mask();
  uint64_t uniform = gpu::match_any(lane_mask, slab->get_global_index());
  void *ptr = slab->allocate(lane_mask, uniform);
  return ptr;
}

void deallocate(void *ptr) {
  if (!ptr)
    return;

  // All non-slab allocations will be aligned on a 2MiB boundary.
  if ((reinterpret_cast<uintptr_t>(ptr) & SLAB_ALIGNMENT) == 0)
    return impl::rpc_free(ptr);

  // The original slab pointer is the 2MiB boundary using the given pointer.
  Slab *slab = reinterpret_cast<Slab *>(
      (reinterpret_cast<uintptr_t>(ptr) & ~SLAB_ALIGNMENT));
  slab->deallocate(ptr);
  release_slab(slab);
}

} // namespace gpu
} // namespace LIBC_NAMESPACE_DECL
