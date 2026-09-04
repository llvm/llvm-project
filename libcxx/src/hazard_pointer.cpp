//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__hazard_pointer/domain.h>

#include <__assert>
#include <__atomic/atomic.h>
#include <__atomic/fence.h>
#include <__atomic/memory_order.h>
#include <__bit/bit_ceil.h>
#include <__chrono/duration.h>
#include <__chrono/steady_clock.h>
#include <__mutex/once_flag.h>
#include <__thread/support.h>
#include <cstddef>
#include <cstdint>
#include <new>

// The engine behind <hazard_pointer> implements the single default domain the Standard specifies.
// Nothing in this file is ABI: only the three functions at the bottom, and the two structs in
// <__hazard_pointer/domain.h> they take/return, are.
//
// Terminology: a *record* is a hazard pointer (its slot holds the protected node address); a *node* is
// the header embedded in every hazard-protectable object; the *available list* holds free records; each
// thread keeps a small cache of free records; retired nodes wait in eight sharded lists until a
// reclamation pass -- run inline by whichever thread's retire() crosses the threshold -- deletes the
// unprotected ones.

#if defined(__SANITIZE_THREAD__)
#  define _LIBCPP_HAZARD_POINTER_TSAN 1
#elif defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define _LIBCPP_HAZARD_POINTER_TSAN 1
#  endif
#endif
#ifndef _LIBCPP_HAZARD_POINTER_TSAN
#  define _LIBCPP_HAZARD_POINTER_TSAN 0
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace {

// Configuration

constexpr int __retired_count_threshold   = 1000; // a pass runs once this many objects are retired ...
constexpr int __threshold_multiplier      = 2;    // ... or twice the record count, whichever is larger ...
constexpr uintptr_t __sync_time_period_s  = 2;    // ... or two seconds after the previous pass
constexpr int __shard_count               = 8;
constexpr int __shard_mask                = __shard_count - 1;
constexpr int __ignored_low_bits          = 8;
constexpr uintptr_t __lock_bit            = 1;
constexpr uint8_t __thread_cache_capacity = 9;
// Records are laid out one per cache line (see __hazard_pointer_record). Where over-aligned allocation is
// not available, std::__libcpp_allocate can only promise fundamental alignment, so fall back to that.
#if !_LIBCPP_HAS_ALIGNED_ALLOCATION
constexpr size_t __cache_line_size = alignof(max_align_t);
#elif defined(__GCC_DESTRUCTIVE_SIZE)
constexpr size_t __cache_line_size = __GCC_DESTRUCTIVE_SIZE;
#else
constexpr size_t __cache_line_size = 64;
#endif

// 64-bit mixer shared by the shard function and the protected set.
inline size_t __mix_pointer(uintptr_t __x) noexcept {
  if constexpr (sizeof(uintptr_t) == 8) {
    __x ^= __x >> 32;
    __x *= 0x9E3779B97F4A7C15ULL;
    __x ^= __x >> 29;
  } else {
    __x ^= __x >> 16;
    __x *= 0x9E3779B1U;
    __x ^= __x >> 15;
  }
  return static_cast<size_t>(__x);
}

// Record

// One cache line per record: records of different threads must not share a line (every protect() writes
// to it); targets without over-aligned allocation fall back to fundamental alignment. Records are created
// by the domain's grow array and never destroyed.
struct alignas(__cache_line_size) __hazard_pointer_record : __hazard_pointer_slot {
  __hazard_pointer_record* __next_available_ = nullptr; // available-list link; meaningful only while free
  __hazard_pointer_record() noexcept : __hazard_pointer_slot{nullptr} {}
};

// Grow-only array

// Concurrent grow-only array with wait-free indexed access and reference stability. operator[](i) is
// valid for every i: if i >= __size() the array grows to bit_ceil(i + 1), default-constructing every
// element up to the new capacity. Growth is a race decided by one CAS; losers destroy what they built.
// Each growth round is one allocation holding {header}{_Tp* list[capacity]}{_Tp slab[capacity - base]};
// older rounds stay alive on a chain, so element addresses never change. __size() is *not* monotonic -- a
// delayed store from a smaller round can lower it after a larger one published -- but it is always a lower
// bound on what has been constructed, which is all operator[]'s fast path needs. Never destroyed (the domain
// is immortal). _Tp must be nothrow default constructible.
template <class _Tp>
class __grow_array {
  struct __round {
    __round* __next_ = nullptr; // the previous (smaller) round; owns elements [0, __next_->__size_)
    size_t __size_   = 0;       // > __next_->__size_
    _Tp** __list() noexcept { return reinterpret_cast<_Tp**>(reinterpret_cast<byte*>(this) + sizeof(__round)); }
  };

  atomic<size_t> __size_{0};
  atomic<__round*> __array_{nullptr};

  static constexpr size_t __alignment() noexcept {
    return alignof(_Tp) > alignof(max_align_t) ? alignof(_Tp) : alignof(max_align_t);
  }
  static constexpr size_t __ceil_to(size_t __value, size_t __alignment) noexcept {
    return (__value + __alignment - 1) / __alignment * __alignment;
  }
  static constexpr size_t __bytes(size_t __capacity, size_t __base) noexcept {
    return __ceil_to(sizeof(__round) + __capacity * sizeof(_Tp*), __alignment()) +
           __ceil_to((__capacity - __base) * sizeof(_Tp), __alignment());
  }
  static _Tp* __slab(__round* __curr) noexcept {
    uintptr_t __list_end = reinterpret_cast<uintptr_t>(__curr->__list() + __curr->__size_);
    return reinterpret_cast<_Tp*>(__ceil_to(static_cast<size_t>(__list_end), __alignment()));
  }

  // Builds a round of __capacity elements on top of __next; returns nullptr (and updates __next) when
  // another thread published a round meanwhile. May throw bad_alloc.
  __round* __new_round(size_t __capacity, __round*& __next) {
    size_t __base   = __next ? __next->__size_ : 0;
    void* __raw     = std::__libcpp_allocate<byte>(__element_count(__bytes(__capacity, __base)), __alignment());
    __round* __curr = ::new (__raw) __round{};
    __curr->__next_ = __next;
    __curr->__size_ = __capacity;
    _Tp* __slab_ptr = __slab(__curr);
    _Tp** __list    = __curr->__list();
    for (size_t __i = 0; __i < __base; ++__i)
      __list[__i] = __next->__list()[__i];
    for (size_t __i = __base; __i < __capacity; ++__i)
      __list[__i] = nullptr;
    for (size_t __i = __base; __i < __capacity; ++__i) {
      __round* __seen = __array_.load(memory_order_acquire);
      if (__seen != __next) { // race lost early: stop building
        __next = __seen;
        __del_round(__curr);
        return nullptr;
      }
      __list[__i] = ::new (static_cast<void*>(&__slab_ptr[__i - __base])) _Tp();
    }
    return __curr;
  }

  void __del_round(__round* __curr) noexcept {
    size_t __built  = __curr->__size_;
    __round* __next = __curr->__next_;
    size_t __base   = __next ? __next->__size_ : 0;
    size_t __size   = __bytes(__curr->__size_, __base);
    _Tp** __list    = __curr->__list();
    while (__built > __base && __list[__built - 1] == nullptr) // never-constructed tail
      --__built;
    for (size_t __i = 0; __i < __built - __base; ++__i)
      __list[__built - 1 - __i]->~_Tp();
    std::__libcpp_deallocate<byte>(reinterpret_cast<byte*>(__curr), __element_count(__size), __alignment());
  }

  __round* __at_slow(size_t __index) {
    // Optimistic concurrency: no mutex, so element constructors may themselves use the array.
    __round* __p      = __array_.load(memory_order_acquire);
    size_t __capacity = std::__bit_ceil(__index + 1);
    while (true) {
      if (__p && __index < __p->__size_)
        return __p;
      __round* __q = __new_round(__capacity, __p); // __p is updated when the race is lost early
      if (__q == nullptr)
        continue;
      if (__array_.compare_exchange_strong(__p, __q, memory_order_acq_rel, memory_order_acquire)) {
        __size_.store(__capacity, memory_order_release);
        return __q;
      }
      __del_round(__q);
    }
  }

public:
  constexpr __grow_array() noexcept            = default;
  __grow_array(const __grow_array&)            = delete;
  __grow_array& operator=(const __grow_array&) = delete;

  size_t __size() const noexcept { return __size_.load(memory_order_acquire); }

  // Reference to element __index, growing the array if needed. May throw bad_alloc.
  _Tp& operator[](size_t __index) {
    bool __fast  = __index < __size_.load(memory_order_acquire);
    __round* __p = __fast ? __array_.load(memory_order_acquire) : __at_slow(__index);
    return *__p->__list()[__index];
  }

  // Pointers to the first min(__count, size) elements constructed so far, in index order; __count is
  // clamped. Returns nullptr when the array is still empty.
  _Tp* const* __pointers(size_t& __count) const noexcept {
    __round* __p = __array_.load(memory_order_acquire);
    if (__p == nullptr) {
      __count = 0;
      return nullptr;
    }
    if (__count > __p->__size_)
      __count = __p->__size_;
    return __p->__list();
  }
};

// Per-thread cache of free records

// A per-thread stack of up to 9 free records. It sits in front of the domain's available list:
// __hazard_pointer_acquire() pops from it and __hazard_pointer_release() pushes to it, so the steady state
// costs no atomics on the domain's shared state (the call_once fast path and __libcpp_tls_get remain).
// Reached through a pthread key (see __hazard_pointer_domain::__thread_cache).
struct __hazard_pointer_thread_cache {
  __hazard_pointer_record* __entries_[__thread_cache_capacity] = {};
  uint8_t __count_                                             = 0;

  __hazard_pointer_record* __try_get() noexcept { return __count_ > 0 ? __entries_[--__count_] : nullptr; }

  bool __try_put(__hazard_pointer_record* __record) noexcept {
    if (__count_ < __thread_cache_capacity) {
      __entries_[__count_++] = __record;
      return true;
    }
    return false;
  }
};

// Sentinel stored in the TLS key by __at_thread_exit once a thread's cache has been torn down. While it is
// in the key, __thread_cache() reports "no cache", so a hazard_pointer destroyed by a destructor that runs
// after ours releases its record straight to the domain instead of touching the freed cache.
//
// The sentinel is armed when the cache is torn down. POSIX clears a TSD value before each destructor
// round, so a later round sees null again, not the sentinel. We deliberately do not re-arm it: re-arming
// would keep this thread in destructor rounds after sanitizer runtimes have already finalized it.
// Consequently, a make_hazard_pointer() called from a foreign TSD destructor in a later round may build a
// fresh cache that is never evicted -- a bounded leak of that cache object plus up to
// __thread_cache_capacity records. Records are immortal, so the cache object is the actual leak.
__hazard_pointer_thread_cache __dead_thread_cache;

// Intrusive lists of retired nodes

using __node = __hazard_pointer_obj_node;

// A (head, tail, count) list of nodes; single-threaded.
struct __node_list {
  __node* __head_ = nullptr;
  __node* __tail_ = nullptr;
  int __count_    = 0;

  bool __empty() const noexcept { return __head_ == nullptr; }

  void __push(__node* __n) noexcept {
    __n->__next_ = nullptr;
    if (__tail_ != nullptr)
      __tail_->__next_ = __n;
    else
      __head_ = __n;
    __tail_ = __n;
    ++__count_;
  }

  void __clear() noexcept {
    __head_  = nullptr;
    __tail_  = nullptr;
    __count_ = 0;
  }
};

// A head-only list shared between threads: lock-free push of a whole __node_list, wait-free pop-all.
struct __shared_node_list {
  atomic<__node*> __head_{nullptr};

  void __push(__node_list& __list) noexcept { // prepends the whole list, kept in order
    if (__list.__empty())
      return;
    __node* __old_head = __head_.load(memory_order_acquire);
    do {
      __list.__tail_->__next_ = __old_head;
    } while (!__head_.compare_exchange_weak(__old_head, __list.__head_, memory_order_acq_rel, memory_order_acquire));
    __list.__clear();
  }

  __node* __pop_all() noexcept { return __head_.exchange(nullptr, memory_order_acq_rel); }

  bool __empty() const noexcept { return __head_.load(memory_order_acquire) == nullptr; }
};

// The set of protected addresses built by every reclamation pass

// Open addressing, power-of-two table, __mix_pointer as the hash. Storage comes from nothrow new;
// __insert() returns false when the table cannot grow, in which case the pass falls back to matching
// each retired node against the records directly (see __hazard_pointer_domain::__match_reclaim).
class __protected_set {
  uintptr_t* __slots_ = nullptr; // 0 = empty; only non-null addresses are inserted
  size_t __capacity_  = 0;
  size_t __mask_      = 0;
  size_t __size_      = 0;

  void __insert_no_grow(uintptr_t __key) noexcept {
    size_t __i = __mix_pointer(__key) & __mask_;
    while (__slots_[__i] != 0) {
      if (__slots_[__i] == __key)
        return;
      __i = (__i + 1) & __mask_;
    }
    __slots_[__i] = __key;
    ++__size_;
  }

  bool __grow() noexcept {
    size_t __new_capacity  = __capacity_ == 0 ? 16 : __capacity_ * 2;
    uintptr_t* __new_slots = ::new (nothrow) uintptr_t[__new_capacity](); // zero-initialised
    if (__new_slots == nullptr)
      return false;
    uintptr_t* __old      = __slots_;
    size_t __old_capacity = __capacity_;
    __slots_              = __new_slots;
    __capacity_           = __new_capacity;
    __mask_               = __new_capacity - 1;
    __size_               = 0;
    for (size_t __i = 0; __i < __old_capacity; ++__i)
      if (__old[__i] != 0)
        __insert_no_grow(__old[__i]);
    delete[] __old;
    return true;
  }

public:
  __protected_set()                                  = default;
  __protected_set(const __protected_set&)            = delete;
  __protected_set& operator=(const __protected_set&) = delete;
  ~__protected_set() { delete[] __slots_; }

  bool __insert(const void* __p) noexcept {
    if ((__size_ + 1) * 2 > __capacity_ && !__grow())
      return false;
    __insert_no_grow(reinterpret_cast<uintptr_t>(__p));
    return true;
  }

  bool __contains(const void* __p) const noexcept {
    if (__size_ == 0)
      return false;
    uintptr_t __key = reinterpret_cast<uintptr_t>(__p);
    size_t __i      = __mix_pointer(__key) & __mask_;
    while (__slots_[__i] != 0) {
      if (__slots_[__i] == __key)
        return true;
      __i = (__i + 1) & __mask_;
    }
    return false;
  }
};

// The domain

class __hazard_pointer_domain {
  using __record = __hazard_pointer_record;

  // Records: every record ever created (never destroyed) and the free ones
  __grow_array<__record> __records_;
  atomic<int> __record_count_{0};    // records handed out so far = the prefix of __records_ the scan reads
  atomic<uintptr_t> __available_{0}; // head of the free list; low bit = lock

  // Per-thread caches
  once_flag __tls_once_;
  __libcpp_tls_key __tls_key_ = {};
  bool __tls_ok_              = false;

  // Retired objects
  __shared_node_list __retired_[__shard_count]; // sharded by node address to spread contention
  atomic<int> __retired_count_{0};              // signed: transiently negative during a pass (see below)
  // steady_clock seconds, kept word-sized: a 64-bit atomic is not lock-free on every 32-bit target, and
  // 2^32 seconds of uptime is over a century, so seconds never wrap. A pass also runs when now >= due.
  // Starting at 0 means the very first retire() in the process is always due and runs a (nearly empty)
  // pass, which is what arms the real due time.
  atomic<uintptr_t> __due_time_{0};

public:
  constexpr __hazard_pointer_domain() noexcept                       = default;
  __hazard_pointer_domain(const __hazard_pointer_domain&)            = delete;
  __hazard_pointer_domain& operator=(const __hazard_pointer_domain&) = delete;

  // Entry points

  // A free, unassociated record for the calling thread. May throw bad_alloc.
  __hazard_pointer_slot* __acquire_slot() {
    if (__hazard_pointer_thread_cache* __cache = __thread_cache(/*__create=*/true))
      if (__record* __rec = __cache->__try_get())
        return __rec;
    return __acquire_records(1);
  }

  // Ends the epoch of __slot and returns its record to the calling thread's cache, if it has one,
  // else to the domain. Never allocates a cache: a thread that only ever releases (e.g. a
  // hazard_pointer moved in from another thread, or one destroyed during thread teardown) must not
  // leave records stranded in a cache nothing will evict.
  void __release_slot(__hazard_pointer_slot* __slot) noexcept {
    __record* __rec = static_cast<__record*>(__slot);
    __rec->__value_.store(nullptr, memory_order_release);
    if (__hazard_pointer_thread_cache* __cache = __thread_cache(/*__create=*/false))
      if (__cache->__try_put(__rec))
        return;
    __push_available(__rec, __rec);
  }

  // Retires __node: the tail of hazard_pointer_obj_base::retire(). May run a reclamation pass (and thus
  // deleters) synchronously.
  void __retire(__node* __n) noexcept {
    // Retire-side fence: orders the user's store to `src` (sequenced before this call) before the
    // reclaimer's scan, even when the reclaimer is another thread. See <__hazard_pointer/domain.h>.
    std::atomic_thread_fence(memory_order_seq_cst);
    __node_list __list;
    __list.__push(__n);
    __retired_[__shard_of(__n)].__push(__list);
    __retired_count_.fetch_add(1, memory_order_release);
    __check_threshold_and_reclaim();
  }

private:
  // Trigger

  static uintptr_t __now_s() noexcept {
    return static_cast<uintptr_t>(
        chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now().time_since_epoch()).count());
  }

  static size_t __shard_of(const __node* __n) noexcept {
    return (__mix_pointer(reinterpret_cast<uintptr_t>(__n)) >> __ignored_low_bits) & __shard_mask;
  }

  int __threshold() const noexcept {
    int __records = __record_count_.load(memory_order_relaxed);
    int __scaled  = __threshold_multiplier * __records;
    return __scaled > __retired_count_threshold ? __scaled : __retired_count_threshold;
  }

  // If the retired count reached the threshold, claims it (resets it to 0) and returns it; else 0.
  int __check_count_threshold() noexcept {
    int __count = __retired_count_.load(memory_order_acquire);
    while (__count >= __threshold()) {
      if (__retired_count_.compare_exchange_weak(__count, 0, memory_order_acq_rel, memory_order_relaxed)) {
        __due_time_.store(__now_s() + __sync_time_period_s, memory_order_release);
        return __count;
      }
    }
    return 0;
  }

  // If the due time has passed, claims the current count (may be small) and returns it; else 0.
  int __check_due_time() noexcept {
    uintptr_t __time = __now_s();
    uintptr_t __due  = __due_time_.load(memory_order_acquire);
    if (__time < __due || !__due_time_.compare_exchange_strong(
                              __due, __time + __sync_time_period_s, memory_order_acq_rel, memory_order_relaxed))
      return 0;
    int __count = __retired_count_.exchange(0, memory_order_acq_rel);
    if (__count < 0) {
      __retired_count_.fetch_add(__count, memory_order_release);
      return 0;
    }
    return __count;
  }

  void __check_threshold_and_reclaim() noexcept {
    int __count = __check_count_threshold();
    if (__count == 0) {
      __count = __check_due_time();
      if (__count == 0)
        return;
    }
    __do_reclamation(__count);
  }

  // Reclamation pass

  bool __retired_empty() const noexcept {
    for (const __shared_node_list& __shard : __retired_)
      if (!__shard.__empty())
        return false;
    return true;
  }

  // Pops every shard into __extracted; returns the number of nodes taken.
  int __extract_retired(__node* __extracted[__shard_count]) noexcept {
    int __taken = 0;
    for (int __s = 0; __s < __shard_count; ++__s) {
      __extracted[__s] = __retired_[__s].__pop_all();
      for (__node* __n = __extracted[__s]; __n != nullptr; __n = __n->__next_)
        ++__taken;
    }
    return __taken;
  }

  // Loads every non-null slot value into __set. Returns false when the set could not be built (allocation
  // failure); the caller then matches linearly.
  bool __load_protected(__protected_set& __set) noexcept {
    int __handed_out        = __record_count_.load(memory_order_relaxed);
    size_t __count          = __handed_out > 0 ? static_cast<size_t>(__handed_out) : 0;
    __record* const* __recs = __records_.__pointers(__count); // clamps __count to what exists
#if _LIBCPP_HAZARD_POINTER_TSAN
    constexpr memory_order __order = memory_order_acquire; // TSan does not model fences
#else
    constexpr memory_order __order = memory_order_relaxed; // the fence below provides the acquire ordering
#endif
    // Load a small batch up front so the branches below do not form one long dependency chain.
    constexpr size_t __chunk = 8;
    size_t __i               = 0;
    for (; __i + __chunk <= __count; __i += __chunk) {
      const void* __values[__chunk];
      for (size_t __j = 0; __j < __chunk; ++__j)
        __values[__j] = __recs[__i + __j]->__value_.load(__order);
      for (const void* __v : __values)
        if (__v != nullptr && !__set.__insert(__v))
          return false;
    }
    for (; __i < __count; ++__i) {
      const void* __v = __recs[__i]->__value_.load(__order);
      if (__v != nullptr && !__set.__insert(__v))
        return false;
    }
    std::atomic_thread_fence(memory_order_acquire);
    return true;
  }

  // Allocation-free fallback: is __n currently held by any record? O(records).
  bool __protected_linear(const __node* __n) const noexcept {
    int __handed_out        = __record_count_.load(memory_order_relaxed);
    size_t __count          = __handed_out > 0 ? static_cast<size_t>(__handed_out) : 0;
    __record* const* __recs = __records_.__pointers(__count);
    for (size_t __i = 0; __i < __count; ++__i)
      if (__recs[__i]->__value_.load(memory_order_acquire) == static_cast<const void*>(__n))
        return true;
    return false;
  }

  // Reclaims the unprotected extracted nodes and pushes the protected ones back. Returns the number
  // pushed back; sets __done = false when new nodes were retired meanwhile (e.g. by a deleter).
  int __match_reclaim(__node* __retired[__shard_count], const __protected_set* __set, bool& __done) noexcept {
    __done = true;
    __node_list __not_reclaimed[__shard_count];
    __node* __heads[__shard_count];
    for (int __s = 0; __s < __shard_count; ++__s)
      __heads[__s] = __retired[__s];
    for (bool __more = true; __more;) { // walk the shards in lock-step (instruction-level parallelism)
      __more = false;
      for (int __s = 0; __s < __shard_count; ++__s) {
        __node* __n = __heads[__s];
        if (__n == nullptr)
          continue;
        __more              = true;
        __heads[__s]        = __n->__next_; // read before the deleter may free __n
        bool __is_protected = __set != nullptr ? __set->__contains(__n) : __protected_linear(__n);
        if (__is_protected)
          __not_reclaimed[__s].__push(__n);
        else
          __n->__reclaim_(__n);
      }
    }
    if (!__retired_empty())
      __done = false;
    int __residue = 0;
    for (int __s = 0; __s < __shard_count; ++__s) {
      __residue += __not_reclaimed[__s].__count_;
      __retired_[__s].__push(__not_reclaimed[__s]);
    }
    return __residue;
  }

  // The bulk reclamation pass. __count is the number of retirements claimed by the trigger. The claim is
  // settled against what the pass actually extracted before any deleter runs, so that from then on
  // __retired_count_ again counts the nodes sitting in the shards (transiently negative when nodes retired
  // concurrently were extracted before their retire() counted them); the protected residue is added back
  // when it is returned to the shards.
  void __do_reclamation(int __count) noexcept {
    while (true) {
      __node* __retired[__shard_count];
      bool __done = true;
      if (int __taken = __extract_retired(__retired)) {
        // Settling now rather than after the deleters matters when two passes overlap: the credit a pass
        // claimed for nodes another pass then extracted would otherwise sit in the counter, above the
        // threshold, for the whole duration of that other pass, and every retire() in between would run
        // a fruitless full pass.
        __count -= __taken;
        if (__count != 0)
          __retired_count_.fetch_add(__count, memory_order_release);
        // Reclaimer-side fence: pairs with the readers' fence in try_protect() and the retirers' fence in
        // __retire(). See <__hazard_pointer/domain.h>.
        std::atomic_thread_fence(memory_order_seq_cst);
        __protected_set __set;
        const __protected_set* __set_ptr = __load_protected(__set) ? &__set : nullptr;
        // Residue bound: residue <= records < max(1000, 2 * records) == __threshold(), because a node
        // survives a pass only while some record protects it and distinct nodes have distinct addresses.
        if (int __residue = __match_reclaim(__retired, __set_ptr, __done))
          __retired_count_.fetch_add(__residue, memory_order_release);
      } else if (__count != 0) {
        // Nothing to hold the claim against (another pass took the nodes): give it back and return
        // rather than claiming it again; the owning pass has settled or will settle its own extract.
        __retired_count_.fetch_add(__count, memory_order_release);
      }
      if (__done)
        break;
      __count = __check_count_threshold();
    }
  }

public:
  static void _LIBCPP_TLS_DESTRUCTOR_CC __at_thread_exit(void* __p);

private:
  // Available list (lock-bit stack of free records)

  uintptr_t __load_available() const noexcept { return __available_.load(memory_order_acquire); }
  void __store_available(uintptr_t __value) noexcept { __available_.store(__value, memory_order_release); }
  bool __cas_available(uintptr_t& __expected, uintptr_t __desired) noexcept {
    return __available_.compare_exchange_weak(__expected, __desired, memory_order_acq_rel, memory_order_acquire);
  }

  // Pops up to __count records; returns the head of a __next_available_-linked chain and sets __popped.
  // __count is only ever 1 today; the batch capability is kept for a future batch-acquire entry point
  // (P3428R4's hazard_pointer_array).
  __record* __try_pop_available(uint8_t __count, uint8_t& __popped) noexcept {
    while (true) {
      uintptr_t __available = __load_available();
      if (__available == 0) {
        __popped = 0;
        return nullptr;
      }
      if ((__available & __lock_bit) == 0) {
        if (__cas_available(__available, __available | __lock_bit)) { // locked
          __record* __head = reinterpret_cast<__record*>(__available);
          __popped         = __pop_available_release_lock(__count, __head);
          return __head;
        }
      } else {
        std::__libcpp_thread_yield();
      }
    }
  }

  // Lock already held: detaches up to __count records starting at __head, then stores the new head
  // (which releases the lock).
  uint8_t __pop_available_release_lock(uint8_t __count, __record* __head) noexcept {
    __record* __tail = __head;
    uint8_t __taken  = 1;
    __record* __next = __tail->__next_available_;
    while (__next != nullptr && __taken < __count) {
      __tail = __next;
      __next = __tail->__next_available_;
      ++__taken;
    }
    _LIBCPP_ASSERT_INTERNAL(
        (reinterpret_cast<uintptr_t>(__next) & __lock_bit) == 0, "hazard pointer record misaligned");
    __store_available(reinterpret_cast<uintptr_t>(__next)); // releases the lock
    __tail->__next_available_ = nullptr;
    return __taken;
  }

  // Pushes the chain [__head, __tail] (tail's link is overwritten) onto the available list.
  void __push_available(__record* __head, __record* __tail) noexcept {
    uintptr_t __new_head = reinterpret_cast<uintptr_t>(__head);
    _LIBCPP_ASSERT_INTERNAL((__new_head & __lock_bit) == 0, "hazard pointer record misaligned");
    while (true) {
      uintptr_t __available = __load_available();
      if ((__available & __lock_bit) == 0) {
        __tail->__next_available_ = reinterpret_cast<__record*>(__available);
        if (__cas_available(__available, __new_head))
          return;
      } else {
        std::__libcpp_thread_yield();
      }
    }
  }

  // Record creation

  __record* __create_record() { // may throw bad_alloc
    // Any scan that must see the new record sees this increment and the grow-array round holding the
    // record: acquiring a hazard pointer precedes the reader's seq_cst fence in try_protect(), and when
    // that fence precedes the reclaimer's seq_cst fence in the fences' total order, the scan's loads observe
    // every write that happens-before the reader's fence -- including the round-publishing CAS of another
    // thread that this one merely observed (C++20 [atomics.order]/4.4, P0668; the same cumulativity RC11
    // and every hardware mapping we target provide).
    //
    // The count is bumped before the element exists. If __records_[] throws, the domain has counted one
    // record more than it built; the scan clamps to what the grow array actually holds (see __pointers),
    // so the over-count only makes __threshold() marginally larger.
    int __index = __record_count_.fetch_add(1, memory_order_relaxed);
    return &__records_[static_cast<size_t>(__index)];
  }

  // Pops __count records from the available list, creating the shortfall. Returns the head of a
  // __next_available_-linked chain of exactly __count records. As with __try_pop_available, __count is
  // always 1 today; batching is kept for a future batch-acquire entry point (P3428R4).
  __record* __acquire_records(uint8_t __count) {
    uint8_t __popped = 0;
    __record* __head = __try_pop_available(__count, __popped);
    for (; __popped < __count; ++__popped) {
      __record* __created          = __create_record();
      __created->__next_available_ = __head;
      __head                       = __created;
    }
    return __head;
  }

  // Thread cache plumbing

  void __init_tls() noexcept {
    std::call_once(__tls_once_, [this] {
      __tls_ok_ = std::__libcpp_tls_create(&__tls_key_, &__hazard_pointer_domain::__at_thread_exit) == 0;
    });
  }

  // The calling thread's cache, or nullptr when the thread has none (TLS unavailable, allocation failed,
  // or its cache was already torn down). __create: allocate one on first use.
  __hazard_pointer_thread_cache* __thread_cache(bool __create) noexcept {
    __init_tls();
    if (!__tls_ok_)
      return nullptr;
    void* __p = std::__libcpp_tls_get(__tls_key_);
    if (__p == static_cast<void*>(&__dead_thread_cache))
      return nullptr;
    if (__p == nullptr && __create) {
      __p = ::new (nothrow) __hazard_pointer_thread_cache();
      if (__p != nullptr && std::__libcpp_tls_set(__tls_key_, __p) != 0) {
        delete static_cast<__hazard_pointer_thread_cache*>(__p);
        __p = nullptr;
      }
    }
    return static_cast<__hazard_pointer_thread_cache*>(__p);
  }

  // Returns every cached record to the available list.
  void __evict(__hazard_pointer_thread_cache& __cache) noexcept {
    if (__cache.__count_ == 0)
      return;
    __record* __head = nullptr;
    __record* __tail = nullptr;
    while (__cache.__count_ > 0) {
      __record* __rec          = __cache.__entries_[--__cache.__count_];
      __rec->__next_available_ = __head;
      __head                   = __rec;
      if (__tail == nullptr)
        __tail = __rec;
    }
    __push_available(__head, __tail);
  }
};

constinit __hazard_pointer_domain __domain;

// The TSD destructor of the thread cache. POSIX clears the key's value before each destructor round, so
// round 1 arrives with the cache -- return its records to the domain, free it, and arm the sentinel -- and
// round 2 arrives with the sentinel and does nothing, which leaves the key null for good. Deliberately no
// re-arming: see the note above __dead_thread_cache.
void _LIBCPP_TLS_DESTRUCTOR_CC __hazard_pointer_domain::__at_thread_exit(void* __p) {
  if (__p == nullptr || __p == static_cast<void*>(&__dead_thread_cache))
    return;
  __hazard_pointer_thread_cache* __cache = static_cast<__hazard_pointer_thread_cache*>(__p);
  __domain.__evict(*__cache);
  delete __cache;
  std::__libcpp_tls_set(__domain.__tls_key_, &__dead_thread_cache);
}

} // namespace

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_EXPORTED_FROM_ABI __hazard_pointer_slot* __hazard_pointer_acquire() { return __domain.__acquire_slot(); }

_LIBCPP_EXPORTED_FROM_ABI void __hazard_pointer_release(__hazard_pointer_slot* __slot) noexcept {
  __domain.__release_slot(__slot);
}

_LIBCPP_EXPORTED_FROM_ABI void __hazard_pointer_retire(__hazard_pointer_obj_node* __n) noexcept {
  __domain.__retire(__n);
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_END_NAMESPACE_STD
