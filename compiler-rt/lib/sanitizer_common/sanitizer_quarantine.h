//===-- sanitizer_quarantine.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Memory quarantine for AddressSanitizer and potentially other tools.
// Quarantine caches some specified amount of memory in per-thread caches,
// then evicts to global FIFO queue. When the queue reaches specified threshold,
// oldest memory is recycled.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_QUARANTINE_H
#define SANITIZER_QUARANTINE_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"
#include "sanitizer_list.h"

namespace __sanitizer {

template<typename Node> class QuarantineCache;

struct QuarantineBatch {
  static const uptr kSize = 1021;
  QuarantineBatch *next;
  uptr size;
  uptr count;
  void *batch[kSize];
};

COMPILER_CHECK(sizeof(QuarantineBatch) <= (1 << 13));  // 8Kb.

// The callback interface is:
// void Callback::Recycle(Node *ptr);
// void *cb.Allocate(uptr size);
// void cb.Deallocate(void *ptr);
template<typename Callback, typename Node>
class Quarantine {
 public:
  typedef QuarantineCache<Callback> Cache;

  explicit Quarantine(LinkerInitialized)
      : cache_(LINKER_INITIALIZED) {
  }

  void Init(uptr size, uptr cache_size) {
    atomic_store(&max_size_, size, memory_order_release);
    atomic_store(&min_size_, size / 10 * 9,
                 memory_order_release); // 90% of max size.
    max_cache_size_ = cache_size;
  }

  uptr GetSize() const { return atomic_load(&max_size_, memory_order_acquire); }

  void Put(Cache *c, Callback cb, Node *ptr, uptr size) {
    c->Enqueue(cb, ptr, size);
    if (c->Size() > max_cache_size_)
      Drain(c, cb);
  }

  void NOINLINE Drain(Cache *c, Callback cb) {
    {
      SpinMutexLock l(&cache_mutex_);
      cache_.Transfer(c);
    }
    if (cache_.Size() > GetSize() && recycle_mutex_.TryLock())
      Recycle(cb);
  }

 private:
  // Read-only data.
  char pad0_[kCacheLineSize];
  atomic_uintptr_t max_size_;
  atomic_uintptr_t min_size_;
  uptr max_cache_size_;
  char pad1_[kCacheLineSize];
  SpinMutex cache_mutex_;
  SpinMutex recycle_mutex_;
  Cache cache_;
  char pad2_[kCacheLineSize];

  void NOINLINE Recycle(Callback cb) {
    Cache tmp;
    uptr min_size = atomic_load(&min_size_, memory_order_acquire);
    {
      SpinMutexLock l(&cache_mutex_);
      while (cache_.Size() > min_size) {
        QuarantineBatch *b = cache_.DequeueBatch();
        tmp.EnqueueBatch(b);
      }
    }
    recycle_mutex_.Unlock();
    DoRecycle(&tmp, cb);
  }

  void NOINLINE DoRecycle(Cache *c, Callback cb) {
    while (QuarantineBatch *b = c->DequeueBatch()) {
      const uptr kPrefetch = 16;
      for (uptr i = 0; i < kPrefetch; i++)
        PREFETCH(b->batch[i]);
      for (uptr i = 0; i < b->count; i++) {
        PREFETCH(b->batch[i + kPrefetch]);
        cb.Recycle((Node*)b->batch[i]);
      }
      cb.Deallocate(b);
    }
  }
};

// Per-thread cache of memory blocks.
template<typename Callback>
class QuarantineCache {
 public:
  explicit QuarantineCache(LinkerInitialized) {
  }

  QuarantineCache()
      : size_() {
    list_.clear();
  }

  uptr Size() const {
    return atomic_load(&size_, memory_order_relaxed);
  }

  void Enqueue(Callback cb, void *ptr, uptr size) {
    if (list_.empty() || list_.back()->count == QuarantineBatch::kSize) {
      AllocBatch(cb);
      size += sizeof(QuarantineBatch);  // Count the batch in Quarantine size.
    }
    QuarantineBatch *b = list_.back();
    CHECK(b);
    b->batch[b->count++] = ptr;
    b->size += size;
    SizeAdd(size);
  }

  void Transfer(QuarantineCache *c) {
    list_.append_back(&c->list_);
    SizeAdd(c->Size());
    atomic_store(&c->size_, 0, memory_order_relaxed);
  }

  void EnqueueBatch(QuarantineBatch *b) {
    list_.push_back(b);
    SizeAdd(b->size);
  }

  QuarantineBatch *DequeueBatch() {
    if (list_.empty())
      return nullptr;
    QuarantineBatch *b = list_.front();
    list_.pop_front();
    SizeSub(b->size);
    return b;
  }

 private:
  IntrusiveList<QuarantineBatch> list_;
  atomic_uintptr_t size_;

  void SizeAdd(uptr add) {
    atomic_store(&size_, Size() + add, memory_order_relaxed);
  }
  void SizeSub(uptr sub) {
    atomic_store(&size_, Size() - sub, memory_order_relaxed);
  }

  NOINLINE QuarantineBatch* AllocBatch(Callback cb) {
    QuarantineBatch *b = (QuarantineBatch *)cb.Allocate(sizeof(*b));
    CHECK(b);
    b->count = 0;
    b->size = 0;
    list_.push_back(b);
    return b;
  }
};
} // namespace __sanitizer

#endif // SANITIZER_QUARANTINE_H
