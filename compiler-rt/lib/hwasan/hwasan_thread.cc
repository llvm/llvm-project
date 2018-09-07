
#include "hwasan.h"
#include "hwasan_mapping.h"
#include "hwasan_thread.h"
#include "hwasan_poisoning.h"
#include "hwasan_interface_internal.h"

#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

namespace __hwasan {

static u32 RandomSeed() {
  u32 seed;
  do {
    if (UNLIKELY(!GetRandom(reinterpret_cast<void *>(&seed), sizeof(seed),
                            /*blocking=*/false))) {
      seed = static_cast<u32>(
          (NanoTime() >> 12) ^
          (reinterpret_cast<uptr>(__builtin_frame_address(0)) >> 4));
    }
  } while (!seed);
  return seed;
}

Thread *Thread::thread_list_head;
SpinMutex Thread::thread_list_mutex;
Thread::ThreadStats Thread::thread_stats;

void Thread::InsertIntoThreadList(Thread *t) {
  CHECK(!t->next_);
  SpinMutexLock l(&thread_list_mutex);
  thread_stats.n_live_threads++;
  thread_stats.total_stack_size += t->stack_size();
  if (!thread_list_head) {
    thread_list_head = t;
    return;
  }
  Thread *last = thread_list_head;
  while (last->next_)
    last = last->next_;
  last->next_ = t;
}

void Thread::RemoveFromThreadList(Thread *t) {
  SpinMutexLock l(&thread_list_mutex);
  thread_stats.n_live_threads--;
  thread_stats.total_stack_size -= t->stack_size();
  if (t == thread_list_head) {
    thread_list_head = t->next_;
    t->next_ = nullptr;
    return;
  }
  Thread *prev = thread_list_head;
  Thread *cur = prev->next_;
  CHECK(cur);
  while (cur) {
    if (cur == t) {
      prev->next_ = cur->next_;
      return;
    }
    prev = cur;
    cur = cur->next_;
  }
  CHECK(0 && "RemoveFromThreadList: thread not found");
}

void Thread::Create() {
  static u64 unique_id;
  uptr PageSize = GetPageSizeCached();
  uptr size = RoundUpTo(sizeof(Thread), PageSize);
  Thread *thread = (Thread*)MmapOrDie(size, __func__);
  thread->destructor_iterations_ = GetPthreadDestructorIterations();
  thread->unique_id_ = unique_id++;
  thread->random_state_ =
      flags()->random_tags ? RandomSeed() : thread->unique_id_;
  if (auto sz = flags()->heap_history_size)
    thread->heap_allocations_ = HeapAllocationsRingBuffer::New(sz);
  SetCurrentThread(thread);
  thread->Init();
  InsertIntoThreadList(thread);
}

uptr Thread::MemoryUsedPerThread() {
  uptr res = sizeof(Thread);
  if (auto sz = flags()->heap_history_size)
    res += HeapAllocationsRingBuffer::SizeInBytes(sz);
  return res;
}

void Thread::Init() {
  // GetPthreadDestructorIterations may call malloc, so disable the tagging.
  ScopedTaggingDisabler disabler;

  // If this process is "init" (pid 1), /proc may not be mounted yet.
  if (IsMainThread() && !FileExists("/proc/self/maps")) {
    stack_top_ = stack_bottom_ = 0;
    tls_begin_ = tls_end_ = 0;
    return;
  }

  uptr tls_size;
  uptr stack_size;
  GetThreadStackAndTls(IsMainThread(), &stack_bottom_, &stack_size, &tls_begin_,
                       &tls_size);
  stack_top_ = stack_bottom_ + stack_size;
  tls_end_ = tls_begin_ + tls_size;

  int local;
  CHECK(AddrIsInStack((uptr)&local));
  CHECK(MemIsApp(stack_bottom_));
  CHECK(MemIsApp(stack_top_ - 1));

  if (stack_bottom_) {
    CHECK(MemIsApp(stack_bottom_));
    CHECK(MemIsApp(stack_top_ - 1));
  }
  if (flags()->verbose_threads) {
    if (IsMainThread()) {
      Printf("sizeof(Thread): %zd sizeof(RB): %zd\n", sizeof(Thread),
             heap_allocations_->SizeInBytes());
    }
    Print("Creating  : ");
  }
}

void Thread::ClearShadowForThreadStackAndTLS() {
  if (stack_top_ != stack_bottom_)
    TagMemory(stack_bottom_, stack_top_ - stack_bottom_, 0);
  if (tls_begin_ != tls_end_)
    TagMemory(tls_begin_, tls_end_ - tls_begin_, 0);
}

void Thread::Destroy() {
  if (flags()->verbose_threads)
    Print("Destroying: ");
  AllocatorSwallowThreadLocalCache(allocator_cache());
  ClearShadowForThreadStackAndTLS();
  RemoveFromThreadList(this);
  uptr size = RoundUpTo(sizeof(Thread), GetPageSizeCached());
  if (heap_allocations_)
    heap_allocations_->Delete();
  UnmapOrDie(this, size);
  DTLS_Destroy();
}

void Thread::Print(const char *Prefix) {
  Printf("%sT%zd %p stack: [%p,%p) sz: %zd tls: [%p,%p)\n", Prefix,
         unique_id_, this, stack_bottom(), stack_top(),
         stack_top() - stack_bottom(),
         tls_begin(), tls_end());
}

static u32 xorshift(u32 state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// Generate a (pseudo-)random non-zero tag.
tag_t Thread::GenerateRandomTag() {
  if (tagging_disabled_) return 0;
  tag_t tag;
  do {
    if (flags()->random_tags) {
      if (!random_buffer_)
        random_buffer_ = random_state_ = xorshift(random_state_);
      CHECK(random_buffer_);
      tag = random_buffer_ & 0xFF;
      random_buffer_ >>= 8;
    } else {
      tag = random_state_ = (random_state_ + 1) & 0xFF;
    }
  } while (!tag);
  return tag;
}

} // namespace __hwasan
