
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

Thread *Thread::main_thread;
SpinMutex Thread::thread_list_mutex;

void Thread::InsertIntoThreadList(Thread *t) {
  CHECK(!t->next_);
  if (!main_thread) {
    main_thread = t;
    return;
  }
  SpinMutexLock l(&thread_list_mutex);
  Thread *last = main_thread;
  while (last->next_)
    last = last->next_;
  last->next_ = t;
}

void Thread::RemoveFromThreadList(Thread *t) {
  CHECK_NE(t, main_thread);
  SpinMutexLock l(&thread_list_mutex);
  Thread *prev = main_thread;
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

Thread *Thread::Create(thread_callback_t start_routine,
                               void *arg) {
  uptr PageSize = GetPageSizeCached();
  uptr size = RoundUpTo(sizeof(Thread), PageSize);
  Thread *thread = (Thread*)MmapOrDie(size, __func__);
  thread->start_routine_ = start_routine;
  thread->arg_ = arg;
  thread->destructor_iterations_ = GetPthreadDestructorIterations();
  thread->random_state_ = flags()->random_tags ? RandomSeed() : 0;
  if (auto sz = flags()->heap_history_size)
    thread->heap_allocations_ = RingBuffer<HeapAllocationRecord>::New(sz);
  InsertIntoThreadList(thread);
  return thread;
}

void Thread::SetThreadStackAndTls() {
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
}

void Thread::Init() {
  SetThreadStackAndTls();
  if (stack_bottom_) {
    CHECK(MemIsApp(stack_bottom_));
    CHECK(MemIsApp(stack_top_ - 1));
  }
}

void Thread::ClearShadowForThreadStackAndTLS() {
  if (stack_top_ != stack_bottom_)
    TagMemory(stack_bottom_, stack_top_ - stack_bottom_, 0);
  if (tls_begin_ != tls_end_)
    TagMemory(tls_begin_, tls_end_ - tls_begin_, 0);
}

void Thread::Destroy() {
  malloc_storage().CommitBack();
  ClearShadowForThreadStackAndTLS();
  RemoveFromThreadList(this);
  uptr size = RoundUpTo(sizeof(Thread), GetPageSizeCached());
  if (heap_allocations_)
    heap_allocations_->Delete();
  UnmapOrDie(this, size);
  DTLS_Destroy();
}

static u32 xorshift(u32 state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// Generate a (pseudo-)random non-zero tag.
tag_t Thread::GenerateRandomTag() {
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
