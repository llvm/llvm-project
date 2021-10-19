
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

void Thread::InitRandomState() {
  random_state_ = flags()->random_tags ? RandomSeed() : unique_id_;

  // Push a random number of zeros onto the ring buffer so that the first stack
  // tag base will be random.
  for (tag_t i = 0, e = GenerateRandomTag(); i != e; ++i)
    stack_allocations_->push(0);
}

void Thread::Init(uptr stack_buffer_start, uptr stack_buffer_size,
                  const InitState *state) {
  CHECK_EQ(0, unique_id_);  // try to catch bad stack reuse
  CHECK_EQ(0, stack_top_);
  CHECK_EQ(0, stack_bottom_);

  static u64 unique_id;
  unique_id_ = unique_id++;
  if (auto sz = flags()->heap_history_size)
    heap_allocations_ = HeapAllocationsRingBuffer::New(sz);

#if !SANITIZER_FUCHSIA
  // Do not initialize the stack ring buffer just yet on Fuchsia. Threads will
  // be initialized before we enter the thread itself, so we will instead call
  // this later.
  InitStackRingBuffer(stack_buffer_start, stack_buffer_size);
#endif
  InitStackAndTls(state);
}

void Thread::InitStackRingBuffer(uptr stack_buffer_start,
                                 uptr stack_buffer_size) {
  HwasanTSDThreadInit();  // Only needed with interceptors.
  uptr *ThreadLong = GetCurrentThreadLongPtr();
  // The following implicitly sets (this) as the current thread.
  stack_allocations_ = new (ThreadLong)
      StackAllocationsRingBuffer((void *)stack_buffer_start, stack_buffer_size);
  // Check that it worked.
  CHECK_EQ(GetCurrentThread(), this);

  // ScopedTaggingDisable needs GetCurrentThread to be set up.
  ScopedTaggingDisabler disabler;

  if (stack_bottom_) {
    int local;
    CHECK(AddrIsInStack((uptr)&local));
    CHECK(MemIsApp(stack_bottom_));
    CHECK(MemIsApp(stack_top_ - 1));
  }

  if (flags()->verbose_threads) {
    if (IsMainThread()) {
      Printf("sizeof(Thread): %zd sizeof(HeapRB): %zd sizeof(StackRB): %zd\n",
             sizeof(Thread), heap_allocations_->SizeInBytes(),
             stack_allocations_->size() * sizeof(uptr));
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
  if (heap_allocations_)
    heap_allocations_->Delete();
  DTLS_Destroy();
  // Unregister this as the current thread.
  // Instrumented code can not run on this thread from this point onwards, but
  // malloc/free can still be served. Glibc may call free() very late, after all
  // TSD destructors are done.
  CHECK_EQ(GetCurrentThread(), this);
  *GetCurrentThreadLongPtr() = 0;
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
tag_t Thread::GenerateRandomTag(uptr num_bits) {
  DCHECK_GT(num_bits, 0);
  if (tagging_disabled_) return 0;
  tag_t tag;
  const uptr tag_mask = (1ULL << num_bits) - 1;
  do {
    if (flags()->random_tags) {
      if (!random_buffer_)
        random_buffer_ = random_state_ = xorshift(random_state_);
      CHECK(random_buffer_);
      tag = random_buffer_ & tag_mask;
      random_buffer_ >>= num_bits;
    } else {
      random_state_ += 1;
      tag = random_state_ & tag_mask;
    }
  } while (!tag);
  return tag;
}

} // namespace __hwasan
