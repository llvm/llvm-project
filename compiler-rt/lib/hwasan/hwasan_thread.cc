
#include "hwasan.h"
#include "hwasan_mapping.h"
#include "hwasan_thread.h"
#include "hwasan_poisoning.h"
#include "hwasan_interface_internal.h"

#include "sanitizer_common/sanitizer_file.h"
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

HwasanThread *HwasanThread::Create(thread_callback_t start_routine,
                               void *arg) {
  uptr PageSize = GetPageSizeCached();
  uptr size = RoundUpTo(sizeof(HwasanThread), PageSize);
  HwasanThread *thread = (HwasanThread*)MmapOrDie(size, __func__);
  thread->start_routine_ = start_routine;
  thread->arg_ = arg;
  thread->destructor_iterations_ = GetPthreadDestructorIterations();
  thread->random_state_ = flags()->random_tags ? RandomSeed() : 0;

  return thread;
}

void HwasanThread::SetThreadStackAndTls() {
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
  CHECK(MEM_IS_APP(stack_bottom_));
  CHECK(MEM_IS_APP(stack_top_ - 1));
}

void HwasanThread::Init() {
  SetThreadStackAndTls();
  if (stack_bottom_) {
    CHECK(MEM_IS_APP(stack_bottom_));
    CHECK(MEM_IS_APP(stack_top_ - 1));
  }
}

void HwasanThread::ClearShadowForThreadStackAndTLS() {
  if (stack_top_ != stack_bottom_)
    TagMemory(stack_bottom_, stack_top_ - stack_bottom_, 0);
  if (tls_begin_ != tls_end_)
    TagMemory(tls_begin_, tls_end_ - tls_begin_, 0);
}

void HwasanThread::Destroy() {
  malloc_storage().CommitBack();
  ClearShadowForThreadStackAndTLS();
  uptr size = RoundUpTo(sizeof(HwasanThread), GetPageSizeCached());
  UnmapOrDie(this, size);
  DTLS_Destroy();
}

thread_return_t HwasanThread::ThreadStart() {
  return start_routine_(arg_);
}

static u32 xorshift(u32 state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// Generate a (pseudo-)random non-zero tag.
tag_t HwasanThread::GenerateRandomTag() {
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
