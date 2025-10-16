//===-- asan_fake_stack.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header for asan_fake_stack.cpp, implements FakeStack.
//===----------------------------------------------------------------------===//

#ifndef ASAN_FAKE_STACK_H
#define ASAN_FAKE_STACK_H

#include "sanitizer_common/sanitizer_common.h"

namespace __asan {

// Fake stack frame contains local variables of one function.
struct FakeFrame {
  uptr magic;  // Modified by the instrumented code.
  uptr descr;  // Modified by the instrumented code.
  uptr pc;     // Modified by the instrumented code.
  uptr real_stack;
};

// For each thread we create a fake stack and place stack objects on this fake
// stack instead of the real stack. The fake stack is not really a stack but
// a fast malloc-like allocator so that when a function exits the fake stack
// is not popped but remains there for quite some time until gets used again.
// So, we poison the objects on the fake stack when function returns.
// It helps us find use-after-return bugs.
// The FakeStack objects is allocated by a single mmap call and has no other
// pointers. The size of the fake stack depends on the actual thread stack size
// and thus can not be a constant.
// stack_size is a power of two greater or equal to the thread's stack size;
// we store it as its logarithm (stack_size_log).
// FakeStack is padded such that GetFrame() is aligned to BytesInSizeClass().
// FakeStack has kNumberOfSizeClasses (11) size classes, each size class
// is a power of two, starting from 64 bytes. Each size class occupies
// stack_size bytes and thus can allocate
// NumberOfFrames=(stack_size/BytesInSizeClass) fake frames (also a power of 2).
// For each size class we have NumberOfFrames allocation flags,
// each flag indicates whether the given frame is currently allocated.
// All flags for size classes 0 .. 10 are stored in a single contiguous region
// followed by another contiguous region which contains the actual memory for
// size classes. The addresses are computed by GetFlags and GetFrame without
// any memory accesses solely based on 'this' and stack_size_log.
// Allocate() flips the appropriate allocation flag atomically, thus achieving
// async-signal safety.
// This allocator does not have quarantine per se, but it tries to allocate the
// frames in round robin fashion to maximize the delay between a deallocation
// and the next allocation.
class FakeStack {
  static const uptr kMinStackFrameSizeLog = 6;  // Min frame is 64B.
  static const uptr kMaxStackFrameSizeLog = 16;  // Max stack frame is 64K.
  static_assert(kMaxStackFrameSizeLog >= kMinStackFrameSizeLog);

  static const u64 kMaxStackFrameSize = 1 << kMaxStackFrameSizeLog;

 public:
  static const uptr kNumberOfSizeClasses =
       kMaxStackFrameSizeLog - kMinStackFrameSizeLog + 1;

  // CTOR: create the FakeStack as a single mmap-ed object.
  static FakeStack *Create(uptr stack_size_log);

  void Destroy(int tid);

  // min_uar_stack_size_log is 16 (stack_size >= 64KB)
  static uptr SizeRequiredForFlags(uptr stack_size_log) {
    return ((uptr)1) << (stack_size_log + 1 - kMinStackFrameSizeLog);
  }

  // Each size class occupies stack_size bytes.
  static uptr SizeRequiredForFrames(uptr stack_size_log) {
    return (((uptr)1) << stack_size_log) * kNumberOfSizeClasses;
  }

  // Number of bytes requires for the whole object.
  static uptr RequiredSize(uptr stack_size_log) {
    return kFlagsOffset + SizeRequiredForFlags(stack_size_log) +
           SizeRequiredForFrames(stack_size_log);
  }

  // Offset of the given flag from the first flag.
  // The flags for class 0 begin at offset  000000000
  // The flags for class 1 begin at offset  100000000
  // ....................2................  110000000
  // ....................3................  111000000
  // and so on.
  static uptr FlagsOffset(uptr stack_size_log, uptr class_id) {
    uptr t = kNumberOfSizeClasses - 1 - class_id;
    const uptr all_ones = (((uptr)1) << (kNumberOfSizeClasses - 1)) - 1;
    return ((all_ones >> t) << t) << (stack_size_log - 15);
  }

  static uptr NumberOfFrames(uptr stack_size_log, uptr class_id) {
    return ((uptr)1) << (stack_size_log - kMinStackFrameSizeLog - class_id);
  }

  // Divide n by the number of frames in size class.
  static uptr ModuloNumberOfFrames(uptr stack_size_log, uptr class_id, uptr n) {
    return n & (NumberOfFrames(stack_size_log, class_id) - 1);
  }

  // The pointer to the flags of the given class_id.
  u8 *GetFlags(uptr stack_size_log, uptr class_id) {
    return reinterpret_cast<u8 *>(this) + kFlagsOffset +
           FlagsOffset(stack_size_log, class_id);
  }

  // Get frame by class_id and pos.
  // Return values are guaranteed to be aligned to BytesInSizeClass(class_id),
  // which is useful in combination with
  // ASanStackFrameLayout::ComputeASanStackFrameLayout().
  //
  // Note that alignment to 1<<kMaxStackFrameSizeLog (aka
  // BytesInSizeClass(max_class_id)) implies alignment to BytesInSizeClass()
  // for any class_id, since the class sizes are increasing powers of 2.
  //
  // 1) (this + kFlagsOffset + SizeRequiredForFlags())) is aligned to
  //    1<<kMaxStackFrameSizeLog (see FakeStack::Create)
  //
  //    Note that SizeRequiredForFlags(16) == 2048. If FakeStack::Create() had
  //    merely returned an address from mmap (4K-aligned), the addition would
  //    not be 4K-aligned.
  // 2) We know that stack_size_log >= kMaxStackFrameSizeLog (otherwise you
  //    couldn't store a single frame of that size in the entire stack)
  //    hence (1<<stack_size_log) is aligned to 1<<kMaxStackFrameSizeLog
  //    and   ((1<<stack_size_log) * class_id) is aligned to
  //          1<<kMaxStackFrameSizeLog
  // 3) BytesInSizeClass(class_id) * pos is aligned to
  //    BytesInSizeClass(class_id)
  // The sum of these is aligned to BytesInSizeClass(class_id).
  u8 *GetFrame(uptr stack_size_log, uptr class_id, uptr pos) {
    return reinterpret_cast<u8 *>(this) + kFlagsOffset +
           SizeRequiredForFlags(stack_size_log) +
           (((uptr)1) << stack_size_log) * class_id +
           BytesInSizeClass(class_id) * pos;
  }

  // Allocate the fake frame.
  FakeFrame *Allocate(uptr stack_size_log, uptr class_id, uptr real_stack);

  // Deallocate the fake frame: read the saved flag address and write 0 there.
  static void Deallocate(uptr x, uptr class_id) {
    **SavedFlagPtr(x, class_id) = 0;
  }

  // Poison the entire FakeStack's shadow with the magic value.
  void PoisonAll(u8 magic);

  // Return the beginning of the FakeFrame or 0 if the address is not ours.
  uptr AddrIsInFakeStack(uptr addr, uptr *frame_beg, uptr *frame_end);
  USED uptr AddrIsInFakeStack(uptr addr) {
    uptr t1, t2;
    return AddrIsInFakeStack(addr, &t1, &t2);
  }

  // Number of bytes in a fake frame of this size class.
  static uptr BytesInSizeClass(uptr class_id) {
    return ((uptr)1) << (class_id + kMinStackFrameSizeLog);
  }

  // The fake frame is guaranteed to have a right redzone.
  // We use the last word of that redzone to store the address of the flag
  // that corresponds to the current frame to make faster deallocation.
  static u8 **SavedFlagPtr(uptr x, uptr class_id) {
    return reinterpret_cast<u8 **>(x + BytesInSizeClass(class_id) - sizeof(x));
  }

  uptr stack_size_log() const { return stack_size_log_; }

  void HandleNoReturn();
  void GC(uptr real_stack);

  void ForEachFakeFrame(RangeIteratorCallback callback, void *arg);

 private:
  FakeStack() { }
  static const uptr kFlagsOffset = 4096;  // This is where the flags begin.
  // Must match the number of uses of DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID
  COMPILER_CHECK(kNumberOfSizeClasses == 11);
  static const uptr kMaxStackMallocSize = ((uptr)1) << kMaxStackFrameSizeLog;

  uptr hint_position_[kNumberOfSizeClasses];
  uptr stack_size_log_;
  bool needs_gc_;
  // We allocated more memory than needed to ensure the FakeStack (and, by
  // extension, each of the fake stack frames) is aligned. We keep track of the
  // true start so that we can unmap it.
  void *true_start;
};

void SetTLSFakeStack(FakeStack* fs);

}  // namespace __asan

#endif  // ASAN_FAKE_STACK_H
