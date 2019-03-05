//===-- asan_stack.cc -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Code for ASan stack trace.
//===----------------------------------------------------------------------===//
#include "asan_internal.h"
#include "asan_stack.h"
#include "sanitizer_common/sanitizer_atomic.h"

namespace __asan {

static atomic_uint32_t malloc_context_size;

void SetMallocContextSize(u32 size) {
  atomic_store(&malloc_context_size, size, memory_order_release);
}

u32 GetMallocContextSize() {
  return atomic_load(&malloc_context_size, memory_order_acquire);
}

namespace {

// ScopedUnwinding is a scope for stacktracing member of a context
class ScopedUnwinding {
 public:
  explicit ScopedUnwinding(AsanThread *t) : thread(t) { t->setUnwinding(true); }
  ~ScopedUnwinding() { thread->setUnwinding(false); }

 private:
  AsanThread *thread;
};

}  // namespace

}  // namespace __asan

void __sanitizer::BufferedStackTrace::UnwindImpl(
    uptr pc, uptr bp, void *context, bool request_fast, u32 max_depth) {
  using namespace __asan;
  size = 0;
  if (UNLIKELY(!asan_inited))
    return;
#if SANITIZER_WINDOWS
  Unwind(max_depth, pc, bp, context, 0, 0, false);
#else
  AsanThread *t = GetCurrentThread();
  if (!t) {
    if (!request_fast) {
      /* If GetCurrentThread() has failed, try to do slow unwind anyways. */
      Unwind(max_depth, pc, bp, context, 0, 0, false);
    }
    return;
  }
  if (t->isUnwinding())
    return;
  uptr stack_top = t->stack_top();
  uptr stack_bottom = t->stack_bottom();
  ScopedUnwinding unwind_scope(t);
  if (SANITIZER_MIPS && !IsValidFrame(bp, stack_top, stack_bottom))
    return;
  if (StackTrace::WillUseFastUnwind(request_fast))
    Unwind(max_depth, pc, bp, nullptr, stack_top, stack_bottom, true);
  else
    Unwind(max_depth, pc, bp, context, 0, 0, false);
#endif // SANITIZER_WINDOWS
}

// ------------------ Interface -------------- {{{1

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  using namespace __asan;
  PRINT_CURRENT_STACK();
}
}  // extern "C"
