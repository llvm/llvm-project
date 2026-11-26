// Check the shadow of fake stack frames for every size class, at every frame
// size in classes 0 to 6 and at a stride in the larger ones: addressable up to
// the frame size, right redzone up to the end of the size class, and, once the
// frame is freed, stack-after-return over the frame, and in classes 0 to 6 over
// the whole size class.

// RUN: %clangxx_asan -O1 %s -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=1 %run %t
// With poison_heap=0 the runtime poisons nothing but freed frames of classes 0
// to 6.
// RUN: %env_asan_opts=detect_stack_use_after_return=1:poison_heap=0 %run %t no-poison
// RUN: %env_asan_opts=detect_stack_use_after_return=1 not %run %t oversize 2>&1 | FileCheck %s

// REQUIRES: shadow-scale-3

#include "defines.h"
#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DECLARE(N)                                                             \
  extern "C" uintptr_t __asan_stack_malloc_##N(uintptr_t size);                \
  extern "C" void __asan_stack_free_##N(uintptr_t ptr, uintptr_t size);
DECLARE(0)
DECLARE(1)
DECLARE(2)
DECLARE(3)
DECLARE(4)
DECLARE(5)
DECLARE(6)
DECLARE(7)
DECLARE(8)
DECLARE(9)
DECLARE(10)

typedef uintptr_t (*MallocFn)(uintptr_t);
typedef void (*FreeFn)(uintptr_t, uintptr_t);
#define FNS(N) {__asan_stack_malloc_##N, __asan_stack_free_##N}
static const struct {
  MallocFn malloc_fn;
  FreeFn free_fn;
} kClasses[] = {FNS(0), FNS(1), FNS(2), FNS(3), FNS(4), FNS(5),
                FNS(6), FNS(7), FNS(8), FNS(9), FNS(10)};

static const uint8_t kRightRedzone = 0xf3;
static const uint8_t kAfterReturn = 0xf5;

static size_t shadow_scale, shadow_offset;

static const uint8_t *Shadow(uintptr_t addr) {
  return reinterpret_cast<const uint8_t *>((addr >> shadow_scale) +
                                           shadow_offset);
}

// Shadow memory has no shadow of its own, so reads of it can't be instrumented.
ATTRIBUTE_NO_SANITIZE_ADDRESS static void
ExpectShadow(uintptr_t beg, uintptr_t end, uint8_t value) {
  for (uintptr_t p = beg; p < end; p += 8) {
    if (*Shadow(p) != value) {
      fprintf(stderr, "shadow of %p is 0x%02x, expected 0x%02x\n", (void *)p,
              *Shadow(p), value);
      assert(false);
    }
  }
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "oversize")) {
    __asan_stack_malloc_0(64 + 8);
    // CHECK: CHECK failed: {{.*}}((size)) < ((class_size))
    return 0;
  }
  __asan_get_shadow_mapping(&shadow_scale, &shadow_offset);
  const bool can_poison = argc == 1;
  const uint8_t right_redzone = can_poison ? kRightRedzone : 0;
  for (int class_id = 0; class_id <= 10; class_id++) {
    // Classes 0 to 6 write the shadow directly: a freed frame gets the magic
    // over the whole size class, with poisoning on or off. Larger classes go
    // through PoisonShadow: only size bytes, and only with poisoning on.
    const bool direct = class_id <= 6;
    const uint8_t after_return = direct || can_poison ? kAfterReturn : 0;
    uintptr_t class_size = (uintptr_t)64 << class_id;
    // Frame sizes are multiples of the shadow granularity. Classes 0 to 6 are
    // checked for every frame size, larger ones at a stride.
    uintptr_t step = direct ? 8 : class_size / 64 + 8;
    for (uintptr_t size = 8; size <= class_size; size += step) {
      uintptr_t frame = kClasses[class_id].malloc_fn(size);
      assert(frame && "no fake stack");
      assert(frame % class_size == 0);
      ExpectShadow(frame, frame + size, 0);
      ExpectShadow(frame + size, frame + class_size, right_redzone);
      kClasses[class_id].free_fn(frame, size);
      ExpectShadow(frame, frame + size, after_return);
      ExpectShadow(frame + size, frame + class_size,
                   direct ? kAfterReturn : right_redzone);
    }
    uintptr_t frame = kClasses[class_id].malloc_fn(class_size);
    assert(frame);
    ExpectShadow(frame, frame + class_size, 0);
    kClasses[class_id].free_fn(frame, class_size);
    ExpectShadow(frame, frame + class_size, after_return);
  }
  return 0;
}
