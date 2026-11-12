// Fake frames abandoned by longjmp must be collected: their memory becomes
// stack-after-return and they can be allocated again. Frames of functions that
// are still active must survive. Checked for every size class, with enough
// frames per class to cover all positions within a group of 8 flags.

// No optimization: the test relies on every call getting its own stack frame.
// The size of the fake stack is fixed, so that it does not follow the stack
// limit: the 64 KiB class has to hold 13 frames.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=1:min_uar_stack_size_log=20:max_uar_stack_size_log=20 %run %t

#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <setjmp.h>

static jmp_buf env;
static const int kMaxDepth = 12;
static char *abandoned[kMaxDepth];

__attribute__((noinline)) static void Escape(char *p) {
  __asm__ __volatile__("" : : "r"(p) : "memory");
}

static bool InFakeStack(char *p) {
  void *fake_stack = __asan_get_current_fake_stack();
  return fake_stack &&
         __asan_addr_is_in_fake_stack(fake_stack, p, nullptr, nullptr);
}

template <int kSize> __attribute__((noinline)) static void Recurse(int depth) {
  char local[kSize];
  Escape(local);
  assert(InFakeStack(local));
  abandoned[depth] = local;
  if (depth == 0)
    longjmp(env, 1);
  Recurse<kSize>(depth - 1);
}

// Pushes the abandoned frames well below the real stack position from which
// AllocateOneFrame runs the collection; only lower frames are collected.
template <int kSize>
__attribute__((noinline, no_sanitize("address"))) static void Enter(int depth) {
  volatile char padding[512];
  padding[0] = 0;
  Recurse<kSize>(depth);
}

__attribute__((noinline)) static void AllocateOneFrame() {
  char local[8];
  Escape(local);
}

// Functions that call setjmp never get a fake frame, so this one is separate
// from Test, whose frame has to survive the collections.
template <int kSize>
__attribute__((noinline)) static void JumpAndCheck(char *live) {
  for (int depth = 0; depth < kMaxDepth; depth++) {
    if (setjmp(env) == 0)
      Enter<kSize>(depth);
    // The next fake frame allocation collects the abandoned frames.
    AllocateOneFrame();
    for (int i = 0; i <= depth; i++) {
      assert(__asan_address_is_poisoned(abandoned[i]));
      assert(__asan_address_is_poisoned(abandoned[i] + kSize - 1));
    }
    assert(!__asan_region_is_poisoned(live, kSize));
  }
}

template <int kSize> __attribute__((noinline)) static void Test() {
  char live[kSize];
  Escape(live);
  assert(InFakeStack(live));
  JumpAndCheck<kSize>(live);
}

int main() {
  Test<8>();     // 64-byte class
  Test<64>();    // 128
  Test<150>();   // 256
  Test<300>();   // 512
  Test<700>();   // 1K
  Test<1500>();  // 2K
  Test<3000>();  // 4K
  Test<6000>();  // 8K
  Test<12000>(); // 16K
  Test<24000>(); // 32K
  Test<48000>(); // 64K
  return 0;
}
