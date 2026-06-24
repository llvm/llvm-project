// RUN: %clangxx -O0 -w -fsanitize=alignment,bool,signed-integer-overflow \
// RUN:   -fno-sanitize-memory-param-retval %s -o %t
// RUN: %env_ubsan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s
//
// The handler hooks are not defined as weak. Redefining them here isn't
// supported on Windows.
//
// UNSUPPORTED: target={{.*windows.*}}
// Linkage issue
// XFAIL: target={{.*openbsd.*}}

#include <climits>
#include <cstdio>
#include <cstdlib>

static int BeginCount{0};
static int EndCount{0};
static bool IsHookActive{false};

#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void __ubsan_handle_error_begin(void) {
  if (IsHookActive) {
    printf("%s ERROR: Unexpected IsHookActive\n", __FUNCTION__);
    exit(1);
  }
  IsHookActive = true;
  ++BeginCount;
}

// Required for dyld macOS 12.0+.
#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void __ubsan_handle_error_end(void) {
  if (!IsHookActive) {
    printf("%s ERROR: Unexpected IsHookActive\n", __FUNCTION__);
    exit(1);
  }

  if (BeginCount != EndCount + 1) {
    printf("%s ERROR: Unexpected count\n", __FUNCTION__);
    exit(1);
  }
  IsHookActive = false;
  ++EndCount;
}

__attribute__((noinline)) int TriggerSignedOverflow() {
  volatile int Max = INT_MAX;
  return Max + 1;
}

__attribute__((noinline)) bool TriggerInvalidBoolLoad() {
  char C = 3;
  return *reinterpret_cast<bool *>(&C);
}

__attribute__((noinline)) int TriggerMisalignedLoad() {
  alignas(int) char Buffer[2 * sizeof(int)] = {};
  int *Misaligned = reinterpret_cast<int *>(Buffer + 1);
  return *Misaligned;
}

int main() {
  volatile int Sink = 0;
  Sink += TriggerSignedOverflow();
  Sink += TriggerInvalidBoolLoad();
  Sink += TriggerMisalignedLoad();

  if (BeginCount != 3) {
    printf("%s ERROR: Unexpected count\n", __FUNCTION__);
    exit(1);
  }

  if (EndCount != 3) {
    printf("%s ERROR: Unexpected count\n", __FUNCTION__);
    exit(1);
  }

  if (IsHookActive) {
    printf("%s ERROR: Unexpected IsHookActive\n", __FUNCTION__);
    exit(1);
  }

  printf("HOOK TEST PASSED: begin=%d end=%d sink=%d\n", BeginCount, EndCount,
         Sink);
  return 0;
}

// CHECK-NOT: ERROR
// CHECK: HOOK TEST PASSED: begin=3
