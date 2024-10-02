// Test ASan detection of stack-overflow condition.

// RUN: %clangxx_asan -O0 %s -DSMALL_FRAME -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DSMALL_FRAME -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O0 %s -DSAVE_ALL_THE_REGISTERS -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DSAVE_ALL_THE_REGISTERS -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O0 %s -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_asan -O0 %s -DTHREAD -DSMALL_FRAME -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DTHREAD -DSMALL_FRAME -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O0 %s -DTHREAD -DSAVE_ALL_THE_REGISTERS -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DTHREAD -DSAVE_ALL_THE_REGISTERS -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O0 %s -DTHREAD -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DTHREAD -pthread -o %t && %env_asan_opts=use_sigaltstack=1 not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: stable-runtime

// Issue #109771
// XFAIL: target={{sparc.*-.*-linux.*}}

// UNSUPPORTED: ios

#include <assert.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sanitizer/asan_interface.h>

const int BS = 1024;
volatile char x;
volatile int y = 1;
volatile int z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13;

void recursive_func(uintptr_t parent_frame_address) {
#if defined(SMALL_FRAME)
  char *buf = 0;
#elif defined(SAVE_ALL_THE_REGISTERS)
  char *buf = 0;
  int t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;
  t0 = z0;
  t1 = z1;
  t2 = z2;
  t3 = z3;
  t4 = z4;
  t5 = z5;
  t6 = z6;
  t7 = z7;
  t8 = z8;
  t9 = z9;
  t10 = z10;
  t11 = z11;
  t12 = z12;
  t13 = z13;

  z0 = t0;
  z1 = t1;
  z2 = t2;
  z3 = t3;
  z4 = t4;
  z5 = t5;
  z6 = t6;
  z7 = t7;
  z8 = t8;
  z9 = t9;
  z10 = t10;
  z11 = t11;
  z12 = t12;
  z13 = t13;
#else
  char buf[BS];
  // Check that the stack grows in the righ direction, unless we use fake stack.
  assert(parent_frame_address > (uintptr_t)__builtin_frame_address(0));
  buf[rand() % BS] = 1;
  buf[rand() % BS] = 2;
  x = buf[rand() % BS];
#endif
  if (y)
    recursive_func((uintptr_t)__builtin_frame_address(0));
  x = 1; // prevent tail call optimization
  // CHECK: {{stack-overflow on address 0x.* \(pc 0x.* bp 0x.* sp 0x.* T.*\)}}
  // If stack overflow happens during function prologue, stack trace may be
  // corrupted. Unwind tables are not always 100% exact there.
  // For this reason, we don't do any further checks.
}

void *ThreadFn(void* unused) {
  recursive_func((uintptr_t)__builtin_frame_address(0));
  return 0;
}

void LimitStackAndReexec(int argc, char **argv) {
  struct rlimit rlim;
  int res = getrlimit(RLIMIT_STACK, &rlim);
  assert(res == 0);
  if (rlim.rlim_cur == RLIM_INFINITY) {
    rlim.rlim_cur = 256 * 1024;
    res = setrlimit(RLIMIT_STACK, &rlim);
    assert(res == 0);

    execv(argv[0], argv);
    assert(0 && "unreachable");
  }
}

int main(int argc, char **argv) {
  LimitStackAndReexec(argc, argv);
#ifdef THREAD
  pthread_t t;
  pthread_create(&t, 0, ThreadFn, 0);
  pthread_join(t, 0);
#else
  recursive_func((uintptr_t)__builtin_frame_address(0));
#endif
  return 0;
}
