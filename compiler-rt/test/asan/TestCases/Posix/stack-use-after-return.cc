// RUN: %clangxx_asan  -O0 %s -pthread -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan  -O1 %s -pthread -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan  -O2 %s -pthread -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan  -O3 %s -pthread -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=detect_stack_use_after_return=0 %run %t
// Regression test for a CHECK failure with small stack size and large frame.
// RUN: %clangxx_asan  -O3 %s -pthread -o %t -DkSize=10000 -DUseThread -DkStackSize=131072 && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck --check-prefix=THREAD %s
//
// Test that we can find UAR in a thread other than main:
// RUN: %clangxx_asan  -DUseThread -O2 %s -pthread -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck --check-prefix=THREAD %s
//
// Test the max_uar_stack_size_log/min_uar_stack_size_log flag.
//
// RUN: %env_asan_opts=detect_stack_use_after_return=1:max_uar_stack_size_log=20:verbosity=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-20 %s
// RUN: %env_asan_opts=detect_stack_use_after_return=1:min_uar_stack_size_log=24:max_uar_stack_size_log=24:verbosity=1 not %run %t 2>&1 | FileCheck --check-prefix=CHECK-24 %s

// This test runs out of stack on AArch64.
// UNSUPPORTED: aarch64
// stack size log lower than expected
// XFAIL: freebsd

// FIXME: Fix this test for dynamic runtime on arm linux.
// UNSUPPORTED: (arm-linux || armhf-linux) && asan-dynamic-runtime

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef kSize
# define kSize 1
#endif

#ifndef UseThread
# define UseThread 0
#endif

#ifndef kStackSize
# define kStackSize 0
#endif

__attribute__((noinline))
char *Ident(char *x) {
  fprintf(stderr, "1: %p\n", x);
  return x;
}

__attribute__((noinline))
char *Func1() {
  char local[kSize];
  return Ident(local);
}

__attribute__((noinline))
void Func2(char *x) {
  fprintf(stderr, "2: %p\n", x);
  *x = 1;
  // CHECK: WRITE of size 1 {{.*}} thread T0
  // CHECK:     #0{{.*}}Func2{{.*}}stack-use-after-return.cc:[[@LINE-2]]
  // CHECK: is located in stack of thread T0 at offset
  // CHECK: 'local'{{.*}} <== Memory access at offset {{16|32}} is inside this variable
  // THREAD: WRITE of size 1 {{.*}} thread T{{[1-9]}}
  // THREAD:     #0{{.*}}Func2{{.*}}stack-use-after-return.cc:[[@LINE-6]]
  // THREAD: is located in stack of thread T{{[1-9]}} at offset
  // THREAD: 'local'{{.*}} <== Memory access at offset {{16|32}} is inside this variable
  // CHECK-20: T0: FakeStack created:{{.*}} stack_size_log: 20
  // CHECK-24: T0: FakeStack created:{{.*}} stack_size_log: 24
}

void *Thread(void *unused)  {
  Func2(Func1());
  return NULL;
}

int main(int argc, char **argv) {
#if UseThread
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  if (kStackSize > 0) {
    size_t desired_stack_size = kStackSize;
    if (desired_stack_size < PTHREAD_STACK_MIN) {
      desired_stack_size = PTHREAD_STACK_MIN;
    }

    int ret = pthread_attr_setstacksize(&attr, desired_stack_size);
    if (ret != 0) {
      fprintf(stderr, "pthread_attr_setstacksize returned %d\n", ret);
      abort();
    }
    
    size_t stacksize_check;
    ret = pthread_attr_getstacksize(&attr, &stacksize_check);
    if (ret != 0) {
      fprintf(stderr, "pthread_attr_getstacksize returned %d\n", ret);
      abort();
    }

    if (stacksize_check != desired_stack_size) {
      fprintf(stderr, "Unable to set stack size to %d, the stack size is %d.\n",
              (int)desired_stack_size, (int)stacksize_check);
      abort(); 
    }
  }
  pthread_t t;
  pthread_create(&t, &attr, Thread, 0);
  pthread_attr_destroy(&attr);
  pthread_join(t, 0);
#else
  Func2(Func1());
#endif
  return 0;
}
