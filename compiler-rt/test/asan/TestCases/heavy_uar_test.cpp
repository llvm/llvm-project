// Most Windows linkers set stack size default to 1 MB. Bump it up to 8 MB.
// DEFINE: %{stack} = %if target={{.*-windows-gnu}} %{ -Wl,--stack,8388608 %} \
// DEFINE:            %if target={{.*-windows-msvc.*}} %{ -Wl,/STACK:8388608 %}

// RUN: %clangxx_asan %{stack} -O0 %s -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan %{stack} -O2 %s -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan %{stack} -O0 %s -o %t -fsanitize-address-use-after-return=always && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan %{stack} -O2 %s -o %t -fsanitize-address-use-after-return=always && not %run %t 2>&1 | FileCheck %s

// FIXME: Fix this test under GCC.
// REQUIRES: Clang

// FIXME: Fix this test for dynamic runtime on arm linux.
// UNSUPPORTED: (arm-linux || armhf-linux) && asan-dynamic-runtime

// UNSUPPORTED: ios

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#  include <windows.h>
#endif

__attribute__((noinline))
char *pretend_to_do_something(char *x) {
  __asm__ __volatile__("" : : "r" (x) : "memory");
  return x;
}

__attribute__((noinline))
char *LeakStack() {
  char x[1024];
  memset(x, 0, sizeof(x));
  return pretend_to_do_something(x);
}

template<size_t kFrameSize>
__attribute__((noinline))
void RecursiveFunctionWithStackFrame(int depth) {
  if (depth <= 0) return;
  char x[kFrameSize];
  x[0] = depth;
  pretend_to_do_something(x);
  RecursiveFunctionWithStackFrame<kFrameSize>(depth - 1);
}

int main(int argc, char **argv) {
#ifdef _WIN32
  // Prevent the crash dialog from showing in case something bad like a stack
  // overflow happens.
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif

  int n_iter = argc >= 2 ? atoi(argv[1]) : 1000;
  int depth  = argc >= 3 ? atoi(argv[2]) : 500;
  for (int i = 0; i < n_iter; i++) {
    RecursiveFunctionWithStackFrame<10>(depth);
    RecursiveFunctionWithStackFrame<100>(depth);
    RecursiveFunctionWithStackFrame<500>(depth);
    RecursiveFunctionWithStackFrame<1024>(depth);
    RecursiveFunctionWithStackFrame<2000>(depth);
    // The stack size is tight for the main thread in multithread
    // environment on FreeBSD and NetBSD.
#if !defined(__FreeBSD__) && !defined(__NetBSD__)
    RecursiveFunctionWithStackFrame<5000>(depth);
    RecursiveFunctionWithStackFrame<10000>(depth);
#endif
  }
  char *stale_stack = LeakStack();
  RecursiveFunctionWithStackFrame<1024>(10);
  stale_stack[100]++;
  // CHECK: ERROR: AddressSanitizer: stack-use-after-return on address
  // CHECK: is located in stack of thread T0 at offset {{116|132}} in frame
  // CHECK:  in LeakStack{{.*}}heavy_uar_test.cpp:
  // CHECK: [{{16|32}}, {{1040|1056}}) 'x'
  return 0;
}
