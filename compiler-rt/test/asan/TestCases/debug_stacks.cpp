// Check that the stack trace debugging API works and returns correct
// malloc and free stacks.
// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// FIXME: Figure out why allocation/free stack traces may be too short on ARM.
// REQUIRES: stable-runtime

#if _WIN64
#define PTR "%llx"
#else
#define PTR "%lx"
#endif

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

char *mem;
void func1() {
  mem = (char *)malloc(10);
}

void func2() {
  free(mem);
}

int main() {
  // Disable stderr buffering. Needed on Windows.
  setvbuf(stderr, NULL, _IONBF, 0);

  func1();
  func2();

  void *trace[100];
  size_t num_frames = 100;
  int thread_id;
  num_frames = __asan_get_alloc_stack(mem, trace, num_frames, &thread_id);

  fprintf(stderr, "alloc stack retval %s\n", (num_frames > 0 && num_frames < 10)
          ? "ok" : "");
  // CHECK: alloc stack retval ok
  fprintf(stderr, "thread id = %d\n", thread_id);
  // CHECK: thread id = 0
  fprintf(stderr, "0x" PTR "\n", trace[0]);
  // CHECK: 0x{{0*}}[[ALLOC_FRAME_0:[0-9a-f]+]]
  fprintf(stderr, "0x" PTR "\n", trace[1]);
  // CHECK: 0x{{0*}}[[ALLOC_FRAME_1:[0-9a-f]+]]

  num_frames = 100;
  num_frames = __asan_get_free_stack(mem, trace, num_frames, &thread_id);

  fprintf(stderr, "free stack retval %s\n", (num_frames > 0 && num_frames < 10)
          ? "ok" : "");
  // CHECK: free stack retval ok
  fprintf(stderr, "thread id = %d\n", thread_id);
  // CHECK: thread id = 0
  fprintf(stderr, "0x" PTR "\n", trace[0]);
  // CHECK: 0x{{0*}}[[FREE_FRAME_0:[0-9a-f]+]]
  fprintf(stderr, "0x" PTR "\n", trace[1]);
  // CHECK: 0x{{0*}}[[FREE_FRAME_1:[0-9a-f]+]]

  mem[0] = 'A'; // BOOM

  // CHECK: ERROR: AddressSanitizer: heap-use-after-free
  // CHECK: WRITE of size 1 at 0x{{.*}}
  // CHECK: freed by thread T0 here:
  // CHECK: #0 0x{{0*}}[[FREE_FRAME_0]]
  // CHECK: #1 0x{{0*}}[[FREE_FRAME_1]]
  // CHECK: previously allocated by thread T0 here:
  // CHECK: #0 0x{{0*}}[[ALLOC_FRAME_0]]
  // CHECK: #1 0x{{0*}}[[ALLOC_FRAME_1]]

  return 0;
}
