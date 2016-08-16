// Test that a module constructor can not map memory over the MSan heap
// (without MAP_FIXED, of course). Current implementation ensures this by
// mapping the heap early, in __msan_init.
//
// RUN: %clangxx_msan -O0 %s -o %t_1
// RUN: %clangxx_msan -O0 -DHEAP_ADDRESS=$(%run %t_1) %s -o %t_2 && %run %t_2
//
// This test only makes sense for the 64-bit allocator. The 32-bit allocator
// does not have a fixed mapping. Exclude platforms that use the 32-bit
// allocator.
// UNSUPPORTED: target-is-mips64,target-is-mips64el,aarch64

#include <assert.h>
#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>

#ifdef HEAP_ADDRESS
struct A {
  A() {
    void *const hint = reinterpret_cast<void *>(HEAP_ADDRESS);
    void *p = mmap(hint, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    // This address must be already mapped. Check that mmap() succeeds, but at a
    // different address.
    assert(p != reinterpret_cast<void *>(-1));
    assert(p != hint);
  }
} a;
#endif

int main() {
  void *p = malloc(10);
  printf("0x%zx\n", reinterpret_cast<size_t>(p) & (~0xfff));
  free(p);
}
