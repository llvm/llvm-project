/// From msan/allocator_mapping.cpp
/// Test that a module constructor can not map memory over the NSan heap
/// (without MAP_FIXED, of course).
// RUN: %clangxx_nsan -O0 %s -o %t_1
// RUN: %clangxx_nsan -O0 -DHEAP_ADDRESS=$(%run %t_1) %s -o %t_2 && %run %t_2

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

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
