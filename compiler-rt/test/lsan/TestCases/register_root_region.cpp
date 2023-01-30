// Test for __lsan_(un)register_root_region().
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 %run %t
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0 not %run %t foo 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=use_stacks=0:use_registers=0:use_root_regions=0 not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <sanitizer/lsan_interface.h>

int main(int argc, char *argv[]) {
  size_t size = getpagesize() * 2;
  void *p =
      mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  assert(p);
  // Make half of the memory inaccessible. LSan must not crash trying to read it.
  assert(0 == mprotect((char *)p + size / 2, size / 2, PROT_NONE));

  __lsan_register_root_region(p, size);
  *((void **)p) = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", p);
  if (argc > 1)
    __lsan_unregister_root_region(p, size);
  return 0;
}
// CHECK: Test alloc: [[ADDR:.*]].
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 1337 byte(s) leaked in 1 allocation(s)
