// Test fixed shadow base functionality.
//
// Default compiler instrumentation works with any shadow base (dynamic or fixed).
// RUN: %clang_hwasan %s -o %t
// RUN: %run %t
// RUN: HWASAN_OPTIONS=fixed_shadow_base=263878495698944 %run %t 2>%t.out || (cat %t.out | FileCheck %s)
// RUN: HWASAN_OPTIONS=fixed_shadow_base=4398046511104 %run %t
//
// If -hwasan-mapping-offset is set, then the fixed_shadow_base needs to match.
// RUN: %clang_hwasan %s -mllvm -hwasan-mapping-offset=263878495698944 -o %t
// RUN: HWASAN_OPTIONS=fixed_shadow_base=263878495698944 %run %t 2>%t.out || (cat %t.out | FileCheck %s)
// RUN: HWASAN_OPTIONS=fixed_shadow_base=4398046511104 not %run %t

// RUN: %clang_hwasan %s -mllvm -hwasan-mapping-offset=4398046511104 -o %t
// RUN: HWASAN_OPTIONS=fixed_shadow_base=4398046511104 %run %t
// RUN: HWASAN_OPTIONS=fixed_shadow_base=263878495698944 not %run %t
//
// Note: if fixed_shadow_base is not set, compiler-rt will dynamically choose a
// shadow base, which has a tiny but non-zero probability of matching the
// compiler instrumentation. To avoid test flake, we do not test this case.
//
// Assume 48-bit VMA
// REQUIRES: aarch64-target-arch
//
// REQUIRES: Clang
//
// UNSUPPORTED: android

// CHECK: FATAL: HWAddressSanitizer: Shadow range {{.*}} is not available

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

int main() {
  __hwasan_enable_allocator_tagging();

  // We test that the compiler instrumentation is able to access shadow memory
  // for many different addresses. If we only test a small number of addresses,
  // it might work by chance even if the shadow base does not match between the
  // compiler instrumentation and compiler-rt.
  void **mmaps[256];
  // 48-bit VMA
  for (int i = 0; i < 256; i++) {
    unsigned long long addr = (i * (1ULL << 40));

    void *p = mmap((void *)addr, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    // We don't use MAP_FIXED, to avoid overwriting critical memory.
    // However, if we don't get allocated the requested address, it
    // isn't a useful test.
    if ((unsigned long long)p != addr) {
      munmap(p, 4096);
      mmaps[i] = MAP_FAILED;
    } else {
      mmaps[i] = p;
    }
  }

  int failures = 0;
  for (int i = 0; i < 256; i++) {
    if (mmaps[i] == MAP_FAILED) {
      failures++;
    } else {
      printf("%d %p\n", i, mmaps[i]);
      munmap(mmaps[i], 4096);
    }
  }

  // We expect roughly 17 failures:
  // - the page at address zero
  // - 16 failures because the shadow memory takes up 1/16th of the address space
  // We could also get unlucky e.g., if libraries or binaries are loaded into the
  // exact addresses where we tried to map.
  // To avoid test flake, we allow some margin of error.
  printf("Failed: %d\n", failures);
  assert(failures < 48);
  return 0;
}
