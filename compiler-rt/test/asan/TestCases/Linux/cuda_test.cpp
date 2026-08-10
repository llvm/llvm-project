// Emulate the behavior of the NVIDIA CUDA driver
// that mmaps memory inside the asan's shadow gap.
//
// REQUIRES: x86_64-target-arch, shadow-scale-3
//
// RUN: %clangxx_asan %s -o %t
// RUN: not %env_asan_opts=protect_shadow_gap=1 %t 2>&1 | FileCheck %s  --check-prefix=CHECK-PROTECT1
// RUN: not                                     %t 2>&1 | FileCheck %s  --check-prefix=CHECK-PROTECT1
// RUN: not %env_asan_opts=protect_shadow_gap=0 %t 2>&1 | FileCheck %s  --check-prefix=CHECK-PROTECT0
#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>

#include "sanitizer/asan_interface.h"

int main(void) {
  uintptr_t Base = 0x200000000;
  uintptr_t Size = 0x1100000000;
  void *addr =
      mmap((void *)Base, Size, PROT_READ | PROT_WRITE,
           MAP_NORESERVE | MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, 0, 0);
  assert(addr == (void*)Base);
  // Make sure we can access memory in shadow gap.
  // W/o protect_shadow_gap=0 we should fail here.
  for (uintptr_t Addr = Base; Addr < Base + Size; Addr += Size / 100)
    *(char*)Addr = 1;
  // CHECK-PROTECT1: AddressSanitizer: SEGV on unknown address 0x0000bfff8000

  // Poison a part of gap's shadow:
  __asan_poison_memory_region((void*)Base, 4096);
  // Now we should fail with use-after-poison.
  *(char*)(Base + 1234) = 1;
  // CHECK-PROTECT0: AddressSanitizer: use-after-poison on address 0x0002000004d2
}
