// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_right_align -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// Right-align mode must still satisfy the default malloc alignment guarantee.

#include <stddef.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

int main() {
  constexpr size_t kAlign = alignof(max_align_t);

  void *p17 = malloc(17);
  void *p48 = malloc(48);
  if (!p17 || !p48) return 1;

  if ((reinterpret_cast<std::uintptr_t>(p17) % kAlign) != 0) return 2;
  if ((reinterpret_cast<std::uintptr_t>(p48) % kAlign) != 0) return 3;

  // CHECK: alignment: ok
  printf("alignment: ok\n");

  free(p48);
  free(p17);
  return 0;
}
