// REQUIRES: x86_64-target-arch
// RUN: %clang -Wno-constant-conversion -Wno-array-bounds -Wno-division-by-zero -Wno-shift-negative-value -Wno-shift-count-negative -Wno-int-to-pointer-cast -O0 -fsanitize=alignment,array-bounds,bool,float-cast-overflow,implicit-integer-sign-change,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation,integer-divide-by-zero,nonnull-attribute,null,nullability-arg,nullability-assign,nullability-return,pointer-overflow,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,unreachable,unsigned-integer-overflow,unsigned-shift-base,vla-bound %s -o %t1 && %run %t1 2>&1 | FileCheck %s

#include <stdint.h>
#include <stdio.h>

// In this test there is an expectation of assignment of _BitInt not producing any output.
uint32_t nullability_arg(_BitInt(37) *_Nonnull x)
    __attribute__((no_sanitize("address")))
    __attribute__((no_sanitize("memory"))) {
  _BitInt(37) y = *(_BitInt(37) *)&x;
  return (y > 0) ? y : 0;
}

// In this test there is an expectation of ubsan not triggeting on returning random address which is inside address space of the process.
_BitInt(37) nonnull_attribute(__attribute__((nonnull)) _BitInt(37) * x)
    __attribute__((no_sanitize("address")))
    __attribute__((no_sanitize("memory"))) {
  return *(_BitInt(37) *)&x;
}

// In this test there is an expectation of assignment of uint32_t from "invalid" _BitInt is not producing any output.
uint32_t nullability_assign(_BitInt(7) * x)
    __attribute__((no_sanitize("address")))
    __attribute__((no_sanitize("memory"))) {
  _BitInt(7) *_Nonnull y = x;
  int32_t r = *(_BitInt(7) *)&y;
  return (r > 0) ? r : 0;
}

// In those examples the file is expected to compile & run with no diagnostics
// CHECK-NOT: runtime error:

int main(int argc, char **argv) {
  // clang-format off
  uint64_t result =
      1ULL +
      nullability_arg((_BitInt(37) *)argc) +
      ((uint64_t)nonnull_attribute((_BitInt(37) *)argc) & 0xFFFFFFFF) +
      nullability_assign((_BitInt(7) *)argc);
  // clang-format on
  printf("%u\n", (uint32_t)(result & 0xFFFFFFFF));
}
