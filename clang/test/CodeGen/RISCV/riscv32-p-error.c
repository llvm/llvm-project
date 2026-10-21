// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-p -S -verify %s -o -

unsigned long long rev16_64(unsigned long long a) {
  return __builtin_riscv_rev16_64(a); // expected-error {{'__builtin_riscv_rev16_64' needs target feature experimental-p,64bit}}
}
