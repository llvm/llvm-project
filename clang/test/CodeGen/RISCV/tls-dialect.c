// REQUIRES: riscv-registered-target
/// cc1 -enable-tlsdesc (due to -mtls-dialect=desc) enables TLSDESC.
// RUN: %clang_cc1 -triple riscv64 -S -mrelocation-model pic -pic-level 1 -enable-tlsdesc %s -o - | FileCheck %s --check-prefix=DESC
// RUN: %clang_cc1 -triple riscv64 -S -mrelocation-model pic -pic-level 1 %s -o - | FileCheck %s --check-prefix=NODESC

__thread int x;

// DESC:       %tlsdesc_hi
// DESC-NOT:   %tls_gd_pcrel_hi
// NODESC:     %tls_gd_pcrel_hi
// NODESC-NOT: %tlsdesc_hi
int use() {
  return x;
}
