// REQUIRES: loongarch-registered-target
/// cc1 -enable-tlsdesc (due to -mtls-dialect=desc) enables TLSDESC.
// RUN: %clang_cc1 -triple loongarch64 -S -mrelocation-model pic -pic-level 1 -enable-tlsdesc %s -o - | FileCheck %s --check-prefix=DESC
// RUN: %clang_cc1 -triple loongarch64 -S -mrelocation-model pic -pic-level 1 %s -o - | FileCheck %s --check-prefix=NODESC

__thread int x;

// DESC:       %desc_pc_hi20
// DESC-NOT:   %gd_pc_hi20
// NODESC:     %gd_pc_hi20
// NODESC-NOT: %desc_pc_hi20
int use() {
  return x;
}
