// RUN: %clang_cc1 -triple powerpc-ibm-aix -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @padding_preserves_37(
// CHECK: ret i32 1
int padding_preserves_37(void) {
  _BitInt(97) value;
  __builtin_memset(&value, 42, sizeof(value));
  value = 37;
  __builtin_clear_padding(&value);
  return value == 37;
}
