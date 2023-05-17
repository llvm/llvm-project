// RUN: not %clang_cc1 -triple loongarch32 -O2 -emit-llvm %s 2>&1 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple loongarch64 -O2 -emit-llvm %s 2>&1 -o - | FileCheck %s

void test_l(void) {
// CHECK: :[[#@LINE+1]]:27: error: value '32768' out of range for constraint 'l'
  asm volatile ("" :: "l"(32768));
// CHECK: :[[#@LINE+1]]:27: error: value '-32769' out of range for constraint 'l'
  asm volatile ("" :: "l"(-32769));
}

void test_I(void) {
// CHECK: :[[#@LINE+1]]:27: error: value '2048' out of range for constraint 'I'
  asm volatile ("" :: "I"(2048));
// CHECK: :[[#@LINE+1]]:27: error: value '-2049' out of range for constraint 'I'
  asm volatile ("" :: "I"(-2049));
}

void test_K(void) {
// CHECK: :[[#@LINE+1]]:27: error: value '4096' out of range for constraint 'K'
  asm volatile ("" :: "K"(4096));
// CHECK: :[[#@LINE+1]]:27: error: value '-1' out of range for constraint 'K'
  asm volatile ("" :: "K"(-1));
}
