// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-linux-pc -target-feature +avx512fp16 %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple spir-unknown-unknown %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple armv7a--none-eabi %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple aarch64-linux-gnu %s | FileCheck %s

void test_float16_builtins(void) {
  volatile _Float16 res;

  // CHECK: store volatile half 0xH7C00, ptr %res, align 2
  res = __builtin_huge_valf16();
  // CHECK: store volatile half 0xH7C00, ptr %res, align 2
  res = __builtin_inff16();
  // CHECK: store volatile half 0xH7E00, ptr %res, align 2
  res = __builtin_nanf16("");
  // CHECK: store volatile half 0xH7D00, ptr %res, align 2
  res = __builtin_nansf16("");
}
