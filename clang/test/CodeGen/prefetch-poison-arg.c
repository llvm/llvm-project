// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm %s -o - | FileCheck %s

void test_poison_rw(void) {
  __builtin_prefetch(0, 2 >> 32, 2 >> 32);
  // CHECK: call void @llvm.prefetch.p0(ptr null, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, i32 1)
}
