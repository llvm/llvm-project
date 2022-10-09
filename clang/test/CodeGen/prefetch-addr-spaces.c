// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm %s -o - | FileCheck %s

void f(int __attribute__((address_space(1))) * a, ...) {
  __builtin_prefetch(a, 0, 1);
  // CHECK: call void @llvm.prefetch.p1(ptr addrspace(1) {{%.+}}, i32 0, i32 1, i32 1)
}
