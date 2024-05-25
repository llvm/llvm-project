// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

extern void foo() __attribute__((weak_import));

// CHECK-LABEL: define void @bar()
// CHECK: br i1 icmp ne (ptr ptrauth (ptr @foo, i32 0), ptr null), label
void bar() {
  if (foo)
    foo();
}
