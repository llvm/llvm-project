// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

extern void foo() __attribute__((weak_import));

// CHECK-LABEL: define void @bar()
// CHECK: icmp ne ptr @foo.ptrauth, null
// CHECK: br i1
void bar() {
  if (foo)
    foo();
}
