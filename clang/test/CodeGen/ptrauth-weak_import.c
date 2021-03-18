// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

extern void foo() __attribute__((weak_import));

// CHECK-LABEL: define void @bar()
// CHECK: br i1 icmp ne (void (...)* bitcast ({ i8*, i32, i64, i64 }* @foo.ptrauth to void (...)*), void (...)* null), label
void bar() {
  if (foo)
    foo();
}
