// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

extern void foo() __attribute__((weak_import));

// CHECK: define {{(dso_local )?}}void @bar()
// CHECK: [[TMP1:%.*]] =  icmp ne ptr ptrauth (ptr @foo, i32 0), null
// CHECK: br i1 [[TMP1]], label
void bar() {
  if (foo)
    foo();
}
