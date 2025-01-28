// RUN: %clang_cc1 -Wno-error=non-pod-varargs -triple i686-pc-win32 -fms-compatibility -emit-llvm -o - %s | FileCheck %s -check-prefix=X86 -check-prefix=CHECK
// RUN: %clang_cc1 -Wno-error=non-pod-varargs -triple x86_64-pc-win32 -fms-compatibility -emit-llvm -o - %s | FileCheck %s -check-prefix=X64 -check-prefix=CHECK

struct X {
  X();
  ~X();
  int data;
};

void vararg(...);

void test(X x) {
  // CHECK-LABEL: define dso_local void @"?test@@YAXUX@@@Z"

  // X86: %[[argmem:[^ ]*]] = alloca inalloca <{ %struct.X }>
  // X86: call void (ptr, ...) @"?vararg@@YAXZZ"(ptr inalloca(<{ %struct.X }>) %[[argmem]])

  // X64: alloca %struct.X

  // X64: %[[agg:[^ ]*]] = alloca %struct.X
  // X64: %[[valptr:[^ ]*]] = getelementptr inbounds nuw %struct.X, ptr %[[agg]], i32 0, i32 0
  // X64: %[[val:[^ ]*]] = load i32, ptr %[[valptr]]
  // X64: call void (...) @"?vararg@@YAXZZ"(i32 %[[val]])

  // CHECK-NOT: llvm.trap
  vararg(x);
  // CHECK: ret void
}
