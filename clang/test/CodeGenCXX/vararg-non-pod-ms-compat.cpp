// RUN: %clang_cc1 -Wno-error=non-pod-varargs -triple i686-pc-win32 -fms-compatibility -emit-llvm -o - %s | FileCheck %s -check-prefix=X86 -check-prefix=CHECK
// RUN: %clang_cc1 -Wno-error=non-pod-varargs -triple x86_64-pc-win32 -fms-compatibility -emit-llvm -o - %s | FileCheck %s -check-prefix=X64 -check-prefix=CHECK

struct X {
  X();
  ~X();
  int data;
};

void vararg(...);

void test(X x) {
  // CHECK-LABEL: define void @"\01?test@@YAXUX@@@Z"

  // X86: %[[argmem:[^ ]*]] = alloca inalloca <{ %struct.X }>
  // X86: call void (<{ %struct.X }>*, ...) bitcast (void (...)* @"\01?vararg@@YAXZZ" to void (<{ %struct.X }>*, ...)*)(<{ %struct.X }>* inalloca %[[argmem]])

  // X64: alloca %struct.X

  // X64: %[[agg:[^ ]*]] = alloca %struct.X
  // X64: %[[valptr:[^ ]*]] = getelementptr inbounds %struct.X, %struct.X* %[[agg]], i32 0, i32 0
  // X64: %[[val:[^ ]*]] = load i32, i32* %[[valptr]]
  // X64: call void (...) @"\01?vararg@@YAXZZ"(i32 %[[val]])

  // CHECK-NOT: llvm.trap
  vararg(x);
  // CHECK: ret void
}
