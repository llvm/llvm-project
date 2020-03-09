// RUN: %clang_cc1 %s       -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s
// RUN: %clang_cc1 -xc++ %s -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s --check-prefixes=CHECK,CXX

#ifdef __cplusplus
extern "C" {
#endif

void f(void);

// CHECK: @f.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @f to i8*), i32 0, i64 0, i64 0 }, section "llvm.ptrauth"

#ifdef __cplusplus

// CXX-LABEL: define internal void @__cxx_global_var_init()
// CXX: store void ()* bitcast (i32* getelementptr inbounds (i32, i32* bitcast ({ i8*, i32, i64, i64 }* @f.ptrauth to i32*), i64 2) to void ()*), void ()** @_ZL2fp, align 8

__attribute__((used))
void (*const fp)(void) = (void (*)(void))((int *)&f + 2); // Error in C mode.

#endif

// CHECK-LABEL: define void @t1()
void t1() {
  // CHECK: [[PF:%.*]] = alloca void ()*
  // CHECK: store void ()* bitcast (i32* getelementptr inbounds (i32, i32* bitcast ({ i8*, i32, i64, i64 }* @f.ptrauth to i32*), i64 2) to void ()*), void ()** [[PF]]

  void (*pf)(void) = (void (*)(void))((int *)&f + 2);
  (void)pf;
}

#ifdef __cplusplus
}
#endif
