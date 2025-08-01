// RUN: %clang_cc1 %s       -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s
// RUN: %clang_cc1 %s       -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s
// RUN: %clang_cc1 -xc++ %s -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s --check-prefixes=CHECK,CXX
// RUN: %clang_cc1 -xc++ %s -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s --check-prefixes=CHECK,CXX

#ifdef __cplusplus
extern "C" {
#endif

void f(void);

#ifdef __cplusplus

// CXX: define {{(dso_local )?}}internal void @__cxx_global_var_init()
// CXX: store ptr getelementptr inbounds (i32, ptr ptrauth (ptr @f, i32 0), i64 2), ptr @_ZL2fp, align 8

// This is rejected in C mode as adding a non-zero constant to a signed pointer
// is unrepresentable in relocations. In C++ mode, this can be done dynamically
// by the global constructor.
__attribute__((used))
void (*const fp)(void) = (void (*)(void))((int *)&f + 2);

#endif

// CHECK: define {{(dso_local )?}}void @t1()
void t1() {
  // CHECK: [[PF:%.*]] = alloca ptr
  // CHECK: store ptr getelementptr inbounds (i32, ptr ptrauth (ptr @f, i32 0), i64 2), ptr [[PF]]

  void (*pf)(void) = (void (*)(void))((int *)&f + 2);
  (void)pf;
}

#ifdef __cplusplus
}
#endif
