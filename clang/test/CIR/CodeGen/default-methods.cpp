// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

// We should emit and call both implicit operator= functions.
struct S {
  struct T {
    int x;
  } t;
};

// CIR-LABEL: cir.func linkonce_odr @_ZN1S1TaSERKS0_({{.*}} {
// CIR-LABEL: cir.func linkonce_odr @_ZN1SaSERKS_(
// CIR:         cir.call @_ZN1S1TaSERKS0_(
// CIR-LABEL: cir.func @_Z1fR1SS0_(
// CIR:         cir.call @_ZN1SaSERKS_(

// LLVM-LABEL: define linkonce_odr ptr @_ZN1S1TaSERKS0_(
// LLVM-LABEL: define linkonce_odr ptr @_ZN1SaSERKS_(
// LLVM:         call ptr @_ZN1S1TaSERKS0_(
// LLVM-LABEL: define dso_local void @_Z1fR1SS0_(
// LLVM:         call ptr @_ZN1SaSERKS_(
void f(S &s1, S &s2) { s1 = s2; }
