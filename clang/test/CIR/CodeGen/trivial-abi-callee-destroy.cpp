// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct __attribute__((trivial_abi)) Handle {
  int *p;
  Handle();
  Handle(const Handle &);
  ~Handle();
};

void take(Handle h) {}

// CIR-LABEL: cir.func {{.*}} @_Z4take6Handle(
// CIR:         %[[H:.+]] = cir.alloca "h"
// CIR:         cir.call @_ZN6HandleD1Ev(%[[H]])
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} void @_Z4take6Handle(
// LLVM:         call void @_ZN6HandleD1Ev(ptr {{.*}} %[[H:.+]])
// LLVM:         ret void

// OGCG-LABEL: define {{.*}} void @_Z4take6Handle(
// OGCG:         call void @_ZN6HandleD1Ev(ptr {{.*}} %h)
// OGCG:         ret void

int deref(Handle h) {
  if (*h.p)
    return 1;
  return 2;
}

// CIR-LABEL: cir.func {{.*}} @_Z5deref6Handle(
// CIR:         %[[H:.+]] = cir.alloca "h"
// CIR:         cir.cleanup.scope {
// CIR:           cir.return
// CIR:           cir.return
// CIR:         } cleanup normal {
// CIR-NEXT:      cir.call @_ZN6HandleD1Ev(%[[H]])

// LLVM-LABEL: define {{.*}} i32 @_Z5deref6Handle(
// LLVM:         call void @_ZN6HandleD1Ev(
// LLVM-NOT:     call void @_ZN6HandleD1Ev(
// LLVM:         ret i32

// OGCG-LABEL: define {{.*}} i32 @_Z5deref6Handle(
// OGCG:         call void @_ZN6HandleD1Ev(
// OGCG-NOT:     call void @_ZN6HandleD1Ev(
// OGCG:         ret i32

void caller() {
  Handle h;
  take(h);
}

// CIR-LABEL: cir.func {{.*}} @_Z6callerv(
// CIR:         cir.call @_Z4take6Handle(
// CIR:         cir.call @_ZN6HandleD1Ev(
// CIR-NOT:     cir.call @_ZN6HandleD1Ev(
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} void @_Z6callerv(
// LLVM:         call void @_Z4take6Handle(
// LLVM:         call void @_ZN6HandleD1Ev(
// LLVM-NOT:     call void @_ZN6HandleD1Ev(
// LLVM:         ret void

// OGCG-LABEL: define {{.*}} void @_Z6callerv(
// OGCG:         call void @_Z4take6Handle(
// OGCG:         call void @_ZN6HandleD1Ev(
// OGCG-NOT:     call void @_ZN6HandleD1Ev(
// OGCG:         ret void
