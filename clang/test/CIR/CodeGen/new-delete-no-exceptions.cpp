// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-cir -std=c++11 -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s \
// RUN:   --implicit-check-not=cleanup.isactive \
// RUN:   --implicit-check-not="cleanup eh" \
// RUN:   --implicit-check-not=@_ZdlPv
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -fclangir -emit-llvm -std=c++11 -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s \
// RUN:   --implicit-check-not=personality \
// RUN:   --implicit-check-not=invoke \
// RUN:   --implicit-check-not=landingpad \
// RUN:   --implicit-check-not=@_ZdlPv
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=x86_64-linux-gnu -emit-llvm -std=c++11 -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s \
// RUN:   --implicit-check-not=personality \
// RUN:   --implicit-check-not=invoke \
// RUN:   --implicit-check-not=landingpad \
// RUN:   --implicit-check-not=@_ZdlPv

struct A {
  A(int);
  ~A();
};

struct B {
  B();
  ~B();
  operator int();
};

B makeB();

A *noex_new() { return new A(makeB()); }

// CIR-LABEL: cir.func {{.*}} @_Z8noex_newv()
// CIR:   cir.call @_Znwm({{.*}}) {{{.*}}builtin}
// CIR:   cir.call @_Z5makeBv
// CIR:   cir.cleanup.scope {
// CIR:     cir.call @_ZN1BcviEv
// CIR:     cir.call @_ZN1AC1Ei
// CIR:   } cleanup normal {
// CIR:     cir.call @_ZN1BD1Ev{{.*}} nothrow
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_Z8noex_newv()
// LLVM:   call {{.*}} @_Znwm(i64 1)
// LLVM:   call void @_Z5makeBv
// LLVM:   call i32 @_ZN1BcviEv
// LLVM:   call void @_ZN1AC1Ei
// LLVM:   call void @_ZN1BD1Ev
// LLVM:   ret ptr
