// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: cir-translate %t.cir -cir-to-llvmir -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct E {
  ~E();
  E operator!();
};

void f() {
  !E();
}

//      CHECK: cir.func private  @_ZN1EC1Ev(!cir.ptr<!ty_E>) extra(#fn_attr)
// CHECK-NEXT: cir.func private  @_ZN1EntEv(!cir.ptr<!ty_E>) -> !ty_E
// CHECK-NEXT: cir.func private  @_ZN1ED1Ev(!cir.ptr<!ty_E>) extra(#fn_attr)
// CHECK-NEXT: cir.func  @_Z1fv() extra(#fn_attr1) {
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %[[ONE:[0-9]+]] = cir.alloca !ty_E, !cir.ptr<!ty_E>, ["agg.tmp.ensured"] {alignment = 1 : i64}
// CHECK-NEXT:     %[[TWO:[0-9]+]] = cir.alloca !ty_E, !cir.ptr<!ty_E>, ["ref.tmp0"] {alignment = 1 : i64}
// CHECK-NEXT:     cir.call @_ZN1EC1Ev(%1) : (!cir.ptr<!ty_E>) -> () extra(#fn_attr)
// CHECK-NEXT:     %[[THREE:[0-9]+]] = cir.call @_ZN1EntEv(%[[TWO]]) : (!cir.ptr<!ty_E>) -> !ty_E
// CHECK-NEXT:     cir.store %[[THREE]], %[[ONE]] : !ty_E, !cir.ptr<!ty_E>
// CHECK-NEXT:     cir.call @_ZN1ED1Ev(%[[ONE]]) : (!cir.ptr<!ty_E>) -> () extra(#fn_attr)
// CHECK-NEXT:     cir.call @_ZN1ED1Ev(%[[TWO]]) : (!cir.ptr<!ty_E>) -> () extra(#fn_attr)
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

const unsigned int n = 1234;
const int &r = (const int&)n;

//      CHECK: cir.global "private"  constant internal @_ZGR1r_ = #cir.int<1234> : !s32i
// CHECK-NEXT: cir.global  external @r = #cir.global_view<@_ZGR1r_> : !cir.ptr<!s32i> {alignment = 8 : i64}

//      LLVM: @_ZGR1r_ = internal constant i32 1234, align 4
// LLVM-NEXT: @r = global ptr @_ZGR1r_, align 8

