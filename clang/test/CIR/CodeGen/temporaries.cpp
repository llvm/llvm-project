// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

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
