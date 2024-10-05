// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -emit-cir %s -o %t.eh.cir
// RUN: FileCheck --input-file=%t.eh.cir %s -check-prefix=CIR_EH
// RUN: cir-translate %t.cir -cir-to-llvmir -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct E {
  ~E();
  E operator!();
};

void f() {
  !E();
}

//      CIR: cir.func private  @_ZN1EC1Ev(!cir.ptr<!ty_E>) extra(#fn_attr)
// CIR-NEXT: cir.func private  @_ZN1EntEv(!cir.ptr<!ty_E>) -> !ty_E
// CIR-NEXT: cir.func private  @_ZN1ED1Ev(!cir.ptr<!ty_E>) extra(#fn_attr)
// CIR-NEXT: cir.func  @_Z1fv() extra(#fn_attr1) {
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     %[[ONE:[0-9]+]] = cir.alloca !ty_E, !cir.ptr<!ty_E>, ["agg.tmp.ensured"] {alignment = 1 : i64}
// CIR-NEXT:     %[[TWO:[0-9]+]] = cir.alloca !ty_E, !cir.ptr<!ty_E>, ["ref.tmp0"] {alignment = 1 : i64}
// CIR-NEXT:     cir.call @_ZN1EC1Ev(%1) : (!cir.ptr<!ty_E>) -> () extra(#fn_attr)
// CIR-NEXT:     %[[THREE:[0-9]+]] = cir.call @_ZN1EntEv(%[[TWO]]) : (!cir.ptr<!ty_E>) -> !ty_E
// CIR-NEXT:     cir.store %[[THREE]], %[[ONE]] : !ty_E, !cir.ptr<!ty_E>
// CIR-NEXT:     cir.call @_ZN1ED1Ev(%[[ONE]]) : (!cir.ptr<!ty_E>) -> () extra(#fn_attr)
// CIR-NEXT:     cir.call @_ZN1ED1Ev(%[[TWO]]) : (!cir.ptr<!ty_E>) -> () extra(#fn_attr)
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// CIR_EH-LABEL: @_Z1fv
// CIR_EH: %[[AGG_TMP:.*]] = cir.alloca {{.*}} ["agg.tmp.ensured"]
// CIR_EH: cir.try synthetic cleanup {
// CIR_EH:   %[[RVAL:.*]] = cir.call exception {{.*}} cleanup {
// CIR_EH:     cir.call @_ZN1ED1Ev
// CIR_EH:     cir.yield
// CIR_EH:   }
// CIR_EH:   cir.store %[[RVAL]], %[[AGG_TMP]]
// CIR_EH:   cir.yield
// CIR_EH: } catch [#cir.unwind {

const unsigned int n = 1234;
const int &r = (const int&)n;

//      CIR: cir.global "private"  constant internal @_ZGR1r_ = #cir.int<1234> : !s32i
// CIR-NEXT: cir.global  constant external @r = #cir.global_view<@_ZGR1r_> : !cir.ptr<!s32i> {alignment = 8 : i64}

//      LLVM: @_ZGR1r_ = internal constant i32 1234, align 4
// LLVM-NEXT: @r = constant ptr @_ZGR1r_, align 8

