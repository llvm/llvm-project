// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

struct C {
  void f();
  void f2(int a, int b);
};

// CIR: !rec_C = !cir.record<struct "C" padded {!u8i}>

void C::f() {}

// CIR: cir.func{{.*}} @_ZN1C1fEv(%[[THIS_ARG:.*]]: !cir.ptr<!rec_C>
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["this", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]] : !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
// CIR:   cir.return
// CIR: }

void C::f2(int a, int b) {}

// CIR:      cir.func{{.*}} @_ZN1C2f2Eii(%[[THIS_ARG:.*]]: !cir.ptr<!rec_C> {{.*}}, %[[A_ARG:.*]]: !s32i {{.*}}, %[[B_ARG:.*]]: !s32i {{.*}})
// CIR-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["this", init]
// CIR-NEXT:   %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR-NEXT:   %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR-NEXT:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]] : !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>
// CIR-NEXT:   cir.store %[[A_ARG]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %[[B_ARG]], %[[B_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
// CIR-NEXT:   cir.return
// CIR-NEXT: }

void test1() {
  C c;
  c.f();
  c.f2(1, 2);
}

// CIR: cir.func{{.*}} @_Z5test1v()
// CIR-NEXT:   %[[C_ADDR:.*]] = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["c"]
// CIR-NEXT:   cir.call @_ZN1C1fEv(%[[C_ADDR]]) : (!cir.ptr<!rec_C>) -> ()
// CIR-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:   %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:   cir.call @_ZN1C2f2Eii(%[[C_ADDR]], %[[ONE]], %[[TWO]]) : (!cir.ptr<!rec_C>, !s32i, !s32i) -> ()
// CIR-NEXT:   cir.return
// CIR-NEXT: }
