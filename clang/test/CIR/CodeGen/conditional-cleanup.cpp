// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

typedef __typeof(sizeof(0)) size_t;

// Declare the reserved global placement new.
void *operator new(size_t, void*);

namespace test7 {
  struct A { A(); ~A(); };
  struct B {
    static void *operator new(size_t size) throw();
    B(const A&, B*);
    ~B();
  };

  B *test() {
    return new B(A(), new B(A(), 0));
  }
}

// CIR-DAG: ![[A:.*]] = !cir.struct<struct "test7::A" {!cir.int<u, 8>}
// CIR-DAG: ![[B:.*]] = !cir.struct<struct "test7::B" {!cir.int<u, 8>}

// CIR-LABEL: _ZN5test74testEv
// CIR:   %[[RET_VAL:.*]] = cir.alloca !cir.ptr<![[B]]>, !cir.ptr<!cir.ptr<![[B]]>>, ["__retval"] {alignment = 8 : i64}
// CIR:   cir.scope {
// CIR:     %[[TMP_A0:.*]] = cir.alloca ![[A]], !cir.ptr<![[A]]>, ["ref.tmp0"] {alignment = 1 : i64}
// CIR:     %[[CLEANUP_COND_OUTER:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR:     %[[TMP_A1:.*]] = cir.alloca ![[A]], !cir.ptr<![[A]]>, ["ref.tmp1"] {alignment = 1 : i64}
// CIR:     %[[CLEANUP_COND_INNER:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR:     %[[FALSE0:.*]] = cir.const #false
// CIR:     %[[TRUE0:.*]] = cir.const #true
// CIR:     %[[FALSE1:.*]] = cir.const #false
// CIR:     %[[TRUE1:.*]] = cir.const #true

// CIR:     %[[NULL_CHECK0:.*]] = cir.cmp(ne
// CIR:     %[[PTR_B0:.*]] = cir.cast(bitcast
// CIR:     cir.store align(1) %[[FALSE1]], %[[CLEANUP_COND_OUTER]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.store align(1) %[[FALSE0]], %[[CLEANUP_COND_INNER]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:     cir.if %[[NULL_CHECK0]] {

// Ctor call: @test7::A::A()
// CIR:       cir.call @_ZN5test71AC1Ev(%[[TMP_A0]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:       cir.store %[[TRUE1]], %[[CLEANUP_COND_OUTER]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR:       %[[NULL_CHECK1:.*]] = cir.cmp(ne
// CIR:       %[[PTR_B1:.*]] = cir.cast(bitcast
// CIR:       cir.if %[[NULL_CHECK1]] {

// Ctor call: @test7::A::A()
// CIR:         cir.call @_ZN5test71AC1Ev(%[[TMP_A1]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:         cir.store %[[TRUE0]], %[[CLEANUP_COND_INNER]] : !cir.bool, !cir.ptr<!cir.bool>
// Ctor call: @test7::B::B()
// CIR:         cir.call @_ZN5test71BC1ERKNS_1AEPS0_(%[[PTR_B1]], %[[TMP_A1]], {{.*}}) : (!cir.ptr<![[B]]>, !cir.ptr<![[A]]>, !cir.ptr<![[B]]>) -> ()
// CIR:       }

// Ctor call: @test7::B::B()
// CIR:       cir.call @_ZN5test71BC1ERKNS_1AEPS0_(%[[PTR_B0]], %[[TMP_A0]], %[[PTR_B1]]) : (!cir.ptr<![[B]]>, !cir.ptr<![[A]]>, !cir.ptr<![[B]]>) -> ()
// CIR:     }
// CIR:     cir.store %[[PTR_B0]], %[[RET_VAL]] : !cir.ptr<![[B]]>, !cir.ptr<!cir.ptr<![[B]]>>
// CIR:     %[[DO_CLEANUP_INNER:.*]] = cir.load %[[CLEANUP_COND_INNER]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[DO_CLEANUP_INNER]] {
// Dtor call: @test7::A::~A()
// CIR:       cir.call @_ZN5test71AD1Ev(%[[TMP_A1]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:     }
// CIR:     %[[DO_CLEANUP_OUTER:.*]] = cir.load %[[CLEANUP_COND_OUTER]] : !cir.ptr<!cir.bool>, !cir.bool
// Dtor call: @test7::A::~A()
// CIR:     cir.if %[[DO_CLEANUP_OUTER]] {
// CIR:       cir.call @_ZN5test71AD1Ev(%[[TMP_A0]]) : (!cir.ptr<![[A]]>) -> ()
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }