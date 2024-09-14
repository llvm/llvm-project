// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -fexceptions -fcxx-exceptions -triple aarch64-none-linux-android21 -Wno-unused-value -fclangir -emit-cir %s -o %t.eh.cir
// RUN: FileCheck --check-prefix=CIR_EH --input-file=%t.eh.cir %s

struct Vec {
  Vec();
  Vec(Vec&&);
  ~Vec();
};

struct S1 {
  Vec v;
};

// CIR-DAG: ![[VecType:.*]] = !cir.struct<struct "Vec" {!cir.int<u, 8>}>
// CIR-DAG: ![[S1:.*]] = !cir.struct<struct "S1" {!cir.struct<struct "Vec" {!cir.int<u, 8>}>}>

// CIR_EH-DAG: ![[VecType:.*]] = !cir.struct<struct "Vec" {!cir.int<u, 8>}>
// CIR_EH-DAG: ![[S1:.*]] = !cir.struct<struct "S1" {!cir.struct<struct "Vec" {!cir.int<u, 8>}>}>

template <int I>
void make1() {
  Vec v;
  S1((Vec&&) v);
// CIR: cir.func linkonce_odr  @_Z5make1ILi0EEvv()
// CIR:   %[[VEC:.*]] = cir.alloca ![[VecType]], !cir.ptr<![[VecType]]>
// CIR:   cir.call @_ZN3VecC1Ev(%[[VEC]]) : (!cir.ptr<![[VecType]]>)
// CIR:   cir.scope {
// CIR:     %[[AGG_TMP:.*]] = cir.alloca ![[S1]], !cir.ptr<![[S1]]>, ["agg.tmp.ensured"]
// CIR:     %[[FIELD:.*]] = cir.get_member %[[AGG_TMP]][0] {name = "v"} : !cir.ptr<![[S1]]> -> !cir.ptr<![[VecType]]>
// CIR:     cir.call @_ZN3VecC1EOS_(%[[FIELD]], %[[VEC]]) : (!cir.ptr<![[VecType]]>, !cir.ptr<![[VecType]]>) -> ()
// CIR:     cir.call @_ZN2S1D1Ev(%[[AGG_TMP]]) : (!cir.ptr<![[S1]]>) -> ()
// CIR:   }
// CIR:   cir.call @_ZN3VecD1Ev(%[[VEC]]) : (!cir.ptr<![[VecType]]>) -> ()
// CIR:   cir.return

// CIR_EH: cir.func linkonce_odr  @_Z5make1ILi0EEvv()
// CIR_EH:  %[[VEC:.*]] = cir.alloca ![[VecType]], !cir.ptr<![[VecType]]>, ["v", init]

// Construct v
// CIR_EH:  cir.call @_ZN3VecC1Ev(%[[VEC]]) : (!cir.ptr<![[VecType]]>) -> ()
// CIR_EH:  cir.scope {
// CIR_EH:    %1 = cir.alloca ![[S1]], !cir.ptr<![[S1]]>, ["agg.tmp.ensured"]
// CIR_EH:    %2 = cir.get_member %1[0] {name = "v"} : !cir.ptr<![[S1]]> -> !cir.ptr<![[VecType]]>
// CIR_EH:    cir.try synthetic cleanup {

// Call v move ctor
// CIR_EH:      cir.call exception @_ZN3VecC1EOS_{{.*}} cleanup {

// Destroy v after v move ctor throws
// CIR_EH:        cir.call @_ZN3VecD1Ev(%[[VEC]])
// CIR_EH:        cir.yield
// CIR_EH:      }
// CIR_EH:      cir.yield
// CIR_EH:    } catch [#cir.unwind {
// CIR_EH:      cir.resume
// CIR_EH:    }]
// CIR_EH:    cir.call @_ZN2S1D1Ev(%1) : (!cir.ptr<![[S1]]>) -> ()
// CIR_EH:  }

// Destroy v after successful cir.try
// CIR_EH:  cir.call @_ZN3VecD1Ev(%[[VEC]]) : (!cir.ptr<![[VecType]]>) -> ()
// CIR_EH:  cir.return
}

void foo() {
  make1<0>();
}