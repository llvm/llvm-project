// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

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
}

void foo() {
  make1<0>();
}