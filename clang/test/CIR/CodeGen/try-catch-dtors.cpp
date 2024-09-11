// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

struct Vec {
  Vec();
  Vec(Vec&&);
  ~Vec();
};

void yo() {
  int r = 1;
  try {
    Vec v;
  } catch (...) {
    r++;
  }
}

// CIR-DAG: ![[VecTy:.*]] = !cir.struct<struct "Vec" {!cir.int<u, 8>}>

// CIR: cir.scope {
// CIR:   %[[VADDR:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v", init]
// CIR:   cir.try {
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[VADDR]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call @_ZN3VecD1Ev(%[[VADDR]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.yield
// CIR:   } cleanup {
// CIR:   } catch [type #cir.all {
// CIR:     cir.catch_param -> !cir.ptr<!void>
// CIR:   }]
// CIR: }
// CIR: cir.return