// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct e { e(int); };
e *g = new e(0);

//CHECK:  {{%.*}} = cir.const(#cir.int<1> : !u64i) : !u64i loc(#loc11)
//CHECK:  {{%.*}} = cir.call @_Znwm(%1) : (!u64i) -> !cir.ptr<!void> loc(#loc6)
