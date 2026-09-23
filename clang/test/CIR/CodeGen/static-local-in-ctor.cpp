// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct Ctor {
  Ctor() {
    static int x = 42;
    (void)x;
  }
};

struct Dtor {
  ~Dtor() {
    static int y = 7;
    (void)y;
  }
};

void use() { Ctor c; Dtor d; }


// Static local inside the constructor body.

// CIR-DAG: cir.global linkonce_odr comdat @_ZZN4CtorC1EvE1x = #cir.int<42> : !s32i {alignment = 4 : i64}
// LLVM-DAG: @_ZZN4CtorC1EvE1x = linkonce_odr global i32 42, comdat, align 4


// Static local inside the destructor body.

// CIR-DAG: cir.global linkonce_odr comdat @_ZZN4DtorD1EvE1y = #cir.int<7> : !s32i {alignment = 4 : i64}
// LLVM-DAG: @_ZZN4DtorD1EvE1y = linkonce_odr global i32 7, comdat, align 4


// The static local is loaded by cir.get_global inside the base-subobject
// constructor / destructor body (Itanium ABI emits both C1/C2 and D1/D2;
// both reference the same static).

// CIR: cir.func{{.*}}@_ZN4CtorC2Ev
// CIR:   cir.get_global @_ZZN4CtorC1EvE1x : !cir.ptr<!s32i>

// CIR: cir.func{{.*}}@_ZN4DtorD2Ev
// CIR:   cir.get_global @_ZZN4DtorD1EvE1y : !cir.ptr<!s32i>
