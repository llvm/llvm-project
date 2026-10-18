// Lowering ClangIR input to LLVM IR, assembly, bitcode or an object gives the
// same result as compiling the source directly. The CIR-to-CIR pipeline is not
// run again on input printed by -emit-cir, whose lowering passes are not
// idempotent; the code below exercises the calling convention, global
// constructor and TLS wrapper lowering.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.direct.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t.cir -o %t.ll
// RUN: diff %t.direct.ll %t.ll

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S %s -o %t.direct.s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -S %t.cir -o %t.s
// RUN: diff %t.direct.s %t.s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-obj %s -o %t.direct.o
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj %t.cir -o %t.o
// RUN: cmp %t.direct.o %t.o

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm-bc %s -o %t.direct.bc
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc %t.cir -o %t.bc
// RUN: cmp %t.direct.bc %t.bc

// Without -o, the output is named after the input.
// RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir
// RUN: cp %t.cir input.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj input.cir
// RUN: cmp %t.direct.o input.o

// Code generation options still come from the command line.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.O2.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm %s -o %t.O2.direct.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm -x cir %t.O2.cir -o %t.O2.ll
// RUN: diff %t.O2.direct.ll %t.O2.ll

typedef float v2f __attribute__((vector_size(8)));

v2f mk(float a) { return (v2f){a, a}; }

struct S {
  S();
  ~S();
  int v;
};

S gs;

thread_local int tl = 3;

int get() {
  static S local;
  return local.v + tl;
}
