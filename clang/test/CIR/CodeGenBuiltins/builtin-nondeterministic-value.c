// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef float float4 __attribute__((ext_vector_type(4)));

// TODO: the bool4 (ext_vector_type _Bool) case from the classic CodeGen test is
// omitted here: CIR does not yet implement storing ext-vector-bool types
// (emitStoreOfScalar ExtVectorBoolType is NYI). The cir.freeze lowering itself
// works for <N x !cir.bool>; only the surrounding store is unsupported.

int clang_nondet_i(int x) {
  return __builtin_nondeterministic_value(x);
}

// CIR-LABEL: cir.func {{.*}}@clang_nondet_i
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !s32i
// CIR: %[[RES:.*]] = cir.freeze %[[POISON]] : !s32i

// LLVM-LABEL: @clang_nondet_i
// LLVM: %[[RES:.*]] = freeze i32 poison

float clang_nondet_f(float x) {
  return __builtin_nondeterministic_value(x);
}

// CIR-LABEL: cir.func {{.*}}@clang_nondet_f
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.float
// CIR: %[[RES:.*]] = cir.freeze %[[POISON]] : !cir.float

// LLVM-LABEL: @clang_nondet_f
// LLVM: %[[RES:.*]] = freeze float poison

double clang_nondet_d(double x) {
  return __builtin_nondeterministic_value(x);
}

// CIR-LABEL: cir.func {{.*}}@clang_nondet_d
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.double
// CIR: %[[RES:.*]] = cir.freeze %[[POISON]] : !cir.double

// LLVM-LABEL: @clang_nondet_d
// LLVM: %[[RES:.*]] = freeze double poison

_Bool clang_nondet_b(_Bool x) {
  return __builtin_nondeterministic_value(x);
}

// CIR-LABEL: cir.func {{.*}}@clang_nondet_b
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.bool
// CIR: %[[RES:.*]] = cir.freeze %[[POISON]] : !cir.bool

// LLVM-LABEL: @clang_nondet_b
// LLVM: %[[RES:.*]] = freeze i1 poison

void clang_nondet_fv(void) {
  float4 x = __builtin_nondeterministic_value(x);
}

// CIR-LABEL: cir.func {{.*}}@clang_nondet_fv
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !cir.float>
// CIR: %[[RES:.*]] = cir.freeze %[[POISON]] : !cir.vector<4 x !cir.float>

// LLVM-LABEL: @clang_nondet_fv
// LLVM: %[[RES:.*]] = freeze <4 x float> poison
