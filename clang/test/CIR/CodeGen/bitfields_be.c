// RUN: %clang_cc1  -triple aarch64_be-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1  -triple aarch64_be-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1  -triple aarch64_be-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef struct {
    int a : 4;
    int b : 11;
    int c : 17;
} S;
S s;

// CIR:  !rec_S = !cir.record<struct "S" {!u32i}>
// LLVM: %struct.S = type { i32 }
// OGCG: %struct.S = type { i32 }
