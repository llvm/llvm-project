// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fenable-matrix -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fenable-matrix -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fenable-matrix -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef float matrix3x3 __attribute__((matrix_type(3, 3)));

matrix3x3 a;

// CIR: cir.global external @a = #cir.zero : !cir.matrix<3 x 3 x !cir.float>
// LLVM: @a = global [9 x float] zeroinitializer, align 4

void local_matrix() {
  matrix3x3 a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// LLVM: %[[A_ADDR:.*]] = alloca [9 x float], align 4
