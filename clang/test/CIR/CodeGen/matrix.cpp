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

void load_and_store() {
  matrix3x3 a;
  matrix3x3 b;
  b = a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[TMP_A:.*]] = load <9 x float>, ptr %[[A_ADDR]], align 4
// LLVM: store <9 x float> %[[TMP_A]], ptr %[[B_ADDR]], align 4

void load_global_store_in_local() {
  matrix3x3 b;
  b = a;
}

// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[GLOBAL_A:.*]] = cir.get_global @a : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[GLOBAL_A]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

// LLVM: %[[B_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[TMP_A:.*]] = load <9 x float>, ptr @a, align 4
// LLVM: store <9 x float> %[[TMP_A]], ptr %[[B_ADDR]], align 4
