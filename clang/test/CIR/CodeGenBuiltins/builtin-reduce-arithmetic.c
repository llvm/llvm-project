// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef int v4si __attribute__((vector_size(16)));
typedef unsigned int v4su __attribute__((vector_size(16)));
typedef float v4sf __attribute__((vector_size(16)));

int test_reduce_add(v4si x) {
  // CIR-LABEL: @test_reduce_add
  // CIR: cir.call_llvm_intrinsic "vector.reduce.add"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_add
  // LLVM: call i32 @llvm.vector.reduce.add.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_add(x);
}

unsigned test_reduce_add_unsigned(v4su x) {
  // CIR-LABEL: @test_reduce_add_unsigned
  // CIR: cir.call_llvm_intrinsic "vector.reduce.add"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_add_unsigned
  // LLVM: call i32 @llvm.vector.reduce.add.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_add(x);
}

int test_reduce_mul(v4si x) {
  // CIR-LABEL: @test_reduce_mul
  // CIR: cir.call_llvm_intrinsic "vector.reduce.mul"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_mul
  // LLVM: call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_mul(x);
}

unsigned test_reduce_mul_unsigned(v4su x) {
  // CIR-LABEL: @test_reduce_mul_unsigned
  // CIR: cir.call_llvm_intrinsic "vector.reduce.mul"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_mul_unsigned
  // LLVM: call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_mul(x);
}

int test_reduce_max(v4si x) {
  // CIR-LABEL: @test_reduce_max
  // CIR: cir.call_llvm_intrinsic "vector.reduce.smax"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_max
  // LLVM: call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_max(x);
}

unsigned test_reduce_max_unsigned(v4su x) {
  // CIR-LABEL: @test_reduce_max_unsigned
  // CIR: cir.call_llvm_intrinsic "vector.reduce.umax"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_max_unsigned
  // LLVM: call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_max(x);
}

float test_reduce_max_float(v4sf x) {
  // CIR-LABEL: @test_reduce_max_float
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmax"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_max_float
  // LLVM: call float @llvm.vector.reduce.fmax.v4f32(<4 x float>
  // LLVM: ret float
  return __builtin_reduce_max(x);
}

int test_reduce_min(v4si x) {
  // CIR-LABEL: @test_reduce_min
  // CIR: cir.call_llvm_intrinsic "vector.reduce.smin"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_min
  // LLVM: call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_min(x);
}

unsigned test_reduce_min_unsigned(v4su x) {
  // CIR-LABEL: @test_reduce_min_unsigned
  // CIR: cir.call_llvm_intrinsic "vector.reduce.umin"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_min_unsigned
  // LLVM: call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_min(x);
}

float test_reduce_min_float(v4sf x) {
  // CIR-LABEL: @test_reduce_min_float
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fmin"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_min_float
  // LLVM: call float @llvm.vector.reduce.fmin.v4f32(<4 x float>
  // LLVM: ret float
  return __builtin_reduce_min(x);
}
