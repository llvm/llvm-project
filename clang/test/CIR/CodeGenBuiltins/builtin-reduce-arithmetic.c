// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef int v4si __attribute__((vector_size(16)));
typedef unsigned int v4su __attribute__((vector_size(16)));
typedef float v4sf __attribute__((vector_size(16)));
typedef double v2df __attribute__((vector_size(16)));

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

float test_reduce_in_order_fadd(v4sf x, float start) {
  // CIR-LABEL: @test_reduce_in_order_fadd
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" {{.*}} : (!cir.float, !cir.vector<4 x !cir.float>) -> !cir.float
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_in_order_fadd
  // LLVM: call float @llvm.vector.reduce.fadd.v4f32(float %{{.*}}, <4 x float>
  // LLVM: ret float
  return __builtin_reduce_in_order_fadd(x, start);
}

float test_reduce_in_order_fadd_cast_start(v4sf x, double start) {
  // CIR-LABEL: @test_reduce_in_order_fadd_cast_start
  // CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" {{.*}} : (!cir.float, !cir.vector<4 x !cir.float>) -> !cir.float
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_in_order_fadd_cast_start
  // LLVM: fptrunc double %{{.*}} to float
  // LLVM: call float @llvm.vector.reduce.fadd.v4f32(float %{{.*}}, <4 x float>
  // LLVM: ret float
  return __builtin_reduce_in_order_fadd(x, start);
}

double test_reduce_in_order_fadd_double(v2df x, double start) {
  // CIR-LABEL: @test_reduce_in_order_fadd_double
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" {{.*}} : (!cir.double, !cir.vector<2 x !cir.double>) -> !cir.double
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_in_order_fadd_double
  // LLVM: call double @llvm.vector.reduce.fadd.v2f64(double %{{.*}}, <2 x double>
  // LLVM: ret double
  return __builtin_reduce_in_order_fadd(x, start);
}

double test_reduce_in_order_fadd_ext_start(v2df x, float start) {
  // CIR-LABEL: @test_reduce_in_order_fadd_ext_start
  // CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double
  // CIR: cir.call_llvm_intrinsic "vector.reduce.fadd" {{.*}} : (!cir.double, !cir.vector<2 x !cir.double>) -> !cir.double
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_in_order_fadd_ext_start
  // LLVM: fpext float %{{.*}} to double
  // LLVM: call double @llvm.vector.reduce.fadd.v2f64(double %{{.*}}, <2 x double>
  // LLVM: ret double
  return __builtin_reduce_in_order_fadd(x, start);
}
