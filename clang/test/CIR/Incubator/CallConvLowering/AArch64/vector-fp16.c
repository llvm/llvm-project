// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +fullfp16 \
// RUN:   -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +fullfp16 \
// RUN:   -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +fullfp16 \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef _Float16 half;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));

// CIR: cir.func {{.*}} @test_half(%arg0: !cir.f16 {{.*}}) -> !cir.f16
// LLVM: define {{.*}}half @test_half(half %{{.*}})
// OGCG: define {{.*}}half @test_half(half {{.*}}%{{.*}})
half test_half(half a) {
  return a;
}

// CIR: cir.func {{.*}} @test_half_add(%arg0: !cir.f16 {{.*}}, %arg1: !cir.f16 {{.*}}) -> !cir.f16
// LLVM: define {{.*}}half @test_half_add(half %{{.*}}, half %{{.*}})
// OGCG: define {{.*}}half @test_half_add(half {{.*}}%{{.*}}, half {{.*}}%{{.*}})
half test_half_add(half a, half b) {
  return a + b;
}

// CIR: cir.func {{.*}} @test_float4(%arg0: !cir.vector<!cir.float x 4> {{.*}}) -> !cir.vector<!cir.float x 4>
// LLVM: define {{.*}}<4 x float> @test_float4(<4 x float> %{{.*}})
// OGCG: define {{.*}}<4 x float> @test_float4(<4 x float> {{.*}}%{{.*}})
float4 test_float4(float4 a) {
  return a;
}

// CIR: cir.func {{.*}} @test_float4_add(%arg0: !cir.vector<!cir.float x 4> {{.*}}, %arg1: !cir.vector<!cir.float x 4> {{.*}}) -> !cir.vector<!cir.float x 4>
// LLVM: define {{.*}}<4 x float> @test_float4_add(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// OGCG: define {{.*}}<4 x float> @test_float4_add(<4 x float> {{.*}}%{{.*}}, <4 x float> {{.*}}%{{.*}})
float4 test_float4_add(float4 a, float4 b) {
  return a + b;
}

// CIR: cir.func {{.*}} @test_int4(%arg0: !cir.vector<!s32i x 4> {{.*}}) -> !cir.vector<!s32i x 4>
// LLVM: define {{.*}}<4 x i32> @test_int4(<4 x i32> %{{.*}})
// OGCG: define {{.*}}<4 x i32> @test_int4(<4 x i32> {{.*}}%{{.*}})
int4 test_int4(int4 a) {
  return a;
}

// CIR: cir.func {{.*}} @test_int4_add(%arg0: !cir.vector<!s32i x 4> {{.*}}, %arg1: !cir.vector<!s32i x 4> {{.*}}) -> !cir.vector<!s32i x 4>
// LLVM: define {{.*}}<4 x i32> @test_int4_add(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// OGCG: define {{.*}}<4 x i32> @test_int4_add(<4 x i32> {{.*}}%{{.*}}, <4 x i32> {{.*}}%{{.*}})
int4 test_int4_add(int4 a, int4 b) {
  return a + b;
}

// CIR: cir.func {{.*}} @test_half4(%arg0: !cir.vector<!cir.f16 x 4> {{.*}}) -> !cir.vector<!cir.f16 x 4>
// LLVM: define {{.*}}<4 x half> @test_half4(<4 x half> %{{.*}})
// OGCG: define {{.*}}<4 x half> @test_half4(<4 x half> {{.*}}%{{.*}})
half4 test_half4(half4 a) {
  return a;
}

// CIR: cir.func {{.*}} @test_half4_add(%arg0: !cir.vector<!cir.f16 x 4> {{.*}}, %arg1: !cir.vector<!cir.f16 x 4> {{.*}}) -> !cir.vector<!cir.f16 x 4>
// LLVM: define {{.*}}<4 x half> @test_half4_add(<4 x half> %{{.*}}, <4 x half> %{{.*}})
// OGCG: define {{.*}}<4 x half> @test_half4_add(<4 x half> {{.*}}%{{.*}}, <4 x half> {{.*}}%{{.*}})
half4 test_half4_add(half4 a, half4 b) {
  return a + b;
}

// CIR: cir.func {{.*}} @test_void_ptr(%arg0: !cir.ptr<!void> {{.*}}) -> !cir.ptr<!void>
// LLVM: define {{.*}}ptr @test_void_ptr(ptr %{{.*}})
// OGCG: define {{.*}}ptr @test_void_ptr(ptr {{.*}}%{{.*}})
void *test_void_ptr(void *p) {
  return p;
}

// CIR: cir.func {{.*}} @test_void_ptr_arith(%arg0: !cir.ptr<!void> {{.*}}, %arg1: !s64i {{.*}}) -> !cir.ptr<!void>
// LLVM: define {{.*}}ptr @test_void_ptr_arith(ptr %{{.*}}, i64 %{{.*}})
// OGCG: define {{.*}}ptr @test_void_ptr_arith(ptr {{.*}}%{{.*}}, i64 {{.*}}%{{.*}})
void *test_void_ptr_arith(void *p, long offset) {
  return (char*)p + offset;
}
