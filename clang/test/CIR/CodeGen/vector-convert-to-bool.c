// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef int int4 __attribute__((ext_vector_type(4)));
typedef long long2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef _Bool bool4 __attribute__((ext_vector_type(4)));
typedef _Bool bool2 __attribute__((ext_vector_type(2)));

// The bool vector is kept in registers (round-tripped back to an integer
// vector) so this isolates the element-wise int/float <-> bool casts.

int4 int_to_bool_vec(int4 a) {
  return __builtin_convertvector(__builtin_convertvector(a, bool4), int4);
}

// CIR-LABEL: cir.func{{.*}} @int_to_bool_vec
// CIR: %[[M:.*]] = cir.cast int_to_bool %{{.+}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.bool>
// CIR: cir.cast bool_to_int %[[M]] : !cir.vector<4 x !cir.bool> -> !cir.vector<4 x !s32i>

// LLVM-LABEL: @int_to_bool_vec
// LLVM: %[[M:.*]] = icmp ne <4 x i32> %{{.+}}, zeroinitializer
// LLVM: zext <4 x i1> %[[M]] to <4 x i32>

long2 long_to_bool_vec(long2 a) {
  return __builtin_convertvector(__builtin_convertvector(a, bool2), long2);
}

// CIR-LABEL: cir.func{{.*}} @long_to_bool_vec
// CIR: %[[M:.*]] = cir.cast int_to_bool %{{.+}} : !cir.vector<2 x !s64i> -> !cir.vector<2 x !cir.bool>
// CIR: cir.cast bool_to_int %[[M]] : !cir.vector<2 x !cir.bool> -> !cir.vector<2 x !s64i>

// LLVM-LABEL: @long_to_bool_vec
// LLVM: %[[M:.*]] = icmp ne <2 x i64> %{{.+}}, zeroinitializer
// LLVM: zext <2 x i1> %[[M]] to <2 x i64>

int4 float_to_bool_vec(float4 a) {
  return __builtin_convertvector(__builtin_convertvector(a, bool4), int4);
}

// CIR-LABEL: cir.func{{.*}} @float_to_bool_vec
// CIR: %[[M:.*]] = cir.cast float_to_bool %{{.+}} : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !cir.bool>
// CIR: cir.cast bool_to_int %[[M]] : !cir.vector<4 x !cir.bool> -> !cir.vector<4 x !s32i>

// LLVM-LABEL: @float_to_bool_vec
// LLVM: %[[M:.*]] = fcmp une <4 x float> %{{.+}}, zeroinitializer
// LLVM: zext <4 x i1> %[[M]] to <4 x i32>

float4 bool_to_float_vec(int4 a) {
  return __builtin_convertvector(__builtin_convertvector(a, bool4), float4);
}

// CIR-LABEL: cir.func{{.*}} @bool_to_float_vec
// CIR: %[[M:.*]] = cir.cast int_to_bool %{{.+}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.bool>
// CIR: cir.cast bool_to_float %[[M]] : !cir.vector<4 x !cir.bool> -> !cir.vector<4 x !cir.float>

// LLVM-LABEL: @bool_to_float_vec
// LLVM: %[[M:.*]] = icmp ne <4 x i32> %{{.+}}, zeroinitializer
// LLVM: uitofp <4 x i1> %[[M]] to <4 x float>
