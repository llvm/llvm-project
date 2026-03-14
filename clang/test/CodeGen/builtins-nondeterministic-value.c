// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef _Bool bool4 __attribute__((ext_vector_type(4)));

int clang_nondet_i( int x ) {
// CHECK-LABEL: entry
// CHECK: [[A:%.*]] = alloca i32
// CHECK: store i32 [[X:%.*]], ptr [[A]]
// CHECK: [[R:%.*]] = freeze i32 poison
// CHECK: ret i32 [[R]]
  return __builtin_nondeterministic_value(x);
}

float clang_nondet_f( float x ) {
// CHECK-LABEL: entry
// CHECK: [[A:%.*]] = alloca float
// CHECK: store float [[X:%.*]], ptr [[A]]
// CHECK: [[R:%.*]] = freeze float poison
// CHECK: ret float [[R]]
  return __builtin_nondeterministic_value(x);
}

double clang_nondet_d( double x ) {
// CHECK-LABEL: entry
// CHECK: [[A:%.*]] = alloca double
// CHECK: store double [[X:%.*]], ptr [[A]]
// CHECK: [[R:%.*]] = freeze double poison
// CHECK: ret double [[R]]
  return __builtin_nondeterministic_value(x);
}

_Bool clang_nondet_b( _Bool x) {
// CHECK-LABEL: entry
// CHECK: [[A:%.*]] = alloca i8
// CHECK: [[B:%.*]] = zext i1 %x to i8
// CHECK: store i8 [[B]], ptr [[A]]
// CHECK: [[R:%.*]] = freeze i1 poison
// CHECK: ret i1 [[R]]
  return __builtin_nondeterministic_value(x);
}

void clang_nondet_fv( ) {
// CHECK-LABEL: entry
// CHECK: [[A:%.*]] = alloca <4 x float>
// CHECK: [[R:%.*]] = freeze <4 x float> poison
// CHECK: store <4 x float> [[R]], ptr [[A]]
// CHECK: ret void
  float4 x = __builtin_nondeterministic_value(x);
}

void clang_nondet_bv( ) {
// CHECK: [[A:%.*]] = alloca i8
// CHECK: [[V:%.*]] = freeze <4 x i1> poison
// CHECK: [[SV:%.*]] = shufflevector <4 x i1> [[V]], <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
// CHECK: [[BC:%.*]] = bitcast <8 x i1> [[SV]] to i8
// CHECK: store i8 [[BC]], ptr [[A]]
// CHECK: ret void
  bool4 x = __builtin_nondeterministic_value(x);
}
