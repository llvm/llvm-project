// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file %t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=OGCG --input-file %t.ll

void test_signbit_positive_zero(){
  double positiveZero = +0.0;
  int result = __builtin_signbit(positiveZero);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["positiveZero", init]
// CIR: cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double 0.000000e+00, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double 0.000000e+00, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}

void test_signbit_negative_zero(){
  double negativeZero = -0.0;
  int result = __builtin_signbit(negativeZero);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["negativeZero", init]
// CIR: cir.const #cir.fp<-0.000000e+00> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double -0.000000e+00, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double -0.000000e+00, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}

void test_signbit_positive_number(){
  double positiveNumber = 1.0;
  int result = __builtin_signbit(positiveNumber);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["positiveNumber", init]
// CIR: cir.const #cir.fp<1.000000e+00> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double 1.000000e+00, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double 1.000000e+00, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}

void test_signbit_negative_number(){
  double negativeNumber = -1.0;
  int result = __builtin_signbit(negativeNumber);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["negativeNumber", init]
// CIR: cir.const #cir.fp<-1.000000e+00> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double -1.000000e+00, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double -1.000000e+00, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}

void test_signbit_positive_nan(){
  double positiveNan = +__builtin_nan("");
  int result = __builtin_signbit(positiveNan);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["positiveNan", init]
// CIR: cir.const #cir.fp<0x7FF8000000000000> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double 0x7FF8000000000000, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double 0x7FF8000000000000, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}

void test_signbit_negative_nan(){
  double negativeNan = -__builtin_nan("");
  int result = __builtin_signbit(negativeNan);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["negativeNan", init]
// CIR: cir.const #cir.fp<0xFFF8000000000000> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double 0xFFF8000000000000, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double 0xFFF8000000000000, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}

void test_signbit_positive_infinity(){
  double positiveInfinity = +__builtin_inf();
  int result = __builtin_signbit(positiveInfinity);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["positiveInfinity", init]
// CIR: cir.const #cir.fp<0x7FF0000000000000> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double 0x7FF0000000000000, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double 0x7FF0000000000000, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}

void test_signbit_negative_infinity(){
  double negativeInfinity = -__builtin_inf();
  int result = __builtin_signbit(negativeInfinity);
// CIR: cir.alloca !cir.double, !cir.ptr<!cir.double>, ["negativeInfinity", init]
// CIR: cir.const #cir.fp<0xFFF0000000000000> : !cir.double
// CIR: cir.signbit {{.*}} : !cir.double -> !cir.bool
// CIR: cir.cast bool_to_int {{.*}} : !cir.bool -> !s32i

// LLVM: store double 0xFFF0000000000000, ptr %{{.*}}
// LLVM: bitcast double %{{.*}} to i64
// LLVM: icmp slt i64 %{{.*}}, 0
// LLVM: zext i1 %{{.*}} to i32

// OGCG: store double 0xFFF0000000000000, ptr %{{.*}}
// OGCG: bitcast double %{{.*}} to i64
// OGCG: icmp slt i64 %{{.*}}, 0
// OGCG: zext i1 %{{.*}} to i32
}
