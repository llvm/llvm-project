// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file %t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=OGCG --input-file %t.ll

#define FP_NAN 3
#define FP_INFINITE 516
#define FP_ZERO 96
#define FP_SUBNORMAL 144
#define FP_NORMAL 264

void test_fpclassify_nan(){
    float nanValue = 0.0f / 0.0f;
    __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL,
                                         FP_SUBNORMAL, FP_ZERO, nanValue);
// CIR: %[[IS_ZERO:.+]] = cir.is_fp_class %{{.+}}, fcZero : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_ZERO]], true {
// CIR: cir.const #cir.int<96> : !s32i
// CIR: %[[IS_NAN:.+]] = cir.is_fp_class %{{.+}}, fcNan : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_NAN]], true {
// CIR: cir.const #cir.int<3> : !s32i
// CIR: %[[IS_INF:.+]] = cir.is_fp_class %{{.+}}, fcInf : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_INF]], true {
// CIR: cir.const #cir.int<516> : !s32i
// CIR: %[[IS_NORMAL:.+]] = cir.is_fp_class %{{.+}}, fcNormal : (!cir.float) -> !cir.bool
// CIR: %[[NORMAL_VAL:.+]] = cir.const #cir.int<264> : !s32i
// CIR: %[[SUBNORMAL_VAL:.+]] = cir.const #cir.int<144> : !s32i
// CIR: cir.select if %[[IS_NORMAL]] then %[[NORMAL_VAL]] else %[[SUBNORMAL_VAL]] : (!cir.bool, !s32i, !s32i) -> !s32i

// LLVM: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS1]]
// LLVM: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS2]]
// LLVM: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS3]]
// LLVM: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: phi i32

// OGCG: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS1]]
// OGCG: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS2]]
// OGCG: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS3]]
// OGCG: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: phi i32
}

void test_fpclassify_inf(){
    float infValue = 1.0f / 0.0f;
    __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL,
                                         FP_SUBNORMAL, FP_ZERO, infValue);
// CIR: %[[IS_ZERO:.+]] = cir.is_fp_class %{{.+}}, fcZero : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_ZERO]], true {
// CIR: cir.const #cir.int<96> : !s32i
// CIR: %[[IS_NAN:.+]] = cir.is_fp_class %{{.+}}, fcNan : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_NAN]], true {
// CIR: cir.const #cir.int<3> : !s32i
// CIR: %[[IS_INF:.+]] = cir.is_fp_class %{{.+}}, fcInf : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_INF]], true {
// CIR: cir.const #cir.int<516> : !s32i
// CIR: %[[IS_NORMAL:.+]] = cir.is_fp_class %{{.+}}, fcNormal : (!cir.float) -> !cir.bool
// CIR: %[[NORMAL_VAL:.+]] = cir.const #cir.int<264> : !s32i
// CIR: %[[SUBNORMAL_VAL:.+]] = cir.const #cir.int<144> : !s32i
// CIR: cir.select if %[[IS_NORMAL]] then %[[NORMAL_VAL]] else %[[SUBNORMAL_VAL]] : (!cir.bool, !s32i, !s32i) -> !s32i

// LLVM: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS1]]
// LLVM: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS2]]
// LLVM: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS3]]
// LLVM: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: phi i32

// OGCG: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS1]]
// OGCG: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS2]]
// OGCG: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS3]]
// OGCG: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: phi i32
}

void test_fpclassify_normal(){
    float normalValue = 1.0f;
    __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL,
                                            FP_SUBNORMAL, FP_ZERO, normalValue);
// CIR: %[[IS_ZERO:.+]] = cir.is_fp_class %{{.+}}, fcZero : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_ZERO]], true {
// CIR: cir.const #cir.int<96> : !s32i
// CIR: %[[IS_NAN:.+]] = cir.is_fp_class %{{.+}}, fcNan : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_NAN]], true {
// CIR: cir.const #cir.int<3> : !s32i
// CIR: %[[IS_INF:.+]] = cir.is_fp_class %{{.+}}, fcInf : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_INF]], true {
// CIR: cir.const #cir.int<516> : !s32i
// CIR: %[[IS_NORMAL:.+]] = cir.is_fp_class %{{.+}}, fcNormal : (!cir.float) -> !cir.bool
// CIR: %[[NORMAL_VAL:.+]] = cir.const #cir.int<264> : !s32i
// CIR: %[[SUBNORMAL_VAL:.+]] = cir.const #cir.int<144> : !s32i
// CIR: cir.select if %[[IS_NORMAL]] then %[[NORMAL_VAL]] else %[[SUBNORMAL_VAL]] : (!cir.bool, !s32i, !s32i) -> !s32i

// LLVM: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS1]]
// LLVM: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS2]]
// LLVM: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS3]]
// LLVM: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: phi i32

// OGCG: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS1]]
// OGCG: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS2]]
// OGCG: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS3]]
// OGCG: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: phi i32
}

void test_fpclassify_subnormal(){
    float subnormalValue = 1.0e-40f;
    __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL,
                                               FP_SUBNORMAL, FP_ZERO, subnormalValue);
// CIR: %[[IS_ZERO:.+]] = cir.is_fp_class %{{.+}}, fcZero : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_ZERO]], true {
// CIR: cir.const #cir.int<96> : !s32i
// CIR: %[[IS_NAN:.+]] = cir.is_fp_class %{{.+}}, fcNan : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_NAN]], true {
// CIR: cir.const #cir.int<3> : !s32i
// CIR: %[[IS_INF:.+]] = cir.is_fp_class %{{.+}}, fcInf : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_INF]], true {
// CIR: cir.const #cir.int<516> : !s32i
// CIR: %[[IS_NORMAL:.+]] = cir.is_fp_class %{{.+}}, fcNormal : (!cir.float) -> !cir.bool
// CIR: %[[NORMAL_VAL:.+]] = cir.const #cir.int<264> : !s32i
// CIR: %[[SUBNORMAL_VAL:.+]] = cir.const #cir.int<144> : !s32i
// CIR: cir.select if %[[IS_NORMAL]] then %[[NORMAL_VAL]] else %[[SUBNORMAL_VAL]] : (!cir.bool, !s32i, !s32i) -> !s32i

// LLVM: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS1]]
// LLVM: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS2]]
// LLVM: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS3]]
// LLVM: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: phi i32

// OGCG: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS1]]
// OGCG: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS2]]
// OGCG: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS3]]
// OGCG: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: phi i32
}

void test_fpclassify_zero(){
    float zeroValue = 0.0f;
    __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL,
                                          FP_SUBNORMAL, FP_ZERO, zeroValue);
// CIR: %[[IS_ZERO:.+]] = cir.is_fp_class %{{.+}}, fcZero : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_ZERO]], true {
// CIR: cir.const #cir.int<96> : !s32i
// CIR: %[[IS_NAN:.+]] = cir.is_fp_class %{{.+}}, fcNan : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_NAN]], true {
// CIR: cir.const #cir.int<3> : !s32i
// CIR: %[[IS_INF:.+]] = cir.is_fp_class %{{.+}}, fcInf : (!cir.float) -> !cir.bool
// CIR: cir.ternary(%[[IS_INF]], true {
// CIR: cir.const #cir.int<516> : !s32i
// CIR: %[[IS_NORMAL:.+]] = cir.is_fp_class %{{.+}}, fcNormal : (!cir.float) -> !cir.bool
// CIR: %[[NORMAL_VAL:.+]] = cir.const #cir.int<264> : !s32i
// CIR: %[[SUBNORMAL_VAL:.+]] = cir.const #cir.int<144> : !s32i
// CIR: cir.select if %[[IS_NORMAL]] then %[[NORMAL_VAL]] else %[[SUBNORMAL_VAL]] : (!cir.bool, !s32i, !s32i) -> !s32i

// LLVM: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS1]]
// LLVM: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS2]]
// LLVM: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: br i1 %[[FPCLASS3]]
// LLVM: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// LLVM: phi i32

// OGCG: %[[FPCLASS1:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL:.*]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS1]]
// OGCG: %[[FPCLASS2:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS2]]
// OGCG: %[[FPCLASS3:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: br i1 %[[FPCLASS3]]
// OGCG: %[[FPCLASS4:.*]] = call i1 @llvm.is.fpclass.f32(float %[[VAL]], i32 {{.*}})
// OGCG: phi i32
}
