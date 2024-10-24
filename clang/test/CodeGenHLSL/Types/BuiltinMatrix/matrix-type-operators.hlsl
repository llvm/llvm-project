// RUN: %clang_cc1 -O0 -triple spirv-unknown-vulkan-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s --check-prefixes=CHECK,NOOPT
// RUN: %clang_cc1 -O1 -triple spirv-unknown-vulkan-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s --check-prefixes=CHECK,OPT
// RUIN: %clang_cc1 -O0 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s --check-prefixes=CHECK,NOOPT
// RUIN: %clang_cc1 -O1 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s --check-prefixes=CHECK,OPT

// Test arithmetic operations on matrix types.
// This is adapted to HLSL from CodeGen/matrix-type-operators.c.

// Floating point matrix/scalar additions.

// CHECK-LABEL: define {{.*}}add_matrix_matrix_double
void add_matrix_matrix_double() {
double4x4 a;
double4x4 b;
double4x4 c;
  // NOOPT:       [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:  [[C:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[C:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fadd <16 x double> [[B]], [[C]]
  // CHECK-NEXT:  store <16 x double> [[RES]], ptr {{.*}}, align 8

  a = b + c;
}

// CHECK-LABEL: define {{.*}}add_compound_assign_matrix_double
void add_compound_assign_matrix_double() {
double4x4 a;
double4x4 b;
  // NOOPT:       [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fadd <16 x double> [[A]], [[B]]
  // CHECK-NEXT:  store <16 x double> [[RES]], ptr {{.*}}, align 8

  a += b;
}

// CHECK-LABEL: define {{.*}}subtract_compound_assign_matrix_double
void subtract_compound_assign_matrix_double() {
double4x4 a;
double4x4 b;
  // NOOPT:       [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fsub <16 x double> [[A]], [[B]]
  // CHECK-NEXT:  store <16 x double> [[RES]], ptr {{.*}}, align 8

  a -= b;
}

// CHECK-LABEL: define {{.*}}add_matrix_matrix_float
void add_matrix_matrix_float() {
float2x3 a;
float2x3 b;
float2x3 c;
  // NOOPT:       [[B:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[C:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[B:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[C:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[B]], [[C]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4

  a = b + c;
}

// CHECK-LABEL: define {{.*}}add_compound_assign_matrix_float
void add_compound_assign_matrix_float() {
float2x3 a;
float2x3 b;
  // NOOPT:       [[B:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[B:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[A]], [[B]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4

  a += b;
}

// CHECK-LABEL: define {{.*}}subtract_compound_assign_matrix_float
void subtract_compound_assign_matrix_float() {
float2x3 a;
float2x3 b;
  // NOOPT:       [[B:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[B:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fsub <6 x float> [[A]], [[B]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4

  a -= b;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_double_float
void add_matrix_scalar_double_float() {
double4x4 a;
float vf;
  // NOOPT:       [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:  [[SCALAR:%.*]] = load float, ptr {{.*}}, align 4{{$}}
  // OPT:         [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[SCALAR:%.*]] = load float, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = fpext float [[SCALAR]] to double
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <16 x double> poison, double [[SCALAR_EXT]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <16 x double> [[SCALAR_EMBED]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <16 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <16 x double> [[RES]], ptr {{.*}}, align 8

  a = a + vf;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_double_float
void add_compound_matrix_scalar_double_float() {
double4x4 a;
float vf;
  // NOOPT:  [[SCALAR:%.*]] = load float, ptr {{.*}}, align 4{{$}}
  // OPT:    [[SCALAR:%.*]] = load float, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = fpext float [[SCALAR]] to double
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <16 x double> poison, double [[SCALAR_EXT]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <16 x double> [[SCALAR_EMBED]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <16 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <16 x double> [[RES]], ptr {{.*}}, align 8

  a += vf;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_double_float
void subtract_compound_matrix_scalar_double_float() {
double4x4 a;
float vf;
  // NOOPT:  [[SCALAR:%.*]] = load float, ptr %vf, align 4{{$}}
  // OPT:    [[SCALAR:%.*]] = load float, ptr %vf, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = fpext float [[SCALAR]] to double
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <16 x double> poison, double [[SCALAR_EXT]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <16 x double> [[SCALAR_EMBED]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <16 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <16 x double> [[RES]], ptr {{.*}}, align 8

  a -= vf;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_double_double
void add_matrix_scalar_double_double() {
double4x4 a;
double vd;
  // NOOPT:       [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:  [[SCALAR:%.*]] = load double, ptr %vd, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[SCALAR:%.*]] = load double, ptr %vd, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <16 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <16 x double> [[SCALAR_EMBED]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <16 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <16 x double> [[RES]], ptr {{.*}}, align 8

  a = a + vd;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_double_double
void add_compound_matrix_scalar_double_double() {
double4x4 a;
double vd;
  // NOOPT:       [[SCALAR:%.*]] = load double, ptr %vd, align 8{{$}}
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[SCALAR:%.*]] = load double, ptr %vd, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <16 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <16 x double> [[SCALAR_EMBED]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <16 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // store <16 x double> [[RES]], ptr {{.*}}, align 8
  a += vd;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_double_double
void subtract_compound_matrix_scalar_double_double() {
double4x4 a;
double vd;
  // NOOPT:       [[SCALAR:%.*]] = load double, ptr %vd, align 8{{$}}
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[SCALAR:%.*]] = load double, ptr %vd, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <16 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <16 x double> [[SCALAR_EMBED]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <16 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // store <16 x double> [[RES]], ptr {{.*}}, align 8
  a -= vd;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_float_float
void add_matrix_scalar_float_float() {
float2x3 b;
float vf;
  // NOOPT:       [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[SCALAR:%.*]] = load float, ptr %vf, align 4{{$}}
  // OPT:         [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[SCALAR:%.*]] = load float, ptr %vf, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> poison, float [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4

  b = b + vf;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_float_float
void add_compound_matrix_scalar_float_float() {
float2x3 b;
float vf;
  // NOOPT:       [[SCALAR:%.*]] = load float, ptr %vf, align 4{{$}}
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <6 x float>, ptr %b, align 4{{$}}
  // OPT:         [[SCALAR:%.*]] = load float, ptr %vf, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <6 x float>, ptr %b, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> poison, float [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4
  b += vf;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_float_float
void subtract_compound_matrix_scalar_float_float() {
float2x3 b;
float vf;
  // NOOPT:       [[SCALAR:%.*]] = load float, ptr %vf, align 4{{$}}
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <6 x float>, ptr %b, align 4{{$}}
  // OPT:         [[SCALAR:%.*]] = load float, ptr %vf, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <6 x float>, ptr %b, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> poison, float [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4
  b -= vf;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_float_double
void add_matrix_scalar_float_double() {
float2x3 b;
double vd;
  // NOOPT:       [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[SCALAR:%.*]] = load double, ptr %vd, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[SCALAR:%.*]] = load double, ptr %vd, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_TRUNC:%.*]] = fptrunc double [[SCALAR]] to float
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> poison, float [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4

  b = b + vd;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_float_double
void add_compound_matrix_scalar_float_double() {
float2x3 b;
double vd;
  // NOOPT:       [[SCALAR:%.*]] = load double, ptr %vd, align 8{{$}}
  // OPT:         [[SCALAR:%.*]] = load double, ptr %vd, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_TRUNC:%.*]] = fptrunc double [[SCALAR]] to float
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> poison, float [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4
  b += vd;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_float_double
void subtract_compound_matrix_scalar_float_double() {
float2x3 b;
double vd;
  // NOOPT:       [[SCALAR:%.*]] = load double, ptr %vd, align 8{{$}}
  // OPT:         [[SCALAR:%.*]] = load double, ptr %vd, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_TRUNC:%.*]] = fptrunc double [[SCALAR]] to float
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <6 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> poison, float [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], ptr {{.*}}, align 4
  b -= vd;
}

// Integer matrix/scalar additions

// CHECK-LABEL: define {{.*}}add_matrix_matrix_int
void add_matrix_matrix_int() {
int4x3 a;
int4x3 b;
int4x3 c;
  // NOOPT:       [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[C:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[C:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = add <12 x i32> [[B]], [[C]]
  // CHECK-NEXT:  store <12 x i32> [[RES]], ptr {{.*}}, align 4
  a = b + c;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_matrix_int
void add_compound_matrix_matrix_int() {
int4x3 a;
int4x3 b;
  // NOOPT:       [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = add <12 x i32> [[A]], [[B]]
  // CHECK-NEXT:  store <12 x i32> [[RES]], ptr {{.*}}, align 4
  a += b;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_matrix_int
void subtract_compound_matrix_matrix_int() {
int4x3 a;
int4x3 b;
  // NOOPT:       [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = sub <12 x i32> [[A]], [[B]]
  // CHECK-NEXT:  store <12 x i32> [[RES]], ptr {{.*}}, align 4
  a -= b;
}

// CHECK-LABEL: define {{.*}}add_matrix_matrix_uint64
void add_matrix_matrix_uint64() {
uint64_t4x2 a;
uint64_t4x2 b;
uint64_t4x2 c;
  // NOOPT:       [[B:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:  [[C:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[B:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[C:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = add <8 x i64> [[B]], [[C]]
  // CHECK-NEXT:  store <8 x i64> [[RES]], ptr {{.*}}, align 8

  a = b + c;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_matrix_uint64
void add_compound_matrix_matrix_uint64() {
uint64_t4x2 a;
uint64_t4x2 b;
  // NOOPT:       [[B:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[B:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = add <8 x i64> [[A]], [[B]]
  // CHECK-NEXT:  store <8 x i64> [[RES]], ptr {{.*}}, align 8

  a += b;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_matrix_uint64
void subtract_compound_matrix_matrix_uint64() {
uint64_t4x2 a;
uint64_t4x2 b;
  // NOOPT:       [[B:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[B:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // NOOPT-NEXT:  [[A:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT-NEXT:    [[A:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = sub <8 x i64> [[A]], [[B]]
  // CHECK-NEXT:  store <8 x i64> [[RES]], ptr {{.*}}, align 8

  a -= b;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_int_int16
void add_matrix_scalar_int_int16() {
int4x3 a;
int16_t vs;
  // NOOPT:        [[MATRIX:%.*]] = load <12 x i32>, ptr [[MAT_ADDR:%.*]], align 4{{$}}
  // NOOPT-NEXT:   [[SCALAR:%.*]] = load i16, ptr %vs, align 2{{$}}
  // OPT:          [[MATRIX:%.*]] = load <12 x i32>, ptr [[MAT_ADDR:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[SCALAR:%.*]] = load i16, ptr %vs, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i32
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_EXT]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a = a + vs;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_int_int16
void add_compound_matrix_scalar_int_int16() {
int4x3 a;
int16_t vs;
  // NOOPT:       [[SCALAR:%.*]] = load i16, ptr %vs, align 2{{$}}
  // OPT:         [[SCALAR:%.*]] = load i16, ptr %vs, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i32
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_EXT:%.*]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = add <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a += vs;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_int_int16
void subtract_compound_matrix_scalar_int_int16() {
int4x3 a;
int16_t
  vs;
  // NOOPT:       [[SCALAR:%.*]] = load i16, ptr %vs, align 2{{$}}
  // OPT:         [[SCALAR:%.*]] = load i16, ptr %vs, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i32
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_EXT:%.*]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = sub <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a -= vs;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_int_int64
void add_matrix_scalar_int_int64() {
int4x3 a;
int64_t vli;
  // NOOPT:        [[MATRIX:%.*]] = load <12 x i32>, ptr [[MAT_ADDR:%.*]], align 4{{$}}
  // NOOPT-NEXT:   [[SCALAR:%.*]] = load i64, ptr %vli, align 8{{$}}
  // OPT:          [[MATRIX:%.*]] = load <12 x i32>, ptr [[MAT_ADDR:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[SCALAR:%.*]] = load i64, ptr %vli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a = a + vli;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_int_int64
void add_compound_matrix_scalar_int_int64() {
int4x3 a;
int64_t vli;
  // NOOPT:       [[SCALAR:%.*]] = load i64, ptr %vli, align 8{{$}}
  // OPT:         [[SCALAR:%.*]] = load i64, ptr %vli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = add <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a += vli;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_int_int64
void subtract_compound_matrix_scalar_int_int64() {
int4x3 a;
int64_t vli;
  // NOOPT:       [[SCALAR:%.*]] = load i64, ptr %vli, align 8{{$}}
  // OPT:         [[SCALAR:%.*]] = load i64, ptr %vli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <12 x i32>, ptr %a, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = sub <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a -= vli;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_int_uint64
void add_matrix_scalar_int_uint64() {
int4x3 a;
uint64_t vulli;
  // NOOPT:        [[MATRIX:%.*]] = load <12 x i32>, ptr [[MAT_ADDR:%.*]], align 4{{$}}
  // NOOPT-NEXT:   [[SCALAR:%.*]] = load i64, ptr %vulli, align 8{{$}}
  // OPT:          [[MATRIX:%.*]] = load <12 x i32>, ptr [[MAT_ADDR:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[SCALAR:%.*]] = load i64, ptr %vulli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a = a + vulli;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_int_uint64
void add_compound_matrix_scalar_int_uint64() {
int4x3 a;
uint64_t vulli;
  // NOOPT:        [[SCALAR:%.*]] = load i64, ptr %vulli, align 8{{$}}
  // OPT:          [[SCALAR:%.*]] = load i64, ptr %vulli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // NOOPT-NEXT:   [[MATRIX:%.*]] = load <12 x i32>, ptr [[MATRIX_ADDR:%.*]], align 4{{$}}
  // OPT-NEXT:     [[MATRIX:%.*]] = load <12 x i32>, ptr [[MATRIX_ADDR:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a += vulli;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_int_uint64
void subtract_compound_matrix_scalar_int_uint64() {
int4x3 a;
uint64_t vulli;
  // NOOPT:        [[SCALAR:%.*]] = load i64, ptr %vulli, align 8{{$}}
  // OPT:          [[SCALAR:%.*]] = load i64, ptr %vulli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // NOOPT-NEXT:   [[MATRIX:%.*]] = load <12 x i32>, ptr [[MATRIX_ADDR:%.*]], align 4{{$}}
  // OPT-NEXT:     [[MATRIX:%.*]] = load <12 x i32>, ptr [[MATRIX_ADDR:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <12 x i32> poison, i32 [[SCALAR_TRUNC]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <12 x i32> [[SCALAR_EMBED]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = sub <12 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <12 x i32> [[RES]], ptr [[MAT_ADDR]], align 4

  a -= vulli;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_uint64_short
void add_matrix_scalar_uint64_short() {
uint64_t4x2 b;
int16_t vs;
  // NOOPT:         [[SCALAR:%.*]] = load i16, ptr %vs, align 2{{$}}
  // OPT:           [[SCALAR:%.*]] = load i16, ptr %vs, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i64
  // NOOPT-NEXT:    [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT-NEXT:      [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR_EXT]], i64 0
  // CHECK-NEXT:    [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = add <8 x i64> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK-NEXT:    store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b = vs + b;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_uint64_short
void add_compound_matrix_scalar_uint64_short() {
uint64_t4x2 b;
int16_t vs;
  // NOOPT:       [[SCALAR:%.*]] = load i16, ptr %vs, align 2{{$}}
  // OPT:         [[SCALAR:%.*]] = load i16, ptr %vs, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i64
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR_EXT]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = add <8 x i64> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b += vs;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_uint64_short
void subtract_compound_matrix_scalar_uint64_short() {
uint64_t4x2 b;
int16_t vs;
  // NOOPT:       [[SCALAR:%.*]] = load i16, ptr %vs, align 2{{$}}
  // OPT:         [[SCALAR:%.*]] = load i16, ptr %vs, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i64
  // NOOPT-NEXT:  [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8{{$}}
  // OPT-NEXT:    [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR_EXT]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = sub <8 x i64> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b -= vs;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_uint64_int
void add_matrix_scalar_uint64_int() {
uint64_t4x2 b;
int64_t vli;
  // NOOPT:         [[SCALAR:%.*]] = load i64, ptr %vli, align 8{{$}}
  // NOOPT-NEXT:    [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT:           [[SCALAR:%.*]] = load i64, ptr %vli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR]], i64 0
  // CHECK-NEXT:    [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = add <8 x i64> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK-NEXT:    store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b = vli + b;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_uint64_int
void add_compound_matrix_scalar_uint64_int() {
uint64_t4x2 b;
int64_t vli;
  // NOOPT:        [[SCALAR:%.*]] = load i64, ptr %vli, align 8{{$}}
  // NOOPT-NEXT:   [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT:          [[SCALAR:%.*]] = load i64, ptr %vli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <8 x i64> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b += vli;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_uint64_int
void subtract_compound_matrix_scalar_uint64_int() {
uint64_t4x2 b;
int64_t vli;
  // NOOPT:        [[SCALAR:%.*]] = load i64, ptr %vli, align 8{{$}}
  // OPT:          [[SCALAR:%.*]] = load i64, ptr %vli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // NOOPT-NEXT:   [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8{{$}}
  // OPT-NEXT:     [[MATRIX:%.*]] = load <8 x i64>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = sub <8 x i64> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b -= vli;
}

// CHECK-LABEL: define {{.*}}add_matrix_scalar_uint64_uint64
void add_matrix_scalar_uint64_uint64() {
uint64_t4x2 b;
uint64_t vulli;
  // NOOPT:        [[SCALAR:%.*]] = load i64, ptr %vulli, align 8{{$}}
  // NOOPT-NEXT:   [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8{{$}}
  // OPT:          [[SCALAR:%.*]] = load i64, ptr %vulli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <8 x i64> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK-NEXT:   store <8 x i64> [[RES]], ptr {{.*}}, align 8
  b = vulli + b;
}

// CHECK-LABEL: define {{.*}}add_compound_matrix_scalar_uint64_uint64
void add_compound_matrix_scalar_uint64_uint64() {
uint64_t4x2 b;
uint64_t vulli;
  // NOOPT:        [[SCALAR:%.*]] = load i64, ptr %vulli, align 8{{$}}
  // NOOPT-NEXT:   [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8{{$}}
  // OPT:          [[SCALAR:%.*]] = load i64, ptr %vulli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <8 x i64> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b += vulli;
}

// CHECK-LABEL: define {{.*}}subtract_compound_matrix_scalar_uint64_uint64
void subtract_compound_matrix_scalar_uint64_uint64() {
uint64_t4x2 b;
uint64_t vulli;
  // NOOPT:        [[SCALAR:%.*]] = load i64, ptr %vulli, align 8{{$}}
  // NOOPT-NEXT:   [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8{{$}}
  // OPT:          [[SCALAR:%.*]] = load i64, ptr %vulli, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[MATRIX:%.*]] = load <8 x i64>, ptr %b, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> poison, i64 [[SCALAR]], i64 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = sub <8 x i64> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <8 x i64> [[RES]], ptr {{.*}}, align 8

  b -= vulli;
}

// Tests for matrix multiplication.

// CHECK-LABEL: define {{.*}}multiply_matrix_matrix_double
void multiply_matrix_matrix_double() {
double4x4 b;
double4x4 c;
  // NOOPT:         [[B:%.*]] = load <16 x double>, ptr %b, align 8{{$}}
  // NOOPT-NEXT:    [[C:%.*]] = load <16 x double>, ptr %c, align 8{{$}}
  // OPT:           [[B:%.*]] = load <16 x double>, ptr %b, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[C:%.*]] = load <16 x double>, ptr %c, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = call <16 x double> @llvm.matrix.multiply.v16f64.v16f64.v16f64(<16 x double> [[B]], <16 x double> [[C]], i32 4, i32 4, i32 4)
  // CHECK-NEXT:    store <16 x double> [[RES]], ptr %a, align 8
  // OPT-NEXT:     call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // OPT-NEXT:     call void @llvm.lifetime.end.p0(i64 128, ptr %c)
  // OPT-NEXT:     call void @llvm.lifetime.end.p0(i64 128, ptr %b)
  // CHECK-NEXT:    ret void

  double4x4 a;
  a = b * c;
}

// CHECK-LABEL: define {{.*}}multiply_compound_matrix_matrix_double
void multiply_compound_matrix_matrix_double() {
double4x4 b;
double4x4 c;
  // NOOPT:        [[C:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:   [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:          [[C:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:     [[B:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[RES:%.*]] = call <16 x double> @llvm.matrix.multiply.v16f64.v16f64.v16f64(<16 x double> [[B]], <16 x double> [[C]], i32 4, i32 4, i32 4)
  // CHECK-NEXT:   store <16 x double> [[RES]], ptr {{.*}}, align 8
  // OPT-NEXT:     call void @llvm.lifetime.end.p0(i64 128, ptr %c)
  // OPT-NEXT:     call void @llvm.lifetime.end.p0(i64 128, ptr %b)
  // CHECK-NEXT:   ret void
  b *= c;
}

// CHECK-LABEL: define {{.*}}multiply_matrix_matrix_int
void multiply_matrix_matrix_int() {
int4x3 b;
int3x4 c;
  // NOOPT:         [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // NOOPT-NEXT:    [[C:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[B:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[C:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = call <16 x i32> @llvm.matrix.multiply.v16i32.v12i32.v12i32(<12 x i32> [[B]], <12 x i32> [[C]], i32 4, i32 3, i32 4)
  // CHECK-NEXT:    store <16 x i32> [[RES]], ptr %a, align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 64, ptr %a)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %c)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %b)
  // CHECK:         ret void
  int4x4 a;
  a = b * c;
}

// CHECK-LABEL: define {{.*}}multiply_double_matrix_scalar_float
void multiply_double_matrix_scalar_float() {
double4x4 a;
float s;
  // NOOPT:         [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load float, ptr %s, align 4{{$}}
  // OPT:           [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load float, ptr %s, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_EXT:%.*]] = fpext float [[S]] to double
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <16 x double> poison, double [[S_EXT]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <16 x double> [[VECINSERT]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fmul <16 x double> [[A]], [[VECSPLAT]]
  // CHECK-NEXT:    store <16 x double> [[RES]], ptr {{.*}}, align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void
  a = a * s;
}

// CHECK-LABEL: define {{.*}}multiply_compound_double_matrix_scalar_float
void multiply_compound_double_matrix_scalar_float() {
double4x4 a;
float s;
  // NOOPT:         [[S:%.*]] = load float, ptr %s, align 4{{$}}
  // OPT:           [[S:%.*]] = load float, ptr %s, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_EXT:%.*]] = fpext float [[S]] to double
  // NOOPT-NEXT:    [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT-NEXT:      [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <16 x double> poison, double [[S_EXT]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <16 x double> [[VECINSERT]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fmul <16 x double> [[A]], [[VECSPLAT]]
  // CHECK-NEXT:    store <16 x double> [[RES]], ptr {{.*}}, align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void
  a *= s;
}

// CHECK-LABEL: define {{.*}}multiply_double_matrix_scalar_double
void multiply_double_matrix_scalar_double() {
double4x4 a;
double s;
  // NOOPT:         [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load double, ptr %s, align 8{{$}}
  // OPT:           [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load double, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <16 x double> poison, double [[S]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <16 x double> [[VECINSERT]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fmul <16 x double> [[A]], [[VECSPLAT]]
  // CHECK-NEXT:    store <16 x double> [[RES]], ptr {{.*}}, align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void
  a = a * s;
}

// CHECK-LABEL: define {{.*}}multiply_compound_double_matrix_scalar_double
void multiply_compound_double_matrix_scalar_double() {
double4x4 a;
double s;
  // NOOPT:         [[S:%.*]] = load double, ptr %s, align 8{{$}}
  // NOOPT-NEXT:    [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:           [[S:%.*]] = load double, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <16 x double> poison, double [[S]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <16 x double> [[VECINSERT]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fmul <16 x double> [[A]], [[VECSPLAT]]
  // CHECK-NEXT:    store <16 x double> [[RES]], ptr {{.*}}, align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void
  a *= s;
}

// CHECK-LABEL: define {{.*}}multiply_float_matrix_scalar_double
void multiply_float_matrix_scalar_double() {
float2x3 b;
double s;
  // NOOPT:         [[S:%.*]] = load double, ptr %s, align 8{{$}}
  // OPT:           [[S:%.*]] = load double, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_TRUNC:%.*]] = fptrunc double [[S]] to float
  // NOOPT-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <6 x float> poison, float [[S_TRUNC]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <6 x float> [[VECINSERT]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fmul <6 x float> [[VECSPLAT]], [[MAT]]
  // CHECK-NEXT:    store <6 x float> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %b)
  // CHECK-NEXT:    ret void
  b = s * b;
}

// CHECK-LABEL: define {{.*}}multiply_compound_float_matrix_scalar_double
void multiply_compound_float_matrix_scalar_double() {
float2x3 b;
double s;
  // NOOPT:         [[S:%.*]] = load double, ptr %s, align 8{{$}}
  // OPT:           [[S:%.*]] = load double, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_TRUNC:%.*]] = fptrunc double [[S]] to float
  // NOOPT-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <6 x float> poison, float [[S_TRUNC]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <6 x float> [[VECINSERT]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fmul <6 x float> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:    store <6 x float> %3, ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %b)
  // ret void
  b *= s;
}

// CHECK-LABEL: define {{.*}}multiply_int_matrix_scalar_int16
void multiply_int_matrix_scalar_int16() {
int4x3 b;
int16_t s;
  // NOOPT:         [[S:%.*]] = load i16, ptr %s, align 2{{$}}
  // OPT:           [[S:%.*]] = load i16, ptr %s, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_EXT:%.*]] = sext i16 [[S]] to i32
  // NOOPT-NEXT:    [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <12 x i32> poison, i32 [[S_EXT]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <12 x i32> [[VECINSERT]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = mul <12 x i32> [[VECSPLAT]], [[MAT]]
  // CHECK-NEXT:    store <12 x i32> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 2, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %b)
  // CHECK-NEXT:    ret void
  b = s * b;
}

// CHECK-LABEL: define {{.*}}multiply_compound_int_matrix_scalar_int16
void multiply_compound_int_matrix_scalar_int16() {
int4x3 b;
int16_t s;
  // NOOPT:        [[S:%.*]] = load i16, ptr %s, align 2{{$}}
  // OPT:          [[S:%.*]] = load i16, ptr %s, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[S_EXT:%.*]] = sext i16 [[S]] to i32
  // NOOPT-NEXT:   [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4{{$}}
  // OPT-NEXT:     [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[VECINSERT:%.*]] = insertelement <12 x i32> poison, i32 [[S_EXT]], i64 0
  // CHECK-NEXT:   [[VECSPLAT:%.*]] = shufflevector <12 x i32> [[VECINSERT]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = mul <12 x i32> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:   store <12 x i32> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 2, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %b)
  // CHECK-NEXT:   ret void
  b *= s;
}

// CHECK-LABEL: define {{.*}}multiply_int_matrix_scalar_ull
void multiply_int_matrix_scalar_ull() {
int4x3 b;
uint64_t s;
  // NOOPT:         [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load i64, ptr %s, align 8{{$}}
  // OPT-NEXT:      [[S:%.*]] = load i64, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_TRUNC:%.*]] = trunc i64 [[S]] to i32
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <12 x i32> poison, i32 [[S_TRUNC]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <12 x i32> [[VECINSERT]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = mul <12 x i32> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:    store <12 x i32> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %b)
  // CHECK-NEXT:    ret void
  b = b * s;
}

// CHECK-LABEL: define {{.*}}multiply_compound_int_matrix_scalar_ull
void multiply_compound_int_matrix_scalar_ull() {
int4x3 b;
uint64_t s;
  // NOOPT:         [[S:%.*]] = load i64, ptr %s, align 8{{$}}
  // OPT:           [[S:%.*]] = load i64, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_TRUNC:%.*]] = trunc i64 [[S]] to i32
  // NOOPT-NEXT:    [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <12 x i32> poison, i32 [[S_TRUNC]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <12 x i32> [[VECINSERT]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = mul <12 x i32> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:    store <12 x i32> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %b)
  // CHECK-NEXT:    ret void

  b *= s;
}

// CHECK-LABEL: define {{.*}}multiply_float_matrix_constant
void multiply_float_matrix_constant() {
float2x3 a;
  // CHECK:         [[A_ADDR:%.*]] = alloca [6 x float], align 4
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 24, ptr %a)
  // NOOPT-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[A_ADDR]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <6 x float>, ptr [[A_ADDR]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = fmul <6 x float> [[MAT]], <float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00>
  // CHECK-NEXT:    store <6 x float> [[RES]], ptr [[A_ADDR]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %a)
  // CHECK-NEXT:    ret void
  a = a * 2.5;
}

// CHECK-LABEL: define {{.*}}multiply_compound_float_matrix_constant
void multiply_compound_float_matrix_constant() {
float2x3 a;
  // CHECK:         [[A_ADDR:%.*]] = alloca [6 x float], align 4
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 24, ptr %a)
  // NOOPT-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[A_ADDR]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <6 x float>, ptr [[A_ADDR]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = fmul <6 x float> [[MAT]], <float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00>
  // CHECK-NEXT:    store <6 x float> [[RES]], ptr [[A_ADDR]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %a)
  // CHECK-NEXT:    ret void
  a *= 2.5;
}

// CHECK-LABEL: define {{.*}}multiply_int_matrix_constant
void multiply_int_matrix_constant() {
int4x3 a;
  // CHECK:         [[A_ADDR:%.*]] = alloca [12 x i32], align 4
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 48, ptr %a)
  // NOOPT-NEXT:    [[MAT:%.*]] = load <12 x i32>, ptr [[A_ADDR]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <12 x i32>, ptr [[A_ADDR]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = mul <12 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>, [[MAT]]
  // CHECK-NEXT:    store <12 x i32> [[RES]], ptr [[A_ADDR]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %a)
  // CHECK-NEXT:    ret void
  a = 5 * a;
}

// CHECK-LABEL: define {{.*}}multiply_compound_int_matrix_constant
void multiply_compound_int_matrix_constant() {
int4x3 a;
  // CHECK:         [[A_ADDR:%.*]] = alloca [12 x i32], align 4
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 48, ptr %a)
  // NOOPT-NEXT:    [[MAT:%.*]] = load <12 x i32>, ptr [[A_ADDR]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <12 x i32>, ptr [[A_ADDR]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = mul <12 x i32> [[MAT]], <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  // CHECK-NEXT:    store <12 x i32> [[RES]], ptr [[A_ADDR]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %a)
  // CHECK-NEXT:    ret void
  a *= 5;
}

// CHECK-LABEL: define {{.*}}divide_double_matrix_scalar_float
void divide_double_matrix_scalar_float() {
double4x4 a;
float s;
  // NOOPT:         [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load float, ptr %s, align 4{{$}}
  // OPT:           [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load float, ptr %s, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_EXT:%.*]] = fpext float [[S]] to double
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <16 x double> poison, double [[S_EXT]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <16 x double> [[VECINSERT]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fdiv <16 x double> [[A]], [[VECSPLAT]]
  // CHECK-NEXT:    store <16 x double> [[RES]], ptr {{.*}}, align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void
  a = a / s;
}

// CHECK-LABEL: define {{.*}}divide_double_matrix_scalar_double
void divide_double_matrix_scalar_double() {
double4x4 a;
double s;
  // NOOPT:         [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load double, ptr %s, align 8{{$}}
  // OPT:           [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load double, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <16 x double> poison, double [[S]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <16 x double> [[VECINSERT]], <16 x double> poison, <16 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fdiv <16 x double> [[A]], [[VECSPLAT]]
  // CHECK-NEXT:    store <16 x double> [[RES]], ptr {{.*}}, align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void
  a = a / s;
}

// CHECK-LABEL: define {{.*}}divide_float_matrix_scalar_double
void divide_float_matrix_scalar_double() {
float2x3 b;
double s;
  // NOOPT:         [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load double, ptr %s, align 8{{$}}
  // OPT:           [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load double, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_TRUNC:%.*]] = fptrunc double [[S]] to float
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <6 x float> poison, float [[S_TRUNC]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <6 x float> [[VECINSERT]], <6 x float> poison, <6 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = fdiv <6 x float> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:    store <6 x float> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %b)
  // CHECK-NEXT:    ret void
  b = b / s;
}

// CHECK-LABEL: define {{.*}}divide_int_matrix_scalar_int16
void divide_int_matrix_scalar_int16() {
int4x3 b;
int16_t s;
  // NOOPT:         [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load i16, ptr %s, align 2{{$}}
  // OPT:           [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load i16, ptr %s, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_EXT:%.*]] = sext i16 [[S]] to i32
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <12 x i32> poison, i32 [[S_EXT]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <12 x i32> [[VECINSERT]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = sdiv <12 x i32> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:    store <12 x i32> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 2, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %b)
  // CHECK-NEXT:    ret void
  b = b / s;
}

// CHECK-LABEL: define {{.*}}divide_int_matrix_scalar_ull
void divide_int_matrix_scalar_ull() {
int4x3 b;
uint64_t s;
  // NOOPT:         [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load i64, ptr %s, align 8{{$}}
  // OPT:           [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load i64, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[S_TRUNC:%.*]] = trunc i64 [[S]] to i32
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <12 x i32> poison, i32 [[S_TRUNC]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <12 x i32> [[VECINSERT]], <12 x i32> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = sdiv <12 x i32> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:    store <12 x i32> [[RES]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %b)
  // CHECK-NEXT:    ret void
  b = b / s;
}

// CHECK-LABEL: define {{.*}}divide_ull_matrix_scalar_ull
void divide_ull_matrix_scalar_ull() {
uint64_t4x2 b;
uint64_t s;
  // NOOPT:         [[MAT:%.*]] = load <8 x i64>, ptr [[B:%.*]], align 8{{$}}
  // NOOPT-NEXT:    [[S:%.*]] = load i64, ptr %s, align 8{{$}}
  // OPT:           [[MAT:%.*]] = load <8 x i64>, ptr [[B:%.*]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[S:%.*]] = load i64, ptr %s, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <8 x i64> poison, i64 [[S]], i64 0
  // CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <8 x i64> [[VECINSERT]], <8 x i64> poison, <8 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = udiv <8 x i64> [[MAT]], [[VECSPLAT]]
  // CHECK-NEXT:    store <8 x i64> [[RES]], ptr [[B]], align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %s)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 64, ptr %b)
  // CHECK-NEXT:    ret void
  b = b / s;
}

// CHECK-LABEL: define {{.*}}divide_float_matrix_constant
void divide_float_matrix_constant() {
float2x3 a;
  // CHECK:         [[A_ADDR:%.*]] = alloca [6 x float], align 4
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 24, ptr %a)
  // NOOPT-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[A_ADDR]], align 4{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <6 x float>, ptr [[A_ADDR]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = fdiv <6 x float> [[MAT]], <float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00>
  // CHECK-NEXT:    store <6 x float> [[RES]], ptr [[A_ADDR]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %a)
  // CHECK-NEXT:    ret void
  a = a / 2.5;
}

  // Tests for the matrix type operators.

  // Check that we can use matrix index expression on different floating point
  // matrixes and indices.
// CHECK-LABEL: define {{.*}}insert_double_matrix_const_idx_ll_u_double
void insert_double_matrix_const_idx_ll_u_double() {
double4x4 a;
double d;
float2x3 b;
float e;
int j;
uint k;
  // NOOPT:         [[D:%.*]] = load double, ptr %d, align 8{{$}}
  // OPT:           [[D:%.*]] = load double, ptr %d, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MAT:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <16 x double> [[MAT]], double [[D]], i64 4
  // CHECK-NEXT:    store <16 x double> [[MATINS]], ptr {{.*}}, align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %k)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %j)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %e)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %b)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %d)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void

  a[0ll][1u] = d;
}

// CHECK-LABEL: define {{.*}}insert_double_matrix_const_idx_i_u_double
void insert_double_matrix_const_idx_i_u_double() {
double4x4 a;
double d;
  // NOOPT:         [[D:%.*]] = load double, ptr %d, align 8{{$}}
  // OPT:           [[D:%.*]] = load double, ptr %d, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MAT:%.*]] = load <16 x double>, ptr [[B:%.*]], align 8{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <16 x double> [[MAT]], double [[D]], i64 13
  // CHECK-NEXT:    store <16 x double> [[MATINS]], ptr [[B]], align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %d)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void

  a[1][3u] = d;
}

// CHECK-LABEL: define {{.*}}insert_float_matrix_const_idx_ull_i_float
void insert_float_matrix_const_idx_ull_i_float() {
float2x3 b;
float e;
  // NOOPT:         [[E:%.*]] = load float, ptr %e, align 4{{$}}
  // OPT:           [[E:%.*]] = load float, ptr %e, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <6 x float> [[MAT]], float [[E]], i64 3
  // CHECK-NEXT:    store <6 x float> [[MATINS]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %e)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %b)
  // CHECK-NEXT:    ret void

  b[1ull][1] = e;
}

// CHECK-LABEL: define {{.*}}insert_float_matrix_idx_i_u_float
void insert_float_matrix_idx_i_u_float() {
float2x3 b;
float e;
int j;
uint k;
  // NOOPT:         [[E:%.*]] = load float, ptr %e, align 4{{$}}
  // NOOPT-NEXT:    [[J:%.*]] = load i32, ptr %j, align 4{{$}}
  // OPT:           [[E:%.*]] = load float, ptr %e, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i32, ptr %j, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[J_EXT:%.*]] = sext i32 [[J]] to i64
  // NOOPT-NEXT:    [[K:%.*]] = load i32, ptr %k, align 4{{$}}
  // OPT-NEXT:      [[K:%.*]] = load i32, ptr %k, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[K_EXT:%.*]] = zext i32 [[K]] to i64
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[K_EXT]], 2
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], [[J_EXT]]
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 6
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <6 x float> [[MAT]], float [[E]], i64 [[IDX2]]
  // CHECK-NEXT:    store <6 x float> [[MATINS]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %k)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %j)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %e)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %b)
  // CHECK-NEXT:    ret void

  b[j][k] = e;
}

// CHECK-LABEL: define {{.*}}insert_float_matrix_idx_s_ull_float
void insert_float_matrix_idx_s_ull_float() {
float2x3 b;
float e;
int16_t j;
uint64_t k;
  // NOOPT:         [[E:%.*]] = load float, ptr %e, align 4{{$}}
  // NOOPT-NEXT:    [[J:%.*]] = load i16, ptr %j, align 2{{$}}
  // OPT:           [[E:%.*]] = load float, ptr %e, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i16, ptr %j, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[J_EXT:%.*]] = sext i16 [[J]] to i64
  // NOOPT-NEXT:    [[K:%.*]] = load i64, ptr %k, align 8{{$}}
  // OPT-NEXT:      [[K:%.*]] = load i64, ptr %k, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[K]], 2
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], [[J_EXT]]
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 6
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr [[B:%.*]], align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <6 x float> [[MAT]], float [[E]], i64 [[IDX2]]
  // CHECK-NEXT:    store <6 x float> [[MATINS]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %k)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 2, ptr %j)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %e)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %b)
  // CHECK-NEXT:    ret void

  (b)[j][k] = e;
}

  // Check that we can can use matrix index expressions on integer matrixes.
// CHECK-LABEL: define {{.*}}insert_int_idx_expr
void insert_int_idx_expr() {
int4x3 a;
int i;
  // NOOPT:         [[I1:%.*]] = load i32, ptr %i, align 4{{$}}
  // NOOPT-NEXT:    [[I2:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT:           [[I1:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[I2:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[I2_ADD:%.*]] = add nsw i32 4, [[I2]]
  // CHECK-NEXT:    [[ADD_EXT:%.*]] = sext i32 [[I2_ADD]] to i64
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 8, [[ADD_EXT]]
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 12
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT:%.*]] = load <12 x i32>, ptr [[B:%.*]], align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <12 x i32> [[MAT]], i32 [[I1]], i64 [[IDX2]]
  // CHECK-NEXT:    store <12 x i32> [[MATINS]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %i)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %a)
  // CHECK-NEXT:    ret void

  a[4 + i][1 + 1u] = i;
}

  // Check that we can can use matrix index expressions on FP and integer
  // matrixes.
// CHECK-LABEL: define {{.*}}insert_float_into_int_matrix
void insert_float_into_int_matrix() {
int4x3 a;
int i;
  // NOOPT:         [[I:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT:           [[I:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MAT:%.*]] = load <12 x i32>, ptr [[MAT_ADDR:%.*]], align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <12 x i32> [[MAT]], i32 [[I]], i64 7
  // CHECK-NEXT:    store <12 x i32> [[MATINS]], ptr [[MAT_ADDR]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %i)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %a)
  // CHECK-NEXT:    ret void

  a[3][1] = i;
}

  // Check that we can use overloaded matrix index expressions on matrixes with
  // matching dimensions, but different element types.
// CHECK-LABEL: define {{.*}}insert_matching_dimensions1
void insert_matching_dimensions1() {
double3x3 a;
double i;
  // NOOPT:         [[I:%.*]] = load double, ptr %i, align 8{{$}}
  // OPT:           [[I:%.*]] = load double, ptr %i, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MAT:%.*]] = load <9 x double>, ptr [[B:%.*]], align 8{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <9 x double> [[MAT]], double [[I]], i64 5
  // CHECK-NEXT:    store <9 x double> [[MATINS]], ptr [[B]], align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %i)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 72, ptr %a)
  // CHECK-NEXT:    ret void

  a[2u][1u] = i;
}

// CHECK-LABEL: define {{.*}}insert_matching_dimensions
void insert_matching_dimensions() {
float3x3 b;
float e;
  // NOOPT:         [[E:%.*]] = load float, ptr %e, align 4{{$}}
  // OPT:           [[E:%.*]] = load float, ptr %e, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MAT:%.*]] = load <9 x float>, ptr [[B:%.*]], align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <9 x float> [[MAT]], float [[E]], i64 7
  // CHECK-NEXT:    store <9 x float> [[MATINS]], ptr [[B]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %e)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 36, ptr %b)
  // CHECK-NEXT:    ret void

  b[1u][2u] = e;
}

// CHECK-LABEL: define {{.*}}extract_double
double extract_double() {
double4x4 a;
  // NOOPT:         [[MAT:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:           [[MAT:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <16 x double> [[MAT]], i64 10
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret double [[MATEXT]]

  return a[2][3 - 1u];
}

// CHECK-LABEL: define {{.*}}extract_float
double extract_float() {
float3x3 b;
  // NOOPT:         [[MAT:%.*]] = load <9 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <9 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <9 x float> [[MAT]], i64 5
  // CHECK-NEXT:    [[TO_DOUBLE:%.*]] = fpext float [[MATEXT]] to double
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 36, ptr %b)
  // CHECK-NEXT:    ret double [[TO_DOUBLE]]

  return b[2][1];
}

// CHECK-LABEL: define {{.*}}extract_int
int extract_int() {
int4x3 c;
uint64_t j;
  // NOOPT:         [[J1:%.*]] = load i64, ptr %j, align 8{{$}}
  // NOOPT-NEXT:    [[J2:%.*]] = load i64, ptr %j, align 8{{$}}
  // OPT:           [[J1:%.*]] = load i64, ptr %j, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[J2:%.*]] = load i64, ptr %j, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[J2]], 4
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], [[J1]]
  // NOOPT-NEXT:    [[MAT:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 12
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // OPT-NEXT:      [[MAT:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <12 x i32> [[MAT]], i64 [[IDX2]]
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %j)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 48, ptr %c)
  // CHECK-NEXT:    ret i32 [[MATEXT]]

  return c[j][j];
}

// CHECK-LABEL: define {{.*}}test_extract_matrix_pointer1
double test_extract_matrix_pointer1() {
double3x2 ptr[3][3];
uint j;
  // NOOPT:         [[J:%.*]] = load i32, ptr %j, align 4{{$}}
  // OPT:           [[J:%.*]] = load i32, ptr %j, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[J_EXT:%.*]] = zext i32 [[J]] to i64
  // CHECK-NEXT:    [[IDX:%.*]] = add i64 3, [[J_EXT]]
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX]], 6
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[ARIX:%.*]] = getelementptr inbounds [3 x [3 x [6 x double]]], ptr %ptr, i64 0, i64 1
  // CHECK-NEXT:    [[ARIX1:%.*]] = getelementptr inbounds [3 x [6 x double]], ptr [[ARIX]], i64 0, i64 2
  // NOOPT-NEXT:    [[MAT:%.*]] = load <6 x double>, ptr [[ARIX1]], align 8{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <6 x double>, ptr [[ARIX1]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <6 x double> [[MAT]], i64 [[IDX]]
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %j)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 432, ptr %ptr)
  // CHECK-NEXT:    ret double [[MATEXT]]

  return ptr[1][2][j][1];
}

// CHECK-LABEL: define {{.*}}test_extract_matrix_pointer2
double test_extract_matrix_pointer2() {
double3x2 ptr[7][7];
  // CHECK:         [[ARIX:%.*]] = getelementptr inbounds [7 x [7 x [6 x double]]], ptr %ptr, i64 0, i64 4
  // CHECK-NEXT:    [[ARIX1:%.*]] = getelementptr inbounds [7 x [6 x double]], ptr [[ARIX]], i64 0, i64 6
  // NOOPT:         [[MAT:%.*]] = load <6 x double>, ptr [[ARIX1]], align 8{{$}}
  // OPT:           [[MAT:%.*]] = load <6 x double>, ptr [[ARIX1]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <6 x double> [[MAT]], i64 5
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 2352, ptr %ptr)
  // CHECK-NEXT:    ret double [[MATEXT]]

  return ptr[4][6][2][1 * 3 - 2];
}

// CHECK-LABEL: define {{.*}}insert_extract
void insert_extract() {
double4x4 a;
float3x3 b;
uint64_t j;
int16_t k;
  // NOOPT:         [[K:%.*]] = load i16, ptr %k, align 2{{$}}
  // OPT:           [[K:%.*]] = load i16, ptr %k, align 2, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[K_EXT:%.*]] = sext i16 [[K]] to i64
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[K_EXT]], 3
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], 0
  // NOOPT-NEXT:    [[MAT:%.*]] = load <9 x float>, ptr [[MAT_ADDR:%.*]], align 4{{$}}
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 9
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // OPT-NEXT:      [[MAT:%.*]] = load <9 x float>, ptr [[MAT_ADDR:%.*]], align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <9 x float> [[MAT]], i64 [[IDX2]]
  // NOOPT-NEXT:    [[J:%.*]] = load i64, ptr %j, align 8{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i64, ptr %j, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[IDX3:%.*]] = mul i64 [[J]], 3
  // CHECK-NEXT:    [[IDX4:%.*]] = add i64 [[IDX3]], 2
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX4]], 9
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT2:%.*]] = load <9 x float>, ptr [[MAT_ADDR]], align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <9 x float> [[MAT2]], float [[MATEXT]], i64 [[IDX4]]
  // CHECK-NEXT:    store <9 x float> [[MATINS]], ptr [[MAT_ADDR]], align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 2, ptr %k)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 8, ptr %j)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 36, ptr %b)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  // CHECK-NEXT:    ret void

  b[2][j] = b[0][k];
}

// CHECK-LABEL: define {{.*}}insert_compound_stmt
void insert_compound_stmt() {
double4x4 a;
  // CHECK:        [[A:%.*]] = load <16 x double>, ptr [[A_PTR:%.*]], align 8{{$}}
  // CHECK-NEXT:   [[EXT:%.*]] = extractelement <16 x double> [[A]], i64 14
  // CHECK-NEXT:   [[SUB:%.*]] = fsub double [[EXT]], 1.000000e+00
  // CHECK-NEXT:   [[A2:%.*]] = load <16 x double>, ptr [[A_PTR]], align 8{{$}}
  // CHECK-NEXT:   [[INS:%.*]] = insertelement <16 x double> [[A2]], double [[SUB]], i64 14
  // CHECK-NEXT:   store <16 x double> [[INS]], ptr [[A_PTR]], align 8
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 128, ptr %a) #5
  // CHECK-NEXT:   ret void

  a[2][3] -= 1.0;
}

struct Foo {
  float2x3 mat;
};

// CHECK-LABEL: define {{.*}}insert_compound_stmt_field
void insert_compound_stmt_field() {
struct Foo a;
float f;
uint i;
uint j;
  // NOOPT:         [[I:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT:           [[I:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[I_EXT:%.*]] = zext i32 [[I]] to i64
  // NOOPT-NEXT:    [[J:%.*]] = load i32, ptr %j, align 4{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i32, ptr %j, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[J_EXT:%.*]] = zext i32 [[J]] to i64
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[J_EXT]], 2
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], [[I_EXT]]
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 6
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT:%.*]] = load <6 x float>, ptr %mat, align 4{{$}}
  // CHECK-NEXT:    [[EXT:%.*]] = extractelement <6 x float> [[MAT]], i64 [[IDX2]]
  // CHECK-NEXT:    [[SUM:%.*]] = fadd float [[EXT]], {{.*}}
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 6
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT2:%.*]] = load <6 x float>, ptr %mat, align 4{{$}}
  // CHECK-NEXT:    [[INS:%.*]] = insertelement <6 x float> [[MAT2]], float [[SUM]], i64 [[IDX2]]
  // CHECK-NEXT:    store <6 x float> [[INS]], ptr %mat, align 4
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %j)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %i)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 4, ptr %f)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 24, ptr %a)
  // CHECK-NEXT:    ret void

  a.mat[i][j] += f;
}

// CHECK-LABEL: define {{.*}}matrix_as_idx
void matrix_as_idx() {
int4x3 a;
int i;
int j;
double4x4 b;
  // NOOPT:       [[I1:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT:         [[I1:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[I1_EXT:%.*]] = sext i32 [[I1]] to i64
  // NOOPT-NEXT:  [[J1:%.*]] = load i32, ptr %j, align 4{{$}}
  // OPT-NEXT:    [[J1:%.*]] = load i32, ptr %j, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[J1_EXT:%.*]] = sext i32 [[J1]] to i64
  // CHECK-NEXT:  [[IDX1_1:%.*]] = mul i64 [[J1_EXT]], 4
  // CHECK-NEXT:  [[IDX1_2:%.*]] = add i64 [[IDX1_1]], [[I1_EXT]]
  // NOOPT-NEXT:  [[A:%.*]] = load <12 x i32>, ptr %a, align 4{{$}}
  // OPT-NEXT:    [[CMP:%.*]] = icmp ult i64 [[IDX1_2]], 12
  // OPT-NEXT:    call void @llvm.assume(i1 [[CMP]])
  // OPT-NEXT:    [[A:%.*]] = load <12 x i32>, ptr %a, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[MI1:%.*]] = extractelement <12 x i32> [[A]], i64 [[IDX1_2]]
  // CHECK-NEXT:  [[MI1_EXT:%.*]] = sext i32 [[MI1]] to i64
  // NOOPT-NEXT:  [[J2:%.*]] = load i32, ptr %j, align 4{{$}}
  // OPT-NEXT:    [[J2:%.*]] = load i32, ptr %j, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[J2_EXT:%.*]] = sext i32 [[J2]] to i64
  // NOOPT-NEXT:  [[I2:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT-NEXT:    [[I2:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[I2_EXT:%.*]] = sext i32 [[I2]] to i64
  // CHECK-NEXT:  [[IDX2_1:%.*]] = mul i64 [[I2_EXT]], 4
  // CHECK-NEXT:  [[IDX2_2:%.*]] = add i64 [[IDX2_1]], [[J2_EXT]]
  // NOOPT-NEXT:  [[A2:%.*]] = load <12 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT-NEXT:    [[CMP:%.*]] = icmp ult i64 [[IDX2_2]], 12
  // OPT-NEXT:    call void @llvm.assume(i1 [[CMP]])
  // OPT-NEXT:    [[A2:%.*]] = load <12 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[MI2:%.*]] = extractelement <12 x i32> [[A2]], i64 [[IDX2_2]]
  // CHECK-NEXT:  [[MI3:%.*]] = add nsw i32 [[MI2]], 2
  // CHECK-NEXT:  [[MI3_EXT:%.*]] = sext i32 [[MI3]] to i64
  // CHECK-NEXT:  [[IDX3_1:%.*]] = mul i64 [[MI3_EXT]], 4
  // CHECK-NEXT:  [[IDX3_2:%.*]] = add i64 [[IDX3_1]], [[MI1_EXT]]
  // OPT-NEXT:    [[CMP:%.*]] = icmp ult i64 [[IDX3_2]], 16
  // OPT-NEXT:    call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:  [[B:%.*]] = load <16 x double>, ptr [[B_PTR:%.*]], align 8{{$}}
  // CHECK-NEXT:  [[INS:%.*]] = insertelement <16 x double> [[B]], double 1.500000e+00, i64 [[IDX3_2]]
  // CHECK-NEXT:  store <16 x double> [[INS]], ptr [[B_PTR]], align 8
  // OPT-NEXT:    call void @llvm.lifetime.end.p0(i64 128, ptr %b) #5
  // OPT-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr %j) #5
  // OPT-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr %i) #5
  // OPT-NEXT:    call void @llvm.lifetime.end.p0(i64 48, ptr %a) #5
  // CHECK-NEXT:  ret void

  b[a[i][j]][a[j][i] + 2] = 1.5;
}


