// RUN: %clang_cc1 -ffast-math -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
typedef float fx2x3_t __attribute__((matrix_type(2, 3)));
typedef int ix9x3_t __attribute__((matrix_type(9, 3)));
typedef unsigned long long ullx4x2_t __attribute__((matrix_type(4, 2)));

// Floating point matrix/scalar additions.

void add_matrix_matrix_double(dx5x5_t a, dx5x5_t b, dx5x5_t c) {
  // CHECK-LABEL: define{{.*}} void @add_matrix_matrix_double(<25 x double> noundef %a, <25 x double> noundef %b, <25 x double> noundef %c)
  // CHECK:       [[B:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[C:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[RES:%.*]] = fadd reassoc nnan ninf nsz arcp afn <25 x double> [[B]], [[C]]
  // CHECK-NEXT:  store <25 x double> [[RES]], ptr {{.*}}, align 8

  a = b + c;
}

void add_compound_assign_matrix_double(dx5x5_t a, dx5x5_t b) {
  // CHECK-LABEL: define{{.*}} void @add_compound_assign_matrix_double(<25 x double> noundef %a, <25 x double> noundef %b)
  // CHECK:       [[B:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[A:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[RES:%.*]] = fadd reassoc nnan ninf nsz arcp afn <25 x double> [[A]], [[B]]
  // CHECK-NEXT:  store <25 x double> [[RES]], ptr {{.*}}, align 8

  a += b;
}

void subtract_compound_assign_matrix_double(dx5x5_t a, dx5x5_t b) {
  // CHECK-LABEL: define{{.*}} void @subtract_compound_assign_matrix_double(<25 x double> noundef %a, <25 x double> noundef %b)
  // CHECK:       [[B:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[A:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[RES:%.*]] = fsub reassoc nnan ninf nsz arcp afn <25 x double> [[A]], [[B]]
  // CHECK-NEXT:  store <25 x double> [[RES]], ptr {{.*}}, align 8

  a -= b;
}

void add_matrix_scalar_double_float(dx5x5_t a, float vf) {
  // CHECK-LABEL: define{{.*}} void @add_matrix_scalar_double_float(<25 x double> noundef %a, float noundef %vf)
  // CHECK:       [[MATRIX:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[SCALAR:%.*]] = load float, ptr %vf.addr, align 4
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = fpext float [[SCALAR]] to double
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <25 x double> poison, double [[SCALAR_EXT]], i32 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <25 x double> [[SCALAR_EMBED]], <25 x double> poison, <25 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd reassoc nnan ninf nsz arcp afn <25 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <25 x double> [[RES]], ptr {{.*}}, align 8

  a = a + vf;
}

void add_compound_matrix_scalar_double_float(dx5x5_t a, float vf) {
  // CHECK-LABEL: define{{.*}} void @add_compound_matrix_scalar_double_float(<25 x double> noundef %a, float noundef %vf)
  // CHECK:  [[SCALAR:%.*]] = load float, ptr %vf.addr, align 4
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = fpext float [[SCALAR]] to double
  // CHECK-NEXT:  [[MATRIX:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <25 x double> poison, double [[SCALAR_EXT]], i32 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <25 x double> [[SCALAR_EMBED]], <25 x double> poison, <25 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd reassoc nnan ninf nsz arcp afn <25 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <25 x double> [[RES]], ptr {{.*}}, align 8

  a += vf;
}

void subtract_compound_matrix_scalar_double_float(dx5x5_t a, float vf) {
  // CHECK-LABEL: define{{.*}} void @subtract_compound_matrix_scalar_double_float(<25 x double> noundef %a, float noundef %vf)
  // CHECK:  [[SCALAR:%.*]] = load float, ptr %vf.addr, align 4
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = fpext float [[SCALAR]] to double
  // CHECK-NEXT:  [[MATRIX:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <25 x double> poison, double [[SCALAR_EXT]], i32 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <25 x double> [[SCALAR_EMBED]], <25 x double> poison, <25 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub reassoc nnan ninf nsz arcp afn <25 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <25 x double> [[RES]], ptr {{.*}}, align 8

  a -= vf;
}

// Tests for matrix multiplication.

void multiply_matrix_matrix_double(dx5x5_t b, dx5x5_t c) {
  // CHECK-LABEL: @multiply_matrix_matrix_double(
  // CHECK:         [[B:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:    [[C:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:    [[RES:%.*]] = call reassoc nnan ninf nsz arcp afn <25 x double> @llvm.matrix.multiply.v25f64.v25f64.v25f64(<25 x double> [[B]], <25 x double> [[C]], i32 5, i32 5, i32 5)
  // CHECK-NEXT:    store <25 x double> [[RES]], ptr %a, align 8
  // CHECK-NEXT:    ret void
  //

  dx5x5_t a;
  a = b * c;
}

void multiply_compound_matrix_matrix_double(dx5x5_t b, dx5x5_t c) {
  // CHECK-LABEL: @multiply_compound_matrix_matrix_double(
  // CHECK:        [[C:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:   [[B:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:   [[RES:%.*]] = call reassoc nnan ninf nsz arcp afn <25 x double> @llvm.matrix.multiply.v25f64.v25f64.v25f64(<25 x double> [[B]], <25 x double> [[C]], i32 5, i32 5, i32 5)
  // CHECK-NEXT:   store <25 x double> [[RES]], ptr {{.*}}, align 8
  // CHECK-NEXT:   ret void
  b *= c;
}

// CHECK-LABEL: @multiply_double_matrix_scalar_float(
// CHECK:         [[A:%.*]] = load <25 x double>, ptr {{.*}}, align 8
// CHECK-NEXT:    [[S:%.*]] = load float, ptr %s.addr, align 4
// CHECK-NEXT:    [[S_EXT:%.*]] = fpext float [[S]] to double
// CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <25 x double> poison, double [[S_EXT]], i32 0
// CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <25 x double> [[VECINSERT]], <25 x double> poison, <25 x i32> zeroinitializer
// CHECK-NEXT:    [[RES:%.*]] = fmul reassoc nnan ninf nsz arcp afn <25 x double> [[A]], [[VECSPLAT]]
// CHECK-NEXT:    store <25 x double> [[RES]], ptr {{.*}}, align 8
// CHECK-NEXT:    ret void
//
void multiply_double_matrix_scalar_float(dx5x5_t a, float s) {
  a = a * s;
}

// CHECK-LABEL: @multiply_compound_double_matrix_scalar_float
// CHECK:         [[S:%.*]] = load float, ptr %s.addr, align 4
// CHECK-NEXT:    [[S_EXT:%.*]] = fpext float [[S]] to double
// CHECK-NEXT:    [[A:%.*]] = load <25 x double>, ptr {{.*}}, align 8
// CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <25 x double> poison, double [[S_EXT]], i32 0
// CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <25 x double> [[VECINSERT]], <25 x double> poison, <25 x i32> zeroinitializer
// CHECK-NEXT:    [[RES:%.*]] = fmul reassoc nnan ninf nsz arcp afn <25 x double> [[A]], [[VECSPLAT]]
// CHECK-NEXT:    store <25 x double> [[RES]], ptr {{.*}}, align 8
// CHECK-NEXT:    ret void
//
void multiply_compound_double_matrix_scalar_float(dx5x5_t a, float s) {
  a *= s;
}

// CHECK-LABEL: @divide_float_matrix_scalar_double(
// CHECK:         [[MAT:%.*]] = load <6 x float>, ptr [[MAT_ADDR:%.*]], align 4
// CHECK-NEXT:    [[S:%.*]] = load double, ptr %s.addr, align 8
// CHECK-NEXT:    [[S_TRUNC:%.*]] = fptrunc double [[S]] to float
// CHECK-NEXT:    [[VECINSERT:%.*]] = insertelement <6 x float> poison, float [[S_TRUNC]], i32 0
// CHECK-NEXT:    [[VECSPLAT:%.*]] = shufflevector <6 x float> [[VECINSERT]], <6 x float> poison, <6 x i32> zeroinitializer
// CHECK-NEXT:    [[RES:%.*]] = fdiv reassoc nnan ninf nsz arcp afn <6 x float> [[MAT]], [[VECSPLAT]]
// CHECK-NEXT:    store <6 x float> [[RES]], ptr [[MAT_ADDR]], align 4
// CHECK-NEXT:    ret void
//
void divide_float_matrix_scalar_double(fx2x3_t b, double s) {
  b = b / s;
}
