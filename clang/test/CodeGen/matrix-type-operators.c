// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
typedef float fx2x3_t __attribute__((matrix_type(2, 3)));
typedef int ix9x3_t __attribute__((matrix_type(9, 3)));
typedef unsigned long long ullx4x2_t __attribute__((matrix_type(4, 2)));

// Floating point matrix/scalar additions.

void add_matrix_matrix_double(dx5x5_t a, dx5x5_t b, dx5x5_t c) {
  // CHECK-LABEL: define void @add_matrix_matrix_double(<25 x double> %a, <25 x double> %b, <25 x double> %c)
  // CHECK:       [[B:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[C:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[RES:%.*]] = fadd <25 x double> [[B]], [[C]]
  // CHECK-NEXT:  store <25 x double> [[RES]], <25 x double>* {{.*}}, align 8

  a = b + c;
}

void add_matrix_matrix_float(fx2x3_t a, fx2x3_t b, fx2x3_t c) {
  // CHECK-LABEL: define void @add_matrix_matrix_float(<6 x float> %a, <6 x float> %b, <6 x float> %c)
  // CHECK:       [[B:%.*]] = load <6 x float>, <6 x float>* {{.*}}, align 4
  // CHECK-NEXT:  [[C:%.*]] = load <6 x float>, <6 x float>* {{.*}}, align 4
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[B]], [[C]]
  // CHECK-NEXT:  store <6 x float> [[RES]], <6 x float>* {{.*}}, align 4

  a = b + c;
}

void add_matrix_scalar_double_float(dx5x5_t a, float vf) {
  // CHECK-LABEL: define void @add_matrix_scalar_double_float(<25 x double> %a, float %vf)
  // CHECK:       [[MATRIX:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[SCALAR:%.*]] = load float, float* %vf.addr, align 4
  // CHECK-NEXT:  [[SCALAR_EXT:%.*]] = fpext float [[SCALAR]] to double
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <25 x double> undef, double [[SCALAR_EXT]], i32 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <25 x double> [[SCALAR_EMBED]], <25 x double> undef, <25 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <25 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <25 x double> [[RES]], <25 x double>* {{.*}}, align 8

  a = a + vf;
}

void add_matrix_scalar_double_double(dx5x5_t a, double vd) {
  // CHECK-LABEL: define void @add_matrix_scalar_double_double(<25 x double> %a, double %vd)
  // CHECK:       [[MATRIX:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[SCALAR:%.*]] = load double, double* %vd.addr, align 8
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <25 x double> undef, double [[SCALAR]], i32 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <25 x double> [[SCALAR_EMBED]], <25 x double> undef, <25 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <25 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <25 x double> [[RES]], <25 x double>* {{.*}}, align 8

  a = a + vd;
}

void add_matrix_scalar_float_float(fx2x3_t b, float vf) {
  // CHECK-LABEL: define void @add_matrix_scalar_float_float(<6 x float> %b, float %vf)
  // CHECK:       [[MATRIX:%.*]] = load <6 x float>, <6 x float>* {{.*}}, align 4
  // CHECK-NEXT:  [[SCALAR:%.*]] = load float, float* %vf.addr, align 4
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> undef, float [[SCALAR]], i32 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> undef, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], <6 x float>* {{.*}}, align 4

  b = b + vf;
}

void add_matrix_scalar_float_double(fx2x3_t b, double vd) {
  // CHECK-LABEL: define void @add_matrix_scalar_float_double(<6 x float> %b, double %vd)
  // CHECK:       [[MATRIX:%.*]] = load <6 x float>, <6 x float>* {{.*}}, align 4
  // CHECK-NEXT:  [[SCALAR:%.*]] = load double, double* %vd.addr, align 8
  // CHECK-NEXT:  [[SCALAR_TRUNC:%.*]] = fptrunc double [[SCALAR]] to float
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <6 x float> undef, float [[SCALAR_TRUNC]], i32 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <6 x float> [[SCALAR_EMBED]], <6 x float> undef, <6 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <6 x float> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:  store <6 x float> [[RES]], <6 x float>* {{.*}}, align 4

  b = b + vd;
}

// Integer matrix/scalar additions

void add_matrix_matrix_int(ix9x3_t a, ix9x3_t b, ix9x3_t c) {
  // CHECK-LABEL: define void @add_matrix_matrix_int(<27 x i32> %a, <27 x i32> %b, <27 x i32> %c)
  // CHECK:       [[B:%.*]] = load <27 x i32>, <27 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  [[C:%.*]] = load <27 x i32>, <27 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  [[RES:%.*]] = add <27 x i32> [[B]], [[C]]
  // CHECK-NEXT:  store <27 x i32> [[RES]], <27 x i32>* {{.*}}, align 4
  a = b + c;
}

void add_matrix_matrix_unsigned_long_long(ullx4x2_t a, ullx4x2_t b, ullx4x2_t c) {
  // CHECK-LABEL: define void @add_matrix_matrix_unsigned_long_long(<8 x i64> %a, <8 x i64> %b, <8 x i64> %c)
  // CHECK:       [[B:%.*]] = load <8 x i64>, <8 x i64>* {{.*}}, align 8
  // CHECK-NEXT:  [[C:%.*]] = load <8 x i64>, <8 x i64>* {{.*}}, align 8
  // CHECK-NEXT:  [[RES:%.*]] = add <8 x i64> [[B]], [[C]]
  // CHECK-NEXT:  store <8 x i64> [[RES]], <8 x i64>* {{.*}}, align 8

  a = b + c;
}

void add_matrix_scalar_int_short(ix9x3_t a, short vs) {
  // CHECK-LABEL: define void @add_matrix_scalar_int_short(<27 x i32> %a, i16 signext %vs)
  // CHECK:        [[MATRIX:%.*]] = load <27 x i32>, <27 x i32>* [[MAT_ADDR:%.*]], align 4
  // CHECK-NEXT:   [[SCALAR:%.*]] = load i16, i16* %vs.addr, align 2
  // CHECK-NEXT:   [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i32
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <27 x i32> undef, i32 [[SCALAR_EXT]], i32 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <27 x i32> [[SCALAR_EMBED]], <27 x i32> undef, <27 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <27 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <27 x i32> [[RES]], <27 x i32>* [[MAT_ADDR]], align 4

  a = a + vs;
}

void add_matrix_scalar_int_long_int(ix9x3_t a, long int vli) {
  // CHECK-LABEL: define void @add_matrix_scalar_int_long_int(<27 x i32> %a, i64 %vli)
  // CHECK:        [[MATRIX:%.*]] = load <27 x i32>, <27 x i32>* [[MAT_ADDR:%.*]], align 4
  // CHECK-NEXT:   [[SCALAR:%.*]] = load i64, i64* %vli.addr, align 8
  // CHECK-NEXT:   [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <27 x i32> undef, i32 [[SCALAR_TRUNC]], i32 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <27 x i32> [[SCALAR_EMBED]], <27 x i32> undef, <27 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <27 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <27 x i32> [[RES]], <27 x i32>* [[MAT_ADDR]], align 4

  a = a + vli;
}

void add_matrix_scalar_int_unsigned_long_long(ix9x3_t a, unsigned long long int vulli) {
  // CHECK-LABEL: define void @add_matrix_scalar_int_unsigned_long_long(<27 x i32> %a, i64 %vulli)
  // CHECK:        [[MATRIX:%.*]] = load <27 x i32>, <27 x i32>* [[MAT_ADDR:%.*]], align 4
  // CHECK-NEXT:   [[SCALAR:%.*]] = load i64, i64* %vulli.addr, align 8
  // CHECK-NEXT:   [[SCALAR_TRUNC:%.*]] = trunc i64 [[SCALAR]] to i32
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <27 x i32> undef, i32 [[SCALAR_TRUNC]], i32 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <27 x i32> [[SCALAR_EMBED]], <27 x i32> undef, <27 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <27 x i32> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK-NEXT:   store <27 x i32> [[RES]], <27 x i32>* [[MAT_ADDR]], align 4

  a = a + vulli;
}

void add_matrix_scalar_long_long_int_short(ullx4x2_t b, short vs) {
  // CHECK-LABEL: define void @add_matrix_scalar_long_long_int_short(<8 x i64> %b, i16 signext %vs)
  // CHECK:         [[SCALAR:%.*]] = load i16, i16* %vs.addr, align 2
  // CHECK-NEXT:    [[SCALAR_EXT:%.*]] = sext i16 [[SCALAR]] to i64
  // CHECK-NEXT:    [[MATRIX:%.*]] = load <8 x i64>, <8 x i64>* {{.*}}, align 8
  // CHECK-NEXT:    [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> undef, i64 [[SCALAR_EXT]], i32 0
  // CHECK-NEXT:    [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> undef, <8 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = add <8 x i64> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK-NEXT:    store <8 x i64> [[RES]], <8 x i64>* {{.*}}, align 8

  b = vs + b;
}

void add_matrix_scalar_long_long_int_int(ullx4x2_t b, long int vli) {
  // CHECK-LABEL: define void @add_matrix_scalar_long_long_int_int(<8 x i64> %b, i64 %vli)
  // CHECK:         [[SCALAR:%.*]] = load i64, i64* %vli.addr, align 8
  // CHECK-NEXT:    [[MATRIX:%.*]] = load <8 x i64>, <8 x i64>* {{.*}}, align 8
  // CHECK-NEXT:    [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> undef, i64 [[SCALAR]], i32 0
  // CHECK-NEXT:    [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> undef, <8 x i32> zeroinitializer
  // CHECK-NEXT:    [[RES:%.*]] = add <8 x i64> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK-NEXT:    store <8 x i64> [[RES]], <8 x i64>* {{.*}}, align 8

  b = vli + b;
}

void add_matrix_scalar_long_long_int_unsigned_long_long(ullx4x2_t b, unsigned long long int vulli) {
  // CHECK-LABEL: define void @add_matrix_scalar_long_long_int_unsigned_long_long
  // CHECK:        [[SCALAR:%.*]] = load i64, i64* %vulli.addr, align 8
  // CHECK-NEXT:   [[MATRIX:%.*]] = load <8 x i64>, <8 x i64>* %0, align 8
  // CHECK-NEXT:   [[SCALAR_EMBED:%.*]] = insertelement <8 x i64> undef, i64 [[SCALAR]], i32 0
  // CHECK-NEXT:   [[SCALAR_EMBED1:%.*]] = shufflevector <8 x i64> [[SCALAR_EMBED]], <8 x i64> undef, <8 x i32> zeroinitializer
  // CHECK-NEXT:   [[RES:%.*]] = add <8 x i64> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK-NEXT:   store <8 x i64> [[RES]], <8 x i64>* {{.*}}, align 8
  b = vulli + b;
}
