// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

typedef char cx5x5 __attribute__((matrix_type(5, 5)));
typedef int ix5x5 __attribute__((matrix_type(5, 5)));
typedef short sx5x5 __attribute__((matrix_type(5, 5)));
typedef float fx5x5 __attribute__((matrix_type(5, 5)));
typedef double dx5x5 __attribute__((matrix_type(5, 5)));
typedef unsigned short unsigned_short_int_5x5 __attribute__((matrix_type(5, 5)));
typedef unsigned int unsigned_int_5x5 __attribute__((matrix_type(5, 5)));
typedef unsigned long unsigned_long_int_5x5 __attribute__((matrix_type(5, 5)));

void cast_char_matrix_to_int(cx5x5 c, ix5x5 i) {
  // CHECK-LABEL: define{{.*}} void @cast_char_matrix_to_int(<25 x i8> %c, <25 x i32> %i)
  // CHECK:       [[C:%.*]] = load <25 x i8>, <25 x i8>* {{.*}}, align 1
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (ix5x5)c;
}

void cast_char_matrix_to_unsigned_int(cx5x5 c, unsigned_int_5x5 u) {
  // CHECK-LABEL: define{{.*}} void @cast_char_matrix_to_unsigned_int(<25 x i8> %c, <25 x i32> %u)
  // CHECK:       [[C:%.*]] = load <25 x i8>, <25 x i8>* {{.*}}, align 1
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  u = (unsigned_int_5x5)c;
}

void cast_unsigned_long_int_matrix_to_short(unsigned_long_int_5x5 u, sx5x5 s) {
  // CHECK-LABEL: define{{.*}} void @cast_unsigned_long_int_matrix_to_short(<25 x i64> %u, <25 x i16> %s)
  // CHECK:       [[U:%.*]] = load <25 x i64>, <25 x i64>* {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i64> [[U]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], <25 x i16>* {{.*}}, align 2
  // CHECK-NEXT:  ret void

  s = (sx5x5)u;
}

void cast_int_matrix_to_short(ix5x5 i, sx5x5 s) {
  // CHECK-LABEL: define{{.*}} void @cast_int_matrix_to_short(<25 x i32> %i, <25 x i16> %s)
  // CHECK:       [[I:%.*]] = load <25 x i32>, <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i32> [[I]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], <25 x i16>* {{.*}}, align 2
  // CHECK-NEXT:  ret void

  s = (sx5x5)i;
}

void cast_int_matrix_to_float(ix5x5 i, fx5x5 f) {
  // CHECK-LABEL: define{{.*}} void @cast_int_matrix_to_float(<25 x i32> %i, <25 x float> %f)
  // CHECK:       [[I:%.*]] = load <25 x i32>, <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sitofp <25 x i32> [[I]] to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], <25 x float>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  f = (fx5x5)i;
}

void cast_unsigned_int_matrix_to_float(unsigned_short_int_5x5 u, fx5x5 f) {
  // CHECK-LABEL: define{{.*}} void @cast_unsigned_int_matrix_to_float(<25 x i16> %u, <25 x float> %f)
  // CHECK:       [[U:%.*]] = load <25 x i16>, <25 x i16>* {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = uitofp <25 x i16> [[U]] to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], <25 x float>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  f = (fx5x5)u;
}

void cast_double_matrix_to_int(dx5x5 d, ix5x5 i) {
  // CHECK-LABEL: define{{.*}} void @cast_double_matrix_to_int(<25 x double> %d, <25 x i32> %i)
  // CHECK:       [[D:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptosi <25 x double> [[D]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (ix5x5)d;
}

void cast_float_matrix_to_unsigned_short_int(fx5x5 f, unsigned_short_int_5x5 i) {
  // CHECK-LABEL: define{{.*}} void @cast_float_matrix_to_unsigned_short_int(<25 x float> %f, <25 x i16> %i)
  // CHECK:       [[F:%.*]] = load <25 x float>, <25 x float>* {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = fptoui <25 x float> [[F]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], <25 x i16>* %1, align 2
  // CHECK-NEXT:  ret void

  i = (unsigned_short_int_5x5)f;
}

void cast_double_matrix_to_float(dx5x5 d, fx5x5 f) {
  // CHECK-LABEL: define{{.*}} void @cast_double_matrix_to_float(<25 x double> %d, <25 x float> %f)
  // CHECK:       [[D:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptrunc <25 x double> [[D]] to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], <25 x float>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  f = (fx5x5)d;
}

void cast_unsigned_short_int_to_unsigned_int(unsigned_short_int_5x5 s, unsigned_int_5x5 i) {
  // CHECK-LABEL: define{{.*}} void @cast_unsigned_short_int_to_unsigned_int(<25 x i16> %s, <25 x i32> %i)
  // CHECK:       [[S:%.*]] = load <25 x i16>, <25 x i16>* {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[S]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (unsigned_int_5x5)s;
}

void cast_unsigned_long_int_to_unsigned_short_int(unsigned_long_int_5x5 l, unsigned_short_int_5x5 s) {
  // CHECK-LABEL: define{{.*}} void @cast_unsigned_long_int_to_unsigned_short_int(<25 x i64> %l, <25 x i16> %s)
  // CHECK:       [[L:%.*]] = load <25 x i64>, <25 x i64>* %0, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i64> [[L]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], <25 x i16>* {{.*}}, align 2
  // CHECK-NEXT:  ret void

  s = (unsigned_short_int_5x5)l;
}

void cast_unsigned_short_int_to_int(unsigned_short_int_5x5 u, ix5x5 i) {
  // CHECK-LABEL: define{{.*}} void @cast_unsigned_short_int_to_int(<25 x i16> %u, <25 x i32> %i)
  // CHECK:       [[U:%.*]] = load <25 x i16>, <25 x i16>* %0, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[U]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (ix5x5)u;
}

void cast_int_to_unsigned_long_int(ix5x5 i, unsigned_long_int_5x5 u) {
  // CHECK-LABEL: define{{.*}} void @cast_int_to_unsigned_long_int(<25 x i32> %i, <25 x i64> %u)
  // CHECK:       [[I:%.*]] = load <25 x i32>, <25 x i32>* %0, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i32> [[I]] to <25 x i64>
  // CHECK-NEXT:  store <25 x i64> [[CONV]], <25 x i64>* {{.*}}, align 8
  // CHECK-NEXT:  ret void

  u = (unsigned_long_int_5x5)i;
}
