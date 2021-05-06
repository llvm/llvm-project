// RUN: %clang_cc1 -std=c++11 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

template <typename X>
using matrix_4_4 = X __attribute__((matrix_type(4, 4)));

template <typename Y>
using matrix_5_5 = Y __attribute__((matrix_type(5, 5)));

// CHECK-LABEL: define{{.*}} void @_Z19CastCharMatrixToIntv
void CastCharMatrixToInt() {
  // CHECK: [[C:%.*]] = load <25 x i8>, <25 x i8>* {{.*}}, align 1
  // CHECK-NEXT: [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT: [[CONV1:%.*]] = bitcast [25 x i32]* {{.*}} to <25 x i32>*
  // CHECK-NEXT: store <25 x i32> [[CONV]], <25 x i32>* [[CONV1]], align 4

  matrix_5_5<char> c;
  matrix_5_5<int> i;
  i = (matrix_5_5<int>)c;
}

// CHECK-LABEL: define{{.*}} void @_Z27CastCharMatrixToUnsignedIntv
void CastCharMatrixToUnsignedInt() {
  // CHECK:       [[C:%.*]] = load <25 x i8>, <25 x i8>* {{.*}}, align 1
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i32]* {{.*}} to <25 x i32>*
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* [[CONV1]], align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<char> c;
  matrix_5_5<unsigned int> u;
  u = (matrix_5_5<unsigned int>)c;
}

// CHECK-LABEL: define{{.*}} void @_Z32CastUnsignedLongIntMatrixToShortv
void CastUnsignedLongIntMatrixToShort() {
  // CHECK:      [[U:%.*]] = load <25 x i64>, <25 x i64>* {{.*}}, align 8
  // CHECK-NEXT: [[CONV:%.*]] = trunc <25 x i64> {{.*}} to <25 x i16>
  // CHECK-NEXT: [[CONV1:%.*]] = bitcast [25 x i16]* {{.*}} to <25 x i16>*
  // CHECK-NEXT: store <25 x i16> [[CONV]], <25 x i16>* [[CONV1]], align 2
  // CHECK-NEXT: ret void

  matrix_5_5<unsigned long int> u;
  matrix_5_5<short int> s;
  s = (matrix_5_5<short int>)u;
}

// CHECK-LABEL: define{{.*}} void @_Z20CastIntMatrixToShortv()
void CastIntMatrixToShort() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i32> [[I]] to <25 x i16>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i16]* {{.*}} to <25 x i16>*
  // CHECK-NEXT:  store <25 x i16> [[CONV]], <25 x i16>* [[CONV1]], align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<short int> s;
  s = (matrix_5_5<short int>)i;
}

// CHECK-LABEL: define{{.*}} void @_Z20CastIntMatrixToFloatv()
void CastIntMatrixToFloat() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  [[CONV]] = sitofp <25 x i32> {{.*}} to <25 x float>
  // CHECK-NEXT:  [[CONV1]] = bitcast [25 x float]* {{.*}} to <25 x float>*
  // CHECK-NEXT:  store <25 x float> [[CONV]], <25 x float>* [[CONV1]], align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<float> f;
  f = (matrix_5_5<float>)i;
}

// CHECK-LABEL: define{{.*}} void @_Z28CastUnsignedIntMatrixToFloatv()
void CastUnsignedIntMatrixToFloat() {
  // CHECK:       [[U:%.*]] = load <25 x i16>, <25 x i16>* {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = uitofp <25 x i16> [[U]] to <25 x float>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x float]* {{.*}} to <25 x float>*
  // CHECK-NEXT:  store <25 x float> [[CONV]], <25 x float>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> u;
  matrix_5_5<float> f;
  f = (matrix_5_5<float>)u;
}

// CHECK-LABEL: define{{.*}} void @_Z21CastDoubleMatrixToIntv()
void CastDoubleMatrixToInt() {
  // CHECK:       [[D:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptosi <25 x double> [[D]] to <25 x i32>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i32]* %i to <25 x i32>*
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<double> d;
  matrix_5_5<int> i;
  i = (matrix_5_5<int>)d;
}

// CHECK-LABEL: define{{.*}} void @_Z33CastFloatMatrixToUnsignedShortIntv()
void CastFloatMatrixToUnsignedShortInt() {
  // CHECK:       [[F:%.*]] = load <25 x float>, <25 x float>* {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = fptoui <25 x float> [[F]] to <25 x i16>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i16]* {{.*}} to <25 x i16>*
  // CHECK-NEXT:  store <25 x i16> [[CONV]], <25 x i16>* [[CONV1]], align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<float> f;
  matrix_5_5<unsigned short int> i;
  i = (matrix_5_5<unsigned short int>)f;
}

// CHECK-LABEL: define{{.*}} void @_Z23CastDoubleMatrixToFloatv()
void CastDoubleMatrixToFloat() {
  // CHECK:       [[D:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptrunc <25 x double> [[D]] to <25 x float>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x float]* {{.*}} to <25 x float>*
  // CHECK-NEXT:  store <25 x float> [[CONV]], <25 x float>* [[CONV1]], align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<double> d;
  matrix_5_5<float> f;
  f = (matrix_5_5<float>)d;
}

// CHECK-LABEL: define{{.*}} void @_Z33CastUnsignedShortIntToUnsignedIntv()
void CastUnsignedShortIntToUnsignedInt() {
  // CHECK:       [[S:%.*]] = load <25 x i16>, <25 x i16>* {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[S]] to <25 x i32>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i32]* {{.*}} to <25 x i32>*
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* [[CONV1]], align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> s;
  matrix_5_5<unsigned int> i;
  i = (matrix_5_5<unsigned int>)s;
}

// CHECK-LABEL: define{{.*}} void @_Z37CastUnsignedLongIntToUnsignedShortIntv()
void CastUnsignedLongIntToUnsignedShortInt() {
  // CHECK:       [[L:%.*]] = load <25 x i64>, <25 x i64>* %0, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i64> [[L]] to <25 x i16>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i16]* {{.*}} to <25 x i16>*
  // CHECK-NEXT:  store <25 x i16> [[CONV]], <25 x i16>* [[CONV1]], align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned long int> l;
  matrix_5_5<unsigned short int> s;
  s = (matrix_5_5<unsigned short int>)l;
}

// CHECK-LABEL: define{{.*}} void @_Z25CastUnsignedShortIntToIntv()
void CastUnsignedShortIntToInt() {
  // CHECK:       [[U:%.*]] = load <25 x i16>, <25 x i16>* %0, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[U]] to <25 x i32>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i32]* {{.*}} to <25 x i32>*
  // CHECK-NEXT:  store <25 x i32> [[CONV]], <25 x i32>* {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> u;
  matrix_5_5<int> i;
  i = (matrix_5_5<int>)u;
}

// CHECK-LABEL: define{{.*}} void @_Z24CastIntToUnsignedLongIntv()
void CastIntToUnsignedLongInt() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, <25 x i32>* %0, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i32> [[I]] to <25 x i64>
  // CHECK-NEXT:  [[CONV1:%.*]] = bitcast [25 x i64]* {{.*}} to <25 x i64>*
  // CHECK-NEXT:  store <25 x i64> [[CONV]], <25 x i64>* {{.*}}, align 8
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<unsigned long int> u;
  u = (matrix_5_5<unsigned long int>)i;
}
