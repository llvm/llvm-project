// RUN: %clang_cc1 -std=c++11 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

template <typename X>
using matrix_4_4 = X __attribute__((matrix_type(4, 4)));

template <typename Y>
using matrix_5_5 = Y __attribute__((matrix_type(5, 5)));

// CHECK-LABEL: define{{.*}} void @_Z25CastCharMatrixToIntCStylev()
void CastCharMatrixToIntCStyle() {
  // CHECK: [[C:%.*]] = load <25 x i8>, ptr {{.*}}, align 1
  // CHECK-NEXT: [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT: store <25 x i32> [[CONV]], ptr {{.*}}, align 4

  matrix_5_5<char> c;
  matrix_5_5<int> i;
  i = (matrix_5_5<int>)c;
}

// CHECK-LABEL: define{{.*}} void @_Z29CastCharMatrixToIntStaticCastv()
void CastCharMatrixToIntStaticCast() {
  // CHECK: [[C:%.*]] = load <25 x i8>, ptr {{.*}}, align 1
  // CHECK-NEXT: [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT: store <25 x i32> [[CONV]], ptr {{.*}}, align 4

  matrix_5_5<char> c;
  matrix_5_5<int> i;
  i = static_cast<matrix_5_5<int>>(c);
}

// CHECK-LABEL: define{{.*}} void @_Z33CastCharMatrixToUnsignedIntCStylev
void CastCharMatrixToUnsignedIntCStyle() {
  // CHECK:       [[C:%.*]] = load <25 x i8>, ptr {{.*}}, align 1
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<char> c;
  matrix_5_5<unsigned int> u;
  u = (matrix_5_5<unsigned int>)c;
}

// CHECK-LABEL: define{{.*}} void @_Z37CastCharMatrixToUnsignedIntStaticCastv
void CastCharMatrixToUnsignedIntStaticCast() {
  // CHECK:       [[C:%.*]] = load <25 x i8>, ptr {{.*}}, align 1
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i8> [[C]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<char> c;
  matrix_5_5<unsigned int> u;
  u = static_cast<matrix_5_5<unsigned int>>(c);
}

// CHECK-LABEL: define{{.*}} void @_Z38CastUnsignedLongIntMatrixToShortCStylev
void CastUnsignedLongIntMatrixToShortCStyle() {
  // CHECK:      [[U:%.*]] = load <25 x i64>, ptr {{.*}}, align 8
  // CHECK-NEXT: [[CONV:%.*]] = trunc <25 x i64> {{.*}} to <25 x i16>
  // CHECK-NEXT: store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT: ret void

  matrix_5_5<unsigned long int> u;
  matrix_5_5<short int> s;
  s = (matrix_5_5<short int>)u;
}

// CHECK-LABEL: define{{.*}} void @_Z42CastUnsignedLongIntMatrixToShortStaticCastv
void CastUnsignedLongIntMatrixToShortStaticCast() {
  // CHECK:      [[U:%.*]] = load <25 x i64>, ptr {{.*}}, align 8
  // CHECK-NEXT: [[CONV:%.*]] = trunc <25 x i64> {{.*}} to <25 x i16>
  // CHECK-NEXT: store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT: ret void

  matrix_5_5<unsigned long int> u;
  matrix_5_5<short int> s;
  s = static_cast<matrix_5_5<short int>>(u);
}

// CHECK-LABEL: define{{.*}} void @_Z26CastIntMatrixToShortCStylev()
void CastIntMatrixToShortCStyle() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i32> [[I]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<short int> s;
  s = (matrix_5_5<short int>)i;
}

// CHECK-LABEL: define{{.*}} void @_Z30CastIntMatrixToShortStaticCastv()
void CastIntMatrixToShortStaticCast() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i32> [[I]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<short int> s;
  s = static_cast<matrix_5_5<short int>>(i);
}

// CHECK-LABEL: define{{.*}} void @_Z26CastIntMatrixToFloatCStylev()
void CastIntMatrixToFloatCStyle() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV]] = sitofp <25 x i32> {{.*}} to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<float> f;
  f = (matrix_5_5<float>)i;
}

// CHECK-LABEL: define{{.*}} void @_Z30CastIntMatrixToFloatStaticCastv()
void CastIntMatrixToFloatStaticCast() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV]] = sitofp <25 x i32> {{.*}} to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<float> f;
  f = static_cast<matrix_5_5<float>>(i);
}

// CHECK-LABEL: define{{.*}} void @_Z34CastUnsignedIntMatrixToFloatCStylev()
void CastUnsignedIntMatrixToFloatCStyle() {
  // CHECK:       [[U:%.*]] = load <25 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = uitofp <25 x i16> [[U]] to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> u;
  matrix_5_5<float> f;
  f = (matrix_5_5<float>)u;
}

// CHECK-LABEL: define{{.*}} void @_Z38CastUnsignedIntMatrixToFloatStaticCastv()
void CastUnsignedIntMatrixToFloatStaticCast() {
  // CHECK:       [[U:%.*]] = load <25 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = uitofp <25 x i16> [[U]] to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> u;
  matrix_5_5<float> f;
  f = static_cast<matrix_5_5<float>>(u);
}

// CHECK-LABEL: define{{.*}} void @_Z27CastDoubleMatrixToIntCStylev()
void CastDoubleMatrixToIntCStyle() {
  // CHECK:       [[D:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptosi <25 x double> [[D]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<double> d;
  matrix_5_5<int> i;
  i = (matrix_5_5<int>)d;
}

// CHECK-LABEL: define{{.*}} void @_Z31CastDoubleMatrixToIntStaticCastv()
void CastDoubleMatrixToIntStaticCast() {
  // CHECK:       [[D:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptosi <25 x double> [[D]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<double> d;
  matrix_5_5<int> i;
  i = static_cast<matrix_5_5<int>>(d);
}

// CHECK-LABEL: define{{.*}} void @_Z39CastFloatMatrixToUnsignedShortIntCStylev()
void CastFloatMatrixToUnsignedShortIntCStyle() {
  // CHECK:       [[F:%.*]] = load <25 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = fptoui <25 x float> [[F]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<float> f;
  matrix_5_5<unsigned short int> i;
  i = (matrix_5_5<unsigned short int>)f;
}

// CHECK-LABEL: define{{.*}} void @_Z43CastFloatMatrixToUnsignedShortIntStaticCastv()
void CastFloatMatrixToUnsignedShortIntStaticCast() {
  // CHECK:       [[F:%.*]] = load <25 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = fptoui <25 x float> [[F]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<float> f;
  matrix_5_5<unsigned short int> i;
  i = static_cast<matrix_5_5<unsigned short int>>(f);
}

// CHECK-LABEL: define{{.*}} void @_Z29CastDoubleMatrixToFloatCStylev()
void CastDoubleMatrixToFloatCStyle() {
  // CHECK:       [[D:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptrunc <25 x double> [[D]] to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<double> d;
  matrix_5_5<float> f;
  f = (matrix_5_5<float>)d;
}

// CHECK-LABEL: define{{.*}} void @_Z33CastDoubleMatrixToFloatStaticCastv()
void CastDoubleMatrixToFloatStaticCast() {
  // CHECK:       [[D:%.*]] = load <25 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptrunc <25 x double> [[D]] to <25 x float>
  // CHECK-NEXT:  store <25 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<double> d;
  matrix_5_5<float> f;
  f = static_cast<matrix_5_5<float>>(d);
}

// CHECK-LABEL: define{{.*}} void @_Z39CastUnsignedShortIntToUnsignedIntCStylev()
void CastUnsignedShortIntToUnsignedIntCStyle() {
  // CHECK:       [[S:%.*]] = load <25 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[S]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> s;
  matrix_5_5<unsigned int> i;
  i = (matrix_5_5<unsigned int>)s;
}

// CHECK-LABEL: define{{.*}} void @_Z43CastUnsignedShortIntToUnsignedIntStaticCastv()
void CastUnsignedShortIntToUnsignedIntStaticCast() {
  // CHECK:       [[S:%.*]] = load <25 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[S]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> s;
  matrix_5_5<unsigned int> i;
  i = static_cast<matrix_5_5<unsigned int>>(s);
}

// CHECK-LABEL: define{{.*}} void @_Z43CastUnsignedLongIntToUnsignedShortIntCStylev()
void CastUnsignedLongIntToUnsignedShortIntCStyle() {
  // CHECK:       [[L:%.*]] = load <25 x i64>, ptr %l, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i64> [[L]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned long int> l;
  matrix_5_5<unsigned short int> s;
  s = (matrix_5_5<unsigned short int>)l;
}

// CHECK-LABEL: define{{.*}} void @_Z47CastUnsignedLongIntToUnsignedShortIntStaticCastv()
void CastUnsignedLongIntToUnsignedShortIntStaticCast() {
  // CHECK:       [[L:%.*]] = load <25 x i64>, ptr %l, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <25 x i64> [[L]] to <25 x i16>
  // CHECK-NEXT:  store <25 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned long int> l;
  matrix_5_5<unsigned short int> s;
  s = static_cast<matrix_5_5<unsigned short int>>(l);
}

// CHECK-LABEL: define{{.*}} void @_Z31CastUnsignedShortIntToIntCStylev()
void CastUnsignedShortIntToIntCStyle() {
  // CHECK:       [[U:%.*]] = load <25 x i16>, ptr %u, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[U]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> u;
  matrix_5_5<int> i;
  i = (matrix_5_5<int>)u;
}

// CHECK-LABEL: define{{.*}} void @_Z35CastUnsignedShortIntToIntStaticCastv()
void CastUnsignedShortIntToIntStaticCast() {
  // CHECK:       [[U:%.*]] = load <25 x i16>, ptr %u, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <25 x i16> [[U]] to <25 x i32>
  // CHECK-NEXT:  store <25 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_5_5<unsigned short int> u;
  matrix_5_5<int> i;
  i = static_cast<matrix_5_5<int>>(u);
}

// CHECK-LABEL: define{{.*}} void @_Z30CastIntToUnsignedLongIntCStylev()
void CastIntToUnsignedLongIntCStyle() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, ptr %i, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i32> [[I]] to <25 x i64>
  // CHECK-NEXT:  store <25 x i64> [[CONV]], ptr {{.*}}, align 8
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<unsigned long int> u;
  u = (matrix_5_5<unsigned long int>)i;
}

// CHECK-LABEL: define{{.*}} void @_Z34CastIntToUnsignedLongIntStaticCastv()
void CastIntToUnsignedLongIntStaticCast() {
  // CHECK:       [[I:%.*]] = load <25 x i32>, ptr %i, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sext <25 x i32> [[I]] to <25 x i64>
  // CHECK-NEXT:  store <25 x i64> [[CONV]], ptr {{.*}}, align 8
  // CHECK-NEXT:  ret void

  matrix_5_5<int> i;
  matrix_5_5<unsigned long int> u;
  u = static_cast<matrix_5_5<unsigned long int>>(i);
}

class Foo {
  int x[10];

public:
  Foo(matrix_5_5<int> x);
};

Foo class_constructor_matrix_ty(matrix_5_5<int> m) {
  // CHECK-LABEL: define void @_Z27class_constructor_matrix_tyu11matrix_typeILm5ELm5EiE(ptr noalias sret(%class.Foo) align 4 %agg.result, <25 x i32> noundef %m)
  // CHECK:         [[M:%.*]]  = load <25 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:    call void @_ZN3FooC1Eu11matrix_typeILm5ELm5EiE(ptr noundef nonnull align 4 dereferenceable(40) %agg.result, <25 x i32> noundef [[M]])
  // CHECK-NEXT:    ret void

  return Foo(m);
}

struct Bar {
  float x[10];
  Bar(matrix_4_4<float> x);
};

Bar struct_constructor_matrix_ty(matrix_4_4<float> m) {
  // CHECK-LABEL: define void @_Z28struct_constructor_matrix_tyu11matrix_typeILm4ELm4EfE(ptr noalias sret(%struct.Bar) align 4 %agg.result, <16 x float> noundef %m)
  // CHECK:         [[M:%.*]] = load <16 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:    call void @_ZN3BarC1Eu11matrix_typeILm4ELm4EfE(ptr noundef nonnull align 4 dereferenceable(40) %agg.result, <16 x float> noundef [[M]])
  // CHECK-NEXT:    ret void

  return Bar(m);
}
