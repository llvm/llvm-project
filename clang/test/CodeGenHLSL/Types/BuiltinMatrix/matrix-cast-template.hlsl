// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - -DSPIRV | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s


template <typename X>
using matrix_3_3 = matrix<X, 3, 3>;

template <typename Y>
using matrix_4_4 = matrix<Y, 4, 4>;

// CHECK-LABEL: define {{.*}}CastCharMatrixToIntCStyle
void CastCharMatrixToIntCStyle() {
  // CHECK: [[C:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT: [[CONV:%.*]] = sext <16 x i16> [[C]] to <16 x i32>
  // CHECK-NEXT: store <16 x i32> [[CONV]], ptr {{.*}}, align 4

  matrix_4_4<int16_t> c;
  matrix_4_4<int> i;
  i = (matrix_4_4<int>)c;
}

// CHECK-LABEL: define {{.*}}CastCharMatrixToIntStaticCast
void CastCharMatrixToIntStaticCast() {
  // CHECK: [[C:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT: [[CONV:%.*]] = sext <16 x i16> [[C]] to <16 x i32>
  // CHECK-NEXT: store <16 x i32> [[CONV]], ptr {{.*}}, align 4

  matrix_4_4<int16_t> c;
  matrix_4_4<int> i;
  i = static_cast<matrix_4_4<int>>(c);
}

// CHECK-LABEL: define {{.*}}CastCharMatrixToUnsignedIntCStyle
void CastCharMatrixToUnsignedIntCStyle() {
  // CHECK:       [[C:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = sext <16 x i16> [[C]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<int16_t> c;
  matrix_4_4<uint> u;
  u = (matrix_4_4<uint>)c;
}

// CHECK-LABEL: define {{.*}}CastCharMatrixToUnsignedIntStaticCast
void CastCharMatrixToUnsignedIntStaticCast() {
  // CHECK:       [[C:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = sext <16 x i16> [[C]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<int16_t> c;
  matrix_4_4<uint> u;
  u = static_cast<matrix_4_4<uint>>(c);
}

// CHECK-LABEL: define {{.*}}CastUnsignedLongIntMatrixToShortCStyle
void CastUnsignedLongIntMatrixToShortCStyle() {
  // CHECK:      [[U:%.*]] = load <16 x i64>, ptr {{.*}}, align 8
  // CHECK-NEXT: [[CONV:%.*]] = trunc <16 x i64> {{.*}} to <16 x i16>
  // CHECK-NEXT: store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT: ret void

  matrix_4_4<uint64_t> u;
  matrix_4_4<int16_t> s;
  s = (matrix_4_4<int16_t>)u;
}

// CHECK-LABEL: define {{.*}}CastUnsignedLongIntMatrixToShortStaticCast
void CastUnsignedLongIntMatrixToShortStaticCast() {
  // CHECK:      [[U:%.*]] = load <16 x i64>, ptr {{.*}}, align 8
  // CHECK-NEXT: [[CONV:%.*]] = trunc <16 x i64> {{.*}} to <16 x i16>
  // CHECK-NEXT: store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT: ret void

  matrix_4_4<uint64_t> u;
  matrix_4_4<int16_t> s;
  s = static_cast<matrix_4_4<int16_t>>(u);
}

// CHECK-LABEL: define {{.*}}CastIntMatrixToShortCStyle
void CastIntMatrixToShortCStyle() {
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <16 x i32> [[I]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_4_4<int> i;
  matrix_4_4<int16_t> s;
  s = (matrix_4_4<int16_t>)i;
}

// CHECK-LABEL: define {{.*}}CastIntMatrixToShortStaticCast
void CastIntMatrixToShortStaticCast() {
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <16 x i32> [[I]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_4_4<int> i;
  matrix_4_4<int16_t> s;
  s = static_cast<matrix_4_4<int16_t>>(i);
}

// CHECK-LABEL: define {{.*}}CastIntMatrixToFloatCStyle
void CastIntMatrixToFloatCStyle() {
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV]] = sitofp <16 x i32> {{.*}} to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<int> i;
  matrix_4_4<float> f;
  f = (matrix_4_4<float>)i;
}

// CHECK-LABEL: define {{.*}}CastIntMatrixToFloatStaticCast
void CastIntMatrixToFloatStaticCast() {
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV]] = sitofp <16 x i32> {{.*}} to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<int> i;
  matrix_4_4<float> f;
  f = static_cast<matrix_4_4<float>>(i);
}

// CHECK-LABEL: define {{.*}}CastUnsignedIntMatrixToFloatCStyle
void CastUnsignedIntMatrixToFloatCStyle() {
  // CHECK:       [[U:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = uitofp <16 x i16> [[U]] to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<uint16_t> u;
  matrix_4_4<float> f;
  f = (matrix_4_4<float>)u;
}

// CHECK-LABEL: define {{.*}}CastUnsignedIntMatrixToFloatStaticCast
void CastUnsignedIntMatrixToFloatStaticCast() {
  // CHECK:       [[U:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = uitofp <16 x i16> [[U]] to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<uint16_t> u;
  matrix_4_4<float> f;
  f = static_cast<matrix_4_4<float>>(u);
}

// CHECK-LABEL: define {{.*}}CastDoubleMatrixToIntCStyle
void CastDoubleMatrixToIntCStyle() {
  // CHECK:       [[D:%.*]] = load <16 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptosi <16 x double> [[D]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<double> d;
  matrix_4_4<int> i;
  i = (matrix_4_4<int>)d;
}

// CHECK-LABEL: define {{.*}}CastDoubleMatrixToIntStaticCast
void CastDoubleMatrixToIntStaticCast() {
  // CHECK:       [[D:%.*]] = load <16 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptosi <16 x double> [[D]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<double> d;
  matrix_4_4<int> i;
  i = static_cast<matrix_4_4<int>>(d);
}

// CHECK-LABEL: define {{.*}}CastFloatMatrixToUnsignedShortIntCStyle
void CastFloatMatrixToUnsignedShortIntCStyle() {
  // CHECK:       [[F:%.*]] = load <16 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = fptoui <16 x float> [[F]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_4_4<float> f;
  matrix_4_4<uint16_t> i;
  i = (matrix_4_4<uint16_t>)f;
}

// CHECK-LABEL: define {{.*}}CastFloatMatrixToUnsignedShortIntStaticCast
void CastFloatMatrixToUnsignedShortIntStaticCast() {
  // CHECK:       [[F:%.*]] = load <16 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = fptoui <16 x float> [[F]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_4_4<float> f;
  matrix_4_4<uint16_t> i;
  i = static_cast<matrix_4_4<uint16_t>>(f);
}

// CHECK-LABEL: define {{.*}}CastDoubleMatrixToFloatCStyle
void CastDoubleMatrixToFloatCStyle() {
  // CHECK:       [[D:%.*]] = load <16 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptrunc <16 x double> [[D]] to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<double> d;
  matrix_4_4<float> f;
  f = (matrix_4_4<float>)d;
}

// CHECK-LABEL: define {{.*}}CastDoubleMatrixToFloatStaticCast
void CastDoubleMatrixToFloatStaticCast() {
  // CHECK:       [[D:%.*]] = load <16 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptrunc <16 x double> [[D]] to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<double> d;
  matrix_4_4<float> f;
  f = static_cast<matrix_4_4<float>>(d);
}

// CHECK-LABEL: define {{.*}}CastUnsignedShortIntToUnsignedIntCStyle
void CastUnsignedShortIntToUnsignedIntCStyle() {
  // CHECK:       [[S:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <16 x i16> [[S]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<uint16_t> s;
  matrix_4_4<uint> i;
  i = (matrix_4_4<uint>)s;
}

// CHECK-LABEL: define {{.*}}CastUnsignedShortIntToUnsignedIntStaticCast
void CastUnsignedShortIntToUnsignedIntStaticCast() {
  // CHECK:       [[S:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <16 x i16> [[S]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<uint16_t> s;
  matrix_4_4<uint> i;
  i = static_cast<matrix_4_4<uint>>(s);
}

// CHECK-LABEL: define {{.*}}CastUnsignedLongIntToUnsignedShortIntCStyle
void CastUnsignedLongIntToUnsignedShortIntCStyle() {
  // CHECK:       [[L:%.*]] = load <16 x i64>, ptr %l, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <16 x i64> [[L]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_4_4<uint64_t> l;
  matrix_4_4<uint16_t> s;
  s = (matrix_4_4<uint16_t>)l;
}

// CHECK-LABEL: define {{.*}}CastUnsignedLongIntToUnsignedShortIntStaticCast
void CastUnsignedLongIntToUnsignedShortIntStaticCast() {
  // CHECK:       [[L:%.*]] = load <16 x i64>, ptr %l, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <16 x i64> [[L]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  matrix_4_4<uint64_t> l;
  matrix_4_4<uint16_t> s;
  s = static_cast<matrix_4_4<uint16_t>>(l);
}

// CHECK-LABEL: define {{.*}}CastUnsignedShortIntToIntCStyle
void CastUnsignedShortIntToIntCStyle() {
  // CHECK:       [[U:%.*]] = load <16 x i16>, ptr %u, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <16 x i16> [[U]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<uint16_t> u;
  matrix_4_4<int> i;
  i = (matrix_4_4<int>)u;
}

// CHECK-LABEL: define {{.*}}CastUnsignedShortIntToIntStaticCast
void CastUnsignedShortIntToIntStaticCast() {
  // CHECK:       [[U:%.*]] = load <16 x i16>, ptr %u, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <16 x i16> [[U]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  matrix_4_4<uint16_t> u;
  matrix_4_4<int> i;
  i = static_cast<matrix_4_4<int>>(u);
}

// CHECK-LABEL: define {{.*}}CastIntToUnsignedLongIntCStyle
void CastIntToUnsignedLongIntCStyle() {
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr %i, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sext <16 x i32> [[I]] to <16 x i64>
  // CHECK-NEXT:  store <16 x i64> [[CONV]], ptr {{.*}}, align 8
  // CHECK-NEXT:  ret void

  matrix_4_4<int> i;
  matrix_4_4<uint64_t> u;
  u = (matrix_4_4<uint64_t>)i;
}

// CHECK-LABEL: define {{.*}}CastIntToUnsignedLongIntStaticCast
void CastIntToUnsignedLongIntStaticCast() {
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr %i, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sext <16 x i32> [[I]] to <16 x i64>
  // CHECK-NEXT:  store <16 x i64> [[CONV]], ptr {{.*}}, align 8
  // CHECK-NEXT:  ret void

  matrix_4_4<int> i;
  matrix_4_4<uint64_t> u;
  u = static_cast<matrix_4_4<uint64_t>>(i);
}

class Foo {
  int x[10];

  Foo(matrix_4_4<int> x);
};

#ifdef SPIRV
// These require mangling. DXIL uses MicrosoftMangle which doesn't support mangling matrices yet.
// SPIRV-LABEL: define {{.*}}class_constructor_matrix_ty
Foo class_constructor_matrix_ty(matrix_4_4<int> m) {
  // SPIRV:         [[M:%.*]]  = load <16 x i32>, ptr {{.*}}, align 4
  // SPIRV-NEXT:    call{{.*}} void @_ZN3FooC1Eu11matrix_typeILj4ELj4EiE(ptr noundef nonnull align 4 dereferenceable(40) %agg.result, <16 x i32> noundef [[M]])
  // SPIRV-NEXT:    ret void

  return Foo(m);
}

struct Bar {
  float x[10];
  Bar(matrix_3_3<float> x);
};

// SPIRV-LABEL: define {{.*}}struct_constructor_matrix_ty
Bar struct_constructor_matrix_ty(matrix_3_3<float> m) {
  // SPIRV:         [[M:%.*]] = load <9 x float>, ptr {{.*}}, align 4
  // SPIRV-NEXT:    call{{.*}} void @_ZN3BarC1Eu11matrix_typeILj3ELj3EfE(ptr noundef nonnull align 4 dereferenceable(40) %agg.result, <9 x float> noundef [[M]])
  // SPIRV-NEXT:    ret void

  return Bar(m);
}
#endif
