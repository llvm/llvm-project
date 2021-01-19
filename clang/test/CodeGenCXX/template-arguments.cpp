// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o - -triple x86_64-linux -DCONSTEXPR= | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o - -triple x86_64-linux -DCONSTEXPR=constexpr | FileCheck %s --check-prefix=CONST

template<typename T> CONSTEXPR T id(T v) { return v; }
template<auto V> auto value = id(V);

// CHECK: call {{.*}} @_Z2idIiET_S0_(i32 1)
// CONST: @_Z5valueILi1EE = weak_odr {{.*}} i32 1,
template int value<1>;

// CHECK: call {{.*}} @_Z2idIyET_S0_(i64 -1)
// CONST: @_Z5valueILy18446744073709551615EE = weak_odr {{.*}} i64 -1,
template unsigned long long value<-1ULL>;

// CHECK: call {{.*}} @_Z2idIfET_S0_(float 1.000000e+00)
// CONST: @_Z5valueILf3f800000EE = weak_odr {{.*}} float 1.000000e+00,
template float value<1.0f>;
// CHECK: call {{.*}} @_Z2idIdET_S0_(double 1.000000e+00)
// CONST: @_Z5valueILd3ff0000000000000EE = weak_odr {{.*}} double 1.000000e+00,
template double value<1.0>;

int n;
// CHECK: call {{.*}} @_Z2idIPiET_S1_(i32* @n)
// CONST: @_Z5valueIXadL_Z1nEEE = weak_odr {{.*}} i32* @n,
template int *value<&n>;

struct A { int a[3]; } a;
// CHECK: call {{.*}} @_Z2idIPiET_S1_(i32* getelementptr inbounds (%struct.A, %struct.A* @a, i32 0, i32 0, i32 0))
// CONST: @_Z5valueIXadsoiL_Z1aEEEE = weak_odr {{.*}} i32* getelementptr inbounds (%struct.A, %struct.A* @a, i32 0, i32 0, i32 0),
template int *value<&a.a[0]>;
// CHECK: call {{.*}} @_Z2idIPiET_S1_(i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 4) to i32*))
// CONST: @_Z5valueIXadsoiL_Z1aE4EEE = weak_odr {{.*}} i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 4) to i32*),
template int *value<&a.a[1]>;
// CHECK: call {{.*}} @_Z2idIPiET_S1_(i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 8) to i32*))
// CONST: @_Z5valueIXadsoiL_Z1aE8EEE = weak_odr {{.*}} i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 8) to i32*),
template int *value<&a.a[2]>;
// CHECK: call {{.*}} @_Z2idIPiET_S1_(i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 12) to i32*))
// CONST: @_Z5valueIXadsoiL_Z1aE12pEEE = weak_odr {{.*}} i32* bitcast (i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 12) to i32*),
template int *value<&a.a[3]>;

struct B { int x, y; };
// CHECK: call {{.*}} @_Z2idIM1BiET_S2_(i64 0)
// CONST: @_Z5valueIXadL_ZN1B1xEEEE = weak_odr {{.*}} i64 0,
template int B::*value<&B::x>;
// CHECK: call {{.*}} @_Z2idIM1BiET_S2_(i64 4)
// CONST: @_Z5valueIXadL_ZN1B1yEEEE = weak_odr {{.*}} i64 4,
template int B::*value<&B::y>;

struct C : A, B { int z; };
// CHECK: call {{.*}} @_Z2idIM1CiET_S2_(i64 12)
// CONST: @_Z5valueIXmcM1CiadL_ZN1B1xEE12EEE = weak_odr {{.*}} i64 12,
template int C::*value<(int C::*)&B::x>;
// CHECK: call {{.*}} @_Z2idIM1BiET_S2_(i64 8)
// CONST: @_Z5valueIXmcM1BiadL_ZN1C1zEEn12EEE = weak_odr {{.*}} i64 8,
template int B::*value<(int B::*)&C::z>;

// CHECK: store i32 1, i32*
// CHECK: store i32 2, i32*
// CHECK: bitcast { i32, i32 }* %{{.*}} to i64*
// CHECK: load i64,
// CHECK: call {{.*}} @_Z2idICiET_S1_(i64 %
// CONST: @_Z5valueIXtlCiLi1ELi2EEEE = weak_odr {{.*}} { i32, i32 } { i32 1, i32 2 },
template _Complex int value<1 + 2j>;

// CHECK: store float 1.000000e+00, float*
// CHECK: store float 2.000000e+00, float*
// CHECK: bitcast { float, float }* %{{.*}} to <2 x float>*
// CHECK: load <2 x float>,
// CHECK: call {{.*}} @_Z2idICfET_S1_(<2 x float> %
// CONST: @_Z5valueIXtlCfLf3f800000ELf40000000EEEE = weak_odr {{.*}} { float, float } { float 1.000000e+00, float 2.000000e+00 },
template _Complex float value<1.0f + 2.0fj>;

using V3i __attribute__((ext_vector_type(3))) = int;
// CHECK: call {{.*}} @_Z2idIDv3_iET_S1_(<3 x i32> <i32 1, i32 2, i32 3>)
// CONST: @_Z5valueIXtlDv3_iLi1ELi2ELi3EEEE = weak_odr {{.*}} <3 x i32> <i32 1, i32 2, i32 3>
template V3i value<V3i{1, 2, 3}>;

using V3f [[gnu::vector_size(12)]] = float;
// CHECK: call {{.*}} @_Z2idIDv3_fET_S1_(<3 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>)
// CONST: @_Z5valueIXtlDv3_fLf3f800000ELf40000000ELf40400000EEEE = weak_odr {{.*}} <3 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
template V3f value<V3f{1, 2, 3}>;
