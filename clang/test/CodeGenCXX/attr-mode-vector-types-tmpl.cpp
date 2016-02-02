// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s

template <class T>
void CheckIntScalarTypes() {
  // T will be substituted with 'int' and 'enum' types.

  typedef T __attribute__((mode(QI))) T1;
  typedef T __attribute__((mode(HI))) T2;
  typedef T __attribute__((mode(SI))) T3;
  typedef T __attribute__((mode(DI))) T4;

  T1 a1;
  T2 a2;
  T3 a3;
  T4 a4;
}

template <class T>
void CheckIntVectorTypes() {
  // T will be substituted with 'int'.

  typedef int __attribute__((mode(QI))) __attribute__((vector_size(8)))  VT_11;
  typedef T   __attribute__((mode(V8QI)))                                VT_12;
  typedef int __attribute__((mode(SI))) __attribute__((vector_size(16))) VT_21;
  typedef T   __attribute__((mode(V4SI)))                                VT_22;
  typedef int __attribute__((mode(DI))) __attribute__((vector_size(64))) VT_31;
  typedef T   __attribute__((mode(V8DI)))                                VT_32;

  VT_11 v11;
  VT_12 v12;

  VT_21 v21;
  VT_22 v22;

  VT_31 v31;
  VT_32 v32;
}

template <class T>
void CheckFloatVectorTypes() {
  // T will be substituted with 'float'.

  typedef float __attribute__((mode(SF))) __attribute__((vector_size(128))) VT_41;
  typedef T     __attribute__((mode(V32SF)))                                VT_42;
  typedef float __attribute__((mode(DF))) __attribute__((vector_size(256))) VT_51;
  typedef T     __attribute__((mode(V32DF)))                                VT_52;

  VT_41 v41;
  VT_42 v42;

  VT_51 v51;
  VT_52 v52;
}

template <class T>
void CheckInstantiationWithModedType() {
  T x1;
}

typedef enum { A1, B1 }                       EnumTy;
typedef int __attribute__((mode(DI)))         Int64Ty1;
typedef enum __attribute__((mode(DI))) { A2 } Int64Ty2;
typedef int __attribute__((mode(V8HI)))       IntVecTy1;

void test() {

  // CHECK: define {{.*}} void @_Z19CheckIntScalarTypesIiEvv()
  // CHECK: %{{.+}} = alloca i8
  // CHECK: %{{.+}} = alloca i16
  // CHECK: %{{.+}} = alloca i32
  // CHECK: %{{.+}} = alloca i64
  CheckIntScalarTypes<int>();

  // CHECK: define {{.*}} void @_Z19CheckIntScalarTypesI6EnumTyEvv()
  // CHECK: %{{.+}} = alloca i8
  // CHECK: %{{.+}} = alloca i16
  // CHECK: %{{.+}} = alloca i32
  // CHECK: %{{.+}} = alloca i64
  CheckIntScalarTypes<EnumTy>();

  // CHECK: define {{.*}} void @_Z19CheckIntVectorTypesIiEvv()
  // CHECK: %{{.+}} = alloca <8 x i8>
  // CHECK: %{{.+}} = alloca <8 x i8>
  // CHECK: %{{.+}} = alloca <4 x i32>
  // CHECK: %{{.+}} = alloca <4 x i32>
  // CHECK: %{{.+}} = alloca <8 x i64>
  // CHECK: %{{.+}} = alloca <8 x i64>
  CheckIntVectorTypes<int>();

  // CHECK: define {{.*}} void @_Z21CheckFloatVectorTypesIfEvv()
  // CHECK: %{{.+}} = alloca <32 x float>
  // CHECK: %{{.+}} = alloca <32 x float>
  // CHECK: %{{.+}} = alloca <32 x double>
  // CHECK: %{{.+}} = alloca <32 x double>
  CheckFloatVectorTypes<float>();

  // CHECK: define {{.*}} void @_Z31CheckInstantiationWithModedTypeIlEvv()
  // CHECK: [[X1:%.+]] = alloca i64
  CheckInstantiationWithModedType<Int64Ty1>();

  // CHECK: define {{.*}} void @_Z31CheckInstantiationWithModedTypeI8Int64Ty2Evv()
  // CHECK: [[X1]] = alloca i64
  CheckInstantiationWithModedType<Int64Ty2>();

  // CHECK: define {{.*}} void @_Z31CheckInstantiationWithModedTypeIDv8_sEvv()
  // CHECK: [[X1]] = alloca <8 x i16>
  CheckInstantiationWithModedType<IntVecTy1>();
}
