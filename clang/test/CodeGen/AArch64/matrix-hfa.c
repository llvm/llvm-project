// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN

typedef float __attribute__((matrix_type(2, 2))) f2x2;
typedef float __attribute__((matrix_type(3, 3))) f3x3;
typedef double __attribute__((matrix_type(2, 2))) d2x2;
typedef int __attribute__((matrix_type(2, 2))) i2x2;

struct MatrixStruct {
  f2x2 m;
};

struct ArrayStruct {
  float arr[4];
};

struct LargeMatrixStruct {
  f3x3 m;
};

struct DoubleMatrixStruct {
  d2x2 m;
};

struct IntMatrixStruct {
  i2x2 m;
};

void take_matrix(f2x2 m) {}
// CHECK-LABEL: define{{.*}} void @take_matrix(<4 x float> noundef %{{.*}})

f2x2 return_matrix(void) {
  f2x2 m;
  return m;
}
// CHECK-LABEL: define{{.*}} <4 x float> @return_matrix

void take_matrix_struct(struct MatrixStruct s) {}
// AAPCS: define{{.*}} void @take_matrix_struct([4 x float] alignstack(8) %{{.*}})
// DARWIN: define{{.*}} void @take_matrix_struct([4 x float] %{{.*}})

struct MatrixStruct return_matrix_struct(void) {
  struct MatrixStruct s;
  return s;
}
// CHECK-LABEL: define{{.*}} %struct.MatrixStruct @return_matrix_struct()

void take_array_struct(struct ArrayStruct s) {}
// AAPCS: define{{.*}} void @take_array_struct([4 x float] alignstack(8) %{{.*}})
// DARWIN: define{{.*}} void @take_array_struct([4 x float] %{{.*}})

struct ArrayStruct return_array_struct(void) {
  struct ArrayStruct s;
  return s;
}
// CHECK-LABEL: define{{.*}} %struct.ArrayStruct @return_array_struct()

void take_double_matrix_struct(struct DoubleMatrixStruct s) {}
// AAPCS: define{{.*}} void @take_double_matrix_struct([4 x double] alignstack(8) %{{.*}})
// DARWIN: define{{.*}} void @take_double_matrix_struct([4 x double] %{{.*}})

struct DoubleMatrixStruct return_double_matrix_struct(void) {
  struct DoubleMatrixStruct s;
  return s;
}
// CHECK-LABEL: define{{.*}} %struct.DoubleMatrixStruct @return_double_matrix_struct()

void take_large_matrix_struct(struct LargeMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_large_matrix_struct(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(36) %{{.*}})

void take_int_matrix_struct(struct IntMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_int_matrix_struct([2 x i64] %{{.*}})
