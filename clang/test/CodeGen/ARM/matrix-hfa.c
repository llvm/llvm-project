// RUN: %clang_cc1 -triple thumbv7-apple-darwin9 -target-abi aapcs \
// RUN:   -target-cpu cortex-a8 -mfloat-abi hard -fenable-matrix -emit-llvm \
// RUN:   -o - %s | FileCheck %s --check-prefixes=CHECK,HFA
// RUN: %clang_cc1 -triple thumbv7-apple-darwin9 -target-abi aapcs \
// RUN:   -target-cpu cortex-a8 -mfloat-abi hard -fenable-matrix \
// RUN:   -fclang-abi-compat=23 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,COMPAT23

// AAPCS-VFP HFA classification of structs that contain Clang matrix types.
// Matrix types are flattened like arrays (T NxM -> N*M members of T). Bare
// matrices are scalars (<N x T>), not HFAs. AAPCS-VFP passes HFAs as the
// original LLVM struct type under arm_aapcs_vfpcc.
//
// -fclang-abi-compat=23 does not flatten matrices, so those structs are
// ordinary composites (GPR coercion or sret).

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

struct OverAlignedMatrixStruct {
  f2x2 m;
} __attribute__((aligned(32)));

void take_matrix(f2x2 m) {}
// CHECK-LABEL: define{{.*}} void @take_matrix(<4 x float> noundef %{{.*}})

f2x2 return_matrix(void) {
  f2x2 m;
  return m;
}
// CHECK-LABEL: define{{.*}} <4 x float> @return_matrix

// Four floats, no padding -> HFA. Compat 23 coerces the 16-byte aggregate
// to [4 x i32] / sret.
void take_matrix_struct(struct MatrixStruct s) {}
// HFA: define{{.*}} void @take_matrix_struct(%struct.MatrixStruct %{{.*}})
// COMPAT23: define{{.*}} void @take_matrix_struct([4 x i32] %{{.*}})

struct MatrixStruct return_matrix_struct(void) {
  struct MatrixStruct s;
  return s;
}
// HFA-LABEL: define{{.*}} %struct.MatrixStruct @return_matrix_struct()
// COMPAT23-LABEL: define{{.*}} void @return_matrix_struct(ptr {{.*}}sret(%struct.MatrixStruct)

// float[4] was already an HFA.
void take_array_struct(struct ArrayStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_array_struct(%struct.ArrayStruct %{{.*}})

struct ArrayStruct return_array_struct(void) {
  struct ArrayStruct s;
  return s;
}
// CHECK-LABEL: define{{.*}} %struct.ArrayStruct @return_array_struct()

// Four doubles -> HFA. Compat 23 uses [4 x i64] / sret.
void take_double_matrix_struct(struct DoubleMatrixStruct s) {}
// HFA: define{{.*}} void @take_double_matrix_struct(%struct.DoubleMatrixStruct %{{.*}})
// COMPAT23: define{{.*}} void @take_double_matrix_struct([4 x i64] %{{.*}})

struct DoubleMatrixStruct return_double_matrix_struct(void) {
  struct DoubleMatrixStruct s;
  return s;
}
// HFA-LABEL: define{{.*}} %struct.DoubleMatrixStruct @return_double_matrix_struct()
// COMPAT23-LABEL: define{{.*}} void @return_double_matrix_struct(ptr {{.*}}sret(%struct.DoubleMatrixStruct)

// Nine floats exceeds the 4-member HFA limit.
void take_large_matrix_struct(struct LargeMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_large_matrix_struct([9 x i32] %{{.*}})

// int is not an HFA base type.
void take_int_matrix_struct(struct IntMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_int_matrix_struct([4 x i32] %{{.*}})

// Tail padding from aligned(32) disqualifies the HFA.
void take_overaligned_matrix_struct(struct OverAlignedMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_overaligned_matrix_struct([8 x i32] %{{.*}})
