// RUN: %clang_cc1 -target-feature +altivec \
// RUN:   -triple powerpc64le-unknown-linux-gnu -fenable-matrix -emit-llvm \
// RUN:   -o - %s | FileCheck %s --check-prefixes=CHECK,HFA
// RUN: %clang_cc1 -target-feature +altivec \
// RUN:   -triple powerpc64le-unknown-linux-gnu -fenable-matrix \
// RUN:   -fclang-abi-compat=23 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,COMPAT23

// PPC64 ELFv2 homogeneous-aggregate classification of structs that contain
// Clang matrix types. Matrix types are flattened like arrays (T NxM -> N*M
// members of T). Bare matrices are scalars (<N x T>), not homogeneous
// aggregates. ELFv2 coerces homogeneous aggregates to [N x Base] for both
// arguments and returns (up to 8 registers).
//
// -fclang-abi-compat=23 does not flatten matrices, so those structs follow
// the ordinary ELFv2 aggregate rules.

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

// Four floats, no padding -> HA [4 x float]. Compat 23 uses [2 x i64] /
// { i64, i64 }.
void take_matrix_struct(struct MatrixStruct s) {}
// HFA: define{{.*}} void @take_matrix_struct([4 x float] %{{.*}})
// COMPAT23: define{{.*}} void @take_matrix_struct([2 x i64] %{{.*}})

struct MatrixStruct return_matrix_struct(void) {
  struct MatrixStruct s;
  return s;
}
// HFA-LABEL: define{{.*}} [4 x float] @return_matrix_struct()
// COMPAT23-LABEL: define{{.*}} { i64, i64 } @return_matrix_struct()

// float[4] was already an HA.
void take_array_struct(struct ArrayStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_array_struct([4 x float] %{{.*}})

struct ArrayStruct return_array_struct(void) {
  struct ArrayStruct s;
  return s;
}
// CHECK-LABEL: define{{.*}} [4 x float] @return_array_struct()

// Four doubles -> HA [4 x double]. Compat 23 uses [4 x i64] / sret.
void take_double_matrix_struct(struct DoubleMatrixStruct s) {}
// HFA: define{{.*}} void @take_double_matrix_struct([4 x double] %{{.*}})
// COMPAT23: define{{.*}} void @take_double_matrix_struct([4 x i64] %{{.*}})

struct DoubleMatrixStruct return_double_matrix_struct(void) {
  struct DoubleMatrixStruct s;
  return s;
}
// HFA-LABEL: define{{.*}} [4 x double] @return_double_matrix_struct()
// COMPAT23-LABEL: define{{.*}} void @return_double_matrix_struct(ptr {{.*}}sret(%struct.DoubleMatrixStruct)

// Nine floats need 9 FPRs; ELFv2 homogeneous aggregates use at most 8.
void take_large_matrix_struct(struct LargeMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_large_matrix_struct([5 x i64] %{{.*}})

// int is not an HA base type.
void take_int_matrix_struct(struct IntMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_int_matrix_struct([2 x i64] %{{.*}})

// Tail padding from aligned(32) disqualifies the HA.
void take_overaligned_matrix_struct(struct OverAlignedMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_overaligned_matrix_struct([2 x i128] %{{.*}})
