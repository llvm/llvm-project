// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HFA,HFA-AAPCS,AAPCS
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HFA,HFA-DARWIN,DARWIN
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclang-abi-compat=23 -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,COMPAT23,AAPCS
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fclang-abi-compat=23 -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,COMPAT23,DARWIN

// AArch64 classifies a type as an HFA (Homogeneous Floating-point Aggregate)
// when it is made of 1-4 identical floating-point (or 64/128-bit vector)
// members. Matrix types are flattened the same way as arrays: a T NxM matrix
// contributes N*M members of T.
//
// Bare matrix values are scalars in CodeGen (LLVM vectors), so they are not
// themselves HFAs. A struct that contains a matrix can be an HFA.
//
// Argument HFAs are coerced to [N x Base]. AAPCS64 also sets the stack
// alignment of that coerced type to 8 (or 16 if the unadjusted alignment is
// at least 16 bytes), which appears as alignstack(8) here. Darwin PCS uses
// the same coerce type without that attribute.
//
// Returned HFAs keep the original LLVM struct type (direct, no coerce).
// Aggregates that are not HFAs follow the usual AArch64 rules: at most 16
// bytes in GPRs (here [2 x i64]), otherwise passed indirectly. Over-alignment
// that inserts tail padding also disqualifies an otherwise homogeneous type.
//
// -fclang-abi-compat=23 restores the pre-Clang-24 behavior: matrix types are
// not flattened for HFA classification, so a struct whose only field is a
// matrix is an opaque aggregate. Arrays of float were already HFAs and are
// unchanged.

typedef float __attribute__((matrix_type(2, 2))) f2x2;
typedef float __attribute__((matrix_type(1, 2))) f1x2;
typedef _Float16 __attribute__((matrix_type(2, 2))) h2x2;
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

struct ArrayOfOneMatrix {
  f2x2 m[1];
};

struct HalfMatrixStruct {
  h2x2 m;
};

struct ArrayOfTwoMatrices {
  f1x2 m[2];
};

struct ArrayOfTwoLargeMatrices {
  f2x2 m[2];
};

// f2x2 is four floats. As a scalar matrix it lowers to <4 x float> and is
// passed in a SIMD register, not classified as an HFA.
void take_matrix(f2x2 m) {}
// CHECK-LABEL: define{{.*}} void @take_matrix(<4 x float> noundef %{{.*}})

// Returned the same way: the LLVM function type uses <4 x float> directly.
f2x2 return_matrix(void) {
  f2x2 m;
  return m;
}
// CHECK-LABEL: define{{.*}} <4 x float> @return_matrix

// One f2x2 field: four floats, no padding -> HFA of 4 floats. The argument
// is therefore coerced to [4 x float]; AAPCS adds alignstack(8).
// With -fclang-abi-compat=23 the matrix is not an HFA member, so the 16-byte
// struct is passed in GPRs as [2 x i64].
void take_matrix_struct(struct MatrixStruct s) {}
// HFA-AAPCS: define{{.*}} void @take_matrix_struct([4 x float] alignstack(8) %{{.*}})
// HFA-DARWIN: define{{.*}} void @take_matrix_struct([4 x float] %{{.*}})
// COMPAT23: define{{.*}} void @take_matrix_struct([2 x i64] %{{.*}})

// Returned HFAs are not coerced, so LLVM keeps %struct.MatrixStruct.
// Compat 23 returns the 16-byte aggregate as [2 x i64].
struct MatrixStruct return_matrix_struct(void) {
  struct MatrixStruct s;
  return s;
}
// HFA-LABEL: define{{.*}} %struct.MatrixStruct @return_matrix_struct()
// COMPAT23-LABEL: define{{.*}} [2 x i64] @return_matrix_struct()

// float[4] is the array analogue of f2x2. Arrays were HFAs before the matrix
// change, so this matches on both latest and -fclang-abi-compat=23.
void take_array_struct(struct ArrayStruct s) {}
// AAPCS: define{{.*}} void @take_array_struct([4 x float] alignstack(8) %{{.*}})
// DARWIN: define{{.*}} void @take_array_struct([4 x float] %{{.*}})

struct ArrayStruct return_array_struct(void) {
  struct ArrayStruct s;
  return s;
}
// CHECK-LABEL: define{{.*}} %struct.ArrayStruct @return_array_struct()

// d2x2 is four doubles (32 bytes). It is still an HFA (4 members of double),
// so it is passed as [4 x double] rather than indirectly.
// Compat 23 does not treat the matrix as an HFA, so 32 bytes are indirect.
void take_double_matrix_struct(struct DoubleMatrixStruct s) {}
// HFA-AAPCS: define{{.*}} void @take_double_matrix_struct([4 x double] alignstack(8) %{{.*}})
// HFA-DARWIN: define{{.*}} void @take_double_matrix_struct([4 x double] %{{.*}})
// COMPAT23: define{{.*}} void @take_double_matrix_struct(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %{{.*}})

struct DoubleMatrixStruct return_double_matrix_struct(void) {
  struct DoubleMatrixStruct s;
  return s;
}
// HFA-LABEL: define{{.*}} %struct.DoubleMatrixStruct @return_double_matrix_struct()
// COMPAT23-LABEL: define{{.*}} void @return_double_matrix_struct(ptr {{.*}}sret(%struct.DoubleMatrixStruct)

// f3x3 is nine floats. HFAs may have at most four members, so this is not an
// HFA. 36 bytes exceeds the 16-byte GPR limit, so it is passed indirectly
// (pointer with dereferenceable(36)). Same under compat 23.
void take_large_matrix_struct(struct LargeMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_large_matrix_struct(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(36) %{{.*}})

// int is not an HFA base type. i2x2 is 16 bytes, so the struct is passed in
// GPRs as [2 x i64] on both AAPCS and Darwin, including compat 23.
void take_int_matrix_struct(struct IntMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_int_matrix_struct([2 x i64] %{{.*}})

// aligned(32) on a 16-byte f2x2 struct adds 16 bytes of tail padding. HFAs
// require size == base_size * members, so this is not an HFA and is passed
// indirectly (32-byte, 32-aligned). Same under compat 23.
void take_overaligned_matrix_struct(struct OverAlignedMatrixStruct s) {}
// CHECK-LABEL: define{{.*}} void @take_overaligned_matrix_struct(ptr nofreeobj noundef align 32 dead_on_return dereferenceable(32) %{{.*}})

// Array of one f2x2: the array branch multiplies the matrix's 4 floats by 1,
// so this is the same HFA as MatrixStruct. Compat 23 does not flatten the
// matrix element, so the 16-byte struct is passed as [2 x i64].
void take_array_of_one_matrix(struct ArrayOfOneMatrix s) {}
// HFA-AAPCS: define{{.*}} void @take_array_of_one_matrix([4 x float] alignstack(8) %{{.*}})
// HFA-DARWIN: define{{.*}} void @take_array_of_one_matrix([4 x float] %{{.*}})
// COMPAT23: define{{.*}} void @take_array_of_one_matrix([2 x i64] %{{.*}})

// One 2x2 of _Float16: four halves, no padding -> HFA of 4 half.
// Compat 23 does not flatten the matrix, so the 8-byte struct is passed as
// i64.
void take_half_matrix_struct(struct HalfMatrixStruct s) {}
// HFA-AAPCS: define{{.*}} void @take_half_matrix_struct([4 x half] alignstack(8) %{{.*}})
// HFA-DARWIN: define{{.*}} void @take_half_matrix_struct([4 x half] %{{.*}})
// COMPAT23: define{{.*}} void @take_half_matrix_struct(i64 %{{.*}})

// Array of two f1x2: each matrix is two floats, so the array is four floats
// and is an HFA. This would be rejected if arrays of matrices were limited
// by the array length rather than the flattened member count. Compat 23
// does not flatten the matrix element, so the 16-byte struct is passed as
// [2 x i64].
void take_array_of_two_matrices(struct ArrayOfTwoMatrices s) {}
// HFA-AAPCS: define{{.*}} void @take_array_of_two_matrices([4 x float] alignstack(8) %{{.*}})
// HFA-DARWIN: define{{.*}} void @take_array_of_two_matrices([4 x float] %{{.*}})
// COMPAT23: define{{.*}} void @take_array_of_two_matrices([2 x i64] %{{.*}})

// Array of two f2x2: eight floats, which exceeds the 4-member HFA limit, so
// this is not an HFA. 32 bytes exceeds the 16-byte GPR limit, so it is
// passed indirectly. Same under compat 23.
void take_array_of_two_large_matrices(struct ArrayOfTwoLargeMatrices s) {}
// CHECK-LABEL: define{{.*}} void @take_array_of_two_large_matrices(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(32) %{{.*}})
