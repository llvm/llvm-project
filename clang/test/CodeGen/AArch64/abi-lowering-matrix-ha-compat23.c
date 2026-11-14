// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefix=LATEST
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=LATEST --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclang-abi-compat=23 -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefix=COMPAT23
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclang-abi-compat=23 -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=COMPAT23 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefix=LATEST
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=LATEST --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fclang-abi-compat=23 -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefix=COMPAT23
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fclang-abi-compat=23 -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=COMPAT23 --implicit-check-not="not yet implemented"

// A struct whose only field is a 2x2 float matrix is a homogeneous aggregate
// and is returned as the struct. Clang 23 did not accept a matrix as an HFA
// base type, so that 16-byte aggregate is returned as [2 x i64]. The ABI
// library matches both classifications.

typedef float fx2x2_t __attribute__((matrix_type(2, 2)));
struct MatrixStruct {
  fx2x2_t m;
};

struct MatrixStruct ret_matrix_struct(void) {
  struct MatrixStruct s;
  return s;
}
// LATEST: define{{.*}} %struct.MatrixStruct @ret_matrix_struct()
// COMPAT23: define{{.*}} [2 x i64] @ret_matrix_struct()
