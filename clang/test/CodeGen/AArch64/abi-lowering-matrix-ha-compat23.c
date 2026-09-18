// REQUIRES: asserts
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=LATEST --implicit-check-not="not yet implemented"
// RUN: not --crash %clang_cc1 -triple aarch64-linux-gnu -fclang-abi-compat=23 -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=COMPAT23
// RUN: %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=LATEST --implicit-check-not="not yet implemented"
// RUN: not --crash %clang_cc1 -triple arm64-apple-ios -target-abi darwinpcs -fclang-abi-compat=23 -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=COMPAT23

// This is a temporary test to exercise ABI compatibility handling of matrix
// types. Clang 23 and earlier did not accept matrix types as HFA base types,
// but they are now accepted as HFA base types now. When the LLVM ABI library
// rejects a matrix type as an HFA base type, it falls through to a
// "not yet implemented" diagnostic, and the Clang asserts because the
// library's classification does not match Clang's classification.
//
// When the AArch64 classification is completed in the ABI library, this test
// will be removed and test cases will be added elsewhere to verify that the
// classification matches Clang's classification when the ABI compatibility
// flag is used.

typedef float fx2x2_t __attribute__((matrix_type(2, 2)));
struct MatrixStruct {
  fx2x2_t m;
};

struct MatrixStruct ret_matrix_struct(void) {
  struct MatrixStruct s;
  return s;
}
// LATEST: define{{.*}} %struct.MatrixStruct @ret_matrix_struct()
// COMPAT23: Aggregate return type handling is not yet implemented for AArch64 in the LLVM ABI library.
