// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -target-feature +sve -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -target-feature +sve -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature -fp-armv8 -target-abi aapcs-soft -target-feature +sve -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature -fp-armv8 -target-abi aapcs-soft -target-feature +sve -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"

// This test is verifying that the LLVM ABI library classifies AArch64 SVE
// tuples in the same way that Clang does without the library.

// DarwinPCS, Win64, and the soft-float ABI pass an SVE tuple directly.
// The tuple is expanded to one scalable vector argument per member, and a
// returned tuple is a struct of those vectors.

void arg_svint32x2(__clang_svint32x2_t v) {}
// CHECK: define{{.*}} void @arg_svint32x2(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})

void arg_svint32x3(__clang_svint32x3_t v) {}
// CHECK: define{{.*}} void @arg_svint32x3(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})

void arg_svint32x4(__clang_svint32x4_t v) {}
// CHECK: define{{.*}} void @arg_svint32x4(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})

void arg_svboolx2(__clang_svboolx2_t p) {}
// CHECK: define{{.*}} void @arg_svboolx2(<vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}})

void arg_svboolx4(__clang_svboolx4_t p) {}
// CHECK: define{{.*}} void @arg_svboolx4(<vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}})

__clang_svint32x2_t ret_svint32x2(__clang_svint32x2_t v) { return v; }
// CHECK: define{{.*}} { <vscale x 4 x i32>, <vscale x 4 x i32> } @ret_svint32x2(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})

__clang_svint32x3_t ret_svint32x3(__clang_svint32x3_t v) { return v; }
// CHECK: define{{.*}} { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @ret_svint32x3(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})

__clang_svint32x4_t ret_svint32x4(__clang_svint32x4_t v) { return v; }
// CHECK: define{{.*}} { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } @ret_svint32x4(<vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32> %{{.*}})

__clang_svboolx2_t ret_svboolx2(__clang_svboolx2_t p) { return p; }
// CHECK: define{{.*}} { <vscale x 16 x i1>, <vscale x 16 x i1> } @ret_svboolx2(<vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}})

__clang_svboolx4_t ret_svboolx4(__clang_svboolx4_t p) { return p; }
// CHECK: define{{.*}} { <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1>, <vscale x 16 x i1> } @ret_svboolx4(<vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}}, <vscale x 16 x i1> %{{.*}})
