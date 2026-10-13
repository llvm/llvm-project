// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -DBITS=512 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -DBITS=512 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -target-feature +sve -mvscale-min=2 -mvscale-max=2 -DBITS=256 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"

// This test is verifying that the LLVM ABI library classifies the AArch64 SVE
// types in the same way that Clang does without the library.

// The test runs at two vector lengths. The classification of a fixed-length
// type does not depend on the vector length, so both use the same check lines.

typedef __SVInt8_t fixed_int8_t __attribute__((arm_sve_vector_bits(BITS)));
typedef __SVInt32_t fixed_int32_t __attribute__((arm_sve_vector_bits(BITS)));
typedef __SVUint32_t fixed_uint32_t __attribute__((arm_sve_vector_bits(BITS)));
typedef __SVFloat64_t fixed_float64_t __attribute__((arm_sve_vector_bits(BITS)));
typedef __SVBool_t fixed_bool_t __attribute__((arm_sve_vector_bits(BITS)));

// The sizeless types are passed and returned in their own registers without
// coercion.

void arg_svint8(__SVInt8_t v) {}
// CHECK: define{{.*}} void @arg_svint8(<vscale x 16 x i8> %{{.*}})

void arg_svint32(__SVInt32_t v) {}
// CHECK: define{{.*}} void @arg_svint32(<vscale x 4 x i32> %{{.*}})

void arg_svfloat64(__SVFloat64_t v) {}
// CHECK: define{{.*}} void @arg_svfloat64(<vscale x 2 x double> %{{.*}})

void arg_svbool(__SVBool_t p) {}
// CHECK: define{{.*}} void @arg_svbool(<vscale x 16 x i1> %{{.*}})

void arg_svcount(__SVCount_t c) {}
// CHECK: define{{.*}} void @arg_svcount(target("aarch64.svcount") %{{.*}})

__SVInt32_t ret_svint32(__SVInt32_t v) { return v; }
// CHECK: define{{.*}} <vscale x 4 x i32> @ret_svint32(<vscale x 4 x i32> %{{.*}})

__SVBool_t ret_svbool(__SVBool_t p) { return p; }
// CHECK: define{{.*}} <vscale x 16 x i1> @ret_svbool(<vscale x 16 x i1> %{{.*}})

__SVCount_t ret_svcount(__SVCount_t c) { return c; }
// CHECK: define{{.*}} target("aarch64.svcount") @ret_svcount(target("aarch64.svcount") %{{.*}})

// The fixed-length types are coerced to the sizeless type that occupies the
// same register, so the element count of the coerced type depends only on the
// element size, not on the vector length the type was declared with.

void arg_fixed_int8(fixed_int8_t v) {}
// CHECK: define{{.*}} void @arg_fixed_int8(<vscale x 16 x i8> noundef %{{.*}})

void arg_fixed_int32(fixed_int32_t v) {}
// CHECK: define{{.*}} void @arg_fixed_int32(<vscale x 4 x i32> noundef %{{.*}})

void arg_fixed_uint32(fixed_uint32_t v) {}
// CHECK: define{{.*}} void @arg_fixed_uint32(<vscale x 4 x i32> noundef %{{.*}})

void arg_fixed_float64(fixed_float64_t v) {}
// CHECK: define{{.*}} void @arg_fixed_float64(<vscale x 2 x double> noundef %{{.*}})

void arg_fixed_bool(fixed_bool_t p) {}
// CHECK: define{{.*}} void @arg_fixed_bool(<vscale x 16 x i1> noundef %{{.*}})

fixed_int32_t ret_fixed_int32(fixed_int32_t v) { return v; }
// CHECK: define{{.*}} <vscale x 4 x i32> @ret_fixed_int32(<vscale x 4 x i32> noundef %{{.*}})

fixed_bool_t ret_fixed_bool(fixed_bool_t p) { return p; }
// CHECK: define{{.*}} <vscale x 16 x i1> @ret_fixed_bool(<vscale x 16 x i1> noundef %{{.*}})

// Sizeless and fixed-length types can be mixed in one signature.

void arg_mixed(__SVInt32_t a, __SVBool_t p, fixed_int32_t b) {}
// CHECK: define{{.*}} void @arg_mixed(<vscale x 4 x i32> %{{.*}}, <vscale x 16 x i1> %{{.*}}, <vscale x 4 x i32> noundef %{{.*}})
