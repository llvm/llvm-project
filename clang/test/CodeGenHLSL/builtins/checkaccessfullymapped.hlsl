// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN: dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN: -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,DX

// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN: spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN: -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,SPV

// CHECK: test_check_access_fully_mapped
// DX: %[[X_ADDR:.*]] = alloca i32, align 4
// DX: store i32 %x, ptr %[[X_ADDR]], align 4
// DX: %[[X_LOAD:.*]] = load i32, ptr %[[X_ADDR]], align 4
// DX: %[[BoolCast:.*]] = icmp ne i32 %[[X_LOAD]], 0
// DX: ret i1 %[[BoolCast]]

// SPV: %[[X_ADDR:.*]] = alloca i32, align 4
// SPV: store i32 %x, ptr %[[X_ADDR]], align 4
// SPV: %[[X_LOAD:.*]] = loasd i32, ptr %[[X_ADDR]], align 4
// SPV: %[[CAFM:.*]] = call i1 @llvm.spv.check.access.fully.mapped(i32 %[[X_LOAD]])
// SPV: ret i1 %[[CAFM]]

bool test_check_access_fully_mapped(uint x)
{
	return CheckAccessFullyMapped(x);
}
