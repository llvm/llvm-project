// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple r600-unknown-unknown -target-cpu cypress -S -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @test_recipsqrt_ieee_f32
// CHECK: call float @llvm.r600.recipsqrt.ieee.f32
void test_recipsqrt_ieee_f32(global float* out, float a)
{
  *out = __builtin_r600_recipsqrt_ieeef(a);
}

#if cl_khr_fp64
// XCHECK-LABEL: @test_recipsqrt_ieee_f64
// XCHECK: call double @llvm.r600.recipsqrt.ieee.f64
void test_recipsqrt_ieee_f64(global double* out, double a)
{
  *out = __builtin_r600_recipsqrt_ieee(a);
}
#endif

// CHECK-LABEL: @test_implicitarg_ptr
// CHECK: call i8 addrspace(7)* @llvm.r600.implicitarg.ptr()
void test_implicitarg_ptr(__attribute__((address_space(7))) unsigned char ** out)
{
  *out = __builtin_r600_implicitarg_ptr();
}

// CHECK-LABEL: @test_get_group_id(
// CHECK: tail call i32 @llvm.r600.read.tgid.x()
// CHECK: tail call i32 @llvm.r600.read.tgid.y()
// CHECK: tail call i32 @llvm.r600.read.tgid.z()
void test_get_group_id(int d, global int *out)
{
	switch (d) {
	case 0: *out = __builtin_r600_read_tgid_x(); break;
	case 1: *out = __builtin_r600_read_tgid_y(); break;
	case 2: *out = __builtin_r600_read_tgid_z(); break;
	default: *out = 0;
	}
}

// CHECK-LABEL: @test_get_local_id(
// CHECK: tail call i32 @llvm.r600.read.tidig.x(), !range [[WI_RANGE:![0-9]*]]
// CHECK: tail call i32 @llvm.r600.read.tidig.y(), !range [[WI_RANGE]]
// CHECK: tail call i32 @llvm.r600.read.tidig.z(), !range [[WI_RANGE]]
void test_get_local_id(int d, global int *out)
{
	switch (d) {
	case 0: *out = __builtin_r600_read_tidig_x(); break;
	case 1: *out = __builtin_r600_read_tidig_y(); break;
	case 2: *out = __builtin_r600_read_tidig_z(); break;
	default: *out = 0;
	}
}

// CHECK-DAG: [[WI_RANGE]] = !{i32 0, i32 1024}
