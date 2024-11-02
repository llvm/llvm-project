// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -cl-std=CL2.0 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck -enable-var-scope %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef ushort __attribute__((ext_vector_type(2))) ushort2;
typedef uint __attribute__((ext_vector_type(4))) uint4;

// CHECK-LABEL: @test_div_scale_f64
// CHECK: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { double, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { double, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i32
// CHECK: store i32 [[FLAGEXT]]
void test_div_scale_f64(global double* out, global int* flagout, double a, double b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scale(a, b, true, &flag);
  *flagout = flag;
}

// CHECK-LABEL: @test_div_scale_f32(
// CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { float, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { float, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i8
// CHECK: store i8 [[FLAGEXT]]
void test_div_scale_f32(global float* out, global bool* flagout, float a, float b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scalef(a, b, true, &flag);
  *flagout = flag;
}

// CHECK-LABEL: @test_div_scale_f32_global_ptr(
// CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { float, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { float, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i8
// CHECK: store i8 [[FLAGEXT]]
void test_div_scale_f32_global_ptr(global float* out, global int* flagout, float a, float b, global bool* flag)
{
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
}

// CHECK-LABEL: @test_div_scale_f32_generic_ptr(
// CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { float, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { float, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i8
// CHECK: store i8 [[FLAGEXT]]
void test_div_scale_f32_generic_ptr(global float* out, global int* flagout, float a, float b, global bool* flag_arg)
{
  generic bool* flag = flag_arg;
  *out = __builtin_amdgcn_div_scalef(a, b, true, flag);
}

// CHECK-LABEL: @test_div_fmas_f32
// CHECK: call float @llvm.amdgcn.div.fmas.f32
void test_div_fmas_f32(global float* out, float a, float b, float c, int d)
{
  *out = __builtin_amdgcn_div_fmasf(a, b, c, d);
}

// CHECK-LABEL: @test_div_fmas_f64
// CHECK: call double @llvm.amdgcn.div.fmas.f64
void test_div_fmas_f64(global double* out, double a, double b, double c, int d)
{
  *out = __builtin_amdgcn_div_fmas(a, b, c, d);
}

// CHECK-LABEL: @test_div_fixup_f32
// CHECK: call float @llvm.amdgcn.div.fixup.f32
void test_div_fixup_f32(global float* out, float a, float b, float c)
{
  *out = __builtin_amdgcn_div_fixupf(a, b, c);
}

// CHECK-LABEL: @test_div_fixup_f64
// CHECK: call double @llvm.amdgcn.div.fixup.f64
void test_div_fixup_f64(global double* out, double a, double b, double c)
{
  *out = __builtin_amdgcn_div_fixup(a, b, c);
}

// CHECK-LABEL: @test_trig_preop_f32
// CHECK: call float @llvm.amdgcn.trig.preop.f32
void test_trig_preop_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgcn_trig_preopf(a, b);
}

// CHECK-LABEL: @test_trig_preop_f64
// CHECK: call double @llvm.amdgcn.trig.preop.f64
void test_trig_preop_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgcn_trig_preop(a, b);
}

// CHECK-LABEL: @test_rcp_f32
// CHECK: call float @llvm.amdgcn.rcp.f32
void test_rcp_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_rcpf(a);
}

// CHECK-LABEL: @test_rcp_f64
// CHECK: call double @llvm.amdgcn.rcp.f64
void test_rcp_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_rcp(a);
}

// CHECK-LABEL: @test_sqrt_f32
// CHECK: call float @llvm.amdgcn.sqrt.f32
void test_sqrt_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_sqrtf(a);
}

// CHECK-LABEL: @test_sqrt_f64
// CHECK: call double @llvm.amdgcn.sqrt.f64
void test_sqrt_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_sqrt(a);
}

// CHECK-LABEL: @test_rsq_f32
// CHECK: call float @llvm.amdgcn.rsq.f32
void test_rsq_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_rsqf(a);
}

// CHECK-LABEL: @test_rsq_f64
// CHECK: call double @llvm.amdgcn.rsq.f64
void test_rsq_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_rsq(a);
}

// CHECK-LABEL: @test_rsq_clamp_f32
// CHECK: call float @llvm.amdgcn.rsq.clamp.f32
void test_rsq_clamp_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_rsq_clampf(a);
}

// CHECK-LABEL: @test_rsq_clamp_f64
// CHECK: call double @llvm.amdgcn.rsq.clamp.f64
void test_rsq_clamp_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_rsq_clamp(a);
}

// CHECK-LABEL: @test_sin_f32
// CHECK: call float @llvm.amdgcn.sin.f32
void test_sin_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_sinf(a);
}

// CHECK-LABEL: @test_cos_f32
// CHECK: call float @llvm.amdgcn.cos.f32
void test_cos_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_cosf(a);
}

// CHECK-LABEL: @test_log_clamp_f32
// CHECK: call float @llvm.amdgcn.log.clamp.f32
void test_log_clamp_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_log_clampf(a);
}

// CHECK-LABEL: @test_ldexp_f32
// CHECK: call float @llvm.amdgcn.ldexp.f32
void test_ldexp_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgcn_ldexpf(a, b);
}

// CHECK-LABEL: @test_ldexp_f64
// CHECK: call double @llvm.amdgcn.ldexp.f64
void test_ldexp_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgcn_ldexp(a, b);
}

// CHECK-LABEL: @test_frexp_mant_f32
// CHECK: call float @llvm.amdgcn.frexp.mant.f32
void test_frexp_mant_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_frexp_mantf(a);
}

// CHECK-LABEL: @test_frexp_mant_f64
// CHECK: call double @llvm.amdgcn.frexp.mant.f64
void test_frexp_mant_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_frexp_mant(a);
}

// CHECK-LABEL: @test_frexp_exp_f32
// CHECK: call i32 @llvm.amdgcn.frexp.exp.i32.f32
void test_frexp_exp_f32(global int* out, float a)
{
  *out = __builtin_amdgcn_frexp_expf(a);
}

// CHECK-LABEL: @test_frexp_exp_f64
// CHECK: call i32 @llvm.amdgcn.frexp.exp.i32.f64
void test_frexp_exp_f64(global int* out, double a)
{
  *out = __builtin_amdgcn_frexp_exp(a);
}

// CHECK-LABEL: @test_fract_f32
// CHECK: call float @llvm.amdgcn.fract.f32
void test_fract_f32(global int* out, float a)
{
  *out = __builtin_amdgcn_fractf(a);
}

// CHECK-LABEL: @test_fract_f64
// CHECK: call double @llvm.amdgcn.fract.f64
void test_fract_f64(global int* out, double a)
{
  *out = __builtin_amdgcn_fract(a);
}

// CHECK-LABEL: @test_lerp
// CHECK: call i32 @llvm.amdgcn.lerp
void test_lerp(global int* out, int a, int b, int c)
{
  *out = __builtin_amdgcn_lerp(a, b, c);
}

// CHECK-LABEL: @test_sicmp_i32
// CHECK: call i64 @llvm.amdgcn.icmp.i64.i32(i32 %a, i32 %b, i32 32)
void test_sicmp_i32(global ulong* out, int a, int b)
{
  *out = __builtin_amdgcn_sicmp(a, b, 32);
}

// CHECK-LABEL: @test_uicmp_i32
// CHECK: call i64 @llvm.amdgcn.icmp.i64.i32(i32 %a, i32 %b, i32 32)
void test_uicmp_i32(global ulong* out, uint a, uint b)
{
  *out = __builtin_amdgcn_uicmp(a, b, 32);
}

// CHECK-LABEL: @test_sicmp_i64
// CHECK: call i64 @llvm.amdgcn.icmp.i64.i64(i64 %a, i64 %b, i32 38)
void test_sicmp_i64(global ulong* out, long a, long b)
{
  *out = __builtin_amdgcn_sicmpl(a, b, 39-1);
}

// CHECK-LABEL: @test_uicmp_i64
// CHECK: call i64 @llvm.amdgcn.icmp.i64.i64(i64 %a, i64 %b, i32 35)
void test_uicmp_i64(global ulong* out, ulong a, ulong b)
{
  *out = __builtin_amdgcn_uicmpl(a, b, 30+5);
}

// CHECK-LABEL: @test_ds_swizzle
// CHECK: call i32 @llvm.amdgcn.ds.swizzle(i32 %a, i32 32)
void test_ds_swizzle(global int* out, int a)
{
  *out = __builtin_amdgcn_ds_swizzle(a, 32);
}

// CHECK-LABEL: @test_ds_permute
// CHECK: call i32 @llvm.amdgcn.ds.permute(i32 %a, i32 %b)
void test_ds_permute(global int* out, int a, int b)
{
  out[0] = __builtin_amdgcn_ds_permute(a, b);
}

// CHECK-LABEL: @test_ds_bpermute
// CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 %b)
void test_ds_bpermute(global int* out, int a, int b)
{
  *out = __builtin_amdgcn_ds_bpermute(a, b);
}

// CHECK-LABEL: @test_readfirstlane
// CHECK: call i32 @llvm.amdgcn.readfirstlane(i32 %a)
void test_readfirstlane(global int* out, int a)
{
  *out = __builtin_amdgcn_readfirstlane(a);
}

// CHECK-LABEL: @test_readlane
// CHECK: call i32 @llvm.amdgcn.readlane(i32 %a, i32 %b)
void test_readlane(global int* out, int a, int b)
{
  *out = __builtin_amdgcn_readlane(a, b);
}

// CHECK-LABEL: @test_fcmp_f32
// CHECK: call i64 @llvm.amdgcn.fcmp.i64.f32(float %a, float %b, i32 5)
void test_fcmp_f32(global ulong* out, float a, float b)
{
  *out = __builtin_amdgcn_fcmpf(a, b, 5);
}

// CHECK-LABEL: @test_fcmp_f64
// CHECK: call i64 @llvm.amdgcn.fcmp.i64.f64(double %a, double %b, i32 6)
void test_fcmp_f64(global ulong* out, double a, double b)
{
  *out = __builtin_amdgcn_fcmp(a, b, 3+3);
}

// CHECK-LABEL: @test_class_f32
// CHECK: call i1 @llvm.amdgcn.class.f32
void test_class_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgcn_classf(a, b);
}

// CHECK-LABEL: @test_class_f64
// CHECK: call i1 @llvm.amdgcn.class.f64
void test_class_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgcn_class(a, b);
}

// CHECK-LABEL: @test_buffer_wbinvl1
// CHECK: call void @llvm.amdgcn.buffer.wbinvl1(
void test_buffer_wbinvl1()
{
  __builtin_amdgcn_buffer_wbinvl1();
}

// CHECK-LABEL: @test_s_dcache_inv
// CHECK: call void @llvm.amdgcn.s.dcache.inv(
void test_s_dcache_inv()
{
  __builtin_amdgcn_s_dcache_inv();
}

// CHECK-LABEL: @test_s_waitcnt
// CHECK: call void @llvm.amdgcn.s.waitcnt(
void test_s_waitcnt()
{
  __builtin_amdgcn_s_waitcnt(0);
}

// CHECK-LABEL: @test_s_sendmsg
// CHECK: call void @llvm.amdgcn.s.sendmsg(
void test_s_sendmsg()
{
  __builtin_amdgcn_s_sendmsg(1, 0);
}

// CHECK-LABEL: @test_s_sendmsg_var
// CHECK: call void @llvm.amdgcn.s.sendmsg(
void test_s_sendmsg_var(int in)
{
  __builtin_amdgcn_s_sendmsg(1, in);
}

// CHECK-LABEL: @test_s_sendmsghalt
// CHECK: call void @llvm.amdgcn.s.sendmsghalt(
void test_s_sendmsghalt()
{
  __builtin_amdgcn_s_sendmsghalt(1, 0);
}

// CHECK-LABEL: @test_s_sendmsghalt
// CHECK: call void @llvm.amdgcn.s.sendmsghalt(
void test_s_sendmsghalt_var(int in)
{
  __builtin_amdgcn_s_sendmsghalt(1, in);
}

// CHECK-LABEL: @test_s_barrier
// CHECK: call void @llvm.amdgcn.s.barrier(
void test_s_barrier()
{
  __builtin_amdgcn_s_barrier();
}

// CHECK-LABEL: @test_wave_barrier
// CHECK: call void @llvm.amdgcn.wave.barrier(
void test_wave_barrier()
{
  __builtin_amdgcn_wave_barrier();
}

// CHECK-LABEL: @test_sched_barrier
// CHECK: call void @llvm.amdgcn.sched.barrier(i32 0)
// CHECK: call void @llvm.amdgcn.sched.barrier(i32 1)
// CHECK: call void @llvm.amdgcn.sched.barrier(i32 4)
// CHECK: call void @llvm.amdgcn.sched.barrier(i32 15)
void test_sched_barrier()
{
  __builtin_amdgcn_sched_barrier(0);
  __builtin_amdgcn_sched_barrier(1);
  __builtin_amdgcn_sched_barrier(4);
  __builtin_amdgcn_sched_barrier(15);
}

// CHECK-LABEL: @test_s_sleep
// CHECK: call void @llvm.amdgcn.s.sleep(i32 1)
// CHECK: call void @llvm.amdgcn.s.sleep(i32 15)
void test_s_sleep()
{
  __builtin_amdgcn_s_sleep(1);
  __builtin_amdgcn_s_sleep(15);
}

// CHECK-LABEL: @test_s_incperflevel
// CHECK: call void @llvm.amdgcn.s.incperflevel(i32 1)
// CHECK: call void @llvm.amdgcn.s.incperflevel(i32 15)
void test_s_incperflevel()
{
  __builtin_amdgcn_s_incperflevel(1);
  __builtin_amdgcn_s_incperflevel(15);
}

// CHECK-LABEL: @test_s_decperflevel
// CHECK: call void @llvm.amdgcn.s.decperflevel(i32 1)
// CHECK: call void @llvm.amdgcn.s.decperflevel(i32 15)
void test_s_decperflevel()
{
  __builtin_amdgcn_s_decperflevel(1);
  __builtin_amdgcn_s_decperflevel(15);
}

// CHECK-LABEL: @test_s_setprio
// CHECK: call void @llvm.amdgcn.s.setprio(i16 0)
// CHECK: call void @llvm.amdgcn.s.setprio(i16 3)
void test_s_setprio()
{
  __builtin_amdgcn_s_setprio(0);
  __builtin_amdgcn_s_setprio(3);
}

// CHECK-LABEL: @test_cubeid(
// CHECK: call float @llvm.amdgcn.cubeid(float %a, float %b, float %c)
void test_cubeid(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubeid(a, b, c);
}

// CHECK-LABEL: @test_cubesc(
// CHECK: call float @llvm.amdgcn.cubesc(float %a, float %b, float %c)
void test_cubesc(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubesc(a, b, c);
}

// CHECK-LABEL: @test_cubetc(
// CHECK: call float @llvm.amdgcn.cubetc(float %a, float %b, float %c)
void test_cubetc(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubetc(a, b, c);
}

// CHECK-LABEL: @test_cubema(
// CHECK: call float @llvm.amdgcn.cubema(float %a, float %b, float %c)
void test_cubema(global float* out, float a, float b, float c) {
  *out = __builtin_amdgcn_cubema(a, b, c);
}

// CHECK-LABEL: @test_read_exec(
// CHECK: call i64 @llvm.read_register.i64(metadata ![[$EXEC:[0-9]+]]) #[[$READ_EXEC_ATTRS:[0-9]+]]
void test_read_exec(global ulong* out) {
  *out = __builtin_amdgcn_read_exec();
}

// CHECK: declare i64 @llvm.read_register.i64(metadata) #[[$NOUNWIND_READONLY:[0-9]+]]

// CHECK-LABEL: @test_read_exec_lo(
// CHECK: call i32 @llvm.read_register.i32(metadata ![[$EXEC_LO:[0-9]+]]) #[[$READ_EXEC_ATTRS]]
void test_read_exec_lo(global uint* out) {
  *out = __builtin_amdgcn_read_exec_lo();
}

// CHECK-LABEL: @test_read_exec_hi(
// CHECK: call i32 @llvm.read_register.i32(metadata ![[$EXEC_HI:[0-9]+]]) #[[$READ_EXEC_ATTRS]]
void test_read_exec_hi(global uint* out) {
  *out = __builtin_amdgcn_read_exec_hi();
}

// CHECK-LABEL: @test_dispatch_ptr
// CHECK: call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
void test_dispatch_ptr(__constant unsigned char ** out)
{
  *out = __builtin_amdgcn_dispatch_ptr();
}

// CHECK-LABEL: @test_queue_ptr
// CHECK: call i8 addrspace(4)* @llvm.amdgcn.queue.ptr()
void test_queue_ptr(__constant unsigned char ** out)
{
  *out = __builtin_amdgcn_queue_ptr();
}

// CHECK-LABEL: @test_kernarg_segment_ptr
// CHECK: call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
void test_kernarg_segment_ptr(__constant unsigned char ** out)
{
  *out = __builtin_amdgcn_kernarg_segment_ptr();
}

// CHECK-LABEL: @test_implicitarg_ptr
// CHECK: call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
void test_implicitarg_ptr(__constant unsigned char ** out)
{
  *out = __builtin_amdgcn_implicitarg_ptr();
}

// CHECK-LABEL: @test_get_group_id(
// CHECK: tail call i32 @llvm.amdgcn.workgroup.id.x()
// CHECK: tail call i32 @llvm.amdgcn.workgroup.id.y()
// CHECK: tail call i32 @llvm.amdgcn.workgroup.id.z()
void test_get_group_id(int d, global int *out)
{
	switch (d) {
	case 0: *out = __builtin_amdgcn_workgroup_id_x(); break;
	case 1: *out = __builtin_amdgcn_workgroup_id_y(); break;
	case 2: *out = __builtin_amdgcn_workgroup_id_z(); break;
	default: *out = 0;
	}
}

// CHECK-LABEL: @test_s_getreg(
// CHECK: tail call i32 @llvm.amdgcn.s.getreg(i32 0)
// CHECK: tail call i32 @llvm.amdgcn.s.getreg(i32 1)
// CHECK: tail call i32 @llvm.amdgcn.s.getreg(i32 65535)
void test_s_getreg(volatile global uint *out)
{
  *out = __builtin_amdgcn_s_getreg(0);
  *out = __builtin_amdgcn_s_getreg(1);
  *out = __builtin_amdgcn_s_getreg(65535);
}

// CHECK-LABEL: @test_get_local_id(
// CHECK: tail call i32 @llvm.amdgcn.workitem.id.x(), !range [[$WI_RANGE:![0-9]*]]
// CHECK: tail call i32 @llvm.amdgcn.workitem.id.y(), !range [[$WI_RANGE]]
// CHECK: tail call i32 @llvm.amdgcn.workitem.id.z(), !range [[$WI_RANGE]]
void test_get_local_id(int d, global int *out)
{
	switch (d) {
	case 0: *out = __builtin_amdgcn_workitem_id_x(); break;
	case 1: *out = __builtin_amdgcn_workitem_id_y(); break;
	case 2: *out = __builtin_amdgcn_workitem_id_z(); break;
	default: *out = 0;
	}
}

// CHECK-LABEL: @test_get_workgroup_size(
// CHECK: call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i64 4
// CHECK: load i16, i16 addrspace(4)* %{{.*}}, align 4, !range [[$WS_RANGE:![0-9]*]], !invariant.load
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i64 6
// CHECK: load i16, i16 addrspace(4)* %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i64 8
// CHECK: load i16, i16 addrspace(4)* %{{.*}}, align 4, !range [[$WS_RANGE:![0-9]*]], !invariant.load
void test_get_workgroup_size(int d, global int *out)
{
	switch (d) {
	case 0: *out = __builtin_amdgcn_workgroup_size_x() + 1; break;
	case 1: *out = __builtin_amdgcn_workgroup_size_y(); break;
	case 2: *out = __builtin_amdgcn_workgroup_size_z(); break;
	default: *out = 0;
	}
}

// CHECK-LABEL: @test_get_grid_size(
// CHECK: call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i64 12
// CHECK: load i32, i32 addrspace(4)* %{{.*}}, align 4, !invariant.load
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i64 16
// CHECK: load i32, i32 addrspace(4)* %{{.*}}, align 4, !invariant.load
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i64 20
// CHECK: load i32, i32 addrspace(4)* %{{.*}}, align 4, !invariant.load
void test_get_grid_size(int d, global int *out)
{
	switch (d) {
	case 0: *out = __builtin_amdgcn_grid_size_x(); break;
	case 1: *out = __builtin_amdgcn_grid_size_y(); break;
	case 2: *out = __builtin_amdgcn_grid_size_z(); break;
	default: *out = 0;
	}
}

// CHECK-LABEL: @test_fmed3_f32
// CHECK: call float @llvm.amdgcn.fmed3.f32(
void test_fmed3_f32(global float* out, float a, float b, float c)
{
  *out = __builtin_amdgcn_fmed3f(a, b, c);
}

// CHECK-LABEL: @test_s_getpc
// CHECK: call i64 @llvm.amdgcn.s.getpc()
void test_s_getpc(global ulong* out)
{
  *out = __builtin_amdgcn_s_getpc();
}

// CHECK-LABEL: @test_ds_append_lds(
// CHECK: call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %ptr, i1 false)
kernel void test_ds_append_lds(global int* out, local int* ptr) {
  *out = __builtin_amdgcn_ds_append(ptr);
}

// CHECK-LABEL: @test_ds_consume_lds(
// CHECK: call i32 @llvm.amdgcn.ds.consume.p3i32(i32 addrspace(3)* %ptr, i1 false)
kernel void test_ds_consume_lds(global int* out, local int* ptr) {
  *out = __builtin_amdgcn_ds_consume(ptr);
}

// CHECK-LABEL: @test_gws_init(
// CHECK: call void @llvm.amdgcn.ds.gws.init(i32 %value, i32 %id)
kernel void test_gws_init(uint value, uint id) {
  __builtin_amdgcn_ds_gws_init(value, id);
}

// CHECK-LABEL: @test_gws_barrier(
// CHECK: call void @llvm.amdgcn.ds.gws.barrier(i32 %value, i32 %id)
kernel void test_gws_barrier(uint value, uint id) {
  __builtin_amdgcn_ds_gws_barrier(value, id);
}

// CHECK-LABEL: @test_gws_sema_v(
// CHECK: call void @llvm.amdgcn.ds.gws.sema.v(i32 %id)
kernel void test_gws_sema_v(uint id) {
  __builtin_amdgcn_ds_gws_sema_v(id);
}

// CHECK-LABEL: @test_gws_sema_br(
// CHECK: call void @llvm.amdgcn.ds.gws.sema.br(i32 %value, i32 %id)
kernel void test_gws_sema_br(uint value, uint id) {
  __builtin_amdgcn_ds_gws_sema_br(value, id);
}

// CHECK-LABEL: @test_gws_sema_p(
// CHECK: call void @llvm.amdgcn.ds.gws.sema.p(i32 %id)
kernel void test_gws_sema_p(uint id) {
  __builtin_amdgcn_ds_gws_sema_p(id);
}

// CHECK-LABEL: @test_mbcnt_lo(
// CHECK: call i32 @llvm.amdgcn.mbcnt.lo(i32 %src0, i32 %src1)
kernel void test_mbcnt_lo(global uint* out, uint src0, uint src1) {
  *out = __builtin_amdgcn_mbcnt_lo(src0, src1);
}

// CHECK-LABEL: @test_mbcnt_hi(
// CHECK: call i32 @llvm.amdgcn.mbcnt.hi(i32 %src0, i32 %src1)
kernel void test_mbcnt_hi(global uint* out, uint src0, uint src1) {
  *out = __builtin_amdgcn_mbcnt_hi(src0, src1);
}

// CHECK-LABEL: @test_alignbit(
// CHECK: tail call i32 @llvm.fshr.i32(i32 %src0, i32 %src1, i32 %src2)
kernel void test_alignbit(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_alignbit(src0, src1, src2);
}

// CHECK-LABEL: @test_alignbyte(
// CHECK: tail call i32 @llvm.amdgcn.alignbyte(i32 %src0, i32 %src1, i32 %src2)
kernel void test_alignbyte(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_alignbyte(src0, src1, src2);
}

// CHECK-LABEL: @test_ubfe(
// CHECK: tail call i32 @llvm.amdgcn.ubfe.i32(i32 %src0, i32 %src1, i32 %src2)
kernel void test_ubfe(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_ubfe(src0, src1, src2);
}

// CHECK-LABEL: @test_sbfe(
// CHECK: tail call i32 @llvm.amdgcn.sbfe.i32(i32 %src0, i32 %src1, i32 %src2)
kernel void test_sbfe(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_sbfe(src0, src1, src2);
}

// CHECK-LABEL: @test_cvt_pkrtz(
// CHECK: tail call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %src0, float %src1)
kernel void test_cvt_pkrtz(global half2* out, float src0, float src1) {
  *out = __builtin_amdgcn_cvt_pkrtz(src0, src1);
}

// CHECK-LABEL: @test_cvt_pknorm_i16(
// CHECK: tail call <2 x i16> @llvm.amdgcn.cvt.pknorm.i16(float %src0, float %src1)
kernel void test_cvt_pknorm_i16(global short2* out, float src0, float src1) {
  *out = __builtin_amdgcn_cvt_pknorm_i16(src0, src1);
}

// CHECK-LABEL: @test_cvt_pknorm_u16(
// CHECK: tail call <2 x i16> @llvm.amdgcn.cvt.pknorm.u16(float %src0, float %src1)
kernel void test_cvt_pknorm_u16(global ushort2* out, float src0, float src1) {
  *out = __builtin_amdgcn_cvt_pknorm_u16(src0, src1);
}

// CHECK-LABEL: @test_cvt_pk_i16(
// CHECK: tail call <2 x i16> @llvm.amdgcn.cvt.pk.i16(i32 %src0, i32 %src1)
kernel void test_cvt_pk_i16(global short2* out, int src0, int src1) {
  *out = __builtin_amdgcn_cvt_pk_i16(src0, src1);
}

// CHECK-LABEL: @test_cvt_pk_u16(
// CHECK: tail call <2 x i16> @llvm.amdgcn.cvt.pk.u16(i32 %src0, i32 %src1)
kernel void test_cvt_pk_u16(global ushort2* out, uint src0, uint src1) {
  *out = __builtin_amdgcn_cvt_pk_u16(src0, src1);
}

// CHECK-LABEL: @test_cvt_pk_u8_f32
// CHECK: tail call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src0, i32 %src1, i32 %src2)
kernel void test_cvt_pk_u8_f32(global uint* out, float src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_cvt_pk_u8_f32(src0, src1, src2);
}

// CHECK-LABEL: @test_sad_u8(
// CHECK: tail call i32 @llvm.amdgcn.sad.u8(i32 %src0, i32 %src1, i32 %src2)
kernel void test_sad_u8(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_sad_u8(src0, src1, src2);
}

// CHECK-LABEL: test_msad_u8(
// CHECK: call i32 @llvm.amdgcn.msad.u8(i32 %src0, i32 %src1, i32 %src2)
kernel void test_msad_u8(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_msad_u8(src0, src1, src2);
}

// CHECK-LABEL: test_sad_hi_u8(
// CHECK: call i32 @llvm.amdgcn.sad.hi.u8(i32 %src0, i32 %src1, i32 %src2)
kernel void test_sad_hi_u8(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_sad_hi_u8(src0, src1, src2);
}

// CHECK-LABEL: @test_sad_u16(
// CHECK: call i32 @llvm.amdgcn.sad.u16(i32 %src0, i32 %src1, i32 %src2)
kernel void test_sad_u16(global uint* out, uint src0, uint src1, uint src2) {
  *out = __builtin_amdgcn_sad_u16(src0, src1, src2);
}

// CHECK-LABEL: @test_qsad_pk_u16_u8(
// CHECK: call i64 @llvm.amdgcn.qsad.pk.u16.u8(i64 %src0, i32 %src1, i64 %src2)
kernel void test_qsad_pk_u16_u8(global ulong* out, ulong src0, uint src1, ulong src2) {
  *out = __builtin_amdgcn_qsad_pk_u16_u8(src0, src1, src2);
}

// CHECK-LABEL: @test_mqsad_pk_u16_u8(
// CHECK: call i64 @llvm.amdgcn.mqsad.pk.u16.u8(i64 %src0, i32 %src1, i64 %src2)
kernel void test_mqsad_pk_u16_u8(global ulong* out, ulong src0, uint src1, ulong src2) {
  *out = __builtin_amdgcn_mqsad_pk_u16_u8(src0, src1, src2);
}

// CHECK-LABEL: test_mqsad_u32_u8(
// CHECK: call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %src0, i32 %src1, <4 x i32> %src2)
kernel void test_mqsad_u32_u8(global uint4* out, ulong src0, uint src1, uint4 src2) {
  *out = __builtin_amdgcn_mqsad_u32_u8(src0, src1, src2);
}

// CHECK-LABEL: test_s_setreg(
// CHECK: call void @llvm.amdgcn.s.setreg(i32 8193, i32 %val)
kernel void test_s_setreg(uint val) {
  __builtin_amdgcn_s_setreg(8193, val);
}

// CHECK-DAG: [[$WI_RANGE]] = !{i32 0, i32 1024}
// CHECK-DAG: [[$WS_RANGE]] = !{i16 1, i16 1025}
// CHECK-DAG: attributes #[[$NOUNWIND_READONLY:[0-9]+]] = { nofree nounwind readonly }
// CHECK-DAG: attributes #[[$READ_EXEC_ATTRS]] = { convergent }
// CHECK-DAG: ![[$EXEC]] = !{!"exec"}
// CHECK-DAG: ![[$EXEC_LO]] = !{!"exec_lo"}
// CHECK-DAG: ![[$EXEC_HI]] = !{!"exec_hi"}
