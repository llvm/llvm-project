// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown-opencl -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned long ulong;
typedef unsigned int uint;

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

// CHECK-LABEL: @test_div_scale_f32
// CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { float, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { float, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i32
// CHECK: store i32 [[FLAGEXT]]
void test_div_scale_f32(global float* out, global int* flagout, float a, float b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scalef(a, b, true, &flag);
  *flagout = flag;
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
// CHECK: call i64 @llvm.amdgcn.icmp.i32(i32 %a, i32 %b, i32 32)
void test_sicmp_i32(global ulong* out, int a, int b)
{
  *out = __builtin_amdgcn_sicmp(a, b, 32);
}

// CHECK-LABEL: @test_uicmp_i32
// CHECK: call i64 @llvm.amdgcn.icmp.i32(i32 %a, i32 %b, i32 32)
void test_uicmp_i32(global ulong* out, uint a, uint b)
{
  *out = __builtin_amdgcn_uicmp(a, b, 32);
}

// CHECK-LABEL: @test_sicmp_i64
// CHECK: call i64 @llvm.amdgcn.icmp.i64(i64 %a, i64 %b, i32 38)
void test_sicmp_i64(global ulong* out, long a, long b)
{
  *out = __builtin_amdgcn_sicmpl(a, b, 39-1);
}

// CHECK-LABEL: @test_uicmp_i64
// CHECK: call i64 @llvm.amdgcn.icmp.i64(i64 %a, i64 %b, i32 35)
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
// CHECK: call i64 @llvm.amdgcn.fcmp.f32(float %a, float %b, i32 5)
void test_fcmp_f32(global ulong* out, float a, float b)
{
  *out = __builtin_amdgcn_fcmpf(a, b, 5);
}

// CHECK-LABEL: @test_fcmp_f64
// CHECK: call i64 @llvm.amdgcn.fcmp.f64(double %a, double %b, i32 6)
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

// CHECK-LABEL: @test_s_memtime
// CHECK: call i64 @llvm.amdgcn.s.memtime()
void test_s_memtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memtime();
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
// CHECK: call i64 @llvm.read_register.i64(metadata ![[EXEC:[0-9]+]]) #[[READ_EXEC_ATTRS:[0-9]+]]
void test_read_exec(global ulong* out) {
  *out = __builtin_amdgcn_read_exec();
}

// CHECK: declare i64 @llvm.read_register.i64(metadata) #[[NOUNWIND_READONLY:[0-9]+]]

// CHECK-LABEL: @test_read_exec_lo(
// CHECK: call i32 @llvm.read_register.i32(metadata ![[EXEC_LO:[0-9]+]]) #[[READ_EXEC_ATTRS]]
void test_read_exec_lo(global uint* out) {
  *out = __builtin_amdgcn_read_exec_lo();
}

// CHECK-LABEL: @test_read_exec_hi(
// CHECK: call i32 @llvm.read_register.i32(metadata ![[EXEC_HI:[0-9]+]]) #[[READ_EXEC_ATTRS]]
void test_read_exec_hi(global uint* out) {
  *out = __builtin_amdgcn_read_exec_hi();
}

// CHECK-LABEL: @test_dispatch_ptr
// CHECK: call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
void test_dispatch_ptr(__attribute__((address_space(2))) unsigned char ** out)
{
  *out = __builtin_amdgcn_dispatch_ptr();
}

// CHECK-LABEL: @test_kernarg_segment_ptr
// CHECK: call i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr()
void test_kernarg_segment_ptr(__attribute__((address_space(2))) unsigned char ** out)
{
  *out = __builtin_amdgcn_kernarg_segment_ptr();
}

// CHECK-LABEL: @test_implicitarg_ptr
// CHECK: call i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr()
void test_implicitarg_ptr(__attribute__((address_space(2))) unsigned char ** out)
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
// CHECK: tail call i32 @llvm.amdgcn.workitem.id.x(), !range [[WI_RANGE:![0-9]*]]
// CHECK: tail call i32 @llvm.amdgcn.workitem.id.y(), !range [[WI_RANGE]]
// CHECK: tail call i32 @llvm.amdgcn.workitem.id.z(), !range [[WI_RANGE]]
void test_get_local_id(int d, global int *out)
{
	switch (d) {
	case 0: *out = __builtin_amdgcn_workitem_id_x(); break;
	case 1: *out = __builtin_amdgcn_workitem_id_y(); break;
	case 2: *out = __builtin_amdgcn_workitem_id_z(); break;
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

// CHECK-DAG: [[WI_RANGE]] = !{i32 0, i32 1024}
// CHECK-DAG: attributes #[[NOUNWIND_READONLY:[0-9]+]] = { nounwind readonly }
// CHECK-DAG: attributes #[[READ_EXEC_ATTRS]] = { convergent }
// CHECK-DAG: ![[EXEC]] = !{!"exec"}
// CHECK-DAG: ![[EXEC_LO]] = !{!"exec_lo"}
// CHECK-DAG: ![[EXEC_HI]] = !{!"exec_hi"}
