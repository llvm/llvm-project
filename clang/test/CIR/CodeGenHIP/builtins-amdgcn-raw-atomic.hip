// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.42-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple amdgpu9.42-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple amdgpu9.42-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s

#define __device__ __attribute__((device))

typedef __attribute__((address_space(3))) float *float_lds_ptr;
typedef __attribute__((address_space(3))) double *double_lds_ptr;
typedef __attribute__((address_space(1))) float *float_global_ptr;
typedef __attribute__((address_space(1))) double *double_global_ptr;
typedef _Float16 half2 __attribute__((ext_vector_type(2)));
typedef short short2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) half2 *half2_lds_ptr;
typedef __attribute__((address_space(1))) half2 *half2_global_ptr;
typedef __attribute__((address_space(3))) short2 *short2_lds_ptr;
typedef __attribute__((address_space(1))) short2 *short2_global_ptr;

// CIR-LABEL: @_Z5inc32PVjj
// CIR: cir.atomic.fetch uinc_wrap seq_cst syncscope(device) fetch_first %{{.+}}, %{{.+}} volatile : (!cir.ptr<!u32i>, !u32i) -> !u32i {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z5inc32PVjj
// LLVM: atomicrmw volatile uinc_wrap ptr %{{.+}}, i32 %{{.+}} syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory
__device__ unsigned int inc32(unsigned int volatile *p, unsigned int v) {
  return __builtin_amdgcn_atomic_inc32(p, v, __ATOMIC_SEQ_CST, "agent");
}

// CIR-LABEL: @_Z17inc32_nonvolatilePjj
// CIR: cir.atomic.fetch uinc_wrap seq_cst syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!u32i>, !u32i) -> !u32i {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z17inc32_nonvolatilePjj
// LLVM: atomicrmw uinc_wrap ptr %{{.+}}, i32 %{{.+}} syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory
__device__ unsigned int inc32_nonvolatile(unsigned int *p, unsigned int v) {
  return __builtin_amdgcn_atomic_inc32(p, v, __ATOMIC_SEQ_CST, "agent");
}

// CIR-LABEL: @_Z5inc64PVmm
// CIR: cir.atomic.fetch uinc_wrap seq_cst syncscope(device) fetch_first %{{.+}}, %{{.+}} volatile : (!cir.ptr<!u64i>, !u64i) -> !u64i {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z5inc64PVmm
// LLVM: atomicrmw volatile uinc_wrap ptr %{{.+}}, i64 %{{.+}} syncscope("agent") seq_cst, align 8, !amdgpu.no.fine.grained.memory
__device__ unsigned long inc64(unsigned long volatile *p, unsigned long v) {
  return __builtin_amdgcn_atomic_inc64(p, v, __ATOMIC_SEQ_CST, "agent");
}

// CIR-LABEL: @_Z5dec32PVjj
// CIR: cir.atomic.fetch udec_wrap seq_cst syncscope(workgroup) fetch_first %{{.+}}, %{{.+}} volatile : (!cir.ptr<!u32i>, !u32i) -> !u32i {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z5dec32PVjj
// LLVM: atomicrmw volatile udec_wrap ptr %{{.+}}, i32 %{{.+}} syncscope("workgroup") seq_cst, align 4, !amdgpu.no.fine.grained.memory
__device__ unsigned int dec32(unsigned int volatile *p, unsigned int v) {
  return __builtin_amdgcn_atomic_dec32(p, v, __ATOMIC_SEQ_CST, "workgroup");
}

// CIR-LABEL: @_Z5dec64PVmm
// CIR: cir.atomic.fetch udec_wrap seq_cst syncscope(system) fetch_first %{{.+}}, %{{.+}} volatile : (!cir.ptr<!u64i>, !u64i) -> !u64i {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z5dec64PVmm
// LLVM: atomicrmw volatile udec_wrap ptr %{{.+}}, i64 %{{.+}} seq_cst, align 8, !amdgpu.no.fine.grained.memory
__device__ unsigned long dec64(unsigned long volatile *p, unsigned long v) {
  return __builtin_amdgcn_atomic_dec64(p, v, __ATOMIC_SEQ_CST, "");
}

// CIR-LABEL: @_Z8ds_faddfPU3AS3ff
// CIR: cir.atomic.fetch add relaxed syncscope(system) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float, target_address_space(3)>, !cir.float) -> !cir.float {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z8ds_faddfPU3AS3ff
// LLVM: atomicrmw fadd ptr addrspace(3) %{{.+}}, float %{{.+}} monotonic, align 4
// LLVM-NOT: amdgpu.no.fine.grained.memory
__device__ float ds_faddf(float_lds_ptr p, float v) {
  return __builtin_amdgcn_ds_faddf(p, v, 0, 0, false);
}

// CIR-LABEL: @_Z17ds_faddf_volatilePU3AS3ff
// CIR: cir.atomic.fetch add relaxed syncscope(system) fetch_first %{{.+}}, %{{.+}} volatile : (!cir.ptr<!cir.float, target_address_space(3)>, !cir.float) -> !cir.float {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z17ds_faddf_volatilePU3AS3ff
// LLVM: atomicrmw volatile fadd ptr addrspace(3) %{{.+}}, float %{{.+}} monotonic, align 4
__device__ float ds_faddf_volatile(float_lds_ptr p, float v) {
  return __builtin_amdgcn_ds_faddf(p, v, 0, 0, true);
}

// CIR-LABEL: @_Z8ds_fminfPU3AS3ff
// CIR: cir.atomic.fetch min relaxed syncscope(system) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float, target_address_space(3)>, !cir.float) -> !cir.float {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z8ds_fminfPU3AS3ff
// LLVM: atomicrmw fmin ptr addrspace(3) %{{.+}}, float %{{.+}} monotonic, align 4
__device__ float ds_fminf(float_lds_ptr p, float v) {
  return __builtin_amdgcn_ds_fminf(p, v, 0, 0, false);
}

// CIR-LABEL: @_Z8ds_fmaxfPU3AS3ff
// CIR: cir.atomic.fetch max relaxed syncscope(system) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float, target_address_space(3)>, !cir.float) -> !cir.float {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z8ds_fmaxfPU3AS3ff
// LLVM: atomicrmw fmax ptr addrspace(3) %{{.+}}, float %{{.+}} monotonic, align 4
__device__ float ds_fmaxf(float_lds_ptr p, float v) {
  return __builtin_amdgcn_ds_fmaxf(p, v, 0, 0, false);
}

// CIR-LABEL: @_Z18ds_atomic_fadd_f32PU3AS3ff
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float, target_address_space(3)>, !cir.float) -> !cir.float {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z18ds_atomic_fadd_f32PU3AS3ff
// LLVM: atomicrmw fadd ptr addrspace(3) %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4
// LLVM-NOT: amdgpu.no.fine.grained.memory
__device__ float ds_atomic_fadd_f32(float_lds_ptr p, float v) {
  return __builtin_amdgcn_ds_atomic_fadd_f32(p, v);
}

// CIR-LABEL: @_Z18ds_atomic_fadd_f64PU3AS3dd
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double, target_address_space(3)>, !cir.double) -> !cir.double {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z18ds_atomic_fadd_f64PU3AS3dd
// LLVM: atomicrmw fadd ptr addrspace(3) %{{.+}}, double %{{.+}} syncscope("agent") monotonic, align 8
// LLVM-NOT: amdgpu.no.fine.grained.memory
__device__ double ds_atomic_fadd_f64(double_lds_ptr p, double v) {
  return __builtin_amdgcn_ds_atomic_fadd_f64(p, v);
}

// CIR-LABEL: @_Z22global_atomic_fadd_f32PU3AS1ff
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float, target_address_space(1)>, !cir.float) -> !cir.float {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z22global_atomic_fadd_f32PU3AS1ff
// LLVM: atomicrmw fadd ptr addrspace(1) %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!atomic.ignore.denormal.mode{{.*}}!amdgpu.no.fine.grained.memory
__device__ float global_atomic_fadd_f32(float_global_ptr p, float v) {
  return __builtin_amdgcn_global_atomic_fadd_f32(p, v);
}

// CIR-LABEL: @_Z22global_atomic_fadd_f64PU3AS1dd
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double, target_address_space(1)>, !cir.double) -> !cir.double {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z22global_atomic_fadd_f64PU3AS1dd
// LLVM: atomicrmw fadd ptr addrspace(1) %{{.+}}, double %{{.+}} syncscope("agent") monotonic, align 8, !amdgpu.no.fine.grained.memory
// LLVM-NOT: atomic.ignore.denormal.mode
__device__ double global_atomic_fadd_f64(double_global_ptr p, double v) {
  return __builtin_amdgcn_global_atomic_fadd_f64(p, v);
}

// CIR-LABEL: @_Z20flat_atomic_fadd_f32Pff
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z20flat_atomic_fadd_f32Pff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!atomic.ignore.denormal.mode{{.*}}!amdgpu.no.fine.grained.memory
__device__ float flat_atomic_fadd_f32(float *p, float v) {
  return __builtin_amdgcn_flat_atomic_fadd_f32(p, v);
}

// CIR-LABEL: @_Z20flat_atomic_fadd_f64Pdd
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double>, !cir.double) -> !cir.double {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z20flat_atomic_fadd_f64Pdd
// LLVM: atomicrmw fadd ptr %{{.+}}, double %{{.+}} syncscope("agent") monotonic, align 8, !amdgpu.no.fine.grained.memory
// LLVM-NOT: atomic.ignore.denormal.mode
__device__ double flat_atomic_fadd_f64(double *p, double v) {
  return __builtin_amdgcn_flat_atomic_fadd_f64(p, v);
}

// CIR-LABEL: @_Z22global_atomic_fmin_f64PU3AS1dd
// CIR: cir.atomic.fetch min relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double, target_address_space(1)>, !cir.double) -> !cir.double {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z22global_atomic_fmin_f64PU3AS1dd
// LLVM: atomicrmw fmin ptr addrspace(1) %{{.+}}, double %{{.+}} syncscope("agent") monotonic, align 8, !amdgpu.no.fine.grained.memory
// LLVM-NOT: atomic.ignore.denormal.mode
__device__ double global_atomic_fmin_f64(double_global_ptr p, double v) {
  return __builtin_amdgcn_global_atomic_fmin_f64(p, v);
}

// CIR-LABEL: @_Z22global_atomic_fmax_f64PU3AS1dd
// CIR: cir.atomic.fetch max relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double, target_address_space(1)>, !cir.double) -> !cir.double {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z22global_atomic_fmax_f64PU3AS1dd
// LLVM: atomicrmw fmax ptr addrspace(1) %{{.+}}, double %{{.+}} syncscope("agent") monotonic, align 8, !amdgpu.no.fine.grained.memory
__device__ double global_atomic_fmax_f64(double_global_ptr p, double v) {
  return __builtin_amdgcn_global_atomic_fmax_f64(p, v);
}

// CIR-LABEL: @_Z20flat_atomic_fmin_f64Pdd
// CIR: cir.atomic.fetch min relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double>, !cir.double) -> !cir.double {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z20flat_atomic_fmin_f64Pdd
// LLVM: atomicrmw fmin ptr %{{.+}}, double %{{.+}} syncscope("agent") monotonic, align 8, !amdgpu.no.fine.grained.memory
__device__ double flat_atomic_fmin_f64(double *p, double v) {
  return __builtin_amdgcn_flat_atomic_fmin_f64(p, v);
}

// CIR-LABEL: @_Z20flat_atomic_fmax_f64Pdd
// CIR: cir.atomic.fetch max relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double>, !cir.double) -> !cir.double {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z20flat_atomic_fmax_f64Pdd
// LLVM: atomicrmw fmax ptr %{{.+}}, double %{{.+}} syncscope("agent") monotonic, align 8, !amdgpu.no.fine.grained.memory
__device__ double flat_atomic_fmax_f64(double *p, double v) {
  return __builtin_amdgcn_flat_atomic_fmax_f64(p, v);
}

// CIR-LABEL: @_Z20ds_atomic_fadd_v2f16
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.vector<2 x !cir.f16>, target_address_space(3)>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16> {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z20ds_atomic_fadd_v2f16
// LLVM: atomicrmw fadd ptr addrspace(3) %{{.+}}, <2 x half> %{{.+}} syncscope("agent") monotonic, align 4
// LLVM-NOT: amdgpu.no.fine.grained.memory
__device__ half2 ds_atomic_fadd_v2f16(half2_lds_ptr p, half2 v) {
  return __builtin_amdgcn_ds_atomic_fadd_v2f16(p, v);
}

// CIR-LABEL: @_Z24global_atomic_fadd_v2f16
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.vector<2 x !cir.f16>, target_address_space(1)>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16> {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z24global_atomic_fadd_v2f16
// LLVM: atomicrmw fadd ptr addrspace(1) %{{.+}}, <2 x half> %{{.+}} syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
__device__ half2 global_atomic_fadd_v2f16(half2_global_ptr p, half2 v) {
  return __builtin_amdgcn_global_atomic_fadd_v2f16(p, v);
}

// CIR-LABEL: @_Z22flat_atomic_fadd_v2f16
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.vector<2 x !cir.f16>>, !cir.vector<2 x !cir.f16>) -> !cir.vector<2 x !cir.f16> {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z22flat_atomic_fadd_v2f16
// LLVM: atomicrmw fadd ptr %{{.+}}, <2 x half> %{{.+}} syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
__device__ half2 flat_atomic_fadd_v2f16(half2 *p, half2 v) {
  return __builtin_amdgcn_flat_atomic_fadd_v2f16(p, v);
}

// CIR-LABEL: @_Z21ds_atomic_fadd_v2bf16
// CIR: %[[VAL:.+]] = cir.cast bitcast %{{.+}} : !cir.vector<2 x !s16i> -> !cir.vector<2 x !cir.bf16>
// CIR: %[[PTR:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.vector<2 x !s16i>, target_address_space(3)> -> !cir.ptr<!cir.vector<2 x !cir.bf16>, target_address_space(3)>
// CIR: %[[RMW:.+]] = cir.atomic.fetch add relaxed syncscope(device) fetch_first %[[PTR]], %[[VAL]] : (!cir.ptr<!cir.vector<2 x !cir.bf16>, target_address_space(3)>, !cir.vector<2 x !cir.bf16>) -> !cir.vector<2 x !cir.bf16> {cir.amdgpu_raw_atomic}
// CIR: cir.cast bitcast %[[RMW]] : !cir.vector<2 x !cir.bf16> -> !cir.vector<2 x !s16i>
// LLVM-LABEL: @_Z21ds_atomic_fadd_v2bf16
// LLVM: %[[VAL:.+]] = bitcast <2 x i16> %{{.+}} to <2 x bfloat>
// LLVM: %[[RMW:.+]] = atomicrmw fadd ptr addrspace(3) %{{.+}}, <2 x bfloat> %[[VAL]] syncscope("agent") monotonic, align 4
// LLVM-NOT: amdgpu.no.fine.grained.memory
// LLVM: bitcast <2 x bfloat> %[[RMW]] to <2 x i16>
__device__ short2 ds_atomic_fadd_v2bf16(short2_lds_ptr p, short2 v) {
  return __builtin_amdgcn_ds_atomic_fadd_v2bf16(p, v);
}

// CIR-LABEL: @_Z25global_atomic_fadd_v2bf16
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.vector<2 x !cir.bf16>, target_address_space(1)>, !cir.vector<2 x !cir.bf16>) -> !cir.vector<2 x !cir.bf16> {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z25global_atomic_fadd_v2bf16
// LLVM: %[[VAL:.+]] = bitcast <2 x i16> %{{.+}} to <2 x bfloat>
// LLVM: %[[RMW:.+]] = atomicrmw fadd ptr addrspace(1) %{{.+}}, <2 x bfloat> %[[VAL]] syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
// LLVM: bitcast <2 x bfloat> %[[RMW]] to <2 x i16>
__device__ short2 global_atomic_fadd_v2bf16(short2_global_ptr p, short2 v) {
  return __builtin_amdgcn_global_atomic_fadd_v2bf16(p, v);
}

// CIR-LABEL: @_Z23flat_atomic_fadd_v2bf16
// CIR: cir.atomic.fetch add relaxed syncscope(device) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.vector<2 x !cir.bf16>>, !cir.vector<2 x !cir.bf16>) -> !cir.vector<2 x !cir.bf16> {cir.amdgpu_raw_atomic}
// LLVM-LABEL: @_Z23flat_atomic_fadd_v2bf16
// LLVM: %[[VAL:.+]] = bitcast <2 x i16> %{{.+}} to <2 x bfloat>
// LLVM: %[[RMW:.+]] = atomicrmw fadd ptr %{{.+}}, <2 x bfloat> %[[VAL]] syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
// LLVM: bitcast <2 x bfloat> %[[RMW]] to <2 x i16>
__device__ short2 flat_atomic_fadd_v2bf16(short2 *p, short2 v) {
  return __builtin_amdgcn_flat_atomic_fadd_v2bf16(p, v);
}

// CIR-LABEL: @_Z5fencev
// CIR: cir.atomic.fence syncscope(device) seq_cst
// LLVM-LABEL: @_Z5fencev
// LLVM: fence syncscope("agent") seq_cst
__device__ void fence() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}
