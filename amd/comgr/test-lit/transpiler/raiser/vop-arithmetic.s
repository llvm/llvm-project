; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir > %t.ll
; RUN: %opt -passes='default<O2>' -verify-each -S %t.ll -o %t.opt.ll
; RUN: %FileCheck %s < %t.ll
; RUN: %FileCheck %s --check-prefix=OPT < %t.opt.ll
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.opt.ll -o %t.target.o
; RUN: %FileCheck %s --check-prefixes=CONST,KEEP,INPUT,IDENTITY-KEEP,FUSION < %t.opt.ll
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM=1 %s -o %t.flush.o
; RUN: %ld.lld -shared %t.flush.o -o %t.flush.hsaco
; RUN: %transpile_cli %t.flush.hsaco --target-isa=gfx942 --emit-ir=constants | %opt -passes='default<O2>' -S | %FileCheck %s --check-prefixes=CONST,FLUSH,INPUT,IDENTITY-FLUSH,FUSION
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM=2 %s -o %t.flush.o
; RUN: %ld.lld -shared %t.flush.o -o %t.flush.hsaco
; RUN: %transpile_cli %t.flush.hsaco --target-isa=gfx942 --emit-ir=constants | %opt -passes='default<O2>' -S | %FileCheck %s --check-prefixes=CONST,KEEP,NOINPUT,IDENTITY-FLUSH,FUSION
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=BAD_OMOD=1 %s -o %t.bad.o
; RUN: %ld.lld -shared %t.bad.o -o %t.bad.hsaco
; RUN: not %transpile_cli %t.bad.hsaco --target-isa=gfx942 --emit-ir=arithmetic32 2>&1 | %FileCheck %s --check-prefix=OMOD
; OMOD: unsupported-instruction-form: v_add_f32 [VOP3]
; OMOD-SAME: output multiplier is not supported
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=BAD_CLAMP=1 %s -o %t.bad.o
; RUN: %ld.lld -shared %t.bad.o -o %t.bad.hsaco
; RUN: not %transpile_cli %t.bad.hsaco --target-isa=gfx942 --emit-ir=arithmetic32 2>&1 | %FileCheck %s --check-prefix=CLAMP
; CLAMP: unsupported-instruction-form: v_add_f32 [VOP3]
; CLAMP-SAME: floating-point output clamp is not supported
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=EXCEPTIONS=1 %s -o %t.except.o
; RUN: %ld.lld -shared %t.except.o -o %t.except.hsaco
; RUN: not %transpile_cli %t.except.hsaco --target-isa=gfx942 --emit-ir=arithmetic32 2>&1 | %FileCheck %s --check-prefix=EXCEPTIONS
; EXCEPTIONS: unsupported-instruction-form: s_getreg_b32
; EXCEPTIONS-SAME: cannot reproduce hardware-register read for id 18
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=FMA_CLAMP=1 %s -o %t.bad.o
; RUN: %ld.lld -shared %t.bad.o -o %t.bad.hsaco
; RUN: not %transpile_cli %t.bad.hsaco --target-isa=gfx942 --emit-ir=arithmetic32 2>&1 | %FileCheck %s --check-prefix=FMA-CLAMP
; FMA-CLAMP: unsupported-instruction-form: v_fma_f32 [VOP3]
; FMA-CLAMP-SAME: floating-point output clamp is not supported
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=F64_OMOD=1 %s -o %t.bad.o
; RUN: %ld.lld -shared %t.bad.o -o %t.bad.hsaco
; RUN: not %transpile_cli %t.bad.hsaco --target-isa=gfx942 --emit-ir=arithmetic64 2>&1 | %FileCheck %s --check-prefix=F64-OMOD
; F64-OMOD: unsupported-instruction-form: v_add_f64 [VOP3]
; F64-OMOD-SAME: output multiplier is not supported
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=ROUND64=1 %s -o %t.bad.o
; RUN: %ld.lld -shared %t.bad.o -o %t.bad.hsaco
; RUN: not %transpile_cli %t.bad.hsaco --target-isa=gfx942 --emit-ir=arithmetic64 2>&1 | %FileCheck %s --check-prefix=ROUND64
; ROUND64: unsupported-floating-point-mode: v_add_f64 [VOP2]
; ROUND64-SAME: f64 rounding mode 1 is unsupported
.ifndef ROUND64
.set ROUND64, 0
.endif
.ifndef BAD_CLAMP
.set BAD_CLAMP, 0
.endif
.ifndef DENORM
.set DENORM, 3
.endif
.amdhsa_code_object_version 6
.text

.globl arithmetic32
.p2align 8
.type arithmetic32,@function
; CHECK-LABEL: define amdgpu_kernel void @arithmetic32(
; OPT-LABEL: define amdgpu_kernel void @arithmetic32(
arithmetic32:
 s_load_b128 s[4:7], s[0:1], 0
 v_lshlrev_b32 v1, 6, v0
 s_wait_kmcnt 0
 global_load_b32 v2, v1, s[4:5]
 global_load_b32 v4, v1, s[4:5] offset:8
 s_wait_loadcnt 0
; CHECK: fadd float
 v_add_f32_e32 v6, v2, v4
.ifdef EXCEPTIONS
 s_getreg_b32 s2, hwreg(HW_REG_WAVE_EXCP_FLAG_USER, 0, 6)
.endif
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b32 v1, v6, s[6:7] offset:0
; CHECK: fmul float
; OPT: or i32 {{%.+}}, 4194304
 v_mul_f32_e32 v6, 1.0, v2
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b32 v1, v6, s[6:7] offset:8
; CHECK: [[ADD_ABS32:%.+]] = call float @llvm.fabs.f32(float {{%.+}})
; CHECK: [[ADD_NEG32:%.+]] = fneg float [[ADD_ABS32]]
; CHECK: [[ADD_RIGHT32:%.+]] = call float @llvm.fabs.f32(float {{%.+}})
; CHECK: fadd float [[ADD_NEG32]], [[ADD_RIGHT32]]
.ifdef BAD_OMOD
 v_add_f32_e64 v6, -|v2|, |v4| mul:2
.elseif BAD_CLAMP
 v_add_f32_e64 v6, -|v2|, |v4| clamp
.else
 v_add_f32_e64 v6, -|v2|, |v4|
.endif
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b32 v1, v6, s[6:7] offset:16
; CHECK: [[MUL_LEFT32:%.+]] = call float @llvm.fabs.f32(float {{%.+}})
; CHECK: [[MUL_RIGHT32:%.+]] = fneg float {{%.+}}
; CHECK: fmul float [[MUL_LEFT32]], [[MUL_RIGHT32]]
 v_mul_f32_e64 v6, |v2|, -v4
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b32 v1, v6, s[6:7] offset:24
; CHECK: [[FMA_LEFT32:%.+]] = fneg float {{%.+}}
; CHECK: [[FMA_RIGHT32:%.+]] = call float @llvm.fabs.f32(float {{%.+}})
; CHECK: [[FMA_ADDEND32:%.+]] = fneg float {{%.+}}
; CHECK: call float @llvm.fma.f32(float [[FMA_LEFT32]], float [[FMA_RIGHT32]], float [[FMA_ADDEND32]])
.ifdef FMA_CLAMP
 v_fma_f32 v6, -v2, |v4|, -v6 clamp
.else
 v_fma_f32 v6, -v2, |v4|, -v6
.endif
; CHECK: store i32 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b32 v1, v6, s[6:7] offset:32
; CHECK: call float @llvm.fma.f32(float {{%.+}}, float {{%.+}}, float 5.000000e-01)
 v_fmaak_f32 v6, v2, v4, 0x3f000000
 global_store_b32 v1, v6, s[6:7] offset:40
; CHECK: [[SUB_NEG:%.+]] = fneg float {{%.+}}
; CHECK: [[SUB_ABS:%.+]] = call float @llvm.fabs.f32(float {{%.+}})
; CHECK: fsub float [[SUB_NEG]], [[SUB_ABS]]
 v_sub_f32_e64 v6, -v2, |v4|
 global_store_b32 v1, v6, s[6:7] offset:48
; CHECK: [[SUBREV_NEG:%.+]] = fneg float {{%.+}}
; CHECK: [[SUBREV_ABS:%.+]] = call float @llvm.fabs.f32(float {{%.+}})
; CHECK: fsub float [[SUBREV_ABS]], [[SUBREV_NEG]]
 v_subrev_f32_e64 v6, -v2, |v4|
 global_store_b32 v1, v6, s[6:7] offset:56
 s_endpgm

.globl arithmetic64
.p2align 8
.type arithmetic64,@function
; CHECK-LABEL: define amdgpu_kernel void @arithmetic64(
; OPT-LABEL: define amdgpu_kernel void @arithmetic64(
arithmetic64:
 s_load_b128 s[4:7], s[0:1], 0
 v_lshlrev_b32 v1, 6, v0
 s_wait_kmcnt 0
 global_load_b64 v[2:3], v1, s[4:5]
 global_load_b64 v[4:5], v1, s[4:5] offset:8
 s_wait_loadcnt 0
; CHECK: fadd double
 v_add_f64_e32 v[6:7], v[2:3], v[4:5]
; CHECK: store i64 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b64 v1, v[6:7], s[6:7] offset:0
; CHECK: fmul double
; OPT: or i64 {{%.+}}, 2251799813685248
 v_mul_f64_e32 v[6:7], 1.0, v[2:3]
; CHECK: store i64 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b64 v1, v[6:7], s[6:7] offset:8
; CHECK: [[ADD_ABS64:%.+]] = call double @llvm.fabs.f64(double {{%.+}})
; CHECK: [[ADD_NEG64:%.+]] = fneg double [[ADD_ABS64]]
; CHECK: [[ADD_RIGHT64:%.+]] = call double @llvm.fabs.f64(double {{%.+}})
; CHECK: fadd double [[ADD_NEG64]], [[ADD_RIGHT64]]
.ifdef F64_OMOD
 v_add_f64_e64 v[6:7], -|v[2:3]|, |v[4:5]| mul:2
.else
 v_add_f64_e64 v[6:7], -|v[2:3]|, |v[4:5]|
.endif
; CHECK: store i64 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b64 v1, v[6:7], s[6:7] offset:16
; CHECK: [[MUL_LEFT64:%.+]] = call double @llvm.fabs.f64(double {{%.+}})
; CHECK: [[MUL_RIGHT64:%.+]] = fneg double {{%.+}}
; CHECK: fmul double [[MUL_LEFT64]], [[MUL_RIGHT64]]
 v_mul_f64_e64 v[6:7], |v[2:3]|, -v[4:5]
; CHECK: store i64 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b64 v1, v[6:7], s[6:7] offset:24
; CHECK: [[FMA_LEFT64:%.+]] = fneg double {{%.+}}
; CHECK: [[FMA_RIGHT64:%.+]] = call double @llvm.fabs.f64(double {{%.+}})
; CHECK: [[FMA_ADDEND64:%.+]] = fneg double {{%.+}}
; CHECK: call double @llvm.fma.f64(double [[FMA_LEFT64]], double [[FMA_RIGHT64]], double [[FMA_ADDEND64]])
 v_fma_f64 v[6:7], -v[2:3], |v[4:5]|, -v[6:7]
; CHECK: store i64 {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b64 v1, v[6:7], s[6:7] offset:32
; CHECK: fmul double 1.500000e+00,
 v_mul_f64_e32 v[6:7], 0x3ff8000000000000, v[2:3]
 global_store_b64 v1, v[6:7], s[6:7] offset:40
; CHECK: fmul double 1.500000e+00,
 v_mul_f64_e64 v[6:7], lit(0x3ff80000), v[2:3]
 global_store_b64 v1, v[6:7], s[6:7] offset:48
 s_endpgm

.globl constants
.p2align 8
.type constants,@function
; CONST-LABEL: define amdgpu_kernel void @constants(
constants:
 s_load_b64 s[4:5], s[0:1], 0
 v_lshlrev_b32 v1, 5, v0
 s_wait_kmcnt 0
 v_mov_b32 v10, 1.0
 v_mul_f32_e32 v2, 0x7f812345, v10
 v_and_b32 v2, 0x7fc00000, v2
; CONST: store i32 2143289344,
 global_store_b32 v1, v2, s[4:5]
 v_mul_f32_e32 v3, 1, v10
; IDENTITY-KEEP: store i32 1,
; IDENTITY-FLUSH: store i32 0,
 global_store_b32 v1, v3, s[4:5] offset:4
 v_mov_b32 v10, 0x00800000
 v_mul_f32_e32 v4, 0.5, v10
; KEEP: store i32 4194304,
; FLUSH: store i32 0,
 global_store_b32 v1, v4, s[4:5] offset:8
 v_mov_b32 v10, 1
 v_mul_f32_e32 v5, 0x7e800000, v10
; INPUT: store i32 872415232,
; NOINPUT: store i32 0,
 global_store_b32 v1, v5, s[4:5] offset:12
 v_mov_b32 v10, 0x3f800001
 v_fma_f32 v6, v10, v10, 0xbf800002
; FUSION: store i32 679477248,
 global_store_b32 v1, v6, s[4:5] offset:16
 v_mul_f32_e32 v7, v10, v10
 v_add_f32_e32 v7, 0xbf800002, v7
; FUSION: store i32 0,
 global_store_b32 v1, v7, s[4:5] offset:20
 v_mov_b32 v10, 1.0
 v_mul_f32_e32 v8, 0x80000000, v10
; CONST: store i32 -2147483648,
 global_store_b32 v1, v8, s[4:5] offset:24
 v_mov_b32 v10, 0x12345678
 v_mov_b32 v11, 0x7ff00000
 v_mul_f64_e32 v[12:13], 1.0, v[10:11]
 v_and_b32 v9, 0x7ff80000, v13
; CONST: store i32 2146959360,
 global_store_b32 v1, v9, s[4:5] offset:28
 s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel arithmetic32
 .amdhsa_kernarg_size 16
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 8
 .amdhsa_next_free_sgpr 8
 .amdhsa_float_denorm_mode_32 DENORM
 .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.amdhsa_kernel arithmetic64
 .amdhsa_float_round_mode_16_64 ROUND64
 .amdhsa_kernarg_size 16
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 8
 .amdhsa_next_free_sgpr 8
 .amdhsa_float_denorm_mode_32 DENORM
 .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdhsa_kernel constants
 .amdhsa_kernarg_size 8
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 14
 .amdhsa_next_free_sgpr 6
 .amdhsa_float_denorm_mode_32 DENORM
 .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
 - .name: arithmetic32
   .symbol: arithmetic32.kd
   .kernarg_segment_size: 16
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 32
   .sgpr_count: 8
   .vgpr_count: 8
   .wavefront_size: 32
 - .name: arithmetic64
   .symbol: arithmetic64.kd
   .kernarg_segment_size: 16
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 32
   .sgpr_count: 8
   .vgpr_count: 8
   .wavefront_size: 32
 - .name: constants
   .symbol: constants.kd
   .kernarg_segment_size: 8
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 32
   .sgpr_count: 6
   .vgpr_count: 14
   .wavefront_size: 32
...
.end_amdgpu_metadata
