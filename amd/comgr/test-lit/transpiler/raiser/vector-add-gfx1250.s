; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s \
; RUN:   -o %t.gfx1250.o
; RUN: %ld.lld -shared %t.gfx1250.o -o %t.gfx1250.hsaco

; RUN: %transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vector_add --allow-replicated-dispatch \
; RUN:   --launch-grid=4096,1,1 --launch-workgroup=256,1,1 > %t.raised.ll
; RUN: %FileCheck %s --check-prefix=IR < %t.raised.ll
; IR: ; launch: vector_add kind=unchanged max_workgroup_size=256 grid=4096,1,1 workgroup=256,1,1
; RUN: %transpile_cli %t.gfx1250.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vector_add --allow-replicated-dispatch \
; RUN:   --launch-grid=768,1,1 --launch-workgroup=48,1,1 \
; RUN:   | %FileCheck %s --check-prefix=PARTIAL
; PARTIAL: ; launch: vector_add kind=unchanged max_workgroup_size=256 grid=768,1,1 workgroup=48,1,1
; RUN: %opt -S -passes='default<O2>' -verify-each %t.raised.ll -o %t.opt.ll
; RUN: %clang -x ir -nogpulib --target=amdgcn-amd-amdhsa -mcpu=gfx942 \
; RUN:   -O2 %t.opt.ll -o %t.device.hsaco

; Both launches use 16 workgroups and a 256-element output stride. The partial
; launch leaves holes after each workgroup, including 16 inactive wave64 lanes.
; RUN: %if comgr-has-hip-gfx942 %{ %clang %hip_cflags \
; RUN:   %S/../Inputs/run-vector-add.c %hip_ldflags -o %t.run-vector-add %}
; RUN: %if comgr-has-hip-gfx942 %{ %t.run-vector-add %t.device.hsaco 256 \
; RUN:   | %FileCheck %s --check-prefix=FULL %}
; FULL: RESULT: PASS (4096 sums, 128 untouched elements)
; RUN: %if comgr-has-hip-gfx942 %{ %t.run-vector-add %t.device.hsaco 48 \
; RUN:   | %FileCheck %s --check-prefix=TAIL %}
; TAIL: RESULT: PASS (768 sums, 3456 untouched elements)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	vector_add
	.p2align	8
	.type	vector_add,@function
; Source entry IDs: v0 contains X; ttmp9 contains the workgroup X ID.
; IR-LABEL: define amdgpu_kernel void @vector_add(
; IR: [[TID:%.+]] = call i32 @llvm.amdgcn.workitem.id.x()
; IR: [[WGID:%.+]] = call i32 @llvm.amdgcn.workgroup.id.x()
vector_add:
; IR: [[A_PTR:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i64, ptr addrspace(1) [[A_PTR]], align 4
	s_load_b64 s[4:5], s[0:1], 0x0
; IR: [[B_PTR:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i64, ptr addrspace(1) [[B_PTR]], align 4
	s_load_b64 s[6:7], s[0:1], 0x8
; IR: [[C_PTR:%.+]] = inttoptr i64 {{%.+}} to ptr addrspace(1)
; IR: load i64, ptr addrspace(1) [[C_PTR]], align 4
	s_load_b64 s[8:9], s[0:1], 0x10

; A fixed stride leaves observable gaps when launching partial workgroups.
; IR: [[BLOCK:%.+]] = shl i32 [[WGID]], 8
	s_lshl_b32 s3, ttmp9, 8
; IR: [[INDEX:%.+]] = add i32 [[BLOCK]], [[TID]]
	v_add_nc_u32 v1, s3, v0
; IR: shl i32 [[INDEX]], 2
	v_lshlrev_b32 v1, 2, v1
	s_wait_kmcnt 0x0

; IR: load i32, ptr addrspace(1) {{%.+}}, align 4
	global_load_b32 v2, v1, s[4:5]
; IR: load i32, ptr addrspace(1) {{%.+}}, align 4
	global_load_b32 v3, v1, s[6:7]
	s_wait_loadcnt 0x0
; IR: [[LHS:%.+]] = bitcast i32 {{%.+}} to float
; IR: [[RHS:%.+]] = bitcast i32 {{%.+}} to float
; IR: fadd float [[LHS]], [[RHS]]
	v_add_f32 v2, v2, v3
; IR: store i32 {{%.+}}, ptr addrspace(1) {{%.+}}, align 4
	global_store_b32 v1, v2, s[8:9]
; IR: ret void
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vector_add
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 10
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 256
    .name:           vector_add
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         vector_add.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
