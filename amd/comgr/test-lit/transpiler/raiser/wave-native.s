; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=masks,wave_id,cmpx,uniform_branch,atomic_counts,yz_predicate,matrix > %t.ll
; RUN: %FileCheck %s --check-prefix=IR < %t.ll
; RUN: %opt -passes='default<O2>' %t.ll -o %t.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.bc -o %t.gfx942.o
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir=masks | %FileCheck %s --check-prefix=SAME

; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=masks --specialize-workgroup=1,1,1 > %t.single.ll
; RUN: %opt -passes='default<O2>' -verify-each -S %t.single.ll -o %t.single.opt.ll
; RUN: %FileCheck %s --check-prefix=SINGLE --implicit-check-not=amdgpu-no-wwm < %t.single.opt.ll
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.single.opt.ll -o %t.single.o
; SINGLE: call i64 @llvm.amdgcn.strict.wwm.i64
; SINGLE: "amdgpu-flat-work-group-size"="1,1"
; RUN: %opt -passes='print<uniformity>' -disable-output %t.single.opt.ll 2>&1 | %FileCheck %s --check-prefix=SINGLE-DIVERGENCE
; SINGLE-DIVERGENCE: DIVERGENT: {{.+}}call i1 @llvm.amdgcn.init.whole.wave()
; SINGLE-DIVERGENCE: DIVERGENT: br i1

; The two source waves select opposite lane parities. Keep their ballots,
; saved EXEC masks, and memory guards connected through source-width SGPRs.
; IR-LABEL: define amdgpu_kernel void @masks(
; IR: [[LANE_LO:%.*]] = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; IR-NEXT: [[LANE:%.*]] = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 [[LANE_LO]])
; IR-NEXT: [[ENTRY_ACTIVE:%.*]] = call i1 @llvm.amdgcn.init.whole.wave()
; IR-NEXT: [[ENTRY_BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[ENTRY_ACTIVE]])
; IR-NEXT: [[ENTRY_WWM:%.+]] = call i64 @llvm.amdgcn.strict.wwm.i64(i64 [[ENTRY_BALLOT]])
; IR-NEXT: [[ENTRY_BASE:%.*]] = and i32 [[LANE]], -32
; IR-NEXT: [[ENTRY_SHIFT:%.*]] = zext i32 [[ENTRY_BASE]] to i64
; IR-NEXT: [[ENTRY_SLICE:%.*]] = lshr i64 [[ENTRY_WWM]], [[ENTRY_SHIFT]]
; IR-NEXT: [[ENTRY_EXEC:%.*]] = trunc i64 [[ENTRY_SLICE]] to i32
; IR: [[SOURCE_LANE:%.*]] = and i32 [[LANE]], 31
; IR-NEXT: [[ENTRY_BITS:%.*]] = lshr i32 [[ENTRY_EXEC]], [[SOURCE_LANE]]
; IR-NEXT: [[ENTRY_BIT:%.*]] = and i32 [[ENTRY_BITS]], 1
; IR-NEXT: [[ENTRY_ENABLED:%.*]] = icmp ne i32 [[ENTRY_BIT]], 0
; IR-NEXT: [[ACTIVE:%.*]] = select i1 [[ENTRY_ACTIVE]], i1 [[ENTRY_ENABLED]], i1 false
; IR: [[PRED:%.*]] = icmp eq i32 {{%.*}}, {{%.*}}
; IR-NEXT: [[CMP_ACTIVE:%.*]] = select i1 [[ACTIVE]], i1 [[PRED]], i1 false
; IR-NEXT: [[CMP_BALLOT:%.*]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[CMP_ACTIVE]])
; IR-NEXT: [[CMP_BASE:%.*]] = and i32 [[LANE]], -32
; IR-NEXT: [[CMP_SHIFT:%.*]] = zext i32 [[CMP_BASE]] to i64
; IR-NEXT: [[CMP_SLICE:%.*]] = lshr i64 [[CMP_BALLOT]], [[CMP_SHIFT]]
; IR-NEXT: [[CMP_MASK:%.*]] = trunc i64 [[CMP_SLICE]] to i32
; IR: [[EXEC:%.*]] = and i32 [[CMP_MASK]], [[ENTRY_EXEC]]
; IR: [[MASK_LANE:%.*]] = and i32 [[LANE]], 31
; IR-NEXT: [[MASK_BITS:%.*]] = lshr i32 [[EXEC]], [[MASK_LANE]]
; IR-NEXT: [[MASK_BIT:%.*]] = and i32 [[MASK_BITS]], 1
; IR-NEXT: [[MASK_ENABLED:%.*]] = icmp ne i32 [[MASK_BIT]], 0
; IR-NEXT: [[MASK_ACTIVE:%.*]] = select i1 [[ENTRY_ACTIVE]], i1 [[MASK_ENABLED]], i1 false
; IR-NEXT: br i1 [[MASK_ACTIVE]], label %[[CMP_COPY:.*]], label %{{.*}}
; IR: [[CMP_VALUE:%.*]] = phi i32 [ [[CMP_MASK]], %[[CMP_COPY]] ], [ undef, %{{.*}} ]
; IR: br i1 [[MASK_ACTIVE]], label %[[CMP_STORE:.*]], label %[[CMP_SKIP:.*]]
; IR: [[CMP_STORE]]:
; IR-NEXT: store i32 [[CMP_VALUE]], ptr addrspace(1) {{%.*}}, align 4
; IR-NEXT: br label %[[CMP_SKIP]]
; IR: [[CMP_SKIP]]:
; IR-NEXT: [[FIRST_BIT:%.*]] = call i32 @llvm.cttz.i32(i32 [[EXEC]], i1 false)
; IR-NEXT: [[EXEC_ZERO:%.*]] = icmp eq i32 [[EXEC]], 0
; IR-NEXT: [[FIRST_SOURCE_LANE:%.*]] = select i1 [[EXEC_ZERO]], i32 0, i32 [[FIRST_BIT]]
; IR-NEXT: [[FIRST_BASE:%.*]] = and i32 [[LANE]], -32
; IR-NEXT: [[FIRST_LANE:%.*]] = or i32 [[FIRST_BASE]], [[FIRST_SOURCE_LANE]]
; IR-NEXT: [[FIRST_ADDR:%.*]] = shl i32 [[FIRST_LANE]], 2
; IR-NEXT: [[FIRST_VALUE:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[FIRST_ADDR]], i32 [[WORKITEM:%.*]])
;
; Restoring s6 must retain the selected mask; restoring s5 must recover entry EXEC.
; IR: [[RESTORED_LANE:%.*]] = and i32 [[LANE]], 31
; IR-NEXT: [[RESTORED_BITS:%.*]] = lshr i32 [[EXEC]], [[RESTORED_LANE]]
; IR-NEXT: [[RESTORED_BIT:%.*]] = and i32 [[RESTORED_BITS]], 1
; IR-NEXT: [[RESTORED_ENABLED:%.*]] = icmp ne i32 [[RESTORED_BIT]], 0
; IR-NEXT: [[RESTORED_ACTIVE:%.*]] = select i1 [[ENTRY_ACTIVE]], i1 [[RESTORED_ENABLED]], i1 false
; IR-NEXT: br i1 [[RESTORED_ACTIVE]], label %[[EXEC_COPY:.*]], label %{{.*}}
; IR: [[EXEC_VALUE:%.*]] = phi i32 [ [[EXEC]], %[[EXEC_COPY]] ], [ [[CMP_VALUE]], %{{.*}} ]
; IR: [[EXEC_PTR:%.*]] = getelementptr i8, ptr addrspace(1) {{%.*}}, i64 1024
; IR-NEXT: br i1 [[RESTORED_ACTIVE]], label %[[EXEC_STORE:.*]], label %[[EXEC_SKIP:.*]]
; IR: [[EXEC_STORE]]:
; IR-NEXT: store i32 [[EXEC_VALUE]], ptr addrspace(1) [[EXEC_PTR]], align 4
; IR-NEXT: br label %[[EXEC_SKIP]]
; IR: [[EXEC_SKIP]]:
; IR-NEXT: [[RESTORED_ENTRY_LANE:%.*]] = and i32 [[LANE]], 31
; IR-NEXT: [[RESTORED_ENTRY_BITS:%.*]] = lshr i32 [[ENTRY_EXEC]], [[RESTORED_ENTRY_LANE]]
; IR-NEXT: [[RESTORED_ENTRY_BIT:%.*]] = and i32 [[RESTORED_ENTRY_BITS]], 1
; IR-NEXT: [[RESTORED_ENTRY_ENABLED:%.*]] = icmp ne i32 [[RESTORED_ENTRY_BIT]], 0
; IR-NEXT: [[RESTORED_ENTRY_ACTIVE:%.*]] = select i1 [[ENTRY_ACTIVE]], i1 [[RESTORED_ENTRY_ENABLED]], i1 false
; IR-NEXT: br i1 [[RESTORED_ENTRY_ACTIVE]], label %[[FIRST_COPY:.*]], label %{{.*}}
; IR: [[FIRST_STORED:%.*]] = phi i32 [ [[FIRST_VALUE]], %[[FIRST_COPY]] ], [ [[EXEC_VALUE]], %{{.*}} ]
; IR: [[FIRST_PTR:%.*]] = getelementptr i8, ptr addrspace(1) {{%.*}}, i64 2048
; IR-NEXT: br i1 [[RESTORED_ENTRY_ACTIVE]], label %[[FIRST_STORE:.*]], label %[[FIRST_SKIP:.*]]
; IR: [[FIRST_STORE]]:
; IR-NEXT: store i32 [[FIRST_STORED]], ptr addrspace(1) [[FIRST_PTR]], align 4
; IR-NEXT: br label %[[FIRST_SKIP]]
; IR: [[FIRST_SKIP]]:
; IR-NEXT: [[READ_BASE:%.*]] = and i32 [[LANE]], -32
; IR-NEXT: [[READ_LANE:%.*]] = or i32 [[READ_BASE]], 7
; IR-NEXT: [[READ_ADDR:%.*]] = shl i32 [[READ_LANE]], 2
; IR-NEXT: [[READ_VALUE:%.*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[READ_ADDR]], i32 [[WORKITEM]])
; IR: [[WRITE_LANE:%.*]] = and i32 [[LANE]], 31
; IR-NEXT: [[WRITE_SELECTED:%.*]] = icmp eq i32 [[WRITE_LANE]], 3
; IR-NEXT: [[WRITE_VALUE:%.*]] = select i1 [[WRITE_SELECTED]], i32 [[READ_VALUE]], i32 [[WORKITEM]]
; IR: [[WRITE_PTR:%.*]] = getelementptr i8, ptr addrspace(1) {{%.*}}, i64 3072
; IR-NEXT: [[WRITE_SOURCE_LANE:%.*]] = and i32 [[LANE]], 31
; IR-NEXT: [[WRITE_BITS:%.*]] = lshr i32 [[ENTRY_EXEC]], [[WRITE_SOURCE_LANE]]
; IR-NEXT: [[WRITE_BIT:%.*]] = and i32 [[WRITE_BITS]], 1
; IR-NEXT: [[WRITE_ENABLED:%.*]] = icmp ne i32 [[WRITE_BIT]], 0
; IR-NEXT: [[WRITE_ACTIVE:%.*]] = select i1 [[ENTRY_ACTIVE]], i1 [[WRITE_ENABLED]], i1 false
; IR-NEXT: br i1 [[WRITE_ACTIVE]], label %[[WRITE_STORE:.*]], label %[[WRITE_SKIP:.*]]
; IR: [[WRITE_STORE]]:
; IR-NEXT: store i32 [[WRITE_VALUE]], ptr addrspace(1) [[WRITE_PTR]], align 4
; IR-NEXT: br label %[[WRITE_SKIP]]
; IR: [[WRITE_SKIP]]:
; IR-NEXT: ret void
; IR-LABEL: define amdgpu_kernel void @wave_id(
; IR: call i32 @llvm.umin.i32
; IR: call i32 @llvm.umin.i32
; IR: udiv i32 {{.*}}, 64
; IR: call i32 @llvm.amdgcn.readfirstlane.i32
; IR: mul i32 {{.*}}, 2
; IR: udiv i32 {{.*}}, 32
; IR: add i32
; IR: store i32
; IR-LABEL: define amdgpu_kernel void @cmpx(
; IR: icmp ult i32 48,
; IR: call i64 @llvm.amdgcn.ballot.i64
; IR: and i32
; IR: store i32
; IR-LABEL: define amdgpu_kernel void @uniform_branch(
; IR: br i1
; IR-LABEL: define amdgpu_kernel void @atomic_counts(
; IR: br i1
; IR: atomicrmw add ptr addrspace(3)
; IR: store i32
; IR-LABEL: define amdgpu_kernel void @yz_predicate(
; IR: store i32
; IR-LABEL: define amdgpu_kernel void @matrix(
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16
; IR: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16
; IR: store i32
; SAME-LABEL: define amdgpu_kernel void @masks(
; SAME-NOT: @llvm.amdgcn.init.whole.wave
; SAME: call i32 @llvm.amdgcn.ballot.i32
; SAME: store i32
; SAME: call i32 @llvm.amdgcn.readlane.i32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text

.globl masks
.p2align 8
.type masks,@function
masks:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  v_and_b32 v0, 1023, v0
  v_lshlrev_b32 v3, 2, v0
  v_lshrrev_b32 v1, 5, v0
  v_and_b32 v1, 1, v1
  v_and_b32 v2, 1, v0
  v_cmp_eq_u32_e64 s4, v1, v2
  s_and_saveexec_b32 s5, s4
  s_mov_b32 s6, exec_lo
  v_mov_b32 v4, s4
  global_store_b32 v3, v4, s[2:3]
  v_readfirstlane_b32 s8, v0
  s_mov_b32 exec_lo, 0
  s_mov_b32 exec_lo, s6
  v_mov_b32 v4, s6
  global_store_b32 v3, v4, s[2:3] offset:1024
  s_mov_b32 exec_lo, s5
  v_mov_b32 v4, s8
  global_store_b32 v3, v4, s[2:3] offset:2048
  v_readlane_b32 s9, v0, 7
  v_mov_b32 v5, v0
  s_mov_b32 exec_lo, 0
  v_writelane_b32 v5, s9, 3
  s_mov_b32 exec_lo, s5
  global_store_b32 v3, v5, s[2:3] offset:3072
  s_endpgm

.globl wave_id
.p2align 8
.type wave_id,@function
wave_id:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  s_bfe_u32 s4, ttmp8, 0x50019
  v_mov_b32 v1, s4
  v_lshrrev_b32 v3, 10, v0
  v_and_b32 v3, 1023, v3
  v_lshlrev_b32 v3, 8, v3
  v_lshrrev_b32 v4, 20, v0
  v_lshlrev_b32 v4, 12, v4
  v_and_b32 v0, 1023, v0
  v_or_b32 v0, v0, v3
  v_or_b32 v0, v0, v4
  v_lshlrev_b32 v2, 2, v0
  global_store_b32 v2, v1, s[2:3]
  s_endpgm

.globl cmpx
.p2align 8
.type cmpx,@function
cmpx:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  s_mov_b32 s4, exec_lo
  v_and_b32 v0, 1023, v0
  v_lshlrev_b32 v2, 2, v0
  v_cmpx_lt_u32 48, v0
  global_store_b32 v2, v0, s[2:3]
  s_mov_b32 exec_lo, s4
  s_endpgm

.globl uniform_branch
.p2align 8
.type uniform_branch,@function
uniform_branch:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  s_cmp_eq_u32 s2, 0
  s_cbranch_scc1 .Lexit_uniform
  s_mov_b32 s5, exec_lo
  s_and_b32 exec_lo, exec_lo, 0x55555555
  v_and_b32 v0, 1023, v0
  v_lshlrev_b32 v1, 2, v0
  global_store_b32 v1, v0, s[2:3]
  s_mov_b32 exec_lo, s5
.Lexit_uniform:
  s_endpgm

.globl atomic_counts
.p2align 8
.type atomic_counts,@function
atomic_counts:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  s_bfe_u32 s4, ttmp8, 0x50019
  v_lshlrev_b32 v1, 2, s4
  v_and_b32 v0, 1023, v0
  v_and_b32 v2, 31, v0
  v_mov_b32 v3, 0
  v_mov_b32 v6, 1
  v_cmp_eq_u32 vcc_lo, 0, v2
  s_and_saveexec_b32 s5, vcc_lo
  ds_store_b32 v1, v3
  s_mov_b32 exec_lo, s5
  s_wait_dscnt 0
  v_and_b32 v2, 1, v0
  s_and_b32 s4, s4, 1
  v_cmp_eq_u32 vcc_lo, s4, v2
  s_and_saveexec_b32 s6, vcc_lo
  ds_add_u32 v1, v6
  s_mov_b32 exec_lo, s6
  s_wait_dscnt 0
  ds_load_b32 v4, v1
  s_wait_dscnt 0
  v_lshlrev_b32 v2, 2, v0
  global_store_b32 v2, v4, s[2:3]
  s_endpgm

.globl yz_predicate
.p2align 8
.type yz_predicate,@function
yz_predicate:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  v_lshrrev_b32 v1, 10, v0
  v_and_b32 v2, 1023, v0
  v_lshlrev_b32 v2, 2, v2
  v_cmp_eq_u32 vcc_lo, 0, v1
  s_and_saveexec_b32 s4, vcc_lo
  global_store_b32 v2, v0, s[2:3]
  s_mov_b32 exec_lo, s4
  s_endpgm


.globl matrix
.p2align 8
.type matrix,@function
matrix:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  v_and_b32 v24, 1023, v0
  v_lshlrev_b32 v25, 2, v24
  v_and_b32 v26, 32, v24
  v_cmp_eq_u32 vcc_lo, 0, v26
  v_mov_b32 v28, 0x3c003c00
  v_cndmask_b32 v27, 0x40004000, v28, vcc_lo
  v_mov_b32 v0, v27
  v_mov_b32 v1, v27
  v_mov_b32 v2, v27
  v_mov_b32 v3, v27
  v_mov_b32 v4, v27
  v_mov_b32 v5, v27
  v_mov_b32 v6, v27
  v_mov_b32 v7, v27
  v_mov_b32 v8, 0x3c003c00
  v_mov_b32 v9, 0x3c003c00
  v_mov_b32 v10, 0x3c003c00
  v_mov_b32 v11, 0x3c003c00
  v_mov_b32 v12, 0x3c003c00
  v_mov_b32 v13, 0x3c003c00
  v_mov_b32 v14, 0x3c003c00
  v_mov_b32 v15, 0x3c003c00
  v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 0
  global_store_b32 v25, v16, s[2:3] offset:0
  global_store_b32 v25, v17, s[2:3] offset:1024
  global_store_b32 v25, v18, s[2:3] offset:2048
  global_store_b32 v25, v19, s[2:3] offset:3072
  global_store_b32 v25, v20, s[2:3] offset:4096
  global_store_b32 v25, v21, s[2:3] offset:5120
  global_store_b32 v25, v22, s[2:3] offset:6144
  global_store_b32 v25, v23, s[2:3] offset:7168
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel masks
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel wave_id
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel cmpx
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel uniform_branch
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel atomic_counts
  .amdhsa_group_segment_fixed_size 32
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel yz_predicate
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel


.amdhsa_kernel matrix
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: masks
    .symbol: masks.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: wave_id
    .symbol: wave_id.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: cmpx
    .symbol: cmpx.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: uniform_branch
    .symbol: uniform_branch.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: atomic_counts
    .symbol: atomic_counts.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 32
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: yz_predicate
    .symbol: yz_predicate.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 24
    .vgpr_count: 32
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
  - .args:
      - .name: output
        .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name: matrix
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .symbol: matrix.kd
    .vgpr_count: 32
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
