; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=1024,1,1 > %t.ll
; RUN: %FileCheck %s --check-prefixes=CHECK,EXEC-LOOP < %t.ll
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym FIXED_TRIPS=1 -filetype=obj %s -o %t.fixed.o
; RUN: %ld.lld -shared %t.fixed.o -o %t.fixed.hsaco
; RUN: %transpile_cli %t.fixed.hsaco --target-isa=gfx942 --emit-ir --specialize-workgroup=37,1,1 | %FileCheck %s --check-prefixes=CHECK,FIXED-LOOP
; RUN: %transpile_cli %t.fixed.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s --check-prefixes=CHECK,FIXED-LOOP
; RUN: %opt -passes='default<O2>' -verify-each %t.ll -o %t.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.bc -o %t.target.o
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym UNSAFE_INPUT=1 -defsym FIXED_TRIPS=1 -filetype=obj %s -o %t.unsafe.o
; RUN: %ld.lld -shared %t.unsafe.o -o %t.unsafe.hsaco
; RUN: not %transpile_cli %t.unsafe.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; REFUSE: unproven-exec-containment: s_or_b32 [SOP2]
; REFUSE-SAME: in kernel 'accumulated_ballot'
; REFUSE-SAME: cannot prove that EXEC only enables lanes active at kernel entry
; RUN: not %transpile_cli %t.unsafe.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=37,1,1 2>&1 | %FileCheck %s --check-prefix=PARTIAL
; PARTIAL: unsupported-launch
; PARTIAL-SAME: requires whole source waves
; RUN: %transpile_cli %t.unsafe.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=FALLBACK
; FALLBACK: launch: accumulated_ballot kind=replicated
; FALLBACK: store i32

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym FIXED_TRIPS=1 -defsym MASK_OPERATION=1 -filetype=obj %s -o %t.ops.o
; RUN: %ld.lld -shared %t.ops.o -o %t.ops.hsaco
; RUN: %transpile_cli %t.ops.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s --check-prefix=OPS
; OPS: xor i32
; OPS: and i32
; OPS: select i1
; OPS: store i32
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym FIXED_TRIPS=1 -defsym MASK_OPERATION=2 -filetype=obj %s -o %t.shift.o
; RUN: %ld.lld -shared %t.shift.o -o %t.shift.hsaco
; RUN: not %transpile_cli %t.shift.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym FIXED_TRIPS=1 -defsym MASK_OPERATION=3 -filetype=obj %s -o %t.cast.o
; RUN: %ld.lld -shared %t.cast.o -o %t.cast.hsaco
; RUN: not %transpile_cli %t.cast.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym FIXED_TRIPS=1 -defsym MASK_OPERATION=4 -filetype=obj %s -o %t.predicate.o
; RUN: %ld.lld -shared %t.predicate.o -o %t.predicate.hsaco
; RUN: not %transpile_cli %t.predicate.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE

.ifndef MASK_OPERATION
.set MASK_OPERATION, 0
.endif
; Each lane accumulates a different iteration count; the two source waves differ.
.amdhsa_code_object_version 6
.text
.globl accumulated_ballot
.p2align 8
.type accumulated_ballot,@function
; CHECK-LABEL: define amdgpu_kernel void @accumulated_ballot(
; CHECK: [[ENTRY_ACTIVE:%.+]] = call i1 @llvm.amdgcn.init.whole.wave()
; CHECK: [[ENTRY_BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[ENTRY_ACTIVE]])
; CHECK: [[ENTRY_BASE:%.+]] = and i32 [[LANE_ID:%.+]], -32
; CHECK: [[ENTRY_OFFSET:%.+]] = zext i32 [[ENTRY_BASE]] to i64
; CHECK: [[ENTRY_SLICE:%.+]] = lshr i64 [[ENTRY_BALLOT]], [[ENTRY_OFFSET]]
; CHECK: [[ENTRY:%.+]] = trunc i64 [[ENTRY_SLICE]] to i32
accumulated_ballot:
 s_load_b64 s[4:5], s[0:1], 0
 v_and_b32 v1, 63, v0
 v_add_nc_u32 v1, 1, v1
 v_mov_b32 v2, 0
 .ifdef UNSAFE_INPUT
 s_mov_b32 s6, -1
 .else
 s_mov_b32 s6, 0
 .endif
 .ifdef FIXED_TRIPS
 s_mov_b32 s7, 64
 .endif
.Lloop:
; CHECK: [[COUNTER:%.+]] = phi i32 [ 0, {{%.+}} ], [ [[COUNTER_NEXT:%.+]], {{%.+}} ]
; CHECK: [[EXEC:%.+]] = phi i32 [ [[ENTRY]], {{%.+}} ], [ [[REMAINING:%.+]], {{%.+}} ]
; CHECK: [[ACC:%.+]] = phi i32 [ 0, {{%.+}} ], [ [[ACC_NEXT:%.+]], {{%.+}} ]
; CHECK: [[INCREMENT:%.+]] = add i32 1, [[COUNTER]]
; CHECK: [[LANE:%.+]] = and i32 [[LANE_ID]], 31
; CHECK: [[SHIFTED:%.+]] = lshr i32 [[EXEC]], [[LANE]]
; CHECK: [[BIT:%.+]] = and i32 [[SHIFTED]], 1
; CHECK: [[ACTIVE:%.+]] = icmp ne i32 [[BIT]], 0
; CHECK: [[DISPATCHED:%.+]] = select i1 [[ENTRY_ACTIVE]], i1 [[ACTIVE]], i1 false
 v_add_nc_u32 v2, 1, v2
 v_cmp_ge_u32 vcc_lo, v2, v1
 s_or_b32 s6, vcc_lo, s6
 s_and_not1_b32 exec_lo, exec_lo, s6
 .ifdef FIXED_TRIPS
 s_sub_u32 s7, s7, 1
 s_cmp_lg_u32 s7, 0
 s_cbranch_scc1 .Lloop
 .else
 s_cbranch_execnz .Lloop
 .endif
 .if MASK_OPERATION == 1
 s_xor_b32 s8, s6, exec_lo
 s_and_b32 s8, s8, s6
 s_cselect_b32 s6, s6, s8
 .elseif MASK_OPERATION == 2
 s_lshl_b32 s6, s6, 1
 .elseif MASK_OPERATION == 3
 s_sext_i32_i16 s6, s6
 .elseif MASK_OPERATION == 4
 s_cselect_b32 s6, 1, 0
 .endif
; CHECK: [[RESTORED:%.+]] = or i32 [[REMAINING]], [[ACC_NEXT]]
 s_or_b32 exec_lo, exec_lo, s6
 v_lshlrev_b32 v3, 3, v0
 v_mov_b32 v4, exec_lo
 s_wait_kmcnt 0
; CHECK: [[COUNTER_NEXT]] = phi i32 [ [[INCREMENT]], {{%.+}} ], [ [[COUNTER]], {{%.+}} ]
; CHECK: [[DONE:%.+]] = icmp uge i32 [[COUNTER_NEXT]], {{%.+}}
; CHECK: [[PRED:%.+]] = select i1 [[DISPATCHED]], i1 [[DONE]], i1 false
; CHECK: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[PRED]])
; CHECK: [[BASE:%.+]] = and i32 [[LANE_ID]], -32
; CHECK: [[OFFSET:%.+]] = zext i32 [[BASE]] to i64
; CHECK: [[SLICE:%.+]] = lshr i64 [[BALLOT]], [[OFFSET]]
; CHECK: [[MASK:%.+]] = trunc i64 [[SLICE]] to i32
; CHECK: [[ACC_NEXT]] = or i32 [[MASK]], [[ACC]]
; CHECK: [[COMPLEMENT:%.+]] = xor i32 [[ACC_NEXT]], -1
; CHECK: [[REMAINING]] = and i32 [[EXEC]], [[COMPLEMENT]]
; FIXED-LOOP: [[SAVED:%.+]] = phi i32 [ [[RESTORED]], {{%.+}} ], [ {{.+}}, {{%.+}} ]
; CHECK: store i32 [[COUNTER_NEXT]], ptr addrspace(1) {{.+}}
 global_store_b32 v3, v2, s[4:5] offset:16
; EXEC-LOOP: store i32 [[RESTORED]], ptr addrspace(1) {{.+}}
; FIXED-LOOP: store i32 [[SAVED]], ptr addrspace(1) {{.+}}
 global_store_b32 v3, v4, s[4:5] offset:20
 s_endpgm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel accumulated_ballot
 .amdhsa_kernarg_size 8
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 5
 .amdhsa_next_free_sgpr 9
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
 - .name: accumulated_ballot
   .symbol: accumulated_ballot.kd
   .kernarg_segment_size: 8
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 1024
   .sgpr_count: 9
   .vgpr_count: 5
   .wavefront_size: 32
...
.end_amdgpu_metadata
