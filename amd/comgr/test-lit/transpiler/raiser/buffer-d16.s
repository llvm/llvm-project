; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=3 -defsym=INITIAL=0x12345678 -filetype=obj %s -o %t.any.o
; RUN: %ld.lld -shared %t.any.o -o %t.any.hsaco
; RUN: %transpile_cli %t.any.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefix=ECC
; RUN: %transpile_cli %t.any.hsaco --target-isa=amdgcn-amd-amdhsa--gfx942:sramecc- --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefix=ECC
; RUN: %transpile_cli %t.any.hsaco --target-isa=gfx1250 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefix=ECC
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=+sramecc -defsym=EXTENT=3 -defsym=INITIAL=0x12345678 -filetype=obj %s -o %t.on.o
; RUN: %ld.lld -shared %t.on.o -o %t.on.hsaco
; RUN: %transpile_cli %t.on.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefix=ECC
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=3 -defsym=INITIAL=0x12345678 -defsym=MASKED=1 -filetype=obj %s -o %t.masked.o
; RUN: %ld.lld -shared %t.masked.o -o %t.masked.hsaco
; RUN: %transpile_cli %t.masked.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefix=MASKED
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=0 -defsym=INITIAL=0x12345678 -filetype=obj %s -o %t.oob.o
; RUN: %ld.lld -shared %t.oob.o -o %t.oob.hsaco
; RUN: %transpile_cli %t.oob.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=OOB,OOB-ACTIVE
; RUN: %transpile_cli %t.oob.hsaco --isa=amdgcn-amd-amdhsa--gfx1250:sramecc+ --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=OOB,OOB-ACTIVE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=0 -defsym=INITIAL=0x12345678 -defsym=MASKED=1 -filetype=obj %s -o %t.masked-oob.o
; RUN: %ld.lld -shared %t.masked-oob.o -o %t.masked-oob.hsaco
; RUN: %transpile_cli %t.masked-oob.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefixes=OOB,OOB-MASKED
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=0 -defsym=INITIAL=0 -filetype=obj %s -o %t.zero.o
; RUN: %ld.lld -shared %t.zero.o -o %t.zero.hsaco
; RUN: %transpile_cli %t.zero.hsaco --target-isa=gfx942 --emit-ir | %opt -S -passes=instcombine,simplifycfg | %FileCheck %s --check-prefix=ZERO
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=-sramecc -defsym=EXTENT=3 -defsym=INITIAL=0x12345678 -filetype=obj %s -o %t.off.o
; RUN: %ld.lld -shared %t.off.o -o %t.off.hsaco
; RUN: not %transpile_cli %t.off.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=OFF
; RUN: not %transpile_cli %t.any.hsaco --isa=amdgcn-amd-amdhsa--gfx1250:sramecc- --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=OFF
; OFF: BadInput
; OFF-SAME: gfx1250 source does not support disabling SRAM ECC

.amdhsa_code_object_version 6
.text
.globl buffer_d16
.p2align 8
.type buffer_d16,@function
; OOB-LABEL: define amdgpu_kernel void @buffer_d16(
; OOB-MASKED: [[SOURCE_EXEC:%.+]] = trunc i64 {{%.+}} to i32
; ZERO-LABEL: define amdgpu_kernel void @buffer_d16(
; ECC-LABEL: define amdgpu_kernel void @buffer_d16(
; MASKED-LABEL: define amdgpu_kernel void @buffer_d16(
buffer_d16:
  s_load_b128 s[4:7], s[0:1], 0
  s_load_b64 s[8:9], s[0:1], 16
  s_wait_kmcnt 0
  s_and_b32 s7, s7, 63
.ifdef EXTENT
  s_and_b32 s5, s5, 0x1ffffff
  s_or_b32 s5, s5, EXTENT << 25
  s_mov_b32 s6, 0
  s_mov_b32 s7, 0
.endif
  v_mov_b32 v0, 0
  v_mov_b32 v4, INITIAL
  v_mov_b32 v5, INITIAL
  v_mov_b32 v6, INITIAL
  v_mov_b32 v7, INITIAL
  v_mov_b32 v8, INITIAL
  v_mov_b32 v9, INITIAL
.ifdef MASKED
  s_and_saveexec_b32 s10, 0x55555555
.endif
; MASKED: [[VALUE0:%.+]] = phi i32 [ {{%.+}}, %{{.+}} ], [ 305419896, %{{.+}} ]
; MASKED: [[VALUE1:%.+]] = phi i32 [ {{%.+}}, %{{.+}} ], [ 305419896, %{{.+}} ]
; MASKED: [[VALUE2:%.+]] = phi i32 [ {{%.+}}, %{{.+}} ], [ 305419896, %{{.+}} ]
; MASKED: [[VALUE3:%.+]] = phi i32 [ {{%.+}}, %{{.+}} ], [ 305419896, %{{.+}} ]
; MASKED: [[VALUE4:%.+]] = phi i32 [ {{%.+}}, %{{.+}} ], [ 305419896, %{{.+}} ]
; MASKED: [[VALUE5:%.+]] = phi i32 [ {{%.+}}, %{{.+}} ], [ 305419896, %{{.+}} ]
; ECC: [[U8:%.+]] = load i8, ptr addrspace(1) {{.+}}, align 1
; ECC: [[LOW_U8:%.+]] = zext i8 [[U8]] to i32
  buffer_load_d16_u8 v4, off, s[4:7], null
; ECC: [[I8:%.+]] = load i8, ptr addrspace(1) {{.+}}, align 1
; ECC: [[SIGNED_I8:%.+]] = sext i8 [[I8]] to i32
; ECC: [[LOW_I8:%.+]] = and i32 [[SIGNED_I8]], 65535
  buffer_load_d16_i8 v5, off, s[4:7], null
; ECC: [[B16:%.+]] = load i16, ptr addrspace(1) {{.+}}, align 1
; ECC: [[LOW_B16:%.+]] = zext i16 [[B16]] to i32
  buffer_load_d16_b16 v6, off, s[4:7], null
; ECC: [[HI_U8:%.+]] = load i8, ptr addrspace(1) {{.+}}, align 1
; ECC: [[EXT_U8:%.+]] = zext i8 [[HI_U8]] to i32
; ECC: [[HIGH_U8:%.+]] = shl{{.+}}i32 [[EXT_U8]], 16
  buffer_load_d16_hi_u8 v7, off, s[4:7], null
; ECC: [[HI_I8:%.+]] = load i8, ptr addrspace(1) {{.+}}, align 1
; ECC: [[EXT_I8:%.+]] = sext i8 [[HI_I8]] to i32
; ECC: [[HIGH_I8:%.+]] = shl{{.+}}i32 [[EXT_I8]], 16
  buffer_load_d16_hi_i8 v8, off, s[4:7], null
; ECC: [[HI_B16:%.+]] = load i16, ptr addrspace(1) {{.+}}, align 1
; ECC: [[EXT_B16:%.+]] = zext i16 [[HI_B16]] to i32
; ECC: [[HIGH_B16:%.+]] = shl{{.+}}i32 [[EXT_B16]], 16
  buffer_load_d16_hi_b16 v9, off, s[4:7], null
  s_wait_loadcnt 0
.ifdef MASKED
  // Observe the destinations of lanes disabled during the loads.
  s_mov_b32 exec_lo, s10
.endif
; OOB-NOT: load i8
; OOB-NOT: load i16
; OOB: [[VALUE0:%.+]] = select i1 [[ACTIVE:%.+]], i32 0, i32 305419896
; OOB: [[VALUE1:%.+]] = select i1 [[ACTIVE]], i32 0, i32 305419896
; OOB: [[VALUE2:%.+]] = select i1 [[ACTIVE]], i32 0, i32 305419896
; OOB: [[VALUE3:%.+]] = select i1 [[ACTIVE]], i32 0, i32 305419896
; OOB: [[VALUE4:%.+]] = select i1 [[ACTIVE]], i32 0, i32 305419896
; OOB: [[VALUE5:%.+]] = select i1 [[ACTIVE]], i32 0, i32 305419896
; OOB-ACTIVE: br i1 [[ACTIVE]], label %[[STORE:.+]], label %[[EXIT:.+]]
; OOB-MASKED: [[RESTORED:%.+]] = lshr i32 [[SOURCE_EXEC]], {{%.+}}
; OOB-MASKED: [[LANE_ACTIVE:%.+]] = trunc i32 [[RESTORED]] to i1
; OOB-MASKED: [[STORE_ACTIVE:%.+]] = select i1 {{%.+}}, i1 [[LANE_ACTIVE]], i1 false
; OOB-MASKED: br i1 [[STORE_ACTIVE]], label %[[STORE:.+]], label %[[EXIT:.+]]
; OOB: [[STORE]]:
; OOB: store i32 [[VALUE0]], ptr addrspace(1)
; OOB: store i32 [[VALUE1]], ptr addrspace(1)
; OOB: store i32 [[VALUE2]], ptr addrspace(1)
; OOB: store i32 [[VALUE3]], ptr addrspace(1)
; OOB: store i32 [[VALUE4]], ptr addrspace(1)
; OOB: store i32 [[VALUE5]], ptr addrspace(1)
; OOB: br label %[[EXIT]]
; OOB: [[EXIT]]:
; OOB-NEXT: ret void
; ZERO-NOT: load i8
; ZERO-NOT: load i16
; ZERO-COUNT-6: store i32 0, ptr addrspace(1)
; ECC: store i32 [[LOW_U8]], ptr addrspace(1)
; MASKED: store i32 [[VALUE0]], ptr addrspace(1)
  global_store_b32 v0, v4, s[8:9]
; ECC: store i32 [[LOW_I8]], ptr addrspace(1)
; MASKED: store i32 [[VALUE1]], ptr addrspace(1)
  global_store_b32 v0, v5, s[8:9] offset:4
; ECC: store i32 [[LOW_B16]], ptr addrspace(1)
; MASKED: store i32 [[VALUE2]], ptr addrspace(1)
  global_store_b32 v0, v6, s[8:9] offset:8
; ECC: store i32 [[HIGH_U8]], ptr addrspace(1)
; MASKED: store i32 [[VALUE3]], ptr addrspace(1)
  global_store_b32 v0, v7, s[8:9] offset:12
; ECC: store i32 [[HIGH_I8]], ptr addrspace(1)
; MASKED: store i32 [[VALUE4]], ptr addrspace(1)
  global_store_b32 v0, v8, s[8:9] offset:16
; ECC: store i32 [[HIGH_B16]], ptr addrspace(1)
; MASKED: store i32 [[VALUE5]], ptr addrspace(1)
  global_store_b32 v0, v9, s[8:9] offset:20
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel buffer_d16
  .amdhsa_kernarg_size 24
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 10
  .amdhsa_next_free_sgpr 11
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: buffer_d16
    .symbol: buffer_d16.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .sgpr_count: 11
    .vgpr_count: 10
    .wavefront_size: 32
...
.end_amdgpu_metadata
