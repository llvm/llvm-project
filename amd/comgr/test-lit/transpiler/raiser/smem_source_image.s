; REQUIRES: comgr-has-transpiler

; The raiser fixtures assemble for gfx1250, which is also the ISA that spells
; the 64-bit scalar arithmetic these PC-relative chains use.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=pcrel_dword_kernel,pcrel_pair_kernel \
; RUN:   --emit-ir=pcrel_triple_kernel,pcrel_quad_kernel,pcrel_eight_kernel \
; RUN:   --emit-ir=pcrel_sixteen_kernel,pcrel_add_kernel,pcrel_sub_kernel \
; RUN:   --emit-ir=pcrel_mov_copy_kernel,pcrel_back_offset_kernel \
; RUN:   --emit-ir=pcrel_rodata_kernel \
; RUN:   --emit-ir=pcrel_lit32_kernel,pcrel_lit64_kernel \
; RUN:   --emit-ir=pcrel_signed_add_kernel,pcrel_signed_back_kernel \
; RUN:   --emit-ir=pcrel_cmp_eq_kernel,pcrel_cmp_lg_kernel \
; RUN:   | %FileCheck %s

; A kernel whose source address the raise cannot resolve to a literal is
; refused one kernel at a time, because the first refusal ends the run.

; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_dynamic_offset_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=DYNAMIC
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_outside_image_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=OUTSIDE
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_below_image_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=BELOW
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_untracked_dst_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=UNTRACKED
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_register_offset_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=REGOFFSET
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_two_addresses_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=TWOADDR
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_reversed_sub_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=REVERSEDSUB
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_other_block_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=OTHERBLOCK
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_clobbered_block_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=CLOBBERED
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_signed_carry_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=SIGNEDCARRY
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_compare_dynamic_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=COMPAREDYNAMIC
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_compare_absolute_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=COMPAREABSOLUTE
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_signed_add_scc_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=SIGNEDADDSCC
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_split_add_scc_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=SPLITADDSCC
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_scc_other_block_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=SCCOTHERBLOCK
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_escape_low_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=ESCAPELOW
; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_escape_high_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=ESCAPEHIGH

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; s_get_pc_i64 names the address the next instruction sits at in the source
; code object, and a scalar load off it reads a literal the source compiled in
; rather than anything the raised kernel has mapped. Each kernel below folds
; one such load and hands the result to a vector add, which keeps the literal
; in the IR where a constant-folded scalar consumer would have absorbed it.

	.globl	pcrel_dword_kernel
	.p2align	8
	.type	pcrel_dword_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_dword_kernel(
pcrel_dword_kernel:
	s_get_pc_i64 s[0:1]
; The capture sits four bytes into the kernel, and the table twenty bytes past
; that.
	s_load_b32 s2, s[0:1], 0x14
	s_wait_kmcnt 0x0
; CHECK: add i32 -559038737,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0xdeadbeef

	.globl	pcrel_pair_kernel
	.p2align	8
	.type	pcrel_pair_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_pair_kernel(
pcrel_pair_kernel:
	s_get_pc_i64 s[0:1]
	s_load_b64 s[2:3], s[0:1], 0x18
	s_wait_kmcnt 0x0
; A pair is one 64-bit constant, which the register file splits back into the
; dwords the source wrote.
; CHECK: add i32 286331153,
	v_add_nc_u32 v0, s2, v0
; CHECK: add i32 572662306,
	v_add_nc_u32 v0, s3, v0
	s_endpgm
	.long	0x11111111
	.long	0x22222222

	.globl	pcrel_triple_kernel
	.p2align	8
	.type	pcrel_triple_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_triple_kernel(
pcrel_triple_kernel:
	s_get_pc_i64 s[0:1]
	s_load_b96 s[4:6], s[0:1], 0x14
	s_wait_kmcnt 0x0
; CHECK: <3 x i32> <i32 1, i32 2, i32 3>
	v_add_nc_u32 v0, s4, v0
	s_endpgm
	.long	1
	.long	2
	.long	3

	.globl	pcrel_quad_kernel
	.p2align	8
	.type	pcrel_quad_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_quad_kernel(
pcrel_quad_kernel:
	s_get_pc_i64 s[0:1]
; A load wider than a pair is one vector constant, which the register file
; distributes across the tuple.
	s_load_b128 s[4:7], s[0:1], 0x14
	s_wait_kmcnt 0x0
; CHECK: <4 x i32> <i32 1, i32 2, i32 3, i32 4>
	v_add_nc_u32 v0, s4, v0
	s_endpgm
	.long	1
	.long	2
	.long	3
	.long	4

	.globl	pcrel_eight_kernel
	.p2align	8
	.type	pcrel_eight_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_eight_kernel(
pcrel_eight_kernel:
	s_get_pc_i64 s[0:1]
	s_load_b256 s[4:11], s[0:1], 0x14
	s_wait_kmcnt 0x0
; CHECK: <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
	v_add_nc_u32 v0, s4, v0
	s_endpgm
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.long	8

	.globl	pcrel_sixteen_kernel
	.p2align	8
	.type	pcrel_sixteen_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_sixteen_kernel(
pcrel_sixteen_kernel:
	s_get_pc_i64 s[0:1]
	s_load_b512 s[4:19], s[0:1], 0x14
	s_wait_kmcnt 0x0
; CHECK: <16 x i32> <i32 1, i32 2, {{.+}} i32 16>
	v_add_nc_u32 v0, s4, v0
	s_endpgm
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.long	8
	.long	9
	.long	10
	.long	11
	.long	12
	.long	13
	.long	14
	.long	15
	.long	16

	.globl	pcrel_add_kernel
	.p2align	8
	.type	pcrel_add_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_add_kernel(
pcrel_add_kernel:
	s_get_pc_i64 s[0:1]
; The captured address stays one wherever the constant sits, so the load still
; reads the table rather than target memory.
	s_add_nc_u64 s[0:1], 8, s[0:1]
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
; CHECK: add i32 287454020,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0x11223344

	.globl	pcrel_sub_kernel
	.p2align	8
	.type	pcrel_sub_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_sub_kernel(
pcrel_sub_kernel:
	s_get_pc_i64 s[0:1]
; Subtracting a negative constant moves the captured address forward by eight.
	s_sub_nc_u64 s[0:1], s[0:1], -8
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
; CHECK: add i32 -1430532899,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0xaabbccdd

	.globl	pcrel_mov_copy_kernel
	.p2align	8
	.type	pcrel_mov_copy_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_mov_copy_kernel(
pcrel_mov_copy_kernel:
	s_get_pc_i64 s[0:1]
; A copy of the captured pair addresses the source code object too, so the
; load off the copy reads the table.
	s_mov_b64 s[2:3], s[0:1]
	s_load_b32 s4, s[2:3], 0x18
	s_wait_kmcnt 0x0
; CHECK: add i32 16909060,
	v_add_nc_u32 v0, s4, v0
	s_endpgm
	.long	0x01020304

	.globl	pcrel_back_offset_kernel
	.p2align	8
	.type	pcrel_back_offset_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_back_offset_kernel(
pcrel_back_offset_kernel:
	s_get_pc_i64 s[0:1]
; The scalar-load offset is signed, so a base the source moved past the table
; reads it again by going back eight bytes.
	s_add_nc_u64 s[0:1], s[0:1], 32
	s_load_b32 s2, s[0:1], -0x8
	s_wait_kmcnt 0x0
; CHECK: add i32 -1412567278,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0xabcdef12

	.globl	pcrel_rodata_kernel
	.p2align	8
	.type	pcrel_rodata_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_rodata_kernel(
pcrel_rodata_kernel:
	s_get_pc_i64 s[0:1]
.Lrodata_capture:
; A literal the source compiled into its read-only data is as much part of the
; captured image as one sitting between instructions.
	s_add_nc_u64 s[0:1], s[0:1], (pcrel_rodata_literal-.Lrodata_capture)
	s_load_b32 s2, s[0:1], 0x0
	s_wait_kmcnt 0x0
; CHECK: add i32 -889262067,
	v_add_nc_u32 v0, s2, v0
	s_endpgm

	.globl	pcrel_lit32_kernel
	.p2align	8
	.type	pcrel_lit32_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_lit32_kernel(
pcrel_lit32_kernel:
	s_get_pc_i64 s[0:1]
; A 32-bit literal spelling a value an inline constant could have held is kept
; apart from that encoding as an expression, not an immediate.
	s_add_nc_u64 s[0:1], s[0:1], lit(0xc)
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
; CHECK: add i32 878082202,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0x3456789a

	.globl	pcrel_lit64_kernel
	.p2align	8
	.type	pcrel_lit64_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_lit64_kernel(
pcrel_lit64_kernel:
	s_get_pc_i64 s[0:1]
; A 64-bit literal reaches the decode as an expression too, and carries the
; displacement just as the immediate forms above do.
	s_add_nc_u64 s[0:1], s[0:1], lit64(0x10)
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
; CHECK: add i32 591751049,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0x23456789

	.globl	pcrel_signed_add_kernel
	.p2align	8
	.type	pcrel_signed_add_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_signed_add_kernel(
pcrel_signed_add_kernel:
	s_get_pc_i64 s[0:1]
; The positive add does not carry into the unchanged high half.
	s_add_i32 s0, s0, 8
	s_load_b32 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
; CHECK: add i32 287454020,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0x11223344

	.globl	pcrel_signed_back_kernel
	.p2align	8
	.type	pcrel_signed_back_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_signed_back_kernel(
pcrel_signed_back_kernel:
	s_get_pc_i64 s[0:1]
; The negative add carries out of the low half, so the unchanged high half
; still represents the correct source address.
	s_add_i32 s0, s0, 36
	s_add_i32 s0, s0, -8
	s_load_b32 s2, s[0:1], 0x0
	s_wait_kmcnt 0x0
; CHECK: add i32 -1412567278,
	v_add_nc_u32 v0, s2, v0
	s_endpgm
	.long	0xabcdef12

	.globl	pcrel_cmp_eq_kernel
	.p2align	8
	.type	pcrel_cmp_eq_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_cmp_eq_kernel(
pcrel_cmp_eq_kernel:
	s_get_pc_i64 s[0:1]
; The source address is known at raise time, so compare it as a constant.
	s_cmp_eq_u64 s[0:1], 0
; CHECK: select i1 false, i32 1, i32 2
	s_cselect_b32 s2, 1, 2
	v_add_nc_u32 v0, s2, v0
	s_endpgm

	.globl	pcrel_cmp_lg_kernel
	.p2align	8
	.type	pcrel_cmp_lg_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @pcrel_cmp_lg_kernel(
pcrel_cmp_lg_kernel:
	s_get_pc_i64 s[0:1]
; The known source address differs from zero.
	s_cmp_lg_u64 s[0:1], 0
; CHECK: select i1 true, i32 1, i32 2
	s_cselect_b32 s2, 1, 2
	v_add_nc_u32 v0, s2, v0
	s_endpgm

; A source address the raise cannot resolve to a literal is refused, since
; letting the load through would read target memory at a source address.

	.globl	refuse_dynamic_offset_kernel
	.p2align	8
	.type	refuse_dynamic_offset_kernel,@function
refuse_dynamic_offset_kernel:
	s_get_pc_i64 s[0:1]
; DYNAMIC: unsupported-instruction-form: s_load_b32 {{.+}} :: reads the source code object at an offset only the running kernel knows
	s_load_b32 s2, s[0:1], s3 offset:0x0
	s_endpgm

	.globl	refuse_outside_image_kernel
	.p2align	8
	.type	refuse_outside_image_kernel,@function
refuse_outside_image_kernel:
	s_get_pc_i64 s[0:1]
	s_add_nc_u64 s[0:1], s[0:1], 0x100000
; OUTSIDE: unsupported-instruction-form: s_load_b32 {{.+}} :: reads a source address that no section of the source code object covers
	s_load_b32 s2, s[0:1], 0x0
	s_endpgm

	.globl	refuse_below_image_kernel
	.p2align	8
	.type	refuse_below_image_kernel,@function
refuse_below_image_kernel:
	s_get_pc_i64 s[0:1]
; BELOW: unsupported-instruction-form: s_sub_nc_u64 {{.+}} :: moves a source address out of the address space
	s_sub_nc_u64 s[0:1], s[0:1], 0x10000000
	s_load_b32 s2, s[0:1], 0x0
	s_endpgm

	.globl	refuse_untracked_dst_kernel
	.p2align	8
	.type	refuse_untracked_dst_kernel,@function
refuse_untracked_dst_kernel:
	s_get_pc_i64 s[0:1]
; UNTRACKED: unsupported-instruction-form: s_add_nc_u64 {{.+}} :: puts a source address outside an SGPR pair, which is not tracked as one
	s_add_nc_u64 vcc, s[0:1], 8
	s_endpgm

	.globl	refuse_register_offset_kernel
	.p2align	8
	.type	refuse_register_offset_kernel,@function
refuse_register_offset_kernel:
	s_get_pc_i64 s[0:1]
; REGOFFSET: unsupported-instruction-form: s_add_nc_u64 {{.+}} :: displaces a source address by a register value only the running kernel knows
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_endpgm

	.globl	refuse_two_addresses_kernel
	.p2align	8
	.type	refuse_two_addresses_kernel,@function
refuse_two_addresses_kernel:
	s_get_pc_i64 s[0:1]
	s_mov_b64 s[2:3], s[0:1]
; TWOADDR: unsupported-instruction-form: s_add_nc_u64 {{.+}} :: combines two source addresses, which names nothing in the source code object
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	s_endpgm

	.globl	refuse_reversed_sub_kernel
	.p2align	8
	.type	refuse_reversed_sub_kernel,@function
refuse_reversed_sub_kernel:
	s_get_pc_i64 s[0:1]
; REVERSEDSUB: unsupported-instruction-form: s_sub_nc_u64 {{.+}} :: subtracts a source address from a constant, which is no source address
	s_sub_nc_u64 s[2:3], 8, s[0:1]
	s_endpgm

	.globl	refuse_other_block_kernel
	.p2align	8
	.type	refuse_other_block_kernel,@function
refuse_other_block_kernel:
; The captured address holds only in the block that captured it, so the load
; the branch leads to names an address the raise can no longer resolve.
	s_get_pc_i64 s[0:1]
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lother_block
	s_endpgm
.Lother_block:
; OTHERBLOCK: unsupported-instruction-form: s_load_b32 {{.+}} :: uses a source address that a block boundary or a write to one half of the pair has dropped
	s_load_b32 s3, s[0:1], 0x0
	s_endpgm

	.globl	refuse_clobbered_block_kernel
	.p2align	8
	.type	refuse_clobbered_block_kernel,@function
refuse_clobbered_block_kernel:
; Writing the pair in one block says nothing about what another block holds
; there, so the load is still refused rather than read from target memory.
	s_get_pc_i64 s[0:1]
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lclobbered_block
	s_mov_b32 s0, 0
	s_endpgm
.Lclobbered_block:
; CLOBBERED: unsupported-instruction-form: s_load_b32 {{.+}} :: uses a source address that a block boundary or a write to one half of the pair has dropped
	s_load_b32 s3, s[0:1], 0x0
	s_endpgm

; A source address stands for a place in the captured image and names nothing
; the raised kernel can address, so a read that would hand either of its halves
; to the target program is refused.

	.globl	refuse_signed_carry_kernel
	.p2align	8
	.type	refuse_signed_carry_kernel,@function
refuse_signed_carry_kernel:
	s_get_pc_i64 s[0:1]
; A signed add does not produce the carry required by s_addc_u32.
	s_add_i32 s0, s0, 8
; SIGNEDCARRY: unsupported-instruction-form: s_add_co_ci_u32 {{.+}} :: operand-read: {{.+}} may hold a source code-object address
	s_addc_u32 s1, s1, 0
	s_endpgm

	.globl	refuse_compare_dynamic_kernel
	.p2align	8
	.type	refuse_compare_dynamic_kernel,@function
refuse_compare_dynamic_kernel:
	s_get_pc_i64 s[0:1]
; The second operand is known only at runtime, so the comparison is refused.
; COMPAREDYNAMIC: unsupported-instruction-form: s_cmp_eq_u64 {{.+}} :: operand-read: {{.+}} may hold a source code-object address
	s_cmp_eq_u64 s[0:1], s[2:3]
	s_endpgm

	.globl	refuse_compare_absolute_kernel
	.p2align	8
	.type	refuse_compare_absolute_kernel,@function
refuse_compare_absolute_kernel:
	s_get_pc_i64 s[0:1]
; A nonzero absolute comparison can change when the code is relocated.
; COMPAREABSOLUTE: unsupported-instruction-form: s_cmp_eq_u64 {{.+}} :: operand-read: {{.+}} may hold a source code-object address
	s_cmp_eq_u64 s[0:1], 1
	s_endpgm

; A displacement moves the whole 64-bit address, which the loader moves with
; it. The carry and the overflow the halves produce along the way answer for
; the address the ELF file records rather than the one the loader hands the
; kernel, so SCC is refused to whatever would read it as a reliable flag.

	.globl	refuse_signed_add_scc_kernel
	.p2align	8
	.type	refuse_signed_add_scc_kernel,@function
refuse_signed_add_scc_kernel:
	s_get_pc_i64 s[0:1]
	s_add_i32 s0, s0, 8
; SIGNEDADDSCC: unsupported-instruction-form: s_cselect_b32 {{.+}} :: operand-read: 'scc' may hold a carry or overflow derived from a source code-object address; relocation may change the result
	s_cselect_b32 s2, 1, 2
	s_endpgm

	.globl	refuse_split_add_scc_kernel
	.p2align	8
	.type	refuse_split_add_scc_kernel,@function
refuse_split_add_scc_kernel:
; The pair displaces the address it holds, and still leaves the high half's own
; carry behind in SCC.
	s_get_pc_i64 s[0:1]
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
; SPLITADDSCC: unsupported-instruction-form: s_cselect_b32 {{.+}} :: operand-read: 'scc' may hold a carry or overflow derived from a source code-object address; relocation may change the result
	s_cselect_b32 s2, 1, 2
	s_endpgm

	.globl	refuse_scc_other_block_kernel
	.p2align	8
	.type	refuse_scc_other_block_kernel,@function
refuse_scc_other_block_kernel:
; A block that does not write SCC itself reads what the blocks before it left,
; so the refusal has to outlive the block the displacement happened in.
	s_get_pc_i64 s[0:1]
	s_add_u32 s0, s0, 8
	s_branch .Lscc_other_block
.Lscc_other_block:
; SCCOTHERBLOCK: unsupported-instruction-form: s_cselect_b32 {{.+}} :: operand-read: 'scc' may hold a carry or overflow derived from a source code-object address; relocation may change the result
	s_cselect_b32 s2, 1, 2
	s_endpgm

	.globl	refuse_escape_low_kernel
	.p2align	8
	.type	refuse_escape_low_kernel,@function
refuse_escape_low_kernel:
	s_get_pc_i64 s[0:1]
; ESCAPELOW: unsupported-instruction-form: s_mov_b32 {{.+}} :: operand-read: {{.+}} may hold a source code-object address
	s_mov_b32 s2, s0
	s_endpgm

	.globl	refuse_escape_high_kernel
	.p2align	8
	.type	refuse_escape_high_kernel,@function
refuse_escape_high_kernel:
	s_get_pc_i64 s[0:1]
; ESCAPEHIGH: unsupported-instruction-form: v_add_nc_u32 {{.+}} :: operand-read: {{.+}} may hold a source code-object address
	v_add_nc_u32 v0, s1, v0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pcrel_dword_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_pair_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_triple_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_quad_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_eight_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_sixteen_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_add_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_sub_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_mov_copy_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_back_offset_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_rodata_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_lit32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_lit64_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_signed_add_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_signed_back_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_signed_carry_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_cmp_eq_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel pcrel_cmp_lg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_compare_dynamic_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_compare_absolute_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_signed_add_scc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_split_add_scc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_scc_other_block_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_dynamic_offset_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_outside_image_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_below_image_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_untracked_dst_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_register_offset_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_two_addresses_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_reversed_sub_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_other_block_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_clobbered_block_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_escape_low_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_escape_high_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
pcrel_rodata_literal:
	.long	0xcafef00d
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_dword_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_dword_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_pair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_pair_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_triple_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_triple_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_quad_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_quad_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_eight_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_eight_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_sixteen_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_sixteen_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_add_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_add_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_sub_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_sub_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_mov_copy_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_mov_copy_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_back_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_back_offset_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_rodata_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_rodata_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_lit32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_lit32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_lit64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_lit64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_signed_add_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_signed_add_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_signed_back_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_signed_back_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_signed_carry_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_signed_carry_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_cmp_eq_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_cmp_eq_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           pcrel_cmp_lg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         pcrel_cmp_lg_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_compare_dynamic_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_compare_dynamic_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_compare_absolute_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_compare_absolute_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_signed_add_scc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_signed_add_scc_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_split_add_scc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_split_add_scc_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_scc_other_block_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_scc_other_block_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_dynamic_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_dynamic_offset_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_outside_image_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_outside_image_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_below_image_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_below_image_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_untracked_dst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_untracked_dst_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_register_offset_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_register_offset_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_two_addresses_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_two_addresses_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_reversed_sub_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_reversed_sub_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_other_block_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_other_block_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_clobbered_block_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_clobbered_block_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_escape_low_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_escape_low_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_escape_high_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_escape_high_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
