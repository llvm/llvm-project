// COM: Test HotSwap trampoline patch: ds_*_addtid_b32 expansion.
// COM:
// COM: On gfx1250 A0 the DS unit truncates M0 to 16 bits, so ADDTID address
// COM: encodings (M0 + lane_id*4 + offset) silently wrap above 64KB
// COM: on gfx1250. The trampoline materialises the lane-id math in the ALU
// COM: using M0 masked to 20 bits (matching B0's DS-unit M0 read width) and
// COM: issues a regular ds_load_b32 / ds_store_b32, bypassing the buggy
// COM: address path.
// COM:
// COM: Coverage:
// COM:   test_addtid_load        : ds_load_addtid_b32 + offset (NOP sled)
// COM:   test_addtid_load_zero   : ds_load_addtid_b32 + offset:0
// COM:   test_addtid_store       : ds_store_addtid_b32 needs a scratch VGPR
// COM:
// COM: DISASM-NEXT vs DISASM convention used throughout this file: the
// COM: original ADDTID site is replaced in place by an s_branch, then the
// COM: NOP-sled padding follows (variable size, depends on how many s_nops
// COM: were available inside the kernel) and only after the sled does the
// COM: trampoline body start. The gap is bridged with a non-consecutive
// COM: 'DISASM:' on v_mbcnt_lo so FileCheck skips over the sled NOPs;
// COM: every instruction inside the trampoline body is then chained with
// COM: 'DISASM-NEXT:' so the body itself is verified bit-for-bit.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// ---- Kernel 1: ds_load_addtid_b32 with offset --------------------------------
//
// COM: Original site is replaced with s_branch (forward to NOP sled). The
// COM: sled body computes lane_id*4 + m0 in the load's destination VGPR (v5),
// COM: masks the result to 20 bits to match B0's DS-unit M0 read width, then
// COM: reads LDS with the original offset folded into the DS encoding, and
// COM: finally s_branch returns to the instruction following the original.
// COM: Operand-pinned matches catch any regression that drops the offset,
// COM: swaps operand order, or changes the shift / add scalar source.

// DISASM-LABEL: <test_addtid_load>:
// DISASM-NOT:   ds_load_addtid_b32
// DISASM:       s_branch
// DISASM:       v_mbcnt_lo_u32_b32 v5, -1, 0
// DISASM-NEXT:  v_mbcnt_hi_u32_b32 v5, -1, v5
// DISASM-NEXT:  v_lshlrev_b32_e32 v5, 2, v5
// DISASM-NEXT:  v_add_nc_u32_e32 v5, m0, v5
// DISASM-NEXT:  v_and_b32_e32 v5, 0xfffff, v5
// DISASM-NEXT:  ds_load_b32 v5, v5 offset:128
// DISASM-NEXT:  s_branch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_addtid_load
.p2align 8
.type test_addtid_load,@function
test_addtid_load:
  ds_load_addtid_b32 v5 offset:128
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_addtid_load_end:
.size test_addtid_load, .Ltest_addtid_load_end-test_addtid_load

.rodata
.p2align 8
.amdhsa_kernel test_addtid_load
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

// ---- Kernel 2: ds_load_addtid_b32 with offset:0 ------------------------------
//
// COM: Same expansion as kernel 1 but with offset:0. The disassembler omits
// COM: the offset suffix entirely when the encoded offset is zero, so the
// COM: ds_load_b32 line drops `offset:N` -- the test pins this so a future
// COM: change that always emits `offset:0` would surface here.

// DISASM-LABEL: <test_addtid_load_zero>:
// DISASM-NOT:   ds_load_addtid_b32
// DISASM:       s_branch
// DISASM:       v_mbcnt_lo_u32_b32 v6, -1, 0
// DISASM-NEXT:  v_mbcnt_hi_u32_b32 v6, -1, v6
// DISASM-NEXT:  v_lshlrev_b32_e32 v6, 2, v6
// DISASM-NEXT:  v_add_nc_u32_e32 v6, m0, v6
// DISASM-NEXT:  v_and_b32_e32 v6, 0xfffff, v6
// DISASM-NEXT:  ds_load_b32 v6, v6
// DISASM-NEXT:  s_branch

.text
.globl test_addtid_load_zero
.p2align 8
.type test_addtid_load_zero,@function
test_addtid_load_zero:
  ds_load_addtid_b32 v6 offset:0
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_addtid_load_zero_end:
.size test_addtid_load_zero, .Ltest_addtid_load_zero_end-test_addtid_load_zero

.rodata
.amdhsa_kernel test_addtid_load_zero
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 1
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

// ---- Kernel 3: ds_store_addtid_b32 ------------------------------------------
//
// COM: Store path needs a separate scratch VGPR for the address-compute
// COM: temporary because the original data VGPR (v8) must be preserved as
// COM: the store source. The scratch register comes from tryAllocScratchVgpr;
// COM: the exact index varies with liveness, so we capture it through a
// COM: FileCheck regex variable [[VTMP]] and pin it across the body. Most
// COM: importantly: the ds_store_b32 operand order must be (addr, data) =
// COM: ([[VTMP]], v8) -- a swap would silently corrupt the LDS layout.

// DISASM-LABEL: <test_addtid_store>:
// DISASM-NOT:   ds_store_addtid_b32
// DISASM:       s_branch
// DISASM:       v_mbcnt_lo_u32_b32 [[VTMP:v[0-9]+]], -1, 0
// DISASM-NEXT:  v_mbcnt_hi_u32_b32 [[VTMP]], -1, [[VTMP]]
// DISASM-NEXT:  v_lshlrev_b32_e32 [[VTMP]], 2, [[VTMP]]
// DISASM-NEXT:  v_add_nc_u32_e32 [[VTMP]], m0, [[VTMP]]
// DISASM-NEXT:  v_and_b32_e32 [[VTMP]], 0xfffff, [[VTMP]]
// DISASM-NEXT:  ds_store_b32 [[VTMP]], v8 offset:64
// DISASM-NEXT:  s_branch

.text
.globl test_addtid_store
.p2align 8
.type test_addtid_store,@function
test_addtid_store:
  ds_store_addtid_b32 v8 offset:64
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_addtid_store_end:
.size test_addtid_store, .Ltest_addtid_store_end-test_addtid_store

.rodata
.amdhsa_kernel test_addtid_store
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 1
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_addtid_load
      .symbol: test_addtid_load.kd
      .sgpr_count: 1
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: test_addtid_load_zero
      .symbol: test_addtid_load_zero.kd
      .sgpr_count: 1
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: test_addtid_store
      .symbol: test_addtid_store.kd
      .sgpr_count: 1
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata

// COM: Idempotency: rewriting the output a second time must produce
// COM: identical bytes (the patched body has no ADDTID mnemonic so the
// COM: dispatcher leaves it untouched on subsequent runs).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES
