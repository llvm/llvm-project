// COM: Test HotSwap trampoline patch: tensor_load_to_lds multicast fix.
// COM: Prepends s_pack_hh_b32_b16 to clear multicast routing bits in
// COM: the descriptor's base SGPR. Base operand variants via NOP sled:
// COM:   dead SGPR  — only s_pack_hh prepended (no save/restore)
// COM:   live SGPR  — v_writelane save, s_pack_hh, tensor, v_readlane restore
// COM:   alt descriptor — different SGPR range (s[16:23]) for pack target
// COM:   SGPR redef — descriptor SGPR overwritten before use (dead path)
// COM: Verifies per-kernel behavior with CHECK-LABEL blocks and explicit
// COM: s_branch checks.
// COM:
// COM: Companion tests:
// COM:   hotswap-trampoline-tensor-nosled.s     — trampoline fallback path
// COM:   hotswap-trampoline-tensor-multi.s      — multi-site stacking
// COM:   hotswap-trampoline-tensor-liveness.s   — isSgprLiveAfter edge cases

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: --- Per-kernel checks ---

// COM: Kernel 1 (dead SGPR): s_branch forward to sled, s_pack_hh and
// COM: tensor_load_to_lds appear in sled area, s_branch back to original
// COM: stream. No v_writelane/v_readlane since descriptor SGPR is dead
// COM: (s_endpgm follows immediately).
// DISASM-LABEL: <test_tensor_dead>:
// DISASM-NOT: v_writelane_b32
// DISASM-NOT: v_readlane_b32
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: s_branch
// DISASM-NOT: v_writelane_b32
// DISASM-NOT: v_readlane_b32

// COM: Kernel 2 (live SGPR): s_branch forward to sled, then save/pack/
// COM: tensor/restore sequence in sled area with branch-back.
// COM: s4 is used after tensor_load_to_lds (s_mov reads it), so
// COM: save/restore via scratch VGPR is required.
// DISASM-LABEL: <test_tensor_live>:
// DISASM: s_branch
// DISASM: s_mov_b32
// DISASM: v_writelane_b32
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: v_readlane_b32
// DISASM: s_branch

// COM: Kernel 3 (alternate descriptor s[16:23]): verifies
// COM: getDescriptorBaseSgpr correctly extracts s16 from a different
// COM: SReg_256 range. s_pack_hh should target s16, not s4.
// COM: SGPR is dead (s_endpgm follows).
// DISASM-LABEL: <test_tensor_alt_descriptor>:
// DISASM-NOT: v_writelane_b32
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: s_pack_hh_b32_b16 s16
// DISASM: tensor_load_to_lds
// DISASM: s_branch

// COM: Kernel 4 (SGPR redefined before use): s4 is overwritten by
// COM: s_mov_b32 s4, 0 immediately after tensor_load, then s_endpgm.
// COM: isSgprLiveAfter sees a def-before-use and takes the dead path
// COM: — no save/restore needed.
// DISASM-LABEL: <test_tensor_sgpr_redef>:
// DISASM-NOT: v_writelane_b32
// DISASM-NOT: v_readlane_b32
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: s_branch
// DISASM-NOT: v_writelane_b32
// DISASM-NOT: v_readlane_b32

// COM: Idempotency: rewriting the output again should produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// ---- Kernel 1: tensor_load_to_lds with dead SGPR (s_endpgm follows) --------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_dead
.p2align 8
.type test_tensor_dead,@function
test_tensor_dead:
  tensor_load_to_lds s[0:3], s[4:11]
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
.Ltest_tensor_dead_end:
.size test_tensor_dead, .Ltest_tensor_dead_end-test_tensor_dead

// ---- Kernel 2: tensor_load_to_lds with live SGPR (s4 used after) -----------

.globl test_tensor_live
.p2align 8
.type test_tensor_live,@function
test_tensor_live:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
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
.Ltest_tensor_live_end:
.size test_tensor_live, .Ltest_tensor_live_end-test_tensor_live

// ---- Kernel 3: tensor_load_to_lds with alternate descriptor s[16:23] -------

.globl test_tensor_alt_descriptor
.p2align 8
.type test_tensor_alt_descriptor,@function
test_tensor_alt_descriptor:
  tensor_load_to_lds s[0:3], s[16:23]
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
.Ltest_tensor_alt_descriptor_end:
.size test_tensor_alt_descriptor, .Ltest_tensor_alt_descriptor_end-test_tensor_alt_descriptor

// ---- Kernel 4: tensor_load_to_lds with SGPR redefined (dead path) ----------

.globl test_tensor_sgpr_redef
.p2align 8
.type test_tensor_sgpr_redef,@function
test_tensor_sgpr_redef:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s4, 0
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
.Ltest_tensor_sgpr_redef_end:
.size test_tensor_sgpr_redef, .Ltest_tensor_sgpr_redef_end-test_tensor_sgpr_redef

.rodata
.p2align 8
.amdhsa_kernel test_tensor_dead
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_live
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_alt_descriptor
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_sgpr_redef
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
