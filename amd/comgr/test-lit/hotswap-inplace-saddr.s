// COM: Test HotSwap cluster_load addressing-form selectivity. The in-place
// COM: B0->A0 replacement templates target the saddr=off (64-bit vaddr)
// COM: encoding. The SGPR-relative (_SADDR) variant shares the display
// COM: mnemonic but is a distinct MC opcode with a different operand layout,
// COM: so reusing the off-form opcode would mis-encode its operands and
// COM: corrupt the address at runtime. The in-place pass must skip _SADDR,
// COM: then the trampoline pass must wrap that remaining cluster_load with
// COM: the A0 M0 wg_mask save/clear/restore sequence.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The original _SADDR site is redirected to a sled/trampoline. The
// COM: remaining cluster_load body is bracketed by M0 save, M0 wg_mask
// COM: clear, and M0 restore.
// DISASM-LABEL: <test_saddr_kernel>:
// DISASM: s_branch
// DISASM-NOT: cluster_load_b32 {{.*}}, off
// COM: The saddr=off site that followed is rewritten to global_load_b32,
// COM: proving the skip is specific to the _SADDR form, not a blanket opt-out.
// DISASM: global_load_b32 v{{[0-9]+}}, v[{{[0-9:]+}}], off
// DISASM: s_endpgm
// DISASM: s_mov_b32 [[SCRATCH:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NEXT: s_mov_b32 m0, [[SCRATCH]]

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_saddr_kernel
.p2align 8
.type test_saddr_kernel,@function
test_saddr_kernel:
  // SGPR-relative (SADDR) form -- must be left unchanged.
  cluster_load_b32 v4, v1, s[2:3]
  s_wait_loadcnt 0x0
  // saddr=off form -- must be swapped to global_load_b32.
  cluster_load_b32 v5, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_saddr_kernel_end:
.size test_saddr_kernel, .Ltest_saddr_kernel_end-test_saddr_kernel

.rodata
.p2align 8
.amdhsa_kernel test_saddr_kernel
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_saddr_kernel
      .symbol: test_saddr_kernel.kd
      .sgpr_count: 4
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
