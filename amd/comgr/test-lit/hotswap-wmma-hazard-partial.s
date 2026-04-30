// COM: Test WMMA hazard with pre-existing v_nops: 3 v_nops already present
// COM: between WMMA (needs 8) and overlapping VALU. Should insert 5 more.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Verify the patched layout. v_wmma_i32_16x16x64_iu8 needs 8 v_nops on
// COM: A0. The kernel body has 3 pre-existing v_nops before the hazardous
// COM: VALU; the patch must keep them, replace the VALU with an s_branch
// COM: to a trampoline, and emit exactly 5 v_nops (the deficit = 8 - 3)
// COM: immediately before the relocated VALU. CHECK-NEXT pins the in-body
// COM: nop count and CHECK-COUNT-5 + CHECK-NEXT pin the trampoline count.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: v_wmma_i32_16x16x64_iu8
// DISASM-NEXT: v_nop
// DISASM-NEXT: v_nop
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_branch
// DISASM: s_endpgm
// DISASM-COUNT-5: v_nop
// DISASM-NEXT: v_add_f32

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_partial
.p2align 8
.type test_wmma_partial,@function
test_wmma_partial:
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  v_nop
  v_nop
  v_nop
  // Only 3 v_nops -- need 8 for A0, so 5 more should be inserted
  v_add_f32 v16, v0, v1
  s_endpgm
.Ltest_wmma_partial_end:
.size test_wmma_partial, .Ltest_wmma_partial_end-test_wmma_partial

.rodata
.p2align 8
.amdhsa_kernel test_wmma_partial
  .amdhsa_next_free_vgpr 24
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
