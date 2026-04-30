// COM: Test HotSwap WMMA co-execution hazard patch: a WMMA integer
// COM: instruction (A0 needs 8 v_nops vs B0's 4) followed by an
// COM: overlapping VALU should get v_nop padding inserted.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Verify the patched layout. v_wmma_i32_16x16x64_iu8 needs 8 v_nops on
// COM: A0; the kernel body has 0 pre-existing safe slots, so the original
// COM: VALU site must be replaced by an s_branch to a trampoline that
// COM: contains exactly 8 v_nops immediately followed by the relocated
// COM: VALU. CHECK-COUNT-8 asserts the count and CHECK-NEXT pins it: any
// COM: deviation (7 or 9 v_nops) breaks the chain.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: v_wmma_i32_16x16x64_iu8
// DISASM-NEXT: s_branch
// DISASM: s_endpgm
// DISASM-COUNT-8: v_nop
// DISASM-NEXT: v_add_f32

// COM: Idempotency: second rewrite should produce identical output
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_hazard
.p2align 8
.type test_wmma_hazard,@function
test_wmma_hazard:
  // WMMA integer instruction: A0 needs 8 nops, B0 needs 4
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  // VALU that overlaps WMMA dest (writes v16) -- should trigger hazard
  v_add_f32 v16, v0, v1
  s_endpgm
.Ltest_wmma_hazard_end:
.size test_wmma_hazard, .Ltest_wmma_hazard_end-test_wmma_hazard

.rodata
.p2align 8
.amdhsa_kernel test_wmma_hazard
  .amdhsa_next_free_vgpr 24
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
