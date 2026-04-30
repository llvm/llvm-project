// COM: Test WMMA hazard with sufficient pre-existing v_nops: 8 v_nops
// COM: already present between WMMA (needs 8) and overlapping VALU.
// COM: No additional padding should be inserted.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: 8 pre-existing v_nops between WMMA and the overlapping VALU already
// COM: meet A0's requirement, so the patch must not insert any padding.
// COM: The disassembly must remain WMMA -> 8 v_nops -> v_add_f32 ->
// COM: s_endpgm with no s_branch anywhere (no in-body branch, no trampoline
// COM: appended after s_endpgm). CHECK-COUNT-8 + CHECK-NEXT chain pins the
// COM: layout exactly; the trailing CHECK-NOT covers the post-kernel range.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: v_wmma_i32_16x16x64_iu8
// DISASM-COUNT-8: v_nop
// DISASM-NEXT: v_add_f32
// DISASM-NEXT: s_endpgm
// DISASM-NOT: s_branch

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_safe
.p2align 8
.type test_wmma_safe,@function
test_wmma_safe:
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  v_nop
  v_nop
  v_nop
  v_nop
  v_nop
  v_nop
  v_nop
  v_nop
  // 8 v_nops -- sufficient for A0, no patch needed
  v_add_f32 v16, v0, v1
  s_endpgm
.Ltest_wmma_safe_end:
.size test_wmma_safe, .Ltest_wmma_safe_end-test_wmma_safe

.rodata
.p2align 8
.amdhsa_kernel test_wmma_safe
  .amdhsa_next_free_vgpr 24
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
