// COM: WMMA-split shares emitToTrampoline with the other patch families. A
// COM: split site beyond s_branch's +-128 KB reach uses SGPR-backed set-PC
// COM: sequences on both edges and never executes s_add_pc_i64. External NOP
// COM: space supplies the forward gateway; non-NOP filler forces the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s

// COM: The site redirects through a safe gateway to two K=64 halves and
// COM: returns through the SCC-neutral set-PC sequence.
// DISASM-LABEL: <test_wsplit_far>:
// DISASM-NEXT: s_branch
// DISASM: s_get_pc_i64 s[0:1]
// DISASM-NEXT: s_add_nc_u64 s[0:1], s[0:1],
// DISASM-NEXT: s_set_pc_i64 s[0:1]
// DISASM: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: s_get_pc_i64 s[0:1]
// DISASM-NEXT: s_add_nc_u64 s[0:1], s[0:1],
// DISASM-NEXT: s_set_pc_i64 s[0:1]

// COM: Idempotency: rewriting the output again must be a no-op.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wsplit_far
.p2align 8
.type test_wsplit_far,@function
test_wsplit_far:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
.size test_wsplit_far, .-test_wsplit_far

// Safe external gateway space after the kernel's no-fallthrough terminator.
.rept 8
  s_nop 0
.endr

  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the WMMA above (forces the long-branch path).
  .rept 40000
    s_mov_b32 s0, s1
  .endr

.rodata
.p2align 8
.amdhsa_kernel test_wsplit_far
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wsplit_far
      .symbol: test_wsplit_far.kd
      .sgpr_count: 0
      .vgpr_count: 40
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
