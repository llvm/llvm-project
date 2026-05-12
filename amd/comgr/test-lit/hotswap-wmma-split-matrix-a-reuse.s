// Test that matrix_a_reuse is stripped on both halves of a K-split.
//
// matrix_a_reuse is a HW data-reuse hint asserting that the A matrix
// is identical to the previous WMMA's A. After a K-split, A is sliced
// into halves -- the data layout assumption no longer holds, so
// preserving the hint would make the hardware reuse stale data. The
// splitter strips the modifier on both halves (no perf hint, but
// correct semantics).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_bf8
// DISASM:       s_branch
// DISASM:       s_endpgm

// COM: Both halves end at `//` immediately after the operand list (no
// COM: matrix_a_reuse modifier suffix on either).
// DISASM:       v_wmma_f32_16x16x64_fp8_bf8 v[40:47], v[0:7], v[8:15], v[40:47]{{[[:space:]]*\/\/}}
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_bf8 v[40:47], v[8:15], v[16:23], v[40:47]{{[[:space:]]*\/\/}}
// DISASM-NEXT:  s_branch

// COM: Sanity: matrix_a_reuse must NOT appear anywhere on the K=64
// COM: replacement instructions for this kernel.
// DISASM-NOT:   v_wmma_f32_16x16x64_fp8_bf8 v[40:47]{{.*}}matrix_a_reuse
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_16x16x128_fp8_bf8 v[40:47], v[0:15], v[8:23], v[40:47] matrix_a_reuse
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
