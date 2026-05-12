// Test multiple K=128 WMMAs in a single kernel. The splitter accumulates
// trampolines into Ctx.OutTrampolines and computes each new trampoline's
// `.text` offset by walking the previously appended trampolines (the
// TrampTextOffset accumulation pattern in applyWmmaSplitPatches). A bug in
// that accumulation would land the s_branch-back from trampoline N at the
// wrong target -- typically jumping into the next trampoline's body or off
// the end of .text. Putting two WMMAs in one kernel and asserting both
// landing pads carry the expected K=64 mnemonics is the smallest input that
// exercises the >1-trampoline path.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// COM: Both source K=128 mnemonics gone from the kernel body, replaced by
// COM: two distinct s_branch instructions (one per trampoline).
// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_bf8
// DISASM:       s_branch
// DISASM:       s_branch
// DISASM:       s_endpgm

// COM: Both trampolines appear after the kernel body, in source order. The
// COM: K=64 fp8_fp8 trampoline is emitted first (its source WMMA appears
// COM: first in the kernel), the K=64 bf8_bf8 trampoline second. Each
// COM: trampoline is two K=64 halves followed by an s_branch back to the
// COM: instruction after its source WMMA -- the accumulating
// COM: TrampTextOffset means the second trampoline's branch target is
// COM: computed relative to a position that already accounts for the
// COM: first trampoline's bytes. Asserting the mnemonics in trampoline
// COM: order (DISASM, not DISASM-DAG) catches a swap or a missing
// COM: trampoline that DAG would mask.
// DISASM:       v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], v[32:39]
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]
// DISASM-NEXT:  s_branch
// DISASM:       v_wmma_f32_16x16x64_bf8_bf8 v[40:47], v[0:7], v[16:23], v[40:47]
// DISASM-NEXT:  v_wmma_f32_16x16x64_bf8_bf8 v[40:47], v[8:15], v[24:31], v[40:47]
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  v_wmma_f32_16x16x128_bf8_bf8 v[40:47], v[0:15], v[16:31], v[40:47]
  s_endpgm
.size kernel, .-kernel

// Idempotency: rewriting the patched output again should produce identical
// bytes (same invariant as the omnibus test, asserted here for the
// >1-trampoline path so a regression specific to the second trampoline
// would surface).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
