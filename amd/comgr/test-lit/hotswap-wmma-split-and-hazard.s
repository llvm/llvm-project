// Test WMMA-split + WMMA-hazard interaction: a single kernel that
// triggers both passes. The splittable K=128 fp8/bf8 WMMA is a float WMMA
// (so it never matches the hazard classifier, which only fires for WMMA
// integer opcodes). The K=64 iu8 WMMA + overlapping VALU triggers the
// hazard pass (8 v_nops on A0). Both passes append to Ctx.OutTrampolines
// and both compute their own trampoline's `.text` offset by walking the
// previously appended trampolines (the TrampTextOffset accumulation
// pattern -- patch-wmma-split.cpp:664-666 and patch-wmma-hazard.cpp:184-186).
// If either pass forgets to account for the other's trampolines, the
// s_branch-back at the tail of one trampoline lands at the wrong target
// and the kernel falls off the end of .text or jumps into the other
// trampoline's body. This test puts both in one kernel so a regression in
// either accumulation surfaces here.
//
// Operand-shape note: same disjoint-VGPR contract as the K-split tests
// (src0=v[0:15], src1=v[16:31], dst=v[32:39] for the splittable WMMA).
// The hazardous WMMA uses the standard hazard test's operand shape
// (dst v[16:23], src0 v[0:7], src1 v[8:15]) -- different VGPR set so the
// two patches are operating on independent kernel state.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// COM: Kernel body: the splittable K=128 WMMA is replaced by an s_branch;
// COM: the K=64 iu8 WMMA stays in place (it's the hazard *source*, not a
// COM: split target); the overlapping v_add_f32 is replaced by an
// COM: s_branch into the hazard trampoline. Two distinct s_branch sites
// COM: in the body, then s_endpgm.
// DISASM-LABEL: <test_split_and_hazard>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
// DISASM:       v_wmma_i32_16x16x64_iu8
// DISASM-NEXT:  s_branch
// DISASM:       s_endpgm
.globl test_split_and_hazard
.p2align 8
.type test_split_and_hazard,@function
test_split_and_hazard:
  // Splittable: K=128 fp8/bf8 -> two K=64 halves in trampoline #1.
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  // Hazard source: WMMA integer needs 8 v_nops on A0.
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  // Hazard target: VALU writes v16, overlapping the WMMA dest.
  v_add_f32 v16, v0, v1
  s_endpgm
.Ltest_split_and_hazard_end:
.size test_split_and_hazard, .Ltest_split_and_hazard_end-test_split_and_hazard

// COM: Trampolines after the kernel body. Order is determined by the
// COM: top-level patch loop in comgr-hotswap-b0a0.cpp:323-361 -- the
// COM: per-instruction passes (which include the splitter) run before
// COM: applyWmmaHazardPatch, so the split trampoline lands first and the
// COM: hazard trampoline second. Asserting in order (DISASM, not
// COM: DISASM-DAG) catches a swap or a missing trampoline that DAG would
// COM: mask -- and, more importantly, asserting the hazard trampoline's
// COM: 8 v_nops + relocated v_add_f32 land *after* the split trampoline's
// COM: bytes is exactly the property that breaks if the hazard pass
// COM: forgets to walk Ctx.OutTrampolines when computing its
// COM: TrampolineTextOffset.

// COM: Split trampoline: two K=64 halves (first half src2 = original
// COM: dst, second half src2 = dst-as-carry), then s_branch back to the
// COM: instruction after the original K=128 WMMA.
// DISASM:       v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], v[32:39]
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]
// DISASM-NEXT:  s_branch

// COM: Hazard trampoline: exactly 8 v_nops (full deficit -- no
// COM: pre-existing nops between WMMA and VALU) followed by the
// COM: relocated v_add_f32.
// DISASM-COUNT-8: v_nop
// DISASM-NEXT:  v_add_f32

.rodata
.p2align 8
.amdhsa_kernel test_split_and_hazard
  .amdhsa_next_free_vgpr 48
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

// Idempotency: rewriting the patched output again should produce identical
// bytes. The splitter only fires on K=128 mnemonics (none left) and the
// hazard pass only fires on WMMA integer + overlapping VALU within the
// hazard window (the v_add_f32 has been relocated past the deficit nops,
// so it's no longer within the window).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
