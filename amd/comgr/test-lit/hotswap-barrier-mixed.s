// COM: Selectivity test for the s_barrier_signal_isfirst -> s_barrier_signal
// COM: in-place patch. A kernel containing isfirst (IMM), non-isfirst, and
// COM: the _M0 form must have only the IMM isfirst sites rewritten; both
// COM: non-isfirst and _M0 sites must pass through unchanged. This guards
// COM: against a regression where the dispatcher accidentally matches the
// COM: non-isfirst mnemonic (e.g. via prefix or contains() rather than
// COM: equality) and documents the intentional _M0 passthrough.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Verify the patched layout. Kernel order is:
// COM:   s_barrier_signal_isfirst -1     (IMM, gets swapped)
// COM:   s_barrier_wait -1               (unchanged)
// COM:   s_barrier_signal -3             (already non-isfirst, unchanged)
// COM:   s_barrier_wait -3               (unchanged)
// COM:   s_barrier_signal_isfirst m0     (_M0 form, intentionally NOT swapped)
// COM:   s_barrier_wait 0xffff           (unchanged)
// COM:   s_barrier_signal_isfirst -1     (IMM, gets swapped)
// COM:   s_barrier_wait -1               (unchanged)
// COM:   s_endpgm
// COM: After patching, the two IMM isfirst sites become s_barrier_signal -1.
// COM: The _M0 site passes through unchanged because the compiler never
// COM: emits it (separate mnemonic, intentional skip with diagnostic).
// COM: CHECK-NEXT chain pins the exact interleaving. The leading CHECK-NOT
// COM: only covers the range before the first match; the _M0 site in the
// COM: middle is verified structurally by the CHECK-NEXT chain itself.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-NOT: s_barrier_signal_isfirst -
// DISASM: s_barrier_signal -1
// DISASM-NEXT: s_barrier_wait 0xffff
// DISASM-NEXT: s_barrier_signal -3
// DISASM-NEXT: s_barrier_wait 0xfffd
// DISASM-NEXT: s_barrier_signal_isfirst m0
// DISASM-NEXT: s_barrier_wait 0xffff
// DISASM-NEXT: s_barrier_signal -1
// DISASM-NEXT: s_barrier_wait 0xffff
// DISASM-NEXT: s_endpgm
// DISASM-NOT: s_barrier_signal_isfirst -

// COM: Idempotency: second rewrite must produce identical bytes (the swapped
// COM: kernel has no isfirst left, so the second pass is a no-op).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_barrier_mixed
.p2align 8
.type test_barrier_mixed,@function
test_barrier_mixed:
  s_barrier_signal_isfirst -1
  s_barrier_wait -1
  // Pre-existing non-isfirst site: verifies the dispatcher matches on
  // equality, not prefix or substring.
  s_barrier_signal -3
  s_barrier_wait -3
  // _M0 form: separate mnemonic, intentionally not patched.
  s_barrier_signal_isfirst m0
  s_barrier_wait -1
  s_barrier_signal_isfirst -1
  s_barrier_wait -1
  s_endpgm
.Ltest_barrier_mixed_end:
.size test_barrier_mixed, .Ltest_barrier_mixed_end-test_barrier_mixed

.rodata
.p2align 8
.amdhsa_kernel test_barrier_mixed
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
