// COM: Passthrough test for the s_barrier_signal_isfirst -> s_barrier_signal
// COM: in-place patch. A kernel that already uses the non-isfirst form must
// COM: be left structurally unchanged: no isfirst should appear anywhere in
// COM: the patched output, and the original s_barrier_signal sites must
// COM: remain in place with their original operands.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Strict no-op verification: original layout preserved; isfirst variant
// COM: never appears. CHECK-NOT covers both pre- and post-kernel ranges.
// COM: Wait operands are shown by llvm-objdump as raw 16-bit hex (signed
// COM: -1 = 0xffff, -3 = 0xfffd).
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-NOT: s_barrier_signal_isfirst
// DISASM: s_barrier_signal -1
// DISASM-NEXT: s_barrier_wait 0xffff
// DISASM-NEXT: s_barrier_signal -3
// DISASM-NEXT: s_barrier_wait 0xfffd
// DISASM-NEXT: s_endpgm
// DISASM-NOT: s_barrier_signal_isfirst

// COM: Idempotency: second rewrite must produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_barrier_no_isfirst
.p2align 8
.type test_barrier_no_isfirst,@function
test_barrier_no_isfirst:
  // Workgroup barrier (-1) and a user cluster barrier (-3); neither uses
  // the isfirst form, so the patch must leave both unchanged.
  s_barrier_signal -1
  s_barrier_wait -1
  s_barrier_signal -3
  s_barrier_wait -3
  s_endpgm
.Ltest_barrier_no_isfirst_end:
.size test_barrier_no_isfirst, .Ltest_barrier_no_isfirst_end-test_barrier_no_isfirst

.rodata
.p2align 8
.amdhsa_kernel test_barrier_no_isfirst
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
