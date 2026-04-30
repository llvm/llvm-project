// COM: Test HotSwap in-place patch s_barrier_signal_isfirst -> s_barrier_signal
// COM: for GFX1250 A0. A0 may return stale SCC before the barrier completes
// COM: when the barrier ID names a user cluster barrier; the non-isfirst
// COM: variant shares encoding size and operand layout but does not write SCC.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Verify the patched layout: both isfirst sites are replaced by the
// COM: non-isfirst variant in place with their operand values preserved
// COM: (-1 stays -1, -3 stays -3). Surrounding waits and endpgm stay put.
// COM: DISASM-NOT brackets ensure no s_barrier_signal_isfirst remains
// COM: anywhere. Wait operands are shown as raw 16-bit hex by llvm-objdump
// COM: (signed -1 = 0xffff, -3 = 0xfffd).
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-NOT: s_barrier_signal_isfirst
// DISASM: s_barrier_signal -1
// DISASM-NEXT: s_barrier_wait 0xffff
// DISASM-NEXT: s_barrier_signal -3
// DISASM-NEXT: s_barrier_wait 0xfffd
// DISASM-NEXT: s_endpgm
// DISASM-NOT: s_barrier_signal_isfirst

// COM: Idempotency: rewriting the patched output again must produce
// COM: identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_barrier_isfirst
.p2align 8
.type test_barrier_isfirst,@function
test_barrier_isfirst:
  // Two isfirst sites with different barrier IDs; both must be swapped
  // to s_barrier_signal in place with their operand values preserved.
  s_barrier_signal_isfirst -1
  s_barrier_wait -1
  s_barrier_signal_isfirst -3
  s_barrier_wait -3
  s_endpgm
.Ltest_barrier_isfirst_end:
.size test_barrier_isfirst, .Ltest_barrier_isfirst_end-test_barrier_isfirst

.rodata
.p2align 8
.amdhsa_kernel test_barrier_isfirst
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
