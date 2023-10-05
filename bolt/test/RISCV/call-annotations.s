/// Test that annotations are properly carried over to fixed calls.
/// Note that --enable-bat is used to force offsets to be kept.

// RUN: llvm-mc -triple riscv64 -filetype obj -o %t.o %s
// RUN: ld.lld --emit-relocs -o %t %t.o
// RUN: llvm-bolt --enable-bat --print-cfg --print-fix-riscv-calls \
// RUN:     -o /dev/null %t | FileCheck %s

  .text
  .global f
  .p2align 1
f:
  ret
  .size f, .-f

// CHECK-LABEL: Binary Function "_start" after building cfg {
// CHECK:      auipc ra, f
// CHECK-NEXT: jalr ra, -4(ra) # Offset: 4
// CHECK-NEXT: jal ra, f # Offset: 8
// CHECK-NEXT: jal zero, f # TAILCALL  # Offset: 12

// CHECK-LABEL: Binary Function "long_tail" after building cfg {
// CHECK:      auipc t1, f
// CHECK-NEXT: jalr zero, -24(t1) # TAILCALL  # Offset: 8

// CHECK-LABEL: Binary Function "compressed_tail" after building cfg {
// CHECK:      jr a0 # TAILCALL  # Offset: 0

// CHECK-LABEL: Binary Function "_start" after fix-riscv-calls {
// CHECK:      call f # Offset: 0
// CHECK-NEXT: call f # Offset: 8
// CHECK-NEXT: tail f # TAILCALL   # Offset: 12

// CHECK-LABEL: Binary Function "long_tail" after fix-riscv-calls {
// CHECK:      tail f # TAILCALL   # Offset: 4

// CHECK-LABEL: Binary Function "compressed_tail" after fix-riscv-calls {
// CHECK:      jr a0 # TAILCALL  # Offset: 0

  .globl _start
  .p2align 1
_start:
  call f
  jal f
  jal zero, f
  .size _start, .-_start

  .globl long_tail
  .p2align 1
long_tail:
    // NOTE: BOLT assumes indirect calls in single-BB functions are tail calls
    // so artificially introduce a second BB to force RISC-V-specific analysis
    // to get triggered.
    beq a0, a1, 1f
1:
    tail f
    .size long_tail, .-long_tail

   .globl compressed_tail
   .p2align 1
   .option rvc
compressed_tail:
    c.jr a0
    .size compressed_tail, .-compressed_tail
