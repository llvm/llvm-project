/// Test that annotations are properly carried over to fixed calls.
/// Note that --enable-bat is used to force offsets to be kept.

// RUN: llvm-mc -triple riscv64 -filetype obj -o %t.o %s
// RUN: ld.lld --emit-relocs -o %t %t.o
// RUN: llvm-bolt --enable-bat --print-cfg --print-fix-riscv-calls \
// RUN:     --print-only=_start -o /dev/null %t | FileCheck %s

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

// CHECK-LABEL: Binary Function "_start" after fix-riscv-calls {
// CHECK:      call f # Offset: 0
// CHECK-NEXT: call f # Offset: 8
// CHECK-NEXT: tail f # TAILCALL   # Offset: 12

  .globl _start
  .p2align 1
_start:
  call f
  jal f
  jal zero, f
  .size _start, .-_start
