// RUN: %clang %cflags64 -o %t %s
// RUN: llvm-bolt --reorder-blocks=reverse -o %t.bolt %t
// RUN: llvm-objdump -d --no-show-raw-insn %t.bolt | FileCheck %s
// RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj -o %t.rv32.o %s
// RUN: ld.lld -q -o %t.rv32 %t.rv32.o
// RUN: llvm-bolt --reorder-blocks=reverse -o %t.rv32.bolt %t.rv32
// RUN: llvm-objdump -d --no-show-raw-insn %t.rv32.bolt | FileCheck %s

  .text
  .globl _start
  .p2align 1
_start:
  nop
  .reloc ., R_RISCV_BRANCH, 1f
  beq t0, t1, 1f
  nop
  beq t0, t2, 2f
1:
  li a0, 5
  j 3f
2:
  li a0, 6
3:
  ret
  .size _start,.-_start

// CHECK: {{.*}}00 <_start>:
// CHECK-NEXT:   {{.*}}00:       beq t0, t1, {{.*}} <[[L0:.Ltmp[0-9]+]]>
// CHECK-NEXT:   {{.*}}04:       j {{.*}} <[[L0]]+0x6>
// CHECK:        {{.*}}08:       ret
// CHECK:        {{.*}}0a:       li a0, 0x6
// CHECK-NEXT:   {{.*}}0c:       j {{.*}}08 <{{(\.Ltmp[0-9]+|_start\+0x8)}}>
// CHECK: {{.*}}10 <[[L0]]>:
// CHECK-NEXT:   {{.*}}10:       li a0, 0x5
// CHECK-NEXT:   {{.*}}12:       j {{.*}}08 <{{(\.Ltmp[0-9]+|_start\+0x8)}}>
// CHECK-NEXT:   {{.*}}16:       beq t0, t2, {{.*}}0a <{{(\.Ltmp[0-9]+|_start\+0xa)}}>
// CHECK-NEXT:   {{.*}}1a:       j {{.*}} <[[L0]]>
