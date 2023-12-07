// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt --reorder-blocks=reverse -o %t.bolt %t
// RUN: llvm-objdump -d --no-show-raw-insn %t.bolt | FileCheck %s

  .text
  .globl _start
  .p2align 1
_start:
  nop
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
// CHECK-NEXT:   {{.*}}00:       beq t0, t1, {{.*}} <_start+0x10>
// CHECK-NEXT:   {{.*}}04:       j {{.*}} <_start+0x16>
// CHECK-NEXT:   {{.*}}08:       ret
// CHECK-NEXT:   {{.*}}0a:       li a0, 6
// CHECK-NEXT:   {{.*}}0c:       j {{.*}} <_start+0x8>
// CHECK-NEXT:   {{.*}}10:       li a0, 5
// CHECK-NEXT:   {{.*}}12:       j {{.*}} <_start+0x8>
// CHECK-NEXT:   {{.*}}16:       beq t0, t2, {{.*}} <_start+0xa>
// CHECK-NEXT:   {{.*}}1a:       j {{.*}} <_start+0x10>
