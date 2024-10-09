## Check that BOLT properly identifies a jump to builtin_unreachable

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld -q -o %t %t.o
# RUN: llvm-bolt %t -o %t.null -lite=0 -print-disasm | FileCheck %s
# CHECK:      callq bar
# CHECK-NEXT: nop

.text
.globl main
.type main, @function
main:
  call foo
  .size main, .-main

.section .mytext.bar, "ax"
.globl  bar
.type	bar, @function
bar:
  ud2
	.size	bar, .-bar

.section .mytext.foo, "ax"
.globl	foo
.type	foo, @function
foo:
.cfi_startproc
  callq bar
  jmp .Lunreachable
  ret
  .cfi_endproc
	.size	foo, .-foo
.Lunreachable:
