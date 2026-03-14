# REQUIRES: x86
## basic-block-sections tests.
## This simple test checks foo is folded into bar with bb sections
## and the jumps are deleted.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --optimize-bb-jumps --icf=all %t.o -o %t.out
# RUN: llvm-objdump -d %t.out| FileCheck %s

# CHECK:      <foo>:
# CHECK-NEXT:  nopl (%rax)
# CHECK-NEXT:  je 0x{{[[:xdigit:]]+}} <aa.BB.foo>
# CHECK-NOT:   jmp

# CHECK:     <a.BB.foo>:
## Explicity check that bar is folded and not emitted.
# CHECK-NOT: <bar>:
# CHECK-NOT: <a.BB.bar>:
# CHECK-NOT: <aa.BB.bar>:

.section	.text.bar,"ax",@progbits
.type	bar,@function
bar:
 nopl (%rax)
 jne	a.BB.bar
 jmp	aa.BB.bar

.section	.text.a.BB.bar,"ax",@progbits,unique,3
a.BB.bar:
 nopl (%rax)

aa.BB.bar:
 ret

.section	.text.foo,"ax",@progbits
.type	foo,@function
foo:
 nopl (%rax)
 jne	a.BB.foo
 jmp	aa.BB.foo

.section	.text.a.BB.foo,"ax",@progbits,unique,2
a.BB.foo:
 nopl (%rax)

aa.BB.foo:
 ret
