# REQUIRES: x86
## basic-block-sections tests.
## This simple test checks if redundant direct jumps are converted to
## implicit fallthrus when PC32 reloc is present.  The jcc's must be converted
## to their inverted opcode, for instance jne to je and jmp must be deleted.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-objdump -dr %t.o| FileCheck %s --check-prefix=RELOC
# RUN: ld.lld  --optimize-bb-jumps %t.o -o %t.out
# RUN: llvm-objdump -d %t.out| FileCheck %s

# RELOC:      jmp
# RELOC-NEXT: R_X86_64_PC32

# CHECK:      <foo>:
# CHECK-NEXT:  nopl (%rax)
# CHECK-NEXT:  jne 0x{{[[:xdigit:]]+}} <r.BB.foo>
# CHECK-NOT:   jmp


.section	.text,"ax",@progbits
.type	foo,@function
foo:
 nopl (%rax)
 je	a.BB.foo
# Encode a jmp r.BB.foo insn using a PC32 reloc
 .byte  0xe9
 .long  r.BB.foo - . - 4

# CHECK:      <a.BB.foo>:
# CHECK-NEXT:  nopl (%rax)

.section	.text,"ax",@progbits,unique,3
a.BB.foo:
 nopl (%rax)
r.BB.foo:
 ret
