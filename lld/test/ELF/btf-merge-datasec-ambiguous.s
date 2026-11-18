# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/input.s -o %t/input.o
# RUN: llvm-objcopy --redefine-sym first=s --redefine-sym second=s %t/input.o %t/duplicate.o
# RUN: ld.lld --btf-merge --gc-sections %t/duplicate.o -o %t/out 2>&1 | FileCheck %s
# CHECK-COUNT-2: warning: {{.*}}:(.BTF): --btf-merge: cannot resolve DATASEC .bss.vars variable 's'; entry dropped

#--- input.s
.globl _start
.text
_start:
  lea first(%rip), %rax
  ret
.pushsection .bss.vars,"aw",@nobits,unique,1
  .zero 8
.type first,@object
first: .zero 4
.popsection
.pushsection .bss.vars,"aw",@nobits,unique,2
  .zero 16
.type second,@object
second: .zero 4
.popsection
.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr - .Ltypes, .Lstr - .Ltypes, .Lend - .Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 0
.long 5, 0x0e000000, 1, 0
.long 7, 0x0f000002, 0, 2, 0, 4, 3, 0, 4
.Lstr:
.asciz "", "int", "s", ".bss.vars"
.Lend:
