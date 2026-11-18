# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/input.s -o %t/input.o
# RUN: llvm-objcopy --redefine-sym first=s --redefine-sym second=s %t/input.o %t/duplicate.o
# RUN: ld.lld --btf-merge --gc-sections %t/duplicate.o -o %t/out 2>&1 | FileCheck %s
# CHECK-COUNT-2: warning: {{.*}}:(.BTF): --btf-merge: cannot resolve DATASEC .bss.vars variable 's'; entry dropped
# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym=KEEP=1 %t/blobs.s -o %t/live.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/blobs.s -o %t/dead.o
# RUN: ld.lld -r --gc-sections -e _start %t/live.o %t/dead.o -o %t/partial.o
# RUN: ld.lld --btf-merge %t/partial.o -o %t/partial-out 2>&1 | FileCheck %s --check-prefix=BLOBS
# BLOBS-COUNT-2: warning: {{.*}}:(.BTF): --btf-merge: cannot resolve DATASEC .bss.vars variable 's'; entry dropped
# RUN: llvm-readelf -x .BTF %t/partial-out | FileCheck %s --check-prefix=NO-DATASEC
# NO-DATASEC: 0x00000000 9feb0100 18000000 00000000 40000000

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

#--- blobs.s
.text
.ifdef KEEP
.globl _start
_start: lea s(%rip), %rax; ret
.endif
.section .bss.vars,"aw",@nobits
.type s,@object
.size s,4
s: .zero 4
.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr-.Ltypes, .Lstr-.Ltypes, .Lend-.Lstr
.Ltypes:
.ifdef KEEP
.long 1, 0x01000000, 4, 32
.else
.long 1, 0x01000000, 4, 0x01000020
.endif
.long 5, 0x0e000000, 1, 0
.long 7, 0x0f000001, 0, 2, 0, 4
.Lstr:
.asciz "", "int", "s", ".bss.vars"
.Lend:
