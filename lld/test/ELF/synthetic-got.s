# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { }" > %t0.script
# RUN: ld.lld --hash-style=sysv -shared %t.o -o %t0.out --script %t0.script
# RUN: llvm-objdump -section-headers %t0.out | FileCheck %s --check-prefix=GOT
# RUN: llvm-objdump -s -section=.got -section=.got.plt %t0.out \
# RUN:   | FileCheck %s --check-prefix=GOTDATA

# GOT:     Sections:
# GOT:      8  .got.plt     00000020 00000000000000d0 DATA
# GOT:      10 .got         00000008 00000000000001c0 DATA
# GOTDATA:     Contents of section .got.plt:
# GOTDATA-NEXT:  00d0 f0000000 00000000 00000000 00000000
# GOTDATA-NEXT:  00e0 00000000 00000000 c6000000 00000000
# GOTDATA-NEXT: Contents of section .got:
# GOTDATA-NEXT:  01c0 00000000 00000000

# RUN: echo "SECTIONS { .mygot : { *(.got) *(.got.plt) } }" > %t1.script
# RUN: ld.lld --hash-style=sysv -shared %t.o -o %t1.out --script %t1.script
# RUN: llvm-objdump -section-headers %t1.out | FileCheck %s --check-prefix=MYGOT
# RUN: llvm-objdump -s -section=.mygot %t1.out | FileCheck %s --check-prefix=MYGOTDATA

# MYGOT:     Sections:
# MYGOT:      8  .mygot     00000028 00000000000000d0 DATA
# MYGOT-NOT:  .got
# MYGOT-NOT:  .got.plt
# MYGOTDATA:      00d0 00000000 00000000 f8000000 00000000
# MYGOTDATA-NEXT: 00e0 00000000 00000000 00000000 00000000
# MYGOTDATA-NEXT: 00f0 c6000000 00000000

mov bar@gotpcrel(%rip), %rax
call foo@plt
