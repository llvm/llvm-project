# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'SECTIONS { .text : { *(.text*) } }' > %t1.script

## Sections within the same output section can be freely folded.
# RUN: ld.lld %t.o --script %t1.script --icf=all --print-icf-sections -o %t | FileCheck --check-prefix=ICF1 %s
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SEC1 %s --implicit-check-not=.text

# ICF1:      selected section {{.*}}.o:(.text.foo0)
# ICF1-NEXT:   removing identical section {{.*}}.o:(.text.foo1)
# ICF1-NEXT:   removing identical section {{.*}}.o:(.text.bar0)
# ICF1-NEXT:   removing identical section {{.*}}.o:(.text.bar1)

# SEC1: .text   PROGBITS 0000000000000000 001000 000001

## Sections with different output sections cannot be folded. Without the
## linker script, .text.foo* and .text.bar* go to the same output section
## .text and they will be folded.
# RUN: echo 'SECTIONS { .text.foo : {*(.text.foo*)} .text.bar : {*(.text.bar*)} }' > %t2.script
# RUN: ld.lld %t.o --script %t2.script --icf=all --print-icf-sections -o %t | FileCheck --check-prefix=ICF2 %s
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SEC2 %s

# ICF2:      selected section {{.*}}.o:(.text.foo0)
# ICF2-NEXT:   removing identical section {{.*}}.o:(.text.foo1)
# ICF2-NEXT: selected section {{.*}}.o:(.text.bar0)
# ICF2-NEXT:   removing identical section {{.*}}.o:(.text.bar1)

# SEC2:      .text.foo   PROGBITS 0000000000000000 001000 000001
# SEC2-NEXT: .text.bar   PROGBITS 0000000000000001 001001 000001

## .text.bar* are orphan sections.
# RUN: echo 'SECTIONS { .text.foo : {*(.text.foo*)} }' > %t3.script
# RUN: ld.lld %t.o -T %t3.script --icf=all --print-icf-sections -o %t3 | FileCheck --check-prefix=ICF3 %s
# RUN: llvm-readelf -S %t3 | FileCheck --check-prefix=SEC3 %s

# ICF3:      selected section {{.*}}.o:(.text.foo0)
# ICF3-NEXT:   removing identical section {{.*}}.o:(.text.foo1)

# SEC3:      Name        Type     Address          Off    Size
# SEC3:      .text.foo   PROGBITS 0000000000000000 001000 000001
# SEC3-NEXT: .text       PROGBITS 0000000000000004 001004 000000
# SEC3-NEXT: .text.bar0  PROGBITS 0000000000000004 001004 000001
# SEC3-NEXT: .text.bar1  PROGBITS 0000000000000005 001005 000001

.section .text.foo0,"ax"
ret
.section .text.foo1,"ax"
ret
.section .text.bar0,"ax"
ret
.section .text.bar1,"ax"
ret
