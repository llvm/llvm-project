# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## SHF_MERGE sections within the same output section can be freely merged.
# RUN: echo 'SECTIONS { .rodata : { *(.rodata.*) }}' > %t.script
# RUN: ld.lld %t.o -T %t.script -o %t
# RUN: llvm-readelf -x .rodata %t | FileCheck --check-prefix=SAME %s --implicit-check-not=section

# SAME:      section '.rodata':
# SAME-NEXT: 0x00000000 01000200 0300

## SHF_MERGE sections with different output sections cannot be merged.
# RUN: echo 'SECTIONS { \
# RUN:   .rodata.foo : { *(.rodata.foo) } \
# RUN:   .rodata.bar : { *(.rodata.bar) } \
# RUN: }' > %t2.script
# RUN: ld.lld %t.o -T %t2.script -o %t2
# RUN: llvm-readelf -x .rodata.foo -x .rodata.bar %t2 | FileCheck --check-prefix=DIFF %s --implicit-check-not=section

# DIFF:      section '.rodata.foo':
# DIFF-NEXT: 0x00000000 01000200 0300
# DIFF:      section '.rodata.bar':
# DIFF-NEXT: 0x00000006 0100

.section .rodata.foo,"aM",@progbits,2,unique,0
.short 1
.short 2
.section .rodata.foo,"aM",@progbits,2,unique,1
.short 1
.short 3

.section .rodata.bar,"aM",@progbits,2,unique,0
.short 1
.section .rodata.bar,"aM",@progbits,2,unique,1
.short 1
