# REQUIRES: x86
## Test that SHT_GROUP sections are retained in relocatable output. The content
## may be rewritten because group members may change their indices. Additionally,
## group member may be combined or discarded (e.g. /DISCARD/ or --gc-sections).

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -r %t.o %t.o -o %t.ro
# RUN: llvm-readelf -g -S %t.ro | FileCheck %s

# CHECK: Name    Type  Address          Off    Size   ES Flg Lk        Inf Al
# CHECK: .group  GROUP 0000000000000000 {{.*}} 000014 04     {{[1-9]}}   1  4

# CHECK: COMDAT group section [{{.*}}] `.group' [abc] contains 4 sections:
# CHECK-NEXT: Name
# CHECK-NEXT: .rodata.bar
# CHECK-NEXT: .rodata.foo
# CHECK-NEXT: .text.bar
# CHECK-NEXT: .text.foo

## Rewrite SHT_GROUP content if some members are combined.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.*)} .text : {*(.text.*)} }' > %t1.lds
# RUN: ld.lld -r -T %t1.lds %t.o %t.o -o %t1.ro
# RUN: llvm-readelf -g -S %t1.ro | FileCheck %s --check-prefix=SCRIPT1

# SCRIPT1: Name    Type  Address          Off    Size   ES Flg Lk        Inf Al
# SCRIPT1: .group  GROUP 0000000000000000 {{.*}} 00000c 04     {{[1-9]}}   1  4

# SCRIPT1: COMDAT group section [{{.*}}] `.group' [abc] contains 2 sections:
# SCRIPT1-NEXT: Name
# SCRIPT1-NEXT: .rodata
# SCRIPT1-NEXT: .text

# RUN: echo 'SECTIONS { /DISCARD/ : {*(.rodata.*)} }' > %t2.lds
# RUN: ld.lld -r -T %t2.lds %t.o %t.o -o %t2.ro
# RUN: llvm-readelf -g -S %t2.ro | FileCheck %s --check-prefix=SCRIPT2

## Handle discarded group members.
# SCRIPT2: [Nr] Name    Type  Address          Off    Size   ES Flg Lk        Inf Al
# SCRIPT2: [ 2] .group  GROUP 0000000000000000 {{.*}} 00000c 04     {{[1-9]}}   1  4

# SCRIPT2: COMDAT group section [{{.*}}] `.group' [abc] contains 2 sections:
# SCRIPT2-NEXT: Name
# SCRIPT2-NEXT: .text.bar
# SCRIPT2-NEXT: .text.foo

.section .rodata.bar,"aG",@progbits,abc,comdat
.byte 42
.section .rodata.foo,"aG",@progbits,abc,comdat
.byte 42

.section .text.bar,"axG",@progbits,abc,comdat
.quad 42
.section .text.foo,"axG",@progbits,abc,comdat
.long 42
