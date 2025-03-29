# REQUIRES: x86
## Test that SHT_GROUP sections are retained in relocatable output. The content
## may be rewritten because group members may change their indices. Additionally,
## group member may be combined or discarded (e.g. /DISCARD/ or --gc-sections).

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: ld.lld -r a.o a.o -o a.ro
# RUN: llvm-readelf -g -S a.ro | FileCheck %s

# CHECK:      Name           Type     Address          Off    Size   ES Flg Lk    Inf   Al
# CHECK:      .group         GROUP    0000000000000000 {{.*}} 00001c 04     [[#]] [[#]]  4
# CHECK-NEXT: .rodata.bar    PROGBITS 0000000000000000 {{.*}} 000001 00  AG  0      0    1
# CHECK-NEXT: .rodata.foo    PROGBITS 0000000000000000 {{.*}} 000001 00  AG  0      0    1
# CHECK-NEXT: .text.bar      PROGBITS 0000000000000000 {{.*}} 000008 00 AXG  0      0    1
# CHECK-NEXT: .rela.text.bar RELA     0000000000000000 {{.*}} 000018 18  IG [[#]] [[#]]  8
# CHECK-NEXT: .text.foo      PROGBITS 0000000000000000 {{.*}} 000008 00 AXG [[#]] [[#]]  1
# CHECK-NEXT: .rela.text.foo RELA     0000000000000000 {{.*}} 000018 18  IG [[#]] [[#]]  8
# CHECK-NEXT: .note.GNU-stack

# CHECK: COMDAT group section [{{.*}}] `.group' [abc] contains 6 sections:
# CHECK-NEXT: Name
# CHECK-NEXT: .rodata.bar
# CHECK-NEXT: .rodata.foo
# CHECK-NEXT: .text.bar
# CHECK-NEXT: .rela.text.bar
# CHECK-NEXT: .text.foo
# CHECK-NEXT: .rela.text.foo

## Rewrite SHT_GROUP content if some members are combined.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.*)} .text : {*(.text.*)} }' > combine.lds
# RUN: ld.lld -r -T combine.lds a.o a.o -o combine.ro
# RUN: llvm-readelf -g -S combine.ro | FileCheck %s --check-prefix=COMBINE

# COMBINE:      Name           Type     Address          Off    Size   ES Flg Lk    Inf   Al
# COMBINE:      .rodata        PROGBITS 0000000000000000 {{.*}} 000002 00  AG  0      0    1
# COMBINE-NEXT: .text          PROGBITS 0000000000000000 {{.*}} 000010 00 AXG  0      0    4
# COMBINE-NEXT: .group         GROUP    0000000000000000 {{.*}} 000014 04     [[#]] [[#]]  4
# COMBINE-NEXT: .rela.text     RELA     0000000000000000 {{.*}} 000018 18  IG [[#]] [[#]]  8
# COMBINE-NEXT: .rela.text     RELA     0000000000000000 {{.*}} 000018 18  IG [[#]] [[#]]  8
# COMBINE-NEXT: .note.GNU-stack

# COMBINE: COMDAT group section [{{.*}}] `.group' [abc] contains 4 sections:
# COMBINE-NEXT: Name
# COMBINE-NEXT: .rodata
# COMBINE-NEXT: .text
# COMBINE-NEXT: .rela.text
# COMBINE-NEXT: .rela.text

## If --force-group-allocation is specified, discard .group and combine .rela.* if their relocated sections are combined.
# RUN: ld.lld -r -T combine.lds a.o a.o --force-group-allocation -o combine-a.ro
# RUN: llvm-readelf -g -S combine-a.ro | FileCheck %s --check-prefix=COMBINE-A

# COMBINE-A:      Name            Type     Address          Off    Size   ES Flg Lk    Inf   Al
# COMBINE-A:      .rodata         PROGBITS 0000000000000000 {{.*}} 000002 00   A  0      0    1
# COMBINE-A-NEXT: .text           PROGBITS 0000000000000000 {{.*}} 000010 00  AX  0      0    4
# COMBINE-A-NEXT: .rela.text      RELA     0000000000000000 {{.*}} 000030 18   I [[#]] [[#]]  8
# COMBINE-A-NEXT: .note.GNU-stack

# RUN: echo 'SECTIONS { /DISCARD/ : {*(.rodata.*)} }' > discard-rodata.lds
# RUN: ld.lld -r -T discard-rodata.lds a.o a.o -o discard-rodata.ro
# RUN: llvm-readelf -g -S discard-rodata.ro | FileCheck %s --check-prefix=NO-RODATA

## Handle discarded group members.
# NO-RODATA:      Name           Type     Address          Off    Size   ES Flg Lk    Inf   Al
# NO-RODATA:      .group         GROUP    0000000000000000 {{.*}} 000014 04     [[#]] [[#]]  4
# NO-RODATA-NEXT: .text.bar      PROGBITS 0000000000000000 {{.*}} 000008 00 AXG  0      0    1
# NO-RODATA-NEXT: .rela.text.bar RELA     0000000000000000 {{.*}} 000018 18  IG [[#]] [[#]]  8
# NO-RODATA-NEXT: .text.foo      PROGBITS 0000000000000000 {{.*}} 000008 00 AXG [[#]] [[#]]  1
# NO-RODATA-NEXT: .rela.text.foo RELA     0000000000000000 {{.*}} 000018 18  IG [[#]] [[#]]  8
# NO-RODATA-NEXT: .note.GNU-stack

# NO-RODATA:      COMDAT group section [{{.*}}] `.group' [abc] contains 4 sections:
# NO-RODATA-NEXT: Name
# NO-RODATA-NEXT: .text.bar
# NO-RODATA-NEXT: .rela.text.bar
# NO-RODATA-NEXT: .text.foo
# NO-RODATA-NEXT: .rela.text.foo

#--- a.s
.weak abc
abc:

.section .rodata.bar,"aG",@progbits,abc,comdat
.byte 42
.section .rodata.foo,"aG",@progbits,abc,comdat
.byte 42

.section .text.bar,"axG",@progbits,abc,comdat
.quad abc
.section .text.foo,"axG",@progbits,abc,comdat
.quad abc
