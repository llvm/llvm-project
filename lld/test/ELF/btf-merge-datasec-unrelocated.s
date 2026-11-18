# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/input.s -o %t.o
# RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=NOREL
# RUN: ld.lld --btf-merge --entry=0 %t.o -o %t/out 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: ld.lld --btf-merge --entry=0 --wrap=g %t.o -o %t/wrap 2>&1 | FileCheck %s --check-prefix=WRAP
# RUN: llvm-readelf -x .BTF %t/out | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/real.s -o %t/real.o
# RUN: ld.lld --btf-merge --entry=0 --wrap=t %t/real.o -o %t/real 2>&1 | FileCheck %s --check-prefix=REAL

## Model GCC's zero offsets with no .BTF relocations. Recover both a global
## and a local offset from .bss, but drop the entry whose symbol is in .data.
# NOREL: There are no relocations in this file.
# WARN: warning: {{.*}}:(.BTF): --btf-merge: cannot resolve DATASEC .bss variable 'other'; entry dropped
# WRAP: warning: {{.*}}:(.BTF): --btf-merge: cannot resolve DATASEC .bss variable 'g'; entry dropped
## --wrap=t redirects __real_t to t; refuse it rather than use t's offset.
# REAL-DAG: warning: {{.*}}:(.BTF): --btf-merge: cannot resolve DATASEC .bss variable '__real_t'; entry dropped
# REAL-DAG: warning: {{.*}}:(.BTF): --btf-merge: cannot resolve DATASEC .bss variable 't'; entry dropped
# CHECK:      0x00000050 01000000 01000000 0f000000 0200000f
# CHECK-NEXT: 0x00000060 0c000000 02000000 04000000 04000000
# CHECK-NEXT: 0x00000070 03000000 08000000 04000000

#--- input.s
.bss
  .zero 4
.globl g
.type g,@object
g: .zero 4
.type s,@object
s: .zero 4
.data
.globl other
.type other,@object
other: .long 1

.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lstr - .Ltypes, .Lstr - .Ltypes, .Lend - .Lstr
.Ltypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 1
.long 7, 0x0e000000, 1, 0
.long 9, 0x0e000000, 1, 1
.long 15, 0x0f000003, 0, 2, 0, 4, 3, 0, 4, 4, 0, 4
.Lstr:
.asciz "", "int", "g", "s", "other", ".bss"
.Lend:

#--- real.s
.bss
.globl __real_t
.type __real_t,@object
__real_t: .zero 4
.globl t
.type t,@object
t: .zero 4

.section .BTF,"",@progbits
.short 0xeb9f
.byte 1, 0
.long 24, 0, .Lrstr - .Lrtypes, .Lrstr - .Lrtypes, .Lrend - .Lrstr
.Lrtypes:
.long 1, 0x01000000, 4, 32
.long 5, 0x0e000000, 1, 1
.long 14, 0x0e000000, 1, 1
.long 16, 0x0f000002, 0, 2, 0, 4, 3, 0, 4
.Lrstr:
.asciz "", "int", "__real_t", "t", ".bss"
.Lrend:
