# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: llvm-objcopy %t1.o %t1copy.o
# RUN: llvm-objcopy --localize-symbol=h1 %t1.o %t1changed.o
# RUN: ld.lld -r %t1.o -o %t1reloc.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %S/Inputs/icf-safe.s -o %t2.o
# RUN: ld.lld %t1.o %t2.o -o %t2 --icf=safe --print-icf-sections | FileCheck %s
# RUN: ld.lld %t1copy.o %t2.o -o %t2 --icf=safe --print-icf-sections | FileCheck %s
# RUN: ld.lld %t1.o %t2.o -o %t3 --icf=safe --print-icf-sections -shared | FileCheck --check-prefix=EXPORT %s
# RUN: ld.lld %t1.o %t2.o -o %t3 --icf=safe --print-icf-sections --export-dynamic | FileCheck --check-prefix=EXPORT %s
# RUN: ld.lld %t1.o %t2.o -o %t2 --icf=all --print-icf-sections | FileCheck --check-prefix=ALL %s
# RUN: ld.lld %t1.o %t2.o -o %t2 --icf=all --print-icf-sections --export-dynamic | FileCheck --check-prefix=ALL-EXPORT %s
# RUN: ld.lld %t1changed.o -o %t4 --icf=safe 2>&1 | FileCheck --check-prefix=SH_LINK_0 %s
# RUN: ld.lld %t1reloc.o -o %t4 --icf=safe 2>&1 | FileCheck --check-prefix=SH_LINK_0 %s

# CHECK-NOT:  {{.}}
# CHECK:      selected section {{.*}}:(.rodata.h3)
# CHECK-NEXT:   removing identical section {{.*}}:(.rodata.h4)
# CHECK-NEXT: selected section {{.*}}:(.text.f3)
# CHECK-NEXT:   removing identical section {{.*}}:(.text.f4)
# CHECK-NEXT: selected section {{.*}}:(.rodata.g3)
# CHECK-NEXT:   removing identical section {{.*}}:(.rodata.g4)
# CHECK-NEXT: selected section {{.*}}:(.rodata.l3)
# CHECK-NEXT:   removing identical section {{.*}}:(.rodata.l4)
# CHECK-NEXT: selected section {{.*}}:(.text)
# CHECK-NEXT:   removing identical section {{.*}}:(.text)
# CHECK-NOT:  {{.}}

# With --icf=all address-significance implies keep-unique only for rodata, not
# text.
# ALL-NOT:  {{.}}
# ALL:      selected section {{.*}}:(.rodata.h3)
# ALL-NEXT:   removing identical section {{.*}}:(.rodata.h4)
# ALL-NEXT: selected section {{.*}}:(.text.f3)
# ALL-NEXT:   removing identical section {{.*}}:(.text.f4)
# ALL-NEXT: selected section {{.*}}:(.text.f1)
# ALL-NEXT:   removing identical section {{.*}}:(.text.f2)
# ALL-NEXT:   removing identical section {{.*}}:(.text.non_addrsig1)
# ALL-NEXT:   removing identical section {{.*}}:(.text.non_addrsig2)
# ALL-NEXT: selected section {{.*}}:(.rodata.g3)
# ALL-NEXT:   removing identical section {{.*}}:(.rodata.g4)
# ALL-NEXT: selected section {{.*}}:(.rodata.l3)
# ALL-NEXT:   removing identical section {{.*}}:(.rodata.l4)
# ALL-NEXT: selected section {{.*}}:(.text)
# ALL-NEXT:   removing identical section {{.*}}:(.text)
# ALL-NOT:  {{.}}

# llvm-mc normally emits an empty .text section into every object file. Since
# nothing actually refers to it via a relocation, it doesn't have any associated
# symbols (thus nor can anything refer to it via a relocation, making it safe to
# merge with the empty section in the other input file). Here we check that the
# only two sections merged are the two empty sections and the sections with only
# STB_LOCAL or STV_HIDDEN symbols. The dynsym entries should have prevented
# anything else from being merged.
# EXPORT-NOT:  {{.}}
# EXPORT:      selected section {{.*}}:(.rodata.h3)
# EXPORT-NEXT:   removing identical section {{.*}}:(.rodata.h4)
# EXPORT-NEXT: selected section {{.*}}:(.rodata.l3)
# EXPORT-NEXT:   removing identical section {{.*}}:(.rodata.l4)
# EXPORT-NOT:  {{.}}

# If --icf=all is specified when exporting we can also merge the exported text
# sections, but not the exported rodata.
# ALL-EXPORT-NOT:  {{.}}
# ALL-EXPORT:      selected section {{.*}}:(.rodata.h3)
# ALL-EXPORT-NEXT:   removing identical section {{.*}}:(.rodata.h4)
# ALL-EXPORT-NEXT: selected section {{.*}}:(.text.f3)
# ALL-EXPORT-NEXT:   removing identical section {{.*}}:(.text.f4)
# ALL-EXPORT-NEXT: selected section {{.*}}:(.text.f1)
# ALL-EXPORT-NEXT:   removing identical section {{.*}}:(.text.f2)
# ALL-EXPORT-NEXT:   removing identical section {{.*}}:(.text.non_addrsig1)
# ALL-EXPORT-NEXT:   removing identical section {{.*}}:(.text.non_addrsig2)
# ALL-EXPORT-NEXT: selected section {{.*}}:(.rodata.l3)
# ALL-EXPORT-NEXT:   removing identical section {{.*}}:(.rodata.l4)
# ALL-EXPORT-NEXT: selected section {{.*}}:(.text)
# ALL-EXPORT-NEXT:   removing identical section {{.*}}:(.text)
# ALL-EXPORT-NOT:  {{.}}

# SH_LINK_0: --icf=safe conservatively ignores SHT_LLVM_ADDRSIG [index [[#]]] with sh_link=0 (likely created using objcopy or ld -r)

.globl _start
_start:

.section .text.f1,"ax",@progbits
.globl f1
f1:
ret

.section .text.f2,"ax",@progbits
.globl f2
f2:
ret

.section .text.f3,"ax",@progbits
.globl f3
f3:
ud2

.section .text.f4,"ax",@progbits
.globl f4
f4:
ud2

.section .rodata.g1,"a",@progbits
.globl g1
g1:
.byte 1

.section .rodata.g2,"a",@progbits
.globl g2
g2:
.byte 1

.section .rodata.g3,"a",@progbits
.globl g3
g3:
.byte 2

.section .rodata.g4,"a",@progbits
.globl g4
g4:
.byte 2

.section .rodata.l1,"a",@progbits
l1:
.byte 3

.section .rodata.l2,"a",@progbits
l2:
.byte 3

.section .rodata.l3,"a",@progbits
l3:
.byte 4

.section .rodata.l4,"a",@progbits
l4:
.byte 4

.section .rodata.h1,"a",@progbits
.globl h1
.hidden h1
h1:
.byte 5

.section .rodata.h2,"a",@progbits
.globl h2
.hidden h2
h2:
.byte 5

.section .rodata.h3,"a",@progbits
.globl h3
.hidden h3
h3:
.byte 6

.section .rodata.h4,"a",@progbits
.globl h4
.hidden h4
h4:
.byte 6

.addrsig
.addrsig_sym f1
.addrsig_sym f2
.addrsig_sym g1
.addrsig_sym g2
.addrsig_sym l1
.addrsig_sym l2
.addrsig_sym h1
.addrsig_sym h2
