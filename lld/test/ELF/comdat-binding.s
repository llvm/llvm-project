# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/g.s -o %t/g.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/w.s -o %t/w.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/u.s -o %t/u.o
# RUN: ld.lld -e 0 %t/w.o %t/u.o -o %t/w
# RUN: llvm-readelf -s %t/w | FileCheck %s --check-prefix=WEAK
# RUN: ld.lld -e 0 %t/u.o %t/w.o -o %t/u
# RUN: llvm-readelf -s %t/u | FileCheck %s --check-prefix=UNIQUE

# RUN: ld.lld -e 0 %t/w.o %t/g.o -o %t/w
# RUN: llvm-readelf -s %t/w | FileCheck %s --check-prefix=WEAK

# WEAK:   NOTYPE WEAK   DEFAULT [[#]] _ZZ1fvE1x
# UNIQUE: OBJECT UNIQUE DEFAULT [[#]] _ZZ1fvE1x

#--- g.s
movq _ZZ1fvE1x@gotpcrel(%rip), %rax

.section .bss._ZZ1fvE1x,"awG",@nobits,_ZZ1fvE1x,comdat
.globl _ZZ1fvE1x
_ZZ1fvE1x:

#--- w.s
.section .bss._ZZ1fvE1x,"awG",@nobits,_ZZ1fvE1x,comdat
.weak _ZZ1fvE1x
_ZZ1fvE1x:

#--- u.s
.section .bss._ZZ1fvE1x,"awG",@nobits,_ZZ1fvE1x,comdat
.type _ZZ1fvE1x, @gnu_unique_object
_ZZ1fvE1x:
