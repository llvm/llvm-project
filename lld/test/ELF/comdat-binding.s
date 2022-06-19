# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/g.s -o %t/g.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/w.s -o %t/w.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/u.s -o %t/u.o
# RUN: ld.lld -e 0 %t/w.o %t/u.o -o %t/w
# RUN: llvm-readelf -s %t/w | FileCheck %s --check-prefix=WEAK
# RUN: ld.lld -e 0 %t/u.o %t/w.o -o %t/u
# RUN: llvm-readelf -s %t/u | FileCheck %s --check-prefix=UNIQUE

## We prefer STB_GLOBAL definition, then changing it to undefined since it is in
## in a non-prevailing COMDAT. Ideally this should be defined, but our behavior
## is fine because valid input cannot form this case.
# RUN: ld.lld -e 0 %t/w.o %t/g.o -o %t/und --noinhibit-exec 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-readelf -s %t/und | FileCheck %s --check-prefix=UND

# WEAK:   NOTYPE WEAK   DEFAULT [[#]] _ZZ1fvE1x
# UNIQUE: OBJECT UNIQUE DEFAULT [[#]] _ZZ1fvE1x
# UND:    NOTYPE GLOBAL DEFAULT UND   _ZZ1fvE1x

# WARN: warning: relocation refers to a symbol in a discarded section: f()::x
# WARN-NEXT: >>> defined in {{.*}}g.o
# WARN-NEXT: >>> section group signature: _ZZ1fvE1x
# WARN-NEXT: >>> prevailing definition is in {{.*}}w.o
# WARN-NEXT: >>> or the symbol in the prevailing group had STB_WEAK binding and the symbol in a non-prevailing group had STB_GLOBAL binding. Mixing groups with STB_WEAK and STB_GLOBAL binding signature is not supported
# WARN-NEXT: >>> referenced by {{.*}}g.o:(.text+0x3)

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
