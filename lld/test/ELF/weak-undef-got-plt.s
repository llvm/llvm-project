# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 -x86-relax-relocations=false a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/shared.s -o s.o
# RUN: ld.lld -shared s.o -o s.so

# RUN: ld.lld a.o -o a
# RUN: llvm-readelf -r a | FileCheck %s --check-prefix=NORELOC
# RUN: ld.lld a.o -o a -z dynamic-undefined-weak
# RUN: llvm-readelf -r a | FileCheck %s --check-prefix=NORELOC
# RUN: ld.lld a.o s.so -o as
# RUN: llvm-objdump -dR as | FileCheck %s
# RUN: ld.lld a.o s.so -o as -z nodynamic-undefined-weak
# RUN: llvm-readelf -r a | FileCheck %s --check-prefix=NORELOC

# RUN: ld.lld -pie a.o s.so -o as.pie
# RUN: llvm-objdump -dR as.pie | FileCheck %s
# RUN: ld.lld -pie a.o s.so -o as.pie -z nodynamic-undefined-weak
# RUN: llvm-readelf -r as.pie | FileCheck --check-prefix=NORELOC %s

# RUN: ld.lld -shared a.o -o a.so
# RUN: llvm-objdump -dR a.so | FileCheck %s

# NORELOC:    no relocation

# CHECK:      TYPE                     VALUE
# CHECK-NEXT: R_X86_64_GLOB_DAT        foo{{$}}
# CHECK-NEXT: R_X86_64_JUMP_SLOT       foo{{$}}
# CHECK-EMPTY:
# CHECK:      <_start>:
# CHECK-NEXT:   movq {{.*}}(%rip), %rax
# CHECK-NEXT:   callq {{.*}} <foo@plt>

#--- a.s
.weak foo

.globl _start
_start:
mov foo@gotpcrel(%rip), %rax
call foo
