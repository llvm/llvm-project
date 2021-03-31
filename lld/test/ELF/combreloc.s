# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'mov aaa@gotpcrel(%rip), %rax' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o

# RUN: ld.lld -shared %t.o %t1.o -o %t.so
# RUN: llvm-readobj -r --dynamic-table %t.so | FileCheck %s
# RUN: ld.lld -shared %t.o %t1.o -o %t.so -z combreloc
# RUN: llvm-readobj -r --dynamic-table %t.so | FileCheck %s

# -z combreloc is the default: sort relocations by (!IsRelative,SymIndex,r_offset),
# and emit DT_RELACOUNT (except on MIPS) to indicate the number of relative
# relocations.

# CHECK:      DynamicSection [
# CHECK:        RELACOUNT 1
# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:     0x3428 R_X86_64_RELATIVE - 0x3430
# CHECK-NEXT:     0x2400 R_X86_64_GLOB_DAT aaa 0x0
# CHECK-NEXT:     0x3408 R_X86_64_64 aaa 0x0
# CHECK-NEXT:     0x3420 R_X86_64_64 aaa 0x0
# CHECK-NEXT:     0x3418 R_X86_64_64 bbb 0x0
# CHECK-NEXT:     0x3410 R_X86_64_64 ccc 0x0
# CHECK-NEXT:   }

# RUN: ld.lld -z nocombreloc -shared %t.o %t1.o -o %t.so
# RUN: llvm-readobj -r --dynamic-table %t.so | FileCheck --check-prefix=NOCOMB %s

# NOCOMB:      DynamicSection [
# NOCOMB-NOT:    RELACOUNT
# NOCOMB:      Relocations [
# NOCOMB-NEXT:   Section ({{.*}}) .rela.dyn {
# NOCOMB-NEXT:     0x33F8 R_X86_64_64 aaa 0x0
# NOCOMB-NEXT:     0x3400 R_X86_64_64 ccc 0x0
# NOCOMB-NEXT:     0x3408 R_X86_64_64 bbb 0x0
# NOCOMB-NEXT:     0x3410 R_X86_64_64 aaa 0x0
# NOCOMB-NEXT:     0x3418 R_X86_64_RELATIVE - 0x3420
# NOCOMB-NEXT:     0x23F0 R_X86_64_GLOB_DAT aaa 0x0
# NOCOMB-NEXT:   }

.globl aaa, bbb, ccc
.data
 .quad aaa
 .quad ccc
 .quad bbb
 .quad aaa
 .quad relative
relative:
