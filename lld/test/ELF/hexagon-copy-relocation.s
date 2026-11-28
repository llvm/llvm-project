# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon shared.s -o shared.o
# RUN: ld.lld -shared shared.o -o shared.so
# RUN: llvm-mc -filetype=obj -triple=hexagon main.s -o main.o
# RUN: ld.lld main.o shared.so -o main
# RUN: llvm-readobj -r main | FileCheck %s
# RUN: llvm-readelf -S -s main | FileCheck --check-prefix=SYM %s

## Non-PIE references to writable and read-only data exported by a DSO need
## COPY relocations so the dynamic loader initializes their executable copies.
# CHECK:      Section ({{.*}}) .rela.dyn {
# CHECK-DAG:    R_HEX_COPY writable 0x0
# CHECK-DAG:    R_HEX_COPY readonly 0x0
# CHECK-NEXT: }

## The read-only object's copy is placed in .bss.rel.ro, the writable one in .bss.
# SYM:      .bss.rel.ro NOBITS
# SYM:      .bss        NOBITS
# SYM-DAG:  4 OBJECT GLOBAL DEFAULT {{.*}} writable
# SYM-DAG:  4 OBJECT GLOBAL DEFAULT {{.*}} readonly

#--- main.s
.globl _start
_start:
  r0 = memw(##writable)
  r1 = memw(##readonly)

#--- shared.s
.data
.globl writable
.type writable,@object
.size writable,4
writable:
  .word 42
.section .rodata,"a",@progbits
.globl readonly
.type readonly,@object
.size readonly,4
readonly:
  .word 17
