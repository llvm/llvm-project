# REQUIRES: x86
## Deduplicate SHF_MERGE|SHF_STRINGS pieces from sections large enough to be
## processed in separate chunks. Placing the same pieces in one section must
## give the same layout.

# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym .LSPLIT=1 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym .LSPLIT=0 %s -o %t1.o
# RUN: ld.lld %t.o -o %t
# RUN: ld.lld %t1.o -o %t1
# RUN: cmp %t %t1
# RUN: llvm-nm %t > %t.txt
# RUN: llvm-readelf -S -p .rodata %t >> %t.txt
# RUN: FileCheck %s -D#SIZE=0x20 < %t.txt
# RUN: ld.lld --gc-sections %t.o -o %t.gc
# RUN: llvm-nm %t.gc > %t.gc.txt
# RUN: llvm-readelf -S -p .rodata %t.gc >> %t.gc.txt
# RUN: FileCheck %s -D#SIZE=0x10 < %t.gc.txt

## Each symbol must point to its own string, and c0 and c3 refer to the same
## deduplicated string.
# CHECK:      [[#%x,A0:]] r a0{{$}}
# CHECK:      [[#%x,A3:]] r a3{{$}}
# CHECK:      [[#%x,B2:]] r b2{{$}}
# CHECK:      [[#%x,C0:]] r c0{{$}}
# CHECK:      [[#%.16x,C0]] r c3{{$}}
# CHECK:      .rodata PROGBITS [[#%x,RODATA:]] [[#%x,]] [[#%.6x,SIZE]] 01 AMS
# CHECK:      String dump of section '.rodata':
# CHECK-DAG:  [{{ *}}[[#%x,A0-RODATA]]] a0{{$}}
# CHECK-DAG:  [{{ *}}[[#%x,C0-RODATA]]] common{{$}}
# CHECK-DAG:  [{{ *}}[[#%x,B2-RODATA]]] b2{{$}}
# CHECK-DAG:  [{{ *}}[[#%x,A3-RODATA]]] a3{{$}}

.globl _start
_start:
  movl $a0, %eax
  movl $c0, %eax
  movl $b2, %eax
  movl $a3, %eax
  movl $c3, %eax

## Each section has 3003 pieces, more than half the minimum chunk size (4096),
## so each one is a separate chunk.
.macro sec k
.if .LSPLIT
.section .rodata.str1.1,"aMS",@progbits,1,unique,\k
.else
.section .rodata.str1.1,"aMS",@progbits,1
.endif
a\k: .asciz "a\k"
b\k: .asciz "b\k"
c\k: .asciz "common"
.zero 3000
.endm
.irp k,0,1,2,3
sec \k
.endr
