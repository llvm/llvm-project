# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=i386 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=i386 %t/crti.s -o %t/crti.o
# RUN: llvm-mc -filetype=obj -triple=i386 %t/elf-init.s -o %t/elf-init.o

## crti.o in i386 glibc<2.32 has .gnu.linkonce.t.__x86.get_pc_thunk.bx that is
## not fully supported. Test that we don't report
## "relocation refers to a symbol in a discarded section: __x86.get_pc_thunk.bx".
# RUN: ld.lld %t/a.o %t/crti.o %t/elf-init.o -o /dev/null
# RUN: ld.lld -shared %t/a.o %t/crti.o %t/elf-init.o -o /dev/null

#--- a.s
.globl _start
_start:

#--- crti.s
.section .gnu.linkonce.t.__x86.get_pc_thunk.bx,"ax"
.globl __x86.get_pc_thunk.bx
.hidden __x86.get_pc_thunk.bx
__x86.get_pc_thunk.bx:
  movl (%esp),%ebx
  ret

#--- elf-init.s
.globl __libc_csu_init
__libc_csu_init:
  call __x86.get_pc_thunk.bx

.section .text.__x86.get_pc_thunk.bx,"axG",@progbits,__x86.get_pc_thunk.bx,comdat
.globl __x86.get_pc_thunk.bx
.hidden __x86.get_pc_thunk.bx
__x86.get_pc_thunk.bx:
  movl (%esp),%ebx
  ret
