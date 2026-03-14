# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -o %t.bolt.exe --funcs=nocfi_function
# RUN: llvm-readelf --section-headers %t.bolt.exe | FileCheck %s

## Check that llvm-bolt does not allocate unmarked space for original .eh_frame
## after .text when no update is needed to .eh_frame.

# CHECK: {{ .text}} PROGBITS [[#%x,ADDR:]] [[#%x,OFFSET:]] [[#%x,SIZE:]]
# CHECK-NEXT: 0000000000000000 [[#%x, OFFSET + SIZE]]

  .text
  .globl nocfi_function
  .type nocfi_function,@function
nocfi_function:
  ret
  .size nocfi_function, .-nocfi_function

  .globl _start
  .type _start,@function
_start:
  .cfi_startproc
  call nocfi_function
  .size _start, .-_start
  .cfi_endproc
