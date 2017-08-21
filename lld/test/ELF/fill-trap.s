# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-readobj -program-headers %t2 | FileCheck %s
# RUN: hexdump -v -s 0x0001ff0 -x %t2 | FileCheck %s -check-prefix=FILL

# CHECK: ProgramHeader {
# CHECK:   Type: PT_LOAD
# CHECK:   Offset: 0x1000
# CHECK-NEXT:   VirtualAddress:
# CHECK-NEXT:   PhysicalAddress:
# CHECK-NEXT:   FileSize: 4096
# CHECK-NEXT:   MemSize:
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     PF_R
# CHECK-NEXT:     PF_X
# CHECK-NEXT:   ]

## Check that executable page is filled with traps at it's end.
# FILL: 0001ff0 cccc cccc cccc cccc cccc cccc cccc cccc

.globl _start
_start:
  nop
