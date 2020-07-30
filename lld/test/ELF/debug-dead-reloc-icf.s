# REQUIRES: x86
## Test we resolve symbolic relocations in .debug_* sections to a tombstone
## value if the referenced section symbol is folded into another section by ICF.
## Otherwise, we would leave entries in multiple CUs claiming ownership of the
## same range of code, which can confuse consumers.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --icf=all %t.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

# CHECK:      Contents of section .debug_info:
# CHECK-NEXT:  0000 {{[0-9a-f]+}}000 00000000 ffffffff ffffffff
# CHECK:      Contents of section .debug_line:
# CHECK-NEXT:  0000 [[ADDR:[0-9a-f]+]] 00000000
# CHECK-SAME:                                   [[ADDR]] 00000000

.globl _start
_start:
  ret

## .text.1 will be folded by ICF.
.section .text.1,"ax"
  ret

.section .debug_info
  .quad .text+8
  .quad .text.1+8

## .debug_line contributions associated with folded-in functions will describe
## different lines to the canonical function. Leaving a tombstone value would
## prevent users from setting breakpoints on the folded-in functions.
## Instead resolve the relocation to the folded .text.1 to .text
.section .debug_line
  .quad .text
  .quad .text.1
