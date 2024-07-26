# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags  -nostartfiles -nodefaultlibs -static -Wl,--no-relax \
# RUN:   -Wl,-q -Wl,-T %S/Inputs/got_end_of_section_symbol.lld_script  \
# RUN:   %t.o -o %t.exe
# RUN: not llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s

# CHECK: BOLT-ERROR: GOT table contains currently unsupported section end
# CHECK-SAME: symbol array_end

.section .array, "a", @progbits
.globl array_start
.globl array_end
array_start:
  .word 0
array_end:

.section .text
.globl _start
.type _start, %function
_start:
  adrp x1, #:got:array_start
  ldr x1, [x1, #:got_lo12:array_start]
  adrp x0, #:got:array_end
  ldr x0, [x0, #:got_lo12:array_end]
  adrp x2, #:got:_start
  ldr x2, [x2, #:got_lo12:_start]
  ret
