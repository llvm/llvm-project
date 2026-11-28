## A page can feed addresses with different low bits. Preserve the absolute
## input page in the ADRP and leave both ADD instructions unchanged.
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %t.o
# RUN: ld.lld --no-relax %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --recover-relocations \
# RUN:   --print-relocation-recovery --print-only=_start 2>&1 | FileCheck %s
# CHECK: after recover-relocations
# CHECK: adrp x0, {{.*}}__BOLT_zero_addr
# CHECK-NEXT: add x1, x0, #0x{{[0-9a-f]+}}
# CHECK-NEXT: add x2, x0, #0x{{[0-9a-f]+}}
.text
.globl _start
.type _start, %function
_start:
.cfi_startproc
  adrp x0, first
  add x1, x0, :lo12:first
  add x2, x0, :lo12:second
  ret
.cfi_endproc
.size _start, .-_start
.type first, %function
first:
.cfi_startproc
  ret
.cfi_endproc
.size first, .-first
.type second, %function
second:
.cfi_startproc
  ret
.cfi_endproc
.size second, .-second
