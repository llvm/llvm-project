## Check that llvm-bolt is able to overwrite LSDA in ULEB128 format in-place for
## all types of binaries.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --no-pie %t.o -o %t.exe -q
# RUN: ld.lld --pie %t.o -o %t.pie -q
# RUN: ld.lld --shared %t.o -o %t.so -q
# RUN: llvm-bolt %t.exe -o %t.bolt --strict \
# RUN:   | FileCheck --check-prefix=CHECK-BOLT %s
# RUN: llvm-bolt %t.pie -o %t.pie.bolt --strict \
# RUN:   | FileCheck --check-prefix=CHECK-BOLT %s
# RUN: llvm-bolt %t.so -o %t.so.bolt --strict \
# RUN:   | FileCheck --check-prefix=CHECK-BOLT %s

# CHECK-BOLT: rewriting .gcc_except_table in-place

# RUN: llvm-readelf -WS %t.bolt | FileCheck --check-prefix=CHECK-ELF %s
# RUN: llvm-readelf -WS %t.pie.bolt | FileCheck --check-prefix=CHECK-ELF %s
# RUN: llvm-readelf -WS %t.so.bolt | FileCheck --check-prefix=CHECK-ELF %s

# CHECK-ELF-NOT: .bolt.org.gcc_except_table

  .text
  .global foo
  .type foo, %function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo

  .globl _start
  .type _start, %function
_start:
.Lfunc_begin0:
  .cfi_startproc
  .cfi_lsda 27, .Lexception0
  call foo
.Ltmp0:
  call foo
.Ltmp1:
  ret

## Landing pads.
.LLP1:
  ret
.LLP0:
  ret

  .cfi_endproc
.Lfunc_end0:
  .size _start, .-_start

## EH table.
  .section  .gcc_except_table,"a",@progbits
  .p2align  2
GCC_except_table0:
.Lexception0:
  .byte 255                             # @LPStart Encoding = omit
  .byte 255                             # @TType Encoding = omit
  .byte 1                               # Call site Encoding = uleb128
  .uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
  .uleb128 .Lfunc_begin0-.Lfunc_begin0  # >> Call Site 1 <<
  .uleb128 .Ltmp0-.Lfunc_begin0         #   Call between .Lfunc_begin0 and .Ltmp0
  .uleb128 .LLP0-.Lfunc_begin0          #   jumps to .LLP0
  .byte 0                               #   On action: cleanup
  .uleb128 .Ltmp0-.Lfunc_begin0         # >> Call Site 2 <<
  .uleb128 .Ltmp1-.Ltmp0                #   Call between .Ltmp0 and .Ltmp1
  .uleb128 .LLP1-.Lfunc_begin0          #     jumps to .LLP1
  .byte 0                               #   On action: cleanup
.Lcst_end0:

