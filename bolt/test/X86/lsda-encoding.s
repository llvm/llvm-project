## Test that BOLT handles large code model LSDA encoding correctly:
## 1. Auto-detection via .ltext sections
## 2. Disabling auto-detection with --large-code-model=0
## 3. Forcing large code model with --large-code-model flag

# REQUIRES: system-linux

## Build two variants: one with .ltext section, one without.
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux \
# RUN:   --defsym LTEXT=1 %s -o %t.ltext.o
# RUN: ld.lld --no-pie %t.ltext.o -o %t.ltext.exe -q -e _start
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux \
# RUN:   --defsym LTEXT=0 %s -o %t.text.o
# RUN: ld.lld --no-pie %t.text.o -o %t.text.exe -q -e _start

## Test 1: Auto-detection via .ltext section.
# RUN: llvm-bolt %t.ltext.exe -o %t.ltext.bolt --reorder-blocks=none 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# CHECK-BOLT: enabling large code model

# RUN: llvm-dwarfdump --eh-frame %t.ltext.bolt \
# RUN:   | FileCheck %s --check-prefix=CHECK-EH

## Test 2: Disable large code model with --large-code-model=0, overriding
## auto-detection even though .ltext is present.
# RUN: llvm-bolt %t.ltext.exe -o %t.ltext.bolt2 --reorder-blocks=none \
# RUN:   --large-code-model=0 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-DISABLED
# CHECK-DISABLED-NOT: enabling large code model

# RUN: llvm-dwarfdump --eh-frame %t.ltext.bolt2 \
# RUN:   | FileCheck %s --check-prefix=CHECK-EH

## Test 3: --large-code-model flag forces large code model on a binary
## without .ltext sections. No auto-detection message is printed since the
## flag is explicit, but the LSDA encoding should still be updated.
# RUN: llvm-bolt %t.text.exe -o %t.text.bolt --reorder-blocks=none \
# RUN:   --large-code-model 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-FORCED
# CHECK-FORCED-NOT: enabling large code model

# RUN: llvm-dwarfdump --eh-frame %t.text.bolt \
# RUN:   | FileCheck %s --check-prefix=CHECK-EH

## Verify the BOLT-emitted CIE uses 8-byte LSDA encoding (DW_EH_PE_absptr
## = 0x00) instead of the default 4-byte encoding (DW_EH_PE_sdata4 = 0x1B).
## In the "zLR" augmentation data: [L-enc] [R-enc].
# CHECK-EH:      Augmentation:          "zLR"
# CHECK-EH:      Augmentation data:     00 1B

  .text
  .globl foo
  .type foo, @function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo

  .globl _start
  .type _start, @function
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
.LLP0:
  ret
.LLP1:
  ret

  .cfi_endproc
.Lfunc_end0:
  .size _start, .Lfunc_end0-_start

## Exception table.
  .section .gcc_except_table,"a",@progbits
  .p2align 2
.Lexception0:
  .byte 255                             # @LPStart Encoding = omit
  .byte 255                             # @TType Encoding = omit
  .byte 1                               # Call site Encoding = uleb128
  .uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
  .uleb128 .Lfunc_begin0-.Lfunc_begin0  # Call Site 1
  .uleb128 .Ltmp0-.Lfunc_begin0
  .uleb128 .LLP0-.Lfunc_begin0          # landing pad
  .byte 0                               # action: cleanup
  .uleb128 .Ltmp0-.Lfunc_begin0         # Call Site 2
  .uleb128 .Ltmp1-.Ltmp0
  .uleb128 .LLP1-.Lfunc_begin0          # landing pad
  .byte 0                               # action: cleanup
.Lcst_end0:

## When LTEXT=1, emit large_func in .ltext to trigger auto-detection.
.if LTEXT
  .section .ltext,"axl",@progbits
  .globl large_func
  .type large_func, @function
large_func:
  .cfi_startproc
  movl $42, %eax
  retq
  .cfi_endproc
  .size large_func, .-large_func
.endif
