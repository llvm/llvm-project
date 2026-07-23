## Check that LDR relaxation will fail since LDR is inside a non-simple
## function and there is no NOP next to it.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym FAIL=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: not llvm-bolt %t.so -o %t.bolt 2>&1 | FileCheck %s --check-prefix=FAIL

# FAIL: BOLT-ERROR: cannot relax LDR in non-simple function _start

.ifdef FAIL
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  br x2
  ldr x0, _foo
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check that LDR relaxation is not needed since the reference is not far away.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym NOT_NEEDED=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=NOT_NEEDED

# NOT_NEEDED: <_start>
# NOT_NEEDED-NEXT: ldr

.ifdef NOT_NEEDED
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  ldr x0, _start
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check that LDR relaxation is done in a simple function, where NOP will
## be inserted as needed.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_SIMPLE=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELAX

# RELAX: adrp
# RELAX-NEXT: ldr

.ifdef RELAX_SIMPLE
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  ldr x0, _foo
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check that LDR relaxation is done in a non-simple function, where NOP
## exists next to LDR.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_NON_SIMPLE=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELAX

.ifdef RELAX_NON_SIMPLE
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  br x2
  ldr x0, _foo
  nop
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check LDR relaxation works on loading W (low 32-bit of X) registers.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_SIMPLE_WREG=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELAXW

# RELAXW: adrp x0
# RELAXW-NEXT: ldr w0

.ifdef RELAX_SIMPLE_WREG
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  ldr w0, _foo
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check LDR relaxation works on LDRSW (literal)

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_SIMPLE_LDRSW=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELAX_LDRSW

# RELAX_LDRSW: adrp
# RELAX_LDRSW-NEXT: ldrsw

.ifdef RELAX_SIMPLE_LDRSW
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  ldrsw x0, _foo
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check LDR relaxation works on loading S registers.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_SIMPLE_LDR_FP32=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELAX_LDR_FP32

# RELAX_LDR_FP32: _start
# RELAX_LDR_FP32-NEXT: stp x16, x17, [sp, #-0x10]!
# RELAX_LDR_FP32-NEXT: adrp x16
# RELAX_LDR_FP32-NEXT: ldr s0
# RELAX_LDR_FP32-NEXT: ldp x16, x17, [sp], #0x10
# RELAX_LDR_FP32-NEXT: ret
.ifdef RELAX_SIMPLE_LDR_FP32
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  ldr s0, _bar
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check LDR relaxation works on loading D registers.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_SIMPLE_LDR_FP64=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELAX_LDR_FP64

# RELAX_LDR_FP64: _start
# RELAX_LDR_FP64-NEXT: stp x16, x17, [sp, #-0x10]!
# RELAX_LDR_FP64-NEXT: adrp x16
# RELAX_LDR_FP64-NEXT: ldr d0
# RELAX_LDR_FP64-NEXT: ldp x16, x17, [sp], #0x10
# RELAX_LDR_FP64-NEXT: ret
.ifdef RELAX_SIMPLE_LDR_FP64
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  ldr d0, _bar
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check LDR relaxation works on loading Q registers.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_SIMPLE_LDR_FP128=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELAX_LDR_FP128

# RELAX_LDR_FP128: _start
# RELAX_LDR_FP128-NEXT: stp x16, x17, [sp, #-0x10]!
# RELAX_LDR_FP128-NEXT: adrp x16
# RELAX_LDR_FP128-NEXT: ldr q0
# RELAX_LDR_FP128-NEXT: ldp x16, x17, [sp], #0x10
# RELAX_LDR_FP128-NEXT: ret
.ifdef RELAX_SIMPLE_LDR_FP128
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  ldr q0, _bar
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check that LDR relaxation is done in a non-simple function when the load
## instruction is preceded by enough nops.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_NON_SIMPLE_PRECEDED_BY_NOPS=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=PRECEDED_BY_NOPS

# PRECEDED_BY_NOPS: _start
# PRECEDED_BY_NOPS-NEXT: br x2
# PRECEDED_BY_NOPS-NEXT: nop
# PRECEDED_BY_NOPS-NEXT: stp x16, x17, [sp, #-0x10]!
# PRECEDED_BY_NOPS-NEXT: adrp x16
# PRECEDED_BY_NOPS-NEXT: ldr q0
# PRECEDED_BY_NOPS-NEXT: ldp x16, x17, [sp], #0x10
# PRECEDED_BY_NOPS-NEXT: ret
.ifdef RELAX_NON_SIMPLE_PRECEDED_BY_NOPS
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  br x2
  nop
  nop
  nop
  nop
  ldr q0, _bar
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check that LDR relaxation is done in a non-simple function when the load
## instruction is followed by enough nops.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_NON_SIMPLE_FOLLOWED_BY_NOPS=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=FOLLOWED_BY_NOPS

# FOLLOWED_BY_NOPS: _start
# FOLLOWED_BY_NOPS-NEXT: br x2
# FOLLOWED_BY_NOPS-NEXT: stp x16, x17, [sp, #-0x10]!
# FOLLOWED_BY_NOPS-NEXT: adrp x16
# FOLLOWED_BY_NOPS-NEXT: ldr q0
# FOLLOWED_BY_NOPS-NEXT: ldp x16, x17, [sp], #0x10
# FOLLOWED_BY_NOPS-NEXT: nop
# FOLLOWED_BY_NOPS-NEXT: ret
.ifdef RELAX_NON_SIMPLE_FOLLOWED_BY_NOPS
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  br x2
  ldr q0, _bar
  nop
  nop
  nop
  nop
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check that LDR relaxation is done in a non-simple function when the load
## instruction is surrouned by enough nops.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_NON_SIMPLE_SURROUNDED_BY_NOPS=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=SURROUNDED_BY_NOPS

# SURROUNDED_BY_NOPS: _start
# SURROUNDED_BY_NOPS-NEXT: br x2
# SURROUNDED_BY_NOPS-NEXT: stp x16, x17, [sp, #-0x10]!
# SURROUNDED_BY_NOPS-NEXT: adrp x16
# SURROUNDED_BY_NOPS-NEXT: ldr q0
# SURROUNDED_BY_NOPS-NEXT: ldp x16, x17, [sp], #0x10
# SURROUNDED_BY_NOPS-NEXT: ret
.ifdef RELAX_NON_SIMPLE_SURROUNDED_BY_NOPS
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  br x2
  nop
  nop
  ldr q0, _bar
  nop
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

## Check that LDR relaxation will fail in a non-simple function when there are
## not enough NOPs around the load instruction to accommodate the relaxed
## instruction sequence in place.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym RELAX_NON_SIMPLE_NOT_ENOUGH_NOPS=1 %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.so -Wl,-q
# RUN: not llvm-bolt %t.so -o %t.bolt 2>&1 | FileCheck %s --check-prefix=NOT_ENOUGH_NOPS

# NOT_ENOUGH_NOPS: BOLT-ERROR: cannot relax LDR in non-simple function _start
.ifdef RELAX_NON_SIMPLE_NOT_ENOUGH_NOPS
  .text
  .global _start
  .type _start, %function
_start:
  .cfi_startproc
  br x2
  nop
  ldr q0, _bar
  nop
  ret
  .cfi_endproc
  .size _start, .-_start
.endif

  .section .text_cold
  .global _foo
  .align 3
_foo:
  .long 0x12345678
  .size _foo, .-_foo
  .global _bar
  .align 4
_bar:
  .xword  0x0000000000000000
  .xword  0x0000000000000000
  .size _bar, .-_bar
