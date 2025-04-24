## Check that in lite mode llvm-bolt updates function references in
## non-optimized code.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite
# RUN: llvm-objdump -d --disassemble-symbols=cold_function %t.exe \
# RUN:   | FileCheck %s --check-prefix=CHECK-INPUT
# RUN: llvm-objdump -d --disassemble-symbols=cold_function %t.bolt \
# RUN:   | FileCheck %s

## In lite mode, optimized code will be separated from the original .text by
## over 128MB, making it impossible for call/bl instructions in cold functions
## to reach optimized functions directly.

  .text
  .globl _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 100
  .cfi_startproc
  cmp  x0, 1
  b.eq  .L0
  bl cold_function
.L0:
  ret  x30
  .cfi_endproc
.size _start, .-_start

## Cold non-optimized function with a reference to a hot function (_start).
# CHECK: Disassembly of section .bolt.org.text:
# CHECK-LABEL: <cold_function>
  .globl cold_function
  .type cold_function, %function
cold_function:
  .cfi_startproc

## Absolute 64-bit function pointer reference.
## We check for the lower 16 bits of _start to be zeros after update.
  movz    x0, :abs_g3:_start
  movk    x0, :abs_g2_nc:_start
  movk    x0, :abs_g1_nc:_start
# CHECK-INPUT-NOT: movk x0, #0x0{{$}}
# CHECK: movk x0, #0x0{{$}}
  movk    x0, :abs_g0_nc:_start

## Relaxable address reference.
# CHECK-INPUT:      nop
# CHECK-INPUT-NEXT: adr x1
# CHECK-NEXT:       adrp x1, [[ADDR:0x[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:       add  x1
  adrp    x1, _start
  add     x1, x1, :lo12:_start

## Non-relaxable address reference.
# CHECK-INPUT-NEXT: adrp x2
# CHECK-INPUT-NEXT: add  x2
# CHECK-NEXT:       adrp x2, [[ADDR]]
# CHECK-NEXT:       add  x2
  adrp    x2, far_func
  add     x2, x2, :lo12:far_func

## Check that fully-relaxed GOT reference is converted into ADRP+ADD.
  adrp    x3, :got:_start
  ldr     x3, [x3, #:got_lo12:_start]
# CHECK-INPUT-NEXT: nop
# CHECK-INPUT-NEXT: adr x3
# CHECK-NEXT:       adrp x3, [[ADDR]]
# CHECK-NEXT:       add  x3

## Check that partially-relaxed GOT reference is converted into ADRP+ADD.
  adrp    x4, :got:far_func
  ldr     x4, [x4, #:got_lo12:far_func]
# CHECK-INPUT-NEXT: adrp x4
# CHECK-INPUT-NEXT: add x4
# CHECK-NEXT:       adrp x4, [[ADDR]]
# CHECK-NEXT:       add  x4

## Check that non-relaxable GOT load is left intact.
  adrp    x5, :got:far_func
  nop
  ldr     x5, [x5, #:got_lo12:far_func]
# CHECK-INPUT-NEXT: adrp x5
# CHECK-INPUT-NEXT: nop
# CHECK-INPUT-NEXT: ldr x5
# CHECK-NEXT:       adrp x5
# CHECK-NOT: [[ADDR]]
# CHECK-NEXT:       nop
# CHECK-NEXT:       ldr x5

  .cfi_endproc
.size cold_function, .-cold_function

## Reserve 1MB of space to make functions that follow unreachable by ADRs in
## code that precedes this gap.
.space 0x100000

  .globl far_func
  .type far_func, %function
far_func:
# FDATA: 0 [unknown] 0 1 far_func 0 0 100
  .cfi_startproc
  ret  x30
  .cfi_endproc
.size far_func, .-far_func

