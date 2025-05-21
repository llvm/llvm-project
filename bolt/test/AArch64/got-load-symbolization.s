## Check that BOLT symbolizer properly handles loads from GOT, including
## instruction sequences changed/relaxed by the linker.

# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/main.s \
# RUN:   -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/near.s \
# RUN:   -o %t/near.o
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/far.s \
# RUN:   -o %t/far.o
# RUN: %clang %cflags %t/main.o %t/near.o %t/far.o -o %t/main.exe -Wl,-q -static
# RUN: llvm-bolt %t/main.exe -o %t/main.bolt --keep-nops --print-disasm \
# RUN:   --print-only=_start | FileCheck %s

#--- main.s

	.text
	.globl _start
	.p2align        2
	.type _start, @function
# CHECK-LABEL: _start
_start:

## Function address load relaxable into nop+adr.
# CHECK: 			nop
# CHECK-NEXT: adr x0, near
	adrp    x0, :got:near
	ldr     x0, [x0, :got_lo12:near]

## Function address load relaxable into adrp+add.
# CHECK-NEXT: adrp x1, far
# CHECK-NEXT: add  x1, x1, :lo12:far
	adrp    x1, :got:far
	ldr     x1, [x1, :got_lo12:far]

## Non-relaxable due to the instruction in-between.
# CHECK-NEXT: adrp x2, __BOLT_got_zero
# CHECK-NEXT: nop
# CHECK-NEXT: ldr  x2, [x2, :lo12:__BOLT_got_zero{{.*}}]
	adrp    x2, :got:near
	nop
	ldr     x2, [x2, :got_lo12:near]

## Load data object with local visibility. Relaxable into adrp+add.
# CHECK-NEXT: adrp x3, "local_far_data/1"
# CHECK-NEXT: add  x3, x3, :lo12:"local_far_data/1"
  adrp    x3, :got:local_far_data
  ldr     x3, [x3, :got_lo12:local_far_data]

## Global data reference relaxable into adrp+add.
# CHECK-NEXT: adrp x4, far_data
# CHECK-NEXT: add  x4, x4, :lo12:far_data
  adrp    x4, :got:far_data
  ldr     x4, [x4, :got_lo12:far_data]

	ret
	.size _start, .-_start

.weak near
.weak far
.weak far_data

## Data object separated by more than 1MB from _start.
  .data
  .type local_far_data, @object
local_far_data:
  .xword 0
.size   local_far_data, .-local_far_data

#--- near.s

	.text
	.globl near
	.type near, @function
near:
  ret
.size   near, .-near

#--- far.s

  .text

## Insert 1MB of empty space to make objects after it unreachable by adr
## instructions in _start.
  .space 0x100000

	.globl far
	.type far, @function
far:
  ret
.size   far, .-far

  .data
  .globl far_data
  .type far_data, @object
far_data:
  .xword 0
.size   far_data, .-far_data

