## Test that llvm-bolt detects internal calls and marks the containing function
## as non-simple.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.null --print-all 2>&1 | FileCheck %s

# CHECK: Binary Function "_start" after building cfg
# CHECK: internal call detected in function _start
# CHECK-NOT: Binary Function "_start" after validate-internal-calls

	.text
  .globl _start
  .type _start, %function
_start:
	.cfi_startproc
.LBB00:
	mov	x11, #0x1fff
	cmp	x1, x11
	b.hi	.Ltmp1

.entry1:
	movi	v4.16b, #0x0
	movi	v5.16b, #0x0
	subs	x1, x1, #0x8
	b.lo	.Ltmp2

.entry2:
	ld1	{ v2.2d, v3.2d }, [x0], #32
	ld1	{ v0.2d, v1.2d }, [x0], #32

.Ltmp2:
	uaddlp	v4.4s, v4.8h
	uaddlp	v4.2d, v4.4s
	mov	x0, v4.d[0]
	mov	x1, v4.d[1]
	add	x0, x0, x1
	ret	x30

.Ltmp1:
	mov	x8, x30

.Lloop:
	add	x5, x0, x9
	mov	x1, #0xface
	movi	v4.16b, #0x0
	movi	v5.16b, #0x0
	bl	.entry2
	add	x4, x4, x0
	mov	x0, x5
	sub	x7, x7, x10
	cmp	x7, x11
	b.hi	.Lloop

	mov	x1, x7
	bl	.entry1
	add	x0, x4, x0
	mov	x30, x8
	ret	x30

	.cfi_endproc
.size _start, .-_start

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
