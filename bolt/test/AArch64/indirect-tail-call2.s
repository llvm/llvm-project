## This test checks that indirect tail call is properly identified by BOLT on aarch64
## when that OpDefCfaOffset=0 is in place without any stack changes between the 
## CFI instruction and the indirect branch.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -O0 %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt --print-all --print-only=indirect  \
# RUN: %t.exe -o %t.bolt | FileCheck %s

#CHECK: Binary Function "indirect" after building cfg {
#CHECK-NOT: # UNKNOWN CONTROL FLOW
#CHECK: End of Function "indirect"
	.text
	.global	indirect
	.type	indirect, %function
indirect:
.LFB0:
	.cfi_startproc
	stp	x29, x30, [sp, -16]!
	.cfi_def_cfa_offset 16
	.cfi_offset 29, -16
	.cfi_offset 30, -8
	mov	w3, w1
	add	w1, w0, w1
	add	w5, w1, w1, lsr 31
	tst	x1, 1
	asr	w5, w5, 1
	csel	w1, w1, w5, eq
	cmp	w0, w3
	beq	.L3
	mov	w0, w3
	mov	x16, x2
	ldp	x29, x30, [sp], 16
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	mov w3, w1
	add w1, w0, w1
	br	x16
.L3:
	ret
	.cfi_endproc
.LFE0:
	.size	indirect, .-indirect

