# Instrumenting binaries with pac-ret hardening is not supported.
# This test makes sure that BOLT will fail when ran with both the
# --allow-experimental-pacret and --instrument flags.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags  %t.o -o %t.exe -Wl,-q

# RUN: not llvm-bolt %t.exe -o %t.bolt --allow-experimental-pacret --instrument 2>&1 | FileCheck %s

# CHECK: BOLT-ERROR: Instrumenting binaries with pac-ret hardening is not supported.

	.text
	.globl	foo
	.p2align	2
	.type	foo,@function
foo:
	.cfi_startproc
	hint	#25
	.cfi_negate_ra_state
	sub	sp, sp, #16
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	str	w0, [sp, #12]
	ldr	w8, [sp, #12]
	add	w0, w8, #1
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #16
	hint	#29
	.cfi_negate_ra_state
	ret
.Lfunc_end1:
	.size	foo, .Lfunc_end1-foo
	.cfi_endproc

	.global _start
	.type _start, %function
_start:
	b foo

.reloc 0, R_AARCH64_NONE
