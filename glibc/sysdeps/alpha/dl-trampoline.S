/* PLT trampolines.  Alpha version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	.set noat

.macro savei regno, offset
	stq	$\regno, \offset($30)
	cfi_rel_offset(\regno, \offset)
.endm

.macro savef regno, offset
	stt	$f\regno, \offset($30)
	cfi_rel_offset(\regno+32, \offset)
.endm

	.align	4
	.globl	_dl_runtime_resolve_new
	.ent	_dl_runtime_resolve_new

#undef FRAMESIZE
#define FRAMESIZE	14*8

_dl_runtime_resolve_new:
	.frame	$30, FRAMESIZE, $26, 0
	.mask	0x4000000, 0

	ldah	$29, 0($27)		!gpdisp!1
	lda	$30, -FRAMESIZE($30)
	stq	$26, 0*8($30)
	stq	$16, 2*8($30)

	stq	$17, 3*8($30)
	lda	$29, 0($29)		!gpdisp!1
	stq	$18, 4*8($30)
	mov	$28, $16		/* link_map from .got.plt */

	stq	$19, 5*8($30)
	mov	$25, $17		/* offset of reloc entry */
	stq	$20, 6*8($30)
	mov	$26, $18		/* return address */

	stq	$21, 7*8($30)
	stt	$f16, 8*8($30)
	stt	$f17, 9*8($30)
	stt	$f18, 10*8($30)

	stt	$f19, 11*8($30)
	stt	$f20, 12*8($30)
	stt	$f21, 13*8($30)
	.prologue 2

	bsr	$26, _dl_fixup		!samegp
	mov	$0, $27

	ldq	$26, 0*8($30)
	ldq	$16, 2*8($30)
	ldq	$17, 3*8($30)
	ldq	$18, 4*8($30)
	ldq	$19, 5*8($30)
	ldq	$20, 6*8($30)
	ldq	$21, 7*8($30)
	ldt	$f16, 8*8($30)
	ldt	$f17, 9*8($30)
	ldt	$f18, 10*8($30)
	ldt	$f19, 11*8($30)
	ldt	$f20, 12*8($30)
	ldt	$f21, 13*8($30)
	lda	$30, FRAMESIZE($30)
	jmp	$31, ($27), 0
	.end	_dl_runtime_resolve_new

	.globl	_dl_runtime_profile_new
	.type	_dl_runtime_profile_new, @function

#undef FRAMESIZE
#define FRAMESIZE	20*8

	/* We save the registers in a different order than desired by
	   .mask/.fmask, so we have to use explicit cfi directives.  */
	cfi_startproc

_dl_runtime_profile_new:
	ldah	$29, 0($27)		!gpdisp!2
	lda	$30, -FRAMESIZE($30)
	savei	26, 0*8
	stq	$16, 2*8($30)

	stq	$17, 3*8($30)
	lda	$29, 0($29)		!gpdisp!2
	stq	$18, 4*8($30)
	lda	$1, FRAMESIZE($30)	/* incoming sp value */

	stq	$1, 1*8($30)
	stq	$19, 5*8($30)
	stq	$20, 6*8($30)
	mov	$28, $16		/* link_map from .got.plt */

	stq	$21, 7*8($30)
	mov	$25, $17		/* offset of reloc entry */
	stt	$f16, 8*8($30)
	mov	$26, $18		/* return address */

	stt	$f17, 9*8($30)
	mov	$30, $19		/* La_alpha_regs address */
	stt	$f18, 10*8($30)
	lda	$20, 14*8($30)		/* framesize address */

	stt	$f19, 11*8($30)
	stt	$f20, 12*8($30)
	stt	$f21, 13*8($30)
	stq	$28, 16*8($30)
	stq	$25, 17*8($30)

	bsr	$26, _dl_profile_fixup	!samegp
	mov	$0, $27

	/* Discover if we're wrapping this call.  */
	ldq	$18, 14*8($30)
	bge	$18, 1f

	ldq	$26, 0*8($30)
	ldq	$16, 2*8($30)
	ldq	$17, 3*8($30)
	ldq	$18, 4*8($30)
	ldq	$19, 5*8($30)
	ldq	$20, 6*8($30)
	ldq	$21, 7*8($30)
	ldt	$f16, 8*8($30)
	ldt	$f17, 9*8($30)
	ldt	$f18, 10*8($30)
	ldt	$f19, 11*8($30)
	ldt	$f20, 12*8($30)
	ldt	$f21, 13*8($30)
	lda	$30, FRAMESIZE($30)
	jmp	$31, ($27), 0

1:
	/* Create a frame pointer and allocate a new argument frame.  */
	savei	15, 15*8
	mov	$30, $15
	cfi_def_cfa_register (15)
	addq	$18, 15, $18
	bic	$18, 15, $18
	subq	$30, $18, $30

	/* Save the call destination around memcpy.  */
	stq	$0, 14*8($30)

	/* Copy the stack arguments into place.  */
	lda	$16, 0($30)
	lda	$17, FRAMESIZE($15)
	jsr	$26, memcpy
	ldgp	$29, 0($26)

	/* Reload the argument registers.  */
	ldq	$27, 14*8($30)
	ldq	$16, 2*8($15)
	ldq	$17, 3*8($15)
	ldq	$18, 4*8($15)
	ldq	$19, 5*8($15)
	ldq	$20, 6*8($15)
	ldq	$21, 7*8($15)
	ldt	$f16, 8*8($15)
	ldt	$f17, 9*8($15)
	ldt	$f18, 10*8($15)
	ldt	$f19, 11*8($15)
	ldt	$f20, 12*8($15)
	ldt	$f21, 13*8($15)

	jsr	$26, ($27), 0
	ldgp	$29, 0($26)

	/* Set up for call to _dl_call_pltexit.  */
	ldq	$16, 16*8($15)
	ldq	$17, 17*8($15)
	stq	$0, 16*8($15)
	lda	$18, 0($15)
	stq	$1, 17*8($15)
	lda	$19, 16*8($15)
	stt	$f0, 18*8($15)
	stt	$f1, 19*8($15)
	bsr	$26, _dl_call_pltexit	!samegp

	mov	$15, $30
	cfi_def_cfa_register (30)
	ldq	$26, 0($30)
	ldq	$15, 15*8($30)
	lda	$30, FRAMESIZE($30)
	ret

	cfi_endproc
	.size	_dl_runtime_profile_new, .-_dl_runtime_profile_new

	.align	4
	.globl	_dl_runtime_resolve_old
	.ent	_dl_runtime_resolve_old

#undef FRAMESIZE
#define FRAMESIZE	44*8

_dl_runtime_resolve_old:
	lda	$30, -FRAMESIZE($30)
	.frame	$30, FRAMESIZE, $26
	/* Preserve all registers that C normally doesn't.  */
	stq	$26, 0*8($30)
	stq	$0, 1*8($30)
	stq	$1, 2*8($30)
	stq	$2, 3*8($30)
	stq	$3, 4*8($30)
	stq	$4, 5*8($30)
	stq	$5, 6*8($30)
	stq	$6, 7*8($30)
	stq	$7, 8*8($30)
	stq	$8, 9*8($30)
	stq	$16, 10*8($30)
	stq	$17, 11*8($30)
	stq	$18, 12*8($30)
	stq	$19, 13*8($30)
	stq	$20, 14*8($30)
	stq	$21, 15*8($30)
	stq	$22, 16*8($30)
	stq	$23, 17*8($30)
	stq	$24, 18*8($30)
	stq	$25, 19*8($30)
	stq	$29, 20*8($30)
	stt	$f0, 21*8($30)
	stt	$f1, 22*8($30)
	stt	$f10, 23*8($30)
	stt	$f11, 24*8($30)
	stt	$f12, 25*8($30)
	stt	$f13, 26*8($30)
	stt	$f14, 27*8($30)
	stt	$f15, 28*8($30)
	stt	$f16, 29*8($30)
	stt	$f17, 30*8($30)
	stt	$f18, 31*8($30)
	stt	$f19, 32*8($30)
	stt	$f20, 33*8($30)
	stt	$f21, 34*8($30)
	stt	$f22, 35*8($30)
	stt	$f23, 36*8($30)
	stt	$f24, 37*8($30)
	stt	$f25, 38*8($30)
	stt	$f26, 39*8($30)
	stt	$f27, 40*8($30)
	stt	$f28, 41*8($30)
	stt	$f29, 42*8($30)
	stt	$f30, 43*8($30)
	.mask	0x27ff01ff, -FRAMESIZE
	.fmask	0xfffffc03, -FRAMESIZE+21*8
	/* Set up our GP.  */
	br	$29, .+4
	ldgp	$29, 0($29)
	.prologue 0
	/* Set up the arguments for _dl_fixup:
	   $16 = link_map out of plt0
	   $17 = offset of reloc entry = ($28 - $27 - 20) /12 * 24
	   $18 = return address
	*/
	subq	$28, $27, $17
	ldq	$16, 8($27)
	subq	$17, 20, $17
	mov	$26, $18
	addq	$17, $17, $17
	bsr	$26, _dl_fixup	!samegp

	/* Move the destination address into position.  */
	mov	$0, $27
	/* Restore program registers.  */
	ldq	$26, 0*8($30)
	ldq	$0, 1*8($30)
	ldq	$1, 2*8($30)
	ldq	$2, 3*8($30)
	ldq	$3, 4*8($30)
	ldq	$4, 5*8($30)
	ldq	$5, 6*8($30)
	ldq	$6, 7*8($30)
	ldq	$7, 8*8($30)
	ldq	$8, 9*8($30)
	ldq	$16, 10*8($30)
	ldq	$17, 11*8($30)
	ldq	$18, 12*8($30)
	ldq	$19, 13*8($30)
	ldq	$20, 14*8($30)
	ldq	$21, 15*8($30)
	ldq	$22, 16*8($30)
	ldq	$23, 17*8($30)
	ldq	$24, 18*8($30)
	ldq	$25, 19*8($30)
	ldq	$29, 20*8($30)
	ldt	$f0, 21*8($30)
	ldt	$f1, 22*8($30)
	ldt	$f10, 23*8($30)
	ldt	$f11, 24*8($30)
	ldt	$f12, 25*8($30)
	ldt	$f13, 26*8($30)
	ldt	$f14, 27*8($30)
	ldt	$f15, 28*8($30)
	ldt	$f16, 29*8($30)
	ldt	$f17, 30*8($30)
	ldt	$f18, 31*8($30)
	ldt	$f19, 32*8($30)
	ldt	$f20, 33*8($30)
	ldt	$f21, 34*8($30)
	ldt	$f22, 35*8($30)
	ldt	$f23, 36*8($30)
	ldt	$f24, 37*8($30)
	ldt	$f25, 38*8($30)
	ldt	$f26, 39*8($30)
	ldt	$f27, 40*8($30)
	ldt	$f28, 41*8($30)
	ldt	$f29, 42*8($30)
	ldt	$f30, 43*8($30)
	/* Flush the Icache after having modified the .plt code.  */
	imb
	/* Clean up and turn control to the destination */
	lda	$30, FRAMESIZE($30)
	jmp	$31, ($27)

	.end	_dl_runtime_resolve_old

	.globl	_dl_runtime_profile_old
	.usepv	_dl_runtime_profile_old, no
	.type	_dl_runtime_profile_old, @function

	/* We save the registers in a different order than desired by
	   .mask/.fmask, so we have to use explicit cfi directives.  */
	cfi_startproc

#undef FRAMESIZE
#define FRAMESIZE	50*8

	.align	4
_dl_runtime_profile_old:
	lda	$30, -FRAMESIZE($30)
	cfi_adjust_cfa_offset (FRAMESIZE)

	/* Preserve all argument registers.  This also constructs the
	   La_alpha_regs structure.  */
	savei	26, 0*8
	savei	16, 2*8
	savei	17, 3*8
	savei	18, 4*8
	savei	19, 5*8
	savei	20, 6*8
	savei	21, 7*8
	lda	$16, FRAMESIZE($30)
	savef	16, 8*8
	savef	17, 9*8
	savef	18, 10*8
	savef	19, 11*8
	savef	20, 12*8
	savef	21, 13*8
	stq	$16, 1*8($30)

	/* Preserve all registers that C normally doesn't.  */
	savei	0, 14*8
	savei	1, 15*8
	savei	2, 16*8
	savei	3, 17*8
	savei	4, 18*8
	savei	5, 19*8
	savei	6, 20*8
	savei	7, 21*8
	savei	8, 22*8
	savei	22, 23*8
	savei	23, 24*8
	savei	24, 25*8
	savei	25, 26*8
	savei	29, 27*8
	savef	0, 28*8
	savef	1, 29*8
	savef	10, 30*8
	savef	11, 31*8
	savef	12, 32*8
	savef	13, 33*8
	savef	14, 34*8
	savef	15, 35*8
	savef	22, 36*8
	savef	23, 37*8
	savef	24, 38*8
	savef	25, 39*8
	savef	26, 40*8
	savef	27, 41*8
	savef	28, 42*8
	savef	29, 43*8
	savef	30, 44*8

	/* Set up our GP.  */
	br	$29, .+4
	ldgp	$29, 0($29)

	/* Set up the arguments for _dl_profile_fixup:
	   $16 = link_map out of plt0
	   $17 = offset of reloc entry = ($28 - $27 - 20) /12 * 24
	   $18 = return address
	   $19 = La_alpha_regs address
	   $20 = framesize address
	*/
	subq	$28, $27, $17
	ldq	$16, 8($27)
	subq	$17, 20, $17
	mov	$26, $18
	addq	$17, $17, $17
	lda	$19, 0($30)
	lda	$20, 45*8($30)
	stq	$16, 48*8($30)
	stq	$17, 49*8($30)

	bsr	$26, _dl_profile_fixup	!samegp

	/* Discover if we're wrapping this call.  */
	ldq	$18, 45*8($30)
	bge	$18, 1f

	/* Move the destination address into position.  */
	mov	$0, $27
	/* Restore program registers.  */
	ldq	$26, 0*8($30)
	ldq	$16, 2*8($30)
	ldq	$17, 3*8($30)
	ldq	$18, 4*8($30)
	ldq	$19, 5*8($30)
	ldq	$20, 6*8($30)
	ldq	$21, 7*8($30)
	ldt	$f16, 8*8($30)
	ldt	$f17, 9*8($30)
	ldt	$f18, 10*8($30)
	ldt	$f19, 11*8($30)
	ldt	$f20, 12*8($30)
	ldt	$f21, 13*8($30)
	ldq	$0, 14*8($30)
	ldq	$1, 15*8($30)
	ldq	$2, 16*8($30)
	ldq	$3, 17*8($30)
	ldq	$4, 18*8($30)
	ldq	$5, 19*8($30)
	ldq	$6, 20*8($30)
	ldq	$7, 21*8($30)
	ldq	$8, 22*8($30)
	ldq	$22, 23*8($30)
	ldq	$23, 24*8($30)
	ldq	$24, 25*8($30)
	ldq	$25, 26*8($30)
	ldq	$29, 27*8($30)
	ldt	$f0, 28*8($30)
	ldt	$f1, 29*8($30)
	ldt	$f10, 30*8($30)
	ldt	$f11, 31*8($30)
	ldt	$f12, 32*8($30)
	ldt	$f13, 33*8($30)
	ldt	$f14, 34*8($30)
	ldt	$f15, 35*8($30)
	ldt	$f22, 36*8($30)
	ldt	$f23, 37*8($30)
	ldt	$f24, 38*8($30)
	ldt	$f25, 39*8($30)
	ldt	$f26, 40*8($30)
	ldt	$f27, 41*8($30)
	ldt	$f28, 42*8($30)
	ldt	$f29, 43*8($30)
	ldt	$f30, 44*8($30)

	/* Clean up and turn control to the destination.  */
	lda	$30, FRAMESIZE($30)
	jmp	$31, ($27)

1:
	/* Create a frame pointer and allocate a new argument frame.  */
	savei	15, 45*8
	mov	$30, $15
	cfi_def_cfa_register (15)
	addq	$18, 15, $18
	bic	$18, 15, $18
	subq	$30, $18, $30

	/* Save the call destination around memcpy.  */
	stq	$0, 46*8($30)

	/* Copy the stack arguments into place.  */
	lda	$16, 0($30)
	lda	$17, FRAMESIZE($15)
	jsr	$26, memcpy
	ldgp	$29, 0($26)

	/* Reload the argument registers.  */
	ldq	$27, 46*8($30)
	ldq	$16, 2*8($15)
	ldq	$17, 3*8($15)
	ldq	$18, 4*8($15)
	ldq	$19, 5*8($15)
	ldq	$20, 6*8($15)
	ldq	$21, 7*8($15)
	ldt	$f16, 8*8($15)
	ldt	$f17, 9*8($15)
	ldt	$f18, 10*8($15)
	ldt	$f19, 11*8($15)
	ldt	$f20, 12*8($15)
	ldt	$f21, 13*8($15)

	jsr	$26, ($27), 0
	ldgp	$29, 0($26)

	/* Set up for call to _dl_call_pltexit.  */
	ldq	$16, 48*8($15)
	ldq	$17, 49*8($15)
	stq	$0, 46*8($15)
	lda	$18, 0($15)
	stq	$1, 47*8($15)
	lda	$19, 46*8($15)
	stt	$f0, 48*8($15)
	stt	$f1, 49*8($15)
	bsr	$26, _dl_call_pltexit	!samegp

	mov	$15, $30
	cfi_def_cfa_register (30)
	ldq	$26, 0($30)
	ldq	$15, 45*8($30)
	lda	$30, FRAMESIZE($30)
	ret

	cfi_endproc
	.size	_dl_runtime_profile_old, .-_dl_runtime_profile_old
