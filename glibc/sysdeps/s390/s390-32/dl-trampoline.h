/* PLT trampolines.  s390 version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* This code is used in dl-runtime.c to call the `fixup' function
   and then redirect to the address it returns.  */

/* The PLT stubs will call _dl_runtime_resolve/_dl_runtime_profile
 * with the following linkage:
 *   r2 - r6 : parameter registers
 *   f0, f2 : floating point parameter registers
 *   v24, v26, v28, v30, v25, v27, v29, v31 : vector parameter registers
 *   24(r15), 28(r15) : PLT arguments PLT1, PLT2
 *   96(r15) : additional stack parameters
 * The slightly tightened normal clobber rules for function calls apply:
 *   r0 : call saved (for __fentry__)
 *   r1 - r5 : call clobbered
 *   r6 - r13 :	call saved
 *   r14 : return address (call clobbered)
 *   r15 : stack pointer (call saved)
 *   f4, f6 : call saved
 *   f0 - f3, f5, f7 - f15 : call clobbered
 *   v0 - v3, v5, v7 - v15 : bytes 0-7 overlap with fprs: call clobbered
               bytes 8-15: call clobbered
 *   v4, v6 : bytes 0-7 overlap with f4, f6: call saved
              bytes 8-15: call clobbered
 *   v16 - v31 : call clobbered
 */

#define CFA_OFF 96
#define FRAME_OFF CFA_OFF + FRAME_SIZE
#define V24_OFF -224
#define V25_OFF -208
#define V26_OFF -192
#define V27_OFF -176
#define V28_OFF -160
#define V29_OFF -144
#define V30_OFF -128
#define V31_OFF -112
#define R0_OFF -76
#define PLT1_OFF -72
#define PLT2_OFF -68
#define R2_OFF -64
#define R3_OFF -60
#define R4_OFF -56
#define R5_OFF -52
#define R14_OFF -48
#define R15_OFF -44
#define F0_OFF -40
#define F2_OFF -32
	.globl _dl_runtime_resolve
	.type _dl_runtime_resolve, @function
	cfi_startproc
	.align 16
_dl_runtime_resolve:
	st     %r0,CFA_OFF+R0_OFF(%r15)
	cfi_offset (r0, R0_OFF)
	stm    %r2,%r5,CFA_OFF+R2_OFF(%r15) # save registers
	cfi_offset (r2, R2_OFF)
	cfi_offset (r3, R3_OFF)
	cfi_offset (r4, R4_OFF)
	cfi_offset (r5, R5_OFF)
	stm    %r14,%r15,CFA_OFF+R14_OFF(%r15)
	cfi_offset (r14, R14_OFF)
	cfi_offset (r15, R15_OFF)
	std    %f0,CFA_OFF+F0_OFF(%r15)
	cfi_offset (f0, F0_OFF)
	std    %f2,CFA_OFF+F2_OFF(%r15)
	cfi_offset (f2, F2_OFF)
	lr     %r0,%r15
	lm     %r2,%r3,CFA_OFF+PLT1_OFF(%r15) # load args saved by PLT
#ifdef RESTORE_VRS
# define FRAME_SIZE (CFA_OFF + 128)
	ahi    %r15,-FRAME_SIZE # create stack frame
	cfi_adjust_cfa_offset (FRAME_SIZE)
	.machine push
	.machine "z13"
	.machinemode "zarch_nohighgprs"
	vstm   %v24,%v31,FRAME_OFF+V24_OFF(%r15) # save call-clobbered vr args
	cfi_offset (v24, V24_OFF)
	cfi_offset (v25, V25_OFF)
	cfi_offset (v26, V26_OFF)
	cfi_offset (v27, V27_OFF)
	cfi_offset (v28, V28_OFF)
	cfi_offset (v29, V29_OFF)
	cfi_offset (v30, V30_OFF)
	cfi_offset (v31, V31_OFF)
	.machine pop
#else
# define FRAME_SIZE CFA_OFF
	ahi    %r15,-FRAME_SIZE # create stack frame
	cfi_adjust_cfa_offset (FRAME_SIZE)
#endif
	st     %r0,0(%r15)		# write backchain
	basr   %r1,0
0:	l      %r14,1f-0b(%r1)
	bas    %r14,0(%r14,%r1)		# call _dl_fixup
	lr     %r1,%r2			# function addr returned in r2
#ifdef RESTORE_VRS
	.machine push
	.machine "z13"
	.machinemode "zarch_nohighgprs"
	vlm    %v24,%v31,FRAME_OFF+V24_OFF(%r15) # restore vector registers
	.machine pop
#endif
	lm     %r14,%r15,FRAME_OFF+R14_OFF(%r15) # restore frame and registers
#undef FRAME_SIZE
	cfi_def_cfa_offset (CFA_OFF)
	ld     %f0,CFA_OFF+F0_OFF(%r15)
	ld     %f2,CFA_OFF+F2_OFF(%r15)
	lm     %r2,%r5,CFA_OFF+R2_OFF(%r15)
	l      %r0,CFA_OFF+R0_OFF(%r15)
	br     %r1
1:	.long  _dl_fixup - 0b
	cfi_endproc
	.size _dl_runtime_resolve, .-_dl_runtime_resolve
#undef V24_OFF
#undef V25_OFF
#undef V26_OFF
#undef V27_OFF
#undef V28_OFF
#undef V29_OFF
#undef V30_OFF
#undef V31_OFF
#undef R0_OFF
#undef PLT1_OFF
#undef PLT2_OFF
#undef R2_OFF
#undef R3_OFF
#undef R4_OFF
#undef R5_OFF
#undef R14_OFF
#undef R15_OFF
#undef F0_OFF
#undef F2_OFF

#ifndef PROF
# define SIZEOF_STRUCT_LA_S390_32_REGS 168
# define REGS_OFF -264
# define R2_OFF -264
# define R3_OFF -260
# define R4_OFF -256
# define R5_OFF -252
# define R6_OFF -248
# define F0_OFF -240
# define F2_OFF -232
# define V24_OFF -224
# define V25_OFF -208
# define V26_OFF -192
# define V27_OFF -176
# define V28_OFF -160
# define V29_OFF -144
# define V30_OFF -128
# define V31_OFF -112
# define R0_OFF -88
# define R12_OFF -84
# define R14_OFF -80
# define FRAMESIZE_OFF -76
# define PLT1_OFF -72
# define PLT2_OFF -68
# define PREGS_OFF -64
# define RETVAL_OFF -56
# define RET_R2_OFF -56
# define RET_R3_OFF -52
# define RET_F0_OFF -48
# define RET_V24_OFF -40
	.globl _dl_runtime_profile
	.type _dl_runtime_profile, @function
	cfi_startproc
	.align 16
_dl_runtime_profile:
	st     %r0,CFA_OFF+R0_OFF(%r15)
	cfi_offset (r0, R0_OFF)
	st     %r12,CFA_OFF+R12_OFF(%r15)	# r12 is used as backup of r15
	cfi_offset (r12, R12_OFF)
	st     %r14,CFA_OFF+R14_OFF(%r15)
	cfi_offset (r14, R14_OFF)
	lr     %r12,%r15			# backup stack pointer
	cfi_def_cfa_register (12)
# define FRAME_SIZE (CFA_OFF + SIZEOF_STRUCT_LA_S390_32_REGS)
	ahi    %r15,-FRAME_SIZE			# create stack frame:
	st     %r12,0(%r15)			# save backchain

	stm    %r2,%r6,FRAME_OFF+R2_OFF(%r15)	# save registers
	cfi_offset (r2, R2_OFF)			# + r6 needed as arg for
	cfi_offset (r3, R3_OFF)			#  _dl_profile_fixup
	cfi_offset (r4, R4_OFF)
	cfi_offset (r5, R5_OFF)
	cfi_offset (r6, R6_OFF)
	std    %f0,FRAME_OFF+F0_OFF(%r15)
	cfi_offset (f0, F0_OFF)
	std    %f2,FRAME_OFF+F2_OFF(%r15)
	cfi_offset (f2, F2_OFF)
# ifdef RESTORE_VRS
	.machine push
	.machine "z13"
	.machinemode "zarch_nohighgprs"
	vstm   %v24,%v31,FRAME_OFF+V24_OFF(%r15)	# store call-clobbered
	cfi_offset (v24, V24_OFF)			# vr arguments
	cfi_offset (v25, V25_OFF)
	cfi_offset (v26, V26_OFF)
	cfi_offset (v27, V27_OFF)
	cfi_offset (v28, V28_OFF)
	cfi_offset (v29, V29_OFF)
	cfi_offset (v30, V30_OFF)
	cfi_offset (v31, V31_OFF)
	.machine pop
# endif

	lm     %r2,%r3,CFA_OFF+PLT1_OFF(%r12)	# load arguments saved by PLT
	lr     %r4,%r14				# return address as third parm
	basr   %r1,0
0:	l      %r14,6f-0b(%r1)
	la     %r5,FRAME_OFF+REGS_OFF(%r15)	# struct La_s390_32_regs *
	la     %r6,CFA_OFF+FRAMESIZE_OFF(%r12)	# long int * framesize
	bas    %r14,0(%r14,%r1)			# call resolver
	lr     %r1,%r2				# function addr returned in r2
	ld     %f0,FRAME_OFF+F0_OFF(%r15)	# restore call-clobbered
	ld     %f2,FRAME_OFF+F2_OFF(%r15)	# arg fprs
# ifdef RESTORE_VRS
	.machine push
	.machine "z13"
	.machinemode "zarch_nohighgprs"		# restore call-clobbered
	vlm    %v24,%v31,FRAME_OFF+V24_OFF(%r15)# arg vrs
	.machine pop
# endif
	icm    %r0,15,CFA_OFF+FRAMESIZE_OFF(%r12)	# load & test framesize
	jnm    2f
						# framesize < 0 means no
	lm     %r2,%r6,FRAME_OFF+R2_OFF(%r15)	# pltexit call, so we can do a
						# tail call without
						# copying the arg overflow area
	lr     %r15,%r12			# remove stack frame
	cfi_def_cfa_register (15)
	l      %r14,CFA_OFF+R14_OFF(%r15)	# restore registers
	l      %r12,CFA_OFF+R12_OFF(%r15)
	l      %r0,CFA_OFF+R0_OFF(%r15)
	br     %r1				# tail call

	cfi_def_cfa_register (12)
2:	la     %r4,FRAME_OFF+REGS_OFF(%r15)	# struct La_s390_32_regs *
	st     %r4,CFA_OFF+PREGS_OFF(%r12)
	jz     4f				# framesize == 0 ?
	ahi    %r0,7				# align framesize to 8
	lhi    %r2,-8
	nr     %r0,%r2
	slr    %r15,%r0				# make room for framesize bytes
	st     %r12,0(%r15)			# save backchain
	la     %r2,FRAME_OFF+REGS_OFF(%r15)
	la     %r3,CFA_OFF(%r12)
	srl    %r0,3
3:	mvc    0(8,%r2),0(%r3)			# copy additional parameters
	la     %r2,8(%r2)
	la     %r3,8(%r3)
	brct   %r0,3b
4:	lm     %r2,%r6,0(%r4)			# load register parameters
	basr   %r14,%r1				# call resolved function
	stm    %r2,%r3,CFA_OFF+RET_R2_OFF(%r12)	# store return vals r2, r3, f0
	std    %f0,CFA_OFF+RET_F0_OFF(%r12)	# to struct La_s390_32_retval
# ifdef RESTORE_VRS
	.machine push
	.machine "z13"
	vst    %v24,CFA_OFF+RET_V24_OFF(%r12)	# store return value v24
	.machine pop
# endif
	lm     %r2,%r4,CFA_OFF+PLT1_OFF(%r12)	# r2, r3: args saved by PLT
						# r4: struct La_s390_32_regs *
	basr   %r1,0
5:	l      %r14,7f-5b(%r1)
	la     %r5,CFA_OFF+RETVAL_OFF(%r12)	# struct La_s390_32_retval *
	bas    %r14,0(%r14,%r1)			# call _dl_call_pltexit

	lr     %r15,%r12			# remove stack frame
# undef FRAME_SIZE
	cfi_def_cfa_register (15)
	l      %r14,CFA_OFF+R14_OFF(%r15)	# restore registers
	l      %r12,CFA_OFF+R12_OFF(%r15)
	l      %r0,CFA_OFF+R0_OFF(%r15)
	lm     %r2,%r3,CFA_OFF+RET_R2_OFF(%r15)	# restore return values
	ld     %f0,CFA_OFF+RET_F0_OFF(%r15)
# ifdef RESTORE_VRS
	.machine push
	.machine "z13"
	vl    %v24,CFA_OFF+RET_V24_OFF(%r15)	# restore return value v24
	.machine pop
# endif
	br     %r14

6:	.long  _dl_profile_fixup - 0b
7:	.long  _dl_call_pltexit - 5b
	cfi_endproc
	.size _dl_runtime_profile, .-_dl_runtime_profile
# undef SIZEOF_STRUCT_LA_S390_32_REGS
# undef REGS_OFF
# undef R2_OFF
# undef R3_OFF
# undef R4_OFF
# undef R5_OFF
# undef R6_OFF
# undef F0_OFF
# undef F2_OFF
# undef V24_OFF
# undef V25_OFF
# undef V26_OFF
# undef V27_OFF
# undef V28_OFF
# undef V29_OFF
# undef V30_OFF
# undef V31_OFF
# undef R0_OFF
# undef R12_OFF
# undef R14_OFF
# undef FRAMESIZE_OFF
# undef PLT1_OFF
# undef PLT2_OFF
# undef PREGS_OFF
# undef RETVAL_OFF
# undef RET_R2_OFF
# undef RET_R3_OFF
# undef RET_F0_OFF
# undef RET_V24_OFF
#endif
