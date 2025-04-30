/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 * \file
 * \brief directives.h - define macros for asm directives
 */

#if     ! defined (__ASSEMBLER__)
#define __ASSEMBLER__
#endif          /* ! defined (__ASSEMBLER__) */

#define	_ASM_CONCAT(l,r)	l##r
#define	ASM_CONCAT(l,r)		_ASM_CONCAT(l,r)
#define	_ASM_CONCAT3(l,m,r)	l##m##r
#define	ASM_CONCAT3(l,m,r)	_ASM_CONCAT3(l,m,r)
#define	_ASM_STRINGIFY(s)	#s
#define	ASM_STRINGIFY(s)	_ASM_STRINGIFY(s)

#if defined(TARGET_WIN_X8664)
#define ENT(n) n
#define ALN_WORD .align 4
#define ALN_FUNC .align 16
#define ALN_DBLE .align 8
#define ALN_QUAD .align 16
#if   	defined(__clang__)
/*
 * https://stackoverflow.com/questions/1317081/gccs-assembly-output-of-an-empty-program-on-x86-win32
 * Debugging info for function 's'
 * .scl 2: storage class external(2)
 * .type 32: symbol is a function
 */
#define ELF_FUNC(s) .def s; .scl 2; .type 32 ; .endef
#define ELF_OBJ(s)
#define ELF_SIZE(s)
#else
#define ELF_FUNC(s) .type ENT(s), @function
#define ELF_OBJ(s) .type ENT(s), @object
#define ELF_SIZE(s) .size ENT(s), .- ENT(s)
#endif
#define ELF_HIDDEN(s)
#define AS_VER .version "01.01"
#define I1 %rcx
#define I1W %ecx
#define I2 %rdx
#define I2W %edx
#define I3 %r8
#define I3W %r8d
#define I4 %r9
#define I4W %r9d
#define F1 %xmm0
#define F2 %xmm1
#define F3 %xmm2
#define F4 %xmm3

#elif defined(LINUX_ELF) || defined(TARGET_LINUX_X86) || defined(TARGET_LINUX_X8664)
/*
 * For X86-64 ELF enabled objects, disable stack execute bit.
 *
 * Assume that this file is one of the first include files listed in assembly
 * source files that need preprocessing.
 */
#if	! defined(NOTE_GNU_STACK)
# define	NOTE_GNU_STACK
	.section .note.GNU-stack,"",%progbits
	.text
#endif		// #if     ! defined(NOTE_GNU_STACK)

#define ENT(n) n
#define ALN_WORD .align 4
#define ALN_FUNC .align 16
#define ALN_DBLE .align 8
#define ALN_QUAD .align 16
#define ELF_FUNC(s) .type ENT(s), @function
#define ELF_OBJ(s) .type ENT(s), @object
#define ELF_SIZE(s) .size ENT(s), .- ENT(s)
#define ELF_HIDDEN(s)	.globl	s ;	.hidden	s
#define AS_VER .version "01.01"
#define I1 %rdi
#define I1W %edi
#define I2 %rsi
#define I2W %esi
#define I3 %rdx
#define I3W %edx
#define I4 %rcx
#define I4W %ecx
#define F1 %xmm0
#define F2 %xmm1
#define F3 %xmm2
#define F4 %xmm3

#elif defined(TARGET_OSX_X8664)
#define ENT(n) ASM_CONCAT(_,n)
#define ALN_WORD .align 2
#define ALN_FUNC .align 4
#define ALN_DBLE .align 3
#define ALN_QUAD .align 4
#define ELF_FUNC(s)
#define ELF_OBJ(s)
#define ELF_SIZE(s)
#define ELF_HIDDEN(s)
#define AS_VER
#define I1 %rdi
#define I1W %edi
#define I2 %rsi
#define I2W %esi
#define I3 %rdx
#define I3W %edx
#define I4 %rcx
#define I4W %ecx
#define F1 %xmm0
#define F2 %xmm1
#define F3 %xmm2
#define F4 %xmm3

#else
#error	X8664 TARGET platform not defined.
#error	TARGET must be one of TARGET_LINUX_X8664, TARGET_OSX_X8664, or TARGET_WIN_X8664.
#endif

/* macros for handling pic and non-pic code */

#if defined(PG_PIC) && ! defined (TARGET_OSX_X8664)
#define GBLTXT(fn) fn @PLT
#define LDL(var, tmp, lreg)                                                    \
  leaq var(% rip), tmp;                                                        \
  movl (tmp), lreg
#define STL(lreg, tmp, var)                                                    \
  leaq var(% rip), tmp;                                                        \
  movl lreg, (tmp)
#define LDQ(var, tmp, qreg)                                                    \
  leaq var(% rip), tmp;                                                        \
  movq (tmp), qreg
#define STQ(qreg, tmp, var)                                                    \
  leaq var(% rip), tmp;                                                        \
  movq qreg, (tmp)
#define LDDQU(var, tmp, qreg)                                                  \
  leaq var(% rip), tmp;                                                        \
  movdqu (tmp), qreg
#define STDQU(qreg, tmp, var)                                                  \
  leaq var(% rip), tmp;                                                        \
  movdqu qreg, (tmp)
#define LEAQ(var, tmp) leaq var(% rip), tmp
#define XCHGL(lreg, tmp, var)                                                  \
  leaq var(% rip), tmp;                                                        \
  xchgl lreg, (tmp)
#define FNSTCW(tmp, var)                                                       \
  leaq var(% rip), tmp;                                                        \
  fnstcw (tmp)
#define FLDCW(var, tmp)                                                        \
  leaq var(% rip), tmp;                                                        \
  fldcw (tmp)
#else
#define GBLTXT(fn) fn
#define LDL(var, tmp, lreg) movl var(% rip), lreg
#define STL(lreg, tmp, var) movl lreg, var(% rip)
#define LDQ(var, tmp, qreg) movq var(% rip), qreg
#define STQ(qreg, tmp, var) movq qreg, var(% rip)
#define LDDQU(var, tmp, qreg) movdqu var(% rip), qreg
#define STDQU(qreg, tmp, var) movdqu qreg, var(% rip)
#define LEAQ(var, tmp) leaq var(% rip), tmp
#define XCHGL(lreg, tmp, var) xchgl lreg, var(% rip)
#define FNSTCW(tmp, var) fnstcw var(% rip)
#define FLDCW(var, tmp) fldcw var(% rip)
#endif

#define CALL(fn) call GBLTXT(fn)
#define JMP(fn) jmp GBLTXT(fn)

/* macros for handling the red zone of a stack, i.e., the 128-byte area
 * below a leaf function's stack pointer as defined in the linux abi.
 * For other enviroments, the red zone must be allocated.
 * Macros:
 *    RZ_PUSH   -- create the 128-byte redzone
 *    RZ_POP    -- relinquish the 128-byte redzone
 *    RZ_OFF(n) -- produce the offset representing n (positive) bytes below the
 *                 stackpointer wrt the linux abi; e.g., for the linux abi
 *                 RZ_OFF(24)(%rsp) => -24(%rsp); for other systems, the
 *                 offset would be computed as 128-24.
 */

#if defined(TARGET_WIN_X8664)
#define RZ_PUSH subq $128, % rsp
#define RZ_POP addq $128, % rsp
#define RZ_OFF(n) (128## - n)
#else
#define RZ_PUSH
#define RZ_POP
#define RZ_OFF(n) -##n
#endif

#if defined(GH_TARGET)
#define LDDP movsd
#else
#define LDDP movlpd
#endif

#ifdef	LBL__
#error	Macro "LBL__" should not be defined
#endif
#define	LBL__(_n,_t) _n ## _ ## _t
#ifdef	LBL_
#error	Macro "LBL_" should not be defined
#endif
#define	LBL_(_n, _m) LBL__(_n, _m)
#ifdef	LBL
#error	Macro "LBL" should not be defined
#endif
#define	LBL(_n) LBL_(_n, NNN)


/*
 *	Define a set of generic vex prefixed floating point instructions
 *	(macros) that can be used for both Intel 3 operand and AMD's 4
 *	operand instruction sets.
 *
 *	Conform to Intel's instruction notation.
 *
 *	Instructions (macros) use the C preprocessor and not the ".macro"
 *	facility builtin with most assemblers because of the inconsistent
 *	syntax and availability of pseudo ops across various platforms.
 *
 *	Selection between FMA3 and FMA4(AMD only) instructions is controlled
 *	by macro "VFMA_IS_FMA3_OR_FMA4" which must be defined to be either
 *	"FMA3" or "FMA4".
 *
 *	Map:
 *	Instruction			  "Pseudo OP" (CPP macro)
 *	vfmadd{132,213,231}{ss,sd,ps,pd}  VFMA_{132,213,231}{SS,SD,PS,PD}
 *	vfnmadd{132,213,231}{ss,sd,ps,pd} VNFMA_{132,213,231}{SS,SD,PS,PD}
 *	vfmsub{132,213,231}{ss,sd,ps,pd}  VFMS_{132,213,231}{SS,SD,PS,PD}
 *	vfnmsub{132,213,231}{ss,sd,ps,pd} VNFMS_{132,213,231}{SS,SD,PS,PD}
 */

#define	VFMA_FMA4_ORDER_132(src_rm,src_r,dst)	src_r,src_rm,dst,dst
#define	VFMA_FMA4_ORDER_213(src_rm,src_r,dst)	src_rm,src_r,dst,dst
#define	VFMA_FMA4_ORDER_231(src_rm,src_r,dst)	dst,src_rm,src_r,dst

#define	IF_VFMA_IS_FMA3(op,order,size,src_rm,src_r,dst) \
	ASM_CONCAT3(op,order,size)	src_rm,src_r,dst
	
#define	IF_VFMA_IS_FMA4(op,order,size,src_rm,src_r,dst) \
	ASM_CONCAT(op,size)	ASM_CONCAT(VFMA_FMA4_ORDER_,order)(src_rm,src_r,dst)

#define	VFMA_FMA3_OR_FMA4(op,order,size,src_rm,src_r,dst) \
	ASM_CONCAT(IF_VFMA_IS_,VFMA_IS_FMA3_OR_FMA4)(op,order,size,src_rm,src_r,dst)

/*
 * A couple of true/false (1/0) macros to use to determine whether FMA3 or
 * FMA4 is being targetted.
 *
 * Note: Macro "VFMA_IS_FMA3_OR_FMA4" must be defined.
 */

#define	VFMA_IS_FMA3_(l,r)	l
#define	VFMA_IS_FMA4_(l,r)	r
#define	VFMA_IS_FMA3	ASM_CONCAT3(VFMA_IS_,VFMA_IS_FMA3_OR_FMA4,_)(1,0)
#define	VFMA_IS_FMA4	ASM_CONCAT3(VFMA_IS_,VFMA_IS_FMA3_OR_FMA4,_)(0,1)


/*
 *	VEX floating multiply add.
 */

#define	VFMA_132SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,132,ss,src_rm,src_r,dst)
#define	VFMA_132SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,132,sd,src_rm,src_r,dst)
#define	VFMA_132PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,132,ps,src_rm,src_r,dst)
#define	VFMA_132PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,132,pd,src_rm,src_r,dst)

#define	VFMA_213SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,213,ss,src_rm,src_r,dst)
#define	VFMA_213SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,213,sd,src_rm,src_r,dst)
#define	VFMA_213PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,213,ps,src_rm,src_r,dst)
#define	VFMA_213PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,213,pd,src_rm,src_r,dst)

#define	VFMA_231SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,231,ss,src_rm,src_r,dst)
#define	VFMA_231SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,231,sd,src_rm,src_r,dst)
#define	VFMA_231PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,231,ps,src_rm,src_r,dst)
#define	VFMA_231PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmadd,231,pd,src_rm,src_r,dst)
/*
 *	VEX floating -multiply add.
 */

#define	VFNMA_132SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,132,ss,src_rm,src_r,dst)
#define	VFNMA_132SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,132,sd,src_rm,src_r,dst)
#define	VFNMA_132PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,132,ps,src_rm,src_r,dst)
#define	VFNMA_132PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,132,pd,src_rm,src_r,dst)

#define	VFNMA_213SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,213,ss,src_rm,src_r,dst)
#define	VFNMA_213SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,213,sd,src_rm,src_r,dst)
#define	VFNMA_213PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,213,ps,src_rm,src_r,dst)
#define	VFNMA_213PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,213,pd,src_rm,src_r,dst)

#define	VFNMA_231SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,231,ss,src_rm,src_r,dst)
#define	VFNMA_231SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,231,sd,src_rm,src_r,dst)
#define	VFNMA_231PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,231,ps,src_rm,src_r,dst)
#define	VFNMA_231PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmadd,231,pd,src_rm,src_r,dst)

/*
 *	VEX floating multiply subtract.
 */

#define	VFMS_132SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,132,ss,src_rm,src_r,dst)
#define	VFMS_132SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,132,sd,src_rm,src_r,dst)
#define	VFMS_132PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,132,ps,src_rm,src_r,dst)
#define	VFMS_132PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,132,pd,src_rm,src_r,dst)

#define	VFMS_213SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,213,ss,src_rm,src_r,dst)
#define	VFMS_213SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,213,sd,src_rm,src_r,dst)
#define	VFMS_213PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,213,ps,src_rm,src_r,dst)
#define	VFMS_213PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,213,pd,src_rm,src_r,dst)

#define	VFMS_231SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,231,ss,src_rm,src_r,dst)
#define	VFMS_231SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,231,sd,src_rm,src_r,dst)
#define	VFMS_231PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,231,ps,src_rm,src_r,dst)
#define	VFMS_231PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfmsub,231,pd,src_rm,src_r,dst)

/*
 *	VEX floating -multiply subtract.
 */

#define	VFNMS_132SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,132,ss,src_rm,src_r,dst)
#define	VFNMS_132SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,132,sd,src_rm,src_r,dst)
#define	VFNMS_132PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,132,ps,src_rm,src_r,dst)
#define	VFNMS_132PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,132,pd,src_rm,src_r,dst)

#define	VFNMS_213SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,213,ss,src_rm,src_r,dst)
#define	VFNMS_213SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,213,sd,src_rm,src_r,dst)
#define	VFNMS_213PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,213,ps,src_rm,src_r,dst)
#define	VFNMS_213PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,213,pd,src_rm,src_r,dst)

#define	VFNMS_231SS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,231,ss,src_rm,src_r,dst)
#define	VFNMS_231SD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,231,sd,src_rm,src_r,dst)
#define	VFNMS_231PS(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,231,ps,src_rm,src_r,dst)
#define	VFNMS_231PD(src_rm,src_r,dst) \
	VFMA_FMA3_OR_FMA4(vfnmsub,231,pd,src_rm,src_r,dst)
