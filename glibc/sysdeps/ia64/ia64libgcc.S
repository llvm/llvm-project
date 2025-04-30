/* From the Intel IA-64 Optimization Guide, choose the minimum latency
   alternative.  */

#include <sysdep.h>
#undef ret

#include <shlib-compat.h>

#if SHLIB_COMPAT(libc, GLIBC_2_2, GLIBC_2_2_6)

/* __divtf3
   Compute a 80-bit IEEE double-extended quotient.
   farg0 holds the dividend.  farg1 holds the divisor.  */

ENTRY(___divtf3)
	cmp.eq p7, p0 = r0, r0
	frcpa.s0 f10, p6 = farg0, farg1
	;;
(p6)	cmp.ne p7, p0 = r0, r0
	.pred.rel.mutex p6, p7
(p6)	fnma.s1 f11 = farg1, f10, f1
(p6)	fma.s1 f12 = farg0, f10, f0
	;;
(p6)	fma.s1 f13 = f11, f11, f0
(p6)	fma.s1 f14 = f11, f11, f11
	;;
(p6)	fma.s1 f11 = f13, f13, f11
(p6)	fma.s1 f13 = f14, f10, f10
	;;
(p6)	fma.s1 f10 = f13, f11, f10
(p6)	fnma.s1 f11 = farg1, f12, farg0
	;;
(p6)	fma.s1 f11 = f11, f10, f12
(p6)	fnma.s1 f12 = farg1, f10, f1
	;;
(p6)	fma.s1 f10 = f12, f10, f10
(p6)	fnma.s1 f12 = farg1, f11, farg0
	;;
(p6)	fma.s0 fret0 = f12, f10, f11
(p7)	mov fret0 = f10
	br.ret.sptk rp
END(___divtf3)
	.symver ___divtf3, __divtf3@GLIBC_2.2

/* __divdf3
   Compute a 64-bit IEEE double quotient.
   farg0 holds the dividend.  farg1 holds the divisor.  */

ENTRY(___divdf3)
	cmp.eq p7, p0 = r0, r0
	frcpa.s0 f10, p6 = farg0, farg1
	;;
(p6)	cmp.ne p7, p0 = r0, r0
	.pred.rel.mutex p6, p7
(p6)	fmpy.s1 f11 = farg0, f10
(p6)	fnma.s1 f12 = farg1, f10, f1
	;;
(p6)	fma.s1 f11 = f12, f11, f11
(p6)	fmpy.s1 f13 = f12, f12
	;;
(p6)	fma.s1 f10 = f12, f10, f10
(p6)	fma.s1 f11 = f13, f11, f11
	;;
(p6)	fmpy.s1 f12 = f13, f13
(p6)	fma.s1 f10 = f13, f10, f10
	;;
(p6)	fma.d.s1 f11 = f12, f11, f11
(p6)	fma.s1 f10 = f12, f10, f10
	;;
(p6)	fnma.d.s1 f8 = farg1, f11, farg0
	;;
(p6)	fma.d fret0 = f8, f10, f11
(p7)	mov fret0 = f10
	br.ret.sptk rp
	;;
END(___divdf3)
	.symver	___divdf3, __divdf3@GLIBC_2.2

/* __divsf3
   Compute a 32-bit IEEE float quotient.
   farg0 holds the dividend.  farg1 holds the divisor.  */

ENTRY(___divsf3)
	cmp.eq p7, p0 = r0, r0
	frcpa.s0 f10, p6 = farg0, farg1
	;;
(p6)	cmp.ne p7, p0 = r0, r0
	.pred.rel.mutex p6, p7
(p6)	fmpy.s1 f8 = farg0, f10
(p6)	fnma.s1 f9 = farg1, f10, f1
	;;
(p6)	fma.s1 f8 = f9, f8, f8
(p6)	fmpy.s1 f9 = f9, f9
	;;
(p6)	fma.s1 f8 = f9, f8, f8
(p6)	fmpy.s1 f9 = f9, f9
	;;
(p6)	fma.d.s1 f10 = f9, f8, f8
	;;
(p6)	fnorm.s.s0 fret0 = f10
(p7)	mov fret0 = f10
	br.ret.sptk rp
	;;
END(___divsf3)
	.symver	___divsf3, __divsf3@GLIBC_2.2

/* __divdi3
   Compute a 64-bit integer quotient.
   in0 holds the dividend.  in1 holds the divisor.  */

ENTRY(___divdi3)
	.regstk 2,0,0,0
	/* Transfer inputs to FP registers.  */
	setf.sig f8 = in0
	setf.sig f9 = in1
	;;
	/* Convert the inputs to FP, so that they won't be treated as
	   unsigned.  */
	fcvt.xf f8 = f8
	fcvt.xf f9 = f9
	;;
	/* Compute the reciprocal approximation.  */
	frcpa.s1 f10, p6 = f8, f9
	;;
	/* 3 Newton-Raphson iterations.  */
(p6)	fnma.s1 f11 = f9, f10, f1
(p6)	fmpy.s1 f12 = f8, f10
	;;
(p6)	fmpy.s1 f13 = f11, f11
(p6)	fma.s1 f12 = f11, f12, f12
	;;
(p6)	fma.s1 f10 = f11, f10, f10
(p6)	fma.s1 f11 = f13, f12, f12
	;;
(p6)	fma.s1 f10 = f13, f10, f10
(p6)	fnma.s1 f12 = f9, f11, f8
	;;
(p6)	fma.s1 f10 = f12, f10, f11
	;;
	/* Round quotient to an integer.  */
	fcvt.fx.trunc.s1 f10 = f10
	;;
	/* Transfer result to GP registers.  */
	getf.sig ret0 = f10
	br.ret.sptk rp
	;;
END(___divdi3)
	.symver	___divdi3, __divdi3@GLIBC_2.2

/* __moddi3
   Compute a 64-bit integer modulus.
   in0 holds the dividend (a).  in1 holds the divisor (b).  */

ENTRY(___moddi3)
	.regstk 2,0,0,0
	/* Transfer inputs to FP registers.  */
	setf.sig f14 = in0
	setf.sig f9 = in1
	;;
	/* Convert the inputs to FP, so that they won't be treated as
	   unsigned.  */
	fcvt.xf f8 = f14
	fcvt.xf f9 = f9
	;;
	/* Compute the reciprocal approximation.  */
	frcpa.s1 f10, p6 = f8, f9
	;;
	/* 3 Newton-Raphson iterations.  */
(p6)	fmpy.s1 f12 = f8, f10
(p6)	fnma.s1 f11 = f9, f10, f1
	;;
(p6)	fma.s1 f12 = f11, f12, f12
(p6)	fmpy.s1 f13 = f11, f11
	;;
(p6)	fma.s1 f10 = f11, f10, f10
(p6)	fma.s1 f11 = f13, f12, f12
	;;
	sub in1 = r0, in1
(p6)	fma.s1 f10 = f13, f10, f10
(p6)	fnma.s1 f12 = f9, f11, f8
	;;
	setf.sig f9 = in1
(p6)	fma.s1 f10 = f12, f10, f11
	;;
	fcvt.fx.trunc.s1 f10 = f10
	;;
	/* r = q * (-b) + a  */
	xma.l f10 = f10, f9, f14
	;;
	/* Transfer result to GP registers.  */
	getf.sig ret0 = f10
	br.ret.sptk rp
	;;
END(___moddi3)
	.symver ___moddi3, __moddi3@GLIBC_2.2

/* __udivdi3
   Compute a 64-bit unsigned integer quotient.
   in0 holds the dividend.  in1 holds the divisor.  */

ENTRY(___udivdi3)
	.regstk 2,0,0,0
	/* Transfer inputs to FP registers.  */
	setf.sig f8 = in0
	setf.sig f9 = in1
	;;
	/* Convert the inputs to FP, to avoid FP software-assist faults.  */
	fcvt.xuf.s1 f8 = f8
	fcvt.xuf.s1 f9 = f9
	;;
	/* Compute the reciprocal approximation.  */
	frcpa.s1 f10, p6 = f8, f9
	;;
	/* 3 Newton-Raphson iterations.  */
(p6)	fnma.s1 f11 = f9, f10, f1
(p6)	fmpy.s1 f12 = f8, f10
	;;
(p6)	fmpy.s1 f13 = f11, f11
(p6)	fma.s1 f12 = f11, f12, f12
	;;
(p6)	fma.s1 f10 = f11, f10, f10
(p6)	fma.s1 f11 = f13, f12, f12
	;;
(p6)	fma.s1 f10 = f13, f10, f10
(p6)	fnma.s1 f12 = f9, f11, f8
	;;
(p6)	fma.s1 f10 = f12, f10, f11
	;;
	/* Round quotient to an unsigned integer.  */
	fcvt.fxu.trunc.s1 f10 = f10
	;;
	/* Transfer result to GP registers.  */
	getf.sig ret0 = f10
	br.ret.sptk rp
	;;
END(___udivdi3)
	.symver	___udivdi3, __udivdi3@GLIBC_2.2

/* __umoddi3
   Compute a 64-bit unsigned integer modulus.
   in0 holds the dividend (a).  in1 holds the divisor (b).  */

ENTRY(___umoddi3)
	.regstk 2,0,0,0
	/* Transfer inputs to FP registers.  */
	setf.sig f14 = in0
	setf.sig f9 = in1
	;;
	/* Convert the inputs to FP, to avoid FP software assist faults.  */
	fcvt.xuf.s1 f8 = f14
	fcvt.xuf.s1 f9 = f9
	;;
	/* Compute the reciprocal approximation.  */
	frcpa.s1 f10, p6 = f8, f9
	;;
	/* 3 Newton-Raphson iterations.  */
(p6)	fmpy.s1 f12 = f8, f10
(p6)	fnma.s1 f11 = f9, f10, f1
	;;
(p6)	fma.s1 f12 = f11, f12, f12
(p6)	fmpy.s1 f13 = f11, f11
	;;
(p6)	fma.s1 f10 = f11, f10, f10
(p6)	fma.s1 f11 = f13, f12, f12
	;;
	sub in1 = r0, in1
(p6)	fma.s1 f10 = f13, f10, f10
(p6)	fnma.s1 f12 = f9, f11, f8
	;;
	setf.sig f9 = in1
(p6)	fma.s1 f10 = f12, f10, f11
	;;
	/* Round quotient to an unsigned integer.  */
	fcvt.fxu.trunc.s1 f10 = f10
	;;
	/* r = q * (-b) + a  */
	xma.l f10 = f10, f9, f14
	;;
	/* Transfer result to GP registers.  */
	getf.sig ret0 = f10
	br.ret.sptk rp
	;;
END(___umoddi3)
	.symver	___umoddi3, __umoddi3@GLIBC_2.2

/* __multi3
   Compute a 128-bit multiply of 128-bit multiplicands.
   in0/in1 holds one multiplicand (a), in2/in3 holds the other one (b).  */

ENTRY(___multi3)
	.regstk 4,0,0,0
	setf.sig f6 = in1
	movl r19 = 0xffffffff
	setf.sig f7 = in2
	;;
	and r14 = r19, in0
	;;
	setf.sig f10 = r14
	and r14 = r19, in2
	xmpy.l f9 = f6, f7
	;;
	setf.sig f6 = r14
	shr.u r14 = in0, 32
	;;
	setf.sig f7 = r14
	shr.u r14 = in2, 32
	;;
	setf.sig f8 = r14
	xmpy.l f11 = f10, f6
	xmpy.l f6 = f7, f6
	;;
	getf.sig r16 = f11
	xmpy.l f7 = f7, f8
	;;
	shr.u r14 = r16, 32
	and r16 = r19, r16
	getf.sig r17 = f6
	setf.sig f6 = in0
	;;
	setf.sig f11 = r14
	getf.sig r21 = f7
	setf.sig f7 = in3
	;;
	xma.l f11 = f10, f8, f11
	xma.l f6 = f6, f7, f9
	;;
	getf.sig r18 = f11
	;;
	add r18 = r18, r17
	;;
	and r15 = r19, r18
	cmp.ltu p7, p6 = r18, r17
	;;
	getf.sig r22 = f6
(p7)	adds r14 = 1, r19
	;;
(p7)	add r21 = r21, r14
	shr.u r14 = r18, 32
	shl r15 = r15, 32
	;;
	add r20 = r21, r14
	;;
	add ret0 = r15, r16
	add ret1 = r22, r20
	br.ret.sptk rp
	;;
END(___multi3)
	.symver	___multi3, __multi3@GLIBC_2.2

#endif
