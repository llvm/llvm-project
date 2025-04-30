/* Function powf vectorized with SSE4.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include "svml_s_powf_data.h"

	.text
ENTRY (_ZGVbN4vv_powf_sse4)
/*
   ALGORITHM DESCRIPTION:

     We are using the next identity: pow(x,y) = 2^(y * log2(x)).

     1) log2(x) calculation
        Here we use the following formula.
        Let |x|=2^k1*X1, where k1 is integer, 1<=X1<2.
        Let C ~= 1/ln(2),
        Rcp1 ~= 1/X1,   X2=Rcp1*X1,
        Rcp2 ~= 1/X2,   X3=Rcp2*X2,
        Rcp3 ~= 1/X3,   Rcp3C ~= C/X3.
        Then
          log2|x| = k1 + log2(1/Rcp1) + log2(1/Rcp2) + log2(C/Rcp3C) +
                    log2(X1*Rcp1*Rcp2*Rcp3C/C),
        where X1*Rcp1*Rcp2*Rcp3C = C*(1+q), q is very small.

        The values of Rcp1, log2(1/Rcp1), Rcp2, log2(1/Rcp2),
        Rcp3C, log2(C/Rcp3C) are taken from tables.
        Values of Rcp1, Rcp2, Rcp3C are such that RcpC=Rcp1*Rcp2*Rcp3C
        is exactly represented in target precision.

        log2(X1*Rcp1*Rcp2*Rcp3C/C) = log2(1+q) = ln(1+q)/ln2 =
             = 1/(ln2)*q - 1/(2ln2)*q^2 + 1/(3ln2)*q^3 - ... =
             = 1/(C*ln2)*cq - 1/(2*C^2*ln2)*cq^2 + 1/(3*C^3*ln2)*cq^3 - ... =
             = (1 + a1)*cq + a2*cq^2 + a3*cq^3 + ...,
        where
             cq=X1*Rcp1*Rcp2*Rcp3C-C,
             a1=1/(C*ln(2))-1 is small,
             a2=1/(2*C^2*ln2),
             a3=1/(3*C^3*ln2),
                  ...
        Log2 result is split by three parts: HH+HL+HLL

     2) Calculation of y*log2(x)
        Split y into YHi+YLo.
        Get high PH and medium PL parts of y*log2|x|.
        Get low PLL part of y*log2|x|.
        Now we have PH+PL+PLL ~= y*log2|x|.

     3) Calculation of 2^(y*log2(x))
        Let's represent PH+PL+PLL in the form N + j/2^expK + Z,
        where expK=7 in this implementation, N and j are integers,
        0<=j<=2^expK-1, |Z|<2^(-expK-1). Hence
        2^(PH+PL+PLL) ~= 2^N * 2^(j/2^expK) * 2^Z,
        where 2^(j/2^expK) is stored in a table, and
        2^Z ~= 1 + B1*Z + B2*Z^2 ... + B5*Z^5.
        We compute 2^(PH+PL+PLL) as follows:
        Break PH into PHH + PHL, where PHH = N + j/2^expK.
        Z = PHL + PL + PLL
        Exp2Poly = B1*Z + B2*Z^2 ... + B5*Z^5
        Get 2^(j/2^expK) from table in the form THI+TLO.
        Now we have 2^(PH+PL+PLL) ~= 2^N * (THI + TLO) * (1 + Exp2Poly).
        Get significand of 2^(PH+PL+PLL) in the form ResHi+ResLo:
        ResHi := THI
        ResLo := THI * Exp2Poly + TLO
        Get exponent ERes of the result:
        Res := ResHi + ResLo:
        Result := ex(Res) + N.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $256, %rsp
        movaps    %xmm0, %xmm3
        movhlps   %xmm0, %xmm3
        movaps    %xmm1, %xmm5
        movups    %xmm8, 112(%rsp)
        movaps    %xmm5, %xmm2
        cvtps2pd  %xmm3, %xmm8
        cvtps2pd  %xmm5, %xmm7
        movups    %xmm9, 96(%rsp)
        movaps    %xmm0, %xmm4
        cvtps2pd  %xmm0, %xmm9
        movq      __svml_spow_data@GOTPCREL(%rip), %rdx
        movups    %xmm10, 176(%rsp)
        movups    %xmm13, 48(%rsp)
        movups _ExpMask(%rdx), %xmm6

/* preserve mantissa, set input exponent to 2^(-10) */
        movaps    %xmm6, %xmm10
        andps     %xmm8, %xmm6
        andps     %xmm9, %xmm10

/* exponent bits selection */
        psrlq     $20, %xmm9
        orps _Two10(%rdx), %xmm6
        psrlq     $20, %xmm8
        orps _Two10(%rdx), %xmm10

/* reciprocal approximation good to at least 11 bits */
        cvtpd2ps  %xmm6, %xmm13
        cvtpd2ps  %xmm10, %xmm1
        movlhps   %xmm13, %xmm13
        movhlps   %xmm5, %xmm2
        movlhps   %xmm1, %xmm1
        movups    %xmm12, 208(%rsp)
        rcpps     %xmm13, %xmm12
        movups    %xmm11, 80(%rsp)
        cvtps2pd  %xmm2, %xmm11
        rcpps     %xmm1, %xmm2
        movups    %xmm14, 144(%rsp)
        cvtps2pd  %xmm12, %xmm14
        movups    %xmm15, 160(%rsp)
        cvtps2pd  %xmm2, %xmm15
        shufps    $221, %xmm8, %xmm9

/* round reciprocal to nearest integer, will have 1+9 mantissa bits */
        roundpd   $0, %xmm14, %xmm14

/* biased exponent in DP format */
        pshufd    $238, %xmm9, %xmm8
        roundpd   $0, %xmm15, %xmm15
        cvtdq2pd  %xmm8, %xmm1
        mulpd     %xmm15, %xmm10
        mulpd     %xmm14, %xmm6
        cvtdq2pd  %xmm9, %xmm2
        subpd _One(%rdx), %xmm10
        subpd _One(%rdx), %xmm6

/* table lookup */
        movaps    %xmm14, %xmm8
        movaps    %xmm15, %xmm9
        psrlq     $40, %xmm8
        psrlq     $40, %xmm9
        movd      %xmm8, %r8d
        movd      %xmm9, %eax
        psubd _NMINNORM(%rdx), %xmm4
        movdqu _ABSMASK(%rdx), %xmm3
        pextrd    $2, %xmm8, %r9d
        pand      %xmm5, %xmm3
        movups _Threshold(%rdx), %xmm8
        pextrd    $2, %xmm9, %ecx
        movaps    %xmm8, %xmm9
        cmpltpd   %xmm15, %xmm9
        cmpltpd   %xmm14, %xmm8
        andps _Bias(%rdx), %xmm9
        movaps    %xmm10, %xmm14
        andps _Bias(%rdx), %xmm8
        movaps    %xmm6, %xmm15
        orps _Bias1(%rdx), %xmm9
        orps _Bias1(%rdx), %xmm8
        subpd     %xmm9, %xmm2
        subpd     %xmm8, %xmm1
        mulpd     %xmm10, %xmm14
        mulpd     %xmm6, %xmm15
        mulpd _L2(%rdx), %xmm2
        mulpd _L2(%rdx), %xmm1
        movups _poly_coeff_3(%rdx), %xmm9
        movaps    %xmm9, %xmm8
        mulpd     %xmm10, %xmm8
        mulpd     %xmm6, %xmm9
        addpd _poly_coeff_4(%rdx), %xmm8
        addpd _poly_coeff_4(%rdx), %xmm9
        mulpd     %xmm14, %xmm8
        mulpd     %xmm15, %xmm9

/* reconstruction */
        addpd     %xmm8, %xmm10
        addpd     %xmm9, %xmm6
        movslq    %eax, %rax
        movslq    %r8d, %r8
        movslq    %ecx, %rcx
        movslq    %r9d, %r9
        movsd     _Log2Rcp_lookup(%rdx,%rax), %xmm13
        movsd     _Log2Rcp_lookup(%rdx,%r8), %xmm12
        movhpd    _Log2Rcp_lookup(%rdx,%rcx), %xmm13
        movhpd    _Log2Rcp_lookup(%rdx,%r9), %xmm12
        addpd     %xmm10, %xmm13
        addpd     %xmm6, %xmm12
        addpd     %xmm13, %xmm2
        addpd     %xmm12, %xmm1
        mulpd     %xmm7, %xmm2
        mulpd     %xmm11, %xmm1
        movups __dbInvLn2(%rdx), %xmm11
        movdqa    %xmm4, %xmm12
        movaps    %xmm11, %xmm10
        mulpd     %xmm2, %xmm10
        mulpd     %xmm1, %xmm11

/* to round down; if dR is an integer we will get R = 1, which is ok */
        movaps    %xmm10, %xmm8
        movaps    %xmm11, %xmm9
        subpd __dbHALF(%rdx), %xmm8
        subpd __dbHALF(%rdx), %xmm9
        addpd __dbShifter(%rdx), %xmm8
        addpd __dbShifter(%rdx), %xmm9
        movaps    %xmm8, %xmm6
        movaps    %xmm9, %xmm7
        subpd __dbShifter(%rdx), %xmm6
        subpd __dbShifter(%rdx), %xmm7

/* [0..1) */
        subpd     %xmm6, %xmm10
        subpd     %xmm7, %xmm11
        mulpd __dbC1(%rdx), %xmm10
        mulpd __dbC1(%rdx), %xmm11

/* hi bits */
        shufps    $221, %xmm1, %xmm2
        movdqu _NMAXVAL(%rdx), %xmm1
        pcmpgtd   %xmm1, %xmm12
        pcmpeqd   %xmm1, %xmm4
        por       %xmm4, %xmm12
        movdqa    %xmm3, %xmm1
        movdqu _INF(%rdx), %xmm4
        pcmpgtd   %xmm4, %xmm1
        pcmpeqd   %xmm4, %xmm3

/* iAbsX = iAbsX&iAbsMask */
        pand __iAbsMask(%rdx), %xmm2
        por       %xmm3, %xmm1

/* iRangeMask = (iAbsX>iDomainRange) */
        pcmpgtd __iDomainRange(%rdx), %xmm2
        por       %xmm1, %xmm12
        movups __lbLOWKBITS(%rdx), %xmm3
        por       %xmm2, %xmm12

/* low K bits */
        movaps    %xmm3, %xmm2
        andps     %xmm9, %xmm3
        andps     %xmm8, %xmm2
        psrlq     $11, %xmm8

/* dpP= _dbT+lJ*T_ITEM_GRAN */
        movd      %xmm2, %r10d
        psrlq     $11, %xmm9
        movd      %xmm3, %ecx

/* NB : including +/- sign for the exponent!! */
        psllq     $52, %xmm8
        psllq     $52, %xmm9
        pextrw    $4, %xmm2, %r11d
        pextrw    $4, %xmm3, %r8d
        movmskps  %xmm12, %eax
        shll      $3, %r10d
        shll      $3, %ecx
        shll      $3, %r11d
        shll      $3, %r8d
        movq      13952(%rdx,%r10), %xmm6
        movq      13952(%rdx,%rcx), %xmm7
        movhpd    13952(%rdx,%r11), %xmm6
        movhpd    13952(%rdx,%r8), %xmm7
        mulpd     %xmm6, %xmm10
        mulpd     %xmm7, %xmm11
        addpd     %xmm10, %xmm6
        addpd     %xmm11, %xmm7
        paddq     %xmm8, %xmm6
        paddq     %xmm9, %xmm7
        cvtpd2ps  %xmm6, %xmm1
        cvtpd2ps  %xmm7, %xmm4
        movlhps   %xmm4, %xmm1
        testl     %eax, %eax
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        movups    112(%rsp), %xmm8
        movaps    %xmm1, %xmm0
        movups    96(%rsp), %xmm9
        movups    176(%rsp), %xmm10
        movups    80(%rsp), %xmm11
        movups    208(%rsp), %xmm12
        movups    48(%rsp), %xmm13
        movups    144(%rsp), %xmm14
        movups    160(%rsp), %xmm15
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        movups    %xmm0, 64(%rsp)
        movups    %xmm5, 128(%rsp)
        movups    %xmm1, 192(%rsp)
        je        .LBL_1_2

        xorb      %cl, %cl
        xorl      %edx, %edx
        movq      %rsi, 8(%rsp)
        movq      %rdi, (%rsp)
        movq      %r12, 40(%rsp)
        cfi_offset_rel_rsp (12, 40)
        movb      %cl, %r12b
        movq      %r13, 32(%rsp)
        cfi_offset_rel_rsp (13, 32)
        movl      %eax, %r13d
        movq      %r14, 24(%rsp)
        cfi_offset_rel_rsp (14, 24)
        movl      %edx, %r14d
        movq      %r15, 16(%rsp)
        cfi_offset_rel_rsp (15, 16)
        cfi_remember_state

.LBL_1_6:
        btl       %r14d, %r13d
        jc        .LBL_1_12

.LBL_1_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_1_10

.LBL_1_8:
        incb      %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_1_6

        movq      8(%rsp), %rsi
        movq      (%rsp), %rdi
        movq      40(%rsp), %r12
        cfi_restore (%r12)
        movq      32(%rsp), %r13
        cfi_restore (%r13)
        movq      24(%rsp), %r14
        cfi_restore (%r14)
        movq      16(%rsp), %r15
        cfi_restore (%r15)
        movups    192(%rsp), %xmm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        movss     68(%rsp,%r15,8), %xmm0
        movss     132(%rsp,%r15,8), %xmm1

        call      JUMPTARGET(powf)

        movss     %xmm0, 196(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        movss     64(%rsp,%r15,8), %xmm0
        movss     128(%rsp,%r15,8), %xmm1

        call      JUMPTARGET(powf)

        movss     %xmm0, 192(%rsp,%r15,8)
        jmp       .LBL_1_7

END (_ZGVbN4vv_powf_sse4)
