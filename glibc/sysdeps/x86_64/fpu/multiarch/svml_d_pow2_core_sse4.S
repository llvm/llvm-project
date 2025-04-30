/* Function pow vectorized with SSE4.
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
#include "svml_d_pow_data.h"

	.text
ENTRY (_ZGVbN2vv_pow_sse4)
/*
   ALGORITHM DESCRIPTION:

     1) Calculating log2|x|
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
     where cq = X1*Rcp1*Rcp2*Rcp3C-C,
     a1=1/(C*ln(2))-1 is small,
     a2=1/(2*C^2*ln2),
     a3=1/(3*C^3*ln2),
     ...
     We get 3 parts of log2 result: HH+HL+HLL ~= log2|x|.

     2)  Calculation of y*(HH+HL+HLL).
     Split y into YHi+YLo.
     Get high PH and medium PL parts of y*log2|x|.
     Get low PLL part of y*log2|x|.
     Now we have PH+PL+PLL ~= y*log2|x|.

     3) Calculation of 2^(PH+PL+PLL).
     Mathematical idea of computing 2^(PH+PL+PLL) is the following.
     Let's represent PH+PL+PLL in the form N + j/2^expK + Z,
     where expK=7 in this implementation, N and j are integers,
     0<=j<=2^expK-1, |Z|<2^(-expK-1).
     Hence 2^(PH+PL+PLL) ~= 2^N * 2^(j/2^expK) * 2^Z,
     where 2^(j/2^expK) is stored in a table, and
     2^Z ~= 1 + B1*Z + B2*Z^2 ... + B5*Z^5.

     We compute 2^(PH+PL+PLL) as follows.
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
        movq      __svml_dpow_data@GOTPCREL(%rip), %rdx
        movups    %xmm14, 80(%rsp)
        movups    %xmm9, 176(%rsp)
        movaps    %xmm1, %xmm9
        pshufd    $221, %xmm0, %xmm1
        movq _iIndexMask(%rdx), %xmm14
        movq _iIndexAdd(%rdx), %xmm6

/* i = (((Hi(x) & 0x000ffe00) + 0x00000200) >> 10); -> i = (b1..b11 + 1) / 2 */
        pand      %xmm1, %xmm14
        paddd     %xmm6, %xmm14
        psrld     $10, %xmm14
        movups    %xmm13, 96(%rsp)

/* Index for reciprocal table */
        movdqa    %xmm14, %xmm13
        pslld     $3, %xmm13

/* Index for log2 table */
        pslld     $4, %xmm14
        movd      %xmm13, %eax
        movups    %xmm10, 160(%rsp)
        movups _iMantissaMask(%rdx), %xmm10
        movslq    %eax, %rax

/* x1 = x; Hi(x1) = (Hi(x1)&0x000fffff)|0x3ff00000 */
        andps     %xmm0, %xmm10
        pextrd    $1, %xmm13, %ecx
        movslq    %ecx, %rcx
        movups    %xmm0, (%rsp)
        movdqa    %xmm1, %xmm0

/* k = Hi(x); k = k - 0x3fe7fe00; k = k >> 20 */
        movq _i3fe7fe00(%rdx), %xmm6
        psubd     %xmm6, %xmm0
        movups _iHighMask(%rdx), %xmm6
        psrad     $20, %xmm0
        movups    %xmm15, 48(%rsp)
        movups    %xmm12, 112(%rsp)
        orps _dbOne(%rdx), %xmm10
        movsd     11712(%rdx,%rax), %xmm12
        movd      %xmm14, %r8d
        movq _i2p20_2p19(%rdx), %xmm15
        movhpd    11712(%rdx,%rcx), %xmm12
        paddd     %xmm15, %xmm0
        pextrd    $1, %xmm14, %r9d

/* x1Hi=x1; Lo(x1Hi)&=0xf8000000; x1Lo = x1-x1Hi */
        movaps    %xmm6, %xmm14
        andps     %xmm10, %xmm14
        movaps    %xmm10, %xmm15
        subpd     %xmm14, %xmm15

/* r1 = x1*rcp1 */
        mulpd     %xmm12, %xmm10

/* E = -r1+__fence(x1Hi*rcp1) */
        mulpd     %xmm12, %xmm14

/* E=E+x1Lo*rcp1 */
        mulpd     %xmm15, %xmm12
        subpd     %xmm10, %xmm14
        pshufd    $80, %xmm0, %xmm0
        movslq    %r8d, %r8
        andps _iffffffff00000000(%rdx), %xmm0
        subpd _db2p20_2p19(%rdx), %xmm0
        addpd     %xmm12, %xmm14
        movslq    %r9d, %r9

/* T_Rh_Eh = T_Rh + E */
        movaps    %xmm14, %xmm15
        movups    %xmm8, 208(%rsp)
        movups    19968(%rdx,%r8), %xmm8
        movups    %xmm11, 144(%rsp)
        movaps    %xmm8, %xmm11

/* cq = c+r1 */
        movups _LHN(%rdx), %xmm13
        movhpd    19968(%rdx,%r9), %xmm11
        addpd     %xmm10, %xmm13

/* T = k + L1hi */
        addpd     %xmm0, %xmm11

/* T_Rh = T + cq */
        movaps    %xmm13, %xmm12
        addpd     %xmm11, %xmm12
        addpd     %xmm12, %xmm15

/* Rl = T-T_Rh; -> -Rh */
        subpd     %xmm12, %xmm11

/* HLL = T_Rh - T_Rh_Eh; -> -Eh */
        subpd     %xmm15, %xmm12

/* Rl=Rl+cq; */
        addpd     %xmm13, %xmm11

/* cq = cq + E */
        addpd     %xmm14, %xmm13

/* HLL+=E;  -> El */
        addpd     %xmm14, %xmm12

/* HLL+=Rl */
        addpd     %xmm12, %xmm11
        unpckhpd  19968(%rdx,%r9), %xmm8

/* T_Rh_Eh_HLLhi = T_Rh_Eh + HLL */
        movaps    %xmm15, %xmm14

/* HLL+=L1lo; */
        addpd     %xmm11, %xmm8
        movups _clv_2(%rdx), %xmm11

/* HH = T_Rh_Eh_HLLhi; Lo(HH)&=0xf8000000 */
        movaps    %xmm6, %xmm12

/* HLL = HLL + (((((((a7)*cq+a6)*cq+a5)*cq+a4)*cq+a3)*cq+a2)*cq+a1)*cq */
        mulpd     %xmm13, %xmm11
        addpd _clv_3(%rdx), %xmm11
        mulpd     %xmm13, %xmm11
        addpd _clv_4(%rdx), %xmm11
        mulpd     %xmm13, %xmm11
        addpd _clv_5(%rdx), %xmm11
        mulpd     %xmm13, %xmm11
        addpd _clv_6(%rdx), %xmm11
        mulpd     %xmm13, %xmm11
        addpd _clv_7(%rdx), %xmm11
        mulpd     %xmm11, %xmm13
        addpd     %xmm13, %xmm8
        addpd     %xmm8, %xmm14

/*
   2^(y*(HH+HL+HLL)) starts here:
   yH = y; Lo(yH)&=0xf8000000
 */
        andps     %xmm9, %xmm6

/* yL = y-yH; */
        movaps    %xmm9, %xmm11
        subpd     %xmm6, %xmm11
        andps     %xmm14, %xmm12

/* HLLhi = T_Rh_Eh_HLLhi - T_Rh_Eh */
        movaps    %xmm14, %xmm10

/* HL = T_Rh_Eh_HLLhi-HH; */
        subpd     %xmm12, %xmm14
        subpd     %xmm15, %xmm10
        movq _HIDELTA(%rdx), %xmm2

/* pH = yH*HH; */
        movaps    %xmm6, %xmm13
        movq _LORANGE(%rdx), %xmm3
        paddd     %xmm2, %xmm1
        pcmpgtd   %xmm1, %xmm3

/* pL=yL*HL+yH*HL; pL+=yL*HH; */
        movaps    %xmm11, %xmm1
        mulpd     %xmm14, %xmm1
        mulpd     %xmm14, %xmm6
        mulpd     %xmm12, %xmm13
        mulpd     %xmm11, %xmm12
        addpd     %xmm6, %xmm1

/* HLL = HLL - HLLhi */
        subpd     %xmm10, %xmm8
        addpd     %xmm12, %xmm1

/* pLL = y*HLL */
        mulpd     %xmm9, %xmm8
        movups _db2p45_2p44(%rdx), %xmm11

/* pHH = pH + *(double*)&db2p45_2p44 */
        movaps    %xmm11, %xmm12
        addpd     %xmm13, %xmm12

/* t=pL+pLL; t+=pHL */
        addpd     %xmm8, %xmm1
        movq _ABSMASK(%rdx), %xmm5
        pshufd    $221, %xmm9, %xmm4
        pand      %xmm5, %xmm4
        movq _INF(%rdx), %xmm7
        movdqa    %xmm4, %xmm2
        pcmpgtd   %xmm7, %xmm2
        pcmpeqd   %xmm7, %xmm4
        pshufd    $136, %xmm12, %xmm7
        por       %xmm4, %xmm2

/* pHH = pHH - *(double*)&db2p45_2p44 */
        subpd     %xmm11, %xmm12
        pshufd    $221, %xmm13, %xmm10
        por       %xmm2, %xmm3

/* pHL = pH - pHH; */
        subpd     %xmm12, %xmm13
        pand      %xmm5, %xmm10
        movq _DOMAINRANGE(%rdx), %xmm5
        movdqa    %xmm10, %xmm4
        addpd     %xmm1, %xmm13
        pcmpgtd   %xmm5, %xmm4
        pcmpeqd   %xmm5, %xmm10
        por       %xmm10, %xmm4
        movq _jIndexMask(%rdx), %xmm6
        por       %xmm4, %xmm3
        movmskps  %xmm3, %eax

/* j = Lo(pHH)&0x0000007f */
        pand      %xmm7, %xmm6
        movq _iOne(%rdx), %xmm3

/* _n = Lo(pHH);
   _n = _n & 0xffffff80;
   _n = _n >> 7;
   Hi(_2n) = (0x3ff+_n)<<20; Lo(_2n) = 0; -> 2^n
 */
        pslld     $13, %xmm7
        paddd     %xmm3, %xmm7
        pslld     $4, %xmm6
        movups _cev_1(%rdx), %xmm3
        movaps    %xmm13, %xmm4
        mulpd     %xmm13, %xmm3

/* T1 = ((double*)exp2_tbl)[ 2*j ] */
        movd      %xmm6, %r10d
        pshufd    $80, %xmm7, %xmm0
        andps _ifff0000000000000(%rdx), %xmm0
        addpd _cev_2(%rdx), %xmm3
        mulpd     %xmm13, %xmm3
        addpd _cev_3(%rdx), %xmm3
        mulpd     %xmm13, %xmm3
        movslq    %r10d, %r10
        andl      $3, %eax
        pextrd    $1, %xmm6, %r11d
        movslq    %r11d, %r11
        addpd _cev_4(%rdx), %xmm3
        movsd     36416(%rdx,%r10), %xmm2
        movhpd    36416(%rdx,%r11), %xmm2
        mulpd     %xmm2, %xmm0
        mulpd     %xmm3, %xmm13
        mulpd     %xmm0, %xmm4
        addpd _cev_5(%rdx), %xmm13
        mulpd     %xmm4, %xmm13
        addpd     %xmm13, %xmm0
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        movups    208(%rsp), %xmm8
        movups    176(%rsp), %xmm9
        movups    160(%rsp), %xmm10
        movups    144(%rsp), %xmm11
        movups    112(%rsp), %xmm12
        movups    96(%rsp), %xmm13
        movups    80(%rsp), %xmm14
        movups    48(%rsp), %xmm15
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        movups    (%rsp), %xmm1
        movups    %xmm1, 64(%rsp)
        movups    %xmm9, 128(%rsp)
        movups    %xmm0, 192(%rsp)
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
        movups    192(%rsp), %xmm0
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        movsd     72(%rsp,%r15), %xmm0
        movsd     136(%rsp,%r15), %xmm1

        call      JUMPTARGET(pow)

        movsd     %xmm0, 200(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        movsd     64(%rsp,%r15), %xmm0
        movsd     128(%rsp,%r15), %xmm1

        call      JUMPTARGET(pow)

        movsd     %xmm0, 192(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVbN2vv_pow_sse4)
