/* Function powf vectorized with AVX2.
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
ENTRY(_ZGVdN8vv_powf_avx2)
/*
   ALGORITHM DESCRIPTION:

     We are using the next identity : pow(x,y) = 2^(y * log2(x)).

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
        subq      $448, %rsp
        lea       __VPACK_ODD_ind.6357.0.1(%rip), %rcx
        vmovups   %ymm14, 320(%rsp)

/* hi bits */
        lea       __VPACK_ODD_ind.6358.0.1(%rip), %rax
        vmovups   %ymm12, 256(%rsp)
        vmovups   %ymm9, 96(%rsp)
        vmovups   %ymm13, 224(%rsp)
        vmovups   %ymm15, 352(%rsp)
        vmovups   %ymm11, 384(%rsp)
        vmovups   %ymm10, 288(%rsp)
        vmovups   (%rcx), %ymm10
        vmovups   %ymm8, 160(%rsp)
        vmovdqa   %ymm1, %ymm9
        movq      __svml_spow_data@GOTPCREL(%rip), %rdx
        vextractf128 $1, %ymm0, %xmm7
        vcvtps2pd %xmm0, %ymm14
        vcvtps2pd %xmm7, %ymm12
        vpsubd _NMINNORM(%rdx), %ymm0, %ymm7

/* preserve mantissa, set input exponent to 2^(-10) */
        vandpd _ExpMask(%rdx), %ymm14, %ymm3
        vandpd _ExpMask(%rdx), %ymm12, %ymm13

/* exponent bits selection */
        vpsrlq    $20, %ymm12, %ymm12
        vpsrlq    $20, %ymm14, %ymm14
        vextractf128 $1, %ymm9, %xmm2
        vcvtps2pd %xmm9, %ymm1
        vpand _ABSMASK(%rdx), %ymm9, %ymm8
        vcvtps2pd %xmm2, %ymm6
        vorpd _Two10(%rdx), %ymm3, %ymm2
        vorpd _Two10(%rdx), %ymm13, %ymm3

/* reciprocal approximation good to at least 11 bits */
        vcvtpd2ps %ymm2, %xmm5
        vcvtpd2ps %ymm3, %xmm15
        vrcpps    %xmm5, %xmm4
        vrcpps    %xmm15, %xmm11
        vcvtps2pd %xmm4, %ymm13
        vcvtps2pd %xmm11, %ymm4
        vpermps   %ymm12, %ymm10, %ymm11

/* round reciprocal to nearest integer, will have 1+9 mantissa bits */
        vroundpd  $0, %ymm13, %ymm12
        vpermps   %ymm14, %ymm10, %ymm5
        vroundpd  $0, %ymm4, %ymm14
        vmovupd _One(%rdx), %ymm4

/* table lookup */
        vpsrlq    $40, %ymm12, %ymm10
        vfmsub213pd %ymm4, %ymm12, %ymm2
        vfmsub213pd %ymm4, %ymm14, %ymm3
        vcmpgt_oqpd _Threshold(%rdx), %ymm12, %ymm12
        vxorpd    %ymm4, %ymm4, %ymm4
        vandpd _Bias(%rdx), %ymm12, %ymm12

/* biased exponent in DP format */
        vcvtdq2pd %xmm11, %ymm13
        vpcmpeqd  %ymm11, %ymm11, %ymm11
        vgatherqpd %ymm11, _Log2Rcp_lookup(%rdx,%ymm10), %ymm4
        vpsrlq    $40, %ymm14, %ymm10
        vcmpgt_oqpd _Threshold(%rdx), %ymm14, %ymm14
        vpcmpeqd  %ymm11, %ymm11, %ymm11
        vandpd _Bias(%rdx), %ymm14, %ymm14
        vcvtdq2pd %xmm5, %ymm15
        vxorpd    %ymm5, %ymm5, %ymm5
        vgatherqpd %ymm11, _Log2Rcp_lookup(%rdx,%ymm10), %ymm5
        vorpd _Bias1(%rdx), %ymm12, %ymm11
        vorpd _Bias1(%rdx), %ymm14, %ymm10
        vsubpd    %ymm11, %ymm15, %ymm11
        vsubpd    %ymm10, %ymm13, %ymm14
        vmovupd _poly_coeff_4(%rdx), %ymm15
        vmovupd _poly_coeff_3(%rdx), %ymm13
        vmulpd    %ymm3, %ymm3, %ymm10
        vfmadd213pd %ymm15, %ymm3, %ymm13
        vmovdqa   %ymm15, %ymm12
        vfmadd231pd _poly_coeff_3(%rdx), %ymm2, %ymm12
        vmulpd    %ymm2, %ymm2, %ymm15

/* reconstruction */
        vfmadd213pd %ymm3, %ymm10, %ymm13
        vfmadd213pd %ymm2, %ymm15, %ymm12
        vaddpd    %ymm5, %ymm13, %ymm13
        vaddpd    %ymm4, %ymm12, %ymm2
        vfmadd231pd _L2(%rdx), %ymm14, %ymm13
        vfmadd132pd _L2(%rdx), %ymm2, %ymm11
        vmulpd    %ymm6, %ymm13, %ymm2
        vmulpd    %ymm1, %ymm11, %ymm10
        vmulpd __dbInvLn2(%rdx), %ymm2, %ymm6
        vmulpd __dbInvLn2(%rdx), %ymm10, %ymm15

/* to round down; if dR is an integer we will get R = 1, which is ok */
        vsubpd __dbHALF(%rdx), %ymm6, %ymm3
        vsubpd __dbHALF(%rdx), %ymm15, %ymm1
        vaddpd __dbShifter(%rdx), %ymm3, %ymm13
        vaddpd __dbShifter(%rdx), %ymm1, %ymm14
        vsubpd __dbShifter(%rdx), %ymm13, %ymm12
        vmovups   (%rax), %ymm1
        vsubpd __dbShifter(%rdx), %ymm14, %ymm11

/* [0..1) */
        vsubpd    %ymm12, %ymm6, %ymm6
        vpermps   %ymm10, %ymm1, %ymm3
        vpermps   %ymm2, %ymm1, %ymm10
        vpcmpgtd _NMAXVAL(%rdx), %ymm7, %ymm4
        vpcmpgtd _INF(%rdx), %ymm8, %ymm1
        vpcmpeqd _NMAXVAL(%rdx), %ymm7, %ymm7
        vpcmpeqd _INF(%rdx), %ymm8, %ymm8
        vpor      %ymm7, %ymm4, %ymm2
        vpor      %ymm8, %ymm1, %ymm1
        vsubpd    %ymm11, %ymm15, %ymm7
        vinsertf128 $1, %xmm10, %ymm3, %ymm10
        vpor      %ymm1, %ymm2, %ymm3

/* iAbsX = iAbsX&iAbsMask */
        vandps __iAbsMask(%rdx), %ymm10, %ymm10

/* iRangeMask = (iAbsX>iDomainRange) */
        vpcmpgtd __iDomainRange(%rdx), %ymm10, %ymm4
        vpor      %ymm4, %ymm3, %ymm5
        vmulpd __dbC1(%rdx), %ymm7, %ymm4
        vmovmskps %ymm5, %ecx
        vmulpd __dbC1(%rdx), %ymm6, %ymm5

/* low K bits */
        vandps __lbLOWKBITS(%rdx), %ymm14, %ymm6

/* dpP= _dbT+lJ*T_ITEM_GRAN */
        vxorpd    %ymm7, %ymm7, %ymm7
        vpcmpeqd  %ymm1, %ymm1, %ymm1
        vandps __lbLOWKBITS(%rdx), %ymm13, %ymm2
        vxorpd    %ymm10, %ymm10, %ymm10
        vpcmpeqd  %ymm3, %ymm3, %ymm3
        vgatherqpd %ymm1, 13952(%rdx,%ymm6,8), %ymm7
        vgatherqpd %ymm3, 13952(%rdx,%ymm2,8), %ymm10
        vpsrlq    $11, %ymm14, %ymm14
        vpsrlq    $11, %ymm13, %ymm13
        vfmadd213pd %ymm7, %ymm4, %ymm7
        vfmadd213pd %ymm10, %ymm5, %ymm10

/* NB : including +/- sign for the exponent!! */
        vpsllq    $52, %ymm14, %ymm8
        vpsllq    $52, %ymm13, %ymm11
        vpaddq    %ymm8, %ymm7, %ymm12
        vpaddq    %ymm11, %ymm10, %ymm1
        vcvtpd2ps %ymm12, %xmm15
        vcvtpd2ps %ymm1, %xmm2
        vinsertf128 $1, %xmm2, %ymm15, %ymm1
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovups   160(%rsp), %ymm8
        vmovups   96(%rsp), %ymm9
        vmovups   288(%rsp), %ymm10
        vmovups   384(%rsp), %ymm11
        vmovups   256(%rsp), %ymm12
        vmovups   224(%rsp), %ymm13
        vmovups   320(%rsp), %ymm14
        vmovups   352(%rsp), %ymm15
        vmovdqa   %ymm1, %ymm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovups   %ymm0, 64(%rsp)
        vmovups   %ymm9, 128(%rsp)
        vmovups   %ymm1, 192(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        movq      %rsi, 8(%rsp)
        movq      %rdi, (%rsp)
        movq      %r12, 40(%rsp)
        cfi_offset_rel_rsp (12, 40)
        movb      %dl, %r12b
        movq      %r13, 32(%rsp)
        cfi_offset_rel_rsp (13, 32)
        movl      %ecx, %r13d
        movq      %r14, 24(%rsp)
        cfi_offset_rel_rsp (14, 24)
        movl      %eax, %r14d
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
        vmovups   192(%rsp), %ymm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        vmovss    68(%rsp,%r15,8), %xmm0
        vmovss    132(%rsp,%r15,8), %xmm1
        vzeroupper

        call      JUMPTARGET(powf)

        vmovss    %xmm0, 196(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    64(%rsp,%r15,8), %xmm0
        vmovss    128(%rsp,%r15,8), %xmm1
        vzeroupper

        call      JUMPTARGET(powf)

        vmovss    %xmm0, 192(%rsp,%r15,8)
        jmp       .LBL_1_7

END(_ZGVdN8vv_powf_avx2)

	.section .rodata, "a"
__VPACK_ODD_ind.6357.0.1:
	.long	1
	.long	3
	.long	5
	.long	7
	.long	0
	.long	0
	.long	0
	.long	0
	.space 32, 0x00
__VPACK_ODD_ind.6358.0.1:
	.long	1
	.long	3
	.long	5
	.long	7
	.long	0
	.long	0
	.long	0
	.long	0
