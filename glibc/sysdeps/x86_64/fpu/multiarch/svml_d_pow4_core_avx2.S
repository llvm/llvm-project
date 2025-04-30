/* Function pow vectorized with AVX2.
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
ENTRY (_ZGVdN4vv_pow_avx2)
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
        subq      $448, %rsp
        movq      __svml_dpow_data@GOTPCREL(%rip), %rax
        vmovups   %ymm11, 160(%rsp)
        vmovups   %ymm8, 224(%rsp)
        vmovups   %ymm10, 352(%rsp)
        vmovups   %ymm9, 384(%rsp)
        vmovups   %ymm13, 288(%rsp)
        vmovapd   %ymm1, %ymm11
        vxorpd    %ymm1, %ymm1, %ymm1
        vextracti128 $1, %ymm0, %xmm5
        vshufps   $221, %xmm5, %xmm0, %xmm5

/* i = (((Hi(x) & 0x000ffe00) + 0x00000200) >> 10); -> i = (b1..b11 + 1) / 2 */
        vandps _iIndexMask(%rax), %xmm5, %xmm3
        vpaddd _iIndexAdd(%rax), %xmm3, %xmm6
        vpsrld    $10, %xmm6, %xmm8

/* Index for reciprocal table */
        vpslld    $3, %xmm8, %xmm9

/* Index for log2 table */
        vpslld    $4, %xmm8, %xmm6

/* x1 = x; Hi(x1) = (Hi(x1)&0x000fffff)|0x3ff00000 */
        vandpd _iMantissaMask(%rax), %ymm0, %ymm4
        vorpd _dbOne(%rax), %ymm4, %ymm13
        vpcmpeqd  %ymm4, %ymm4, %ymm4
        vpcmpeqd  %ymm8, %ymm8, %ymm8

/* k = Hi(x); k = k - 0x3fe7fe00; k = k >> 20 */
        vpsubd _i3fe7fe00(%rax), %xmm5, %xmm3
        vpaddd _HIDELTA(%rax), %xmm5, %xmm5
        vextracti128 $1, %ymm11, %xmm7
        vshufps   $221, %xmm7, %xmm11, %xmm2
        vpand _ABSMASK(%rax), %xmm2, %xmm10
        vpcmpeqd  %ymm2, %ymm2, %ymm2
        vgatherdpd %ymm2, 11712(%rax,%xmm9), %ymm1
        vmovups _LORANGE(%rax), %xmm7
        vxorpd    %ymm2, %ymm2, %ymm2
        vgatherdpd %ymm4, 19968(%rax,%xmm6), %ymm2
        vxorpd    %ymm4, %ymm4, %ymm4
        vgatherdpd %ymm8, 19976(%rax,%xmm6), %ymm4
        vpsrad    $20, %xmm3, %xmm6
        vpaddd _i2p20_2p19(%rax), %xmm6, %xmm9
        vpshufd   $80, %xmm9, %xmm8
        vpshufd   $250, %xmm9, %xmm3

/* x1Hi=x1; Lo(x1Hi)&=0xf8000000; x1Lo = x1-x1Hi */
        vandpd _iHighMask(%rax), %ymm13, %ymm9
        vinserti128 $1, %xmm3, %ymm8, %ymm6
        vandpd _iffffffff00000000(%rax), %ymm6, %ymm8

/* r1 = x1*rcp1 */
        vmulpd    %ymm1, %ymm13, %ymm6
        vsubpd    %ymm9, %ymm13, %ymm3
        vsubpd _db2p20_2p19(%rax), %ymm8, %ymm8

/* cq = c+r1 */
        vaddpd _LHN(%rax), %ymm6, %ymm13

/* E = -r1+__fence(x1Hi*rcp1) */
        vfmsub213pd %ymm6, %ymm1, %ymm9

/* E=E+x1Lo*rcp1 */
        vfmadd213pd %ymm9, %ymm1, %ymm3

/* T = k + L1hi */
        vaddpd    %ymm2, %ymm8, %ymm1

/* T_Rh = T + cq */
        vaddpd    %ymm13, %ymm1, %ymm8

/* Rl = T-T_Rh; -> -Rh */
        vsubpd    %ymm8, %ymm1, %ymm6

/* Rl=Rl+cq */
        vaddpd    %ymm6, %ymm13, %ymm1

/* T_Rh_Eh = T_Rh + E */
        vaddpd    %ymm3, %ymm8, %ymm6

/* cq = cq + E */
        vaddpd    %ymm3, %ymm13, %ymm13

/* HLL = T_Rh - T_Rh_Eh; -> -Eh */
        vsubpd    %ymm6, %ymm8, %ymm9

/* HLL+=E;  -> El */
        vaddpd    %ymm9, %ymm3, %ymm2

/* HLL+=Rl */
        vaddpd    %ymm1, %ymm2, %ymm8

/* HLL+=L1lo */
        vaddpd    %ymm4, %ymm8, %ymm4
        vmovupd _clv_2(%rax), %ymm8

/* HLL = HLL + (((((((a7)*cq+a6)*cq+a5)*cq+a4)*cq+a3)*cq+a2)*cq+a1)*cq */
        vfmadd213pd _clv_3(%rax), %ymm13, %ymm8
        vfmadd213pd _clv_4(%rax), %ymm13, %ymm8
        vfmadd213pd _clv_5(%rax), %ymm13, %ymm8
        vfmadd213pd _clv_6(%rax), %ymm13, %ymm8
        vfmadd213pd _clv_7(%rax), %ymm13, %ymm8
        vfmadd213pd %ymm4, %ymm13, %ymm8

/* T_Rh_Eh_HLLhi = T_Rh_Eh + HLL */
        vaddpd    %ymm8, %ymm6, %ymm9

/* HH = T_Rh_Eh_HLLhi; Lo(HH)&=0xf8000000 */
        vandpd _iHighMask(%rax), %ymm9, %ymm2

/*
   2^(y*(HH+HL+HLL)) starts here:
   yH = y; Lo(yH)&=0xf8000000;
 */
        vandpd _iHighMask(%rax), %ymm11, %ymm1

/* HLLhi = T_Rh_Eh_HLLhi - T_Rh_Eh */
        vsubpd    %ymm6, %ymm9, %ymm13

/* HL = T_Rh_Eh_HLLhi-HH */
        vsubpd    %ymm2, %ymm9, %ymm4

/* pH = yH*HH */
        vmulpd    %ymm2, %ymm1, %ymm9

/* HLL = HLL - HLLhi */
        vsubpd    %ymm13, %ymm8, %ymm6

/* yL = y-yH */
        vsubpd    %ymm1, %ymm11, %ymm8
        vextracti128 $1, %ymm9, %xmm3
        vshufps   $221, %xmm3, %xmm9, %xmm13
        vpand _ABSMASK(%rax), %xmm13, %xmm3
        vpcmpgtd  %xmm5, %xmm7, %xmm13
        vpcmpgtd _INF(%rax), %xmm10, %xmm7
        vpcmpeqd _INF(%rax), %xmm10, %xmm10
        vpor      %xmm10, %xmm7, %xmm7
        vpor      %xmm7, %xmm13, %xmm5

/* pL=yL*HL+yH*HL; pL+=yL*HH */
        vmulpd    %ymm4, %ymm8, %ymm7
        vpcmpgtd _DOMAINRANGE(%rax), %xmm3, %xmm13
        vpcmpeqd _DOMAINRANGE(%rax), %xmm3, %xmm10
        vpor      %xmm10, %xmm13, %xmm3
        vpor      %xmm3, %xmm5, %xmm13
        vfmadd213pd %ymm7, %ymm4, %ymm1

/* pLL = y*HLL;
   pHH = pH + *(double*)&db2p45_2p44
 */
        vaddpd _db2p45_2p44(%rax), %ymm9, %ymm7
        vmovmskps %xmm13, %ecx
        vfmadd213pd %ymm1, %ymm2, %ymm8

/* t=pL+pLL; t+=pHL */
        vfmadd231pd %ymm11, %ymm6, %ymm8
        vextracti128 $1, %ymm7, %xmm1
        vshufps   $136, %xmm1, %xmm7, %xmm10

/* _n = Lo(pHH);
   _n = _n & 0xffffff80;
   _n = _n >> 7;
   Hi(_2n) = (0x3ff+_n)<<20; Lo(_2n) = 0; -> 2^n
 */
        vpslld    $13, %xmm10, %xmm2
        vpaddd _iOne(%rax), %xmm2, %xmm13
        vpshufd   $80, %xmm13, %xmm4
        vpshufd   $250, %xmm13, %xmm1

/* j = Lo(pHH)&0x0000007f */
        vandps _jIndexMask(%rax), %xmm10, %xmm3

/* T1 = ((double*)exp2_tbl)[ 2*j ] */
        vpcmpeqd  %ymm10, %ymm10, %ymm10
        vpslld    $4, %xmm3, %xmm5

/* pHH = pHH - *(double*)&db2p45_2p44 */
        vsubpd _db2p45_2p44(%rax), %ymm7, %ymm7

/* pHL = pH - pHH */
        vsubpd    %ymm7, %ymm9, %ymm9
        vaddpd    %ymm9, %ymm8, %ymm6
        vinserti128 $1, %xmm1, %ymm4, %ymm2
        vxorpd    %ymm1, %ymm1, %ymm1
        vgatherdpd %ymm10, 36416(%rax,%xmm5), %ymm1
        vandpd _ifff0000000000000(%rax), %ymm2, %ymm13
        vmovupd _cev_1(%rax), %ymm2
        vmulpd    %ymm1, %ymm13, %ymm1
        vfmadd213pd _cev_2(%rax), %ymm6, %ymm2
        vmulpd    %ymm6, %ymm1, %ymm8
        vfmadd213pd _cev_3(%rax), %ymm6, %ymm2
        vfmadd213pd _cev_4(%rax), %ymm6, %ymm2
        vfmadd213pd _cev_5(%rax), %ymm6, %ymm2
        vfmadd213pd %ymm1, %ymm8, %ymm2
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovups   224(%rsp), %ymm8
        vmovups   384(%rsp), %ymm9
        vmovups   352(%rsp), %ymm10
        vmovups   160(%rsp), %ymm11
        vmovups   288(%rsp), %ymm13
        vmovdqa   %ymm2, %ymm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovupd   %ymm0, 192(%rsp)
        vmovupd   %ymm11, 256(%rsp)
        vmovupd   %ymm2, 320(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        vmovups   %ymm12, 64(%rsp)
        vmovups   %ymm14, 32(%rsp)
        vmovups   %ymm15, (%rsp)
        movq      %rsi, 104(%rsp)
        movq      %rdi, 96(%rsp)
        movq      %r12, 136(%rsp)
        cfi_offset_rel_rsp (12, 136)
        movb      %dl, %r12b
        movq      %r13, 128(%rsp)
        cfi_offset_rel_rsp (13, 128)
        movl      %ecx, %r13d
        movq      %r14, 120(%rsp)
        cfi_offset_rel_rsp (14, 120)
        movl      %eax, %r14d
        movq      %r15, 112(%rsp)
        cfi_offset_rel_rsp (15, 112)
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

        vmovups   64(%rsp), %ymm12
        vmovups   32(%rsp), %ymm14
        vmovups   (%rsp), %ymm15
        vmovupd   320(%rsp), %ymm2
        movq      104(%rsp), %rsi
        movq      96(%rsp), %rdi
        movq      136(%rsp), %r12
        cfi_restore (%r12)
        movq      128(%rsp), %r13
        cfi_restore (%r13)
        movq      120(%rsp), %r14
        cfi_restore (%r14)
        movq      112(%rsp), %r15
        cfi_restore (%r15)
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    200(%rsp,%r15), %xmm0
        vmovsd    264(%rsp,%r15), %xmm1
        vzeroupper

        call      JUMPTARGET(pow)

        vmovsd    %xmm0, 328(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    192(%rsp,%r15), %xmm0
        vmovsd    256(%rsp,%r15), %xmm1
        vzeroupper

        call      JUMPTARGET(pow)

        vmovsd    %xmm0, 320(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVdN4vv_pow_avx2)
