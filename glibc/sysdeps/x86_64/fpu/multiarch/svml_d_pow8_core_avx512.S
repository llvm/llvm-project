/* Function pow vectorized with AVX-512. KNL and SKX versions.
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
#include "svml_d_wrapper_impl.h"

/* ALGORITHM DESCRIPTION:

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

	.text
ENTRY (_ZGVeN8vv_pow_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_ff _ZGVdN4vv_pow
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        vpsrlq    $32, %zmm0, %zmm13
        vmovaps   %zmm1, %zmm12
        movq      __svml_dpow_data@GOTPCREL(%rip), %rax
        movl      $255, %edx
        vpmovqd   %zmm13, %ymm10
        vpsrlq    $32, %zmm12, %zmm14
        kmovw     %edx, %k1
        movl      $-1, %ecx
        vpmovqd   %zmm14, %ymm15

/* x1 = x; Hi(x1) = (Hi(x1)&0x000fffff)|0x3ff00000 */
        vmovups _dbOne(%rax), %zmm6

/* i = (((Hi(x) & 0x000ffe00) + 0x00000200) >> 10); -> i = (b1..b11 + 1) / 2 */
        vmovaps   %zmm10, %zmm5

/* k = Hi(x); k = k - 0x3fe7fe00; k = k >> 20 */
        vpsubd    _i3fe7fe00(%rax), %zmm10, %zmm14{%k1}
        vpandd    _iIndexMask(%rax), %zmm10, %zmm5{%k1}
        vpsrad    $20, %zmm14, %zmm14{%k1}
        vpxord    %zmm9, %zmm9, %zmm9
        vpaddd    _HIDELTA(%rax), %zmm10, %zmm3{%k1}
        vpaddd    _iIndexAdd(%rax), %zmm5, %zmm5{%k1}
        vpxord    %zmm7, %zmm7, %zmm7
        vpaddd    _i2p20_2p19(%rax), %zmm14, %zmm14{%k1}
        vpcmpd    $1, _LORANGE(%rax), %zmm3, %k2{%k1}
        vpsrld    $10, %zmm5, %zmm5{%k1}
        vpandd    _ABSMASK(%rax), %zmm15, %zmm2{%k1}
        vpbroadcastd %ecx, %zmm1{%k2}{z}

/* Index for reciprocal table */
        vpslld     $3, %zmm5, %zmm8{%k1}
        kxnorw     %k2, %k2, %k2
        vgatherdpd 11712(%rax,%ymm8), %zmm9{%k2}
        vpmovzxdq  %ymm14, %zmm10

/* Index for log2 table */
        vpslld    $4, %zmm5, %zmm13{%k1}
        kxnorw    %k2, %k2, %k2
        vpsllq    $32, %zmm10, %zmm3
        vpxord    %zmm8, %zmm8, %zmm8
        vpcmpd    $5, _INF(%rax), %zmm2, %k3{%k1}
        vpbroadcastd %ecx, %zmm4{%k3}{z}
        vpternlogq $248, _iMantissaMask(%rax), %zmm0, %zmm6
        kxnorw     %k3, %k3, %k3
        vpternlogq $168, _iffffffff00000000(%rax), %zmm10, %zmm3

/* x1Hi=x1; Lo(x1Hi)&=0xf8000000; x1Lo = x1-x1Hi */
        vpandq     _iHighMask(%rax), %zmm6, %zmm2
        vgatherdpd 19976(%rax,%ymm13), %zmm8{%k2}
        vpord      %zmm4, %zmm1, %zmm11{%k1}
        vsubpd     _db2p20_2p19(%rax), %zmm3, %zmm1
        vsubpd     %zmm2, %zmm6, %zmm5

/* r1 = x1*rcp1 */
        vmulpd     %zmm9, %zmm6, %zmm6
        vgatherdpd 19968(%rax,%ymm13), %zmm7{%k3}

/* cq = c+r1 */
        vaddpd _LHN(%rax), %zmm6, %zmm4

/* E = -r1+__fence(x1Hi*rcp1) */
        vfmsub213pd %zmm6, %zmm9, %zmm2

/* T = k + L1hi */
        vaddpd    %zmm7, %zmm1, %zmm7

/* E=E+x1Lo*rcp1 */
        vfmadd213pd %zmm2, %zmm9, %zmm5

/* T_Rh = T + cq */
        vaddpd    %zmm4, %zmm7, %zmm3

/* Rl = T-T_Rh; -> -Rh */
        vsubpd    %zmm3, %zmm7, %zmm9

/* Rl=Rl+cq */
        vaddpd    %zmm9, %zmm4, %zmm6

/* T_Rh_Eh = T_Rh + E */
        vaddpd    %zmm5, %zmm3, %zmm9

/* HLL = T_Rh - T_Rh_Eh; -> -Eh */
        vsubpd    %zmm9, %zmm3, %zmm2

/* cq = cq + E; */
        vaddpd    %zmm5, %zmm4, %zmm4

/* HLL+=E;  -> El */
        vaddpd    %zmm2, %zmm5, %zmm1
        vmovups _clv_2(%rax), %zmm5

/* HLL = HLL + (((((((a7)*cq+a6)*cq+a5)*cq+a4)*cq+a3)*cq+a2)*cq+a1)*cq */
        vfmadd213pd _clv_3(%rax), %zmm4, %zmm5

/* HLL+=Rl */
        vaddpd    %zmm6, %zmm1, %zmm7

/* 2^(y*(HH+HL+HLL)) starts here:
   yH = y; Lo(yH)&=0xf8000000
 */
        vpandq    _iHighMask(%rax), %zmm12, %zmm6

/* yL = y-yH */
        vsubpd      %zmm6, %zmm12, %zmm2
        vfmadd213pd _clv_4(%rax), %zmm4, %zmm5

/* HLL+=L1lo */
        vaddpd      %zmm8, %zmm7, %zmm8
        vfmadd213pd _clv_5(%rax), %zmm4, %zmm5
        vfmadd213pd _clv_6(%rax), %zmm4, %zmm5
        vfmadd213pd _clv_7(%rax), %zmm4, %zmm5
        vfmadd213pd %zmm8, %zmm4, %zmm5

/* T_Rh_Eh_HLLhi = T_Rh_Eh + HLL */
        vaddpd    %zmm5, %zmm9, %zmm13

/* HLLhi = T_Rh_Eh_HLLhi - T_Rh_Eh */
        vsubpd    %zmm9, %zmm13, %zmm10

/* HLL = HLL - HLLhi */
        vsubpd    %zmm10, %zmm5, %zmm3

/* HH = T_Rh_Eh_HLLhi; Lo(HH)&=0xf8000000 */
        vpandq _iHighMask(%rax), %zmm13, %zmm5

/* pH = yH*HH */
        vmulpd    %zmm5, %zmm6, %zmm1

/* HL = T_Rh_Eh_HLLhi-HH */
        vsubpd    %zmm5, %zmm13, %zmm4
        vpsrlq    $32, %zmm1, %zmm14

/* pLL = y*HLL;
   pHH = pH + *(double*)&db2p45_2p44
 */
        vaddpd    _db2p45_2p44(%rax), %zmm1, %zmm10
        vpmovqd   %zmm14, %ymm15
        vpandd    _ABSMASK(%rax), %zmm15, %zmm14{%k1}
        vpcmpd    $5, _DOMAINRANGE(%rax), %zmm14, %k3{%k1}

/* T1 = ((double*)exp2_tbl)[ 2*j ] */
        vpxord    %zmm14, %zmm14, %zmm14
        vpbroadcastd %ecx, %zmm13{%k3}{z}
        vpord     %zmm13, %zmm11, %zmm11{%k1}
        vptestmd  %zmm11, %zmm11, %k0{%k1}

/* pL=yL*HL+yH*HL; pL+=yL*HH */
        vmulpd      %zmm4, %zmm2, %zmm11
        kmovw       %k0, %ecx
        vfmadd213pd %zmm11, %zmm4, %zmm6

/* pHH = pHH - *(double*)&db2p45_2p44 */
        vsubpd    _db2p45_2p44(%rax), %zmm10, %zmm11
        vpmovqd   %zmm10, %ymm4
        movzbl    %cl, %ecx

/* _n = Lo(pHH);
   _n = _n & 0xffffff80;
   _n = _n >> 7;
   Hi(_2n) = (0x3ff+_n)<<20; Lo(_2n) = 0; -> 2^n
 */
        vpslld    $13, %zmm4, %zmm7{%k1}

/* j = Lo(pHH)&0x0000007f */
        vpandd      _jIndexMask(%rax), %zmm4, %zmm9{%k1}
        vfmadd213pd %zmm6, %zmm5, %zmm2

/* pHL = pH - pHH */
        vsubpd    %zmm11, %zmm1, %zmm1
        vpaddd    _iOne(%rax), %zmm7, %zmm7{%k1}

/* t=pL+pLL; t+=pHL */
        vfmadd231pd %zmm12, %zmm3, %zmm2
        vpslld      $4, %zmm9, %zmm9{%k1}
        kxnorw      %k1, %k1, %k1
        vgatherdpd  36416(%rax,%ymm9), %zmm14{%k1}
        vpmovzxdq   %ymm7, %zmm8
        vaddpd      %zmm1, %zmm2, %zmm2
        vmovups     _cev_1(%rax), %zmm1
        vpsllq      $32, %zmm8, %zmm13
        vpternlogq  $168, _ifff0000000000000(%rax), %zmm8, %zmm13
        vfmadd213pd _cev_2(%rax), %zmm2, %zmm1
        vmulpd      %zmm14, %zmm13, %zmm15
        vfmadd213pd _cev_3(%rax), %zmm2, %zmm1
        vmulpd      %zmm2, %zmm15, %zmm3
        vfmadd213pd _cev_4(%rax), %zmm2, %zmm1
        vfmadd213pd _cev_5(%rax), %zmm2, %zmm1
        vfmadd213pd %zmm15, %zmm3, %zmm1
        testl       %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovups   %zmm0, 1152(%rsp)
        vmovups   %zmm12, 1216(%rsp)
        vmovups   %zmm1, 1280(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        kmovw     %k4, 1048(%rsp)
        xorl      %eax, %eax
        kmovw     %k5, 1040(%rsp)
        kmovw     %k6, 1032(%rsp)
        kmovw     %k7, 1024(%rsp)
        vmovups   %zmm16, 960(%rsp)
        vmovups   %zmm17, 896(%rsp)
        vmovups   %zmm18, 832(%rsp)
        vmovups   %zmm19, 768(%rsp)
        vmovups   %zmm20, 704(%rsp)
        vmovups   %zmm21, 640(%rsp)
        vmovups   %zmm22, 576(%rsp)
        vmovups   %zmm23, 512(%rsp)
        vmovups   %zmm24, 448(%rsp)
        vmovups   %zmm25, 384(%rsp)
        vmovups   %zmm26, 320(%rsp)
        vmovups   %zmm27, 256(%rsp)
        vmovups   %zmm28, 192(%rsp)
        vmovups   %zmm29, 128(%rsp)
        vmovups   %zmm30, 64(%rsp)
        vmovups   %zmm31, (%rsp)
        movq      %rsi, 1064(%rsp)
        movq      %rdi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %ecx, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %eax, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        cfi_remember_state

.LBL_1_6:
        btl       %r14d, %r13d
        jc        .LBL_1_12

.LBL_1_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_1_10

.LBL_1_8:
        addb      $1, %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_1_6

        kmovw     1048(%rsp), %k4
        movq      1064(%rsp), %rsi
        kmovw     1040(%rsp), %k5
        movq      1056(%rsp), %rdi
        kmovw     1032(%rsp), %k6
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
        movq      1088(%rsp), %r13
        cfi_restore (%r13)
        kmovw     1024(%rsp), %k7
        vmovups   960(%rsp), %zmm16
        vmovups   896(%rsp), %zmm17
        vmovups   832(%rsp), %zmm18
        vmovups   768(%rsp), %zmm19
        vmovups   704(%rsp), %zmm20
        vmovups   640(%rsp), %zmm21
        vmovups   576(%rsp), %zmm22
        vmovups   512(%rsp), %zmm23
        vmovups   448(%rsp), %zmm24
        vmovups   384(%rsp), %zmm25
        vmovups   320(%rsp), %zmm26
        vmovups   256(%rsp), %zmm27
        vmovups   192(%rsp), %zmm28
        vmovups   128(%rsp), %zmm29
        vmovups   64(%rsp), %zmm30
        vmovups   (%rsp), %zmm31
        movq      1080(%rsp), %r14
        cfi_restore (%r14)
        movq      1072(%rsp), %r15
        cfi_restore (%r15)
        vmovups   1280(%rsp), %zmm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1160(%rsp,%r15), %xmm0
        vmovsd    1224(%rsp,%r15), %xmm1
        call      JUMPTARGET(pow)
        vmovsd    %xmm0, 1288(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1152(%rsp,%r15), %xmm0
        vmovsd    1216(%rsp,%r15), %xmm1
        call      JUMPTARGET(pow)
        vmovsd    %xmm0, 1280(%rsp,%r15)
        jmp       .LBL_1_7

#endif
END (_ZGVeN8vv_pow_knl)

ENTRY (_ZGVeN8vv_pow_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_ff _ZGVdN4vv_pow
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        vpsrlq    $32, %zmm0, %zmm10
        kxnorw    %k1, %k1, %k1
        kxnorw    %k2, %k2, %k2
        kxnorw    %k3, %k3, %k3
        vpmovqd   %zmm10, %ymm7
        movq      __svml_dpow_data@GOTPCREL(%rip), %rax
        vmovaps   %zmm1, %zmm6
        vpsrlq    $32, %zmm6, %zmm13

/* i = (((Hi(x) & 0x000ffe00) + 0x00000200) >> 10); -> i = (b1..b11 + 1) / 2 */
        vpand _iIndexMask(%rax), %ymm7, %ymm15
        vpaddd _HIDELTA(%rax), %ymm7, %ymm2

/* k = Hi(x); k = k - 0x3fe7fe00; k = k >> 20 */
        vpsubd _i3fe7fe00(%rax), %ymm7, %ymm7
        vmovdqu _ABSMASK(%rax), %ymm4
        vmovdqu _LORANGE(%rax), %ymm3

/* x1 = x; Hi(x1) = (Hi(x1)&0x000fffff)|0x3ff00000 */
        vmovups _dbOne(%rax), %zmm11
        vmovdqu _INF(%rax), %ymm5
        vpaddd _iIndexAdd(%rax), %ymm15, %ymm12
        vpmovqd   %zmm13, %ymm14
        vpternlogq $248, _iMantissaMask(%rax), %zmm0, %zmm11
        vpsrld    $10, %ymm12, %ymm10
        vpsrad    $20, %ymm7, %ymm13

/* Index for reciprocal table */
        vpslld    $3, %ymm10, %ymm8

/* Index for log2 table */
        vpslld    $4, %ymm10, %ymm1
        vpcmpgtd  %ymm2, %ymm3, %ymm3
        vpand     %ymm4, %ymm14, %ymm2
        vpaddd _i2p20_2p19(%rax), %ymm13, %ymm14
        vpmovzxdq %ymm14, %zmm15
        vpsllq    $32, %zmm15, %zmm7
        vpternlogq $168, _iffffffff00000000(%rax), %zmm15, %zmm7
        vsubpd _db2p20_2p19(%rax), %zmm7, %zmm13
        vpxord    %zmm9, %zmm9, %zmm9
        vgatherdpd 11712(%rax,%ymm8), %zmm9{%k1}

/* T1 = ((double*)exp2_tbl)[ 2*j ] */
        kxnorw    %k1, %k1, %k1
        vpxord    %zmm12, %zmm12, %zmm12
        vpxord    %zmm8, %zmm8, %zmm8
        vgatherdpd 19968(%rax,%ymm1), %zmm12{%k2}
        vgatherdpd 19976(%rax,%ymm1), %zmm8{%k3}
        vmovups _iHighMask(%rax), %zmm1

/* x1Hi=x1; Lo(x1Hi)&=0xf8000000; x1Lo = x1-x1Hi */
        vandpd    %zmm1, %zmm11, %zmm10
        vsubpd    %zmm10, %zmm11, %zmm15

/* r1 = x1*rcp1 */
        vmulpd    %zmm9, %zmm11, %zmm11

/* E = -r1+__fence(x1Hi*rcp1) */
        vfmsub213pd %zmm11, %zmm9, %zmm10

/* cq = c+r1 */
        vaddpd _LHN(%rax), %zmm11, %zmm14

/* E=E+x1Lo*rcp1 */
        vfmadd213pd %zmm10, %zmm9, %zmm15

/* T = k + L1hi */
        vaddpd    %zmm12, %zmm13, %zmm9

/* T_Rh = T + cq */
        vaddpd    %zmm14, %zmm9, %zmm11

/* T_Rh_Eh = T_Rh + E */
        vaddpd    %zmm15, %zmm11, %zmm13

/* Rl = T-T_Rh; -> -Rh */
        vsubpd    %zmm11, %zmm9, %zmm12

/* HLL = T_Rh - T_Rh_Eh; -> -Eh */
        vsubpd    %zmm13, %zmm11, %zmm9

/* Rl=Rl+cq */
        vaddpd    %zmm12, %zmm14, %zmm10

/* HLL+=E;  -> El */
        vaddpd    %zmm9, %zmm15, %zmm7

/* HLL+=Rl */
        vaddpd    %zmm10, %zmm7, %zmm12

/* 2^(y*(HH+HL+HLL)) starts here:
   yH = y; Lo(yH)&=0xf8000000
 */
        vandpd    %zmm1, %zmm6, %zmm7

/* HLL+=L1lo */
        vaddpd    %zmm8, %zmm12, %zmm12

/* cq = cq + E */
        vaddpd    %zmm15, %zmm14, %zmm8
        vmovups _clv_2(%rax), %zmm14

/* HLL = HLL + (((((((a7)*cq+a6)*cq+a5)*cq+a4)*cq+a3)*cq+a2)*cq+a1)*cq */
        vfmadd213pd _clv_3(%rax), %zmm8, %zmm14
        vfmadd213pd _clv_4(%rax), %zmm8, %zmm14
        vfmadd213pd _clv_5(%rax), %zmm8, %zmm14
        vfmadd213pd _clv_6(%rax), %zmm8, %zmm14
        vfmadd213pd _clv_7(%rax), %zmm8, %zmm14
        vfmadd213pd %zmm12, %zmm8, %zmm14

/* yL = y-yH */
        vsubpd    %zmm7, %zmm6, %zmm8

/* T_Rh_Eh_HLLhi = T_Rh_Eh + HLL */
        vaddpd    %zmm14, %zmm13, %zmm15

/* HH = T_Rh_Eh_HLLhi; Lo(HH)&=0xf8000000 */
        vandpd    %zmm1, %zmm15, %zmm11

/* HLLhi = T_Rh_Eh_HLLhi - T_Rh_Eh */
        vsubpd    %zmm13, %zmm15, %zmm13

/* pH = yH*HH */
        vmulpd    %zmm11, %zmm7, %zmm9

/* HLL = HLL - HLLhi */
        vsubpd    %zmm13, %zmm14, %zmm12

/* HL = T_Rh_Eh_HLLhi-HH */
        vsubpd    %zmm11, %zmm15, %zmm10
        vpsrlq    $32, %zmm9, %zmm1
        vmovdqu _DOMAINRANGE(%rax), %ymm13
        vpmovqd   %zmm1, %ymm1
        vpand     %ymm4, %ymm1, %ymm1
        vpcmpgtd  %ymm5, %ymm2, %ymm4
        vpcmpeqd  %ymm5, %ymm2, %ymm5
        vpternlogd $254, %ymm5, %ymm4, %ymm3
        vpcmpgtd  %ymm13, %ymm1, %ymm2
        vpcmpeqd  %ymm13, %ymm1, %ymm4
        vpternlogd $254, %ymm4, %ymm2, %ymm3

/* pLL = y*HLL */
        vmovups _db2p45_2p44(%rax), %zmm2

/* pHH = pH + *(double*)&db2p45_2p44 */
        vaddpd    %zmm2, %zmm9, %zmm1
        vpmovqd   %zmm1, %ymm5

/* j = Lo(pHH)&0x0000007f */
        vpand _jIndexMask(%rax), %ymm5, %ymm14
        vpslld    $4, %ymm14, %ymm15
        vmovmskps %ymm3, %ecx

/* pL=yL*HL+yH*HL; pL+=yL*HH */
        vmulpd    %zmm10, %zmm8, %zmm3
        vfmadd213pd %zmm3, %zmm10, %zmm7
        vfmadd213pd %zmm7, %zmm11, %zmm8

/* _n = Lo(pHH)
   _n = _n & 0xffffff80
   _n = _n >> 7
   Hi(_2n) = (0x3ff+_n)<<20; Lo(_2n) = 0; -> 2^n
 */
        vpslld    $13, %ymm5, %ymm7

/* t=pL+pLL; t+=pHL */
        vfmadd231pd %zmm6, %zmm12, %zmm8
        vpaddd _iOne(%rax), %ymm7, %ymm10
        vpmovzxdq %ymm10, %zmm11
        vpsllq    $32, %zmm11, %zmm3
        vpternlogq $168, _ifff0000000000000(%rax), %zmm11, %zmm3

/* pHH = pHH - *(double*)&db2p45_2p44 */
        vsubpd    %zmm2, %zmm1, %zmm11
        vmovups _cev_1(%rax), %zmm2

/* pHL = pH - pHH */
        vsubpd    %zmm11, %zmm9, %zmm9
        vaddpd    %zmm9, %zmm8, %zmm8
        vfmadd213pd _cev_2(%rax), %zmm8, %zmm2
        vfmadd213pd _cev_3(%rax), %zmm8, %zmm2
        vfmadd213pd _cev_4(%rax), %zmm8, %zmm2
        vfmadd213pd _cev_5(%rax), %zmm8, %zmm2
        vpxord    %zmm4, %zmm4, %zmm4
        vgatherdpd 36416(%rax,%ymm15), %zmm4{%k1}
        vmulpd    %zmm4, %zmm3, %zmm1
        vmulpd    %zmm8, %zmm1, %zmm12
        vfmadd213pd %zmm1, %zmm12, %zmm2
        testl     %ecx, %ecx
        jne       .LBL_2_3

.LBL_2_2:
        cfi_remember_state
        vmovaps   %zmm2, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_2_3:
        cfi_restore_state
        vmovups   %zmm0, 1152(%rsp)
        vmovups   %zmm6, 1216(%rsp)
        vmovups   %zmm2, 1280(%rsp)
        je        .LBL_2_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        kmovw     %k4, 1048(%rsp)
        kmovw     %k5, 1040(%rsp)
        kmovw     %k6, 1032(%rsp)
        kmovw     %k7, 1024(%rsp)
        vmovups   %zmm16, 960(%rsp)
        vmovups   %zmm17, 896(%rsp)
        vmovups   %zmm18, 832(%rsp)
        vmovups   %zmm19, 768(%rsp)
        vmovups   %zmm20, 704(%rsp)
        vmovups   %zmm21, 640(%rsp)
        vmovups   %zmm22, 576(%rsp)
        vmovups   %zmm23, 512(%rsp)
        vmovups   %zmm24, 448(%rsp)
        vmovups   %zmm25, 384(%rsp)
        vmovups   %zmm26, 320(%rsp)
        vmovups   %zmm27, 256(%rsp)
        vmovups   %zmm28, 192(%rsp)
        vmovups   %zmm29, 128(%rsp)
        vmovups   %zmm30, 64(%rsp)
        vmovups   %zmm31, (%rsp)
        movq      %rsi, 1064(%rsp)
        movq      %rdi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %ecx, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %eax, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        cfi_remember_state

.LBL_2_6:
        btl       %r14d, %r13d
        jc        .LBL_2_12

.LBL_2_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_2_10

.LBL_2_8:
        incb      %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_2_6

        kmovw     1048(%rsp), %k4
        kmovw     1040(%rsp), %k5
        kmovw     1032(%rsp), %k6
        kmovw     1024(%rsp), %k7
        vmovups   960(%rsp), %zmm16
        vmovups   896(%rsp), %zmm17
        vmovups   832(%rsp), %zmm18
        vmovups   768(%rsp), %zmm19
        vmovups   704(%rsp), %zmm20
        vmovups   640(%rsp), %zmm21
        vmovups   576(%rsp), %zmm22
        vmovups   512(%rsp), %zmm23
        vmovups   448(%rsp), %zmm24
        vmovups   384(%rsp), %zmm25
        vmovups   320(%rsp), %zmm26
        vmovups   256(%rsp), %zmm27
        vmovups   192(%rsp), %zmm28
        vmovups   128(%rsp), %zmm29
        vmovups   64(%rsp), %zmm30
        vmovups   (%rsp), %zmm31
        vmovups   1280(%rsp), %zmm2
        movq      1064(%rsp), %rsi
        movq      1056(%rsp), %rdi
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
        movq      1088(%rsp), %r13
        cfi_restore (%r13)
        movq      1080(%rsp), %r14
        cfi_restore (%r14)
        movq      1072(%rsp), %r15
        cfi_restore (%r15)
        jmp       .LBL_2_2

.LBL_2_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1224(%rsp,%r15), %xmm1
        vzeroupper
        vmovsd    1160(%rsp,%r15), %xmm0

        call      JUMPTARGET(pow)

        vmovsd    %xmm0, 1288(%rsp,%r15)
        jmp       .LBL_2_8

.LBL_2_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1216(%rsp,%r15), %xmm1
        vzeroupper
        vmovsd    1152(%rsp,%r15), %xmm0

        call      JUMPTARGET(pow)

        vmovsd    %xmm0, 1280(%rsp,%r15)
        jmp       .LBL_2_7

#endif
END (_ZGVeN8vv_pow_skx)
