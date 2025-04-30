/* Function powf vectorized with AVX-512. KNL and SKX versions.
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
#include "svml_s_wrapper_impl.h"

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

	.text
ENTRY (_ZGVeN16vv_powf_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_ff _ZGVdN8vv_powf
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        movq      __svml_spow_data@GOTPCREL(%rip), %rdx
        vmovaps   %zmm1, %zmm9
        vshuff32x4 $238, %zmm0, %zmm0, %zmm7
        kxnorw    %k3, %k3, %k3
        vcvtps2pd %ymm0, %zmm14
        vcvtps2pd %ymm7, %zmm10
        movl      $-1, %eax
        movq      $-1, %rcx
        vpandd    _ABSMASK(%rdx), %zmm9, %zmm4
        vmovups   _ExpMask(%rdx), %zmm6

/* exponent bits selection */
        vpsrlq    $20, %zmm14, %zmm13
        vshuff32x4 $238, %zmm9, %zmm9, %zmm8
        vpcmpd    $5, _INF(%rdx), %zmm4, %k2
        vpsrlq    $32, %zmm13, %zmm15
        vcvtps2pd %ymm8, %zmm2
        vmovups   _Two10(%rdx), %zmm4
        vpmovqd   %zmm15, %ymm12
        vcvtps2pd %ymm9, %zmm1
        vpsubd    _NMINNORM(%rdx), %zmm0, %zmm3
        vpbroadcastd %eax, %zmm8{%k2}{z}
        vpcmpd    $5, _NMAXVAL(%rdx), %zmm3, %k1

/* preserve mantissa, set input exponent to 2^(-10) */
        vmovaps   %zmm6, %zmm3
        vpternlogq $248, %zmm6, %zmm10, %zmm4
        vpsrlq    $20, %zmm10, %zmm10
        vpternlogq $234, _Two10(%rdx), %zmm14, %zmm3

/* reciprocal approximation good to at least 11 bits */
        vrcp28pd  %zmm4, %zmm11
        vpsrlq    $32, %zmm10, %zmm14
        vpbroadcastd %eax, %zmm7{%k1}{z}
        kxnorw    %k1, %k1, %k1
        vrcp28pd  %zmm3, %zmm5
        vpmovqd   %zmm14, %ymm6
        vshufi32x4 $68, %zmm6, %zmm12, %zmm13
        vmovups   _One(%rdx), %zmm6

/* round reciprocal to nearest integer, will have 1+9 mantissa bits */
        vrndscalepd $8, %zmm5, %zmm14

/* biased exponent in DP format */
        vshuff32x4 $238, %zmm13, %zmm13, %zmm5
        vrndscalepd $8, %zmm11, %zmm11
        vcmppd    $30, _Threshold(%rdx), %zmm14, %k2
        vcvtdq2pd %ymm13, %zmm10
        vcvtdq2pd %ymm5, %zmm15

/* table lookup */
        vpsrlq    $40, %zmm14, %zmm13
        vpxord    %zmm5, %zmm5, %zmm5
        vgatherqpd _Log2Rcp_lookup(%rdx,%zmm13), %zmm5{%k3}
        vfmsub213pd %zmm6, %zmm14, %zmm3
        vfmsub213pd %zmm6, %zmm11, %zmm4
        vcmppd    $30, _Threshold(%rdx), %zmm11, %k3
        vpbroadcastq %rcx, %zmm14{%k2}{z}

/* dpP= _dbT+lJ*T_ITEM_GRAN */
        kxnorw    %k2, %k2, %k2
        vpsrlq    $40, %zmm11, %zmm12
        vpxord    %zmm6, %zmm6, %zmm6
        vpbroadcastq %rcx, %zmm11{%k3}{z}
        kxnorw    %k3, %k3, %k3
        vgatherqpd _Log2Rcp_lookup(%rdx,%zmm12), %zmm6{%k1}
        vmovups   _Bias1(%rdx), %zmm12
        vpternlogq $236, _Bias(%rdx), %zmm12, %zmm14
        vpternlogq $248, _Bias(%rdx), %zmm11, %zmm12
        vsubpd    %zmm14, %zmm10, %zmm13
        vsubpd    %zmm12, %zmm15, %zmm10
        vmovups   _poly_coeff_3(%rdx), %zmm11
        vmovups   _poly_coeff_4(%rdx), %zmm15
        vfmadd213pd %zmm15, %zmm4, %zmm11
        vmulpd    %zmm4, %zmm4, %zmm12
        vmovaps   %zmm15, %zmm14
        vmulpd    %zmm3, %zmm3, %zmm15
        vfmadd231pd _poly_coeff_3(%rdx), %zmm3, %zmm14

/* reconstruction */
        vfmadd213pd %zmm4, %zmm12, %zmm11
        vfmadd213pd %zmm3, %zmm15, %zmm14
        vaddpd    %zmm6, %zmm11, %zmm11
        vaddpd    %zmm5, %zmm14, %zmm3
        vfmadd231pd _L2(%rdx), %zmm10, %zmm11
        vfmadd132pd _L2(%rdx), %zmm3, %zmm13
        vmulpd    %zmm2, %zmm11, %zmm12
        vmulpd    %zmm1, %zmm13, %zmm10
        vmulpd    __dbInvLn2(%rdx), %zmm12, %zmm6

/* hi bits */
        vpsrlq    $32, %zmm12, %zmm12
        vmulpd    __dbInvLn2(%rdx), %zmm10, %zmm1

/* to round down; if dR is an integer we will get R = 1, which is ok */
        vsubpd    __dbHALF(%rdx), %zmm6, %zmm4
        vpsrlq    $32, %zmm10, %zmm11
        vpmovqd   %zmm11, %ymm3
        vsubpd    __dbHALF(%rdx), %zmm1, %zmm2
        vaddpd    __dbShifter(%rdx), %zmm4, %zmm14
        vpmovqd   %zmm12, %ymm4
        vshufi32x4 $68, %zmm4, %zmm3, %zmm5
        vpxord    %zmm4, %zmm4, %zmm4
        vaddpd    __dbShifter(%rdx), %zmm2, %zmm2

/* iAbsX = iAbsX&iAbsMask; */
        vpandd    __iAbsMask(%rdx), %zmm5, %zmm11
        vpxord    %zmm5, %zmm5, %zmm5
        vsubpd    __dbShifter(%rdx), %zmm14, %zmm13

/* iRangeMask = (iAbsX>iDomainRange) */
        vpcmpgtd     __iDomainRange(%rdx), %zmm11, %k1
        vsubpd       __dbShifter(%rdx), %zmm2, %zmm15
        vpbroadcastd %eax, %zmm10{%k1}{z}
        vpternlogd   $254, %zmm8, %zmm7, %zmm10

/* [0..1) */
        vsubpd    %zmm15, %zmm1, %zmm1

/* low K bits */
        vpandq    __lbLOWKBITS(%rdx), %zmm14, %zmm11
        vgatherqpd 13952(%rdx,%zmm11,8), %zmm5{%k3}
        vsubpd    %zmm13, %zmm6, %zmm7
        vptestmd  %zmm10, %zmm10, %k0
        vpandq    __lbLOWKBITS(%rdx), %zmm2, %zmm10
        vmulpd    __dbC1(%rdx), %zmm1, %zmm1
        vmulpd    __dbC1(%rdx), %zmm7, %zmm3
        vpsrlq    $11, %zmm2, %zmm8
        vpsrlq    $11, %zmm14, %zmm2

/* NB : including +/- sign for the exponent!! */
        vpsllq    $52, %zmm8, %zmm8
        kmovw     %k0, %ecx
        vpsllq    $52, %zmm2, %zmm6
        vfmadd213pd %zmm5, %zmm3, %zmm5
        vgatherqpd 13952(%rdx,%zmm10,8), %zmm4{%k2}
        vfmadd213pd %zmm4, %zmm1, %zmm4
        vpaddq    %zmm6, %zmm5, %zmm10
        vcvtpd2ps %zmm10, %ymm12
        vpaddq    %zmm8, %zmm4, %zmm7
        vcvtpd2ps %zmm7, %ymm11
        vshuff32x4 $68, %zmm12, %zmm11, %zmm1
        testl     %ecx, %ecx
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
        vmovups   %zmm9, 1216(%rsp)
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
        vmovss    1156(%rsp,%r15,8), %xmm0
        vmovss    1220(%rsp,%r15,8), %xmm1
        call      JUMPTARGET(powf)
        vmovss    %xmm0, 1284(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        vmovss    1216(%rsp,%r15,8), %xmm1
        call      JUMPTARGET(powf)
        vmovss    %xmm0, 1280(%rsp,%r15,8)
        jmp       .LBL_1_7
#endif
END (_ZGVeN16vv_powf_knl)

ENTRY (_ZGVeN16vv_powf_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_ff _ZGVdN8vv_powf
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        movq      __svml_spow_data@GOTPCREL(%rip), %rax
        vextractf32x8 $1, %zmm1, %ymm14
        vextractf32x8 $1, %zmm0, %ymm15
        vpsubd _NMINNORM(%rax), %zmm0, %zmm9
        vmovups   %zmm26, 1280(%rsp)
        vmovups _ExpMask(%rax), %zmm6
        vpcmpd    $1, _NMAXVAL(%rax), %zmm9, %k1
        vcvtps2pd %ymm0, %zmm5
        vcvtps2pd %ymm1, %zmm12
        kxnorw    %k3, %k3, %k3

/* exponent bits selection */
        vpsrlq    $20, %zmm5, %zmm3
        vpsrlq    $32, %zmm3, %zmm2
        vpmovqd   %zmm2, %ymm11
        vcvtps2pd %ymm14, %zmm13
        vmovups   .L_2il0floatpacket.23(%rip), %zmm14
        vmovaps   %zmm14, %zmm26
        vpandd _ABSMASK(%rax), %zmm1, %zmm8
        vpcmpd    $1, _INF(%rax), %zmm8, %k2
        vpandnd   %zmm9, %zmm9, %zmm26{%k1}
        vmovups _Two10(%rax), %zmm9
        kxnorw    %k1, %k1, %k1
        vcvtps2pd %ymm15, %zmm4
        vmovaps   %zmm14, %zmm15

/* preserve mantissa, set input exponent to 2^(-10) */
        vpternlogq $248, %zmm6, %zmm4, %zmm9
        vpsrlq    $20, %zmm4, %zmm4

/* reciprocal approximation good to at least 11 bits */
        vrcp14pd  %zmm9, %zmm10

/* round reciprocal to nearest integer, will have 1+9 mantissa bits */
        vrndscalepd $8, %zmm10, %zmm3
        vmovups _One(%rax), %zmm10
        vfmsub213pd %zmm10, %zmm3, %zmm9
        vpandnd   %zmm8, %zmm8, %zmm15{%k2}
        vmovaps   %zmm6, %zmm8
        vpternlogq $234, _Two10(%rax), %zmm5, %zmm8
        vpsrlq    $32, %zmm4, %zmm5
        vrcp14pd  %zmm8, %zmm7
        vpmovqd   %zmm5, %ymm6
        vrndscalepd $8, %zmm7, %zmm2
        vfmsub213pd %zmm10, %zmm2, %zmm8

/* table lookup */
        vpsrlq    $40, %zmm2, %zmm10
        vinserti32x8 $1, %ymm6, %zmm11, %zmm4
        vpsrlq    $40, %zmm3, %zmm11

/* biased exponent in DP format */
        vextracti32x8 $1, %zmm4, %ymm7
        vcvtdq2pd %ymm4, %zmm6
        vpmovqd   %zmm10, %ymm4
        vpmovqd   %zmm11, %ymm5
        vpxord    %zmm10, %zmm10, %zmm10
        vgatherdpd _Log2Rcp_lookup(%rax,%ymm4), %zmm10{%k3}
        vpbroadcastq .L_2il0floatpacket.24(%rip), %zmm4
        vpxord    %zmm11, %zmm11, %zmm11
        vcvtdq2pd %ymm7, %zmm7
        vgatherdpd _Log2Rcp_lookup(%rax,%ymm5), %zmm11{%k1}
        vmovups _Threshold(%rax), %zmm5
        vcmppd    $21, %zmm2, %zmm5, %k2
        vcmppd    $21, %zmm3, %zmm5, %k3
        vmovups _Bias1(%rax), %zmm3
        vmovaps   %zmm4, %zmm2
        vpandnq   %zmm5, %zmm5, %zmm2{%k2}
        vpternlogq $236, _Bias(%rax), %zmm3, %zmm2

/* dpP= _dbT+lJ*T_ITEM_GRAN */
        kxnorw    %k2, %k2, %k2
        vpandnq   %zmm5, %zmm5, %zmm4{%k3}
        vpternlogq $248, _Bias(%rax), %zmm4, %zmm3
        vsubpd    %zmm2, %zmm6, %zmm4
        vmovups _poly_coeff_3(%rax), %zmm6
        vmovups _poly_coeff_4(%rax), %zmm2
        vsubpd    %zmm3, %zmm7, %zmm5
        vmulpd    %zmm8, %zmm8, %zmm7
        vfmadd213pd %zmm2, %zmm9, %zmm6
        kxnorw    %k3, %k3, %k3
        vmovaps   %zmm2, %zmm3
        vmulpd    %zmm9, %zmm9, %zmm2
        vfmadd231pd _poly_coeff_3(%rax), %zmm8, %zmm3

/* reconstruction */
        vfmadd213pd %zmm9, %zmm2, %zmm6
        vfmadd213pd %zmm8, %zmm7, %zmm3
        vaddpd    %zmm11, %zmm6, %zmm8
        vaddpd    %zmm10, %zmm3, %zmm9
        vfmadd231pd _L2(%rax), %zmm5, %zmm8
        vfmadd132pd _L2(%rax), %zmm9, %zmm4
        vmulpd    %zmm13, %zmm8, %zmm13
        vmulpd    %zmm12, %zmm4, %zmm3
        vmulpd __dbInvLn2(%rax), %zmm13, %zmm10
        vmulpd __dbInvLn2(%rax), %zmm3, %zmm8

/* hi bits */
        vpsrlq    $32, %zmm3, %zmm4
        vpsrlq    $32, %zmm13, %zmm13

/* to round down; if dR is an integer we will get R = 1, which is ok */
        vsubpd __dbHALF(%rax), %zmm8, %zmm12
        vpmovqd   %zmm4, %ymm5
        vpmovqd   %zmm13, %ymm2
        vsubpd __dbHALF(%rax), %zmm10, %zmm9
        vaddpd __dbShifter(%rax), %zmm12, %zmm7
        vaddpd __dbShifter(%rax), %zmm9, %zmm9
        vsubpd __dbShifter(%rax), %zmm7, %zmm11
        vsubpd __dbShifter(%rax), %zmm9, %zmm12
        vinserti32x8 $1, %ymm2, %zmm5, %zmm3

/* iAbsX = iAbsX&iAbsMask */
        vpandd __iAbsMask(%rax), %zmm3, %zmm4

/* iRangeMask = (iAbsX>iDomainRange) */
        vpcmpd    $2, __iDomainRange(%rax), %zmm4, %k1
        vpandnd   %zmm4, %zmm4, %zmm14{%k1}
        vpternlogd $254, %zmm15, %zmm26, %zmm14

/* [0..1) */
        vsubpd    %zmm11, %zmm8, %zmm15
        vsubpd    %zmm12, %zmm10, %zmm26
        vptestmd  %zmm14, %zmm14, %k0
        vpsrlq    $11, %zmm7, %zmm8
        vpsrlq    $11, %zmm9, %zmm10
        vmulpd __dbC1(%rax), %zmm26, %zmm26
        vmulpd __dbC1(%rax), %zmm15, %zmm15

/* NB : including +/- sign for the exponent!! */
        vpsllq    $52, %zmm10, %zmm13
        vpsllq    $52, %zmm8, %zmm12
        kmovw     %k0, %ecx

/* low K bits */
        vpandq __lbLOWKBITS(%rax), %zmm9, %zmm14
        vpandq __lbLOWKBITS(%rax), %zmm7, %zmm6
        vpmovqd   %zmm14, %ymm7
        vpmovqd   %zmm6, %ymm9
        vpxord    %zmm2, %zmm2, %zmm2
        vgatherdpd 13952(%rax,%ymm7,8), %zmm2{%k3}
        vfmadd213pd %zmm2, %zmm26, %zmm2
        vpaddq    %zmm13, %zmm2, %zmm2
        vcvtpd2ps %zmm2, %ymm4
        vpxord    %zmm11, %zmm11, %zmm11
        vgatherdpd 13952(%rax,%ymm9,8), %zmm11{%k2}
        vfmadd213pd %zmm11, %zmm15, %zmm11
        vpaddq    %zmm12, %zmm11, %zmm3
        vcvtpd2ps %zmm3, %ymm5
        vinsertf32x8 $1, %ymm4, %zmm5, %zmm2
        testl     %ecx, %ecx
        jne       .LBL_2_3

.LBL_2_2:
        cfi_remember_state
        vmovups   1280(%rsp), %zmm26
        vmovaps   %zmm2, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_2_3:
        cfi_restore_state
        vmovups   %zmm0, 1088(%rsp)
        vmovups   %zmm1, 1152(%rsp)
        vmovups   %zmm2, 1216(%rsp)
        je        .LBL_2_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        kmovw     %k4, 984(%rsp)
        kmovw     %k5, 976(%rsp)
        kmovw     %k6, 968(%rsp)
        kmovw     %k7, 960(%rsp)
        vmovups   %zmm16, 896(%rsp)
        vmovups   %zmm17, 832(%rsp)
        vmovups   %zmm18, 768(%rsp)
        vmovups   %zmm19, 704(%rsp)
        vmovups   %zmm20, 640(%rsp)
        vmovups   %zmm21, 576(%rsp)
        vmovups   %zmm22, 512(%rsp)
        vmovups   %zmm23, 448(%rsp)
        vmovups   %zmm24, 384(%rsp)
        vmovups   %zmm25, 320(%rsp)
        vmovups   %zmm27, 256(%rsp)
        vmovups   %zmm28, 192(%rsp)
        vmovups   %zmm29, 128(%rsp)
        vmovups   %zmm30, 64(%rsp)
        vmovups   %zmm31, (%rsp)
        movq      %rsi, 1000(%rsp)
        movq      %rdi, 992(%rsp)
        movq      %r12, 1032(%rsp)
        cfi_offset_rel_rsp (12, 1032)
        movb      %dl, %r12b
        movq      %r13, 1024(%rsp)
        cfi_offset_rel_rsp (13, 1024)
        movl      %ecx, %r13d
        movq      %r14, 1016(%rsp)
        cfi_offset_rel_rsp (14, 1016)
        movl      %eax, %r14d
        movq      %r15, 1008(%rsp)
        cfi_offset_rel_rsp (15, 1008)
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

        kmovw     984(%rsp), %k4
        kmovw     976(%rsp), %k5
        kmovw     968(%rsp), %k6
        kmovw     960(%rsp), %k7
        vmovups   896(%rsp), %zmm16
        vmovups   832(%rsp), %zmm17
        vmovups   768(%rsp), %zmm18
        vmovups   704(%rsp), %zmm19
        vmovups   640(%rsp), %zmm20
        vmovups   576(%rsp), %zmm21
        vmovups   512(%rsp), %zmm22
        vmovups   448(%rsp), %zmm23
        vmovups   384(%rsp), %zmm24
        vmovups   320(%rsp), %zmm25
        vmovups   256(%rsp), %zmm27
        vmovups   192(%rsp), %zmm28
        vmovups   128(%rsp), %zmm29
        vmovups   64(%rsp), %zmm30
        vmovups   (%rsp), %zmm31
        vmovups   1216(%rsp), %zmm2
        movq      1000(%rsp), %rsi
        movq      992(%rsp), %rdi
        movq      1032(%rsp), %r12
        cfi_restore (%r12)
        movq      1024(%rsp), %r13
        cfi_restore (%r13)
        movq      1016(%rsp), %r14
        cfi_restore (%r14)
        movq      1008(%rsp), %r15
        cfi_restore (%r15)
        jmp       .LBL_2_2

.LBL_2_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        vmovss    1156(%rsp,%r15,8), %xmm1
        vzeroupper
        vmovss    1092(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(powf)
        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_2_8

.LBL_2_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm1
        vzeroupper
        vmovss    1088(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(powf)
        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_2_7
#endif
END (_ZGVeN16vv_powf_skx)

	.section .rodata, "a"
.L_2il0floatpacket.23:
	.long	0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.23,@object
.L_2il0floatpacket.24:
	.long	0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.24,@object
