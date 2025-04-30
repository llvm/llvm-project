/* Function logf vectorized with AVX-512. KNL and SKX versions.
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
#include "svml_s_logf_data.h"
#include "svml_s_wrapper_impl.h"

	.text
ENTRY (_ZGVeN16v_logf_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_logf
#else
/*
   ALGORITHM DESCRIPTION:

     log(x) = exponent_x*log(2) + log(mantissa_x),         if mantissa_x<4/3
     log(x) = (exponent_x+1)*log(2) + log(0.5*mantissa_x), if mantissa_x>4/3

     R = mantissa_x - 1,     if mantissa_x<4/3
     R = 0.5*mantissa_x - 1, if mantissa_x>4/3
     |R|< 1/3

     log(1+R) is approximated as a polynomial: degree 9 for 1-ulp,
     degree 7 for 4-ulp, degree 3 for half-precision.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1280, %rsp
        movq      __svml_slog_data@GOTPCREL(%rip), %rax
        movl      $-1, %ecx

/* reduction: compute r,n */
        vpsubd    _iBrkValue(%rax), %zmm0, %zmm2
        vmovups   _sPoly_7(%rax), %zmm7
        vpandd    _iOffExpoMask(%rax), %zmm2, %zmm3

/* exponent_x (mantissa_x<4/3) or exponent_x+1 (mantissa_x>4/3) */
        vpsrad    $23, %zmm2, %zmm4

/* check for working range,
   set special argument mask (denormals/zero/Inf/NaN)
 */
        vpaddd    _iHiDelta(%rax), %zmm0, %zmm1

/* mantissa_x (mantissa_x<4/3), or 0.5*mantissa_x (mantissa_x>4/3) */
        vpaddd    _iBrkValue(%rax), %zmm3, %zmm6
        vpcmpd    $1, _iLoRange(%rax), %zmm1, %k1
        vcvtdq2ps {rn-sae}, %zmm4, %zmm1

/* reduced argument R */
        vsubps       _sOne(%rax), %zmm6, %zmm8
        vpbroadcastd %ecx, %zmm5{%k1}{z}

/* polynomial evaluation starts here */
        vfmadd213ps _sPoly_6(%rax), %zmm8, %zmm7
        vptestmd    %zmm5, %zmm5, %k0
        kmovw       %k0, %ecx
        vfmadd213ps _sPoly_5(%rax), %zmm8, %zmm7
        vfmadd213ps _sPoly_4(%rax), %zmm8, %zmm7
        vfmadd213ps _sPoly_3(%rax), %zmm8, %zmm7
        vfmadd213ps _sPoly_2(%rax), %zmm8, %zmm7
        vfmadd213ps _sPoly_1(%rax), %zmm8, %zmm7
        vmulps      %zmm8, %zmm7, %zmm9

/* polynomial evaluation end */
        vfmadd213ps %zmm8, %zmm8, %zmm9

/*
   final reconstruction:
   add exponent_value*log2 to polynomial result
 */
        vfmadd132ps _sLn2(%rax), %zmm9, %zmm1
        testl       %ecx, %ecx
        jne         .LBL_1_3

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
        vmovups   %zmm1, 1216(%rsp)
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
        vmovups   1216(%rsp), %zmm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        vmovss    1156(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(logf)
        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(logf)
        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_1_7
#endif
END (_ZGVeN16v_logf_knl)

ENTRY (_ZGVeN16v_logf_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_logf
#else
/*
   ALGORITHM DESCRIPTION:

     log(x) = exponent_x*log(2) + log(mantissa_x),         if mantissa_x<4/3
     log(x) = (exponent_x+1)*log(2) + log(0.5*mantissa_x), if mantissa_x>4/3

     R = mantissa_x - 1,     if mantissa_x<4/3
     R = 0.5*mantissa_x - 1, if mantissa_x>4/3
     |R|< 1/3

     log(1+R) is approximated as a polynomial: degree 9 for 1-ulp,
     degree 7 for 4-ulp, degree 3 for half-precision.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1280, %rsp
        movq      __svml_slog_data@GOTPCREL(%rip), %rax
        vmovups   .L_2il0floatpacket.7(%rip), %zmm6
        vmovups _iBrkValue(%rax), %zmm4
        vmovups _sPoly_7(%rax), %zmm8

/*
   check for working range,
   set special argument mask (denormals/zero/Inf/NaN)
 */
        vpaddd _iHiDelta(%rax), %zmm0, %zmm1

/* reduction: compute r,n */
        vpsubd    %zmm4, %zmm0, %zmm2
        vpcmpd    $5, _iLoRange(%rax), %zmm1, %k1

/* exponent_x (mantissa_x<4/3) or exponent_x+1 (mantissa_x>4/3) */
        vpsrad    $23, %zmm2, %zmm5
        vpandd _iOffExpoMask(%rax), %zmm2, %zmm3

/* mantissa_x (mantissa_x<4/3), or 0.5*mantissa_x (mantissa_x>4/3) */
        vpaddd    %zmm4, %zmm3, %zmm7

/* reduced argument R */
        vsubps _sOne(%rax), %zmm7, %zmm9

/* polynomial evaluation starts here */
        vfmadd213ps _sPoly_6(%rax), %zmm9, %zmm8
        vfmadd213ps _sPoly_5(%rax), %zmm9, %zmm8
        vfmadd213ps _sPoly_4(%rax), %zmm9, %zmm8
        vfmadd213ps _sPoly_3(%rax), %zmm9, %zmm8
        vfmadd213ps _sPoly_2(%rax), %zmm9, %zmm8
        vfmadd213ps _sPoly_1(%rax), %zmm9, %zmm8
        vmulps    %zmm9, %zmm8, %zmm10

/* polynomial evaluation end */
        vfmadd213ps %zmm9, %zmm9, %zmm10
        vpandnd   %zmm1, %zmm1, %zmm6{%k1}
        vptestmd  %zmm6, %zmm6, %k0
        vcvtdq2ps {rn-sae}, %zmm5, %zmm1
        kmovw     %k0, %ecx

/*
   final reconstruction:
   add exponent_value*log2 to polynomial result
 */
        vfmadd132ps _sLn2(%rax), %zmm10, %zmm1
        testl     %ecx, %ecx
        jne       .LBL_2_3

.LBL_2_2:
        cfi_remember_state
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_2_3:
        cfi_restore_state
        vmovups   %zmm0, 1152(%rsp)
        vmovups   %zmm1, 1216(%rsp)
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
        vmovups   1216(%rsp), %zmm1
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
        vmovss    1156(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1156(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(logf)

        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_2_8

.LBL_2_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1152(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(logf)

        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_2_7

#endif
END (_ZGVeN16v_logf_skx)

	.section .rodata, "a"
.L_2il0floatpacket.7:
	.long	0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.7,@object
