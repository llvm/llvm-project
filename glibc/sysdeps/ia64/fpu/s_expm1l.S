.file "expl_m1.s"


// Copyright (c) 2000 - 2003, Intel Corporation
// All rights reserved.
//
// Contributed 2000 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//
// History
//==============================================================
// 02/02/00 Initial Version
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 07/07/01 Improved speed of all paths
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double table values
// 03/11/03 Improved accuracy and performance, corrected missing inexact flags
// 04/17/03 Eliminated misplaced and unused data label
// 12/15/03 Eliminated call to error support on expm1l underflow
//
//*********************************************************************
//
// Function:   Combined expl(x) and expm1l(x), where
//                        x
//             expl(x) = e , for double-extended precision x values
//                          x
//             expm1l(x) = e  - 1  for double-extended precision x values
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8  (Input and Return Value)
//                              f9-f15,f32-f77
//
//    General Purpose Registers:
//      r14-r38
//      r35-r38 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    Denormal  fault raised on denormal inputs
//    Overflow exceptions raised when appropriate for exp and expm1
//    Underflow exceptions raised when appropriate for exp and expm1
//    (Error Handling Routine called for overflow and Underflow)
//    Inexact raised when appropriate by algorithm
//
//    exp(inf) = inf
//    exp(-inf) = +0
//    exp(SNaN) = QNaN
//    exp(QNaN) = QNaN
//    exp(0) = 1
//    exp(EM_special Values) = QNaN
//    exp(inf) = inf
//    expm1(-inf) = -1
//    expm1(SNaN) = QNaN
//    expm1(QNaN) = QNaN
//    expm1(0) = 0
//    expm1(EM_special Values) = QNaN
//
//*********************************************************************
//
// Implementation and Algorithm Notes:
//
//  ker_exp_64( in_FR  : X,
//            out_FR : Y_hi,
//            out_FR : Y_lo,
//            out_FR : scale,
//            out_PR : Safe )
//
// On input, X is in register format
// p6 for exp,
// p7 for expm1,
//
// On output,
//
//   scale*(Y_hi + Y_lo)  approximates  exp(X)       if exp
//   scale*(Y_hi + Y_lo)  approximates  exp(X)-1     if expm1
//
// The accuracy is sufficient for a highly accurate 64 sig.
// bit implementation.  Safe is set if there is no danger of
// overflow/underflow when the result is composed from scale,
// Y_hi and Y_lo. Thus, we can have a fast return if Safe is set.
// Otherwise, one must prepare to handle the possible exception
// appropriately.  Note that SAFE not set (false) does not mean
// that overflow/underflow will occur; only the setting of SAFE
// guarantees the opposite.
//
// **** High Level Overview ****
//
// The method consists of three cases.
//
// If           |X| < Tiny	use case exp_tiny;
// else if	|X| < 2^(-m)	use case exp_small; m=12 for exp, m=7 for expm1
// else		use case exp_regular;
//
// Case exp_tiny:
//
//   1 + X     can be used to approximate exp(X)
//   X + X^2/2 can be used to approximate exp(X) - 1
//
// Case exp_small:
//
//   Here, exp(X) and exp(X) - 1 can all be
//   approximated by a relatively simple polynomial.
//
//   This polynomial resembles the truncated Taylor series
//
//	exp(w) = 1 + w + w^2/2! + w^3/3! + ... + w^n/n!
//
// Case exp_regular:
//
//   Here we use a table lookup method. The basic idea is that in
//   order to compute exp(X), we accurately decompose X into
//
//   X = N * log(2)/(2^12)  + r,	|r| <= log(2)/2^13.
//
//   Hence
//
//   exp(X) = 2^( N / 2^12 ) * exp(r).
//
//   The value 2^( N / 2^12 ) is obtained by simple combinations
//   of values calculated beforehand and stored in table; exp(r)
//   is approximated by a short polynomial because |r| is small.
//
//   We elaborate this method in 4 steps.
//
//   Step 1: Reduction
//
//   The value 2^12/log(2) is stored as a double-extended number
//   L_Inv.
//
//   N := round_to_nearest_integer( X * L_Inv )
//
//   The value log(2)/2^12 is stored as two numbers L_hi and L_lo so
//   that r can be computed accurately via
//
//   r := (X - N*L_hi) - N*L_lo
//
//   We pick L_hi such that N*L_hi is representable in 64 sig. bits
//   and thus the FMA   X - N*L_hi   is error free. So r is the
//   1 rounding error from an exact reduction with respect to
//
//   L_hi + L_lo.
//
//   In particular, L_hi has 30 significant bit and can be stored
//   as a double-precision number; L_lo has 64 significant bits and
//   stored as a double-extended number.
//
//   Step 2: Approximation
//
//   exp(r) - 1 is approximated by a short polynomial of the form
//
//   r + A_1 r^2 + A_2 r^3 + A_3 r^4 .
//
//   Step 3: Composition from Table Values
//
//   The value 2^( N / 2^12 ) can be composed from a couple of tables
//   of precalculated values. First, express N as three integers
//   K, M_1, and M_2 as
//
//     N  =  K * 2^12  + M_1 * 2^6 + M_2
//
//   Where 0 <= M_1, M_2 < 2^6; and K can be positive or negative.
//   When N is represented in 2's complement, M_2 is simply the 6
//   lsb's, M_1 is the next 6, and K is simply N shifted right
//   arithmetically (sign extended) by 12 bits.
//
//   Now, 2^( N / 2^12 ) is simply
//
//      2^K * 2^( M_1 / 2^6 ) * 2^( M_2 / 2^12 )
//
//   Clearly, 2^K needs no tabulation. The other two values are less
//   trivial because if we store each accurately to more than working
//   precision, than its product is too expensive to calculate. We
//   use the following method.
//
//   Define two mathematical values, delta_1 and delta_2, implicitly
//   such that
//
//     T_1 = exp( [M_1 log(2)/2^6]  -  delta_1 )
//     T_2 = exp( [M_2 log(2)/2^12] -  delta_2 )
//
//   are representable as 24 significant bits. To illustrate the idea,
//   we show how we define delta_1:
//
//     T_1     := round_to_24_bits( exp( M_1 log(2)/2^6 ) )
//     delta_1  = (M_1 log(2)/2^6) - log( T_1 )
//
//   The last equality means mathematical equality. We then tabulate
//
//     W_1 := exp(delta_1) - 1
//     W_2 := exp(delta_2) - 1
//
//   Both in double precision.
//
//   From the tabulated values T_1, T_2, W_1, W_2, we compose the values
//   T and W via
//
//     T := T_1 * T_2			...exactly
//     W := W_1 + (1 + W_1)*W_2
//
//   W approximates exp( delta ) - 1  where delta = delta_1 + delta_2.
//   The mathematical product of T and (W+1) is an accurate representation
//   of 2^(M_1/2^6) * 2^(M_2/2^12).
//
//   Step 4. Reconstruction
//
//   Finally, we can reconstruct exp(X), exp(X) - 1.
//   Because
//
//	X = K * log(2) + (M_1*log(2)/2^6  - delta_1)
//		       + (M_2*log(2)/2^12 - delta_2)
//		       + delta_1 + delta_2 + r 		...accurately
//   We have
//
//	exp(X) ~=~ 2^K * ( T + T*[exp(delta_1+delta_2+r) - 1] )
//	       ~=~ 2^K * ( T + T*[exp(delta + r) - 1]         )
//	       ~=~ 2^K * ( T + T*[(exp(delta)-1)
//				 + exp(delta)*(exp(r)-1)]   )
//             ~=~ 2^K * ( T + T*( W + (1+W)*poly(r) ) )
//             ~=~ 2^K * ( Y_hi  +  Y_lo )
//
//   where Y_hi = T  and Y_lo = T*(W + (1+W)*poly(r))
//
//   For exp(X)-1, we have
//
//	exp(X)-1 ~=~ 2^K * ( Y_hi + Y_lo ) - 1
//		 ~=~ 2^K * ( Y_hi + Y_lo - 2^(-K) )
//
//   and we combine Y_hi + Y_lo - 2^(-N)  into the form of two
//   numbers  Y_hi + Y_lo carefully.
//
//   **** Algorithm Details ****
//
//   A careful algorithm must be used to realize the mathematical ideas
//   accurately. We describe each of the three cases. We assume SAFE
//   is preset to be TRUE.
//
//   Case exp_tiny:
//
//   The important points are to ensure an accurate result under
//   different rounding directions and a correct setting of the SAFE
//   flag.
//
//   If expm1 is 1, then
//      SAFE  := False	...possibility of underflow
//      Scale := 1.0
//      Y_hi  := X
//      Y_lo  := 2^(-17000)
//   Else
//      Scale := 1.0
//      Y_hi  := 1.0
//      Y_lo  := X	...for different rounding modes
//   Endif
//
//   Case exp_small:
//
//   Here we compute a simple polynomial. To exploit parallelism, we split
//   the polynomial into several portions.
//
//   Let r = X
//
//   If exp 	...i.e. exp( argument )
//
//      rsq := r * r;
//      r4  := rsq*rsq
//      poly_lo := P_3 + r*(P_4 + r*(P_5 + r*P_6))
//      poly_hi := r + rsq*(P_1 + r*P_2)
//      Y_lo    := poly_hi + r4 * poly_lo
//      Y_hi    := 1.0
//      Scale   := 1.0
//
//   Else			...i.e. exp( argument ) - 1
//
//      rsq := r * r
//      r4  := rsq * rsq
//      poly_lo := Q_7 + r*(Q_8 + r*Q_9))
//      poly_med:= Q_3 + r*Q_4 + rsq*(Q_5 + r*Q_6)
//      poly_med:= poly_med + r4*poly_lo
//      poly_hi := Q_1 + r*Q_2
//      Y_lo    := rsq*(poly_hi +  rsq*poly_lo)
//      Y_hi    := X
//      Scale   := 1.0
//
//   Endif
//
//  Case exp_regular:
//
//  The previous description contain enough information except the
//  computation of poly and the final Y_hi and Y_lo in the case for
//  exp(X)-1.
//
//  The computation of poly for Step 2:
//
//   rsq := r*r
//   poly := r + rsq*(A_1 + r*(A_2 + r*A_3))
//
//  For the case exp(X) - 1, we need to incorporate 2^(-K) into
//  Y_hi and Y_lo at the end of Step 4.
//
//   If K > 10 then
//      Y_lo := Y_lo - 2^(-K)
//   Else
//      If K < -10 then
//	 Y_lo := Y_hi + Y_lo
//	 Y_hi := -2^(-K)
//      Else
//	 Y_hi := Y_hi - 2^(-K)
//      End If
//   End If
//
//=======================================================
// General Purpose Registers
//
GR_ad_Arg           = r14
GR_ad_A             = r15
GR_sig_inv_ln2      = r15
GR_rshf_2to51       = r16
GR_ad_PQ            = r16
GR_ad_Q             = r16
GR_signexp_x        = r17
GR_exp_x            = r17
GR_small_exp        = r18
GR_rshf             = r18
GR_exp_mask         = r19
GR_ad_W1            = r20
GR_exp_2tom51       = r20
GR_ad_W2            = r21
GR_exp_underflow    = r21
GR_M2               = r22
GR_huge_exp         = r22
GR_M1               = r23
GR_huge_signif      = r23
GR_K                = r24
GR_one              = r24
GR_minus_one        = r24
GR_exp_bias         = r25
GR_ad_Limits        = r26
GR_N_fix            = r26
GR_exp_2_mk         = r26
GR_ad_P             = r27
GR_exp_2_k          = r27
GR_big_expo_neg     = r28
GR_very_small_exp   = r29
GR_exp_half         = r29
GR_ad_T1            = r30
GR_ad_T2            = r31

GR_SAVE_PFS         = r32
GR_SAVE_B0          = r33
GR_SAVE_GP          = r34
GR_Parameter_X      = r35
GR_Parameter_Y      = r36
GR_Parameter_RESULT = r37
GR_Parameter_TAG    = r38

// Floating Point Registers
//
FR_norm_x           = f9
FR_RSHF_2TO51       = f10
FR_INV_LN2_2TO63    = f11
FR_W_2TO51_RSH      = f12
FR_2TOM51           = f13
FR_RSHF             = f14
FR_Y_hi             = f34
FR_Y_lo             = f35
FR_scale            = f36
FR_tmp              = f37
FR_float_N          = f38
FR_N_signif         = f39
FR_L_hi             = f40
FR_L_lo             = f41
FR_r                = f42
FR_W1               = f43
FR_T1               = f44
FR_W2               = f45
FR_T2               = f46
FR_W1_p1            = f47
FR_rsq              = f48
FR_A2               = f49
FR_r4               = f50
FR_A3               = f51
FR_poly             = f52
FR_T                = f53
FR_W                = f54
FR_Wp1              = f55
FR_p21              = f59
FR_p210             = f59
FR_p65              = f60
FR_p654             = f60
FR_p6543            = f60
FR_2_mk             = f61
FR_P4Q7             = f61
FR_P4               = f61
FR_Q7               = f61
FR_P3Q6             = f62
FR_P3               = f62
FR_Q6               = f62
FR_q65              = f62
FR_q6543            = f62
FR_P2Q5             = f63
FR_P2               = f63
FR_Q5               = f63
FR_P1Q4             = f64
FR_P1               = f64
FR_Q4               = f64
FR_q43              = f64
FR_Q3               = f65
FR_Q2               = f66
FR_q21              = f66
FR_Q1               = f67
FR_A1               = f68
FR_P6Q9             = f68
FR_P6               = f68
FR_Q9               = f68
FR_P5Q8             = f69
FR_P5               = f69
FR_Q8               = f69
FR_q987             = f69
FR_q98              = f69
FR_q9876543         = f69
FR_min_oflow_x      = f70
FR_huge_exp         = f70
FR_zero_uflow_x     = f71
FR_huge_signif      = f71
FR_huge             = f72
FR_small            = f72
FR_half             = f73
FR_T_scale          = f74
FR_result_lo        = f75
FR_W_T_scale        = f76
FR_Wp1_T_scale      = f77
FR_ftz              = f77
FR_half_x           = f77
//

FR_X                = f9
FR_Y                = f0
FR_RESULT           = f15

// ************* DO NOT CHANGE ORDER OF THESE TABLES ********************

// double-extended 1/ln(2)
// 3fff b8aa 3b29 5c17 f0bb be87fed0691d3e88
// 3fff b8aa 3b29 5c17 f0bc
// For speed the significand will be loaded directly with a movl and setf.sig
//   and the exponent will be bias+63 instead of bias+0.  Thus subsequent
//   computations need to scale appropriately.
// The constant 2^12/ln(2) is needed for the computation of N.  This is also
//   obtained by scaling the computations.
//
// Two shifting constants are loaded directly with movl and setf.d.
//   1. RSHF_2TO51 = 1.1000..00 * 2^(63-12)
//        This constant is added to x*1/ln2 to shift the integer part of
//        x*2^12/ln2 into the rightmost bits of the significand.
//        The result of this fma is N_signif.
//   2. RSHF       = 1.1000..00 * 2^(63)
//        This constant is subtracted from N_signif * 2^(-51) to give
//        the integer part of N, N_fix, as a floating-point number.
//        The result of this fms is float_N.

RODATA
.align 64
LOCAL_OBJECT_START(Constants_exp_64_Arg)
//data8 0xB8AA3B295C17F0BC,0x0000400B // Inv_L = 2^12/log(2)
data8 0xB17217F400000000,0x00003FF2 // L_hi = hi part log(2)/2^12
data8 0xF473DE6AF278ECE6,0x00003FD4 // L_lo = lo part log(2)/2^12
LOCAL_OBJECT_END(Constants_exp_64_Arg)

LOCAL_OBJECT_START(Constants_exp_64_Limits)
data8 0xb17217f7d1cf79ac,0x0000400c // Smallest long dbl oflow x
data8 0xb220000000000000,0x0000c00c // Small long dbl uflow zero x
LOCAL_OBJECT_END(Constants_exp_64_Limits)

LOCAL_OBJECT_START(Constants_exp_64_A)
data8 0xAAAAAAABB1B736A0,0x00003FFA // A3
data8 0xAAAAAAAB90CD6327,0x00003FFC // A2
data8 0xFFFFFFFFFFFFFFFF,0x00003FFD // A1
LOCAL_OBJECT_END(Constants_exp_64_A)

LOCAL_OBJECT_START(Constants_exp_64_P)
data8 0xD00D6C8143914A8A,0x00003FF2 // P6
data8 0xB60BC4AC30304B30,0x00003FF5 // P5
data8 0x888888887474C518,0x00003FF8 // P4
data8 0xAAAAAAAA8DAE729D,0x00003FFA // P3
data8 0xAAAAAAAAAAAAAF61,0x00003FFC // P2
data8 0x80000000000004C7,0x00003FFE // P1
LOCAL_OBJECT_END(Constants_exp_64_P)

LOCAL_OBJECT_START(Constants_exp_64_Q)
data8 0x93F2AC5F7471F32E, 0x00003FE9 // Q9
data8 0xB8DA0F3550B3E764, 0x00003FEC // Q8
data8 0xD00D00D0028E89C4, 0x00003FEF // Q7
data8 0xD00D00DAEB8C4E91, 0x00003FF2 // Q6
data8 0xB60B60B60B60B6F5, 0x00003FF5 // Q5
data8 0x888888888886CC23, 0x00003FF8 // Q4
data8 0xAAAAAAAAAAAAAAAB, 0x00003FFA // Q3
data8 0xAAAAAAAAAAAAAAAB, 0x00003FFC // Q2
data8 0x8000000000000000, 0x00003FFE // Q1
LOCAL_OBJECT_END(Constants_exp_64_Q)

LOCAL_OBJECT_START(Constants_exp_64_T1)
data4 0x3F800000,0x3F8164D2,0x3F82CD87,0x3F843A29
data4 0x3F85AAC3,0x3F871F62,0x3F88980F,0x3F8A14D5
data4 0x3F8B95C2,0x3F8D1ADF,0x3F8EA43A,0x3F9031DC
data4 0x3F91C3D3,0x3F935A2B,0x3F94F4F0,0x3F96942D
data4 0x3F9837F0,0x3F99E046,0x3F9B8D3A,0x3F9D3EDA
data4 0x3F9EF532,0x3FA0B051,0x3FA27043,0x3FA43516
data4 0x3FA5FED7,0x3FA7CD94,0x3FA9A15B,0x3FAB7A3A
data4 0x3FAD583F,0x3FAF3B79,0x3FB123F6,0x3FB311C4
data4 0x3FB504F3,0x3FB6FD92,0x3FB8FBAF,0x3FBAFF5B
data4 0x3FBD08A4,0x3FBF179A,0x3FC12C4D,0x3FC346CD
data4 0x3FC5672A,0x3FC78D75,0x3FC9B9BE,0x3FCBEC15
data4 0x3FCE248C,0x3FD06334,0x3FD2A81E,0x3FD4F35B
data4 0x3FD744FD,0x3FD99D16,0x3FDBFBB8,0x3FDE60F5
data4 0x3FE0CCDF,0x3FE33F89,0x3FE5B907,0x3FE8396A
data4 0x3FEAC0C7,0x3FED4F30,0x3FEFE4BA,0x3FF28177
data4 0x3FF5257D,0x3FF7D0DF,0x3FFA83B3,0x3FFD3E0C
LOCAL_OBJECT_END(Constants_exp_64_T1)

LOCAL_OBJECT_START(Constants_exp_64_T2)
data4 0x3F800000,0x3F80058C,0x3F800B18,0x3F8010A4
data4 0x3F801630,0x3F801BBD,0x3F80214A,0x3F8026D7
data4 0x3F802C64,0x3F8031F2,0x3F803780,0x3F803D0E
data4 0x3F80429C,0x3F80482B,0x3F804DB9,0x3F805349
data4 0x3F8058D8,0x3F805E67,0x3F8063F7,0x3F806987
data4 0x3F806F17,0x3F8074A8,0x3F807A39,0x3F807FCA
data4 0x3F80855B,0x3F808AEC,0x3F80907E,0x3F809610
data4 0x3F809BA2,0x3F80A135,0x3F80A6C7,0x3F80AC5A
data4 0x3F80B1ED,0x3F80B781,0x3F80BD14,0x3F80C2A8
data4 0x3F80C83C,0x3F80CDD1,0x3F80D365,0x3F80D8FA
data4 0x3F80DE8F,0x3F80E425,0x3F80E9BA,0x3F80EF50
data4 0x3F80F4E6,0x3F80FA7C,0x3F810013,0x3F8105AA
data4 0x3F810B41,0x3F8110D8,0x3F81166F,0x3F811C07
data4 0x3F81219F,0x3F812737,0x3F812CD0,0x3F813269
data4 0x3F813802,0x3F813D9B,0x3F814334,0x3F8148CE
data4 0x3F814E68,0x3F815402,0x3F81599C,0x3F815F37
LOCAL_OBJECT_END(Constants_exp_64_T2)

LOCAL_OBJECT_START(Constants_exp_64_W1)
data8 0x0000000000000000, 0xBE384454171EC4B4
data8 0xBE6947414AA72766, 0xBE5D32B6D42518F8
data8 0x3E68D96D3A319149, 0xBE68F4DA62415F36
data8 0xBE6DDA2FC9C86A3B, 0x3E6B2E50F49228FE
data8 0xBE49C0C21188B886, 0x3E64BFC21A4C2F1F
data8 0xBE6A2FBB2CB98B54, 0x3E5DC5DE9A55D329
data8 0x3E69649039A7AACE, 0x3E54728B5C66DBA5
data8 0xBE62B0DBBA1C7D7D, 0x3E576E0409F1AF5F
data8 0x3E6125001A0DD6A1, 0xBE66A419795FBDEF
data8 0xBE5CDE8CE1BD41FC, 0xBE621376EA54964F
data8 0x3E6370BE476E76EE, 0x3E390D1A3427EB92
data8 0x3E1336DE2BF82BF8, 0xBE5FF1CBD0F7BD9E
data8 0xBE60A3550CEB09DD, 0xBE5CA37E0980F30D
data8 0xBE5C541B4C082D25, 0xBE5BBECA3B467D29
data8 0xBE400D8AB9D946C5, 0xBE5E2A0807ED374A
data8 0xBE66CB28365C8B0A, 0x3E3AAD5BD3403BCA
data8 0x3E526055C7EA21E0, 0xBE442C75E72880D6
data8 0x3E58B2BB85222A43, 0xBE5AAB79522C42BF
data8 0xBE605CB4469DC2BC, 0xBE589FA7A48C40DC
data8 0xBE51C2141AA42614, 0xBE48D087C37293F4
data8 0x3E367A1CA2D673E0, 0xBE51BEBB114F7A38
data8 0xBE6348E5661A4B48, 0xBDF526431D3B9962
data8 0x3E3A3B5E35A78A53, 0xBE46C46C1CECD788
data8 0xBE60B7EC7857D689, 0xBE594D3DD14F1AD7
data8 0xBE4F9C304C9A8F60, 0xBE52187302DFF9D2
data8 0xBE5E4C8855E6D68F, 0xBE62140F667F3DC4
data8 0xBE36961B3BF88747, 0x3E602861C96EC6AA
data8 0xBE3B5151D57FD718, 0x3E561CD0FC4A627B
data8 0xBE3A5217CA913FEA, 0x3E40A3CC9A5D193A
data8 0xBE5AB71310A9C312, 0x3E4FDADBC5F57719
data8 0x3E361428DBDF59D5, 0x3E5DB5DB61B4180D
data8 0xBE42AD5F7408D856, 0x3E2A314831B2B707
LOCAL_OBJECT_END(Constants_exp_64_W1)

LOCAL_OBJECT_START(Constants_exp_64_W2)
data8 0x0000000000000000, 0xBE641F2537A3D7A2
data8 0xBE68DD57AD028C40, 0xBE5C77D8F212B1B6
data8 0x3E57878F1BA5B070, 0xBE55A36A2ECAE6FE
data8 0xBE620608569DFA3B, 0xBE53B50EA6D300A3
data8 0x3E5B5EF2223F8F2C, 0xBE56A0D9D6DE0DF4
data8 0xBE64EEF3EAE28F51, 0xBE5E5AE2367EA80B
data8 0x3E47CB1A5FCBC02D, 0xBE656BA09BDAFEB7
data8 0x3E6E70C6805AFEE7, 0xBE6E0509A3415EBA
data8 0xBE56856B49BFF529, 0x3E66DD3300508651
data8 0x3E51165FC114BC13, 0x3E53333DC453290F
data8 0x3E6A072B05539FDA, 0xBE47CD877C0A7696
data8 0xBE668BF4EB05C6D9, 0xBE67C3E36AE86C93
data8 0xBE533904D0B3E84B, 0x3E63E8D9556B53CE
data8 0x3E212C8963A98DC8, 0xBE33138F032A7A22
data8 0x3E530FA9BC584008, 0xBE6ADF82CCB93C97
data8 0x3E5F91138370EA39, 0x3E5443A4FB6A05D8
data8 0x3E63DACD181FEE7A, 0xBE62B29DF0F67DEC
data8 0x3E65C4833DDE6307, 0x3E5BF030D40A24C1
data8 0x3E658B8F14E437BE, 0xBE631C29ED98B6C7
data8 0x3E6335D204CF7C71, 0x3E529EEDE954A79D
data8 0x3E5D9257F64A2FB8, 0xBE6BED1B854ED06C
data8 0x3E5096F6D71405CB, 0xBE3D4893ACB9FDF5
data8 0xBDFEB15801B68349, 0x3E628D35C6A463B9
data8 0xBE559725ADE45917, 0xBE68C29C042FC476
data8 0xBE67593B01E511FA, 0xBE4A4313398801ED
data8 0x3E699571DA7C3300, 0x3E5349BE08062A9E
data8 0x3E5229C4755BB28E, 0x3E67E42677A1F80D
data8 0xBE52B33F6B69C352, 0xBE6B3550084DA57F
data8 0xBE6DB03FD1D09A20, 0xBE60CBC42161B2C1
data8 0x3E56ED9C78A2B771, 0xBE508E319D0FA795
data8 0xBE59482AFD1A54E9, 0xBE2A17CEB07FD23E
data8 0x3E68BF5C17365712, 0x3E3956F9B3785569
LOCAL_OBJECT_END(Constants_exp_64_W2)


.section .text

GLOBAL_IEEE754_ENTRY(expm1l)

//
//    Set p7 true for expm1, p6 false
//

{ .mlx
      getf.exp GR_signexp_x = f8  // Get sign and exponent of x, redo if unorm
      movl GR_sig_inv_ln2 = 0xb8aa3b295c17f0bc  // significand of 1/ln2
}
{ .mlx
      addl GR_ad_Arg = @ltoff(Constants_exp_64_Arg#),gp
      movl GR_rshf_2to51 = 0x4718000000000000 // 1.10000 2^(63+51)
}
;;

{ .mfi
      ld8  GR_ad_Arg = [GR_ad_Arg]       // Point to Arg table
      fclass.m p8, p0 =  f8, 0x1E7       // Test x for natval, nan, inf, zero
      cmp.eq  p7, p6 =  r0, r0
}
{ .mfb
      mov GR_exp_half = 0x0FFFE          // Exponent of 0.5, for very small path
      fnorm.s1 FR_norm_x = f8            // Normalize x
      br.cond.sptk exp_continue
}
;;

GLOBAL_IEEE754_END(expm1l)
libm_alias_ldouble_other (__expm1, expm1)


GLOBAL_IEEE754_ENTRY(expl)
//
//    Set p7 false for exp, p6 true
//
{ .mlx
      getf.exp GR_signexp_x = f8  // Get sign and exponent of x, redo if unorm
      movl GR_sig_inv_ln2 = 0xb8aa3b295c17f0bc  // significand of 1/ln2
}
{ .mlx
      addl GR_ad_Arg = @ltoff(Constants_exp_64_Arg#),gp
      movl GR_rshf_2to51 = 0x4718000000000000 // 1.10000 2^(63+51)
}
;;

{ .mfi
      ld8  GR_ad_Arg = [GR_ad_Arg]       // Point to Arg table
      fclass.m p8, p0 =  f8, 0x1E7       // Test x for natval, nan, inf, zero
      cmp.eq  p6, p7 =  r0, r0
}
{ .mfi
      mov GR_exp_half = 0x0FFFE          // Exponent of 0.5, for very small path
      fnorm.s1 FR_norm_x = f8            // Normalize x
      nop.i 999
}
;;

exp_continue:
// Form two constants we need
//  1/ln2 * 2^63  to compute  w = x * 1/ln2 * 128
//  1.1000..000 * 2^(63+63-12) to right shift int(N) into the significand

{ .mfi
      setf.sig  FR_INV_LN2_2TO63 = GR_sig_inv_ln2 // form 1/ln2 * 2^63
      fclass.nm.unc p9, p0 =  f8, 0x1FF  // Test x for unsupported
      mov GR_exp_2tom51 = 0xffff-51
}
{ .mlx
      setf.d  FR_RSHF_2TO51 = GR_rshf_2to51 // Form const 1.1000 * 2^(63+51)
      movl GR_rshf = 0x43e8000000000000  // 1.10000 2^63 for right shift
}
;;

{ .mfi
      setf.exp FR_half = GR_exp_half     // Form 0.5 for very small path
      fma.s1 FR_scale = f1,f1,f0         // Scale = 1.0
      mov GR_exp_bias = 0x0FFFF          // Set exponent bias
}
{ .mib
      add GR_ad_Limits = 0x20, GR_ad_Arg // Point to Limits table
      mov GR_exp_mask = 0x1FFFF          // Form exponent mask
(p8)  br.cond.spnt EXP_64_SPECIAL        // Branch if natval, nan, inf, zero
}
;;

{ .mfi
      setf.exp FR_2TOM51 = GR_exp_2tom51 // Form 2^-51 for scaling float_N
      nop.f 999
      add GR_ad_A = 0x40, GR_ad_Arg      // Point to A table
}
{ .mib
      setf.d  FR_RSHF = GR_rshf          // Form right shift const 1.1000 * 2^63
      add GR_ad_T1 = 0x160, GR_ad_Arg    // Point to T1 table
(p9)  br.cond.spnt EXP_64_UNSUPPORTED    // Branch if unsupported
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      ldfe FR_L_hi = [GR_ad_Arg],16      // Get L_hi
      fcmp.eq.s0 p9,p0 =  f8, f0         // Dummy op to flag denormals
(p6)  add GR_ad_PQ = 0x30, GR_ad_A       // Point to P table for exp
}
{ .mfi
      ldfe FR_min_oflow_x = [GR_ad_Limits],16 // Get min x to cause overflow
      fmpy.s1 FR_rsq = f8, f8            // rsq = x * x for small path
(p7)  add GR_ad_PQ = 0x90, GR_ad_A       // Point to Q table for expm1
};;

{ .mmi
      ldfe FR_L_lo = [GR_ad_Arg],16      // Get L_lo
      ldfe FR_zero_uflow_x = [GR_ad_Limits],16 // Get x for zero uflow result
      add GR_ad_W1 = 0x200, GR_ad_T1     // Point to W1 table
}
;;

{ .mfi
      ldfe FR_P6Q9 = [GR_ad_PQ],16       // P6(exp) or Q9(expm1) for small path
      mov FR_r = FR_norm_x               // r = X for small path
      mov GR_very_small_exp = -60        // Exponent of x for very small path
}
{ .mfi
      add GR_ad_W2 = 0x400, GR_ad_T1     // Point to W2 table
      nop.f 999
(p7)  mov GR_small_exp = -7              // Exponent of x for small path expm1
}
;;

{ .mmi
      ldfe FR_P5Q8 = [GR_ad_PQ],16       // P5(exp) or Q8(expm1) for small path
      and  GR_exp_x = GR_signexp_x, GR_exp_mask
(p6)  mov GR_small_exp = -12             // Exponent of x for small path exp
}
;;

// N_signif = X * Inv_log2_by_2^12
// By adding 1.10...0*2^63 we shift and get round_int(N_signif) in significand.
// We actually add 1.10...0*2^51 to X * Inv_log2 to do the same thing.
{ .mfi
      ldfe FR_P4Q7 = [GR_ad_PQ],16       // P4(exp) or Q7(expm1) for small path
      fma.s1 FR_N_signif = FR_norm_x, FR_INV_LN2_2TO63, FR_RSHF_2TO51
      nop.i 999
}
{ .mfi
      sub GR_exp_x = GR_exp_x, GR_exp_bias // Get exponent
      fmpy.s1 FR_r4 = FR_rsq, FR_rsq     // Form r4 for small path
      cmp.eq.unc  p15, p0 =  r0, r0      // Set Safe as default
}
;;

{ .mmi
      ldfe FR_P3Q6 = [GR_ad_PQ],16       // P3(exp) or Q6(expm1) for small path
      cmp.lt  p14, p0 =  GR_exp_x, GR_very_small_exp // Is |x| < 2^-60?
      nop.i 999
}
;;

{ .mfi
      ldfe FR_P2Q5 = [GR_ad_PQ],16       // P2(exp) or Q5(expm1) for small path
      fmpy.s1 FR_half_x = FR_half, FR_norm_x // 0.5 * x for very small path
      cmp.lt  p13, p0 =  GR_exp_x, GR_small_exp // Is |x| < 2^-m?
}
{ .mib
      nop.m 999
      nop.i 999
(p14) br.cond.spnt EXP_VERY_SMALL        // Branch if |x| < 2^-60
}
;;

{ .mfi
      ldfe FR_A3 = [GR_ad_A],16          // Get A3 for normal path
      fcmp.ge.s1 p10,p0 = FR_norm_x, FR_min_oflow_x // Will result overflow?
      mov GR_big_expo_neg = -16381       // -0x3ffd
}
{ .mfb
      ldfe FR_P1Q4 = [GR_ad_PQ],16       // P1(exp) or Q4(expm1) for small path
      nop.f 999
(p13) br.cond.spnt EXP_SMALL             // Branch if |x| < 2^-m
                                         // m=12 for exp, m=7 for expm1
}
;;

// Now we are on the main path for |x| >= 2^-m, m=12 for exp, m=7 for expm1
//
// float_N = round_int(N_signif)
// The signficand of N_signif contains the rounded integer part of X * 2^12/ln2,
// as a twos complement number in the lower bits (that is, it may be negative).
// That twos complement number (called N) is put into GR_N.

// Since N_signif is scaled by 2^51, it must be multiplied by 2^-51
// before the shift constant 1.10000 * 2^63 is subtracted to yield float_N.
// Thus, float_N contains the floating point version of N


{ .mfi
      ldfe FR_A2 = [GR_ad_A],16          // Get A2 for main path
      fcmp.lt.s1 p11,p0 = FR_norm_x, FR_zero_uflow_x // Certain zero, uflow?
      add GR_ad_T2 = 0x100, GR_ad_T1     // Point to T2 table
}
{ .mfi
      nop.m 999
      fms.s1 FR_float_N = FR_N_signif, FR_2TOM51, FR_RSHF // Form float_N
      nop.i 999
}
;;

{ .mbb
      getf.sig GR_N_fix = FR_N_signif    // Get N from significand
(p10) br.cond.spnt  EXP_OVERFLOW         // Branch if result will overflow
(p11) br.cond.spnt  EXP_CERTAIN_UNDERFLOW_ZERO // Branch if certain zero, uflow
}
;;

{ .mfi
      ldfe FR_A1 = [GR_ad_A],16          // Get A1 for main path
      fnma.s1 FR_r = FR_L_hi, FR_float_N, FR_norm_x  // r = -L_hi * float_N + x
      extr.u GR_M1 = GR_N_fix, 6, 6      // Extract index M_1
}
{ .mfi
      and GR_M2 = 0x3f, GR_N_fix         // Extract index M_2
      nop.f 999
      nop.i 999
}
;;

// N_fix is only correct up to 50 bits because of our right shift technique.
// Actually in the normal path we will have restricted K to about 14 bits.
// Somewhat arbitrarily we extract 32 bits.
{ .mfi
      shladd GR_ad_W1 = GR_M1,3,GR_ad_W1 // Point to W1
      nop.f 999
      extr GR_K = GR_N_fix, 12, 32       // Extract limited range K
}
{ .mfi
      shladd GR_ad_T1 = GR_M1,2,GR_ad_T1 // Point to T1
      nop.f 999
      shladd GR_ad_T2 = GR_M2,2,GR_ad_T2 // Point to T2
}
;;

{ .mmi
      ldfs  FR_T1 = [GR_ad_T1],0         // Get T1
      ldfd  FR_W1 = [GR_ad_W1],0         // Get W1
      add GR_exp_2_k = GR_exp_bias, GR_K // Form exponent of 2^k
}
;;

{ .mmi
      ldfs  FR_T2 = [GR_ad_T2],0         // Get T2
      shladd GR_ad_W2 = GR_M2,3,GR_ad_W2 // Point to W2
      sub GR_exp_2_mk = GR_exp_bias, GR_K // Form exponent of 2^-k
}
;;

{ .mmf
      ldfd  FR_W2 = [GR_ad_W2],0         // Get W2
      setf.exp FR_scale = GR_exp_2_k     // Set scale = 2^k
      fnma.s1 FR_r = FR_L_lo, FR_float_N, FR_r // r = -L_lo * float_N + r
}
;;

{ .mfi
      setf.exp FR_2_mk = GR_exp_2_mk     // Form 2^-k
      fma.s1 FR_poly = FR_r, FR_A3, FR_A2 // poly = r * A3 + A2
      cmp.lt p8,p15 = GR_K,GR_big_expo_neg // Set Safe if K > big_expo_neg
}
{ .mfi
      nop.m 999
      fmpy.s1 FR_rsq = FR_r, FR_r         // rsq = r * r
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_T = FR_T1, FR_T2         // T = T1 * T2
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_W1_p1 = FR_W1, f1        // W1_p1 = W1 + 1.0
      nop.i 999
}
;;

{ .mfi
(p7)  cmp.lt.unc  p8, p9 =  10, GR_K       // If expm1, set p8 if K > 10
      fma.s1 FR_poly = FR_r, FR_poly, FR_A1 // poly = r * poly + A1
      nop.i 999
}
;;

{ .mfi
(p7)  cmp.eq  p15, p0 =  r0, r0            // If expm1, set Safe flag
      fma.s1 FR_T_scale = FR_T, FR_scale, f0 // T_scale = T * scale
(p9)  cmp.gt.unc  p9, p10 =  -10, GR_K     // If expm1, set p9 if K < -10
                                           // If expm1, set p10 if -10<=K<=10
}
{ .mfi
      nop.m 999
      fma.s1 FR_W = FR_W2, FR_W1_p1, FR_W1 // W = W2 * (W1+1.0) + W1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      mov FR_Y_hi = FR_T                   // Assume Y_hi = T
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_poly = FR_rsq, FR_poly, FR_r // poly = rsq * poly + r
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_Wp1_T_scale = FR_W, FR_T_scale, FR_T_scale // (W+1)*T*scale
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 FR_W_T_scale = FR_W, FR_T_scale, f0 // W*T*scale
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fsub.s1 FR_Y_hi = f0, FR_2_mk      // If expm1, if K < -10 set Y_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p10) fsub.s1 FR_Y_hi = FR_T, FR_2_mk    // If expm1, if |K|<=10 set Y_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 FR_result_lo = FR_Wp1_T_scale, FR_poly, FR_W_T_scale
      nop.i 999
}
;;

.pred.rel "mutex",p8,p9
// If K > 10 adjust result_lo = result_lo - scale * 2^-k
// If |K| <= 10 adjust result_lo = result_lo + scale * T
{ .mfi
      nop.m 999
(p8)  fnma.s1 FR_result_lo = FR_scale, FR_2_mk, FR_result_lo // If K > 10
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1 FR_result_lo = FR_T_scale, f1, FR_result_lo // If |K| <= 10
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s0 FR_tmp = FR_A1, FR_A1         // Dummy op to set inexact
      nop.i 999
}
{ .mfb
      nop.m 999
(p15) fma.s0 f8 = FR_Y_hi, FR_scale, FR_result_lo  // Safe result
(p15) br.ret.sptk b0                        // Safe exit for normal path
}
;;

// Here if unsafe, will only be here for exp with K < big_expo_neg
{ .mfb
      nop.m 999
      fma.s0 FR_RESULT = FR_Y_hi, FR_scale, FR_result_lo  // Prelim result
      br.cond.sptk EXP_POSSIBLE_UNDERFLOW  // Branch to unsafe code
}
;;


EXP_SMALL:
// Here if 2^-60 < |x| < 2^-m, m=12 for exp, m=7 for expm1
{ .mfi
(p7)  ldfe FR_Q3 = [GR_ad_Q],16          // Get Q3 for small path, if expm1
(p6)  fma.s1 FR_p65 = FR_P6, FR_r, FR_P5  // If exp, p65 = P6 * r + P5
      nop.i 999
}
{ .mfi
      mov GR_minus_one = -1
(p7)  fma.s1 FR_q98 = FR_Q9, FR_r, FR_Q8  // If expm1, q98 = Q9 * r + Q8
      nop.i 999
}
;;

{ .mfi
(p7)  ldfe FR_Q2 = [GR_ad_Q],16           // Get Q2 for small path, if expm1
(p7)  fma.s1 FR_q65 = FR_Q6, FR_r, FR_Q5  // If expm1, q65 = Q6 * r + Q5
      nop.i 999
}
;;

{ .mfi
      setf.sig FR_tmp = GR_minus_one      // Create value to force inexact
(p6)  fma.s1 FR_p21 = FR_P2, FR_r, FR_P1  // If exp, p21 = P2 * r + P1
      nop.i 999
}
{ .mfi
(p7)  ldfe FR_Q1 = [GR_ad_Q],16           // Get Q1 for small path, if expm1
(p7)  fma.s1 FR_q43 = FR_Q4, FR_r, FR_Q3  // If expm1, q43 = Q4 * r + Q3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fma.s1 FR_p654 = FR_p65, FR_r, FR_P4 // If exp, p654 = p65 * r + P4
      nop.i 999
}
{ .mfi
      nop.m 999
(p7)  fma.s1 FR_q987 = FR_q98, FR_r, FR_Q7 // If expm1, q987 = q98 * r + Q7
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p7)  fma.s1 FR_q21 = FR_Q2, FR_r, FR_Q1  // If expm1, q21 = Q2 * r + Q1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fma.s1 FR_p210 = FR_p21, FR_rsq, FR_r // If exp, p210 = p21 * r + P0
      nop.i 999
}
{ .mfi
      nop.m 999
(p7)  fma.s1 FR_q6543 = FR_q65, FR_rsq, FR_q43 // If expm1, q6543 = q65*r2+q43
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fma.s1 FR_p6543 = FR_p654, FR_r, FR_P3 // If exp, p6543 = p654 * r + P3
      nop.i 999
}
{ .mfi
      nop.m 999
(p7)  fma.s1 FR_q9876543 = FR_q987, FR_r4, FR_q6543 // If expm1, q9876543 = ...
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fma.s1 FR_Y_lo = FR_p6543, FR_r4, FR_p210 // If exp, form Y_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p7)  fma.s1 FR_Y_lo = FR_q9876543, FR_rsq, FR_q21 // If expm1, form Y_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s0  FR_tmp = FR_tmp, FR_tmp   // Dummy op to set inexact
      nop.i 999
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      nop.m 999
(p6)  fma.s0 f8 = FR_Y_lo, f1, f1          // If exp, result = 1 + Y_lo
      nop.i 999
}
{ .mfb
      nop.m 999
(p7)  fma.s0 f8 = FR_Y_lo, FR_rsq, FR_norm_x // If expm1, result = Y_lo*r2+x
      br.ret.sptk  b0                      // Exit for 2^-60 <= |x| < 2^-m
                                           // m=12 for exp, m=7 for expm1
}
;;


EXP_VERY_SMALL:
//
// Here if 0 < |x| < 2^-60
// If exp, result = 1.0 + x
// If expm1, result = x +x*x/2, but have to check for possible underflow
//

{ .mfi
(p7)  mov GR_exp_underflow = -16381        // Exponent for possible underflow
(p6)  fadd.s0 f8 = f1, FR_norm_x           // If exp, result = 1+x
      nop.i 999
}
{ .mfi
      nop.m 999
(p7)  fmpy.s1 FR_result_lo = FR_half_x, FR_norm_x  // If expm1 result_lo = x*x/2
      nop.i 999
}
;;

{ .mfi
(p7)  cmp.lt.unc p0, p8 = GR_exp_x, GR_exp_underflow // Unsafe if expm1 x small
(p7)  mov FR_Y_hi = FR_norm_x              // If expm1, Y_hi = x
(p7)  cmp.lt p0, p15 = GR_exp_x, GR_exp_underflow // Unsafe if expm1 x small
}
;;

{ .mfb
      nop.m 999
(p8)  fma.s0 f8 = FR_norm_x, f1, FR_result_lo // If expm1, result=x+x*x/2
(p15) br.ret.sptk b0                       // If Safe, exit
}
;;

// Here if expm1 and 0 < |x| < 2^-16381;  may be possible underflow
{ .mfb
      nop.m 999
      fma.s0 FR_RESULT = FR_Y_hi, FR_scale, FR_result_lo // Prelim result
      br.cond.sptk EXP_POSSIBLE_UNDERFLOW  // Branch to unsafe code
}
;;

EXP_CERTAIN_UNDERFLOW_ZERO:
// Here if x < zero_uflow_x
// For exp, set result to tiny+0.0 and set I, U, and branch to error handling
// For expm1, set result to tiny-1.0 and set I, and exit
{ .mmi
      alloc GR_SAVE_PFS = ar.pfs,0,3,4,0
      nop.m 999
      mov GR_one = 1
}
;;

{ .mmi
      setf.exp FR_small = GR_one               // Form small value
      nop.m 999
(p6)  mov GR_Parameter_TAG = 13                // Error tag for exp underflow
}
;;

{ .mfi
      nop.m 999
      fmerge.s FR_X = f8,f8                    // Save x for error call
      nop.i 999
}
;;

.pred.rel "mutex",p6,p7
{ .mfb
      nop.m 999
(p6)  fma.s0 FR_RESULT = FR_small, FR_small, f0 // If exp, set I,U, tiny result
(p6)  br.cond.sptk __libm_error_region          // If exp, go to error handling
}
{ .mfb
      nop.m 999
(p7)  fms.s0 f8 = FR_small, FR_small, f1        // If expm1, set I, result -1.0
(p7)  br.ret.sptk  b0                           // If expm1, exit
}
;;


EXP_OVERFLOW:
// Here if x >= min_oflow_x
{ .mmi
      alloc GR_SAVE_PFS = ar.pfs,0,3,4,0
      mov GR_huge_exp = 0x1fffe
      nop.i 999
}
{ .mfi
      mov GR_huge_signif = -0x1
      nop.f 999
(p6)  mov GR_Parameter_TAG = 12                // Error tag for exp overflow
}
;;

{ .mmf
      setf.exp FR_huge_exp = GR_huge_exp       // Create huge value
      setf.sig FR_huge_signif = GR_huge_signif // Create huge value
      fmerge.s FR_X = f8,f8                    // Save x for error call
}
;;

{ .mfi
      nop.m 999
      fmerge.se FR_huge = FR_huge_exp, FR_huge_signif
(p7)  mov GR_Parameter_TAG = 39                // Error tag for expm1 overflow
}
;;

{ .mfb
      nop.m 999
      fma.s0 FR_RESULT = FR_huge, FR_huge, FR_huge // Force I, O, and Inf
      br.cond.sptk __libm_error_region         // Branch to error handling
}
;;



EXP_POSSIBLE_UNDERFLOW:
// Here if exp and zero_uflow_x < x < about -11356 [where k < -16381]
// Here if expm1 and |x| < 2^-16381
{ .mfi
      alloc GR_SAVE_PFS = ar.pfs,0,3,4,0
      fsetc.s2 0x7F,0x41                   // Set FTZ and disable traps
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s2 FR_ftz = FR_Y_hi, FR_scale, FR_result_lo   // Result with FTZ
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsetc.s2 0x7F,0x40                   // Disable traps (set s2 default)
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m.unc p11, p0 = FR_ftz, 0x00F // If exp, FTZ result denorm or zero?
      nop.i 999
}
;;

{ .mfb
(p11) mov   GR_Parameter_TAG = 13             // exp underflow
      fmerge.s FR_X = f8,f8                   // Save x for error call
(p11) br.cond.spnt __libm_error_region        // Branch on exp underflow
}
;;

{ .mfb
      nop.m 999
      mov   f8     = FR_RESULT                // Was safe after all
      br.ret.sptk   b0
}
;;


EXP_64_SPECIAL:
// Here if x natval, nan, inf, zero
// If x natval, +inf, or if expm1 and x zero, just return x.
// The other cases must be tested for, and results set.
// These cases do not generate exceptions.
{ .mfi
      nop.m 999
      fclass.m p8, p0 =  f8, 0x0c3            // Is x nan?
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m.unc p13, p0 =  f8, 0x007       // If exp, is x zero?
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m.unc p11, p0 =  f8, 0x022       // If exp, is x -inf?
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fadd.s0 f8 = f8, f1                     // If x nan, result quietized x
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p7)  fclass.m.unc p10, p0 =  f8, 0x022       // If expm1, is x -inf?
      nop.i 999
}
{ .mfi
      nop.m 999
(p13) fadd.s0 f8 = f0, f1                     // If exp and x zero, result 1.0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p11) mov f8 = f0                             // If exp and x -inf, result 0
      nop.i 999
}
;;

{ .mfb
      nop.m 999
(p10) fsub.s1 f8 = f0, f1                     // If expm1, x -inf, result -1.0
      br.ret.sptk b0                          // Exit special cases
}
;;


EXP_64_UNSUPPORTED:
// Here if x unsupported type
{ .mfb
      nop.m 999
      fmpy.s0 f8 = f8, f0                     // Return nan
      br.ret.sptk   b0
}
;;

GLOBAL_IEEE754_END(expl)
libm_alias_ldouble_other (__exp, exp)

LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                           // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                       // Save gp
};;
{ .mmi
        stfe [GR_Parameter_Y] = FR_Y,16         // Save Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfe [GR_Parameter_X] = FR_X            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfe [GR_Parameter_Y] = FR_RESULT      // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#  // Call error handling function
};;
{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;
{ .mmi
        ldfe  f8 = [GR_Parameter_RESULT]       // Get return result off stack
.restore sp
        add   sp = 64,sp                       // Restore stack pointer
        mov   b0 = GR_SAVE_B0                  // Restore return address
};;
{ .mib
        mov   gp = GR_SAVE_GP                  // Restore gp
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
        br.ret.sptk     b0                     // Return
};;
LOCAL_LIBM_END(__libm_error_region#)

.type   __libm_error_support#,@function
.global __libm_error_support#
