.file "logl.s"


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
//*********************************************************************
//
// History:
// 05/21/01 Extracted logl and log10l from log1pl.s file, and optimized
//          all paths.
// 06/20/01 Fixed error tag for x=-inf.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double table values
//
//*********************************************************************
//
//*********************************************************************
//
// Function:   Combined logl(x) and log10l(x) where
//             logl(x)   = ln(x), for double-extended precision x values
//             log10l(x) = log (x), for double-extended precision x values
//                           10
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8 (Input and Return Value)
//                              f34-f76
//
//    General Purpose Registers:
//      r32-r56
//      r53-r56 (Used to pass arguments to error handling routine)
//
//    Predicate Registers:      p6-p14
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    Denormal  fault raised on denormal inputs
//    Overflow exceptions cannot occur
//    Underflow exceptions raised when appropriate for log1p
//    (Error Handling Routine called for underflow)
//    Inexact raised when appropriate by algorithm
//
//    logl(inf) = inf
//    logl(-inf) = QNaN
//    logl(+/-0) = -inf
//    logl(SNaN) = QNaN
//    logl(QNaN) = QNaN
//    logl(EM_special Values) = QNaN
//    log10l(inf) = inf
//    log10l(-inf) = QNaN
//    log10l(+/-0) = -inf
//    log10l(SNaN) = QNaN
//    log10l(QNaN) = QNaN
//    log10l(EM_special Values) = QNaN
//
//*********************************************************************
//
// Overview
//
// The method consists of two cases.
//
// If      |X-1| < 2^(-7)	use case log_near1;
// else      			use case log_regular;
//
// Case log_near1:
//
//   logl( 1 + X ) can be approximated by a simple polynomial
//   in W = X-1. This polynomial resembles the truncated Taylor
//   series W - W^/2 + W^3/3 - ...
//
// Case log_regular:
//
//   Here we use a table lookup method. The basic idea is that in
//   order to compute logl(Arg) for an argument Arg in [1,2), we
//   construct a value G such that G*Arg is close to 1 and that
//   logl(1/G) is obtainable easily from a table of values calculated
//   beforehand. Thus
//
//      logl(Arg) = logl(1/G) + logl(G*Arg)
//      	 = logl(1/G) + logl(1 + (G*Arg - 1))
//
//   Because |G*Arg - 1| is small, the second term on the right hand
//   side can be approximated by a short polynomial. We elaborate
//   this method in four steps.
//
//   Step 0: Initialization
//
//   We need to calculate logl( X ). Obtain N, S_hi such that
//
//      X = 2^N * S_hi 	exactly
//
//   where S_hi in [1,2)
//
//   Step 1: Argument Reduction
//
//   Based on S_hi, obtain G_1, G_2, G_3 from a table and calculate
//
//      G := G_1 * G_2 * G_3
//      r := (G * S_hi - 1)
//
//   These G_j's have the property that the product is exactly
//   representable and that |r| < 2^(-12) as a result.
//
//   Step 2: Approximation
//
//
//   logl(1 + r) is approximated by a short polynomial poly(r).
//
//   Step 3: Reconstruction
//
//
//   Finally, logl( X ) is given by
//
//   logl( X )   =   logl( 2^N * S_hi )
//                 ~=~  N*logl(2) + logl(1/G) + logl(1 + r)
//                 ~=~  N*logl(2) + logl(1/G) + poly(r).
//
// **** Algorithm ****
//
// Case log_near1:
//
// Here we compute a simple polynomial. To exploit parallelism, we split
// the polynomial into two portions.
//
//       W := X - 1
//       Wsq := W * W
//       W4  := Wsq*Wsq
//       W6  := W4*Wsq
//       Y_hi := W + Wsq*(P_1 + W*(P_2 + W*(P_3 + W*P_4))
//       Y_lo := W6*(P_5 + W*(P_6 + W*(P_7 + W*P_8)))
//
// Case log_regular:
//
// We present the algorithm in four steps.
//
//   Step 0. Initialization
//   ----------------------
//
//   Z := X
//   N := unbaised exponent of Z
//   S_hi := 2^(-N) * Z
//
//   Step 1. Argument Reduction
//   --------------------------
//
//   Let
//
//      Z = 2^N * S_hi = 2^N * 1.d_1 d_2 d_3 ... d_63
//
//   We obtain G_1, G_2, G_3 by the following steps.
//
//
//      Define		X_0 := 1.d_1 d_2 ... d_14. This is extracted
//      		from S_hi.
//
//      Define		A_1 := 1.d_1 d_2 d_3 d_4. This is X_0 truncated
//      		to lsb = 2^(-4).
//
//      Define		index_1 := [ d_1 d_2 d_3 d_4 ].
//
//      Fetch 		Z_1 := (1/A_1) rounded UP in fixed point with
//      fixed point	lsb = 2^(-15).
//      		Z_1 looks like z_0.z_1 z_2 ... z_15
//      	        Note that the fetching is done using index_1.
//      		A_1 is actually not needed in the implementation
//      		and is used here only to explain how is the value
//      		Z_1 defined.
//
//      Fetch		G_1 := (1/A_1) truncated to 21 sig. bits.
//      floating pt.	Again, fetching is done using index_1. A_1
//      		explains how G_1 is defined.
//
//      Calculate	X_1 := X_0 * Z_1 truncated to lsb = 2^(-14)
//      		     = 1.0 0 0 0 d_5 ... d_14
//      		This is accomplished by integer multiplication.
//      		It is proved that X_1 indeed always begin
//      		with 1.0000 in fixed point.
//
//
//      Define		A_2 := 1.0 0 0 0 d_5 d_6 d_7 d_8. This is X_1
//      		truncated to lsb = 2^(-8). Similar to A_1,
//      		A_2 is not needed in actual implementation. It
//      		helps explain how some of the values are defined.
//
//      Define		index_2 := [ d_5 d_6 d_7 d_8 ].
//
//      Fetch 		Z_2 := (1/A_2) rounded UP in fixed point with
//      fixed point	lsb = 2^(-15). Fetch done using index_2.
//      		Z_2 looks like z_0.z_1 z_2 ... z_15
//
//      Fetch		G_2 := (1/A_2) truncated to 21 sig. bits.
//      floating pt.
//
//      Calculate	X_2 := X_1 * Z_2 truncated to lsb = 2^(-14)
//      		     = 1.0 0 0 0 0 0 0 0 d_9 d_10 ... d_14
//      		This is accomplished by integer multiplication.
//      		It is proved that X_2 indeed always begin
//      		with 1.00000000 in fixed point.
//
//
//      Define		A_3 := 1.0 0 0 0 0 0 0 0 d_9 d_10 d_11 d_12 d_13 1.
//      		This is 2^(-14) + X_2 truncated to lsb = 2^(-13).
//
//      Define		index_3 := [ d_9 d_10 d_11 d_12 d_13 ].
//
//      Fetch		G_3 := (1/A_3) truncated to 21 sig. bits.
//      floating pt.	Fetch is done using index_3.
//
//      Compute		G := G_1 * G_2 * G_3.
//
//      This is done exactly since each of G_j only has 21 sig. bits.
//
//      Compute
//
//      	r := (G*S_hi - 1)
//
//
//  Step 2. Approximation
//  ---------------------
//
//   This step computes an approximation to logl( 1 + r ) where r is the
//   reduced argument just obtained. It is proved that |r| <= 1.9*2^(-13);
//   thus logl(1+r) can be approximated by a short polynomial:
//
//      logl(1+r) ~=~ poly = r + Q1 r^2 + ... + Q4 r^5
//
//
//  Step 3. Reconstruction
//  ----------------------
//
//   This step computes the desired result of logl(X):
//
//      logl(X)  =   logl( 2^N * S_hi )
//      	  =   N*logl(2) + logl( S_hi )
//      	  =   N*logl(2) + logl(1/G) +
//      	      logl(1 + G*S_hi - 1 )
//
//   logl(2), logl(1/G_j) are stored as pairs of (single,double) numbers:
//   log2_hi, log2_lo, log1byGj_hi, log1byGj_lo. The high parts are
//   single-precision numbers and the low parts are double precision
//   numbers. These have the property that
//
//      N*log2_hi + SUM ( log1byGj_hi )
//
//   is computable exactly in double-extended precision (64 sig. bits).
//   Finally
//
//      Y_hi := N*log2_hi + SUM ( log1byGj_hi )
//      Y_lo := poly_hi + [ poly_lo +
//              ( SUM ( log1byGj_lo ) + N*log2_lo ) ]
//

RODATA
.align 64

// ************* DO NOT CHANGE THE ORDER OF THESE TABLES *************

// P_8, P_7, P_6, P_5, P_4, P_3, P_2, and P_1

LOCAL_OBJECT_START(Constants_P)
data8  0xE3936754EFD62B15,0x00003FFB
data8  0x8003B271A5E56381,0x0000BFFC
data8  0x9249248C73282DB0,0x00003FFC
data8  0xAAAAAA9F47305052,0x0000BFFC
data8  0xCCCCCCCCCCD17FC9,0x00003FFC
data8  0x8000000000067ED5,0x0000BFFD
data8  0xAAAAAAAAAAAAAAAA,0x00003FFD
data8  0xFFFFFFFFFFFFFFFE,0x0000BFFD
LOCAL_OBJECT_END(Constants_P)

// log2_hi, log2_lo, Q_4, Q_3, Q_2, and Q_1

LOCAL_OBJECT_START(Constants_Q)
data8  0xB172180000000000,0x00003FFE
data8  0x82E308654361C4C6,0x0000BFE2
data8  0xCCCCCAF2328833CB,0x00003FFC
data8  0x80000077A9D4BAFB,0x0000BFFD
data8  0xAAAAAAAAAAABE3D2,0x00003FFD
data8  0xFFFFFFFFFFFFDAB7,0x0000BFFD
LOCAL_OBJECT_END(Constants_Q)

// 1/ln10_hi, 1/ln10_lo

LOCAL_OBJECT_START(Constants_1_by_LN10)
data8  0xDE5BD8A937287195,0x00003FFD
data8  0xD56EAABEACCF70C8,0x00003FBB
LOCAL_OBJECT_END(Constants_1_by_LN10)


// Z1 - 16 bit fixed

LOCAL_OBJECT_START(Constants_Z_1)
data4  0x00008000
data4  0x00007879
data4  0x000071C8
data4  0x00006BCB
data4  0x00006667
data4  0x00006187
data4  0x00005D18
data4  0x0000590C
data4  0x00005556
data4  0x000051EC
data4  0x00004EC5
data4  0x00004BDB
data4  0x00004925
data4  0x0000469F
data4  0x00004445
data4  0x00004211
LOCAL_OBJECT_END(Constants_Z_1)

// G1 and H1 - IEEE single and h1 - IEEE double

LOCAL_OBJECT_START(Constants_G_H_h1)
data4  0x3F800000,0x00000000
data8  0x0000000000000000
data4  0x3F70F0F0,0x3D785196
data8  0x3DA163A6617D741C
data4  0x3F638E38,0x3DF13843
data8  0x3E2C55E6CBD3D5BB
data4  0x3F579430,0x3E2FF9A0
data8  0xBE3EB0BFD86EA5E7
data4  0x3F4CCCC8,0x3E647FD6
data8  0x3E2E6A8C86B12760
data4  0x3F430C30,0x3E8B3AE7
data8  0x3E47574C5C0739BA
data4  0x3F3A2E88,0x3EA30C68
data8  0x3E20E30F13E8AF2F
data4  0x3F321640,0x3EB9CEC8
data8  0xBE42885BF2C630BD
data4  0x3F2AAAA8,0x3ECF9927
data8  0x3E497F3497E577C6
data4  0x3F23D708,0x3EE47FC5
data8  0x3E3E6A6EA6B0A5AB
data4  0x3F1D89D8,0x3EF8947D
data8  0xBDF43E3CD328D9BE
data4  0x3F17B420,0x3F05F3A1
data8  0x3E4094C30ADB090A
data4  0x3F124920,0x3F0F4303
data8  0xBE28FBB2FC1FE510
data4  0x3F0D3DC8,0x3F183EBF
data8  0x3E3A789510FDE3FA
data4  0x3F088888,0x3F20EC80
data8  0x3E508CE57CC8C98F
data4  0x3F042108,0x3F29516A
data8  0xBE534874A223106C
LOCAL_OBJECT_END(Constants_G_H_h1)

// Z2 - 16 bit fixed

LOCAL_OBJECT_START(Constants_Z_2)
data4  0x00008000
data4  0x00007F81
data4  0x00007F02
data4  0x00007E85
data4  0x00007E08
data4  0x00007D8D
data4  0x00007D12
data4  0x00007C98
data4  0x00007C20
data4  0x00007BA8
data4  0x00007B31
data4  0x00007ABB
data4  0x00007A45
data4  0x000079D1
data4  0x0000795D
data4  0x000078EB
LOCAL_OBJECT_END(Constants_Z_2)

// G2 and H2 - IEEE single and h2 - IEEE double

LOCAL_OBJECT_START(Constants_G_H_h2)
data4  0x3F800000,0x00000000
data8  0x0000000000000000
data4  0x3F7F00F8,0x3B7F875D
data8  0x3DB5A11622C42273
data4  0x3F7E03F8,0x3BFF015B
data8  0x3DE620CF21F86ED3
data4  0x3F7D08E0,0x3C3EE393
data8  0xBDAFA07E484F34ED
data4  0x3F7C0FC0,0x3C7E0586
data8  0xBDFE07F03860BCF6
data4  0x3F7B1880,0x3C9E75D2
data8  0x3DEA370FA78093D6
data4  0x3F7A2328,0x3CBDC97A
data8  0x3DFF579172A753D0
data4  0x3F792FB0,0x3CDCFE47
data8  0x3DFEBE6CA7EF896B
data4  0x3F783E08,0x3CFC15D0
data8  0x3E0CF156409ECB43
data4  0x3F774E38,0x3D0D874D
data8  0xBE0B6F97FFEF71DF
data4  0x3F766038,0x3D1CF49B
data8  0xBE0804835D59EEE8
data4  0x3F757400,0x3D2C531D
data8  0x3E1F91E9A9192A74
data4  0x3F748988,0x3D3BA322
data8  0xBE139A06BF72A8CD
data4  0x3F73A0D0,0x3D4AE46F
data8  0x3E1D9202F8FBA6CF
data4  0x3F72B9D0,0x3D5A1756
data8  0xBE1DCCC4BA796223
data4  0x3F71D488,0x3D693B9D
data8  0xBE049391B6B7C239
LOCAL_OBJECT_END(Constants_G_H_h2)

// G3 and H3 - IEEE single and h3 - IEEE double

LOCAL_OBJECT_START(Constants_G_H_h3)
data4  0x3F7FFC00,0x38800100
data8  0x3D355595562224CD
data4  0x3F7FF400,0x39400480
data8  0x3D8200A206136FF6
data4  0x3F7FEC00,0x39A00640
data8  0x3DA4D68DE8DE9AF0
data4  0x3F7FE400,0x39E00C41
data8  0xBD8B4291B10238DC
data4  0x3F7FDC00,0x3A100A21
data8  0xBD89CCB83B1952CA
data4  0x3F7FD400,0x3A300F22
data8  0xBDB107071DC46826
data4  0x3F7FCC08,0x3A4FF51C
data8  0x3DB6FCB9F43307DB
data4  0x3F7FC408,0x3A6FFC1D
data8  0xBD9B7C4762DC7872
data4  0x3F7FBC10,0x3A87F20B
data8  0xBDC3725E3F89154A
data4  0x3F7FB410,0x3A97F68B
data8  0xBD93519D62B9D392
data4  0x3F7FAC18,0x3AA7EB86
data8  0x3DC184410F21BD9D
data4  0x3F7FA420,0x3AB7E101
data8  0xBDA64B952245E0A6
data4  0x3F7F9C20,0x3AC7E701
data8  0x3DB4B0ECAABB34B8
data4  0x3F7F9428,0x3AD7DD7B
data8  0x3D9923376DC40A7E
data4  0x3F7F8C30,0x3AE7D474
data8  0x3DC6E17B4F2083D3
data4  0x3F7F8438,0x3AF7CBED
data8  0x3DAE314B811D4394
data4  0x3F7F7C40,0x3B03E1F3
data8  0xBDD46F21B08F2DB1
data4  0x3F7F7448,0x3B0BDE2F
data8  0xBDDC30A46D34522B
data4  0x3F7F6C50,0x3B13DAAA
data8  0x3DCB0070B1F473DB
data4  0x3F7F6458,0x3B1BD766
data8  0xBDD65DDC6AD282FD
data4  0x3F7F5C68,0x3B23CC5C
data8  0xBDCDAB83F153761A
data4  0x3F7F5470,0x3B2BC997
data8  0xBDDADA40341D0F8F
data4  0x3F7F4C78,0x3B33C711
data8  0x3DCD1BD7EBC394E8
data4  0x3F7F4488,0x3B3BBCC6
data8  0xBDC3532B52E3E695
data4  0x3F7F3C90,0x3B43BAC0
data8  0xBDA3961EE846B3DE
data4  0x3F7F34A0,0x3B4BB0F4
data8  0xBDDADF06785778D4
data4  0x3F7F2CA8,0x3B53AF6D
data8  0x3DCC3ED1E55CE212
data4  0x3F7F24B8,0x3B5BA620
data8  0xBDBA31039E382C15
data4  0x3F7F1CC8,0x3B639D12
data8  0x3D635A0B5C5AF197
data4  0x3F7F14D8,0x3B6B9444
data8  0xBDDCCB1971D34EFC
data4  0x3F7F0CE0,0x3B7393BC
data8  0x3DC7450252CD7ADA
data4  0x3F7F04F0,0x3B7B8B6D
data8  0xBDB68F177D7F2A42
LOCAL_OBJECT_END(Constants_G_H_h3)


// Floating Point Registers

FR_Input_X      = f8

FR_Y_hi         = f34
FR_Y_lo         = f35

FR_Scale        = f36
FR_X_Prime      = f37
FR_S_hi         = f38
FR_W            = f39
FR_G            = f40

FR_H            = f41
FR_wsq          = f42
FR_w4           = f43
FR_h            = f44
FR_w6           = f45

FR_G2           = f46
FR_H2           = f47
FR_poly_lo      = f48
FR_P8           = f49
FR_poly_hi      = f50

FR_P7           = f51
FR_h2           = f52
FR_rsq          = f53
FR_P6           = f54
FR_r            = f55

FR_log2_hi      = f56
FR_log2_lo      = f57
FR_p87          = f58
FR_p876         = f58
FR_p8765        = f58
FR_float_N      = f59
FR_Q4           = f60

FR_p43          = f61
FR_p432         = f61
FR_p4321        = f61
FR_P4           = f62
FR_G3           = f63
FR_H3           = f64
FR_h3           = f65

FR_Q3           = f66
FR_P3           = f67
FR_Q2           = f68
FR_P2           = f69
FR_1LN10_hi     = f70

FR_Q1           = f71
FR_P1           = f72
FR_1LN10_lo     = f73
FR_P5           = f74
FR_rcub         = f75

FR_Output_X_tmp = f76

FR_X                = f8
FR_Y                = f0
FR_RESULT           = f76


// General Purpose Registers

GR_ad_p         = r33
GR_Index1       = r34
GR_Index2       = r35
GR_signif       = r36
GR_X_0          = r37
GR_X_1          = r38
GR_X_2          = r39
GR_Z_1          = r40
GR_Z_2          = r41
GR_N            = r42
GR_Bias         = r43
GR_M            = r44
GR_Index3       = r45
GR_ad_p2        = r46
GR_exp_mask     = r47
GR_exp_2tom7    = r48
GR_ad_ln10      = r49
GR_ad_tbl_1     = r50
GR_ad_tbl_2     = r51
GR_ad_tbl_3     = r52
GR_ad_q         = r53
GR_ad_z_1       = r54
GR_ad_z_2       = r55
GR_ad_z_3       = r56

//
// Added for unwind support
//

GR_SAVE_PFS         = r50
GR_SAVE_B0          = r51
GR_SAVE_GP          = r52
GR_Parameter_X      = r53
GR_Parameter_Y      = r54
GR_Parameter_RESULT = r55
GR_Parameter_TAG    = r56

.section .text

GLOBAL_IEEE754_ENTRY(logl)
{ .mfi
      alloc r32 = ar.pfs,0,21,4,0
      fclass.m p6, p0 =  FR_Input_X, 0x1E3  // Test for natval, nan, inf
      cmp.eq  p7, p14 = r0, r0              // Set p7 if logl
}
{ .mfb
      addl GR_ad_z_1 = @ltoff(Constants_Z_1#),gp
      fnorm.s1 FR_X_Prime = FR_Input_X      // Normalize x
      br.cond.sptk LOGL_BEGIN
}
;;

GLOBAL_IEEE754_END(logl)
libm_alias_ldouble_other (__log, log)


GLOBAL_IEEE754_ENTRY(log10l)
{ .mfi
      alloc r32 = ar.pfs,0,21,4,0
      fclass.m p6, p0 =  FR_Input_X, 0x1E3  // Test for natval, nan, inf
      cmp.ne  p7, p14 = r0, r0              // Set p14 if log10l
}
{ .mfb
      addl GR_ad_z_1 = @ltoff(Constants_Z_1#),gp
      fnorm.s1 FR_X_Prime = FR_Input_X      // Normalize x
      nop.b 999
}
;;


// Common code for logl and log10
LOGL_BEGIN:
{ .mfi
      ld8    GR_ad_z_1 = [GR_ad_z_1]          // Get pointer to Constants_Z_1
      fclass.m p10, p0 =  FR_Input_X, 0x0b    // Test for denormal
      mov GR_exp_2tom7 = 0x0fff8              // Exponent of 2^-7
}
;;

{ .mfb
      getf.sig GR_signif = FR_Input_X         // Get significand of x
      fcmp.eq.s1 p9, p0 =  FR_Input_X, f1     // Test for x=1.0
(p6)  br.cond.spnt LOGL_64_special            // Branch for nan, inf, natval
}
;;

{ .mfi
      add   GR_ad_tbl_1 = 0x040, GR_ad_z_1    // Point to Constants_G_H_h1
      fcmp.lt.s1 p13, p0 =  FR_Input_X, f0    // Test for x<0
      add   GR_ad_p = -0x100, GR_ad_z_1       // Point to Constants_P
}
{ .mib
      add   GR_ad_z_2 = 0x140, GR_ad_z_1      // Point to Constants_Z_2
      add   GR_ad_tbl_2 = 0x180, GR_ad_z_1    // Point to Constants_G_H_h2
(p10) br.cond.spnt LOGL_64_denormal           // Branch for denormal
}
;;

LOGL_64_COMMON:
{ .mfi
      add   GR_ad_q = 0x080, GR_ad_p          // Point to Constants_Q
      fcmp.eq.s1 p8, p0 =  FR_Input_X, f0     // Test for x=0
      extr.u GR_Index1 = GR_signif, 59, 4     // Get high 4 bits of signif
}
{ .mfb
      add   GR_ad_tbl_3 = 0x280, GR_ad_z_1    // Point to Constants_G_H_h3
(p9)  fma.s0  f8 = FR_Input_X, f0, f0         // If x=1, return +0.0
(p9)  br.ret.spnt  b0                         // Exit if x=1
}
;;

{ .mfi
      shladd GR_ad_z_1 = GR_Index1, 2, GR_ad_z_1  // Point to Z_1
      fclass.nm p10, p0 =  FR_Input_X, 0x1FF  // Test for unsupported
      extr.u GR_X_0 = GR_signif, 49, 15       // Get high 15 bits of significand
}
{ .mfi
      ldfe FR_P8 = [GR_ad_p],16               // Load P_8 for near1 path
      fsub.s1 FR_W = FR_X_Prime, f1           // W = x - 1
      add   GR_ad_ln10 = 0x060, GR_ad_q       // Point to Constants_1_by_LN10
}
;;

{ .mfi
      ld4 GR_Z_1 = [GR_ad_z_1]                // Load Z_1
      nop.f 999
      mov GR_exp_mask = 0x1FFFF               // Create exponent mask
}
{ .mib
      shladd GR_ad_tbl_1 = GR_Index1, 4, GR_ad_tbl_1  // Point to G_1
      mov GR_Bias = 0x0FFFF                   // Create exponent bias
(p13) br.cond.spnt LOGL_64_negative           // Branch if x<0
}
;;

{ .mfb
      ldfps  FR_G, FR_H = [GR_ad_tbl_1],8     // Load G_1, H_1
      fmerge.se FR_S_hi =  f1,FR_X_Prime      // Form |x|
(p8)  br.cond.spnt LOGL_64_zero               // Branch if x=0
}
;;

{ .mmb
      getf.exp GR_N =  FR_X_Prime             // Get N = exponent of x
      ldfd  FR_h = [GR_ad_tbl_1]              // Load h_1
(p10) br.cond.spnt LOGL_64_unsupported        // Branch for unsupported type
}
;;

{ .mfi
      ldfe FR_log2_hi = [GR_ad_q],16          // Load log2_hi
      fcmp.eq.s0 p8, p0 =  FR_Input_X, f0     // Dummy op to flag denormals
      pmpyshr2.u GR_X_1 = GR_X_0,GR_Z_1,15    // Get bits 30-15 of X_0 * Z_1
}
;;

//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mmi
      ldfe FR_log2_lo = [GR_ad_q],16          // Load log2_lo
(p14) ldfe FR_1LN10_hi = [GR_ad_ln10],16      // If log10l, load 1/ln10_hi
      sub GR_N = GR_N, GR_Bias
}
;;

{ .mmi
      ldfe FR_Q4 = [GR_ad_q],16               // Load Q4
(p14) ldfe FR_1LN10_lo = [GR_ad_ln10]         // If log10l, load 1/ln10_lo
      nop.i 999
}
;;

{ .mmi
      ldfe FR_Q3 = [GR_ad_q],16               // Load Q3
      setf.sig FR_float_N = GR_N   // Put integer N into rightmost significand
      nop.i 999
}
;;

{ .mmi
      getf.exp GR_M = FR_W                    // Get signexp of w = x - 1
      ldfe FR_Q2 = [GR_ad_q],16               // Load Q2
      extr.u GR_Index2 = GR_X_1, 6, 4         // Extract bits 6-9 of X_1
}
;;

{ .mmi
      ldfe FR_Q1 = [GR_ad_q]                  // Load Q1
      shladd GR_ad_z_2 = GR_Index2, 2, GR_ad_z_2  // Point to Z_2
      add GR_ad_p2  = 0x30,GR_ad_p            // Point to P_4
}
;;

{ .mmi
      ld4 GR_Z_2 = [GR_ad_z_2]                // Load Z_2
      shladd GR_ad_tbl_2 = GR_Index2, 4, GR_ad_tbl_2  // Point to G_2
      and GR_M = GR_exp_mask, GR_M            // Get exponent of w = x - 1
}
;;

{ .mmi
      ldfps  FR_G2, FR_H2 = [GR_ad_tbl_2],8   // Load G_2, H_2
      cmp.lt  p8, p9 =  GR_M, GR_exp_2tom7    // Test |x-1| < 2^-7
      nop.i 999
}
;;

// Paths are merged.
//  p8 is for the near1 path: |x-1| < 2^-7
//  p9 is for regular path:   |x-1| >= 2^-7

{ .mmi
      ldfd  FR_h2 = [GR_ad_tbl_2]             // Load h_2
      nop.m 999
      nop.i 999
}
;;

{ .mmi
(p8)  ldfe FR_P7 = [GR_ad_p],16               // Load P_7 for near1 path
(p8)  ldfe FR_P4 = [GR_ad_p2],16              // Load P_4 for near1 path
(p9)  pmpyshr2.u GR_X_2 = GR_X_1,GR_Z_2,15    // Get bits 30-15 of X_1 * Z_2
}
;;

//
//    For performance, don't use result of pmpyshr2.u for 4 cycles.
//
{ .mmi
(p8)  ldfe FR_P6 = [GR_ad_p],16               // Load P_6 for near1 path
(p8)  ldfe FR_P3 = [GR_ad_p2],16              // Load P_3 for near1 path
      nop.i 999
}
;;

{ .mmf
(p8)  ldfe FR_P5 = [GR_ad_p],16               // Load P_5 for near1 path
(p8)  ldfe FR_P2 = [GR_ad_p2],16              // Load P_2 for near1 path
(p8)  fmpy.s1 FR_wsq = FR_W, FR_W             // wsq = w * w for near1 path
}
;;

{ .mmi
(p8)  ldfe FR_P1 = [GR_ad_p2],16 ;;           // Load P_1 for near1 path
      nop.m 999
(p9)  extr.u GR_Index3 = GR_X_2, 1, 5         // Extract bits 1-5 of X_2
}
;;

{ .mfi
(p9)  shladd GR_ad_tbl_3 = GR_Index3, 4, GR_ad_tbl_3  // Point to G_3
(p9)  fcvt.xf FR_float_N = FR_float_N
      nop.i 999
}
;;

{ .mfi
(p9)  ldfps  FR_G3, FR_H3 = [GR_ad_tbl_3],8   // Load G_3, H_3
      nop.f 999
      nop.i 999
}
;;

{ .mfi
(p9)  ldfd  FR_h3 = [GR_ad_tbl_3]             // Load h_3
(p9)  fmpy.s1 FR_G = FR_G, FR_G2              // G = G_1 * G_2
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fadd.s1 FR_H = FR_H, FR_H2              // H = H_1 + H_2
      nop.i 999
}
;;

{ .mmf
      nop.m 999
      nop.m 999
(p9)  fadd.s1 FR_h = FR_h, FR_h2              // h = h_1 + h_2
}
;;

{ .mfi
      nop.m 999
(p8)  fmpy.s1 FR_w4 = FR_wsq, FR_wsq          // w4 = w^4 for near1 path
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1 FR_p87 = FR_W, FR_P8, FR_P7      // p87 = w * P8 + P7
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fma.s1 FR_p43 = FR_W, FR_P4, FR_P3      // p43 = w * P4 + P3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fmpy.s1 FR_G = FR_G, FR_G3              // G = (G_1 * G_2) * G_3
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fadd.s1 FR_H = FR_H, FR_H3              // H = (H_1 + H_2) + H_3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fadd.s1 FR_h = FR_h, FR_h3              // h = (h_1 + h_2) + h_3
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fmpy.s1 FR_w6 = FR_w4, FR_wsq           // w6 = w^6 for near1 path
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fma.s1 FR_p432 = FR_W, FR_p43, FR_P2    // p432 = w * p43 + P2
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1 FR_p876 = FR_W, FR_p87, FR_P6    // p876 = w * p87 + P6
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fms.s1 FR_r = FR_G, FR_S_hi, f1         // r = G * S_hi - 1
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1 FR_Y_hi = FR_float_N, FR_log2_hi, FR_H // Y_hi = N * log2_hi + H
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fma.s1 FR_h = FR_float_N, FR_log2_lo, FR_h  // h = N * log2_lo + h
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fma.s1 FR_p4321 = FR_W, FR_p432, FR_P1      // p4321 = w * p432 + P1
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1 FR_p8765 = FR_W, FR_p876, FR_P5      // p8765 = w * p876 + P5
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fma.s1 FR_poly_lo = FR_r, FR_Q4, FR_Q3      // poly_lo = r * Q4 + Q3
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fmpy.s1 FR_rsq = FR_r, FR_r                 // rsq = r * r
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fma.s1 FR_Y_lo = FR_wsq, FR_p4321, f0       // Y_lo = wsq * p4321
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fma.s1 FR_Y_hi = FR_W, f1, f0               // Y_hi = w for near1 path
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fma.s1 FR_poly_lo = FR_poly_lo, FR_r, FR_Q2 // poly_lo = poly_lo * r + Q2
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fma.s1 FR_rcub = FR_rsq, FR_r, f0           // rcub = r^3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fma.s1 FR_Y_lo = FR_w6, FR_p8765,FR_Y_lo // Y_lo = w6 * p8765 + w2 * p4321
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fma.s1 FR_poly_hi = FR_Q1, FR_rsq, FR_r     // poly_hi = Q1 * rsq + r
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fma.s1 FR_poly_lo = FR_poly_lo, FR_rcub, FR_h // poly_lo = poly_lo*r^3 + h
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fadd.s1 FR_Y_lo = FR_poly_hi, FR_poly_lo    // Y_lo = poly_hi + poly_lo
      nop.i 999
}
;;

// Remainder of code is common for near1 and regular paths
{ .mfi
      nop.m 999
(p7)  fadd.s0  f8 = FR_Y_lo,FR_Y_hi               // If logl, result=Y_lo+Y_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p14) fmpy.s1 FR_Output_X_tmp = FR_Y_lo,FR_1LN10_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p14) fma.s1  FR_Output_X_tmp = FR_Y_hi,FR_1LN10_lo,FR_Output_X_tmp
      nop.i 999
}
;;

{ .mfb
      nop.m 999
(p14) fma.s0  f8 = FR_Y_hi,FR_1LN10_hi,FR_Output_X_tmp
      br.ret.sptk   b0                        // Common exit for 0 < x < inf
}
;;


// Here if x=+-0
LOGL_64_zero:
//
//    If x=+-0 raise divide by zero and return -inf
//
{ .mfi
(p7)  mov   GR_Parameter_TAG = 0
      fsub.s1 FR_Output_X_tmp = f0, f1
      nop.i 999
}
;;

{ .mfb
(p14) mov   GR_Parameter_TAG = 6
      frcpa.s0 FR_Output_X_tmp, p8 =  FR_Output_X_tmp, f0
      br.cond.sptk __libm_error_region
}
;;

LOGL_64_special:
{ .mfi
      nop.m 999
      fclass.m.unc p8, p0 =  FR_Input_X, 0x1E1  // Test for natval, nan, +inf
      nop.i 999
}
;;

//
//    For SNaN raise invalid and return QNaN.
//    For QNaN raise invalid and return QNaN.
//    For +Inf return +Inf.
//
{ .mfb
      nop.m 999
(p8)  fmpy.s0 f8 =  FR_Input_X, f1
(p8)  br.ret.sptk   b0                          // Return for natval, nan, +inf
}
;;

//
//    For -Inf raise invalid and return QNaN.
//
{ .mmi
(p7)  mov   GR_Parameter_TAG = 1
      nop.m 999
      nop.i 999
}
;;

{ .mfb
(p14) mov   GR_Parameter_TAG = 7
      fmpy.s0 FR_Output_X_tmp =  FR_Input_X, f0
      br.cond.sptk __libm_error_region
}
;;

// Here if x denormal or unnormal
LOGL_64_denormal:
{ .mmi
      getf.sig GR_signif = FR_X_Prime   // Get significand of normalized input
      nop.m 999
      nop.i 999
}
;;

{ .mmb
      getf.exp GR_N =  FR_X_Prime       // Get exponent of normalized input
      nop.m 999
      br.cond.sptk   LOGL_64_COMMON     // Branch back to common code
}
;;

LOGL_64_unsupported:
//
//    Return generated NaN or other value.
//
{ .mfb
      nop.m 999
      fmpy.s0 f8 = FR_Input_X, f0
      br.ret.sptk   b0
}
;;

// Here if -inf < x < 0
LOGL_64_negative:
//
//    Deal with x < 0 in a special way - raise
//    invalid and produce QNaN indefinite.
//
{ .mfi
(p7)  mov   GR_Parameter_TAG = 1
      frcpa.s0 FR_Output_X_tmp, p8 =  f0, f0
      nop.i 999
}
;;

{ .mib
(p14) mov   GR_Parameter_TAG = 7
      nop.i 999
      br.cond.sptk __libm_error_region
}
;;


GLOBAL_IEEE754_END(log10l)
libm_alias_ldouble_other (__log10, log10)

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
        nop.m 999
        nop.m 999
        add   GR_Parameter_RESULT = 48,sp
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
