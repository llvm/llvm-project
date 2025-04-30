.file "atan2f.s"


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

// History
//==============================================================
// 06/01/00 Initial version
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 08/17/00 Changed predicate register macro-usage to direct predicate
//          names due to an assembler bug.
// 01/05/01 Fixed flag settings for denormal input.
// 01/19/01 Added documentation
// 01/30/01 Improved speed
// 02/06/02 Corrected .section statement
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align

// Description
//=========================================
// The atan2 function computes the principle value of the arc tangent of y/x using
// the signs of both arguments to determine the quadrant of the return value.
// A domain error may occur if both arguments are zero.

// The atan2 function returns the arc tangent of y/x in the range [-pi,+pi] radians.

//..
//..Let (v,u) = (y,x) if |y| <= |x|, and (v,u) = (x,y) otherwise. Note that
//..v and u can be negative. We state the relationship between atan2(y,x) and
//..atan(v/u).
//..
//..Let swap = false if v = y, and swap = true if v = x.
//..Define C according to the matrix
//..
//..                   TABLE FOR C
//..                              x +ve       x -ve
//..   no swap (swap = false)    sgn(y)*0     sgn(y)*pi
//..   swap    (swap = true )    sgn(y)*pi/2  sgn(y)*pi/2
//..
//..   atan2(y,x) =  C +  atan(v/u)  if no swap
//..   atan2(y,x) =  C -  atan(v/u)  if  swap
//..
//..These relationship is more efficient to compute as we accommodate signs in v and u
//..saving the need to obtain the absolute value before computation can proceed.
//..
//..Suppose (v,u) = (y,x), we calculate atan(v/u) as follows:
//..A = y * frcpa(x)    (so A = (y/x)(1 - beta))
//..atan(y/x) = atan(A) + atan( ((y/x)-A))/(1 + (y/x)A) ), the second term is
//..a correction.
//..atan(A) is approximated by a polynomial
//..A + p1 A^3 + p2 A^5 + ... + p10 A^21,
//..atan(G) is approximated as follows:
//..Let G = (y - Ax)/(x + Ay), atan(G) can be approximated by G + g * p1
//..where g is a limited precision approximation to G via g = (y - Ax)*frcpa(x + Ay).
//..
//..Suppose (v,u) = (x,y), we calculate atan(v/u) as follows:
//..Z = x * frcpa(y)    (so Z = (x/y)(1 - beta))
//..atan(x/y) = atan(Z) + atan( ((x/y)-Z))/(1 + (x/y)Z) ), the second term is
//..a correction.
//..atan(Z) is approximated by a polynomial
//..Z + p1 Z^3 + p2 Z^5 + ... + p10 Z^21,
//..atan(T) is approximated as follows:
//..Let T = (x - Ay)/(y + Ax), atan(T) can be approximated by T + t * p1
//..where t is a limited precision approximation to T via t = (x - Ay)*frcpa(y + Ax).
//..
//..
//..A = y * frcpa(x)
//..atan(A) ~=~ A + p1 A^3 + ... + P10 A^21
//..
//..This polynomial is computed as follows:
//..Asq = A*A; Acub = A*Asq, A4 = Asq*Asq
//..A5 = Asq*Acub, A6 = Asq*A4; A11 = A5 * A6
//..
//..poly_A1 = p9 + Asq*p10, poly_A2 = p7 + Asq*p8, poly_A3 = p5 + Asq*p6
//..poly_A1 = poly_A2 + A4 * poly_A1
//..poly_A1 = poly_A3 + A4 * poly_A1
//..
//..poly_A4 = p1 * A
//,,poly_A5 = p3 + Asq * p4, poly_A4 = A + Asq*poly_A4
//..poly_A5 = p2 + Asq * poly_A5
//..poly_A4 = poly_A4 + A5 * poly_A5
//..
//..atan_A = poly_A4 + A11 * poly_A1
//..
//..atan(G) is approximated as follows:
//..G_numer = y - A*x, G_denom = x + A*y
//..H1 = frcpa(G_denom)
//..H_beta = 1 - H1 * G_denom
//..H2 = H1 + H1 * H_beta
//..H_beta2 = H_beta*H_beta
//..H3 = H2 + H2*H_beta2
//..g = H1 * G_numer; gsq = g*g; atan_G = g*p1, atan_G = atan_G*gsq
//..atan_G = G_numer*H3 + atan_G
//..
//..
//..A = y * frcpa(x)
//..atan(A) ~=~ A + p1 A^3 + ... + P10 A^21
//..
//..This polynomial is computed as follows:
//..Asq = A*A; Acub = A*Asq, A4 = Asq*Asq
//..A5 = Asq*Acub, A6 = Asq*A4; A11 = A5 * A6
//..
//..poly_A1 = p9 + Asq*p10, poly_A2 = p7 + Asq*p8, poly_A3 = p5 + Asq*p6
//..poly_A1 = poly_A2 + A4 * poly_A1
//..poly_A1 = poly_A3 + A4 * poly_A1
//..
//..poly_A4 = p1 * A
//,,poly_A5 = p3 + Asq * p4, poly_A4 = A + Asq*poly_A4
//..poly_A5 = p2 + Asq * poly_A5
//..poly_A4 = poly_A4 + A5 * poly_A5
//..
//..atan_A = poly_A4 + A11 * poly_A1
//..
//..
//..====================================================================
//..	COEFFICIENTS USED IN THE COMPUTATION
//..====================================================================

//coef_pj, j = 1,2,...,10;  atan(A) ~=~ A + p1 A^3 + p2 A^5 + ... + p10 A^21
//
//  coef_p1          =      -.3333332707155439167401311806315789E+00
//  coef_p1   in dbl = BFD5 5555 1219 1621
//
//  coef_p2          =       .1999967670926658391827857030875748E+00
//  coef_p2   in dbl = 3FC9 997E 7AFB FF4E
//
//  coef_p3          =      -.1427989384500152360161563301087296E+00
//  coef_p3   in dbl = BFC2 473C 5145 EE38
//
//  coef_p4          =       .1105852823460720770079031213661163E+00
//  coef_p4   in dbl = 3FBC 4F51 2B18 65F5
//
//  coef_p5          =      -.8811839915595312348625710228448363E-01
//  coef_p5   in dbl = BFB6 8EED 6A8C FA32
//
//  coef_p6          =       .6742329836955067042153645159059714E-01
//  coef_p6   in dbl = 3FB1 42A7 3D7C 54E3
//
//  coef_p7          =      -.4468571068774672908561591262231909E-01
//  coef_p7   in dbl = BFA6 E10B A401 393F
//
//  coef_p8          =       .2252333246746511135532726960586493E-01
//  coef_p8   in dbl = 3F97 105B 4160 F86B
//
//  coef_p9          =      -.7303884867007574742501716845542314E-02
//  coef_p9   in dbl = BF7D EAAD AA33 6451
//
//  coef_p10         =       .1109686868355312093949039454619058E-02
//  coef_p10  in dbl = 3F52 2E5D 33BC 9BAA
//

// Special values
//==============================================================
//              Y                 x          Result
//             +number           +inf        +0
//             -number           +inf        -0
//             +number           -inf        +pi
//             -number           -inf        -pi
//
//             +inf              +number     +pi/2
//             -inf              +number     -pi/2
//             +inf              -number     +pi/2
//             -inf              -number     -pi/2
//
//             +inf              +inf        +pi/4
//             -inf              +inf        -pi/4
//             +inf              -inf        +3pi/4
//             -inf              -inf        -3pi/4
//
//             +1                +1          +pi/4
//             -1                +1          -pi/4
//             +1                -1          +3pi/4
//             -1                -1          -3pi/4
//
//             +number           +0          +pi/2    // does not raise DBZ
//             -number           +0          -pi/2    // does not raise DBZ
//             +number           -0          +pi/2    // does not raise DBZ
//             -number           -0          -pi/2    // does not raise DBZ
//
//             +0                +number     +0
//             -0                +number     -0
//             +0                -number     +pi
//             -0                -number     -pi
//
//             +0                +0          +0      // does not raise invalid
//             -0                +0          -0      // does not raise invalid
//             +0                -0          +pi     // does not raise invalid
//             -0                -0          -pi     // does not raise invalid
//
//            Nan             anything      quiet Y
//            anything        NaN           quiet X

// atan2(+-0/+-0) sets double error tag to 37
// atan2f(+-0/+-0) sets single error tag to 38
// These are domain errors.


//
// Assembly macros
//=========================================


// integer registers
atan2f_GR_Addr_1              = r33
atan2f_GR_Addr_2              = r34
GR_SAVE_B0                    = r35

GR_SAVE_PFS                   = r36
GR_SAVE_GP                    = r37

GR_Parameter_X                = r38
GR_Parameter_Y                = r39
GR_Parameter_RESULT           = r40
GR_Parameter_TAG              = r41

// floating point registers
atan2f_coef_p1         = f32
atan2f_coef_p10        = f33
atan2f_coef_p7         = f34
atan2f_coef_p6         = f35

atan2f_coef_p3         = f36
atan2f_coef_p2         = f37
atan2f_coef_p9         = f38
atan2f_coef_p8         = f39
atan2f_coef_p5         = f40

atan2f_coef_p4         = f41
atan2f_const_piby2     = f42
atan2f_const_pi        = f43
atan2f_const_piby4     = f44
atan2f_const_3piby4    = f45

atan2f_xsq             = f46
atan2f_ysq             = f47
atan2f_xy              = f48
atan2f_const_1         = f49
atan2f_sgn_Y           = f50

atan2f_Z0              = f51
atan2f_A0              = f52
atan2f_Z               = f53
atan2f_A               = f54
atan2f_C               = f55

atan2f_U               = f56
atan2f_Usq             = f57
atan2f_U4              = f58
atan2f_U6              = f59
atan2f_U8              = f60

atan2f_poly_u109       = f61
atan2f_poly_u87        = f62
atan2f_poly_u65        = f63
atan2f_poly_u43        = f64
atan2f_poly_u21        = f65

atan2f_poly_u10to7     = f66
atan2f_poly_u6to3      = f67
atan2f_poly_u10to3     = f68
atan2f_poly_u10to0     = f69
atan2f_poly_u210       = f70

atan2f_T_numer         = f71
atan2f_T_denom         = f72
atan2f_G_numer         = f73
atan2f_G_denom         = f74
atan2f_p1rnum          = f75

atan2f_R_denom         = f76
atan2f_R_numer         = f77
atan2f_pR              = f78
atan2f_pRC             = f79
atan2f_pQRC            = f80

atan2f_Q1              = f81
atan2f_Q_beta          = f82
atan2f_Q2              = f83
atan2f_Q_beta2         = f84
atan2f_Q3              = f85

atan2f_r               = f86
atan2f_rsq             = f87
atan2f_poly_atan_U     = f88


// predicate registers
//atan2f_Pred_Swap     = p6 // |y| >  |x|
//atan2f_Pred_noSwap   = p7 // |y| <= |x|
//atan2f_Pred_Xpos     = p8 //  x  >=  0
//atan2f_Pred_Xneg     = p9 //  x  <   0


RODATA

.align 16

LOCAL_OBJECT_START(atan2f_coef_table1)
data8 0xBFD5555512191621 // p1
data8 0x3F522E5D33BC9BAA // p10
data8 0xBFA6E10BA401393F // p7
data8 0x3FB142A73D7C54E3 // p6
data8 0xBFC2473C5145EE38 // p3
data8 0x3FC9997E7AFBFF4E // p2
LOCAL_OBJECT_END(atan2f_coef_table1)

LOCAL_OBJECT_START(atan2f_coef_table2)
data8 0xBF7DEAADAA336451 // p9
data8 0x3F97105B4160F86B // p8
data8 0xBFB68EED6A8CFA32 // p5
data8 0x3FBC4F512B1865F5 // p4
data8 0x3ff921fb54442d18 // pi/2
data8 0x400921fb54442d18 // pi
data8 0x3fe921fb54442d18 // pi/4
data8 0x4002d97c7f3321d2 // 3pi/4
LOCAL_OBJECT_END(atan2f_coef_table2)



.section .text
GLOBAL_IEEE754_ENTRY(atan2f)

{     .mfi
     alloc      r32           = ar.pfs,1,5,4,0
     frcpa.s1  atan2f_Z0,p0     =    f1,f8   // Approx to 1/y
     nop.i  999
}
{     .mfi
     addl      atan2f_GR_Addr_1    =    @ltoff(atan2f_coef_table1),gp
     fma.s1    atan2f_xsq     =    f9,f9,f0
     nop.i  999 ;;
}


{     .mfi
     ld8       atan2f_GR_Addr_1    =    [atan2f_GR_Addr_1]
     frcpa.s1  atan2f_A0,p0     =    f1,f9   // Approx to 1/x
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_ysq     =    f8,f8,f0
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fcmp.ge.s1     p8,p9  =    f9,f0  // Set p8 if x>=0, p9 if x<0
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_xy     =    f9,f8,f0
     nop.i  999 ;;
}


{     .mfi
     add   atan2f_GR_Addr_2 = 0x30, atan2f_GR_Addr_1
     fmerge.s  atan2f_sgn_Y   =    f8,f1
     nop.i  999 ;;
}

{     .mmf
     ldfpd     atan2f_coef_p1,atan2f_coef_p10 =    [atan2f_GR_Addr_1],16
     ldfpd     atan2f_coef_p9,atan2f_coef_p8 =    [atan2f_GR_Addr_2],16
     fclass.m  p10,p0 =    f9,0xe7	// Test x @inf|@snan|@qnan|@zero
}
;;

{     .mfi
     ldfpd     atan2f_coef_p7,atan2f_coef_p6 =    [atan2f_GR_Addr_1],16
     fma.s1    atan2f_T_denom =    atan2f_Z0,atan2f_xsq,f8
     nop.i  999
}
{     .mfi
     ldfpd     atan2f_coef_p5,atan2f_coef_p4     =    [atan2f_GR_Addr_2],16
     fma.s1    atan2f_Z                      =    atan2f_Z0,f9,f0
     nop.i  999 ;;
}


{     .mfi
     ldfpd     atan2f_coef_p3,atan2f_coef_p2 =    [atan2f_GR_Addr_1],16
     fma.s1    atan2f_G_denom =    atan2f_A0,atan2f_ysq,f9
     nop.i  999
}
{     .mfi
     ldfpd     atan2f_const_piby2,atan2f_const_pi =    [atan2f_GR_Addr_2],16
     fma.s1    atan2f_A                           =    atan2f_A0,f8,f0
     nop.i  999 ;;
}

{     .mfi
     ldfpd     atan2f_const_piby4,atan2f_const_3piby4 = [atan2f_GR_Addr_2]
     fclass.m  p11,p0 = f8,0xe7	// Test y @inf|@snan|@qnan|@zero
     nop.i  999
}
{     .mfb
     nop.m  999
     fnma.s1   atan2f_T_numer =    atan2f_Z0,atan2f_xy,f9
(p10) br.cond.spnt ATAN2F_XY_INF_NAN_ZERO ;;   // Branch on x nan,inf,zero
}


// p6 if |y|>|x|, p7 if |x|>=|y| , use xsq and ysq for test
{     .mfi
     nop.m  999
     fcmp.gt.s1 p6,p7 = atan2f_ysq,atan2f_xsq
     nop.i  999
}
{     .mfb
     nop.m  999
     fnma.s1   atan2f_G_numer =    atan2f_A0,atan2f_xy,f8
(p11) br.cond.spnt ATAN2F_XY_INF_NAN_ZERO ;;  // Branch on y nan,inf,zero
}


{     .mfi
     nop.m  999
(p8) fma.s1    atan2f_const_1 =    atan2f_sgn_Y,f0,f0
     nop.i  999
}
{     .mfi
     nop.m  999
(p9) fma.s1    atan2f_const_1 =    atan2f_sgn_Y,f1,f0
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
(p6) fnma.s1    atan2f_U       =    atan2f_Z,f1,f0
     nop.i  999
}
{     .mfi
     nop.m  999
(p6) fma.s1    atan2f_Usq     =    atan2f_Z,atan2f_Z,f0
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
(p7) fma.s1    atan2f_U       =    atan2f_A,f1,f0
     nop.i  999
}
{     .mfi
     nop.m  999
(p7) fma.s1    atan2f_Usq     =    atan2f_A,atan2f_A,f0
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
(p6) frcpa.s1  atan2f_Q1,p0    =    f1,atan2f_T_denom
     nop.i  999
}
{     .mfi
     nop.m  999
(p6) fma.s1    atan2f_R_denom =   atan2f_T_denom,f1,f0
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
(p7) frcpa.s1  atan2f_Q1,p0    =    f1,atan2f_G_denom
     nop.i  999
}
{     .mfi
     nop.m  999
(p7) fma.s1    atan2f_R_denom =   atan2f_G_denom,f1,f0
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
(p6) fnma.s1    atan2f_R_numer =   atan2f_T_numer,f1,f0
     nop.i  999
}
{     .mfi
     nop.m  999
(p7) fma.s1    atan2f_R_numer =   atan2f_G_numer,f1,f0
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
(p6) fnma.s1    atan2f_p1rnum =   atan2f_T_numer,atan2f_coef_p1,f0
     nop.i  999 ;;
}
{     .mfi
     nop.m  999
(p7) fma.s1    atan2f_p1rnum =   atan2f_G_numer,atan2f_coef_p1,f0
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
     fma.s1    atan2f_U4 =    atan2f_Usq,atan2f_Usq,f0
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u109 = atan2f_Usq,atan2f_coef_p10,atan2f_coef_p9
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u87 =    atan2f_Usq,atan2f_coef_p8,atan2f_coef_p7
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u65 =    atan2f_Usq,atan2f_coef_p6,atan2f_coef_p5
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u43 =    atan2f_Usq,atan2f_coef_p4,atan2f_coef_p3
     nop.i  999
}
{     .mfi
     nop.m  999
     fnma.s1   atan2f_Q_beta  =    atan2f_Q1,atan2f_R_denom,f1
     nop.i  999 ;;
}


{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u21 =    atan2f_Usq,atan2f_coef_p2,atan2f_coef_p1
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_r  =    atan2f_Q1,atan2f_R_numer,f0
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
(p6) fma.s1    atan2f_C  =    atan2f_sgn_Y,atan2f_const_piby2,f0
     nop.i  999
}
{     .mfi
     nop.m  999
(p7) fma.s1    atan2f_C  =    atan2f_const_1,atan2f_const_pi,f0
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1    atan2f_U6 =    atan2f_U4,atan2f_Usq,f0
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_U8 =    atan2f_U4,atan2f_U4,f0
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u10to7 = atan2f_U4,atan2f_poly_u109,atan2f_poly_u87
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_pR = atan2f_p1rnum,atan2f_Q1,f0
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u6to3 = atan2f_U4,atan2f_poly_u65,atan2f_poly_u43
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_Q2 =    atan2f_Q1,atan2f_Q_beta,atan2f_Q1
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1    atan2f_Q_beta2 =    atan2f_Q_beta,atan2f_Q_beta,f0
     nop.i  999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_rsq     =    atan2f_r,atan2f_r,f0
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1    atan2f_poly_u210 = atan2f_Usq,atan2f_poly_u21,f1
     nop.i  999 ;;
}

{     .mfi
     nop.m 999
     fcmp.eq.s0 p8,p0 = f8,f9      // Dummy op to set flag on denormal inputs
     nop.i 999
}
{     .mfi
     nop.m  999
     fma.s1 atan2f_poly_u10to3 = atan2f_U8,atan2f_poly_u10to7,atan2f_poly_u6to3
     nop.i  999 ;;
}

{     .mfi
     nop.m                 999
     fma.s1    atan2f_Q3 =    atan2f_Q2,atan2f_Q_beta2,atan2f_Q2
     nop.i                 999
}
{     .mfi
     nop.m  999
     fma.s1    atan2f_pRC = atan2f_rsq,atan2f_pR,atan2f_C
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1 atan2f_poly_u10to0 = atan2f_U6,atan2f_poly_u10to3,atan2f_poly_u210
     nop.i  999 ;;
}

{     .mfi
     nop.m  999
     fma.s1    atan2f_pQRC = atan2f_R_numer,atan2f_Q3,atan2f_pRC
     nop.i  999 ;;
}

{     .mfb
     nop.m  999
     fma.s.s0    f8 = atan2f_U,atan2f_poly_u10to0,atan2f_pQRC
     br.ret.sptk b0 ;;
}



ATAN2F_XY_INF_NAN_ZERO:

{ .mfi
      nop.m 999
      fclass.m   p10,p0 = f8,0xc3	// Is y nan
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m   p12,p0 = f9,0xc3	// Is x nan
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m   p6,p0 = f9,0x21	// Is x +inf
      nop.i 999
}
{ .mfb
      nop.m 999
(p10) fma.s.s0 f8  = f9,f8,f0          // Result quietized y if y is nan
(p10) br.ret.spnt b0                // Exit if y is nan
}
;;


{ .mfi
      nop.m 999
(p6)  fclass.m.unc   p7,p8 = f8,0x23	// x +inf, is y inf
      nop.i 999
}
{ .mfb
      nop.m 999
(p12) fnorm.s.s0 f8 = f9               // Result quietized x if x is nan, y not nan
(p12) br.ret.spnt b0                // Exit if x is nan, y not nan
}
;;

// Here if x or y inf, or x or y zero
{ .mfi
      nop.m 999
      fcmp.eq.s0 p15,p0 = f8,f9     // Dummy op to set flag on denormal inputs
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m   p11,p12 = f9,0x22	// Is x -inf
      nop.i 999
}
{ .mfb
      nop.m 999
(p7)  fma.s.s0 f8 = atan2f_sgn_Y, atan2f_const_piby4,f0 // Result +-pi/4
(p7)  br.ret.spnt b0            // Exit if x +inf and y inf
}
;;

{ .mfb
      nop.m 999
(p8)  fmerge.s   f8 = f8,f0     // If x +inf and y not inf, result +-0
(p8)  br.ret.spnt b0            // Exit if x +inf and y not inf
}
;;

{ .mfi
      nop.m 999
(p12) fclass.m.unc   p13,p0 = f8,0x23	// x not -inf, is y inf
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p11) fclass.m.unc   p14,p15 = f8,0x23	// x -inf, is y inf
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m  p6,p7 = f9,0x7	// Is x zero
      nop.i 999
}
{ .mfb
      nop.m 999
(p13) fma.s.s0   f8 = atan2f_sgn_Y, atan2f_const_piby2,f0 // Result +-pi/2
(p13) br.ret.spnt b0           // Exit if x not -inf and y inf
}
;;

{ .mfi
      nop.m 999
(p14) fma.s.s0   f8 = atan2f_sgn_Y, atan2f_const_3piby4,f0 // Result +-3pi/4
      nop.i 999
}
{ .mfb
      nop.m 999
(p15) fma.s.s0   f8 = atan2f_sgn_Y, atan2f_const_pi,f0 // Result +-pi
(p11) br.ret.spnt b0           // Exit if x -inf
}
;;

// Here if x or y zero
{ .mfi
      nop.m 999
(p7)  fclass.m.unc   p8,p9 = f9,0x19	// x not zero, y zero, is x > zero
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m.unc   p10,p11 = f8,0x7	// x zero, is y zero
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p8)  fmerge.s  f8 = f8, f0  // x > zero and y zero, result is +-zero
      nop.i 999
}
{ .mfb
      nop.m 999
(p9)  fma.s.s0  f8 = atan2f_sgn_Y, atan2f_const_pi,f0 // x < 0, y 0, result +-pi
(p10) br.cond.spnt   __libm_error_region // Branch if x zero and y zero
}
;;

{ .mfb
      nop.m 999
(p11) fma.s.s0  f8 = atan2f_sgn_Y, atan2f_const_piby2,f0 // x zero, y not zero
      br.ret.sptk b0         // Final special case exit
}
;;


GLOBAL_IEEE754_END(atan2f)
libm_alias_float_other (__atan2, atan2)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
         mov            GR_Parameter_TAG      = 38
         fclass.m       p10,p11               = f9,0x5	// @zero | @pos
;;
(p10)    fmerge.s       f10                   = f8, f0
(p11)    fma.s.s0          f10                   = atan2f_sgn_Y, atan2f_const_pi,f0
;;

{ .mfi
        add   GR_Parameter_Y=-32,sp             // Parameter 2 value
        nop.f 999
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
}

{ .mfi
.fframe 64
        add sp=-64,sp                           // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                       // Save gp
}
;;

{ .mmi
        stfs [GR_Parameter_Y] = f9,16         // Store Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
}
;;


.body
{ .mib
        stfs [GR_Parameter_X] = f8            // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfs [GR_Parameter_Y] = f10       // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#   // Call error handling function
}
;;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;

{ .mmi
        ldfs  f8 = [GR_Parameter_RESULT]       // Get return result off stack
.restore sp
        add   sp = 64,sp                       // Restore stack pointer
        mov   b0 = GR_SAVE_B0                  // Restore return address
}
;;

{ .mib
        mov   gp = GR_SAVE_GP                  // Restore gp
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
        br.ret.sptk     b0                     // Return
}
;;

LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#
