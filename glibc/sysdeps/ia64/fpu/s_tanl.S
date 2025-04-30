.file "tancotl.s"


// Copyright (c) 2000 - 2004, Intel Corporation
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
//
// 02/02/00 (hand-optimized)
// 04/04/00 Unwind support added
// 12/28/00 Fixed false invalid flags
// 02/06/02 Improved speed
// 05/07/02 Changed interface to __libm_pi_by_2_reduce
// 05/30/02 Added cotl
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double table values
// 05/15/03 Reformatted data tables
// 10/26/04 Avoided using r14-31 as scratch so not clobbered by dynamic loader
//
//*********************************************************************
//
// Functions:   tanl(x) = tangent(x), for double-extended precision x values
//              cotl(x) = cotangent(x), for double-extended precision x values
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8 (Input and Return Value)
//                              f9-f15
//                              f32-f121
//
//    General Purpose Registers:
//      r32-r70
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions for tanl:
//
//    Denormal  fault raised on denormal inputs
//    Overflow exceptions do not occur
//    Underflow exceptions raised when appropriate for tan
//    (No specialized error handling for this routine)
//    Inexact raised when appropriate by algorithm
//
//    tanl(SNaN) = QNaN
//    tanl(QNaN) = QNaN
//    tanl(inf) = QNaN
//    tanl(+/-0) = +/-0
//
//*********************************************************************
//
// IEEE Special Conditions for cotl:
//
//    Denormal  fault raised on denormal inputs
//    Overflow exceptions occur at zero and near zero
//    Underflow exceptions do not occur
//    Inexact raised when appropriate by algorithm
//
//    cotl(SNaN) = QNaN
//    cotl(QNaN) = QNaN
//    cotl(inf) = QNaN
//    cotl(+/-0) = +/-Inf and error handling is called
//
//*********************************************************************
//
//    Below are mathematical and algorithmic descriptions for tanl.
//    For cotl we use next identity cot(x) = -tan(x + Pi/2).
//    So, to compute cot(x) we just need to increment N (N = N + 1)
//    and invert sign of the computed result.
//
//*********************************************************************
//
// Mathematical Description
//
// We consider the computation of FPTANL of Arg. Now, given
//
//      Arg = N pi/2  + alpha,          |alpha| <= pi/4,
//
// basic mathematical relationship shows that
//
//      tan( Arg ) =  tan( alpha )     if N is even;
//                 = -cot( alpha )      otherwise.
//
// The value of alpha is obtained by argument reduction and
// represented by two working precision numbers r and c where
//
//      alpha =  r  +  c     accurately.
//
// The reduction method is described in a previous write up.
// The argument reduction scheme identifies 4 cases. For Cases 2
// and 4, because |alpha| is small, tan(r+c) and -cot(r+c) can be
// computed very easily by 2 or 3 terms of the Taylor series
// expansion as follows:
//
// Case 2:
// -------
//
//      tan(r + c) = r + c + r^3/3          ...accurately
//     -cot(r + c) = -1/(r+c) + r/3          ...accurately
//
// Case 4:
// -------
//
//      tan(r + c) = r + c + r^3/3 + 2r^5/15     ...accurately
//     -cot(r + c) = -1/(r+c) + r/3 + r^3/45     ...accurately
//
//
// The only cases left are Cases 1 and 3 of the argument reduction
// procedure. These two cases will be merged since after the
// argument is reduced in either cases, we have the reduced argument
// represented as r + c and that the magnitude |r + c| is not small
// enough to allow the usage of a very short approximation.
//
// The greatest challenge of this task is that the second terms of
// the Taylor series for tan(r) and -cot(r)
//
//      r + r^3/3 + 2 r^5/15 + ...
//
// and
//
//      -1/r + r/3 + r^3/45 + ...
//
// are not very small when |r| is close to pi/4 and the rounding
// errors will be a concern if simple polynomial accumulation is
// used. When |r| < 2^(-2), however, the second terms will be small
// enough (5 bits or so of right shift) that a normal Horner
// recurrence suffices. Hence there are two cases that we consider
// in the accurate computation of tan(r) and cot(r), |r| <= pi/4.
//
// Case small_r: |r| < 2^(-2)
// --------------------------
//
// Since Arg = N pi/4 + r + c accurately, we have
//
//      tan(Arg) =  tan(r+c)            for N even,
//               = -cot(r+c)            otherwise.
//
// Here for this case, both tan(r) and -cot(r) can be approximated
// by simple polynomials:
//
//      tan(r) =    r + P1_1 r^3 + P1_2 r^5 + ... + P1_9 r^19
//     -cot(r) = -1/r + Q1_1 r   + Q1_2 r^3 + ... + Q1_7 r^13
//
// accurately. Since |r| is relatively small, tan(r+c) and
// -cot(r+c) can be accurately approximated by replacing r with
// r+c only in the first two terms of the corresponding polynomials.
//
// Note that P1_1 (and Q1_1 for that matter) approximates 1/3 to
// almost 64 sig. bits, thus
//
//      P1_1 (r+c)^3 =  P1_1 r^3 + c * r^2     accurately.
//
// Hence,
//
//      tan(r+c) =    r + P1_1 r^3 + P1_2 r^5 + ... + P1_9 r^19
//                     + c*(1 + r^2)
//
//        -cot(r+c) = -1/(r+c) + Q1_1 r   + Q1_2 r^3 + ... + Q1_7 r^13
//               + Q1_1*c
//
//
// Case normal_r: 2^(-2) <= |r| <= pi/4
// ------------------------------------
//
// This case is more likely than the previous one if one considers
// r to be uniformly distributed in [-pi/4 pi/4].
//
// The required calculation is either
//
//      tan(r + c)  =  tan(r)  +  correction,  or
//     -cot(r + c)  = -cot(r)  +  correction.
//
// Specifically,
//
//      tan(r + c) =  tan(r) + c tan'(r)  + O(c^2)
//                 =  tan(r) + c sec^2(r) + O(c^2)
//                 =  tan(r) + c SEC_sq     ...accurately
//                as long as SEC_sq approximates sec^2(r)
//                to, say, 5 bits or so.
//
// Similarly,
//
//     -cot(r + c) = -cot(r) - c cot'(r)  + O(c^2)
//                 = -cot(r) + c csc^2(r) + O(c^2)
//                 = -cot(r) + c CSC_sq     ...accurately
//                as long as CSC_sq approximates csc^2(r)
//                to, say, 5 bits or so.
//
// We therefore concentrate on accurately calculating tan(r) and
// cot(r) for a working-precision number r, |r| <= pi/4 to within
// 0.1% or so.
//
// We will employ a table-driven approach. Let
//
//      r = sgn_r * 2^k * 1.b_1 b_2 ... b_5 ... b_63
//        = sgn_r * ( B + x )
//
// where
//
//      B = 2^k * 1.b_1 b_2 ... b_5 1
//      x = |r| - B
//
// Now,
//                   tan(B)  +   tan(x)
//      tan( B + x ) =  ------------------------
//                   1 -  tan(B)*tan(x)
//
//               /                         \
//               |   tan(B)  +   tan(x)          |

//      = tan(B) +  | ------------------------ - tan(B) |
//               |     1 -  tan(B)*tan(x)          |
//               \                         /
//
//                 sec^2(B) * tan(x)
//      = tan(B) + ------------------------
//                 1 -  tan(B)*tan(x)
//
//                (1/[sin(B)*cos(B)]) * tan(x)
//      = tan(B) + --------------------------------
//                      cot(B)  -  tan(x)
//
//
// Clearly, the values of tan(B), cot(B) and 1/(sin(B)*cos(B)) are
// calculated beforehand and stored in a table. Since
//
//      |x| <= 2^k * 2^(-6)  <= 2^(-7)  (because k = -1, -2)
//
// a very short polynomial will be sufficient to approximate tan(x)
// accurately. The details involved in computing the last expression
// will be given in the next section on algorithm description.
//
//
// Now, we turn to the case where cot( B + x ) is needed.
//
//
//                   1 - tan(B)*tan(x)
//      cot( B + x ) =  ------------------------
//                   tan(B)  +  tan(x)
//
//               /                           \
//               |   1 - tan(B)*tan(x)              |

//      = cot(B) +  | ----------------------- - cot(B) |
//               |     tan(B)  +  tan(x)            |
//               \                           /
//
//               [tan(B) + cot(B)] * tan(x)
//      = cot(B) - ----------------------------
//                   tan(B)  +  tan(x)
//
//                (1/[sin(B)*cos(B)]) * tan(x)
//      = cot(B) - --------------------------------
//                      tan(B)  +  tan(x)
//
//
// Note that the values of tan(B), cot(B) and 1/(sin(B)*cos(B)) that
// are needed are the same set of values needed in the previous
// case.
//
// Finally, we can put all the ingredients together as follows:
//
//      Arg = N * pi/2 +  r + c          ...accurately
//
//      tan(Arg) =  tan(r) + correction    if N is even;
//               = -cot(r) + correction    otherwise.
//
// For Cases 2 and 4,
//
//     Case 2:
//     tan(Arg) =  tan(r + c) = r + c + r^3/3           N even
//              = -cot(r + c) = -1/(r+c) + r/3           N odd
//     Case 4:
//     tan(Arg) =  tan(r + c) = r + c + r^3/3 + 2r^5/15  N even
//              = -cot(r + c) = -1/(r+c) + r/3 + r^3/45  N odd
//
//
// For Cases 1 and 3,
//
//     Case small_r: |r| < 2^(-2)
//
//      tan(Arg) =  r + P1_1 r^3 + P1_2 r^5 + ... + P1_9 r^19
//                     + c*(1 + r^2)               N even
//
//               = -1/(r+c) + Q1_1 r   + Q1_2 r^3 + ... + Q1_7 r^13
//                     + Q1_1*c                    N odd
//
//     Case normal_r: 2^(-2) <= |r| <= pi/4
//
//      tan(Arg) =  tan(r) + c * sec^2(r)     N even
//               = -cot(r) + c * csc^2(r)     otherwise
//
//     For N even,
//
//      tan(Arg) = tan(r) + c*sec^2(r)
//               = tan( sgn_r * (B+x) ) + c * sec^2(|r|)
//               = sgn_r * ( tan(B+x)  + sgn_r*c*sec^2(|r|) )
//               = sgn_r * ( tan(B+x)  + sgn_r*c*sec^2(B) )
//
// since B approximates |r| to 2^(-6) in relative accuracy.
//
//                 /            (1/[sin(B)*cos(B)]) * tan(x)
//    tan(Arg) = sgn_r * | tan(B) + --------------------------------
//                 \                     cot(B)  -  tan(x)
//                                        \
//                       + CORR  |

//                                     /
// where
//
//    CORR = sgn_r*c*tan(B)*SC_inv(B);  SC_inv(B) = 1/(sin(B)*cos(B)).
//
// For N odd,
//
//      tan(Arg) = -cot(r) + c*csc^2(r)
//               = -cot( sgn_r * (B+x) ) + c * csc^2(|r|)
//               = sgn_r * ( -cot(B+x)  + sgn_r*c*csc^2(|r|) )
//               = sgn_r * ( -cot(B+x)  + sgn_r*c*csc^2(B) )
//
// since B approximates |r| to 2^(-6) in relative accuracy.
//
//                 /            (1/[sin(B)*cos(B)]) * tan(x)
//    tan(Arg) = sgn_r * | -cot(B) + --------------------------------
//                 \                     tan(B)  +  tan(x)
//                                        \
//                       + CORR  |

//                                     /
// where
//
//    CORR = sgn_r*c*cot(B)*SC_inv(B);  SC_inv(B) = 1/(sin(B)*cos(B)).
//
//
// The actual algorithm prescribes how all the mathematical formulas
// are calculated.
//
//
// 2. Algorithmic Description
// ==========================
//
// 2.1 Computation for Cases 2 and 4.
// ----------------------------------
//
// For Case 2, we use two-term polynomials.
//
//    For N even,
//
//    rsq := r * r
//    Poly := c + r * rsq * P1_1
//    Result := r + Poly          ...in user-defined rounding
//
//    For N odd,
//    S_hi  := -frcpa(r)               ...8 bits
//    S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...16 bits
//    S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...32 bits
//    S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...64 bits
//    S_lo  := S_hi*( (1 + S_hi*r) + S_hi*c )
//    ...S_hi + S_lo is -1/(r+c) to extra precision
//    S_lo  := S_lo + Q1_1*r
//
//    Result := S_hi + S_lo     ...in user-defined rounding
//
// For Case 4, we use three-term polynomials
//
//    For N even,
//
//    rsq := r * r
//    Poly := c + r * rsq * (P1_1 + rsq * P1_2)
//    Result := r + Poly          ...in user-defined rounding
//
//    For N odd,
//    S_hi  := -frcpa(r)               ...8 bits
//    S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...16 bits
//    S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...32 bits
//    S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...64 bits
//    S_lo  := S_hi*( (1 + S_hi*r) + S_hi*c )
//    ...S_hi + S_lo is -1/(r+c) to extra precision
//    rsq   := r * r
//    P      := Q1_1 + rsq*Q1_2
//    S_lo  := S_lo + r*P
//
//    Result := S_hi + S_lo     ...in user-defined rounding
//
//
// Note that the coefficients P1_1, P1_2, Q1_1, and Q1_2 are
// the same as those used in the small_r case of Cases 1 and 3
// below.
//
//
// 2.2 Computation for Cases 1 and 3.
// ----------------------------------
// This is further divided into the case of small_r,
// where |r| < 2^(-2), and the case of normal_r, where |r| lies between
// 2^(-2) and pi/4.
//
// Algorithm for the case of small_r
// ---------------------------------
//
// For N even,
//      rsq   := r * r
//      Poly1 := rsq*(P1_1 + rsq*(P1_2 + rsq*P1_3))
//      r_to_the_8    := rsq * rsq
//      r_to_the_8    := r_to_the_8 * r_to_the_8
//      Poly2 := P1_4 + rsq*(P1_5 + rsq*(P1_6 + ... rsq*P1_9))
//      CORR  := c * ( 1 + rsq )
//      Poly  := Poly1 + r_to_the_8*Poly2
//      Poly := r*Poly + CORR
//      Result := r + Poly     ...in user-defined rounding
//      ...note that Poly1 and r_to_the_8 can be computed in parallel
//      ...with Poly2 (Poly1 is intentionally set to be much
//      ...shorter than Poly2 so that r_to_the_8 and CORR can be hidden)
//
// For N odd,
//      S_hi  := -frcpa(r)               ...8 bits
//      S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...16 bits
//      S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...32 bits
//      S_hi  := S_hi + S_hi*(1 + S_hi*r)     ...64 bits
//      S_lo  := S_hi*( (1 + S_hi*r) + S_hi*c )
//      ...S_hi + S_lo is -1/(r+c) to extra precision
//      S_lo  := S_lo + Q1_1*c
//
//      ...S_hi and S_lo are computed in parallel with
//      ...the following
//      rsq := r*r
//      P   := Q1_1 + rsq*(Q1_2 + rsq*(Q1_3 + ... + rsq*Q1_7))
//
//      Poly :=  r*P + S_lo
//      Result :=  S_hi  +  Poly      ...in user-defined rounding
//
//
// Algorithm for the case of normal_r
// ----------------------------------
//
// Here, we first consider the computation of tan( r + c ). As
// presented in the previous section,
//
//      tan( r + c )  =  tan(r) + c * sec^2(r)
//                 =  sgn_r * [ tan(B+x) + CORR ]
//      CORR = sgn_r * c * tan(B) * 1/[sin(B)*cos(B)]
//
// because sec^2(r) = sec^(|r|), and B approximate |r| to 6.5 bits.
//
//      tan( r + c ) =
//           /           (1/[sin(B)*cos(B)]) * tan(x)
//      sgn_r * | tan(B) + --------------------------------  +
//           \                     cot(B)  -  tan(x)
//                                \
//                          CORR  |

//                                /
//
// The values of tan(B), cot(B) and 1/(sin(B)*cos(B)) are
// calculated beforehand and stored in a table. Specifically,
// the table values are
//
//      tan(B)             as  T_hi  +  T_lo;
//      cot(B)             as  C_hi  +  C_lo;
//      1/[sin(B)*cos(B)]  as  SC_inv
//
// T_hi, C_hi are in  double-precision  memory format;
// T_lo, C_lo are in  single-precision  memory format;
// SC_inv     is  in extended-precision memory format.
//
// The value of tan(x) will be approximated by a short polynomial of
// the form
//
//      tan(x)  as  x  +  x * P, where
//           P  =   x^2 * (P2_1 + x^2 * (P2_2 + x^2 * P2_3))
//
// Because |x| <= 2^(-7), cot(B) - x approximates cot(B) - tan(x)
// to a relative accuracy better than 2^(-20). Thus, a good
// initial guess of 1/( cot(B) - tan(x) ) to initiate the iterative
// division is:
//
//      1/(cot(B) - tan(x))      is approximately
//      1/(cot(B) -   x)         is
//      tan(B)/(1 - x*tan(B))    is approximately
//      T_hi / ( 1 - T_hi * x )  is approximately
//
//      T_hi * [ 1 + (Thi * x) + (T_hi * x)^2 ]
//
// The calculation of tan(r+c) therefore proceed as follows:
//
//      Tx     := T_hi * x
//      xsq     := x * x
//
//      V_hi     := T_hi*(1 + Tx*(1 + Tx))
//      P     := xsq * (P1_1 + xsq*(P1_2 + xsq*P1_3))
//      ...V_hi serves as an initial guess of 1/(cot(B) - tan(x))
//         ...good to about 20 bits of accuracy
//
//      tanx     := x + x*P
//      D     := C_hi - tanx
//      ...D is a double precision denominator: cot(B) - tan(x)
//
//      V_hi     := V_hi + V_hi*(1 - V_hi*D)
//      ....V_hi approximates 1/(cot(B)-tan(x)) to 40 bits
//
//      V_lo     := V_hi * ( [ (1 - V_hi*C_hi) + V_hi*tanx ]
//                           - V_hi*C_lo )   ...observe all order
//         ...V_hi + V_lo approximates 1/(cot(B) - tan(x))
//      ...to extra accuracy
//
//      ...               SC_inv(B) * (x + x*P)
//      ...   tan(B) +      ------------------------- + CORR
//         ...                cot(B) - (x + x*P)
//      ...
//      ... = tan(B) + SC_inv(B)*(x + x*P)*(V_hi + V_lo) + CORR
//      ...
//
//      Sx     := SC_inv * x
//      CORR     := sgn_r * c * SC_inv * T_hi
//
//      ...put the ingredients together to compute
//      ...               SC_inv(B) * (x + x*P)
//      ...   tan(B) +      ------------------------- + CORR
//         ...                cot(B) - (x + x*P)
//      ...
//      ... = tan(B) + SC_inv(B)*(x + x*P)*(V_hi + V_lo) + CORR
//      ...
//      ... = T_hi + T_lo + CORR +
//      ...    Sx * V_hi + Sx * V_lo + Sx * P *(V_hi + V_lo)
//
//      CORR := CORR + T_lo
//      tail := V_lo + P*(V_hi + V_lo)
//         tail := Sx * tail  +  CORR
//      tail := Sx * V_hi  +  tail
//         T_hi := sgn_r * T_hi
//
//         ...T_hi + sgn_r*tail  now approximate
//      ...sgn_r*(tan(B+x) + CORR) accurately
//
//      Result :=  T_hi + sgn_r*tail  ...in user-defined
//                           ...rounding control
//      ...It is crucial that independent paths be fully
//      ...exploited for performance's sake.
//
//
// Next, we consider the computation of -cot( r + c ). As
// presented in the previous section,
//
//        -cot( r + c )  =  -cot(r) + c * csc^2(r)
//                 =  sgn_r * [ -cot(B+x) + CORR ]
//      CORR = sgn_r * c * cot(B) * 1/[sin(B)*cos(B)]
//
// because csc^2(r) = csc^(|r|), and B approximate |r| to 6.5 bits.
//
//        -cot( r + c ) =
//           /             (1/[sin(B)*cos(B)]) * tan(x)
//      sgn_r * | -cot(B) + --------------------------------  +
//           \                     tan(B)  +  tan(x)
//                                \
//                          CORR  |

//                                /
//
// The values of tan(B), cot(B) and 1/(sin(B)*cos(B)) are
// calculated beforehand and stored in a table. Specifically,
// the table values are
//
//      tan(B)             as  T_hi  +  T_lo;
//      cot(B)             as  C_hi  +  C_lo;
//      1/[sin(B)*cos(B)]  as  SC_inv
//
// T_hi, C_hi are in  double-precision  memory format;
// T_lo, C_lo are in  single-precision  memory format;
// SC_inv     is  in extended-precision memory format.
//
// The value of tan(x) will be approximated by a short polynomial of
// the form
//
//      tan(x)  as  x  +  x * P, where
//           P  =   x^2 * (P2_1 + x^2 * (P2_2 + x^2 * P2_3))
//
// Because |x| <= 2^(-7), tan(B) + x approximates tan(B) + tan(x)
// to a relative accuracy better than 2^(-18). Thus, a good
// initial guess of 1/( tan(B) + tan(x) ) to initiate the iterative
// division is:
//
//      1/(tan(B) + tan(x))      is approximately
//      1/(tan(B) +   x)         is
//      cot(B)/(1 + x*cot(B))    is approximately
//      C_hi / ( 1 + C_hi * x )  is approximately
//
//      C_hi * [ 1 - (C_hi * x) + (C_hi * x)^2 ]
//
// The calculation of -cot(r+c) therefore proceed as follows:
//
//      Cx     := C_hi * x
//      xsq     := x * x
//
//      V_hi     := C_hi*(1 - Cx*(1 - Cx))
//      P     := xsq * (P1_1 + xsq*(P1_2 + xsq*P1_3))
//      ...V_hi serves as an initial guess of 1/(tan(B) + tan(x))
//         ...good to about 18 bits of accuracy
//
//      tanx     := x + x*P
//      D     := T_hi + tanx
//      ...D is a double precision denominator: tan(B) + tan(x)
//
//      V_hi     := V_hi + V_hi*(1 - V_hi*D)
//      ....V_hi approximates 1/(tan(B)+tan(x)) to 40 bits
//
//      V_lo     := V_hi * ( [ (1 - V_hi*T_hi) - V_hi*tanx ]
//                           - V_hi*T_lo )   ...observe all order
//         ...V_hi + V_lo approximates 1/(tan(B) + tan(x))
//      ...to extra accuracy
//
//      ...               SC_inv(B) * (x + x*P)
//      ...  -cot(B) +      ------------------------- + CORR
//         ...                tan(B) + (x + x*P)
//      ...
//      ... =-cot(B) + SC_inv(B)*(x + x*P)*(V_hi + V_lo) + CORR
//      ...
//
//      Sx     := SC_inv * x
//      CORR     := sgn_r * c * SC_inv * C_hi
//
//      ...put the ingredients together to compute
//      ...               SC_inv(B) * (x + x*P)
//      ...  -cot(B) +      ------------------------- + CORR
//         ...                tan(B) + (x + x*P)
//      ...
//      ... =-cot(B) + SC_inv(B)*(x + x*P)*(V_hi + V_lo) + CORR
//      ...
//      ... =-C_hi - C_lo + CORR +
//      ...    Sx * V_hi + Sx * V_lo + Sx * P *(V_hi + V_lo)
//
//      CORR := CORR - C_lo
//      tail := V_lo + P*(V_hi + V_lo)
//         tail := Sx * tail  +  CORR
//      tail := Sx * V_hi  +  tail
//         C_hi := -sgn_r * C_hi
//
//         ...C_hi + sgn_r*tail now approximates
//      ...sgn_r*(-cot(B+x) + CORR) accurately
//
//      Result :=  C_hi + sgn_r*tail   in user-defined rounding control
//      ...It is crucial that independent paths be fully
//      ...exploited for performance's sake.
//
// 3. Implementation Notes
// =======================
//
//   Table entries T_hi, T_lo; C_hi, C_lo; SC_inv
//
//   Recall that 2^(-2) <= |r| <= pi/4;
//
//      r = sgn_r * 2^k * 1.b_1 b_2 ... b_63
//
//   and
//
//        B = 2^k * 1.b_1 b_2 b_3 b_4 b_5 1
//
//   Thus, for k = -2, possible values of B are
//
//          B = 2^(-2) * ( 1 + index/32  +  1/64 ),
//      index ranges from 0 to 31
//
//   For k = -1, however, since |r| <= pi/4 = 0.78...
//   possible values of B are
//
//        B = 2^(-1) * ( 1 + index/32  +  1/64 )
//      index ranges from 0 to 19.
//
//

RODATA
.align 16

LOCAL_OBJECT_START(TANL_BASE_CONSTANTS)

tanl_table_1:
data8    0xA2F9836E4E44152A, 0x00003FFE // two_by_pi
data8    0xC84D32B0CE81B9F1, 0x00004016 // P_0
data8    0xC90FDAA22168C235, 0x00003FFF // P_1
data8    0xECE675D1FC8F8CBB, 0x0000BFBD // P_2
data8    0xB7ED8FBBACC19C60, 0x0000BF7C // P_3
LOCAL_OBJECT_END(TANL_BASE_CONSTANTS)

LOCAL_OBJECT_START(tanl_table_2)
data8    0xC90FDAA22168C234, 0x00003FFE // PI_BY_4
data8    0xA397E5046EC6B45A, 0x00003FE7 // Inv_P_0
data8    0x8D848E89DBD171A1, 0x0000BFBF // d_1
data8    0xD5394C3618A66F8E, 0x0000BF7C // d_2
data4    0x3E800000 // two**-2
data4    0xBE800000 // -two**-2
data4    0x00000000 // pad
data4    0x00000000 // pad
LOCAL_OBJECT_END(tanl_table_2)

LOCAL_OBJECT_START(tanl_table_p1)
data8    0xAAAAAAAAAAAAAABD, 0x00003FFD // P1_1
data8    0x8888888888882E6A, 0x00003FFC // P1_2
data8    0xDD0DD0DD0F0177B6, 0x00003FFA // P1_3
data8    0xB327A440646B8C6D, 0x00003FF9 // P1_4
data8    0x91371B251D5F7D20, 0x00003FF8 // P1_5
data8    0xEB69A5F161C67914, 0x00003FF6 // P1_6
data8    0xBEDD37BE019318D2, 0x00003FF5 // P1_7
data8    0x9979B1463C794015, 0x00003FF4 // P1_8
data8    0x8EBD21A38C6EB58A, 0x00003FF3 // P1_9
LOCAL_OBJECT_END(tanl_table_p1)

LOCAL_OBJECT_START(tanl_table_q1)
data8    0xAAAAAAAAAAAAAAB4, 0x00003FFD // Q1_1
data8    0xB60B60B60B5FC93E, 0x00003FF9 // Q1_2
data8    0x8AB355E00C9BBFBF, 0x00003FF6 // Q1_3
data8    0xDDEBBC89CBEE3D4C, 0x00003FF2 // Q1_4
data8    0xB3548A685F80BBB6, 0x00003FEF // Q1_5
data8    0x913625604CED5BF1, 0x00003FEC // Q1_6
data8    0xF189D95A8EE92A83, 0x00003FE8 // Q1_7
LOCAL_OBJECT_END(tanl_table_q1)

LOCAL_OBJECT_START(tanl_table_p2)
data8    0xAAAAAAAAAAAB362F, 0x00003FFD // P2_1
data8    0x88888886E97A6097, 0x00003FFC // P2_2
data8    0xDD108EE025E716A1, 0x00003FFA // P2_3
LOCAL_OBJECT_END(tanl_table_p2)

LOCAL_OBJECT_START(tanl_table_tm2)
//
//  Entries T_hi   double-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//  Entries T_lo  single-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//
data8 0x3FD09BC362400794
data4 0x23A05C32, 0x00000000
data8 0x3FD124A9DFFBC074
data4 0x240078B2, 0x00000000
data8 0x3FD1AE235BD4920F
data4 0x23826B8E, 0x00000000
data8 0x3FD2383515E2701D
data4 0x22D31154, 0x00000000
data8 0x3FD2C2E463739C2D
data4 0x2265C9E2, 0x00000000
data8 0x3FD34E36AFEEA48B
data4 0x245C05EB, 0x00000000
data8 0x3FD3DA317DBB35D1
data4 0x24749F2D, 0x00000000
data8 0x3FD466DA67321619
data4 0x2462CECE, 0x00000000
data8 0x3FD4F4371F94A4D5
data4 0x246D0DF1, 0x00000000
data8 0x3FD5824D740C3E6D
data4 0x240A85B5, 0x00000000
data8 0x3FD611234CB1E73D
data4 0x23F96E33, 0x00000000
data8 0x3FD6A0BEAD9EA64B
data4 0x247C5393, 0x00000000
data8 0x3FD73125B804FD01
data4 0x241F3B29, 0x00000000
data8 0x3FD7C25EAB53EE83
data4 0x2479989B, 0x00000000
data8 0x3FD8546FE6640EED
data4 0x23B343BC, 0x00000000
data8 0x3FD8E75FE8AF1892
data4 0x241454D1, 0x00000000
data8 0x3FD97B3553928BDA
data4 0x238613D9, 0x00000000
data8 0x3FDA0FF6EB9DE4DE
data4 0x22859FA7, 0x00000000
data8 0x3FDAA5AB99ECF92D
data4 0x237A6D06, 0x00000000
data8 0x3FDB3C5A6D8F1796
data4 0x23952F6C, 0x00000000
data8 0x3FDBD40A9CFB8BE4
data4 0x2280FC95, 0x00000000
data8 0x3FDC6CC387943100
data4 0x245D2EC0, 0x00000000
data8 0x3FDD068CB736C500
data4 0x23C4AD7D, 0x00000000
data8 0x3FDDA16DE1DDBC31
data4 0x23D076E6, 0x00000000
data8 0x3FDE3D6EEB515A93
data4 0x244809A6, 0x00000000
data8 0x3FDEDA97E6E9E5F1
data4 0x220856C8, 0x00000000
data8 0x3FDF78F11963CE69
data4 0x244BE993, 0x00000000
data8 0x3FE00C417D635BCE
data4 0x23D21799, 0x00000000
data8 0x3FE05CAB1C302CD3
data4 0x248A1B1D, 0x00000000
data8 0x3FE0ADB9DB6A1FA0
data4 0x23D53E33, 0x00000000
data8 0x3FE0FF724A20BA81
data4 0x24DB9ED5, 0x00000000
data8 0x3FE151D9153FA6F5
data4 0x24E9E451, 0x00000000
LOCAL_OBJECT_END(tanl_table_tm2)

LOCAL_OBJECT_START(tanl_table_tm1)
//
//  Entries T_hi   double-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//  Entries T_lo  single-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//
data8 0x3FE1CEC4BA1BE39E
data4 0x24B60F9E, 0x00000000
data8 0x3FE277E45ABD9B2D
data4 0x248C2474, 0x00000000
data8 0x3FE324180272B110
data4 0x247B8311, 0x00000000
data8 0x3FE3D38B890E2DF0
data4 0x24C55751, 0x00000000
data8 0x3FE4866D46236871
data4 0x24E5BC34, 0x00000000
data8 0x3FE53CEE45E044B0
data4 0x24001BA4, 0x00000000
data8 0x3FE5F74282EC06E4
data4 0x24B973DC, 0x00000000
data8 0x3FE6B5A125DF43F9
data4 0x24895440, 0x00000000
data8 0x3FE77844CAFD348C
data4 0x240021CA, 0x00000000
data8 0x3FE83F6BCEED6B92
data4 0x24C45372, 0x00000000
data8 0x3FE90B58A34F3665
data4 0x240DAD33, 0x00000000
data8 0x3FE9DC522C1E56B4
data4 0x24F846CE, 0x00000000
data8 0x3FEAB2A427041578
data4 0x2323FB6E, 0x00000000
data8 0x3FEB8E9F9DD8C373
data4 0x24B3090B, 0x00000000
data8 0x3FEC709B65C9AA7B
data4 0x2449F611, 0x00000000
data8 0x3FED58F4ACCF8435
data4 0x23616A7E, 0x00000000
data8 0x3FEE480F97635082
data4 0x24C2FEAE, 0x00000000
data8 0x3FEF3E57F0ACC544
data4 0x242CE964, 0x00000000
data8 0x3FF01E20F7E06E4B
data4 0x2480D3EE, 0x00000000
data8 0x3FF0A1258A798A69
data4 0x24DB8967, 0x00000000
LOCAL_OBJECT_END(tanl_table_tm1)

LOCAL_OBJECT_START(tanl_table_cm2)
//
//  Entries C_hi   double-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//  Entries C_lo  single-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//
data8 0x400ED3E2E63EFBD0
data4 0x259D94D4, 0x00000000
data8 0x400DDDB4C515DAB5
data4 0x245F0537, 0x00000000
data8 0x400CF57ABE19A79F
data4 0x25D4EA9F, 0x00000000
data8 0x400C1A06D15298ED
data4 0x24AE40A0, 0x00000000
data8 0x400B4A4C164B2708
data4 0x25A5AAB6, 0x00000000
data8 0x400A855A5285B068
data4 0x25524F18, 0x00000000
data8 0x4009CA5A3FFA549F
data4 0x24C999C0, 0x00000000
data8 0x4009188A646AF623
data4 0x254FD801, 0x00000000
data8 0x40086F3C6084D0E7
data4 0x2560F5FD, 0x00000000
data8 0x4007CDD2A29A76EE
data4 0x255B9D19, 0x00000000
data8 0x400733BE6C8ECA95
data4 0x25CB021B, 0x00000000
data8 0x4006A07E1F8DDC52
data4 0x24AB4722, 0x00000000
data8 0x4006139BC298AD58
data4 0x252764E2, 0x00000000
data8 0x40058CABBAD7164B
data4 0x24DAF5DB, 0x00000000
data8 0x40050B4BAE31A5D3
data4 0x25EA20F4, 0x00000000
data8 0x40048F2189F85A8A
data4 0x2583A3E8, 0x00000000
data8 0x400417DAA862380D
data4 0x25DCC4CC, 0x00000000
data8 0x4003A52B1088FCFE
data4 0x2430A492, 0x00000000
data8 0x400336CCCD3527D5
data4 0x255F77CF, 0x00000000
data8 0x4002CC7F5760766D
data4 0x25DA0BDA, 0x00000000
data8 0x4002660711CE02E3
data4 0x256FF4A2, 0x00000000
data8 0x4002032CD37BBE04
data4 0x25208AED, 0x00000000
data8 0x4001A3BD7F050775
data4 0x24B72DD6, 0x00000000
data8 0x40014789A554848A
data4 0x24AB4DAA, 0x00000000
data8 0x4000EE65323E81B7
data4 0x2584C440, 0x00000000
data8 0x4000982721CF1293
data4 0x25C9428D, 0x00000000
data8 0x400044A93D415EEB
data4 0x25DC8482, 0x00000000
data8 0x3FFFE78FBD72C577
data4 0x257F5070, 0x00000000
data8 0x3FFF4AC375EFD28E
data4 0x23EBBF7A, 0x00000000
data8 0x3FFEB2AF60B52DDE
data4 0x22EECA07, 0x00000000
data8 0x3FFE1F1935204180
data4 0x24191079, 0x00000000
data8 0x3FFD8FCA54F7E60A
data4 0x248D3058, 0x00000000
LOCAL_OBJECT_END(tanl_table_cm2)

LOCAL_OBJECT_START(tanl_table_cm1)
//
//  Entries C_hi   double-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//  Entries C_lo  single-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//
data8 0x3FFCC06A79F6FADE
data4 0x239C7886, 0x00000000
data8 0x3FFBB91F891662A6
data4 0x250BD191, 0x00000000
data8 0x3FFABFB6529F155D
data4 0x256CC3E6, 0x00000000
data8 0x3FF9D3002E964AE9
data4 0x250843E3, 0x00000000
data8 0x3FF8F1EF89DCB383
data4 0x2277C87E, 0x00000000
data8 0x3FF81B937C87DBD6
data4 0x256DA6CF, 0x00000000
data8 0x3FF74F141042EDE4
data4 0x2573D28A, 0x00000000
data8 0x3FF68BAF1784B360
data4 0x242E489A, 0x00000000
data8 0x3FF5D0B57C923C4C
data4 0x2532D940, 0x00000000
data8 0x3FF51D88F418EF20
data4 0x253C7DD6, 0x00000000
data8 0x3FF4719A02F88DAE
data4 0x23DB59BF, 0x00000000
data8 0x3FF3CC6649DA0788
data4 0x252B4756, 0x00000000
data8 0x3FF32D770B980DB8
data4 0x23FE585F, 0x00000000
data8 0x3FF2945FE56C987A
data4 0x25378A63, 0x00000000
data8 0x3FF200BDB16523F6
data4 0x247BB2E0, 0x00000000
data8 0x3FF172358CE27778
data4 0x24446538, 0x00000000
data8 0x3FF0E873FDEFE692
data4 0x2514638F, 0x00000000
data8 0x3FF0632C33154062
data4 0x24A7FC27, 0x00000000
data8 0x3FEFC42EB3EF115F
data4 0x248FD0FE, 0x00000000
data8 0x3FEEC9E8135D26F6
data4 0x2385C719, 0x00000000
LOCAL_OBJECT_END(tanl_table_cm1)

LOCAL_OBJECT_START(tanl_table_scim2)
//
//  Entries SC_inv in Swapped IEEE format (extended)
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//
data8    0x839D6D4A1BF30C9E, 0x00004001
data8    0x80092804554B0EB0, 0x00004001
data8    0xF959F94CA1CF0DE9, 0x00004000
data8    0xF3086BA077378677, 0x00004000
data8    0xED154515CCD4723C, 0x00004000
data8    0xE77909441C27CF25, 0x00004000
data8    0xE22D037D8DDACB88, 0x00004000
data8    0xDD2B2D8A89C73522, 0x00004000
data8    0xD86E1A23BB2C1171, 0x00004000
data8    0xD3F0E288DFF5E0F9, 0x00004000
data8    0xCFAF16B1283BEBD5, 0x00004000
data8    0xCBA4AFAA0D88DD53, 0x00004000
data8    0xC7CE03CCCA67C43D, 0x00004000
data8    0xC427BC820CA0DDB0, 0x00004000
data8    0xC0AECD57F13D8CAB, 0x00004000
data8    0xBD606C3871ECE6B1, 0x00004000
data8    0xBA3A0A96A44C4929, 0x00004000
data8    0xB7394F6FE5CCCEC1, 0x00004000
data8    0xB45C12039637D8BC, 0x00004000
data8    0xB1A0552892CB051B, 0x00004000
data8    0xAF04432B6BA2FFD0, 0x00004000
data8    0xAC862A237221235F, 0x00004000
data8    0xAA2478AF5F00A9D1, 0x00004000
data8    0xA7DDBB0C81E082BF, 0x00004000
data8    0xA5B0987D45684FEE, 0x00004000
data8    0xA39BD0F5627A8F53, 0x00004000
data8    0xA19E3B036EC5C8B0, 0x00004000
data8    0x9FB6C1F091CD7C66, 0x00004000
data8    0x9DE464101FA3DF8A, 0x00004000
data8    0x9C263139A8F6B888, 0x00004000
data8    0x9A7B4968C27B0450, 0x00004000
data8    0x98E2DB7E5EE614EE, 0x00004000
LOCAL_OBJECT_END(tanl_table_scim2)

LOCAL_OBJECT_START(tanl_table_scim1)
//
//  Entries SC_inv in Swapped IEEE format (extended)
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//
data8    0x969F335C13B2B5BA, 0x00004000
data8    0x93D446D9D4C0F548, 0x00004000
data8    0x9147094F61B798AF, 0x00004000
data8    0x8EF317CC758787AC, 0x00004000
data8    0x8CD498B3B99EEFDB, 0x00004000
data8    0x8AE82A7DDFF8BC37, 0x00004000
data8    0x892AD546E3C55D42, 0x00004000
data8    0x8799FEA9D15573C1, 0x00004000
data8    0x86335F88435A4B4C, 0x00004000
data8    0x84F4FB6E3E93A87B, 0x00004000
data8    0x83DD195280A382FB, 0x00004000
data8    0x82EA3D7FA4CB8C9E, 0x00004000
data8    0x821B247C6861D0A8, 0x00004000
data8    0x816EBED163E8D244, 0x00004000
data8    0x80E42D9127E4CFC6, 0x00004000
data8    0x807ABF8D28E64AFD, 0x00004000
data8    0x8031EF26863B4FD8, 0x00004000
data8    0x800960ADAE8C11FD, 0x00004000
data8    0x8000E1475FDBEC21, 0x00004000
data8    0x80186650A07791FA, 0x00004000
LOCAL_OBJECT_END(tanl_table_scim1)

Arg                 = f8
Save_Norm_Arg       = f8        // For input to reduction routine
Result              = f8
r                   = f8        // For output from reduction routine
c                   = f9        // For output from reduction routine
U_2                 = f10
rsq                 = f11
C_hi                = f12
C_lo                = f13
T_hi                = f14
T_lo                = f15

d_1                 = f33
N_0                 = f34
tail                = f35
tanx                = f36
Cx                  = f37
Sx                  = f38
sgn_r               = f39
CORR                = f40
P                   = f41
D                   = f42
ArgPrime            = f43
P_0                 = f44

P2_1                = f45
P2_2                = f46
P2_3                = f47

P1_1                = f45
P1_2                = f46
P1_3                = f47

P1_4                = f48
P1_5                = f49
P1_6                = f50
P1_7                = f51
P1_8                = f52
P1_9                = f53

x                   = f56
xsq                 = f57
Tx                  = f58
Tx1                 = f59
Set                 = f60
poly1               = f61
poly2               = f62
Poly                = f63
Poly1               = f64
Poly2               = f65
r_to_the_8          = f66
B                   = f67
SC_inv              = f68
Pos_r               = f69
N_0_fix             = f70
d_2                 = f71
PI_BY_4             = f72
TWO_TO_NEG14        = f74
TWO_TO_NEG33        = f75
NEGTWO_TO_NEG14     = f76
NEGTWO_TO_NEG33     = f77
two_by_PI           = f78
N                   = f79
N_fix               = f80
P_1                 = f81
P_2                 = f82
P_3                 = f83
s_val               = f84
w                   = f85
B_mask1             = f86
B_mask2             = f87
w2                  = f88
A                   = f89
a                   = f90
t                   = f91
U_1                 = f92
NEGTWO_TO_NEG2      = f93
TWO_TO_NEG2         = f94
Q1_1                = f95
Q1_2                = f96
Q1_3                = f97
Q1_4                = f98
Q1_5                = f99
Q1_6                = f100
Q1_7                = f101
Q1_8                = f102
S_hi                = f103
S_lo                = f104
V_hi                = f105
V_lo                = f106
U_hi                = f107
U_lo                = f108
U_hiabs             = f109
V_hiabs             = f110
V                   = f111
Inv_P_0             = f112

FR_inv_pi_2to63     = f113
FR_rshf_2to64       = f114
FR_2tom64           = f115
FR_rshf             = f116
Norm_Arg            = f117
Abs_Arg             = f118
TWO_TO_NEG65        = f119
fp_tmp              = f120
mOne                = f121

GR_SAVE_B0     = r33
GR_SAVE_GP     = r34
GR_SAVE_PFS    = r35
table_base     = r36
table_ptr1     = r37
table_ptr2     = r38
table_ptr3     = r39
lookup         = r40
N_fix_gr       = r41
GR_exp_2tom2   = r42
GR_exp_2tom65  = r43
exp_r          = r44
sig_r          = r45
bmask1         = r46
table_offset   = r47
bmask2         = r48
gr_tmp         = r49
cot_flag       = r50

GR_sig_inv_pi  = r51
GR_rshf_2to64  = r52
GR_exp_2tom64  = r53
GR_rshf        = r54
GR_exp_2_to_63 = r55
GR_exp_2_to_24 = r56
GR_signexp_x   = r57
GR_exp_x       = r58
GR_exp_mask    = r59
GR_exp_2tom14  = r60
GR_exp_m2tom14 = r61
GR_exp_2tom33  = r62
GR_exp_m2tom33 = r63

GR_SAVE_B0                  = r64
GR_SAVE_PFS                 = r65
GR_SAVE_GP                  = r66

GR_Parameter_X              = r67
GR_Parameter_Y              = r68
GR_Parameter_RESULT         = r69
GR_Parameter_Tag            = r70


.section .text
.global __libm_tanl#
.global __libm_cotl#

.proc __libm_cotl#
__libm_cotl:
.endp __libm_cotl#
LOCAL_LIBM_ENTRY(cotl)

{ .mlx
      alloc r32 = ar.pfs, 0,35,4,0
      movl GR_sig_inv_pi = 0xa2f9836e4e44152a // significand of 1/pi
}
{ .mlx
      mov GR_exp_mask = 0x1ffff            // Exponent mask
      movl GR_rshf_2to64 = 0x47e8000000000000 // 1.1000 2^(63+64)
}
;;

//     Check for NatVals, Infs , NaNs, and Zeros
{ .mfi
      getf.exp GR_signexp_x = Arg          // Get sign and exponent of x
      fclass.m  p6,p0 = Arg, 0x1E7         // Test for natval, nan, inf, zero
      mov cot_flag = 0x1
}
{ .mfb
      addl table_base = @ltoff(TANL_BASE_CONSTANTS), gp // Pointer to table ptr
      fnorm.s1 Norm_Arg = Arg              // Normalize x
      br.cond.sptk COMMON_PATH
};;

LOCAL_LIBM_END(cotl)


.proc __libm_tanl#
__libm_tanl:
.endp __libm_tanl#
GLOBAL_IEEE754_ENTRY(tanl)

{ .mlx
      alloc r32 = ar.pfs, 0,35,4,0
      movl GR_sig_inv_pi = 0xa2f9836e4e44152a // significand of 1/pi
}
{ .mlx
      mov GR_exp_mask = 0x1ffff            // Exponent mask
      movl GR_rshf_2to64 = 0x47e8000000000000 // 1.1000 2^(63+64)
}
;;

//     Check for NatVals, Infs , NaNs, and Zeros
{ .mfi
      getf.exp GR_signexp_x = Arg          // Get sign and exponent of x
      fclass.m  p6,p0 = Arg, 0x1E7         // Test for natval, nan, inf, zero
      mov cot_flag = 0x0
}
{ .mfi
      addl table_base = @ltoff(TANL_BASE_CONSTANTS), gp // Pointer to table ptr
      fnorm.s1 Norm_Arg = Arg              // Normalize x
      nop.i 0
};;

// Common path for both tanl and cotl
COMMON_PATH:
{ .mfi
      setf.sig FR_inv_pi_2to63 = GR_sig_inv_pi // Form 1/pi * 2^63
      fclass.m p9, p0 = Arg, 0x0b          // Test x denormal
      mov GR_exp_2tom64 = 0xffff - 64      // Scaling constant to compute N
}
{ .mlx
      setf.d FR_rshf_2to64 = GR_rshf_2to64 // Form const 1.1000 * 2^(63+64)
      movl GR_rshf = 0x43e8000000000000    // Form const 1.1000 * 2^63
}
;;

// Check for everything - if false, then must be pseudo-zero or pseudo-nan.
// Branch out to deal with special values.
{ .mfi
      addl gr_tmp = -1,r0
      fclass.nm  p7,p0 = Arg, 0x1FF        // Test x unsupported
      mov GR_exp_2_to_63 = 0xffff + 63     // Exponent of 2^63
}
{ .mfb
      ld8 table_base = [table_base]        // Get pointer to constant table
      fms.s1 mOne = f0, f0, f1
(p6)  br.cond.spnt TANL_SPECIAL            // Branch if x natval, nan, inf, zero
}
;;

{ .mmb
      setf.sig fp_tmp = gr_tmp   // Make a constant so fmpy produces inexact
      mov GR_exp_2_to_24 = 0xffff + 24     // Exponent of 2^24
(p9)  br.cond.spnt TANL_DENORMAL           // Branch if x denormal
}
;;

TANL_COMMON:
// Return to here if x denormal
//
// Do fcmp to generate Denormal exception
//  - can't do FNORM (will generate Underflow when U is unmasked!)
// Branch out to deal with unsupporteds values.
{ .mfi
      setf.exp FR_2tom64 = GR_exp_2tom64 // Form 2^-64 for scaling N_float
      fcmp.eq.s0 p0, p6 = Arg, f1        // Dummy to flag denormals
      add table_ptr1 = 0, table_base     // Point to tanl_table_1
}
{ .mib
      setf.d FR_rshf = GR_rshf           // Form right shift const 1.1000 * 2^63
      add table_ptr2 = 80, table_base    // Point to tanl_table_2
(p7)  br.cond.spnt TANL_UNSUPPORTED      // Branch if x unsupported type
}
;;

{ .mfi
      and GR_exp_x = GR_exp_mask, GR_signexp_x // Get exponent of x
      fmpy.s1 Save_Norm_Arg = Norm_Arg, f1     // Save x if large arg reduction
      dep.z bmask1 = 0x7c, 56, 8               // Form mask to get 5 msb of r
                                               // bmask1 = 0x7c00000000000000
}
;;

//
//     Decide about the paths to take:
//     Set PR_6 if |Arg| >= 2**63
//     Set PR_9 if |Arg| < 2**24 - CASE 1 OR 2
//     OTHERWISE Set PR_8 - CASE 3 OR 4
//
//     Branch out if the magnitude of the input argument is >= 2^63
//     - do this branch before the next.
{ .mfi
      ldfe two_by_PI = [table_ptr1],16        // Load 2/pi
      nop.f 999
      dep.z bmask2 = 0x41, 57, 7              // Form mask to OR to produce B
                                              // bmask2 = 0x8200000000000000
}
{ .mib
      ldfe PI_BY_4 = [table_ptr2],16          // Load pi/4
      cmp.ge p6,p0 = GR_exp_x, GR_exp_2_to_63 // Is |x| >= 2^63
(p6)  br.cond.spnt TANL_ARG_TOO_LARGE         // Branch if |x| >= 2^63
}
;;

{ .mmi
      ldfe P_0 = [table_ptr1],16              // Load P_0
      ldfe Inv_P_0 = [table_ptr2],16          // Load Inv_P_0
      nop.i 999
}
;;

{ .mfi
      ldfe P_1 = [table_ptr1],16              // Load P_1
      fmerge.s Abs_Arg = f0, Norm_Arg         // Get |x|
      mov GR_exp_m2tom33 = 0x2ffff - 33       // Form signexp of -2^-33
}
{ .mfi
      ldfe d_1 = [table_ptr2],16              // Load d_1 for 2^24 <= |x| < 2^63
      nop.f 999
      mov GR_exp_2tom33 = 0xffff - 33         // Form signexp of 2^-33
}
;;

{ .mmi
      ldfe P_2 = [table_ptr1],16              // Load P_2
      ldfe d_2 = [table_ptr2],16              // Load d_2 for 2^24 <= |x| < 2^63
      cmp.ge p8,p0 = GR_exp_x, GR_exp_2_to_24 // Is |x| >= 2^24
}
;;

// Use special scaling to right shift so N=Arg * 2/pi is in rightmost bits
// Branch to Cases 3 or 4 if Arg <= -2**24 or Arg >= 2**24
{ .mfb
      ldfe   P_3 = [table_ptr1],16            // Load P_3
      fma.s1      N_fix = Norm_Arg, FR_inv_pi_2to63, FR_rshf_2to64
(p8)  br.cond.spnt TANL_LARGER_ARG            // Branch if 2^24 <= |x| < 2^63
}
;;

// Here if 0 < |x| < 2^24
//     ARGUMENT REDUCTION CODE - CASE 1 and 2
//
{ .mmf
      setf.exp TWO_TO_NEG33 = GR_exp_2tom33      // Form 2^-33
      setf.exp NEGTWO_TO_NEG33 = GR_exp_m2tom33  // Form -2^-33
      fmerge.s r = Norm_Arg,Norm_Arg          // Assume r=x, ok if |x| < pi/4
}
;;

//
// If |Arg| < pi/4,  set PR_8, else  pi/4 <=|Arg| < 2^24 - set PR_9.
//
//     Case 2: Convert integer N_fix back to normalized floating-point value.
{ .mfi
      getf.sig sig_r = Norm_Arg               // Get sig_r if 1/4 <= |x| < pi/4
      fcmp.lt.s1 p8,p9= Abs_Arg,PI_BY_4       // Test |x| < pi/4
      mov GR_exp_2tom2 = 0xffff - 2           // Form signexp of 2^-2
}
{ .mfi
      ldfps TWO_TO_NEG2, NEGTWO_TO_NEG2 = [table_ptr2] // Load 2^-2, -2^-2
      fms.s1 N = N_fix, FR_2tom64, FR_rshf    // Use scaling to get N floated
      mov N_fix_gr = r0                       // Assume N=0, ok if |x| < pi/4
}
;;

//
//     Case 1: Is |r| < 2**(-2).
//     Arg is the same as r in this case.
//     r = Arg
//     c = 0
//
//     Case 2: Place integer part of N in GP register.
{ .mfi
(p9)  getf.sig N_fix_gr = N_fix
      fmerge.s c = f0, f0                     // Assume c=0, ok if |x| < pi/4
      cmp.lt p10, p0 = GR_exp_x, GR_exp_2tom2 // Test if |x| < 1/4
}
;;

{ .mfi
      setf.sig B_mask1 = bmask1               // Form mask to get 5 msb of r
      nop.f 999
      mov exp_r = GR_exp_x                    // Get exp_r if 1/4 <= |x| < pi/4
}
{ .mbb
      setf.sig B_mask2 = bmask2               // Form mask to form B from r
(p10) br.cond.spnt TANL_SMALL_R               // Branch if 0 < |x| < 1/4
(p8)  br.cond.spnt TANL_NORMAL_R              // Branch if 1/4 <= |x| < pi/4
}
;;

// Here if pi/4 <= |x| < 2^24
//
//     Case 1: PR_3 is only affected  when PR_1 is set.
//
//
//     Case 2: w = N * P_2
//     Case 2: s_val = -N * P_1  + Arg
//

{ .mfi
      nop.m 999
      fnma.s1 s_val = N, P_1, Norm_Arg
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 w = N, P_2                     // w = N * P_2 for |s| >= 2^-33
      nop.i 999
}
;;

//     Case 2_reduce: w = N * P_3 (change sign)
{ .mfi
      nop.m 999
      fmpy.s1 w2 = N, P_3                    // w = N * P_3 for |s| < 2^-33
      nop.i 999
}
;;

//     Case 1_reduce: r = s + w (change sign)
{ .mfi
      nop.m 999
      fsub.s1 r = s_val, w                   // r = s_val - w for |s| >= 2^-33
      nop.i 999
}
;;

//     Case 2_reduce: U_1 = N * P_2 + w
{ .mfi
      nop.m 999
      fma.s1  U_1 = N, P_2, w2              // U_1 = N * P_2 + w for |s| < 2^-33
      nop.i 999
}
;;

//
//     Decide between case_1 and case_2 reduce:
//     Case 1_reduce:  |s| >= 2**(-33)
//     Case 2_reduce:  |s| < 2**(-33)
//
{ .mfi
      nop.m 999
      fcmp.lt.s1 p9, p8 = s_val, TWO_TO_NEG33
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fcmp.gt.s1 p9, p8 = s_val, NEGTWO_TO_NEG33
      nop.i 999
}
;;

//     Case 1_reduce: c = s - r
{ .mfi
      nop.m 999
      fsub.s1 c = s_val, r                     // c = s_val - r for |s| >= 2^-33
      nop.i 999
}
;;

//     Case 2_reduce: r is complete here - continue to calculate c .
//     r = s - U_1
{ .mfi
      nop.m 999
(p9)  fsub.s1 r = s_val, U_1
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fms.s1 U_2 = N, P_2, U_1
      nop.i 999
}
;;

//
//     Case 1_reduce: Is |r| < 2**(-2), if so set PR_10
//     else set PR_13.
//

{ .mfi
      nop.m 999
      fand B = B_mask1, r
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fcmp.lt.unc.s1 p10, p13 = r, TWO_TO_NEG2
      nop.i 999
}
;;

{ .mfi
(p8)  getf.sig sig_r = r               // Get signif of r if |s| >= 2^-33
      nop.f 999
      nop.i 999
}
;;

{ .mfi
(p8)  getf.exp exp_r = r               // Extract signexp of r if |s| >= 2^-33
(p10) fcmp.gt.s1 p10, p13 = r, NEGTWO_TO_NEG2
      nop.i 999
}
;;

//     Case 1_reduce: c is complete here.
//     Case 1: Branch to SMALL_R or NORMAL_R.
//     c = c + w (w has not been negated.)
{ .mfi
      nop.m 999
(p8)  fsub.s1 c = c, w                         // c = c - w for |s| >= 2^-33
      nop.i 999
}
{ .mbb
      nop.m 999
(p10) br.cond.spnt TANL_SMALL_R     // Branch if pi/4 < |x| < 2^24 and |r|<1/4
(p13) br.cond.sptk TANL_NORMAL_R_A  // Branch if pi/4 < |x| < 2^24 and |r|>=1/4
}
;;


// Here if pi/4 < |x| < 2^24 and |s| < 2^-33
//
//     Is i_1 = lsb of N_fix_gr even or odd?
//     if i_1 == 0, set p11, else set p12.
//
{ .mfi
      nop.m 999
      fsub.s1 s_val = s_val, r
      add N_fix_gr = N_fix_gr, cot_flag // N = N + 1 (for cotl)
}
{ .mfi
      nop.m 999
//
//     Case 2_reduce:
//     U_2 = N * P_2 - U_1
//     Not needed until later.
//
      fadd.s1 U_2 = U_2, w2
//
//     Case 2_reduce:
//     s = s - r
//     U_2 = U_2 + w
//
      nop.i 999
}
;;

//
//     Case 2_reduce:
//     c = c - U_2
//     c is complete here
//     Argument reduction ends here.
//
{ .mfi
      nop.m 999
      fmpy.s1 rsq = r, r
      tbit.z p11, p12 = N_fix_gr, 0 ;;    // Set p11 if N even, p12 if odd
}

{ .mfi
      nop.m 999
(p12) frcpa.s1 S_hi,p0 = f1, r
      nop.i 999
}
{ .mfi
      nop.m 999
      fsub.s1 c = s_val, U_1
      nop.i 999
}
;;

{ .mmi
      add table_ptr1 = 160, table_base ;;  // Point to tanl_table_p1
      ldfe P1_1 = [table_ptr1],144
      nop.i 999 ;;
}
//
//     Load P1_1 and point to Q1_1 .
//
{ .mfi
      ldfe Q1_1 = [table_ptr1]
//
//     N even: rsq = r * Z
//     N odd:  S_hi = frcpa(r)
//
(p12) fmerge.ns S_hi = S_hi, S_hi
      nop.i 999
}
{ .mfi
      nop.m 999
//
//     Case 2_reduce:
//     c = s - U_1
//
(p9)  fsub.s1 c = c, U_2
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1  poly1 = S_hi, r, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N odd:  Change sign of S_hi
//
(p11) fmpy.s1 rsq = rsq, P1_1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N even: rsq = rsq * P1_1
//     N odd:  poly1 =  1.0 +  S_hi * r    16 bits partial  account for necessary
//
(p11) fma.s1 Poly = r, rsq, c
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N even: Poly = c  + r * rsq
//     N odd:  S_hi  = S_hi + S_hi*poly1  16 bits account for necessary
//
(p12) fma.s1 poly1 = S_hi, r, f1
(p11) tbit.z.unc p14, p15 = cot_flag, 0 ;; // p14=1 for tanl; p15=1 for cotl
}
{ .mfi
      nop.m 999
//
//     N even: Result = Poly + r
//     N odd:  poly1  = 1.0 + S_hi * r        32 bits partial
//
(p14) fadd.s0 Result = r, Poly             // for tanl
      nop.i 999
}
{ .mfi
      nop.m 999
(p15) fms.s0 Result = r, mOne, Poly        // for cotl
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p12) fma.s1  S_hi = S_hi, poly1, S_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N even: Result1 = Result + r
//     N odd:   S_hi  = S_hi * poly1 + S_hi   32 bits
//
(p12) fma.s1 poly1 = S_hi, r, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N odd:  poly1  =  S_hi * r + 1.0       64 bits partial
//
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N odd:  poly1  =  S_hi * poly + 1.0    64 bits
//
(p12) fma.s1 poly1 = S_hi, r, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N odd:  poly1  =  S_hi * r + 1.0
//
(p12) fma.s1 poly1 = S_hi, c, poly1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N odd:  poly1  =  S_hi * c + poly1
//
(p12) fmpy.s1 S_lo = S_hi, poly1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N odd:  S_lo  =  S_hi *  poly1
//
(p12) fma.s1 S_lo = Q1_1, r, S_lo
(p12) tbit.z.unc p14, p15 = cot_flag, 0 // p14=1 for tanl; p15=1 for cotl
}
{ .mfi
      nop.m 999
//
//     N odd:  Result =  S_hi + S_lo
//
      fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//     N odd:  S_lo  =  S_lo + Q1_1 * r
//
(p14) fadd.s0 Result = S_hi, S_lo          // for tanl
      nop.i 999
}
{ .mfb
      nop.m 999
(p15) fms.s0 Result = S_hi, mOne, S_lo     // for cotl
      br.ret.sptk b0 ;;          // Exit for pi/4 <= |x| < 2^24 and |s| < 2^-33
}


TANL_LARGER_ARG:
// Here if 2^24 <= |x| < 2^63
//
// ARGUMENT REDUCTION CODE - CASE 3 and 4
//

{ .mmf
      mov GR_exp_2tom14 = 0xffff - 14          // Form signexp of 2^-14
      mov GR_exp_m2tom14 = 0x2ffff - 14        // Form signexp of -2^-14
      fmpy.s1 N_0 = Norm_Arg, Inv_P_0
}
;;

{ .mmi
      setf.exp TWO_TO_NEG14 = GR_exp_2tom14    // Form 2^-14
      setf.exp NEGTWO_TO_NEG14 = GR_exp_m2tom14// Form -2^-14
      nop.i 999
}
;;


//
//    Adjust table_ptr1 to beginning of table.
//    N_0 = Arg * Inv_P_0
//
{ .mmi
      add table_ptr2 = 144, table_base ;;     // Point to 2^-2
      ldfps TWO_TO_NEG2, NEGTWO_TO_NEG2 = [table_ptr2]
      nop.i 999
}
;;

//
//    N_0_fix  = integer part of N_0 .
//
//
//    Make N_0 the integer part.
//
{ .mfi
      nop.m 999
      fcvt.fx.s1 N_0_fix = N_0
      nop.i 999 ;;
}
{ .mfi
      setf.sig B_mask1 = bmask1               // Form mask to get 5 msb of r
      fcvt.xf N_0 = N_0_fix
      nop.i 999 ;;
}
{ .mfi
      setf.sig B_mask2 = bmask2               // Form mask to form B from r
      fnma.s1 ArgPrime = N_0, P_0, Norm_Arg
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 w = N_0, d_1
      nop.i 999 ;;
}
//
//    ArgPrime = -N_0 * P_0 + Arg
//    w  = N_0 * d_1
//
//
//    N = ArgPrime * 2/pi
//
//      fcvt.fx.s1 N_fix = N
// Use special scaling to right shift so N=Arg * 2/pi is in rightmost bits
// Branch to Cases 3 or 4 if Arg <= -2**24 or Arg >= 2**24
{ .mfi
      nop.m 999
      fma.s1      N_fix = ArgPrime, FR_inv_pi_2to63, FR_rshf_2to64

      nop.i 999 ;;
}
//     Convert integer N_fix back to normalized floating-point value.
{ .mfi
      nop.m 999
      fms.s1 N = N_fix, FR_2tom64, FR_rshf    // Use scaling to get N floated
      nop.i 999
}
;;

//
//    N is the integer part of the reduced-reduced argument.
//    Put the integer in a GP register.
//
{ .mfi
      getf.sig N_fix_gr = N_fix
      nop.f 999
      nop.i 999
}
;;

//
//    s_val = -N*P_1 + ArgPrime
//    w = -N*P_2 + w
//
{ .mfi
      nop.m 999
      fnma.s1 s_val = N, P_1, ArgPrime
      nop.i 999
}
{ .mfi
      nop.m 999
      fnma.s1 w = N, P_2, w
      nop.i 999
}
;;

//    Case 4: V_hi = N * P_2
//    Case 4: U_hi = N_0 * d_1
{ .mfi
      nop.m 999
      fmpy.s1 V_hi = N, P_2               // V_hi = N * P_2 for |s| < 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 U_hi = N_0, d_1             // U_hi = N_0 * d_1 for |s| < 2^-14
      nop.i 999
}
;;

//    Case 3: r = s_val + w (Z complete)
//    Case 4: w = N * P_3
{ .mfi
      nop.m 999
      fadd.s1 r = s_val, w                // r = s_val + w for |s| >= 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 w2 = N, P_3                 // w = N * P_3 for |s| < 2^-14
      nop.i 999
}
;;

//    Case 4: A =  U_hi + V_hi
//    Note: Worry about switched sign of V_hi, so subtract instead of add.
//    Case 4: V_lo = -N * P_2 - V_hi (U_hi is in place of V_hi in writeup)
//    Note: the (-) is still missing for V_hi.
{ .mfi
      nop.m 999
      fsub.s1 A = U_hi, V_hi           // A = U_hi - V_hi for |s| < 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
      fnma.s1 V_lo = N, P_2, V_hi      // V_lo = V_hi - N * P_2 for |s| < 2^-14
      nop.i 999
}
;;

//    Decide between case 3 and 4:
//    Case 3:  |s| >= 2**(-14)     Set p10
//    Case 4:  |s| <  2**(-14)     Set p11
//
//    Case 4: U_lo = N_0 * d_1 - U_hi
{ .mfi
      nop.m 999
      fms.s1 U_lo = N_0, d_1, U_hi     // U_lo = N_0*d_1 - U_hi for |s| < 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
      fcmp.lt.s1 p11, p10 = s_val, TWO_TO_NEG14
      nop.i 999
}
;;

//    Case 4: We need abs of both U_hi and V_hi - dont
//    worry about switched sign of V_hi.
{ .mfi
      nop.m 999
      fabs V_hiabs = V_hi              // |V_hi| for |s| < 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fcmp.gt.s1 p11, p10 = s_val, NEGTWO_TO_NEG14
      nop.i 999
}
;;

//    Case 3: c = s_val - r
{ .mfi
      nop.m 999
      fabs U_hiabs = U_hi              // |U_hi| for |s| < 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
      fsub.s1 c = s_val, r             // c = s_val - r    for |s| >= 2^-14
      nop.i 999
}
;;

// For Case 3, |s| >= 2^-14, determine if |r| < 1/4
//
//    Case 4: C_hi = s_val + A
//
{ .mfi
      nop.m 999
(p11) fadd.s1 C_hi = s_val, A              // C_hi = s_val + A for |s| < 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
(p10) fcmp.lt.unc.s1 p14, p15 = r, TWO_TO_NEG2
      nop.i 999
}
;;

{ .mfi
      getf.sig sig_r = r               // Get signif of r if |s| >= 2^-33
      fand B = B_mask1, r
      nop.i 999
}
;;

//    Case 4: t = U_lo + V_lo
{ .mfi
      getf.exp exp_r = r               // Extract signexp of r if |s| >= 2^-33
(p11) fadd.s1 t = U_lo, V_lo               // t = U_lo + V_lo for |s| < 2^-14
      nop.i 999
}
{ .mfi
      nop.m 999
(p14) fcmp.gt.s1 p14, p15 = r, NEGTWO_TO_NEG2
      nop.i 999
}
;;

//    Case 3: c = (s - r) + w (c complete)
{ .mfi
      nop.m 999
(p10) fadd.s1 c = c, w              // c = c + w for |s| >= 2^-14
      nop.i 999
}
{ .mbb
      nop.m 999
(p14) br.cond.spnt TANL_SMALL_R     // Branch if 2^24 <= |x| < 2^63 and |r|< 1/4
(p15) br.cond.sptk TANL_NORMAL_R_A  // Branch if 2^24 <= |x| < 2^63 and |r|>=1/4
}
;;


// Here if 2^24 <= |x| < 2^63 and |s| < 2^-14  >>>>>>>  Case 4.
//
//    Case 4: Set P_12 if U_hiabs >= V_hiabs
//    Case 4: w = w + N_0 * d_2
//    Note: the (-) is now incorporated in w .
{ .mfi
      add table_ptr1 = 160, table_base           // Point to tanl_table_p1
      fcmp.ge.unc.s1 p12, p13 = U_hiabs, V_hiabs
      nop.i 999
}
{ .mfi
      nop.m 999
      fms.s1 w2 = N_0, d_2, w2
      nop.i 999
}
;;

//    Case 4: C_lo = s_val - C_hi
{ .mfi
      ldfe P1_1 = [table_ptr1], 16               // Load P1_1
      fsub.s1 C_lo = s_val, C_hi
      nop.i 999
}
;;

//
//    Case 4: a = U_hi - A
//            a = V_hi - A (do an add to account for missing (-) on V_hi
//
{ .mfi
      ldfe P1_2 = [table_ptr1], 128              // Load P1_2
(p12) fsub.s1 a = U_hi, A
      nop.i 999
}
{ .mfi
      nop.m 999
(p13) fadd.s1 a = V_hi, A
      nop.i 999
}
;;

//    Case 4: t = U_lo + V_lo  + w
{ .mfi
      ldfe Q1_1 = [table_ptr1], 16               // Load Q1_1
      fadd.s1 t = t, w2
      nop.i 999
}
;;

//    Case 4: a = (U_hi - A)  + V_hi
//            a = (V_hi - A)  + U_hi
//    In each case account for negative missing form V_hi .
//
{ .mfi
      ldfe Q1_2 = [table_ptr1], 16               // Load Q1_2
(p12) fsub.s1 a = a, V_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p13) fsub.s1 a = U_hi, a
      nop.i 999
}
;;

//
//    Case 4: C_lo = (s_val - C_hi) + A
//
{ .mfi
      nop.m 999
      fadd.s1 C_lo = C_lo, A
      nop.i 999 ;;
}
//
//    Case 4: t = t + a
//
{ .mfi
      nop.m 999
      fadd.s1 t = t, a
      nop.i 999
}
;;

//    Case 4: C_lo = C_lo + t
//    Case 4: r = C_hi + C_lo
{ .mfi
      nop.m 999
      fadd.s1 C_lo = C_lo, t
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 r = C_hi, C_lo
      nop.i 999
}
;;

//
//    Case 4: c = C_hi - r
//
{ .mfi
      nop.m 999
      fsub.s1 c = C_hi, r
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 rsq = r, r
      add N_fix_gr = N_fix_gr, cot_flag // N = N + 1 (for cotl)
}
;;

//    Case 4: c = c + C_lo  finished.
//
//    Is i_1 = lsb of N_fix_gr even or odd?
//    if i_1 == 0, set PR_11, else set PR_12.
//
{ .mfi
      nop.m 999
      fadd.s1 c = c , C_lo
      tbit.z p11, p12 =  N_fix_gr, 0
}
;;

// r and c have been computed.
{ .mfi
      nop.m 999
(p12) frcpa.s1 S_hi, p0 = f1, r
      nop.i 999
}
{ .mfi
      nop.m 999
//
//    N odd: Change sign of S_hi
//
(p11) fma.s1 Poly = rsq, P1_2, P1_1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 P = rsq, Q1_2, Q1_1
      nop.i 999
}
{ .mfi
      nop.m 999
//
//    N odd:  Result  =  S_hi + S_lo      (User supplied rounding mode for C1)
//
       fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: rsq = r * r
//    N odd:  S_hi = frcpa(r)
//
(p12) fmerge.ns S_hi = S_hi, S_hi
      nop.i 999
}
{ .mfi
      nop.m 999
//
//    N even: rsq = rsq * P1_2 + P1_1
//    N odd:  poly1 =  1.0 +  S_hi * r    16 bits partial  account for necessary
//
(p11) fmpy.s1 Poly = rsq, Poly
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly1 = S_hi, r,f1
(p11) tbit.z.unc p14, p15 = cot_flag, 0 // p14=1 for tanl; p15=1 for cotl
}
{ .mfi
      nop.m 999
//
//    N even: Poly =  Poly * rsq
//    N odd:  S_hi  = S_hi + S_hi*poly1  16 bits account for necessary
//
(p11) fma.s1 Poly = r, Poly, c
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
      nop.i 999
}
{ .mfi
      nop.m 999
//
//    N odd:   S_hi  = S_hi * poly1 + S_hi   32 bits
//
(p14) fadd.s0 Result = r, Poly          // for tanl
      nop.i 999 ;;
}

.pred.rel "mutex",p15,p12
{ .mfi
      nop.m 999
(p15) fms.s0 Result = r, mOne, Poly     // for cotl
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly1 =  S_hi, r, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: Poly = Poly * r + c
//    N odd:  poly1  = 1.0 + S_hi * r        32 bits partial
//
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly1 = S_hi, r, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: Result = Poly + r  (Rounding mode S0)
//    N odd:  poly1  =  S_hi * r + 1.0       64 bits partial
//
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N odd:  poly1  =  S_hi * poly + S_hi    64 bits
//
(p12) fma.s1 poly1 = S_hi, r, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N odd:  poly1  =  S_hi * r + 1.0
//
(p12) fma.s1 poly1 = S_hi, c, poly1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N odd:  poly1  =  S_hi * c + poly1
//
(p12) fmpy.s1 S_lo = S_hi, poly1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N odd:  S_lo  =  S_hi *  poly1
//
(p12) fma.s1 S_lo = P, r, S_lo
(p12) tbit.z.unc p14, p15 = cot_flag, 0 ;; // p14=1 for tanl; p15=1 for cotl
}

{ .mfi
      nop.m 999
(p14) fadd.s0 Result = S_hi, S_lo           // for tanl
      nop.i 999
}
{ .mfb
      nop.m 999
//
//    N odd:  S_lo  =  S_lo + r * P
//
(p15) fms.s0 Result = S_hi, mOne, S_lo      // for cotl
      br.ret.sptk b0 ;;      // Exit for 2^24 <= |x| < 2^63 and |s| < 2^-14
}


TANL_SMALL_R:
// Here if |r| < 1/4
// r and c have been computed.
// *****************************************************************
// *****************************************************************
// *****************************************************************
//    N odd:  S_hi = frcpa(r)
//    Get [i_1] - lsb of N_fix_gr.  Set p11 if N even, p12 if N odd.
//    N even: rsq = r * r
{ .mfi
      add table_ptr1 = 160, table_base    // Point to tanl_table_p1
      frcpa.s1 S_hi, p0 = f1, r           // S_hi for N odd
      add N_fix_gr = N_fix_gr, cot_flag   // N = N + 1 (for cotl)
}
{ .mfi
      add table_ptr2 = 400, table_base    // Point to Q1_7
      fmpy.s1 rsq = r, r
      nop.i 999
}
;;

{ .mmi
      ldfe P1_1 = [table_ptr1], 16
;;
      ldfe P1_2 = [table_ptr1], 16
      tbit.z p11, p12 = N_fix_gr, 0
}
;;


{ .mfi
      ldfe P1_3 = [table_ptr1], 96
      nop.f 999
      nop.i 999
}
;;

{ .mfi
(p11) ldfe P1_9 = [table_ptr1], -16
(p12) fmerge.ns S_hi = S_hi, S_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fmpy.s1 r_to_the_8 = rsq, rsq
      nop.i 999
}
;;

//
//    N even: Poly2 = P1_7 + Poly2 * rsq
//    N odd:  poly2 = Q1_5 + poly2 * rsq
//
{ .mfi
(p11) ldfe P1_8 = [table_ptr1], -16
(p11) fadd.s1 CORR = rsq, f1
      nop.i 999
}
;;

//
//    N even: Poly1 = P1_2 + P1_3 * rsq
//    N odd:  poly1 =  1.0 +  S_hi * r
//    16 bits partial  account for necessary (-1)
//
{ .mmi
(p11) ldfe P1_7 = [table_ptr1], -16
;;
(p11) ldfe P1_6 = [table_ptr1], -16
      nop.i 999
}
;;

//
//    N even: Poly1 = P1_1 + Poly1 * rsq
//    N odd:  S_hi  =  S_hi + S_hi * poly1)     16 bits account for necessary
//
//
//    N even: Poly2 = P1_5 + Poly2 * rsq
//    N odd:  poly2 = Q1_3 + poly2 * rsq
//
{ .mfi
(p11) ldfe P1_5 = [table_ptr1], -16
(p11) fmpy.s1 r_to_the_8 = r_to_the_8, r_to_the_8
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly1 =  S_hi, r, f1
      nop.i 999
}
;;

//
//    N even: Poly1 =  Poly1 * rsq
//    N odd:  poly1  = 1.0 + S_hi * r         32 bits partial
//

//
//    N even: CORR =  CORR * c
//    N odd:  S_hi  =  S_hi * poly1 + S_hi    32 bits
//

//
//    N even: Poly2 = P1_6 + Poly2 * rsq
//    N odd:  poly2 = Q1_4 + poly2 * rsq
//

{ .mmf
(p11) ldfe P1_4 = [table_ptr1], -16
      nop.m 999
(p11) fmpy.s1 CORR =  CORR, c
}
;;

{ .mfi
      nop.m 999
(p11) fma.s1 Poly1 = P1_3, rsq, P1_2
      nop.i 999 ;;
}
{ .mfi
(p12) ldfe Q1_7 = [table_ptr2], -16
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
      nop.i 999 ;;
}
{ .mfi
(p12) ldfe Q1_6 = [table_ptr2], -16
(p11) fma.s1 Poly2 = P1_9, rsq, P1_8
      nop.i 999 ;;
}
{ .mmi
(p12) ldfe Q1_5 = [table_ptr2], -16 ;;
(p12) ldfe Q1_4 = [table_ptr2], -16
      nop.i 999 ;;
}
{ .mfi
(p12) ldfe Q1_3 = [table_ptr2], -16
//
//    N even: Poly2 = P1_8 + P1_9 * rsq
//    N odd:  poly2 = Q1_6 + Q1_7 * rsq
//
(p11) fma.s1 Poly1 = Poly1, rsq, P1_1
      nop.i 999 ;;
}
{ .mfi
(p12) ldfe Q1_2 = [table_ptr2], -16
(p12) fma.s1 poly1 = S_hi, r, f1
      nop.i 999 ;;
}
{ .mfi
(p12) ldfe Q1_1 = [table_ptr2], -16
(p11) fma.s1 Poly2 = Poly2, rsq, P1_7
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: CORR =  rsq + 1
//    N even: r_to_the_8 =  rsq * rsq
//
(p11) fmpy.s1 Poly1 = Poly1, rsq
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly2 = Q1_7, rsq, Q1_6
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p11) fma.s1 Poly2 = Poly2, rsq, P1_6
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly1 = S_hi, r, f1
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly2 = poly2, rsq, Q1_5
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p11) fma.s1 Poly2= Poly2, rsq, P1_5
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 S_hi =  S_hi, poly1, S_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly2 = poly2, rsq, Q1_4
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: r_to_the_8 = r_to_the_8 * r_to_the_8
//    N odd:  poly1  =  S_hi * r + 1.0       64 bits partial
//
(p11) fma.s1 Poly2 = Poly2, rsq, P1_4
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: Poly = CORR + Poly * r
//    N odd:  P = Q1_1 + poly2 * rsq
//
(p12) fma.s1 poly1 = S_hi, r, f1
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly2 = poly2, rsq, Q1_3
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: Poly2 = P1_4 + Poly2 * rsq
//    N odd:  poly2 = Q1_2 + poly2 * rsq
//
(p11) fma.s1 Poly = Poly2, r_to_the_8, Poly1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly1 = S_hi, c, poly1
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fma.s1 poly2 = poly2, rsq, Q1_2
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//    N even: Poly = Poly1 + Poly2 * r_to_the_8
//    N odd:  S_hi =  S_hi * poly1 + S_hi    64 bits
//
(p11) fma.s1 Poly = Poly, r, CORR
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: Result =  r + Poly  (User supplied rounding mode)
//    N odd:  poly1  =  S_hi * c + poly1
//
(p12) fmpy.s1 S_lo = S_hi, poly1
(p11) tbit.z.unc p14, p15 = cot_flag, 0 // p14=1 for tanl; p15=1 for cotl
}
{ .mfi
      nop.m 999
(p12) fma.s1 P = poly2, rsq, Q1_1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N odd:  poly1  =  S_hi * r + 1.0
//
//
//    N odd:  S_lo  =  S_hi *  poly1
//
(p14) fadd.s0 Result = Poly, r          // for tanl
      nop.i 999
}
{ .mfi
      nop.m 999
(p15) fms.s0 Result = Poly, mOne, r     // for cotl
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//    N odd:  S_lo  =  Q1_1 * c + S_lo
//
(p12) fma.s1 S_lo = Q1_1, c, S_lo
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N odd:  Result =  S_lo + r * P
//
(p12) fma.s1 Result = P, r, S_lo
(p12) tbit.z.unc p14, p15 = cot_flag, 0 ;; // p14=1 for tanl; p15=1 for cotl
}

//
//    N odd:  Result = Result + S_hi  (user supplied rounding mode)
//
{ .mfi
      nop.m 999
(p14) fadd.s0 Result = Result, S_hi         // for tanl
      nop.i 999
}
{ .mfb
      nop.m 999
(p15) fms.s0 Result = Result, mOne, S_hi    // for cotl
      br.ret.sptk b0 ;;              // Exit |r| < 1/4 path
}


TANL_NORMAL_R:
// Here if 1/4 <= |x| < pi/4  or  if |x| >= 2^63 and |r| >= 1/4
// *******************************************************************
// *******************************************************************
// *******************************************************************
//
//    r and c have been computed.
//
{ .mfi
      nop.m 999
      fand B = B_mask1, r
      nop.i 999
}
;;

TANL_NORMAL_R_A:
// Enter here if pi/4 <= |x| < 2^63 and |r| >= 1/4
//    Get the 5 bits or r for the lookup.   1.xxxxx ....
{ .mmi
      add table_ptr1 = 416, table_base     // Point to tanl_table_p2
      mov GR_exp_2tom65 = 0xffff - 65      // Scaling constant for B
      extr.u lookup = sig_r, 58, 5
}
;;

{ .mmi
      ldfe P2_1 = [table_ptr1], 16
      setf.exp TWO_TO_NEG65 = GR_exp_2tom65  // 2^-65 for scaling B if exp_r=-2
      add N_fix_gr = N_fix_gr, cot_flag      // N = N + 1 (for cotl)
}
;;

.pred.rel "mutex",p11,p12
//    B =  2^63 * 1.xxxxx 100...0
{ .mfi
      ldfe P2_2 = [table_ptr1], 16
      for B = B_mask2, B
      mov table_offset = 512               // Assume table offset is 512
}
;;

{ .mfi
      ldfe P2_3 = [table_ptr1], 16
      fmerge.s  Pos_r = f1, r
      tbit.nz p8,p9 = exp_r, 0
}
;;

//    Is  B = 2** -2 or  B= 2** -1? If 2**-1, then
//    we want an offset of 512 for table addressing.
{ .mii
      add table_ptr2 = 1296, table_base     // Point to tanl_table_cm2
(p9)  shladd table_offset = lookup, 4, table_offset
(p8)  shladd table_offset = lookup, 4, r0
}
;;

{ .mmi
      add table_ptr1 = table_ptr1, table_offset  // Point to T_hi
      add table_ptr2 = table_ptr2, table_offset  // Point to C_hi
      add table_ptr3 = 2128, table_base     // Point to tanl_table_scim2
}
;;

{ .mmi
      ldfd T_hi = [table_ptr1], 8                // Load T_hi
;;
      ldfd C_hi = [table_ptr2], 8                // Load C_hi
      add table_ptr3 = table_ptr3, table_offset  // Point to SC_inv
}
;;

//
//    x = |r| - B
//
//   Convert B so it has the same exponent as Pos_r before subtracting
{ .mfi
      ldfs T_lo = [table_ptr1]                   // Load T_lo
(p9)  fnma.s1 x = B, FR_2tom64, Pos_r
      nop.i 999
}
{ .mfi
      nop.m 999
(p8)  fnma.s1 x = B, TWO_TO_NEG65, Pos_r
      nop.i 999
}
;;

{ .mfi
      ldfs C_lo = [table_ptr2]                   // Load C_lo
      nop.f 999
      nop.i 999
}
;;

{ .mfi
      ldfe SC_inv = [table_ptr3]                 // Load SC_inv
      fmerge.s  sgn_r = r, f1
      tbit.z p11, p12 = N_fix_gr, 0              // p11 if N even, p12 if odd

}
;;

//
//    xsq = x * x
//    N even: Tx = T_hi * x
//
//    N even: Tx1 = Tx + 1
//    N odd:  Cx1 = 1 - Cx
//

{ .mfi
      nop.m 999
      fmpy.s1 xsq = x, x
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fmpy.s1 Tx = T_hi, x
      nop.i 999
}
;;

//
//    N odd: Cx = C_hi * x
//
{ .mfi
      nop.m 999
(p12) fmpy.s1 Cx = C_hi, x
      nop.i 999
}
;;
//
//    N even and odd: P = P2_3 + P2_2 * xsq
//
{ .mfi
      nop.m 999
      fma.s1 P = P2_3, xsq, P2_2
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fadd.s1 Tx1 = Tx, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: D = C_hi - tanx
//    N odd: D = T_hi + tanx
//
(p11) fmpy.s1 CORR = SC_inv, T_hi
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 Sx = SC_inv, x
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fmpy.s1 CORR = SC_inv, C_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fsub.s1 V_hi = f1, Cx
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
      fma.s1 P = P, xsq, P2_1
      nop.i 999
}
{ .mfi
      nop.m 999
//
//    N even and odd: P = P2_1 + P * xsq
//
(p11) fma.s1 V_hi = Tx, Tx1, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: Result  = sgn_r * tail + T_hi (user rounding mode for C1)
//    N odd:  Result  = sgn_r * tail + C_hi (user rounding mode for C1)
//
      fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
      fmpy.s1 CORR = CORR, c
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fnma.s1 V_hi = Cx,V_hi,f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: V_hi = Tx * Tx1 + 1
//    N odd: Cx1 = 1 - Cx * Cx1
//
      fmpy.s1 P = P, xsq
      nop.i 999
}
{ .mfi
      nop.m 999
//
//    N even and odd: P = P * xsq
//
(p11) fmpy.s1 V_hi = V_hi, T_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even and odd: tail = P * tail + V_lo
//
(p11) fmpy.s1 T_hi = sgn_r, T_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
      fmpy.s1 CORR = CORR, sgn_r
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
(p12) fmpy.s1 V_hi = V_hi,C_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: V_hi = T_hi * V_hi
//    N odd: V_hi  = C_hi * V_hi
//
      fma.s1 tanx = P, x, x
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fnmpy.s1 C_hi = sgn_r, C_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: V_lo = 1 - V_hi + C_hi
//    N odd: V_lo = 1 - V_hi + T_hi
//
(p11) fadd.s1 CORR = CORR, T_lo
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fsub.s1 CORR = CORR, C_lo
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even and odd: tanx = x + x * P
//    N even and odd: Sx = SC_inv * x
//
(p11) fsub.s1 D = C_hi, tanx
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fadd.s1 D = T_hi, tanx
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N odd: CORR = SC_inv * C_hi
//    N even: CORR = SC_inv * T_hi
//
      fnma.s1 D = V_hi, D, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even and odd: D = 1 - V_hi * D
//    N even and odd: CORR = CORR * c
//
      fma.s1 V_hi = V_hi, D, V_hi
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even and odd: V_hi = V_hi + V_hi * D
//    N even and odd: CORR = sgn_r * CORR
//
(p11) fnma.s1 V_lo = V_hi, C_hi, f1
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fnma.s1 V_lo = V_hi, T_hi, f1
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: CORR = COOR + T_lo
//    N odd: CORR = CORR - C_lo
//
(p11) fma.s1 V_lo = tanx, V_hi, V_lo
      tbit.nz p15, p0 = cot_flag, 0       // p15=1 if we compute cotl
}
{ .mfi
      nop.m 999
(p12) fnma.s1 V_lo = tanx, V_hi, V_lo
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p15) fms.s1 T_hi = f0, f0, T_hi        // to correct result's sign for cotl
      nop.i 999
}
{ .mfi
      nop.m 999
(p15) fms.s1 C_hi = f0, f0, C_hi        // to correct result's sign for cotl
      nop.i 999
};;

{ .mfi
      nop.m 999
(p15) fms.s1 sgn_r = f0, f0, sgn_r      // to correct result's sign for cotl
      nop.i 999
};;

{ .mfi
      nop.m 999
//
//    N even: V_lo = V_lo + V_hi * tanx
//    N odd: V_lo = V_lo - V_hi * tanx
//
(p11) fnma.s1 V_lo = C_lo, V_hi, V_lo
      nop.i 999
}
{ .mfi
      nop.m 999
(p12) fnma.s1 V_lo = T_lo, V_hi, V_lo
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N  even: V_lo = V_lo - V_hi * C_lo
//    N  odd: V_lo = V_lo - V_hi * T_lo
//
      fmpy.s1 V_lo = V_hi, V_lo
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even and odd: V_lo = V_lo * V_hi
//
      fadd.s1 tail = V_hi, V_lo
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even and odd: tail = V_hi + V_lo
//
      fma.s1 tail = tail, P, V_lo
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even: T_hi = sgn_r * T_hi
//    N odd : C_hi = -sgn_r * C_hi
//
      fma.s1 tail = tail, Sx, CORR
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even and odd: tail = Sx * tail + CORR
//
      fma.s1 tail = V_hi, Sx, tail
      nop.i 999 ;;
}
{ .mfi
      nop.m 999
//
//    N even an odd: tail = Sx * V_hi + tail
//
(p11) fma.s0 Result = sgn_r, tail, T_hi
      nop.i 999
}
{ .mfb
      nop.m 999
(p12) fma.s0 Result = sgn_r, tail, C_hi
      br.ret.sptk b0 ;;                 // Exit for 1/4 <= |r| < pi/4
}

TANL_DENORMAL:
// Here if x denormal
{ .mfb
      getf.exp GR_signexp_x = Norm_Arg          // Get sign and exponent of x
      nop.f 999
      br.cond.sptk TANL_COMMON                  // Return to common code
}
;;


TANL_SPECIAL:
TANL_UNSUPPORTED:
//
//     Code for NaNs, Unsupporteds, Infs, or +/- zero ?
//     Invalid raised for Infs and SNaNs.
//

{ .mfi
      nop.m 999
      fmerge.s  f10 = f8, f8            // Save input for error call
      tbit.nz p6, p7 = cot_flag, 0      // p6=1 if we compute cotl
}
;;

{ .mfi
      nop.m 999
(p6)  fclass.m p6, p7 = f8, 0x7         // Test for zero (cotl only)
      nop.i 999
}
;;

.pred.rel "mutex", p6, p7
{ .mfi
(p6)  mov GR_Parameter_Tag = 225        // (cotl)
(p6)  frcpa.s0  f8, p0 = f1, f8         // cotl(+-0) = +-Inf
      nop.i 999
}
{ .mfb
      nop.m 999
(p7)  fmpy.s0 f8 = f8, f0
(p7)  br.ret.sptk b0
}
;;

GLOBAL_IEEE754_END(tanl)
libm_alias_ldouble_other (__tan, tan)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue

// (1)
{ .mfi
      add           GR_Parameter_Y=-32,sp        // Parameter 2 value
      nop.f         0
.save   ar.pfs,GR_SAVE_PFS
      mov           GR_SAVE_PFS=ar.pfs           // Save ar.pfs
}
{ .mfi
.fframe 64
      add sp=-64,sp                              // Create new stack
      nop.f 0
      mov GR_SAVE_GP=gp                          // Save gp
};;

// (2)
{ .mmi
      stfe [GR_Parameter_Y] = f1,16              // STORE Parameter 2 on stack
      add GR_Parameter_X = 16,sp                 // Parameter 1 address
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                          // Save b0
};;

.body
// (3)
{ .mib
      stfe [GR_Parameter_X] = f10                // STORE Parameter 1 on stack
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
      nop.b 0
}
{ .mib
      stfe [GR_Parameter_Y] = f8                 // STORE Parameter 3 on stack
      add   GR_Parameter_Y = -16,GR_Parameter_Y
      br.call.sptk b0=__libm_error_support#      // Call error handling function
};;
{ .mmi
      nop.m 0
      nop.m 0
      add   GR_Parameter_RESULT = 48,sp
};;

// (4)
{ .mmi
      ldfe  f8 = [GR_Parameter_RESULT]           // Get return result off stack
.restore sp
      add   sp = 64,sp                           // Restore stack pointer
      mov   b0 = GR_SAVE_B0                      // Restore return address
};;
{ .mib
      mov   gp = GR_SAVE_GP                      // Restore gp
      mov   ar.pfs = GR_SAVE_PFS                 // Restore ar.pfs
      br.ret.sptk     b0                         // Return
};;

LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#


// *******************************************************************
// *******************************************************************
// *******************************************************************
//
//     Special Code to handle very large argument case.
//     Call int __libm_pi_by_2_reduce(x,r,c) for |arguments| >= 2**63
//     The interface is custom:
//       On input:
//         (Arg or x) is in f8
//       On output:
//         r is in f8
//         c is in f9
//         N is in r8
//     We know also that __libm_pi_by_2_reduce preserves f10-15, f71-127.  We
//     use this to eliminate save/restore of key fp registers in this calling
//     function.
//
// *******************************************************************
// *******************************************************************
// *******************************************************************

LOCAL_LIBM_ENTRY(__libm_callout)
TANL_ARG_TOO_LARGE:
.prologue
{ .mfi
      add table_ptr2 = 144, table_base        // Point to 2^-2
      nop.f 999
.save   ar.pfs,GR_SAVE_PFS
      mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
}
;;

//     Load 2^-2, -2^-2
{ .mmi
      ldfps  TWO_TO_NEG2, NEGTWO_TO_NEG2 = [table_ptr2]
      setf.sig B_mask1 = bmask1               // Form mask to get 5 msb of r
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                       // Save b0
};;

.body
//
//     Call argument reduction with x in f8
//     Returns with N in r8, r in f8, c in f9
//     Assumes f71-127 are preserved across the call
//
{ .mib
      setf.sig B_mask2 = bmask2               // Form mask to form B from r
      mov GR_SAVE_GP=gp                       // Save gp
      br.call.sptk b0=__libm_pi_by_2_reduce#
}
;;

//
//     Is |r| < 2**(-2)
//
{ .mfi
      getf.sig sig_r = r                     // Extract significand of r
      fcmp.lt.s1  p6, p0 = r, TWO_TO_NEG2
      mov   gp = GR_SAVE_GP                  // Restore gp
}
;;

{ .mfi
      getf.exp exp_r = r                     // Extract signexp of r
      nop.f 999
      mov    b0 = GR_SAVE_B0                 // Restore return address
}
;;

//
//     Get N_fix_gr
//
{ .mfi
      mov   N_fix_gr = r8
(p6)  fcmp.gt.unc.s1  p6, p0 = r, NEGTWO_TO_NEG2
      mov   ar.pfs = GR_SAVE_PFS             // Restore pfs
}
;;

{ .mbb
      nop.m 999
(p6)  br.cond.spnt TANL_SMALL_R              // Branch if |r| < 1/4
      br.cond.sptk TANL_NORMAL_R             // Branch if 1/4 <= |r| < pi/4
}
;;

LOCAL_LIBM_END(__libm_callout)

.type __libm_pi_by_2_reduce#,@function
.global __libm_pi_by_2_reduce#
