.file "libm_tan.s"

// Copyright (C) 2000, 2001, Intel Corporation
// All rights reserved.
//
// Contributed 2/2/2000 by John Harrison, Ted Kubaska, Bob Norin, Shane Story,
// and Ping Tak Peter Tang of the Computational Software Lab, Intel Corporation.
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
//
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
// http://developer.intel.com/opensource.
//
// *********************************************************************
//
// History:
// 02/02/00 Initial Version
// 4/04/00  Unwind support added
// 12/28/00 Fixed false invalid flags
//
// *********************************************************************
//
// Function:   tan(x) = tangent(x), for double precision x values
//
// *********************************************************************
//
// Accuracy:       Very accurate for double-precision values
//
// *********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8 (Input and Return Value)
//                              f9-f15
//                              f32-f112
//
//    General Purpose Registers:
//      r32-r48
//      r49-r50 (Used to pass arguments to pi_by_2 reduce routine)
//
//    Predicate Registers:      p6-p15
//
// *********************************************************************
//
// IEEE Special Conditions:
//
//    Denormal  fault raised on denormal inputs
//    Overflow exceptions do not occur
//    Underflow exceptions raised when appropriate for tan
//    (No specialized error handling for this routine)
//    Inexact raised when appropriate by algorithm
//
//    tan(SNaN) = QNaN
//    tan(QNaN) = QNaN
//    tan(inf) = QNaN
//    tan(+/-0) = +/-0
//
// *********************************************************************
//
// Mathematical Description
//
// We consider the computation of FPTAN of Arg. Now, given
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
//        -cot(r + c) = -1/(r+c) + r/3          ...accurately
//
// Case 4:
// -------
//
//      tan(r + c) = r + c + r^3/3 + 2r^5/15     ...accurately
//        -cot(r + c) = -1/(r+c) + r/3 + r^3/45     ...accurately
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
//            = -cot(r+c)          otherwise.
//
// Here for this case, both tan(r) and -cot(r) can be approximated
// by simple polynomials:
//
//      tan(r) =    r + P1_1 r^3 + P1_2 r^5 + ... + P1_9 r^19
//        -cot(r) = -1/r + Q1_1 r   + Q1_2 r^3 + ... + Q1_7 r^13
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
//        -cot(r + c)  = -cot(r)  +  correction.
//
// Specifically,
//
//      tan(r + c) =  tan(r) + c tan'(r)  + O(c^2)
//              =  tan(r) + c sec^2(r) + O(c^2)
//              =  tan(r) + c SEC_sq     ...accurately
//                as long as SEC_sq approximates sec^2(r)
//                to, say, 5 bits or so.
//
// Similarly,
//
//        -cot(r + c) = -cot(r) - c cot'(r)  + O(c^2)
//              = -cot(r) + c csc^2(r) + O(c^2)
//              = -cot(r) + c CSC_sq     ...accurately
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
//         x = |r| - B
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
//            = -cot(r) + correction    otherwise.
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
//                  = -1/(r+c) + Q1_1 r   + Q1_2 r^3 + ... + Q1_7 r^13
//               + Q1_1*c                    N odd
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
//                  = sgn_r * ( tan(B+x)  + sgn_r*c*sec^2(|r|) )
//                  = sgn_r * ( tan(B+x)  + sgn_r*c*sec^2(B) )
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
//                  = sgn_r * ( -cot(B+x)  + sgn_r*c*csc^2(|r|) )
//                  = sgn_r * ( -cot(B+x)  + sgn_r*c*csc^2(B) )
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
//    Result := c + r * rsq * P1_1
//    Result := r + Result          ...in user-defined rounding
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
//    Result := c + r * rsq * (P1_1 + rsq * P1_2)
//    Result := r + Result          ...in user-defined rounding
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
//      Result := r*Poly + CORR
//      Result := r + Result     ...in user-defined rounding
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
//      Result :=  r*P + S_lo
//      Result :=  S_hi  +  Result      ...in user-defined rounding
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
//      tan(B)                as  T_hi  +  T_lo;
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
//      tan(B)                as  T_hi  +  T_lo;
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

#include "libm_support.h"

#ifdef _LIBC
.rodata
#else
.data
#endif

.align 128

TAN_BASE_CONSTANTS:
.type TAN_BASE_CONSTANTS, @object
data4    0x4B800000, 0xCB800000, 0x38800000, 0xB8800000 // two**24, -two**24
                                                        // two**-14, -two**-14
data4    0x4E44152A, 0xA2F9836E, 0x00003FFE, 0x00000000 // two_by_pi
data4    0xCE81B9F1, 0xC84D32B0, 0x00004016, 0x00000000 // P_0
data4    0x2168C235, 0xC90FDAA2, 0x00003FFF, 0x00000000 // P_1
data4    0xFC8F8CBB, 0xECE675D1, 0x0000BFBD, 0x00000000 // P_2
data4    0xACC19C60, 0xB7ED8FBB, 0x0000BF7C, 0x00000000 // P_3
data4    0x5F000000, 0xDF000000, 0x00000000, 0x00000000 // two_to_63, -two_to_63
data4    0x6EC6B45A, 0xA397E504, 0x00003FE7, 0x00000000 // Inv_P_0
data4    0xDBD171A1, 0x8D848E89, 0x0000BFBF, 0x00000000 // d_1
data4    0x18A66F8E, 0xD5394C36, 0x0000BF7C, 0x00000000 // d_2
data4    0x2168C234, 0xC90FDAA2, 0x00003FFE, 0x00000000 // PI_BY_4
data4    0x2168C234, 0xC90FDAA2, 0x0000BFFE, 0x00000000 // MPI_BY_4
data4    0x3E800000, 0xBE800000, 0x00000000, 0x00000000 // two**-2, -two**-2
data4    0x2F000000, 0xAF000000, 0x00000000, 0x00000000 // two**-33, -two**-33
data4    0xAAAAAABD, 0xAAAAAAAA, 0x00003FFD, 0x00000000 // P1_1
data4    0x88882E6A, 0x88888888, 0x00003FFC, 0x00000000 // P1_2
data4    0x0F0177B6, 0xDD0DD0DD, 0x00003FFA, 0x00000000 // P1_3
data4    0x646B8C6D, 0xB327A440, 0x00003FF9, 0x00000000 // P1_4
data4    0x1D5F7D20, 0x91371B25, 0x00003FF8, 0x00000000 // P1_5
data4    0x61C67914, 0xEB69A5F1, 0x00003FF6, 0x00000000 // P1_6
data4    0x019318D2, 0xBEDD37BE, 0x00003FF5, 0x00000000 // P1_7
data4    0x3C794015, 0x9979B146, 0x00003FF4, 0x00000000 // P1_8
data4    0x8C6EB58A, 0x8EBD21A3, 0x00003FF3, 0x00000000 // P1_9
data4    0xAAAAAAB4, 0xAAAAAAAA, 0x00003FFD, 0x00000000 // Q1_1
data4    0x0B5FC93E, 0xB60B60B6, 0x00003FF9, 0x00000000 // Q1_2
data4    0x0C9BBFBF, 0x8AB355E0, 0x00003FF6, 0x00000000 // Q1_3
data4    0xCBEE3D4C, 0xDDEBBC89, 0x00003FF2, 0x00000000 // Q1_4
data4    0x5F80BBB6, 0xB3548A68, 0x00003FEF, 0x00000000 // Q1_5
data4    0x4CED5BF1, 0x91362560, 0x00003FEC, 0x00000000 // Q1_6
data4    0x8EE92A83, 0xF189D95A, 0x00003FE8, 0x00000000 // Q1_7
data4    0xAAAB362F, 0xAAAAAAAA, 0x00003FFD, 0x00000000 // P2_1
data4    0xE97A6097, 0x88888886, 0x00003FFC, 0x00000000 // P2_2
data4    0x25E716A1, 0xDD108EE0, 0x00003FFA, 0x00000000 // P2_3
//
//  Entries T_hi   double-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//  Entries T_lo  single-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//
data4    0x62400794, 0x3FD09BC3, 0x23A05C32, 0x00000000
data4    0xDFFBC074, 0x3FD124A9, 0x240078B2, 0x00000000
data4    0x5BD4920F, 0x3FD1AE23, 0x23826B8E, 0x00000000
data4    0x15E2701D, 0x3FD23835, 0x22D31154, 0x00000000
data4    0x63739C2D, 0x3FD2C2E4, 0x2265C9E2, 0x00000000
data4    0xAFEEA48B, 0x3FD34E36, 0x245C05EB, 0x00000000
data4    0x7DBB35D1, 0x3FD3DA31, 0x24749F2D, 0x00000000
data4    0x67321619, 0x3FD466DA, 0x2462CECE, 0x00000000
data4    0x1F94A4D5, 0x3FD4F437, 0x246D0DF1, 0x00000000
data4    0x740C3E6D, 0x3FD5824D, 0x240A85B5, 0x00000000
data4    0x4CB1E73D, 0x3FD61123, 0x23F96E33, 0x00000000
data4    0xAD9EA64B, 0x3FD6A0BE, 0x247C5393, 0x00000000
data4    0xB804FD01, 0x3FD73125, 0x241F3B29, 0x00000000
data4    0xAB53EE83, 0x3FD7C25E, 0x2479989B, 0x00000000
data4    0xE6640EED, 0x3FD8546F, 0x23B343BC, 0x00000000
data4    0xE8AF1892, 0x3FD8E75F, 0x241454D1, 0x00000000
data4    0x53928BDA, 0x3FD97B35, 0x238613D9, 0x00000000
data4    0xEB9DE4DE, 0x3FDA0FF6, 0x22859FA7, 0x00000000
data4    0x99ECF92D, 0x3FDAA5AB, 0x237A6D06, 0x00000000
data4    0x6D8F1796, 0x3FDB3C5A, 0x23952F6C, 0x00000000
data4    0x9CFB8BE4, 0x3FDBD40A, 0x2280FC95, 0x00000000
data4    0x87943100, 0x3FDC6CC3, 0x245D2EC0, 0x00000000
data4    0xB736C500, 0x3FDD068C, 0x23C4AD7D, 0x00000000
data4    0xE1DDBC31, 0x3FDDA16D, 0x23D076E6, 0x00000000
data4    0xEB515A93, 0x3FDE3D6E, 0x244809A6, 0x00000000
data4    0xE6E9E5F1, 0x3FDEDA97, 0x220856C8, 0x00000000
data4    0x1963CE69, 0x3FDF78F1, 0x244BE993, 0x00000000
data4    0x7D635BCE, 0x3FE00C41, 0x23D21799, 0x00000000
data4    0x1C302CD3, 0x3FE05CAB, 0x248A1B1D, 0x00000000
data4    0xDB6A1FA0, 0x3FE0ADB9, 0x23D53E33, 0x00000000
data4    0x4A20BA81, 0x3FE0FF72, 0x24DB9ED5, 0x00000000
data4    0x153FA6F5, 0x3FE151D9, 0x24E9E451, 0x00000000
//
//  Entries T_hi   double-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//  Entries T_lo  single-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//
data4    0xBA1BE39E, 0x3FE1CEC4, 0x24B60F9E, 0x00000000
data4    0x5ABD9B2D, 0x3FE277E4, 0x248C2474, 0x00000000
data4    0x0272B110, 0x3FE32418, 0x247B8311, 0x00000000
data4    0x890E2DF0, 0x3FE3D38B, 0x24C55751, 0x00000000
data4    0x46236871, 0x3FE4866D, 0x24E5BC34, 0x00000000
data4    0x45E044B0, 0x3FE53CEE, 0x24001BA4, 0x00000000
data4    0x82EC06E4, 0x3FE5F742, 0x24B973DC, 0x00000000
data4    0x25DF43F9, 0x3FE6B5A1, 0x24895440, 0x00000000
data4    0xCAFD348C, 0x3FE77844, 0x240021CA, 0x00000000
data4    0xCEED6B92, 0x3FE83F6B, 0x24C45372, 0x00000000
data4    0xA34F3665, 0x3FE90B58, 0x240DAD33, 0x00000000
data4    0x2C1E56B4, 0x3FE9DC52, 0x24F846CE, 0x00000000
data4    0x27041578, 0x3FEAB2A4, 0x2323FB6E, 0x00000000
data4    0x9DD8C373, 0x3FEB8E9F, 0x24B3090B, 0x00000000
data4    0x65C9AA7B, 0x3FEC709B, 0x2449F611, 0x00000000
data4    0xACCF8435, 0x3FED58F4, 0x23616A7E, 0x00000000
data4    0x97635082, 0x3FEE480F, 0x24C2FEAE, 0x00000000
data4    0xF0ACC544, 0x3FEF3E57, 0x242CE964, 0x00000000
data4    0xF7E06E4B, 0x3FF01E20, 0x2480D3EE, 0x00000000
data4    0x8A798A69, 0x3FF0A125, 0x24DB8967, 0x00000000
//
//  Entries C_hi   double-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//  Entries C_lo  single-precision memory format
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//
data4    0xE63EFBD0, 0x400ED3E2, 0x259D94D4, 0x00000000
data4    0xC515DAB5, 0x400DDDB4, 0x245F0537, 0x00000000
data4    0xBE19A79F, 0x400CF57A, 0x25D4EA9F, 0x00000000
data4    0xD15298ED, 0x400C1A06, 0x24AE40A0, 0x00000000
data4    0x164B2708, 0x400B4A4C, 0x25A5AAB6, 0x00000000
data4    0x5285B068, 0x400A855A, 0x25524F18, 0x00000000
data4    0x3FFA549F, 0x4009CA5A, 0x24C999C0, 0x00000000
data4    0x646AF623, 0x4009188A, 0x254FD801, 0x00000000
data4    0x6084D0E7, 0x40086F3C, 0x2560F5FD, 0x00000000
data4    0xA29A76EE, 0x4007CDD2, 0x255B9D19, 0x00000000
data4    0x6C8ECA95, 0x400733BE, 0x25CB021B, 0x00000000
data4    0x1F8DDC52, 0x4006A07E, 0x24AB4722, 0x00000000
data4    0xC298AD58, 0x4006139B, 0x252764E2, 0x00000000
data4    0xBAD7164B, 0x40058CAB, 0x24DAF5DB, 0x00000000
data4    0xAE31A5D3, 0x40050B4B, 0x25EA20F4, 0x00000000
data4    0x89F85A8A, 0x40048F21, 0x2583A3E8, 0x00000000
data4    0xA862380D, 0x400417DA, 0x25DCC4CC, 0x00000000
data4    0x1088FCFE, 0x4003A52B, 0x2430A492, 0x00000000
data4    0xCD3527D5, 0x400336CC, 0x255F77CF, 0x00000000
data4    0x5760766D, 0x4002CC7F, 0x25DA0BDA, 0x00000000
data4    0x11CE02E3, 0x40026607, 0x256FF4A2, 0x00000000
data4    0xD37BBE04, 0x4002032C, 0x25208AED, 0x00000000
data4    0x7F050775, 0x4001A3BD, 0x24B72DD6, 0x00000000
data4    0xA554848A, 0x40014789, 0x24AB4DAA, 0x00000000
data4    0x323E81B7, 0x4000EE65, 0x2584C440, 0x00000000
data4    0x21CF1293, 0x40009827, 0x25C9428D, 0x00000000
data4    0x3D415EEB, 0x400044A9, 0x25DC8482, 0x00000000
data4    0xBD72C577, 0x3FFFE78F, 0x257F5070, 0x00000000
data4    0x75EFD28E, 0x3FFF4AC3, 0x23EBBF7A, 0x00000000
data4    0x60B52DDE, 0x3FFEB2AF, 0x22EECA07, 0x00000000
data4    0x35204180, 0x3FFE1F19, 0x24191079, 0x00000000
data4    0x54F7E60A, 0x3FFD8FCA, 0x248D3058, 0x00000000
//
//  Entries C_hi   double-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//  Entries C_lo  single-precision memory format
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//
data4    0x79F6FADE, 0x3FFCC06A, 0x239C7886, 0x00000000
data4    0x891662A6, 0x3FFBB91F, 0x250BD191, 0x00000000
data4    0x529F155D, 0x3FFABFB6, 0x256CC3E6, 0x00000000
data4    0x2E964AE9, 0x3FF9D300, 0x250843E3, 0x00000000
data4    0x89DCB383, 0x3FF8F1EF, 0x2277C87E, 0x00000000
data4    0x7C87DBD6, 0x3FF81B93, 0x256DA6CF, 0x00000000
data4    0x1042EDE4, 0x3FF74F14, 0x2573D28A, 0x00000000
data4    0x1784B360, 0x3FF68BAF, 0x242E489A, 0x00000000
data4    0x7C923C4C, 0x3FF5D0B5, 0x2532D940, 0x00000000
data4    0xF418EF20, 0x3FF51D88, 0x253C7DD6, 0x00000000
data4    0x02F88DAE, 0x3FF4719A, 0x23DB59BF, 0x00000000
data4    0x49DA0788, 0x3FF3CC66, 0x252B4756, 0x00000000
data4    0x0B980DB8, 0x3FF32D77, 0x23FE585F, 0x00000000
data4    0xE56C987A, 0x3FF2945F, 0x25378A63, 0x00000000
data4    0xB16523F6, 0x3FF200BD, 0x247BB2E0, 0x00000000
data4    0x8CE27778, 0x3FF17235, 0x24446538, 0x00000000
data4    0xFDEFE692, 0x3FF0E873, 0x2514638F, 0x00000000
data4    0x33154062, 0x3FF0632C, 0x24A7FC27, 0x00000000
data4    0xB3EF115F, 0x3FEFC42E, 0x248FD0FE, 0x00000000
data4    0x135D26F6, 0x3FEEC9E8, 0x2385C719, 0x00000000
//
//  Entries SC_inv in Swapped IEEE format (extended)
//  Index = 0,1,...,31  B = 2^(-2)*(1+Index/32+1/64)
//
data4    0x1BF30C9E, 0x839D6D4A, 0x00004001, 0x00000000
data4    0x554B0EB0, 0x80092804, 0x00004001, 0x00000000
data4    0xA1CF0DE9, 0xF959F94C, 0x00004000, 0x00000000
data4    0x77378677, 0xF3086BA0, 0x00004000, 0x00000000
data4    0xCCD4723C, 0xED154515, 0x00004000, 0x00000000
data4    0x1C27CF25, 0xE7790944, 0x00004000, 0x00000000
data4    0x8DDACB88, 0xE22D037D, 0x00004000, 0x00000000
data4    0x89C73522, 0xDD2B2D8A, 0x00004000, 0x00000000
data4    0xBB2C1171, 0xD86E1A23, 0x00004000, 0x00000000
data4    0xDFF5E0F9, 0xD3F0E288, 0x00004000, 0x00000000
data4    0x283BEBD5, 0xCFAF16B1, 0x00004000, 0x00000000
data4    0x0D88DD53, 0xCBA4AFAA, 0x00004000, 0x00000000
data4    0xCA67C43D, 0xC7CE03CC, 0x00004000, 0x00000000
data4    0x0CA0DDB0, 0xC427BC82, 0x00004000, 0x00000000
data4    0xF13D8CAB, 0xC0AECD57, 0x00004000, 0x00000000
data4    0x71ECE6B1, 0xBD606C38, 0x00004000, 0x00000000
data4    0xA44C4929, 0xBA3A0A96, 0x00004000, 0x00000000
data4    0xE5CCCEC1, 0xB7394F6F, 0x00004000, 0x00000000
data4    0x9637D8BC, 0xB45C1203, 0x00004000, 0x00000000
data4    0x92CB051B, 0xB1A05528, 0x00004000, 0x00000000
data4    0x6BA2FFD0, 0xAF04432B, 0x00004000, 0x00000000
data4    0x7221235F, 0xAC862A23, 0x00004000, 0x00000000
data4    0x5F00A9D1, 0xAA2478AF, 0x00004000, 0x00000000
data4    0x81E082BF, 0xA7DDBB0C, 0x00004000, 0x00000000
data4    0x45684FEE, 0xA5B0987D, 0x00004000, 0x00000000
data4    0x627A8F53, 0xA39BD0F5, 0x00004000, 0x00000000
data4    0x6EC5C8B0, 0xA19E3B03, 0x00004000, 0x00000000
data4    0x91CD7C66, 0x9FB6C1F0, 0x00004000, 0x00000000
data4    0x1FA3DF8A, 0x9DE46410, 0x00004000, 0x00000000
data4    0xA8F6B888, 0x9C263139, 0x00004000, 0x00000000
data4    0xC27B0450, 0x9A7B4968, 0x00004000, 0x00000000
data4    0x5EE614EE, 0x98E2DB7E, 0x00004000, 0x00000000
//
//  Entries SC_inv in Swapped IEEE format (extended)
//  Index = 0,1,...,19  B = 2^(-1)*(1+Index/32+1/64)
//
data4    0x13B2B5BA, 0x969F335C, 0x00004000, 0x00000000
data4    0xD4C0F548, 0x93D446D9, 0x00004000, 0x00000000
data4    0x61B798AF, 0x9147094F, 0x00004000, 0x00000000
data4    0x758787AC, 0x8EF317CC, 0x00004000, 0x00000000
data4    0xB99EEFDB, 0x8CD498B3, 0x00004000, 0x00000000
data4    0xDFF8BC37, 0x8AE82A7D, 0x00004000, 0x00000000
data4    0xE3C55D42, 0x892AD546, 0x00004000, 0x00000000
data4    0xD15573C1, 0x8799FEA9, 0x00004000, 0x00000000
data4    0x435A4B4C, 0x86335F88, 0x00004000, 0x00000000
data4    0x3E93A87B, 0x84F4FB6E, 0x00004000, 0x00000000
data4    0x80A382FB, 0x83DD1952, 0x00004000, 0x00000000
data4    0xA4CB8C9E, 0x82EA3D7F, 0x00004000, 0x00000000
data4    0x6861D0A8, 0x821B247C, 0x00004000, 0x00000000
data4    0x63E8D244, 0x816EBED1, 0x00004000, 0x00000000
data4    0x27E4CFC6, 0x80E42D91, 0x00004000, 0x00000000
data4    0x28E64AFD, 0x807ABF8D, 0x00004000, 0x00000000
data4    0x863B4FD8, 0x8031EF26, 0x00004000, 0x00000000
data4    0xAE8C11FD, 0x800960AD, 0x00004000, 0x00000000
data4    0x5FDBEC21, 0x8000E147, 0x00004000, 0x00000000
data4    0xA07791FA, 0x80186650, 0x00004000, 0x00000000

Arg                 = f8
Result              = f8
fp_tmp              = f9
U_2                 = f10
rsq                =  f11
C_hi                = f12
C_lo                = f13
T_hi                = f14
T_lo                = f15

N_0                 = f32
d_1                 = f33
MPI_BY_4            = f34
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

TWO_TO_63           = f54
NEGTWO_TO_63        = f55
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
PI_BY_4             = f71
NEGTWO_TO_NEG2      = f72
TWO_TO_24           = f73
TWO_TO_NEG14        = f74
TWO_TO_NEG33        = f75
NEGTWO_TO_24        = f76
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
c                   = f86
r                   = f87
Z                   = f88
A                   = f89
a                   = f90
t                   = f91
U_1                 = f92
d_2                 = f93
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

GR_SAVE_B0     = r33
GR_SAVE_GP     = r34
GR_SAVE_PFS    = r35

delta1         = r36
table_ptr1     = r37
table_ptr2     = r38
i_0            = r39
i_1            = r40
N_fix_gr       = r41
N_inc          = r42
exp_Arg        = r43
exp_r          = r44
sig_r          = r45
lookup         = r46
table_offset   = r47
Create_B       = r48
gr_tmp         = r49

GR_Parameter_X = r49
GR_Parameter_r = r50



.global __libm_tan
.section .text
.proc __libm_tan


__libm_tan:

{ .mfi
alloc r32 = ar.pfs, 0,17,2,0
(p0)   fclass.m.unc  p6,p0 = Arg, 0x1E7
      addl gr_tmp = -1,r0
}
;;

{ .mfi
       nop.m 999
(p0)   fclass.nm.unc  p7,p0 = Arg, 0x1FF
       nop.i 999
}
;;

{ .mfi
(p0)  addl           table_ptr1   = @ltoff(TAN_BASE_CONSTANTS), gp
       nop.f 999
       nop.i 999
}
;;

{ .mmi
      ld8 table_ptr1 = [table_ptr1]
      setf.sig fp_tmp = gr_tmp   // Make a constant so fmpy produces inexact
      nop.i 999
}
;;

//
//     Check for NatVals, Infs , NaNs, and Zeros
//     Check for everything - if false, then must be pseudo-zero
//     or pseudo-nan.
//     Local table pointer
//

{ .mbb
(p0)   add table_ptr2 = 96, table_ptr1
(p6)   br.cond.spnt __libm_TAN_SPECIAL
(p7)   br.cond.spnt __libm_TAN_SPECIAL ;;
}
//
//     Point to Inv_P_0
//     Branch out to deal with unsupporteds and special values.
//

{ .mmf
(p0)   ldfs TWO_TO_24 = [table_ptr1],4
(p0)   ldfs TWO_TO_63 = [table_ptr2],4
//
//     Load -2**24, load -2**63.
//
(p0)   fcmp.eq.s0 p0, p6 = Arg, f1 ;;
}

{ .mfi
(p0)   ldfs NEGTWO_TO_63 = [table_ptr2],12
(p0)   fnorm.s1     Arg = Arg
	nop.i 999
}
//
//     Load 2**24, Load 2**63.
//

{ .mmi
(p0)   ldfs NEGTWO_TO_24 = [table_ptr1],12 ;;
//
//     Do fcmp to generate Denormal exception
//     - can't do FNORM (will generate Underflow when U is unmasked!)
//     Normalize input argument.
//
(p0)   ldfe two_by_PI = [table_ptr1],16
	nop.i 999
}

{ .mmi
(p0)   ldfe Inv_P_0 = [table_ptr2],16 ;;
(p0)   ldfe d_1 = [table_ptr2],16
	nop.i 999
}
//
//     Decide about the paths to take:
//     PR_1 and PR_3 set if -2**24 < Arg < 2**24 - CASE 1 OR 2
//     OTHERWISE - CASE 3 OR 4
//     Load inverse of P_0 .
//     Set PR_6 if Arg <= -2**63
//     Are there any Infs, NaNs, or zeros?
//

{ .mmi
(p0)   ldfe P_0 = [table_ptr1],16 ;;
(p0)   ldfe d_2 = [table_ptr2],16
	nop.i 999
}
//
//     Set PR_8 if Arg <= -2**24
//     Set PR_6 if Arg >=  2**63
//

{ .mmi
(p0)   ldfe P_1 = [table_ptr1],16 ;;
(p0)   ldfe PI_BY_4 = [table_ptr2],16
	nop.i 999
}
//
//     Set PR_8 if Arg >= 2**24
//

{ .mmi
(p0)   ldfe P_2 = [table_ptr1],16 ;;
(p0)   ldfe   MPI_BY_4 = [table_ptr2],16
	nop.i 999
}
//
//     Load  P_2 and PI_BY_4
//

{ .mfi
(p0)   ldfe   P_3 = [table_ptr1],16
	nop.f 999
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p0)   fcmp.le.unc.s1 p6,p7 = Arg,NEGTWO_TO_63
	nop.i 999
}

{ .mfi
	nop.m 999
(p0)   fcmp.le.unc.s1 p8,p9 = Arg,NEGTWO_TO_24
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p7)   fcmp.ge.s1 p6,p0 = Arg,TWO_TO_63
	nop.i 999
}

{ .mfi
	nop.m 999
(p9)   fcmp.ge.s1 p8,p0 = Arg,TWO_TO_24
	nop.i 999 ;;
}

{ .mib
	nop.m 999
	nop.i 999
//
//     Load  P_3 and -PI_BY_4
//
(p6)   br.cond.spnt TAN_ARG_TOO_LARGE ;;
}

{ .mib
	nop.m 999
	nop.i 999
//
//     Load 2**(-2).
//     Load -2**(-2).
//     Branch out if we have a special argument.
//     Branch out if the magnitude of the input argument is too large
//     - do this branch before the next.
//
(p8)   br.cond.spnt TAN_LARGER_ARG ;;
}
//
//     Branch to Cases 3 or 4 if Arg <= -2**24 or Arg >= 2**24
//

{ .mfi
(p0)   ldfs TWO_TO_NEG2 = [table_ptr2],4
//     ARGUMENT REDUCTION CODE - CASE 1 and 2
//     Load 2**(-2).
//     Load -2**(-2).
(p0)   fmpy.s1 N = Arg,two_by_PI
	nop.i 999 ;;
}

{ .mfi
(p0)   ldfs NEGTWO_TO_NEG2 = [table_ptr2],12
//
//     N = Arg * 2/pi
//
(p0)   fcmp.lt.unc.s1 p8,p9= Arg,PI_BY_4
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     if Arg < pi/4,  set PR_8.
//
(p8)   fcmp.gt.s1 p8,p9= Arg,MPI_BY_4
	nop.i 999 ;;
}
//
//     Case 1: Is |r| < 2**(-2).
//     Arg is the same as r in this case.
//     r = Arg
//     c = 0
//

{ .mfi
(p8)   mov N_fix_gr = r0
//
//     if Arg > -pi/4, reset PR_8.
//     Select the case when |Arg| < pi/4 - set PR[8] = true.
//     Else Select the case when |Arg| >= pi/4 - set PR[9] = true.
//
(p0)   fcvt.fx.s1 N_fix = N
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Grab the integer part of N .
//
(p8)   mov r = Arg
	nop.i 999
}

{ .mfi
	nop.m 999
(p8)   mov c = f0
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p8)   fcmp.lt.unc.s1 p10, p11 = Arg, TWO_TO_NEG2
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p10)  fcmp.gt.s1 p10,p0 = Arg, NEGTWO_TO_NEG2
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Case 2: Place integer part of N in GP register.
//
(p9)   fcvt.xf N = N_fix
	nop.i 999 ;;
}

{ .mib
(p9)   getf.sig N_fix_gr = N_fix
	nop.i 999
//
//     Case 2: Convert integer N_fix back to normalized floating-point value.
//
(p10)  br.cond.spnt TAN_SMALL_R ;;
}

{ .mib
	nop.m 999
	nop.i 999
(p8)   br.cond.sptk TAN_NORMAL_R ;;
}
//
//     Case 1: PR_3 is only affected  when PR_1 is set.
//

{ .mmi
(p9)   ldfs TWO_TO_NEG33 = [table_ptr2], 4 ;;
//
//     Case 2: Load 2**(-33).
//
(p9)   ldfs NEGTWO_TO_NEG33 = [table_ptr2], 4
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Case 2: Load -2**(-33).
//
(p9)   fnma.s1 s_val = N, P_1, Arg
	nop.i 999
}

{ .mfi
	nop.m 999
(p9)   fmpy.s1 w = N, P_2
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Case 2: w = N * P_2
//     Case 2: s_val = -N * P_1  + Arg
//
(p0)   fcmp.lt.unc.s1 p9,p8 = s_val, TWO_TO_NEG33
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Decide between case_1 and case_2 reduce:
//
(p9)   fcmp.gt.s1 p9, p8 = s_val, NEGTWO_TO_NEG33
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Case 1_reduce:  s <= -2**(-33) or s >= 2**(-33)
//     Case 2_reduce: -2**(-33) < s < 2**(-33)
//
(p8)   fsub.s1 r = s_val, w
	nop.i 999
}

{ .mfi
	nop.m 999
(p9)   fmpy.s1 w = N, P_3
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p9)   fma.s1  U_1 = N, P_2, w
	nop.i 999
}

{ .mfi
	nop.m 999
//
//     Case 1_reduce: Is |r| < 2**(-2), if so set PR_10
//     else set PR_11.
//
(p8)   fsub.s1 c = s_val, r
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Case 1_reduce: r = s + w (change sign)
//     Case 2_reduce: w = N * P_3 (change sign)
//
(p8)   fcmp.lt.unc.s1 p10, p11 = r, TWO_TO_NEG2
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p10)  fcmp.gt.s1 p10, p11 = r, NEGTWO_TO_NEG2
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p9)   fsub.s1 r = s_val, U_1
	nop.i 999
}

{ .mfi
	nop.m 999
//
//     Case 1_reduce: c is complete here.
//     c = c + w (w has not been negated.)
//     Case 2_reduce: r is complete here - continue to calculate c .
//     r = s - U_1
//
(p9)   fms.s1 U_2 = N, P_2, U_1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     Case 1_reduce: c = s - r
//     Case 2_reduce: U_1 = N * P_2 + w
//
(p8)   fsub.s1 c = c, w
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p9)   fsub.s1 s_val = s_val, r
	nop.i 999
}

{ .mfb
	nop.m 999
//
//     Case 2_reduce:
//     U_2 = N * P_2 - U_1
//     Not needed until later.
//
(p9)   fadd.s1 U_2 = U_2, w
//
//     Case 2_reduce:
//     s = s - r
//     U_2 = U_2 + w
//
(p10)  br.cond.spnt TAN_SMALL_R ;;
}

{ .mib
	nop.m 999
	nop.i 999
(p11)  br.cond.sptk TAN_NORMAL_R ;;
}

{ .mii
	nop.m 999
//
//     Case 2_reduce:
//     c = c - U_2
//     c is complete here
//     Argument reduction ends here.
//
(p9)   extr.u i_1 = N_fix_gr, 0, 1 ;;
(p9)   cmp.eq.unc p11, p12 = 0x0000,i_1 ;;
}

{ .mfi
	nop.m 999
//
//     Is i_1  even or odd?
//     if i_1 == 0, set p11, else set p12.
//
(p11)  fmpy.s1 rsq = r, Z
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12)  frcpa.s1 S_hi,p0 = f1, r
	nop.i 999
}

//
//     Case 1: Branch to SMALL_R or NORMAL_R.
//     Case 1 is done now.
//

{ .mfi
(p9)  addl           table_ptr1   = @ltoff(TAN_BASE_CONSTANTS), gp
(p9)   fsub.s1 c = s_val, U_1
	nop.i 999 ;;
}
;;

{ .mmi
(p9)  ld8 table_ptr1 = [table_ptr1]
      nop.m 999
      nop.i 999
}
;;

{ .mmi
(p9)   add table_ptr1 = 224, table_ptr1 ;;
(p9)   ldfe P1_1 = [table_ptr1],144
	nop.i 999 ;;
}
//
//     Get [i_1] -  lsb of N_fix_gr .
//     Load P1_1 and point to Q1_1 .
//

{ .mfi
(p9)   ldfe Q1_1 = [table_ptr1] , 0
//
//     N even: rsq = r * Z
//     N odd:  S_hi = frcpa(r)
//
(p12)  fmerge.ns S_hi = S_hi, S_hi
	nop.i 999
}

{ .mfi
	nop.m 999
//
//     Case 2_reduce:
//     c = s - U_1
//
(p9)   fsub.s1 c = c, U_2
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12)  fma.s1  poly1 = S_hi, r, f1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N odd:  Change sign of S_hi
//
(p11)  fmpy.s1 rsq = rsq, P1_1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12)  fma.s1 S_hi = S_hi, poly1, S_hi
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N even: rsq = rsq * P1_1
//     N odd:  poly1 =  1.0 +  S_hi * r    16 bits partial  account for necessary
//
(p11)  fma.s1 Result = r, rsq, c
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N even: Result = c  + r * rsq
//     N odd:  S_hi  = S_hi + S_hi*poly1  16 bits account for necessary
//
(p12)  fma.s1 poly1 = S_hi, r, f1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N even: Result = Result + r
//     N odd:  poly1  = 1.0 + S_hi * r        32 bits partial
//
(p11)  fadd.s0 Result = r, Result
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12)  fma.s1  S_hi = S_hi, poly1, S_hi
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N even: Result1 = Result + r
//     N odd:   S_hi  = S_hi * poly1 + S_hi   32 bits
//
(p12)  fma.s1 poly1 = S_hi, r, f1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N odd:  poly1  =  S_hi * r + 1.0       64 bits partial
//
(p12)  fma.s1 S_hi = S_hi, poly1, S_hi
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N odd:  poly1  =  S_hi * poly + 1.0    64 bits
//
(p12)  fma.s1 poly1 = S_hi, r, f1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N odd:  poly1  =  S_hi * r + 1.0
//
(p12)  fma.s1 poly1 = S_hi, c, poly1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N odd:  poly1  =  S_hi * c + poly1
//
(p12)  fmpy.s1 S_lo = S_hi, poly1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//     N odd:  S_lo  =  S_hi *  poly1
//
(p12)  fma.s1 S_lo = Q1_1, r, S_lo
	nop.i 999
}

{ .mfi
	nop.m 999
//
//     N odd:  Result =  S_hi + S_lo
//
(p0)   fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
	nop.i 999 ;;
}

{ .mfb
	nop.m 999
//
//     N odd:  S_lo  =  S_lo + Q1_1 * r
//
(p12)  fadd.s0 Result = S_hi, S_lo
(p0)   br.ret.sptk b0 ;;
}


TAN_LARGER_ARG:

{ .mmf
(p0)  addl           table_ptr1   = @ltoff(TAN_BASE_CONSTANTS), gp
      nop.m 999
(p0)  fmpy.s1 N_0 = Arg, Inv_P_0
}
;;

//
// ARGUMENT REDUCTION CODE - CASE 3 and 4
//
//
//    Adjust table_ptr1 to beginning of table.
//    N_0 = Arg * Inv_P_0
//


{ .mmi
(p0)  ld8 table_ptr1 = [table_ptr1]
      nop.m 999
      nop.i 999
}
;;


{ .mmi
(p0)  add table_ptr1 = 8, table_ptr1 ;;
//
//    Point to  2*-14
//
(p0)  ldfs TWO_TO_NEG14 = [table_ptr1], 4
	nop.i 999 ;;
}
//
//    Load 2**(-14).
//

{ .mmi
(p0)  ldfs NEGTWO_TO_NEG14 = [table_ptr1], 180 ;;
//
//    N_0_fix  = integer part of N_0 .
//    Adjust table_ptr1 to beginning of table.
//
(p0)  ldfs TWO_TO_NEG2 = [table_ptr1], 4
	nop.i 999 ;;
}
//
//    Make N_0 the integer part.
//

{ .mfi
(p0)  ldfs NEGTWO_TO_NEG2 = [table_ptr1]
//
//    Load -2**(-14).
//
(p0)  fcvt.fx.s1 N_0_fix = N_0
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p0)  fcvt.xf N_0 = N_0_fix
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p0)  fnma.s1 ArgPrime = N_0, P_0, Arg
	nop.i 999
}

{ .mfi
	nop.m 999
(p0)  fmpy.s1 w = N_0, d_1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    ArgPrime = -N_0 * P_0 + Arg
//    w  = N_0 * d_1
//
(p0)  fmpy.s1 N = ArgPrime, two_by_PI
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N = ArgPrime * 2/pi
//
(p0)  fcvt.fx.s1 N_fix = N
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N_fix is the integer part.
//
(p0)  fcvt.xf N = N_fix
	nop.i 999 ;;
}

{ .mfi
(p0)  getf.sig N_fix_gr = N_fix
	nop.f 999
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N is the integer part of the reduced-reduced argument.
//    Put the integer in a GP register.
//
(p0)  fnma.s1 s_val = N, P_1, ArgPrime
	nop.i 999
}

{ .mfi
	nop.m 999
(p0)  fnma.s1 w = N, P_2, w
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    s_val = -N*P_1 + ArgPrime
//    w = -N*P_2 + w
//
(p0)  fcmp.lt.unc.s1 p11, p10 = s_val, TWO_TO_NEG14
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p11) fcmp.gt.s1 p11, p10 = s_val, NEGTWO_TO_NEG14
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 3: r = s_val + w (Z complete)
//    Case 4: U_hi = N_0 * d_1
//
(p10) fmpy.s1 V_hi = N, P_2
	nop.i 999
}

{ .mfi
	nop.m 999
(p11) fmpy.s1 U_hi = N_0, d_1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 3: r = s_val + w (Z complete)
//    Case 4: U_hi = N_0 * d_1
//
(p11) fmpy.s1 V_hi = N, P_2
	nop.i 999
}

{ .mfi
	nop.m 999
(p11) fmpy.s1 U_hi = N_0, d_1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Decide between case 3 and 4:
//    Case 3:  s <= -2**(-14) or s >= 2**(-14)
//    Case 4: -2**(-14) < s < 2**(-14)
//
(p10) fadd.s1 r = s_val, w
	nop.i 999
}

{ .mfi
	nop.m 999
(p11) fmpy.s1 w = N, P_3
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 4: We need abs of both U_hi and V_hi - dont
//    worry about switched sign of V_hi .
//
(p11) fsub.s1 A = U_hi, V_hi
	nop.i 999
}

{ .mfi
	nop.m 999
//
//    Case 4: A =  U_hi + V_hi
//    Note: Worry about switched sign of V_hi, so subtract instead of add.
//
(p11) fnma.s1 V_lo = N, P_2, V_hi
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p11) fms.s1 U_lo = N_0, d_1, U_hi
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p11) fabs V_hiabs = V_hi
	nop.i 999
}

{ .mfi
	nop.m 999
//
//    Case 4: V_hi = N * P_2
//            w = N * P_3
//    Note the product does not include the (-) as in the writeup
//    so (-) missing for V_hi and w .
(p10) fadd.s1 r = s_val, w
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 3: c = s_val - r
//    Case 4: U_lo = N_0 * d_1 - U_hi
//
(p11) fabs U_hiabs = U_hi
	nop.i 999
}

{ .mfi
	nop.m 999
(p11) fmpy.s1 w = N, P_3
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 4: Set P_12 if U_hiabs >= V_hiabs
//
(p11) fadd.s1 C_hi = s_val, A
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 4: C_hi = s_val + A
//
(p11) fadd.s1 t = U_lo, V_lo
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 3: Is |r| < 2**(-2), if so set PR_7
//    else set PR_8.
//    Case 3: If PR_7 is set, prepare to branch to Small_R.
//    Case 3: If PR_8 is set, prepare to branch to Normal_R.
//
(p10) fsub.s1 c = s_val, r
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 3: c = (s - r) + w (c complete)
//
(p11) fcmp.ge.unc.s1 p12, p13 = U_hiabs, V_hiabs
	nop.i 999
}

{ .mfi
	nop.m 999
(p11) fms.s1 w = N_0, d_2, w
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 4: V_hi = N * P_2
//            w = N * P_3
//    Note the product does not include the (-) as in the writeup
//    so (-) missing for V_hi and w .
//
(p10) fcmp.lt.unc.s1 p14, p15 = r, TWO_TO_NEG2
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p14) fcmp.gt.s1 p14, p15 = r, NEGTWO_TO_NEG2
	nop.i 999 ;;
}

{ .mfb
	nop.m 999
//
//    Case 4: V_lo = -N * P_2 - V_hi (U_hi is in place of V_hi in writeup)
//    Note: the (-) is still missing for V_hi .
//    Case 4: w = w + N_0 * d_2
//    Note: the (-) is now incorporated in w .
//
(p10) fadd.s1 c = c, w
//
//    Case 4: t = U_lo + V_lo
//    Note: remember V_lo should be (-), subtract instead of add. NO
//
(p14) br.cond.spnt TAN_SMALL_R ;;
}

{ .mib
	nop.m 999
	nop.i 999
(p15) br.cond.spnt TAN_NORMAL_R ;;
}

{ .mfi
	nop.m 999
//
//    Case 3: Vector off when |r| < 2**(-2).  Recall that PR_3 will be true.
//    The remaining stuff is for Case 4.
//
(p12) fsub.s1 a = U_hi, A
(p11) extr.u i_1 = N_fix_gr, 0, 1 ;;
}

{ .mfi
	nop.m 999
//
//    Case 4: C_lo = s_val - C_hi
//
(p11) fadd.s1 t = t, w
	nop.i 999
}

{ .mfi
	nop.m 999
(p13) fadd.s1 a = V_hi, A
	nop.i 999 ;;
}

//
//    Case 4: a = U_hi - A
//            a = V_hi - A (do an add to account for missing (-) on V_hi
//

{ .mfi
(p11)  addl           table_ptr1   = @ltoff(TAN_BASE_CONSTANTS), gp
(p11) fsub.s1 C_lo = s_val, C_hi
	nop.i 999
}
;;

{ .mmi
(p11) ld8 table_ptr1 = [table_ptr1]
      nop.m 999
      nop.i 999
}
;;

//
//    Case 4: a = (U_hi - A)  + V_hi
//            a = (V_hi - A)  + U_hi
//    In each case account for negative missing form V_hi .
//
//
//    Case 4: C_lo = (s_val - C_hi) + A
//

{ .mmi
(p11) add table_ptr1 = 224, table_ptr1 ;;
(p11) ldfe P1_1 = [table_ptr1], 16
	nop.i 999 ;;
}

{ .mfi
(p11) ldfe P1_2 = [table_ptr1], 128
//
//    Case 4: w = U_lo + V_lo  + w
//
(p12) fsub.s1 a = a, V_hi
	nop.i 999 ;;
}
//
//    Case 4: r = C_hi + C_lo
//

{ .mfi
(p11) ldfe Q1_1 = [table_ptr1], 16
(p11) fadd.s1 C_lo = C_lo, A
	nop.i 999 ;;
}
//
//    Case 4: c = C_hi - r
//    Get [i_1] - lsb of N_fix_gr.
//

{ .mfi
(p11) ldfe Q1_2 = [table_ptr1], 16
	nop.f 999
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p13) fsub.s1 a = U_hi, a
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p11) fadd.s1 t = t, a
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 4: t = t + a
//
(p11) fadd.s1 C_lo = C_lo, t
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    Case 4: C_lo = C_lo + t
//
(p11) fadd.s1 r = C_hi, C_lo
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p11) fsub.s1 c = C_hi, r
	nop.i 999
}

{ .mfi
	nop.m 999
//
//    Case 4: c = c + C_lo  finished.
//    Is i_1  even or odd?
//    if i_1 == 0, set PR_4, else set PR_5.
//
// r and c have been computed.
// We known whether this is the sine or cosine routine.
// Make sure ftz mode is set - should be automatic when using wre
(p0)  fmpy.s1 rsq = r, r
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p11) fadd.s1 c = c , C_lo
(p11) cmp.eq.unc p11, p12 =  0x0000, i_1 ;;
}

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
(p11) fma.s1 Result = rsq, P1_2, P1_1
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
(p0)   fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
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
(p11) fmpy.s1 Result = rsq, Result
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12) fma.s1 poly1 = S_hi, r,f1
	nop.i 999
}

{ .mfi
	nop.m 999
//
//    N even: Result =  Result * rsq
//    N odd:  S_hi  = S_hi + S_hi*poly1  16 bits account for necessary
//
(p11) fma.s1 Result = r, Result, c
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
(p11) fadd.s0 Result= r, Result
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12) fma.s1 poly1 =  S_hi, r, f1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N even: Result = Result * r + c
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
//    N even: Result1 = Result + r  (Rounding mode S0)
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
	nop.i 999 ;;
}

{ .mfb
	nop.m 999
//
//    N odd:  S_lo  =  S_lo + r * P
//
(p12) fadd.s0 Result = S_hi, S_lo
(p0)   br.ret.sptk b0 ;;
}


TAN_SMALL_R:

{ .mii
	nop.m 999
(p0)  extr.u i_1 = N_fix_gr, 0, 1 ;;
(p0)  cmp.eq.unc p11, p12 = 0x0000, i_1
}

{ .mfi
	nop.m 999
(p0)  fmpy.s1 rsq = r, r
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12) frcpa.s1 S_hi, p0 = f1, r
	nop.i 999
}

{ .mfi
(p0)  addl           table_ptr1   = @ltoff(TAN_BASE_CONSTANTS), gp
        nop.f 999
        nop.i 999
}
;;

{ .mmi
(p0)  ld8 table_ptr1 = [table_ptr1]
      nop.m 999
      nop.i 999
}
;;

// *****************************************************************
// *****************************************************************
// *****************************************************************

{ .mmi
(p0)  add table_ptr1 = 224, table_ptr1 ;;
(p0)  ldfe P1_1 = [table_ptr1], 16
	nop.i 999 ;;
}
//    r and c have been computed.
//    We known whether this is the sine or cosine routine.
//    Make sure ftz mode is set - should be automatic when using wre
//    |r| < 2**(-2)

{ .mfi
(p0)  ldfe P1_2 = [table_ptr1], 16
(p11) fmpy.s1 r_to_the_8 = rsq, rsq
	nop.i 999 ;;
}
//
//    Set table_ptr1 to beginning of constant table.
//    Get [i_1] - lsb of N_fix_gr.
//

{ .mfi
(p0)  ldfe P1_3 = [table_ptr1], 96
//
//    N even: rsq = r * r
//    N odd:  S_hi = frcpa(r)
//
(p12) fmerge.ns S_hi = S_hi, S_hi
	nop.i 999 ;;
}
//
//    Is i_1  even or odd?
//    if i_1 == 0, set PR_11.
//    if i_1 != 0, set PR_12.
//

{ .mfi
(p11) ldfe P1_9 = [table_ptr1], -16
//
//    N even: Poly2 = P1_7 + Poly2 * rsq
//    N odd:  poly2 = Q1_5 + poly2 * rsq
//
(p11) fadd.s1 CORR = rsq, f1
	nop.i 999 ;;
}

{ .mmi
(p11) ldfe P1_8 = [table_ptr1], -16 ;;
//
//    N even: Poly1 = P1_2 + P1_3 * rsq
//    N odd:  poly1 =  1.0 +  S_hi * r
//    16 bits partial  account for necessary (-1)
//
(p11) ldfe P1_7 = [table_ptr1], -16
	nop.i 999 ;;
}
//
//    N even: Poly1 = P1_1 + Poly1 * rsq
//    N odd:  S_hi  =  S_hi + S_hi * poly1)     16 bits account for necessary
//

{ .mfi
(p11) ldfe P1_6 = [table_ptr1], -16
//
//    N even: Poly2 = P1_5 + Poly2 * rsq
//    N odd:  poly2 = Q1_3 + poly2 * rsq
//
(p11) fmpy.s1 r_to_the_8 = r_to_the_8, r_to_the_8
	nop.i 999 ;;
}
//
//    N even: Poly1 =  Poly1 * rsq
//    N odd:  poly1  = 1.0 + S_hi * r         32 bits partial
//

{ .mfi
(p11) ldfe P1_5 = [table_ptr1], -16
(p12) fma.s1 poly1 =  S_hi, r, f1
	nop.i 999 ;;
}
//
//    N even: CORR =  CORR * c
//    N odd:  S_hi  =  S_hi * poly1 + S_hi    32 bits
//

//
//    N even: Poly2 = P1_6 + Poly2 * rsq
//    N odd:  poly2 = Q1_4 + poly2 * rsq
//
{ .mmf
(p0)  addl           table_ptr2   = @ltoff(TAN_BASE_CONSTANTS), gp
(p11) ldfe P1_4 = [table_ptr1], -16
(p11) fmpy.s1 CORR =  CORR, c
}
;;


{ .mmi
(p0)  ld8 table_ptr2 = [table_ptr2]
      nop.m 999
      nop.i 999
}
;;


{ .mii
(p0)  add table_ptr2 = 464, table_ptr2
	nop.i 999 ;;
	nop.i 999
}

{ .mfi
	nop.m 999
(p11) fma.s1 Poly1 = P1_3, rsq, P1_2
	nop.i 999 ;;
}

{ .mfi
(p0)  ldfe Q1_7 = [table_ptr2], -16
(p12) fma.s1 S_hi = S_hi, poly1, S_hi
	nop.i 999 ;;
}

{ .mfi
(p0)  ldfe Q1_6 = [table_ptr2], -16
(p11) fma.s1 Poly2 = P1_9, rsq, P1_8
	nop.i 999 ;;
}

{ .mmi
(p0)  ldfe Q1_5 = [table_ptr2], -16 ;;
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
//    N even: Result = CORR + Poly * r
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
(p11) fma.s1 Result = Poly, r, CORR
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N even: Result =  r + Result  (User supplied rounding mode)
//    N odd:  poly1  =  S_hi * c + poly1
//
(p12) fmpy.s1 S_lo = S_hi, poly1
	nop.i 999
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
(p11) fadd.s0 Result = Result, r
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N odd:  S_lo  =  S_hi *  poly1
//
(p12) fma.s1 S_lo = Q1_1, c, S_lo
	nop.i 999
}

{ .mfi
	nop.m 999
//
//    N odd:  Result = Result + S_hi  (user supplied rounding mode)
//
(p0)   fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N odd:  S_lo  =  Q1_1 * c + S_lo
//
(p12) fma.s1 Result = P, r, S_lo
	nop.i 999 ;;
}

{ .mfb
	nop.m 999
//
//    N odd:  Result =  S_lo + r * P
//
(p12) fadd.s0 Result = Result, S_hi
(p0)   br.ret.sptk b0 ;;
}


TAN_NORMAL_R:

{ .mfi
(p0)  getf.sig sig_r = r
// *******************************************************************
// *******************************************************************
// *******************************************************************
//
//    r and c have been computed.
//    Make sure ftz mode is set - should be automatic when using wre
//
//
//    Get [i_1] -  lsb of N_fix_gr alone.
//
(p0)  fmerge.s  Pos_r = f1, r
(p0)  extr.u i_1 = N_fix_gr, 0, 1 ;;
}

{ .mfi
	nop.m 999
(p0)  fmerge.s  sgn_r =  r, f1
(p0)  cmp.eq.unc p11, p12 = 0x0000, i_1 ;;
}

{ .mfi
	nop.m 999
	nop.f 999
(p0)  extr.u lookup = sig_r, 58, 5
}

{ .mlx
	nop.m 999
(p0)  movl Create_B = 0x8200000000000000 ;;
}

{ .mfi
(p0)  addl           table_ptr1   = @ltoff(TAN_BASE_CONSTANTS), gp
	nop.f 999
(p0)  dep Create_B = lookup, Create_B, 58, 5
}
;;

//
//    Get [i_1] -  lsb of N_fix_gr alone.
//    Pos_r = abs (r)
//


{ .mmi
      ld8 table_ptr1 = [table_ptr1]
      nop.m 999
      nop.i 999
}
;;


{ .mmi
	nop.m 999
(p0)  setf.sig B = Create_B
//
//    Set table_ptr1 and table_ptr2 to base address of
//    constant table.
//
(p0)  add table_ptr1 = 480, table_ptr1 ;;
}

{ .mmb
	nop.m 999
//
//    Is i_1 or i_0  == 0 ?
//    Create the constant  1 00000 1000000000000000000000...
//
(p0)  ldfe P2_1 = [table_ptr1], 16
	nop.b 999
}

{ .mmi
	nop.m 999 ;;
(p0)  getf.exp exp_r = Pos_r
	nop.i 999
}
//
//    Get r's exponent
//    Get r's significand
//

{ .mmi
(p0)  ldfe P2_2 = [table_ptr1], 16 ;;
//
//    Get the 5 bits or r for the lookup.   1.xxxxx ....
//    from sig_r.
//    Grab  lsb of exp of B
//
(p0)  ldfe P2_3 = [table_ptr1], 16
	nop.i 999 ;;
}

{ .mii
	nop.m 999
(p0)  andcm table_offset = 0x0001, exp_r ;;
(p0)  shl table_offset = table_offset, 9 ;;
}

{ .mii
	nop.m 999
//
//    Deposit   0 00000 1000000000000000000000... on
//              1 xxxxx yyyyyyyyyyyyyyyyyyyyyy...,
//    getting rid of the ys.
//    Is  B = 2** -2 or  B= 2** -1? If 2**-1, then
//    we want an offset of 512 for table addressing.
//
(p0)  shladd table_offset = lookup, 4, table_offset ;;
//
//    B =  ........ 1xxxxx 1000000000000000000...
//
(p0)  add table_ptr1 = table_ptr1, table_offset ;;
}

{ .mmb
	nop.m 999
//
//   B =  ........ 1xxxxx 1000000000000000000...
//   Convert B so it has the same exponent as Pos_r
//
(p0)  ldfd T_hi = [table_ptr1], 8
	nop.b 999 ;;
}

//
//    x = |r| - B
//    Load T_hi.
//    Load C_hi.
//

{ .mmf
(p0)  addl           table_ptr2   = @ltoff(TAN_BASE_CONSTANTS), gp
(p0)  ldfs T_lo = [table_ptr1]
(p0)  fmerge.se B = Pos_r, B
}
;;

{ .mmi
      ld8 table_ptr2 = [table_ptr2]
      nop.m 999
      nop.i 999
}
;;

{ .mii
(p0)  add table_ptr2 = 1360, table_ptr2
	nop.i 999 ;;
(p0)  add table_ptr2 = table_ptr2, table_offset ;;
}

{ .mfi
(p0)  ldfd C_hi = [table_ptr2], 8
(p0)  fsub.s1 x = Pos_r, B
	nop.i 999 ;;
}

{ .mii
(p0)  ldfs C_lo = [table_ptr2],255
	nop.i 999 ;;
//
//    xsq = x * x
//    N even: Tx = T_hi * x
//    Load T_lo.
//    Load C_lo - increment pointer to get SC_inv
//    - cant get all the way, do an add later.
//
(p0)  add table_ptr2 = 569, table_ptr2 ;;
}
//
//    N even: Tx1 = Tx + 1
//    N odd:  Cx1 = 1 - Cx
//

{ .mfi
(p0)  ldfe SC_inv = [table_ptr2], 0
	nop.f 999
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p0)  fmpy.s1 xsq = x, x
	nop.i 999
}

{ .mfi
	nop.m 999
(p11) fmpy.s1 Tx = T_hi, x
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p12) fmpy.s1 Cx = C_hi, x
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N odd: Cx = C_hi * x
//
(p0)  fma.s1 P = P2_3, xsq, P2_2
	nop.i 999
}

{ .mfi
	nop.m 999
//
//    N even and odd: P = P2_3 + P2_2 * xsq
//
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
(p0)  fmpy.s1 Sx = SC_inv, x
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
(p0)  fma.s1 P = P, xsq, P2_1
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
(p0)   fmpy.s0 fp_tmp = fp_tmp, fp_tmp  // Dummy mult to set inexact
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
(p0)  fmpy.s1 CORR = CORR, c
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
(p0)  fmpy.s1 P = P, xsq
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
(p0)  fmpy.s1 CORR = CORR, sgn_r
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
(p0)  fma.s1 tanx = P, x, x
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
(p0)  fnma.s1 D = V_hi, D, f1
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N even and odd: D = 1 - V_hi * D
//    N even and odd: CORR = CORR * c
//
(p0)  fma.s1 V_hi = V_hi, D, V_hi
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
	nop.i 999
}

{ .mfi
	nop.m 999
(p12) fnma.s1 V_lo = tanx, V_hi, V_lo
	nop.i 999 ;;
}

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
(p0)  fmpy.s1 V_lo = V_hi, V_lo
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N even and odd: V_lo = V_lo * V_hi
//
(p0)  fadd.s1 tail = V_hi, V_lo
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N even and odd: tail = V_hi + V_lo
//
(p0)  fma.s1 tail = tail, P, V_lo
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N even: T_hi = sgn_r * T_hi
//    N odd : C_hi = -sgn_r * C_hi
//
(p0)  fma.s1 tail = tail, Sx, CORR
	nop.i 999 ;;
}

{ .mfi
	nop.m 999
//
//    N even and odd: tail = Sx * tail + CORR
//
(p0)  fma.s1 tail = V_hi, Sx, tail
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
(p0)   br.ret.sptk b0 ;;
}

.endp __libm_tan
ASM_SIZE_DIRECTIVE(__libm_tan)



// *******************************************************************
// *******************************************************************
// *******************************************************************
//
//     Special Code to handle very large argument case.
//     Call int pi_by_2_reduce(&x,&r)
//     for |arguments| >= 2**63
//     (Arg or x) is in f8
//     Address to save r and c as double

//                 (1)                    (2)                 (3) (call)         (4)
//            sp -> +               psp -> +            psp -> +           sp ->  +
//                  |                      |                   |                  |
//                  |                r50 ->| <- r50      f0  ->|           r50 -> | -> c
//                  |                      |                   |                  |
//         sp-32 -> | <- r50          f0 ->|             f0  ->| <- r50    r49 -> | -> r
//                  |                      |                   |                  |
//                  |               r49  ->| <- r49     Arg  ->| <- r49           | -> x
//                  |                      |                   |                  |
//         sp -64 ->|             sp -64 ->|          sp -64 ->|                  |
//
//            save pfs           save b0                                     restore gp
//            save gp                                                        restore b0
//                                                                           restore pfs



.proc __libm_callout
__libm_callout:
TAN_ARG_TOO_LARGE:
.prologue
// (1)
{ .mfi
        add   GR_Parameter_r =-32,sp                        // Parameter: r address
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

// (2)
{ .mmi
        stfe [GR_Parameter_r ] = f0,16                      // Clear Parameter r on stack
        add  GR_Parameter_X = 16,sp                        // Parameter x address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;

// (3)
.body
{ .mib
        stfe [GR_Parameter_r ] = f0,-16                     // Clear Parameter c on stack
        nop.i 0
        nop.b 0
}
{ .mib
        stfe [GR_Parameter_X] = Arg                        // Store Parameter x on stack
        nop.i 0
(p0)    br.call.sptk b0=__libm_pi_by_2_reduce#
}
;;


// (4)
{ .mmi
        mov   gp = GR_SAVE_GP                  // Restore gp
(p0)    mov   N_fix_gr = r8
        nop.i 999
}
;;

{ .mmi
(p0)    ldfe  Arg        =[GR_Parameter_X],16
(p0)    ldfs  TWO_TO_NEG2 = [table_ptr2],4
        nop.i 999
}
;;


{ .mmb
(p0)    ldfe  r =[GR_Parameter_r ],16
(p0)    ldfs  NEGTWO_TO_NEG2 = [table_ptr2],4
        nop.b 999 ;;
}

{ .mfi
(p0)    ldfe  c =[GR_Parameter_r ]
        nop.f 999
        nop.i 999 ;;
}

{ .mfi
        nop.m 999
//
//     Is |r| < 2**(-2)
//
(p0)   fcmp.lt.unc.s1  p6, p0 = r, TWO_TO_NEG2
        mov   b0 = GR_SAVE_B0                  // Restore return address
}
;;

{ .mfi
       nop.m 999
(p6)   fcmp.gt.unc.s1  p6, p0 = r, NEGTWO_TO_NEG2
       mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
}
;;

{ .mbb
.restore sp
        add   sp = 64,sp                       // Restore stack pointer
(p6)   br.cond.spnt TAN_SMALL_R
(p0)   br.cond.sptk TAN_NORMAL_R
}
;;
.endp __libm_callout
ASM_SIZE_DIRECTIVE(__libm_callout)


.proc __libm_TAN_SPECIAL
__libm_TAN_SPECIAL:

//
//     Code for NaNs, Unsupporteds, Infs, or +/- zero ?
//     Invalid raised for Infs and SNaNs.
//

{ .mfb
	nop.m 999
(p0)   fmpy.s0 Arg = Arg, f0
(p0)   br.ret.sptk b0
}
.endp __libm_TAN_SPECIAL
ASM_SIZE_DIRECTIVE(__libm_TAN_SPECIAL)


.type __libm_pi_by_2_reduce#,@function
.global __libm_pi_by_2_reduce#
