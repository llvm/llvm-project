.file "libm_sincos_large.s"


// Copyright (c) 2002 - 2003, Intel Corporation
// All rights reserved.
//
// Contributed 2002 by the Intel Numerics Group, Intel Corporation
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
// 02/15/02 Initial version
// 05/13/02 Changed interface to __libm_pi_by_2_reduce
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double table values
// 05/15/03 Reformatted data tables
//
//
// Overview of operation
//==============================================================
//
// These functions calculate the sin and cos for inputs
// greater than 2^10
//
// __libm_sin_large#
// __libm_cos_large#
// They accept argument in f8
// and return result in f8 without final rounding
//
// __libm_sincos_large#
// It accepts argument in f8
// and returns cos in f8 and sin in f9 without final rounding
//
//
//*********************************************************************
//
// Accuracy:       Within .7 ulps for 80-bit floating point values
//                 Very accurate for double precision values
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8 as Input Value, f8 and f9 as Return Values
//                              f32-f103
//
//    General Purpose Registers:
//      r32-r43
//      r44-r45 (Used to pass arguments to pi_by_2 reduce routine)
//
//    Predicate Registers:      p6-p13
//
//*********************************************************************
//
//  IEEE Special Conditions:
//
//    Denormal  fault raised on denormal inputs
//    Overflow exceptions do not occur
//    Underflow exceptions raised when appropriate for sin
//    (No specialized error handling for this routine)
//    Inexact raised when appropriate by algorithm
//
//    sin(SNaN) = QNaN
//    sin(QNaN) = QNaN
//    sin(inf) = QNaN
//    sin(+/-0) = +/-0
//    cos(inf) = QNaN
//    cos(SNaN) = QNaN
//    cos(QNaN) = QNaN
//    cos(0) = 1
//
//*********************************************************************
//
//  Mathematical Description
//  ========================
//
//  The computation of FSIN and FCOS is best handled in one piece of
//  code. The main reason is that given any argument Arg, computation
//  of trigonometric functions first calculate N and an approximation
//  to alpha where
//
//  Arg = N pi/2 + alpha, |alpha| <= pi/4.
//
//  Since
//
//  cos( Arg ) = sin( (N+1) pi/2 + alpha ),
//
//  therefore, the code for computing sine will produce cosine as long
//  as 1 is added to N immediately after the argument reduction
//  process.
//
//  Let M = N if sine
//      N+1 if cosine.
//
//  Now, given
//
//  Arg = M pi/2  + alpha, |alpha| <= pi/4,
//
//  let I = M mod 4, or I be the two lsb of M when M is represented
//  as 2's complement. I = [i_0 i_1]. Then
//
//  sin( Arg ) = (-1)^i_0  sin( alpha ) if i_1 = 0,
//             = (-1)^i_0  cos( alpha )     if i_1 = 1.
//
//  For example:
//       if M = -1, I = 11
//         sin ((-pi/2 + alpha) = (-1) cos (alpha)
//       if M = 0, I = 00
//         sin (alpha) = sin (alpha)
//       if M = 1, I = 01
//         sin (pi/2 + alpha) = cos (alpha)
//       if M = 2, I = 10
//         sin (pi + alpha) = (-1) sin (alpha)
//       if M = 3, I = 11
//         sin ((3/2)pi + alpha) = (-1) cos (alpha)
//
//  The value of alpha is obtained by argument reduction and
//  represented by two working precision numbers r and c where
//
//  alpha =  r  +  c     accurately.
//
//  The reduction method is described in a previous write up.
//  The argument reduction scheme identifies 4 cases. For Cases 2
//  and 4, because |alpha| is small, sin(r+c) and cos(r+c) can be
//  computed very easily by 2 or 3 terms of the Taylor series
//  expansion as follows:
//
//  Case 2:
//  -------
//
//  sin(r + c) = r + c - r^3/6  accurately
//  cos(r + c) = 1 - 2^(-67)    accurately
//
//  Case 4:
//  -------
//
//  sin(r + c) = r + c - r^3/6 + r^5/120    accurately
//  cos(r + c) = 1 - r^2/2 + r^4/24     accurately
//
//  The only cases left are Cases 1 and 3 of the argument reduction
//  procedure. These two cases will be merged since after the
//  argument is reduced in either cases, we have the reduced argument
//  represented as r + c and that the magnitude |r + c| is not small
//  enough to allow the usage of a very short approximation.
//
//  The required calculation is either
//
//  sin(r + c)  =  sin(r)  +  correction,  or
//  cos(r + c)  =  cos(r)  +  correction.
//
//  Specifically,
//
//  sin(r + c) = sin(r) + c sin'(r) + O(c^2)
//         = sin(r) + c cos (r) + O(c^2)
//         = sin(r) + c(1 - r^2/2)  accurately.
//  Similarly,
//
//  cos(r + c) = cos(r) - c sin(r) + O(c^2)
//         = cos(r) - c(r - r^3/6)  accurately.
//
//  We therefore concentrate on accurately calculating sin(r) and
//  cos(r) for a working-precision number r, |r| <= pi/4 to within
//  0.1% or so.
//
//  The greatest challenge of this task is that the second terms of
//  the Taylor series
//
//  r - r^3/3! + r^r/5! - ...
//
//  and
//
//  1 - r^2/2! + r^4/4! - ...
//
//  are not very small when |r| is close to pi/4 and the rounding
//  errors will be a concern if simple polynomial accumulation is
//  used. When |r| < 2^-3, however, the second terms will be small
//  enough (6 bits or so of right shift) that a normal Horner
//  recurrence suffices. Hence there are two cases that we consider
//  in the accurate computation of sin(r) and cos(r), |r| <= pi/4.
//
//  Case small_r: |r| < 2^(-3)
//  --------------------------
//
//  Since Arg = M pi/4 + r + c accurately, and M mod 4 is [i_0 i_1],
//  we have
//
//  sin(Arg) = (-1)^i_0 * sin(r + c)    if i_1 = 0
//       = (-1)^i_0 * cos(r + c)    if i_1 = 1
//
//  can be accurately approximated by
//
//  sin(Arg) = (-1)^i_0 * [sin(r) + c]  if i_1 = 0
//           = (-1)^i_0 * [cos(r) - c*r] if i_1 = 1
//
//  because |r| is small and thus the second terms in the correction
//  are unnecessary.
//
//  Finally, sin(r) and cos(r) are approximated by polynomials of
//  moderate lengths.
//
//  sin(r) =  r + S_1 r^3 + S_2 r^5 + ... + S_5 r^11
//  cos(r) =  1 + C_1 r^2 + C_2 r^4 + ... + C_5 r^10
//
//  We can make use of predicates to selectively calculate
//  sin(r) or cos(r) based on i_1.
//
//  Case normal_r: 2^(-3) <= |r| <= pi/4
//  ------------------------------------
//
//  This case is more likely than the previous one if one considers
//  r to be uniformly distributed in [-pi/4 pi/4]. Again,
//
//  sin(Arg) = (-1)^i_0 * sin(r + c)    if i_1 = 0
//           = (-1)^i_0 * cos(r + c)    if i_1 = 1.
//
//  Because |r| is now larger, we need one extra term in the
//  correction. sin(Arg) can be accurately approximated by
//
//  sin(Arg) = (-1)^i_0 * [sin(r) + c(1-r^2/2)]      if i_1 = 0
//           = (-1)^i_0 * [cos(r) - c*r*(1 - r^2/6)]    i_1 = 1.
//
//  Finally, sin(r) and cos(r) are approximated by polynomials of
//  moderate lengths.
//
//  sin(r) =  r + PP_1_hi r^3 + PP_1_lo r^3 +
//                PP_2 r^5 + ... + PP_8 r^17
//
//  cos(r) =  1 + QQ_1 r^2 + QQ_2 r^4 + ... + QQ_8 r^16
//
//  where PP_1_hi is only about 16 bits long and QQ_1 is -1/2.
//  The crux in accurate computation is to calculate
//
//  r + PP_1_hi r^3   or  1 + QQ_1 r^2
//
//  accurately as two pieces: U_hi and U_lo. The way to achieve this
//  is to obtain r_hi as a 10 sig. bit number that approximates r to
//  roughly 8 bits or so of accuracy. (One convenient way is
//
//  r_hi := frcpa( frcpa( r ) ).)
//
//  This way,
//
//  r + PP_1_hi r^3 =  r + PP_1_hi r_hi^3 +
//                          PP_1_hi (r^3 - r_hi^3)
//              =  [r + PP_1_hi r_hi^3]  +
//             [PP_1_hi (r - r_hi)
//                (r^2 + r_hi r + r_hi^2) ]
//              =  U_hi  +  U_lo
//
//  Since r_hi is only 10 bit long and PP_1_hi is only 16 bit long,
//  PP_1_hi * r_hi^3 is only at most 46 bit long and thus computed
//  exactly. Furthermore, r and PP_1_hi r_hi^3 are of opposite sign
//  and that there is no more than 8 bit shift off between r and
//  PP_1_hi * r_hi^3. Hence the sum, U_hi, is representable and thus
//  calculated without any error. Finally, the fact that
//
//  |U_lo| <= 2^(-8) |U_hi|
//
//  says that U_hi + U_lo is approximating r + PP_1_hi r^3 to roughly
//  8 extra bits of accuracy.
//
//  Similarly,
//
//  1 + QQ_1 r^2  =  [1 + QQ_1 r_hi^2]  +
//                      [QQ_1 (r - r_hi)(r + r_hi)]
//            =  U_hi  +  U_lo.
//
//  Summarizing, we calculate r_hi = frcpa( frcpa( r ) ).
//
//  If i_1 = 0, then
//
//    U_hi := r + PP_1_hi * r_hi^3
//    U_lo := PP_1_hi * (r - r_hi) * (r^2 + r*r_hi + r_hi^2)
//    poly := PP_1_lo r^3 + PP_2 r^5 + ... + PP_8 r^17
//    correction := c * ( 1 + C_1 r^2 )
//
//  Else ...i_1 = 1
//
//    U_hi := 1 + QQ_1 * r_hi * r_hi
//    U_lo := QQ_1 * (r - r_hi) * (r + r_hi)
//    poly := QQ_2 * r^4 + QQ_3 * r^6 + ... + QQ_8 r^16
//    correction := -c * r * (1 + S_1 * r^2)
//
//  End
//
//  Finally,
//
//  V := poly + ( U_lo + correction )
//
//                 /    U_hi  +  V         if i_0 = 0
//  result := |
//                 \  (-U_hi) -  V         if i_0 = 1
//
//  It is important that in the last step, negation of U_hi is
//  performed prior to the subtraction which is to be performed in
//  the user-set rounding mode.
//
//
//  Algorithmic Description
//  =======================
//
//  The argument reduction algorithm is tightly integrated into FSIN
//  and FCOS which share the same code. The following is complete and
//  self-contained. The argument reduction description given
//  previously is repeated below.
//
//
//  Step 0. Initialization.
//
//   If FSIN is invoked, set N_inc := 0; else if FCOS is invoked,
//   set N_inc := 1.
//
//  Step 1. Check for exceptional and special cases.
//
//   * If Arg is +-0, +-inf, NaN, NaT, go to Step 10 for special
//     handling.
//   * If |Arg| < 2^24, go to Step 2 for reduction of moderate
//     arguments. This is the most likely case.
//   * If |Arg| < 2^63, go to Step 8 for pre-reduction of large
//     arguments.
//   * If |Arg| >= 2^63, go to Step 10 for special handling.
//
//  Step 2. Reduction of moderate arguments.
//
//  If |Arg| < pi/4     ...quick branch
//     N_fix := N_inc   (integer)
//     r     := Arg
//     c     := 0.0
//     Branch to Step 4, Case_1_complete
//  Else        ...cf. argument reduction
//     N     := Arg * two_by_PI (fp)
//     N_fix := fcvt.fx( N )    (int)
//     N     := fcvt.xf( N_fix )
//     N_fix := N_fix + N_inc
//     s     := Arg - N * P_1   (first piece of pi/2)
//     w     := -N * P_2    (second piece of pi/2)
//
//     If |s| >= 2^(-33)
//        go to Step 3, Case_1_reduce
//     Else
//        go to Step 7, Case_2_reduce
//     Endif
//  Endif
//
//  Step 3. Case_1_reduce.
//
//  r := s + w
//  c := (s - r) + w    ...observe order
//
//  Step 4. Case_1_complete
//
//  ...At this point, the reduced argument alpha is
//  ...accurately represented as r + c.
//  If |r| < 2^(-3), go to Step 6, small_r.
//
//  Step 5. Normal_r.
//
//  Let [i_0 i_1] by the 2 lsb of N_fix.
//  FR_rsq  := r * r
//  r_hi := frcpa( frcpa( r ) )
//  r_lo := r - r_hi
//
//  If i_1 = 0, then
//    poly := r*FR_rsq*(PP_1_lo + FR_rsq*(PP_2 + ... FR_rsq*PP_8))
//    U_hi := r + PP_1_hi*r_hi*r_hi*r_hi    ...any order
//    U_lo := PP_1_hi*r_lo*(r*r + r*r_hi + r_hi*r_hi)
//    correction := c + c*C_1*FR_rsq        ...any order
//  Else
//    poly := FR_rsq*FR_rsq*(QQ_2 + FR_rsq*(QQ_3 + ... + FR_rsq*QQ_8))
//    U_hi := 1 + QQ_1 * r_hi * r_hi        ...any order
//    U_lo := QQ_1 * r_lo * (r + r_hi)
//    correction := -c*(r + S_1*FR_rsq*r)   ...any order
//  Endif
//
//  V := poly + (U_lo + correction) ...observe order
//
//  result := (i_0 == 0?   1.0 : -1.0)
//
//  Last instruction in user-set rounding mode
//
//  result := (i_0 == 0?   result*U_hi + V :
//                        result*U_hi - V)
//
//  Return
//
//  Step 6. Small_r.
//
//  ...Use flush to zero mode without causing exception
//    Let [i_0 i_1] be the two lsb of N_fix.
//
//  FR_rsq := r * r
//
//  If i_1 = 0 then
//     z := FR_rsq*FR_rsq; z := FR_rsq*z *r
//     poly_lo := S_3 + FR_rsq*(S_4 + FR_rsq*S_5)
//     poly_hi := r*FR_rsq*(S_1 + FR_rsq*S_2)
//     correction := c
//     result := r
//  Else
//     z := FR_rsq*FR_rsq; z := FR_rsq*z
//     poly_lo := C_3 + FR_rsq*(C_4 + FR_rsq*C_5)
//     poly_hi := FR_rsq*(C_1 + FR_rsq*C_2)
//     correction := -c*r
//     result := 1
//  Endif
//
//  poly := poly_hi + (z * poly_lo + correction)
//
//  If i_0 = 1, result := -result
//
//  Last operation. Perform in user-set rounding mode
//
//  result := (i_0 == 0?     result + poly :
//                          result - poly )
//  Return
//
//  Step 7. Case_2_reduce.
//
//  ...Refer to the write up for argument reduction for
//  ...rationale. The reduction algorithm below is taken from
//  ...argument reduction description and integrated this.
//
//  w := N*P_3
//  U_1 := N*P_2 + w        ...FMA
//  U_2 := (N*P_2 - U_1) + w    ...2 FMA
//  ...U_1 + U_2 is  N*(P_2+P_3) accurately
//
//  r := s - U_1
//  c := ( (s - r) - U_1 ) - U_2
//
//  ...The mathematical sum r + c approximates the reduced
//  ...argument accurately. Note that although compared to
//  ...Case 1, this case requires much more work to reduce
//  ...the argument, the subsequent calculation needed for
//  ...any of the trigonometric function is very little because
//  ...|alpha| < 1.01*2^(-33) and thus two terms of the
//  ...Taylor series expansion suffices.
//
//  If i_1 = 0 then
//     poly := c + S_1 * r * r * r  ...any order
//     result := r
//  Else
//     poly := -2^(-67)
//     result := 1.0
//  Endif
//
//  If i_0 = 1, result := -result
//
//  Last operation. Perform in user-set rounding mode
//
//  result := (i_0 == 0?     result + poly :
//                           result - poly )
//
//  Return
//
//
//  Step 8. Pre-reduction of large arguments.
//
//  ...Again, the following reduction procedure was described
//  ...in the separate write up for argument reduction, which
//  ...is tightly integrated here.

//  N_0 := Arg * Inv_P_0
//  N_0_fix := fcvt.fx( N_0 )
//  N_0 := fcvt.xf( N_0_fix)

//  Arg' := Arg - N_0 * P_0
//  w := N_0 * d_1
//  N := Arg' * two_by_PI
//  N_fix := fcvt.fx( N )
//  N := fcvt.xf( N_fix )
//  N_fix := N_fix + N_inc
//
//  s := Arg' - N * P_1
//  w := w - N * P_2
//
//  If |s| >= 2^(-14)
//     go to Step 3
//  Else
//     go to Step 9
//  Endif
//
//  Step 9. Case_4_reduce.
//
//    ...first obtain N_0*d_1 and -N*P_2 accurately
//   U_hi := N_0 * d_1      V_hi := -N*P_2
//   U_lo := N_0 * d_1 - U_hi   V_lo := -N*P_2 - U_hi   ...FMAs
//
//   ...compute the contribution from N_0*d_1 and -N*P_3
//   w := -N*P_3
//   w := w + N_0*d_2
//   t := U_lo + V_lo + w       ...any order
//
//   ...at this point, the mathematical value
//   ...s + U_hi + V_hi  + t approximates the true reduced argument
//   ...accurately. Just need to compute this accurately.
//
//   ...Calculate U_hi + V_hi accurately:
//   A := U_hi + V_hi
//   if |U_hi| >= |V_hi| then
//      a := (U_hi - A) + V_hi
//   else
//      a := (V_hi - A) + U_hi
//   endif
//   ...order in computing "a" must be observed. This branch is
//   ...best implemented by predicates.
//   ...A + a  is U_hi + V_hi accurately. Moreover, "a" is
//   ...much smaller than A: |a| <= (1/2)ulp(A).
//
//   ...Just need to calculate   s + A + a + t
//   C_hi := s + A      t := t + a
//   C_lo := (s - C_hi) + A
//   C_lo := C_lo + t
//
//   ...Final steps for reduction
//   r := C_hi + C_lo
//   c := (C_hi - r) + C_lo
//
//   ...At this point, we have r and c
//   ...And all we need is a couple of terms of the corresponding
//   ...Taylor series.
//
//   If i_1 = 0
//      poly := c + r*FR_rsq*(S_1 + FR_rsq*S_2)
//      result := r
//   Else
//      poly := FR_rsq*(C_1 + FR_rsq*C_2)
//      result := 1
//   Endif
//
//   If i_0 = 1, result := -result
//
//   Last operation. Perform in user-set rounding mode
//
//   result := (i_0 == 0?     result + poly :
//                            result - poly )
//   Return
//
//   Large Arguments: For arguments above 2**63, a Payne-Hanek
//   style argument reduction is used and pi_by_2 reduce is called.
//


RODATA
.align 16

LOCAL_OBJECT_START(FSINCOS_CONSTANTS)

data4 0x4B800000 // two**24
data4 0xCB800000 // -two**24
data4 0x00000000 // pad
data4 0x00000000 // pad
data8 0xA2F9836E4E44152A, 0x00003FFE // Inv_pi_by_2
data8 0xC84D32B0CE81B9F1, 0x00004016 // P_0
data8 0xC90FDAA22168C235, 0x00003FFF // P_1
data8 0xECE675D1FC8F8CBB, 0x0000BFBD // P_2
data8 0xB7ED8FBBACC19C60, 0x0000BF7C // P_3
data4 0x5F000000 // two**63
data4 0xDF000000 // -two**63
data4 0x00000000 // pad
data4 0x00000000 // pad
data8 0xA397E5046EC6B45A, 0x00003FE7 // Inv_P_0
data8 0x8D848E89DBD171A1, 0x0000BFBF // d_1
data8 0xD5394C3618A66F8E, 0x0000BF7C // d_2
data8 0xC90FDAA22168C234, 0x00003FFE // pi_by_4
data8 0xC90FDAA22168C234, 0x0000BFFE // neg_pi_by_4
data4 0x3E000000 // two**-3
data4 0xBE000000 // -two**-3
data4 0x00000000 // pad
data4 0x00000000 // pad
data4 0x2F000000 // two**-33
data4 0xAF000000 // -two**-33
data4 0x9E000000 // -two**-67
data4 0x00000000 // pad
data8 0xCC8ABEBCA21C0BC9, 0x00003FCE // PP_8
data8 0xD7468A05720221DA, 0x0000BFD6 // PP_7
data8 0xB092382F640AD517, 0x00003FDE // PP_6
data8 0xD7322B47D1EB75A4, 0x0000BFE5 // PP_5
data8 0xFFFFFFFFFFFFFFFE, 0x0000BFFD // C_1
data8 0xAAAA000000000000, 0x0000BFFC // PP_1_hi
data8 0xB8EF1D2ABAF69EEA, 0x00003FEC // PP_4
data8 0xD00D00D00D03BB69, 0x0000BFF2 // PP_3
data8 0x8888888888888962, 0x00003FF8 // PP_2
data8 0xAAAAAAAAAAAB0000, 0x0000BFEC // PP_1_lo
data8 0xD56232EFC2B0FE52, 0x00003FD2 // QQ_8
data8 0xC9C99ABA2B48DCA6, 0x0000BFDA // QQ_7
data8 0x8F76C6509C716658, 0x00003FE2 // QQ_6
data8 0x93F27DBAFDA8D0FC, 0x0000BFE9 // QQ_5
data8 0xAAAAAAAAAAAAAAAA, 0x0000BFFC // S_1
data8 0x8000000000000000, 0x0000BFFE // QQ_1
data8 0xD00D00D00C6E5041, 0x00003FEF // QQ_4
data8 0xB60B60B60B607F60, 0x0000BFF5 // QQ_3
data8 0xAAAAAAAAAAAAAA9B, 0x00003FFA // QQ_2
data8 0xFFFFFFFFFFFFFFFE, 0x0000BFFD // C_1
data8 0xAAAAAAAAAAAA719F, 0x00003FFA // C_2
data8 0xB60B60B60356F994, 0x0000BFF5 // C_3
data8 0xD00CFFD5B2385EA9, 0x00003FEF // C_4
data8 0x93E4BD18292A14CD, 0x0000BFE9 // C_5
data8 0xAAAAAAAAAAAAAAAA, 0x0000BFFC // S_1
data8 0x88888888888868DB, 0x00003FF8 // S_2
data8 0xD00D00D0055EFD4B, 0x0000BFF2 // S_3
data8 0xB8EF1C5D839730B9, 0x00003FEC // S_4
data8 0xD71EA3A4E5B3F492, 0x0000BFE5 // S_5
data4 0x38800000 // two**-14
data4 0xB8800000 // -two**-14
LOCAL_OBJECT_END(FSINCOS_CONSTANTS)

// sin and cos registers

// FR
FR_Input_X        = f8

FR_r              = f8
FR_c              = f9

FR_Two_to_63      = f32
FR_Two_to_24      = f33
FR_Pi_by_4        = f33
FR_Two_to_M14     = f34
FR_Two_to_M33     = f35
FR_Neg_Two_to_24  = f36
FR_Neg_Pi_by_4    = f36
FR_Neg_Two_to_M14 = f37
FR_Neg_Two_to_M33 = f38
FR_Neg_Two_to_M67 = f39
FR_Inv_pi_by_2    = f40
FR_N_float        = f41
FR_N_fix          = f42
FR_P_1            = f43
FR_P_2            = f44
FR_P_3            = f45
FR_s              = f46
FR_w              = f47
FR_d_2            = f48
FR_prelim         = f49
FR_Z              = f50
FR_A              = f51
FR_a              = f52
FR_t              = f53
FR_U_1            = f54
FR_U_2            = f55
FR_C_1            = f56
FR_C_2            = f57
FR_C_3            = f58
FR_C_4            = f59
FR_C_5            = f60
FR_S_1            = f61
FR_S_2            = f62
FR_S_3            = f63
FR_S_4            = f64
FR_S_5            = f65
FR_poly_hi        = f66
FR_poly_lo        = f67
FR_r_hi           = f68
FR_r_lo           = f69
FR_rsq            = f70
FR_r_cubed        = f71
FR_C_hi           = f72
FR_N_0            = f73
FR_d_1            = f74
FR_V              = f75
FR_V_hi           = f75
FR_V_lo           = f76
FR_U_hi           = f77
FR_U_lo           = f78
FR_U_hiabs        = f79
FR_V_hiabs        = f80
FR_PP_8           = f81
FR_QQ_8           = f81
FR_PP_7           = f82
FR_QQ_7           = f82
FR_PP_6           = f83
FR_QQ_6           = f83
FR_PP_5           = f84
FR_QQ_5           = f84
FR_PP_4           = f85
FR_QQ_4           = f85
FR_PP_3           = f86
FR_QQ_3           = f86
FR_PP_2           = f87
FR_QQ_2           = f87
FR_QQ_1           = f88
FR_N_0_fix        = f89
FR_Inv_P_0        = f90
FR_corr           = f91
FR_poly           = f92
FR_Neg_Two_to_M3  = f93
FR_Two_to_M3      = f94
FR_Neg_Two_to_63  = f94
FR_P_0            = f95
FR_C_lo           = f96
FR_PP_1           = f97
FR_PP_1_lo        = f98
FR_ArgPrime       = f99

// GR
GR_Table_Base     = r32
GR_Table_Base1    = r33
GR_i_0            = r34
GR_i_1            = r35
GR_N_Inc          = r36
GR_Sin_or_Cos     = r37

GR_SAVE_B0        = r39
GR_SAVE_GP        = r40
GR_SAVE_PFS       = r41

// sincos combined routine registers

// GR
GR_SINCOS_SAVE_PFS    = r32
GR_SINCOS_SAVE_B0     = r33
GR_SINCOS_SAVE_GP     = r34

// FR
FR_SINCOS_ARG         = f100
FR_SINCOS_RES_SIN     = f101


.section .text


GLOBAL_LIBM_ENTRY(__libm_sincos_large)

{ .mfi
        alloc GR_SINCOS_SAVE_PFS = ar.pfs,0,3,0,0
        fma.s1 FR_SINCOS_ARG     = f8, f1, f0  // Save argument for sin and cos
        mov GR_SINCOS_SAVE_B0    = b0
};;

{ .mfb
        mov GR_SINCOS_SAVE_GP    = gp
        nop.f  0
        br.call.sptk b0          = __libm_sin_large // Call sin
};;

{ .mfi
        nop.m  0
        fma.s1 FR_SINCOS_RES_SIN = f8, f1, f0 // Save sin result
        nop.i  0
};;

{ .mfb
        nop.m  0
        fma.s1 f8                = FR_SINCOS_ARG, f1, f0 // Arg for cos
        br.call.sptk b0          = __libm_cos_large // Call cos
};;

{ .mfi
        mov    gp                = GR_SINCOS_SAVE_GP
        fma.s1 f9                = FR_SINCOS_RES_SIN, f1, f0 // Out sin result
        mov    b0                = GR_SINCOS_SAVE_B0
};;

{ .mib
        nop.m  0
        mov ar.pfs               = GR_SINCOS_SAVE_PFS
        br.ret.sptk                b0 // sincos_large exit
};;

GLOBAL_LIBM_END(__libm_sincos_large)




GLOBAL_LIBM_ENTRY(__libm_sin_large)

{ .mlx
alloc GR_Table_Base = ar.pfs,0,12,2,0
       movl GR_Sin_or_Cos = 0x0 ;;
}

{ .mmi
      nop.m 999
      addl           GR_Table_Base   = @ltoff(FSINCOS_CONSTANTS#), gp
      nop.i 999
}
;;

{ .mmi
      ld8 GR_Table_Base = [GR_Table_Base]
      nop.m 999
      nop.i 999
}
;;


{ .mib
      nop.m 999
      nop.i 999
       br.cond.sptk SINCOS_CONTINUE ;;
}

GLOBAL_LIBM_END(__libm_sin_large)

GLOBAL_LIBM_ENTRY(__libm_cos_large)

{ .mlx
alloc GR_Table_Base= ar.pfs,0,12,2,0
       movl GR_Sin_or_Cos = 0x1 ;;
}

{ .mmi
      nop.m 999
      addl           GR_Table_Base   = @ltoff(FSINCOS_CONSTANTS#), gp
      nop.i 999
}
;;

{ .mmi
      ld8 GR_Table_Base = [GR_Table_Base]
      nop.m 999
      nop.i 999
}
;;

//
//     Load Table Address
//
SINCOS_CONTINUE:

{ .mmi
       add GR_Table_Base1 = 96, GR_Table_Base
       ldfs FR_Two_to_24 = [GR_Table_Base], 4
       nop.i 999
}
;;

{ .mmi
      nop.m 999
//
//     Load 2**24, load 2**63.
//
       ldfs FR_Neg_Two_to_24 = [GR_Table_Base], 12
       mov   r41 = ar.pfs ;;
}

{ .mfi
       ldfs FR_Two_to_63 = [GR_Table_Base1], 4
//
//     Check for unnormals - unsupported operands. We do not want
//     to generate denormal exception
//     Check for NatVals, QNaNs, SNaNs, +/-Infs
//     Check for EM unsupporteds
//     Check for Zero
//
       fclass.m.unc  p6, p8 =  FR_Input_X, 0x1E3
       mov   r40 = gp ;;
}

{ .mfi
      nop.m 999
       fclass.nm.unc p8, p0 =  FR_Input_X, 0x1FF
// GR_Sin_or_Cos denotes
       mov   r39 = b0
}

{ .mfb
       ldfs FR_Neg_Two_to_63 = [GR_Table_Base1], 12
       fclass.m.unc p10, p0 = FR_Input_X, 0x007
(p6)   br.cond.spnt SINCOS_SPECIAL ;;
}

{ .mib
      nop.m 999
      nop.i 999
(p8)   br.cond.spnt SINCOS_SPECIAL ;;
}

{ .mib
      nop.m 999
      nop.i 999
//
//     Branch if +/- NaN, Inf.
//     Load -2**24, load -2**63.
//
(p10)  br.cond.spnt SINCOS_ZERO ;;
}

{ .mmb
       ldfe FR_Inv_pi_by_2 = [GR_Table_Base], 16
       ldfe FR_Inv_P_0 = [GR_Table_Base1], 16
      nop.b 999 ;;
}

{ .mmb
      nop.m 999
       ldfe     FR_d_1 = [GR_Table_Base1], 16
      nop.b 999 ;;
}
//
//     Raise possible denormal operand flag with useful fcmp
//     Is x <= -2**63
//     Load Inv_P_0 for pre-reduction
//     Load Inv_pi_by_2
//

{ .mmb
       ldfe     FR_P_0 = [GR_Table_Base], 16
       ldfe FR_d_2 = [GR_Table_Base1], 16
      nop.b 999 ;;
}
//
//     Load P_0
//     Load d_1
//     Is x >= 2**63
//     Is x <= -2**24?
//

{ .mmi
       ldfe FR_P_1 = [GR_Table_Base], 16 ;;
//
//     Load P_1
//     Load d_2
//     Is x >= 2**24?
//
       ldfe FR_P_2 = [GR_Table_Base], 16
      nop.i 999 ;;
}

{ .mmf
      nop.m 999
       ldfe FR_P_3 = [GR_Table_Base], 16
       fcmp.le.unc.s1   p7, p8 = FR_Input_X, FR_Neg_Two_to_24
}

{ .mfi
      nop.m 999
//
//     Branch if +/- zero.
//     Decide about the paths to take:
//     If -2**24 < FR_Input_X < 2**24 - CASE 1 OR 2
//     OTHERWISE - CASE 3 OR 4
//
       fcmp.le.unc.s1   p10, p11 = FR_Input_X, FR_Neg_Two_to_63
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p8)   fcmp.ge.s1 p7, p0 = FR_Input_X, FR_Two_to_24
      nop.i 999
}

{ .mfi
       ldfe FR_Pi_by_4 = [GR_Table_Base1], 16
(p11)  fcmp.ge.s1   p10, p0 = FR_Input_X, FR_Two_to_63
      nop.i 999 ;;
}

{ .mmi
       ldfe FR_Neg_Pi_by_4 = [GR_Table_Base1], 16 ;;
       ldfs FR_Two_to_M3 = [GR_Table_Base1], 4
      nop.i 999 ;;
}

{ .mib
       ldfs FR_Neg_Two_to_M3 = [GR_Table_Base1], 12
      nop.i 999
//
//     Load P_2
//     Load P_3
//     Load pi_by_4
//     Load neg_pi_by_4
//     Load 2**(-3)
//     Load -2**(-3).
//
(p10)  br.cond.spnt SINCOS_ARG_TOO_LARGE ;;
}

{ .mib
      nop.m 999
      nop.i 999
//
//     Branch out if x >= 2**63. Use Payne-Hanek Reduction
//
(p7)   br.cond.spnt SINCOS_LARGER_ARG ;;
}

{ .mfi
      nop.m 999
//
//     Branch if Arg <= -2**24 or Arg >= 2**24 and use pre-reduction.
//
       fma.s1   FR_N_float = FR_Input_X, FR_Inv_pi_by_2, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
       fcmp.lt.unc.s1   p6, p7 = FR_Input_X, FR_Pi_by_4
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Select the case when |Arg| < pi/4
//     Else Select the case when |Arg| >= pi/4
//
       fcvt.fx.s1 FR_N_fix = FR_N_float
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     N  = Arg * 2/pi
//     Check if Arg < pi/4
//
(p6)   fcmp.gt.s1 p6, p7 = FR_Input_X, FR_Neg_Pi_by_4
      nop.i 999 ;;
}
//
//     Case 2: Convert integer N_fix back to normalized floating-point value.
//     Case 1: p8 is only affected  when p6 is set
//

{ .mfi
(p7)   ldfs FR_Two_to_M33 = [GR_Table_Base1], 4
//
//     Grab the integer part of N and call it N_fix
//
(p6)   fmerge.se FR_r = FR_Input_X, FR_Input_X
//     If |x| < pi/4, r = x and c = 0
//     lf |x| < pi/4, is x < 2**(-3).
//     r = Arg
//     c = 0
(p6)   mov GR_N_Inc = GR_Sin_or_Cos ;;
}

{ .mmf
      nop.m 999
(p7)   ldfs FR_Neg_Two_to_M33 = [GR_Table_Base1], 4
(p6)   fmerge.se FR_c = f0, f0
}

{ .mfi
      nop.m 999
(p6)   fcmp.lt.unc.s1   p8, p9 = FR_Input_X, FR_Two_to_M3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     lf |x| < pi/4, is -2**(-3)< x < 2**(-3) - set p8.
//     If |x| >= pi/4,
//     Create the right N for |x| < pi/4 and otherwise
//     Case 2: Place integer part of N in GP register
//
(p7)   fcvt.xf FR_N_float = FR_N_fix
      nop.i 999 ;;
}

{ .mmf
      nop.m 999
(p7)   getf.sig GR_N_Inc = FR_N_fix
(p8)   fcmp.gt.s1 p8, p0 = FR_Input_X, FR_Neg_Two_to_M3 ;;
}

{ .mib
      nop.m 999
      nop.i 999
//
//     Load 2**(-33), -2**(-33)
//
(p8)   br.cond.spnt SINCOS_SMALL_R ;;
}

{ .mib
      nop.m 999
      nop.i 999
(p6)   br.cond.sptk SINCOS_NORMAL_R ;;
}
//
//     if |x| < pi/4, branch based on |x| < 2**(-3) or otherwise.
//
//
//     In this branch, |x| >= pi/4.
//

{ .mfi
       ldfs FR_Neg_Two_to_M67 = [GR_Table_Base1], 8
//
//     Load -2**(-67)
//
       fnma.s1  FR_s = FR_N_float, FR_P_1, FR_Input_X
//
//     w = N * P_2
//     s = -N * P_1  + Arg
//
       add GR_N_Inc = GR_N_Inc, GR_Sin_or_Cos
}

{ .mfi
      nop.m 999
       fma.s1   FR_w = FR_N_float, FR_P_2, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Adjust N_fix by N_inc to determine whether sine or
//     cosine is being calculated
//
       fcmp.lt.unc.s1 p7, p6 = FR_s, FR_Two_to_M33
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p7)   fcmp.gt.s1 p7, p6 = FR_s, FR_Neg_Two_to_M33
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//     Remember x >= pi/4.
//     Is s <= -2**(-33) or s >= 2**(-33) (p6)
//     or -2**(-33) < s < 2**(-33) (p7)
(p6)   fms.s1 FR_r = FR_s, f1, FR_w
      nop.i 999
}

{ .mfi
      nop.m 999
(p7)   fma.s1 FR_w = FR_N_float, FR_P_3, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p7)   fma.s1 FR_U_1 = FR_N_float, FR_P_2, FR_w
      nop.i 999
}

{ .mfi
      nop.m 999
(p6)   fms.s1 FR_c = FR_s, f1, FR_r
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     For big s: r = s - w: No futher reduction is necessary
//     For small s: w = N * P_3 (change sign) More reduction
//
(p6)   fcmp.lt.unc.s1 p8, p9 = FR_r, FR_Two_to_M3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p8)   fcmp.gt.s1 p8, p9 = FR_r, FR_Neg_Two_to_M3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p7)   fms.s1 FR_r = FR_s, f1, FR_U_1
      nop.i 999
}

{ .mfb
      nop.m 999
//
//     For big s: Is |r| < 2**(-3)?
//     For big s: c = S - r
//     For small s: U_1 = N * P_2 + w
//
//     If p8 is set, prepare to branch to Small_R.
//     If p9 is set, prepare to branch to Normal_R.
//     For big s,  r is complete here.
//
(p6)   fms.s1 FR_c = FR_c, f1, FR_w
//
//     For big s: c = c + w (w has not been negated.)
//     For small s: r = S - U_1
//
(p8)   br.cond.spnt SINCOS_SMALL_R ;;
}

{ .mib
      nop.m 999
      nop.i 999
(p9)   br.cond.sptk SINCOS_NORMAL_R ;;
}

{ .mfi
(p7)   add GR_Table_Base1 = 224, GR_Table_Base1
//
//     Branch to SINCOS_SMALL_R or SINCOS_NORMAL_R
//
(p7)   fms.s1 FR_U_2 = FR_N_float, FR_P_2, FR_U_1
//
//     c = S - U_1
//     r = S_1 * r
//
//
(p7)   extr.u   GR_i_1 = GR_N_Inc, 0, 1
}

{ .mmi
      nop.m 999 ;;
//
//     Get [i_0,i_1] - two lsb of N_fix_gr.
//     Do dummy fmpy so inexact is always set.
//
(p7)   cmp.eq.unc p9, p10 = 0x0, GR_i_1
(p7)   extr.u   GR_i_0 = GR_N_Inc, 1, 1 ;;
}
//
//     For small s: U_2 = N * P_2 - U_1
//     S_1 stored constant - grab the one stored with the
//     coefficients.
//

{ .mfi
(p7)   ldfe FR_S_1 = [GR_Table_Base1], 16
//
//     Check if i_1 and i_0  != 0
//
(p10)  fma.s1   FR_poly = f0, f1, FR_Neg_Two_to_M67
(p7)   cmp.eq.unc p11, p12 = 0x0, GR_i_0 ;;
}

{ .mfi
      nop.m 999
(p7)   fms.s1   FR_s = FR_s, f1, FR_r
      nop.i 999
}

{ .mfi
      nop.m 999
//
//     S = S - r
//     U_2 = U_2 + w
//     load S_1
//
(p7)   fma.s1   FR_rsq = FR_r, FR_r, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p7)   fma.s1   FR_U_2 = FR_U_2, f1, FR_w
      nop.i 999
}

{ .mfi
      nop.m 999
//(p7)   fmerge.se FR_Input_X = FR_r, FR_r
(p7)   fmerge.se FR_prelim = FR_r, FR_r
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//(p10)  fma.s1 FR_Input_X = f0, f1, f1
(p10)  fma.s1 FR_prelim = f0, f1, f1
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     FR_rsq = r * r
//     Save r as the result.
//
(p7)   fms.s1   FR_c = FR_s, f1, FR_U_1
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     if ( i_1 ==0) poly = c + S_1*r*r*r
//     else Result = 1
//
//(p12)  fnma.s1 FR_Input_X = FR_Input_X, f1, f0
(p12)  fnma.s1 FR_prelim = FR_prelim, f1, f0
      nop.i 999
}

{ .mfi
      nop.m 999
(p7)   fma.s1   FR_r = FR_S_1, FR_r, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p7)   fma.d.s1 FR_S_1 = FR_S_1, FR_S_1, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     If i_1 != 0, poly = 2**(-67)
//
(p7)   fms.s1 FR_c = FR_c, f1, FR_U_2
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     c = c - U_2
//
(p9)   fma.s1 FR_poly = FR_r, FR_rsq, FR_c
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     i_0 != 0, so Result = -Result
//
(p11)  fma.s1 FR_Input_X = FR_prelim, f1, FR_poly
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
(p12)  fms.s1 FR_Input_X = FR_prelim, f1, FR_poly
//
//     if (i_0 == 0),  Result = Result + poly
//     else            Result = Result - poly
//
       br.ret.sptk   b0 ;;
}
SINCOS_LARGER_ARG:

{ .mfi
      nop.m 999
       fma.s1 FR_N_0 = FR_Input_X, FR_Inv_P_0, f0
      nop.i 999
}
;;

//     This path for argument > 2*24
//     Adjust table_ptr1 to beginning of table.
//

{ .mmi
      nop.m 999
      addl           GR_Table_Base   = @ltoff(FSINCOS_CONSTANTS#), gp
      nop.i 999
}
;;

{ .mmi
      ld8 GR_Table_Base = [GR_Table_Base]
      nop.m 999
      nop.i 999
}
;;


//
//     Point to  2*-14
//     N_0 = Arg * Inv_P_0
//

{ .mmi
       add GR_Table_Base = 688, GR_Table_Base ;;
       ldfs FR_Two_to_M14 = [GR_Table_Base], 4
      nop.i 999 ;;
}

{ .mfi
       ldfs FR_Neg_Two_to_M14 = [GR_Table_Base], 0
      nop.f 999
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Load values 2**(-14) and -2**(-14)
//
       fcvt.fx.s1 FR_N_0_fix = FR_N_0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     N_0_fix  = integer part of N_0
//
       fcvt.xf FR_N_0 = FR_N_0_fix
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Make N_0 the integer part
//
       fnma.s1 FR_ArgPrime = FR_N_0, FR_P_0, FR_Input_X
      nop.i 999
}

{ .mfi
      nop.m 999
       fma.s1 FR_w = FR_N_0, FR_d_1, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Arg' = -N_0 * P_0 + Arg
//     w  = N_0 * d_1
//
       fma.s1 FR_N_float = FR_ArgPrime, FR_Inv_pi_by_2, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     N = A' * 2/pi
//
       fcvt.fx.s1 FR_N_fix = FR_N_float
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     N_fix is the integer part
//
       fcvt.xf FR_N_float = FR_N_fix
      nop.i 999 ;;
}

{ .mfi
       getf.sig GR_N_Inc = FR_N_fix
      nop.f 999
      nop.i 999 ;;
}

{ .mii
      nop.m 999
      nop.i 999 ;;
       add GR_N_Inc = GR_N_Inc, GR_Sin_or_Cos ;;
}

{ .mfi
      nop.m 999
//
//     N is the integer part of the reduced-reduced argument.
//     Put the integer in a GP register
//
       fnma.s1 FR_s = FR_N_float, FR_P_1, FR_ArgPrime
      nop.i 999
}

{ .mfi
      nop.m 999
       fnma.s1 FR_w = FR_N_float, FR_P_2, FR_w
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     s = -N*P_1 + Arg'
//     w = -N*P_2 + w
//     N_fix_gr = N_fix_gr + N_inc
//
       fcmp.lt.unc.s1 p9, p8 = FR_s, FR_Two_to_M14
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)   fcmp.gt.s1 p9, p8 = FR_s, FR_Neg_Two_to_M14
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     For |s|  > 2**(-14) r = S + w (r complete)
//     Else       U_hi = N_0 * d_1
//
(p9)   fma.s1 FR_V_hi = FR_N_float, FR_P_2, f0
      nop.i 999
}

{ .mfi
      nop.m 999
(p9)   fma.s1 FR_U_hi = FR_N_0, FR_d_1, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Either S <= -2**(-14) or S >= 2**(-14)
//     or -2**(-14) < s < 2**(-14)
//
(p8)   fma.s1 FR_r = FR_s, f1, FR_w
      nop.i 999
}

{ .mfi
      nop.m 999
(p9)   fma.s1 FR_w = FR_N_float, FR_P_3, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     We need abs of both U_hi and V_hi - don't
//     worry about switched sign of V_hi.
//
(p9)   fms.s1 FR_A = FR_U_hi, f1, FR_V_hi
      nop.i 999
}

{ .mfi
      nop.m 999
//
//     Big s: finish up c = (S - r) + w (c complete)
//     Case 4: A =  U_hi + V_hi
//     Note: Worry about switched sign of V_hi, so subtract instead of add.
//
(p9)   fnma.s1 FR_V_lo = FR_N_float, FR_P_2, FR_V_hi
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)   fms.s1 FR_U_lo = FR_N_0, FR_d_1, FR_U_hi
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)   fmerge.s FR_V_hiabs = f0, FR_V_hi
      nop.i 999
}

{ .mfi
      nop.m 999
//     For big s: c = S - r
//     For small s do more work: U_lo = N_0 * d_1 - U_hi
//
(p9)   fmerge.s FR_U_hiabs = f0, FR_U_hi
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     For big s: Is |r| < 2**(-3)
//     For big s: if p12 set, prepare to branch to Small_R.
//     For big s: If p13 set, prepare to branch to Normal_R.
//
(p8)   fms.s1 FR_c = FR_s, f1, FR_r
      nop.i 999
}

{ .mfi
      nop.m 999
//
//     For small S: V_hi = N * P_2
//                  w = N * P_3
//     Note the product does not include the (-) as in the writeup
//     so (-) missing for V_hi and w.
//
(p8)   fcmp.lt.unc.s1 p12, p13 = FR_r, FR_Two_to_M3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p12)  fcmp.gt.s1 p12, p13 = FR_r, FR_Neg_Two_to_M3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p8)   fma.s1 FR_c = FR_c, f1, FR_w
      nop.i 999
}

{ .mfb
      nop.m 999
(p9)   fms.s1 FR_w = FR_N_0, FR_d_2, FR_w
(p12)  br.cond.spnt SINCOS_SMALL_R ;;
}

{ .mib
      nop.m 999
      nop.i 999
(p13)  br.cond.sptk SINCOS_NORMAL_R ;;
}

{ .mfi
      nop.m 999
//
//     Big s: Vector off when |r| < 2**(-3).  Recall that p8 will be true.
//     The remaining stuff is for Case 4.
//     Small s: V_lo = N * P_2 + U_hi (U_hi is in place of V_hi in writeup)
//     Note: the (-) is still missing for V_lo.
//     Small s: w = w + N_0 * d_2
//     Note: the (-) is now incorporated in w.
//
(p9)   fcmp.ge.unc.s1 p10, p11 = FR_U_hiabs, FR_V_hiabs
       extr.u   GR_i_1 = GR_N_Inc, 0, 1 ;;
}

{ .mfi
      nop.m 999
//
//     C_hi = S + A
//
(p9)   fma.s1 FR_t = FR_U_lo, f1, FR_V_lo
       extr.u   GR_i_0 = GR_N_Inc, 1, 1 ;;
}

{ .mfi
      nop.m 999
//
//     t = U_lo + V_lo
//
//
(p10)  fms.s1 FR_a = FR_U_hi, f1, FR_A
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p11)  fma.s1 FR_a = FR_V_hi, f1, FR_A
      nop.i 999
}
;;

{ .mmi
      nop.m 999
      addl           GR_Table_Base   = @ltoff(FSINCOS_CONSTANTS#), gp
      nop.i 999
}
;;

{ .mmi
      ld8 GR_Table_Base = [GR_Table_Base]
      nop.m 999
      nop.i 999
}
;;


{ .mfi
       add GR_Table_Base = 528, GR_Table_Base
//
//     Is U_hiabs >= V_hiabs?
//
(p9)   fma.s1 FR_C_hi = FR_s, f1, FR_A
      nop.i 999 ;;
}

{ .mmi
       ldfe FR_C_1 = [GR_Table_Base], 16 ;;
       ldfe FR_C_2 = [GR_Table_Base], 64
      nop.i 999 ;;
}

{ .mmf
      nop.m 999
//
//     c = c + C_lo  finished.
//     Load  C_2
//
       ldfe FR_S_1 = [GR_Table_Base], 16
//
//     C_lo = S - C_hi
//
       fma.s1 FR_t = FR_t, f1, FR_w ;;
}
//
//     r and c have been computed.
//     Make sure ftz mode is set - should be automatic when using wre
//     |r| < 2**(-3)
//     Get [i_0,i_1] - two lsb of N_fix.
//     Load S_1
//

{ .mfi
       ldfe FR_S_2 = [GR_Table_Base], 64
//
//     t = t + w
//
(p10)  fms.s1 FR_a = FR_a, f1, FR_V_hi
       cmp.eq.unc p9, p10 = 0x0, GR_i_0
}

{ .mfi
      nop.m 999
//
//     For larger u than v: a = U_hi - A
//     Else a = V_hi - A (do an add to account for missing (-) on V_hi
//
       fms.s1 FR_C_lo = FR_s, f1, FR_C_hi
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p11)  fms.s1 FR_a = FR_U_hi, f1, FR_a
       cmp.eq.unc p11, p12 = 0x0, GR_i_1
}

{ .mfi
      nop.m 999
//
//     If u > v: a = (U_hi - A)  + V_hi
//     Else      a = (V_hi - A)  + U_hi
//     In each case account for negative missing from V_hi.
//
       fma.s1 FR_C_lo = FR_C_lo, f1, FR_A
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     C_lo = (S - C_hi) + A
//
       fma.s1 FR_t = FR_t, f1, FR_a
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     t = t + a
//
       fma.s1 FR_C_lo = FR_C_lo, f1, FR_t
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     C_lo = C_lo + t
//     Adjust Table_Base to beginning of table
//
       fma.s1 FR_r = FR_C_hi, f1, FR_C_lo
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Load S_2
//
       fma.s1 FR_rsq = FR_r, FR_r, f0
      nop.i 999
}

{ .mfi
      nop.m 999
//
//     Table_Base points to C_1
//     r = C_hi + C_lo
//
       fms.s1 FR_c = FR_C_hi, f1, FR_r
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     if i_1 ==0: poly = S_2 * FR_rsq + S_1
//     else        poly = C_2 * FR_rsq + C_1
//
//(p11)  fma.s1 FR_Input_X = f0, f1, FR_r
(p11)  fma.s1 FR_prelim = f0, f1, FR_r
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//(p12)  fma.s1 FR_Input_X = f0, f1, f1
(p12)  fma.s1 FR_prelim = f0, f1, f1
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     Compute r_cube = FR_rsq * r
//
(p11)  fma.s1 FR_poly = FR_rsq, FR_S_2, FR_S_1
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p12)  fma.s1 FR_poly = FR_rsq, FR_C_2, FR_C_1
      nop.i 999
}

{ .mfi
      nop.m 999
//
//     Compute FR_rsq = r * r
//     Is i_1 == 0 ?
//
       fma.s1 FR_r_cubed = FR_rsq, FR_r, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     c = C_hi - r
//     Load  C_1
//
       fma.s1 FR_c = FR_c, f1, FR_C_lo
      nop.i 999
}

{ .mfi
      nop.m 999
//
//     if i_1 ==0: poly = r_cube * poly + c
//     else        poly = FR_rsq * poly
//
//(p10)  fms.s1 FR_Input_X = f0, f1, FR_Input_X
(p10)  fms.s1 FR_prelim = f0, f1, FR_prelim
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     if i_1 ==0: Result = r
//     else        Result = 1.0
//
(p11)  fma.s1 FR_poly = FR_r_cubed, FR_poly, FR_c
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p12)  fma.s1 FR_poly = FR_rsq, FR_poly, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//     if i_0 !=0: Result = -Result
//
(p9)   fma.s1 FR_Input_X = FR_prelim, f1, FR_poly
      nop.i 999 ;;
}

{ .mfb
      nop.m 999
(p10)  fms.s1 FR_Input_X = FR_prelim, f1, FR_poly
//
//     if i_0 == 0: Result = Result + poly
//     else         Result = Result - poly
//
       br.ret.sptk   b0 ;;
}
SINCOS_SMALL_R:

{ .mii
      nop.m 999
        extr.u  GR_i_1 = GR_N_Inc, 0, 1 ;;
//
//
//      Compare both i_1 and i_0 with 0.
//      if i_1 == 0, set p9.
//      if i_0 == 0, set p11.
//
        cmp.eq.unc p9, p10 = 0x0, GR_i_1 ;;
}

{ .mfi
      nop.m 999
        fma.s1 FR_rsq = FR_r, FR_r, f0
        extr.u  GR_i_0 = GR_N_Inc, 1, 1 ;;
}

{ .mfi
      nop.m 999
//
//  Z = Z * FR_rsq
//
(p10)   fnma.s1 FR_c = FR_c, FR_r, f0
        cmp.eq.unc p11, p12 = 0x0, GR_i_0
}
;;

// ******************************************************************
// ******************************************************************
// ******************************************************************
//      r and c have been computed.
//      We know whether this is the sine or cosine routine.
//      Make sure ftz mode is set - should be automatic when using wre
//      |r| < 2**(-3)
//
//      Set table_ptr1 to beginning of constant table.
//      Get [i_0,i_1] - two lsb of N_fix_gr.
//

{ .mmi
      nop.m 999
      addl           GR_Table_Base   = @ltoff(FSINCOS_CONSTANTS#), gp
      nop.i 999
}
;;

{ .mmi
      ld8 GR_Table_Base = [GR_Table_Base]
      nop.m 999
      nop.i 999
}
;;


//
//      Set table_ptr1 to point to S_5.
//      Set table_ptr1 to point to C_5.
//      Compute FR_rsq = r * r
//

{ .mfi
(p9)    add GR_Table_Base = 672, GR_Table_Base
(p10)   fmerge.s FR_r = f1, f1
(p10)   add GR_Table_Base = 592, GR_Table_Base ;;
}
//
//      Set table_ptr1 to point to S_5.
//      Set table_ptr1 to point to C_5.
//

{ .mmi
(p9)    ldfe FR_S_5 = [GR_Table_Base], -16 ;;
//
//      if (i_1 == 0) load S_5
//      if (i_1 != 0) load C_5
//
(p9)    ldfe FR_S_4 = [GR_Table_Base], -16
      nop.i 999 ;;
}

{ .mmf
(p10)   ldfe FR_C_5 = [GR_Table_Base], -16
//
//      Z = FR_rsq * FR_rsq
//
(p9)    ldfe FR_S_3 = [GR_Table_Base], -16
//
//      Compute FR_rsq = r * r
//      if (i_1 == 0) load S_4
//      if (i_1 != 0) load C_4
//
        fma.s1 FR_Z = FR_rsq, FR_rsq, f0 ;;
}
//
//      if (i_1 == 0) load S_3
//      if (i_1 != 0) load C_3
//

{ .mmi
(p9)    ldfe FR_S_2 = [GR_Table_Base], -16 ;;
//
//      if (i_1 == 0) load S_2
//      if (i_1 != 0) load C_2
//
(p9)    ldfe FR_S_1 = [GR_Table_Base], -16
      nop.i 999
}

{ .mmi
(p10)   ldfe FR_C_4 = [GR_Table_Base], -16 ;;
(p10)   ldfe FR_C_3 = [GR_Table_Base], -16
      nop.i 999 ;;
}

{ .mmi
(p10)   ldfe FR_C_2 = [GR_Table_Base], -16 ;;
(p10)   ldfe FR_C_1 = [GR_Table_Base], -16
      nop.i 999
}

{ .mfi
      nop.m 999
//
//      if (i_1 != 0):
//      poly_lo = FR_rsq * C_5 + C_4
//      poly_hi = FR_rsq * C_2 + C_1
//
(p9)    fma.s1 FR_Z = FR_Z, FR_r, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1 == 0) load S_1
//      if (i_1 != 0) load C_1
//
(p9)    fma.s1 FR_poly_lo = FR_rsq, FR_S_5, FR_S_4
      nop.i 999
}

{ .mfi
      nop.m 999
//
//      c = -c * r
//      dummy fmpy's to flag inexact.
//
(p9)    fma.d.s1 FR_S_4 = FR_S_4, FR_S_4, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      poly_lo = FR_rsq * poly_lo + C_3
//      poly_hi = FR_rsq * poly_hi
//
        fma.s1  FR_Z = FR_Z, FR_rsq, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)    fma.s1 FR_poly_hi = FR_rsq, FR_S_2, FR_S_1
      nop.i 999
}

{ .mfi
      nop.m 999
//
//      if (i_1 == 0):
//      poly_lo = FR_rsq * S_5 + S_4
//      poly_hi = FR_rsq * S_2 + S_1
//
(p10)   fma.s1 FR_poly_lo = FR_rsq, FR_C_5, FR_C_4
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1 == 0):
//      Z = Z * r  for only one of the small r cases - not there
//      in original implementation notes.
//
(p9)    fma.s1 FR_poly_lo = FR_rsq, FR_poly_lo, FR_S_3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly_hi = FR_rsq, FR_C_2, FR_C_1
      nop.i 999
}

{ .mfi
      nop.m 999
(p10)   fma.d.s1 FR_C_1 = FR_C_1, FR_C_1, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)    fma.s1 FR_poly_hi = FR_poly_hi, FR_rsq, f0
      nop.i 999
}

{ .mfi
      nop.m 999
//
//      poly_lo = FR_rsq * poly_lo + S_3
//      poly_hi = FR_rsq * poly_hi
//
(p10)   fma.s1 FR_poly_lo = FR_rsq, FR_poly_lo, FR_C_3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly_hi = FR_poly_hi, FR_rsq, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//  if (i_1 == 0): dummy fmpy's to flag inexact
//  r = 1
//
(p9)    fma.s1 FR_poly_hi = FR_r, FR_poly_hi, f0
      nop.i 999
}

{ .mfi
      nop.m 999
//
//  poly_hi = r * poly_hi
//
        fma.s1  FR_poly = FR_Z, FR_poly_lo, FR_c
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p12)   fms.s1  FR_r = f0, f1, FR_r
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      poly_hi = Z * poly_lo + c
//  if i_0 == 1: r = -r
//
        fma.s1  FR_poly = FR_poly, f1, FR_poly_hi
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p12)   fms.s1 FR_Input_X = FR_r, f1, FR_poly
      nop.i 999
}

{ .mfb
      nop.m 999
//
//      poly = poly + poly_hi
//
(p11)   fma.s1 FR_Input_X = FR_r, f1, FR_poly
//
//      if (i_0 == 0) Result = r + poly
//      if (i_0 != 0) Result = r - poly
//
       br.ret.sptk   b0 ;;
}
SINCOS_NORMAL_R:

{ .mii
      nop.m 999
        extr.u  GR_i_1 = GR_N_Inc, 0, 1 ;;
//
//      Set table_ptr1 and table_ptr2 to base address of
//      constant table.
        cmp.eq.unc p9, p10 = 0x0, GR_i_1 ;;
}

{ .mfi
      nop.m 999
        fma.s1  FR_rsq = FR_r, FR_r, f0
        extr.u  GR_i_0 = GR_N_Inc, 1, 1 ;;
}

{ .mfi
      nop.m 999
        frcpa.s1 FR_r_hi, p6 = f1, FR_r
        cmp.eq.unc p11, p12 = 0x0, GR_i_0
}
;;

// ******************************************************************
// ******************************************************************
// ******************************************************************
//
//      r and c have been computed.
//      We known whether this is the sine or cosine routine.
//      Make sure ftz mode is set - should be automatic when using wre
//      Get [i_0,i_1] - two lsb of N_fix_gr alone.
//

{ .mmi
      nop.m 999
      addl           GR_Table_Base   = @ltoff(FSINCOS_CONSTANTS#), gp
      nop.i 999
}
;;

{ .mmi
      ld8 GR_Table_Base = [GR_Table_Base]
      nop.m 999
      nop.i 999
}
;;


{ .mfi
(p10)   add GR_Table_Base = 384, GR_Table_Base
//(p12)   fms.s1 FR_Input_X = f0, f1, f1
(p12)   fms.s1 FR_prelim = f0, f1, f1
(p9)    add GR_Table_Base = 224, GR_Table_Base ;;
}

{ .mmf
      nop.m 999
(p10)   ldfe FR_QQ_8 = [GR_Table_Base], 16
//
//      if (i_1==0) poly = poly * FR_rsq + PP_1_lo
//      else        poly = FR_rsq * poly
//
//(p11)   fma.s1 FR_Input_X = f0, f1, f1 ;;
(p11)   fma.s1 FR_prelim = f0, f1, f1 ;;
}

{ .mmf
(p10)   ldfe FR_QQ_7 = [GR_Table_Base], 16
//
//  Adjust table pointers based on i_0
//      Compute rsq = r * r
//
(p9)    ldfe FR_PP_8 = [GR_Table_Base], 16
        fma.s1 FR_r_cubed = FR_r, FR_rsq, f0 ;;
}

{ .mmf
(p9)    ldfe FR_PP_7 = [GR_Table_Base], 16
(p10)   ldfe FR_QQ_6 = [GR_Table_Base], 16
//
//      Load PP_8 and QQ_8; PP_7 and QQ_7
//
        frcpa.s1 FR_r_hi, p6 = f1, FR_r_hi ;;
}
//
//      if (i_1==0) poly =   PP_7 + FR_rsq * PP_8.
//      else        poly =   QQ_7 + FR_rsq * QQ_8.
//

{ .mmb
(p9)    ldfe FR_PP_6 = [GR_Table_Base], 16
(p10)   ldfe FR_QQ_5 = [GR_Table_Base], 16
      nop.b 999 ;;
}

{ .mmb
(p9)    ldfe FR_PP_5 = [GR_Table_Base], 16
(p10)   ldfe FR_S_1 = [GR_Table_Base], 16
      nop.b 999 ;;
}

{ .mmb
(p10)   ldfe FR_QQ_1 = [GR_Table_Base], 16
(p9)    ldfe FR_C_1 = [GR_Table_Base], 16
      nop.b 999 ;;
}

{ .mmi
(p10)   ldfe FR_QQ_4 = [GR_Table_Base], 16 ;;
(p9)    ldfe FR_PP_1 = [GR_Table_Base], 16
      nop.i 999 ;;
}

{ .mmf
(p10)   ldfe FR_QQ_3 = [GR_Table_Base], 16
//
//      if (i_1=0) corr = corr + c*c
//      else       corr = corr * c
//
(p9)    ldfe FR_PP_4 = [GR_Table_Base], 16
(p10)   fma.s1 FR_poly = FR_rsq, FR_QQ_8, FR_QQ_7 ;;
}
//
//      if (i_1=0) poly = rsq * poly + PP_5
//      else       poly = rsq * poly + QQ_5
//      Load PP_4 or QQ_4
//

{ .mmf
(p9)    ldfe FR_PP_3 = [GR_Table_Base], 16
(p10)   ldfe FR_QQ_2 = [GR_Table_Base], 16
//
//      r_hi =   frcpa(frcpa(r)).
//      r_cube = r * FR_rsq.
//
(p9)    fma.s1 FR_poly = FR_rsq, FR_PP_8, FR_PP_7 ;;
}
//
//      Do dummy multiplies so inexact is always set.
//

{ .mfi
(p9)    ldfe FR_PP_2 = [GR_Table_Base], 16
//
//      r_lo = r - r_hi
//
(p9)    fma.s1 FR_U_lo = FR_r_hi, FR_r_hi, f0
      nop.i 999 ;;
}

{ .mmf
      nop.m 999
(p9)    ldfe FR_PP_1_lo = [GR_Table_Base], 16
(p10)   fma.s1 FR_corr = FR_S_1, FR_r_cubed, FR_r
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly = FR_rsq, FR_poly, FR_QQ_6
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1=0) U_lo = r_hi * r_hi
//      else       U_lo = r_hi + r
//
(p9)    fma.s1 FR_corr = FR_C_1, FR_rsq, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1=0) corr = C_1 * rsq
//      else       corr = S_1 * r_cubed + r
//
(p9)    fma.s1 FR_poly = FR_rsq, FR_poly, FR_PP_6
      nop.i 999
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_U_lo = FR_r_hi, f1, FR_r
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1=0) U_hi = r_hi + U_hi
//      else       U_hi = QQ_1 * U_hi + 1
//
(p9)    fma.s1 FR_U_lo = FR_r, FR_r_hi, FR_U_lo
      nop.i 999
}

{ .mfi
      nop.m 999
//
//      U_hi = r_hi * r_hi
//
        fms.s1 FR_r_lo = FR_r, f1, FR_r_hi
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      Load PP_1, PP_6, PP_5, and C_1
//      Load QQ_1, QQ_6, QQ_5, and S_1
//
        fma.s1 FR_U_hi = FR_r_hi, FR_r_hi, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly = FR_rsq, FR_poly, FR_QQ_5
      nop.i 999
}

{ .mfi
      nop.m 999
(p10)   fnma.s1 FR_corr = FR_corr, FR_c, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1=0) U_lo = r * r_hi + U_lo
//      else       U_lo = r_lo * U_lo
//
(p9)    fma.s1 FR_corr = FR_corr, FR_c, FR_c
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)    fma.s1 FR_poly = FR_rsq, FR_poly, FR_PP_5
      nop.i 999
}

{ .mfi
      nop.m 999
//
//      if (i_1 =0) U_hi = r + U_hi
//      if (i_1 =0) U_lo = r_lo * U_lo
//
//
(p9)    fma.d.s1 FR_PP_5 = FR_PP_5, FR_PP_4, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)    fma.s1 FR_U_lo = FR_r, FR_r, FR_U_lo
      nop.i 999
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_U_lo = FR_r_lo, FR_U_lo, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1=0) poly = poly * rsq + PP_6
//      else       poly = poly * rsq + QQ_6
//
(p9)    fma.s1 FR_U_hi = FR_r_hi, FR_U_hi, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly = FR_rsq, FR_poly, FR_QQ_4
      nop.i 999
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_U_hi = FR_QQ_1, FR_U_hi, f1
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.d.s1 FR_QQ_5 = FR_QQ_5, FR_QQ_5, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1!=0) U_hi = PP_1 * U_hi
//      if (i_1!=0) U_lo = r * r  + U_lo
//      Load PP_3 or QQ_3
//
(p9)    fma.s1 FR_poly = FR_rsq, FR_poly, FR_PP_4
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)    fma.s1 FR_U_lo = FR_r_lo, FR_U_lo, f0
      nop.i 999
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_U_lo = FR_QQ_1,FR_U_lo, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p9)    fma.s1 FR_U_hi = FR_PP_1, FR_U_hi, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly = FR_rsq, FR_poly, FR_QQ_3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      Load PP_2, QQ_2
//
(p9)    fma.s1 FR_poly = FR_rsq, FR_poly, FR_PP_3
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1==0) poly = FR_rsq * poly  + PP_3
//      else        poly = FR_rsq * poly  + QQ_3
//      Load PP_1_lo
//
(p9)    fma.s1 FR_U_lo = FR_PP_1, FR_U_lo, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1 =0) poly = poly * rsq + pp_r4
//      else        poly = poly * rsq + qq_r4
//
(p9)    fma.s1 FR_U_hi = FR_r, f1, FR_U_hi
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly = FR_rsq, FR_poly, FR_QQ_2
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1==0) U_lo =  PP_1_hi * U_lo
//      else        U_lo =  QQ_1 * U_lo
//
(p9)    fma.s1 FR_poly = FR_rsq, FR_poly, FR_PP_2
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_0==0)  Result = 1
//      else         Result = -1
//
        fma.s1 FR_V = FR_U_lo, f1, FR_corr
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly = FR_rsq, FR_poly, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1==0) poly =  FR_rsq * poly + PP_2
//      else poly =  FR_rsq * poly + QQ_2
//
(p9)    fma.s1 FR_poly = FR_rsq, FR_poly, FR_PP_1_lo
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
(p10)   fma.s1 FR_poly = FR_rsq, FR_poly, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      V = U_lo + corr
//
(p9)    fma.s1 FR_poly = FR_r_cubed, FR_poly, f0
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//
//      if (i_1==0) poly = r_cube * poly
//      else        poly = FR_rsq * poly
//
        fma.s1  FR_V = FR_poly, f1, FR_V
      nop.i 999 ;;
}

{ .mfi
      nop.m 999
//(p12)   fms.s1 FR_Input_X = FR_Input_X, FR_U_hi, FR_V
(p12)   fms.s1 FR_Input_X = FR_prelim, FR_U_hi, FR_V
      nop.i 999
}

{ .mfb
      nop.m 999
//
//      V = V + poly
//
//(p11)   fma.s1 FR_Input_X = FR_Input_X, FR_U_hi, FR_V
(p11)   fma.s1 FR_Input_X = FR_prelim, FR_U_hi, FR_V
//
//      if (i_0==0) Result = Result * U_hi + V
//      else        Result = Result * U_hi - V
//
       br.ret.sptk   b0 ;;
}

//
//      If cosine, FR_Input_X = 1
//      If sine, FR_Input_X = +/-Zero (Input FR_Input_X)
//      Results are exact, no exceptions
//
SINCOS_ZERO:

{ .mmb
        cmp.eq.unc p6, p7 = 0x1, GR_Sin_or_Cos
      nop.m 999
      nop.b 999 ;;
}

{ .mfi
      nop.m 999
(p7)    fmerge.s FR_Input_X = FR_Input_X, FR_Input_X
      nop.i 999
}

{ .mfb
      nop.m 999
(p6)    fmerge.s FR_Input_X = f1, f1
       br.ret.sptk   b0 ;;
}

SINCOS_SPECIAL:

//
//      Path for Arg = +/- QNaN, SNaN, Inf
//      Invalid can be raised. SNaNs
//      become QNaNs
//

{ .mfb
      nop.m 999
        fmpy.s1 FR_Input_X = FR_Input_X, f0
        br.ret.sptk   b0 ;;
}
GLOBAL_LIBM_END(__libm_cos_large)


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
//     Be sure to allocate at least 2 GP registers as output registers for
//     __libm_pi_by_2_reduce.  This routine uses r49-50. These are used as
//     scratch registers within the __libm_pi_by_2_reduce routine (for speed).
//
//     We know also that __libm_pi_by_2_reduce preserves f10-15, f71-127.  We
//     use this to eliminate save/restore of key fp registers in this calling
//     function.
//
// *******************************************************************
// *******************************************************************
// *******************************************************************

LOCAL_LIBM_ENTRY(__libm_callout_2)
SINCOS_ARG_TOO_LARGE:

.prologue
//      Readjust Table ptr
{ .mfi
        adds  GR_Table_Base1 = -16, GR_Table_Base1
        nop.f 999
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
};;

{ .mmi
        ldfs FR_Two_to_M3 = [GR_Table_Base1],4
        mov GR_SAVE_GP=gp                       // Save gp
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
        ldfs FR_Neg_Two_to_M3 = [GR_Table_Base1],0
        nop.i 0
        br.call.sptk b0=__libm_pi_by_2_reduce#
};;

{ .mfi
        add   GR_N_Inc = GR_Sin_or_Cos,r8
        fcmp.lt.unc.s1  p6, p0 = FR_r, FR_Two_to_M3
        mov   b0 = GR_SAVE_B0                  // Restore return address
};;

{ .mfi
        mov   gp = GR_SAVE_GP                  // Restore gp
(p6)    fcmp.gt.unc.s1  p6, p0 = FR_r, FR_Neg_Two_to_M3
        mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
};;

{ .mbb
        nop.m 999
(p6)    br.cond.spnt SINCOS_SMALL_R            // Branch if |r| < 1/4
        br.cond.sptk SINCOS_NORMAL_R ;;        // Branch if 1/4 <= |r| < pi/4
}

LOCAL_LIBM_END(__libm_callout_2)

.type   __libm_pi_by_2_reduce#,@function
.global __libm_pi_by_2_reduce#
