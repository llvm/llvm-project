.file "atanl.s"


// Copyright (c) 2000 - 2005, Intel Corporation
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
//
//*********************************************************************
//
// History
// 02/02/00 (hand-optimized)
// 04/04/00 Unwind support added
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 03/13/01 Fixed flags when denormal raised on intermediate result
// 01/08/02 Improved speed.
// 02/06/02 Corrected .section statement
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double table values
// 03/31/05 Reformatted delimiters between data tables
//
//*********************************************************************
//
// Function:   atanl(x) = inverse tangent(x), for double extended x values
// Function:   atan2l(y,x) = atan(y/x), for double extended y, x values
//
// API
//
//  long double atanl  (long double x)
//  long double atan2l (long double y, long double x)
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers: f8 (Input and Return Value)
//                              f9 (Input for atan2l)
//                              f10-f15, f32-f83
//
//    General Purpose Registers:
//      r32-r51
//      r49-r52 (Arguments to error support for 0,0 case)
//
//    Predicate Registers:      p6-p15
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    Denormal fault raised on denormal inputs
//    Underflow exceptions may occur
//    Special error handling for the y=0 and x=0 case
//    Inexact raised when appropriate by algorithm
//
//    atanl(SNaN) = QNaN
//    atanl(QNaN) = QNaN
//    atanl(+/-0) = +/- 0
//    atanl(+/-Inf) = +/-pi/2
//
//    atan2l(Any NaN for x or y) = QNaN
//    atan2l(+/-0,x) = +/-0 for x > 0
//    atan2l(+/-0,x) = +/-pi for x < 0
//    atan2l(+/-0,+0) = +/-0
//    atan2l(+/-0,-0) = +/-pi
//    atan2l(y,+/-0) = pi/2 y > 0
//    atan2l(y,+/-0) = -pi/2 y < 0
//    atan2l(+/-y, Inf) = +/-0 for finite y > 0
//    atan2l(+/-Inf, x) = +/-pi/2 for finite x
//    atan2l(+/-y, -Inf) = +/-pi for finite  y > 0
//    atan2l(+/-Inf, Inf) = +/-pi/4
//    atan2l(+/-Inf, -Inf) = +/-3pi/4
//
//*********************************************************************
//
// Mathematical Description
// ---------------------------
//
// The function ATANL( Arg_Y, Arg_X ) returns the "argument"
// or the "phase" of the complex number
//
//           Arg_X + i Arg_Y
//
// or equivalently, the angle in radians from the positive
// x-axis to the line joining the origin and the point
// (Arg_X,Arg_Y)
//
//
//        (Arg_X, Arg_Y) x
//                        \
//                \
//                 \
//                  \
//                   \ angle between is ATANL(Arg_Y,Arg_X)




//                    \
//                     ------------------> X-axis

//                   Origin
//
// Moreover, this angle is reported in the range [-pi,pi] thus
//
//      -pi <= ATANL( Arg_Y, Arg_X ) <= pi.
//
// From the geometry, it is easy to define ATANL when one of
// Arg_X or Arg_Y is +-0 or +-inf:
//
//
//      \ Y |
//     X \  |  +0  | -0  |  +inf |  -inf  |  finite non-zero
//        \ |      |     |       |        |
//    ______________________________________________________
//          |            |       |        |
//     +-0  |   Invalid/ |  pi/2 | -pi/2  |  sign(Y)*pi/2
//          |    qNaN    |       |        |
//  --------------------------------------------------------
//          |      |     |       |        |
//     +inf |  +0  | -0  |  pi/4 | -pi/4  |  sign(Y)*0
//  --------------------------------------------------------
//          |      |     |       |        |
//     -inf |  +pi | -pi | 3pi/4 | -3pi/4 |  sign(Y)*pi
//  --------------------------------------------------------
//   finite |    X>0?    |  pi/2 | -pi/2  |  normal case
//  non-zero| sign(Y)*0: |       |        |
//       | sign(Y)*pi |       |        |
//
//
// One must take note that ATANL is NOT the arctangent of the
// value Arg_Y/Arg_X; but rather ATANL and arctan are related
// in a slightly more complicated way as follows:
//
// Let U := max(|Arg_X|, |Arg_Y|);  V := min(|Arg_X|, |Arg_Y|);
// sign_X be the sign bit of Arg_X, i.e., sign_X is 0 or 1;
// s_X    be the sign     of Arg_X, i.e., s_X = (-1)^sign_X;
//
// sign_Y be the sign bit of Arg_Y, i.e., sign_Y is 0 or 1;
// s_Y    be the sign     of Arg_Y, i.e., s_Y = (-1)^sign_Y;
//
// swap   be 0  if |Arg_X| >= |Arg_Y|  and 1 otherwise.
//
// Then, ATANL(Arg_Y, Arg_X) =
//
//       /    arctan(V/U)     \      sign_X = 0 & swap = 0
//       | pi/2 - arctan(V/U) |      sign_X = 0 & swap = 1
// s_Y * |                    |
//       |  pi  - arctan(V/U) |      sign_X = 1 & swap = 0
//       \ pi/2 + arctan(V/U) /      sign_X = 1 & swap = 1
//
//
// This relationship also suggest that the algorithm's major
// task is to calculate arctan(V/U) for 0 < V <= U; and the
// final Result is given by
//
//      s_Y * { (P_hi + P_lo) + sigma * arctan(V/U) }
//
// where
//
//   (P_hi,P_lo) represents M(sign_X,swap)*(pi/2) accurately
//
//   M(sign_X,swap) = 0  for sign_X = 0 and swap = 0
//              1  for swap   = 1
//              2  for sign_X = 1 and swap = 0
//
// and
//
//   sigma = { (sign_X  XOR  swap) :  -1.0 : 1.0 }
//
//      =  (-1) ^ ( sign_X XOR swap )
//
// Both (P_hi,P_lo) and sigma can be stored in a table and fetched
// using (sign_X,swap) as an index. (P_hi, P_lo) can be stored as a
// double-precision, and single-precision pair; and sigma can
// obviously be just a single-precision number.
//
// In the algorithm we propose, arctan(V/U) is calculated to high accuracy
// as A_hi + A_lo. Consequently, the Result ATANL( Arg_Y, Arg_X ) is
// given by
//
//    s_Y*P_hi + s_Y*sigma*A_hi + s_Y*(sigma*A_lo + P_lo)
//
// We now discuss the calculation of arctan(V/U) for 0 < V <= U.
//
// For (V/U) < 2^(-3), we use a simple polynomial of the form
//
//      z + z^3*(P_1 + z^2*(P_2 + z^2*(P_3 + ... + P_8)))
//
// where z = V/U.
//
// For the sake of accuracy, the first term "z" must approximate V/U to
// extra precision. For z^3 and higher power, a working precision
// approximation to V/U suffices. Thus, we obtain:
//
//      z_hi + z_lo = V/U  to extra precision and
//      z           = V/U  to working precision
//
// The value arctan(V/U) is delivered as two pieces (A_hi, A_lo)
//
//      (A_hi,A_lo) = (z_hi, z^3*(P_1 + ... + P_8) + z_lo).
//
//
// For 2^(-3) <= (V/U) <= 1, we use a table-driven approach.
// Consider
//
//      (V/U) = 2^k * 1.b_1 b_2 .... b_63 b_64 b_65 ....
//
// Define
//
//       z_hi = 2^k * 1.b_1 b_2 b_3 b_4 1
//
// then
//                                            /                \
//                                            |  (V/U) - z_hi  |

//      arctan(V/U) = arctan(z_hi) + acrtan| -------------- |
//                                            | 1 + (V/U)*z_hi |
//                                            \                /
//
//                                            /                \
//                                            |   V - z_hi*U   |

//                  = arctan(z_hi) + acrtan| -------------- |
//                                            |   U + V*z_hi   |
//                                            \                /
//
//                  = arctan(z_hi) + acrtan( V' / U' )
//
//
// where
//
//      V' = V - U*z_hi;   U' = U + V*z_hi.
//
// Let
//
//      w_hi + w_lo  = V'/U' to extra precision and
//           w       = V'/U' to working precision
//
// then we can approximate arctan(V'/U') by
//
//      arctan(V'/U') = w_hi + w_lo
//                     + w^3*(Q_1 + w^2*(Q_2 + w^2*(Q_3 + w^2*Q_4)))
//
//                       = w_hi + w_lo + poly
//
// Finally, arctan(z_hi) is calculated beforehand and stored in a table
// as Tbl_hi, Tbl_lo. Thus,
//
//      (A_hi, A_lo) = (Tbl_hi, w_hi+(poly+(w_lo+Tbl_lo)))
//
// This completes the mathematical description.
//
//
// Algorithm
// -------------
//
// Step 0. Check for unsupported format.
//
//    If
//       ( expo(Arg_X) not zero AND msb(Arg_X) = 0 ) OR
//       ( expo(Arg_Y) not zero AND msb(Arg_Y) = 0 )
//
//    then one of the arguments is unsupported. Generate an
//    invalid and return qNaN.
//
// Step 1. Initialize
//
//    Normalize Arg_X and Arg_Y and set the following
//
//    sign_X :=  sign_bit(Arg_X)
//    s_Y    := (sign_bit(Arg_Y)==0? 1.0 : -1.0)
//    swap   := (|Arg_X| >= |Arg_Y|?   0 :  1  )
//    U      := max( |Arg_X|, |Arg_Y| )
//    V      := min( |Arg_X|, |Arg_Y| )
//
//    execute: frcpa E, pred, V, U
//    If pred is 0, go to Step 5 for special cases handling.
//
// Step 2. Decide on branch.
//
//    Q := E * V
//    If Q < 2^(-3) go to Step 4 for simple polynomial case.
//
// Step 3. Table-driven algorithm.
//
//    Q is represented as
//
//      2^(-k) * 1.b_1 b_2 b_3 ... b_63; k = 0,-1,-2,-3
//
// and that if k = 0, b_1 = b_2 = b_3 = b_4 = 0.
//
// Define
//
//      z_hi := 2^(-k) * 1.b_1 b_2 b_3 b_4 1
//
// (note that there are 49 possible values of z_hi).
//
//      ...We now calculate V' and U'. While V' is representable
//      ...as a 64-bit number because of cancellation, U' is
//      ...not in general a 64-bit number. Obtaining U' accurately
//      ...requires two working precision numbers
//
//      U_prime_hi := U + V * z_hi            ...WP approx. to U'
//      U_prime_lo := ( U - U_prime_hi ) + V*z_hi ...observe order
//      V_prime    := V - U * z_hi             ...this is exact
//
//         C_hi := frcpa (1.0, U_prime_hi)  ...C_hi approx 1/U'_hi
//
//      loop 3 times
//         C_hi := C_hi + C_hi*(1.0 - C_hi*U_prime_hi)
//
//      ...at this point C_hi is (1/U_prime_hi) to roughly 64 bits
//
//      w_hi := V_prime * C_hi     ...w_hi is V_prime/U_prime to
//                     ...roughly working precision
//
//         ...note that we want w_hi + w_lo to approximate
//      ...V_prime/(U_prime_hi + U_prime_lo) to extra precision
//         ...but for now, w_hi is good enough for the polynomial
//      ...calculation.
//
//         wsq  := w_hi*w_hi
//      poly := w_hi*wsq*(Q_1 + wsq*(Q_2 + wsq*(Q_3 + wsq*Q_4)))
//
//      Fetch
//      (Tbl_hi, Tbl_lo) = atan(z_hi) indexed by (k,b_1,b_2,b_3,b_4)
//      ...Tbl_hi is a double-precision number
//      ...Tbl_lo is a single-precision number
//
//         (P_hi, P_lo) := M(sign_X,swap)*(Pi_by_2_hi, Pi_by_2_lo)
//      ...as discussed previous. Again; the implementation can
//      ...chose to fetch P_hi and P_lo from a table indexed by
//      ...(sign_X, swap).
//      ...P_hi is a double-precision number;
//      ...P_lo is a single-precision number.
//
//      ...calculate w_lo so that w_hi + w_lo is V'/U' accurately
//         w_lo := ((V_prime - w_hi*U_prime_hi) -
//              w_hi*U_prime_lo) * C_hi     ...observe order
//
//
//      ...Ready to deliver arctan(V'/U') as A_hi, A_lo
//      A_hi := Tbl_hi
//      A_lo := w_hi + (poly + (Tbl_lo + w_lo)) ...observe order
//
//      ...Deliver final Result
//      ...s_Y*P_hi + s_Y*sigma*A_hi + s_Y*(sigma*A_lo + P_lo)
//
//      sigma := ( (sign_X XOR swap) ? -1.0 : 1.0 )
//      ...sigma can be obtained by a table lookup using
//      ...(sign_X,swap) as index and stored as single precision
//         ...sigma should be calculated earlier
//
//      P_hi := s_Y*P_hi
//      A_hi := s_Y*A_hi
//
//      Res_hi := P_hi + sigma*A_hi     ...this is exact because
//                          ...both P_hi and Tbl_hi
//                          ...are double-precision
//                          ...and |Tbl_hi| > 2^(-4)
//                          ...P_hi is either 0 or
//                          ...between (1,4)
//
//      Res_lo := sigma*A_lo + P_lo
//
//      Return Res_hi + s_Y*Res_lo in user-defined rounding control
//
// Step 4. Simple polynomial case.
//
//    ...E and Q are inherited from Step 2.
//
//    A_hi := Q     ...Q is inherited from Step 2 Q approx V/U
//
//    loop 3 times
//       E := E + E2(1.0 - E*U1
//    ...at this point E approximates 1/U to roughly working precision
//
//    z := V * E     ...z approximates V/U to roughly working precision
//    zsq := z * z
//    z4 := zsq * zsq; z8 := z4 * z4
//
//    poly1 := P_4 + zsq*(P_5 + zsq*(P_6 + zsq*(P_7 + zsq*P_8)))
//    poly2 := zsq*(P_1 + zsq*(P_2 + zsq*P_3))
//
//    poly  := poly1 + z8*poly2
//
//    z_lo := (V - A_hi*U)*E
//
//    A_lo := z*poly + z_lo
//    ...A_hi, A_lo approximate arctan(V/U) accurately
//
//    (P_hi, P_lo) := M(sign_X,swap)*(Pi_by_2_hi, Pi_by_2_lo)
//    ...one can store the M(sign_X,swap) as single precision
//    ...values
//
//    ...Deliver final Result
//    ...s_Y*P_hi + s_Y*sigma*A_hi + s_Y*(sigma*A_lo + P_lo)
//
//    sigma := ( (sign_X XOR swap) ? -1.0 : 1.0 )
//    ...sigma can be obtained by a table lookup using
//    ...(sign_X,swap) as index and stored as single precision
//    ...sigma should be calculated earlier
//
//    P_hi := s_Y*P_hi
//    A_hi := s_Y*A_hi
//
//    Res_hi := P_hi + sigma*A_hi          ...need to compute
//                          ...P_hi + sigma*A_hi
//                          ...exactly
//
//    tmp    := (P_hi - Res_hi) + sigma*A_hi
//
//    Res_lo := s_Y*(sigma*A_lo + P_lo) + tmp
//
//    Return Res_hi + Res_lo in user-defined rounding control
//
// Step 5. Special Cases
//
//    These are detected early in the function by fclass instructions.
//
//    We are in one of those special cases when X or Y is 0,+-inf or NaN
//
//    If one of X and Y is NaN, return X+Y (which will generate
//    invalid in case one is a signaling NaN). Otherwise,
//    return the Result as described in the table
//
//
//
//      \ Y |
//     X \  |  +0  | -0  |  +inf |  -inf  |  finite non-zero
//        \ |      |     |       |        |
//    ______________________________________________________
//          |            |       |        |
//     +-0  |   Invalid/ |  pi/2 | -pi/2  |  sign(Y)*pi/2
//          |    qNaN    |       |        |
//  --------------------------------------------------------
//          |      |     |       |        |
//     +inf |  +0  | -0  |  pi/4 | -pi/4  |  sign(Y)*0
//  --------------------------------------------------------
//          |      |     |       |        |
//     -inf |  +pi | -pi | 3pi/4 | -3pi/4 |  sign(Y)*pi
//  --------------------------------------------------------
//   finite |    X>0?    |  pi/2 | -pi/2  |
//  non-zero| sign(Y)*0: |       |        |      N/A
//       | sign(Y)*pi |       |        |
//
//

ArgY_orig   =   f8
Result      =   f8
FR_RESULT   =   f8
ArgX_orig   =   f9
ArgX        =   f10
FR_X        =   f10
ArgY        =   f11
FR_Y        =   f11
s_Y         =   f12
U           =   f13
V           =   f14
E           =   f15
Q           =   f32
z_hi        =   f33
U_prime_hi  =   f34
U_prime_lo  =   f35
V_prime     =   f36
C_hi        =   f37
w_hi        =   f38
w_lo        =   f39
wsq         =   f40
poly        =   f41
Tbl_hi      =   f42
Tbl_lo      =   f43
P_hi        =   f44
P_lo        =   f45
A_hi        =   f46
A_lo        =   f47
sigma       =   f48
Res_hi      =   f49
Res_lo      =   f50
Z           =   f52
zsq         =   f53
z4          =   f54
z8          =   f54
poly1       =   f55
poly2       =   f56
z_lo        =   f57
tmp         =   f58
P_1         =   f59
Q_1         =   f60
P_2         =   f61
Q_2         =   f62
P_3         =   f63
Q_3         =   f64
P_4         =   f65
Q_4         =   f66
P_5         =   f67
P_6         =   f68
P_7         =   f69
P_8         =   f70
U_hold      =   f71
TWO_TO_NEG3 =   f72
C_hi_hold   =   f73
E_hold      =   f74
M           =   f75
ArgX_abs    =   f76
ArgY_abs    =   f77
Result_lo   =   f78
A_temp      =   f79
FR_temp     =   f80
Xsq         =   f81
Ysq         =   f82
tmp_small   =   f83

GR_SAVE_PFS   = r33
GR_SAVE_B0    = r34
GR_SAVE_GP    = r35
sign_X        = r36
sign_Y        = r37
swap          = r38
table_ptr1    = r39
table_ptr2    = r40
k             = r41
lookup        = r42
exp_ArgX      = r43
exp_ArgY      = r44
exponent_Q    = r45
significand_Q = r46
special       = r47
sp_exp_Q      = r48
sp_exp_4sig_Q = r49
table_base    = r50
int_temp      = r51

GR_Parameter_X      = r49
GR_Parameter_Y      = r50
GR_Parameter_RESULT = r51
GR_Parameter_TAG    = r52
GR_temp             = r52

RODATA
.align 16

LOCAL_OBJECT_START(Constants_atan)
//       double pi/2
data8 0x3FF921FB54442D18
//       single lo_pi/2, two**(-3)
data4 0x248D3132, 0x3E000000
data8 0xAAAAAAAAAAAAAAA3, 0xBFFD // P_1
data8 0xCCCCCCCCCCCC54B2, 0x3FFC // P_2
data8 0x9249249247E4D0C2, 0xBFFC // P_3
data8 0xE38E38E058870889, 0x3FFB // P_4
data8 0xBA2E895B290149F8, 0xBFFB // P_5
data8 0x9D88E6D4250F733D, 0x3FFB // P_6
data8 0x884E51FFFB8745A0, 0xBFFB // P_7
data8 0xE1C7412B394396BD, 0x3FFA // P_8
data8 0xAAAAAAAAAAAAA52F, 0xBFFD // Q_1
data8 0xCCCCCCCCC75B60D3, 0x3FFC // Q_2
data8 0x924923AD011F1940, 0xBFFC // Q_3
data8 0xE36F716D2A5F89BD, 0x3FFB // Q_4
//
//    Entries Tbl_hi  (double precision)
//    B = 1+Index/16+1/32  Index = 0
//    Entries Tbl_lo (single precision)
//    B = 1+Index/16+1/32  Index = 0
//
data8 0x3FE9A000A935BD8E
data4 0x23ACA08F, 0x00000000
//
//    Entries Tbl_hi  (double precision) Index = 0,1,...,15
//    B = 2^(-1)*(1+Index/16+1/32)
//    Entries Tbl_lo (single precision)
//    Index = 0,1,...,15  B = 2^(-1)*(1+Index/16+1/32)
//
data8 0x3FDE77EB7F175A34
data4 0x238729EE, 0x00000000
data8 0x3FE0039C73C1A40B
data4 0x249334DB, 0x00000000
data8 0x3FE0C6145B5B43DA
data4 0x22CBA7D1, 0x00000000
data8 0x3FE1835A88BE7C13
data4 0x246310E7, 0x00000000
data8 0x3FE23B71E2CC9E6A
data4 0x236210E5, 0x00000000
data8 0x3FE2EE628406CBCA
data4 0x2462EAF5, 0x00000000
data8 0x3FE39C391CD41719
data4 0x24B73EF3, 0x00000000
data8 0x3FE445065B795B55
data4 0x24C11260, 0x00000000
data8 0x3FE4E8DE5BB6EC04
data4 0x242519EE, 0x00000000
data8 0x3FE587D81F732FBA
data4 0x24D4346C, 0x00000000
data8 0x3FE6220D115D7B8D
data4 0x24ED487B, 0x00000000
data8 0x3FE6B798920B3D98
data4 0x2495FF1E, 0x00000000
data8 0x3FE748978FBA8E0F
data4 0x223D9531, 0x00000000
data8 0x3FE7D528289FA093
data4 0x242B0411, 0x00000000
data8 0x3FE85D69576CC2C5
data4 0x2335B374, 0x00000000
data8 0x3FE8E17AA99CC05D
data4 0x24C27CFB, 0x00000000
//
//    Entries Tbl_hi  (double precision) Index = 0,1,...,15
//    B = 2^(-2)*(1+Index/16+1/32)
//    Entries Tbl_lo (single precision)
//    Index = 0,1,...,15  B = 2^(-2)*(1+Index/16+1/32)
//
data8 0x3FD025FA510665B5
data4 0x24263482, 0x00000000
data8 0x3FD1151A362431C9
data4 0x242C8DC9, 0x00000000
data8 0x3FD2025567E47C95
data4 0x245CF9BA, 0x00000000
data8 0x3FD2ED987A823CFE
data4 0x235C892C, 0x00000000
data8 0x3FD3D6D129271134
data4 0x2389BE52, 0x00000000
data8 0x3FD4BDEE586890E6
data4 0x24436471, 0x00000000
data8 0x3FD5A2E0175E0F4E
data4 0x2389DBD4, 0x00000000
data8 0x3FD685979F5FA6FD
data4 0x2476D43F, 0x00000000
data8 0x3FD7660752817501
data4 0x24711774, 0x00000000
data8 0x3FD84422B8DF95D7
data4 0x23EBB501, 0x00000000
data8 0x3FD91FDE7CD0C662
data4 0x23883A0C, 0x00000000
data8 0x3FD9F93066168001
data4 0x240DF63F, 0x00000000
data8 0x3FDAD00F5422058B
data4 0x23FE261A, 0x00000000
data8 0x3FDBA473378624A5
data4 0x23A8CD0E, 0x00000000
data8 0x3FDC76550AAD71F8
data4 0x2422D1D0, 0x00000000
data8 0x3FDD45AEC9EC862B
data4 0x2344A109, 0x00000000
//
//    Entries Tbl_hi  (double precision) Index = 0,1,...,15
//    B = 2^(-3)*(1+Index/16+1/32)
//    Entries Tbl_lo (single precision)
//    Index = 0,1,...,15  B = 2^(-3)*(1+Index/16+1/32)
//
data8 0x3FC068D584212B3D
data4 0x239874B6, 0x00000000
data8 0x3FC1646541060850
data4 0x2335E774, 0x00000000
data8 0x3FC25F6E171A535C
data4 0x233E36BE, 0x00000000
data8 0x3FC359E8EDEB99A3
data4 0x239680A3, 0x00000000
data8 0x3FC453CEC6092A9E
data4 0x230FB29E, 0x00000000
data8 0x3FC54D18BA11570A
data4 0x230C1418, 0x00000000
data8 0x3FC645BFFFB3AA73
data4 0x23F0564A, 0x00000000
data8 0x3FC73DBDE8A7D201
data4 0x23D4A5E1, 0x00000000
data8 0x3FC8350BE398EBC7
data4 0x23D4ADDA, 0x00000000
data8 0x3FC92BA37D050271
data4 0x23BCB085, 0x00000000
data8 0x3FCA217E601081A5
data4 0x23BC841D, 0x00000000
data8 0x3FCB1696574D780B
data4 0x23CF4A8E, 0x00000000
data8 0x3FCC0AE54D768466
data4 0x23BECC90, 0x00000000
data8 0x3FCCFE654E1D5395
data4 0x2323DCD2, 0x00000000
data8 0x3FCDF110864C9D9D
data4 0x23F53F3A, 0x00000000
data8 0x3FCEE2E1451D980C
data4 0x23CCB11F, 0x00000000
//
data8 0x400921FB54442D18, 0x3CA1A62633145C07 // PI two doubles
data8 0x3FF921FB54442D18, 0x3C91A62633145C07 // PI_by_2 two dbles
data8 0x3FE921FB54442D18, 0x3C81A62633145C07 // PI_by_4 two dbles
data8 0x4002D97C7F3321D2, 0x3C9A79394C9E8A0A // 3PI_by_4 two dbles
LOCAL_OBJECT_END(Constants_atan)


.section .text
GLOBAL_IEEE754_ENTRY(atanl)

// Use common code with atan2l after setting x=1.0
{ .mfi
      alloc r32 = ar.pfs, 0, 17, 4, 0
      fma.s1 Ysq = ArgY_orig, ArgY_orig, f0          // Form y*y
      nop.i 999
}
{ .mfi
      addl table_ptr1 = @ltoff(Constants_atan#), gp  // Address of table pointer
      fma.s1 Xsq = f1, f1, f0                        // Form x*x
      nop.i 999
}
;;

{ .mfi
      ld8 table_ptr1 = [table_ptr1]                  // Get table pointer
      fnorm.s1 ArgY = ArgY_orig
      nop.i 999
}
{ .mfi
      nop.m 999
      fnorm.s1 ArgX = f1
      nop.i 999
}
;;

{ .mfi
      getf.exp sign_X = f1               // Get signexp of x
      fmerge.s ArgX_abs = f0, f1         // Form |x|
      nop.i 999
}
{ .mfi
      nop.m 999
      fnorm.s1 ArgX_orig = f1
      nop.i 999
}
;;

{ .mfi
      getf.exp sign_Y = ArgY_orig        // Get signexp of y
      fmerge.s ArgY_abs = f0, ArgY_orig  // Form |y|
      mov table_base = table_ptr1        // Save base pointer to tables
}
;;

{ .mfi
      ldfd P_hi = [table_ptr1],8         // Load double precision hi part of pi
      fclass.m p8,p0 = ArgY_orig, 0x1e7  // Test y natval, nan, inf, zero
      nop.i 999
}
;;

{ .mfi
      ldfps P_lo, TWO_TO_NEG3 = [table_ptr1], 8 // Load P_lo and constant 2^-3
      nop.f 999
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 M = f1, f1, f0              // Set M = 1.0
      nop.i 999
}
;;

//
//     Check for everything - if false, then must be pseudo-zero
//     or pseudo-nan (IA unsupporteds).
//
{ .mfb
      nop.m 999
      fclass.m p0,p12 = f1, 0x1FF        // Test x unsupported
(p8)  br.cond.spnt ATANL_Y_SPECIAL       // Branch if y natval, nan, inf, zero
}
;;

//     U = max(ArgX_abs,ArgY_abs)
//     V = min(ArgX_abs,ArgY_abs)
{ .mfi
      nop.m 999
      fcmp.ge.s1 p6,p7 = Xsq, Ysq        // Test for |x| >= |y| using squares
      nop.i 999
}
{ .mfb
      nop.m 999
      fma.s1 V = ArgX_abs, f1, f0        // Set V assuming |x| < |y|
      br.cond.sptk ATANL_COMMON          // Branch to common code
}
;;

GLOBAL_IEEE754_END(atanl)
libm_alias_ldouble_other (__atan, atan)

GLOBAL_IEEE754_ENTRY(atan2l)

{ .mfi
      alloc r32 = ar.pfs, 0, 17, 4, 0
      fma.s1 Ysq = ArgY_orig, ArgY_orig, f0          // Form y*y
      nop.i 999
}
{ .mfi
      addl table_ptr1 = @ltoff(Constants_atan#), gp  // Address of table pointer
      fma.s1 Xsq = ArgX_orig, ArgX_orig, f0          // Form x*x
      nop.i 999
}
;;

{ .mfi
      ld8 table_ptr1 = [table_ptr1]                  // Get table pointer
      fnorm.s1 ArgY = ArgY_orig
      nop.i 999
}
{ .mfi
      nop.m 999
      fnorm.s1 ArgX = ArgX_orig
      nop.i 999
}
;;

{ .mfi
      getf.exp sign_X = ArgX_orig        // Get signexp of x
      fmerge.s ArgX_abs = f0, ArgX_orig  // Form |x|
      nop.i 999
}
;;

{ .mfi
      getf.exp sign_Y = ArgY_orig        // Get signexp of y
      fmerge.s ArgY_abs = f0, ArgY_orig  // Form |y|
      mov table_base = table_ptr1        // Save base pointer to tables
}
;;

{ .mfi
      ldfd P_hi = [table_ptr1],8         // Load double precision hi part of pi
      fclass.m p8,p0 = ArgY_orig, 0x1e7  // Test y natval, nan, inf, zero
      nop.i 999
}
;;

{ .mfi
      ldfps P_lo, TWO_TO_NEG3 = [table_ptr1], 8 // Load P_lo and constant 2^-3
      fclass.m p9,p0 = ArgX_orig, 0x1e7  // Test x natval, nan, inf, zero
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 M = f1, f1, f0              // Set M = 1.0
      nop.i 999
}
;;

//
//     Check for everything - if false, then must be pseudo-zero
//     or pseudo-nan (IA unsupporteds).
//
{ .mfb
      nop.m 999
      fclass.m p0,p12 = ArgX_orig, 0x1FF // Test x unsupported
(p8)  br.cond.spnt ATANL_Y_SPECIAL       // Branch if y natval, nan, inf, zero
}
;;

//     U = max(ArgX_abs,ArgY_abs)
//     V = min(ArgX_abs,ArgY_abs)
{ .mfi
      nop.m 999
      fcmp.ge.s1 p6,p7 = Xsq, Ysq        // Test for |x| >= |y| using squares
      nop.i 999
}
{ .mfb
      nop.m 999
      fma.s1 V = ArgX_abs, f1, f0        // Set V assuming |x| < |y|
(p9)  br.cond.spnt ATANL_X_SPECIAL       // Branch if x natval, nan, inf, zero
}
;;

// Now common code for atanl and atan2l
ATANL_COMMON:
{ .mfi
      nop.m 999
      fclass.m p0,p13 = ArgY_orig, 0x1FF // Test y unsupported
      shr sign_X = sign_X, 17            // Get sign bit of x
}
{ .mfi
      nop.m 999
      fma.s1 U = ArgY_abs, f1, f0        // Set U assuming |x| < |y|
      adds table_ptr1 = 176, table_ptr1  // Point to Q4
}
;;

{ .mfi
(p6)  add swap = r0, r0                  // Set swap=0 if |x| >= |y|
(p6)  frcpa.s1 E, p0 = ArgY_abs, ArgX_abs // Compute E if |x| >= |y|
      shr sign_Y = sign_Y, 17            // Get sign bit of y
}
{ .mfb
      nop.m 999
(p6)  fma.s1 V = ArgY_abs, f1, f0        // Set V if |x| >= |y|
(p12) br.cond.spnt ATANL_UNSUPPORTED     // Branch if x unsupported
}
;;

// Set p8 if y >=0
// Set p9 if y < 0
// Set p10 if |x| >= |y| and x >=0
// Set p11 if |x| >= |y| and x < 0
{ .mfi
      cmp.eq p8, p9 = 0, sign_Y          // Test for y >= 0
(p7)  frcpa.s1 E, p0 = ArgX_abs, ArgY_abs // Compute E if |x| < |y|
(p7)  add swap = 1, r0                   // Set swap=1 if |x| < |y|
}
{ .mfb
(p6)  cmp.eq.unc p10, p11 = 0, sign_X    // If |x| >= |y|, test for x >= 0
(p6)  fma.s1 U = ArgX_abs, f1, f0        // Set U if |x| >= |y|
(p13) br.cond.spnt ATANL_UNSUPPORTED     // Branch if y unsupported
}
;;

//
//     if p8, s_Y = 1.0
//     if p9, s_Y = -1.0
//
.pred.rel "mutex",p8,p9
{ .mfi
      nop.m 999
(p8)  fadd.s1 s_Y = f0, f1               // If y >= 0 set s_Y = 1.0
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fsub.s1 s_Y = f0, f1               // If y < 0 set s_Y = -1.0
      nop.i 999
}
;;

.pred.rel "mutex",p10,p11
{ .mfi
      nop.m 999
(p10) fsub.s1 M = M, f1                  // If |x| >= |y| and x >=0, set M=0
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fadd.s1 M = M, f1                  // If |x| >= |y| and x < 0, set M=2.0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fcmp.eq.s0 p0, p9 = ArgX_orig, ArgY_orig // Dummy to set denormal flag
      nop.i 999
}
// *************************************************
// ********************* STEP2 *********************
// *************************************************
//
//     Q = E * V
//
{ .mfi
      nop.m 999
      fmpy.s1 Q = E, V
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fnma.s1 E_hold = E, U, f1           // E_hold = 1.0 - E*U (1) if POLY path
      nop.i 999
}
;;

// Create a single precision representation of the signexp of Q with the
// 4 most significant bits of the significand followed by a 1 and then 18 0's
{ .mfi
      nop.m 999
      fmpy.s1 P_hi = M, P_hi
      dep.z special = 0x1, 18, 1           // Form 0x0000000000040000
}
{ .mfi
      nop.m 999
      fmpy.s1 P_lo = M, P_lo
      add table_ptr2 = 32, table_ptr1
}
;;

{ .mfi
      nop.m 999
      fma.s1 A_temp = Q, f1, f0            // Set A_temp if POLY path
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 E = E, E_hold, E              // E = E + E*E_hold (1) if POLY path
      nop.i 999
}
;;

//
//     Is Q < 2**(-3)?
//     swap = xor(swap,sign_X)
//
{ .mfi
      nop.m 999
      fcmp.lt.s1 p9, p0 = Q, TWO_TO_NEG3    // Test Q < 2^-3
      xor swap = sign_X, swap
}
;;

//     P_hi = s_Y * P_hi
{ .mmf
      getf.exp exponent_Q =  Q              // Get signexp of Q
      cmp.eq.unc p7, p6 = 0x00000, swap
      fmpy.s1 P_hi = s_Y, P_hi
}
;;

//
//     if (PR_1) sigma = -1.0
//     if (PR_2) sigma =  1.0
//
{ .mfi
      getf.sig significand_Q = Q            // Get significand of Q
(p6)  fsub.s1 sigma = f0, f1
      nop.i 999
}
{ .mfb
(p9)  add table_ptr1 = 128, table_base      // Point to P8 if POLY path
(p7)  fadd.s1 sigma = f0, f1
(p9)  br.cond.spnt ATANL_POLY               // Branch to POLY if 0 < Q < 2^-3
}
;;

//
// *************************************************
// ******************** STEP3 **********************
// *************************************************
//
//     lookup = b_1 b_2 b_3 B_4
//
{ .mmi
      nop.m 999
      nop.m 999
      andcm k = 0x0003, exponent_Q  // k=0,1,2,3 for exp_Q=0,-1,-2,-3
}
;;

//
//  Generate sign_exp_Q b_1 b_2 b_3 b_4 1 0 0 0 ... 0  in single precision
//  representation.  Note sign of Q is always 0.
//
{ .mfi
      cmp.eq p8, p9 = 0x0000, k             // Test k=0
      nop.f 999
      extr.u lookup = significand_Q, 59, 4  // Extract b_1 b_2 b_3 b_4 for index
}
{ .mfi
      sub sp_exp_Q = 0x7f, k                // Form single prec biased exp of Q
      nop.f 999
      sub k = k, r0, 1                      // Decrement k
}
;;

//     Form pointer to B index table
{ .mfi
      ldfe Q_4 = [table_ptr1], -16          // Load Q_4
      nop.f 999
(p9)  shl k = k, 8                          // k = 0, 256, or 512
}
{ .mfi
(p9)  shladd table_ptr2 = lookup, 4, table_ptr2
      nop.f 999
      shladd sp_exp_4sig_Q = sp_exp_Q, 4, lookup // Shift and add in 4 high bits
}
;;

{ .mmi
(p8)  add table_ptr2 = -16, table_ptr2      // Pointer if original k was 0
(p9)  add table_ptr2 = k, table_ptr2        // Pointer if k was 1, 2, 3
      dep special = sp_exp_4sig_Q, special, 19, 13 // Form z_hi as single prec
}
;;

//     z_hi = s exp 1.b_1 b_2 b_3 b_4 1 0 0 0 ... 0
{ .mmi
      ldfd Tbl_hi = [table_ptr2], 8         // Load Tbl_hi from index table
;;
      setf.s z_hi = special                 // Form z_hi
      nop.i 999
}
{ .mmi
      ldfs Tbl_lo = [table_ptr2], 8         // Load Tbl_lo from index table
;;
      ldfe Q_3 = [table_ptr1], -16          // Load Q_3
      nop.i 999
}
;;

{ .mmi
      ldfe Q_2 = [table_ptr1], -16          // Load Q_2
      nop.m 999
      nop.i 999
}
;;

{ .mmf
      ldfe Q_1 = [table_ptr1], -16          // Load Q_1
      nop.m 999
      nop.f 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 U_prime_hi = V, z_hi, U        // U_prime_hi = U + V * z_hi
      nop.i 999
}
{ .mfi
      nop.m 999
      fnma.s1 V_prime = U, z_hi, V          // V_prime =  V - U * z_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      mov A_hi = Tbl_hi                     // Start with A_hi = Tbl_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsub.s1 U_hold = U, U_prime_hi        // U_hold = U - U_prime_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      frcpa.s1 C_hi, p0 = f1, U_prime_hi    // C_hi = frcpa(1,U_prime_hi)
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 A_hi = s_Y, A_hi              // A_hi = s_Y * A_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 U_prime_lo = z_hi, V, U_hold   // U_prime_lo =  U_hold + V * z_hi
      nop.i 999
}
;;

//     C_hi_hold = 1 - C_hi * U_prime_hi (1)
{ .mfi
      nop.m 999
      fnma.s1 C_hi_hold = C_hi, U_prime_hi, f1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 Res_hi = sigma, A_hi, P_hi   // Res_hi = P_hi + sigma * A_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 C_hi = C_hi_hold, C_hi, C_hi // C_hi = C_hi + C_hi * C_hi_hold (1)
      nop.i 999
}
;;

//     C_hi_hold = 1 - C_hi * U_prime_hi (2)
{ .mfi
      nop.m 999
      fnma.s1 C_hi_hold = C_hi, U_prime_hi, f1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 C_hi = C_hi_hold, C_hi, C_hi // C_hi = C_hi + C_hi * C_hi_hold (2)
      nop.i 999
}
;;

//     C_hi_hold = 1 - C_hi * U_prime_hi (3)
{ .mfi
      nop.m 999
      fnma.s1 C_hi_hold = C_hi, U_prime_hi, f1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 C_hi = C_hi_hold, C_hi, C_hi // C_hi = C_hi + C_hi * C_hi_hold (3)
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 w_hi = V_prime, C_hi           // w_hi = V_prime * C_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 wsq = w_hi, w_hi               // wsq = w_hi * w_hi
      nop.i 999
}
{ .mfi
      nop.m 999
      fnma.s1 w_lo = w_hi, U_prime_hi, V_prime // w_lo = V_prime-w_hi*U_prime_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 poly =  wsq, Q_4, Q_3           // poly = Q_3 + wsq * Q_4
      nop.i 999
}
{ .mfi
      nop.m 999
      fnma.s1 w_lo = w_hi, U_prime_lo, w_lo  // w_lo = w_lo - w_hi * U_prime_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 poly = wsq, poly, Q_2           // poly = Q_2 + wsq * poly
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 w_lo = C_hi, w_lo              // w_lo =  = w_lo * C_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 poly = wsq, poly, Q_1           // poly = Q_1 + wsq * poly
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 A_lo = Tbl_lo, w_lo            // A_lo = Tbl_lo + w_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s0 Q_1 =  Q_1, Q_1                // Dummy operation to raise inexact
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 poly = wsq, poly               // poly = wsq * poly
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 poly = w_hi, poly              // poly = w_hi * poly
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 A_lo = A_lo, poly              // A_lo = A_lo + poly
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 A_lo = A_lo, w_hi              // A_lo = A_lo + w_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 Res_lo = sigma, A_lo, P_lo      // Res_lo = P_lo + sigma * A_lo
      nop.i 999
}
;;

//
//     Result  =  Res_hi + Res_lo * s_Y  (User Supplied Rounding Mode)
//
{ .mfb
      nop.m 999
      fma.s0 Result = Res_lo, s_Y, Res_hi
      br.ret.sptk   b0                        // Exit table path 2^-3 <= V/U < 1
}
;;


ATANL_POLY:
// Here if 0 < V/U < 2^-3
//
// ***********************************************
// ******************** STEP4 ********************
// ***********************************************

//
//     Following:
//     Iterate 3 times E = E + E*(1.0 - E*U)
//     Also load P_8, P_7, P_6, P_5, P_4
//
{ .mfi
      ldfe P_8 = [table_ptr1], -16            // Load P_8
      fnma.s1 z_lo = A_temp, U, V             // z_lo = V - A_temp * U
      nop.i 999
}
{ .mfi
      nop.m 999
      fnma.s1 E_hold = E, U, f1               // E_hold = 1.0 - E*U (2)
      nop.i 999
}
;;

{ .mmi
      ldfe P_7 = [table_ptr1], -16            // Load P_7
;;
      ldfe P_6 = [table_ptr1], -16            // Load P_6
      nop.i 999
}
;;

{ .mfi
      ldfe P_5 = [table_ptr1], -16            // Load P_5
      fma.s1 E = E, E_hold, E                 // E = E + E_hold*E (2)
      nop.i 999
}
;;

{ .mmi
      ldfe P_4 = [table_ptr1], -16            // Load P_4
;;
      ldfe P_3 = [table_ptr1], -16            // Load P_3
      nop.i 999
}
;;

{ .mfi
      ldfe P_2 = [table_ptr1], -16            // Load P_2
      fnma.s1 E_hold = E, U, f1               // E_hold = 1.0 - E*U (3)
      nop.i 999
}
{ .mlx
      nop.m 999
      movl         int_temp = 0x24005         // Signexp for small neg number
}
;;

{ .mmf
      ldfe P_1 = [table_ptr1], -16            // Load P_1
      setf.exp     tmp_small = int_temp       // Form small neg number
      fma.s1 E = E, E_hold, E                 // E = E + E_hold*E (3)
}
;;

//
//
// At this point E approximates 1/U to roughly working precision
// Z = V*E approximates V/U
//
{ .mfi
      nop.m 999
      fmpy.s1 Z = V, E                         // Z = V * E
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 z_lo = z_lo, E                   // z_lo = z_lo * E
      nop.i 999
}
;;

//
//     Now what we want to do is
//     poly1 = P_4 + zsq*(P_5 + zsq*(P_6 + zsq*(P_7 + zsq*P_8)))
//     poly2 = zsq*(P_1 + zsq*(P_2 + zsq*P_3))
//
//
//     Fixup added to force inexact later -
//     A_hi = A_temp + z_lo
//     z_lo = (A_temp - A_hi) + z_lo
//
{ .mfi
      nop.m 999
      fmpy.s1 zsq = Z, Z                        // zsq = Z * Z
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 A_hi = A_temp, z_lo               // A_hi = A_temp + z_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 poly1 = zsq, P_8, P_7              // poly1 = P_7 + zsq * P_8
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 poly2 = zsq, P_3, P_2              // poly2 = P_2 + zsq * P_3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 z4 = zsq, zsq                     // z4 = zsq * zsq
      nop.i 999
}
{ .mfi
      nop.m 999
      fsub.s1 A_temp = A_temp, A_hi             // A_temp = A_temp - A_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmerge.s     tmp = A_hi, A_hi             // Copy tmp = A_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 poly1 = zsq, poly1, P_6            // poly1 = P_6 + zsq * poly1
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 poly2 = zsq, poly2, P_1            // poly2 = P_2 + zsq * poly2
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 z8 = z4, z4                       // z8 = z4 * z4
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 z_lo = A_temp, z_lo               // z_lo = (A_temp - A_hi) + z_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 poly1 = zsq, poly1, P_5            // poly1 = P_5 + zsq * poly1
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 poly2 = poly2, zsq                // poly2 = zsq * poly2
      nop.i 999
}
;;

//     Create small GR double in case need to raise underflow
{ .mfi
      nop.m 999
      fma.s1 poly1 = zsq, poly1, P_4            // poly1 = P_4 + zsq * poly1
      dep GR_temp = -1,r0,0,53
}
;;

//     Create small double in case need to raise underflow
{ .mfi
      setf.d FR_temp = GR_temp
      fma.s1 poly = z8, poly1, poly2            // poly = poly2 + z8 * poly1
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 A_lo = Z, poly, z_lo               // A_lo = z_lo + Z * poly
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1      A_hi = tmp, A_lo             // A_hi = tmp + A_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsub.s1      tmp = tmp, A_hi              // tmp = tmp - A_hi
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s1 A_hi = s_Y, A_hi                  // A_hi = s_Y * A_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1      A_lo = tmp, A_lo             // A_lo = tmp + A_lo
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 Res_hi = sigma, A_hi, P_hi         // Res_hi = P_hi + sigma * A_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsub.s1 tmp =  P_hi, Res_hi               // tmp = P_hi - Res_hi
      nop.i 999
}
;;

//
//     Test if A_lo is zero
//
{ .mfi
      nop.m 999
      fclass.m p6,p0 = A_lo, 0x007              // Test A_lo = 0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p6)  mov          A_lo = tmp_small             // If A_lo zero, make very small
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 tmp = A_hi, sigma, tmp             // tmp = sigma * A_hi  + tmp
      nop.i 999
}
{ .mfi
      nop.m 999
      fma.s1 sigma =  A_lo, sigma, P_lo         // sigma = A_lo * sigma  + P_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fma.s1 Res_lo = s_Y, sigma, tmp           // Res_lo = s_Y * sigma + tmp
      nop.i 999
}
;;

//
//     Test if Res_lo is denormal
//
{ .mfi
      nop.m 999
      fclass.m p14, p15 = Res_lo, 0x0b
      nop.i 999
}
;;

//
//     Compute Result = Res_lo + Res_hi.  Use s3 if Res_lo is denormal.
//
{ .mfi
      nop.m 999
(p14) fadd.s3 Result = Res_lo, Res_hi     // Result for Res_lo denormal
      nop.i 999
}
{ .mfi
      nop.m 999
(p15) fadd.s0 Result = Res_lo, Res_hi     // Result for Res_lo normal
      nop.i 999
}
;;

//
//     If Res_lo is denormal test if Result equals zero
//
{ .mfi
      nop.m 999
(p14) fclass.m.unc p14, p0 = Result, 0x07
      nop.i 999
}
;;

//
//     If Res_lo is denormal and Result equals zero, raise inexact, underflow
//     by squaring small double
//
{ .mfb
      nop.m 999
(p14) fmpy.d.s0 FR_temp = FR_temp, FR_temp
      br.ret.sptk   b0                     // Exit POLY path, 0 < Q < 2^-3
}
;;


ATANL_UNSUPPORTED:
{ .mfb
      nop.m 999
      fmpy.s0 Result = ArgX,ArgY
      br.ret.sptk   b0
}
;;

// Here if y natval, nan, inf, zero
ATANL_Y_SPECIAL:
// Here if x natval, nan, inf, zero
ATANL_X_SPECIAL:
{ .mfi
      nop.m 999
      fclass.m p13,p12 = ArgY_orig, 0x0c3  // Test y nan
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p15,p14 = ArgY_orig, 0x103  // Test y natval
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p12) fclass.m p13,p0 = ArgX_orig, 0x0c3  // Test x nan
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p14) fclass.m p15,p0 = ArgX_orig, 0x103  // Test x natval
      nop.i 999
}
;;

{ .mfb
      nop.m 999
(p13) fmpy.s0 Result = ArgX_orig, ArgY_orig // Result nan if x or y nan
(p13) br.ret.spnt b0                      // Exit if x or y nan
}
;;

{ .mfb
      nop.m 999
(p15) fmpy.s0 Result = ArgX_orig, ArgY_orig // Result natval if x or y natval
(p15) br.ret.spnt b0                      // Exit if x or y natval
}
;;


// Here if x or y inf or zero
ATANL_SPECIAL_HANDLING:
{ .mfi
      nop.m 999
      fclass.m p6, p7 = ArgY_orig, 0x007        // Test y zero
      mov special = 992                         // Offset to table
}
;;

{ .mfb
      add table_ptr1 = table_base, special      // Point to 3pi/4
      fcmp.eq.s0 p0, p9 = ArgX_orig, ArgY_orig  // Dummy to set denormal flag
(p7)  br.cond.spnt ATANL_ArgY_Not_ZERO          // Branch if y not zero
}
;;

// Here if y zero
{ .mmf
      ldfd  Result = [table_ptr1], 8            // Get pi high
      nop.m 999
      fclass.m p14, p0 = ArgX, 0x035            // Test for x>=+0
}
;;

{ .mmf
      nop.m 999
      ldfd  Result_lo = [table_ptr1], -8        // Get pi lo
      fclass.m p15, p0 = ArgX, 0x036            // Test for x<=-0
}
;;

//
//     Return sign_Y * 0 when  ArgX > +0
//
{ .mfi
      nop.m 999
(p14) fmerge.s Result = ArgY, f0               // If x>=+0, y=0, hi sgn(y)*0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p13, p0 = ArgX, 0x007           // Test for x=0
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p14) fmerge.s Result_lo = ArgY, f0            // If x>=+0, y=0, lo sgn(y)*0
      nop.i 999
}
;;

{ .mfi
(p13) mov GR_Parameter_TAG = 36                // Error tag for x=0, y=0
      nop.f 999
      nop.i 999
}
;;

//
//     Return sign_Y * pi when  ArgX < -0
//
{ .mfi
      nop.m 999
(p15) fmerge.s Result = ArgY, Result           // If x<0, y=0, hi=sgn(y)*pi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p15) fmerge.s Result_lo = ArgY, Result_lo     // If x<0, y=0, lo=sgn(y)*pi
      nop.i 999
}
;;

//
//     Call error support function for atan(0,0)
//
{ .mfb
      nop.m 999
      fadd.s0 Result = Result, Result_lo
(p13) br.cond.spnt __libm_error_region         // Branch if atan(0,0)
}
;;

{ .mib
      nop.m 999
      nop.i 999
      br.ret.sptk   b0                         // Exit for y=0, x not 0
}
;;

// Here if y not zero
ATANL_ArgY_Not_ZERO:
{ .mfi
      nop.m 999
      fclass.m p0, p10 = ArgY, 0x023           // Test y inf
      nop.i 999
}
;;

{ .mfb
      nop.m 999
      fclass.m p6, p0 = ArgX, 0x017            // Test for 0 <= |x| < inf
(p10) br.cond.spnt  ATANL_ArgY_Not_INF         // Branch if 0 < |y| < inf
}
;;

// Here if y=inf
//
//     Return +PI/2 when ArgY = +Inf and ArgX = +/-0 or normal
//     Return -PI/2 when ArgY = -Inf and ArgX = +/-0 or normal
//     Return +PI/4 when ArgY = +Inf and ArgX = +Inf
//     Return -PI/4 when ArgY = -Inf and ArgX = +Inf
//     Return +3PI/4 when ArgY = +Inf and ArgX = -Inf
//     Return -3PI/4 when ArgY = -Inf and ArgX = -Inf
//
{ .mfi
      nop.m 999
      fclass.m p7, p0 = ArgX, 0x021            // Test for x=+inf
      nop.i 999
}
;;

{ .mfi
(p6)  add table_ptr1 =  16, table_ptr1         // Point to pi/2, if x finite
      fclass.m p8, p0 = ArgX, 0x022            // Test for x=-inf
      nop.i 999
}
;;

{ .mmi
(p7)  add table_ptr1 =  32, table_ptr1         // Point to pi/4 if x=+inf
;;
(p8)  add table_ptr1 =  48, table_ptr1         // Point to 3pi/4 if x=-inf

      nop.i 999
}
;;

{ .mmi
      ldfd Result = [table_ptr1], 8            // Load pi/2, pi/4, or 3pi/4 hi
;;
      ldfd Result_lo = [table_ptr1], -8        // Load pi/2, pi/4, or 3pi/4 lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmerge.s Result = ArgY, Result           // Merge sgn(y) in hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmerge.s Result_lo = ArgY, Result_lo     // Merge sgn(y) in lo
      nop.i 999
}
;;

{ .mfb
      nop.m 999
      fadd.s0 Result = Result, Result_lo       // Compute complete result
      br.ret.sptk   b0                         // Exit for y=inf
}
;;

// Here if y not INF, and x=0 or INF
ATANL_ArgY_Not_INF:
//
//     Return +PI/2 when ArgY NOT Inf, ArgY > 0 and ArgX = +/-0
//     Return -PI/2 when ArgY NOT Inf, ArgY < 0 and ArgX = +/-0
//     Return +0    when ArgY NOT Inf, ArgY > 0 and ArgX = +Inf
//     Return -0    when ArgY NOT Inf, ArgY > 0 and ArgX = +Inf
//     Return +PI   when ArgY NOT Inf, ArgY > 0 and ArgX = -Inf
//     Return -PI   when ArgY NOT Inf, ArgY > 0 and ArgX = -Inf
//
{ .mfi
      nop.m 999
      fclass.m p7, p9 = ArgX, 0x021            // Test for x=+inf
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fclass.m p6, p0 = ArgX, 0x007            // Test for x=0
      nop.i 999
}
;;

{ .mfi
(p6)  add table_ptr1 = 16, table_ptr1          // Point to pi/2
      fclass.m p8, p0 = ArgX, 0x022            // Test for x=-inf
      nop.i 999
}
;;

.pred.rel "mutex",p7,p9
{ .mfi
(p9)  ldfd Result = [table_ptr1], 8           // Load pi or pi/2 hi
(p7)  fmerge.s Result = ArgY, f0              // If y not inf, x=+inf, sgn(y)*0
      nop.i 999
}
;;

{ .mfi
(p9)  ldfd Result_lo = [table_ptr1], -8       // Load pi or pi/2 lo
(p7)  fnorm.s0 Result = Result                // If y not inf, x=+inf normalize
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fmerge.s Result = ArgY, Result          // Merge sgn(y) in hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fmerge.s Result_lo = ArgY, Result_lo    // Merge sgn(y) in lo
      nop.i 999
}
;;

{ .mfb
      nop.m 999
(p9)  fadd.s0 Result = Result, Result_lo      // Compute complete result
      br.ret.spnt   b0                        // Exit for y not inf, x=0,inf
}
;;

GLOBAL_IEEE754_END(atan2l)
libm_alias_ldouble_other (__atan2, atan2)

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
        nop.m 0
        nop.m 0
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
