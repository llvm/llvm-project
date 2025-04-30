.file "libm_reduce.s"


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
// History:
// 02/02/00 Initial Version
// 05/13/02 Rescheduled for speed, changed interface to pass
//          parameters in fp registers
// 02/10/03 Reordered header: .section, .global, .proc, .align;
//          used data8 for long double data storage
//
//*********************************************************************
//*********************************************************************
//
// Function:   __libm_pi_by_two_reduce(x) return r, c, and N where
//             x = N * pi/4 + (r+c) , where |r+c| <= pi/4.
//             This function is not designed to be used by the
//             general user.
//
//*********************************************************************
//
// Accuracy:       Returns double-precision values
//
//*********************************************************************
//
// Resources Used:
//
//    Floating-Point Registers:
//      f8  = Input x, return value r
//      f9  = return value c
//      f32-f70
//
//    General Purpose Registers:
//      r8  = return value N
//      r34-r64
//
//    Predicate Registers:      p6-p14
//
//*********************************************************************
//
// IEEE Special Conditions:
//
//    No conditions should be raised.
//
//*********************************************************************
//
// I. Introduction
// ===============
//
// For the forward trigonometric functions sin, cos, sincos, and
// tan, the original algorithms for IA 64 handle arguments up to
// 1 ulp less than 2^63 in magnitude. For double-extended arguments x,
// |x| >= 2^63, this routine returns N and r_hi, r_lo where
//
//    x  is accurately approximated by
//    2*K*pi  +  N * pi/2  +  r_hi + r_lo,  |r_hi+r_lo| <= pi/4.
//    CASE = 1 or 2.
//    CASE is 1 unless |r_hi + r_lo| < 2^(-33).
//
// The exact value of K is not determined, but that information is
// not required in trigonometric function computations.
//
// We first assume the argument x in question satisfies x >= 2^(63).
// In particular, it is positive. Negative x can be handled by symmetry:
//
//   -x  is accurately approximated by
//         -2*K*pi  +  (-N) * pi/2  -  (r_hi + r_lo),  |r_hi+r_lo| <= pi/4.
//
// The idea of the reduction is that
//
//       x  *  2/pi   =   N_big  +  N  +  f,      |f| <= 1/2
//
// Moreover, for double extended x, |f| >= 2^(-75). (This is an
// non-obvious fact found by enumeration using a special algorithm
// involving continued fraction.) The algorithm described below
// calculates N and an accurate approximation of f.
//
// Roughly speaking, an appropriate 256-bit (4 X 64) portion of
// 2/pi is multiplied with x to give the desired information.
//
// II. Representation of 2/PI
// ==========================
//
// The value of 2/pi in binary fixed-point is
//
//            .101000101111100110......
//
// We store 2/pi in a table, starting at the position corresponding
// to bit position 63
//
//   bit position  63 62 ... 0   -1 -2 -3 -4 -5 -6 -7  ....  -16576
//
//              0  0  ... 0  . 1  0  1  0  1  0  1  ....    X
//
//                              ^
//                               |__ implied binary pt
//
// III. Algorithm
// ==============
//
// This describes the algorithm in the most natural way using
// unsigned interger multiplication. The implementation section
// describes how the integer arithmetic is simulated.
//
// STEP 0. Initialization
// ----------------------
//
// Let the input argument x be
//
//     x = 2^m * ( 1. b_1 b_2 b_3 ... b_63 ),  63 <= m <= 16383.
//
// The first crucial step is to fetch four 64-bit portions of 2/pi.
// To fulfill this goal, we calculate the bit position L of the
// beginning of these 256-bit quantity by
//
//     L :=  62 - m.
//
// Note that -16321 <= L <= -1 because 63 <= m <= 16383; and that
// the storage of 2/pi is adequate.
//
// Fetch P_1, P_2, P_3, P_4 beginning at bit position L thus:
//
//      bit position  L  L-1  L-2    ...  L-63
//
//      P_1    =      b   b    b     ...    b
//
// each b can be 0 or 1. Also, let P_0 be the two bits correspoding to
// bit positions L+2 and L+1. So, when each of the P_j is interpreted
// with appropriate scaling, we have
//
//      2/pi  =  P_big  + P_0 + (P_1 + P_2 + P_3 + P_4)  +  P_small
//
// Note that P_big and P_small can be ignored. The reasons are as follow.
// First, consider P_big. If P_big = 0, we can certainly ignore it.
// Otherwise, P_big >= 2^(L+3). Now,
//
//        P_big * ulp(x) >=  2^(L+3) * 2^(m-63)
//                   >=  2^(65-m  +  m-63 )
//                   >=  2^2
//
// Thus, P_big * x is an integer of the form 4*K. So
//
//       x = 4*K * (pi/2) + x*(P_0 + P_1 + P_2 + P_3 + P_4)*(pi/2)
//                + x*P_small*(pi/2).
//
// Hence, P_big*x corresponds to information that can be ignored for
// trigonometic function evaluation.
//
// Next, we must estimate the effect of ignoring P_small. The absolute
// error made by ignoring P_small is bounded by
//
//       |P_small * x|  <=  ulp(P_4) * x
//                  <=  2^(L-255) * 2^(m+1)
//                  <=  2^(62-m-255 + m + 1)
//                  <=  2^(-192)
//
// Since for double-extended precision, x * 2/pi = integer + f,
// 0.5 >= |f| >= 2^(-75), the relative error introduced by ignoring
// P_small is bounded by 2^(-192+75) <= 2^(-117), which is acceptable.
//
// Further note that if x is split into x_hi + x_lo where x_lo is the
// two bits corresponding to bit positions 2^(m-62) and 2^(m-63); then
//
//       P_0 * x_hi
//
// is also an integer of the form 4*K; and thus can also be ignored.
// Let M := P_0 * x_lo which is a small integer. The main part of the
// calculation is really the multiplication of x with the four pieces
// P_1, P_2, P_3, and P_4.
//
// Unless the reduced argument is extremely small in magnitude, it
// suffices to carry out the multiplication of x with P_1, P_2, and
// P_3. x*P_4 will be carried out and added on as a correction only
// when it is found to be needed. Note also that x*P_4 need not be
// computed exactly. A straightforward multiplication suffices since
// the rounding error thus produced would be bounded by 2^(-3*64),
// that is 2^(-192) which is small enough as the reduced argument
// is bounded from below by 2^(-75).
//
// Now that we have four 64-bit data representing 2/pi and a
// 64-bit x. We first need to calculate a highly accurate product
// of x and P_1, P_2, P_3. This is best understood as integer
// multiplication.
//
//
// STEP 1. Multiplication
// ----------------------
//
//
//                     ---------   ---------   ---------
//                    |  P_1  |   |  P_2  |   |  P_3  |
//                    ---------   ---------   ---------
//
//                                            ---------
//             X                              |   X   |
//                                            ---------
//      ----------------------------------------------------
//
//                                 ---------   ---------
//                               |  A_hi |   |  A_lo |
//                               ---------   ---------
//
//
//                    ---------   ---------
//                   |  B_hi |   |  B_lo |
//                   ---------   ---------
//
//
//        ---------   ---------
//       |  C_hi |   |  C_lo |
//       ---------   ---------
//
//      ====================================================
//       ---------   ---------   ---------   ---------
//       |  S_0  |   |  S_1  |   |  S_2  |   |  S_3  |
//       ---------   ---------   ---------   ---------
//
//
//
// STEP 2. Get N and f
// -------------------
//
// Conceptually, after the individual pieces S_0, S_1, ..., are obtained,
// we have to sum them and obtain an integer part, N, and a fraction, f.
// Here, |f| <= 1/2, and N is an integer. Note also that N need only to
// be known to module 2^k, k >= 2. In the case when |f| is small enough,
// we would need to add in the value x*P_4.
//
//
// STEP 3. Get reduced argument
// ----------------------------
//
// The value f is not yet the reduced argument that we seek. The
// equation
//
//       x * 2/pi = 4K  + N  + f
//
// says that
//
//         x   =  2*K*pi  + N * pi/2  +  f * (pi/2).
//
// Thus, the reduced argument is given by
//
//       reduced argument =  f * pi/2.
//
// This multiplication must be performed to extra precision.
//
// IV. Implementation
// ==================
//
// Step 0. Initialization
// ----------------------
//
// Set sgn_x := sign(x); x := |x|; x_lo := 2 lsb of x.
//
// In memory, 2/pi is stored contiguously as
//
//  0x00000000 0x00000000 0xA2F....
//                       ^
//                       |__ implied binary bit
//
// Given x = 2^m * 1.xxxx...xxx; we calculate L := 62 - m. Thus
// -1 <= L <= -16321. We fetch from memory 5 integer pieces of data.
//
// P_0 is the two bits corresponding to bit positions L+2 and L+1
// P_1 is the 64-bit starting at bit position  L
// P_2 is the 64-bit starting at bit position  L-64
// P_3 is the 64-bit starting at bit position  L-128
// P_4 is the 64-bit starting at bit position  L-192
//
// For example, if m = 63, P_0 would be 0 and P_1 would look like
// 0xA2F...
//
// If m = 65, P_0 would be the two msb of 0xA, thus, P_0 is 10 in binary.
// P_1 in binary would be  1 0 0 0 1 0 1 1 1 1 ....
//
// Step 1. Multiplication
// ----------------------
//
// At this point, P_1, P_2, P_3, P_4 are integers. They are
// supposed to be interpreted as
//
//  2^(L-63)     * P_1;
//  2^(L-63-64)  * P_2;
//  2^(L-63-128) * P_3;
// 2^(L-63-192) * P_4;
//
// Since each of them need to be multiplied to x, we would scale
// both x and the P_j's by some convenient factors: scale each
// of P_j's up by 2^(63-L), and scale x down by 2^(L-63).
//
//   p_1 := fcvt.xf ( P_1 )
//   p_2 := fcvt.xf ( P_2 ) * 2^(-64)
//   p_3 := fcvt.xf ( P_3 ) * 2^(-128)
//   p_4 := fcvt.xf ( P_4 ) * 2^(-192)
//   x   := replace exponent of x by -1
//          because 2^m    * 1.xxxx...xxx  * 2^(L-63)
//          is      2^(-1) * 1.xxxx...xxx
//
// We are now faced with the task of computing the following
//
//                     ---------   ---------   ---------
//                    |  P_1  |   |  P_2  |   |  P_3  |
//                    ---------   ---------   ---------
//
//                                             ---------
//             X                              |   X   |
//                                            ---------
//       ----------------------------------------------------
//
//                                 ---------   ---------
//                                |  A_hi |   |  A_lo |
//                                ---------   ---------
//
//                     ---------   ---------
//                    |  B_hi |   |  B_lo |
//                    ---------   ---------
//
//         ---------   ---------
//        |  C_hi |   |  C_lo |
//        ---------   ---------
//
//      ====================================================
//       -----------   ---------   ---------   ---------
//       |    S_0  |   |  S_1  |   |  S_2  |   |  S_3  |
//       -----------   ---------   ---------   ---------
//        ^          ^
//        |          |___ binary point
//        |
//        |___ possibly one more bit
//
// Let FPSR3 be set to round towards zero with widest precision
// and exponent range. Unless an explicit FPSR is given,
// round-to-nearest with widest precision and exponent range is
// used.
//
// Define sigma_C := 2^63; sigma_B := 2^(-1); sigma_C := 2^(-65).
//
// Tmp_C := fmpy.fpsr3( x, p_1 );
// If Tmp_C >= sigma_C then
//    C_hi := Tmp_C;
//    C_lo := x*p_1 - C_hi ...fma, exact
// Else
//    C_hi := fadd.fpsr3(sigma_C, Tmp_C) - sigma_C
//                   ...subtraction is exact, regardless
//                   ...of rounding direction
//    C_lo := x*p_1 - C_hi ...fma, exact
// End If
//
// Tmp_B := fmpy.fpsr3( x, p_2 );
// If Tmp_B >= sigma_B then
//    B_hi := Tmp_B;
//    B_lo := x*p_2 - B_hi ...fma, exact
// Else
//    B_hi := fadd.fpsr3(sigma_B, Tmp_B) - sigma_B
//                   ...subtraction is exact, regardless
//                   ...of rounding direction
//    B_lo := x*p_2 - B_hi ...fma, exact
// End If
//
// Tmp_A := fmpy.fpsr3( x, p_3 );
// If Tmp_A >= sigma_A then
//    A_hi := Tmp_A;
//    A_lo := x*p_3 - A_hi ...fma, exact
// Else
//    A_hi := fadd.fpsr3(sigma_A, Tmp_A) - sigma_A
//                   ...subtraction is exact, regardless
//                   ...of rounding direction
//    A_lo := x*p_3 - A_hi ...fma, exact
// End If
//
// ...Note that C_hi is of integer value. We need only the
// ...last few bits. Thus we can ensure C_hi is never a big
// ...integer, freeing us from overflow worry.
//
// Tmp_C := fadd.fpsr3( C_hi, 2^(70) ) - 2^(70);
// ...Tmp_C is the upper portion of C_hi
// C_hi := C_hi - Tmp_C
// ...0 <= C_hi < 2^7
//
// Step 2. Get N and f
// -------------------
//
// At this point, we have all the components to obtain
// S_0, S_1, S_2, S_3 and thus N and f. We start by adding
// C_lo and B_hi. This sum together with C_hi gives a good
// estimation of N and f.
//
// A := fadd.fpsr3( B_hi, C_lo )
// B := max( B_hi, C_lo )
// b := min( B_hi, C_lo )
//
// a := (B - A) + b      ...exact. Note that a is either 0
//                   ...or 2^(-64).
//
// N := round_to_nearest_integer_value( A );
// f := A - N;            ...exact because lsb(A) >= 2^(-64)
//                   ...and |f| <= 1/2.
//
// f := f + a            ...exact because a is 0 or 2^(-64);
//                   ...the msb of the sum is <= 1/2
//                   ...lsb >= 2^(-64).
//
// N := convert to integer format( C_hi + N );
// M := P_0 * x_lo;
// N := N + M;
//
// If sgn_x == 1 (that is original x was negative)
// N := 2^10 - N
// ...this maintains N to be non-negative, but still
// ...equivalent to the (negated N) mod 4.
// End If
//
// If |f| >= 2^(-33)
//
// ...Case 1
// CASE := 1
// g := A_hi + B_lo;
// s_hi := f + g;
// s_lo := (f - s_hi) + g;
//
// Else
//
// ...Case 2
// CASE := 2
// A := fadd.fpsr3( A_hi, B_lo )
// B := max( A_hi, B_lo )
// b := min( A_hi, B_lo )
//
// a := (B - A) + b      ...exact. Note that a is either 0
//                   ...or 2^(-128).
//
// f_hi := A + f;
// f_lo := (f - f_hi) + A;
// ...this is exact.
// ...f-f_hi is exact because either |f| >= |A|, in which
// ...case f-f_hi is clearly exact; or otherwise, 0<|f|<|A|
// ...means msb(f) <= msb(A) = 2^(-64) => |f| = 2^(-64).
// ...If f = 2^(-64), f-f_hi involves cancellation and is
// ...exact. If f = -2^(-64), then A + f is exact. Hence
// ...f-f_hi is -A exactly, giving f_lo = 0.
//
// f_lo := f_lo + a;
//
// If |f| >= 2^(-50) then
//    s_hi := f_hi;
//    s_lo := f_lo;
// Else
//    f_lo := (f_lo + A_lo) + x*p_4
//    s_hi := f_hi + f_lo
//    s_lo := (f_hi - s_hi) + f_lo
// End If
//
// End If
//
// Step 3. Get reduced argument
// ----------------------------
//
// If sgn_x == 0 (that is original x is positive)
//
// D_hi := Pi_by_2_hi
// D_lo := Pi_by_2_lo
// ...load from table
//
// Else
//
// D_hi := neg_Pi_by_2_hi
// D_lo := neg_Pi_by_2_lo
// ...load from table
// End If
//
// r_hi :=  s_hi*D_hi
// r_lo :=  s_hi*D_hi - r_hi         ...fma
// r_lo := (s_hi*D_lo + r_lo) + s_lo*D_hi
//
// Return  N, r_hi, r_lo
//
FR_input_X = f8
FR_r_hi    = f8
FR_r_lo    = f9

FR_X       = f32
FR_N       = f33
FR_p_1     = f34
FR_TWOM33  = f35
FR_TWOM50  = f36
FR_g       = f37
FR_p_2     = f38
FR_f       = f39
FR_s_lo    = f40
FR_p_3     = f41
FR_f_abs   = f42
FR_D_lo    = f43
FR_p_4     = f44
FR_D_hi    = f45
FR_Tmp2_C  = f46
FR_s_hi    = f47
FR_sigma_A = f48
FR_A       = f49
FR_sigma_B = f50
FR_B       = f51
FR_sigma_C = f52
FR_b       = f53
FR_ScaleP2 = f54
FR_ScaleP3 = f55
FR_ScaleP4 = f56
FR_Tmp_A   = f57
FR_Tmp_B   = f58
FR_Tmp_C   = f59
FR_A_hi    = f60
FR_f_hi    = f61
FR_RSHF    = f62
FR_A_lo    = f63
FR_B_hi    = f64
FR_a       = f65
FR_B_lo    = f66
FR_f_lo    = f67
FR_N_fix   = f68
FR_C_hi    = f69
FR_C_lo    = f70

GR_N       = r8
GR_Exp_x   = r36
GR_Temp    = r37
GR_BIASL63 = r38
GR_CASE    = r39
GR_x_lo    = r40
GR_sgn_x   = r41
GR_M       = r42
GR_BASE    = r43
GR_LENGTH1 = r44
GR_LENGTH2 = r45
GR_ASUB    = r46
GR_P_0     = r47
GR_P_1     = r48
GR_P_2     = r49
GR_P_3     = r50
GR_P_4     = r51
GR_START   = r52
GR_SEGMENT = r53
GR_A       = r54
GR_B       = r55
GR_C       = r56
GR_D       = r57
GR_E       = r58
GR_TEMP1   = r59
GR_TEMP2   = r60
GR_TEMP3   = r61
GR_TEMP4   = r62
GR_TEMP5   = r63
GR_TEMP6   = r64
GR_rshf    = r64

RODATA
.align 64

LOCAL_OBJECT_START(Constants_Bits_of_2_by_pi)
data8 0x0000000000000000,0xA2F9836E4E441529
data8 0xFC2757D1F534DDC0,0xDB6295993C439041
data8 0xFE5163ABDEBBC561,0xB7246E3A424DD2E0
data8 0x06492EEA09D1921C,0xFE1DEB1CB129A73E
data8 0xE88235F52EBB4484,0xE99C7026B45F7E41
data8 0x3991D639835339F4,0x9C845F8BBDF9283B
data8 0x1FF897FFDE05980F,0xEF2F118B5A0A6D1F
data8 0x6D367ECF27CB09B7,0x4F463F669E5FEA2D
data8 0x7527BAC7EBE5F17B,0x3D0739F78A5292EA
data8 0x6BFB5FB11F8D5D08,0x56033046FC7B6BAB
data8 0xF0CFBC209AF4361D,0xA9E391615EE61B08
data8 0x6599855F14A06840,0x8DFFD8804D732731
data8 0x06061556CA73A8C9,0x60E27BC08C6B47C4
data8 0x19C367CDDCE8092A,0x8359C4768B961CA6
data8 0xDDAF44D15719053E,0xA5FF07053F7E33E8
data8 0x32C2DE4F98327DBB,0xC33D26EF6B1E5EF8
data8 0x9F3A1F35CAF27F1D,0x87F121907C7C246A
data8 0xFA6ED5772D30433B,0x15C614B59D19C3C2
data8 0xC4AD414D2C5D000C,0x467D862D71E39AC6
data8 0x9B0062337CD2B497,0xA7B4D55537F63ED7
data8 0x1810A3FC764D2A9D,0x64ABD770F87C6357
data8 0xB07AE715175649C0,0xD9D63B3884A7CB23
data8 0x24778AD623545AB9,0x1F001B0AF1DFCE19
data8 0xFF319F6A1E666157,0x9947FBACD87F7EB7
data8 0x652289E83260BFE6,0xCDC4EF09366CD43F
data8 0x5DD7DE16DE3B5892,0x9BDE2822D2E88628
data8 0x4D58E232CAC616E3,0x08CB7DE050C017A7
data8 0x1DF35BE01834132E,0x6212830148835B8E
data8 0xF57FB0ADF2E91E43,0x4A48D36710D8DDAA
data8 0x425FAECE616AA428,0x0AB499D3F2A6067F
data8 0x775C83C2A3883C61,0x78738A5A8CAFBDD7
data8 0x6F63A62DCBBFF4EF,0x818D67C12645CA55
data8 0x36D9CAD2A8288D61,0xC277C9121426049B
data8 0x4612C459C444C5C8,0x91B24DF31700AD43
data8 0xD4E5492910D5FDFC,0xBE00CC941EEECE70
data8 0xF53E1380F1ECC3E7,0xB328F8C79405933E
data8 0x71C1B3092EF3450B,0x9C12887B20AB9FB5
data8 0x2EC292472F327B6D,0x550C90A7721FE76B
data8 0x96CB314A1679E279,0x4189DFF49794E884
data8 0xE6E29731996BED88,0x365F5F0EFDBBB49A
data8 0x486CA46742727132,0x5D8DB8159F09E5BC
data8 0x25318D3974F71C05,0x30010C0D68084B58
data8 0xEE2C90AA4702E774,0x24D6BDA67DF77248
data8 0x6EEF169FA6948EF6,0x91B45153D1F20ACF
data8 0x3398207E4BF56863,0xB25F3EDD035D407F
data8 0x8985295255C06437,0x10D86D324832754C
data8 0x5BD4714E6E5445C1,0x090B69F52AD56614
data8 0x9D072750045DDB3B,0xB4C576EA17F9877D
data8 0x6B49BA271D296996,0xACCCC65414AD6AE2
data8 0x9089D98850722CBE,0xA4049407777030F3
data8 0x27FC00A871EA49C2,0x663DE06483DD9797
data8 0x3FA3FD94438C860D,0xDE41319D39928C70
data8 0xDDE7B7173BDF082B,0x3715A0805C93805A
data8 0x921110D8E80FAF80,0x6C4BFFDB0F903876
data8 0x185915A562BBCB61,0xB989C7BD401004F2
data8 0xD2277549F6B6EBBB,0x22DBAA140A2F2689
data8 0x768364333B091A94,0x0EAA3A51C2A31DAE
data8 0xEDAF12265C4DC26D,0x9C7A2D9756C0833F
data8 0x03F6F0098C402B99,0x316D07B43915200C
data8 0x5BC3D8C492F54BAD,0xC6A5CA4ECD37A736
data8 0xA9E69492AB6842DD,0xDE6319EF8C76528B
data8 0x6837DBFCABA1AE31,0x15DFA1AE00DAFB0C
data8 0x664D64B705ED3065,0x29BF56573AFF47B9
data8 0xF96AF3BE75DF9328,0x3080ABF68C6615CB
data8 0x040622FA1DE4D9A4,0xB33D8F1B5709CD36
data8 0xE9424EA4BE13B523,0x331AAAF0A8654FA5
data8 0xC1D20F3F0BCD785B,0x76F923048B7B7217
data8 0x8953A6C6E26E6F00,0xEBEF584A9BB7DAC4
data8 0xBA66AACFCF761D02,0xD12DF1B1C1998C77
data8 0xADC3DA4886A05DF7,0xF480C62FF0AC9AEC
data8 0xDDBC5C3F6DDED01F,0xC790B6DB2A3A25A3
data8 0x9AAF009353AD0457,0xB6B42D297E804BA7
data8 0x07DA0EAA76A1597B,0x2A12162DB7DCFDE5
data8 0xFAFEDB89FDBE896C,0x76E4FCA90670803E
data8 0x156E85FF87FD073E,0x2833676186182AEA
data8 0xBD4DAFE7B36E6D8F,0x3967955BBF3148D7
data8 0x8416DF30432DC735,0x6125CE70C9B8CB30
data8 0xFD6CBFA200A4E46C,0x05A0DD5A476F21D2
data8 0x1262845CB9496170,0xE0566B0152993755
data8 0x50B7D51EC4F1335F,0x6E13E4305DA92E85
data8 0xC3B21D3632A1A4B7,0x08D4B1EA21F716E4
data8 0x698F77FF2780030C,0x2D408DA0CD4F99A5
data8 0x20D3A2B30A5D2F42,0xF9B4CBDA11D0BE7D
data8 0xC1DB9BBD17AB81A2,0xCA5C6A0817552E55
data8 0x0027F0147F8607E1,0x640B148D4196DEBE
data8 0x872AFDDAB6256B34,0x897BFEF3059EBFB9
data8 0x4F6A68A82A4A5AC4,0x4FBCF82D985AD795
data8 0xC7F48D4D0DA63A20,0x5F57A4B13F149538
data8 0x800120CC86DD71B6,0xDEC9F560BF11654D
data8 0x6B0701ACB08CD0C0,0xB24855510EFB1EC3
data8 0x72953B06A33540C0,0x7BDC06CC45E0FA29
data8 0x4EC8CAD641F3E8DE,0x647CD8649B31BED9
data8 0xC397A4D45877C5E3,0x6913DAF03C3ABA46
data8 0x18465F7555F5BDD2,0xC6926E5D2EACED44
data8 0x0E423E1C87C461E9,0xFD29F3D6E7CA7C22
data8 0x35916FC5E0088DD7,0xFFE26A6EC6FDB0C1
data8 0x0893745D7CB2AD6B,0x9D6ECD7B723E6A11
data8 0xC6A9CFF7DF7329BA,0xC9B55100B70DB2E2
data8 0x24BA74607DE58AD8,0x742C150D0C188194
data8 0x667E162901767A9F,0xBEFDFDEF4556367E
data8 0xD913D9ECB9BA8BFC,0x97C427A831C36EF1
data8 0x36C59456A8D8B5A8,0xB40ECCCF2D891234
data8 0x576F89562CE3CE99,0xB920D6AA5E6B9C2A
data8 0x3ECC5F114A0BFDFB,0xF4E16D3B8E2C86E2
data8 0x84D4E9A9B4FCD1EE,0xEFC9352E61392F44
data8 0x2138C8D91B0AFC81,0x6A4AFBD81C2F84B4
data8 0x538C994ECC2254DC,0x552AD6C6C096190B
data8 0xB8701A649569605A,0x26EE523F0F117F11
data8 0xB5F4F5CBFC2DBC34,0xEEBC34CC5DE8605E
data8 0xDD9B8E67EF3392B8,0x17C99B5861BC57E1
data8 0xC68351103ED84871,0xDDDD1C2DA118AF46
data8 0x2C21D7F359987AD9,0xC0549EFA864FFC06
data8 0x56AE79E536228922,0xAD38DC9367AAE855
data8 0x3826829BE7CAA40D,0x51B133990ED7A948
data8 0x0569F0B265A7887F,0x974C8836D1F9B392
data8 0x214A827B21CF98DC,0x9F405547DC3A74E1
data8 0x42EB67DF9DFE5FD4,0x5EA4677B7AACBAA2
data8 0xF65523882B55BA41,0x086E59862A218347
data8 0x39E6E389D49EE540,0xFB49E956FFCA0F1C
data8 0x8A59C52BFA94C5C1,0xD3CFC50FAE5ADB86
data8 0xC5476243853B8621,0x94792C8761107B4C
data8 0x2A1A2C8012BF4390,0x2688893C78E4C4A8
data8 0x7BDBE5C23AC4EAF4,0x268A67F7BF920D2B
data8 0xA365B1933D0B7CBD,0xDC51A463DD27DDE1
data8 0x6919949A9529A828,0xCE68B4ED09209F44
data8 0xCA984E638270237C,0x7E32B90F8EF5A7E7
data8 0x561408F1212A9DB5,0x4D7E6F5119A5ABF9
data8 0xB5D6DF8261DD9602,0x36169F3AC4A1A283
data8 0x6DED727A8D39A9B8,0x825C326B5B2746ED
data8 0x34007700D255F4FC,0x4D59018071E0E13F
data8 0x89B295F364A8F1AE,0xA74B38FC4CEAB2BB
LOCAL_OBJECT_END(Constants_Bits_of_2_by_pi)

LOCAL_OBJECT_START(Constants_Bits_of_pi_by_2)
data8 0xC90FDAA22168C234,0x00003FFF
data8 0xC4C6628B80DC1CD1,0x00003FBF
LOCAL_OBJECT_END(Constants_Bits_of_pi_by_2)

.section .text
.global __libm_pi_by_2_reduce#
.proc __libm_pi_by_2_reduce#
.align 32

__libm_pi_by_2_reduce:

//    X is in f8
//    Place the two-piece result r (r_hi) in f8 and c (r_lo) in f9
//    N is returned in r8

{ .mfi
      alloc  r34 = ar.pfs,2,34,0,0
      fsetc.s3 0x00,0x7F     // Set sf3 to round to zero, 82-bit prec, td, ftz
      nop.i 999
}
{ .mfi
      addl           GR_BASE   = @ltoff(Constants_Bits_of_2_by_pi#), gp
      nop.f 999
      mov GR_BIASL63 = 0x1003E
}
;;


//    L         -1-2-3-4
//    0 0 0 0 0. 1 0 1 0
//    M          0 1 2 .... 63, 64 65 ... 127, 128
//     ---------------------------------------------
//    Segment 0.        1     ,      2       ,    3
//    START = M - 63                        M = 128 becomes 65
//    LENGTH1  = START & 0x3F               65 become position 1
//    SEGMENT  = shr(START,6) + 1      0 maps to 1,   64 maps to 2,
//    LENGTH2  = 64 - LENGTH1
//    Address_BASE = shladd(SEGMENT,3) + BASE


{ .mmi
      getf.exp GR_Exp_x = FR_input_X
      ld8 GR_BASE = [GR_BASE]
      mov GR_TEMP5 = 0x0FFFE
}
;;

//    Define sigma_C := 2^63; sigma_B := 2^(-1); sigma_A := 2^(-65).
{ .mmi
      getf.sig GR_x_lo = FR_input_X
      mov GR_TEMP6 = 0x0FFBE
      nop.i 999
}
;;

//    Special Code for testing DE arguments
//          movl GR_BIASL63 = 0x0000000000013FFE
//          movl GR_x_lo = 0xFFFFFFFFFFFFFFFF
//          setf.exp FR_X = GR_BIASL63
//          setf.sig FR_ScaleP3 = GR_x_lo
//          fmerge.se FR_X = FR_X,FR_ScaleP3
//    Set sgn_x := sign(x); x := |x|; x_lo := 2 lsb of x.
//    2/pi is stored contiguously as
//    0x00000000 0x00000000.0xA2F....
//    M = EXP - BIAS  ( M >= 63)
//    Given x = 2^m * 1.xxxx...xxx; we calculate L := 62 - m.
//    Thus -1 <= L <= -16321.
{ .mmi
      setf.exp FR_sigma_B = GR_TEMP5
      setf.exp FR_sigma_A = GR_TEMP6
      extr.u GR_M = GR_Exp_x,0,17
}
;;

{ .mii
      and  GR_x_lo = 0x03,GR_x_lo
      sub  GR_START = GR_M,GR_BIASL63
      add  GR_BASE = 8,GR_BASE           // To effectively add 1 to SEGMENT
}
;;

{ .mii
      and  GR_LENGTH1 = 0x3F,GR_START
      shr.u  GR_SEGMENT = GR_START,6
      nop.i 999
}
;;

{ .mmi
      shladd GR_BASE = GR_SEGMENT,3,GR_BASE
      sub  GR_LENGTH2 = 0x40,GR_LENGTH1
      cmp.le p6,p7 = 0x2,GR_LENGTH1
}
;;

//    P_0 is the two bits corresponding to bit positions L+2 and L+1
//    P_1 is the 64-bit starting at bit position  L
//    P_2 is the 64-bit starting at bit position  L-64
//    P_3 is the 64-bit starting at bit position  L-128
//    P_4 is the 64-bit starting at bit position  L-192
//    P_1 is made up of Alo and Bhi
//    P_1 = deposit Alo, position 0, length2  into P_1,position length1
//          deposit Bhi, position length2, length1 into P_1, position 0
//    P_2 is made up of Blo and Chi
//    P_2 = deposit Blo, position 0, length2  into P_2, position length1
//          deposit Chi, position length2, length1 into P_2, position 0
//    P_3 is made up of Clo and Dhi
//    P_3 = deposit Clo, position 0, length2  into P_3, position length1
//          deposit Dhi, position length2, length1 into P_3, position 0
//    P_4 is made up of Clo and Dhi
//    P_4 = deposit Dlo, position 0, length2  into P_4, position length1
//          deposit Ehi, position length2, length1 into P_4, position 0
{ .mfi
      ld8 GR_A = [GR_BASE],8
      fabs FR_X = FR_input_X
(p7)  cmp.eq.unc p8,p9 = 0x1,GR_LENGTH1
}
;;

//    ld_64 A at Base and increment Base by 8
//    ld_64 B at Base and increment Base by 8
//    ld_64 C at Base and increment Base by 8
//    ld_64 D at Base and increment Base by 8
//    ld_64 E at Base and increment Base by 8
//                                          A/B/C/D
//                                    ---------------------
//    A, B, C, D, and E look like    | length1 | length2   |
//                                    ---------------------
//                                       hi        lo
{ .mlx
      ld8 GR_B = [GR_BASE],8
      movl GR_rshf = 0x43e8000000000000   // 1.10000 2^63 for right shift N_fix
}
;;

{ .mmi
      ld8 GR_C = [GR_BASE],8
      nop.m 999
(p8)  extr.u GR_Temp = GR_A,63,1
}
;;

//    If length1 >= 2,
//       P_0 = deposit Ahi, position length2, 2 bit into P_0 at position 0.
{ .mii
      ld8 GR_D = [GR_BASE],8
      shl GR_TEMP1 = GR_A,GR_LENGTH1   // MM instruction
(p6)  shr.u GR_P_0 = GR_A,GR_LENGTH2   // MM instruction
}
;;

{ .mii
      ld8 GR_E = [GR_BASE],-40
      shl GR_TEMP2 = GR_B,GR_LENGTH1   // MM instruction
      shr.u GR_P_1 = GR_B,GR_LENGTH2   // MM instruction
}
;;

//    Else
//       Load 16 bit of ASUB from (Base_Address_of_A - 2)
//       P_0 = ASUB & 0x3
//       If length1 == 0,
//          P_0 complete
//       Else
//          Deposit element 63 from Ahi and place in element 0 of P_0.
//       Endif
//    Endif

{ .mii
(p7)  ld2 GR_ASUB = [GR_BASE],8
      shl GR_TEMP3 = GR_C,GR_LENGTH1   // MM instruction
      shr.u GR_P_2 = GR_C,GR_LENGTH2   // MM instruction
}
;;

{ .mii
      setf.d FR_RSHF = GR_rshf         // Form right shift const 1.100 * 2^63
      shl GR_TEMP4 = GR_D,GR_LENGTH1   // MM instruction
      shr.u GR_P_3 = GR_D,GR_LENGTH2   // MM instruction
}
;;

{ .mmi
(p7)  and GR_P_0 = 0x03,GR_ASUB
(p6)  and GR_P_0 = 0x03,GR_P_0
      shr.u GR_P_4 = GR_E,GR_LENGTH2   // MM instruction
}
;;

{ .mmi
      nop.m 999
      or GR_P_1 = GR_P_1,GR_TEMP1
(p8)  and GR_P_0 = 0x1,GR_P_0
}
;;

{ .mmi
      setf.sig FR_p_1 = GR_P_1
      or GR_P_2 = GR_P_2,GR_TEMP2
(p8)  shladd GR_P_0 = GR_P_0,1,GR_Temp
}
;;

{ .mmf
      setf.sig FR_p_2 = GR_P_2
      or GR_P_3 = GR_P_3,GR_TEMP3
      fmerge.se FR_X = FR_sigma_B,FR_X
}
;;

{ .mmi
      setf.sig FR_p_3 = GR_P_3
      or GR_P_4 = GR_P_4,GR_TEMP4
      pmpy2.r GR_M = GR_P_0,GR_x_lo
}
;;

//    P_1, P_2, P_3, P_4 are integers. They should be
//    2^(L-63)     * P_1;
//    2^(L-63-64)  * P_2;
//    2^(L-63-128) * P_3;
//    2^(L-63-192) * P_4;
//    Since each of them need to be multiplied to x, we would scale
//    both x and the P_j's by some convenient factors: scale each
//    of P_j's up by 2^(63-L), and scale x down by 2^(L-63).
//    p_1 := fcvt.xf ( P_1 )
//    p_2 := fcvt.xf ( P_2 ) * 2^(-64)
//    p_3 := fcvt.xf ( P_3 ) * 2^(-128)
//    p_4 := fcvt.xf ( P_4 ) * 2^(-192)
//    x= Set x's exp to -1 because 2^m*1.x...x *2^(L-63)=2^(-1)*1.x...xxx
//             ---------   ---------   ---------
//             |  P_1  |   |  P_2  |   |  P_3  |
//             ---------   ---------   ---------
//                                           ---------
//            X                              |   X   |
//                                           ---------
//      ----------------------------------------------------
//                               ---------   ---------
//                               |  A_hi |   |  A_lo |
//                               ---------   ---------
//                   ---------   ---------
//                   |  B_hi |   |  B_lo |
//                   ---------   ---------
//       ---------   ---------
//       |  C_hi |   |  C_lo |
//       ---------   ---------
//     ====================================================
//    -----------   ---------   ---------   ---------
//    |    S_0  |   |  S_1  |   |  S_2  |   |  S_3  |
//    -----------   ---------   ---------   ---------
//    |            |___ binary point
//    |___ possibly one more bit
//
//    Let FPSR3 be set to round towards zero with widest precision
//    and exponent range. Unless an explicit FPSR is given,
//    round-to-nearest with widest precision and exponent range is
//    used.
{ .mmi
      setf.sig FR_p_4 = GR_P_4
      mov GR_TEMP1 = 0x0FFBF
      nop.i 999
}
;;

{ .mmi
      setf.exp FR_ScaleP2 = GR_TEMP1
      mov GR_TEMP2 = 0x0FF7F
      nop.i 999
}
;;

{ .mmi
      setf.exp FR_ScaleP3 = GR_TEMP2
      mov GR_TEMP4 = 0x1003E
      nop.i 999
}
;;

{ .mmf
      setf.exp FR_sigma_C = GR_TEMP4
      mov GR_Temp = 0x0FFDE
      fcvt.xuf.s1 FR_p_1 = FR_p_1
}
;;

{ .mfi
      setf.exp FR_TWOM33 = GR_Temp
      fcvt.xuf.s1 FR_p_2 = FR_p_2
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fcvt.xuf.s1 FR_p_3 = FR_p_3
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fcvt.xuf.s1 FR_p_4 = FR_p_4
      nop.i 999
}
;;

//    Tmp_C := fmpy.fpsr3( x, p_1 );
//    Tmp_B := fmpy.fpsr3( x, p_2 );
//    Tmp_A := fmpy.fpsr3( x, p_3 );
//    If Tmp_C >= sigma_C then
//      C_hi := Tmp_C;
//      C_lo := x*p_1 - C_hi ...fma, exact
//    Else
//      C_hi := fadd.fpsr3(sigma_C, Tmp_C) - sigma_C
//      C_lo := x*p_1 - C_hi ...fma, exact
//    End If
//    If Tmp_B >= sigma_B then
//      B_hi := Tmp_B;
//      B_lo := x*p_2 - B_hi ...fma, exact
//    Else
//      B_hi := fadd.fpsr3(sigma_B, Tmp_B) - sigma_B
//      B_lo := x*p_2 - B_hi ...fma, exact
//    End If
//    If Tmp_A >= sigma_A then
//      A_hi := Tmp_A;
//      A_lo := x*p_3 - A_hi ...fma, exact
//    Else
//      A_hi := fadd.fpsr3(sigma_A, Tmp_A) - sigma_A
//      Exact, regardless ...of rounding direction
//      A_lo := x*p_3 - A_hi ...fma, exact
//    Endif
{ .mfi
      nop.m 999
      fmpy.s3 FR_Tmp_C = FR_X,FR_p_1
      nop.i 999
}
;;

{ .mfi
      mov GR_TEMP3 = 0x0FF3F
      fmpy.s1 FR_p_2 = FR_p_2,FR_ScaleP2
      nop.i 999
}
;;

{ .mmf
      setf.exp FR_ScaleP4 = GR_TEMP3
      mov GR_TEMP4 = 0x10045
      fmpy.s1 FR_p_3 = FR_p_3,FR_ScaleP3
}
;;

{ .mfi
      nop.m 999
      fadd.s3 FR_C_hi = FR_sigma_C,FR_Tmp_C   // For Tmp_C < sigma_C case
      nop.i 999
}
;;

{ .mmf
      setf.exp FR_Tmp2_C = GR_TEMP4
      nop.m 999
      fmpy.s3 FR_Tmp_B = FR_X,FR_p_2
}
;;

{ .mfi
      addl           GR_BASE   = @ltoff(Constants_Bits_of_pi_by_2#), gp
      fcmp.ge.s1 p12,  p9 = FR_Tmp_C,FR_sigma_C
      nop.i 999
}
{ .mfi
      nop.m 999
      fmpy.s3 FR_Tmp_A = FR_X,FR_p_3
      nop.i 99
}
;;

{ .mfi
      ld8 GR_BASE = [GR_BASE]
(p12) mov FR_C_hi = FR_Tmp_C
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fsub.s1 FR_C_hi = FR_C_hi,FR_sigma_C
      nop.i 999
}
;;



//   End If
//   Step 3. Get reduced argument
//   If sgn_x == 0 (that is original x is positive)
//      D_hi := Pi_by_2_hi
//      D_lo := Pi_by_2_lo
//      Load from table
//   Else
//      D_hi := neg_Pi_by_2_hi
//      D_lo := neg_Pi_by_2_lo
//      Load from table
//   End If

{ .mfi
      nop.m 999
      fmpy.s1 FR_p_4 = FR_p_4,FR_ScaleP4
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s3 FR_B_hi = FR_sigma_B,FR_Tmp_B     // For Tmp_B < sigma_B case
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s3 FR_A_hi = FR_sigma_A,FR_Tmp_A     // For Tmp_A < sigma_A case
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fcmp.ge.s1 p13, p10 = FR_Tmp_B,FR_sigma_B
      nop.i 999
}
{ .mfi
      nop.m 999
      fms.s1 FR_C_lo = FR_X,FR_p_1,FR_C_hi
      nop.i 999
}
;;

{ .mfi
      ldfe FR_D_hi = [GR_BASE],16
      fcmp.ge.s1 p14, p11 = FR_Tmp_A,FR_sigma_A
      nop.i 999
}
;;

{ .mfi
      ldfe FR_D_lo = [GR_BASE]
(p13) mov FR_B_hi = FR_Tmp_B
      nop.i 999
}
{ .mfi
      nop.m 999
(p10) fsub.s1 FR_B_hi = FR_B_hi,FR_sigma_B
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p14) mov FR_A_hi = FR_Tmp_A
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fsub.s1 FR_A_hi = FR_A_hi,FR_sigma_A
      nop.i 999
}
;;

//    Note that C_hi is of integer value. We need only the
//    last few bits. Thus we can ensure C_hi is never a big
//    integer, freeing us from overflow worry.
//    Tmp_C := fadd.fpsr3( C_hi, 2^(70) ) - 2^(70);
//    Tmp_C is the upper portion of C_hi
{ .mfi
      nop.m 999
      fadd.s3 FR_Tmp_C = FR_C_hi,FR_Tmp2_C
      tbit.z p12,p9 = GR_Exp_x, 17
}
;;

{ .mfi
      nop.m 999
      fms.s1 FR_B_lo = FR_X,FR_p_2,FR_B_hi
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s3 FR_A = FR_B_hi,FR_C_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fms.s1 FR_A_lo = FR_X,FR_p_3,FR_A_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsub.s1 FR_Tmp_C = FR_Tmp_C,FR_Tmp2_C
      nop.i 999
}
;;

//    *******************
//    Step 2. Get N and f
//    *******************
//    We have all the components to obtain
//    S_0, S_1, S_2, S_3 and thus N and f. We start by adding
//    C_lo and B_hi. This sum together with C_hi estimates
//    N and f well.
//    A := fadd.fpsr3( B_hi, C_lo )
//    B := max( B_hi, C_lo )
//    b := min( B_hi, C_lo )
{ .mfi
      nop.m 999
      fmax.s1 FR_B = FR_B_hi,FR_C_lo
      nop.i 999
}
;;

// We use a right-shift trick to get the integer part of A into the rightmost
// bits of the significand by adding 1.1000..00 * 2^63.  This operation is good
// if |A| < 2^61, which it is in this case.  We are doing this to save a few
// cycles over using fcvt.fx followed by fnorm.  The second step of the trick
// is to subtract the same constant to float the rounded integer into a fp reg.

{ .mfi
      nop.m 999
//    N := round_to_nearest_integer_value( A );
      fma.s1 FR_N_fix = FR_A, f1, FR_RSHF
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmin.s1 FR_b = FR_B_hi,FR_C_lo
      nop.i 999
}
{ .mfi
      nop.m 999
//    C_hi := C_hi - Tmp_C ...0 <= C_hi < 2^7
      fsub.s1 FR_C_hi = FR_C_hi,FR_Tmp_C
      nop.i 999
}
;;

{ .mfi
      nop.m 999
//    a := (B - A) + b: Exact - note that a is either 0 or 2^(-64).
      fsub.s1 FR_a = FR_B,FR_A
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fms.s1 FR_N = FR_N_fix, f1, FR_RSHF
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 FR_a = FR_a,FR_b
      nop.i 999
}
;;

//    f := A - N; Exact because lsb(A) >= 2^(-64) and |f| <= 1/2.
//    N := convert to integer format( C_hi + N );
//    M := P_0 * x_lo;
//    N := N + M;
{ .mfi
      nop.m 999
      fsub.s1 FR_f = FR_A,FR_N
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_N = FR_N,FR_C_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p9)  fsub.s1 FR_D_hi = f0, FR_D_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p9)  fsub.s1 FR_D_lo = f0, FR_D_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 FR_g = FR_A_hi,FR_B_lo          // For Case 1, g=A_hi+B_lo
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s3 FR_A = FR_A_hi,FR_B_lo          // For Case 2, A=A_hi+B_lo w/ sf3
      nop.i 999
}
;;

{ .mfi
      mov GR_Temp = 0x0FFCD                   // For Case 2, exponent of 2^-50
      fmax.s1 FR_B = FR_A_hi,FR_B_lo          // For Case 2, B=max(A_hi,B_lo)
      nop.i 999
}
;;

//    f = f + a      Exact because a is 0 or 2^(-64);
//    the msb of the sum is <= 1/2 and lsb >= 2^(-64).
{ .mfi
      setf.exp FR_TWOM50 = GR_Temp            // For Case 2, form 2^-50
      fcvt.fx.s1 FR_N = FR_N
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_f = FR_f,FR_a
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmin.s1 FR_b = FR_A_hi,FR_B_lo          // For Case 2, b=min(A_hi,B_lo)
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsub.s1 FR_a = FR_B,FR_A                // For Case 2, a=B-A
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fadd.s1 FR_s_hi = FR_f,FR_g             // For Case 1, s_hi=f+g
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_f_hi = FR_A,FR_f             // For Case 2, f_hi=A+f
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fabs FR_f_abs = FR_f
      nop.i 999
}
;;

{ .mfi
      getf.sig GR_N = FR_N
      fsetc.s3 0x7F,0x40                 // Reset sf3 to user settings + td
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fsub.s1 FR_s_lo = FR_f,FR_s_hi          // For Case 1, s_lo=f-s_hi
      nop.i 999
}
{ .mfi
      nop.m 999
      fsub.s1 FR_f_lo = FR_f,FR_f_hi          // For Case 2, f_lo=f-f_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
      fmpy.s1 FR_r_hi = FR_s_hi,FR_D_hi       // For Case 1, r_hi=s_hi*D_hi
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_a = FR_a,FR_b                // For Case 2, a=a+b
      nop.i 999
}
;;


//    If sgn_x == 1 (that is original x was negative)
//       N := 2^10 - N
//       this maintains N to be non-negative, but still
//       equivalent to the (negated N) mod 4.
//    End If
{ .mfi
      add GR_N = GR_N,GR_M
      fcmp.ge.s1 p13, p10 = FR_f_abs,FR_TWOM33
      mov GR_Temp = 0x00400
}
;;

{ .mfi
(p9)  sub GR_N = GR_Temp,GR_N
      fadd.s1 FR_s_lo = FR_s_lo,FR_g           // For Case 1, s_lo=s_lo+g
      nop.i 999
}
{ .mfi
      nop.m 999
      fadd.s1 FR_f_lo = FR_f_lo,FR_A           // For Case 2, f_lo=f_lo+A
      nop.i 999
}
;;

//       a := (B - A) + b      Exact.
//       Note that a is either 0 or 2^(-128).
//       f_hi := A + f;
//       f_lo := (f - f_hi) + A
//       f_lo=f-f_hi is exact because either |f| >= |A|, in which
//       case f-f_hi is clearly exact; or otherwise, 0<|f|<|A|
//       means msb(f) <= msb(A) = 2^(-64) => |f| = 2^(-64).
//       If f = 2^(-64), f-f_hi involves cancellation and is
//       exact. If f = -2^(-64), then A + f is exact. Hence
//       f-f_hi is -A exactly, giving f_lo = 0.
//       f_lo := f_lo + a;

//    If |f| >= 2^(-33)
//       Case 1
//       CASE := 1
//       g := A_hi + B_lo;
//       s_hi := f + g;
//       s_lo := (f - s_hi) + g;
//   Else
//       Case 2
//       CASE := 2
//       A := fadd.fpsr3( A_hi, B_lo )
//       B := max( A_hi, B_lo )
//       b := min( A_hi, B_lo )

{ .mfi
      nop.m 999
(p10) fcmp.ge.unc.s1 p14, p11 = FR_f_abs,FR_TWOM50
      nop.i 999
}
{ .mfi
      nop.m 999
(p13) fms.s1 FR_r_lo = FR_s_hi,FR_D_hi,FR_r_hi //For Case 1, r_lo=s_hi*D_hi+r_hi
      nop.i 999
}
;;

//       If |f| >= 2^(-50) then
//          s_hi := f_hi;
//          s_lo := f_lo;
//       Else
//          f_lo := (f_lo + A_lo) + x*p_4
//          s_hi := f_hi + f_lo
//          s_lo := (f_hi - s_hi) + f_lo
//       End If
{ .mfi
      nop.m 999
(p14) mov FR_s_hi = FR_f_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p10) fadd.s1 FR_f_lo = FR_f_lo,FR_a
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p14) mov FR_s_lo = FR_f_lo
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fadd.s1 FR_f_lo = FR_f_lo,FR_A_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p11) fma.s1 FR_f_lo = FR_X,FR_p_4,FR_f_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p13) fma.s1 FR_r_lo = FR_s_hi,FR_D_lo,FR_r_lo //For Case 1, r_lo=s_hi*D_lo+r_lo
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fadd.s1 FR_s_hi = FR_f_hi,FR_f_lo
      nop.i 999
}
;;

//   r_hi :=  s_hi*D_hi
//   r_lo :=  s_hi*D_hi - r_hi  with fma
//   r_lo := (s_hi*D_lo + r_lo) + s_lo*D_hi
{ .mfi
      nop.m 999
(p10) fmpy.s1 FR_r_hi = FR_s_hi,FR_D_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fsub.s1 FR_s_lo = FR_f_hi,FR_s_hi
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p10) fms.s1 FR_r_lo = FR_s_hi,FR_D_hi,FR_r_hi
      nop.i 999
}
{ .mfi
      nop.m 999
(p11) fadd.s1 FR_s_lo = FR_s_lo,FR_f_lo
      nop.i 999
}
;;

{ .mfi
      nop.m 999
(p10) fma.s1 FR_r_lo = FR_s_hi,FR_D_lo,FR_r_lo
      nop.i 999
}
;;

//   Return  N, r_hi, r_lo
//   We do not return CASE
{ .mfb
      nop.m 999
      fma.s1 FR_r_lo = FR_s_lo,FR_D_hi,FR_r_lo
      br.ret.sptk   b0
}
;;

.endp __libm_pi_by_2_reduce#
