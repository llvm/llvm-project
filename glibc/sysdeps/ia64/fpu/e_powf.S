.file "powf.s"


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
// History
//==============================================================
// 02/02/00 Initial version
// 02/03/00 Added p12 to definite over/under path. With odd power we did not
//          maintain the sign of x in this path.
// 04/04/00 Unwind support added
// 04/19/00 pow(+-1,inf) now returns NaN
//          pow(+-val, +-inf) returns 0 or inf, but now does not call error
//          support
//          Added s1 to fcvt.fx because invalid flag was incorrectly set.
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 09/07/00 Improved performance by eliminating bank conflicts and other stalls,
//          and tweaking the critical path
// 09/08/00 Per c99, pow(+-1,inf) now returns 1, and pow(+1,nan) returns 1
// 09/28/00 Updated NaN**0 path
// 01/20/01 Fixed denormal flag settings.
// 02/13/01 Improved speed.
// 03/19/01 Reordered exp polynomial to improve speed and eliminate monotonicity
//          problem in round up, down, and to zero modes.  Also corrected
//          overflow result when x negative, y odd in round up, down, zero.
// 06/14/01 Added brace missing from bundle
// 12/10/01 Corrected case where x negative, 2^23 <= |y| < 2^24, y odd integer.
// 02/08/02 Fixed overflow/underflow cases that were not calling error support.
// 05/20/02 Cleaned up namespace and sf0 syntax
// 08/29/02 Improved Itanium 2 performance
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 10/09/03 Modified algorithm to improve performance, reduce table size, and
//          fix boundary case powf(2.0,-150.0)
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// float powf(float x, float y)
//
// Overview of operation
//==============================================================
//
// Three steps...
// 1. Log(x)
// 2. y Log(x)
// 3. exp(y log(x))
//
// This means we work with the absolute value of x and merge in the sign later.
//      Log(x) = G + delta + r -rsq/2 + p
// G,delta depend on the exponent of x and table entries. The table entries are
// indexed by the exponent of x, called K.
//
// The G and delta come out of the reduction; r is the reduced x.
//
// B = frcpa(x)
// xB-1 is small means that B is the approximate inverse of x.
//
//      Log(x) = Log( (1/B)(Bx) )
//             = Log(1/B) + Log(Bx)
//             = Log(1/B) + Log( 1 + (Bx-1))
//
//      x  = 2^K 1.x_1x_2.....x_52
//      B= frcpa(x) = 2^-k Cm
//      Log(1/B) = Log(1/(2^-K Cm))
//      Log(1/B) = Log((2^K/ Cm))
//      Log(1/B) = K Log(2) + Log(1/Cm)
//
//      Log(x)   = K Log(2) + Log(1/Cm) + Log( 1 + (Bx-1))
//
// If you take the significand of x, set the exponent to true 0, then Cm is
// the frcpa. We tabulate the Log(1/Cm) values. There are 256 of them.
// The frcpa table is indexed by 8 bits, the x_1 thru x_8.
// m = x_1x_2...x_8 is an 8-bit index.
//
//      Log(1/Cm) = log(1/frcpa(1+m/256)) where m goes from 0 to 255.
//
// We tabluate as one double, T for single precision power
//
//      Log(x)   = (K Log(2)_hi + T) + (K Log(2)_lo) + Log( 1 + (Bx-1))
//      Log(x)   =  G                +     delta     + Log( 1 + (Bx-1))
//
// The Log( 1 + (Bx-1)) can be calculated as a series in r = Bx-1.
//
//      Log( 1 + (Bx-1)) = r - rsq/2 + p
//        where p = r^3(P0 + P1*r + P2*r^2)
//
// Then,
//
//      yLog(x) = yG + y delta + y(r-rsq/2) + yp
//      yLog(x) = Z1 + e3      + Z2         + Z3
//
//
//     exp(yLog(x)) = exp(Z1 + Z2) exp(Z3) exp(e3)
//
//
//       exp(Z3) is another series.
//       exp(e3) is approximated as f3 = 1 +  e3
//
//       exp(Z1 + Z2) = exp(Z)
//       Z (128/log2) = number of log2/128 in Z is N
//
//       s = Z - N log2/128
//
//       exp(Z)       = exp(s) exp(N log2/128)
//
//       exp(r)       = exp(Z - N log2/128)
//
//      r = s + d = (Z - N (log2/128)_hi) -N (log2/128)_lo
//                =  Z - N (log2/128)
//
//      Z         = s+d +N (log2/128)
//
//      exp(Z)    = exp(s) (1+d) exp(N log2/128)
//
//      N = M 128 + n
//
//      N log2/128 = M log2 + n log2/128
//
//      n is 8 binary digits = n_7n_6...n_1
//
//      n log2/128 = n_7n_6n_5 16 log2/128 + n_4n_3n_2n_1 log2/128
//      n log2/128 = n_7n_6n_5 log2/8 + n_4n_3n_2n_1 log2/128
//      n log2/128 = I2 log2/8 + I1 log2/128
//
//      N log2/128 = M log2 + I2 log2/8 + I1 log2/128
//
//      exp(Z)    = exp(s) (1+d) exp(log(2^M) + log(2^I2/8) + log(2^I1/128))
//      exp(Z)    = exp(s) f12 (2^M) 2^I2/8 2^I1/128
//
// I1, I2 are table indices. Use a series for exp(s).
// Then get exp(Z)
//
//     exp(yLog(x)) = exp(Z) exp(Z3) f3
//     exp(yLog(x)) = exp(Z)f3 exp(Z3)
//     exp(yLog(x)) = A exp(Z3)
//
// We actually calculate exp(Z3) -1.
// Then,
//     exp(yLog(x)) = A + A( exp(Z3)   -1)
//

// Table Generation
//==============================================================

// The log values
// ==============
// The operation (K*log2_hi) must be exact. K is the true exponent of x.
// If we allow gradual underflow (denormals), K can be represented in 12 bits
// (as a two's complement number). We assume 13 bits as an engineering
// precaution.
//
//           +------------+----------------+-+
//           |  13 bits   | 50 bits        | |
//           +------------+----------------+-+
//           0            1                66
//                        2                34
//
// So we want the lsb(log2_hi) to be 2^-50
// We get log2 as a quad-extended (15-bit exponent, 128-bit significand)
//
//      0 fffe b17217f7d1cf79ab c9e3b39803f2f6af (4...)
//
// Consider numbering the bits left to right, starting at 0 thru 127.
// Bit 0 is the 2^-1 bit; bit 49 is the 2^-50 bit.
//
//  ...79ab
//     0111 1001 1010 1011
//     44
//     89
//
// So if we shift off the rightmost 14 bits, then (shift back only
// the top half) we get
//
//      0 fffe b17217f7d1cf4000 e6af278ece600fcb dabc000000000000
//
// Put the right 64-bit signficand in an FR register, convert to double;
// it is exact. Put the next 128 bits into a quad register and round to double.
// The true exponent of the low part is -51.
//
// hi is 0 fffe b17217f7d1cf4000
// lo is 0 ffcc e6af278ece601000
//
// Convert to double memory format and get
//
// hi is 0x3fe62e42fefa39e8
// lo is 0x3cccd5e4f1d9cc02
//
// log2_hi + log2_lo is an accurate value for log2.
//
//
// The T and t values
// ==================
// A similar method is used to generate the T and t values.
//
// K * log2_hi + T  must be exact.
//
// Smallest T,t
// ----------
// The smallest T,t is
//       T                   t
// 0x3f60040155d58800, 0x3c93bce0ce3ddd81  log(1/frcpa(1+0/256))=  +1.95503e-003
//
// The exponent is 0x3f6 (biased)  or -9 (true).
// For the smallest T value, what we want is to clip the significand such that
// when it is shifted right by 9, its lsb is in the bit for 2^-51. The 9 is the
// specific for the first entry. In general, it is 0xffff - (biased 15-bit
// exponent).

// Independently, what we have calculated is the table value as a quad
// precision number.
// Table entry 1 is
// 0 fff6 80200aaeac44ef38 338f77605fdf8000
//
// We store this quad precision number in a data structure that is
//    sign:           1
//    exponent:      15
//    signficand_hi: 64 (includes explicit bit)
//    signficand_lo: 49
// Because the explicit bit is included, the significand is 113 bits.
//
// Consider significand_hi for table entry 1.
//
//
// +-+--- ... -------+--------------------+
// | |
// +-+--- ... -------+--------------------+
// 0 1               4444444455555555556666
//                   2345678901234567890123
//
// Labeled as above, bit 0 is 2^0, bit 1 is 2^-1, etc.
// Bit 42 is 2^-42. If we shift to the right by 9, the bit in
// bit 42 goes in 51.
//
// So what we want to do is shift bits 43 thru 63 into significand_lo.
// This is shifting bit 42 into bit 63, taking care to retain shifted-off bits.
// Then shifting (just with signficaand_hi) back into bit 42.
//
// The shift_value is 63-42 = 21. In general, this is
//      63 - (51 -(0xffff - 0xfff6))
// For this example, it is
//      63 - (51 - 9) = 63 - 42  = 21
//
// This means we are shifting 21 bits into significand_lo. We must maintain more
// that a 128-bit signficand not to lose bits. So before the shift we put the
// 128-bit significand into a 256-bit signficand and then shift.
// The 256-bit significand has four parts: hh, hl, lh, and ll.
//
// Start off with
//      hh         hl         lh         ll
//      <64>       <49><15_0> <64_0>     <64_0>
//
// After shift by 21 (then return for significand_hi),
//      <43><21_0> <21><43>   <6><58_0>  <64_0>
//
// Take the hh part and convert to a double. There is no rounding here.
// The conversion is exact. The true exponent of the high part is the same as
// the true exponent of the input quad.
//
// We have some 64 plus significand bits for the low part. In this example, we
// have 70 bits. We want to round this to a double. Put them in a quad and then
// do a quad fnorm.
// For this example the true exponent of the low part is
//      true_exponent_of_high - 43 = true_exponent_of_high - (64-21)
// In general, this is
//      true_exponent_of_high - (64 - shift_value)
//
//
// Largest T,t
// ----------
// The largest T,t is
// 0x3fe62643fecf9742, 0x3c9e3147684bd37d  log(1/frcpa(1+255/256))=+6.92171e-001
//
// Table entry 256 is
// 0 fffe b1321ff67cba178c 51da12f4df5a0000
//
// The shift value is
//      63 - (51 -(0xffff - 0xfffe)) = 13
//
// The true exponent of the low part is
//      true_exponent_of_high - (64 - shift_value)
//      -1 - (64-13) = -52
// Biased as a double, this is 0x3cb
//
//
//
// So then lsb(T) must be >= 2^-51
// msb(Klog2_hi) <= 2^12
//
//              +--------+---------+
//              |       51 bits    | <== largest T
//              +--------+---------+
//              | 9 bits | 42 bits | <== smallest T
// +------------+----------------+-+
// |  13 bits   | 50 bits        | |
// +------------+----------------+-+
//
// Note: For powf only the table of T is needed


// Special Cases
//==============================================================

//                                   double     float
// overflow                          error 24   30

// underflow                         error 25   31

// X zero  Y zero
//  +0     +0                 +1     error 26   32
//  -0     +0                 +1     error 26   32
//  +0     -0                 +1     error 26   32
//  -0     -0                 +1     error 26   32

// X zero  Y negative
//  +0     -odd integer       +inf   error 27   33  divide-by-zero
//  -0     -odd integer       -inf   error 27   33  divide-by-zero
//  +0     !-odd integer      +inf   error 27   33  divide-by-zero
//  -0     !-odd integer      +inf   error 27   33  divide-by-zero
//  +0     -inf               +inf   error 27   33  divide-by-zero
//  -0     -inf               +inf   error 27   33  divide-by-zero

// X zero  Y positve
//  +0     +odd integer       +0
//  -0     +odd integer       -0
//  +0     !+odd integer      +0
//  -0     !+odd integer      +0
//  +0     +inf               +0
//  -0     +inf               +0
//  +0     Y NaN              quiet Y               invalid if Y SNaN
//  -0     Y NaN              quiet Y               invalid if Y SNaN

// X one
//  -1     Y inf              +1
//  -1     Y NaN              quiet Y               invalid if Y SNaN
//  +1     Y NaN              +1                    invalid if Y SNaN
//  +1     Y any else         +1

// X -     Y not integer      QNAN   error 28   34  invalid

// X NaN   Y 0                +1     error 29   35
// X NaN   Y NaN              quiet X               invalid if X or Y SNaN
// X NaN   Y any else         quiet X               invalid if X SNaN
// X !+1   Y NaN              quiet Y               invalid if Y SNaN


// X +inf  Y >0               +inf
// X -inf  Y >0, !odd integer +inf
// X -inf  Y >0, odd integer  -inf

// X +inf  Y <0               +0
// X -inf  Y <0, !odd integer +0
// X -inf  Y <0, odd integer  -0

// X +inf  Y =0               +1
// X -inf  Y =0               +1

// |X|<1   Y +inf             +0
// |X|<1   Y -inf             +inf
// |X|>1   Y +inf             +inf
// |X|>1   Y -inf             +0

// X any   Y =0               +1

// Assembly macros
//==============================================================

// integer registers used

pow_GR_exp_half           = r10
pow_GR_signexp_Xm1        = r11
pow_GR_tmp                = r11

pow_GR_signexp_X          = r14
pow_GR_17ones             = r15
pow_GR_Fpsr               = r15
pow_AD_P                  = r16
pow_GR_rcs0_mask          = r16
pow_GR_exp_2tom8          = r17
pow_GR_rcs0               = r17
pow_GR_sig_X              = r18
pow_GR_10033              = r19
pow_GR_16ones             = r20

pow_AD_Tt                 = r21
pow_GR_exp_X              = r22
pow_AD_Q                  = r23
pow_GR_true_exp_X         = r24
pow_GR_y_zero             = r25

pow_GR_exp_Y              = r26
pow_AD_tbl1               = r27
pow_AD_tbl2               = r28
pow_GR_offset             = r29
pow_GR_exp_Xm1            = r30
pow_GR_xneg_yodd          = r31

pow_GR_int_N              = r38
pow_GR_index1             = r39
pow_GR_index2             = r40

pow_AD_T1                 = r41
pow_AD_T2                 = r42
pow_int_GR_M              = r43
pow_GR_sig_int_Y          = r44
pow_GR_sign_Y_Gpr         = r45

pow_GR_17ones_m1          = r46
pow_GR_one                = r47
pow_GR_sign_Y             = r48
pow_GR_signexp_Y_Gpr      = r49
pow_GR_exp_Y_Gpr          = r50

pow_GR_true_exp_Y_Gpr     = r51
pow_GR_signexp_Y          = r52
pow_GR_x_one              = r53
pow_GR_big_pos            = r55

pow_GR_big_neg            = r56

GR_SAVE_B0                = r50
GR_SAVE_GP                = r51
GR_SAVE_PFS               = r52

GR_Parameter_X            = r53
GR_Parameter_Y            = r54
GR_Parameter_RESULT       = r55
pow_GR_tag                = r56


// floating point registers used

POW_B                     = f32
POW_NORM_X                = f33
POW_Xm1                   = f34
POW_r1                    = f34

POW_NORM_Y                = f37
POW_Q2                    = f38
POW_eps                   = f39
POW_P2                    = f40

POW_P0                    = f42
POW_log2_lo               = f43
POW_r                     = f44
POW_Q0_half               = f45

POW_tmp                   = f47
POW_log2_hi               = f48
POW_Q1                    = f49
POW_P1                    = f50

POW_log2_by_128_hi        = f51
POW_inv_log2_by_128       = f52
POW_rsq                   = f53
POW_Yrcub                 = f54
POW_log2_by_128_lo        = f55

POW_xsq                   = f57
POW_v2                    = f59
POW_T                     = f60

POW_RSHF                  = f62
POW_v210                  = f63
POW_twoV                  = f65

POW_U                     = f66
POW_G                     = f67
POW_delta                 = f68
POW_V                     = f70

POW_p                     = f71
POW_Z                     = f72
POW_e3                    = f73
POW_Z2                    = f75

POW_W1                    = f77
POW_Z3                    = f80

POW_Z3sq                  = f85

POW_Nfloat                = f87
POW_f3                    = f89
POW_q                     = f90

POW_T1                    = f96
POW_T2                    = f97
POW_2M                    = f98
POW_s                     = f99
POW_f12                   = f100

POW_ssq                   = f101
POW_T1T2                  = f102
POW_1ps                   = f103
POW_A                     = f104
POW_es                    = f105

POW_Xp1                   = f106
POW_int_K                 = f107
POW_K                     = f108
POW_f123                  = f109
POW_Gpr                   = f110

POW_Y_Gpr                 = f111
POW_int_Y                 = f112
POW_2Mqp1                 = f113

POW_float_int_Y           = f116
POW_ftz_urm_f8            = f117
POW_wre_urm_f8            = f118
POW_big_neg               = f119
POW_big_pos               = f120

// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(pow_table_P)
data8 0x80000000000018E5, 0x0000BFFD  // P_1
data8 0xb8aa3b295c17f0bc, 0x00004006  // inv_ln2_by_128
//
//
data8 0x3FA5555555554A9E // Q_2
data8 0x0000000000000000 // Pad
data8 0x3FC5555555554733 // Q_1
data8 0x43e8000000000000 // Right shift constant for exp
data8 0xc9e3b39803f2f6af, 0x00003fb7  // ln2_by_128_lo
LOCAL_OBJECT_END(pow_table_P)

LOCAL_OBJECT_START(pow_table_Q)
data8 0xCCCCCCCC4ED2BA7F, 0x00003FFC  // P_2
data8 0xAAAAAAAAAAAAB505, 0x00003FFD  // P_0
data8 0x3fe62e42fefa39e8, 0x3cccd5e4f1d9cc02 // log2 hi lo =  +6.93147e-001
data8 0xb17217f7d1cf79ab, 0x00003ff7  // ln2_by_128_hi
LOCAL_OBJECT_END(pow_table_Q)


LOCAL_OBJECT_START(pow_Tt)
data8 0x3f60040155d58800 // log(1/frcpa(1+0/256))=  +1.95503e-003
data8 0x3f78121214586a00 // log(1/frcpa(1+1/256))=  +5.87661e-003
data8 0x3f841929f9683200 // log(1/frcpa(1+2/256))=  +9.81362e-003
data8 0x3f8c317384c75f00 // log(1/frcpa(1+3/256))=  +1.37662e-002
data8 0x3f91a6b91ac73380 // log(1/frcpa(1+4/256))=  +1.72376e-002
data8 0x3f95ba9a5d9ac000 // log(1/frcpa(1+5/256))=  +2.12196e-002
data8 0x3f99d2a807432580 // log(1/frcpa(1+6/256))=  +2.52177e-002
data8 0x3f9d6b2725979800 // log(1/frcpa(1+7/256))=  +2.87291e-002
data8 0x3fa0c58fa19dfa80 // log(1/frcpa(1+8/256))=  +3.27573e-002
data8 0x3fa2954c78cbce00 // log(1/frcpa(1+9/256))=  +3.62953e-002
data8 0x3fa4a94d2da96c40 // log(1/frcpa(1+10/256))=  +4.03542e-002
data8 0x3fa67c94f2d4bb40 // log(1/frcpa(1+11/256))=  +4.39192e-002
data8 0x3fa85188b630f040 // log(1/frcpa(1+12/256))=  +4.74971e-002
data8 0x3faa6b8abe73af40 // log(1/frcpa(1+13/256))=  +5.16017e-002
data8 0x3fac441e06f72a80 // log(1/frcpa(1+14/256))=  +5.52072e-002
data8 0x3fae1e6713606d00 // log(1/frcpa(1+15/256))=  +5.88257e-002
data8 0x3faffa6911ab9300 // log(1/frcpa(1+16/256))=  +6.24574e-002
data8 0x3fb0ec139c5da600 // log(1/frcpa(1+17/256))=  +6.61022e-002
data8 0x3fb1dbd2643d1900 // log(1/frcpa(1+18/256))=  +6.97605e-002
data8 0x3fb2cc7284fe5f00 // log(1/frcpa(1+19/256))=  +7.34321e-002
data8 0x3fb3bdf5a7d1ee60 // log(1/frcpa(1+20/256))=  +7.71173e-002
data8 0x3fb4b05d7aa012e0 // log(1/frcpa(1+21/256))=  +8.08161e-002
data8 0x3fb580db7ceb5700 // log(1/frcpa(1+22/256))=  +8.39975e-002
data8 0x3fb674f089365a60 // log(1/frcpa(1+23/256))=  +8.77219e-002
data8 0x3fb769ef2c6b5680 // log(1/frcpa(1+24/256))=  +9.14602e-002
data8 0x3fb85fd927506a40 // log(1/frcpa(1+25/256))=  +9.52125e-002
data8 0x3fb9335e5d594980 // log(1/frcpa(1+26/256))=  +9.84401e-002
data8 0x3fba2b0220c8e5e0 // log(1/frcpa(1+27/256))=  +1.02219e-001
data8 0x3fbb0004ac1a86a0 // log(1/frcpa(1+28/256))=  +1.05469e-001
data8 0x3fbbf968769fca00 // log(1/frcpa(1+29/256))=  +1.09274e-001
data8 0x3fbccfedbfee13a0 // log(1/frcpa(1+30/256))=  +1.12548e-001
data8 0x3fbda727638446a0 // log(1/frcpa(1+31/256))=  +1.15832e-001
data8 0x3fbea3257fe10f60 // log(1/frcpa(1+32/256))=  +1.19677e-001
data8 0x3fbf7be9fedbfde0 // log(1/frcpa(1+33/256))=  +1.22985e-001
data8 0x3fc02ab352ff25f0 // log(1/frcpa(1+34/256))=  +1.26303e-001
data8 0x3fc097ce579d2040 // log(1/frcpa(1+35/256))=  +1.29633e-001
data8 0x3fc1178e8227e470 // log(1/frcpa(1+36/256))=  +1.33531e-001
data8 0x3fc185747dbecf30 // log(1/frcpa(1+37/256))=  +1.36885e-001
data8 0x3fc1f3b925f25d40 // log(1/frcpa(1+38/256))=  +1.40250e-001
data8 0x3fc2625d1e6ddf50 // log(1/frcpa(1+39/256))=  +1.43627e-001
data8 0x3fc2d1610c868130 // log(1/frcpa(1+40/256))=  +1.47015e-001
data8 0x3fc340c597411420 // log(1/frcpa(1+41/256))=  +1.50414e-001
data8 0x3fc3b08b6757f2a0 // log(1/frcpa(1+42/256))=  +1.53825e-001
data8 0x3fc40dfb08378000 // log(1/frcpa(1+43/256))=  +1.56677e-001
data8 0x3fc47e74e8ca5f70 // log(1/frcpa(1+44/256))=  +1.60109e-001
data8 0x3fc4ef51f6466de0 // log(1/frcpa(1+45/256))=  +1.63553e-001
data8 0x3fc56092e02ba510 // log(1/frcpa(1+46/256))=  +1.67010e-001
data8 0x3fc5d23857cd74d0 // log(1/frcpa(1+47/256))=  +1.70478e-001
data8 0x3fc6313a37335d70 // log(1/frcpa(1+48/256))=  +1.73377e-001
data8 0x3fc6a399dabbd380 // log(1/frcpa(1+49/256))=  +1.76868e-001
data8 0x3fc70337dd3ce410 // log(1/frcpa(1+50/256))=  +1.79786e-001
data8 0x3fc77654128f6120 // log(1/frcpa(1+51/256))=  +1.83299e-001
data8 0x3fc7e9d82a0b0220 // log(1/frcpa(1+52/256))=  +1.86824e-001
data8 0x3fc84a6b759f5120 // log(1/frcpa(1+53/256))=  +1.89771e-001
data8 0x3fc8ab47d5f5a300 // log(1/frcpa(1+54/256))=  +1.92727e-001
data8 0x3fc91fe490965810 // log(1/frcpa(1+55/256))=  +1.96286e-001
data8 0x3fc981634011aa70 // log(1/frcpa(1+56/256))=  +1.99261e-001
data8 0x3fc9f6c407089660 // log(1/frcpa(1+57/256))=  +2.02843e-001
data8 0x3fca58e729348f40 // log(1/frcpa(1+58/256))=  +2.05838e-001
data8 0x3fcabb55c31693a0 // log(1/frcpa(1+59/256))=  +2.08842e-001
data8 0x3fcb1e104919efd0 // log(1/frcpa(1+60/256))=  +2.11855e-001
data8 0x3fcb94ee93e367c0 // log(1/frcpa(1+61/256))=  +2.15483e-001
data8 0x3fcbf851c0675550 // log(1/frcpa(1+62/256))=  +2.18516e-001
data8 0x3fcc5c0254bf23a0 // log(1/frcpa(1+63/256))=  +2.21558e-001
data8 0x3fccc000c9db3c50 // log(1/frcpa(1+64/256))=  +2.24609e-001
data8 0x3fcd244d99c85670 // log(1/frcpa(1+65/256))=  +2.27670e-001
data8 0x3fcd88e93fb2f450 // log(1/frcpa(1+66/256))=  +2.30741e-001
data8 0x3fcdedd437eaef00 // log(1/frcpa(1+67/256))=  +2.33820e-001
data8 0x3fce530effe71010 // log(1/frcpa(1+68/256))=  +2.36910e-001
data8 0x3fceb89a1648b970 // log(1/frcpa(1+69/256))=  +2.40009e-001
data8 0x3fcf1e75fadf9bd0 // log(1/frcpa(1+70/256))=  +2.43117e-001
data8 0x3fcf84a32ead7c30 // log(1/frcpa(1+71/256))=  +2.46235e-001
data8 0x3fcfeb2233ea07c0 // log(1/frcpa(1+72/256))=  +2.49363e-001
data8 0x3fd028f9c7035c18 // log(1/frcpa(1+73/256))=  +2.52501e-001
data8 0x3fd05c8be0d96358 // log(1/frcpa(1+74/256))=  +2.55649e-001
data8 0x3fd085eb8f8ae790 // log(1/frcpa(1+75/256))=  +2.58174e-001
data8 0x3fd0b9c8e32d1910 // log(1/frcpa(1+76/256))=  +2.61339e-001
data8 0x3fd0edd060b78080 // log(1/frcpa(1+77/256))=  +2.64515e-001
data8 0x3fd122024cf00638 // log(1/frcpa(1+78/256))=  +2.67701e-001
data8 0x3fd14be2927aecd0 // log(1/frcpa(1+79/256))=  +2.70257e-001
data8 0x3fd180618ef18ad8 // log(1/frcpa(1+80/256))=  +2.73461e-001
data8 0x3fd1b50bbe2fc638 // log(1/frcpa(1+81/256))=  +2.76675e-001
data8 0x3fd1df4cc7cf2428 // log(1/frcpa(1+82/256))=  +2.79254e-001
data8 0x3fd214456d0eb8d0 // log(1/frcpa(1+83/256))=  +2.82487e-001
data8 0x3fd23ec5991eba48 // log(1/frcpa(1+84/256))=  +2.85081e-001
data8 0x3fd2740d9f870af8 // log(1/frcpa(1+85/256))=  +2.88333e-001
data8 0x3fd29ecdabcdfa00 // log(1/frcpa(1+86/256))=  +2.90943e-001
data8 0x3fd2d46602adcce8 // log(1/frcpa(1+87/256))=  +2.94214e-001
data8 0x3fd2ff66b04ea9d0 // log(1/frcpa(1+88/256))=  +2.96838e-001
data8 0x3fd335504b355a30 // log(1/frcpa(1+89/256))=  +3.00129e-001
data8 0x3fd360925ec44f58 // log(1/frcpa(1+90/256))=  +3.02769e-001
data8 0x3fd38bf1c3337e70 // log(1/frcpa(1+91/256))=  +3.05417e-001
data8 0x3fd3c25277333180 // log(1/frcpa(1+92/256))=  +3.08735e-001
data8 0x3fd3edf463c16838 // log(1/frcpa(1+93/256))=  +3.11399e-001
data8 0x3fd419b423d5e8c0 // log(1/frcpa(1+94/256))=  +3.14069e-001
data8 0x3fd44591e0539f48 // log(1/frcpa(1+95/256))=  +3.16746e-001
data8 0x3fd47c9175b6f0a8 // log(1/frcpa(1+96/256))=  +3.20103e-001
data8 0x3fd4a8b341552b08 // log(1/frcpa(1+97/256))=  +3.22797e-001
data8 0x3fd4d4f390890198 // log(1/frcpa(1+98/256))=  +3.25498e-001
data8 0x3fd501528da1f960 // log(1/frcpa(1+99/256))=  +3.28206e-001
data8 0x3fd52dd06347d4f0 // log(1/frcpa(1+100/256))=  +3.30921e-001
data8 0x3fd55a6d3c7b8a88 // log(1/frcpa(1+101/256))=  +3.33644e-001
data8 0x3fd5925d2b112a58 // log(1/frcpa(1+102/256))=  +3.37058e-001
data8 0x3fd5bf406b543db0 // log(1/frcpa(1+103/256))=  +3.39798e-001
data8 0x3fd5ec433d5c35a8 // log(1/frcpa(1+104/256))=  +3.42545e-001
data8 0x3fd61965cdb02c18 // log(1/frcpa(1+105/256))=  +3.45300e-001
data8 0x3fd646a84935b2a0 // log(1/frcpa(1+106/256))=  +3.48063e-001
data8 0x3fd6740add31de90 // log(1/frcpa(1+107/256))=  +3.50833e-001
data8 0x3fd6a18db74a58c0 // log(1/frcpa(1+108/256))=  +3.53610e-001
data8 0x3fd6cf31058670e8 // log(1/frcpa(1+109/256))=  +3.56396e-001
data8 0x3fd6f180e852f0b8 // log(1/frcpa(1+110/256))=  +3.58490e-001
data8 0x3fd71f5d71b894e8 // log(1/frcpa(1+111/256))=  +3.61289e-001
data8 0x3fd74d5aefd66d58 // log(1/frcpa(1+112/256))=  +3.64096e-001
data8 0x3fd77b79922bd378 // log(1/frcpa(1+113/256))=  +3.66911e-001
data8 0x3fd7a9b9889f19e0 // log(1/frcpa(1+114/256))=  +3.69734e-001
data8 0x3fd7d81b037eb6a0 // log(1/frcpa(1+115/256))=  +3.72565e-001
data8 0x3fd8069e33827230 // log(1/frcpa(1+116/256))=  +3.75404e-001
data8 0x3fd82996d3ef8bc8 // log(1/frcpa(1+117/256))=  +3.77538e-001
data8 0x3fd85855776dcbf8 // log(1/frcpa(1+118/256))=  +3.80391e-001
data8 0x3fd8873658327cc8 // log(1/frcpa(1+119/256))=  +3.83253e-001
data8 0x3fd8aa75973ab8c8 // log(1/frcpa(1+120/256))=  +3.85404e-001
data8 0x3fd8d992dc8824e0 // log(1/frcpa(1+121/256))=  +3.88280e-001
data8 0x3fd908d2ea7d9510 // log(1/frcpa(1+122/256))=  +3.91164e-001
data8 0x3fd92c59e79c0e50 // log(1/frcpa(1+123/256))=  +3.93332e-001
data8 0x3fd95bd750ee3ed0 // log(1/frcpa(1+124/256))=  +3.96231e-001
data8 0x3fd98b7811a3ee58 // log(1/frcpa(1+125/256))=  +3.99138e-001
data8 0x3fd9af47f33d4068 // log(1/frcpa(1+126/256))=  +4.01323e-001
data8 0x3fd9df270c1914a0 // log(1/frcpa(1+127/256))=  +4.04245e-001
data8 0x3fda0325ed14fda0 // log(1/frcpa(1+128/256))=  +4.06442e-001
data8 0x3fda33440224fa78 // log(1/frcpa(1+129/256))=  +4.09379e-001
data8 0x3fda57725e80c380 // log(1/frcpa(1+130/256))=  +4.11587e-001
data8 0x3fda87d0165dd198 // log(1/frcpa(1+131/256))=  +4.14539e-001
data8 0x3fdaac2e6c03f890 // log(1/frcpa(1+132/256))=  +4.16759e-001
data8 0x3fdadccc6fdf6a80 // log(1/frcpa(1+133/256))=  +4.19726e-001
data8 0x3fdb015b3eb1e790 // log(1/frcpa(1+134/256))=  +4.21958e-001
data8 0x3fdb323a3a635948 // log(1/frcpa(1+135/256))=  +4.24941e-001
data8 0x3fdb56fa04462908 // log(1/frcpa(1+136/256))=  +4.27184e-001
data8 0x3fdb881aa659bc90 // log(1/frcpa(1+137/256))=  +4.30182e-001
data8 0x3fdbad0bef3db160 // log(1/frcpa(1+138/256))=  +4.32437e-001
data8 0x3fdbd21297781c28 // log(1/frcpa(1+139/256))=  +4.34697e-001
data8 0x3fdc039236f08818 // log(1/frcpa(1+140/256))=  +4.37718e-001
data8 0x3fdc28cb1e4d32f8 // log(1/frcpa(1+141/256))=  +4.39990e-001
data8 0x3fdc4e19b84723c0 // log(1/frcpa(1+142/256))=  +4.42267e-001
data8 0x3fdc7ff9c74554c8 // log(1/frcpa(1+143/256))=  +4.45311e-001
data8 0x3fdca57b64e9db00 // log(1/frcpa(1+144/256))=  +4.47600e-001
data8 0x3fdccb130a5ceba8 // log(1/frcpa(1+145/256))=  +4.49895e-001
data8 0x3fdcf0c0d18f3268 // log(1/frcpa(1+146/256))=  +4.52194e-001
data8 0x3fdd232075b5a200 // log(1/frcpa(1+147/256))=  +4.55269e-001
data8 0x3fdd490246defa68 // log(1/frcpa(1+148/256))=  +4.57581e-001
data8 0x3fdd6efa918d25c8 // log(1/frcpa(1+149/256))=  +4.59899e-001
data8 0x3fdd9509707ae528 // log(1/frcpa(1+150/256))=  +4.62221e-001
data8 0x3fddbb2efe92c550 // log(1/frcpa(1+151/256))=  +4.64550e-001
data8 0x3fddee2f3445e4a8 // log(1/frcpa(1+152/256))=  +4.67663e-001
data8 0x3fde148a1a2726c8 // log(1/frcpa(1+153/256))=  +4.70004e-001
data8 0x3fde3afc0a49ff38 // log(1/frcpa(1+154/256))=  +4.72350e-001
data8 0x3fde6185206d5168 // log(1/frcpa(1+155/256))=  +4.74702e-001
data8 0x3fde882578823d50 // log(1/frcpa(1+156/256))=  +4.77060e-001
data8 0x3fdeaedd2eac9908 // log(1/frcpa(1+157/256))=  +4.79423e-001
data8 0x3fded5ac5f436be0 // log(1/frcpa(1+158/256))=  +4.81792e-001
data8 0x3fdefc9326d16ab8 // log(1/frcpa(1+159/256))=  +4.84166e-001
data8 0x3fdf2391a21575f8 // log(1/frcpa(1+160/256))=  +4.86546e-001
data8 0x3fdf4aa7ee031928 // log(1/frcpa(1+161/256))=  +4.88932e-001
data8 0x3fdf71d627c30bb0 // log(1/frcpa(1+162/256))=  +4.91323e-001
data8 0x3fdf991c6cb3b378 // log(1/frcpa(1+163/256))=  +4.93720e-001
data8 0x3fdfc07ada69a908 // log(1/frcpa(1+164/256))=  +4.96123e-001
data8 0x3fdfe7f18eb03d38 // log(1/frcpa(1+165/256))=  +4.98532e-001
data8 0x3fe007c053c5002c // log(1/frcpa(1+166/256))=  +5.00946e-001
data8 0x3fe01b942198a5a0 // log(1/frcpa(1+167/256))=  +5.03367e-001
data8 0x3fe02f74400c64e8 // log(1/frcpa(1+168/256))=  +5.05793e-001
data8 0x3fe04360be7603ac // log(1/frcpa(1+169/256))=  +5.08225e-001
data8 0x3fe05759ac47fe30 // log(1/frcpa(1+170/256))=  +5.10663e-001
data8 0x3fe06b5f1911cf50 // log(1/frcpa(1+171/256))=  +5.13107e-001
data8 0x3fe078bf0533c568 // log(1/frcpa(1+172/256))=  +5.14740e-001
data8 0x3fe08cd9687e7b0c // log(1/frcpa(1+173/256))=  +5.17194e-001
data8 0x3fe0a10074cf9018 // log(1/frcpa(1+174/256))=  +5.19654e-001
data8 0x3fe0b5343a234474 // log(1/frcpa(1+175/256))=  +5.22120e-001
data8 0x3fe0c974c89431cc // log(1/frcpa(1+176/256))=  +5.24592e-001
data8 0x3fe0ddc2305b9884 // log(1/frcpa(1+177/256))=  +5.27070e-001
data8 0x3fe0eb524bafc918 // log(1/frcpa(1+178/256))=  +5.28726e-001
data8 0x3fe0ffb54213a474 // log(1/frcpa(1+179/256))=  +5.31214e-001
data8 0x3fe114253da97d9c // log(1/frcpa(1+180/256))=  +5.33709e-001
data8 0x3fe128a24f1d9afc // log(1/frcpa(1+181/256))=  +5.36210e-001
data8 0x3fe1365252bf0864 // log(1/frcpa(1+182/256))=  +5.37881e-001
data8 0x3fe14ae558b4a92c // log(1/frcpa(1+183/256))=  +5.40393e-001
data8 0x3fe15f85a19c7658 // log(1/frcpa(1+184/256))=  +5.42910e-001
data8 0x3fe16d4d38c119f8 // log(1/frcpa(1+185/256))=  +5.44592e-001
data8 0x3fe18203c20dd130 // log(1/frcpa(1+186/256))=  +5.47121e-001
data8 0x3fe196c7bc4b1f38 // log(1/frcpa(1+187/256))=  +5.49656e-001
data8 0x3fe1a4a738b7a33c // log(1/frcpa(1+188/256))=  +5.51349e-001
data8 0x3fe1b981c0c9653c // log(1/frcpa(1+189/256))=  +5.53895e-001
data8 0x3fe1ce69e8bb1068 // log(1/frcpa(1+190/256))=  +5.56447e-001
data8 0x3fe1dc619de06944 // log(1/frcpa(1+191/256))=  +5.58152e-001
data8 0x3fe1f160a2ad0da0 // log(1/frcpa(1+192/256))=  +5.60715e-001
data8 0x3fe2066d7740737c // log(1/frcpa(1+193/256))=  +5.63285e-001
data8 0x3fe2147dba47a390 // log(1/frcpa(1+194/256))=  +5.65001e-001
data8 0x3fe229a1bc5ebac0 // log(1/frcpa(1+195/256))=  +5.67582e-001
data8 0x3fe237c1841a502c // log(1/frcpa(1+196/256))=  +5.69306e-001
data8 0x3fe24cfce6f80d98 // log(1/frcpa(1+197/256))=  +5.71898e-001
data8 0x3fe25b2c55cd5760 // log(1/frcpa(1+198/256))=  +5.73630e-001
data8 0x3fe2707f4d5f7c40 // log(1/frcpa(1+199/256))=  +5.76233e-001
data8 0x3fe285e0842ca380 // log(1/frcpa(1+200/256))=  +5.78842e-001
data8 0x3fe294294708b770 // log(1/frcpa(1+201/256))=  +5.80586e-001
data8 0x3fe2a9a2670aff0c // log(1/frcpa(1+202/256))=  +5.83207e-001
data8 0x3fe2b7fb2c8d1cc0 // log(1/frcpa(1+203/256))=  +5.84959e-001
data8 0x3fe2c65a6395f5f4 // log(1/frcpa(1+204/256))=  +5.86713e-001
data8 0x3fe2dbf557b0df40 // log(1/frcpa(1+205/256))=  +5.89350e-001
data8 0x3fe2ea64c3f97654 // log(1/frcpa(1+206/256))=  +5.91113e-001
data8 0x3fe3001823684d70 // log(1/frcpa(1+207/256))=  +5.93762e-001
data8 0x3fe30e97e9a8b5cc // log(1/frcpa(1+208/256))=  +5.95531e-001
data8 0x3fe32463ebdd34e8 // log(1/frcpa(1+209/256))=  +5.98192e-001
data8 0x3fe332f4314ad794 // log(1/frcpa(1+210/256))=  +5.99970e-001
data8 0x3fe348d90e7464cc // log(1/frcpa(1+211/256))=  +6.02643e-001
data8 0x3fe35779f8c43d6c // log(1/frcpa(1+212/256))=  +6.04428e-001
data8 0x3fe36621961a6a98 // log(1/frcpa(1+213/256))=  +6.06217e-001
data8 0x3fe37c299f3c3668 // log(1/frcpa(1+214/256))=  +6.08907e-001
data8 0x3fe38ae2171976e4 // log(1/frcpa(1+215/256))=  +6.10704e-001
data8 0x3fe399a157a603e4 // log(1/frcpa(1+216/256))=  +6.12504e-001
data8 0x3fe3afccfe77b9d0 // log(1/frcpa(1+217/256))=  +6.15210e-001
data8 0x3fe3be9d503533b4 // log(1/frcpa(1+218/256))=  +6.17018e-001
data8 0x3fe3cd7480b4a8a0 // log(1/frcpa(1+219/256))=  +6.18830e-001
data8 0x3fe3e3c43918f76c // log(1/frcpa(1+220/256))=  +6.21554e-001
data8 0x3fe3f2acb27ed6c4 // log(1/frcpa(1+221/256))=  +6.23373e-001
data8 0x3fe4019c2125ca90 // log(1/frcpa(1+222/256))=  +6.25197e-001
data8 0x3fe4181061389720 // log(1/frcpa(1+223/256))=  +6.27937e-001
data8 0x3fe42711518df544 // log(1/frcpa(1+224/256))=  +6.29769e-001
data8 0x3fe436194e12b6bc // log(1/frcpa(1+225/256))=  +6.31604e-001
data8 0x3fe445285d68ea68 // log(1/frcpa(1+226/256))=  +6.33442e-001
data8 0x3fe45bcc464c8938 // log(1/frcpa(1+227/256))=  +6.36206e-001
data8 0x3fe46aed21f117fc // log(1/frcpa(1+228/256))=  +6.38053e-001
data8 0x3fe47a1527e8a2d0 // log(1/frcpa(1+229/256))=  +6.39903e-001
data8 0x3fe489445efffcc8 // log(1/frcpa(1+230/256))=  +6.41756e-001
data8 0x3fe4a018bcb69834 // log(1/frcpa(1+231/256))=  +6.44543e-001
data8 0x3fe4af5a0c9d65d4 // log(1/frcpa(1+232/256))=  +6.46405e-001
data8 0x3fe4bea2a5bdbe84 // log(1/frcpa(1+233/256))=  +6.48271e-001
data8 0x3fe4cdf28f10ac44 // log(1/frcpa(1+234/256))=  +6.50140e-001
data8 0x3fe4dd49cf994058 // log(1/frcpa(1+235/256))=  +6.52013e-001
data8 0x3fe4eca86e64a680 // log(1/frcpa(1+236/256))=  +6.53889e-001
data8 0x3fe503c43cd8eb68 // log(1/frcpa(1+237/256))=  +6.56710e-001
data8 0x3fe513356667fc54 // log(1/frcpa(1+238/256))=  +6.58595e-001
data8 0x3fe522ae0738a3d4 // log(1/frcpa(1+239/256))=  +6.60483e-001
data8 0x3fe5322e26867854 // log(1/frcpa(1+240/256))=  +6.62376e-001
data8 0x3fe541b5cb979808 // log(1/frcpa(1+241/256))=  +6.64271e-001
data8 0x3fe55144fdbcbd60 // log(1/frcpa(1+242/256))=  +6.66171e-001
data8 0x3fe560dbc45153c4 // log(1/frcpa(1+243/256))=  +6.68074e-001
data8 0x3fe5707a26bb8c64 // log(1/frcpa(1+244/256))=  +6.69980e-001
data8 0x3fe587f60ed5b8fc // log(1/frcpa(1+245/256))=  +6.72847e-001
data8 0x3fe597a7977c8f30 // log(1/frcpa(1+246/256))=  +6.74763e-001
data8 0x3fe5a760d634bb88 // log(1/frcpa(1+247/256))=  +6.76682e-001
data8 0x3fe5b721d295f10c // log(1/frcpa(1+248/256))=  +6.78605e-001
data8 0x3fe5c6ea94431ef8 // log(1/frcpa(1+249/256))=  +6.80532e-001
data8 0x3fe5d6bb22ea86f4 // log(1/frcpa(1+250/256))=  +6.82462e-001
data8 0x3fe5e6938645d38c // log(1/frcpa(1+251/256))=  +6.84397e-001
data8 0x3fe5f673c61a2ed0 // log(1/frcpa(1+252/256))=  +6.86335e-001
data8 0x3fe6065bea385924 // log(1/frcpa(1+253/256))=  +6.88276e-001
data8 0x3fe6164bfa7cc068 // log(1/frcpa(1+254/256))=  +6.90222e-001
data8 0x3fe62643fecf9740 // log(1/frcpa(1+255/256))=  +6.92171e-001
LOCAL_OBJECT_END(pow_Tt)


// Table 1 is 2^(index_1/128) where
// index_1 goes from 0 to 15
LOCAL_OBJECT_START(pow_tbl1)
data8 0x8000000000000000 , 0x00003FFF
data8 0x80B1ED4FD999AB6C , 0x00003FFF
data8 0x8164D1F3BC030773 , 0x00003FFF
data8 0x8218AF4373FC25EC , 0x00003FFF
data8 0x82CD8698AC2BA1D7 , 0x00003FFF
data8 0x8383594EEFB6EE37 , 0x00003FFF
data8 0x843A28C3ACDE4046 , 0x00003FFF
data8 0x84F1F656379C1A29 , 0x00003FFF
data8 0x85AAC367CC487B15 , 0x00003FFF
data8 0x8664915B923FBA04 , 0x00003FFF
data8 0x871F61969E8D1010 , 0x00003FFF
data8 0x87DB357FF698D792 , 0x00003FFF
data8 0x88980E8092DA8527 , 0x00003FFF
data8 0x8955EE03618E5FDD , 0x00003FFF
data8 0x8A14D575496EFD9A , 0x00003FFF
data8 0x8AD4C6452C728924 , 0x00003FFF
LOCAL_OBJECT_END(pow_tbl1)


// Table 2 is 2^(index_1/8) where
// index_2 goes from 0 to 7
LOCAL_OBJECT_START(pow_tbl2)
data8 0x8000000000000000 , 0x00003FFF
data8 0x8B95C1E3EA8BD6E7 , 0x00003FFF
data8 0x9837F0518DB8A96F , 0x00003FFF
data8 0xA5FED6A9B15138EA , 0x00003FFF
data8 0xB504F333F9DE6484 , 0x00003FFF
data8 0xC5672A115506DADD , 0x00003FFF
data8 0xD744FCCAD69D6AF4 , 0x00003FFF
data8 0xEAC0C6E7DD24392F , 0x00003FFF
LOCAL_OBJECT_END(pow_tbl2)

.section .text
WEAK_LIBM_ENTRY(powf)

// Get exponent of x.  Will be used to calculate K.
{ .mfi
          getf.exp     pow_GR_signexp_X = f8
          fms.s1 POW_Xm1 = f8,f1,f1     // Will be used for r1 if x>0
          mov           pow_GR_17ones   = 0x1FFFF
}
{ .mfi
          addl          pow_AD_P        = @ltoff(pow_table_P), gp
          fma.s1 POW_Xp1 = f8,f1,f1     // Will be used for r1 if x<0
          nop.i 999
}
;;

// Get significand of x.  Will be used to get index to fetch T, Tt.
{ .mfi
          getf.sig      pow_GR_sig_X    = f8
          frcpa.s1      POW_B, p6       = f1,f8
          mov           pow_GR_exp_half = 0xFFFE   // Exponent for 0.5
}
{ .mfi
          ld8 pow_AD_P = [pow_AD_P]
          fma.s1        POW_NORM_X      = f8,f1,f0
          mov          pow_GR_exp_2tom8 = 0xFFF7
}
;;

// DOUBLE 0x10033  exponent limit at which y is an integer
{ .mfi
          nop.m 999
          fcmp.lt.s1 p8,p9 = f8, f0     // Test for x<0
          addl pow_GR_10033             = 0x10033, r0
}
{ .mfi
          mov           pow_GR_16ones   = 0xFFFF
          fma.s1        POW_NORM_Y      = f9,f1,f0
          nop.i 999
}
;;

// p13 = TRUE ==> X is unorm
{ .mfi
          setf.exp      POW_Q0_half     = pow_GR_exp_half  // Form 0.5
          fclass.m  p13,p0              = f8, 0x0b  // Test for x unorm
          adds          pow_AD_Tt       = pow_Tt - pow_table_P,  pow_AD_P
}
{ .mfi
          adds          pow_AD_Q        = pow_table_Q - pow_table_P,  pow_AD_P
          nop.f 999
          nop.i 999
}
;;

// p14 = TRUE ==> X is ZERO
{ .mfi
          ldfe          POW_P2          = [pow_AD_Q], 16
          fclass.m  p14,p0              = f8, 0x07
          nop.i 999
}
// Note POW_Xm1 and POW_r1 are used interchangably
{ .mfb
          nop.m 999
(p8)      fnma.s1        POW_Xm1        = POW_Xp1,f1,f0
(p13)     br.cond.spnt POW_X_DENORM
}
;;

// Continue normal and denormal paths here
POW_COMMON:
// p11 = TRUE ==> Y is a NAN
{ .mfi
          and           pow_GR_exp_X    = pow_GR_signexp_X, pow_GR_17ones
          fclass.m  p11,p0              = f9, 0xc3
          nop.i 999
}
{ .mfi
          nop.m 999
          fms.s1        POW_r           = POW_B, POW_NORM_X,f1
          mov pow_GR_y_zero = 0
}
;;

// Get exponent of |x|-1 to use in comparison to 2^-8
{ .mmi
          getf.exp  pow_GR_signexp_Xm1  = POW_Xm1
          sub       pow_GR_true_exp_X   = pow_GR_exp_X, pow_GR_16ones
          extr.u        pow_GR_offset   = pow_GR_sig_X, 55, 8
}
;;

{ .mfi
          alloc         r32=ar.pfs,2,19,4,0
          fcvt.fx.s1   POW_int_Y        = POW_NORM_Y
          shladd pow_AD_Tt = pow_GR_offset, 3, pow_AD_Tt
}
{ .mfi
          setf.sig POW_int_K            = pow_GR_true_exp_X
          nop.f 999
          nop.i 999
}
;;

// p12 = TRUE if Y is ZERO
// Compute xsq to decide later if |x|=1
{ .mfi
          ldfe          POW_P1          = [pow_AD_P], 16
          fclass.m      p12,p0          = f9, 0x07
          nop.i 999
}
{ .mfb
          ldfe          POW_P0          = [pow_AD_Q], 16
          fma.s1        POW_xsq = POW_NORM_X, POW_NORM_X, f0
(p11)     br.cond.spnt  POW_Y_NAN       // Branch if y=nan
}
;;

{ .mmf
          getf.exp  pow_GR_signexp_Y    = POW_NORM_Y
          ldfd  POW_T                   = [pow_AD_Tt]
          fma.s1        POW_rsq         = POW_r, POW_r,f0
}
;;

// p11 = TRUE ==> X is a NAN
{ .mfi
          ldfpd         POW_log2_hi, POW_log2_lo  = [pow_AD_Q], 16
          fclass.m      p11,p0          = POW_NORM_X, 0xc3
          nop.i 999
}
{ .mfi
          ldfe          POW_inv_log2_by_128 = [pow_AD_P], 16
          fma.s1 POW_delta              = f0,f0,f0 // delta=0 in case |x| near 1
(p12)     mov pow_GR_y_zero = 1
}
;;

{ .mfi
          ldfd   POW_Q2                 = [pow_AD_P], 16
          fnma.s1 POW_twoV              = POW_r, POW_Q0_half,f1
          and       pow_GR_exp_Xm1      = pow_GR_signexp_Xm1, pow_GR_17ones
}
{ .mfi
          nop.m 999
          fma.s1 POW_U                  = POW_NORM_Y,POW_r,f0
          nop.i 999
}
;;

// Determine if we will use the |x| near 1 path (p6) or normal path (p7)
{ .mfi
          nop.m 999
          fcvt.xf POW_K                 = POW_int_K
          cmp.lt p6,p7                  = pow_GR_exp_Xm1, pow_GR_exp_2tom8
}
{ .mfb
          nop.m 999
          fma.s1 POW_G                  = f0,f0,f0  // G=0 in case |x| near 1
(p11)     br.cond.spnt  POW_X_NAN       // Branch if x=nan and y not nan
}
;;

// If on the x near 1 path, assign r1 to r
{ .mfi
          ldfpd  POW_Q1, POW_RSHF       = [pow_AD_P], 16
(p6)      fma.s1    POW_r               = POW_r1, f1, f0
          nop.i 999
}
{ .mfb
          nop.m 999
(p6)      fma.s1    POW_rsq             = POW_r1, POW_r1, f0
(p14)     br.cond.spnt POW_X_0          // Branch if x zero and y not nan
}
;;

{ .mfi
          getf.sig pow_GR_sig_int_Y     = POW_int_Y
(p6)      fnma.s1 POW_twoV              = POW_r1, POW_Q0_half,f1
          and pow_GR_exp_Y              = pow_GR_signexp_Y, pow_GR_17ones
}
{ .mfb
          andcm pow_GR_sign_Y           = pow_GR_signexp_Y, pow_GR_17ones
(p6)      fma.s1 POW_U                  = POW_NORM_Y,POW_r1,f0
(p12)     br.cond.spnt POW_Y_0   // Branch if y=zero, x not zero or nan
}
;;

{ .mfi
          ldfe      POW_log2_by_128_lo  = [pow_AD_P], 16
(p7)      fma.s1 POW_Z2                 = POW_twoV, POW_U, f0
          nop.i 999
}
{ .mfi
          ldfe      POW_log2_by_128_hi  = [pow_AD_Q], 16
          nop.f 999
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          fcvt.xf   POW_float_int_Y     = POW_int_Y
          nop.i 999
}
{ .mfi
          nop.m 999
(p7)      fma.s1 POW_G                  = POW_K, POW_log2_hi, POW_T
          adds          pow_AD_tbl1     = pow_tbl1 - pow_Tt,  pow_AD_Q
}
;;

// p11 = TRUE ==> X is NEGATIVE but not inf
{ .mfi
          nop.m 999
          fclass.m  p11,p0              = POW_NORM_X, 0x1a
          nop.i 999
}
{ .mfi
          nop.m 999
(p7)      fma.s1 POW_delta              = POW_K, POW_log2_lo, f0
          adds pow_AD_tbl2              = pow_tbl2 - pow_tbl1,  pow_AD_tbl1
}
;;

{ .mfi
          nop.m 999
(p6)      fma.s1 POW_Z                  = POW_twoV, POW_U, f0
          nop.i 999
}
{ .mfi
          nop.m 999
          fma.s1 POW_v2                 = POW_P1, POW_r,  POW_P0
          nop.i 999
}
;;

// p11 = TRUE ==> X is NEGATIVE but not inf
//    p12 = TRUE ==> X is NEGATIVE  AND  Y  already even int
//    p13 = TRUE ==> X is NEGATIVE  AND  Y possible int
{ .mfi
          nop.m 999
(p7)      fma.s1 POW_Z                  = POW_NORM_Y, POW_G, POW_Z2
(p11)     cmp.gt.unc  p12,p13           = pow_GR_exp_Y, pow_GR_10033
}
{ .mfi
          nop.m 999
          fma.s1 POW_Gpr                = POW_G, f1, POW_r
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          fma.s1 POW_Yrcub              = POW_rsq, POW_U, f0
          nop.i 999
}
{ .mfi
          nop.m 999
          fma.s1 POW_p                  = POW_rsq, POW_P2, POW_v2
          nop.i 999
}
;;

// Test if x inf
{ .mfi
          nop.m 999
          fclass.m p15,p0 = POW_NORM_X,  0x23
          nop.i 999
}
// By adding RSHF (1.1000...*2^63) we put integer part in rightmost significand
{ .mfi
          nop.m 999
          fma.s1 POW_W1  = POW_Z, POW_inv_log2_by_128, POW_RSHF
          nop.i 999
}
;;

// p13 = TRUE ==> X is NEGATIVE  AND  Y possible int
//     p10 = TRUE ==> X is NEG and Y is an int
//     p12 = TRUE ==> X is NEG and Y is not an int
{ .mfi
          nop.m 999
(p13)     fcmp.eq.unc.s1 p10,p12        = POW_float_int_Y,  POW_NORM_Y
          mov pow_GR_xneg_yodd = 0
}
{ .mfi
          nop.m 999
          fma.s1 POW_Y_Gpr              = POW_NORM_Y, POW_Gpr, f0
          nop.i 999
}
;;

// p11 = TRUE ==> X is +1.0
{ .mfi
          nop.m 999
          fcmp.eq.s1 p11,p0 = POW_NORM_X, f1
          nop.i 999
}
;;

// Extract rounded integer from rightmost significand of POW_W1
// By subtracting RSHF we get rounded integer POW_Nfloat
{ .mfi
          getf.sig pow_GR_int_N        = POW_W1
          fms.s1 POW_Nfloat  = POW_W1, f1, POW_RSHF
          nop.i 999
}
{ .mfb
          nop.m 999
          fma.s1 POW_Z3                 = POW_p, POW_Yrcub, f0
(p12)     br.cond.spnt POW_X_NEG_Y_NONINT  // Branch if x neg, y not integer
}
;;

// p7  = TRUE ==> Y is +1.0
// p12 = TRUE ==> X is NEGATIVE  AND Y is an odd integer
{ .mfi
          getf.exp pow_GR_signexp_Y_Gpr = POW_Y_Gpr
          fcmp.eq.s1 p7,p0 = POW_NORM_Y, f1  // Test for y=1.0
(p10)     tbit.nz.unc  p12,p0           = pow_GR_sig_int_Y,0
}
{ .mfb
          nop.m 999
(p11)     fma.s.s0 f8 = f1,f1,f0    // If x=1, result is +1
(p15)     br.cond.spnt POW_X_INF
}
;;

// Test x and y and flag denormal
{ .mfi
          nop.m 999
          fcmp.eq.s0 p15,p0 = f8,f9
          nop.i 999
}
{ .mfb
          nop.m 999
          fma.s1 POW_e3                 = POW_NORM_Y, POW_delta, f0
(p11)     br.ret.spnt b0            // Early exit if x=1.0, result is +1
}
;;

{ .mfi
(p12)     mov pow_GR_xneg_yodd = 1
          fnma.s1 POW_f12  = POW_Nfloat, POW_log2_by_128_lo, f1
          nop.i 999
}
{ .mfb
          nop.m 999
          fnma.s1 POW_s  = POW_Nfloat, POW_log2_by_128_hi, POW_Z
(p7)      br.ret.spnt b0        // Early exit if y=1.0, result is x
}
;;

{ .mmi
          and pow_GR_index1             = 0x0f, pow_GR_int_N
          and pow_GR_index2             = 0x70, pow_GR_int_N
          shr pow_int_GR_M              = pow_GR_int_N, 7    // M = N/128
}
;;

{ .mfi
          shladd pow_AD_T1              = pow_GR_index1, 4, pow_AD_tbl1
          fma.s1 POW_q                  = POW_Z3, POW_Q1, POW_Q0_half
          add pow_int_GR_M              = pow_GR_16ones, pow_int_GR_M
}
{ .mfi
          add pow_AD_T2                 = pow_AD_tbl2, pow_GR_index2
          fma.s1 POW_Z3sq               = POW_Z3, POW_Z3, f0
          nop.i 999
}
;;

{ .mmi
          ldfe POW_T1                   = [pow_AD_T1]
          ldfe POW_T2                   = [pow_AD_T2]
          nop.i 999
}
;;

// f123 = f12*(e3+1) = f12*e3+f12
{ .mfi
          setf.exp POW_2M               = pow_int_GR_M
          fma.s1 POW_f123               = POW_e3,POW_f12,POW_f12
          nop.i 999
}
{ .mfi
          nop.m 999
          fma.s1 POW_ssq                = POW_s, POW_s, f0
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          fma.s1 POW_v2                 = POW_s, POW_Q2, POW_Q1
          and pow_GR_exp_Y_Gpr          = pow_GR_signexp_Y_Gpr, pow_GR_17ones
}
;;

{ .mfi
          cmp.ne p12,p13 = pow_GR_xneg_yodd, r0
          fma.s1 POW_q                  = POW_Z3sq, POW_q, POW_Z3
          sub pow_GR_true_exp_Y_Gpr     = pow_GR_exp_Y_Gpr, pow_GR_16ones
}
;;

// p8 TRUE ==> |Y(G + r)| >= 7

// single
//     -2^7   -2^6             2^6   2^7
// -----+-----+----+ ... +-----+-----+-----
//  p8  |             p9             |  p8
//      |     |       p10      |     |

// Form signexp of constants to indicate overflow
{ .mfi
          mov         pow_GR_big_pos    = 0x1007f
          nop.f 999
          cmp.le p8,p9                  = 7, pow_GR_true_exp_Y_Gpr
}
{ .mfi
          mov         pow_GR_big_neg    = 0x3007f
          nop.f 999
          andcm pow_GR_sign_Y_Gpr       = pow_GR_signexp_Y_Gpr, pow_GR_17ones
}
;;

// Form big positive and negative constants to test for possible overflow
// Scale both terms of the polynomial by POW_f123
{ .mfi
          setf.exp POW_big_pos          = pow_GR_big_pos
          fma.s1 POW_ssq                = POW_ssq, POW_f123, f0
(p9)      cmp.le.unc p0,p10             = 6, pow_GR_true_exp_Y_Gpr
}
{ .mfb
          setf.exp POW_big_neg          = pow_GR_big_neg
          fma.s1 POW_1ps                = POW_s, POW_f123, POW_f123
(p8)      br.cond.spnt POW_OVER_UNDER_X_NOT_INF
}
;;

{ .mfi
          nop.m 999
(p12)     fnma.s1 POW_T1T2              = POW_T1, POW_T2, f0
          nop.i 999
}
{ .mfi
          nop.m 999
(p13)     fma.s1 POW_T1T2               = POW_T1, POW_T2, f0
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          fma.s1 POW_v210               = POW_s, POW_v2, POW_Q0_half
          nop.i 999
}
{ .mfi
          nop.m 999
          fma.s1 POW_2Mqp1              = POW_2M, POW_q, POW_2M
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          fma.s1 POW_es                 = POW_ssq, POW_v210, POW_1ps
          nop.i 999
}
{ .mfi
          nop.m 999
          fma.s1 POW_A                  = POW_T1T2, POW_2Mqp1, f0
          nop.i 999
}
;;

// Dummy op to set inexact
{ .mfi
          nop.m 999
          fma.s0 POW_tmp                = POW_2M, POW_q, POW_2M
          nop.i 999
}
;;

{ .mfb
          nop.m 999
          fma.s.s0 f8                   = POW_A, POW_es, f0
(p10)     br.ret.sptk     b0            // Exit main branch if no over/underflow
}
;;

// POSSIBLE_OVER_UNDER
// p6 = TRUE ==> Y_Gpr negative
// Result is already computed.  We just need to know if over/underflow occurred.

{ .mfb
        cmp.eq p0,p6                    = pow_GR_sign_Y_Gpr, r0
        nop.f 999
(p6)    br.cond.spnt POW_POSSIBLE_UNDER
}
;;

// POSSIBLE_OVER
// We got an answer.
// overflow is a possibility, not a certainty


// We define an overflow when the answer with
//    WRE set
//    user-defined rounding mode

// double
// Largest double is 7FE (biased double)
//                   7FE - 3FF + FFFF = 103FE
// Create + largest_double_plus_ulp
// Create - largest_double_plus_ulp
// Calculate answer with WRE set.

// single
// Largest single is FE (biased double)
//                   FE - 7F + FFFF = 1007E
// Create + largest_single_plus_ulp
// Create - largest_single_plus_ulp
// Calculate answer with WRE set.

// Cases when answer is ldn+1  are as follows:
//  ldn                   ldn+1
// --+----------|----------+------------
//              |
//    +inf          +inf      -inf
//                  RN         RN
//                             RZ

// Put in s2 (td set, wre set)
{ .mfi
        nop.m 999
        fsetc.s2 0x7F,0x42
        nop.i 999
}
;;

{ .mfi
        nop.m 999
        fma.s.s2 POW_wre_urm_f8         = POW_A, POW_es, f0
        nop.i 999
}
;;

// Return s2 to default
{ .mfi
        nop.m 999
        fsetc.s2 0x7F,0x40
        nop.i 999
}
;;

// p7 = TRUE ==> yes, we have an overflow
{ .mfi
        nop.m 999
        fcmp.ge.s1 p7, p8               =  POW_wre_urm_f8, POW_big_pos
        nop.i 999
}
;;

{ .mfi
        nop.m 999
(p8)    fcmp.le.s1 p7, p0               =  POW_wre_urm_f8, POW_big_neg
        nop.i 999
}
;;

{ .mbb
(p7)   mov pow_GR_tag                   = 30
(p7)   br.cond.spnt __libm_error_region // Branch if overflow
       br.ret.sptk     b0               // Exit if did not overflow
}
;;


POW_POSSIBLE_UNDER:
// We got an answer. input was < -2^9 but > -2^10 (double)
// We got an answer. input was < -2^6 but > -2^7  (float)
// underflow is a possibility, not a certainty

// We define an underflow when the answer with
//    ftz set
// is zero (tiny numbers become zero)
// Notice (from below) that if we have an unlimited exponent range,
// then there is an extra machine number E between the largest denormal and
// the smallest normal.
// So if with unbounded exponent we round to E or below, then we are
// tiny and underflow has occurred.
// But notice that you can be in a situation where we are tiny, namely
// rounded to E, but when the exponent is bounded we round to smallest
// normal. So the answer can be the smallest normal with underflow.
//                           E
// -----+--------------------+--------------------+-----
//      |                    |                    |
//   1.1...10 2^-3fff    1.1...11 2^-3fff    1.0...00 2^-3ffe
//   0.1...11 2^-3ffe                                   (biased, 1)
//    largest dn                               smallest normal

// Form small constant (2^-170) to correct underflow result near region of
// smallest denormal in round-nearest.

// Put in s2 (td set, ftz set)
.pred.rel "mutex",p12,p13
{ .mfi
        mov pow_GR_Fpsr = ar40          // Read the fpsr--need to check rc.s0
        fsetc.s2 0x7F,0x41
        mov pow_GR_rcs0_mask            = 0x0c00 // Set mask for rc.s0
}
{ .mfi
(p12)   mov pow_GR_tmp                  = 0x2ffff - 170
        nop.f 999
(p13)   mov pow_GR_tmp                  = 0x0ffff - 170
}
;;

{ .mfi
        setf.exp POW_eps                = pow_GR_tmp        // Form 2^-170
        fma.s.s2 POW_ftz_urm_f8         = POW_A, POW_es, f0
        nop.i 999
}
;;

// Return s2 to default
{ .mfi
        nop.m 999
        fsetc.s2 0x7F,0x40
        nop.i 999
}
;;

// p7 = TRUE ==> yes, we have an underflow
{ .mfi
        nop.m 999
        fcmp.eq.s1 p7, p0               =  POW_ftz_urm_f8, f0
        nop.i 999
}
;;

{ .mmi
(p7)    and pow_GR_rcs0  = pow_GR_rcs0_mask, pow_GR_Fpsr  // Isolate rc.s0
;;
(p7)    cmp.eq.unc p6,p0 = pow_GR_rcs0, r0    // Test for round to nearest
        nop.i 999
}
;;

// Tweak result slightly if underflow to get correct rounding near smallest
// denormal if round-nearest
{ .mfi
        nop.m 999
(p6)    fms.s.s0 f8                     = POW_A, POW_es, POW_eps
        nop.i 999
}
{ .mbb
(p7)    mov pow_GR_tag                  = 31
(p7)    br.cond.spnt __libm_error_region // Branch if underflow
        br.ret.sptk     b0               // Exit if did not underflow
}
;;

POW_X_DENORM:
// Here if x unorm. Use the NORM_X for getf instructions, and then back
// to normal path
{ .mfi
        getf.exp      pow_GR_signexp_X  = POW_NORM_X
        nop.f 999
        nop.i 999
}
;;

{ .mib
        getf.sig      pow_GR_sig_X      = POW_NORM_X
        nop.i 999
        br.cond.sptk    POW_COMMON
}
;;

POW_X_0:
// Here if x=0 and y not nan
//
// We have the following cases:
//  p6  x=0  and  y>0 and is an integer (may be even or odd)
//  p7  x=0  and  y>0 and is NOT an integer, return +0
//  p8  x=0  and  y>0 and so big as to always be an even integer, return +0
//  p9  x=0  and  y>0 and may not be integer
//  p10 x=0  and  y>0 and is an odd  integer, return x
//  p11 x=0  and  y>0 and is an even integer, return +0
//  p12 used in dummy fcmp to set denormal flag if y=unorm
//  p13 x=0  and  y>0
//  p14 x=0  and  y=0, branch to code for calling error handling
//  p15 x=0  and  y<0, branch to code for calling error handling
//
{ .mfi
        getf.sig pow_GR_sig_int_Y = POW_int_Y // Get signif of int_Y
        fcmp.lt.s1 p15,p13 = f9, f0           // Test for y<0
        and pow_GR_exp_Y = pow_GR_signexp_Y, pow_GR_17ones
}
{ .mfb
        cmp.ne p14,p0 = pow_GR_y_zero,r0      // Test for y=0
        fcvt.xf   POW_float_int_Y = POW_int_Y
(p14)   br.cond.spnt POW_X_0_Y_0              // Branch if x=0 and y=0
}
;;

// If x=0 and y>0, test y and flag denormal
{ .mfb
(p13)   cmp.gt.unc p8,p9 = pow_GR_exp_Y, pow_GR_10033 // Test y +big = even int
(p13)   fcmp.eq.s0 p12,p0 = f9,f0    // If x=0, y>0 dummy op to flag denormal
(p15)   br.cond.spnt POW_X_0_Y_NEG // Branch if x=0 and y<0
}
;;

// Here if x=0 and y>0
{ .mfi
        nop.m 999
(p9)    fcmp.eq.unc.s1 p6,p7 = POW_float_int_Y,  POW_NORM_Y // Test y=int
        nop.i 999
}
{ .mfi
        nop.m 999
(p8)    fma.s.s0 f8 = f0,f0,f0 // If x=0, y>0 and large even int, return +0
        nop.i 999
}
;;

{ .mfi
        nop.m 999
(p7)    fma.s.s0 f8  = f0,f0,f0   // Result +0 if x=0 and y>0 and not integer
(p6)    tbit.nz.unc p10,p11 = pow_GR_sig_int_Y,0 // If y>0 int, test y even/odd
}
;;

// Note if x=0, y>0 and odd integer, just return x
{ .mfb
        nop.m 999
(p11)   fma.s.s0 f8  = f0,f0,f0   // Result +0 if x=0 and y even integer
        br.ret.sptk b0            // Exit if x=0 and y>0
}
;;

POW_X_0_Y_0:
// When X is +-0 and Y is +-0, IEEE returns 1.0
// We call error support with this value

{ .mfb
        mov pow_GR_tag                  = 32
        fma.s.s0 f8                     = f1,f1,f0
        br.cond.sptk __libm_error_region
}
;;

POW_X_0_Y_NEG:
// When X is +-0 and Y is negative, IEEE returns
// X     Y           answer
// +0    -odd int    +inf
// -0    -odd int    -inf

// +0    !-odd int   +inf
// -0    !-odd int   +inf

// p6 == Y is a floating point number outside the integer.
//       Hence it is an integer and is even.
//       return +inf

// p7 == Y is a floating point number within the integer range.
//      p9  == (int_Y = NORM_Y), Y is an integer, which may be odd or even.
//           p11 odd
//              return (sign_of_x)inf
//           p12 even
//              return +inf
//      p10 == Y is not an integer
//         return +inf
//

{ .mfi
          nop.m 999
          nop.f 999
          cmp.gt  p6,p7                 = pow_GR_exp_Y, pow_GR_10033
}
;;

{ .mfi
          mov pow_GR_tag                = 33
(p7)      fcmp.eq.unc.s1 p9,p10         = POW_float_int_Y,  POW_NORM_Y
          nop.i 999
}
;;

{ .mfb
          nop.m 999
(p6)      frcpa.s0 f8,p13               = f1, f0
(p6)      br.cond.sptk __libm_error_region   // x=0, y<0, y large neg int
}
;;

{ .mfb
          nop.m 999
(p10)     frcpa.s0 f8,p13               = f1, f0
(p10)     br.cond.sptk __libm_error_region   // x=0, y<0, y not int
}
;;

// x=0, y<0, y an int
{ .mib
          nop.m 999
(p9)      tbit.nz.unc p11,p12           = pow_GR_sig_int_Y,0
          nop.b 999
}
;;

{ .mfi
          nop.m 999
(p12)     frcpa.s0 f8,p13               = f1,f0
          nop.i 999
}
;;

{ .mfb
          nop.m 999
(p11)     frcpa.s0 f8,p13               = f1,f8
          br.cond.sptk __libm_error_region
}
;;


POW_Y_0:
// Here for y zero, x anything but zero and nan
// Set flag if x denormal
// Result is +1.0
{ .mfi
        nop.m 999
        fcmp.eq.s0 p6,p0 = f8,f0    // Sets flag if x denormal
        nop.i 999
}
{ .mfb
        nop.m 999
        fma.s.s0 f8 = f1,f1,f0
        br.ret.sptk b0
}
;;


POW_X_INF:
// Here when X is +-inf

// X +inf  Y +inf             +inf
// X -inf  Y +inf             +inf

// X +inf  Y >0               +inf
// X -inf  Y >0, !odd integer +inf     <== (-inf)^0.5 = +inf !!
// X -inf  Y >0,  odd integer -inf

// X +inf  Y -inf             +0
// X -inf  Y -inf             +0

// X +inf  Y <0               +0
// X -inf  Y <0, !odd integer +0
// X -inf  Y <0, odd integer  -0

// X + inf Y=+0                +1
// X + inf Y=-0                +1
// X - inf Y=+0                +1
// X - inf Y=-0                +1

// p13 == Y negative
// p14 == Y positive

// p6 == Y is a floating point number outside the integer.
//       Hence it is an integer and is even.
//       p13 == (Y negative)
//          return +inf
//       p14 == (Y positive)
//          return +0

// p7 == Y is a floating point number within the integer range.
//      p9  == (int_Y = NORM_Y), Y is an integer, which may be odd or even.
//           p11 odd
//              p13 == (Y negative)
//                 return (sign_of_x)inf
//              p14 == (Y positive)
//                 return (sign_of_x)0
//           pxx even
//              p13 == (Y negative)
//                 return +inf
//              p14 == (Y positive)
//                 return +0

//      pxx == Y is not an integer
//           p13 == (Y negative)
//                 return +inf
//           p14 == (Y positive)
//                 return +0
//

// If x=inf, test y and flag denormal
{ .mfi
          nop.m 999
          fcmp.eq.s0 p10,p11 = f9,f0
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          fcmp.lt.s0 p13,p14            = POW_NORM_Y,f0
          cmp.gt  p6,p7                 = pow_GR_exp_Y, pow_GR_10033
}
{ .mfi
          nop.m 999
          fclass.m p12,p0               = f9, 0x23 //@inf
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          fclass.m p15,p0               = f9, 0x07 //@zero
          nop.i 999
}
;;

{ .mfb
          nop.m 999
(p15)     fmerge.s f8 = f1,f1      // Return +1.0 if x=inf, y=0
(p15)     br.ret.spnt b0           // Exit if x=inf, y=0
}
;;

{ .mfi
          nop.m 999
(p14)     frcpa.s1 f8,p10 = f1,f0  // If x=inf, y>0, assume result +inf
          nop.i 999
}
{ .mfb
          nop.m 999
(p13)     fma.s.s0 f8 = f0,f0,f0   // If x=inf, y<0, assume result +0.0
(p12)     br.ret.spnt b0           // Exit if x=inf, y=inf
}
;;

// Here if x=inf, and 0 < |y| < inf.  Need to correct results if y odd integer.
{ .mfi
          nop.m 999
(p7)      fcmp.eq.unc.s1 p9,p0 = POW_float_int_Y,  POW_NORM_Y // Is y integer?
          nop.i 999
}
;;

{ .mfi
          nop.m 999
          nop.f 999
(p9)      tbit.nz.unc p11,p0 = pow_GR_sig_int_Y,0  // Test for y odd integer
}
;;

{ .mfb
          nop.m 999
(p11)     fmerge.s f8 = POW_NORM_X,f8    // If y odd integer use sign of x
          br.ret.sptk b0                 // Exit for x=inf, 0 < |y| < inf
}
;;


POW_X_NEG_Y_NONINT:
// When X is negative and Y is a non-integer, IEEE
// returns a qnan indefinite.
// We call error support with this value

{ .mfb
         mov pow_GR_tag                 = 34
         frcpa.s0 f8,p6                 = f0,f0
         br.cond.sptk __libm_error_region
}
;;

POW_X_NAN:
// Here if x=nan, y not nan
{ .mfi
         nop.m 999
         fclass.m  p9,p13 = f9, 0x07 // Test y=zero
         nop.i 999
}
;;

{ .mfb
         nop.m 999
(p13)    fma.s.s0 f8 = f8,f1,f0
(p13)    br.ret.sptk  b0            // Exit if x nan, y anything but zero or nan
}
;;

POW_X_NAN_Y_0:
// When X is a NAN and Y is zero, IEEE returns 1.
// We call error support with this value.
{ .mfi
         nop.m 999
         fcmp.eq.s0 p6,p0 = f8,f0       // Dummy op to set invalid on snan
         nop.i 999
}
{ .mfb
         mov pow_GR_tag                 = 35
         fma.s.s0 f8 = f0,f0,f1
         br.cond.sptk __libm_error_region
}
;;


POW_OVER_UNDER_X_NOT_INF:

// p8 is TRUE for overflow
// p9 is TRUE for underflow

// if y is infinity, we should not over/underflow

{ .mfi
          nop.m 999
          fcmp.eq.s1     p14, p13       = POW_xsq,f1  // Test |x|=1
          cmp.eq p8,p9                  = pow_GR_sign_Y_Gpr, r0
}
;;

{ .mfi
          nop.m 999
(p14)     fclass.m.unc       p15, p0    = f9, 0x23 // If |x|=1, test y=inf
          nop.i 999
}
{ .mfi
          nop.m 999
(p13)     fclass.m.unc       p11,p0     = f9, 0x23 // If |x| not 1, test y=inf
          nop.i 999
}
;;

// p15 = TRUE if |x|=1, y=inf, return +1
{ .mfb
          nop.m 999
(p15)     fma.s.s0          f8          = f1,f1,f0 // If |x|=1, y=inf, result +1
(p15)     br.ret.spnt b0                // Exit if |x|=1, y=inf
}
;;

.pred.rel "mutex",p8,p9
{  .mfb
(p8)      setf.exp           f8 = pow_GR_17ones // If exp(+big), result inf
(p9)      fmerge.s           f8 = f0,f0         // If exp(-big), result 0
(p11)     br.ret.sptk b0                // Exit if |x| not 1, y=inf
}
;;

{ .mfb
          nop.m 999
          nop.f 999
          br.cond.sptk POW_OVER_UNDER_ERROR // Branch if y not inf
}
;;


POW_Y_NAN:
// Here if y=nan, x anything
// If x = +1 then result is +1, else result is quiet Y
{ .mfi
       nop.m 999
       fcmp.eq.s1         p10,p9        = POW_NORM_X, f1
       nop.i 999
}
;;

{ .mfi
       nop.m 999
(p10)  fcmp.eq.s0 p6,p0 = f9,f1   // Set invalid, even if x=+1
       nop.i 999
}
;;

{ .mfi
       nop.m 999
(p10)  fma.s.s0 f8 = f1,f1,f0
       nop.i 999
}
{ .mfb
       nop.m 999
(p9)   fma.s.s0 f8 = f9,f8,f0
       br.ret.sptk b0             // Exit y=nan
}
;;


POW_OVER_UNDER_ERROR:
// Here if we have overflow or underflow.
// Enter with p12 true if x negative and y odd int to force -0 or -inf

{ .mfi
         sub   pow_GR_17ones_m1         = pow_GR_17ones, r0, 1
         nop.f 999
         mov pow_GR_one                 = 0x1
}
;;

// overflow, force inf with O flag
{ .mmb
(p8)     mov pow_GR_tag                 = 30
(p8)     setf.exp POW_tmp               = pow_GR_17ones_m1
         nop.b 999
}
;;

// underflow, force zero with I, U flags
{ .mmi
(p9)    mov pow_GR_tag                  = 31
(p9)    setf.exp POW_tmp                = pow_GR_one
        nop.i 999
}
;;

{ .mfi
        nop.m 999
        fma.s.s0 f8                     = POW_tmp, POW_tmp, f0
        nop.i 999
}
;;

// p12 x is negative and y is an odd integer, change sign of result
{ .mfi
        nop.m 999
(p12)   fnma.s.s0 f8                    = POW_tmp, POW_tmp, f0
        nop.i 999
}
;;

WEAK_LIBM_END(powf)
libm_alias_float_other (__pow, pow)
#ifdef SHARED
.symver powf,powf@@GLIBC_2.27
.weak __powf_compat
.set __powf_compat,__powf
.symver __powf_compat,powf@GLIBC_2.2
#endif


LOCAL_LIBM_ENTRY(__libm_error_region)

.prologue
{ .mfi
        add   GR_Parameter_Y=-32,sp     // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS=ar.pfs         // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp=-64,sp                   // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp               // Save gp
};;

{ .mmi
        stfs [GR_Parameter_Y] = POW_NORM_Y,16 // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp      // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0               // Save b0
};;

.body
{ .mib
        stfs [GR_Parameter_X] = POW_NORM_X // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y    // Parameter 3 address
        nop.b 0
}
{ .mib
        stfs [GR_Parameter_Y] = f8      // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support# // Call error handling function
};;

{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;

{ .mmi
        ldfs  f8 = [GR_Parameter_RESULT] // Get return result off stack
.restore sp
        add   sp = 64,sp                 // Restore stack pointer
        mov   b0 = GR_SAVE_B0            // Restore return address
};;

{ .mib
        mov   gp = GR_SAVE_GP            // Restore gp
        mov   ar.pfs = GR_SAVE_PFS       // Restore ar.pfs
        br.ret.sptk     b0               // Return
};;

LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#
