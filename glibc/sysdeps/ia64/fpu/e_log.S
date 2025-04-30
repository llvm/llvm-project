.file "log.s"


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
// 04/04/00 Unwind support added
// 06/16/00 Updated table to be rounded correctly
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 08/17/00 Improved speed of main path by 5 cycles
//          Shortened path for x=1.0
// 01/09/01 Improved speed, fixed flags for neg denormals
// 05/20/02 Cleaned up namespace and sf0 syntax
// 05/23/02 Modified algorithm. Now only one polynomial is used
//          for |x-1| >= 1/256 and for |x-1| < 1/256
// 12/11/02 Improved performance for Itanium 2
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// double log(double)
// double log10(double)
//
//
// Overview of operation
//==============================================================
// Background
// ----------
//
// This algorithm is based on fact that
// log(a b) = log(a) + log(b).
// In our case we have x = 2^N f, where 1 <= f < 2.
// So
//   log(x) = log(2^N f) = log(2^N) + log(f) = n*log(2) + log(f)
//
// To calculate log(f) we do following
//   log(f) = log(f * frcpa(f) / frcpa(f)) =
//          = log(f * frcpa(f)) + log(1/frcpa(f))
//
// According to definition of IA-64's frcpa instruction it's a
// floating point that approximates 1/f using a lookup on the
// top of 8 bits of the input number's significand with relative
// error < 2^(-8.886). So we have following
//
// |(1/f - frcpa(f)) / (1/f))| = |1 - f*frcpa(f)| < 1/256
//
// and
//
// log(f) = log(f * frcpa(f)) + log(1/frcpa(f)) =
//        = log(1 + r) + T
//
// The first value can be computed by polynomial P(r) approximating
// log(1 + r) on |r| < 1/256 and the second is precomputed tabular
// value defined by top 8 bit of f.
//
// Finally we have that  log(x) ~ (N*log(2) + T) + P(r)
//
// Note that if input argument is close to 1.0 (in our case it means
// that |1 - x| < 1/256) we can use just polynomial approximation
// because x = 2^0 * f = f = 1 + r and
// log(x) = log(1 + r) ~ P(r)
//
//
// To compute log10(x) we use the simple identity
//
//  log10(x) = log(x)/log(10)
//
// so we have that
//
//  log10(x) = (N*log(2) + T  + log(1+r)) / log(10) =
//           = N*(log(2)/log(10)) + (T/log(10)) + log(1 + r)/log(10)
//
//
// Implementation
// --------------
// It can be seen that formulas for log and log10 differ from one another
// only by coefficients and tabular values. Namely as log as log10 are
// calculated as (N*L1 + T) + L2*Series(r) where in case of log
//   L1 = log(2)
//   T  = log(1/frcpa(x))
//   L2 = 1.0
// and in case of log10
//   L1 = log(2)/log(10)
//   T  = log(1/frcpa(x))/log(10)
//   L2 = 1.0/log(10)
//
// So common code with two different entry points those set pointers
// to the base address of coresponding data sets containing values
// of L2,T and prepare integer representation of L1 needed for following
// setf instruction.
//
// Note that both log and log10 use common approximation polynomial
// it means we need only one set of coefficients of approximation.
//
//
// 1. |x-1| >= 1/256
//   InvX = frcpa(x)
//   r = InvX*x - 1
//   P(r) = r*((r*A3 - A2) + r^4*((A4 + r*A5) + r^2*(A6 + r*A7)),
//   all coefficients are calculated in quad and rounded to double
//   precision. A7,A6,A5,A4 are stored in memory whereas A3 and A2
//   created with setf.
//
//   N = float(n) where n is true unbiased exponent of x
//
//   T is tabular value of log(1/frcpa(x)) calculated in quad precision
//   and represented by two floating-point numbers 64-bit Thi and 32-bit Tlo.
//   To load Thi,Tlo we get bits from 55 to 62 of register format significand
//   as index and calculate two addresses
//     ad_Thi = Thi_table_base_addr + 8 * index
//     ad_Tlo = Tlo_table_base_addr + 4 * index
//
//   L2 (1.0 or 1.0/log(10) depending on function) is calculated in quad
//   precision and rounded to double extended; it's loaded from memory.
//
//   L1 (log(2) or log10(2) depending on function) is calculated in quad
//   precision and represented by two floating-point 64-bit numbers L1hi,L1lo
//   stored in memory.
//
//   And final result = ((L1hi*N + Thi) + (N*L1lo + Tlo)) + L2*P(r)
//
//
// 2. |x-1| < 1/256
//   r = x - 1
//   P(r) = r*((r*A3 - A2) + r^4*((A4 + r*A5) + r^2*(A6 + r*A7)),
//   A7,A6,A5A4,A3,A2 are the same as in case |x-1| >= 1/256
//
//   And final results
//     log(x)   = P(r)
//     log10(x) = L2*P(r)
//
// 3. How we define is input argument such that |x-1| < 1/256 or not.
//
//    To do it we analyze biased exponent and integer representation of
//    input argument
//
//      a) First we test is biased exponent equal to 0xFFFE or 0xFFFF (i.e.
//         we test is 0.5 <= x < 2). This comparison can be performed using
//         unsigned version of cmp instruction in such a way
//         biased_exponent_of_x - 0xFFFE < 2
//
//
//      b) Second (in case when result of a) is true) we need to compare x
//         with 1-1/256 and 1+1/256 or in double precision memory representation
//         with 0x3FEFE00000000000 and 0x3FF0100000000000 correspondingly.
//         This comparison can be made like in a), using unsigned
//         version of cmp i.e. ix - 0x3FEFE00000000000 < 0x0000300000000000.
//         0x0000300000000000 is difference between 0x3FF0100000000000 and
//         0x3FEFE00000000000
//
//    Note: NaT, any NaNs, +/-INF, +/-0, negatives and unnormalized numbers are
//          filtered and processed on special branches.
//

//
// Special values
//==============================================================
//
// log(+0)    = -inf
// log(-0)    = -inf
//
// log(+qnan) = +qnan
// log(-qnan) = -qnan
// log(+snan) = +qnan
// log(-snan) = -qnan
//
// log(-n)    = QNAN Indefinite
// log(-inf)  = QNAN Indefinite
//
// log(+inf)  = +inf
//
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f7 -> f15,  f32 -> f42
//
// General registers used:
// r8  -> r11
// r14 -> r23
//
// Predicate registers used:
// p6 -> p15

// Assembly macros
//==============================================================
GR_TAG                 = r8
GR_ad_1                = r8
GR_ad_2                = r9
GR_Exp                 = r10
GR_N                   = r11

GR_x                   = r14
GR_dx                  = r15
GR_NearOne             = r15
GR_xorg                = r16
GR_mask                = r16
GR_05                  = r17
GR_A3                  = r18
GR_Sig                 = r19
GR_Ind                 = r19
GR_Nm1                 = r20
GR_bias                = r21
GR_ad_3                = r22
GR_rexp                = r23


GR_SAVE_B0             = r33
GR_SAVE_PFS            = r34
GR_SAVE_GP             = r35
GR_SAVE_SP             = r36

GR_Parameter_X         = r37
GR_Parameter_Y         = r38
GR_Parameter_RESULT    = r39
GR_Parameter_TAG       = r40



FR_NormX               = f7
FR_RcpX                = f9
FR_tmp                 = f9
FR_r                   = f10
FR_r2                  = f11
FR_r4                  = f12
FR_N                   = f13
FR_Ln2hi               = f14
FR_Ln2lo               = f15

FR_A7                  = f32
FR_A6                  = f33
FR_A5                  = f34
FR_A4                  = f35
FR_A3                  = f36
FR_A2                  = f37

FR_Thi                 = f38
FR_NxLn2hipThi         = f38
FR_NxLn2pT             = f38
FR_Tlo                 = f39
FR_NxLn2lopTlo         = f39

FR_InvLn10             = f40
FR_A32                 = f41
FR_A321                = f42


FR_Y                   = f1
FR_X                   = f10
FR_RESULT              = f8


// Data
//==============================================================
RODATA
.align 16

LOCAL_OBJECT_START(log_data)
// coefficients of polynomial approximation
data8 0x3FC2494104381A8E // A7
data8 0xBFC5556D556BBB69 // A6
//
// two parts of ln(2)
data8 0x3FE62E42FEF00000,0x3DD473DE6AF278ED
//
data8 0x8000000000000000,0x3FFF // 1.0
//
data8 0x3FC999999988B5E9 // A5
data8 0xBFCFFFFFFFF6FFF5 // A4
//
// hi parts of ln(1/frcpa(1+i/256)), i=0...255
data8 0x3F60040155D5889D // 0
data8 0x3F78121214586B54 // 1
data8 0x3F841929F96832EF // 2
data8 0x3F8C317384C75F06 // 3
data8 0x3F91A6B91AC73386 // 4
data8 0x3F95BA9A5D9AC039 // 5
data8 0x3F99D2A8074325F3 // 6
data8 0x3F9D6B2725979802 // 7
data8 0x3FA0C58FA19DFAA9 // 8
data8 0x3FA2954C78CBCE1A // 9
data8 0x3FA4A94D2DA96C56 // 10
data8 0x3FA67C94F2D4BB58 // 11
data8 0x3FA85188B630F068 // 12
data8 0x3FAA6B8ABE73AF4C // 13
data8 0x3FAC441E06F72A9E // 14
data8 0x3FAE1E6713606D06 // 15
data8 0x3FAFFA6911AB9300 // 16
data8 0x3FB0EC139C5DA600 // 17
data8 0x3FB1DBD2643D190B // 18
data8 0x3FB2CC7284FE5F1C // 19
data8 0x3FB3BDF5A7D1EE64 // 20
data8 0x3FB4B05D7AA012E0 // 21
data8 0x3FB580DB7CEB5701 // 22
data8 0x3FB674F089365A79 // 23
data8 0x3FB769EF2C6B568D // 24
data8 0x3FB85FD927506A47 // 25
data8 0x3FB9335E5D594988 // 26
data8 0x3FBA2B0220C8E5F4 // 27
data8 0x3FBB0004AC1A86AB // 28
data8 0x3FBBF968769FCA10 // 29
data8 0x3FBCCFEDBFEE13A8 // 30
data8 0x3FBDA727638446A2 // 31
data8 0x3FBEA3257FE10F79 // 32
data8 0x3FBF7BE9FEDBFDE5 // 33
data8 0x3FC02AB352FF25F3 // 34
data8 0x3FC097CE579D204C // 35
data8 0x3FC1178E8227E47B // 36
data8 0x3FC185747DBECF33 // 37
data8 0x3FC1F3B925F25D41 // 38
data8 0x3FC2625D1E6DDF56 // 39
data8 0x3FC2D1610C868139 // 40
data8 0x3FC340C59741142E // 41
data8 0x3FC3B08B6757F2A9 // 42
data8 0x3FC40DFB08378003 // 43
data8 0x3FC47E74E8CA5F7C // 44
data8 0x3FC4EF51F6466DE4 // 45
data8 0x3FC56092E02BA516 // 46
data8 0x3FC5D23857CD74D4 // 47
data8 0x3FC6313A37335D76 // 48
data8 0x3FC6A399DABBD383 // 49
data8 0x3FC70337DD3CE41A // 50
data8 0x3FC77654128F6127 // 51
data8 0x3FC7E9D82A0B022D // 52
data8 0x3FC84A6B759F512E // 53
data8 0x3FC8AB47D5F5A30F // 54
data8 0x3FC91FE49096581B // 55
data8 0x3FC981634011AA75 // 56
data8 0x3FC9F6C407089664 // 57
data8 0x3FCA58E729348F43 // 58
data8 0x3FCABB55C31693AC // 59
data8 0x3FCB1E104919EFD0 // 60
data8 0x3FCB94EE93E367CA // 61
data8 0x3FCBF851C067555E // 62
data8 0x3FCC5C0254BF23A5 // 63
data8 0x3FCCC000C9DB3C52 // 64
data8 0x3FCD244D99C85673 // 65
data8 0x3FCD88E93FB2F450 // 66
data8 0x3FCDEDD437EAEF00 // 67
data8 0x3FCE530EFFE71012 // 68
data8 0x3FCEB89A1648B971 // 69
data8 0x3FCF1E75FADF9BDE // 70
data8 0x3FCF84A32EAD7C35 // 71
data8 0x3FCFEB2233EA07CD // 72
data8 0x3FD028F9C7035C1C // 73
data8 0x3FD05C8BE0D9635A // 74
data8 0x3FD085EB8F8AE797 // 75
data8 0x3FD0B9C8E32D1911 // 76
data8 0x3FD0EDD060B78080 // 77
data8 0x3FD122024CF0063F // 78
data8 0x3FD14BE2927AECD4 // 79
data8 0x3FD180618EF18ADF // 80
data8 0x3FD1B50BBE2FC63B // 81
data8 0x3FD1DF4CC7CF242D // 82
data8 0x3FD214456D0EB8D4 // 83
data8 0x3FD23EC5991EBA49 // 84
data8 0x3FD2740D9F870AFB // 85
data8 0x3FD29ECDABCDFA03 // 86
data8 0x3FD2D46602ADCCEE // 87
data8 0x3FD2FF66B04EA9D4 // 88
data8 0x3FD335504B355A37 // 89
data8 0x3FD360925EC44F5C // 90
data8 0x3FD38BF1C3337E74 // 91
data8 0x3FD3C25277333183 // 92
data8 0x3FD3EDF463C1683E // 93
data8 0x3FD419B423D5E8C7 // 94
data8 0x3FD44591E0539F48 // 95
data8 0x3FD47C9175B6F0AD // 96
data8 0x3FD4A8B341552B09 // 97
data8 0x3FD4D4F39089019F // 98
data8 0x3FD501528DA1F967 // 99
data8 0x3FD52DD06347D4F6 // 100
data8 0x3FD55A6D3C7B8A89 // 101
data8 0x3FD5925D2B112A59 // 102
data8 0x3FD5BF406B543DB1 // 103
data8 0x3FD5EC433D5C35AD // 104
data8 0x3FD61965CDB02C1E // 105
data8 0x3FD646A84935B2A1 // 106
data8 0x3FD6740ADD31DE94 // 107
data8 0x3FD6A18DB74A58C5 // 108
data8 0x3FD6CF31058670EC // 109
data8 0x3FD6F180E852F0B9 // 110
data8 0x3FD71F5D71B894EF // 111
data8 0x3FD74D5AEFD66D5C // 112
data8 0x3FD77B79922BD37D // 113
data8 0x3FD7A9B9889F19E2 // 114
data8 0x3FD7D81B037EB6A6 // 115
data8 0x3FD8069E33827230 // 116
data8 0x3FD82996D3EF8BCA // 117
data8 0x3FD85855776DCBFA // 118
data8 0x3FD8873658327CCE // 119
data8 0x3FD8AA75973AB8CE // 120
data8 0x3FD8D992DC8824E4 // 121
data8 0x3FD908D2EA7D9511 // 122
data8 0x3FD92C59E79C0E56 // 123
data8 0x3FD95BD750EE3ED2 // 124
data8 0x3FD98B7811A3EE5B // 125
data8 0x3FD9AF47F33D406B // 126
data8 0x3FD9DF270C1914A7 // 127
data8 0x3FDA0325ED14FDA4 // 128
data8 0x3FDA33440224FA78 // 129
data8 0x3FDA57725E80C382 // 130
data8 0x3FDA87D0165DD199 // 131
data8 0x3FDAAC2E6C03F895 // 132
data8 0x3FDADCCC6FDF6A81 // 133
data8 0x3FDB015B3EB1E790 // 134
data8 0x3FDB323A3A635948 // 135
data8 0x3FDB56FA04462909 // 136
data8 0x3FDB881AA659BC93 // 137
data8 0x3FDBAD0BEF3DB164 // 138
data8 0x3FDBD21297781C2F // 139
data8 0x3FDC039236F08818 // 140
data8 0x3FDC28CB1E4D32FC // 141
data8 0x3FDC4E19B84723C1 // 142
data8 0x3FDC7FF9C74554C9 // 143
data8 0x3FDCA57B64E9DB05 // 144
data8 0x3FDCCB130A5CEBAF // 145
data8 0x3FDCF0C0D18F326F // 146
data8 0x3FDD232075B5A201 // 147
data8 0x3FDD490246DEFA6B // 148
data8 0x3FDD6EFA918D25CD // 149
data8 0x3FDD9509707AE52F // 150
data8 0x3FDDBB2EFE92C554 // 151
data8 0x3FDDEE2F3445E4AE // 152
data8 0x3FDE148A1A2726CD // 153
data8 0x3FDE3AFC0A49FF3F // 154
data8 0x3FDE6185206D516D // 155
data8 0x3FDE882578823D51 // 156
data8 0x3FDEAEDD2EAC990C // 157
data8 0x3FDED5AC5F436BE2 // 158
data8 0x3FDEFC9326D16AB8 // 159
data8 0x3FDF2391A21575FF // 160
data8 0x3FDF4AA7EE03192C // 161
data8 0x3FDF71D627C30BB0 // 162
data8 0x3FDF991C6CB3B379 // 163
data8 0x3FDFC07ADA69A90F // 164
data8 0x3FDFE7F18EB03D3E // 165
data8 0x3FE007C053C5002E // 166
data8 0x3FE01B942198A5A0 // 167
data8 0x3FE02F74400C64EA // 168
data8 0x3FE04360BE7603AC // 169
data8 0x3FE05759AC47FE33 // 170
data8 0x3FE06B5F1911CF51 // 171
data8 0x3FE078BF0533C568 // 172
data8 0x3FE08CD9687E7B0E // 173
data8 0x3FE0A10074CF9019 // 174
data8 0x3FE0B5343A234476 // 175
data8 0x3FE0C974C89431CD // 176
data8 0x3FE0DDC2305B9886 // 177
data8 0x3FE0EB524BAFC918 // 178
data8 0x3FE0FFB54213A475 // 179
data8 0x3FE114253DA97D9F // 180
data8 0x3FE128A24F1D9AFF // 181
data8 0x3FE1365252BF0864 // 182
data8 0x3FE14AE558B4A92D // 183
data8 0x3FE15F85A19C765B // 184
data8 0x3FE16D4D38C119FA // 185
data8 0x3FE18203C20DD133 // 186
data8 0x3FE196C7BC4B1F3A // 187
data8 0x3FE1A4A738B7A33C // 188
data8 0x3FE1B981C0C9653C // 189
data8 0x3FE1CE69E8BB106A // 190
data8 0x3FE1DC619DE06944 // 191
data8 0x3FE1F160A2AD0DA3 // 192
data8 0x3FE2066D7740737E // 193
data8 0x3FE2147DBA47A393 // 194
data8 0x3FE229A1BC5EBAC3 // 195
data8 0x3FE237C1841A502E // 196
data8 0x3FE24CFCE6F80D9A // 197
data8 0x3FE25B2C55CD5762 // 198
data8 0x3FE2707F4D5F7C40 // 199
data8 0x3FE285E0842CA383 // 200
data8 0x3FE294294708B773 // 201
data8 0x3FE2A9A2670AFF0C // 202
data8 0x3FE2B7FB2C8D1CC0 // 203
data8 0x3FE2C65A6395F5F5 // 204
data8 0x3FE2DBF557B0DF42 // 205
data8 0x3FE2EA64C3F97654 // 206
data8 0x3FE3001823684D73 // 207
data8 0x3FE30E97E9A8B5CC // 208
data8 0x3FE32463EBDD34E9 // 209
data8 0x3FE332F4314AD795 // 210
data8 0x3FE348D90E7464CF // 211
data8 0x3FE35779F8C43D6D // 212
data8 0x3FE36621961A6A99 // 213
data8 0x3FE37C299F3C366A // 214
data8 0x3FE38AE2171976E7 // 215
data8 0x3FE399A157A603E7 // 216
data8 0x3FE3AFCCFE77B9D1 // 217
data8 0x3FE3BE9D503533B5 // 218
data8 0x3FE3CD7480B4A8A2 // 219
data8 0x3FE3E3C43918F76C // 220
data8 0x3FE3F2ACB27ED6C6 // 221
data8 0x3FE4019C2125CA93 // 222
data8 0x3FE4181061389722 // 223
data8 0x3FE42711518DF545 // 224
data8 0x3FE436194E12B6BF // 225
data8 0x3FE445285D68EA69 // 226
data8 0x3FE45BCC464C893A // 227
data8 0x3FE46AED21F117FC // 228
data8 0x3FE47A1527E8A2D3 // 229
data8 0x3FE489445EFFFCCB // 230
data8 0x3FE4A018BCB69835 // 231
data8 0x3FE4AF5A0C9D65D7 // 232
data8 0x3FE4BEA2A5BDBE87 // 233
data8 0x3FE4CDF28F10AC46 // 234
data8 0x3FE4DD49CF994058 // 235
data8 0x3FE4ECA86E64A683 // 236
data8 0x3FE503C43CD8EB68 // 237
data8 0x3FE513356667FC57 // 238
data8 0x3FE522AE0738A3D7 // 239
data8 0x3FE5322E26867857 // 240
data8 0x3FE541B5CB979809 // 241
data8 0x3FE55144FDBCBD62 // 242
data8 0x3FE560DBC45153C6 // 243
data8 0x3FE5707A26BB8C66 // 244
data8 0x3FE587F60ED5B8FF // 245
data8 0x3FE597A7977C8F31 // 246
data8 0x3FE5A760D634BB8A // 247
data8 0x3FE5B721D295F10E // 248
data8 0x3FE5C6EA94431EF9 // 249
data8 0x3FE5D6BB22EA86F5 // 250
data8 0x3FE5E6938645D38F // 251
data8 0x3FE5F673C61A2ED1 // 252
data8 0x3FE6065BEA385926 // 253
data8 0x3FE6164BFA7CC06B // 254
data8 0x3FE62643FECF9742 // 255
//
// lo parts of ln(1/frcpa(1+i/256)), i=0...255
data4 0x20E70672 // 0
data4 0x1F60A5D0 // 1
data4 0x218EABA0 // 2
data4 0x21403104 // 3
data4 0x20E9B54E // 4
data4 0x21EE1382 // 5
data4 0x226014E3 // 6
data4 0x2095E5C9 // 7
data4 0x228BA9D4 // 8
data4 0x22932B86 // 9
data4 0x22608A57 // 10
data4 0x220209F3 // 11
data4 0x212882CC // 12
data4 0x220D46E2 // 13
data4 0x21FA4C28 // 14
data4 0x229E5BD9 // 15
data4 0x228C9838 // 16
data4 0x2311F954 // 17
data4 0x221365DF // 18
data4 0x22BD0CB3 // 19
data4 0x223D4BB7 // 20
data4 0x22A71BBE // 21
data4 0x237DB2FA // 22
data4 0x23194C9D // 23
data4 0x22EC639E // 24
data4 0x2367E669 // 25
data4 0x232E1D5F // 26
data4 0x234A639B // 27
data4 0x2365C0E0 // 28
data4 0x234646C1 // 29
data4 0x220CBF9C // 30
data4 0x22A00FD4 // 31
data4 0x2306A3F2 // 32
data4 0x23745A9B // 33
data4 0x2398D756 // 34
data4 0x23DD0B6A // 35
data4 0x23DE338B // 36
data4 0x23A222DF // 37
data4 0x223164F8 // 38
data4 0x23B4E87B // 39
data4 0x23D6CCB8 // 40
data4 0x220C2099 // 41
data4 0x21B86B67 // 42
data4 0x236D14F1 // 43
data4 0x225A923F // 44
data4 0x22748723 // 45
data4 0x22200D13 // 46
data4 0x23C296EA // 47
data4 0x2302AC38 // 48
data4 0x234B1996 // 49
data4 0x2385E298 // 50
data4 0x23175BE5 // 51
data4 0x2193F482 // 52
data4 0x23BFEA90 // 53
data4 0x23D70A0C // 54
data4 0x231CF30A // 55
data4 0x235D9E90 // 56
data4 0x221AD0CB // 57
data4 0x22FAA08B // 58
data4 0x23D29A87 // 59
data4 0x20C4B2FE // 60
data4 0x2381B8B7 // 61
data4 0x23F8D9FC // 62
data4 0x23EAAE7B // 63
data4 0x2329E8AA // 64
data4 0x23EC0322 // 65
data4 0x2357FDCB // 66
data4 0x2392A9AD // 67
data4 0x22113B02 // 68
data4 0x22DEE901 // 69
data4 0x236A6D14 // 70
data4 0x2371D33E // 71
data4 0x2146F005 // 72
data4 0x23230B06 // 73
data4 0x22F1C77D // 74
data4 0x23A89FA3 // 75
data4 0x231D1241 // 76
data4 0x244DA96C // 77
data4 0x23ECBB7D // 78
data4 0x223E42B4 // 79
data4 0x23801BC9 // 80
data4 0x23573263 // 81
data4 0x227C1158 // 82
data4 0x237BD749 // 83
data4 0x21DDBAE9 // 84
data4 0x23401735 // 85
data4 0x241D9DEE // 86
data4 0x23BC88CB // 87
data4 0x2396D5F1 // 88
data4 0x23FC89CF // 89
data4 0x2414F9A2 // 90
data4 0x2474A0F5 // 91
data4 0x24354B60 // 92
data4 0x23C1EB40 // 93
data4 0x2306DD92 // 94
data4 0x24353B6B // 95
data4 0x23CD1701 // 96
data4 0x237C7A1C // 97
data4 0x245793AA // 98
data4 0x24563695 // 99
data4 0x23C51467 // 100
data4 0x24476B68 // 101
data4 0x212585A9 // 102
data4 0x247B8293 // 103
data4 0x2446848A // 104
data4 0x246A53F8 // 105
data4 0x246E496D // 106
data4 0x23ED1D36 // 107
data4 0x2314C258 // 108
data4 0x233244A7 // 109
data4 0x245B7AF0 // 110
data4 0x24247130 // 111
data4 0x22D67B38 // 112
data4 0x2449F620 // 113
data4 0x23BBC8B8 // 114
data4 0x237D3BA0 // 115
data4 0x245E8F13 // 116
data4 0x2435573F // 117
data4 0x242DE666 // 118
data4 0x2463BC10 // 119
data4 0x2466587D // 120
data4 0x2408144B // 121
data4 0x2405F0E5 // 122
data4 0x22381CFF // 123
data4 0x24154F9B // 124
data4 0x23A4E96E // 125
data4 0x24052967 // 126
data4 0x2406963F // 127
data4 0x23F7D3CB // 128
data4 0x2448AFF4 // 129
data4 0x24657A21 // 130
data4 0x22FBC230 // 131
data4 0x243C8DEA // 132
data4 0x225DC4B7 // 133
data4 0x23496EBF // 134
data4 0x237C2B2B // 135
data4 0x23A4A5B1 // 136
data4 0x2394E9D1 // 137
data4 0x244BC950 // 138
data4 0x23C7448F // 139
data4 0x2404A1AD // 140
data4 0x246511D5 // 141
data4 0x24246526 // 142
data4 0x23111F57 // 143
data4 0x22868951 // 144
data4 0x243EB77F // 145
data4 0x239F3DFF // 146
data4 0x23089666 // 147
data4 0x23EBFA6A // 148
data4 0x23C51312 // 149
data4 0x23E1DD5E // 150
data4 0x232C0944 // 151
data4 0x246A741F // 152
data4 0x2414DF8D // 153
data4 0x247B5546 // 154
data4 0x2415C980 // 155
data4 0x24324ABD // 156
data4 0x234EB5E5 // 157
data4 0x2465E43E // 158
data4 0x242840D1 // 159
data4 0x24444057 // 160
data4 0x245E56F0 // 161
data4 0x21AE30F8 // 162
data4 0x23FB3283 // 163
data4 0x247A4D07 // 164
data4 0x22AE314D // 165
data4 0x246B7727 // 166
data4 0x24EAD526 // 167
data4 0x24B41DC9 // 168
data4 0x24EE8062 // 169
data4 0x24A0C7C4 // 170
data4 0x24E8DA67 // 171
data4 0x231120F7 // 172
data4 0x24401FFB // 173
data4 0x2412DD09 // 174
data4 0x248C131A // 175
data4 0x24C0A7CE // 176
data4 0x243DD4C8 // 177
data4 0x24457FEB // 178
data4 0x24DEEFBB // 179
data4 0x243C70AE // 180
data4 0x23E7A6FA // 181
data4 0x24C2D311 // 182
data4 0x23026255 // 183
data4 0x2437C9B9 // 184
data4 0x246BA847 // 185
data4 0x2420B448 // 186
data4 0x24C4CF5A // 187
data4 0x242C4981 // 188
data4 0x24DE1525 // 189
data4 0x24F5CC33 // 190
data4 0x235A85DA // 191
data4 0x24A0B64F // 192
data4 0x244BA0A4 // 193
data4 0x24AAF30A // 194
data4 0x244C86F9 // 195
data4 0x246D5B82 // 196
data4 0x24529347 // 197
data4 0x240DD008 // 198
data4 0x24E98790 // 199
data4 0x2489B0CE // 200
data4 0x22BC29AC // 201
data4 0x23F37C7A // 202
data4 0x24987FE8 // 203
data4 0x22AFE20B // 204
data4 0x24C8D7C2 // 205
data4 0x24B28B7D // 206
data4 0x23B6B271 // 207
data4 0x24C77CB6 // 208
data4 0x24EF1DCA // 209
data4 0x24A4F0AC // 210
data4 0x24CF113E // 211
data4 0x2496BBAB // 212
data4 0x23C7CC8A // 213
data4 0x23AE3961 // 214
data4 0x2410A895 // 215
data4 0x23CE3114 // 216
data4 0x2308247D // 217
data4 0x240045E9 // 218
data4 0x24974F60 // 219
data4 0x242CB39F // 220
data4 0x24AB8D69 // 221
data4 0x23436788 // 222
data4 0x24305E9E // 223
data4 0x243E71A9 // 224
data4 0x23C2A6B3 // 225
data4 0x23FFE6CF // 226
data4 0x2322D801 // 227
data4 0x24515F21 // 228
data4 0x2412A0D6 // 229
data4 0x24E60D44 // 230
data4 0x240D9251 // 231
data4 0x247076E2 // 232
data4 0x229B101B // 233
data4 0x247B12DE // 234
data4 0x244B9127 // 235
data4 0x2499EC42 // 236
data4 0x21FC3963 // 237
data4 0x23E53266 // 238
data4 0x24CE102D // 239
data4 0x23CC45D2 // 240
data4 0x2333171D // 241
data4 0x246B3533 // 242
data4 0x24931129 // 243
data4 0x24405FFA // 244
data4 0x24CF464D // 245
data4 0x237095CD // 246
data4 0x24F86CBD // 247
data4 0x24E2D84B // 248
data4 0x21ACBB44 // 249
data4 0x24F43A8C // 250
data4 0x249DB931 // 251
data4 0x24A385EF // 252
data4 0x238B1279 // 253
data4 0x2436213E // 254
data4 0x24F18A3B // 255
LOCAL_OBJECT_END(log_data)


LOCAL_OBJECT_START(log10_data)
// coefficients of polynoimal approximation
data8 0x3FC2494104381A8E // A7
data8 0xBFC5556D556BBB69 // A6
//
// two parts of ln(2)/ln(10)
data8 0x3FD3441350900000, 0x3DCEF3FDE623E256
//
data8 0xDE5BD8A937287195,0x3FFD // 1/ln(10)
//
data8 0x3FC999999988B5E9 // A5
data8 0xBFCFFFFFFFF6FFF5 // A4
//
// Hi parts of ln(1/frcpa(1+i/256))/ln(10), i=0...255
data8 0x3F4BD27045BFD024 // 0
data8 0x3F64E84E793A474A // 1
data8 0x3F7175085AB85FF0 // 2
data8 0x3F787CFF9D9147A5 // 3
data8 0x3F7EA9D372B89FC8 // 4
data8 0x3F82DF9D95DA961C // 5
data8 0x3F866DF172D6372B // 6
data8 0x3F898D79EF5EEDEF // 7
data8 0x3F8D22ADF3F9579C // 8
data8 0x3F9024231D30C398 // 9
data8 0x3F91F23A98897D49 // 10
data8 0x3F93881A7B818F9E // 11
data8 0x3F951F6E1E759E35 // 12
data8 0x3F96F2BCE7ADC5B4 // 13
data8 0x3F988D362CDF359E // 14
data8 0x3F9A292BAF010981 // 15
data8 0x3F9BC6A03117EB97 // 16
data8 0x3F9D65967DE3AB08 // 17
data8 0x3F9F061167FC31E7 // 18
data8 0x3FA05409E4F7819B // 19
data8 0x3FA125D0432EA20D // 20
data8 0x3FA1F85D440D299B // 21
data8 0x3FA2AD755749617C // 22
data8 0x3FA381772A00E603 // 23
data8 0x3FA45643E165A70A // 24
data8 0x3FA52BDD034475B8 // 25
data8 0x3FA5E3966B7E9295 // 26
data8 0x3FA6BAAF47C5B244 // 27
data8 0x3FA773B3E8C4F3C7 // 28
data8 0x3FA84C51EBEE8D15 // 29
data8 0x3FA906A6786FC1CA // 30
data8 0x3FA9C197ABF00DD6 // 31
data8 0x3FAA9C78712191F7 // 32
data8 0x3FAB58C09C8D637C // 33
data8 0x3FAC15A8BCDD7B7E // 34
data8 0x3FACD331E2C2967B // 35
data8 0x3FADB11ED766ABF4 // 36
data8 0x3FAE70089346A9E6 // 37
data8 0x3FAF2F96C6754AED // 38
data8 0x3FAFEFCA8D451FD5 // 39
data8 0x3FB0585283764177 // 40
data8 0x3FB0B913AAC7D3A6 // 41
data8 0x3FB11A294F2569F5 // 42
data8 0x3FB16B51A2696890 // 43
data8 0x3FB1CD03ADACC8BD // 44
data8 0x3FB22F0BDD7745F5 // 45
data8 0x3FB2916ACA38D1E7 // 46
data8 0x3FB2F4210DF7663C // 47
data8 0x3FB346A6C3C49065 // 48
data8 0x3FB3A9FEBC605409 // 49
data8 0x3FB3FD0C10A3AA54 // 50
data8 0x3FB46107D3540A81 // 51
data8 0x3FB4C55DD16967FE // 52
data8 0x3FB51940330C000A // 53
data8 0x3FB56D620EE7115E // 54
data8 0x3FB5D2ABCF26178D // 55
data8 0x3FB6275AA5DEBF81 // 56
data8 0x3FB68D4EAF26D7EE // 57
data8 0x3FB6E28C5C54A28D // 58
data8 0x3FB7380B9665B7C7 // 59
data8 0x3FB78DCCC278E85B // 60
data8 0x3FB7F50C2CF25579 // 61
data8 0x3FB84B5FD5EAEFD7 // 62
data8 0x3FB8A1F6BAB2B226 // 63
data8 0x3FB8F8D144557BDF // 64
data8 0x3FB94FEFDCD61D92 // 65
data8 0x3FB9A752EF316149 // 66
data8 0x3FB9FEFAE7611EDF // 67
data8 0x3FBA56E8325F5C86 // 68
data8 0x3FBAAF1B3E297BB3 // 69
data8 0x3FBB079479C372AC // 70
data8 0x3FBB6054553B12F7 // 71
data8 0x3FBBB95B41AB5CE5 // 72
data8 0x3FBC12A9B13FE079 // 73
data8 0x3FBC6C4017382BEA // 74
data8 0x3FBCB41FBA42686C // 75
data8 0x3FBD0E38CE73393E // 76
data8 0x3FBD689B2193F132 // 77
data8 0x3FBDC3472B1D285F // 78
data8 0x3FBE0C06300D528B // 79
data8 0x3FBE6738190E394B // 80
data8 0x3FBEC2B50D208D9A // 81
data8 0x3FBF0C1C2B936827 // 82
data8 0x3FBF68216C9CC726 // 83
data8 0x3FBFB1F6381856F3 // 84
data8 0x3FC00742AF4CE5F8 // 85
data8 0x3FC02C64906512D2 // 86
data8 0x3FC05AF1E63E03B4 // 87
data8 0x3FC0804BEA723AA8 // 88
data8 0x3FC0AF1FD6711526 // 89
data8 0x3FC0D4B2A88059FF // 90
data8 0x3FC0FA5EF136A06C // 91
data8 0x3FC1299A4FB3E305 // 92
data8 0x3FC14F806253C3EC // 93
data8 0x3FC175805D1587C1 // 94
data8 0x3FC19B9A637CA294 // 95
data8 0x3FC1CB5FC26EDE16 // 96
data8 0x3FC1F1B4E65F2590 // 97
data8 0x3FC218248B5DC3E5 // 98
data8 0x3FC23EAED62ADC76 // 99
data8 0x3FC26553EBD337BC // 100
data8 0x3FC28C13F1B118FF // 101
data8 0x3FC2BCAA14381385 // 102
data8 0x3FC2E3A740B7800E // 103
data8 0x3FC30ABFD8F333B6 // 104
data8 0x3FC331F403985096 // 105
data8 0x3FC35943E7A6068F // 106
data8 0x3FC380AFAC6E7C07 // 107
data8 0x3FC3A8377997B9E5 // 108
data8 0x3FC3CFDB771C9ADB // 109
data8 0x3FC3EDA90D39A5DE // 110
data8 0x3FC4157EC09505CC // 111
data8 0x3FC43D7113FB04C0 // 112
data8 0x3FC4658030AD1CCE // 113
data8 0x3FC48DAC404638F5 // 114
data8 0x3FC4B5F56CBBB869 // 115
data8 0x3FC4DE5BE05E7582 // 116
data8 0x3FC4FCBC0776FD85 // 117
data8 0x3FC525561E9256EE // 118
data8 0x3FC54E0DF3198865 // 119
data8 0x3FC56CAB7112BDE2 // 120
data8 0x3FC59597BA735B15 // 121
data8 0x3FC5BEA23A506FD9 // 122
data8 0x3FC5DD7E08DE382E // 123
data8 0x3FC606BDD3F92355 // 124
data8 0x3FC6301C518A501E // 125
data8 0x3FC64F3770618915 // 126
data8 0x3FC678CC14C1E2D7 // 127
data8 0x3FC6981005ED2947 // 128
data8 0x3FC6C1DB5F9BB335 // 129
data8 0x3FC6E1488ECD2880 // 130
data8 0x3FC70B4B2E7E41B8 // 131
data8 0x3FC72AE209146BF8 // 132
data8 0x3FC7551C81BD8DCF // 133
data8 0x3FC774DD76CC43BD // 134
data8 0x3FC79F505DB00E88 // 135
data8 0x3FC7BF3BDE099F30 // 136
data8 0x3FC7E9E7CAC437F8 // 137
data8 0x3FC809FE4902D00D // 138
data8 0x3FC82A2757995CBD // 139
data8 0x3FC85525C625E098 // 140
data8 0x3FC8757A79831887 // 141
data8 0x3FC895E2058D8E02 // 142
data8 0x3FC8C13437695531 // 143
data8 0x3FC8E1C812EF32BE // 144
data8 0x3FC9026F112197E8 // 145
data8 0x3FC923294888880A // 146
data8 0x3FC94EEA4B8334F2 // 147
data8 0x3FC96FD1B639FC09 // 148
data8 0x3FC990CCA66229AB // 149
data8 0x3FC9B1DB33334842 // 150
data8 0x3FC9D2FD740E6606 // 151
data8 0x3FC9FF49EEDCB553 // 152
data8 0x3FCA209A84FBCFF7 // 153
data8 0x3FCA41FF1E43F02B // 154
data8 0x3FCA6377D2CE9377 // 155
data8 0x3FCA8504BAE0D9F5 // 156
data8 0x3FCAA6A5EEEBEFE2 // 157
data8 0x3FCAC85B878D7878 // 158
data8 0x3FCAEA259D8FFA0B // 159
data8 0x3FCB0C0449EB4B6A // 160
data8 0x3FCB2DF7A5C50299 // 161
data8 0x3FCB4FFFCA70E4D1 // 162
data8 0x3FCB721CD17157E2 // 163
data8 0x3FCB944ED477D4EC // 164
data8 0x3FCBB695ED655C7C // 165
data8 0x3FCBD8F2364AEC0F // 166
data8 0x3FCBFB63C969F4FF // 167
data8 0x3FCC1DEAC134D4E9 // 168
data8 0x3FCC4087384F4F80 // 169
data8 0x3FCC6339498F09E1 // 170
data8 0x3FCC86010FFC076B // 171
data8 0x3FCC9D3D065C5B41 // 172
data8 0x3FCCC029375BA079 // 173
data8 0x3FCCE32B66978BA4 // 174
data8 0x3FCD0643AFD51404 // 175
data8 0x3FCD29722F0DEA45 // 176
data8 0x3FCD4CB70070FE43 // 177
data8 0x3FCD6446AB3F8C95 // 178
data8 0x3FCD87B0EF71DB44 // 179
data8 0x3FCDAB31D1FE99A6 // 180
data8 0x3FCDCEC96FDC888E // 181
data8 0x3FCDE69088763579 // 182
data8 0x3FCE0A4E4A25C1FF // 183
data8 0x3FCE2E2315755E32 // 184
data8 0x3FCE461322D1648A // 185
data8 0x3FCE6A0E95C7787B // 186
data8 0x3FCE8E216243DD60 // 187
data8 0x3FCEA63AF26E007C // 188
data8 0x3FCECA74ED15E0B7 // 189
data8 0x3FCEEEC692CCD259 // 190
data8 0x3FCF070A36B8D9C0 // 191
data8 0x3FCF2B8393E34A2D // 192
data8 0x3FCF5014EF538A5A // 193
data8 0x3FCF68833AF1B17F // 194
data8 0x3FCF8D3CD9F3F04E // 195
data8 0x3FCFA5C61ADD93E9 // 196
data8 0x3FCFCAA8567EBA79 // 197
data8 0x3FCFE34CC8743DD8 // 198
data8 0x3FD0042BFD74F519 // 199
data8 0x3FD016BDF6A18017 // 200
data8 0x3FD023262F907322 // 201
data8 0x3FD035CCED8D32A1 // 202
data8 0x3FD042430E869FFB // 203
data8 0x3FD04EBEC842B2DF // 204
data8 0x3FD06182E84FD4AB // 205
data8 0x3FD06E0CB609D383 // 206
data8 0x3FD080E60BEC8F12 // 207
data8 0x3FD08D7E0D894735 // 208
data8 0x3FD0A06CC96A2055 // 209
data8 0x3FD0AD131F3B3C55 // 210
data8 0x3FD0C01771E775FB // 211
data8 0x3FD0CCCC3CAD6F4B // 212
data8 0x3FD0D986D91A34A8 // 213
data8 0x3FD0ECA9B8861A2D // 214
data8 0x3FD0F972F87FF3D5 // 215
data8 0x3FD106421CF0E5F7 // 216
data8 0x3FD11983EBE28A9C // 217
data8 0x3FD12661E35B7859 // 218
data8 0x3FD13345D2779D3B // 219
data8 0x3FD146A6F597283A // 220
data8 0x3FD15399E81EA83D // 221
data8 0x3FD16092E5D3A9A6 // 222
data8 0x3FD17413C3B7AB5D // 223
data8 0x3FD1811BF629D6FA // 224
data8 0x3FD18E2A47B46685 // 225
data8 0x3FD19B3EBE1A4418 // 226
data8 0x3FD1AEE9017CB450 // 227
data8 0x3FD1BC0CED7134E1 // 228
data8 0x3FD1C93712ABC7FF // 229
data8 0x3FD1D66777147D3E // 230
data8 0x3FD1EA3BD1286E1C // 231
data8 0x3FD1F77BED932C4C // 232
data8 0x3FD204C25E1B031F // 233
data8 0x3FD2120F28CE69B1 // 234
data8 0x3FD21F6253C48D00 // 235
data8 0x3FD22CBBE51D60A9 // 236
data8 0x3FD240CE4C975444 // 237
data8 0x3FD24E37F8ECDAE7 // 238
data8 0x3FD25BA8215AF7FC // 239
data8 0x3FD2691ECC29F042 // 240
data8 0x3FD2769BFFAB2DFF // 241
data8 0x3FD2841FC23952C9 // 242
data8 0x3FD291AA1A384978 // 243
data8 0x3FD29F3B0E15584A // 244
data8 0x3FD2B3A0EE479DF7 // 245
data8 0x3FD2C142842C09E5 // 246
data8 0x3FD2CEEACCB7BD6C // 247
data8 0x3FD2DC99CE82FF20 // 248
data8 0x3FD2EA4F902FD7D9 // 249
data8 0x3FD2F80C186A25FC // 250
data8 0x3FD305CF6DE7B0F6 // 251
data8 0x3FD3139997683CE7 // 252
data8 0x3FD3216A9BB59E7C // 253
data8 0x3FD32F4281A3CEFE // 254
data8 0x3FD33D2150110091 // 255
//
// Lo parts of ln(1/frcpa(1+i/256))/ln(10), i=0...255
data4 0x1FB0EB5A // 0
data4 0x206E5EE3 // 1
data4 0x208F3609 // 2
data4 0x2070EB03 // 3
data4 0x1F314BAE // 4
data4 0x217A889D // 5
data4 0x21E63650 // 6
data4 0x21C2F4A3 // 7
data4 0x2192A10C // 8
data4 0x1F84B73E // 9
data4 0x2243FBCA // 10
data4 0x21BD9C51 // 11
data4 0x213C542B // 12
data4 0x21047386 // 13
data4 0x21217D8F // 14
data4 0x226791B7 // 15
data4 0x204CCE66 // 16
data4 0x2234CE9F // 17
data4 0x220675E2 // 18
data4 0x22B8E5BA // 19
data4 0x22C12D14 // 20
data4 0x211D41F0 // 21
data4 0x228507F3 // 22
data4 0x22F7274B // 23
data4 0x22A7FDD1 // 24
data4 0x2244A06E // 25
data4 0x215DCE69 // 26
data4 0x22F5C961 // 27
data4 0x22EBEF29 // 28
data4 0x222A2CB6 // 29
data4 0x22B9FE00 // 30
data4 0x22E79EB7 // 31
data4 0x222F9607 // 32
data4 0x2189D87F // 33
data4 0x2236DB45 // 34
data4 0x22ED77FB // 35
data4 0x21CB70F0 // 36
data4 0x21B8ACE8 // 37
data4 0x22EC58C1 // 38
data4 0x22CFCC1C // 39
data4 0x2343E77A // 40
data4 0x237FBC7F // 41
data4 0x230D472E // 42
data4 0x234686FB // 43
data4 0x23770425 // 44
data4 0x223977EC // 45
data4 0x2345800A // 46
data4 0x237BC351 // 47
data4 0x23191502 // 48
data4 0x232BAC12 // 49
data4 0x22692421 // 50
data4 0x234D409D // 51
data4 0x22EC3214 // 52
data4 0x2376C916 // 53
data4 0x22B00DD1 // 54
data4 0x2309D910 // 55
data4 0x22F925FD // 56
data4 0x22A63A7B // 57
data4 0x2106264A // 58
data4 0x234227F9 // 59
data4 0x1ECB1978 // 60
data4 0x23460A62 // 61
data4 0x232ED4B1 // 62
data4 0x226DDC38 // 63
data4 0x1F101A73 // 64
data4 0x21B1F82B // 65
data4 0x22752F19 // 66
data4 0x2320BC15 // 67
data4 0x236EEC5E // 68
data4 0x23404D3E // 69
data4 0x2304C517 // 70
data4 0x22F7441A // 71
data4 0x230D3D7A // 72
data4 0x2264A9DF // 73
data4 0x22410CC8 // 74
data4 0x2342CCCB // 75
data4 0x23560BD4 // 76
data4 0x237BBFFE // 77
data4 0x2373A206 // 78
data4 0x22C871B9 // 79
data4 0x2354B70C // 80
data4 0x232EDB33 // 81
data4 0x235DB680 // 82
data4 0x230EF422 // 83
data4 0x235316CA // 84
data4 0x22EEEE8B // 85
data4 0x2375C88C // 86
data4 0x235ABD21 // 87
data4 0x23A0D232 // 88
data4 0x23F5FFB5 // 89
data4 0x23D3CEC8 // 90
data4 0x22A92204 // 91
data4 0x238C64DF // 92
data4 0x23B82896 // 93
data4 0x22D633B8 // 94
data4 0x23861E93 // 95
data4 0x23CB594B // 96
data4 0x2330387E // 97
data4 0x21CD4702 // 98
data4 0x2284C505 // 99
data4 0x23D6995C // 100
data4 0x23F6C807 // 101
data4 0x239CEF5C // 102
data4 0x239442B0 // 103
data4 0x22B35EE5 // 104
data4 0x2391E9A4 // 105
data4 0x23A390F5 // 106
data4 0x2349AC9C // 107
data4 0x23FA5535 // 108
data4 0x21E3A46A // 109
data4 0x23B44ABA // 110
data4 0x23CEA8E0 // 111
data4 0x23F647DC // 112
data4 0x2390D1A8 // 113
data4 0x23D0CFA2 // 114
data4 0x236E0872 // 115
data4 0x23B88B91 // 116
data4 0x2283C359 // 117
data4 0x232F647F // 118
data4 0x23122CD7 // 119
data4 0x232CF564 // 120
data4 0x232630FD // 121
data4 0x23BEE1C8 // 122
data4 0x23B2BD30 // 123
data4 0x2301F1C0 // 124
data4 0x23CE4D67 // 125
data4 0x23A353C9 // 126
data4 0x238086E8 // 127
data4 0x22D0D29E // 128
data4 0x23A3B3C8 // 129
data4 0x23F69F4B // 130
data4 0x23EA3C21 // 131
data4 0x23951C88 // 132
data4 0x2372AFFC // 133
data4 0x23A6D1A8 // 134
data4 0x22BBBAF4 // 135
data4 0x227FA3DD // 136
data4 0x23804D9B // 137
data4 0x232D771F // 138
data4 0x239CB57B // 139
data4 0x2303CF34 // 140
data4 0x22218C2A // 141
data4 0x23991BEE // 142
data4 0x23EB3596 // 143
data4 0x230487FA // 144
data4 0x2135DF4C // 145
data4 0x2380FD2D // 146
data4 0x23EB75E9 // 147
data4 0x211C62C8 // 148
data4 0x23F518F1 // 149
data4 0x23FEF882 // 150
data4 0x239097C7 // 151
data4 0x223E2BDA // 152
data4 0x23988F89 // 153
data4 0x22E4A4AD // 154
data4 0x23F03D9C // 155
data4 0x23F5018F // 156
data4 0x23E1E250 // 157
data4 0x23FD3D90 // 158
data4 0x22DEE2FF // 159
data4 0x238342AB // 160
data4 0x22E6736F // 161
data4 0x233AFC28 // 162
data4 0x2395F661 // 163
data4 0x23D8B991 // 164
data4 0x23CD58D5 // 165
data4 0x21941FD6 // 166
data4 0x23352915 // 167
data4 0x235D09EE // 168
data4 0x22DC7EF9 // 169
data4 0x238BC9F3 // 170
data4 0x2397DF8F // 171
data4 0x2380A7BB // 172
data4 0x23EFF48C // 173
data4 0x21E67408 // 174
data4 0x236420F7 // 175
data4 0x22C8DFB5 // 176
data4 0x239B5D35 // 177
data4 0x23BDC09D // 178
data4 0x239E822C // 179
data4 0x23984F0A // 180
data4 0x23EF2119 // 181
data4 0x23F738B8 // 182
data4 0x23B66187 // 183
data4 0x23B06AD7 // 184
data4 0x2369140F // 185
data4 0x218DACE6 // 186
data4 0x21DF23F1 // 187
data4 0x235D8B34 // 188
data4 0x23460333 // 189
data4 0x23F11D62 // 190
data4 0x23C37147 // 191
data4 0x22B2AE2A // 192
data4 0x23949211 // 193
data4 0x23B69799 // 194
data4 0x23DBEC75 // 195
data4 0x229A6FB3 // 196
data4 0x23FC6C60 // 197
data4 0x22D01FFC // 198
data4 0x235985F0 // 199
data4 0x23F7ECA5 // 200
data4 0x23F924D3 // 201
data4 0x2381B92F // 202
data4 0x243A0FBE // 203
data4 0x24712D72 // 204
data4 0x24594E2F // 205
data4 0x220CD12A // 206
data4 0x23D87FB0 // 207
data4 0x2338288A // 208
data4 0x242BB2CC // 209
data4 0x220F6265 // 210
data4 0x23BB7FE3 // 211
data4 0x2301C0A2 // 212
data4 0x246709AB // 213
data4 0x23A619E2 // 214
data4 0x24030E3B // 215
data4 0x233C36CC // 216
data4 0x241AAB77 // 217
data4 0x243D41A3 // 218
data4 0x23834A60 // 219
data4 0x236AC7BF // 220
data4 0x23B6D597 // 221
data4 0x210E9474 // 222
data4 0x242156E6 // 223
data4 0x243A1D68 // 224
data4 0x2472187C // 225
data4 0x23834E86 // 226
data4 0x23CA0807 // 227
data4 0x24745887 // 228
data4 0x23E2B0E1 // 229
data4 0x2421EB67 // 230
data4 0x23DCC64E // 231
data4 0x22DF71D1 // 232
data4 0x238D5ECA // 233
data4 0x23CDE86F // 234
data4 0x24131F45 // 235
data4 0x240FE4E2 // 236
data4 0x2317731A // 237
data4 0x24015C76 // 238
data4 0x2301A4E8 // 239
data4 0x23E52A6D // 240
data4 0x247D8A0D // 241
data4 0x23DFEEBA // 242
data4 0x22139FEC // 243
data4 0x2454A112 // 244
data4 0x23C21E28 // 245
data4 0x2460D813 // 246
data4 0x24258924 // 247
data4 0x2425680F // 248
data4 0x24194D1E // 249
data4 0x24242C2F // 250
data4 0x243DDE5E // 251
data4 0x23DEB388 // 252
data4 0x23E0E6EB // 253
data4 0x24393E74 // 254
data4 0x241B1863 // 255
LOCAL_OBJECT_END(log10_data)



// Code
//==============================================================

// log   has p13 true, p14 false
// log10 has p14 true, p13 false

.section .text
GLOBAL_IEEE754_ENTRY(log10)
{ .mfi
      getf.exp      GR_Exp = f8 // if x is unorm then must recompute
      frcpa.s1      FR_RcpX,p0 = f1,f8
      mov           GR_05 = 0xFFFE // biased exponent of A2=0.5
}
{ .mlx
      addl          GR_ad_1 = @ltoff(log10_data),gp
      movl          GR_A3 = 0x3fd5555555555557 // double precision memory
                                               // representation of A3
};;

{ .mfi
      getf.sig      GR_Sig = f8 // get significand to calculate index
      fclass.m      p8,p0 = f8,9 // is x positive unorm?
      mov           GR_xorg = 0x3fefe // double precision memory msb of 255/256
}
{ .mib
      ld8           GR_ad_1 = [GR_ad_1]
      cmp.eq        p14,p13 = r0,r0 // set p14 to 1 for log10
      br.cond.sptk  log_log10_common
};;
GLOBAL_IEEE754_END(log10)
libm_alias_double_other (__log10, log10)


GLOBAL_IEEE754_ENTRY(log)
{ .mfi
      getf.exp      GR_Exp = f8 // if x is unorm then must recompute
      frcpa.s1      FR_RcpX,p0 = f1,f8
      mov           GR_05 = 0xfffe
}
{ .mlx
      addl          GR_ad_1 = @ltoff(log_data),gp
      movl          GR_A3 = 0x3fd5555555555557 // double precision memory
                                               // representation of A3
};;

{ .mfi
      getf.sig      GR_Sig = f8 // get significand to calculate index
      fclass.m      p8,p0 = f8,9 // is x positive unorm?
      mov           GR_xorg = 0x3fefe // double precision memory msb of 255/256
}
{ .mfi
      ld8           GR_ad_1 = [GR_ad_1]
      nop.f         0
      cmp.eq        p13,p14 = r0,r0 // set p13 to 1 for log
};;

log_log10_common:
{ .mfi
      getf.d        GR_x = f8 // double precision memory representation of x
      fclass.m      p9,p0 = f8,0x1E1 // is x NaN, NaT or +Inf?
      dep.z         GR_dx = 3, 44, 2 // Create 0x0000300000000000
                                     // Difference between double precision
                                     // memory representations of 257/256 and
                                     // 255/256
}
{ .mfi
      setf.exp      FR_A2 = GR_05 // create A2
      fnorm.s1      FR_NormX = f8
      mov           GR_bias = 0xffff
};;

{ .mfi
      setf.d        FR_A3 = GR_A3 // create A3
      fcmp.eq.s1    p12,p0 = f1,f8 // is x equal to 1.0?
      dep.z         GR_xorg = GR_xorg, 44, 19 // 0x3fefe00000000000
                                              // double precision memory
                                              // representation of 255/256
}
{ .mib
      add           GR_ad_2 = 0x30,GR_ad_1 // address of A5,A4
      add           GR_ad_3 = 0x840,GR_ad_1 // address of ln(1/frcpa) lo parts
(p8)  br.cond.spnt  log_positive_unorms
};;

log_core:
{ .mfi
      ldfpd         FR_A7,FR_A6 = [GR_ad_1],16
      fclass.m      p10,p0 = f8,0x3A // is x < 0?
      sub           GR_Nm1 = GR_Exp,GR_05 // unbiased_exponent_of_x - 1
}
{ .mfi
      ldfpd         FR_A5,FR_A4 = [GR_ad_2],16
(p9)  fma.d.s0      f8 = f8,f1,f0 // set V-flag
      sub           GR_N = GR_Exp,GR_bias // unbiased_exponent_of_x
};;

{ .mfi
      setf.sig      FR_N = GR_N // copy unbiased exponent of x to significand
      fms.s1        FR_r = FR_RcpX,f8,f1 // range reduction for |x-1|>=1/256
      extr.u        GR_Ind = GR_Sig,55,8 // get bits from 55 to 62 as index
}
{ .mib
      sub           GR_x = GR_x, GR_xorg // get diff between x and 255/256
      cmp.gtu       p6, p7 = 2, GR_Nm1 // p6 true if 0.5 <= x < 2
(p9)  br.ret.spnt   b0 // exit for NaN, NaT and +Inf
};;

{ .mfi
      ldfpd         FR_Ln2hi,FR_Ln2lo = [GR_ad_1],16
      fclass.m      p11,p0 = f8,0x07 // is x = 0?
      shladd        GR_ad_3 = GR_Ind,2,GR_ad_3 // address of Tlo
}
{ .mib
      shladd        GR_ad_2 = GR_Ind,3,GR_ad_2 // address of Thi
(p6)  cmp.leu       p6, p7 = GR_x, GR_dx       // 255/256 <= x <= 257/256
(p10) br.cond.spnt  log_negatives // jump if x is negative
};;

// p6 is true if |x-1| < 1/256
// p7 is true if |x-1| >= 1/256
{ .mfi
      ldfd          FR_Thi = [GR_ad_2]
(p6)  fms.s1        FR_r = f8,f1,f1 // range reduction for |x-1|<1/256
      nop.i         0
};;

{ .mmi
(p7)  ldfs          FR_Tlo = [GR_ad_3]
      nop.m         0
      nop.i         0
}
{ .mfb
      nop.m         0
(p12) fma.d.s0      f8 = f0,f0,f0
(p12) br.ret.spnt   b0 // exit for +1.0
};;

.pred.rel "mutex",p6,p7
{ .mfi
(p6)  mov           GR_NearOne = 1
      fms.s1        FR_A32 = FR_A3,FR_r,FR_A2 // A3*r-A2
(p7)  mov           GR_NearOne = 0
}
{ .mfb
      ldfe          FR_InvLn10 = [GR_ad_1],16
      fma.s1        FR_r2 = FR_r,FR_r,f0 // r^2
(p11) br.cond.spnt  log_zeroes // jump if x is zero
};;

{ .mfi
      nop.m         0
      fma.s1        FR_A6 = FR_A7,FR_r,FR_A6 // A7*r+A6
      nop.i         0
}
{ .mfi
(p7)  cmp.eq.unc    p9,p0 = r0,r0  // set p9 if |x-1| > 1/256
      fma.s1        FR_A4 = FR_A5,FR_r,FR_A4 // A5*r+A4
(p14) cmp.eq.unc    p8,p0 = 1,GR_NearOne // set p8 to 1 if it's log10
                                         // and argument near 1.0
};;

{ .mfi
(p6)  getf.exp      GR_rexp = FR_r  // Get signexp of x-1
(p7)  fcvt.xf       FR_N = FR_N
(p8)  cmp.eq        p9,p6 = r0,r0        // Also set p9 and clear p6 if log10
                                         // and arg near 1
};;

{ .mfi
      nop.m         0
      fma.s1        FR_r4 = FR_r2,FR_r2,f0 // r^4
      nop.i         0
}
{ .mfi
      nop.m         0
(p8)  fma.s1        FR_NxLn2pT = f0,f0,f0  // Clear NxLn2pT if log10 near 1
      nop.i         0
};;

{ .mfi
      nop.m         0
      // (A3*r+A2)*r^2+r
      fma.s1        FR_A321 = FR_A32,FR_r2,FR_r
      mov           GR_mask = 0x1ffff
}
{ .mfi
      nop.m         0
      // (A7*r+A6)*r^2+(A5*r+A4)
      fma.s1        FR_A4 = FR_A6,FR_r2,FR_A4
      nop.i         0
};;

{ .mfi
(p6)  and           GR_rexp = GR_rexp, GR_mask
      // N*Ln2hi+Thi
(p7)  fma.s1        FR_NxLn2hipThi = FR_N,FR_Ln2hi,FR_Thi
      nop.i         0
}
{ .mfi
      nop.m         0
      // N*Ln2lo+Tlo
(p7)  fma.s1        FR_NxLn2lopTlo = FR_N,FR_Ln2lo,FR_Tlo
      nop.i         0
};;

{ .mfi
(p6)  sub           GR_rexp = GR_rexp, GR_bias // unbiased exponent of x-1
(p9)  fma.s1        f8 = FR_A4,FR_r4,FR_A321 // P(r) if |x-1| >= 1/256 or
                                             // log10 and |x-1| < 1/256
      nop.i         0
}
{ .mfi
      nop.m         0
      // (N*Ln2hi+Thi) + (N*Ln2lo+Tlo)
(p7)  fma.s1        FR_NxLn2pT = FR_NxLn2hipThi,f1,FR_NxLn2lopTlo
      nop.i         0
};;

{ .mfi
(p6)  cmp.gt.unc    p10, p6 = -40, GR_rexp // Test |x-1| < 2^-40
      nop.f         0
      nop.i         0
};;

{ .mfi
      nop.m         0
(p10) fma.d.s0      f8 = FR_A32,FR_r2,FR_r // log(x) if |x-1| < 2^-40
      nop.i         0
};;

.pred.rel "mutex",p6,p9
{ .mfi
      nop.m         0
(p6)  fma.d.s0      f8 = FR_A4,FR_r4,FR_A321 // log(x) if 2^-40 <= |x-1| < 1/256
      nop.i         0
}
{ .mfb
      nop.m         0
(p9)  fma.d.s0      f8 = f8,FR_InvLn10,FR_NxLn2pT // result if |x-1| >= 1/256
                                                  // or log10 and |x-1| < 1/256
      br.ret.sptk   b0
};;

.align 32
log_positive_unorms:
{ .mmf
      getf.exp      GR_Exp = FR_NormX // recompute biased exponent
      getf.d        GR_x = FR_NormX   // recompute double precision x
      fcmp.eq.s1    p12,p0 = f1,FR_NormX // is x equal to 1.0?
};;

{ .mfb
      getf.sig      GR_Sig = FR_NormX // recompute significand
      fcmp.eq.s0    p15, p0 = f8, f0  // set denormal flag
      br.cond.sptk  log_core
};;

.align 32
log_zeroes:
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8 // keep input argument for subsequent
                                 // call of __libm_error_support#
      nop.i         0
}
{ .mfi
      nop.m         0
      fms.s1        FR_tmp = f0,f0,f1 // -1.0
      nop.i         0
};;

.pred.rel "mutex",p13,p14
{ .mfi
(p13) mov           GR_TAG = 2 // set libm error in case of log
      frcpa.s0      f8,p0 = FR_tmp,f0 // log(+/-0) should be equal to -INF.
                                      // We can get it using frcpa because it
                                      // sets result to the IEEE-754 mandated
                                      // quotient of FR_tmp/f0.
                                      // As far as FR_tmp is -1 it'll be -INF
      nop.i         0
}
{ .mib
(p14) mov           GR_TAG = 8 // set libm error in case of log10
      nop.i         0
      br.cond.sptk  log_libm_err
};;

.align 32
log_negatives:
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8
      nop.i         0
};;

.pred.rel "mutex",p13,p14
{ .mfi
(p13) mov           GR_TAG = 3 // set libm error in case of log
      frcpa.s0      f8,p0 = f0,f0 // log(negatives) should be equal to NaN.
                                  // We can get it using frcpa because it
                                  // sets result to the IEEE-754 mandated
                                  // quotient of f0/f0 i.e. NaN.
(p14) mov           GR_TAG = 9 // set libm error in case of log10
};;

.align 32
log_libm_err:
{ .mmi
      alloc         r32 = ar.pfs,1,4,4,0
      mov           GR_Parameter_TAG = GR_TAG
      nop.i         0
};;
GLOBAL_IEEE754_END(log)
libm_alias_double_other (__log, log)
#ifdef SHARED
.symver log,log@@GLIBC_2.29
.weak __log_compat
.set __log_compat,__log
.symver __log_compat,log@GLIBC_2.2
#endif


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
        add   GR_Parameter_Y = -32,sp         // Parameter 2 value
        nop.f 0
.save   ar.pfs,GR_SAVE_PFS
        mov  GR_SAVE_PFS = ar.pfs             // Save ar.pfs
}
{ .mfi
.fframe 64
        add sp = -64,sp                       // Create new stack
        nop.f 0
        mov GR_SAVE_GP = gp                   // Save gp
};;

{ .mmi
        stfd [GR_Parameter_Y] = FR_Y,16       // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp            // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0 = b0                   // Save b0
};;

.body
{ .mib
        stfd [GR_Parameter_X] = FR_X          // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT     // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support# // Call error handling function
};;

{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;

{ .mmi
        ldfd  f8 = [GR_Parameter_RESULT]      // Get return result off stack
.restore sp
        add   sp = 64,sp                      // Restore stack pointer
        mov   b0 = GR_SAVE_B0                 // Restore return address
};;

{ .mib
        mov   gp = GR_SAVE_GP                 // Restore gp
        mov   ar.pfs = GR_SAVE_PFS            // Restore ar.pfs
        br.ret.sptk     b0                    // Return
};;
LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#
