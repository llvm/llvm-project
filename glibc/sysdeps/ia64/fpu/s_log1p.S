.file "log1p.s"


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
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 06/29/01 Improved speed of all paths
// 05/20/02 Cleaned up namespace and sf0 syntax
// 10/02/02 Improved performance by basing on log algorithm
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 04/18/03 Eliminate possible WAW dependency warning
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// double log1p(double)
//
// log1p(x) = log(x+1)
//
// Overview of operation
//==============================================================
// Background
// ----------
//
// This algorithm is based on fact that
// log1p(x) = log(1+x) and
// log(a b) = log(a) + log(b).
// In our case we have 1+x = 2^N f, where 1 <= f < 2.
// So
//   log(1+x) = log(2^N f) = log(2^N) + log(f) = n*log(2) + log(f)
//
// To calculate log(f) we do following
//   log(f) = log(f * frcpa(f) / frcpa(f)) =
//          = log(f * frcpa(f)) + log(1/frcpa(f))
//
// According to definition of IA-64's frcpa instruction it's a
// floating point that approximates 1/f using a lookup on the
// top of 8 bits of the input number's + 1 significand with relative
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
// Finally we have that  log(1+x) ~ (N*log(2) + T) + P(r)
//
// Note that if input argument is close to 0.0 (in our case it means
// that |x| < 1/256) we can use just polynomial approximation
// because 1+x = 2^0 * f = f = 1 + r and
// log(1+x) = log(1 + r) ~ P(r)
//
//
// Implementation
// --------------
//
// 1. |x| >= 2^(-8), and x > -1
//   InvX = frcpa(x+1)
//   r = InvX*(x+1) - 1
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
//   L1 (log(2)) is calculated in quad
//   precision and represented by two floating-point 64-bit numbers L1hi,L1lo
//   stored in memory.
//
//   And final result = ((L1hi*N + Thi) + (N*L1lo + Tlo)) + P(r)
//
//
// 2. 2^(-80) <= |x| < 2^(-8)
//   r = x
//   P(r) = r*((r*A3 - A2) + r^4*((A4 + r*A5) + r^2*(A6 + r*A7)),
//   A7,A6,A5,A4,A3,A2 are the same as in case |x| >= 1/256
//
//   And final results
//     log(1+x)   = P(r)
//
// 3. 0 < |x| < 2^(-80)
//   Although log1p(x) is basically x, we would like to preserve the inexactness
//   nature as well as consistent behavior under different rounding modes.
//   We can do this by computing the result as
//
//     log1p(x) = x - x*x
//
//
//    Note: NaT, any NaNs, +/-INF, +/-0, negatives and unnormalized numbers are
//          filtered and processed on special branches.
//

//
// Special values
//==============================================================
//
// log1p(-1)    = -inf            // Call error support
//
// log1p(+qnan) = +qnan
// log1p(-qnan) = -qnan
// log1p(+snan) = +qnan
// log1p(-snan) = -qnan
//
// log1p(x),x<-1= QNAN Indefinite // Call error support
// log1p(-inf)  = QNAN Indefinite
// log1p(+inf)  = +inf
// log1p(+/-0)  = +/-0
//
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f7 -> f15,  f32 -> f40
//
// General registers used:
// r8  -> r11
// r14 -> r20
//
// Predicate registers used:
// p6 -> p12

// Assembly macros
//==============================================================
GR_TAG                 = r8
GR_ad_1                = r8
GR_ad_2                = r9
GR_Exp                 = r10
GR_N                   = r11

GR_signexp_x           = r14
GR_exp_mask            = r15
GR_exp_bias            = r16
GR_05                  = r17
GR_A3                  = r18
GR_Sig                 = r19
GR_Ind                 = r19
GR_exp_x               = r20


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

FR_Xp1                 = f40


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
// two parts of ln(2)
data8 0x3FE62E42FEF00000,0x3DD473DE6AF278ED
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


// Code
//==============================================================

.section .text
GLOBAL_IEEE754_ENTRY(log1p)
{ .mfi
      getf.exp      GR_signexp_x = f8 // if x is unorm then must recompute
      fadd.s1       FR_Xp1 = f8, f1       // Form 1+x
      mov           GR_05 = 0xfffe
}
{ .mlx
      addl          GR_ad_1 = @ltoff(log_data),gp
      movl          GR_A3 = 0x3fd5555555555557 // double precision memory
                                               // representation of A3
}
;;

{ .mfi
      ld8           GR_ad_1 = [GR_ad_1]
      fclass.m      p8,p0 = f8,0xb // Is x unorm?
      mov           GR_exp_mask = 0x1ffff
}
{ .mfi
      nop.m         0
      fnorm.s1      FR_NormX = f8              // Normalize x
      mov           GR_exp_bias = 0xffff
}
;;

{ .mfi
      setf.exp      FR_A2 = GR_05 // create A2 = 0.5
      fclass.m      p9,p0 = f8,0x1E1 // is x NaN, NaT or +Inf?
      nop.i         0
}
{ .mib
      setf.d        FR_A3 = GR_A3 // create A3
      add           GR_ad_2 = 16,GR_ad_1 // address of A5,A4
(p8)  br.cond.spnt  log1p_unorm          // Branch if x=unorm
}
;;

log1p_common:
{ .mfi
      nop.m         0
      frcpa.s1      FR_RcpX,p0 = f1,FR_Xp1
      nop.i         0
}
{ .mfb
      nop.m         0
(p9)  fma.d.s0      f8 = f8,f1,f0 // set V-flag
(p9)  br.ret.spnt   b0 // exit for NaN, NaT and +Inf
}
;;

{ .mfi
      getf.exp      GR_Exp = FR_Xp1            // signexp of x+1
      fclass.m      p10,p0 = FR_Xp1,0x3A // is 1+x < 0?
      and           GR_exp_x = GR_exp_mask, GR_signexp_x // biased exponent of x
}
{ .mfi
      ldfpd         FR_A7,FR_A6 = [GR_ad_1]
      nop.f         0
      nop.i         0
}
;;

{ .mfi
      getf.sig      GR_Sig = FR_Xp1 // get significand to calculate index
                                    // for Thi,Tlo if |x| >= 2^-8
      fcmp.eq.s1    p12,p0 = f8,f0     // is x equal to 0?
      sub           GR_exp_x = GR_exp_x, GR_exp_bias // true exponent of x
}
;;

{ .mfi
      sub           GR_N = GR_Exp,GR_exp_bias // true exponent of x+1
      fcmp.eq.s1    p11,p0 = FR_Xp1,f0     // is x = -1?
      cmp.gt        p6,p7 = -8, GR_exp_x  // Is |x| < 2^-8
}
{ .mfb
      ldfpd         FR_A5,FR_A4 = [GR_ad_2],16
      nop.f         0
(p10) br.cond.spnt  log1p_lt_minus_1   // jump if x < -1
}
;;

// p6 is true if |x| < 1/256
// p7 is true if |x| >= 1/256
.pred.rel "mutex",p6,p7
{ .mfi
(p7)  add           GR_ad_1 = 0x820,GR_ad_1 // address of log(2) parts
(p6)  fms.s1        FR_r = f8,f1,f0 // range reduction for |x|<1/256
(p6)  cmp.gt.unc    p10,p0 = -80, GR_exp_x  // Is |x| < 2^-80
}
{ .mfb
(p7)  setf.sig      FR_N = GR_N // copy unbiased exponent of x to the
                                // significand field of FR_N
(p7)  fms.s1        FR_r = FR_RcpX,FR_Xp1,f1 // range reduction for |x|>=1/256
(p12) br.ret.spnt   b0 // exit for x=0, return x
}
;;

{ .mib
(p7)  ldfpd         FR_Ln2hi,FR_Ln2lo = [GR_ad_1],16
(p7)  extr.u        GR_Ind = GR_Sig,55,8 // get bits from 55 to 62 as index
(p11) br.cond.spnt  log1p_eq_minus_1 // jump if x = -1
}
;;

{ .mmf
(p7)  shladd        GR_ad_2 = GR_Ind,3,GR_ad_2 // address of Thi
(p7)  shladd        GR_ad_1 = GR_Ind,2,GR_ad_1 // address of Tlo
(p10) fnma.d.s0     f8 = f8,f8,f8   // If |x| very small, result=x-x*x
}
;;

{ .mmb
(p7)  ldfd          FR_Thi = [GR_ad_2]
(p7)  ldfs          FR_Tlo = [GR_ad_1]
(p10) br.ret.spnt   b0                   // Exit if |x| < 2^(-80)
}
;;

{ .mfi
      nop.m         0
      fma.s1        FR_r2 = FR_r,FR_r,f0 // r^2
      nop.i         0
}
{ .mfi
      nop.m         0
      fms.s1        FR_A2 = FR_A3,FR_r,FR_A2 // A3*r+A2
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        FR_A6 = FR_A7,FR_r,FR_A6 // A7*r+A6
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_A4 = FR_A5,FR_r,FR_A4 // A5*r+A4
      nop.i         0
}
;;

{ .mfi
      nop.m         0
(p7)  fcvt.xf       FR_N = FR_N
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        FR_r4 = FR_r2,FR_r2,f0 // r^4
      nop.i         0
}
{ .mfi
      nop.m         0
      // (A3*r+A2)*r^2+r
      fma.s1        FR_A2 = FR_A2,FR_r2,FR_r
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // (A7*r+A6)*r^2+(A5*r+A4)
      fma.s1        FR_A4 = FR_A6,FR_r2,FR_A4
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // N*Ln2hi+Thi
(p7)  fma.s1        FR_NxLn2hipThi = FR_N,FR_Ln2hi,FR_Thi
      nop.i         0
}
{ .mfi
      nop.m         0
      // N*Ln2lo+Tlo
(p7)  fma.s1        FR_NxLn2lopTlo = FR_N,FR_Ln2lo,FR_Tlo
      nop.i         0
}
;;

{ .mfi
      nop.m         0
(p7)  fma.s1        f8 = FR_A4,FR_r4,FR_A2 // P(r) if |x| >= 1/256
      nop.i         0
}
{ .mfi
      nop.m         0
      // (N*Ln2hi+Thi) + (N*Ln2lo+Tlo)
(p7)  fma.s1        FR_NxLn2pT = FR_NxLn2hipThi,f1,FR_NxLn2lopTlo
      nop.i         0
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fma.d.s0      f8 = FR_A4,FR_r4,FR_A2 // result if 2^(-80) <= |x| < 1/256
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fma.d.s0      f8 = f8,f1,FR_NxLn2pT  // result if |x| >= 1/256
      br.ret.sptk   b0                     // Exit if |x| >= 2^(-80)
}
;;

.align 32
log1p_unorm:
// Here if x=unorm
{ .mfb
      getf.exp      GR_signexp_x = FR_NormX // recompute biased exponent
      nop.f         0
      br.cond.sptk  log1p_common
}
;;

.align 32
log1p_eq_minus_1:
// Here if x=-1
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8 // keep input argument for subsequent
                                 // call of __libm_error_support#
      nop.i         0
}
;;

{ .mfi
      mov           GR_TAG = 140  // set libm error in case of log1p(-1).
      frcpa.s0      f8,p0 = f8,f0 // log1p(-1) should be equal to -INF.
                                      // We can get it using frcpa because it
                                      // sets result to the IEEE-754 mandated
                                      // quotient of f8/f0.
      nop.i         0
}
{ .mib
      nop.m         0
      nop.i         0
      br.cond.sptk  log_libm_err
}
;;

.align 32
log1p_lt_minus_1:
// Here if x < -1
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8
      nop.i         0
}
;;

{ .mfi
      mov           GR_TAG = 141  // set libm error in case of x < -1.
      frcpa.s0      f8,p0 = f0,f0 // log1p(x) x < -1 should be equal to NaN.
                                  // We can get it using frcpa because it
                                  // sets result to the IEEE-754 mandated
                                  // quotient of f0/f0 i.e. NaN.
      nop.i         0
}
;;

.align 32
log_libm_err:
{ .mmi
      alloc         r32 = ar.pfs,1,4,4,0
      mov           GR_Parameter_TAG = GR_TAG
      nop.i         0
}
;;

GLOBAL_IEEE754_END(log1p)
libm_alias_double_other (__log1p, log1p)


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
