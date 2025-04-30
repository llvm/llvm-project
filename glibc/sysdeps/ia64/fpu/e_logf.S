.file "logf.s"


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
// 03/01/00 Initial version
// 08/15/00 Bundle added after call to __libm_error_support to properly
//          set [the previously overwritten] GR_Parameter_RESULT.
// 01/10/01 Improved speed, fixed flags for neg denormals
// 05/20/02 Cleaned up namespace and sf0 syntax
// 05/23/02 Modified algorithm. Now only one polynomial is used
//          for |x-1| >= 1/256 and for |x-1| < 1/256
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 03/31/05 Reformatted delimiters between data tables
//
// API
//==============================================================
// float logf(float)
// float log10f(float)
//
//
// Overview of operation
//==============================================================
// Background
// ----------
//
// This algorithm is based on fact that
// log(a b) = log(a) + log(b).
//
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
// To compute log10(x) we just use identity:
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
// setf instruction can be used.
//
// Note that both log and log10 use common approximation polynomial
// it means we need only one set of coefficients of approximation.
//
// 1. Computation of log(x) for |x-1| >= 1/256
//   InvX = frcpa(x)
//   r = InvX*x - 1
//   P(r) = r*((1 - A2*r) + r^2*(A3 - A4*r)) = r*P2(r),
//   A4,A3,A2 are created with setf inctruction.
//   We use Taylor series and so A4 = 1/4, A3 = 1/3,
//   A2 = 1/2 rounded to double.
//
//   N = float(n) where n is true unbiased exponent of x
//
//   T is tabular value of log(1/frcpa(x)) calculated in quad precision
//   and rounded to double. To T we get bits from 55 to 62 of register
//   format significand of x and calculate address
//     ad_T = table_base_addr + 8 * index
//
//   L2 (1.0 or 1.0/log(10) depending on function) is calculated in quad
//   precision and rounded to double; it's loaded from memory
//
//   L1 (log(2) or log10(2) depending on function) is calculated in quad
//   precision and rounded to double; it's created with setf.
//
//   And final result = P2(r)*(r*L2) + (T + N*L1)
//
//
// 2. Computation of log(x) for |x-1| < 1/256
//   r = x - 1
//   P(r) = r*((1 - A2*r) + r^2*(A3 - A4*r)) = r*P2(r),
//   A4,A3,A2 are the same as in case |x-1| >= 1/256
//
//   And final result = P2(r)*(r*L2)
//
// 3. How we define is input argument such that |x-1| < 1/256 or not.
//
//    To do it we analyze biased exponent and significand of input argument.
//
//      a) First we test is biased exponent equal to 0xFFFE or 0xFFFF (i.e.
//         we test is 0.5 <= x < 2). This comparison can be performed using
//         unsigned version of cmp instruction in such a way
//         biased_exponent_of_x - 0xFFFE < 2
//
//
//      b) Second (in case when result of a) is true) we need to compare x
//         with 1-1/256 and 1+1/256 or in register format representation with
//         0xFFFEFF00000000000000 and 0xFFFF8080000000000000 correspondingly.
//         As far as biased exponent of x here can be equal only to 0xFFFE or
//         0xFFFF we need to test only last bit of it. Also signifigand always
//         has implicit bit set to 1 that can be exluded from comparison.
//         Thus it's quite enough to generate 64-bit integer bits of that are
//         ix[63] = biased_exponent_of_x[0] and ix[62-0] = significand_of_x[62-0]
//         and compare it with 0x7F00000000000000 and 0x80800000000000000 (those
//         obtained like ix from register representatinos of 255/256 and
//         257/256). This comparison can be made like in a), using unsigned
//         version of cmp i.e. ix - 0x7F00000000000000 < 0x0180000000000000.
//         0x0180000000000000 is difference between 0x80800000000000000 and
//         0x7F00000000000000.
//
//    Note: NaT, any NaNs, +/-INF, +/-0, negatives and unnormalized numbers are
//          filtered and processed on special branches.
//
//
// Special values
//==============================================================
//
// logf(+0)    = -inf
// logf(-0)    = -inf
//
// logf(+qnan) = +qnan
// logf(-qnan) = -qnan
// logf(+snan) = +qnan
// logf(-snan) = -qnan
//
// logf(-n)    = QNAN Indefinite
// logf(-inf)  = QNAN Indefinite
//
// logf(+inf)  = +inf
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f12 -> f14,  f33 -> f39
//
// General registers used:
// r8  -> r11
// r14 -> r19
//
// Predicate registers used:
// p6 -> p12


// Assembly macros
//==============================================================

GR_TAG                 = r8
GR_ad_T                = r8
GR_N                   = r9
GR_Exp                 = r10
GR_Sig                 = r11

GR_025                 = r14
GR_05                  = r15
GR_A3                  = r16
GR_Ind                 = r17
GR_dx                  = r15
GR_Ln2                 = r19
GR_de                  = r20
GR_x                   = r21
GR_xorg                = r22

GR_SAVE_B0             = r33
GR_SAVE_PFS            = r34
GR_SAVE_GP             = r35
GR_SAVE_SP             = r36

GR_Parameter_X         = r37
GR_Parameter_Y         = r38
GR_Parameter_RESULT    = r39
GR_Parameter_TAG       = r40


FR_A2                  = f12
FR_A3                  = f13
FR_A4                  = f14

FR_RcpX                = f33
FR_r                   = f34
FR_r2                  = f35
FR_tmp                 = f35
FR_Ln2                 = f36
FR_T                   = f37
FR_N                   = f38
FR_NxLn2pT             = f38
FR_NormX               = f39
FR_InvLn10             = f40


FR_Y                   = f1
FR_X                   = f10
FR_RESULT              = f8


// Data tables
//==============================================================
RODATA
.align 16
LOCAL_OBJECT_START(logf_data)
data8 0x3FF0000000000000 // 1.0
//
// ln(1/frcpa(1+i/256)), i=0...255
data8 0x3F60040155D5889E // 0
data8 0x3F78121214586B54 // 1
data8 0x3F841929F96832F0 // 2
data8 0x3F8C317384C75F06 // 3
data8 0x3F91A6B91AC73386 // 4
data8 0x3F95BA9A5D9AC039 // 5
data8 0x3F99D2A8074325F4 // 6
data8 0x3F9D6B2725979802 // 7
data8 0x3FA0C58FA19DFAAA // 8
data8 0x3FA2954C78CBCE1B // 9
data8 0x3FA4A94D2DA96C56 // 10
data8 0x3FA67C94F2D4BB58 // 11
data8 0x3FA85188B630F068 // 12
data8 0x3FAA6B8ABE73AF4C // 13
data8 0x3FAC441E06F72A9E // 14
data8 0x3FAE1E6713606D07 // 15
data8 0x3FAFFA6911AB9301 // 16
data8 0x3FB0EC139C5DA601 // 17
data8 0x3FB1DBD2643D190B // 18
data8 0x3FB2CC7284FE5F1C // 19
data8 0x3FB3BDF5A7D1EE64 // 20
data8 0x3FB4B05D7AA012E0 // 21
data8 0x3FB580DB7CEB5702 // 22
data8 0x3FB674F089365A7A // 23
data8 0x3FB769EF2C6B568D // 24
data8 0x3FB85FD927506A48 // 25
data8 0x3FB9335E5D594989 // 26
data8 0x3FBA2B0220C8E5F5 // 27
data8 0x3FBB0004AC1A86AC // 28
data8 0x3FBBF968769FCA11 // 29
data8 0x3FBCCFEDBFEE13A8 // 30
data8 0x3FBDA727638446A2 // 31
data8 0x3FBEA3257FE10F7A // 32
data8 0x3FBF7BE9FEDBFDE6 // 33
data8 0x3FC02AB352FF25F4 // 34
data8 0x3FC097CE579D204D // 35
data8 0x3FC1178E8227E47C // 36
data8 0x3FC185747DBECF34 // 37
data8 0x3FC1F3B925F25D41 // 38
data8 0x3FC2625D1E6DDF57 // 39
data8 0x3FC2D1610C86813A // 40
data8 0x3FC340C59741142E // 41
data8 0x3FC3B08B6757F2A9 // 42
data8 0x3FC40DFB08378003 // 43
data8 0x3FC47E74E8CA5F7C // 44
data8 0x3FC4EF51F6466DE4 // 45
data8 0x3FC56092E02BA516 // 46
data8 0x3FC5D23857CD74D5 // 47
data8 0x3FC6313A37335D76 // 48
data8 0x3FC6A399DABBD383 // 49
data8 0x3FC70337DD3CE41B // 50
data8 0x3FC77654128F6127 // 51
data8 0x3FC7E9D82A0B022D // 52
data8 0x3FC84A6B759F512F // 53
data8 0x3FC8AB47D5F5A310 // 54
data8 0x3FC91FE49096581B // 55
data8 0x3FC981634011AA75 // 56
data8 0x3FC9F6C407089664 // 57
data8 0x3FCA58E729348F43 // 58
data8 0x3FCABB55C31693AD // 59
data8 0x3FCB1E104919EFD0 // 60
data8 0x3FCB94EE93E367CB // 61
data8 0x3FCBF851C067555F // 62
data8 0x3FCC5C0254BF23A6 // 63
data8 0x3FCCC000C9DB3C52 // 64
data8 0x3FCD244D99C85674 // 65
data8 0x3FCD88E93FB2F450 // 66
data8 0x3FCDEDD437EAEF01 // 67
data8 0x3FCE530EFFE71012 // 68
data8 0x3FCEB89A1648B971 // 69
data8 0x3FCF1E75FADF9BDE // 70
data8 0x3FCF84A32EAD7C35 // 71
data8 0x3FCFEB2233EA07CD // 72
data8 0x3FD028F9C7035C1C // 73
data8 0x3FD05C8BE0D9635A // 74
data8 0x3FD085EB8F8AE797 // 75
data8 0x3FD0B9C8E32D1911 // 76
data8 0x3FD0EDD060B78081 // 77
data8 0x3FD122024CF0063F // 78
data8 0x3FD14BE2927AECD4 // 79
data8 0x3FD180618EF18ADF // 80
data8 0x3FD1B50BBE2FC63B // 81
data8 0x3FD1DF4CC7CF242D // 82
data8 0x3FD214456D0EB8D4 // 83
data8 0x3FD23EC5991EBA49 // 84
data8 0x3FD2740D9F870AFB // 85
data8 0x3FD29ECDABCDFA04 // 86
data8 0x3FD2D46602ADCCEE // 87
data8 0x3FD2FF66B04EA9D4 // 88
data8 0x3FD335504B355A37 // 89
data8 0x3FD360925EC44F5D // 90
data8 0x3FD38BF1C3337E75 // 91
data8 0x3FD3C25277333184 // 92
data8 0x3FD3EDF463C1683E // 93
data8 0x3FD419B423D5E8C7 // 94
data8 0x3FD44591E0539F49 // 95
data8 0x3FD47C9175B6F0AD // 96
data8 0x3FD4A8B341552B09 // 97
data8 0x3FD4D4F3908901A0 // 98
data8 0x3FD501528DA1F968 // 99
data8 0x3FD52DD06347D4F6 // 100
data8 0x3FD55A6D3C7B8A8A // 101
data8 0x3FD5925D2B112A59 // 102
data8 0x3FD5BF406B543DB2 // 103
data8 0x3FD5EC433D5C35AE // 104
data8 0x3FD61965CDB02C1F // 105
data8 0x3FD646A84935B2A2 // 106
data8 0x3FD6740ADD31DE94 // 107
data8 0x3FD6A18DB74A58C5 // 108
data8 0x3FD6CF31058670EC // 109
data8 0x3FD6F180E852F0BA // 110
data8 0x3FD71F5D71B894F0 // 111
data8 0x3FD74D5AEFD66D5C // 112
data8 0x3FD77B79922BD37E // 113
data8 0x3FD7A9B9889F19E2 // 114
data8 0x3FD7D81B037EB6A6 // 115
data8 0x3FD8069E33827231 // 116
data8 0x3FD82996D3EF8BCB // 117
data8 0x3FD85855776DCBFB // 118
data8 0x3FD8873658327CCF // 119
data8 0x3FD8AA75973AB8CF // 120
data8 0x3FD8D992DC8824E5 // 121
data8 0x3FD908D2EA7D9512 // 122
data8 0x3FD92C59E79C0E56 // 123
data8 0x3FD95BD750EE3ED3 // 124
data8 0x3FD98B7811A3EE5B // 125
data8 0x3FD9AF47F33D406C // 126
data8 0x3FD9DF270C1914A8 // 127
data8 0x3FDA0325ED14FDA4 // 128
data8 0x3FDA33440224FA79 // 129
data8 0x3FDA57725E80C383 // 130
data8 0x3FDA87D0165DD199 // 131
data8 0x3FDAAC2E6C03F896 // 132
data8 0x3FDADCCC6FDF6A81 // 133
data8 0x3FDB015B3EB1E790 // 134
data8 0x3FDB323A3A635948 // 135
data8 0x3FDB56FA04462909 // 136
data8 0x3FDB881AA659BC93 // 137
data8 0x3FDBAD0BEF3DB165 // 138
data8 0x3FDBD21297781C2F // 139
data8 0x3FDC039236F08819 // 140
data8 0x3FDC28CB1E4D32FD // 141
data8 0x3FDC4E19B84723C2 // 142
data8 0x3FDC7FF9C74554C9 // 143
data8 0x3FDCA57B64E9DB05 // 144
data8 0x3FDCCB130A5CEBB0 // 145
data8 0x3FDCF0C0D18F326F // 146
data8 0x3FDD232075B5A201 // 147
data8 0x3FDD490246DEFA6B // 148
data8 0x3FDD6EFA918D25CD // 149
data8 0x3FDD9509707AE52F // 150
data8 0x3FDDBB2EFE92C554 // 151
data8 0x3FDDEE2F3445E4AF // 152
data8 0x3FDE148A1A2726CE // 153
data8 0x3FDE3AFC0A49FF40 // 154
data8 0x3FDE6185206D516E // 155
data8 0x3FDE882578823D52 // 156
data8 0x3FDEAEDD2EAC990C // 157
data8 0x3FDED5AC5F436BE3 // 158
data8 0x3FDEFC9326D16AB9 // 159
data8 0x3FDF2391A2157600 // 160
data8 0x3FDF4AA7EE03192D // 161
data8 0x3FDF71D627C30BB0 // 162
data8 0x3FDF991C6CB3B379 // 163
data8 0x3FDFC07ADA69A910 // 164
data8 0x3FDFE7F18EB03D3E // 165
data8 0x3FE007C053C5002E // 166
data8 0x3FE01B942198A5A1 // 167
data8 0x3FE02F74400C64EB // 168
data8 0x3FE04360BE7603AD // 169
data8 0x3FE05759AC47FE34 // 170
data8 0x3FE06B5F1911CF52 // 171
data8 0x3FE078BF0533C568 // 172
data8 0x3FE08CD9687E7B0E // 173
data8 0x3FE0A10074CF9019 // 174
data8 0x3FE0B5343A234477 // 175
data8 0x3FE0C974C89431CE // 176
data8 0x3FE0DDC2305B9886 // 177
data8 0x3FE0EB524BAFC918 // 178
data8 0x3FE0FFB54213A476 // 179
data8 0x3FE114253DA97D9F // 180
data8 0x3FE128A24F1D9AFF // 181
data8 0x3FE1365252BF0865 // 182
data8 0x3FE14AE558B4A92D // 183
data8 0x3FE15F85A19C765B // 184
data8 0x3FE16D4D38C119FA // 185
data8 0x3FE18203C20DD133 // 186
data8 0x3FE196C7BC4B1F3B // 187
data8 0x3FE1A4A738B7A33C // 188
data8 0x3FE1B981C0C9653D // 189
data8 0x3FE1CE69E8BB106B // 190
data8 0x3FE1DC619DE06944 // 191
data8 0x3FE1F160A2AD0DA4 // 192
data8 0x3FE2066D7740737E // 193
data8 0x3FE2147DBA47A394 // 194
data8 0x3FE229A1BC5EBAC3 // 195
data8 0x3FE237C1841A502E // 196
data8 0x3FE24CFCE6F80D9A // 197
data8 0x3FE25B2C55CD5762 // 198
data8 0x3FE2707F4D5F7C41 // 199
data8 0x3FE285E0842CA384 // 200
data8 0x3FE294294708B773 // 201
data8 0x3FE2A9A2670AFF0C // 202
data8 0x3FE2B7FB2C8D1CC1 // 203
data8 0x3FE2C65A6395F5F5 // 204
data8 0x3FE2DBF557B0DF43 // 205
data8 0x3FE2EA64C3F97655 // 206
data8 0x3FE3001823684D73 // 207
data8 0x3FE30E97E9A8B5CD // 208
data8 0x3FE32463EBDD34EA // 209
data8 0x3FE332F4314AD796 // 210
data8 0x3FE348D90E7464D0 // 211
data8 0x3FE35779F8C43D6E // 212
data8 0x3FE36621961A6A99 // 213
data8 0x3FE37C299F3C366A // 214
data8 0x3FE38AE2171976E7 // 215
data8 0x3FE399A157A603E7 // 216
data8 0x3FE3AFCCFE77B9D1 // 217
data8 0x3FE3BE9D503533B5 // 218
data8 0x3FE3CD7480B4A8A3 // 219
data8 0x3FE3E3C43918F76C // 220
data8 0x3FE3F2ACB27ED6C7 // 221
data8 0x3FE4019C2125CA93 // 222
data8 0x3FE4181061389722 // 223
data8 0x3FE42711518DF545 // 224
data8 0x3FE436194E12B6BF // 225
data8 0x3FE445285D68EA69 // 226
data8 0x3FE45BCC464C893A // 227
data8 0x3FE46AED21F117FC // 228
data8 0x3FE47A1527E8A2D3 // 229
data8 0x3FE489445EFFFCCC // 230
data8 0x3FE4A018BCB69835 // 231
data8 0x3FE4AF5A0C9D65D7 // 232
data8 0x3FE4BEA2A5BDBE87 // 233
data8 0x3FE4CDF28F10AC46 // 234
data8 0x3FE4DD49CF994058 // 235
data8 0x3FE4ECA86E64A684 // 236
data8 0x3FE503C43CD8EB68 // 237
data8 0x3FE513356667FC57 // 238
data8 0x3FE522AE0738A3D8 // 239
data8 0x3FE5322E26867857 // 240
data8 0x3FE541B5CB979809 // 241
data8 0x3FE55144FDBCBD62 // 242
data8 0x3FE560DBC45153C7 // 243
data8 0x3FE5707A26BB8C66 // 244
data8 0x3FE587F60ED5B900 // 245
data8 0x3FE597A7977C8F31 // 246
data8 0x3FE5A760D634BB8B // 247
data8 0x3FE5B721D295F10F // 248
data8 0x3FE5C6EA94431EF9 // 249
data8 0x3FE5D6BB22EA86F6 // 250
data8 0x3FE5E6938645D390 // 251
data8 0x3FE5F673C61A2ED2 // 252
data8 0x3FE6065BEA385926 // 253
data8 0x3FE6164BFA7CC06B // 254
data8 0x3FE62643FECF9743 // 255
LOCAL_OBJECT_END(logf_data)

LOCAL_OBJECT_START(log10f_data)
data8 0x3FDBCB7B1526E50E // 1/ln(10)
//
// ln(1/frcpa(1+i/256))/ln(10), i=0...255
data8 0x3F4BD27045BFD025 // 0
data8 0x3F64E84E793A474A // 1
data8 0x3F7175085AB85FF0 // 2
data8 0x3F787CFF9D9147A5 // 3
data8 0x3F7EA9D372B89FC8 // 4
data8 0x3F82DF9D95DA961C // 5
data8 0x3F866DF172D6372C // 6
data8 0x3F898D79EF5EEDF0 // 7
data8 0x3F8D22ADF3F9579D // 8
data8 0x3F9024231D30C398 // 9
data8 0x3F91F23A98897D4A // 10
data8 0x3F93881A7B818F9E // 11
data8 0x3F951F6E1E759E35 // 12
data8 0x3F96F2BCE7ADC5B4 // 13
data8 0x3F988D362CDF359E // 14
data8 0x3F9A292BAF010982 // 15
data8 0x3F9BC6A03117EB97 // 16
data8 0x3F9D65967DE3AB09 // 17
data8 0x3F9F061167FC31E8 // 18
data8 0x3FA05409E4F7819C // 19
data8 0x3FA125D0432EA20E // 20
data8 0x3FA1F85D440D299B // 21
data8 0x3FA2AD755749617D // 22
data8 0x3FA381772A00E604 // 23
data8 0x3FA45643E165A70B // 24
data8 0x3FA52BDD034475B8 // 25
data8 0x3FA5E3966B7E9295 // 26
data8 0x3FA6BAAF47C5B245 // 27
data8 0x3FA773B3E8C4F3C8 // 28
data8 0x3FA84C51EBEE8D15 // 29
data8 0x3FA906A6786FC1CB // 30
data8 0x3FA9C197ABF00DD7 // 31
data8 0x3FAA9C78712191F7 // 32
data8 0x3FAB58C09C8D637C // 33
data8 0x3FAC15A8BCDD7B7E // 34
data8 0x3FACD331E2C2967C // 35
data8 0x3FADB11ED766ABF4 // 36
data8 0x3FAE70089346A9E6 // 37
data8 0x3FAF2F96C6754AEE // 38
data8 0x3FAFEFCA8D451FD6 // 39
data8 0x3FB0585283764178 // 40
data8 0x3FB0B913AAC7D3A7 // 41
data8 0x3FB11A294F2569F6 // 42
data8 0x3FB16B51A2696891 // 43
data8 0x3FB1CD03ADACC8BE // 44
data8 0x3FB22F0BDD7745F5 // 45
data8 0x3FB2916ACA38D1E8 // 46
data8 0x3FB2F4210DF7663D // 47
data8 0x3FB346A6C3C49066 // 48
data8 0x3FB3A9FEBC60540A // 49
data8 0x3FB3FD0C10A3AA54 // 50
data8 0x3FB46107D3540A82 // 51
data8 0x3FB4C55DD16967FE // 52
data8 0x3FB51940330C000B // 53
data8 0x3FB56D620EE7115E // 54
data8 0x3FB5D2ABCF26178E // 55
data8 0x3FB6275AA5DEBF81 // 56
data8 0x3FB68D4EAF26D7EE // 57
data8 0x3FB6E28C5C54A28D // 58
data8 0x3FB7380B9665B7C8 // 59
data8 0x3FB78DCCC278E85B // 60
data8 0x3FB7F50C2CF2557A // 61
data8 0x3FB84B5FD5EAEFD8 // 62
data8 0x3FB8A1F6BAB2B226 // 63
data8 0x3FB8F8D144557BDF // 64
data8 0x3FB94FEFDCD61D92 // 65
data8 0x3FB9A752EF316149 // 66
data8 0x3FB9FEFAE7611EE0 // 67
data8 0x3FBA56E8325F5C87 // 68
data8 0x3FBAAF1B3E297BB4 // 69
data8 0x3FBB079479C372AD // 70
data8 0x3FBB6054553B12F7 // 71
data8 0x3FBBB95B41AB5CE6 // 72
data8 0x3FBC12A9B13FE079 // 73
data8 0x3FBC6C4017382BEA // 74
data8 0x3FBCB41FBA42686D // 75
data8 0x3FBD0E38CE73393F // 76
data8 0x3FBD689B2193F133 // 77
data8 0x3FBDC3472B1D2860 // 78
data8 0x3FBE0C06300D528B // 79
data8 0x3FBE6738190E394C // 80
data8 0x3FBEC2B50D208D9B // 81
data8 0x3FBF0C1C2B936828 // 82
data8 0x3FBF68216C9CC727 // 83
data8 0x3FBFB1F6381856F4 // 84
data8 0x3FC00742AF4CE5F8 // 85
data8 0x3FC02C64906512D2 // 86
data8 0x3FC05AF1E63E03B4 // 87
data8 0x3FC0804BEA723AA9 // 88
data8 0x3FC0AF1FD6711527 // 89
data8 0x3FC0D4B2A8805A00 // 90
data8 0x3FC0FA5EF136A06C // 91
data8 0x3FC1299A4FB3E306 // 92
data8 0x3FC14F806253C3ED // 93
data8 0x3FC175805D1587C1 // 94
data8 0x3FC19B9A637CA295 // 95
data8 0x3FC1CB5FC26EDE17 // 96
data8 0x3FC1F1B4E65F2590 // 97
data8 0x3FC218248B5DC3E5 // 98
data8 0x3FC23EAED62ADC76 // 99
data8 0x3FC26553EBD337BD // 100
data8 0x3FC28C13F1B11900 // 101
data8 0x3FC2BCAA14381386 // 102
data8 0x3FC2E3A740B7800F // 103
data8 0x3FC30ABFD8F333B6 // 104
data8 0x3FC331F403985097 // 105
data8 0x3FC35943E7A60690 // 106
data8 0x3FC380AFAC6E7C07 // 107
data8 0x3FC3A8377997B9E6 // 108
data8 0x3FC3CFDB771C9ADB // 109
data8 0x3FC3EDA90D39A5DF // 110
data8 0x3FC4157EC09505CD // 111
data8 0x3FC43D7113FB04C1 // 112
data8 0x3FC4658030AD1CCF // 113
data8 0x3FC48DAC404638F6 // 114
data8 0x3FC4B5F56CBBB869 // 115
data8 0x3FC4DE5BE05E7583 // 116
data8 0x3FC4FCBC0776FD85 // 117
data8 0x3FC525561E9256EE // 118
data8 0x3FC54E0DF3198865 // 119
data8 0x3FC56CAB7112BDE2 // 120
data8 0x3FC59597BA735B15 // 121
data8 0x3FC5BEA23A506FDA // 122
data8 0x3FC5DD7E08DE382F // 123
data8 0x3FC606BDD3F92355 // 124
data8 0x3FC6301C518A501F // 125
data8 0x3FC64F3770618916 // 126
data8 0x3FC678CC14C1E2D8 // 127
data8 0x3FC6981005ED2947 // 128
data8 0x3FC6C1DB5F9BB336 // 129
data8 0x3FC6E1488ECD2881 // 130
data8 0x3FC70B4B2E7E41B9 // 131
data8 0x3FC72AE209146BF9 // 132
data8 0x3FC7551C81BD8DCF // 133
data8 0x3FC774DD76CC43BE // 134
data8 0x3FC79F505DB00E88 // 135
data8 0x3FC7BF3BDE099F30 // 136
data8 0x3FC7E9E7CAC437F9 // 137
data8 0x3FC809FE4902D00D // 138
data8 0x3FC82A2757995CBE // 139
data8 0x3FC85525C625E098 // 140
data8 0x3FC8757A79831887 // 141
data8 0x3FC895E2058D8E03 // 142
data8 0x3FC8C13437695532 // 143
data8 0x3FC8E1C812EF32BE // 144
data8 0x3FC9026F112197E8 // 145
data8 0x3FC923294888880B // 146
data8 0x3FC94EEA4B8334F3 // 147
data8 0x3FC96FD1B639FC09 // 148
data8 0x3FC990CCA66229AC // 149
data8 0x3FC9B1DB33334843 // 150
data8 0x3FC9D2FD740E6607 // 151
data8 0x3FC9FF49EEDCB553 // 152
data8 0x3FCA209A84FBCFF8 // 153
data8 0x3FCA41FF1E43F02B // 154
data8 0x3FCA6377D2CE9378 // 155
data8 0x3FCA8504BAE0D9F6 // 156
data8 0x3FCAA6A5EEEBEFE3 // 157
data8 0x3FCAC85B878D7879 // 158
data8 0x3FCAEA259D8FFA0B // 159
data8 0x3FCB0C0449EB4B6B // 160
data8 0x3FCB2DF7A5C50299 // 161
data8 0x3FCB4FFFCA70E4D1 // 162
data8 0x3FCB721CD17157E3 // 163
data8 0x3FCB944ED477D4ED // 164
data8 0x3FCBB695ED655C7D // 165
data8 0x3FCBD8F2364AEC0F // 166
data8 0x3FCBFB63C969F4FF // 167
data8 0x3FCC1DEAC134D4E9 // 168
data8 0x3FCC4087384F4F80 // 169
data8 0x3FCC6339498F09E2 // 170
data8 0x3FCC86010FFC076C // 171
data8 0x3FCC9D3D065C5B42 // 172
data8 0x3FCCC029375BA07A // 173
data8 0x3FCCE32B66978BA4 // 174
data8 0x3FCD0643AFD51404 // 175
data8 0x3FCD29722F0DEA45 // 176
data8 0x3FCD4CB70070FE44 // 177
data8 0x3FCD6446AB3F8C96 // 178
data8 0x3FCD87B0EF71DB45 // 179
data8 0x3FCDAB31D1FE99A7 // 180
data8 0x3FCDCEC96FDC888F // 181
data8 0x3FCDE6908876357A // 182
data8 0x3FCE0A4E4A25C200 // 183
data8 0x3FCE2E2315755E33 // 184
data8 0x3FCE461322D1648A // 185
data8 0x3FCE6A0E95C7787B // 186
data8 0x3FCE8E216243DD60 // 187
data8 0x3FCEA63AF26E007C // 188
data8 0x3FCECA74ED15E0B7 // 189
data8 0x3FCEEEC692CCD25A // 190
data8 0x3FCF070A36B8D9C1 // 191
data8 0x3FCF2B8393E34A2D // 192
data8 0x3FCF5014EF538A5B // 193
data8 0x3FCF68833AF1B180 // 194
data8 0x3FCF8D3CD9F3F04F // 195
data8 0x3FCFA5C61ADD93E9 // 196
data8 0x3FCFCAA8567EBA7A // 197
data8 0x3FCFE34CC8743DD8 // 198
data8 0x3FD0042BFD74F519 // 199
data8 0x3FD016BDF6A18017 // 200
data8 0x3FD023262F907322 // 201
data8 0x3FD035CCED8D32A1 // 202
data8 0x3FD042430E869FFC // 203
data8 0x3FD04EBEC842B2E0 // 204
data8 0x3FD06182E84FD4AC // 205
data8 0x3FD06E0CB609D383 // 206
data8 0x3FD080E60BEC8F12 // 207
data8 0x3FD08D7E0D894735 // 208
data8 0x3FD0A06CC96A2056 // 209
data8 0x3FD0AD131F3B3C55 // 210
data8 0x3FD0C01771E775FB // 211
data8 0x3FD0CCCC3CAD6F4B // 212
data8 0x3FD0D986D91A34A9 // 213
data8 0x3FD0ECA9B8861A2D // 214
data8 0x3FD0F972F87FF3D6 // 215
data8 0x3FD106421CF0E5F7 // 216
data8 0x3FD11983EBE28A9D // 217
data8 0x3FD12661E35B785A // 218
data8 0x3FD13345D2779D3B // 219
data8 0x3FD146A6F597283A // 220
data8 0x3FD15399E81EA83D // 221
data8 0x3FD16092E5D3A9A6 // 222
data8 0x3FD17413C3B7AB5E // 223
data8 0x3FD1811BF629D6FB // 224
data8 0x3FD18E2A47B46686 // 225
data8 0x3FD19B3EBE1A4418 // 226
data8 0x3FD1AEE9017CB450 // 227
data8 0x3FD1BC0CED7134E2 // 228
data8 0x3FD1C93712ABC7FF // 229
data8 0x3FD1D66777147D3F // 230
data8 0x3FD1EA3BD1286E1C // 231
data8 0x3FD1F77BED932C4C // 232
data8 0x3FD204C25E1B031F // 233
data8 0x3FD2120F28CE69B1 // 234
data8 0x3FD21F6253C48D01 // 235
data8 0x3FD22CBBE51D60AA // 236
data8 0x3FD240CE4C975444 // 237
data8 0x3FD24E37F8ECDAE8 // 238
data8 0x3FD25BA8215AF7FC // 239
data8 0x3FD2691ECC29F042 // 240
data8 0x3FD2769BFFAB2E00 // 241
data8 0x3FD2841FC23952C9 // 242
data8 0x3FD291AA1A384978 // 243
data8 0x3FD29F3B0E15584B // 244
data8 0x3FD2B3A0EE479DF7 // 245
data8 0x3FD2C142842C09E6 // 246
data8 0x3FD2CEEACCB7BD6D // 247
data8 0x3FD2DC99CE82FF21 // 248
data8 0x3FD2EA4F902FD7DA // 249
data8 0x3FD2F80C186A25FD // 250
data8 0x3FD305CF6DE7B0F7 // 251
data8 0x3FD3139997683CE7 // 252
data8 0x3FD3216A9BB59E7C // 253
data8 0x3FD32F4281A3CEFF // 254
data8 0x3FD33D2150110092 // 255
LOCAL_OBJECT_END(log10f_data)


// Code
//==============================================================
.section .text

// logf   has p13 true, p14 false
// log10f has p14 true, p13 false

GLOBAL_IEEE754_ENTRY(log10f)
{ .mfi
      getf.exp      GR_Exp = f8 // if x is unorm then must recompute
      frcpa.s1      FR_RcpX,p0 = f1,f8
      mov           GR_05 = 0xFFFE // biased exponent of A2=0.5
}
{ .mlx
      addl          GR_ad_T = @ltoff(log10f_data),gp
      movl          GR_A3 = 0x3FD5555555555555 // double precision memory
                                               // representation of A3
};;
{ .mfi
      getf.sig      GR_Sig = f8 // if x is unorm then must recompute
      fclass.m      p8,p0 = f8,9 // is x positive unorm?
      sub           GR_025 = GR_05,r0,1 // biased exponent of A4=0.25
}
{ .mlx
      ld8           GR_ad_T = [GR_ad_T]
      movl          GR_Ln2 = 0x3FD34413509F79FF // double precision memory
                                                // representation of
                                                // log(2)/ln(10)
};;
{ .mfi
      setf.d        FR_A3 = GR_A3 // create A3
      fcmp.eq.s1    p14,p13 = f0,f0 // set p14 to 1 for log10f
      dep.z         GR_xorg = GR_05,55,8 // 0x7F00000000000000 integer number
                                         // bits of that are
                                         // GR_xorg[63]   = last bit of biased
                                         //            exponent of 255/256
                                         // GR_xorg[62-0] = bits from 62 to 0
                                         //            of significand of 255/256
}
{ .mib
      setf.exp      FR_A2 = GR_05 // create A2
      sub           GR_de = GR_Exp,GR_05 // biased_exponent_of_x - 0xFFFE
                                         // needed for comparison with 0.5 and 2.0
      br.cond.sptk  logf_log10f_common
};;
GLOBAL_IEEE754_END(log10f)
libm_alias_float_other (__log10, log10)

GLOBAL_IEEE754_ENTRY(logf)
{ .mfi
      getf.exp      GR_Exp = f8 // if x is unorm then must recompute
      frcpa.s1      FR_RcpX,p0 = f1,f8
      mov           GR_05 = 0xFFFE // biased exponent of A2=-0.5
}
{ .mlx
      addl          GR_ad_T = @ltoff(logf_data),gp
      movl          GR_A3 = 0x3FD5555555555555 // double precision memory
                                               // representation of A3
};;
{ .mfi
      getf.sig      GR_Sig = f8 // if x is unorm then must recompute
      fclass.m      p8,p0 = f8,9 // is x positive unorm?
      dep.z         GR_xorg = GR_05,55,8 // 0x7F00000000000000 integer number
                                         // bits of that are
                                         // GR_xorg[63]   = last bit of biased
                                         //            exponent of 255/256
                                         // GR_xorg[62-0] = bits from 62 to 0
                                         //            of significand of 255/256
}
{ .mfi
      ld8           GR_ad_T = [GR_ad_T]
      nop.f         0
      sub           GR_025 = GR_05,r0,1 // biased exponent of A4=0.25
};;
{ .mfi
      setf.d        FR_A3 = GR_A3 // create A3
      fcmp.eq.s1    p13,p14 = f0,f0 // p13 - true for logf
      sub           GR_de = GR_Exp,GR_05 // biased_exponent_of_x - 0xFFFE
                                         // needed for comparison with 0.5 and 2.0
}
{ .mlx
      setf.exp      FR_A2 = GR_05 // create A2
      movl          GR_Ln2 = 0x3FE62E42FEFA39EF // double precision memory
                                                // representation of log(2)
};;
logf_log10f_common:
{ .mfi
      setf.exp      FR_A4 = GR_025 // create A4=0.25
      fclass.m      p9,p0 = f8,0x3A // is x < 0 (including negateve unnormals)?
      dep           GR_x = GR_Exp,GR_Sig,63,1 // produce integer that bits are
                                              // GR_x[63] = GR_Exp[0]
                                              // GR_x[62-0] = GR_Sig[62-0]
}
{ .mib
      sub           GR_N = GR_Exp,GR_05,1 // unbiased exponent of x
      cmp.gtu       p6,p7 = 2,GR_de // is 0.5 <= x < 2.0?
(p8)  br.cond.spnt  logf_positive_unorm
};;
logf_core:
{ .mfi
      setf.sig      FR_N = GR_N // copy unbiased exponent of x to the
                                // significand field of FR_N
      fclass.m      p10,p0 = f8,0x1E1 // is x NaN, NaT or +Inf?
      dep.z         GR_dx = GR_05,54,3 // 0x0180000000000000 - difference
                                       // between our integer representations
                                       // of 257/256 and 255/256
}
{ .mfi
      nop.m         0
      nop.f         0
      sub           GR_x = GR_x,GR_xorg // difference between representations
                                        // of x and 255/256
};;
{ .mfi
      ldfd          FR_InvLn10 = [GR_ad_T],8
      fcmp.eq.s1    p11,p0 = f8,f1 // is x equal to 1.0?
      extr.u        GR_Ind = GR_Sig,55,8 // get bits from 55 to 62 as index
}
{ .mib
      setf.d        FR_Ln2 = GR_Ln2 // create log(2) or log10(2)
(p6)  cmp.gtu       p6,p7 = GR_dx,GR_x // set p6 if 255/256 <= x < 257/256
(p9)  br.cond.spnt  logf_negatives // jump if input argument is negative number
};;
// p6 is true if |x-1| < 1/256
// p7 is true if |x-1| >= 1/256
.pred.rel "mutex",p6,p7
{ .mfi
      shladd        GR_ad_T = GR_Ind,3,GR_ad_T // calculate address of T
(p7)  fms.s1        FR_r = FR_RcpX,f8,f1 // range reduction for |x-1|>=1/256
      extr.u        GR_Exp = GR_Exp,0,17 // exponent without sign
}
{ .mfb
      nop.m         0
(p6)  fms.s1        FR_r = f8,f1,f1 // range reduction for |x-1|<1/256
(p10) br.cond.spnt  logf_nan_nat_pinf // exit for NaN, NaT or +Inf
};;
{ .mfb
      ldfd          FR_T = [GR_ad_T] // load T
(p11) fma.s.s0      f8 = f0,f0,f0
(p11) br.ret.spnt   b0 // exit for x = 1.0
};;
{ .mib
      nop.m         0
      cmp.eq        p12,p0 = r0,GR_Exp // is x +/-0? (here it's quite enough
                                       // only to compare exponent with 0
                                       // because all unnormals already
                                       // have been filtered)
(p12) br.cond.spnt  logf_zeroes        // Branch if input argument is +/-0
};;
{ .mfi
      nop.m         0
      fnma.s1       FR_A2 = FR_A2,FR_r,f1 // A2*r+1
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_r2 = FR_r,FR_r,f0  // r^2
      nop.i         0
};;
{ .mfi
      nop.m         0
      fcvt.xf       FR_N = FR_N // convert integer N in significand of FR_N
                                // to floating-point representation
      nop.i         0
}
{ .mfi
      nop.m         0
      fnma.s1       FR_A3 = FR_A4,FR_r,FR_A3 // A4*r+A3
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_r = FR_r,FR_InvLn10,f0 // For log10f we have r/log(10)
      nop.i         0
}
{ .mfi
      nop.m         0
      nop.f         0
      nop.i         0
};;
{ .mfi
      nop.m         0
      fma.s1        FR_A2 = FR_A3,FR_r2,FR_A2 // (A4*r+A3)*r^2+(A2*r+1)
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        FR_NxLn2pT = FR_N,FR_Ln2,FR_T // N*Ln2+T
      nop.i         0
};;
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p7)  fma.s.s0      f8 = FR_A2,FR_r,FR_NxLn2pT // result for |x-1|>=1/256
      nop.i         0
}
{ .mfb
      nop.m         0
(p6)  fma.s.s0      f8 = FR_A2,FR_r,f0 // result for |x-1|<1/256
      br.ret.sptk   b0
};;

.align 32
logf_positive_unorm:
{ .mfi
      nop.m         0
(p8)  fma.s0        f8 = f8,f1,f0 // Normalize & set D-flag
      nop.i         0
};;
{ .mfi
      getf.exp      GR_Exp = f8    // recompute biased exponent
      nop.f         0
      cmp.ne        p6,p7 = r0,r0  // p6 <- 0, p7 <- 1 because
                                   // in case of unorm we are out
                                   // interval [255/256; 257/256]
};;
{ .mfi
      getf.sig      GR_Sig = f8 // recompute significand
      nop.f         0
      nop.i         0
};;
{ .mib
      sub           GR_N = GR_Exp,GR_05,1 // unbiased exponent N
      nop.i         0
      br.cond.sptk  logf_core // return into main path
};;

.align 32
logf_nan_nat_pinf:
{ .mfi
      nop.m         0
      fma.s.s0      f8 = f8,f1,f0 // set V-flag
      nop.i         0
}
{ .mfb
      nop.m         0
      nop.f         0
      br.ret.sptk   b0 // exit for NaN, NaT or +Inf
};;

.align 32
logf_zeroes:
{ .mfi
      nop.m         0
      fmerge.s      FR_X = f8,f8 // keep input argument for subsequent
                                 // call of __libm_error_support#
      nop.i         0
}
{ .mfi
(p13) mov           GR_TAG = 4 // set libm error in case of logf
      fms.s1        FR_tmp = f0,f0,f1 // -1.0
      nop.i         0
};;
{ .mfi
      nop.m         0
      frcpa.s0      f8,p0 = FR_tmp,f0 // log(+/-0) should be equal to -INF.
                                      // We can get it using frcpa because it
                                      // sets result to the IEEE-754 mandated
                                      // quotient of FR_tmp/f0.
                                      // As far as FR_tmp is -1 it'll be -INF
      nop.i         0
}
{ .mib
(p14) mov           GR_TAG = 10 // set libm error in case of log10f
      nop.i         0
      br.cond.sptk  logf_libm_err
};;

.align 32
logf_negatives:
{ .mfi
(p13) mov           GR_TAG = 5 // set libm error in case of logf
      fmerge.s      FR_X = f8,f8 // keep input argument for subsequent
                                 // call of __libm_error_support#
      nop.i         0
};;
{ .mfi
(p14) mov           GR_TAG = 11 // set libm error in case of log10f
      frcpa.s0      f8,p0 = f0,f0 // log(negatives) should be equal to NaN.
                                  // We can get it using frcpa because it
                                  // sets result to the IEEE-754 mandated
                                  // quotient of f0/f0 i.e. NaN.
      nop.i         0
};;

.align 32
logf_libm_err:
{ .mmi
      alloc         r32 = ar.pfs,1,4,4,0
      mov           GR_Parameter_TAG = GR_TAG
      nop.i         0
};;
GLOBAL_IEEE754_END(logf)
libm_alias_float_other (__log, log)
#ifdef SHARED
.symver logf,logf@@GLIBC_2.27
.weak __logf_compat
.set __logf_compat,__logf
.symver __logf_compat,logf@GLIBC_2.2
#endif


// Stack operations when calling error support.
//       (1)               (2)                          (3) (call)              (4)
//   sp   -> +          psp -> +                     psp -> +                   sp -> +
//           |                 |                            |                         |
//           |                 | <- GR_Y               R3 ->| <- GR_RESULT            | -> f8
//           |                 |                            |                         |
//           | <-GR_Y      Y2->|                       Y2 ->| <- GR_Y                 |
//           |                 |                            |                         |
//           |                 | <- GR_X               X1 ->|                         |
//           |                 |                            |                         |
//  sp-64 -> +          sp ->  +                     sp ->  +                         +
//    save ar.pfs          save b0                                               restore gp
//    save gp                                                                    restore ar.pfs

LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{ .mfi
      add   GR_Parameter_Y=-32,sp             // Parameter 2 value
      nop.f 0
.save ar.pfs,GR_SAVE_PFS
      mov  GR_SAVE_PFS=ar.pfs                 // Save ar.pfs
}
{ .mfi
.fframe 64
      add sp=-64,sp                           // Create new stack
      nop.f 0
      mov GR_SAVE_GP=gp                       // Save gp
};;
{ .mmi
      stfs [GR_Parameter_Y] = FR_Y,16         // STORE Parameter 2 on stack
      add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
      stfs [GR_Parameter_X] = FR_X                  // STORE Parameter 1 on stack
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
      nop.b 0
}
{ .mib
      stfs [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3 on stack
      add   GR_Parameter_Y = -16,GR_Parameter_Y
      br.call.sptk b0=__libm_error_support#         // Call error handling function
};;
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
};;
{ .mib
      mov   gp = GR_SAVE_GP                  // Restore gp
      mov   ar.pfs = GR_SAVE_PFS             // Restore ar.pfs
      br.ret.sptk     b0                     // Return
};;

LOCAL_LIBM_END(__libm_error_region)

.type   __libm_error_support#,@function
.global __libm_error_support#
