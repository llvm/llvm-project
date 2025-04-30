.file "log1pf.s"


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
// 12/16/03 Fixed parameter passing to/from error handling routine
//
// API
//==============================================================
// float log1pf(float)
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
//   P(r) = r*((1 - A2*4) + r^2*(A3 - A4*r)) = r*P2(r),
//   A4,A3,A2 are created with setf instruction.
//   We use Taylor series and so A4 = 1/4, A3 = 1/3,
//   A2 = 1/2 rounded to double.
//
//   N = float(n) where n is true unbiased exponent of x
//
//   T is tabular value of log(1/frcpa(x)) calculated in quad precision
//   and rounded to double.  To load T we get bits from 55 to 62 of register
//   format significand as index and calculate address
//     ad_T = table_base_addr + 8 * index
//
//   L1 (log(2)) is calculated in quad precision and rounded to double;
//   it's created with setf
//
//   And final result = P2(r)*r + (T + N*L1)
//
//
// 2. 2^(-40) <= |x| < 2^(-8)
//   r = x
//   P(r) = r*((1 - A2*4) + r^2*(A3 - A4*r)) = r*P2(r),
//   A4,A3,A2 are the same as in case |x| >= 1/256
//
//   And final result = P2(r)*r
//
// 3. 0 < |x| < 2^(-40)
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
// f7 -> f15,  f32 -> f36
//
// General registers used:
// r8  -> r11
// r14 -> r22
//
// Predicate registers used:
// p6 -> p12

// Assembly macros
//==============================================================
GR_TAG                 = r8
GR_ad_T                = r9
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
GR_Ln2                 = r21
GR_025                 = r22


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
FR_Ln2                 = f14
FR_Xp1                 = f15

FR_A4                  = f33
FR_A3                  = f34
FR_A2                  = f35

FR_T                   = f36
FR_NxLn2pT             = f36



FR_Y                   = f1
FR_X                   = f10
FR_RESULT              = f8


// Data
//==============================================================
RODATA
.align 16

LOCAL_OBJECT_START(log_data)
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
LOCAL_OBJECT_END(log_data)


// Code
//==============================================================

.section .text
GLOBAL_IEEE754_ENTRY(log1pf)
{ .mfi
      getf.exp      GR_signexp_x = f8 // if x is unorm then must recompute
      fadd.s1       FR_Xp1 = f8, f1       // Form 1+x
      mov           GR_05 = 0xfffe
}
{ .mlx
      addl          GR_ad_T = @ltoff(log_data),gp
      movl          GR_A3 = 0x3fd5555555555555 // double precision memory
                                               // representation of A3
}
;;

{ .mfi
      ld8           GR_ad_T = [GR_ad_T]
      fclass.m      p8,p0 = f8,0xb // Is x unorm?
      mov           GR_exp_mask = 0x1ffff
}
{ .mfi
      mov           GR_025 = 0xfffd            // Exponent of 0.25
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
      nop.i         0
(p8)  br.cond.spnt  log1p_unorm          // Branch if x=unorm
}
;;

log1p_common:
{ .mfi
      setf.exp      FR_A4 = GR_025 // create A4 = 0.25
      frcpa.s1      FR_RcpX,p0 = f1,FR_Xp1
      nop.i         0
}
{ .mfb
      nop.m         0
(p9)  fma.s.s0      f8 = f8,f1,f0 // set V-flag
(p9)  br.ret.spnt   b0 // exit for NaN, NaT and +Inf
}
;;

{ .mfi
      getf.exp      GR_Exp = FR_Xp1            // signexp of x+1
      fclass.m      p10,p0 = FR_Xp1,0x3A // is 1+x < 0?
      and           GR_exp_x = GR_exp_mask, GR_signexp_x // biased exponent of x
}
{ .mlx
      nop.m         0
      movl          GR_Ln2 = 0x3FE62E42FEFA39EF // double precision memory
                                                // representation of log(2)
}
;;

{ .mfi
      getf.sig      GR_Sig = FR_Xp1 // get significand to calculate index
                                    // for T if |x| >= 2^-8
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
      nop.m         0
      nop.f         0
(p10) br.cond.spnt  log1p_lt_minus_1   // jump if x < -1
}
;;

// p6 is true if |x| < 1/256
// p7 is true if |x| >= 1/256
.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fms.s1        FR_r = f8,f1,f0 // range reduction for |x|<1/256
(p6)  cmp.gt.unc    p10,p0 = -40, GR_exp_x  // Is |x| < 2^-40
}
{ .mfb
(p7)  setf.sig      FR_N = GR_N // copy unbiased exponent of x to the
                                // significand field of FR_N
(p7)  fms.s1        FR_r = FR_RcpX,FR_Xp1,f1 // range reduction for |x|>=1/256
(p12) br.ret.spnt   b0 // exit for x=0, return x
}
;;

{ .mib
      setf.d        FR_Ln2 = GR_Ln2 // create log(2)
(p7)  extr.u        GR_Ind = GR_Sig,55,8 // get bits from 55 to 62 as index
(p11) br.cond.spnt  log1p_eq_minus_1 // jump if x = -1
}
;;

{ .mmf
(p7)  shladd        GR_ad_T = GR_Ind,3,GR_ad_T // address of T
      nop.m         0
(p10) fnma.s.s0     f8 = f8,f8,f8   // If |x| very small, result=x-x*x
}
;;

{ .mmb
(p7)  ldfd          FR_T = [GR_ad_T]
      nop.m         0
(p10) br.ret.spnt   b0              // Exit if |x| < 2^-40
}
;;

{ .mfi
      nop.m         0
      fma.s1        FR_r2 = FR_r,FR_r,f0 // r^2
      nop.i         0
}
{ .mfi
      nop.m         0
      fnma.s1       FR_A2 = FR_A2,FR_r,f1      // 1.0 - A2*r
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fnma.s1       FR_A3 = FR_A4,FR_r,FR_A3 // A3 - A4*r
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
      // (A3*r+A2)*r^2+r
      fma.s1        FR_A2 = FR_A3,FR_r2,FR_A2 // (A4*r+A3)*r^2+(A2*r+1)
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // N*Ln2hi+T
(p7)  fma.s1        FR_NxLn2pT = FR_N,FR_Ln2,FR_T
      nop.i         0
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fma.s.s0      f8 = FR_A2,FR_r,f0 // result if 2^(-40) <= |x| < 1/256
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fma.s.s0      f8 = FR_A2,FR_r,FR_NxLn2pT  // result if |x| >= 1/256
      br.ret.sptk   b0                          // Exit if |x| >= 2^(-40)
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
      mov           GR_TAG = 142  // set libm error in case of log1p(-1).
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
      mov           GR_TAG = 143  // set libm error in case of x < -1.
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

GLOBAL_IEEE754_END(log1pf)
libm_alias_float_other (__log1p, log1p)


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
        stfs [GR_Parameter_Y] = FR_Y,16       // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp            // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0 = b0                   // Save b0
};;
.body
{ .mib
        stfs [GR_Parameter_X] = FR_X          // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y // Parameter 3 address
        nop.b 0
}
{ .mib
        stfs [GR_Parameter_Y] = FR_RESULT     // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support# // Call error handling function
};;
{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;
{ .mmi
        ldfs  f8 = [GR_Parameter_RESULT]      // Get return result off stack
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
