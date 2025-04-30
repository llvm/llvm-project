.file "asinhf.s"


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
// ==============================================================
// History
// ==============================================================
// 04/02/01 Initial version
// 04/19/01 Improved speed of the paths #1,2,3,4,5
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align
// 05/21/03 Improved performance, fixed to handle unorms
//
// API
// ==============================================================
// float asinhf(float)
//
// Overview of operation
// ==============================================================
//
// There are 7 paths:
// 1. x = 0.0
//    Return asinhf(x) = 0.0
// 2. 0.0 <|x| < 2^(-5)
//    Return asinhf(x) = Pol5(x), where Pol5(x) = ((x^2)*C1 + C0)*x^3 + x

// 3. 2^(-5) <= |x| < 2^51
//    Return asinhf(x) = sign(x)*(log(|x| + sqrt(x^2 + 1.0)))
//    To compute x + sqrt(x^2 + 1.0) modified Newton Raphson method is used
//    (2 iterations)
//    Algorithm description for log function see below.
//
// 4. 2^51 <= |x| < +INF
//    Return asinhf(x) = sign(x)*log(2*|x|)
//    Algorithm description for log function see below.
//
// 5. x = INF
//    Return asinhf(x) = INF
//
// 6. x = [S,Q]NaN
//    Return asinhf(x) = QNaN
//
// 7. x = denormal
//    Return asinhf(x) = x
//
//==============================================================
// Algorithm Description for log(x) function
// Below we are using the fact that inequality x - 1.0 > 2^(-6) is always
// true for this asinh implementation
//
// Consider  x = 2^N 1.f1 f2 f3 f4...f63
// Log(x) = log(frcpa(x) x/frcpa(x))
//        = log(1/frcpa(x)) + log(frcpa(x) x)
//        = -log(frcpa(x)) + log(frcpa(x) x)
//
// frcpa(x)       = 2^-N frcpa((1.f1 f2 ... f63)
//
// -log(frcpa(x)) = -log(C)
//                = -log(2^-N) - log(frcpa(1.f1 f2 ... f63))
//
// -log(frcpa(x)) = -log(C)
//                = +Nlog2 - log(frcpa(1.f1 f2 ... f63))
//
// -log(frcpa(x)) = -log(C)
//                = +Nlog2 + log(frcpa(1.f1 f2 ... f63))
//
// Log(x) = log(1/frcpa(x)) + log(frcpa(x) x)
//
// Log(x) =  +Nlog2 + log(1./frcpa(1.f1 f2 ... f63)) + log(frcpa(x) x)
// Log(x) =  +Nlog2 - log(/frcpa(1.f1 f2 ... f63))   + log(frcpa(x) x)
// Log(x) =  +Nlog2 + T                              + log(frcpa(x) x)
//
// Log(x) =  +Nlog2 + T                     + log(C x)
//
// Cx = 1 + r
//
// Log(x) =  +Nlog2 + T  + log(1+r)
// Log(x) =  +Nlog2 + T  + Series( r - r^2/2 + r^3/3 - r^4/4 ....)
//
// 1.f1 f2 ... f8 has 256 entries.
// They are 1 + k/2^8, k = 0 ... 255
// These 256 values are the table entries.
//
// Implementation
//==============================================================
// C = frcpa(x)
// r = C * x - 1
//
// Form rseries = r + P1*r^2 + P2*r^3 + P3*r^4
//
// x = f * 2*n where f is 1.f_1f_2f_3....f_63
// Nfloat = float(n)  where n is the true unbiased exponent
// pre-index = f_1f_2....f_8
// index = pre_index * 8
// get the dxt table entry at index + offset = T
//
// result = (T + Nfloat * log(2)) + rseries
//
// The T table is calculated as follows
// Form x_k = 1 + k/2^8 where k goes from 0... 255
//      y_k = frcpa(x_k)
//      log(1/y_k)  in quad and round to double-extended
//
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f9 -> f15,  f32 -> f55

// General registers used:
// r14 -> r27

// Predicate registers used:
// p6 -> p14

// p6 to filter out case when x = [Q,S]NaN or INF or zero
// p7 to filter out case when x < 0.0
// p8 to select path #2

// p11 to filter out case when x >= 0
// p12 to filter out case when x = + denormal
// p13 to select path #4
// p14 to filtef out case when x = - denormal
// Assembly macros
//==============================================================
log_GR_exp_17_ones    = r14
log_GR_signexp_f8     = r15
log_table_address2    = r16
log_GR_exp_16_ones    = r17
log_GR_exp_f8         = r18
log_GR_true_exp_f8    = r19
log_GR_significand_f8 = r20
log_GR_index          = r21
log_GR_comp2          = r22
asinh_GR_f8           = r23
asinh_GR_comp         = r24
asinh_GR_f8           = r25
log_table_address3    = r26
NR_table_address      = r27

//==============================================================
log_y            = f9
NR1              = f10
NR2              = f11
log_y_rs         = f12
log_y_rs_iter    = f13
log_y_rs_iter1   = f14
fNormX           = f15
asinh_w_sq       = f32
log_arg_early    = f33
log_y_rs_iter2   = f34
log_P3           = f35
log_P2           = f36
log_P1           = f37
log2             = f38
log_C0           = f39
log_C1           = f40
asinh_f8         = f41
log_C            = f42
log_arg          = f43
asinh_w_cube     = f44
log_int_Nfloat   = f45
log_r            = f46
log_rsq          = f47
asinh_w_1        = f48
log_rp_p32       = f49
log_rcube        = f50
log_rp_p10       = f51
log_rp_p2        = f52
log_Nfloat       = f53
log_T            = f54
log_T_plus_Nlog2 = f55

// Data tables
//==============================================================

RODATA
.align 16

LOCAL_OBJECT_START(log_table_1)

data8 0xbfd0001008f39d59 // p3
data8 0x3fd5556073e0c45a // p2
data8 0xbfdffffffffaea15 // p1
data8 0x3fe62e42fefa39ef // log(2)
LOCAL_OBJECT_END(log_table_1)

LOCAL_OBJECT_START(log_table_2)
data8 0x3FE0000000000000 // 0.5
data8 0x4008000000000000 // 3.0
data8 0x9979C79685A5EB16, 0x00003FFB // C1 3FFB9979C79685A5EB16
data8 0xAAAAA96F80786D62, 0x0000BFFC // C0 BFFCAAAAA96F80786D62
LOCAL_OBJECT_END(log_table_2)

LOCAL_OBJECT_START(log_table_3)
data8 0x3F60040155D5889E    //log(1/frcpa(1+   0/256)
data8 0x3F78121214586B54    //log(1/frcpa(1+   1/256)
data8 0x3F841929F96832F0    //log(1/frcpa(1+   2/256)
data8 0x3F8C317384C75F06    //log(1/frcpa(1+   3/256)
data8 0x3F91A6B91AC73386    //log(1/frcpa(1+   4/256)
data8 0x3F95BA9A5D9AC039    //log(1/frcpa(1+   5/256)
data8 0x3F99D2A8074325F4    //log(1/frcpa(1+   6/256)
data8 0x3F9D6B2725979802    //log(1/frcpa(1+   7/256)
data8 0x3FA0C58FA19DFAAA    //log(1/frcpa(1+   8/256)
data8 0x3FA2954C78CBCE1B    //log(1/frcpa(1+   9/256)
data8 0x3FA4A94D2DA96C56    //log(1/frcpa(1+  10/256)
data8 0x3FA67C94F2D4BB58    //log(1/frcpa(1+  11/256)
data8 0x3FA85188B630F068    //log(1/frcpa(1+  12/256)
data8 0x3FAA6B8ABE73AF4C    //log(1/frcpa(1+  13/256)
data8 0x3FAC441E06F72A9E    //log(1/frcpa(1+  14/256)
data8 0x3FAE1E6713606D07    //log(1/frcpa(1+  15/256)
data8 0x3FAFFA6911AB9301    //log(1/frcpa(1+  16/256)
data8 0x3FB0EC139C5DA601    //log(1/frcpa(1+  17/256)
data8 0x3FB1DBD2643D190B    //log(1/frcpa(1+  18/256)
data8 0x3FB2CC7284FE5F1C    //log(1/frcpa(1+  19/256)
data8 0x3FB3BDF5A7D1EE64    //log(1/frcpa(1+  20/256)
data8 0x3FB4B05D7AA012E0    //log(1/frcpa(1+  21/256)
data8 0x3FB580DB7CEB5702    //log(1/frcpa(1+  22/256)
data8 0x3FB674F089365A7A    //log(1/frcpa(1+  23/256)
data8 0x3FB769EF2C6B568D    //log(1/frcpa(1+  24/256)
data8 0x3FB85FD927506A48    //log(1/frcpa(1+  25/256)
data8 0x3FB9335E5D594989    //log(1/frcpa(1+  26/256)
data8 0x3FBA2B0220C8E5F5    //log(1/frcpa(1+  27/256)
data8 0x3FBB0004AC1A86AC    //log(1/frcpa(1+  28/256)
data8 0x3FBBF968769FCA11    //log(1/frcpa(1+  29/256)
data8 0x3FBCCFEDBFEE13A8    //log(1/frcpa(1+  30/256)
data8 0x3FBDA727638446A2    //log(1/frcpa(1+  31/256)
data8 0x3FBEA3257FE10F7A    //log(1/frcpa(1+  32/256)
data8 0x3FBF7BE9FEDBFDE6    //log(1/frcpa(1+  33/256)
data8 0x3FC02AB352FF25F4    //log(1/frcpa(1+  34/256)
data8 0x3FC097CE579D204D    //log(1/frcpa(1+  35/256)
data8 0x3FC1178E8227E47C    //log(1/frcpa(1+  36/256)
data8 0x3FC185747DBECF34    //log(1/frcpa(1+  37/256)
data8 0x3FC1F3B925F25D41    //log(1/frcpa(1+  38/256)
data8 0x3FC2625D1E6DDF57    //log(1/frcpa(1+  39/256)
data8 0x3FC2D1610C86813A    //log(1/frcpa(1+  40/256)
data8 0x3FC340C59741142E    //log(1/frcpa(1+  41/256)
data8 0x3FC3B08B6757F2A9    //log(1/frcpa(1+  42/256)
data8 0x3FC40DFB08378003    //log(1/frcpa(1+  43/256)
data8 0x3FC47E74E8CA5F7C    //log(1/frcpa(1+  44/256)
data8 0x3FC4EF51F6466DE4    //log(1/frcpa(1+  45/256)
data8 0x3FC56092E02BA516    //log(1/frcpa(1+  46/256)
data8 0x3FC5D23857CD74D5    //log(1/frcpa(1+  47/256)
data8 0x3FC6313A37335D76    //log(1/frcpa(1+  48/256)
data8 0x3FC6A399DABBD383    //log(1/frcpa(1+  49/256)
data8 0x3FC70337DD3CE41B    //log(1/frcpa(1+  50/256)
data8 0x3FC77654128F6127    //log(1/frcpa(1+  51/256)
data8 0x3FC7E9D82A0B022D    //log(1/frcpa(1+  52/256)
data8 0x3FC84A6B759F512F    //log(1/frcpa(1+  53/256)
data8 0x3FC8AB47D5F5A310    //log(1/frcpa(1+  54/256)
data8 0x3FC91FE49096581B    //log(1/frcpa(1+  55/256)
data8 0x3FC981634011AA75    //log(1/frcpa(1+  56/256)
data8 0x3FC9F6C407089664    //log(1/frcpa(1+  57/256)
data8 0x3FCA58E729348F43    //log(1/frcpa(1+  58/256)
data8 0x3FCABB55C31693AD    //log(1/frcpa(1+  59/256)
data8 0x3FCB1E104919EFD0    //log(1/frcpa(1+  60/256)
data8 0x3FCB94EE93E367CB    //log(1/frcpa(1+  61/256)
data8 0x3FCBF851C067555F    //log(1/frcpa(1+  62/256)
data8 0x3FCC5C0254BF23A6    //log(1/frcpa(1+  63/256)
data8 0x3FCCC000C9DB3C52    //log(1/frcpa(1+  64/256)
data8 0x3FCD244D99C85674    //log(1/frcpa(1+  65/256)
data8 0x3FCD88E93FB2F450    //log(1/frcpa(1+  66/256)
data8 0x3FCDEDD437EAEF01    //log(1/frcpa(1+  67/256)
data8 0x3FCE530EFFE71012    //log(1/frcpa(1+  68/256)
data8 0x3FCEB89A1648B971    //log(1/frcpa(1+  69/256)
data8 0x3FCF1E75FADF9BDE    //log(1/frcpa(1+  70/256)
data8 0x3FCF84A32EAD7C35    //log(1/frcpa(1+  71/256)
data8 0x3FCFEB2233EA07CD    //log(1/frcpa(1+  72/256)
data8 0x3FD028F9C7035C1C    //log(1/frcpa(1+  73/256)
data8 0x3FD05C8BE0D9635A    //log(1/frcpa(1+  74/256)
data8 0x3FD085EB8F8AE797    //log(1/frcpa(1+  75/256)
data8 0x3FD0B9C8E32D1911    //log(1/frcpa(1+  76/256)
data8 0x3FD0EDD060B78081    //log(1/frcpa(1+  77/256)
data8 0x3FD122024CF0063F    //log(1/frcpa(1+  78/256)
data8 0x3FD14BE2927AECD4    //log(1/frcpa(1+  79/256)
data8 0x3FD180618EF18ADF    //log(1/frcpa(1+  80/256)
data8 0x3FD1B50BBE2FC63B    //log(1/frcpa(1+  81/256)
data8 0x3FD1DF4CC7CF242D    //log(1/frcpa(1+  82/256)
data8 0x3FD214456D0EB8D4    //log(1/frcpa(1+  83/256)
data8 0x3FD23EC5991EBA49    //log(1/frcpa(1+  84/256)
data8 0x3FD2740D9F870AFB    //log(1/frcpa(1+  85/256)
data8 0x3FD29ECDABCDFA04    //log(1/frcpa(1+  86/256)
data8 0x3FD2D46602ADCCEE    //log(1/frcpa(1+  87/256)
data8 0x3FD2FF66B04EA9D4    //log(1/frcpa(1+  88/256)
data8 0x3FD335504B355A37    //log(1/frcpa(1+  89/256)
data8 0x3FD360925EC44F5D    //log(1/frcpa(1+  90/256)
data8 0x3FD38BF1C3337E75    //log(1/frcpa(1+  91/256)
data8 0x3FD3C25277333184    //log(1/frcpa(1+  92/256)
data8 0x3FD3EDF463C1683E    //log(1/frcpa(1+  93/256)
data8 0x3FD419B423D5E8C7    //log(1/frcpa(1+  94/256)
data8 0x3FD44591E0539F49    //log(1/frcpa(1+  95/256)
data8 0x3FD47C9175B6F0AD    //log(1/frcpa(1+  96/256)
data8 0x3FD4A8B341552B09    //log(1/frcpa(1+  97/256)
data8 0x3FD4D4F3908901A0    //log(1/frcpa(1+  98/256)
data8 0x3FD501528DA1F968    //log(1/frcpa(1+  99/256)
data8 0x3FD52DD06347D4F6    //log(1/frcpa(1+ 100/256)
data8 0x3FD55A6D3C7B8A8A    //log(1/frcpa(1+ 101/256)
data8 0x3FD5925D2B112A59    //log(1/frcpa(1+ 102/256)
data8 0x3FD5BF406B543DB2    //log(1/frcpa(1+ 103/256)
data8 0x3FD5EC433D5C35AE    //log(1/frcpa(1+ 104/256)
data8 0x3FD61965CDB02C1F    //log(1/frcpa(1+ 105/256)
data8 0x3FD646A84935B2A2    //log(1/frcpa(1+ 106/256)
data8 0x3FD6740ADD31DE94    //log(1/frcpa(1+ 107/256)
data8 0x3FD6A18DB74A58C5    //log(1/frcpa(1+ 108/256)
data8 0x3FD6CF31058670EC    //log(1/frcpa(1+ 109/256)
data8 0x3FD6F180E852F0BA    //log(1/frcpa(1+ 110/256)
data8 0x3FD71F5D71B894F0    //log(1/frcpa(1+ 111/256)
data8 0x3FD74D5AEFD66D5C    //log(1/frcpa(1+ 112/256)
data8 0x3FD77B79922BD37E    //log(1/frcpa(1+ 113/256)
data8 0x3FD7A9B9889F19E2    //log(1/frcpa(1+ 114/256)
data8 0x3FD7D81B037EB6A6    //log(1/frcpa(1+ 115/256)
data8 0x3FD8069E33827231    //log(1/frcpa(1+ 116/256)
data8 0x3FD82996D3EF8BCB    //log(1/frcpa(1+ 117/256)
data8 0x3FD85855776DCBFB    //log(1/frcpa(1+ 118/256)
data8 0x3FD8873658327CCF    //log(1/frcpa(1+ 119/256)
data8 0x3FD8AA75973AB8CF    //log(1/frcpa(1+ 120/256)
data8 0x3FD8D992DC8824E5    //log(1/frcpa(1+ 121/256)
data8 0x3FD908D2EA7D9512    //log(1/frcpa(1+ 122/256)
data8 0x3FD92C59E79C0E56    //log(1/frcpa(1+ 123/256)
data8 0x3FD95BD750EE3ED3    //log(1/frcpa(1+ 124/256)
data8 0x3FD98B7811A3EE5B    //log(1/frcpa(1+ 125/256)
data8 0x3FD9AF47F33D406C    //log(1/frcpa(1+ 126/256)
data8 0x3FD9DF270C1914A8    //log(1/frcpa(1+ 127/256)
data8 0x3FDA0325ED14FDA4    //log(1/frcpa(1+ 128/256)
data8 0x3FDA33440224FA79    //log(1/frcpa(1+ 129/256)
data8 0x3FDA57725E80C383    //log(1/frcpa(1+ 130/256)
data8 0x3FDA87D0165DD199    //log(1/frcpa(1+ 131/256)
data8 0x3FDAAC2E6C03F896    //log(1/frcpa(1+ 132/256)
data8 0x3FDADCCC6FDF6A81    //log(1/frcpa(1+ 133/256)
data8 0x3FDB015B3EB1E790    //log(1/frcpa(1+ 134/256)
data8 0x3FDB323A3A635948    //log(1/frcpa(1+ 135/256)
data8 0x3FDB56FA04462909    //log(1/frcpa(1+ 136/256)
data8 0x3FDB881AA659BC93    //log(1/frcpa(1+ 137/256)
data8 0x3FDBAD0BEF3DB165    //log(1/frcpa(1+ 138/256)
data8 0x3FDBD21297781C2F    //log(1/frcpa(1+ 139/256)
data8 0x3FDC039236F08819    //log(1/frcpa(1+ 140/256)
data8 0x3FDC28CB1E4D32FD    //log(1/frcpa(1+ 141/256)
data8 0x3FDC4E19B84723C2    //log(1/frcpa(1+ 142/256)
data8 0x3FDC7FF9C74554C9    //log(1/frcpa(1+ 143/256)
data8 0x3FDCA57B64E9DB05    //log(1/frcpa(1+ 144/256)
data8 0x3FDCCB130A5CEBB0    //log(1/frcpa(1+ 145/256)
data8 0x3FDCF0C0D18F326F    //log(1/frcpa(1+ 146/256)
data8 0x3FDD232075B5A201    //log(1/frcpa(1+ 147/256)
data8 0x3FDD490246DEFA6B    //log(1/frcpa(1+ 148/256)
data8 0x3FDD6EFA918D25CD    //log(1/frcpa(1+ 149/256)
data8 0x3FDD9509707AE52F    //log(1/frcpa(1+ 150/256)
data8 0x3FDDBB2EFE92C554    //log(1/frcpa(1+ 151/256)
data8 0x3FDDEE2F3445E4AF    //log(1/frcpa(1+ 152/256)
data8 0x3FDE148A1A2726CE    //log(1/frcpa(1+ 153/256)
data8 0x3FDE3AFC0A49FF40    //log(1/frcpa(1+ 154/256)
data8 0x3FDE6185206D516E    //log(1/frcpa(1+ 155/256)
data8 0x3FDE882578823D52    //log(1/frcpa(1+ 156/256)
data8 0x3FDEAEDD2EAC990C    //log(1/frcpa(1+ 157/256)
data8 0x3FDED5AC5F436BE3    //log(1/frcpa(1+ 158/256)
data8 0x3FDEFC9326D16AB9    //log(1/frcpa(1+ 159/256)
data8 0x3FDF2391A2157600    //log(1/frcpa(1+ 160/256)
data8 0x3FDF4AA7EE03192D    //log(1/frcpa(1+ 161/256)
data8 0x3FDF71D627C30BB0    //log(1/frcpa(1+ 162/256)
data8 0x3FDF991C6CB3B379    //log(1/frcpa(1+ 163/256)
data8 0x3FDFC07ADA69A910    //log(1/frcpa(1+ 164/256)
data8 0x3FDFE7F18EB03D3E    //log(1/frcpa(1+ 165/256)
data8 0x3FE007C053C5002E    //log(1/frcpa(1+ 166/256)
data8 0x3FE01B942198A5A1    //log(1/frcpa(1+ 167/256)
data8 0x3FE02F74400C64EB    //log(1/frcpa(1+ 168/256)
data8 0x3FE04360BE7603AD    //log(1/frcpa(1+ 169/256)
data8 0x3FE05759AC47FE34    //log(1/frcpa(1+ 170/256)
data8 0x3FE06B5F1911CF52    //log(1/frcpa(1+ 171/256)
data8 0x3FE078BF0533C568    //log(1/frcpa(1+ 172/256)
data8 0x3FE08CD9687E7B0E    //log(1/frcpa(1+ 173/256)
data8 0x3FE0A10074CF9019    //log(1/frcpa(1+ 174/256)
data8 0x3FE0B5343A234477    //log(1/frcpa(1+ 175/256)
data8 0x3FE0C974C89431CE    //log(1/frcpa(1+ 176/256)
data8 0x3FE0DDC2305B9886    //log(1/frcpa(1+ 177/256)
data8 0x3FE0EB524BAFC918    //log(1/frcpa(1+ 178/256)
data8 0x3FE0FFB54213A476    //log(1/frcpa(1+ 179/256)
data8 0x3FE114253DA97D9F    //log(1/frcpa(1+ 180/256)
data8 0x3FE128A24F1D9AFF    //log(1/frcpa(1+ 181/256)
data8 0x3FE1365252BF0865    //log(1/frcpa(1+ 182/256)
data8 0x3FE14AE558B4A92D    //log(1/frcpa(1+ 183/256)
data8 0x3FE15F85A19C765B    //log(1/frcpa(1+ 184/256)
data8 0x3FE16D4D38C119FA    //log(1/frcpa(1+ 185/256)
data8 0x3FE18203C20DD133    //log(1/frcpa(1+ 186/256)
data8 0x3FE196C7BC4B1F3B    //log(1/frcpa(1+ 187/256)
data8 0x3FE1A4A738B7A33C    //log(1/frcpa(1+ 188/256)
data8 0x3FE1B981C0C9653D    //log(1/frcpa(1+ 189/256)
data8 0x3FE1CE69E8BB106B    //log(1/frcpa(1+ 190/256)
data8 0x3FE1DC619DE06944    //log(1/frcpa(1+ 191/256)
data8 0x3FE1F160A2AD0DA4    //log(1/frcpa(1+ 192/256)
data8 0x3FE2066D7740737E    //log(1/frcpa(1+ 193/256)
data8 0x3FE2147DBA47A394    //log(1/frcpa(1+ 194/256)
data8 0x3FE229A1BC5EBAC3    //log(1/frcpa(1+ 195/256)
data8 0x3FE237C1841A502E    //log(1/frcpa(1+ 196/256)
data8 0x3FE24CFCE6F80D9A    //log(1/frcpa(1+ 197/256)
data8 0x3FE25B2C55CD5762    //log(1/frcpa(1+ 198/256)
data8 0x3FE2707F4D5F7C41    //log(1/frcpa(1+ 199/256)
data8 0x3FE285E0842CA384    //log(1/frcpa(1+ 200/256)
data8 0x3FE294294708B773    //log(1/frcpa(1+ 201/256)
data8 0x3FE2A9A2670AFF0C    //log(1/frcpa(1+ 202/256)
data8 0x3FE2B7FB2C8D1CC1    //log(1/frcpa(1+ 203/256)
data8 0x3FE2C65A6395F5F5    //log(1/frcpa(1+ 204/256)
data8 0x3FE2DBF557B0DF43    //log(1/frcpa(1+ 205/256)
data8 0x3FE2EA64C3F97655    //log(1/frcpa(1+ 206/256)
data8 0x3FE3001823684D73    //log(1/frcpa(1+ 207/256)
data8 0x3FE30E97E9A8B5CD    //log(1/frcpa(1+ 208/256)
data8 0x3FE32463EBDD34EA    //log(1/frcpa(1+ 209/256)
data8 0x3FE332F4314AD796    //log(1/frcpa(1+ 210/256)
data8 0x3FE348D90E7464D0    //log(1/frcpa(1+ 211/256)
data8 0x3FE35779F8C43D6E    //log(1/frcpa(1+ 212/256)
data8 0x3FE36621961A6A99    //log(1/frcpa(1+ 213/256)
data8 0x3FE37C299F3C366A    //log(1/frcpa(1+ 214/256)
data8 0x3FE38AE2171976E7    //log(1/frcpa(1+ 215/256)
data8 0x3FE399A157A603E7    //log(1/frcpa(1+ 216/256)
data8 0x3FE3AFCCFE77B9D1    //log(1/frcpa(1+ 217/256)
data8 0x3FE3BE9D503533B5    //log(1/frcpa(1+ 218/256)
data8 0x3FE3CD7480B4A8A3    //log(1/frcpa(1+ 219/256)
data8 0x3FE3E3C43918F76C    //log(1/frcpa(1+ 220/256)
data8 0x3FE3F2ACB27ED6C7    //log(1/frcpa(1+ 221/256)
data8 0x3FE4019C2125CA93    //log(1/frcpa(1+ 222/256)
data8 0x3FE4181061389722    //log(1/frcpa(1+ 223/256)
data8 0x3FE42711518DF545    //log(1/frcpa(1+ 224/256)
data8 0x3FE436194E12B6BF    //log(1/frcpa(1+ 225/256)
data8 0x3FE445285D68EA69    //log(1/frcpa(1+ 226/256)
data8 0x3FE45BCC464C893A    //log(1/frcpa(1+ 227/256)
data8 0x3FE46AED21F117FC    //log(1/frcpa(1+ 228/256)
data8 0x3FE47A1527E8A2D3    //log(1/frcpa(1+ 229/256)
data8 0x3FE489445EFFFCCC    //log(1/frcpa(1+ 230/256)
data8 0x3FE4A018BCB69835    //log(1/frcpa(1+ 231/256)
data8 0x3FE4AF5A0C9D65D7    //log(1/frcpa(1+ 232/256)
data8 0x3FE4BEA2A5BDBE87    //log(1/frcpa(1+ 233/256)
data8 0x3FE4CDF28F10AC46    //log(1/frcpa(1+ 234/256)
data8 0x3FE4DD49CF994058    //log(1/frcpa(1+ 235/256)
data8 0x3FE4ECA86E64A684    //log(1/frcpa(1+ 236/256)
data8 0x3FE503C43CD8EB68    //log(1/frcpa(1+ 237/256)
data8 0x3FE513356667FC57    //log(1/frcpa(1+ 238/256)
data8 0x3FE522AE0738A3D8    //log(1/frcpa(1+ 239/256)
data8 0x3FE5322E26867857    //log(1/frcpa(1+ 240/256)
data8 0x3FE541B5CB979809    //log(1/frcpa(1+ 241/256)
data8 0x3FE55144FDBCBD62    //log(1/frcpa(1+ 242/256)
data8 0x3FE560DBC45153C7    //log(1/frcpa(1+ 243/256)
data8 0x3FE5707A26BB8C66    //log(1/frcpa(1+ 244/256)
data8 0x3FE587F60ED5B900    //log(1/frcpa(1+ 245/256)
data8 0x3FE597A7977C8F31    //log(1/frcpa(1+ 246/256)
data8 0x3FE5A760D634BB8B    //log(1/frcpa(1+ 247/256)
data8 0x3FE5B721D295F10F    //log(1/frcpa(1+ 248/256)
data8 0x3FE5C6EA94431EF9    //log(1/frcpa(1+ 249/256)
data8 0x3FE5D6BB22EA86F6    //log(1/frcpa(1+ 250/256)
data8 0x3FE5E6938645D390    //log(1/frcpa(1+ 251/256)
data8 0x3FE5F673C61A2ED2    //log(1/frcpa(1+ 252/256)
data8 0x3FE6065BEA385926    //log(1/frcpa(1+ 253/256)
data8 0x3FE6164BFA7CC06B    //log(1/frcpa(1+ 254/256)
data8 0x3FE62643FECF9743    //log(1/frcpa(1+ 255/256)
LOCAL_OBJECT_END(log_table_3)


.section .text
GLOBAL_LIBM_ENTRY(asinhf)

{ .mfi
      getf.exp   asinh_GR_f8 = f8        // Must recompute later if x unorm
      fclass.m   p12,p0 = f8, 0x0b       // Test x unorm
      mov        log_GR_exp_17_ones = 0x1ffff
}
{ .mfi
      addl       NR_table_address = @ltoff(log_table_1), gp
      fma.s1     log_y = f8, f8, f1      // y = x^2 + 1
      mov        asinh_GR_comp = 0xfffa
}
;;

{ .mfi
      mov        log_GR_exp_16_ones = 0xffff //BIAS
      fclass.m   p6,p0 = f8, 0xe7        // Test for x = NaN and inf and zero
      mov        log_GR_comp2 = 0x10032
}
{ .mfi
      ld8        NR_table_address = [NR_table_address]
      fma.s1     asinh_w_sq = f8,f8,f0   // x^2
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fcmp.lt.s1 p7,p11 = f8,f0          // if x<0
      nop.i      0
}
{ .mfb
      nop.m      0
      fnorm.s1   fNormX = f8             // Normalize x
(p12) br.cond.spnt ASINH_UNORM           // Branch if x=unorm
}
;;

ASINH_COMMON:
// Return here if x=unorm and not denorm
{ .mfi
      //to get second table address
      adds       log_table_address2 = 0x20, NR_table_address
      fma.s1     log_arg = f8,f1,f8
}
{ .mfb
      nop.m      0
(p6)  fma.s.s0   f8 = f8,f1,f8           // quietize nan result if x=nan
(p6)  br.ret.spnt b0                     // Exit for x=nan and inf and zero
}
;;

{ .mfi
      ldfpd      NR1,NR2 = [log_table_address2],16
      frsqrta.s1 log_y_rs,p0 = log_y     // z=1/sqrt(y)
      nop.i      0
}
;;

{ .mfi
      ldfe       log_C1 = [log_table_address2],16
      nop.f      0
      and        asinh_GR_f8 = asinh_GR_f8,log_GR_exp_17_ones
}
;;

{ .mib
      ldfe       log_C0 = [log_table_address2],16
      cmp.le     p13,p0 = log_GR_comp2,asinh_GR_f8
(p13) br.cond.spnt LOG_COMMON1           // Branch if path 4: |x| >= 2^51
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_y_rs_iter = log_y_rs,log_y,f0  // y*z
      nop.i      0
}
;;

.pred.rel "mutex",p7,p11
{ .mfi
      nop.m      0
(p11) mov        asinh_f8 = fNormX
      nop.i      0
}
{ .mfb
      cmp.gt     p8,p0 = asinh_GR_comp,asinh_GR_f8
(p7)  fnma.s1    asinh_f8 = fNormX,f1,f0
(p8)  br.cond.spnt ASINH_NEAR_ZERO       // Branch if path 2: 0 < |x| < 2^-5
}
;;

// Here if main path, 2^-5 <= |x| < 2^51
///////////////////////////////// The first iteration /////////////////////////
{ .mfi
      ldfpd      log_P3,log_P2 = [NR_table_address],16
      fnma.s1    log_y_rs_iter2 = log_y_rs_iter,log_y_rs,NR2    // 3-(y*z)*z
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_y_rs_iter1 = log_y_rs,NR1,f0               // 0.5*z
      nop.i      0
}
;;

{ .mfi
      ldfpd      log_P1,log2 = [NR_table_address],16
      // (0.5*z)*(3-(y*z)*z)
      fma.s1     log_y_rs_iter = log_y_rs_iter1,log_y_rs_iter2,f0
      nop.i      0
}
{ .mfi
      nop.m      0
      // (0.5*z)*(3-(y*z)*z)
      fma.s1     log_arg_early = log_y_rs_iter1,log_y_rs_iter2,f0
      nop.i      0
}
;;

////////////////////////////////// The second iteration ////////////////////////
{ .mfi
      nop.m      0
      fma.s1     log_y_rs = log_y_rs_iter,log_y,f0
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_y_rs_iter1 = log_y_rs_iter,NR1,f0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_arg_early = log_arg_early,log_y,asinh_f8
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fnma.s1    log_y_rs = log_y_rs,log_y_rs_iter,NR2
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_y_rs_iter1 = log_y_rs_iter1,log_y,f0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      frcpa.s1   log_C,p0 = f1,log_arg_early
      nop.i      0
}
;;

{ .mfi
      getf.exp   log_GR_signexp_f8 = log_arg_early
      nop.f      0
      nop.i      0
}
;;

{ .mfi
      getf.sig   log_GR_significand_f8 = log_arg_early
      // (0.5*z)*(3-(y*z)*z)*y + |x|
      fma.s1     log_arg = log_y_rs_iter1,log_y_rs,asinh_f8
      //to get third table address
      adds       log_table_address3 = 0x30, NR_table_address
}
;;

/////////////////////////////////////////// The end NR iterations /////////////

{ .mfi
      nop.m      0
      nop.f      0
      //significant bit destruction
      and        log_GR_exp_f8 = log_GR_signexp_f8, log_GR_exp_17_ones
}
;;

{ .mfi
      //BIAS subtraction
      sub        log_GR_true_exp_f8 = log_GR_exp_f8, log_GR_exp_16_ones
(p7)  fnma.s1    log2 = log2,f1,f0
      nop.i      0
}
;;

{ .mfi
      setf.sig   log_int_Nfloat = log_GR_true_exp_f8
      fms.s1     log_r = log_C,log_arg,f1  //C = frcpa(x); r = C * x - 1
      extr.u     log_GR_index = log_GR_significand_f8,55,8 //Extract 8 bits
}
;;

{ .mmi
      //pre-index*16 + index
      shladd     log_table_address3 = log_GR_index,3,log_table_address3
;;
      ldfd       log_T = [log_table_address3]
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rsq = log_r, log_r, f0          //r^2
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_rp_p32 = log_P3, log_r, log_P2  //P3*r + P2
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rp_p10 = log_P1, log_r, f1
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      //convert N to the floating-point format
      fcvt.xf    log_Nfloat = log_int_Nfloat
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rp_p2 = log_rp_p32, log_rsq, log_rp_p10
      nop.i      0
}
;;

.pred.rel "mutex",p7,p11
{ .mfi
      nop.m      0
(p11) fma.s1     log_T_plus_Nlog2 = log_Nfloat,log2,log_T  //N*log2 + T if x>0
      nop.i      0
}
{ .mfi
      nop.m      0
(p7)  fms.s1     log_T_plus_Nlog2 = log_Nfloat,log2,log_T  //N*log2 - T if x<0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
(p11) fma.s.s0   f8 = log_rp_p2,log_r,log_T_plus_Nlog2
      nop.i      0
}
{ .mfb
      nop.m      0
(p7)  fnma.s.s0  f8 = log_rp_p2,log_r,log_T_plus_Nlog2
      br.ret.sptk b0          // Exit main path, path 3: 2^-5 <= |x| < 2^51
}
;;


// Here if path 4, |x| >= 2^51
LOG_COMMON1:
{ .mfi
      ldfpd      log_P3,log_P2 = [NR_table_address],16
      nop.f      0
      nop.i      0
}
;;

{ .mfi
      ldfpd      log_P1,log2 = [NR_table_address],16
      frcpa.s1   log_C,p0 = f1,log_arg
      nop.i      0
}
;;

{ .mfi
      getf.exp   log_GR_signexp_f8 = log_arg
      nop.f      0
      //to get third table address
      adds       log_table_address3 = 0x30, NR_table_address
}
;;

{ .mfi
      getf.sig   log_GR_significand_f8 = log_arg
      nop.f      0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      nop.f      0
      //to destroy the most bit in the significant area
      and        log_GR_exp_f8 = log_GR_signexp_f8, log_GR_exp_17_ones
}
;;

{ .mmf
      nop.m      0
      //BIAS subtraction
      sub        log_GR_true_exp_f8 = log_GR_exp_f8, log_GR_exp_16_ones
      fms.s1     log_r = log_C,log_arg,f1  //C = frcpa(x); r = C * x - 1
}
;;

{ .mfi
      setf.sig   log_int_Nfloat = log_GR_true_exp_f8
      nop.f      0
      extr.u     log_GR_index = log_GR_significand_f8,55,8 //Extract 8 bits
}
;;

{ .mmi
      //pre-index*16 + index
      shladd     log_table_address3 = log_GR_index,3,log_table_address3
;;
      ldfd       log_T = [log_table_address3]
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rsq = log_r, log_r, f0          //r^2
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_rp_p32 = log_P3, log_r, log_P2  //P3*r + P2
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rp_p10 = log_P1, log_r, f1
      nop.i      0
}
{ .mfi
      nop.m      0
(p7)  fnma.s1    log2 = log2,f1,f0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      //convert N to the floating-point format
      fcvt.xf    log_Nfloat = log_int_Nfloat
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_rp_p2 = log_rp_p32, log_rsq, log_rp_p10
      nop.i      0
}
;;

.pred.rel "mutex",p7,p11
{ .mfi
      nop.m      0
(p11) fma.s1     log_T_plus_Nlog2 = log_Nfloat,log2,log_T  //N*log2 + T if x>0
      nop.i      0
}
{ .mfi
      nop.m      0
(p7)  fms.s1     log_T_plus_Nlog2 = log_Nfloat,log2,log_T  //N*log2 - T if x<0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
(p11) fma.s.s0   f8 = log_rp_p2,log_r,log_T_plus_Nlog2
      nop.i      0
}
{ .mfb
      nop.m      0
(p7)  fnma.s.s0  f8 = log_rp_p2,log_r,log_T_plus_Nlog2
      br.ret.sptk b0           // Exit path 4, |x| >= 2^51
}
;;

// Here if path 2, 0 < |x| < 2^-5
ASINH_NEAR_ZERO:
{ .mfi
      nop.m      0
      fma.s1     asinh_w_1 = asinh_w_sq,log_C1,log_C0
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     asinh_w_cube = asinh_w_sq,fNormX,f0
      nop.i      0
}
;;

{ .mfb
      nop.m      0
      fma.s.s0   f8 = asinh_w_1,asinh_w_cube,fNormX
      br.ret.sptk b0          // Exit path 2, 0 < |x| < 2^-5
}
;;

ASINH_UNORM:
// Here if x=unorm
{ .mfi
      getf.exp   asinh_GR_f8 = fNormX  // Recompute if x unorm
      fclass.m   p0,p13 = fNormX, 0x0b // Test x denorm
      nop.i      0
}
;;

{ .mfb
      nop.m      0
      fcmp.eq.s0 p14,p0 = f8, f0       // Dummy to set denormal flag
(p13) br.cond.sptk ASINH_COMMON        // Continue if x unorm and not denorm
}
;;

.pred.rel "mutex",p7,p11
{ .mfi
      nop.m      0
(p7)  fma.s.s0   f8 = f8,f8,f8         // Result x+x^2 if x=-denorm
      nop.i      0
}
{ .mfb
      nop.m      0
(p11) fnma.s.s0  f8 = f8,f8,f8         // Result x-x^2 if x=+denorm
      br.ret.spnt b0                   // Exit if denorm
}
;;

GLOBAL_LIBM_END(asinhf)
libm_alias_float_other (asinh, asinh)
