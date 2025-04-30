.file "atanhf.s"


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
// 05/22/01 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 08/06/02 Improved Itanium 2 performance
// 02/06/03 Reordered header: .section, .global, .proc, .align
// 05/26/03 Improved performance, fixed to handle unorms
//
// API
//==============================================================
// float atanhf(float)
//
// Overview of operation
//==============================================================
// Background
//
//
// There are 7 paths:
// 1. x = +/-0.0
//    Return atanhf(x) = +/-0.0
//
// 2. 0.0 < |x| <= MAX_DENORMAL_ABS
//    Return atanhf(x) = x + sign(x)*x^2
//
// 3. MAX_DENORMAL_ABS < |x| < 2^(-20)
//    Return atanhf(x) = Pol3(x), where Pol3(x) = x + x^3
//
// 4. 2^(-20) <= |x| < 1
//    Return atanhf(x) = 0.5 * (log(1 + x) - log(1 - x))
//    Algorithm description for log function see below.
//
// 5. |x| = 1
//    Return atanhf(x) = sign(x) * +INF
//
// 6. 1 < |x| <= +INF
//    Return atanhf(x) = QNaN
//
// 7. x = [S,Q]NaN
//    Return atanhf(x) = QNaN
//
//==============================================================
// Algorithm Description for log(x) function
//
// Consider  x = 2^N * 1.f1 f2 f3 f4...f63
// log(x) = log(x * frcpa(x) / frcpa(x))
//        = log(x * frcpa(x)) + log(1/frcpa(x))
//        = log(x * frcpa(x)) - log(frcpa(x))
//
// frcpa(x) = 2^(-N) * frcpa(1.f1 f2 ... f63)
//
// -log(frcpa(x)) = -log(C)
//                = -log(2^(-N)) - log(frcpa(1.f1 f2 ... f63))
//
// -log(frcpa(x)) = -log(C)
//                = N*log2 - log(frcpa(1.f1 f2 ... f63))
//
//
// log(x) = log(1/frcpa(x)) + log(frcpa(x) x)
//
// log(x) = N*log2 + log(1./frcpa(1.f1 f2 ... f63)) + log(x * frcpa(x))
// log(x) = N*log2 + T                              + log(frcpa(x) x)
//
// Log(x) = N*log2 + T                              + log(C * x)
//
// C * x = 1 + r
//
// log(x) = N*log2 + T + log(1 + r)
// log(x) = N*log2 + T + Series(r)
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
// x = f * 2*N where f is 1.f_1f_2f_3...f_63
// Nfloat = float(n)  where n is the true unbiased exponent
// pre-index = f_1f_2....f_8
// index = pre_index * 16
// get the dxt table entry at index + offset = T
//
// result = (T + Nfloat * log(2)) + rseries
//
// The T table is calculated as follows
// Form x_k = 1 + k/2^8 where k goes from 0... 255
//      y_k = frcpa(x_k)
//      log(1/y_k)  in quad and round to double-extended

// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f32 -> f59

// General registers used:
// r14 -> r29, r32 -> r39

// Predicate registers used:
// p6 -> p9

// p6           to filter out case when |x| >= 1
// p7           to filter out case when x = [Q,S]NaN or +/-0
// p8           to filter out case when |x| < 2^(-20)
// p9           to filter out case when x = denormal


// Assembly macros
//==============================================================
DataPtr               = r14
RcpTablePtrM          = r15
RcpTablePtrP          = r16
rExpbMask             = r17
rBias                 = r18
rNearZeroBound        = r19
rArgSExpb             = r20
rArgExpb              = r21
rExpbm                = r22
rExpbp                = r23
rSigm                 = r24
rSigp                 = r25
rNm                   = r26
rNp                   = r27
rIndm                 = r28
rIndp                 = r29

GR_SAVE_B0            = r33
GR_SAVE_GP            = r34
GR_SAVE_PFS           = r35

GR_Parameter_X        = r36
GR_Parameter_Y        = r37
GR_Parameter_RESULT   = r38
atanh_GR_tag          = r39

//==============================================================
fOneMx                = f33
fOnePx                = f34
fRm2                  = f35
fRm3                  = f36
fRp2                  = f37
fRp3                  = f38
fRcpM                 = f39
fRcpP                 = f40
fRp                   = f41
fRm                   = f42
fN4CvtM               = f43
fN4CvtP               = f44
fNm                   = f45
fNp                   = f46
fLogTm                = f47
fLogTp                = f48
fLog2                 = f49
fArgAbs               = f50
fNormX                = f50
fP32m                 = f51
fP32p                 = f52
fP10m                 = f53
fP10p                 = f54
fX2                   = f55
fP3                   = f56
fP2                   = f57
fP1                   = f58
fHalf                 = f59



// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(atanhf_data)
data8 0xbfc0001008f39d59    // P3*0.5
data8 0x3fc5556073e0c45a    // P2*0.5
data8 0xbfcffffffffaea15    // P1*0.5
data8 0x3fe0000000000000    // 0.5
data8 0x3fd62e42fefa39ef    // 0.5*ln(2)
data8 0x0000000000000000    // pad
LOCAL_OBJECT_END(atanhf_data)

LOCAL_OBJECT_START(atanhf_data2)
data8 0x3f50040155d5889e    //log(1/frcpa(1+0/256))/2
data8 0x3f68121214586b54    //log(1/frcpa(1+1/256))/2
data8 0x3f741929f96832f0    //log(1/frcpa(1+2/256))/2
data8 0x3f7c317384c75f06    //log(1/frcpa(1+3/256))/2
data8 0x3f81a6b91ac73386    //log(1/frcpa(1+4/256))/2
data8 0x3f85ba9a5d9ac039    //log(1/frcpa(1+5/256))/2
data8 0x3f89d2a8074325f4    //log(1/frcpa(1+6/256))/2
data8 0x3f8d6b2725979802    //log(1/frcpa(1+7/256))/2
data8 0x3f90c58fa19dfaaa    //log(1/frcpa(1+8/256))/2
data8 0x3f92954c78cbce1b    //log(1/frcpa(1+9/256))/2
data8 0x3f94a94d2da96c56    //log(1/frcpa(1+10/256))/2
data8 0x3f967c94f2d4bb58    //log(1/frcpa(1+11/256))/2
data8 0x3f985188b630f068    //log(1/frcpa(1+12/256))/2
data8 0x3f9a6b8abe73af4c    //log(1/frcpa(1+13/256))/2
data8 0x3f9c441e06f72a9e    //log(1/frcpa(1+14/256))/2
data8 0x3f9e1e6713606d07    //log(1/frcpa(1+15/256))/2
data8 0x3f9ffa6911ab9301    //log(1/frcpa(1+16/256))/2
data8 0x3fa0ec139c5da601    //log(1/frcpa(1+17/256))/2
data8 0x3fa1dbd2643d190b    //log(1/frcpa(1+18/256))/2
data8 0x3fa2cc7284fe5f1c    //log(1/frcpa(1+19/256))/2
data8 0x3fa3bdf5a7d1ee64    //log(1/frcpa(1+20/256))/2
data8 0x3fa4b05d7aa012e0    //log(1/frcpa(1+21/256))/2
data8 0x3fa580db7ceb5702    //log(1/frcpa(1+22/256))/2
data8 0x3fa674f089365a7a    //log(1/frcpa(1+23/256))/2
data8 0x3fa769ef2c6b568d    //log(1/frcpa(1+24/256))/2
data8 0x3fa85fd927506a48    //log(1/frcpa(1+25/256))/2
data8 0x3fa9335e5d594989    //log(1/frcpa(1+26/256))/2
data8 0x3faa2b0220c8e5f5    //log(1/frcpa(1+27/256))/2
data8 0x3fab0004ac1a86ac    //log(1/frcpa(1+28/256))/2
data8 0x3fabf968769fca11    //log(1/frcpa(1+29/256))/2
data8 0x3faccfedbfee13a8    //log(1/frcpa(1+30/256))/2
data8 0x3fada727638446a2    //log(1/frcpa(1+31/256))/2
data8 0x3faea3257fe10f7a    //log(1/frcpa(1+32/256))/2
data8 0x3faf7be9fedbfde6    //log(1/frcpa(1+33/256))/2
data8 0x3fb02ab352ff25f4    //log(1/frcpa(1+34/256))/2
data8 0x3fb097ce579d204d    //log(1/frcpa(1+35/256))/2
data8 0x3fb1178e8227e47c    //log(1/frcpa(1+36/256))/2
data8 0x3fb185747dbecf34    //log(1/frcpa(1+37/256))/2
data8 0x3fb1f3b925f25d41    //log(1/frcpa(1+38/256))/2
data8 0x3fb2625d1e6ddf57    //log(1/frcpa(1+39/256))/2
data8 0x3fb2d1610c86813a    //log(1/frcpa(1+40/256))/2
data8 0x3fb340c59741142e    //log(1/frcpa(1+41/256))/2
data8 0x3fb3b08b6757f2a9    //log(1/frcpa(1+42/256))/2
data8 0x3fb40dfb08378003    //log(1/frcpa(1+43/256))/2
data8 0x3fb47e74e8ca5f7c    //log(1/frcpa(1+44/256))/2
data8 0x3fb4ef51f6466de4    //log(1/frcpa(1+45/256))/2
data8 0x3fb56092e02ba516    //log(1/frcpa(1+46/256))/2
data8 0x3fb5d23857cd74d5    //log(1/frcpa(1+47/256))/2
data8 0x3fb6313a37335d76    //log(1/frcpa(1+48/256))/2
data8 0x3fb6a399dabbd383    //log(1/frcpa(1+49/256))/2
data8 0x3fb70337dd3ce41b    //log(1/frcpa(1+50/256))/2
data8 0x3fb77654128f6127    //log(1/frcpa(1+51/256))/2
data8 0x3fb7e9d82a0b022d    //log(1/frcpa(1+52/256))/2
data8 0x3fb84a6b759f512f    //log(1/frcpa(1+53/256))/2
data8 0x3fb8ab47d5f5a310    //log(1/frcpa(1+54/256))/2
data8 0x3fb91fe49096581b    //log(1/frcpa(1+55/256))/2
data8 0x3fb981634011aa75    //log(1/frcpa(1+56/256))/2
data8 0x3fb9f6c407089664    //log(1/frcpa(1+57/256))/2
data8 0x3fba58e729348f43    //log(1/frcpa(1+58/256))/2
data8 0x3fbabb55c31693ad    //log(1/frcpa(1+59/256))/2
data8 0x3fbb1e104919efd0    //log(1/frcpa(1+60/256))/2
data8 0x3fbb94ee93e367cb    //log(1/frcpa(1+61/256))/2
data8 0x3fbbf851c067555f    //log(1/frcpa(1+62/256))/2
data8 0x3fbc5c0254bf23a6    //log(1/frcpa(1+63/256))/2
data8 0x3fbcc000c9db3c52    //log(1/frcpa(1+64/256))/2
data8 0x3fbd244d99c85674    //log(1/frcpa(1+65/256))/2
data8 0x3fbd88e93fb2f450    //log(1/frcpa(1+66/256))/2
data8 0x3fbdedd437eaef01    //log(1/frcpa(1+67/256))/2
data8 0x3fbe530effe71012    //log(1/frcpa(1+68/256))/2
data8 0x3fbeb89a1648b971    //log(1/frcpa(1+69/256))/2
data8 0x3fbf1e75fadf9bde    //log(1/frcpa(1+70/256))/2
data8 0x3fbf84a32ead7c35    //log(1/frcpa(1+71/256))/2
data8 0x3fbfeb2233ea07cd    //log(1/frcpa(1+72/256))/2
data8 0x3fc028f9c7035c1c    //log(1/frcpa(1+73/256))/2
data8 0x3fc05c8be0d9635a    //log(1/frcpa(1+74/256))/2
data8 0x3fc085eb8f8ae797    //log(1/frcpa(1+75/256))/2
data8 0x3fc0b9c8e32d1911    //log(1/frcpa(1+76/256))/2
data8 0x3fc0edd060b78081    //log(1/frcpa(1+77/256))/2
data8 0x3fc122024cf0063f    //log(1/frcpa(1+78/256))/2
data8 0x3fc14be2927aecd4    //log(1/frcpa(1+79/256))/2
data8 0x3fc180618ef18adf    //log(1/frcpa(1+80/256))/2
data8 0x3fc1b50bbe2fc63b    //log(1/frcpa(1+81/256))/2
data8 0x3fc1df4cc7cf242d    //log(1/frcpa(1+82/256))/2
data8 0x3fc214456d0eb8d4    //log(1/frcpa(1+83/256))/2
data8 0x3fc23ec5991eba49    //log(1/frcpa(1+84/256))/2
data8 0x3fc2740d9f870afb    //log(1/frcpa(1+85/256))/2
data8 0x3fc29ecdabcdfa04    //log(1/frcpa(1+86/256))/2
data8 0x3fc2d46602adccee    //log(1/frcpa(1+87/256))/2
data8 0x3fc2ff66b04ea9d4    //log(1/frcpa(1+88/256))/2
data8 0x3fc335504b355a37    //log(1/frcpa(1+89/256))/2
data8 0x3fc360925ec44f5d    //log(1/frcpa(1+90/256))/2
data8 0x3fc38bf1c3337e75    //log(1/frcpa(1+91/256))/2
data8 0x3fc3c25277333184    //log(1/frcpa(1+92/256))/2
data8 0x3fc3edf463c1683e    //log(1/frcpa(1+93/256))/2
data8 0x3fc419b423d5e8c7    //log(1/frcpa(1+94/256))/2
data8 0x3fc44591e0539f49    //log(1/frcpa(1+95/256))/2
data8 0x3fc47c9175b6f0ad    //log(1/frcpa(1+96/256))/2
data8 0x3fc4a8b341552b09    //log(1/frcpa(1+97/256))/2
data8 0x3fc4d4f3908901a0    //log(1/frcpa(1+98/256))/2
data8 0x3fc501528da1f968    //log(1/frcpa(1+99/256))/2
data8 0x3fc52dd06347d4f6    //log(1/frcpa(1+100/256))/2
data8 0x3fc55a6d3c7b8a8a    //log(1/frcpa(1+101/256))/2
data8 0x3fc5925d2b112a59    //log(1/frcpa(1+102/256))/2
data8 0x3fc5bf406b543db2    //log(1/frcpa(1+103/256))/2
data8 0x3fc5ec433d5c35ae    //log(1/frcpa(1+104/256))/2
data8 0x3fc61965cdb02c1f    //log(1/frcpa(1+105/256))/2
data8 0x3fc646a84935b2a2    //log(1/frcpa(1+106/256))/2
data8 0x3fc6740add31de94    //log(1/frcpa(1+107/256))/2
data8 0x3fc6a18db74a58c5    //log(1/frcpa(1+108/256))/2
data8 0x3fc6cf31058670ec    //log(1/frcpa(1+109/256))/2
data8 0x3fc6f180e852f0ba    //log(1/frcpa(1+110/256))/2
data8 0x3fc71f5d71b894f0    //log(1/frcpa(1+111/256))/2
data8 0x3fc74d5aefd66d5c    //log(1/frcpa(1+112/256))/2
data8 0x3fc77b79922bd37e    //log(1/frcpa(1+113/256))/2
data8 0x3fc7a9b9889f19e2    //log(1/frcpa(1+114/256))/2
data8 0x3fc7d81b037eb6a6    //log(1/frcpa(1+115/256))/2
data8 0x3fc8069e33827231    //log(1/frcpa(1+116/256))/2
data8 0x3fc82996d3ef8bcb    //log(1/frcpa(1+117/256))/2
data8 0x3fc85855776dcbfb    //log(1/frcpa(1+118/256))/2
data8 0x3fc8873658327ccf    //log(1/frcpa(1+119/256))/2
data8 0x3fc8aa75973ab8cf    //log(1/frcpa(1+120/256))/2
data8 0x3fc8d992dc8824e5    //log(1/frcpa(1+121/256))/2
data8 0x3fc908d2ea7d9512    //log(1/frcpa(1+122/256))/2
data8 0x3fc92c59e79c0e56    //log(1/frcpa(1+123/256))/2
data8 0x3fc95bd750ee3ed3    //log(1/frcpa(1+124/256))/2
data8 0x3fc98b7811a3ee5b    //log(1/frcpa(1+125/256))/2
data8 0x3fc9af47f33d406c    //log(1/frcpa(1+126/256))/2
data8 0x3fc9df270c1914a8    //log(1/frcpa(1+127/256))/2
data8 0x3fca0325ed14fda4    //log(1/frcpa(1+128/256))/2
data8 0x3fca33440224fa79    //log(1/frcpa(1+129/256))/2
data8 0x3fca57725e80c383    //log(1/frcpa(1+130/256))/2
data8 0x3fca87d0165dd199    //log(1/frcpa(1+131/256))/2
data8 0x3fcaac2e6c03f896    //log(1/frcpa(1+132/256))/2
data8 0x3fcadccc6fdf6a81    //log(1/frcpa(1+133/256))/2
data8 0x3fcb015b3eb1e790    //log(1/frcpa(1+134/256))/2
data8 0x3fcb323a3a635948    //log(1/frcpa(1+135/256))/2
data8 0x3fcb56fa04462909    //log(1/frcpa(1+136/256))/2
data8 0x3fcb881aa659bc93    //log(1/frcpa(1+137/256))/2
data8 0x3fcbad0bef3db165    //log(1/frcpa(1+138/256))/2
data8 0x3fcbd21297781c2f    //log(1/frcpa(1+139/256))/2
data8 0x3fcc039236f08819    //log(1/frcpa(1+140/256))/2
data8 0x3fcc28cb1e4d32fd    //log(1/frcpa(1+141/256))/2
data8 0x3fcc4e19b84723c2    //log(1/frcpa(1+142/256))/2
data8 0x3fcc7ff9c74554c9    //log(1/frcpa(1+143/256))/2
data8 0x3fcca57b64e9db05    //log(1/frcpa(1+144/256))/2
data8 0x3fcccb130a5cebb0    //log(1/frcpa(1+145/256))/2
data8 0x3fccf0c0d18f326f    //log(1/frcpa(1+146/256))/2
data8 0x3fcd232075b5a201    //log(1/frcpa(1+147/256))/2
data8 0x3fcd490246defa6b    //log(1/frcpa(1+148/256))/2
data8 0x3fcd6efa918d25cd    //log(1/frcpa(1+149/256))/2
data8 0x3fcd9509707ae52f    //log(1/frcpa(1+150/256))/2
data8 0x3fcdbb2efe92c554    //log(1/frcpa(1+151/256))/2
data8 0x3fcdee2f3445e4af    //log(1/frcpa(1+152/256))/2
data8 0x3fce148a1a2726ce    //log(1/frcpa(1+153/256))/2
data8 0x3fce3afc0a49ff40    //log(1/frcpa(1+154/256))/2
data8 0x3fce6185206d516e    //log(1/frcpa(1+155/256))/2
data8 0x3fce882578823d52    //log(1/frcpa(1+156/256))/2
data8 0x3fceaedd2eac990c    //log(1/frcpa(1+157/256))/2
data8 0x3fced5ac5f436be3    //log(1/frcpa(1+158/256))/2
data8 0x3fcefc9326d16ab9    //log(1/frcpa(1+159/256))/2
data8 0x3fcf2391a2157600    //log(1/frcpa(1+160/256))/2
data8 0x3fcf4aa7ee03192d    //log(1/frcpa(1+161/256))/2
data8 0x3fcf71d627c30bb0    //log(1/frcpa(1+162/256))/2
data8 0x3fcf991c6cb3b379    //log(1/frcpa(1+163/256))/2
data8 0x3fcfc07ada69a910    //log(1/frcpa(1+164/256))/2
data8 0x3fcfe7f18eb03d3e    //log(1/frcpa(1+165/256))/2
data8 0x3fd007c053c5002e    //log(1/frcpa(1+166/256))/2
data8 0x3fd01b942198a5a1    //log(1/frcpa(1+167/256))/2
data8 0x3fd02f74400c64eb    //log(1/frcpa(1+168/256))/2
data8 0x3fd04360be7603ad    //log(1/frcpa(1+169/256))/2
data8 0x3fd05759ac47fe34    //log(1/frcpa(1+170/256))/2
data8 0x3fd06b5f1911cf52    //log(1/frcpa(1+171/256))/2
data8 0x3fd078bf0533c568    //log(1/frcpa(1+172/256))/2
data8 0x3fd08cd9687e7b0e    //log(1/frcpa(1+173/256))/2
data8 0x3fd0a10074cf9019    //log(1/frcpa(1+174/256))/2
data8 0x3fd0b5343a234477    //log(1/frcpa(1+175/256))/2
data8 0x3fd0c974c89431ce    //log(1/frcpa(1+176/256))/2
data8 0x3fd0ddc2305b9886    //log(1/frcpa(1+177/256))/2
data8 0x3fd0eb524bafc918    //log(1/frcpa(1+178/256))/2
data8 0x3fd0ffb54213a476    //log(1/frcpa(1+179/256))/2
data8 0x3fd114253da97d9f    //log(1/frcpa(1+180/256))/2
data8 0x3fd128a24f1d9aff    //log(1/frcpa(1+181/256))/2
data8 0x3fd1365252bf0865    //log(1/frcpa(1+182/256))/2
data8 0x3fd14ae558b4a92d    //log(1/frcpa(1+183/256))/2
data8 0x3fd15f85a19c765b    //log(1/frcpa(1+184/256))/2
data8 0x3fd16d4d38c119fa    //log(1/frcpa(1+185/256))/2
data8 0x3fd18203c20dd133    //log(1/frcpa(1+186/256))/2
data8 0x3fd196c7bc4b1f3b    //log(1/frcpa(1+187/256))/2
data8 0x3fd1a4a738b7a33c    //log(1/frcpa(1+188/256))/2
data8 0x3fd1b981c0c9653d    //log(1/frcpa(1+189/256))/2
data8 0x3fd1ce69e8bb106b    //log(1/frcpa(1+190/256))/2
data8 0x3fd1dc619de06944    //log(1/frcpa(1+191/256))/2
data8 0x3fd1f160a2ad0da4    //log(1/frcpa(1+192/256))/2
data8 0x3fd2066d7740737e    //log(1/frcpa(1+193/256))/2
data8 0x3fd2147dba47a394    //log(1/frcpa(1+194/256))/2
data8 0x3fd229a1bc5ebac3    //log(1/frcpa(1+195/256))/2
data8 0x3fd237c1841a502e    //log(1/frcpa(1+196/256))/2
data8 0x3fd24cfce6f80d9a    //log(1/frcpa(1+197/256))/2
data8 0x3fd25b2c55cd5762    //log(1/frcpa(1+198/256))/2
data8 0x3fd2707f4d5f7c41    //log(1/frcpa(1+199/256))/2
data8 0x3fd285e0842ca384    //log(1/frcpa(1+200/256))/2
data8 0x3fd294294708b773    //log(1/frcpa(1+201/256))/2
data8 0x3fd2a9a2670aff0c    //log(1/frcpa(1+202/256))/2
data8 0x3fd2b7fb2c8d1cc1    //log(1/frcpa(1+203/256))/2
data8 0x3fd2c65a6395f5f5    //log(1/frcpa(1+204/256))/2
data8 0x3fd2dbf557b0df43    //log(1/frcpa(1+205/256))/2
data8 0x3fd2ea64c3f97655    //log(1/frcpa(1+206/256))/2
data8 0x3fd3001823684d73    //log(1/frcpa(1+207/256))/2
data8 0x3fd30e97e9a8b5cd    //log(1/frcpa(1+208/256))/2
data8 0x3fd32463ebdd34ea    //log(1/frcpa(1+209/256))/2
data8 0x3fd332f4314ad796    //log(1/frcpa(1+210/256))/2
data8 0x3fd348d90e7464d0    //log(1/frcpa(1+211/256))/2
data8 0x3fd35779f8c43d6e    //log(1/frcpa(1+212/256))/2
data8 0x3fd36621961a6a99    //log(1/frcpa(1+213/256))/2
data8 0x3fd37c299f3c366a    //log(1/frcpa(1+214/256))/2
data8 0x3fd38ae2171976e7    //log(1/frcpa(1+215/256))/2
data8 0x3fd399a157a603e7    //log(1/frcpa(1+216/256))/2
data8 0x3fd3afccfe77b9d1    //log(1/frcpa(1+217/256))/2
data8 0x3fd3be9d503533b5    //log(1/frcpa(1+218/256))/2
data8 0x3fd3cd7480b4a8a3    //log(1/frcpa(1+219/256))/2
data8 0x3fd3e3c43918f76c    //log(1/frcpa(1+220/256))/2
data8 0x3fd3f2acb27ed6c7    //log(1/frcpa(1+221/256))/2
data8 0x3fd4019c2125ca93    //log(1/frcpa(1+222/256))/2
data8 0x3fd4181061389722    //log(1/frcpa(1+223/256))/2
data8 0x3fd42711518df545    //log(1/frcpa(1+224/256))/2
data8 0x3fd436194e12b6bf    //log(1/frcpa(1+225/256))/2
data8 0x3fd445285d68ea69    //log(1/frcpa(1+226/256))/2
data8 0x3fd45bcc464c893a    //log(1/frcpa(1+227/256))/2
data8 0x3fd46aed21f117fc    //log(1/frcpa(1+228/256))/2
data8 0x3fd47a1527e8a2d3    //log(1/frcpa(1+229/256))/2
data8 0x3fd489445efffccc    //log(1/frcpa(1+230/256))/2
data8 0x3fd4a018bcb69835    //log(1/frcpa(1+231/256))/2
data8 0x3fd4af5a0c9d65d7    //log(1/frcpa(1+232/256))/2
data8 0x3fd4bea2a5bdbe87    //log(1/frcpa(1+233/256))/2
data8 0x3fd4cdf28f10ac46    //log(1/frcpa(1+234/256))/2
data8 0x3fd4dd49cf994058    //log(1/frcpa(1+235/256))/2
data8 0x3fd4eca86e64a684    //log(1/frcpa(1+236/256))/2
data8 0x3fd503c43cd8eb68    //log(1/frcpa(1+237/256))/2
data8 0x3fd513356667fc57    //log(1/frcpa(1+238/256))/2
data8 0x3fd522ae0738a3d8    //log(1/frcpa(1+239/256))/2
data8 0x3fd5322e26867857    //log(1/frcpa(1+240/256))/2
data8 0x3fd541b5cb979809    //log(1/frcpa(1+241/256))/2
data8 0x3fd55144fdbcbd62    //log(1/frcpa(1+242/256))/2
data8 0x3fd560dbc45153c7    //log(1/frcpa(1+243/256))/2
data8 0x3fd5707a26bb8c66    //log(1/frcpa(1+244/256))/2
data8 0x3fd587f60ed5b900    //log(1/frcpa(1+245/256))/2
data8 0x3fd597a7977c8f31    //log(1/frcpa(1+246/256))/2
data8 0x3fd5a760d634bb8b    //log(1/frcpa(1+247/256))/2
data8 0x3fd5b721d295f10f    //log(1/frcpa(1+248/256))/2
data8 0x3fd5c6ea94431ef9    //log(1/frcpa(1+249/256))/2
data8 0x3fd5d6bb22ea86f6    //log(1/frcpa(1+250/256))/2
data8 0x3fd5e6938645d390    //log(1/frcpa(1+251/256))/2
data8 0x3fd5f673c61a2ed2    //log(1/frcpa(1+252/256))/2
data8 0x3fd6065bea385926    //log(1/frcpa(1+253/256))/2
data8 0x3fd6164bfa7cc06b    //log(1/frcpa(1+254/256))/2
data8 0x3fd62643fecf9743    //log(1/frcpa(1+255/256))/2
LOCAL_OBJECT_END(atanhf_data2)


.section .text
GLOBAL_LIBM_ENTRY(atanhf)

{ .mfi
      getf.exp      rArgSExpb = f8
      fclass.m      p9,p0 = f8, 0x0b        // is arg denormal ?
      mov           rExpbMask = 0x1ffff
}
{ .mfi
      addl          DataPtr = @ltoff(atanhf_data), gp
      fnma.s1       fOneMx = f8, f1, f1     // 1 - x
      mov           rBias = 0xffff
}
;;

{ .mfi
      nop.m         0
      fclass.m      p7,p0 = f8, 0xc7        // is arg NaN or +/-0 ?
      mov           rNearZeroBound = 0xffeb // 2^(-20)
}
{ .mfi
      ld8           DataPtr = [DataPtr]
      fma.s1        fOnePx = f8, f1, f1     // 1 + x
      nop.i         0
}
;;

{ .mfb
      nop.m         0
      fnorm.s1      fNormX = f8                     // Normalize x
(p9)  br.cond.spnt  ATANH_UNORM                     // Branch if x=unorm
}
;;

ATANH_COMMON:
// Return here if x=unorm and not denorm
{ .mfi
      ldfpd         fP3, fP2 = [DataPtr], 16
      fma.s1        fX2 = f8, f8, f0        // x^2
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fma.s.s0      f8 =  f8,f1,f8          // NaN or +/-0
(p7)  br.ret.spnt   b0
}
;;

{ .mfi
      ldfpd         fP1, fHalf = [DataPtr], 16
      frcpa.s1      fRcpM, p9 = f1, fOneMx  // rcpm = frcpa(1 - x)
      nop.i         0
}
;;

{ .mfi
      getf.exp      rExpbm = fOneMx
      frcpa.s1      fRcpP, p0 = f1, fOnePx  // rcpp = frcpa(1 + x)
      // biased exponent
      and           rArgExpb = rArgSExpb, rExpbMask
}
;;

{ .mmi
      getf.exp      rExpbp = fOnePx
      // is |x| < 2^(-20) ?
      cmp.gt        p8,p0 = rNearZeroBound, rArgExpb
      cmp.ge        p6,p0 = rArgExpb, rBias // is |x| >= 1 ?
}
;;

{ .mmb
      getf.sig      rSigm = fOneMx
      nop.m         0
(p6)  br.cond.spnt  atanhf_ge_one
}
;;

{ .mfb
      getf.sig      rSigp = fOnePx
(p8)  fma.s.s0      f8 =  fX2, f8, f8  // x + x^3
(p8)  br.ret.spnt   b0                 // Exit for MAX_DENORM_ABS < |x| < 2^-20
}
;;

{ .mfi
      ldfd          fLog2 = [DataPtr], 16
      fms.s1        fRm = fRcpM, fOneMx, f1 // rm = rcpm * (1 - x) - 1
      nop.i         0
}
;;

{ .mmf
      // (1 - x) is always positive here and we need not mask sign bit
      sub           rNm = rExpbm, rBias
      // (1 + x) is always positive here and we need not mask sign bit
      sub           rNp = rExpbp, rBias
      fms.s1        fRp = fRcpP, fOnePx, f1 // rp = rcpp * (1 + x) - 1
}
;;

{ .mmi
      setf.sig      fN4CvtM = rNm
      setf.sig      fN4CvtP = rNp
      extr.u        rIndm = rSigm,55,8                // Extract 8 bits
}
;;

{ .mmi
      shladd        RcpTablePtrM = rIndm, 3, DataPtr
      nop.m         0
      extr.u        rIndp = rSigp,55,8                // Extract 8 bits
}
;;

{ .mmi
      ldfd          fLogTm = [RcpTablePtrM]
      shladd        RcpTablePtrP = rIndp, 3, DataPtr
      nop.i         0
}
;;

{ .mfi
      ldfd          fLogTp = [RcpTablePtrP]
      fma.s1        fRm2 = fRm, fRm, f0     // rm^2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        fP32m = fP3, fRm, fP2   // P3*rm + P2
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fRp2 = fRp, fRp, f0     // rp^2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        fP10m = fP1, fRm, fHalf   // P1*rm + 1
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fP32p = fP3, fRp, fP2   // P3*rp + P2
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        fP10p = fP1, fRp, fHalf   // P1*rp + 1
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fcvt.xf       fNm = fN4CvtM
      nop.i         0
}
{ .mfi
      nop.m         0
      fcvt.xf       fNp = fN4CvtP
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // (P3*rm + P2)*rm^2 + (P1*rm + 1)
      fma.s1        fP32m = fP32m, fRm2, fP10m
      nop.i         0
}
{ .mfi
      nop.m         0
      // (P3*rp + P2)*rp^2 + (P1*rp + 1)
      fma.s1        fP32p = fP32p, fRp2, fP10p
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // Nm*ln(2)/2 + Tm/2
      fma.s1        fLogTm = fNm, fLog2, fLogTm
      nop.i         0
}
{ .mfi
      nop.m         0
      // Np*ln(2)/2 + Tp/2
      fma.s1        fLogTp = fNp, fLog2, fLogTp
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // ((P3*rm + P2)*rm^2 + (P3*rm + 1))*0.5*rm + (Nm*ln(2)/2 + Tm/2)
      fma.d.s1      fP32m = fP32m, fRm, fLogTm
      nop.i         0
}
{ .mfi
      nop.m         0
      // ((P3*rp + P2)*rp^2 + (P3*rp + 1))*0.5*rp + (Np*ln(2)/2 + Tp/2)
      fma.d.s1      fP32p = fP32p, fRp, fLogTp
      nop.i         0
}
;;

{ .mfb
      nop.m         0
      // atanhf(x) = 0.5 * (log(1 + x) - log(1 - x))
      fnma.s.s0     f8 = fP32m, f1, fP32p
      br.ret.sptk   b0                      // Exit for 2^(-20) <= |x| < 1.0
}
;;


ATANH_UNORM:
// Here if x=unorm
{ .mfi
      getf.exp      rArgSExpb = fNormX           // Recompute if x unorm
      fclass.m      p0,p9 = fNormX, 0x0b         // Test x denorm
      nop.i         0
}
;;

{ .mfb
      nop.m         0
      fcmp.lt.s0    p10,p11 = f8, f0      // Set denormal flag
(p9)  br.cond.sptk  ATANH_COMMON          // Continue if x unorm and not denorm
}
;;

.pred.rel "mutex",p6,p7
{ .mfi
      nop.m         0
(p6)  fnma.s.s0     f8 = f8,f8,f8                // Result x-x^2 if x=-denorm
      nop.i         0
}
{ .mfb
      nop.m         0
(p7)  fma.s.s0      f8 = f8,f8,f8                // Result x+x^2 if x=+denorm
      br.ret.spnt   b0                           // Exit if denorm
}
;;

// Here if |x| >= 1.0
atanhf_ge_one:
{ .mfi
      alloc         r32 = ar.pfs,1,3,4,0
      fmerge.s      fArgAbs = f0, f8        // Form |x|
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fmerge.s      f10 = f8, f8            // Save input for error call
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fcmp.eq.s1    p6,p7 = fArgAbs, f1     // Test for |x| = 1.0
      nop.i         0
}
;;

// Set error tag and result, and raise invalid flag if |x| > 1.0
{ .mfi
(p7)  mov           atanh_GR_tag = 133
(p7)  frcpa.s0      f8, p0 = f0, f0         // Get QNaN, and raise invalid
      nop.i         0
}
;;

// Set error tag and result, and raise Z flag if |x| = 1.0
{ .mfi
      nop.m         0
(p6)  frcpa.s0      fRm, p0 = f1, f0        // Get inf, and raise Z flag
      nop.i         0
}
;;

{ .mfb
(p6)  mov           atanh_GR_tag = 134
(p6)  fmerge.s      f8 = f8, fRm            // result is +-inf
      br.cond.sptk  __libm_error_region     // Exit if |x| >= 1.0
}
;;

GLOBAL_LIBM_END(atanhf)
libm_alias_float_other (atanh, atanh)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue

{ .mfi
      add           GR_Parameter_Y=-32,sp   // Parameter 2 value
      nop.f         0
.save   ar.pfs,GR_SAVE_PFS
      mov           GR_SAVE_PFS=ar.pfs      // Save ar.pfs
}
{ .mfi
.fframe 64
      add sp=-64,sp                         // Create new stack
      nop.f 0
      mov GR_SAVE_GP=gp                     // Save gp
};;

{ .mmi
      stfs [GR_Parameter_Y] = f1,16         // STORE Parameter 2 on stack
      add GR_Parameter_X = 16,sp            // Parameter 1 address
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                     // Save b0
};;

.body
{ .mib
      stfs [GR_Parameter_X] = f10           // STORE Parameter 1 on stack
      // Parameter 3 address
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y
      nop.b 0
}
{ .mib
      stfs [GR_Parameter_Y] = f8            // STORE Parameter 3 on stack
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
