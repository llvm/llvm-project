.file "atanh.s"


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
// ==============================================================
// History
// ==============================================================
// 05/03/01  Initial version
// 05/20/02  Cleaned up namespace and sf0 syntax
// 02/06/03  Reordered header: .section, .global, .proc, .align
// 05/26/03  Improved performance, fixed to handle unorms
// 03/31/05  Reformatted delimiters between data tables
//
// API
// ==============================================================
// double atanh(double)
//
// Overview of operation
// ==============================================================
//
// There are 7 paths:
// 1. x = +/-0.0
//    Return atanh(x) = +/-0.0
//
// 2. 0.0 < |x| < 1/4
//    Return atanh(x) = Po2l(x),
//    where Po2l(x) = (((((((((C9*x^2 + C8)*x^2 + C7)*x^2 + C6)*x^2 +
//          C5)*x^2 + C4)*x^2 + C3)*x^2 + C2)*x^2 + C1)* x^2 + C0)*x^3 + x
// 3. 1/4 <= |x| < 1
//    Return atanh(x) = sign(x) * log((1 + |x|)/(1 - |x|))
//    To compute (1 + |x|)/(1 - |x|) modified Newton Raphson method is used
//    (3 iterations)
//    Algorithm description for log function see below.
//
// 4. |x| = 1
//    Return atanh(x) = sign(x) * +INF
//
// 5. 1 < |x| <= +INF
//    Return atanh(x) = QNaN
//
// 6. x = [S,Q]NaN
//    Return atanh(x) = QNaN
//
// 7. x = denormal
//    Return atanh(x) = x
//
//==============================================================
// Algorithm Description for log(x) function
// Below we are using the fact that inequality x - 1.0 > 2^(-6) is always true
// for this atanh implementation
//
// Consider  x = 2^N 1.f1 f2 f3 f4...f63
// Log(x) = log(x * frcpa(x) / frcpa(x))
//        = log(x * frcpa(x)) + log(1/frcpa(x))
//        = log(x * frcpa(x)) - log(frcpa(x))
//
// frcpa(x)       = 2^-N * frcpa(1.f1 f2 ... f63)
//
// -log(frcpa(x)) = -log(C)
//                = -log(2^-N) - log(frcpa(1.f1 f2 ... f63))
//
// -log(frcpa(x)) = -log(C)
//                = N*log2 - log(frcpa(1.f1 f2 ... f63))
//
//
// Log(x) = log(1/frcpa(x)) + log(frcpa(x) x)
//
// Log(x) = N*log2 + log(1./frcpa(1.f1 f2 ... f63)) + log(x * frcpa(x))
// Log(x) = N*log2 + T                              + log(frcpa(x) x)
//
// Log(x) = N*log2 + T                              + log(C * x)
//
// C * x = 1 + r
//
// Log(x) = N*log2 + T + log(1 + r)
// Log(x) = N*log2 + T + Series(r - r^2/2 + r^3/3 - r^4/4 + ...)
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
// Form rseries = r + P1*r^2 + P2*r^3 + P3*r^4 + P4*r^5 + P5*r^6
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
//
//
// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f32 -> f77

// General registers used:
// r14 -> r27, r33 -> r39

// Predicate registers used:
// p6 -> p14

// p10, p11      to indicate is argument positive or negative
// p12           to filter out case when x = [Q,S]NaN or +/-0
// p13           to filter out case when x = denormal
// p6, p7        to filter out case when |x| >= 1
// p8            to filter out case when |x| < 1/4

// Assembly macros
//==============================================================
Data2Ptr              = r14
Data3Ptr              = r15
RcpTablePtr           = r16
rExpbMask             = r17
rBias                 = r18
rNearZeroBound        = r19
rArgSExpb             = r20
rArgExpb              = r21
rSExpb                = r22
rExpb                 = r23
rSig                  = r24
rN                    = r25
rInd                  = r26
DataPtr               = r27

GR_SAVE_B0            = r33
GR_SAVE_GP            = r34
GR_SAVE_PFS           = r35

GR_Parameter_X        = r36
GR_Parameter_Y        = r37
GR_Parameter_RESULT   = r38
atanh_GR_tag          = r39

//==============================================================
fAbsX                 = f32
fOneMx                = f33
fOnePx                = f34
fY                    = f35
fR                    = f36
fR2                   = f37
fR3                   = f38
fRcp                  = f39
fY4Rcp                = f40
fRcp0                 = f41
fRcp0n                = f42
fRcp1                 = f43
fRcp2                 = f44
fRcp3                 = f45
fN4Cvt                = f46
fN                    = f47
fY2                   = f48
fLog2                 = f49
fLogT                 = f50
fLogT_N               = f51
fX2                   = f52
fX3                   = f53
fX4                   = f54
fX8                   = f55
fP0                   = f56
fP5                   = f57
fP4                   = f58
fP3                   = f59
fP2                   = f60
fP1                   = f61
fNormX                = f62
fC9                   = f63
fC8                   = f64
fC7                   = f65
fC6                   = f66
fC5                   = f67
fC4                   = f68
fC3                   = f69
fC2                   = f70
fC1                   = f71
fC0                   = f72
fP98                  = f73
fP76                  = f74
fP54                  = f75
fP32                  = f76
fP10                  = f77

// Data tables
//==============================================================
RODATA
.align 16

LOCAL_OBJECT_START(atanh_data)
data8 0xBFC5555DA7212371              //   P5
data8 0x3FC999A19EEF5826              //   P4
data8 0xBFCFFFFFFFFEF009              //   P3
data8 0x3FD555555554ECB2              //   P2
data8 0xBFE0000000000000              //   P1 = -0.5
data8 0x0000000000000000              //   pad
data8 0xb17217f7d1cf79ac , 0x00003ffd //   0.5*log(2)
data8 0x0000000000000000 , 0x00000000 //   pad to eliminate bank conflicts
LOCAL_OBJECT_END(atanh_data)

LOCAL_OBJECT_START(atanh_data_2)
data8 0x8649FB89D3AD51FB , 0x00003FFB //   C9
data8 0xCC10AABEF160077A , 0x00003FFA //   C8
data8 0xF1EDB99AC0819CE2 , 0x00003FFA //   C7
data8 0x8881E53A809AD24D , 0x00003FFB //   C6
data8 0x9D8A116EF212F271 , 0x00003FFB //   C5
data8 0xBA2E8A6D1D756453 , 0x00003FFB //   C4
data8 0xE38E38E7A0945692 , 0x00003FFB //   C3
data8 0x924924924536891A , 0x00003FFC //   C2
data8 0xCCCCCCCCCCD08D51 , 0x00003FFC //   C1
data8 0xAAAAAAAAAAAAAA0C , 0x00003FFD //   C0
LOCAL_OBJECT_END(atanh_data_2)


LOCAL_OBJECT_START(atanh_data_3)
data8 0x80200aaeac44ef38 , 0x00003ff5 //   log(1/frcpa(1+0/2^-8))/2
//
data8 0xc09090a2c35aa070 , 0x00003ff6 //   log(1/frcpa(1+1/2^-8))/2
data8 0xa0c94fcb41977c75 , 0x00003ff7 //   log(1/frcpa(1+2/2^-8))/2
data8 0xe18b9c263af83301 , 0x00003ff7 //   log(1/frcpa(1+3/2^-8))/2
data8 0x8d35c8d6399c30ea , 0x00003ff8 //   log(1/frcpa(1+4/2^-8))/2
data8 0xadd4d2ecd601cbb8 , 0x00003ff8 //   log(1/frcpa(1+5/2^-8))/2
//
data8 0xce95403a192f9f01 , 0x00003ff8 //   log(1/frcpa(1+6/2^-8))/2
data8 0xeb59392cbcc01096 , 0x00003ff8 //   log(1/frcpa(1+7/2^-8))/2
data8 0x862c7d0cefd54c5d , 0x00003ff9 //   log(1/frcpa(1+8/2^-8))/2
data8 0x94aa63c65e70d499 , 0x00003ff9 //   log(1/frcpa(1+9/2^-8))/2
data8 0xa54a696d4b62b382 , 0x00003ff9 //   log(1/frcpa(1+10/2^-8))/2
//
data8 0xb3e4a796a5dac208 , 0x00003ff9 //   log(1/frcpa(1+11/2^-8))/2
data8 0xc28c45b1878340a9 , 0x00003ff9 //   log(1/frcpa(1+12/2^-8))/2
data8 0xd35c55f39d7a6235 , 0x00003ff9 //   log(1/frcpa(1+13/2^-8))/2
data8 0xe220f037b954f1f5 , 0x00003ff9 //   log(1/frcpa(1+14/2^-8))/2
data8 0xf0f3389b036834f3 , 0x00003ff9 //   log(1/frcpa(1+15/2^-8))/2
//
data8 0xffd3488d5c980465 , 0x00003ff9 //   log(1/frcpa(1+16/2^-8))/2
data8 0x87609ce2ed300490 , 0x00003ffa //   log(1/frcpa(1+17/2^-8))/2
data8 0x8ede9321e8c85927 , 0x00003ffa //   log(1/frcpa(1+18/2^-8))/2
data8 0x96639427f2f8e2f4 , 0x00003ffa //   log(1/frcpa(1+19/2^-8))/2
data8 0x9defad3e8f73217b , 0x00003ffa //   log(1/frcpa(1+20/2^-8))/2
//
data8 0xa582ebd50097029c , 0x00003ffa //   log(1/frcpa(1+21/2^-8))/2
data8 0xac06dbe75ab80fee , 0x00003ffa //   log(1/frcpa(1+22/2^-8))/2
data8 0xb3a78449b2d3ccca , 0x00003ffa //   log(1/frcpa(1+23/2^-8))/2
data8 0xbb4f79635ab46bb2 , 0x00003ffa //   log(1/frcpa(1+24/2^-8))/2
data8 0xc2fec93a83523f3f , 0x00003ffa //   log(1/frcpa(1+25/2^-8))/2
//
data8 0xc99af2eaca4c4571 , 0x00003ffa //   log(1/frcpa(1+26/2^-8))/2
data8 0xd1581106472fa653 , 0x00003ffa //   log(1/frcpa(1+27/2^-8))/2
data8 0xd8002560d4355f2e , 0x00003ffa //   log(1/frcpa(1+28/2^-8))/2
data8 0xdfcb43b4fe508632 , 0x00003ffa //   log(1/frcpa(1+29/2^-8))/2
data8 0xe67f6dff709d4119 , 0x00003ffa //   log(1/frcpa(1+30/2^-8))/2
//
data8 0xed393b1c22351280 , 0x00003ffa //   log(1/frcpa(1+31/2^-8))/2
data8 0xf5192bff087bcc35 , 0x00003ffa //   log(1/frcpa(1+32/2^-8))/2
data8 0xfbdf4ff6dfef2fa3 , 0x00003ffa //   log(1/frcpa(1+33/2^-8))/2
data8 0x81559a97f92f9cc7 , 0x00003ffb //   log(1/frcpa(1+34/2^-8))/2
data8 0x84be72bce90266e8 , 0x00003ffb //   log(1/frcpa(1+35/2^-8))/2
//
data8 0x88bc74113f23def2 , 0x00003ffb //   log(1/frcpa(1+36/2^-8))/2
data8 0x8c2ba3edf6799d11 , 0x00003ffb //   log(1/frcpa(1+37/2^-8))/2
data8 0x8f9dc92f92ea08b1 , 0x00003ffb //   log(1/frcpa(1+38/2^-8))/2
data8 0x9312e8f36efab5a7 , 0x00003ffb //   log(1/frcpa(1+39/2^-8))/2
data8 0x968b08643409ceb6 , 0x00003ffb //   log(1/frcpa(1+40/2^-8))/2
//
data8 0x9a062cba08a1708c , 0x00003ffb //   log(1/frcpa(1+41/2^-8))/2
data8 0x9d845b3abf95485c , 0x00003ffb //   log(1/frcpa(1+42/2^-8))/2
data8 0xa06fd841bc001bb4 , 0x00003ffb //   log(1/frcpa(1+43/2^-8))/2
data8 0xa3f3a74652fbe0db , 0x00003ffb //   log(1/frcpa(1+44/2^-8))/2
data8 0xa77a8fb2336f20f5 , 0x00003ffb //   log(1/frcpa(1+45/2^-8))/2
//
data8 0xab0497015d28b0a0 , 0x00003ffb //   log(1/frcpa(1+46/2^-8))/2
data8 0xae91c2be6ba6a615 , 0x00003ffb //   log(1/frcpa(1+47/2^-8))/2
data8 0xb189d1b99aebb20b , 0x00003ffb //   log(1/frcpa(1+48/2^-8))/2
data8 0xb51cced5de9c1b2c , 0x00003ffb //   log(1/frcpa(1+49/2^-8))/2
data8 0xb819bee9e720d42f , 0x00003ffb //   log(1/frcpa(1+50/2^-8))/2
//
data8 0xbbb2a0947b093a5d , 0x00003ffb //   log(1/frcpa(1+51/2^-8))/2
data8 0xbf4ec1505811684a , 0x00003ffb //   log(1/frcpa(1+52/2^-8))/2
data8 0xc2535bacfa8975ff , 0x00003ffb //   log(1/frcpa(1+53/2^-8))/2
data8 0xc55a3eafad187eb8 , 0x00003ffb //   log(1/frcpa(1+54/2^-8))/2
data8 0xc8ff2484b2c0da74 , 0x00003ffb //   log(1/frcpa(1+55/2^-8))/2
//
data8 0xcc0b1a008d53ab76 , 0x00003ffb //   log(1/frcpa(1+56/2^-8))/2
data8 0xcfb6203844b3209b , 0x00003ffb //   log(1/frcpa(1+57/2^-8))/2
data8 0xd2c73949a47a19f5 , 0x00003ffb //   log(1/frcpa(1+58/2^-8))/2
data8 0xd5daae18b49d6695 , 0x00003ffb //   log(1/frcpa(1+59/2^-8))/2
data8 0xd8f08248cf7e8019 , 0x00003ffb //   log(1/frcpa(1+60/2^-8))/2
//
data8 0xdca7749f1b3e540e , 0x00003ffb //   log(1/frcpa(1+61/2^-8))/2
data8 0xdfc28e033aaaf7c7 , 0x00003ffb //   log(1/frcpa(1+62/2^-8))/2
data8 0xe2e012a5f91d2f55 , 0x00003ffb //   log(1/frcpa(1+63/2^-8))/2
data8 0xe600064ed9e292a8 , 0x00003ffb //   log(1/frcpa(1+64/2^-8))/2
data8 0xe9226cce42b39f60 , 0x00003ffb //   log(1/frcpa(1+65/2^-8))/2
//
data8 0xec4749fd97a28360 , 0x00003ffb //   log(1/frcpa(1+66/2^-8))/2
data8 0xef6ea1bf57780495 , 0x00003ffb //   log(1/frcpa(1+67/2^-8))/2
data8 0xf29877ff38809091 , 0x00003ffb //   log(1/frcpa(1+68/2^-8))/2
data8 0xf5c4d0b245cb89be , 0x00003ffb //   log(1/frcpa(1+69/2^-8))/2
data8 0xf8f3afd6fcdef3aa , 0x00003ffb //   log(1/frcpa(1+70/2^-8))/2
//
data8 0xfc2519756be1abc7 , 0x00003ffb //   log(1/frcpa(1+71/2^-8))/2
data8 0xff59119f503e6832 , 0x00003ffb //   log(1/frcpa(1+72/2^-8))/2
data8 0x8147ce381ae0e146 , 0x00003ffc //   log(1/frcpa(1+73/2^-8))/2
data8 0x82e45f06cb1ad0f2 , 0x00003ffc //   log(1/frcpa(1+74/2^-8))/2
data8 0x842f5c7c573cbaa2 , 0x00003ffc //   log(1/frcpa(1+75/2^-8))/2
//
data8 0x85ce471968c8893a , 0x00003ffc //   log(1/frcpa(1+76/2^-8))/2
data8 0x876e8305bc04066d , 0x00003ffc //   log(1/frcpa(1+77/2^-8))/2
data8 0x891012678031fbb3 , 0x00003ffc //   log(1/frcpa(1+78/2^-8))/2
data8 0x8a5f1493d766a05f , 0x00003ffc //   log(1/frcpa(1+79/2^-8))/2
data8 0x8c030c778c56fa00 , 0x00003ffc //   log(1/frcpa(1+80/2^-8))/2
//
data8 0x8da85df17e31d9ae , 0x00003ffc //   log(1/frcpa(1+81/2^-8))/2
data8 0x8efa663e7921687e , 0x00003ffc //   log(1/frcpa(1+82/2^-8))/2
data8 0x90a22b6875c6a1f8 , 0x00003ffc //   log(1/frcpa(1+83/2^-8))/2
data8 0x91f62cc8f5d24837 , 0x00003ffc //   log(1/frcpa(1+84/2^-8))/2
data8 0x93a06cfc3857d980 , 0x00003ffc //   log(1/frcpa(1+85/2^-8))/2
//
data8 0x94f66d5e6fd01ced , 0x00003ffc //   log(1/frcpa(1+86/2^-8))/2
data8 0x96a330156e6772f2 , 0x00003ffc //   log(1/frcpa(1+87/2^-8))/2
data8 0x97fb3582754ea25b , 0x00003ffc //   log(1/frcpa(1+88/2^-8))/2
data8 0x99aa8259aad1bbf2 , 0x00003ffc //   log(1/frcpa(1+89/2^-8))/2
data8 0x9b0492f6227ae4a8 , 0x00003ffc //   log(1/frcpa(1+90/2^-8))/2
//
data8 0x9c5f8e199bf3a7a5 , 0x00003ffc //   log(1/frcpa(1+91/2^-8))/2
data8 0x9e1293b9998c1daa , 0x00003ffc //   log(1/frcpa(1+92/2^-8))/2
data8 0x9f6fa31e0b41f308 , 0x00003ffc //   log(1/frcpa(1+93/2^-8))/2
data8 0xa0cda11eaf46390e , 0x00003ffc //   log(1/frcpa(1+94/2^-8))/2
data8 0xa22c8f029cfa45aa , 0x00003ffc //   log(1/frcpa(1+95/2^-8))/2
//
data8 0xa3e48badb7856b34 , 0x00003ffc //   log(1/frcpa(1+96/2^-8))/2
data8 0xa5459a0aa95849f9 , 0x00003ffc //   log(1/frcpa(1+97/2^-8))/2
data8 0xa6a79c84480cfebd , 0x00003ffc //   log(1/frcpa(1+98/2^-8))/2
data8 0xa80a946d0fcb3eb2 , 0x00003ffc //   log(1/frcpa(1+99/2^-8))/2
data8 0xa96e831a3ea7b314 , 0x00003ffc //   log(1/frcpa(1+100/2^-8))/2
//
data8 0xaad369e3dc544e3b , 0x00003ffc //   log(1/frcpa(1+101/2^-8))/2
data8 0xac92e9588952c815 , 0x00003ffc //   log(1/frcpa(1+102/2^-8))/2
data8 0xadfa035aa1ed8fdc , 0x00003ffc //   log(1/frcpa(1+103/2^-8))/2
data8 0xaf6219eae1ad6e34 , 0x00003ffc //   log(1/frcpa(1+104/2^-8))/2
data8 0xb0cb2e6d8160f753 , 0x00003ffc //   log(1/frcpa(1+105/2^-8))/2
//
data8 0xb2354249ad950f72 , 0x00003ffc //   log(1/frcpa(1+106/2^-8))/2
data8 0xb3a056e98ef4a3b4 , 0x00003ffc //   log(1/frcpa(1+107/2^-8))/2
data8 0xb50c6dba52c6292a , 0x00003ffc //   log(1/frcpa(1+108/2^-8))/2
data8 0xb679882c33876165 , 0x00003ffc //   log(1/frcpa(1+109/2^-8))/2
data8 0xb78c07429785cedc , 0x00003ffc //   log(1/frcpa(1+110/2^-8))/2
//
data8 0xb8faeb8dc4a77d24 , 0x00003ffc //   log(1/frcpa(1+111/2^-8))/2
data8 0xba6ad77eb36ae0d6 , 0x00003ffc //   log(1/frcpa(1+112/2^-8))/2
data8 0xbbdbcc915e9bee50 , 0x00003ffc //   log(1/frcpa(1+113/2^-8))/2
data8 0xbd4dcc44f8cf12ef , 0x00003ffc //   log(1/frcpa(1+114/2^-8))/2
data8 0xbec0d81bf5b531fa , 0x00003ffc //   log(1/frcpa(1+115/2^-8))/2
//
data8 0xc034f19c139186f4 , 0x00003ffc //   log(1/frcpa(1+116/2^-8))/2
data8 0xc14cb69f7c5e55ab , 0x00003ffc //   log(1/frcpa(1+117/2^-8))/2
data8 0xc2c2abbb6e5fd56f , 0x00003ffc //   log(1/frcpa(1+118/2^-8))/2
data8 0xc439b2c193e6771e , 0x00003ffc //   log(1/frcpa(1+119/2^-8))/2
data8 0xc553acb9d5c67733 , 0x00003ffc //   log(1/frcpa(1+120/2^-8))/2
//
data8 0xc6cc96e441272441 , 0x00003ffc //   log(1/frcpa(1+121/2^-8))/2
data8 0xc8469753eca88c30 , 0x00003ffc //   log(1/frcpa(1+122/2^-8))/2
data8 0xc962cf3ce072b05c , 0x00003ffc //   log(1/frcpa(1+123/2^-8))/2
data8 0xcadeba8771f694aa , 0x00003ffc //   log(1/frcpa(1+124/2^-8))/2
data8 0xcc5bc08d1f72da94 , 0x00003ffc //   log(1/frcpa(1+125/2^-8))/2
//
data8 0xcd7a3f99ea035c29 , 0x00003ffc //   log(1/frcpa(1+126/2^-8))/2
data8 0xcef93860c8a53c35 , 0x00003ffc //   log(1/frcpa(1+127/2^-8))/2
data8 0xd0192f68a7ed23df , 0x00003ffc //   log(1/frcpa(1+128/2^-8))/2
data8 0xd19a201127d3c645 , 0x00003ffc //   log(1/frcpa(1+129/2^-8))/2
data8 0xd2bb92f4061c172c , 0x00003ffc //   log(1/frcpa(1+130/2^-8))/2
//
data8 0xd43e80b2ee8cc8fc , 0x00003ffc //   log(1/frcpa(1+131/2^-8))/2
data8 0xd56173601fc4ade4 , 0x00003ffc //   log(1/frcpa(1+132/2^-8))/2
data8 0xd6e6637efb54086f , 0x00003ffc //   log(1/frcpa(1+133/2^-8))/2
data8 0xd80ad9f58f3c8193 , 0x00003ffc //   log(1/frcpa(1+134/2^-8))/2
data8 0xd991d1d31aca41f8 , 0x00003ffc //   log(1/frcpa(1+135/2^-8))/2
//
data8 0xdab7d02231484a93 , 0x00003ffc //   log(1/frcpa(1+136/2^-8))/2
data8 0xdc40d532cde49a54 , 0x00003ffc //   log(1/frcpa(1+137/2^-8))/2
data8 0xdd685f79ed8b265e , 0x00003ffc //   log(1/frcpa(1+138/2^-8))/2
data8 0xde9094bbc0e17b1d , 0x00003ffc //   log(1/frcpa(1+139/2^-8))/2
data8 0xe01c91b78440c425 , 0x00003ffc //   log(1/frcpa(1+140/2^-8))/2
//
data8 0xe14658f26997e729 , 0x00003ffc //   log(1/frcpa(1+141/2^-8))/2
data8 0xe270cdc2391e0d23 , 0x00003ffc //   log(1/frcpa(1+142/2^-8))/2
data8 0xe3ffce3a2aa64922 , 0x00003ffc //   log(1/frcpa(1+143/2^-8))/2
data8 0xe52bdb274ed82887 , 0x00003ffc //   log(1/frcpa(1+144/2^-8))/2
data8 0xe6589852e75d7df6 , 0x00003ffc //   log(1/frcpa(1+145/2^-8))/2
//
data8 0xe786068c79937a7d , 0x00003ffc //   log(1/frcpa(1+146/2^-8))/2
data8 0xe91903adad100911 , 0x00003ffc //   log(1/frcpa(1+147/2^-8))/2
data8 0xea481236f7d35bb0 , 0x00003ffc //   log(1/frcpa(1+148/2^-8))/2
data8 0xeb77d48c692e6b14 , 0x00003ffc //   log(1/frcpa(1+149/2^-8))/2
data8 0xeca84b83d7297b87 , 0x00003ffc //   log(1/frcpa(1+150/2^-8))/2
//
data8 0xedd977f4962aa158 , 0x00003ffc //   log(1/frcpa(1+151/2^-8))/2
data8 0xef7179a22f257754 , 0x00003ffc //   log(1/frcpa(1+152/2^-8))/2
data8 0xf0a450d139366ca7 , 0x00003ffc //   log(1/frcpa(1+153/2^-8))/2
data8 0xf1d7e0524ff9ffdb , 0x00003ffc //   log(1/frcpa(1+154/2^-8))/2
data8 0xf30c29036a8b6cae , 0x00003ffc //   log(1/frcpa(1+155/2^-8))/2
//
data8 0xf4412bc411ea8d92 , 0x00003ffc //   log(1/frcpa(1+156/2^-8))/2
data8 0xf576e97564c8619d , 0x00003ffc //   log(1/frcpa(1+157/2^-8))/2
data8 0xf6ad62fa1b5f172f , 0x00003ffc //   log(1/frcpa(1+158/2^-8))/2
data8 0xf7e499368b55c542 , 0x00003ffc //   log(1/frcpa(1+159/2^-8))/2
data8 0xf91c8d10abaffe22 , 0x00003ffc //   log(1/frcpa(1+160/2^-8))/2
//
data8 0xfa553f7018c966f3 , 0x00003ffc //   log(1/frcpa(1+161/2^-8))/2
data8 0xfb8eb13e185d802c , 0x00003ffc //   log(1/frcpa(1+162/2^-8))/2
data8 0xfcc8e3659d9bcbed , 0x00003ffc //   log(1/frcpa(1+163/2^-8))/2
data8 0xfe03d6d34d487fd2 , 0x00003ffc //   log(1/frcpa(1+164/2^-8))/2
data8 0xff3f8c7581e9f0ae , 0x00003ffc //   log(1/frcpa(1+165/2^-8))/2
//
data8 0x803e029e280173ae , 0x00003ffd //   log(1/frcpa(1+166/2^-8))/2
data8 0x80dca10cc52d0757 , 0x00003ffd //   log(1/frcpa(1+167/2^-8))/2
data8 0x817ba200632755a1 , 0x00003ffd //   log(1/frcpa(1+168/2^-8))/2
data8 0x821b05f3b01d6774 , 0x00003ffd //   log(1/frcpa(1+169/2^-8))/2
data8 0x82bacd623ff19d06 , 0x00003ffd //   log(1/frcpa(1+170/2^-8))/2
//
data8 0x835af8c88e7a8f47 , 0x00003ffd //   log(1/frcpa(1+171/2^-8))/2
data8 0x83c5f8299e2b4091 , 0x00003ffd //   log(1/frcpa(1+172/2^-8))/2
data8 0x8466cb43f3d87300 , 0x00003ffd //   log(1/frcpa(1+173/2^-8))/2
data8 0x850803a67c80ca4b , 0x00003ffd //   log(1/frcpa(1+174/2^-8))/2
data8 0x85a9a1d11a23b461 , 0x00003ffd //   log(1/frcpa(1+175/2^-8))/2
//
data8 0x864ba644a18e6e05 , 0x00003ffd //   log(1/frcpa(1+176/2^-8))/2
data8 0x86ee1182dcc432f7 , 0x00003ffd //   log(1/frcpa(1+177/2^-8))/2
data8 0x875a925d7e48c316 , 0x00003ffd //   log(1/frcpa(1+178/2^-8))/2
data8 0x87fdaa109d23aef7 , 0x00003ffd //   log(1/frcpa(1+179/2^-8))/2
data8 0x88a129ed4becfaf2 , 0x00003ffd //   log(1/frcpa(1+180/2^-8))/2
//
data8 0x89451278ecd7f9cf , 0x00003ffd //   log(1/frcpa(1+181/2^-8))/2
data8 0x89b29295f8432617 , 0x00003ffd //   log(1/frcpa(1+182/2^-8))/2
data8 0x8a572ac5a5496882 , 0x00003ffd //   log(1/frcpa(1+183/2^-8))/2
data8 0x8afc2d0ce3b2dadf , 0x00003ffd //   log(1/frcpa(1+184/2^-8))/2
data8 0x8b6a69c608cfd3af , 0x00003ffd //   log(1/frcpa(1+185/2^-8))/2
//
data8 0x8c101e106e899a83 , 0x00003ffd //   log(1/frcpa(1+186/2^-8))/2
data8 0x8cb63de258f9d626 , 0x00003ffd //   log(1/frcpa(1+187/2^-8))/2
data8 0x8d2539c5bd19e2b1 , 0x00003ffd //   log(1/frcpa(1+188/2^-8))/2
data8 0x8dcc0e064b29e6f1 , 0x00003ffd //   log(1/frcpa(1+189/2^-8))/2
data8 0x8e734f45d88357ae , 0x00003ffd //   log(1/frcpa(1+190/2^-8))/2
//
data8 0x8ee30cef034a20db , 0x00003ffd //   log(1/frcpa(1+191/2^-8))/2
data8 0x8f8b0515686d1d06 , 0x00003ffd //   log(1/frcpa(1+192/2^-8))/2
data8 0x90336bba039bf32f , 0x00003ffd //   log(1/frcpa(1+193/2^-8))/2
data8 0x90a3edd23d1c9d58 , 0x00003ffd //   log(1/frcpa(1+194/2^-8))/2
data8 0x914d0de2f5d61b32 , 0x00003ffd //   log(1/frcpa(1+195/2^-8))/2
//
data8 0x91be0c20d28173b5 , 0x00003ffd //   log(1/frcpa(1+196/2^-8))/2
data8 0x9267e737c06cd34a , 0x00003ffd //   log(1/frcpa(1+197/2^-8))/2
data8 0x92d962ae6abb1237 , 0x00003ffd //   log(1/frcpa(1+198/2^-8))/2
data8 0x9383fa6afbe2074c , 0x00003ffd //   log(1/frcpa(1+199/2^-8))/2
data8 0x942f0421651c1c4e , 0x00003ffd //   log(1/frcpa(1+200/2^-8))/2
//
data8 0x94a14a3845bb985e , 0x00003ffd //   log(1/frcpa(1+201/2^-8))/2
data8 0x954d133857f861e7 , 0x00003ffd //   log(1/frcpa(1+202/2^-8))/2
data8 0x95bfd96468e604c4 , 0x00003ffd //   log(1/frcpa(1+203/2^-8))/2
data8 0x9632d31cafafa858 , 0x00003ffd //   log(1/frcpa(1+204/2^-8))/2
data8 0x96dfaabd86fa1647 , 0x00003ffd //   log(1/frcpa(1+205/2^-8))/2
//
data8 0x9753261fcbb2a594 , 0x00003ffd //   log(1/frcpa(1+206/2^-8))/2
data8 0x9800c11b426b996d , 0x00003ffd //   log(1/frcpa(1+207/2^-8))/2
data8 0x9874bf4d45ae663c , 0x00003ffd //   log(1/frcpa(1+208/2^-8))/2
data8 0x99231f5ee9a74f79 , 0x00003ffd //   log(1/frcpa(1+209/2^-8))/2
data8 0x9997a18a56bcad28 , 0x00003ffd //   log(1/frcpa(1+210/2^-8))/2
//
data8 0x9a46c873a3267e79 , 0x00003ffd //   log(1/frcpa(1+211/2^-8))/2
data8 0x9abbcfc621eb6cb6 , 0x00003ffd //   log(1/frcpa(1+212/2^-8))/2
data8 0x9b310cb0d354c990 , 0x00003ffd //   log(1/frcpa(1+213/2^-8))/2
data8 0x9be14cf9e1b3515c , 0x00003ffd //   log(1/frcpa(1+214/2^-8))/2
data8 0x9c5710b8cbb73a43 , 0x00003ffd //   log(1/frcpa(1+215/2^-8))/2
//
data8 0x9ccd0abd301f399c , 0x00003ffd //   log(1/frcpa(1+216/2^-8))/2
data8 0x9d7e67f3bdce8888 , 0x00003ffd //   log(1/frcpa(1+217/2^-8))/2
data8 0x9df4ea81a99daa01 , 0x00003ffd //   log(1/frcpa(1+218/2^-8))/2
data8 0x9e6ba405a54514ba , 0x00003ffd //   log(1/frcpa(1+219/2^-8))/2
data8 0x9f1e21c8c7bb62b3 , 0x00003ffd //   log(1/frcpa(1+220/2^-8))/2
//
data8 0x9f956593f6b6355c , 0x00003ffd //   log(1/frcpa(1+221/2^-8))/2
data8 0xa00ce1092e5498c3 , 0x00003ffd //   log(1/frcpa(1+222/2^-8))/2
data8 0xa0c08309c4b912c1 , 0x00003ffd //   log(1/frcpa(1+223/2^-8))/2
data8 0xa1388a8c6faa2afa , 0x00003ffd //   log(1/frcpa(1+224/2^-8))/2
data8 0xa1b0ca7095b5f985 , 0x00003ffd //   log(1/frcpa(1+225/2^-8))/2
//
data8 0xa22942eb47534a00 , 0x00003ffd //   log(1/frcpa(1+226/2^-8))/2
data8 0xa2de62326449d0a3 , 0x00003ffd //   log(1/frcpa(1+227/2^-8))/2
data8 0xa357690f88bfe345 , 0x00003ffd //   log(1/frcpa(1+228/2^-8))/2
data8 0xa3d0a93f45169a4b , 0x00003ffd //   log(1/frcpa(1+229/2^-8))/2
data8 0xa44a22f7ffe65f30 , 0x00003ffd //   log(1/frcpa(1+230/2^-8))/2
//
data8 0xa500c5e5b4c1aa36 , 0x00003ffd //   log(1/frcpa(1+231/2^-8))/2
data8 0xa57ad064eb2ebbc2 , 0x00003ffd //   log(1/frcpa(1+232/2^-8))/2
data8 0xa5f5152dedf4384e , 0x00003ffd //   log(1/frcpa(1+233/2^-8))/2
data8 0xa66f9478856233ec , 0x00003ffd //   log(1/frcpa(1+234/2^-8))/2
data8 0xa6ea4e7cca02c32e , 0x00003ffd //   log(1/frcpa(1+235/2^-8))/2
//
data8 0xa765437325341ccf , 0x00003ffd //   log(1/frcpa(1+236/2^-8))/2
data8 0xa81e21e6c75b4020 , 0x00003ffd //   log(1/frcpa(1+237/2^-8))/2
data8 0xa899ab333fe2b9ca , 0x00003ffd //   log(1/frcpa(1+238/2^-8))/2
data8 0xa9157039c51ebe71 , 0x00003ffd //   log(1/frcpa(1+239/2^-8))/2
data8 0xa991713433c2b999 , 0x00003ffd //   log(1/frcpa(1+240/2^-8))/2
//
data8 0xaa0dae5cbcc048b3 , 0x00003ffd //   log(1/frcpa(1+241/2^-8))/2
data8 0xaa8a27ede5eb13ad , 0x00003ffd //   log(1/frcpa(1+242/2^-8))/2
data8 0xab06de228a9e3499 , 0x00003ffd //   log(1/frcpa(1+243/2^-8))/2
data8 0xab83d135dc633301 , 0x00003ffd //   log(1/frcpa(1+244/2^-8))/2
data8 0xac3fb076adc7fe7a , 0x00003ffd //   log(1/frcpa(1+245/2^-8))/2
//
data8 0xacbd3cbbe47988f1 , 0x00003ffd //   log(1/frcpa(1+246/2^-8))/2
data8 0xad3b06b1a5dc57c3 , 0x00003ffd //   log(1/frcpa(1+247/2^-8))/2
data8 0xadb90e94af887717 , 0x00003ffd //   log(1/frcpa(1+248/2^-8))/2
data8 0xae3754a218f7c816 , 0x00003ffd //   log(1/frcpa(1+249/2^-8))/2
data8 0xaeb5d9175437afa2 , 0x00003ffd //   log(1/frcpa(1+250/2^-8))/2
//
data8 0xaf349c322e9c7cee , 0x00003ffd //   log(1/frcpa(1+251/2^-8))/2
data8 0xafb39e30d1768d1c , 0x00003ffd //   log(1/frcpa(1+252/2^-8))/2
data8 0xb032df51c2c93116 , 0x00003ffd //   log(1/frcpa(1+253/2^-8))/2
data8 0xb0b25fd3e6035ad9 , 0x00003ffd //   log(1/frcpa(1+254/2^-8))/2
data8 0xb1321ff67cba178c , 0x00003ffd //   log(1/frcpa(1+255/2^-8))/2
LOCAL_OBJECT_END(atanh_data_3)



.section .text
GLOBAL_LIBM_ENTRY(atanh)

{ .mfi
      getf.exp      rArgSExpb = f8                  // Must recompute if x unorm
      fclass.m      p13,p0 = f8, 0x0b               // is arg denormal ?
      mov           rExpbMask = 0x1ffff
}
{ .mfi
      addl          DataPtr = @ltoff(atanh_data), gp
      fnma.s1       fOneMx = f8, f1, f1             // fOneMx = 1 - x
      mov           rBias = 0xffff
}
;;

{ .mfi
      mov           rNearZeroBound = 0xfffd         // biased exp of 1/4
      fclass.m      p12,p0 = f8, 0xc7               // is arg NaN or +/-0 ?
      nop.i         0
}
{ .mfi
      ld8           DataPtr = [DataPtr]
      fma.s1        fOnePx = f8, f1, f1             // fOnePx = 1 + x
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fcmp.lt.s1    p10,p11 = f8,f0                 // is x < 0 ?
      nop.i         0
}
{ .mfb
      nop.m         0
      fnorm.s1      fNormX = f8                     // Normalize x
(p13) br.cond.spnt  ATANH_UNORM                     // Branch if x=unorm
}
;;

ATANH_COMMON:
// Return here if x=unorm and not denorm
{ .mfi
      adds          Data2Ptr = 0x50, DataPtr
      fma.s1        fX2 = f8, f8, f0                // x^2
      nop.i         0
}
{ .mfb
      adds          Data3Ptr = 0xC0, DataPtr
(p12) fma.d.s0      f8 = f8,f1,f8                   // NaN or +/-0
(p12) br.ret.spnt   b0                              // Exit for x Nan or zero
}
;;

{ .mfi
      ldfe          fC9 = [Data2Ptr], 16
(p11) frcpa.s1      fRcp0, p0 = f1, fOneMx
      nop.i         0
}
;;

{ .mfi
      ldfe          fC8 = [Data2Ptr], 16
(p10) frcpa.s1      fRcp0n, p0 = f1, fOnePx
      and           rArgExpb = rArgSExpb, rExpbMask // biased exponent
}
{ .mfi
      nop.m         0
(p10) fma.s1        fOneMx = fOnePx, f1, f0         // fOnePx = 1 - |x|
      nop.i         0
}
;;

{ .mfi
      ldfe          fC7 = [Data2Ptr], 16
(p10) fnma.s1       fOnePx = fNormX, f1, f1         // fOnePx = 1 + |x|
      cmp.ge        p6,p0 = rArgExpb, rBias         // is Expb(Arg) >= Expb(1) ?
}
{ .mfb
      nop.m         0
      nop.f         0
(p6)  br.cond.spnt  atanh_ge_one                    // Branch if |x| >=1.0
}
;;

{ .mfi
      ldfe          fC6 = [Data2Ptr], 16
      nop.f         0
      nop.i         0
}
;;

{ .mfi
      ldfe          fC5 = [Data2Ptr], 16
      fma.s1        fX4 = fX2, fX2, f0              // x^4
      cmp.gt        p8,p0 = rNearZeroBound, rArgExpb
}
{ .mfb
      ldfe          fC2 = [Data3Ptr], 16
      fma.s1        fX3 = fX2, fNormX, f0           // x^3
(p8)  br.cond.spnt  atanh_near_zero                 // Exit if 0 < |x| < 0.25
}
;;

// Main path: 0.25 <= |x| < 1.0
// NR method: iteration #1
.pred.rel "mutex",p11,p10
{ .mfi
      ldfpd         fP5, fP4 = [DataPtr], 16
(p11) fnma.s1       fRcp1 = fRcp0, fOneMx, f1       // t = 1 - r0*x
      nop.i         0
}
{ .mfi
      nop.m         0
(p10) fnma.s1       fRcp1 = fRcp0n, fOneMx, f1      // t = 1 - r0*x
      nop.i         0
}
;;

{ .mfi
      ldfpd         fP3, fP2 = [DataPtr], 16
      // r1 = r0 + r0*t = r0 + r0*(1 - r0*x)
(p11) fma.s1        fRcp1 = fRcp0, fRcp1, fRcp0
      nop.i         0
}
{ .mfi
      nop.m         0
      // r1 = r0 + r0*t = r0 + r0*(1 - r0*x)
(p10) fma.s1        fRcp1 = fRcp0n, fRcp1, fRcp0n
      nop.i         0
}
;;

// NR method: iteration #2
{ .mfi
      ldfd          fP1 = [DataPtr], 16
      fnma.s1       fRcp2 = fRcp1, fOneMx, f1       // t = 1 - r1*x
      nop.i         0
}
;;

{ .mfi
      ldfe          fLog2 = [DataPtr], 16
      // r2 = r1 + r1*t = r1 + r1*(1 - r1*x)
      fma.s1        fRcp2 = fRcp1, fRcp2, fRcp1
      nop.i         0
}
;;

// NR method: iteration #3
{ .mfi
      adds          RcpTablePtr = 0xB0, DataPtr
      fnma.s1       fRcp3 = fRcp2, fOneMx, f1       // t = 1 - r2*x
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        fY4Rcp = fRcp2, fOnePx, f0      // fY4Rcp = r2*(1 + x)
      nop.i         0
}
;;

// polynomial approximation & final reconstruction
{ .mfi
      nop.m         0
      frcpa.s1      fRcp, p0 = f1, fY4Rcp
      nop.i         0
}
{ .mfi
      nop.m         0
      // y = r2 * (1 + x) + r2 * (1 + x) * t = (1 + x) * (r2 + r2*(1 - r2*x))
      fma.s1        fY = fY4Rcp, fRcp3, fY4Rcp
      nop.i         0
}
;;

{ .mmi
      getf.exp      rSExpb = fY4Rcp                 // biased exponent and sign
;;
      getf.sig      rSig = fY4Rcp                   // significand
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fms.s1        fR = fY, fRcp, f1               // fR = fY * fRcp - 1
      nop.i         0
}
;;

{ .mmi
      and           rExpb = rSExpb, rExpbMask
;;
      sub           rN = rExpb, rBias               // exponent
      extr.u        rInd = rSig,55,8                // Extract 8 bits
}
;;

{ .mmi
      setf.sig      fN4Cvt = rN
      shladd        RcpTablePtr = rInd, 4, RcpTablePtr
      nop.i         0
}
;;

{ .mfi
      ldfe          fLogT = [RcpTablePtr]
      fma.s1        fR2 = fR, fR, f0                // r^2
      nop.i         0
}
{
      nop.m         0
      fma.s1        fP54 = fP5, fR, fP4             // P5*r + P4
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fP32 = fP3, fR, fP2             // P3*r + P2
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fR3 = fR2, fR, f0               // r^3
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        fP10 = fP1, fR2, fR             // P1*r^2 + r
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fcvt.xf       fN = fN4Cvt
      nop.i         0
}
{ .mfi
      nop.m         0
      fma.s1        fP54 = fP54, fR2, fP32      // (P5*r + P4)*r^2 + P3*r + P2
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fLogT_N = fN, fLog2, fLogT      // N*Log2 + LogT
      nop.i         0
}
{ .mfi
      nop.m         0
      // ((P5*r + P4)*r^2 + P3*r + P2)*r^3 + P1*r^2 + r
      fma.s1        fP54 = fP54, fR3, fP10
      nop.i         0
}
;;

.pred.rel "mutex",p11,p10
{ .mfi
      nop.m         0
      // 0.5*(((P5*r + P4)*r^2 + P3*r + P2)*r^3 + P1*r^2 + r) + 0.5*(N*Log2 + T)
(p11) fnma.d.s0     f8 = fP54, fP1, fLogT_N
      nop.i         0
}
{ .mfb
      nop.m         0
     // -0.5*(((P5*r + P4)*r^2 + P3*r + P2)*r^3 + P1*r^2 + r) - 0.5*(N*Log2 + T)
(p10) fms.d.s0      f8 = fP54, fP1, fLogT_N
      br.ret.sptk   b0                          // Exit for 0.25 <= |x| < 1.0
}
;;

// Here if 0 < |x| < 0.25
atanh_near_zero:
{ .mfi
      ldfe          fC4 = [Data2Ptr], 16
      fma.s1        fP98 = fC9, fX2, fC8           // C9*x^2 + C8
      nop.i         0
}
{ .mfi
      ldfe          fC1 = [Data3Ptr], 16
      fma.s1        fP76 = fC7, fX2, fC6           // C7*x^2 + C6
      nop.i         0
}
;;

{ .mfi
      ldfe          fC3 = [Data2Ptr], 16
      fma.s1        fX8 = fX4, fX4, f0             // x^8
      nop.i         0
}
{ .mfi
      ldfe          fC0 = [Data3Ptr], 16
      nop.f         0
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fP98 = fP98, fX4, fP76     // C9*x^6 + C8*x^4 + C7*x^2 + C6
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fP54 = fC5, fX2, fC4           // C5*x^2 + C4
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fP32 = fC3, fX2, fC2           // C3*x^2 + C2
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fP10 = fC1, fX2, fC0           // C1*x^2 + C0
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      fma.s1        fP54 = fP54, fX4, fP32      // C5*x^6 + C4*x^4 + C3*x^2 + C2
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // C9*x^14 + C8*x^12 + C7*x^10 + C6*x^8 + C5*x^6 + C4*x^4 + C3*x^2 + C2
      fma.s1        fP98 = fP98, fX8, fP54
      nop.i         0
}
;;

{ .mfi
      nop.m         0
      // C9*x^18 + C8*x^16 + C7*x^14 + C6*x^12 + C5*x^10 + C4*x^8 + C3*x^6 +
      // C2*x^4 + C1*x^2 + C0
      fma.s1        fP98 = fP98, fX4, fP10
      nop.i         0
}
;;

{ .mfb
      nop.m         0
      // C9*x^21 + C8*x^19 + C7*x^17 + C6*x^15 + C5*x^13 + C4*x^11 + C3*x^9 +
      // C2*x^7 + C1*x^5 + C0*x^3 + x
      fma.d.s0      f8 = fP98, fX3, fNormX
      br.ret.sptk   b0                           // Exit for 0 < |x| < 0.25
}
;;

ATANH_UNORM:
// Here if x=unorm
{ .mfi
      getf.exp      rArgSExpb = fNormX           // Recompute if x unorm
      fclass.m      p0,p13 = fNormX, 0x0b        // Test x denorm
      nop.i         0
}
;;

{ .mfb
      nop.m         0
      fcmp.eq.s0    p7,p0 = f8, f0        // Dummy to set denormal flag
(p13) br.cond.sptk  ATANH_COMMON          // Continue if x unorm and not denorm
}
;;

.pred.rel "mutex",p10,p11
{ .mfi
      nop.m         0
(p10) fnma.d.s0     f8 = f8,f8,f8                // Result x-x^2 if x=-denorm
      nop.i         0
}
{ .mfb
      nop.m         0
(p11) fma.d.s0      f8 = f8,f8,f8                // Result x+x^2 if x=+denorm
      br.ret.spnt   b0                           // Exit if denorm
}
;;

// Here if |x| >= 1.0
atanh_ge_one:
{ .mfi
      alloc         r32 = ar.pfs,1,3,4,0
      fmerge.s      fAbsX = f0, f8          // Form |x|
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
      fcmp.eq.s1    p6,p7 = fAbsX, f1       // Test for |x| = 1.0
      nop.i         0
}
;;

// Set error tag and result, and raise invalid flag if |x| > 1.0
{ .mfi
(p7)  mov           atanh_GR_tag = 131
(p7)  frcpa.s0      f8, p0 = f0, f0         // Get QNaN, and raise invalid
      nop.i         0
}
;;

// Set error tag and result, and raise Z flag if |x| = 1.0
{ .mfi
      nop.m         0
(p6)  frcpa.s0      fRcp, p0 = f1, f0       // Get inf, and raise Z flag
      nop.i         0
}
;;

{ .mfb
(p6)  mov           atanh_GR_tag = 132
(p6)  fmerge.s      f8 = f8, fRcp           // result is +-inf
      br.cond.sptk  __libm_error_region     // Exit if |x| >= 1.0
}
;;

GLOBAL_LIBM_END(atanh)
libm_alias_double_other (atanh, atanh)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue

{ .mfi
      add           GR_Parameter_Y=-32,sp        // Parameter 2 value
      nop.f         0
.save   ar.pfs,GR_SAVE_PFS
      mov           GR_SAVE_PFS=ar.pfs           // Save ar.pfs
}
{ .mfi
.fframe 64
      add sp=-64,sp                              // Create new stack
      nop.f 0
      mov GR_SAVE_GP=gp                          // Save gp
};;

{ .mmi
      stfd [GR_Parameter_Y] = f1,16              // STORE Parameter 2 on stack
      add GR_Parameter_X = 16,sp                 // Parameter 1 address
.save   b0, GR_SAVE_B0
      mov GR_SAVE_B0=b0                          // Save b0
};;

.body
{ .mib
      stfd [GR_Parameter_X] = f10                // STORE Parameter 1 on stack
      add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
      nop.b 0
}
{ .mib
      stfd [GR_Parameter_Y] = f8                 // STORE Parameter 3 on stack
      add   GR_Parameter_Y = -16,GR_Parameter_Y
      br.call.sptk b0=__libm_error_support#      // Call error handling function
};;

{ .mmi
      add   GR_Parameter_RESULT = 48,sp
      nop.m 0
      nop.i 0
};;

{ .mmi
      ldfd  f8 = [GR_Parameter_RESULT]           // Get return result off stack
.restore sp
      add   sp = 64,sp                           // Restore stack pointer
      mov   b0 = GR_SAVE_B0                      // Restore return address
};;

{ .mib
      mov   gp = GR_SAVE_GP                      // Restore gp
      mov   ar.pfs = GR_SAVE_PFS                 // Restore ar.pfs
      br.ret.sptk     b0                         // Return
};;

LOCAL_LIBM_END(__libm_error_region)


.type   __libm_error_support#,@function
.global __libm_error_support#
