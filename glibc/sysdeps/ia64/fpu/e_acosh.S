.file "acosh.s"


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
// 03/23/01 Initial version
// 04/19/01 Improved speed of the paths #1,2,3,4,5
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align
// 05/14/03 Improved performance, set denormal flag for unorms >= 1.0
// 03/31/05 Reformatted delimiters between data tables
//
// API
// ==============================================================
// double acosh(double)
//
// Overview of operation
// ==============================================================
//
// There are 7 paths:
// 1. x = 1.0
//    Return acosh(x) = 0.0
// 2. 1.0 < x < 1.000499725341796875(0x3FF0020C00000000)
//    Return acosh(x) = sqrt(x-1) * Pol4(x), where Pol4(x) =
//      (((x*C4 + C3)*(x-1) + C2)*(x-1) + C1)*(x-1) + C0

// 3. 1.000499725341796875(0x3FF0020C00000000) <= x < 2^63
//    Return acosh(x) = log(x + sqrt(x^2 -1.0))
//    To compute x + sqrt(x^2 -1.0) modified Newton Raphson method is used
//      (3 iterations)
//    Algorithm description for log function see below.
//
// 4. 2^63 <= x < +INF
//    Return acosh(x) = log(2*x)
//    Algorithm description for log function see below.
//
// 5. x = +INF
//    Return acosh(x) = +INF
//
// 6. x = [S,Q]NaN
//    Return acosh(x) = QNaN
//
// 7. x < 1.0
//    It's domain error. Error handler with tag = 136 is called
//
//==============================================================
// Algorithm Description for log(x) function
// Below we are using the fact that inequality x - 1.0 > 2^(-6) is always
//   true for this acosh implementation
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
// Form rseries = r + P1*r^2 + P2*r^3 + P3*r^4 + P4*r^5 + P5*r^6
//
// x = f * 2*n where f is 1.f_1f_2f_3....f_63
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

// Registers used
//==============================================================
// Floating Point registers used:
// f8, input
// f9 -> f15,  f32 -> f65

// General registers used:
// r14 -> r27, r32 -> r39

// Predicate registers used:
// p6 -> p15

// p6 to filter out case when x = [Q,S]NaN
// p7,p8 to filter out case when x < 1.0
// p10 to select path #1
// p11 to filter out case when x = +INF
// p12 used in the frcpa
// p13 to select path #4
// p14,p15 to select path #2

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
acosh_GR_f8           = r23
log_GR_comp           = r24
acosh_GR_f8_sig       = r25
log_table_address3    = r26
NR_table_address      = r27

GR_SAVE_B0            = r33
GR_SAVE_GP            = r34
GR_SAVE_PFS           = r35

GR_Parameter_X        = r36
GR_Parameter_Y        = r37
GR_Parameter_RESULT   = r38
acosh_GR_tag          = r39

//==============================================================
log_y            = f9
NR1              = f10
NR2              = f11
log_y_rs         = f12
log_y_rs_iter    = f13
log_y_rs_iter1   = f14
log_NORM_f8      = f15
acosh_comp       = f32
log_w            = f34
log_P5           = f35
log_P4           = f36
log_P3           = f37
log_P2           = f38
log_P1           = f39
log_C0           = f40
log_C1           = f41
log_C2           = f42
log2             = f43
acosh_w_rs       = f44
log_C            = f45
log_arg          = f46
acosh_w_iter1    = f47
acosh_w_iter2    = f48
log_int_Nfloat   = f49
log_r            = f50
log_rsq          = f51
log_rp_p4        = f52
log_rp_p32       = f53
log_rcube        = f54
log_rp_p10       = f55
log_rp_p2        = f56
log_Nfloat       = f57
log_T            = f58
log_r2P_r        = f59
log_T_plus_Nlog2 = f60
acosh_w_sqrt     = f61
acosh_w_1        = f62
log_C3           = f63
log_C4           = f64
log_arg_early    = f65


// Data tables
//==============================================================

RODATA
.align 16

LOCAL_OBJECT_START(log_table_1)
data8 0x3FF0020C49BA5E35 // 1.0005
data8 0xBFC5555DA7212371 // P5
data8 0x3FC999A19EEF5826 // P4
data8 0xBFCFFFFFFFFEF009 // P3
data8 0x3FD555555554ECB2 // P2
data8 0xBFE0000000000000 // P1 = -0.5
//
data8 0xb17217f7d1cf79ac, 0x00003ffe  // log2
LOCAL_OBJECT_END(log_table_1)

LOCAL_OBJECT_START(log_table_2)
data8 0x3FE0000000000000 // 0.5
data8 0x4008000000000000 // 3.0
//
data8 0xAFE8F9203939CCF8, 0x00003FF6 // C4 3FF6AFE8F9203939CCF8
data8 0xAD46EB6AE752D809, 0x0000BFF8 // C3 BFF8AD46EB6AE752D809
data8 0xD93923D7F53F3627, 0x00003FF9 // C2 3FF9D93923D7F53F3627
data8 0xF15BEEEFF7D32D36, 0x0000BFFB // C1 BFFBF15BEEEFF7D32D36
data8 0xB504F333F9DE6484, 0x00003FFF // C0 3FFFB504F333F9DE6484
LOCAL_OBJECT_END(log_table_2)


LOCAL_OBJECT_START(log_table_3)
data8 0x80200aaeac44ef38 , 0x00003ff6 //   log(1/frcpa(1+  0/2^-8))
//
data8 0xc09090a2c35aa070 , 0x00003ff7 //   log(1/frcpa(1+  1/2^-8))
data8 0xa0c94fcb41977c75 , 0x00003ff8 //   log(1/frcpa(1+  2/2^-8))
data8 0xe18b9c263af83301 , 0x00003ff8 //   log(1/frcpa(1+  3/2^-8))
data8 0x8d35c8d6399c30ea , 0x00003ff9 //   log(1/frcpa(1+  4/2^-8))
data8 0xadd4d2ecd601cbb8 , 0x00003ff9 //   log(1/frcpa(1+  5/2^-8))
//
data8 0xce95403a192f9f01 , 0x00003ff9 //   log(1/frcpa(1+  6/2^-8))
data8 0xeb59392cbcc01096 , 0x00003ff9 //   log(1/frcpa(1+  7/2^-8))
data8 0x862c7d0cefd54c5d , 0x00003ffa //   log(1/frcpa(1+  8/2^-8))
data8 0x94aa63c65e70d499 , 0x00003ffa //   log(1/frcpa(1+  9/2^-8))
data8 0xa54a696d4b62b382 , 0x00003ffa //   log(1/frcpa(1+ 10/2^-8))
//
data8 0xb3e4a796a5dac208 , 0x00003ffa //   log(1/frcpa(1+ 11/2^-8))
data8 0xc28c45b1878340a9 , 0x00003ffa //   log(1/frcpa(1+ 12/2^-8))
data8 0xd35c55f39d7a6235 , 0x00003ffa //   log(1/frcpa(1+ 13/2^-8))
data8 0xe220f037b954f1f5 , 0x00003ffa //   log(1/frcpa(1+ 14/2^-8))
data8 0xf0f3389b036834f3 , 0x00003ffa //   log(1/frcpa(1+ 15/2^-8))
//
data8 0xffd3488d5c980465 , 0x00003ffa //   log(1/frcpa(1+ 16/2^-8))
data8 0x87609ce2ed300490 , 0x00003ffb //   log(1/frcpa(1+ 17/2^-8))
data8 0x8ede9321e8c85927 , 0x00003ffb //   log(1/frcpa(1+ 18/2^-8))
data8 0x96639427f2f8e2f4 , 0x00003ffb //   log(1/frcpa(1+ 19/2^-8))
data8 0x9defad3e8f73217b , 0x00003ffb //   log(1/frcpa(1+ 20/2^-8))
//
data8 0xa582ebd50097029c , 0x00003ffb //   log(1/frcpa(1+ 21/2^-8))
data8 0xac06dbe75ab80fee , 0x00003ffb //   log(1/frcpa(1+ 22/2^-8))
data8 0xb3a78449b2d3ccca , 0x00003ffb //   log(1/frcpa(1+ 23/2^-8))
data8 0xbb4f79635ab46bb2 , 0x00003ffb //   log(1/frcpa(1+ 24/2^-8))
data8 0xc2fec93a83523f3f , 0x00003ffb //   log(1/frcpa(1+ 25/2^-8))
//
data8 0xc99af2eaca4c4571 , 0x00003ffb //   log(1/frcpa(1+ 26/2^-8))
data8 0xd1581106472fa653 , 0x00003ffb //   log(1/frcpa(1+ 27/2^-8))
data8 0xd8002560d4355f2e , 0x00003ffb //   log(1/frcpa(1+ 28/2^-8))
data8 0xdfcb43b4fe508632 , 0x00003ffb //   log(1/frcpa(1+ 29/2^-8))
data8 0xe67f6dff709d4119 , 0x00003ffb //   log(1/frcpa(1+ 30/2^-8))
//
data8 0xed393b1c22351280 , 0x00003ffb //   log(1/frcpa(1+ 31/2^-8))
data8 0xf5192bff087bcc35 , 0x00003ffb //   log(1/frcpa(1+ 32/2^-8))
data8 0xfbdf4ff6dfef2fa3 , 0x00003ffb //   log(1/frcpa(1+ 33/2^-8))
data8 0x81559a97f92f9cc7 , 0x00003ffc //   log(1/frcpa(1+ 34/2^-8))
data8 0x84be72bce90266e8 , 0x00003ffc //   log(1/frcpa(1+ 35/2^-8))
//
data8 0x88bc74113f23def2 , 0x00003ffc //   log(1/frcpa(1+ 36/2^-8))
data8 0x8c2ba3edf6799d11 , 0x00003ffc //   log(1/frcpa(1+ 37/2^-8))
data8 0x8f9dc92f92ea08b1 , 0x00003ffc //   log(1/frcpa(1+ 38/2^-8))
data8 0x9312e8f36efab5a7 , 0x00003ffc //   log(1/frcpa(1+ 39/2^-8))
data8 0x968b08643409ceb6 , 0x00003ffc //   log(1/frcpa(1+ 40/2^-8))
//
data8 0x9a062cba08a1708c , 0x00003ffc //   log(1/frcpa(1+ 41/2^-8))
data8 0x9d845b3abf95485c , 0x00003ffc //   log(1/frcpa(1+ 42/2^-8))
data8 0xa06fd841bc001bb4 , 0x00003ffc //   log(1/frcpa(1+ 43/2^-8))
data8 0xa3f3a74652fbe0db , 0x00003ffc //   log(1/frcpa(1+ 44/2^-8))
data8 0xa77a8fb2336f20f5 , 0x00003ffc //   log(1/frcpa(1+ 45/2^-8))
//
data8 0xab0497015d28b0a0 , 0x00003ffc //   log(1/frcpa(1+ 46/2^-8))
data8 0xae91c2be6ba6a615 , 0x00003ffc //   log(1/frcpa(1+ 47/2^-8))
data8 0xb189d1b99aebb20b , 0x00003ffc //   log(1/frcpa(1+ 48/2^-8))
data8 0xb51cced5de9c1b2c , 0x00003ffc //   log(1/frcpa(1+ 49/2^-8))
data8 0xb819bee9e720d42f , 0x00003ffc //   log(1/frcpa(1+ 50/2^-8))
//
data8 0xbbb2a0947b093a5d , 0x00003ffc //   log(1/frcpa(1+ 51/2^-8))
data8 0xbf4ec1505811684a , 0x00003ffc //   log(1/frcpa(1+ 52/2^-8))
data8 0xc2535bacfa8975ff , 0x00003ffc //   log(1/frcpa(1+ 53/2^-8))
data8 0xc55a3eafad187eb8 , 0x00003ffc //   log(1/frcpa(1+ 54/2^-8))
data8 0xc8ff2484b2c0da74 , 0x00003ffc //   log(1/frcpa(1+ 55/2^-8))
//
data8 0xcc0b1a008d53ab76 , 0x00003ffc //   log(1/frcpa(1+ 56/2^-8))
data8 0xcfb6203844b3209b , 0x00003ffc //   log(1/frcpa(1+ 57/2^-8))
data8 0xd2c73949a47a19f5 , 0x00003ffc //   log(1/frcpa(1+ 58/2^-8))
data8 0xd5daae18b49d6695 , 0x00003ffc //   log(1/frcpa(1+ 59/2^-8))
data8 0xd8f08248cf7e8019 , 0x00003ffc //   log(1/frcpa(1+ 60/2^-8))
//
data8 0xdca7749f1b3e540e , 0x00003ffc //   log(1/frcpa(1+ 61/2^-8))
data8 0xdfc28e033aaaf7c7 , 0x00003ffc //   log(1/frcpa(1+ 62/2^-8))
data8 0xe2e012a5f91d2f55 , 0x00003ffc //   log(1/frcpa(1+ 63/2^-8))
data8 0xe600064ed9e292a8 , 0x00003ffc //   log(1/frcpa(1+ 64/2^-8))
data8 0xe9226cce42b39f60 , 0x00003ffc //   log(1/frcpa(1+ 65/2^-8))
//
data8 0xec4749fd97a28360 , 0x00003ffc //   log(1/frcpa(1+ 66/2^-8))
data8 0xef6ea1bf57780495 , 0x00003ffc //   log(1/frcpa(1+ 67/2^-8))
data8 0xf29877ff38809091 , 0x00003ffc //   log(1/frcpa(1+ 68/2^-8))
data8 0xf5c4d0b245cb89be , 0x00003ffc //   log(1/frcpa(1+ 69/2^-8))
data8 0xf8f3afd6fcdef3aa , 0x00003ffc //   log(1/frcpa(1+ 70/2^-8))
//
data8 0xfc2519756be1abc7 , 0x00003ffc //   log(1/frcpa(1+ 71/2^-8))
data8 0xff59119f503e6832 , 0x00003ffc //   log(1/frcpa(1+ 72/2^-8))
data8 0x8147ce381ae0e146 , 0x00003ffd //   log(1/frcpa(1+ 73/2^-8))
data8 0x82e45f06cb1ad0f2 , 0x00003ffd //   log(1/frcpa(1+ 74/2^-8))
data8 0x842f5c7c573cbaa2 , 0x00003ffd //   log(1/frcpa(1+ 75/2^-8))
//
data8 0x85ce471968c8893a , 0x00003ffd //   log(1/frcpa(1+ 76/2^-8))
data8 0x876e8305bc04066d , 0x00003ffd //   log(1/frcpa(1+ 77/2^-8))
data8 0x891012678031fbb3 , 0x00003ffd //   log(1/frcpa(1+ 78/2^-8))
data8 0x8a5f1493d766a05f , 0x00003ffd //   log(1/frcpa(1+ 79/2^-8))
data8 0x8c030c778c56fa00 , 0x00003ffd //   log(1/frcpa(1+ 80/2^-8))
//
data8 0x8da85df17e31d9ae , 0x00003ffd //   log(1/frcpa(1+ 81/2^-8))
data8 0x8efa663e7921687e , 0x00003ffd //   log(1/frcpa(1+ 82/2^-8))
data8 0x90a22b6875c6a1f8 , 0x00003ffd //   log(1/frcpa(1+ 83/2^-8))
data8 0x91f62cc8f5d24837 , 0x00003ffd //   log(1/frcpa(1+ 84/2^-8))
data8 0x93a06cfc3857d980 , 0x00003ffd //   log(1/frcpa(1+ 85/2^-8))
//
data8 0x94f66d5e6fd01ced , 0x00003ffd //   log(1/frcpa(1+ 86/2^-8))
data8 0x96a330156e6772f2 , 0x00003ffd //   log(1/frcpa(1+ 87/2^-8))
data8 0x97fb3582754ea25b , 0x00003ffd //   log(1/frcpa(1+ 88/2^-8))
data8 0x99aa8259aad1bbf2 , 0x00003ffd //   log(1/frcpa(1+ 89/2^-8))
data8 0x9b0492f6227ae4a8 , 0x00003ffd //   log(1/frcpa(1+ 90/2^-8))
//
data8 0x9c5f8e199bf3a7a5 , 0x00003ffd //   log(1/frcpa(1+ 91/2^-8))
data8 0x9e1293b9998c1daa , 0x00003ffd //   log(1/frcpa(1+ 92/2^-8))
data8 0x9f6fa31e0b41f308 , 0x00003ffd //   log(1/frcpa(1+ 93/2^-8))
data8 0xa0cda11eaf46390e , 0x00003ffd //   log(1/frcpa(1+ 94/2^-8))
data8 0xa22c8f029cfa45aa , 0x00003ffd //   log(1/frcpa(1+ 95/2^-8))
//
data8 0xa3e48badb7856b34 , 0x00003ffd //   log(1/frcpa(1+ 96/2^-8))
data8 0xa5459a0aa95849f9 , 0x00003ffd //   log(1/frcpa(1+ 97/2^-8))
data8 0xa6a79c84480cfebd , 0x00003ffd //   log(1/frcpa(1+ 98/2^-8))
data8 0xa80a946d0fcb3eb2 , 0x00003ffd //   log(1/frcpa(1+ 99/2^-8))
data8 0xa96e831a3ea7b314 , 0x00003ffd //   log(1/frcpa(1+100/2^-8))
//
data8 0xaad369e3dc544e3b , 0x00003ffd //   log(1/frcpa(1+101/2^-8))
data8 0xac92e9588952c815 , 0x00003ffd //   log(1/frcpa(1+102/2^-8))
data8 0xadfa035aa1ed8fdc , 0x00003ffd //   log(1/frcpa(1+103/2^-8))
data8 0xaf6219eae1ad6e34 , 0x00003ffd //   log(1/frcpa(1+104/2^-8))
data8 0xb0cb2e6d8160f753 , 0x00003ffd //   log(1/frcpa(1+105/2^-8))
//
data8 0xb2354249ad950f72 , 0x00003ffd //   log(1/frcpa(1+106/2^-8))
data8 0xb3a056e98ef4a3b4 , 0x00003ffd //   log(1/frcpa(1+107/2^-8))
data8 0xb50c6dba52c6292a , 0x00003ffd //   log(1/frcpa(1+108/2^-8))
data8 0xb679882c33876165 , 0x00003ffd //   log(1/frcpa(1+109/2^-8))
data8 0xb78c07429785cedc , 0x00003ffd //   log(1/frcpa(1+110/2^-8))
//
data8 0xb8faeb8dc4a77d24 , 0x00003ffd //   log(1/frcpa(1+111/2^-8))
data8 0xba6ad77eb36ae0d6 , 0x00003ffd //   log(1/frcpa(1+112/2^-8))
data8 0xbbdbcc915e9bee50 , 0x00003ffd //   log(1/frcpa(1+113/2^-8))
data8 0xbd4dcc44f8cf12ef , 0x00003ffd //   log(1/frcpa(1+114/2^-8))
data8 0xbec0d81bf5b531fa , 0x00003ffd //   log(1/frcpa(1+115/2^-8))
//
data8 0xc034f19c139186f4 , 0x00003ffd //   log(1/frcpa(1+116/2^-8))
data8 0xc14cb69f7c5e55ab , 0x00003ffd //   log(1/frcpa(1+117/2^-8))
data8 0xc2c2abbb6e5fd56f , 0x00003ffd //   log(1/frcpa(1+118/2^-8))
data8 0xc439b2c193e6771e , 0x00003ffd //   log(1/frcpa(1+119/2^-8))
data8 0xc553acb9d5c67733 , 0x00003ffd //   log(1/frcpa(1+120/2^-8))
//
data8 0xc6cc96e441272441 , 0x00003ffd //   log(1/frcpa(1+121/2^-8))
data8 0xc8469753eca88c30 , 0x00003ffd //   log(1/frcpa(1+122/2^-8))
data8 0xc962cf3ce072b05c , 0x00003ffd //   log(1/frcpa(1+123/2^-8))
data8 0xcadeba8771f694aa , 0x00003ffd //   log(1/frcpa(1+124/2^-8))
data8 0xcc5bc08d1f72da94 , 0x00003ffd //   log(1/frcpa(1+125/2^-8))
//
data8 0xcd7a3f99ea035c29 , 0x00003ffd //   log(1/frcpa(1+126/2^-8))
data8 0xcef93860c8a53c35 , 0x00003ffd //   log(1/frcpa(1+127/2^-8))
data8 0xd0192f68a7ed23df , 0x00003ffd //   log(1/frcpa(1+128/2^-8))
data8 0xd19a201127d3c645 , 0x00003ffd //   log(1/frcpa(1+129/2^-8))
data8 0xd2bb92f4061c172c , 0x00003ffd //   log(1/frcpa(1+130/2^-8))
//
data8 0xd43e80b2ee8cc8fc , 0x00003ffd //   log(1/frcpa(1+131/2^-8))
data8 0xd56173601fc4ade4 , 0x00003ffd //   log(1/frcpa(1+132/2^-8))
data8 0xd6e6637efb54086f , 0x00003ffd //   log(1/frcpa(1+133/2^-8))
data8 0xd80ad9f58f3c8193 , 0x00003ffd //   log(1/frcpa(1+134/2^-8))
data8 0xd991d1d31aca41f8 , 0x00003ffd //   log(1/frcpa(1+135/2^-8))
//
data8 0xdab7d02231484a93 , 0x00003ffd //   log(1/frcpa(1+136/2^-8))
data8 0xdc40d532cde49a54 , 0x00003ffd //   log(1/frcpa(1+137/2^-8))
data8 0xdd685f79ed8b265e , 0x00003ffd //   log(1/frcpa(1+138/2^-8))
data8 0xde9094bbc0e17b1d , 0x00003ffd //   log(1/frcpa(1+139/2^-8))
data8 0xe01c91b78440c425 , 0x00003ffd //   log(1/frcpa(1+140/2^-8))
//
data8 0xe14658f26997e729 , 0x00003ffd //   log(1/frcpa(1+141/2^-8))
data8 0xe270cdc2391e0d23 , 0x00003ffd //   log(1/frcpa(1+142/2^-8))
data8 0xe3ffce3a2aa64922 , 0x00003ffd //   log(1/frcpa(1+143/2^-8))
data8 0xe52bdb274ed82887 , 0x00003ffd //   log(1/frcpa(1+144/2^-8))
data8 0xe6589852e75d7df6 , 0x00003ffd //   log(1/frcpa(1+145/2^-8))
//
data8 0xe786068c79937a7d , 0x00003ffd //   log(1/frcpa(1+146/2^-8))
data8 0xe91903adad100911 , 0x00003ffd //   log(1/frcpa(1+147/2^-8))
data8 0xea481236f7d35bb0 , 0x00003ffd //   log(1/frcpa(1+148/2^-8))
data8 0xeb77d48c692e6b14 , 0x00003ffd //   log(1/frcpa(1+149/2^-8))
data8 0xeca84b83d7297b87 , 0x00003ffd //   log(1/frcpa(1+150/2^-8))
//
data8 0xedd977f4962aa158 , 0x00003ffd //   log(1/frcpa(1+151/2^-8))
data8 0xef7179a22f257754 , 0x00003ffd //   log(1/frcpa(1+152/2^-8))
data8 0xf0a450d139366ca7 , 0x00003ffd //   log(1/frcpa(1+153/2^-8))
data8 0xf1d7e0524ff9ffdb , 0x00003ffd //   log(1/frcpa(1+154/2^-8))
data8 0xf30c29036a8b6cae , 0x00003ffd //   log(1/frcpa(1+155/2^-8))
//
data8 0xf4412bc411ea8d92 , 0x00003ffd //   log(1/frcpa(1+156/2^-8))
data8 0xf576e97564c8619d , 0x00003ffd //   log(1/frcpa(1+157/2^-8))
data8 0xf6ad62fa1b5f172f , 0x00003ffd //   log(1/frcpa(1+158/2^-8))
data8 0xf7e499368b55c542 , 0x00003ffd //   log(1/frcpa(1+159/2^-8))
data8 0xf91c8d10abaffe22 , 0x00003ffd //   log(1/frcpa(1+160/2^-8))
//
data8 0xfa553f7018c966f3 , 0x00003ffd //   log(1/frcpa(1+161/2^-8))
data8 0xfb8eb13e185d802c , 0x00003ffd //   log(1/frcpa(1+162/2^-8))
data8 0xfcc8e3659d9bcbed , 0x00003ffd //   log(1/frcpa(1+163/2^-8))
data8 0xfe03d6d34d487fd2 , 0x00003ffd //   log(1/frcpa(1+164/2^-8))
data8 0xff3f8c7581e9f0ae , 0x00003ffd //   log(1/frcpa(1+165/2^-8))
//
data8 0x803e029e280173ae , 0x00003ffe //   log(1/frcpa(1+166/2^-8))
data8 0x80dca10cc52d0757 , 0x00003ffe //   log(1/frcpa(1+167/2^-8))
data8 0x817ba200632755a1 , 0x00003ffe //   log(1/frcpa(1+168/2^-8))
data8 0x821b05f3b01d6774 , 0x00003ffe //   log(1/frcpa(1+169/2^-8))
data8 0x82bacd623ff19d06 , 0x00003ffe //   log(1/frcpa(1+170/2^-8))
//
data8 0x835af8c88e7a8f47 , 0x00003ffe //   log(1/frcpa(1+171/2^-8))
data8 0x83c5f8299e2b4091 , 0x00003ffe //   log(1/frcpa(1+172/2^-8))
data8 0x8466cb43f3d87300 , 0x00003ffe //   log(1/frcpa(1+173/2^-8))
data8 0x850803a67c80ca4b , 0x00003ffe //   log(1/frcpa(1+174/2^-8))
data8 0x85a9a1d11a23b461 , 0x00003ffe //   log(1/frcpa(1+175/2^-8))
//
data8 0x864ba644a18e6e05 , 0x00003ffe //   log(1/frcpa(1+176/2^-8))
data8 0x86ee1182dcc432f7 , 0x00003ffe //   log(1/frcpa(1+177/2^-8))
data8 0x875a925d7e48c316 , 0x00003ffe //   log(1/frcpa(1+178/2^-8))
data8 0x87fdaa109d23aef7 , 0x00003ffe //   log(1/frcpa(1+179/2^-8))
data8 0x88a129ed4becfaf2 , 0x00003ffe //   log(1/frcpa(1+180/2^-8))
//
data8 0x89451278ecd7f9cf , 0x00003ffe //   log(1/frcpa(1+181/2^-8))
data8 0x89b29295f8432617 , 0x00003ffe //   log(1/frcpa(1+182/2^-8))
data8 0x8a572ac5a5496882 , 0x00003ffe //   log(1/frcpa(1+183/2^-8))
data8 0x8afc2d0ce3b2dadf , 0x00003ffe //   log(1/frcpa(1+184/2^-8))
data8 0x8b6a69c608cfd3af , 0x00003ffe //   log(1/frcpa(1+185/2^-8))
//
data8 0x8c101e106e899a83 , 0x00003ffe //   log(1/frcpa(1+186/2^-8))
data8 0x8cb63de258f9d626 , 0x00003ffe //   log(1/frcpa(1+187/2^-8))
data8 0x8d2539c5bd19e2b1 , 0x00003ffe //   log(1/frcpa(1+188/2^-8))
data8 0x8dcc0e064b29e6f1 , 0x00003ffe //   log(1/frcpa(1+189/2^-8))
data8 0x8e734f45d88357ae , 0x00003ffe //   log(1/frcpa(1+190/2^-8))
//
data8 0x8ee30cef034a20db , 0x00003ffe //   log(1/frcpa(1+191/2^-8))
data8 0x8f8b0515686d1d06 , 0x00003ffe //   log(1/frcpa(1+192/2^-8))
data8 0x90336bba039bf32f , 0x00003ffe //   log(1/frcpa(1+193/2^-8))
data8 0x90a3edd23d1c9d58 , 0x00003ffe //   log(1/frcpa(1+194/2^-8))
data8 0x914d0de2f5d61b32 , 0x00003ffe //   log(1/frcpa(1+195/2^-8))
//
data8 0x91be0c20d28173b5 , 0x00003ffe //   log(1/frcpa(1+196/2^-8))
data8 0x9267e737c06cd34a , 0x00003ffe //   log(1/frcpa(1+197/2^-8))
data8 0x92d962ae6abb1237 , 0x00003ffe //   log(1/frcpa(1+198/2^-8))
data8 0x9383fa6afbe2074c , 0x00003ffe //   log(1/frcpa(1+199/2^-8))
data8 0x942f0421651c1c4e , 0x00003ffe //   log(1/frcpa(1+200/2^-8))
//
data8 0x94a14a3845bb985e , 0x00003ffe //   log(1/frcpa(1+201/2^-8))
data8 0x954d133857f861e7 , 0x00003ffe //   log(1/frcpa(1+202/2^-8))
data8 0x95bfd96468e604c4 , 0x00003ffe //   log(1/frcpa(1+203/2^-8))
data8 0x9632d31cafafa858 , 0x00003ffe //   log(1/frcpa(1+204/2^-8))
data8 0x96dfaabd86fa1647 , 0x00003ffe //   log(1/frcpa(1+205/2^-8))
//
data8 0x9753261fcbb2a594 , 0x00003ffe //   log(1/frcpa(1+206/2^-8))
data8 0x9800c11b426b996d , 0x00003ffe //   log(1/frcpa(1+207/2^-8))
data8 0x9874bf4d45ae663c , 0x00003ffe //   log(1/frcpa(1+208/2^-8))
data8 0x99231f5ee9a74f79 , 0x00003ffe //   log(1/frcpa(1+209/2^-8))
data8 0x9997a18a56bcad28 , 0x00003ffe //   log(1/frcpa(1+210/2^-8))
//
data8 0x9a46c873a3267e79 , 0x00003ffe //   log(1/frcpa(1+211/2^-8))
data8 0x9abbcfc621eb6cb6 , 0x00003ffe //   log(1/frcpa(1+212/2^-8))
data8 0x9b310cb0d354c990 , 0x00003ffe //   log(1/frcpa(1+213/2^-8))
data8 0x9be14cf9e1b3515c , 0x00003ffe //   log(1/frcpa(1+214/2^-8))
data8 0x9c5710b8cbb73a43 , 0x00003ffe //   log(1/frcpa(1+215/2^-8))
//
data8 0x9ccd0abd301f399c , 0x00003ffe //   log(1/frcpa(1+216/2^-8))
data8 0x9d7e67f3bdce8888 , 0x00003ffe //   log(1/frcpa(1+217/2^-8))
data8 0x9df4ea81a99daa01 , 0x00003ffe //   log(1/frcpa(1+218/2^-8))
data8 0x9e6ba405a54514ba , 0x00003ffe //   log(1/frcpa(1+219/2^-8))
data8 0x9f1e21c8c7bb62b3 , 0x00003ffe //   log(1/frcpa(1+220/2^-8))
//
data8 0x9f956593f6b6355c , 0x00003ffe //   log(1/frcpa(1+221/2^-8))
data8 0xa00ce1092e5498c3 , 0x00003ffe //   log(1/frcpa(1+222/2^-8))
data8 0xa0c08309c4b912c1 , 0x00003ffe //   log(1/frcpa(1+223/2^-8))
data8 0xa1388a8c6faa2afa , 0x00003ffe //   log(1/frcpa(1+224/2^-8))
data8 0xa1b0ca7095b5f985 , 0x00003ffe //   log(1/frcpa(1+225/2^-8))
//
data8 0xa22942eb47534a00 , 0x00003ffe //   log(1/frcpa(1+226/2^-8))
data8 0xa2de62326449d0a3 , 0x00003ffe //   log(1/frcpa(1+227/2^-8))
data8 0xa357690f88bfe345 , 0x00003ffe //   log(1/frcpa(1+228/2^-8))
data8 0xa3d0a93f45169a4b , 0x00003ffe //   log(1/frcpa(1+229/2^-8))
data8 0xa44a22f7ffe65f30 , 0x00003ffe //   log(1/frcpa(1+230/2^-8))
//
data8 0xa500c5e5b4c1aa36 , 0x00003ffe //   log(1/frcpa(1+231/2^-8))
data8 0xa57ad064eb2ebbc2 , 0x00003ffe //   log(1/frcpa(1+232/2^-8))
data8 0xa5f5152dedf4384e , 0x00003ffe //   log(1/frcpa(1+233/2^-8))
data8 0xa66f9478856233ec , 0x00003ffe //   log(1/frcpa(1+234/2^-8))
data8 0xa6ea4e7cca02c32e , 0x00003ffe //   log(1/frcpa(1+235/2^-8))
//
data8 0xa765437325341ccf , 0x00003ffe //   log(1/frcpa(1+236/2^-8))
data8 0xa81e21e6c75b4020 , 0x00003ffe //   log(1/frcpa(1+237/2^-8))
data8 0xa899ab333fe2b9ca , 0x00003ffe //   log(1/frcpa(1+238/2^-8))
data8 0xa9157039c51ebe71 , 0x00003ffe //   log(1/frcpa(1+239/2^-8))
data8 0xa991713433c2b999 , 0x00003ffe //   log(1/frcpa(1+240/2^-8))
//
data8 0xaa0dae5cbcc048b3 , 0x00003ffe //   log(1/frcpa(1+241/2^-8))
data8 0xaa8a27ede5eb13ad , 0x00003ffe //   log(1/frcpa(1+242/2^-8))
data8 0xab06de228a9e3499 , 0x00003ffe //   log(1/frcpa(1+243/2^-8))
data8 0xab83d135dc633301 , 0x00003ffe //   log(1/frcpa(1+244/2^-8))
data8 0xac3fb076adc7fe7a , 0x00003ffe //   log(1/frcpa(1+245/2^-8))
//
data8 0xacbd3cbbe47988f1 , 0x00003ffe //   log(1/frcpa(1+246/2^-8))
data8 0xad3b06b1a5dc57c3 , 0x00003ffe //   log(1/frcpa(1+247/2^-8))
data8 0xadb90e94af887717 , 0x00003ffe //   log(1/frcpa(1+248/2^-8))
data8 0xae3754a218f7c816 , 0x00003ffe //   log(1/frcpa(1+249/2^-8))
data8 0xaeb5d9175437afa2 , 0x00003ffe //   log(1/frcpa(1+250/2^-8))
//
data8 0xaf349c322e9c7cee , 0x00003ffe //   log(1/frcpa(1+251/2^-8))
data8 0xafb39e30d1768d1c , 0x00003ffe //   log(1/frcpa(1+252/2^-8))
data8 0xb032df51c2c93116 , 0x00003ffe //   log(1/frcpa(1+253/2^-8))
data8 0xb0b25fd3e6035ad9 , 0x00003ffe //   log(1/frcpa(1+254/2^-8))
data8 0xb1321ff67cba178c , 0x00003ffe //   log(1/frcpa(1+255/2^-8))
LOCAL_OBJECT_END(log_table_3)


.section .text
GLOBAL_LIBM_ENTRY(acosh)

{ .mfi
      getf.exp   acosh_GR_f8 = f8
      fclass.m   p6,p0 = f8, 0xc3                    // Test for x = NaN
      mov        log_GR_comp2 = 0x1003e
}
{ .mfi
      addl       NR_table_address = @ltoff(log_table_1), gp
      fms.s1     log_y = f8, f8, f1                  // y = x^2-1
      nop.i      0
}
;;

{ .mfi
      getf.sig   acosh_GR_f8_sig = f8
      fclass.m   p11,p0 = f8, 0x21                   // Test for x=+inf
      mov        log_GR_exp_17_ones = 0x1ffff
}
{ .mfi
      ld8        NR_table_address = [NR_table_address]
      fms.s1     log_w = f8,f1,f1                    // w = x - 1
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fcmp.lt.s1 p7,p8 = f8, f1            // Test for x<1.0
      addl       log_GR_comp = 0x10020C,r0 // Upper 21 bits of signif of 1.0005
}
{ .mfb
      mov        log_GR_exp_16_ones = 0xffff         //BIAS
(p6)  fma.d.s0   f8 = f8,f1,f0      // quietize nan result if x=nan
(p6)  br.ret.spnt b0                // Exit for x=nan
}
;;

{ .mfb
      //get second table address
      adds       log_table_address2 = 0x40, NR_table_address
      fcmp.eq.s1 p10,p0 = f8, f1      // Test for x=+1.0
(p11) br.ret.spnt b0                  // Exit for x=+inf
}
;;

{ .mfi
      ldfpd      NR1,NR2 = [log_table_address2],16
      frsqrta.s1 log_y_rs,p0 = log_y  // z=1/sqrt(y)
      nop.i      0
}
{ .mfb
      nop.m      0
      fma.s1     log_arg = f8,f1,f8
(p7)  br.cond.spnt ACOSH_LESS_ONE     // Branch if path 7, x < 1.0
}
;;

{ .mfi
      ldfe       log_C4 = [log_table_address2],16
(p8)  fcmp.eq.s0 p6,p0 = f8, f0       // Dummy op sets denorm flag if unorm>=1.0
      nop.i      0
}
{ .mfb
(p8)  cmp.le.unc p13,p0 = log_GR_comp2,acosh_GR_f8
      nop.f      0
(p13) br.cond.spnt LOG_COMMON1        // Branch if path 4, x >= 2^63
}
;;

{ .mfi
      ldfe       log_C3 = [log_table_address2],16
(p10) fmerge.s   f8 = f0, f0          // Return 0 if x=1.0
      shr.u      acosh_GR_f8_sig = acosh_GR_f8_sig,43
}
{ .mib
      cmp.eq     p14,p0 = log_GR_exp_16_ones,acosh_GR_f8
      nop.i      0
(p10) br.ret.spnt b0                  // Exit for x=1.0
}
;;

{ .mfi
      ldfe       log_C2 = [log_table_address2],16
      frsqrta.s1 acosh_w_rs,p0 = log_w // t=1/sqrt(w)
      nop.i      0
}
{ .mfb
(p14) cmp.lt.unc p15,p0 = acosh_GR_f8_sig,log_GR_comp
      nop.f      0
(p15) br.cond.spnt ACOSH_NEAR_ONE     // Branch if path 2, 1.0 < x < 1.0005
}
;;

// Here is main path, 1.0005 <= x < 2^63
/////////////// The first iteration //////////////////////////////////
{ .mfi
      ldfpd      acosh_comp,log_P5 = [NR_table_address],16
      fma.s1     log_y_rs_iter = log_y_rs,log_y,f0              // y*z
      nop.i      0
}
;;

{ .mfi
      ldfpd      log_P4,log_P3 = [NR_table_address],16
      fnma.s1    log_y_rs_iter = log_y_rs_iter,log_y_rs,NR2     // 3-(y*z)*z
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_y_rs_iter1 = log_y_rs,NR1,f0               // 0.5*z
      nop.i      0
}
;;

{ .mfi
      ldfpd      log_P2,log_P1 = [NR_table_address],16
      //(0.5*z)*(3-(y*z)*z)
      fma.s1     log_y_rs_iter = log_y_rs_iter1,log_y_rs_iter,f0
      nop.i      0
}
;;

/////////////////////////// The second iteration /////////////////////////////
{ .mfi
      nop.m      0
      fma.s1     log_y_rs = log_y_rs_iter,log_y,f0
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
      fma.s1     log_y_rs_iter1 = log_y_rs_iter,NR1,f0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      //(0.5*z)*(3-(y*z)*z)
      fma.s1     log_y_rs_iter = log_y_rs_iter1,log_y_rs,f0
      nop.i      0
}
{ .mfi
      nop.m      0
      //(0.5*z)*(3-(y*z)*z)
      fma.s1     log_arg_early = log_y_rs_iter1,log_y_rs,f0
      nop.i      0
}
;;

//////////////////////////////////////// The third iteration /////////////////
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
      fma.s1     log_arg_early = log_arg_early,log_y,f8
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
      fma.s1     log_arg = log_y_rs_iter1,log_y_rs,f8 // (0.5*z)*(3-(y*z)*z)
      adds       log_table_address3 = 0x70, NR_table_address
}
;;

///////////////////////////////// The end NR iterations /////////////////////
{ .mfi
      ldfe       log2 = [NR_table_address],16
      nop.f      0
      nop.i      0
}
;;

{ .mmi
      //significant bit destruction
      and        log_GR_exp_f8 = log_GR_signexp_f8, log_GR_exp_17_ones
;;
      //BIAS subtraction
      sub        log_GR_true_exp_f8 = log_GR_exp_f8, log_GR_exp_16_ones
      nop.i      0
}
;;

{ .mfi
      setf.sig   log_int_Nfloat = log_GR_true_exp_f8
      fms.s1     log_r = log_C,log_arg,f1  // C = frcpa(x); r = C * x - 1
      extr.u     log_GR_index = log_GR_significand_f8,55,8 //Extract 8 bits
}
;;

{ .mmi
      //pre-index*16 + index
      shladd     log_table_address3 = log_GR_index,4,log_table_address3
;;
      ldfe       log_T = [log_table_address3]
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rsq = log_r, log_r, f0         //r^2
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_rp_p4 = log_P5, log_r, log_P4  //P5*r + P4
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rp_p32 = log_P3, log_r, log_P2 //P3*r + P2
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      //convert N to the floating-point format log_Nfloat
      fcvt.xf    log_Nfloat = log_int_Nfloat
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rcube = log_rsq, log_r, f0      //r^3
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_rp_p10 = log_rsq, log_P1, log_r //P1*r^2 + r
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      //(P5*r + P4)*r^2 + P3*r + P2
      fma.s1     log_rp_p2 = log_rp_p4, log_rsq, log_rp_p32
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_T_plus_Nlog2 = log_Nfloat,log2,log_T    //N*log2 + T
      nop.i      0
}
{ .mfi
      nop.m      0
      //((P5*r + P4)*r^2 + P3*r + P2)*r^3 + P1*r^2 + r
      fma.s1     log_r2P_r = log_rp_p2, log_rcube, log_rp_p10
      nop.i      0
}
;;

{ .mfb
      nop.m      0
      // N*log2 + T + ((P5*r + P4)*r^2 + P3*r + P2)*w^3 + P1*r^2 + r
      fadd.d.s0  f8 = log_T_plus_Nlog2, log_r2P_r
      br.ret.sptk b0           // Exit main path, path 3: 1.0005 <= x < 2^63
}
;;

// Here if path 2, 1.0 < x < 1.0005
ACOSH_NEAR_ONE:
// The first NR iteration
{ .mfi
      ldfe       log_C1 = [log_table_address2],16
      fma.s1     acosh_w_iter1 = acosh_w_rs,log_w,f0  //t*w
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     acosh_w_1 = f8,log_C4,log_C3         //x*C4 + C3
      nop.i      0
}
;;

{ .mfi
      ldfe       log_C0 = [log_table_address2],16
      fma.s1     acosh_w_iter2 = acosh_w_rs,NR1,f0    //t*0.5
      nop.i      0
}
{ .mfi
      nop.m      0
      fnma.s1    acosh_w_iter1 = acosh_w_iter1,acosh_w_rs,NR2 //3-t*t*w
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      //(3-t*t*w)*t*0.5
      fma.s1     acosh_w_iter2 = acosh_w_iter2,acosh_w_iter1,f0
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     acosh_w_1 = acosh_w_1,log_w,log_C2 //(x*C4 + C3)*(x-1) + C2
      nop.i      0
}
;;

// The second NR iteration
{ .mfi
      nop.m      0
      fma.s1     acosh_w_rs = acosh_w_iter2,log_w,f0  //t*w
      nop.i      0
}
{ .mfi
      nop.m      0
      //((x*C4 + C3)*(x-1) + C2)*(x-1) + C1
      fma.s1     acosh_w_1 = acosh_w_1,log_w,log_C1
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fnma.s1    acosh_w_iter1 = acosh_w_iter2,acosh_w_rs,NR2
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     acosh_w_iter2 = acosh_w_iter2,NR1,f0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     acosh_w_iter2 = acosh_w_iter2,acosh_w_iter1,f0
      nop.i      0
}
{ .mfi
      nop.m      0
      //(((x*C4 + C3)*(x-1) + C2)*(x-1) + C1)*(x-1) + C0
      fma.s1     acosh_w_1 = acosh_w_1,log_w,log_C0
      nop.i      0
}
;;

//The third NR iteration
{ .mfi
      nop.m      0
      fma.s1     acosh_w_rs = acosh_w_iter2,log_w,f0  //t*w
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fnma.s1    acosh_w_iter1 = acosh_w_iter2,acosh_w_rs,NR2
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     acosh_w_iter2 = acosh_w_iter2,NR1,f0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     acosh_w_iter2 = acosh_w_iter2,acosh_w_iter1,f0
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     acosh_w_sqrt = acosh_w_iter2,log_w,f0
      nop.i      0
}
;;

{ .mfb
      nop.m      0
      fma.d.s0   f8 = acosh_w_1,acosh_w_sqrt,f0
      br.ret.sptk b0               // Exit path 2, 1.0 < x < 1.0005
}
;;

// Here if path 4, x >= 2^63
LOG_COMMON1:
{ .mfi
      ldfpd      acosh_comp,log_P5 = [NR_table_address],16
      frcpa.s1   log_C,p0 = f1,log_arg
      nop.i      0
}
;;

{ .mmi
      getf.exp   log_GR_signexp_f8 = log_arg
      ldfpd      log_P4,log_P3 = [NR_table_address],16
      nop.i      0
}
;;

{ .mmi
      getf.sig   log_GR_significand_f8 = log_arg
      ldfpd      log_P2,log_P1 = [NR_table_address],16
      nop.i      0
}
;;

{ .mfi
      adds       log_table_address3 = 0x70, NR_table_address
      nop.f      0
      //significant bit destruction
      and        log_GR_exp_f8 = log_GR_signexp_f8, log_GR_exp_17_ones
}
;;

{ .mmf
      ldfe       log2 = [NR_table_address],16
      //BIAS subtraction
      sub        log_GR_true_exp_f8 = log_GR_exp_f8, log_GR_exp_16_ones
      fms.s1     log_r = log_C,log_arg,f1  // C = frcpa(x); r = C * x - 1
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
      shladd     log_table_address3 = log_GR_index,4,log_table_address3
;;
      ldfe       log_T = [log_table_address3]
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rsq = log_r, log_r, f0         //r^2
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_rp_p4 = log_P5, log_r, log_P4  //P5*r + P4
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rp_p32 = log_P3, log_r, log_P2 //P3*r + P2
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_rcube = log_rsq, log_r, f0     //r^3
      nop.i      0
}
{ .mfi
      nop.m      0
      fma.s1     log_rp_p10 = log_rsq, log_P1, log_r //P1*r^2 + r
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      //convert N to the floating-point format log_Nfloat
      fcvt.xf    log_Nfloat = log_int_Nfloat
      nop.i      0
}
{ .mfi
      nop.m      0
      //(P5*r + P4)*r^2 + P3*r + P2
      fma.s1     log_rp_p2 = log_rp_p4, log_rsq, log_rp_p32
      nop.i      0
}
;;

{ .mfi
      nop.m      0
      fma.s1     log_T_plus_Nlog2 = log_Nfloat,log2,log_T    //N*log2 + T
      nop.i      0
}
{ .mfi
      nop.m      0
      //((P5*r + P4)*r^2 + P3*r + P2)*w^3 + P1*r^2 + r
      fma.s1     log_r2P_r = log_rp_p2, log_rcube, log_rp_p10
      nop.i      0
}
;;

{ .mfb
      nop.m      0
      //  N*log2 + T + ((P5*r + P4)*r^2 + P3*r + P2)*w^3 + P1*r^2 + r
      fadd.d.s0  f8 = log_T_plus_Nlog2, log_r2P_r
      br.ret.sptk b0              // Exit path 4, x >= 2^63
}
;;

// Here if path 7, x < 1.0
ACOSH_LESS_ONE:
{ .mfi
      alloc      r32 = ar.pfs,1,3,4,0
      fmerge.s   f10 = f8,f8
      nop.i      0
}
;;

{ .mfb
      mov        acosh_GR_tag = 136
      frcpa.s0   f8,p0 = f0,f0
      br.cond.sptk __libm_error_region
}
;;

GLOBAL_LIBM_END(acosh)
libm_alias_double_other (acosh, acosh)


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
        add sp=-64,sp                          // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                      // Save gp
};;

{ .mmi
        stfd [GR_Parameter_Y] = f1,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp            // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                     // Save b0
};;

.body
{ .mib
        stfd [GR_Parameter_X] = f10           // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
        nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = f8            // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support# // Call error handling function
};;

{ .mmi
        add   GR_Parameter_RESULT = 48,sp
        nop.m 0
        nop.i 0
};;

{ .mmi
        ldfd  f8 = [GR_Parameter_RESULT]       // Get return result off stack
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
