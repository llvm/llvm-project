.file "exp10l.s"


// Copyright (c) 2000 - 2004, Intel Corporation
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
// 08/25/00 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align
// 05/08/03 Reformatted assembly source; corrected overflow result for round to
//          -inf and round to zero; exact results now don't set inexact flag
// 12/16/04 Call error handling on underflow.
//
// API
//==============================================================
// long double exp10l(long double)
//
// Overview of operation
//==============================================================
// Background
//
// Implementation
//
// Let x= (K + f + r)/log2(10), where
// K is an integer, f= 0.b1 b2... b8 (f>= 0),
// and |r|<2^{-9}
// T is a table that stores 2^f (256 entries) rounded to
// double extended precision (only mantissa is stored)
// D stores (2^f/T [ f ] - 1), rounded to single precision
//
// 10^x is approximated as
// 2^K * T [ f ] * ((1+c1*r+c2*r^2+...+c6*r^6)*(1+c1*e)+D [ f ] ),
// where e= log2(10)_lo*x+(log2(10)_hi*x-RN(log2(10)_hi*x))
//



// Special values
//==============================================================
// exp10(0)= 1
// exp10(+inf)= inf
// exp10(-inf)= 0
//


// Registers used
//==============================================================
// f6-f15, f32-f63
// r14-r30, r32-r40
// p6-p8, p11-p14
//

#include <shlib-compat.h>


       FR_X        = f10
       FR_Y        = f1
       FR_RESULT   = f8

       FR_COEFF1   = f6
       FR_COEFF2   = f7
       FR_KF0      = f9
       FR_LOG10    = f10
       FR_CONST1   = f11
       FR_XL10     = f12
       FR_COEFF3   = f13
       FR_COEFF4   = f14
       FR_UF_TEST  = f15
       FR_OF_TEST  = f32
       FR_L10_LOW  = f33
       FR_COEFF5   = f34
       FR_COEFF6   = f35
       FR_L10      = f36
       FR_C_L10    = f37
       FR_XL10_H   = f38
       FR_XL10_L   = f39
       FR_KF       = f40
       FR_E        = f41
       FR_T        = f42
       FR_D        = f43
       FR_EXP_M_63 = f44
       FR_R        = f45
       FR_E1       = f46
       FR_COEFF2   = f47
       FR_P34      = f48
       FR_P56      = f49
       FR_R2       = f50
       FR_RE       = f51
       FR_D1       = f52
       FR_P36      = f53
       FR_R3E      = f54
       FR_P1       = f55
       FR_P        = f56
       FR_T1       = f57
       FR_XINT     = f58
       FR_XINTF    = f59
       FR_4        = f60
       FR_28       = f61
       FR_32       = f62
       FR_SNORM_LIMIT = f63


       GR_ADDR0    = r14
       GR_D_ADDR   = r15
       GR_ADDR     = r16
       GR_B63      = r17
       GR_KBITS    = r18
       GR_F        = r19
       GR_K        = r20
       GR_D        = r21
       GR_BM63     = r22
       GR_T        = r23
       GR_CONST1   = r24
       GR_EMIN     = r25
       GR_CONST2   = r26
       GR_BM8      = r27
       GR_SREG     = r28
       GR_4_BIAS   = r29
       GR_32_BIAS  = r30

       GR_SAVE_B0  = r33
       GR_SAVE_PFS = r34
       GR_SAVE_GP  = r35
       GR_SAVE_SP  = r36

       GR_Parameter_X     = r37
       GR_Parameter_Y     = r38
       GR_Parameter_RESULT= r39
       GR_Parameter_TAG   = r40


// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(poly_coeffs)

       data8 0xd49a784bcd1b8afe, 0x00004008 // log2(10)*2^8
       data8 0x9a209a84fbcff798, 0x0000400b // overflow threshold
       data8 0xb17217f7d1cf79ab, 0x00003ffe // C_1
       data8 0xf5fdeffc162c7541, 0x00003ffc // C_2
       data8 0x3fac6b08d704a0c0 // C_3
       data8 0x3f83b2ab6fba4e77 // C_4
       data8 0x3f55d87fe78a6731 // C_5
       data8 0x3f2430912f86c787 // C_6
       data8 0x9257edfe9b5fb698, 0x00003fbf // log2(10)_low (bits 64...127)
       data8 0x9a1bc98027a81918, 0x0000c00b // Smallest normal threshold
LOCAL_OBJECT_END(poly_coeffs)


LOCAL_OBJECT_START(T_table)

       // 2^{0.b1 b2 b3 b4 b5 b6 b7 b8}
       data8 0x8000000000000000, 0x8058d7d2d5e5f6b1
       data8 0x80b1ed4fd999ab6c, 0x810b40a1d81406d4
       data8 0x8164d1f3bc030773, 0x81bea1708dde6056
       data8 0x8218af4373fc25ec, 0x8272fb97b2a5894c
       data8 0x82cd8698ac2ba1d7, 0x83285071e0fc4547
       data8 0x8383594eefb6ee37, 0x83dea15b9541b132
       data8 0x843a28c3acde4046, 0x8495efb3303efd30
       data8 0x84f1f656379c1a29, 0x854e3cd8f9c8c95d
       data8 0x85aac367cc487b15, 0x86078a2f23642a9f
       data8 0x8664915b923fba04, 0x86c1d919caef5c88
       data8 0x871f61969e8d1010, 0x877d2afefd4e256c
       data8 0x87db357ff698d792, 0x88398146b919f1d4
       data8 0x88980e8092da8527, 0x88f6dd5af155ac6b
       data8 0x8955ee03618e5fdd, 0x89b540a7902557a4
       data8 0x8a14d575496efd9a, 0x8a74ac9a79896e47
       data8 0x8ad4c6452c728924, 0x8b3522a38e1e1032
       data8 0x8b95c1e3ea8bd6e7, 0x8bf6a434adde0085
       data8 0x8c57c9c4646f4dde, 0x8cb932c1bae97a95
       data8 0x8d1adf5b7e5ba9e6, 0x8d7ccfc09c50e2f8
       data8 0x8ddf042022e69cd6, 0x8e417ca940e35a01
       data8 0x8ea4398b45cd53c0, 0x8f073af5a2013520
       data8 0x8f6a8117e6c8e5c4, 0x8fce0c21c6726481
       data8 0x9031dc431466b1dc, 0x9095f1abc540ca6b
       data8 0x90fa4c8beee4b12b, 0x915eed13c89689d3
       data8 0x91c3d373ab11c336, 0x9228ffdc10a051ad
       data8 0x928e727d9531f9ac, 0x92f42b88f673aa7c
       data8 0x935a2b2f13e6e92c, 0x93c071a0eef94bc1
       data8 0x9426ff0fab1c04b6, 0x948dd3ac8ddb7ed3
       data8 0x94f4efa8fef70961, 0x955c5336887894d5
       data8 0x95c3fe86d6cc7fef, 0x962bf1cbb8d97560
       data8 0x96942d3720185a00, 0x96fcb0fb20ac4ba3
       data8 0x97657d49f17ab08e, 0x97ce9255ec4357ab
       data8 0x9837f0518db8a96f, 0x98a1976f7597e996
       data8 0x990b87e266c189aa, 0x9975c1dd47518c77
       data8 0x99e0459320b7fa65, 0x9a4b13371fd166ca
       data8 0x9ab62afc94ff864a, 0x9b218d16f441d63d
       data8 0x9b8d39b9d54e5539, 0x9bf93118f3aa4cc1
       data8 0x9c6573682ec32c2d, 0x9cd200db8a0774cb
       data8 0x9d3ed9a72cffb751, 0x9dabfdff6367a2aa
       data8 0x9e196e189d472420, 0x9e872a276f0b98ff
       data8 0x9ef5326091a111ae, 0x9f6386f8e28ba651
       data8 0x9fd228256400dd06, 0xa041161b3d0121be
       data8 0xa0b0510fb9714fc2, 0xa11fd9384a344cf7
       data8 0xa18faeca8544b6e4, 0xa1ffd1fc25cea188
       data8 0xa27043030c496819, 0xa2e102153e918f9e
       data8 0xa3520f68e802bb93, 0xa3c36b345991b47c
       data8 0xa43515ae09e6809e, 0xa4a70f0c95768ec5
       data8 0xa5195786be9ef339, 0xa58bef536dbeb6ee
       data8 0xa5fed6a9b15138ea, 0xa6720dc0be08a20c
       data8 0xa6e594cfeee86b1e, 0xa7596c0ec55ff55b
       data8 0xa7cd93b4e965356a, 0xa8420bfa298f70d1
       data8 0xa8b6d5167b320e09, 0xa92bef41fa77771b
       data8 0xa9a15ab4ea7c0ef8, 0xaa1717a7b5693979
       data8 0xaa8d2652ec907629, 0xab0386ef48868de1
       data8 0xab7a39b5a93ed337, 0xabf13edf162675e9
       data8 0xac6896a4be3fe929, 0xace0413ff83e5d04
       data8 0xad583eea42a14ac6, 0xadd08fdd43d01491
       data8 0xae493452ca35b80e, 0xaec22c84cc5c9465
       data8 0xaf3b78ad690a4375, 0xafb51906e75b8661
       data8 0xb02f0dcbb6e04584, 0xb0a957366fb7a3c9
       data8 0xb123f581d2ac2590, 0xb19ee8e8c94feb09
       data8 0xb21a31a66618fe3b, 0xb295cff5e47db4a4
       data8 0xb311c412a9112489, 0xb38e0e38419fae18
       data8 0xb40aaea2654b9841, 0xb487a58cf4a9c180
       data8 0xb504f333f9de6484, 0xb58297d3a8b9f0d2
       data8 0xb60093a85ed5f76c, 0xb67ee6eea3b22b8f
       data8 0xb6fd91e328d17791, 0xb77c94c2c9d725e9
       data8 0xb7fbefca8ca41e7c, 0xb87ba337a1743834
       data8 0xb8fbaf4762fb9ee9, 0xb97c143756844dbf
       data8 0xb9fcd2452c0b9deb, 0xba7de9aebe5fea09
       data8 0xbaff5ab2133e45fb, 0xbb81258d5b704b6f
       data8 0xbc034a7ef2e9fb0d, 0xbc85c9c560e7b269
       data8 0xbd08a39f580c36bf, 0xbd8bd84bb67ed483
       data8 0xbe0f6809860993e2, 0xbe935317fc378238
       data8 0xbf1799b67a731083, 0xbf9c3c248e2486f8
       data8 0xc0213aa1f0d08db0, 0xc0a6956e8836ca8d
       data8 0xc12c4cca66709456, 0xc1b260f5ca0fbb33
       data8 0xc238d2311e3d6673, 0xc2bfa0bcfad907c9
       data8 0xc346ccda24976407, 0xc3ce56c98d21b15d
       data8 0xc4563ecc5334cb33, 0xc4de8523c2c07baa
       data8 0xc5672a115506dadd, 0xc5f02dd6b0bbc3d9
       data8 0xc67990b5aa245f79, 0xc70352f04336c51e
       data8 0xc78d74c8abb9b15d, 0xc817f681416452b2
       data8 0xc8a2d85c8ffe2c45, 0xc92e1a9d517f0ecc
       data8 0xc9b9bd866e2f27a3, 0xca45c15afcc72624
       data8 0xcad2265e4290774e, 0xcb5eecd3b38597c9
       data8 0xcbec14fef2727c5d, 0xcc799f23d11510e5
       data8 0xcd078b86503dcdd2, 0xcd95da6a9ff06445
       data8 0xce248c151f8480e4, 0xceb3a0ca5dc6a55d
       data8 0xcf4318cf191918c1, 0xcfd2f4683f94eeb5
       data8 0xd06333daef2b2595, 0xd0f3d76c75c5db8d
       data8 0xd184df6251699ac6, 0xd2164c023056bcab
       data8 0xd2a81d91f12ae45a, 0xd33a5457a3029054
       data8 0xd3ccf099859ac379, 0xd45ff29e0972c561
       data8 0xd4f35aabcfedfa1f, 0xd5872909ab75d18a
       data8 0xd61b5dfe9f9bce07, 0xd6aff9d1e13ba2fe
       data8 0xd744fccad69d6af4, 0xd7da67311797f56a
       data8 0xd870394c6db32c84, 0xd9067364d44a929c
       data8 0xd99d15c278afd7b6, 0xda3420adba4d8704
       data8 0xdacb946f2ac9cc72, 0xdb63714f8e295255
       data8 0xdbfbb797daf23755, 0xdc9467913a4f1c92
       data8 0xdd2d818508324c20, 0xddc705bcd378f7f0
       data8 0xde60f4825e0e9124, 0xdefb4e1f9d1037f2
       data8 0xdf9612deb8f04420, 0xe031430a0d99e627
       data8 0xe0ccdeec2a94e111, 0xe168e6cfd3295d23
       data8 0xe2055afffe83d369, 0xe2a23bc7d7d91226
       data8 0xe33f8972be8a5a51, 0xe3dd444c46499619
       data8 0xe47b6ca0373da88d, 0xe51a02ba8e26d681
       data8 0xe5b906e77c8348a8, 0xe658797368b3a717
       data8 0xe6f85aaaee1fce22, 0xe798aadadd5b9cbf
       data8 0xe8396a503c4bdc68, 0xe8da9958464b42ab
       data8 0xe97c38406c4f8c57, 0xea1e4756550eb27b
       data8 0xeac0c6e7dd24392f, 0xeb63b74317369840
       data8 0xec0718b64c1cbddc, 0xecaaeb8ffb03ab41
       data8 0xed4f301ed9942b84, 0xedf3e6b1d418a491
       data8 0xee990f980da3025b, 0xef3eab20e032bc6b
       data8 0xefe4b99bdcdaf5cb, 0xf08b3b58cbe8b76a
       data8 0xf13230a7ad094509, 0xf1d999d8b7708cc1
       data8 0xf281773c59ffb13a, 0xf329c9233b6bae9c
       data8 0xf3d28fde3a641a5b, 0xf47bcbbe6db9fddf
       data8 0xf5257d152486cc2c, 0xf5cfa433e6537290
       data8 0xf67a416c733f846e, 0xf7255510c4288239
       data8 0xf7d0df730ad13bb9, 0xf87ce0e5b2094d9c
       data8 0xf92959bb5dd4ba74, 0xf9d64a46eb939f35
       data8 0xfa83b2db722a033a, 0xfb3193cc4227c3f4
       data8 0xfbdfed6ce5f09c49, 0xfc8ec01121e447bb
       data8 0xfd3e0c0cf486c175, 0xfdedd1b496a89f35
       data8 0xfe9e115c7b8f884c, 0xff4ecb59511ec8a5
LOCAL_OBJECT_END(T_table)


LOCAL_OBJECT_START(D_table)
       data4 0x00000000, 0x9f55c08f, 0x1e93ffa3, 0x1dcd43a8
       data4 0x1f751f79, 0x9f3cdd88, 0x9f43d155, 0x1eda222c
       data4 0x1ef35513, 0x9f597895, 0x9e698881, 0x1ec71073
       data4 0x1e50e371, 0x9dc01e19, 0x1de74133, 0x1e2f028c
       data4 0x9edefb47, 0x1ebbac48, 0x9e8b0330, 0x9e9e9314
       data4 0x1edc1d11, 0x1f098529, 0x9f52827c, 0x1f50050d
       data4 0x1f301e8e, 0x1f5b64d1, 0x9f45e3ee, 0x9ef64d6d
       data4 0x1d6ec5e8, 0x9e61ad9a, 0x1d44ccbb, 0x9e4a8bbb
       data4 0x9cf11576, 0x9dcce7e7, 0x9d02ac90, 0x1f26ccf0
       data4 0x9f0877c6, 0x9ddd62ae, 0x9f4b7fc3, 0x1ea8ef6b
       data4 0x1ea4378d, 0x1ef6fc38, 0x1db99fd9, 0x1f22bf6f
       data4 0x1f53e172, 0x1e85504a, 0x9f37cc75, 0x1f0c5e17
       data4 0x1dde8aac, 0x9cb42bb2, 0x1e153cd7, 0x1eb62bba
       data4 0x9e9b941b, 0x9ea80e3c, 0x1f508823, 0x1ec3fd36
       data4 0x1e9ffaa1, 0x1e21e2eb, 0x9d948b1d, 0x9e8ac93a
       data4 0x1ef7ee6f, 0x9e80dda3, 0x1f0814be, 0x1dc5ddfe
       data4 0x1eedb9d1, 0x9f2aaa26, 0x9ea5b0fc, 0x1edf702e
       data4 0x9e391201, 0x1f1316bb, 0x1ea27fb7, 0x9e05ed18
       data4 0x9f199ed2, 0x1ee7fd7c, 0x1f003db6, 0x9eac3793
       data4 0x9e5b8c10, 0x9f3af17c, 0x1bc9a8be, 0x1ee3c004
       data4 0x9f19b1b2, 0x9f242ce9, 0x9ce67dd1, 0x9e4f6275
       data4 0x1e20742c, 0x1eb9328a, 0x9f477153, 0x1d969718
       data4 0x9f1e6c43, 0x1f2f67f4, 0x9f39c7e4, 0x9e3c4feb
       data4 0x1da3956b, 0x9e7c685d, 0x1f280911, 0x9f0d8afb
       data4 0x1e314b40, 0x9eb4f250, 0x9f1a34ad, 0x1ef5d5e7
       data4 0x9f145496, 0x1e604827, 0x9f1e5195, 0x1e9c1fc0
       data4 0x1efde521, 0x1e69b385, 0x1f316830, 0x9f244eae
       data4 0x1f1787ec, 0x9e939971, 0x1f0bb393, 0x9f0511d6
       data4 0x1ed919de, 0x1d8b7b28, 0x1e5ca4a9, 0x1e7c357b
       data4 0x9e3ff8e8, 0x1eef53b5, 0x9ed22ed7, 0x1f16659b
       data4 0x9f2db102, 0x9e2c6a78, 0x1f328d7d, 0x9f2fec3c
       data4 0x1eb395bd, 0x9f242b84, 0x9e2683e6, 0x1ed71e68
       data4 0x1efd1df5, 0x9e9eeafd, 0x9ed2249c, 0x1eef129a
       data4 0x1d1ea44c, 0x9e81f7ff, 0x1eaf77c9, 0x9ee7a285
       data4 0x1e1864ed, 0x9ee7edbb, 0x9e15a27d, 0x9ae61655
       data4 0x1f1ff1a2, 0x1da29755, 0x9e5f46fb, 0x1e901236
       data4 0x9eecfb9b, 0x9f204d2f, 0x1ec64685, 0x9eb809bd
       data4 0x9e0026c5, 0x1d9f1da1, 0x1f142b49, 0x9f20f22e
       data4 0x1f24b067, 0x1f185a4c, 0x9f09765c, 0x9ece902f
       data4 0x1e2ca5db, 0x1e6de464, 0x9f071f67, 0x1f1518c3
       data4 0x1ea13ded, 0x1f0b8414, 0x1edb6ad4, 0x9e548740
       data4 0x9ea10efb, 0x1ee48a60, 0x1e7954c5, 0x9edad013
       data4 0x9f21517d, 0x9e9b6e0c, 0x9ee7f9a6, 0x9ebd4298
       data4 0x9d65b24e, 0x1eed751f, 0x9f1573ea, 0x9d430377
       data4 0x9e13fc0c, 0x1e47008a, 0x1e3d5c1d, 0x1ef41a91
       data4 0x9e4a4ef7, 0x9e952f18, 0x1d620566, 0x1d9b8d33
       data4 0x1db06247, 0x1e94b31e, 0x1f0730ad, 0x9d79ffb4
       data4 0x1ed64d51, 0x9e91fd11, 0x9e28d35a, 0x9dea0ed9
       data4 0x1e891def, 0x9ee28ac0, 0x1e1db99b, 0x9ee1ce38
       data4 0x9bdd9bca, 0x1eb72cb9, 0x9e8c53c6, 0x1e0df6ca
       data4 0x1e8f2ccd, 0x9e9b0886, 0x1eeb3bc7, 0x1ec7e772
       data4 0x9e210776, 0x9daf246c, 0x1ea1f151, 0x1ece4dc6
       data4 0x1ce741c8, 0x1ed3c88f, 0x9ec9a4fd, 0x9e0c8d30
       data4 0x1d2fbb26, 0x9ef212a7, 0x1ee44f1c, 0x9e445550
       data4 0x1e075f77, 0x9d9291a3, 0x1f09c2ee, 0x9e012c88
       data4 0x1f057d62, 0x9e7bb0dc, 0x9d8758ee, 0x1ee8d6c1
       data4 0x9e509a57, 0x9e4ca7b7, 0x1e2cb341, 0x9ec35106
       data4 0x1ecf3baf, 0x1e11781c, 0x1ea0cc78, 0x1eb75ca6
       data4 0x1e961e1a, 0x1eb88853, 0x1e7abf50, 0x1ee38704
       data4 0x9dc5ab0f, 0x1afe197b, 0x9ec07523, 0x9d9b7f78
       data4 0x1f011618, 0x1ed43b0b, 0x9f035945, 0x9e3fd014
       data4 0x9bbda5cd, 0x9e83f8ab, 0x1e58a928, 0x1e392d61
       data4 0x1efdbb52, 0x1ee310a8, 0x9ec7ecc1, 0x1e8c9ed6
       data4 0x9ef82dee, 0x9e70545b, 0x9ea53fc4, 0x1e40f419
LOCAL_OBJECT_END(D_table)



.section .text
GLOBAL_IEEE754_ENTRY(exp10l)

{.mfi
       alloc GR_SREG = ar.pfs, 1, 4, 4, 0
       // will continue only for normal/denormal numbers
       fclass.nm.unc p12, p7 = f8, 0x1b
       // GR_ADDR0 = pointer to log2(10), C_1...C_6 followed by T_table
       addl GR_ADDR0 = @ltoff(poly_coeffs), gp ;;
}

{.mfi
       // load start address for C_1...C_6 followed by T_table
       ld8 GR_ADDR0 = [ GR_ADDR0 ]
       // X<0 ?
       fcmp.lt.s1 p6, p8 = f8, f0
       // GR_BM8 = bias-8
       mov GR_BM8 = 0xffff-8
}
{.mlx
       nop.m 0
       // GR_EMIN = (-2^14-62)*2^{8}
       movl GR_EMIN = 0xca807c00 ;;
}

{.mmb
       // FR_CONST1 = 2^{-8}
       setf.exp FR_CONST1 = GR_BM8
       // load log2(10)*2^8
       ldfe FR_LOG10 = [ GR_ADDR0 ], 16
 (p12) br.cond.spnt SPECIAL_EXP10 ;;
}

{.mmf
       setf.s FR_UF_TEST = GR_EMIN
       // load overflow threshold
       ldfe FR_OF_TEST = [ GR_ADDR0 ], 16
       // normalize x
       fma.s0 f8 = f8, f1, f0 ;;
}

{.mmi
       // load C_1
       ldfe FR_COEFF1 = [ GR_ADDR0 ], 16 ;;
       // load C_2
       ldfe FR_COEFF2 = [ GR_ADDR0 ], 16
       nop.i 0 ;;
}

{.mmf
       // GR_D_ADDR = pointer to D table
       add GR_D_ADDR = 2048-64+96+32, GR_ADDR0
       // load C_3, C_4
       ldfpd FR_COEFF3, FR_COEFF4 = [ GR_ADDR0 ], 16
       // y = x*log2(10)*2^8
       fma.s1 FR_XL10 = f8, FR_LOG10, f0 ;;
}

{.mfi
       // load C_5, C_6
       ldfpd FR_COEFF5, FR_COEFF6 = [ GR_ADDR0 ], 16
       // get int(x)
       fcvt.fx.trunc.s1 FR_XINT = f8
       nop.i 0
}
{.mfi
       nop.m 0
       // FR_LOG10 = log2(10)
       fma.s1 FR_L10 = FR_LOG10, FR_CONST1, f0
       nop.i 0 ;;
}

{.mfi
       // load log2(10)_low
       ldfe FR_L10_LOW = [ GR_ADDR0 ], 16
       // y0 = x*log2(10) = x*log2(10)_hi
       fma.s1 FR_LOG10 = f8, FR_L10, f0
       mov GR_EMIN = 0xffff-63
}
{.mfi
       mov GR_32_BIAS = 0xffff + 5
       // (K+f)*2^8 = round_to_int(y)
       fcvt.fx.s1 FR_KF0 = FR_XL10
       mov GR_4_BIAS = 0xffff + 2;;
}

{.mfi
       // load smallest normal limit
       ldfe FR_SNORM_LIMIT = [ GR_ADDR0 ], 16
       // x>overflow threshold ?
       fcmp.gt.s1 p12, p7 = f8, FR_OF_TEST
       nop.i 0 ;;
}

{.mfi
       setf.exp FR_32 = GR_32_BIAS
       // x<underflow threshold ?
  (p7) fcmp.lt.s1 p12, p7 = FR_XL10, FR_UF_TEST
       nop.i 0 ;;
}

{.mfi
       setf.exp FR_4 = GR_4_BIAS
       fcvt.xf FR_XINTF = FR_XINT
       nop.i 0
}
{.mfi
       nop.m 0
       // FR_L10 = log2(10)_h*x-RN(log2(10)_h*x)
       fms.s1 FR_L10 = f8, FR_L10, FR_LOG10
       nop.i 0 ;;
}

{.mfi
       getf.sig GR_BM8 = FR_KF0
       fcvt.xf FR_KF0 = FR_KF0
       mov GR_CONST2 = 255 ;;
}

{.mfi
       // GR_CONST2 = f
       and GR_CONST2 = GR_CONST2, GR_BM8
       // FR_L10_LOW = e = log2(10)_l*x+(log2(10)_h*x-RN(log2(10)_h*x))
       fma.s1 FR_L10_LOW = FR_L10_LOW, f8, FR_L10
       // GR_BM8 = K
       shr GR_BM8 = GR_BM8, 8 ;;
}

{.mmi
       // address of D
       shladd GR_D_ADDR = GR_CONST2, 2, GR_D_ADDR
       // K+ = bias-63
       add GR_BM8 = GR_BM8, GR_EMIN
       // address of T
       shladd GR_ADDR0 = GR_CONST2, 3, GR_ADDR0 ;;
}

{.mfb
       // load D
       ldfs FR_OF_TEST = [ GR_D_ADDR ]
       // is input an integer ?
       fcmp.eq.s1 p13, p14 = f8, FR_XINTF
 (p12) br.cond.spnt OUT_RANGE_EXP10 ;;
}

{.mmf
       // load T
       ldf8 FR_UF_TEST = [ GR_ADDR0 ]
       // FR_XL10 = 2^{K-63}
       setf.exp FR_XL10 = GR_BM8
       // r = x*log2(10)_hi-2^{-10}* [ (K+f)*2^{10} ]
       fnma.s1 FR_KF0 = FR_KF0, FR_CONST1, FR_LOG10 ;;
}

{.mfi
       nop.m 0
       // get 28.0
       fms.s1 FR_28 = FR_32, f1, FR_4
       nop.i 0
}
{.mfi
       nop.m 0
       // E = 1+C_1*e
       fma.s1 FR_L10 = FR_L10_LOW, FR_COEFF1, f1
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // P12 = C_1+C_2*r
       fma.s1 FR_COEFF2 = FR_COEFF2, FR_KF0, FR_COEFF1
       nop.i 0
}
{.mfi
       nop.m 0
       // P34 = C_3+C_4*r
       fma.s1 FR_COEFF4 = FR_COEFF4, FR_KF0, FR_COEFF3
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // P56 = C_5+C_6*r
       fma.s1 FR_COEFF5 = FR_COEFF6, FR_KF0, FR_COEFF5
       nop.i 0
}
{.mfi
       nop.m 0
       // GR_ADDR0 = r*r
       fma.s1 FR_COEFF3 = FR_KF0, FR_KF0, f0
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // if input is integer, is it positive ?
 (p13) fcmp.ge.s1 p13, p14 = f8, f0
       nop.i 0
}
{.mfi
       nop.m 0
       // r' = r*E
       fma.s1 FR_KF0 = FR_KF0, FR_L10, f0
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // D' = D+C_1*e
       fma.s1 FR_OF_TEST = FR_L10_LOW, FR_COEFF1, FR_OF_TEST
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // test if x >= smallest normal limit
       fcmp.ge.s1 p11, p0 = f8, FR_SNORM_LIMIT
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // P36 = P34+r2*P56
       fma.s1 FR_COEFF4 = FR_COEFF5, FR_COEFF3, FR_COEFF4
       nop.i 0
}
{.mfi
       nop.m 0
       // GR_D_ADDR = r'*r2
       fma.s1 FR_COEFF3 = FR_COEFF3, FR_KF0, f0
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // is input below 28.0 ?
 (p13) fcmp.lt.s1 p13, p14 = f8, FR_28
       nop.i 0
}
{.mfi
       nop.m 0
       // P' = P12*r'+D'
       fma.s1 FR_COEFF2 = FR_COEFF2, FR_KF0, FR_OF_TEST
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // P = P'+r3*P36
       fma.s1 FR_COEFF3 = FR_COEFF3, FR_COEFF4, FR_COEFF2
       nop.i 0
}
{.mfi
       nop.m 0
       // T = 2^{K-63}*T
       fma.s1 FR_UF_TEST = FR_UF_TEST, FR_XL10, f0
       nop.i 0 ;;
}

.pred.rel "mutex",p13,p14
{.mfi
       nop.m 0
 (p13) fma.s1 f8 = FR_COEFF3, FR_UF_TEST, FR_UF_TEST
       nop.i 0
}
{.mfb
       nop.m 0
       // result = T+T*P
 (p14) fma.s0 f8 = FR_COEFF3, FR_UF_TEST, FR_UF_TEST
       // return
 (p11) br.ret.sptk b0 ;;                  // return, if result normal
}

// Here if result in denormal range (and not zero)
{.mib
       nop.m 0
       mov GR_Parameter_TAG= 264
       br.cond.sptk __libm_error_region           // Branch to error handling
}
;;

SPECIAL_EXP10:

{.mfi
       nop.m 0
       // x = -Infinity ?
       fclass.m p6, p0 = f8, 0x22
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // x = +Infinity ?
       fclass.m p7, p0 = f8, 0x21
       nop.i 0 ;;
}

{.mfi
       nop.m 0
       // x = +/-Zero ?
       fclass.m p8, p0 = f8, 0x7
       nop.i 0
}
{.mfb
       nop.m 0
       // exp10(-Infinity) = 0
  (p6) mov f8 = f0
  (p6) br.ret.spnt b0 ;;
}

{.mfb
       nop.m 0
       // exp10(+Infinity) = +Infinity
       nop.f 0
  (p7) br.ret.spnt b0 ;;
}

{.mfb
       nop.m 0
       // exp10(+/-0) = 1
  (p8) mov f8 = f1
  (p8) br.ret.spnt b0 ;;
}

{.mfb
       nop.m 0
       // Remaining cases: NaNs
       fma.s0 f8 = f8, f1, f0
       br.ret.sptk b0 ;;
}


OUT_RANGE_EXP10:

// underflow: p6 = 1
// overflow: p8 = 1

.pred.rel "mutex",p6,p8
{.mmi
  (p8) mov GR_CONST1 = 0x1fffe
  (p6) mov GR_CONST1 = 1
       nop.i 0
}
;;

{.mii
       setf.exp FR_KF0 = GR_CONST1
  (p8) mov GR_Parameter_TAG = 165
  (p6) mov GR_Parameter_TAG = 264
}
;;

{.mfb
       nop.m 999
       fma.s0 f8 = FR_KF0, FR_KF0, f0             // Create overflow/underflow
       br.cond.sptk __libm_error_region           // Branch to error handling
}
;;

GLOBAL_IEEE754_END(exp10l)
libm_alias_ldouble_other (__exp10, exp10)
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_27)
compat_symbol (libm, exp10l, pow10l, GLIBC_2_2)
#endif


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
{.mfi
       add GR_Parameter_Y = -32, sp // Parameter 2 value
       nop.f 0
.save ar.pfs, GR_SAVE_PFS
       mov GR_SAVE_PFS = ar.pfs // Save ar.pfs
}

{.mfi
.fframe 64
       add sp = -64, sp // Create new stack
       nop.f 0
       mov GR_SAVE_GP = gp ;; // Save gp
}

{.mmi
       stfe [ GR_Parameter_Y ] = FR_Y, 16 // STORE Parameter 2 on stack
       add GR_Parameter_X = 16, sp // Parameter 1 address
.save b0, GR_SAVE_B0
       mov GR_SAVE_B0 = b0 ;; // Save b0
}

.body
{.mib
       stfe [ GR_Parameter_X ] = FR_X // STORE Parameter 1 on stack
       add GR_Parameter_RESULT = 0, GR_Parameter_Y // Parameter 3 address
       nop.b 0
}
{.mib
       stfe [ GR_Parameter_Y ] = FR_RESULT // STORE Parameter 3 on stack
       add GR_Parameter_Y = -16, GR_Parameter_Y
       br.call.sptk b0 = __libm_error_support# ;; // Call error handling function
}

{.mmi
       add GR_Parameter_RESULT = 48, sp
       nop.m 0
       nop.i 0 ;;
}

{.mmi
       ldfe f8 = [ GR_Parameter_RESULT ] // Get return result off stack
.restore sp
       add sp = 64, sp // Restore stack pointer
       mov b0 = GR_SAVE_B0 ;; // Restore return address
}

{.mib
       mov gp = GR_SAVE_GP // Restore gp
       mov ar.pfs = GR_SAVE_PFS // Restore ar.pfs
       br.ret.sptk b0 ;; // Return
}


LOCAL_LIBM_END(__libm_error_region)
.type __libm_error_support#, @function
.global __libm_error_support#
