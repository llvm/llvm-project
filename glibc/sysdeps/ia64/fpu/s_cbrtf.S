.file "cbrtf.s"


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
// 05/18/00 New version (modified algorithm)
// 05/20/02 Cleaned up namespace and sf0 syntax
// 01/28/03 Rescheduled some instructions for better performance
//          on Itanium 2, and reformatted
//
// API
//==============================================================
// float cbrtf(float)
//
// Overview of operation
//==============================================================
// Background
//
// Implementation
//
// Let y= frcpa(a), where a is the argument
//
// cbrt(a)= cbrt(a*y)/cbrt(y) = cbrt(1 - (1-a*y)) * (1/cbrt(y))
//
// For all values of y, the 3 possible significands of 1/cbrt(y)
// are stored in a table (T0) to 64 bits of accuracy. (There are
// 3 possible significands because the exponent of y modulo 3
// can be 0, 1, or 2.)
//
//
// * cbrt(1 - (1-a*y)) is approximated by a degree-2 polynomial
//
// 1 - (1/3)*r - (1/9)*r^2
//
// in r = 1-a*y.
//
// The table values are stored for three exponent values and are
// then multiplied by 2^(e/3) where e is the exponent of the input number.
// This computation is carried out in parallel with the polynomial
// evaluation:
//
// T= 2^(e/3) * T0





//===============
// input= x
// C= frcpa(x)
// r= 1 - C * x
//
// Special values
//==============================================================



// Registers used
//==============================================================
// p6, p7, p8, p12

       FR_R      = f6
       FR_COEFF1 = f7
       FR_COEFF2 = f9
       FR_T0     = f10
       FR_T1     = f11
       FR_T2     = f12
       FR_2M63   = f13
       FR_ARG    = f14
       FR_Y      = f15

       GR_GP     = r2
       GR_ADDR   = r2
       GR_TMP5   = r3
       GR_CONST  = r8
       GR_TMP63  = r8
       GR_SIGN   = r9
       GR_CT2    = r10
       GR_CT3    = r11
       GR_TMP4   = r14
       GR_EBIAS3 = r15
       GR_REM    = r16
       GR_SEXP   = r17
       GR_2P63   = r18
       GR_SIGNIF = r19
       GR_I1     = r20
       GR_EBIAS  = r21
       GR_EXP    = r22
       GR_IT1    = r23
       GR_E5     = r24
       GR_IT1_3  = r25
       GR_TP1    = r26
       GR_TMP    = r27
       GR_TMP2   = r28
       GR_TMP3   = r29
       GR_EXP3   = r30
       GR_ARGEXP = r31




// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(poly_coeffs)

       data8 0xaaaab19b7e1f5ef9, 0x00003ffd // ~ 1/3
       data8 0xe38e5192a5a8e56c, 0x00003ffb // ~ 1/9
LOCAL_OBJECT_END(poly_coeffs)

//   For every entry B in the frcpa table, this table contains
// the significands of cbrt(1/B), cbrt(2/B), cbrt(4/B).
// The index to this table is the same as the frcpa index.

LOCAL_OBJECT_START(T_table)

       data8 0x80155c748c374836, 0xa160019ed37fb4ae
       data8 0xcb51ddcb9e93095e, 0x8040404b0879f7f9
       data8 0xa1960b5966da4608, 0xcb95f333968ad59b
       data8 0x806b5dce4b405c10, 0xa1cc5dbe6dc2aab4
       data8 0xcbda64292d3ffd97, 0x8096b586974669b1
       data8 0xa202f97995b69c0d, 0xcc1f3184af961596
       data8 0x80bcd273d952a028, 0xa232fe6eb0c0577d
       data8 0xcc5bb1ac954d33e2, 0x80e898c52813f2f3
       data8 0xa26a2582012f6e17, 0xcca12e9831fc6402
       data8 0x81149add67c2d208, 0xa2a197e5d10465cb
       data8 0xcce70a67b64f24ad, 0x813b4e2c856b6e9a
       data8 0xa2d25a532efefbc8, 0xcd24794726477ea5
       data8 0x8167c1dde03de7aa, 0xa30a5bd6e49e4ab8
       data8 0xcd6b096a0b70ee87, 0x818ed973b811135e
       data8 0xa33b9c9b59879e24, 0xcda9177738b15a90
       data8 0x81bbc0c33e13ec98, 0xa3742fca6a3c1f21
       data8 0xcdf05f2247dffab9, 0x81e33e69fbe7504a
       data8 0xa3a5f1273887bf22, 0xce2f0f347f96f906
       data8 0x820aec524e3c23e9, 0xa3d7ef508ff11574
       data8 0xce6e0be0cd551a61, 0x823880f78e70b805
       data8 0xa4115ce30548bc15, 0xceb666b2c347d1de
       data8 0x826097a62a8e5200, 0xa443df0e53df577a
       data8 0xcef609b0cb874f00, 0x8288dfe00e9b5eaf
       data8 0xa4769fa5913c0ec3, 0xcf35fb5447e5c765
       data8 0x82b15a10c5371624, 0xa4a99f303bc7def5
       data8 0xcf763c47ee869f00, 0x82da06a527b18937
       data8 0xa4dcde37779adf4b, 0xcfb6cd3888d71785
       data8 0x8302e60b635ab394, 0xa5105d46152c938a
       data8 0xcff7aed4fbfbb447, 0x832bf8b2feec2f0e
       data8 0xa5441ce89825cb8d, 0xd038e1ce5167e3c6
       data8 0x83553f0ce00e276b, 0xa5781dad3e54d899
       data8 0xd07a66d7bfa0ebba, 0x837eb98b50f8322a
       data8 0xa5ac602406c4e68c, 0xd0bc3ea6b32d1b21
       data8 0x83a270f44c84f699, 0xa5d9601d95c2c0bc
       data8 0xd0f4f0e8f36c1bf8, 0x83cc4d7cfcfac5ca
       data8 0xa60e1e1a2de14745, 0xd1376458e34b037e
       data8 0x83f65f78a8872b4c, 0xa6431f6e3fbd9658
       data8 0xd17a2ca133f78572, 0x8420a75f2f7b53c8
       data8 0xa67864b0d432fda4, 0xd1bd4a80301c5715
       data8 0x844510461ff14209, 0xa6a6444aa0243c0b
       data8 0xd1f71682b2fa4575, 0x846fbd91b930bed2
       data8 0xa6dc094d10f25792, 0xd23ad555f773f059
       data8 0x84947e18234f3294, 0xa70a574cc02bba69
       data8 0xd2752c7039a5bf73, 0x84bf92755825045a
       data8 0xa7409e2af9549084, 0xd2b98ee008c06b59
       data8 0x84e4ac0ee112ba51, 0xa76f5c64ca2cf13b
       data8 0xd2f4735ffd700280, 0x8509ef44b86f20be
       data8 0xa79e4f0babab5dc0, 0xd32f99ed6d9ac0e1
       data8 0x85359d5d91768427, 0xa7d5579ae5164b85
       data8 0xd374f0666c75d51c, 0x855b3bd5b7384357
       data8 0xa804bd3c6fe61cc8, 0xd3b0a7d13618e4a1
       data8 0x858104f0c415f79a, 0xa8345895e5250a5a
       data8 0xd3eca2ea53bcec0c, 0x85a6f90390d29864
       data8 0xa8642a122b44ef0b, 0xd428e23874f13a17
       data8 0x85d3772fcd56a1dd, 0xa89c38ca18f6108b
       data8 0xd46f82fe293bc6d3, 0x85f9c982fcc002f3
       data8 0xa8cc81063b6e87ca, 0xd4ac57e9b7186420
       data8 0x862047e0e7ea554b, 0xa8fd00bfa409285e
       data8 0xd4e972becb04e8b8, 0x8646f2a26f7f5852
       data8 0xa92db8664d5516da, 0xd526d40a7a9b43a3
       data8 0x866dca21754096b5, 0xa95ea86b75cc2c20
       data8 0xd5647c5b73917370, 0x8694ceb8dfd17a37
       data8 0xa98fd141a4992deb, 0xd5a26c4201bd6d13
       data8 0x86bc00c49e9307e8, 0xa9c1335cae7446ba
       data8 0xd5e0a45015350a7e, 0x86dccd74fce79610
       data8 0xa9ea8686f556f645, 0xd614b539c6194104
       data8 0x870453c845acf90f, 0xaa1c52d17906bb19
       data8 0xd6537310e224283f, 0x872c089a1e90342c
       data8 0xaa4e59b046dab887, 0xd6927ab62244c917
       data8 0x8753ec4a92d16c5e, 0xaa809b9c60d1890b
       data8 0xd6d1ccc1fc4ef4b7, 0x877bff3aca19f6b4
       data8 0xaab319102f3f9b33, 0xd71169cea98fdded
       data8 0x879d88b6fe1c324c, 0xaadd5a18c1e21274
       data8 0xd746a66a5bc9f6d9, 0x87c5f346dbf98c3a
       data8 0xab1045f2ac31bdf5, 0xd786ce8f0fae5317
       data8 0x87e7c653efacef2c, 0xab3ae3ab2df7231e
       data8 0xd7bc7ff214c4e75a, 0x881089d4e73ffefc
       data8 0xab6e3f945d1e96fc, 0xd7fd35467a517ed1
       data8 0x88397e6a366f2a8a, 0xaba1d953a08fa94e
       data8 0xd83e38838648d815, 0x885bc559e5e1c081
       data8 0xabcd090db7ef4c3f, 0xd874a1db598b8951
       data8 0x887e2ee392bb7a93, 0xabf864602d7c323d
       data8 0xd8ab42205b80edaf, 0x88a7a8587e404257
       data8 0xac2ca5886ccf9b57, 0xd8ed1849d202f965
       data8 0x88ca5eda67594784, 0xac5861d4aa441f0f
       data8 0xd92432bd5a173685, 0x88f4356166bd590e
       data8 0xac8d183fe3a2fbed, 0xd9669ca45b03c23e
       data8 0x89173a0acf5ce026, 0xacb93703ff51571e
       data8 0xd99e3327cf89574e, 0x893a62a098b6a57b
       data8 0xace5830ad0c3f14b, 0xd9d602b19b100466
       data8 0x895daf637236ae2c, 0xad11fca5d78b3ff2
       data8 0xda0e0ba86c096841, 0x89883b9d1c2fa9c5
       data8 0xad4797fddf91a798, 0xda5195fcdb1c3dce
       data8 0x89abd8dd374a5d7b, 0xad747701e559ebcb
       data8 0xda8a1eb87a491f6c, 0x89cf9b1dcd197fa0
       data8 0xada184a47e9c7613, 0xdac2e230b91c3f84
       data8 0x89f382a258ea79de, 0xadcec13ab0dda8ff
       data8 0xdafbe0d0b66aea30, 0x8a178faf06648f29
       data8 0xadfc2d1a5fd21ba8, 0xdb351b04a8fafced
       data8 0x8a3bc288b3e1d18a, 0xae29c89a5053c33a
       data8 0xdb6e9139e33cdd8e, 0x8a601b74f4d1f835
       data8 0xae5794122b638df9, 0xdba843ded7151ea1
       data8 0x8a849aba14274764, 0xae858fda8137ae0a
       data8 0xdbe2336319b61fc8, 0x8aa9409f16cdbc9b
       data8 0xaeb3bc4ccc56d3d1, 0xdc1c60376789fa68
       data8 0x8ace0d6bbe2cb316, 0xaee219c374c09920
       data8 0xdc56cacda82d0cd5, 0x8af301688ab33558
       data8 0xaf10a899d3235fe7, 0xdc917398f2797814
       data8 0x8b181cdebe6f3206, 0xaf3f692c341fe8b4
       data8 0xdccc5b0d90a3e628, 0x8b3d60185fafcb7c
       data8 0xaf6e5bd7db9ae6c2, 0xdd0781a10469f0f2
       data8 0x8b62cb603bb2fad0, 0xaf9d80fb081cd91b
       data8 0xdd42e7ca0b52838f, 0x8b80d7d6bc4104de
       data8 0xafc35ce063eb3787, 0xdd729ad01c69114d
       data8 0x8ba68bf73ac74f39, 0xaff2ddcb5f28f03d
       data8 0xddae749c001fbf5e, 0x8bcc68fb9f9f7335
       data8 0xb022923b148e05c5, 0xddea8f50a51c69b1
       data8 0x8bf26f31c534fca2, 0xb0527a919adbf58b
       data8 0xde26eb69a0f0f111, 0x8c10f86e13a1a1f9
       data8 0xb078f3ab1d701c65, 0xde576480262399bc
       data8 0x8c3749916cc6abb5, 0xb0a93a6870649f31
       data8 0xde943789645933c8, 0x8c5dc4c4f7706032
       data8 0xb0d9b624d62ec856, 0xded14d58139a28af
       data8 0x8c7cac3a8c42e3e0, 0xb100a5f53fb3c8e1
       data8 0xdf025c00bbf2b5c7, 0x8ca373f1b7bf2716
       data8 0xb131821882f5540a, 0xdf3feb44d723a713
       data8 0x8cc29907fb951294, 0xb158bf8e4cb04055
       data8 0xdf715bc16c159be0, 0x8ce9ae4e9492aac8
       data8 0xb189fd69d56b238f, 0xdfaf66240e29cda8
       data8 0x8d0911dddbfdad0e, 0xb1b189958e8108e4
       data8 0xdfe139cbf6e19bdc, 0x8d3075c4f20f04ee
       data8 0xb1e32a8165b09832, 0xe01fc0fe94d9fc52
       data8 0x8d5018a9d4de77d5, 0xb20b0678fc271eec
       data8 0xe051f92ffcc0bd60, 0x8d77cc47dd143515
       data8 0xb23d0bd3f7592b6e, 0xe090feec9c9a06ac
       data8 0x8d97af6352739cb7, 0xb26538b2db8420dc
       data8 0xe0c39d0c9ff862d6, 0x8db7af523167800f
       data8 0xb28d89e339ceca14, 0xe0f668eeb99f188d
       data8 0x8ddfd80bc68c32ff, 0xb2c022ca12e55a16
       data8 0xe1362890eb663139, 0x8e00197e1e7c88fe
       data8 0xb2e8c6852c6b03f1, 0xe1695c7212aecbaa
       data8 0x8e207859f77e20e7, 0xb3118f4eda9fe40f
       data8 0xe19cbf0391bbbbe9, 0x8e40f4ce60c9f8e2
       data8 0xb33a7d6268109ebe, 0xe1d050901c531e85
       data8 0x8e69ba46cf2fde4d, 0xb36ddbc5ea70ec55
       data8 0xe2110903b4f4047a, 0x8e8a7a00bd7ae63e
       data8 0xb3971e9b39264023, 0xe2450559b4d80b6d
       data8 0x8eab57ef1cf2f529, 0xb3c0877ecc18e24a
       data8 0xe27931a231554ef3, 0x8ecc5442cffb1dad
       data8 0xb3ea16ae3a6c905f, 0xe2ad8e2ac3c5b04b
       data8 0x8eed6f2d2a4acbfe, 0xb413cc67aa0e4d2d
       data8 0xe2e21b41b9694cce, 0x8f0ea8dff24441ff
       data8 0xb43da8e9d163e1af, 0xe316d93615862714
       data8 0x8f385c95d696b817, 0xb47233773b84d425
       data8 0xe3590bd86a0d30f9, 0x8f59dc43edd930f3
       data8 0xb49c6825430fe730, 0xe38e38e38e38e38e
       data8 0x8f7b7b5f5ffad1c4, 0xb4c6c46bcdb27dcf
       data8 0xe3c397d1e6db7839, 0x8f9d3a1bea165f38
       data8 0xb4f1488c0b35d26f, 0xe3f928f5953feb9e
       data8 0x8fbf18adc34b66da, 0xb51bf4c7c51f0168
       data8 0xe42eeca17c62886c, 0x8fe117499e356095
       data8 0xb546c9616087ab9c, 0xe464e32943446305
       data8 0x90033624aa685f8d, 0xb571c69bdffd9a70
       data8 0xe49b0ce15747a8a2, 0x9025757495f36b86
       data8 0xb59cecbae56984c3, 0xe4d16a1eee94e9d4
       data8 0x903f3a5dcc091203, 0xb5bd64512bb14bb7
       data8 0xe4fa52107353f67d, 0x9061b2fceb2bdbab
       data8 0xb5e8d2a4bf5ba416, 0xe5310a471f4d2dc3
       data8 0x90844ca7211032a7, 0xb6146a9a1bc47819
       data8 0xe567f6f1c2b9c224, 0x90a7079403e6a15d
       data8 0xb6402c7749d621c0, 0xe59f18689a9e4c9a
       data8 0x90c9e3fbafd63799, 0xb66c1882fb435ea2
       data8 0xe5d66f04b8a68ecf, 0x90ece216c8a16ee4
       data8 0xb6982f048c999a56, 0xe60dfb2005c192e9
       data8 0x9110021e7b516f0a, 0xb6c47044075b4142
       data8 0xe645bd1544c7ea51, 0x912a708a39be9075
       data8 0xb6e5bd6bfd02bafd, 0xe66fb21b505b20a0
       data8 0x914dcc7b31146370, 0xb7124a2736ff8ef2
       data8 0xe6a7d32af4a7c59a, 0x91714af8cfe984d5
       data8 0xb73f026a01e94177, 0xe6e02b129c6a5ae4
       data8 0x918c00a6f3795e97, 0xb760a959f1d0a7a7
       data8 0xe70a9136a7403039, 0x91afbc299ed0295d
       data8 0xb78dae7e06868ab0, 0xe74349fb2d92a589
       data8 0x91d39add3e958db0, 0xb7badff8ad9e4e02
       data8 0xe77c3a9c86ed7d42, 0x91ee9920a8974d92
       data8 0xb7dce25b8e17ae9f, 0xe7a713f88151518a
       data8 0x9212b5fcac537c19, 0xb80a6226904045e2
       data8 0xe7e067453317ed2b, 0x9236f6b256923fcf
       data8 0xb8380f1cafd73c1c, 0xe819f37a81871bb5
       data8 0x92523ee6f90dcfc3, 0xb85a6ea8e321b4d8
       data8 0xe8454236bfaeca14, 0x9276bef031e6eb79
       data8 0xb8886b684ae7d2fa, 0xe87f32f24c3fc90e
       data8 0x929236ec237a24ad, 0xb8ab0726fa00cf5d
       data8 0xe8aacd8688892ba6, 0x92b6f70b7efe9dc3
       data8 0xb8d954a4d13b7cb1, 0xe8e523fd32f606f7
       data8 0x92d29f61eec7dc2b, 0xb8fc2d4f6cd9f04a
       data8 0xe9110b5311407927, 0x92f7a05d5b8ba92f
       data8 0xb92acc851476b1ab, 0xe94bc8bf0c108fa3
       data8 0x931379a403be5c16, 0xb94de2d841a184c2
       data8 0xe977fdc439c2ca3c, 0x9338bc44de2e3f34
       data8 0xb97cd4c36c92693c, 0xe9b3236528fc349e
       data8 0x9354c71412c69486, 0xb9a0297f172665e3
       data8 0xe9dfa70b745ac1b4, 0x937a4c273907e262
       data8 0xb9cf6f21e36c3924, 0xea1b36268d0eaa38
       data8 0x93968919f6e7975d, 0xb9f3030951267208
       data8 0xea480963fd394197, 0x93bc516fdd4680c9
       data8 0xba229d6a618e7c59, 0xea84034425f27484
       data8 0x93d8c123d9be59b2, 0xba467144459f9855
       data8 0xeab12713138dd1cc, 0x93f546c955e60076
       data8 0xba6a60c3c48f1a4b, 0xeade6db73a5e503b
       data8 0x941b70a65879079f, 0xba9a76056b67ee7a
       data8 0xeb1b0268343b121b, 0x943829f337410591
       data8 0xbabea699563ada6e, 0xeb489b0b2bdb5f14
       data8 0x9454f995765bc4d2, 0xbae2f350b262cc4b
       data8 0xeb765721e85f03d0, 0x947b86b57f5842ed
       data8 0xbb1385a23be24e57, 0xebb389645f222f62
       data8 0x94988aeb23470f86, 0xbb3814975e17c680
       data8 0xebe198f090607e0c, 0x94b5a5dc9695f42a
       data8 0xbb5cc031009bf467, 0xec0fcc9321024509
       data8 0x94d2d7a9170d8b42, 0xbb81889680024764
       data8 0xec3e247da8b82f61, 0x94f9e87dd78bf019
       data8 0xbbb2c0d8703ae95d, 0xec7c27d21321c9f7
       data8 0x95175019a503d89e, 0xbbd7cd09ba3c5463
       data8 0xecaad5278824e453, 0x9534cefa625fcb3a
       data8 0xbbfcf68c4977718f, 0xecd9a76d097d4e77
       data8 0x955265405c491a25, 0xbc223d88cfc88eee
       data8 0xed089ed5dcd99446, 0x9570130c1f9bb857
       data8 0xbc47a2284fee4ff8, 0xed37bb95add09a1c
       data8 0x9597ca4119525184, 0xbc79ac0916ed7b8a
       data8 0xed76c70508f904b6, 0x95b5af6fb5aa4d3c
       data8 0xbc9f5670d1a13030, 0xeda63bb05e7f93c6
       data8 0x95d3ac9273aafd7a, 0xbcc51f068cb95c1d
       data8 0xedd5d661daed2dc4, 0x95f1c1cafdfd3684
       data8 0xbceb05f4b30a9bc0, 0xee05974eef86b903
       data8 0x960fef3b430b8d5f, 0xbd110b6604c7d306
       data8 0xee357ead791fc670, 0x962e350575b409c5
       data8 0xbd372f8598620f19, 0xee658cb3c134a463
       data8 0x964c934c0dfc1708, 0xbd5d727edb6b3c7e
       data8 0xee95c1987f080211, 0x966b0a31c9c6bc7d
       data8 0xbd83d47d937bbc6d, 0xeec61d92d8c4314f
       data8 0x968999d9ad8d264e, 0xbdaa55addf1ae47d
       data8 0xeef6a0da64a014ac, 0x96a8426705198795
       data8 0xbdd0f63c36aa73f0, 0xef274ba72a07c811
       data8 0x96c703fd64445ee5, 0xbdf7b6556d550a15
       data8 0xef581e31a2c91260, 0x96e5dec0a7b4268d
       data8 0xbe1e9626b1ffa96b, 0xef8918b2bc43aec6
       data8 0x9704d2d4f59f79f3, 0xbe4595dd903e5371
       data8 0xefba3b63d89d7cbf, 0x9723e05ebe91b9b0
       data8 0xbe6cb5a7f14bc935, 0xefeb867ecffaa607
       data8 0x97430782be323831, 0xbe93f5b41d047cf7
       data8 0xf01cfa3df1b9c9fa, 0x97624865fc0df8bf
       data8 0xbebb5630bae4c15f, 0xf04e96dc05b43e2d
       data8 0x9781a32dcc640b2a, 0xbee2d74cd30a430c
       data8 0xf0805c944d827454, 0x97a117ffd0f48e46
       data8 0xbf0a7937cf38d981, 0xf0b24ba285c495cb
       data8 0x97c0a701f9d263c9, 0xbf323c217be2bc8c
       data8 0xf0e46442e76f6569, 0x97e0505a8637a036
       data8 0xbf5a203a09342bbb, 0xf116a6b2291d7896
       data8 0x97f57a9fb0b08c6e, 0xbf74cad1c14ebfc4
       data8 0xf1383fa9e9b5b381, 0x9815503365914a9d
       data8 0xbf9ce6a497a89f78, 0xf16ac84f90083b9b
       data8 0x98354085054fd204, 0xbfc52428bec6e72f
       data8 0xf19d7b686dcb03d7, 0x98554bbbf8a77902
       data8 0xbfed838fddab024b, 0xf1d0593311db1757
       data8 0x987571fffb7f94f6, 0xc016050c0420981a
       data8 0xf20361ee8f1c711e, 0x9895b3791dd03c23
       data8 0xc03ea8cfabddc330, 0xf23695da7de51d3f
       data8 0x98ab43a5fc65d0c8, 0xc059d3cbd65ddbce
       data8 0xf258d095e465cc35, 0x98cbb2d196bd713d
       data8 0xc082b122a3c78c9d, 0xf28c4d0bfc982b34
       data8 0x98ec3d9ec7b6f21a, 0xc0abb1499ae736c4
       data8 0xf2bff55eb3f0ea71, 0x990ce436db5e8344
       data8 0xc0d4d474c3aedaaf, 0xf2f3c9cf9884636e
       data8 0x9922b8218160967a, 0xc0f054ca33eb3437
       data8 0xf31670135ab9cc0f, 0x99438d686f75779d
       data8 0xc119b2c67e600ed0, 0xf34a8e9f0b54cdfb
       data8 0x99647eea131fa20b, 0xc1433453de2033ff
       data8 0xf37ed9fa6b8add3f, 0x997a85045a47c6d0
       data8 0xc15ef3e44e10032d, 0xf3a1cfe884ef6bb6
       data8 0x999ba5f14f8add02, 0xc188b130431d80e6
       data8 0xf3d66689dcc8e8d3, 0x99bce38b5465ecae
       data8 0xc1b2929d6067730e, 0xf40b2ab069d5c96a
       data8 0x99d31ca0887f30f9, 0xc1ce9268f31cc734
       data8 0xf42e718b90c8bc16, 0x99f48a669c74c09e
       data8 0xc1f8b0877c1b0c08, 0xf463822a0a3b4b00
       data8 0x9a16154eb445c873, 0xc222f35a87b415ba
       data8 0xf498c1076015faf8, 0x9a2c822ec198d667
       data8 0xc23f3467349e5c88, 0xf4bc5a19a33990b5
       data8 0x9a4e3e080cd91b78, 0xc269b4e40e088c01
       data8 0xf4f1e6a7d6f5425f, 0x9a70177afe52322e
       data8 0xc2945aac24daaf6e, 0xf527a232cf6be334
       data8 0x9a86b8fa94eebe10, 0xc2b0de05e43c1d66
       data8 0xf54b8ecdcda90851, 0x9aa8c42866ae2958
       data8 0xc2dbc275e1229d09, 0xf5819949c7ad87b4
       data8 0x9abf86f9e12fc45e, 0xc2f86fca9d80eeff
       data8 0xf5a5bac9213b48a9, 0x9ae1c462fc05f49d
       data8 0xc323938449a2587e, 0xf5dc1501f324a812
       data8 0x9af8a8dc936b84d0, 0xc3406b40a538ed20
       data8 0xf6006bee86b5589e, 0x9b1b19033be35730
       data8 0xc36bcee8211d15e0, 0xf63716b2fa067fa4
       data8 0x9b3da7daf04c2892, 0xc397593adf2ba366
       data8 0xf66df22fb6132b9c, 0x9b54c2e4c8a9012b
       data8 0xc3b475b6206155d5, 0xf6929fb98225deb1
       data8 0x9b77854e6c661200, 0xc3e0410243b97383
       data8 0xf6c9cd13021e3fea, 0x9b8ec2e678d56d2f
       data8 0xc3fd890709833d37, 0xf6eeb177472cedae
       data8 0x9ba60e6a5ca133b6, 0xc41ae295f7e7fa06
       data8 0xf713abf4cb0b3afb, 0x9bc919ea66a151a4
       data8 0xc44709f7bb8a4dd2, 0xf74b4d5333684ef1
       data8 0x9be0887c09ef82bb, 0xc4648fb0e0bec4c1
       data8 0xf7707f75a72f8e94, 0x9c03c8d5fffc3503
       data8 0xc490f9a94695ba14, 0xf7a874b97927af44
       data8 0x9c1b5ad21a81cbb9, 0xc4aeac0173b7d390
       data8 0xf7cddf140aedf1d8, 0x9c3ed09216e9ca02
       data8 0xc4db5941007aa853, 0xf806291bacb7f7a9
       data8 0x9c568656c0423def, 0xc4f938aec206291a
       data8 0xf82bcc43b92eafef, 0x9c7a320af242ce60
       data8 0xc52629e899dfd622, 0xf8646bf0defb759e
       data8 0x9c920bf7a8c01dc2, 0xc54436e44043b965
       data8 0xf88a487dfc3ff5f7, 0x9ca9f475d98b159c
       data8 0xc562563abf9ea07f, 0xf8b03c2b46cdc17f
       data8 0x9ccdeca60e80b5f8, 0xc58fa7d1dc42921c
       data8 0xf8e95541c152ae7a, 0x9ce5f9d4653d4902
       data8 0xc5adf561b91e110a, 0xf90f832c2700c160
       data8 0x9cfe15cb38bfdd8e, 0xc5cc5591bdbd82fa
       data8 0xf935c88e0c7f419b, 0x9d225b983f6c1f96
       data8 0xc5fa08f1ff20593c, 0xf96f5cd84fd86873
       data8 0x9d3a9cca32261ed7, 0xc618980a79ce6862
       data8 0xf995dd53ebdd9d6d, 0x9d52ecfccebe1768
       data8 0xc6373a09e34b50fa, 0xf9bc75a034436a41
       data8 0x9d77818d95b82f86, 0xc66550a6e0baaf35
       data8 0xf9f686f26d5518de, 0x9d8ff7893fa4706c
       data8 0xc6842241926342c9, 0xfa1d5b39b910a8c5
       data8 0x9da87cbef36f2a5e, 0xc6a3070b7c93bb9e
       data8 0xfa4447acc4ecbfd2, 0x9dcd6140b4a35aeb
       data8 0xc6d18260bb84081b, 0xfa7ed7e51e6fdfb4
       data8 0x9de60cd06dc6e2d4, 0xc6f0977c9416828b
       data8 0xfaa601394d49a1a0, 0x9dfec7d4cc43b76f
       data8 0xc70fc0117c641630, 0xfacd431644ce0e40
       data8 0x9e17925ec9fccc4a, 0xc72efc34d7e615be
       data8 0xfaf49d96f7a75909, 0x9e3cdf6db57dc075
       data8 0xc75dfb441594141e, 0xfb2fd3c65e562fd5
       data8 0x9e55d110b63637a8, 0xc77d68aa019bda4c
       data8 0xfb576c5762024805, 0x9e6ed27594550d2e
       data8 0xc79ce9ea478dbc4f, 0xfb7f1debc22c4040
       data8 0x9e87e3adc385d393, 0xc7bc7f1ae453219d
       data8 0xfba6e89f32d0190a, 0x9ead9b54b37a1055
       data8 0xc7ec0476e15e141a, 0xfbe2c803a0894893
       data8 0x9ec6d46a3d7de215, 0xc80bcbe16f1d540f
       data8 0xfc0ad1ff0ed9ecf0, 0x9ee01d9108be3154
       data8 0xc82ba78a5d349735, 0xfc32f57bdfbcbe7f
       data8 0x9ef976db07288d04, 0xc84b978847a06b87
       data8 0xfc5b32968f99b21c, 0x9f12e05a4759ec25
       data8 0xc86b9bf1ee817bc6, 0xfc83896bc861ab08
       data8 0x9f2c5a20f4da6668, 0xc88bb4de3667cdf4
       data8 0xfcabfa1861ed4815, 0x9f52af78ed1733ca
       data8 0xc8bc00e7fe9e23a3, 0xfce8d3cea7d3163e
       data8 0x9f6c52426a39d003, 0xc8dc4d7ff2d25232
       data8 0xfd118595143ee273, 0x9f860593d42fd7f3
       data8 0xc8fcaeebcb40eb47, 0xfd3a519943d4865a
       data8 0x9f9fc97fdb96bd51, 0xc91d25431426a663
       data8 0xfd6337f8e1ae5a4b, 0x9fb99e194f4a7037
       data8 0xc93db09d7fdb2949, 0xfd8c38d1c8e927eb
       data8 0x9fd383731ca51db9, 0xc95e5112e721582a
       data8 0xfdb5544205095a53, 0x9fed79a04fbf9423
       data8 0xc97f06bb49787677, 0xfdde8a67d2613531
       data8 0xa00780b413b24ee8, 0xc99fd1aecd6e1b06
       data8 0xfe07db619e781611, 0xa02eab2c4474b0cd
       data8 0xc9d12a3e27bb1625, 0xfe460768d80bf758
       data8 0xa048dcd51ccfd142, 0xc9f22ad82ba3d5f0
       data8 0xfe6f9bfb06cd32f6, 0xa0631fa894b11b8d
       data8 0xca134113105e67b2, 0xfe994bcd3d14fcc2
       data8 0xa07d73ba65e680af, 0xca346d07b045a876
       data8 0xfec316fecaf3f2ab, 0xa097d91e6aaf71b0
       data8 0xca55aecf0e94bb88, 0xfeecfdaf33fadb80
       data8 0xa0b24fe89e02602f, 0xca77068257be9bab
       data8 0xff16fffe2fa8fad6, 0xa0ccd82d1bd2f68b
       data8 0xca98743ae1c693a8, 0xff411e0ba9db886d
       data8 0xa0e77200215909e6, 0xcab9f8122c99a101
       data8 0xff6b57f7c33e4e9a, 0xa1021d760d584855
       data8 0xcadb9221e268c3b5, 0xff95ade2d1bd7358
       data8 0xa11cdaa36068a57d, 0xcafd4283d8043dfd
       data8 0xffc01fed60f86fb5, 0xa137a99cbd3f880b
       data8 0xcb1f09520d37c6fb, 0xffeaae3832b63956
LOCAL_OBJECT_END(T_table)







.section .text
GLOBAL_LIBM_ENTRY(cbrtf)


{.mfi
       getf.sig GR_SIGNIF = f8
       // will continue only for normal/denormal numbers
       fclass.nm.unc p12, p7 = f8, 0x1b
       // GR_GP = pointer to C_1, C_2 followed by T_table
       nop.i 0
}
{.mfi
       addl GR_GP = @ltoff(poly_coeffs), gp
       // normalize a
       fma.s1 FR_ARG = f8, f1, f0
       // GR_CT3 = bias-((2^8-1)/3) -63 = 0xffff-0x55-0x3f = 0xff6b
       mov GR_CT3 = 0xff6b ;;
}

{.mmi
       // get exponent
       getf.exp GR_ARGEXP = f8
       // load start address for C_1, C_2 followed by T_table
       ld8 GR_ADDR = [ GR_GP ]
       nop.i 0 ;;
}

{.mlx
       // check if input significand is 0
 (p7)  cmp.eq p12, p7 = GR_SIGNIF, r0
       // GR_2P63 = 2^63
       movl GR_2P63 = 0x8000000000000000 ;;
}

{.mfi
       nop.m 0
       // y = frcpa(a)
       // p7 = 1 for normal and denormal (but non-zero) arguments
 (p7)  frcpa.s0 FR_Y, p0 = f1, f8
       // p9 = 1 if denormal input
       cmp.gtu p9, p0 = GR_2P63, GR_SIGNIF
}
{.mfb
       // load C_1
       ldfe FR_COEFF1 = [ GR_ADDR ], 16
       // if argument is 0, +/-Infinity, or NaN, return
 (p12) fma.s.s0 f8 = f8, f1, f0
 (p12) br.ret.spnt b0 ;;
}

{.mmi
       // get normalized significand (for denormal inputs only)
 (p9)  getf.sig GR_SIGNIF = FR_ARG
       // load C_2
       ldfe FR_COEFF2 = [ GR_ADDR ], 16
       // GR_CT2 = bias-(2^8-1)
       mov GR_CT2 = 0xff00
}

{.mii
       // get exponent (for denormal inputs only)
 (p9)  getf.exp GR_ARGEXP = FR_ARG
       nop.i 0
       mov GR_CONST = 0x20000 ;;
}


{.mii
       // get GR_SIGN = sign
       and GR_SIGN = GR_ARGEXP, GR_CONST
       // eliminate leading 1 from GR_I1 = 1st table index
       shl GR_I1 = GR_SIGNIF, 1
       // eliminate sign from exponent
       andcm GR_EBIAS = GR_ARGEXP, GR_CONST ;;
}


{.mfi
       // subtract bias from GR_EXP = exponent
       sub GR_EXP = GR_EBIAS, GR_CT2
       // r = 1-a*y
       fnma.s1 FR_R = FR_Y, FR_ARG, f1
       // GR_IT1 = 1st table index (y_index8 bits)
       shr.u GR_IT1 = GR_I1, 56 ;;
}


{.mii
       // 1: exponent* = 5; // (2^{16}-1)/3 = 0x5555
       shladd GR_E5 = GR_EXP, 2, GR_EXP
       // GR_IT1_3 = 3*y_index
       shladd GR_IT1_3 = GR_IT1, 1, GR_IT1
       nop.i 0 ;;
}


{.mmi
       // GR_TMP5 = (5*expon)*16+5*expon = (0x55)*expon
       shladd GR_TMP5 = GR_E5, 4, GR_E5
       // adjust T_table pointer by 1st index
       shladd GR_TP1 = GR_IT1_3, 3, GR_ADDR
       nop.i 0 ;;
}


{.mmi
       // FR_T0 = T [ 0 ] [ y ]
       ldf8 FR_T0 = [ GR_TP1 ], 8
       // get 2^{-63}
       mov GR_TMP63 = 0xffff + 63
       // GR_TMP = (0x5500)*expon
       shl GR_TMP = GR_TMP5, 8 ;;
}


{.mfi
       // FR_T1 = T [ 1 ] [ y ]
       ldf8 FR_T1 = [ GR_TP1 ], 8
       // P_1 = C_1+C_2*r
       fma.s1 FR_COEFF1 = FR_COEFF2, FR_R, FR_COEFF1
       // GR_TMP2 = (0x5555)*expon
       add GR_TMP2 = GR_TMP, GR_TMP5 ;;
}


{.mmi
       // GR_TMP3 = (0x5556)*expon // 0x5556 = (2^{16}+2)/3
       add GR_TMP3 = GR_TMP2, GR_EXP ;;
       // FR_T2 = T [ 2 ] [ y ]
       ldf8 FR_T2 = [ GR_TP1 ]
       // GR_EXP3 = floor(expon/3)
       shr GR_EXP3 = GR_TMP3, 16 ;;
}


{.mmi
       setf.exp FR_2M63 = GR_TMP63
       // GR_TMP4 = 3*exponent
       shladd GR_TMP4 = GR_EXP3, 1, GR_EXP3
       // bias exponent
       add GR_EBIAS3 = GR_CT3, GR_EXP3 ;;
}


{.mmf
       // get remainder of exponent/3
       sub GR_REM = GR_EXP, GR_TMP4
       // add sign to exponent
       or GR_SEXP = GR_EBIAS3, GR_SIGN
       // P_2 = -r*P_1
       fnma.s1 FR_R = FR_COEFF1, FR_R, f0 ;;
}



{.mmi
       // FR_ARG = sign*2^{exponent/3}
       setf.exp FR_ARG = GR_SEXP
       nop.m 0
       // remainder = 0 ?
       // p7=1 if input exponent is 3*j (remainder is 0)
       cmp.eq.unc p7, p8 = r0, GR_REM ;;
}


{.mfi
       // remainder = 1 ?
       // p8=1 if input exponent is 3*j+1 (remainder is 1)
       // p12=1 if input exponent is 3*j+2 (remainder is 2)
  (p8) cmp.eq.unc p8, p12 = 1, GR_REM
       // p7=1 -> remainder = 0 -> use T = FR_T0
  (p7)  fma.s1 f8 = FR_T0, FR_R, FR_T0
       // argument is of the form 2^(3*k) ?
       // ( GR_I1 holds significand bits, without the leading 1)
       or GR_I1 = GR_I1, GR_REM ;;
}


.pred.rel "mutex", p12, p8
{.mfi
       nop.m 0
       // p8=1 -> remainder = 1 -> use FR_T1
  (p8) fma.s1 f8 = FR_T1, FR_R, FR_T1
       // argument is of the form 2^(3*k) ?
       cmp.eq p14, p7 = GR_I1, r0
}


{.mfi
       nop.m 0
       // p12=1 -> remainder=2 -> result = T+T*P_2
 (p12) fma.s1 f8 = FR_T2, FR_R, FR_T2
       nop.i 0 ;;
}


.pred.rel "mutex", p14, p7
{.mfi
       nop.m 0
       // if argument is sgn*2^{3*(expon/3)}
 (p14) fma.s.s0 f8 = FR_2M63, FR_ARG, f0
       nop.i 0
}
{.mfb
       nop.m 0
       // T* = sgn*2^{expon/3}
  (p7) fma.s.s0 f8 = f8, FR_ARG, f0
       br.ret.sptk b0 ;;
}


GLOBAL_LIBM_END(cbrtf)
libm_alias_float_other (cbrt, cbrt)
