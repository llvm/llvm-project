.file "cbrtl.s"


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
// 04/28/00 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header:.section,.global,.proc,.align
// 11/23/04 Reformatted routine and improved speed
//
// API
//==============================================================
// long double cbrtl(long double)
//
// Overview of operation
//==============================================================
// Background
//
// Implementation
//
// The result is computed as
// cbrt(x)= cbrt(1 - (1 - x*y)) * (1/cbrt(y))
// where y = frcpa(x) = (-1)^sgn_y * 2^(3*k+j) * m_y,
//       m_y in [1,2),  j in {0,1,2}
//
//  cbrt(1 - (1 - x*y)) is approximated by a degree-6 polynomial
//  in r= 1 - x*y :
// P = 1 - c_1 r - c_2 * r^2 - c_3 * r^3 - c_4 * r^4 - c_5 * r^5 - c_6 * r^6
//
//
// The values (1/cbrt(y)) are stored as two tables of constants T_hi
// (double-extended precision) and D (single precision) as follows:
// T_hi (1 + D)= 1/cbrt(y) to about 80 bits of accuracy
//
// The tables are only stored for three exponent values (i.e.
// only for 2^j * m_y, where j in {0,1,2} and m_y covers the 256
// possible mantissas for an frcpa result); the index is formed
// by the 8 leading mantissa bits of x, which is the same index used
// by the hardware to get frcpa(x).
//
// The table values are multiplied by 2^k where e is the exponent of
// the input number.  This multiplication is carried out in parallel with
// the polynomial evaluation:
// T= 2^(k) * T_hi
//
//=======================================================================

//===============
// Special values
//==============================================================

// Registers used
//==============================================================
// p6, p7, p12
           FR_R       =  f6
           FR_C1      =  f7
           FR_C2      =  f9
           FR_C3      =  f10
           FR_C4      =  f11
           FR_C5      =  f12
           FR_C6      =  f13
           FR_XNORM   =  f14
           FR_D       =  f15
           FR_SPECIAL =  f32
           FR_RCP     =  f33
           FR_R2      =  f34
           FR_P1      =  f35
           FR_P2      =  f36
           FR_P3      =  f37
           FR_P4      =  f38
           FR_P5      =  f39
           FR_R3      =  f40
           FR_T       =  f41
           FR_TF      =  f42
           FR_P       =  f43
           FR_SGNEXP  =  f44

           GR_ADDR       = r2
           GR_C_START    = r2
           GR_ARGSIG     = r3
           GR_NORMSIG    = r15
           GR_D_ADDR     = r16
           GR_D_START    = r16
           GR_INDEX2     = r17
           GR_IX2        = r17
           GR_NORMEXP    = r18
           GR_EXP5       = r19
           GR_EXP3       = r20
           GR_EXP6       = r20
           GR_EXP17      = r21
           GR_TMP1       = r21
           GR_SGNMASK    = r22
           GR_T_INDEX    = r23
           GR_IX_T       = r23
           GR_IX_D       = r24
           GR_D_INDEX    = r24
           GR_TMP2       = r25
           GR_TMP3       = r25
           GR_TMP4       = r25
           GR_EXP_RES    = r26
           GR_BIAS23     = r27
           GR_EXPBIAS    = r27
           GR_EXP_MOD_3  = r28
           GR_SIGN       = r29
           GR_EXPSIGNRES = r29
           GR_REMTMP     = r30
           GR_NORMEXPSGN = r31


// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(poly_coeffs)

       data8 0xaaaaaaaaaaaaaab1, 0x00003ffd // C_1
       data8 0xe38e38e38e38e3e0, 0x00003ffb // C_2
       data8 0x3faf9add3c0be9a6, 0x3fa511e8d2b1f749 // C_3, C_4
       data8 0x3f9ee71b2c6ebe99, 0x3f9809180fd0340c // C_5, C_6
LOCAL_OBJECT_END(poly_coeffs)


LOCAL_OBJECT_START(T_table)

       data8 0x80155c748c374836, 0x8040404b0879f7f9
       data8 0x806b5dce4b405c10, 0x8096b586974669b1
       data8 0x80bcd273d952a028, 0x80e898c52813f2f3
       data8 0x81149add67c2d208, 0x813b4e2c856b6e9a
       data8 0x8167c1dde03de7aa, 0x818ed973b811135e
       data8 0x81bbc0c33e13ec98, 0x81e33e69fbe7504a
       data8 0x820aec524e3c23e9, 0x823880f78e70b805
       data8 0x826097a62a8e5200, 0x8288dfe00e9b5eaf
       data8 0x82b15a10c5371624, 0x82da06a527b18937
       data8 0x8302e60b635ab394, 0x832bf8b2feec2f0e
       data8 0x83553f0ce00e276b, 0x837eb98b50f8322a
       data8 0x83a270f44c84f699, 0x83cc4d7cfcfac5ca
       data8 0x83f65f78a8872b4c, 0x8420a75f2f7b53c8
       data8 0x844510461ff14209, 0x846fbd91b930bed2
       data8 0x84947e18234f3294, 0x84bf92755825045a
       data8 0x84e4ac0ee112ba51, 0x8509ef44b86f20be
       data8 0x85359d5d91768427, 0x855b3bd5b7384357
       data8 0x858104f0c415f79a, 0x85a6f90390d29864
       data8 0x85d3772fcd56a1dd, 0x85f9c982fcc002f3
       data8 0x862047e0e7ea554b, 0x8646f2a26f7f5852
       data8 0x866dca21754096b5, 0x8694ceb8dfd17a37
       data8 0x86bc00c49e9307e8, 0x86dccd74fce79610
       data8 0x870453c845acf90f, 0x872c089a1e90342c
       data8 0x8753ec4a92d16c5e, 0x877bff3aca19f6b4
       data8 0x879d88b6fe1c324c, 0x87c5f346dbf98c3a
       data8 0x87e7c653efacef2c, 0x881089d4e73ffefc
       data8 0x88397e6a366f2a8a, 0x885bc559e5e1c081
       data8 0x887e2ee392bb7a93, 0x88a7a8587e404257
       data8 0x88ca5eda67594784, 0x88f4356166bd590e
       data8 0x89173a0acf5ce026, 0x893a62a098b6a57b
       data8 0x895daf637236ae2c, 0x89883b9d1c2fa9c5
       data8 0x89abd8dd374a5d7b, 0x89cf9b1dcd197fa0
       data8 0x89f382a258ea79de, 0x8a178faf06648f29
       data8 0x8a3bc288b3e1d18a, 0x8a601b74f4d1f835
       data8 0x8a849aba14274764, 0x8aa9409f16cdbc9b
       data8 0x8ace0d6bbe2cb316, 0x8af301688ab33558
       data8 0x8b181cdebe6f3206, 0x8b3d60185fafcb7c
       data8 0x8b62cb603bb2fad0, 0x8b80d7d6bc4104de
       data8 0x8ba68bf73ac74f39, 0x8bcc68fb9f9f7335
       data8 0x8bf26f31c534fca2, 0x8c10f86e13a1a1f9
       data8 0x8c3749916cc6abb5, 0x8c5dc4c4f7706032
       data8 0x8c7cac3a8c42e3e0, 0x8ca373f1b7bf2716
       data8 0x8cc29907fb951294, 0x8ce9ae4e9492aac8
       data8 0x8d0911dddbfdad0e, 0x8d3075c4f20f04ee
       data8 0x8d5018a9d4de77d5, 0x8d77cc47dd143515
       data8 0x8d97af6352739cb7, 0x8db7af523167800f
       data8 0x8ddfd80bc68c32ff, 0x8e00197e1e7c88fe
       data8 0x8e207859f77e20e7, 0x8e40f4ce60c9f8e2
       data8 0x8e69ba46cf2fde4d, 0x8e8a7a00bd7ae63e
       data8 0x8eab57ef1cf2f529, 0x8ecc5442cffb1dad
       data8 0x8eed6f2d2a4acbfe, 0x8f0ea8dff24441ff
       data8 0x8f385c95d696b817, 0x8f59dc43edd930f3
       data8 0x8f7b7b5f5ffad1c4, 0x8f9d3a1bea165f38
       data8 0x8fbf18adc34b66da, 0x8fe117499e356095
       data8 0x90033624aa685f8d, 0x9025757495f36b86
       data8 0x903f3a5dcc091203, 0x9061b2fceb2bdbab
       data8 0x90844ca7211032a7, 0x90a7079403e6a15d
       data8 0x90c9e3fbafd63799, 0x90ece216c8a16ee4
       data8 0x9110021e7b516f0a, 0x912a708a39be9075
       data8 0x914dcc7b31146370, 0x91714af8cfe984d5
       data8 0x918c00a6f3795e97, 0x91afbc299ed0295d
       data8 0x91d39add3e958db0, 0x91ee9920a8974d92
       data8 0x9212b5fcac537c19, 0x9236f6b256923fcf
       data8 0x92523ee6f90dcfc3, 0x9276bef031e6eb79
       data8 0x929236ec237a24ad, 0x92b6f70b7efe9dc3
       data8 0x92d29f61eec7dc2b, 0x92f7a05d5b8ba92f
       data8 0x931379a403be5c16, 0x9338bc44de2e3f34
       data8 0x9354c71412c69486, 0x937a4c273907e262
       data8 0x93968919f6e7975d, 0x93bc516fdd4680c9
       data8 0x93d8c123d9be59b2, 0x93f546c955e60076
       data8 0x941b70a65879079f, 0x943829f337410591
       data8 0x9454f995765bc4d2, 0x947b86b57f5842ed
       data8 0x94988aeb23470f86, 0x94b5a5dc9695f42a
       data8 0x94d2d7a9170d8b42, 0x94f9e87dd78bf019
       data8 0x95175019a503d89e, 0x9534cefa625fcb3a
       data8 0x955265405c491a25, 0x9570130c1f9bb857
       data8 0x9597ca4119525184, 0x95b5af6fb5aa4d3c
       data8 0x95d3ac9273aafd7a, 0x95f1c1cafdfd3684
       data8 0x960fef3b430b8d5f, 0x962e350575b409c5
       data8 0x964c934c0dfc1708, 0x966b0a31c9c6bc7d
       data8 0x968999d9ad8d264e, 0x96a8426705198795
       data8 0x96c703fd64445ee5, 0x96e5dec0a7b4268d
       data8 0x9704d2d4f59f79f3, 0x9723e05ebe91b9b0
       data8 0x97430782be323831, 0x97624865fc0df8bf
       data8 0x9781a32dcc640b2a, 0x97a117ffd0f48e46
       data8 0x97c0a701f9d263c9, 0x97e0505a8637a036
       data8 0x97f57a9fb0b08c6e, 0x9815503365914a9d
       data8 0x98354085054fd204, 0x98554bbbf8a77902
       data8 0x987571fffb7f94f6, 0x9895b3791dd03c23
       data8 0x98ab43a5fc65d0c8, 0x98cbb2d196bd713d
       data8 0x98ec3d9ec7b6f21a, 0x990ce436db5e8344
       data8 0x9922b8218160967a, 0x99438d686f75779d
       data8 0x99647eea131fa20b, 0x997a85045a47c6d0
       data8 0x999ba5f14f8add02, 0x99bce38b5465ecae
       data8 0x99d31ca0887f30f9, 0x99f48a669c74c09e
       data8 0x9a16154eb445c873, 0x9a2c822ec198d667
       data8 0x9a4e3e080cd91b78, 0x9a70177afe52322e
       data8 0x9a86b8fa94eebe10, 0x9aa8c42866ae2958
       data8 0x9abf86f9e12fc45e, 0x9ae1c462fc05f49d
       data8 0x9af8a8dc936b84d0, 0x9b1b19033be35730
       data8 0x9b3da7daf04c2892, 0x9b54c2e4c8a9012b
       data8 0x9b77854e6c661200, 0x9b8ec2e678d56d2f
       data8 0x9ba60e6a5ca133b6, 0x9bc919ea66a151a4
       data8 0x9be0887c09ef82bb, 0x9c03c8d5fffc3503
       data8 0x9c1b5ad21a81cbb9, 0x9c3ed09216e9ca02
       data8 0x9c568656c0423def, 0x9c7a320af242ce60
       data8 0x9c920bf7a8c01dc2, 0x9ca9f475d98b159c
       data8 0x9ccdeca60e80b5f8, 0x9ce5f9d4653d4902
       data8 0x9cfe15cb38bfdd8e, 0x9d225b983f6c1f96
       data8 0x9d3a9cca32261ed7, 0x9d52ecfccebe1768
       data8 0x9d77818d95b82f86, 0x9d8ff7893fa4706c
       data8 0x9da87cbef36f2a5e, 0x9dcd6140b4a35aeb
       data8 0x9de60cd06dc6e2d4, 0x9dfec7d4cc43b76f
       data8 0x9e17925ec9fccc4a, 0x9e3cdf6db57dc075
       data8 0x9e55d110b63637a8, 0x9e6ed27594550d2e
       data8 0x9e87e3adc385d393, 0x9ead9b54b37a1055
       data8 0x9ec6d46a3d7de215, 0x9ee01d9108be3154
       data8 0x9ef976db07288d04, 0x9f12e05a4759ec25
       data8 0x9f2c5a20f4da6668, 0x9f52af78ed1733ca
       data8 0x9f6c52426a39d003, 0x9f860593d42fd7f3
       data8 0x9f9fc97fdb96bd51, 0x9fb99e194f4a7037
       data8 0x9fd383731ca51db9, 0x9fed79a04fbf9423
       data8 0xa00780b413b24ee8, 0xa02eab2c4474b0cd
       data8 0xa048dcd51ccfd142, 0xa0631fa894b11b8d
       data8 0xa07d73ba65e680af, 0xa097d91e6aaf71b0
       data8 0xa0b24fe89e02602f, 0xa0ccd82d1bd2f68b
       data8 0xa0e77200215909e6, 0xa1021d760d584855
       data8 0xa11cdaa36068a57d, 0xa137a99cbd3f880b
       data8 0xa160019ed37fb4ae, 0xa1960b5966da4608
       data8 0xa1cc5dbe6dc2aab4, 0xa202f97995b69c0d
       data8 0xa232fe6eb0c0577d, 0xa26a2582012f6e17
       data8 0xa2a197e5d10465cb, 0xa2d25a532efefbc8
       data8 0xa30a5bd6e49e4ab8, 0xa33b9c9b59879e24
       data8 0xa3742fca6a3c1f21, 0xa3a5f1273887bf22
       data8 0xa3d7ef508ff11574, 0xa4115ce30548bc15
       data8 0xa443df0e53df577a, 0xa4769fa5913c0ec3
       data8 0xa4a99f303bc7def5, 0xa4dcde37779adf4b
       data8 0xa5105d46152c938a, 0xa5441ce89825cb8d
       data8 0xa5781dad3e54d899, 0xa5ac602406c4e68c
       data8 0xa5d9601d95c2c0bc, 0xa60e1e1a2de14745
       data8 0xa6431f6e3fbd9658, 0xa67864b0d432fda4
       data8 0xa6a6444aa0243c0b, 0xa6dc094d10f25792
       data8 0xa70a574cc02bba69, 0xa7409e2af9549084
       data8 0xa76f5c64ca2cf13b, 0xa79e4f0babab5dc0
       data8 0xa7d5579ae5164b85, 0xa804bd3c6fe61cc8
       data8 0xa8345895e5250a5a, 0xa8642a122b44ef0b
       data8 0xa89c38ca18f6108b, 0xa8cc81063b6e87ca
       data8 0xa8fd00bfa409285e, 0xa92db8664d5516da
       data8 0xa95ea86b75cc2c20, 0xa98fd141a4992deb
       data8 0xa9c1335cae7446ba, 0xa9ea8686f556f645
       data8 0xaa1c52d17906bb19, 0xaa4e59b046dab887
       data8 0xaa809b9c60d1890b, 0xaab319102f3f9b33
       data8 0xaadd5a18c1e21274, 0xab1045f2ac31bdf5
       data8 0xab3ae3ab2df7231e, 0xab6e3f945d1e96fc
       data8 0xaba1d953a08fa94e, 0xabcd090db7ef4c3f
       data8 0xabf864602d7c323d, 0xac2ca5886ccf9b57
       data8 0xac5861d4aa441f0f, 0xac8d183fe3a2fbed
       data8 0xacb93703ff51571e, 0xace5830ad0c3f14b
       data8 0xad11fca5d78b3ff2, 0xad4797fddf91a798
       data8 0xad747701e559ebcb, 0xada184a47e9c7613
       data8 0xadcec13ab0dda8ff, 0xadfc2d1a5fd21ba8
       data8 0xae29c89a5053c33a, 0xae5794122b638df9
       data8 0xae858fda8137ae0a, 0xaeb3bc4ccc56d3d1
       data8 0xaee219c374c09920, 0xaf10a899d3235fe7
       data8 0xaf3f692c341fe8b4, 0xaf6e5bd7db9ae6c2
       data8 0xaf9d80fb081cd91b, 0xafc35ce063eb3787
       data8 0xaff2ddcb5f28f03d, 0xb022923b148e05c5
       data8 0xb0527a919adbf58b, 0xb078f3ab1d701c65
       data8 0xb0a93a6870649f31, 0xb0d9b624d62ec856
       data8 0xb100a5f53fb3c8e1, 0xb131821882f5540a
       data8 0xb158bf8e4cb04055, 0xb189fd69d56b238f
       data8 0xb1b189958e8108e4, 0xb1e32a8165b09832
       data8 0xb20b0678fc271eec, 0xb23d0bd3f7592b6e
       data8 0xb26538b2db8420dc, 0xb28d89e339ceca14
       data8 0xb2c022ca12e55a16, 0xb2e8c6852c6b03f1
       data8 0xb3118f4eda9fe40f, 0xb33a7d6268109ebe
       data8 0xb36ddbc5ea70ec55, 0xb3971e9b39264023
       data8 0xb3c0877ecc18e24a, 0xb3ea16ae3a6c905f
       data8 0xb413cc67aa0e4d2d, 0xb43da8e9d163e1af
       data8 0xb47233773b84d425, 0xb49c6825430fe730
       data8 0xb4c6c46bcdb27dcf, 0xb4f1488c0b35d26f
       data8 0xb51bf4c7c51f0168, 0xb546c9616087ab9c
       data8 0xb571c69bdffd9a70, 0xb59cecbae56984c3
       data8 0xb5bd64512bb14bb7, 0xb5e8d2a4bf5ba416
       data8 0xb6146a9a1bc47819, 0xb6402c7749d621c0
       data8 0xb66c1882fb435ea2, 0xb6982f048c999a56
       data8 0xb6c47044075b4142, 0xb6e5bd6bfd02bafd
       data8 0xb7124a2736ff8ef2, 0xb73f026a01e94177
       data8 0xb760a959f1d0a7a7, 0xb78dae7e06868ab0
       data8 0xb7badff8ad9e4e02, 0xb7dce25b8e17ae9f
       data8 0xb80a6226904045e2, 0xb8380f1cafd73c1c
       data8 0xb85a6ea8e321b4d8, 0xb8886b684ae7d2fa
       data8 0xb8ab0726fa00cf5d, 0xb8d954a4d13b7cb1
       data8 0xb8fc2d4f6cd9f04a, 0xb92acc851476b1ab
       data8 0xb94de2d841a184c2, 0xb97cd4c36c92693c
       data8 0xb9a0297f172665e3, 0xb9cf6f21e36c3924
       data8 0xb9f3030951267208, 0xba229d6a618e7c59
       data8 0xba467144459f9855, 0xba6a60c3c48f1a4b
       data8 0xba9a76056b67ee7a, 0xbabea699563ada6e
       data8 0xbae2f350b262cc4b, 0xbb1385a23be24e57
       data8 0xbb3814975e17c680, 0xbb5cc031009bf467
       data8 0xbb81889680024764, 0xbbb2c0d8703ae95d
       data8 0xbbd7cd09ba3c5463, 0xbbfcf68c4977718f
       data8 0xbc223d88cfc88eee, 0xbc47a2284fee4ff8
       data8 0xbc79ac0916ed7b8a, 0xbc9f5670d1a13030
       data8 0xbcc51f068cb95c1d, 0xbceb05f4b30a9bc0
       data8 0xbd110b6604c7d306, 0xbd372f8598620f19
       data8 0xbd5d727edb6b3c7e, 0xbd83d47d937bbc6d
       data8 0xbdaa55addf1ae47d, 0xbdd0f63c36aa73f0
       data8 0xbdf7b6556d550a15, 0xbe1e9626b1ffa96b
       data8 0xbe4595dd903e5371, 0xbe6cb5a7f14bc935
       data8 0xbe93f5b41d047cf7, 0xbebb5630bae4c15f
       data8 0xbee2d74cd30a430c, 0xbf0a7937cf38d981
       data8 0xbf323c217be2bc8c, 0xbf5a203a09342bbb
       data8 0xbf74cad1c14ebfc4, 0xbf9ce6a497a89f78
       data8 0xbfc52428bec6e72f, 0xbfed838fddab024b
       data8 0xc016050c0420981a, 0xc03ea8cfabddc330
       data8 0xc059d3cbd65ddbce, 0xc082b122a3c78c9d
       data8 0xc0abb1499ae736c4, 0xc0d4d474c3aedaaf
       data8 0xc0f054ca33eb3437, 0xc119b2c67e600ed0
       data8 0xc1433453de2033ff, 0xc15ef3e44e10032d
       data8 0xc188b130431d80e6, 0xc1b2929d6067730e
       data8 0xc1ce9268f31cc734, 0xc1f8b0877c1b0c08
       data8 0xc222f35a87b415ba, 0xc23f3467349e5c88
       data8 0xc269b4e40e088c01, 0xc2945aac24daaf6e
       data8 0xc2b0de05e43c1d66, 0xc2dbc275e1229d09
       data8 0xc2f86fca9d80eeff, 0xc323938449a2587e
       data8 0xc3406b40a538ed20, 0xc36bcee8211d15e0
       data8 0xc397593adf2ba366, 0xc3b475b6206155d5
       data8 0xc3e0410243b97383, 0xc3fd890709833d37
       data8 0xc41ae295f7e7fa06, 0xc44709f7bb8a4dd2
       data8 0xc4648fb0e0bec4c1, 0xc490f9a94695ba14
       data8 0xc4aeac0173b7d390, 0xc4db5941007aa853
       data8 0xc4f938aec206291a, 0xc52629e899dfd622
       data8 0xc54436e44043b965, 0xc562563abf9ea07f
       data8 0xc58fa7d1dc42921c, 0xc5adf561b91e110a
       data8 0xc5cc5591bdbd82fa, 0xc5fa08f1ff20593c
       data8 0xc618980a79ce6862, 0xc6373a09e34b50fa
       data8 0xc66550a6e0baaf35, 0xc6842241926342c9
       data8 0xc6a3070b7c93bb9e, 0xc6d18260bb84081b
       data8 0xc6f0977c9416828b, 0xc70fc0117c641630
       data8 0xc72efc34d7e615be, 0xc75dfb441594141e
       data8 0xc77d68aa019bda4c, 0xc79ce9ea478dbc4f
       data8 0xc7bc7f1ae453219d, 0xc7ec0476e15e141a
       data8 0xc80bcbe16f1d540f, 0xc82ba78a5d349735
       data8 0xc84b978847a06b87, 0xc86b9bf1ee817bc6
       data8 0xc88bb4de3667cdf4, 0xc8bc00e7fe9e23a3
       data8 0xc8dc4d7ff2d25232, 0xc8fcaeebcb40eb47
       data8 0xc91d25431426a663, 0xc93db09d7fdb2949
       data8 0xc95e5112e721582a, 0xc97f06bb49787677
       data8 0xc99fd1aecd6e1b06, 0xc9d12a3e27bb1625
       data8 0xc9f22ad82ba3d5f0, 0xca134113105e67b2
       data8 0xca346d07b045a876, 0xca55aecf0e94bb88
       data8 0xca77068257be9bab, 0xca98743ae1c693a8
       data8 0xcab9f8122c99a101, 0xcadb9221e268c3b5
       data8 0xcafd4283d8043dfd, 0xcb1f09520d37c6fb
       data8 0xcb51ddcb9e93095e, 0xcb95f333968ad59b
       data8 0xcbda64292d3ffd97, 0xcc1f3184af961596
       data8 0xcc5bb1ac954d33e2, 0xcca12e9831fc6402
       data8 0xcce70a67b64f24ad, 0xcd24794726477ea5
       data8 0xcd6b096a0b70ee87, 0xcda9177738b15a90
       data8 0xcdf05f2247dffab9, 0xce2f0f347f96f906
       data8 0xce6e0be0cd551a61, 0xceb666b2c347d1de
       data8 0xcef609b0cb874f00, 0xcf35fb5447e5c765
       data8 0xcf763c47ee869f00, 0xcfb6cd3888d71785
       data8 0xcff7aed4fbfbb447, 0xd038e1ce5167e3c6
       data8 0xd07a66d7bfa0ebba, 0xd0bc3ea6b32d1b21
       data8 0xd0f4f0e8f36c1bf8, 0xd1376458e34b037e
       data8 0xd17a2ca133f78572, 0xd1bd4a80301c5715
       data8 0xd1f71682b2fa4575, 0xd23ad555f773f059
       data8 0xd2752c7039a5bf73, 0xd2b98ee008c06b59
       data8 0xd2f4735ffd700280, 0xd32f99ed6d9ac0e1
       data8 0xd374f0666c75d51c, 0xd3b0a7d13618e4a1
       data8 0xd3eca2ea53bcec0c, 0xd428e23874f13a17
       data8 0xd46f82fe293bc6d3, 0xd4ac57e9b7186420
       data8 0xd4e972becb04e8b8, 0xd526d40a7a9b43a3
       data8 0xd5647c5b73917370, 0xd5a26c4201bd6d13
       data8 0xd5e0a45015350a7e, 0xd614b539c6194104
       data8 0xd6537310e224283f, 0xd6927ab62244c917
       data8 0xd6d1ccc1fc4ef4b7, 0xd71169cea98fdded
       data8 0xd746a66a5bc9f6d9, 0xd786ce8f0fae5317
       data8 0xd7bc7ff214c4e75a, 0xd7fd35467a517ed1
       data8 0xd83e38838648d815, 0xd874a1db598b8951
       data8 0xd8ab42205b80edaf, 0xd8ed1849d202f965
       data8 0xd92432bd5a173685, 0xd9669ca45b03c23e
       data8 0xd99e3327cf89574e, 0xd9d602b19b100466
       data8 0xda0e0ba86c096841, 0xda5195fcdb1c3dce
       data8 0xda8a1eb87a491f6c, 0xdac2e230b91c3f84
       data8 0xdafbe0d0b66aea30, 0xdb351b04a8fafced
       data8 0xdb6e9139e33cdd8e, 0xdba843ded7151ea1
       data8 0xdbe2336319b61fc8, 0xdc1c60376789fa68
       data8 0xdc56cacda82d0cd5, 0xdc917398f2797814
       data8 0xdccc5b0d90a3e628, 0xdd0781a10469f0f2
       data8 0xdd42e7ca0b52838f, 0xdd729ad01c69114d
       data8 0xddae749c001fbf5e, 0xddea8f50a51c69b1
       data8 0xde26eb69a0f0f111, 0xde576480262399bc
       data8 0xde943789645933c8, 0xded14d58139a28af
       data8 0xdf025c00bbf2b5c7, 0xdf3feb44d723a713
       data8 0xdf715bc16c159be0, 0xdfaf66240e29cda8
       data8 0xdfe139cbf6e19bdc, 0xe01fc0fe94d9fc52
       data8 0xe051f92ffcc0bd60, 0xe090feec9c9a06ac
       data8 0xe0c39d0c9ff862d6, 0xe0f668eeb99f188d
       data8 0xe1362890eb663139, 0xe1695c7212aecbaa
       data8 0xe19cbf0391bbbbe9, 0xe1d050901c531e85
       data8 0xe2110903b4f4047a, 0xe2450559b4d80b6d
       data8 0xe27931a231554ef3, 0xe2ad8e2ac3c5b04b
       data8 0xe2e21b41b9694cce, 0xe316d93615862714
       data8 0xe3590bd86a0d30f9, 0xe38e38e38e38e38e
       data8 0xe3c397d1e6db7839, 0xe3f928f5953feb9e
       data8 0xe42eeca17c62886c, 0xe464e32943446305
       data8 0xe49b0ce15747a8a2, 0xe4d16a1eee94e9d4
       data8 0xe4fa52107353f67d, 0xe5310a471f4d2dc3
       data8 0xe567f6f1c2b9c224, 0xe59f18689a9e4c9a
       data8 0xe5d66f04b8a68ecf, 0xe60dfb2005c192e9
       data8 0xe645bd1544c7ea51, 0xe66fb21b505b20a0
       data8 0xe6a7d32af4a7c59a, 0xe6e02b129c6a5ae4
       data8 0xe70a9136a7403039, 0xe74349fb2d92a589
       data8 0xe77c3a9c86ed7d42, 0xe7a713f88151518a
       data8 0xe7e067453317ed2b, 0xe819f37a81871bb5
       data8 0xe8454236bfaeca14, 0xe87f32f24c3fc90e
       data8 0xe8aacd8688892ba6, 0xe8e523fd32f606f7
       data8 0xe9110b5311407927, 0xe94bc8bf0c108fa3
       data8 0xe977fdc439c2ca3c, 0xe9b3236528fc349e
       data8 0xe9dfa70b745ac1b4, 0xea1b36268d0eaa38
       data8 0xea480963fd394197, 0xea84034425f27484
       data8 0xeab12713138dd1cc, 0xeade6db73a5e503b
       data8 0xeb1b0268343b121b, 0xeb489b0b2bdb5f14
       data8 0xeb765721e85f03d0, 0xebb389645f222f62
       data8 0xebe198f090607e0c, 0xec0fcc9321024509
       data8 0xec3e247da8b82f61, 0xec7c27d21321c9f7
       data8 0xecaad5278824e453, 0xecd9a76d097d4e77
       data8 0xed089ed5dcd99446, 0xed37bb95add09a1c
       data8 0xed76c70508f904b6, 0xeda63bb05e7f93c6
       data8 0xedd5d661daed2dc4, 0xee05974eef86b903
       data8 0xee357ead791fc670, 0xee658cb3c134a463
       data8 0xee95c1987f080211, 0xeec61d92d8c4314f
       data8 0xeef6a0da64a014ac, 0xef274ba72a07c811
       data8 0xef581e31a2c91260, 0xef8918b2bc43aec6
       data8 0xefba3b63d89d7cbf, 0xefeb867ecffaa607
       data8 0xf01cfa3df1b9c9fa, 0xf04e96dc05b43e2d
       data8 0xf0805c944d827454, 0xf0b24ba285c495cb
       data8 0xf0e46442e76f6569, 0xf116a6b2291d7896
       data8 0xf1383fa9e9b5b381, 0xf16ac84f90083b9b
       data8 0xf19d7b686dcb03d7, 0xf1d0593311db1757
       data8 0xf20361ee8f1c711e, 0xf23695da7de51d3f
       data8 0xf258d095e465cc35, 0xf28c4d0bfc982b34
       data8 0xf2bff55eb3f0ea71, 0xf2f3c9cf9884636e
       data8 0xf31670135ab9cc0f, 0xf34a8e9f0b54cdfb
       data8 0xf37ed9fa6b8add3f, 0xf3a1cfe884ef6bb6
       data8 0xf3d66689dcc8e8d3, 0xf40b2ab069d5c96a
       data8 0xf42e718b90c8bc16, 0xf463822a0a3b4b00
       data8 0xf498c1076015faf8, 0xf4bc5a19a33990b5
       data8 0xf4f1e6a7d6f5425f, 0xf527a232cf6be334
       data8 0xf54b8ecdcda90851, 0xf5819949c7ad87b4
       data8 0xf5a5bac9213b48a9, 0xf5dc1501f324a812
       data8 0xf6006bee86b5589e, 0xf63716b2fa067fa4
       data8 0xf66df22fb6132b9c, 0xf6929fb98225deb1
       data8 0xf6c9cd13021e3fea, 0xf6eeb177472cedae
       data8 0xf713abf4cb0b3afb, 0xf74b4d5333684ef1
       data8 0xf7707f75a72f8e94, 0xf7a874b97927af44
       data8 0xf7cddf140aedf1d8, 0xf806291bacb7f7a9
       data8 0xf82bcc43b92eafef, 0xf8646bf0defb759e
       data8 0xf88a487dfc3ff5f7, 0xf8b03c2b46cdc17f
       data8 0xf8e95541c152ae7a, 0xf90f832c2700c160
       data8 0xf935c88e0c7f419b, 0xf96f5cd84fd86873
       data8 0xf995dd53ebdd9d6d, 0xf9bc75a034436a41
       data8 0xf9f686f26d5518de, 0xfa1d5b39b910a8c5
       data8 0xfa4447acc4ecbfd2, 0xfa7ed7e51e6fdfb4
       data8 0xfaa601394d49a1a0, 0xfacd431644ce0e40
       data8 0xfaf49d96f7a75909, 0xfb2fd3c65e562fd5
       data8 0xfb576c5762024805, 0xfb7f1debc22c4040
       data8 0xfba6e89f32d0190a, 0xfbe2c803a0894893
       data8 0xfc0ad1ff0ed9ecf0, 0xfc32f57bdfbcbe7f
       data8 0xfc5b32968f99b21c, 0xfc83896bc861ab08
       data8 0xfcabfa1861ed4815, 0xfce8d3cea7d3163e
       data8 0xfd118595143ee273, 0xfd3a519943d4865a
       data8 0xfd6337f8e1ae5a4b, 0xfd8c38d1c8e927eb
       data8 0xfdb5544205095a53, 0xfdde8a67d2613531
       data8 0xfe07db619e781611, 0xfe460768d80bf758
       data8 0xfe6f9bfb06cd32f6, 0xfe994bcd3d14fcc2
       data8 0xfec316fecaf3f2ab, 0xfeecfdaf33fadb80
       data8 0xff16fffe2fa8fad6, 0xff411e0ba9db886d
       data8 0xff6b57f7c33e4e9a, 0xff95ade2d1bd7358
       data8 0xffc01fed60f86fb5, 0xffeaae3832b63956
LOCAL_OBJECT_END(T_table)


LOCAL_OBJECT_START(D_table)

       data4 0x1e50f488, 0x1ebdc559, 0x1e649ec1, 0x9eed9b2c
       data4 0x9e511c44, 0x9ec6d551, 0x9eefe248, 0x9e313854
       data4 0x9f54ff18, 0x9d231411, 0x1ee5d63c, 0x9edf6b95
       data4 0x9f332aaa, 0x1dc92a84, 0x1f73fb7b, 0x1e32f100
       data4 0x9ea636f5, 0x9f6c3353, 0x9f405552, 0x1f33fd97
       data4 0x1e975291, 0x9e59a11e, 0x1e47b0ba, 0x9d8ad33e
       data4 0x1ea51bf6, 0x1f25d782, 0x9ecf534d, 0x1f55436f
       data4 0x1d0975e4, 0x9f0633a1, 0x1f3e840a, 0x1f523a4c
       data4 0x9f53cbbc, 0x9c8b5661, 0x9f6bc8eb, 0x1f4f6c7b
       data4 0x9ed9b376, 0x9f5b30b6, 0x1f64fa5e, 0x1cbcc3e0
       data4 0x1f343548, 0x1f62a6a2, 0x9f336abb, 0x9f1d15af
       data4 0x1f476c83, 0x1ea86421, 0x1f33b2cf, 0x9e8f1348
       data4 0x1f6fa829, 0x9f30ee3a, 0x9ebd6146, 0x1f2db598
       data4 0x1ef9600d, 0x1f5b1427, 0x9edd741b, 0x1f51ef4e
       data4 0x9f1aa57d, 0x9ee9b5e0, 0x9f17ecd7, 0x1ead71ff
       data4 0x1f6c910e, 0x9e1837df, 0x9f0f17d9, 0x9e8350dd
       data4 0x9d292f1b, 0x9e33b3ab, 0x9d6f0fe8, 0x9ed8c7cc
       data4 0x9ec598c8, 0x9d56758c, 0x1e090c1e, 0x9ed4b941
       data4 0x9f1fc4cf, 0x1f63513a, 0x9edd0abc, 0x1e3924dd
       data4 0x1f60d56f, 0x1ea84424, 0x9e88f4fb, 0x1f205c09
       data4 0x1ec9ae4e, 0x1d2d5738, 0x9f2c9f6d, 0x1e0765c2
       data4 0x1e8bbdd7, 0x9f16d9f1, 0x9ea62627, 0x1f13904c
       data4 0x1e566ab8, 0x9dca3d1a, 0x9e91f2a1, 0x9f14641c
       data4 0x9f278946, 0x1f490c1e, 0x1f575eb6, 0x1f50b3fd
       data4 0x9da32efb, 0x1ea95e59, 0x9e41e058, 0x9eada15f
       data4 0x9e4fe66c, 0x1f3abc98, 0x1f1b8d1e, 0x9ece97e4
       data4 0x1d188aed, 0x9e89b6ee, 0x1f287478, 0x9e8a161a
       data4 0x1e4749f7, 0x9e68084a, 0x1e867f33, 0x9f462b63
       data4 0x1db30792, 0x1f59a767, 0x9d1da4ae, 0x9f472a33
       data4 0x1d1e91cd, 0x9f414824, 0x9f473d4f, 0x1f4b5783
       data4 0x9f5b04b8, 0x9f5c205b, 0x1f309617, 0x9f0d6852
       data4 0x9d96a609, 0x9f0965c2, 0x9e23f467, 0x9f089884
       data4 0x9ec71458, 0x9ed6e955, 0x1e5e8691, 0x1f5b2bbc
       data4 0x9f128268, 0x1ed40f5b, 0x1dc430ce, 0x1f345986
       data4 0x1d778f72, 0x1e9b11d6, 0x9f5a40be, 0x9e07f61a
       data4 0x9ed641a7, 0x9f334787, 0x1e952fd0, 0x1edeb5e2
       data4 0x9e9f3eb1, 0x9e379fd9, 0x1f13102a, 0x9e5e80e1
       data4 0x1c757944, 0x1dae2260, 0x1f183ab7, 0x1e55d576
       data4 0x9e6bb99f, 0x9f52d7cb, 0x9e73a0f5, 0x1d4e1d14
       data4 0x9dd05b53, 0x1f2261e4, 0x9d4ee73d, 0x1ede515e
       data4 0x1f22a573, 0x9ecac348, 0x1e6a2ac0, 0x1e2787d2
       data4 0x9eb64b87, 0x1f0c69c6, 0x9f470a01, 0x9d7c1686
       data4 0x1e468ebe, 0x9f21ee2f, 0x9ee52116, 0x9e20f715
       data4 0x1ed18533, 0x9f005b38, 0x9f20cb95, 0x1da72967
       data4 0x1f1ba5d7, 0x1e2f8b16, 0x9c794f96, 0x9ca74ea3
       data4 0x1f410555, 0x9eff2b96, 0x1ce8f0b1, 0x1f0cee77
       data4 0x1f191edd, 0x9ed5fcbc, 0x1f30f242, 0x9e0ad369
       data4 0x1ed8f3c8, 0x1f52bb0e, 0x9e9ce408, 0x1f18907f
       data4 0x9ecdad40, 0x9e8af91d, 0x1d46698a, 0x9f4b93d6
       data4 0x9f3f5d33, 0x1e2e52f7, 0x9f13aeec, 0x9f3b1969
       data4 0x1f0996f4, 0x9f2a03df, 0x1e264767, 0x1f3ab1fb
       data4 0x9f3193c9, 0x9f21ce22, 0x9eab624c, 0x9ecd8fb1
       data4 0x1eaf9a85, 0x1f0c6a2c, 0x1eecbe61, 0x1f3fead9
       data4 0x1f1d3a29, 0x1e9099ce, 0x1eadd875, 0x1e4dbfb8
       data4 0x9dc640d2, 0x1f413680, 0x9f3f57b3, 0x1dfa1553
       data4 0x1ec71c6b, 0x1e00cc00, 0x9f271e55, 0x1e5a88bb
       data4 0x1f46cc2b, 0x1ee80ff9, 0x9e29c6f3, 0x1f15e229
       data4 0x9ea83d66, 0x1f37408e, 0x9dacb66e, 0x1e6f6259
       data4 0x9f106973, 0x1dd4e5ac, 0x1cbfdcc8, 0x9f231c9f
       data4 0x9e8677e4, 0x9e9e695a, 0x1efd782b, 0x9dd26959
       data4 0x9e80af69, 0x1f386fb3, 0x1f022e8c, 0x9e839967
       data4 0x1ce6796f, 0x1e4c22c2, 0x1e57ef24, 0x1e919804
       data4 0x9d7ea090, 0x1e40140a, 0x1f261b46, 0x1db75be2
       data4 0x1f145019, 0x9e3102b9, 0x9e22507b, 0x1eae813c
       data4 0x1f117e97, 0x1f282296, 0x1f3814b3, 0x1e17977b
       data4 0x1f39d6ff, 0x9f1c81b9, 0x9eb5bcad, 0x1f0f596e
       data4 0x1e757fd5, 0x9f090daa, 0x9f2532fc, 0x9eebafbb
       data4 0x1f086556, 0x9eeedde8, 0x9f32e174, 0x1e33c030
       data4 0x1f1f145a, 0x1e6e556c, 0x1e419ffb, 0x9eb6019a
       data4 0x9e872a2e, 0x1e113136, 0x1e93096f, 0x1f39be40
       data4 0x1f1665ad, 0x9db81d7d, 0x9cd29091, 0x1e3f4af7
       data4 0x9f23176c, 0x9eccf9b3, 0x1f34fc6c, 0x9ed36894
       data4 0x1ef08e06, 0x9f3b46bb, 0x9f2c850b, 0x1f1565a4
       data4 0x1e887bc3, 0x1e92629c, 0x9f11ac9e, 0x9e5579f3
       data4 0x1e4d5790, 0x9ee1c3d1, 0x9e916aec, 0x9eb8d9b8
       data4 0x1db46105, 0x1e168663, 0x1f26a942, 0x9f0f0383
       data4 0x9f079032, 0x9ecae1d8, 0x1ed3b34c, 0x9edc5ee6
       data4 0x9e8a75a7, 0x1f3c3de2, 0x9ee5041e, 0x1f08c727
       data4 0x1d02d7ae, 0x9f36adda, 0x9ef9a857, 0x9ef5cb3a
       data4 0x9eee73da, 0x9da5d629, 0x1e0e99be, 0x1e5159b9
       data4 0x1f2eac89, 0x9e8eedc5, 0x1dd0ec90, 0x1f229aff
       data4 0x1ed9c3e6, 0x1e95c55a, 0x9f0c24e4, 0x1e8afed6
       data4 0x1e599a96, 0x1e881b21, 0x1eab84b9, 0x9ba2bb0e
       data4 0x9e33ab10, 0x1f1710b5, 0x1ebfa271, 0x9e90bbc5
       data4 0x9f32515b, 0x9b32aae8, 0x1eda455c, 0x1da8186e
       data4 0x9e8917ff, 0x1ec4d08e, 0x1c90069d, 0x9f2f1d29
       data4 0x9ecee86d, 0x9f234d1f, 0x1f370724, 0x1da87496
       data4 0x1e7959f0, 0x9e8ada34, 0x1f1c7f6f, 0x1edd576b
       data4 0x9de91e8b, 0x1ec4ef89, 0x1f32078a, 0x1e9925e2
       data4 0x9d8eeccb, 0x9ea3d011, 0x1f231fdf, 0x9f1dbdfa
       data4 0x1e7507a3, 0x1ec42614, 0x9e8693cb, 0x9ec68398
       data4 0x1d5b05fb, 0x1de32119, 0x9f003429, 0x9ec16d92
       data4 0x9f095315, 0x9f119d2c, 0x9ed0c984, 0x9f090662
       data4 0x9e59aa1f, 0x9ed4e64a, 0x9f2798a7, 0x9f23624d
       data4 0x1e0467d9, 0x1f22e7e7, 0x1e915256, 0x9cb4df70
       data4 0x9e6f687c, 0x9e3c35e5, 0x9e5757ab, 0x9f031fa1
       data4 0x1f25bff7, 0x1f0e58c2, 0x1ef3ce04, 0x1f002ecb
       data4 0x9ebdc836, 0x9ed657dd, 0x9f149441, 0x9e8544b2
       data4 0x1cd8ff1e, 0x1e9bb463, 0x1eaa1c5c, 0x1f200c1a
       data4 0x1edbfbaf, 0x1f18724d, 0x9ed63c22, 0x9f08e045
       data4 0x1f13ad07, 0x9e949311, 0x9f0c50d4, 0x1e824516
       data4 0x1d5e52ba, 0x1d583fbd, 0x1e3b60a9, 0x9effe6d3
       data4 0x1f0d0508, 0x1f00be77, 0x9e404bfa, 0x9e1ca381
       data4 0x9f084dd8, 0x9e6db85d, 0x1db698e4, 0x9ebd1871
       data4 0x9ecc2679, 0x1ee68442, 0x1edb1050, 0x9dbc96a4
       data4 0x9f27c1f4, 0x1c99b756, 0x1eb4400a, 0x9f24390a
       data4 0x1d927875, 0x9f074faa, 0x1e9dc2c3, 0x1f13c0d2
       data4 0x1e3c9685, 0x9e6b6f75, 0x9db9cb31, 0x1ea5f3aa
       data4 0x9d992c61, 0x1f1015e4, 0x1f194f70, 0x9e19d2b3
       data4 0x9d89116c, 0x1f23cd35, 0x1e33d3a2, 0x1ee331b8
       data4 0x1d5ba7ec, 0x9f273788, 0x9e6907f4, 0x9ed5f912
       data4 0x9edd458d, 0x1e2ca7b2, 0x1ef81fe4, 0x1dc7ade6
       data4 0x1e876e51, 0x9f04ec89, 0x1f1da63a, 0x1ec02bd0
       data4 0x9e71326f, 0x1e7847b4, 0x1f0de618, 0x9e036cb6
       data4 0x1eec61e2, 0x1ef1758b, 0x9ee880a3, 0x1ed269d7
       data4 0x1e27edd3, 0x9e8a81a1, 0x1eacb84d, 0x9e1aad37
       data4 0x1f1aa8f7, 0x1e9bbd90, 0x1ea1b61f, 0x9ed41c2f
       data4 0x1dbb5dd6, 0x1f0ec733, 0x9df06b1b, 0x1e06fef1
       data4 0x9edede3a, 0x1edeb5e2, 0x1f0e63ee, 0x9db316bb
       data4 0x9efc1ad3, 0x1f01fbb5, 0x9cc0d078, 0x1ea28b36
       data4 0x9e9dd205, 0x9e791534, 0x1da1c8d5, 0x9e8195cc
       data4 0x1f0681a4, 0x1eeaf1e2, 0x9ef83b37, 0x9f22a92b
       data4 0x1eabc4ce, 0x1f10eefb, 0x1e06d9aa, 0x1e7cacd5
       data4 0x1f1ea087, 0x1eb21983, 0x9f100c78, 0x1e840abe
       data4 0x9efab66c, 0x1f183fa8, 0x9e84ee68, 0x9eea083d
       data4 0x9ee23a74, 0x1f1351d7, 0x9ec5d42a, 0x9f071f57
       data4 0x9ef578d9, 0x9f1aa7e7, 0x1eb02044, 0x1f151a2e
       data4 0x9c0dc8b2, 0x9ef4087a, 0x1ec12b93, 0x1c1a946b
       data4 0x1e89946f, 0x9dafe8c3, 0x1d295288, 0x9e8497ab
       data4 0x1ec000c6, 0x1e102f29, 0x1e542256, 0x1e67d44d
       data4 0x1ef688d8, 0x1f0e0f29, 0x1e67861f, 0x1e869748
       data4 0x1ee6aa6e, 0x9e4d228b, 0x9e50be5b, 0x1e9fe225
       data4 0x9ea34102, 0x9e628a3b, 0x9ed9fd83, 0x1ecd7109
       data4 0x1f1864ff, 0x1ea19b76, 0x1db0d1c9, 0x9dff519b
       data4 0x1e8fea71, 0x9ee82e9a, 0x9f08919b, 0x9ef5c8ae
       data4 0x9ee446a4, 0x1ea59444, 0x1eb74230, 0x1ea13fbf
       data4 0x9ea6a3ea, 0x1e5f2797, 0x9e0adb07, 0x9d3adadd
       data4 0x1ebf2ee2, 0x1da19bfa, 0x1e8dea6d, 0x1ec4fea9
       data4 0x1e669f22, 0x1dc5f919, 0x9ed25caa, 0x1ee475b1
       data4 0x1ed0603e, 0x9eacb35c, 0x1dc00b27, 0x1e2f9991
       data4 0x1e7b0406, 0x1eaa3387, 0x9d865bde, 0x1eb78a48
       data4 0x1c40ae2e, 0x1ee9838b, 0x9f0f0d7f, 0x1e3e5d26
       data4 0x1e99e7a6, 0x9e681ccf, 0x9e93ed65, 0x9eeb6a66
       data4 0x1e29e9af, 0x9e96f923, 0x9e74f11d, 0x9f1474da
       data4 0x1eec2ea7, 0x1ebf7aa3, 0x9c25dcca, 0x9f0553c2
       data4 0x9e599efd, 0x1d2ab490, 0x1e95d7cd, 0x9ee4b20e
       data4 0x9d988ce5, 0x9ef9787e, 0x9dbbba5b, 0x9f12c304
       data4 0x1e3b9d70, 0x1e7bcae8, 0x9d98bb6e, 0x9e8e6b01
       data4 0x9f07d03b, 0x9d67c822, 0x9f0ef69e, 0x1c7c0fe3
       data4 0x9e9bfbb9, 0x9e83b84b, 0x1efbf15e, 0x9ecfa6a6
       data4 0x9c91158e, 0x9ecf6770, 0x1ee1e3a8, 0x9dc95ec0
       data4 0x1ef603f7, 0x1d5e52ba, 0x1c477d1b, 0x9e955cd8
       data4 0x1ed665b0, 0x9e8376c4, 0x9c0ee88e, 0x1e8c989e
       data4 0x1ea2df29, 0x9d961e5c, 0x1e101813, 0x1e7fffff
       data4 0x9e5abff4, 0x1dbddd71, 0x1eb69100, 0x1e71f114
       data4 0x1e9ca798, 0x1ef62c8d, 0x9db4e55a, 0x1dbe69ce
       data4 0x9ef1c01f, 0x1f044a2a, 0x9eb9e0d7, 0x9ee59745
       data4 0x9e874803, 0x1ea0b418, 0x9e13572a, 0x1ddbb3a2
       data4 0x9ec0e391, 0x1e89fba1, 0x1ee8b261, 0x9e5d25f0
       data4 0x9ef222cb, 0x9ef135ec, 0x1ea04b9a, 0x9f04291f
       data4 0x9e969254, 0x9ee32f08, 0x9ed909d3, 0x9e362640
       data4 0x9ec20735, 0x1e50131b, 0x9ed4e049, 0x1ee8e817
       data4 0x1e1e09c0, 0x9ea643c5, 0x9e5a1ab6, 0x9e389059
       data4 0x1e560947, 0x1d02b877, 0x1e4475ab, 0x9ea9aaf6
       data4 0x1e95bc5e, 0x1eaf6afd, 0x1d43067d, 0x9d043821
       data4 0x9e97baa9, 0x1de5c4f9, 0x9e9a0069, 0x9e1b9944
       data4 0x1eb13686, 0x9eb907eb, 0x1e059589, 0x1cbd0f93
       data4 0x9eb7e6ae, 0x1e9fa175, 0x1ee5bdf4, 0x1e8052f7
       data4 0x9c80d1e3, 0x1bfbe28e, 0x9e672b3b, 0x9ecacf19
       data4 0x9e3c04be, 0x1dfe8c5c, 0x1e1ba9cb, 0x1eb40b1e
       data4 0x1ec7e7f6, 0x9d0d45b3, 0x1ef0113b, 0x9a155fa3
       data4 0x1e28ec3b, 0x1e7ca8df, 0x9d2f91b4, 0x1eccd9ed
       data4 0x9ed943bc, 0x9ccaab19, 0x9e8a5c58, 0x1ec3bca8
       data4 0x1ed78dc7, 0x9ed391a8, 0x9e938f6e, 0x9ec4a030
       data4 0x9e80346e, 0x1e7a4686, 0x9e284315, 0x9e39584c
       data4 0x1ebdc9b4, 0x9e9cfce5, 0x9ef55c65, 0x1e2941e7
       data4 0x9efbe59f, 0x1d87c41b, 0x1e40befc, 0x1e3d05b5
       data4 0x1de9ea67, 0x1ec9a21c, 0x1decb69a, 0x1df6e75a
       data4 0x9e8030ab, 0x9db20540, 0x9ef1e977, 0x1e3cdc43
       data4 0x1e0492b0, 0x9e91d872, 0x1e775346, 0x9e939978
       data4 0x1eb2714e, 0x1e49a203, 0x9e10195a, 0x1ef1ffc3
       data4 0x9ea8b709, 0x9e832e27, 0x1ed5ac3b, 0x1edb20a6
       data4 0x1e4dbd4e, 0x1efbb932, 0x1d8170ec, 0x1e6c4849
       data4 0x1f008e17, 0x1e8000c4, 0x1d855ecf, 0x9e37cb85
       data4 0x1ecffdf5, 0x1eba6519, 0x9edbe600, 0x1ea3e5e7
       data4 0x1ed4fb39, 0x1f00be77, 0x1e6f4484, 0x9e9e7107
       data4 0x9e30b29d, 0x9ee6e174, 0x1e3a2656, 0x9dd72f3f
       data4 0x9ee12138, 0x1ed16fed, 0x9ece8a02, 0x9ca5b249
       data4 0x9eafd508, 0x9ef0e9fc, 0x1d1307ac, 0x1eecee20
       data4 0x1cf60c6f, 0x9d556216, 0x9eaed175, 0x9ec919f4
       data4 0x1ec2c988, 0x1cd82772, 0x9dc99456, 0x1eab0467
       data4 0x1e89b36f, 0x1c757944, 0x1eef9abd, 0x9e98664d
LOCAL_OBJECT_END(D_table)


.section .text
GLOBAL_LIBM_ENTRY(cbrtl)

{ .mfi
       getf.sig GR_ARGSIG = f8
       // will continue on main path only for normal/denormal numbers
       // all other values will be filtered out and will exit early
       fclass.nm.unc p12, p7 = f8, 0x1b
       // GR_ADDR = pointer to C_1...C_6 followed by T_table
       addl GR_ADDR = @ltoff(poly_coeffs), gp
}
{ .mfi
       // GR_BIAS23 = 2/3*bias -63 = 0xaaaa-0x3f = 0xaa6b
       mov GR_BIAS23 = 0xaa6b
       // normalize a
       fma.s1 FR_XNORM = f8, f1, f0
       // GR_D_ADDR = pointer to D table
       addl GR_D_ADDR = @ltoff(D_table), gp
}
;;

{ .mmf
       // load start address for C_1...C_6 followed by T_table
       ld8 GR_C_START = [ GR_ADDR ]
       // load start address of D table
       ld8 GR_D_START = [ GR_D_ADDR ]
       // y = frcpa(a)
       frcpa.s1 FR_RCP, p6 = f1, f8
}
;;

{ .mmi
        // get normalized significand
       getf.sig GR_NORMSIG = FR_XNORM
        // get exponent
       getf.exp GR_NORMEXPSGN = FR_XNORM
 (p7)  cmp.eq p12, p0 = GR_ARGSIG, r0
}
;;

{ .mii
       // load C_1
       ldfe FR_C1 = [ GR_C_START ], 16
       mov GR_SGNMASK = 0x20000
       nop.i 0
}
;;

{ .mfb
       // load C_2
       ldfe FR_C2 = [ GR_C_START ], 16
 (p12) fma.s0 f8 = f8, f1, f0
       // NaN/Infinities exit early
 (p12) br.ret.spnt b0
}
;;

{ .mfi
       // load C_3, C_4
       ldfpd FR_C3, FR_C4 = [ GR_C_START ], 16
       // y = frcpa(a), set flags and result when argument is 0
       // only used when p6=0
       frcpa.s0 f8, p0 = f1, f8
       nop.i 0
}
;;

{ .mii
       // get GR_SIGN = sign
       and GR_SIGN = GR_NORMEXPSGN, GR_SGNMASK
       // eliminate leading 1 from GR_NORMSIG = 2nd table index
       shl GR_INDEX2 = GR_NORMSIG, 1
       // eliminate sign from exponent
       andcm GR_NORMEXP = GR_NORMEXPSGN, GR_SGNMASK
}
;;

{ .mfi
       // load C_5, C_6
 (p6)  ldfpd FR_C5, FR_C6 = [ GR_C_START ], 16
       // r = 1-a*y
 (p6)  fnma.s1 FR_R = FR_RCP, FR_XNORM, f1
       // Start computation of floor(exponent/3) by
       // computing (2^20+2)/3*exponent = exponent*0x55556
       // 1: exponent* = 5;
       // (2^{16}-1)/3 = 0x5555:
       // will form 0x5555*exponent by using shladd's
       shladd GR_EXP5 = GR_NORMEXP, 2, GR_NORMEXP
}
;;

{ .mib
       // Next several integer steps compute floor(exponent/3)
       // GR_TMP1 = (5*expon)*16
       shladd GR_TMP1 = GR_EXP5, 4, r0
       // GR_EXP3 = 3*exponent
       shladd GR_EXP3 = GR_NORMEXP, 1, GR_NORMEXP
       nop.b 0
}
;;

{ .mmi
       // GR_EXP6 = 6*exponent
       shladd GR_EXP6 = GR_EXP3, 1, r0
       // GR_EXP17 = 17*expon
       add GR_EXP17 = GR_EXP5, GR_TMP1
       // GR_IX2 = 2nd table index (8 bits)
       shr.u GR_IX2 = GR_INDEX2, 56
}
;;

{ .mmi
       // adjust T_table pointer by 2nd index
       shladd GR_T_INDEX = GR_IX2, 3, GR_C_START
       // adjust D_table pointer by 2nd index
       shladd GR_D_INDEX = GR_IX2, 2, GR_D_START
       // GR_TMP2 = (17*expon)*16^2
       shl GR_TMP2 = GR_EXP17, 8
}
;;

{ .mmi
       // GR_TMP3 = expon*(2^16-1)/3
       add GR_TMP3 = GR_EXP17, GR_TMP2
;;
       // GR_TMP4 = expon*(2^20+2)/3 = expon*0x55556
       shladd GR_TMP4 = GR_TMP3, 4, GR_EXP6
       nop.i 0
}
;;

{ .mii
       nop.m 0
       // GR_EXP_RES = floor(expon/3)
       shr.u GR_EXP_RES = GR_TMP4, 20
       nop.i 0
}
;;

{ .mmi
       nop.m 0
       // r16 = 3*exponent
       shladd r16 = GR_EXP_RES, 1, GR_EXP_RES
       // bias exponent
       add GR_EXPBIAS = GR_BIAS23, GR_EXP_RES
}
;;

{ .mmi
       // get remainder of exponent/3
       sub GR_EXP_MOD_3 = GR_NORMEXP, r16
;;
       // add sign to exponent
       or GR_EXPSIGNRES = GR_EXPBIAS, GR_SIGN
       // remainder << = 8
       shl GR_REMTMP = GR_EXP_MOD_3, 8
}
;;

{ .mfi
       // adjust D_table pointer by 1st index
       shladd GR_IX_D = GR_REMTMP, 2, GR_D_INDEX
       // P_1 = C_1+C_2*r
 (p6)  fma.s1 FR_P1 = FR_C2, FR_R, FR_C1
       // adjust T_table pointer by 1st index
       shladd GR_IX_T = GR_REMTMP, 3, GR_T_INDEX
}
{ .mfi
       // FR_SGNEXP = sign*2^{exponent/3}
 (p6)  setf.exp FR_SGNEXP = GR_EXPSIGNRES
       // r^2 = r*r
 (p6)  fma.s1 FR_R2 = FR_R, FR_R, f0
       nop.i 0
}
;;

{ .mfi
       // load D
 (p6)  ldfs FR_D = [ GR_IX_D ]
       // P_2 = C_3+C_4*r
 (p6)  fma.s1 FR_P2 = FR_C4, FR_R, FR_C3
       nop.i 0
}
{ .mfi
       // load T
 (p6)  ldf8 FR_T = [ GR_IX_T ]
       // P_3 = C_5+C_6*r
 (p6)  fma.s1 FR_P3 = FR_C6, FR_R, FR_C5
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // P_4 = D-r*P_1
 (p6)  fnma.s1 FR_P4 = FR_R, FR_P1, FR_D
       nop.i 0
}
{ .mfi
       nop.m 0
       // r^3 = r*r^2
 (p6)  fma.s1 FR_R3 = FR_R, FR_R2, f0
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // P_5 = P_2+r2*P_3
 (p6)  fma.s1 FR_P5 = FR_R2, FR_P3, FR_P2
       nop.i 0
}
;;

{ .mfi
       nop.m 0
       // T = T*(sign*2^{exponent/3})
 (p6)  fma.s1 FR_TF = FR_T, FR_SGNEXP, f0
       nop.i 0
}
{ .mfi
       nop.m 0
       // P = P_4-r3*P_5
 (p6)  fnma.s1 FR_P = FR_R3, FR_P5, FR_P4
       nop.i 0
}
;;

{ .mfb
       nop.m 0
       // result = T+T*p
 (p6)  fma.s0 f8 = FR_TF, FR_P, FR_TF
       br.ret.sptk b0
}
;;

GLOBAL_LIBM_END(cbrtl)
libm_alias_ldouble_other (cbrt, cbrt)
