.file "acosl.s"


// Copyright (c) 2001 - 2003, Intel Corporation
// All rights reserved.
//
// Contributed 2001 by the Intel Numerics Group, Intel Corporation
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
// 08/28/01 New version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/06/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// long double acosl(long double)
//
// Overview of operation
//==============================================================
// Background
//
// Implementation
//
// For |s| in [2^{-4}, sqrt(2)/2]:
// Let t= 2^k*1.b1 b2..b6 1, where s= 2^k*1.b1 b2.. b52
// acos(s)= pi/2-asin(t)-asin(r), where r= s*sqrt(1-t^2)-t*sqrt(1-s^2), i.e.
// r= (s-t)*sqrt(1-t^2)-t*sqrt(1-t^2)*(sqrt((1-s^2)/(1-t^2))-1)
// asin(r)-r evaluated as 9-degree polynomial (c3*r^3+c5*r^5+c7*r^7+c9*r^9)
// The 64-bit significands of sqrt(1-t^2), 1/(1-t^2) are read from the table,
// along with the high and low parts of asin(t) (stored as two double precision
// values)
//
// |s| in (sqrt(2)/2, sqrt(255/256)):
// Let t= 2^k*1.b1 b2..b6 1, where (1-s^2)*frsqrta(1-s^2)= 2^k*1.b1 b2..b6..
// acos(|s|)= asin(t)-asin(r)
// acos(-|s|)=pi-asin(t)+asin(r),   r= s*t-sqrt(1-s^2)*sqrt(1-t^2)
// To minimize accumulated errors, r is computed as
// r= (t*s)_s-t^2*y*z+z*y*(t^2-1+s^2)_s+z*y*(1-s^2)_s*x+z'*y*(1-s^2)*PS29+
// +(t*s-(t*s)_s)+z*y*((t^2-1-(t^2-1+s^2)_s)+s^2)+z*y*(1-s^2-(1-s^2)_s)+
// +ez*z'*y*(1-s^2)*(1-x),
// where y= frsqrta(1-s^2), z= (sqrt(1-t^2))_s (rounded to 24 significant bits)
// z'= sqrt(1-t^2), x= ((1-s^2)*y^2-1)/2
//
// |s|<2^{-4}: evaluate asin(s) as 17-degree polynomial, return pi/2-asin(s)
// (or simply return pi/2-s, if|s|<2^{-64})
//
// |s| in [sqrt(255/256), 1): acos(|s|)= asin(sqrt(1-s^2))
// acos(-|s|)= pi-asin(sqrt(1-s^2))
// use 17-degree polynomial for asin(sqrt(1-s^2)),
// 9-degree polynomial to evaluate sqrt(1-s^2)
// High order term is (pi)_high-(y*(1-s^2))_high, for s<0,
// or y*(1-s^2)_s, for s>0
//



// Registers used
//==============================================================
// f6-f15, f32-f36
// r2-r3, r23-r23
// p6, p7, p8, p12
//


       GR_SAVE_B0= r33
       GR_SAVE_PFS= r34
       GR_SAVE_GP= r35 // This reg. can safely be used
       GR_SAVE_SP= r36

       GR_Parameter_X= r37
       GR_Parameter_Y= r38
       GR_Parameter_RESULT= r39
       GR_Parameter_TAG= r40

       FR_X= f10
       FR_Y= f1
       FR_RESULT= f8



RODATA

.align 16

LOCAL_OBJECT_START(T_table)

// stores 64-bit significand of 1/(1-t^2), 64-bit significand of sqrt(1-t^2),
// asin(t)_high (double precision), asin(t)_low (double precision)

data8 0x80828692b71c4391, 0xff7ddcec2d87e879
data8 0x3fb022bc0ae531a0, 0x3c9f599c7bb42af6
data8 0x80869f0163d0b082, 0xff79cad2247914d3
data8 0x3fb062dd26afc320, 0x3ca4eff21bd49c5c
data8 0x808ac7d5a8690705, 0xff75a89ed6b626b9
data8 0x3fb0a2ff4a1821e0, 0x3cb7e33b58f164cc
data8 0x808f0112ad8ad2e0, 0xff7176517c2cc0cb
data8 0x3fb0e32279319d80, 0x3caee31546582c43
data8 0x80934abba8a1da0a, 0xff6d33e949b1ed31
data8 0x3fb12346b8101da0, 0x3cb8bfe463d087cd
data8 0x8097a4d3dbe63d8f, 0xff68e16571015c63
data8 0x3fb1636c0ac824e0, 0x3c8870a7c5a3556f
data8 0x809c0f5e9662b3dd, 0xff647ec520bca0f0
data8 0x3fb1a392756ed280, 0x3c964f1a927461ae
data8 0x80a08a5f33fadc66, 0xff600c07846a6830
data8 0x3fb1e3b9fc19e580, 0x3c69eb3576d56332
data8 0x80a515d91d71acd4, 0xff5b892bc475affa
data8 0x3fb223e2a2dfbe80, 0x3c6a4e19fd972fb6
data8 0x80a9b1cfc86ff7cd, 0xff56f631062cf93d
data8 0x3fb2640c6dd76260, 0x3c62041160e0849e
data8 0x80ae5e46b78b0d68, 0xff5253166bc17794
data8 0x3fb2a43761187c80, 0x3cac61651af678c0
data8 0x80b31b417a4b756b, 0xff4d9fdb14463dc8
data8 0x3fb2e46380bb6160, 0x3cb06ef23eeba7a1
data8 0x80b7e8c3ad33c369, 0xff48dc7e1baf6738
data8 0x3fb32490d0d910c0, 0x3caa05f480b300d5
data8 0x80bcc6d0f9c784d6, 0xff4408fe9ad13e37
data8 0x3fb364bf558b3820, 0x3cb01e7e403aaab9
data8 0x80c1b56d1692492d, 0xff3f255ba75f5f4e
data8 0x3fb3a4ef12ec3540, 0x3cb4fe8fcdf5f5f1
data8 0x80c6b49bc72ec446, 0xff3a319453ebd961
data8 0x3fb3e5200d171880, 0x3caf2dc089b2b7e2
data8 0x80cbc460dc4e0ae8, 0xff352da7afe64ac6
data8 0x3fb425524827a720, 0x3cb75a855e7c6053
data8 0x80d0e4c033bee9c4, 0xff301994c79afb32
data8 0x3fb46585c83a5e00, 0x3cb3264981c019ab
data8 0x80d615bdb87556db, 0xff2af55aa431f291
data8 0x3fb4a5ba916c73c0, 0x3c994251d94427b5
data8 0x80db575d6291fd8a, 0xff25c0f84bae0cb9
data8 0x3fb4e5f0a7dbdb20, 0x3cbee2fcc4c786cb
data8 0x80e0a9a33769e535, 0xff207c6cc0ec09fd
data8 0x3fb526280fa74620, 0x3c940656e5549b91
data8 0x80e60c93498e32cd, 0xff1b27b703a19c98
data8 0x3fb56660ccee2740, 0x3ca7082374d7b2cd
data8 0x80eb8031b8d4052d, 0xff15c2d6105c72f8
data8 0x3fb5a69ae3d0b520, 0x3c7c4d46e09ac68a
data8 0x80f10482b25c6c8a, 0xff104dc8e0813ed4
data8 0x3fb5e6d6586fec20, 0x3c9aa84ffd9b4958
data8 0x80f6998a709c7cfb, 0xff0ac88e6a4ab926
data8 0x3fb627132eed9140, 0x3cbced2cbbbe7d16
data8 0x80fc3f4d3b657c44, 0xff053325a0c8a2ec
data8 0x3fb667516b6c34c0, 0x3c6489c5fc68595a
data8 0x8101f5cf67ed2af8, 0xfeff8d8d73dec2bb
data8 0x3fb6a791120f33a0, 0x3cbe12acf159dfad
data8 0x8107bd1558d6291f, 0xfef9d7c4d043df29
data8 0x3fb6e7d226fabba0, 0x3ca386d099cd0dc7
data8 0x810d95237e38766a, 0xfef411ca9f80b5f7
data8 0x3fb72814ae53cc20, 0x3cb9f35731e71dd6
data8 0x81137dfe55aa0e29, 0xfeee3b9dc7eef009
data8 0x3fb76858ac403a00, 0x3c74df3dd959141a
data8 0x811977aa6a479f0f, 0xfee8553d2cb8122c
data8 0x3fb7a89e24e6b0e0, 0x3ca6034406ee42bc
data8 0x811f822c54bd5ef8, 0xfee25ea7add46a91
data8 0x3fb7e8e51c6eb6a0, 0x3cb82f8f78e68ed7
data8 0x81259d88bb4ffac1, 0xfedc57dc2809fb1d
data8 0x3fb8292d9700ad60, 0x3cbebb73c0e653f9
data8 0x812bc9c451e5a257, 0xfed640d974eb6068
data8 0x3fb8697798c5d620, 0x3ca2feee76a9701b
data8 0x813206e3da0f3124, 0xfed0199e6ad6b585
data8 0x3fb8a9c325e852e0, 0x3cb9e88f2f4d0efe
data8 0x813854ec231172f9, 0xfec9e229dcf4747d
data8 0x3fb8ea1042932a00, 0x3ca5ff40d81f66fd
data8 0x813eb3e209ee858f, 0xfec39a7a9b36538b
data8 0x3fb92a5ef2f247c0, 0x3cb5e3bece4d6b07
data8 0x814523ca796f56ce, 0xfebd428f72561efe
data8 0x3fb96aaf3b3281a0, 0x3cb7b9e499436d7c
data8 0x814ba4aa6a2d3ff9, 0xfeb6da672bd48fe4
data8 0x3fb9ab011f819860, 0x3cb9168143cc1a7f
data8 0x81523686e29bbdd7, 0xfeb062008df81f50
data8 0x3fb9eb54a40e3ac0, 0x3cb6e544197eb1e1
data8 0x8158d964f7124614, 0xfea9d95a5bcbd65a
data8 0x3fba2ba9cd080800, 0x3ca9a717be8f7446
data8 0x815f8d49c9d639e4, 0xfea34073551e1ac8
data8 0x3fba6c009e9f9260, 0x3c741e989a60938a
data8 0x8166523a8b24f626, 0xfe9c974a367f785c
data8 0x3fbaac591d0661a0, 0x3cb2c1290107e57d
data8 0x816d283c793e0114, 0xfe95ddddb94166cb
data8 0x3fbaecb34c6ef600, 0x3c9c7d5fbaec405d
data8 0x81740f54e06d55bd, 0xfe8f142c93750c50
data8 0x3fbb2d0f310cca00, 0x3cbc09479a9cbcfb
data8 0x817b07891b15cd5e, 0xfe883a3577e9fceb
data8 0x3fbb6d6ccf1455e0, 0x3cb9450bff4ee307
data8 0x818210de91bba6c8, 0xfe814ff7162cf62f
data8 0x3fbbadcc2abb1180, 0x3c9227fda12a8d24
data8 0x81892b5abb0f2bf9, 0xfe7a55701a8697b1
data8 0x3fbbee2d48377700, 0x3cb6fad72acfe356
data8 0x819057031bf7760e, 0xfe734a9f2dfa1810
data8 0x3fbc2e902bc10600, 0x3cb4465b588d16ad
data8 0x819793dd479d4fbe, 0xfe6c2f82f643f68b
data8 0x3fbc6ef4d9904580, 0x3c8b9ac54823960d
data8 0x819ee1eedf76367a, 0xfe65041a15d8a92c
data8 0x3fbcaf5b55dec6a0, 0x3ca2b8d28a954db2
data8 0x81a6413d934f7a66, 0xfe5dc8632be3477f
data8 0x3fbcefc3a4e727a0, 0x3c9380da83713ab4
data8 0x81adb1cf21597d4b, 0xfe567c5cd44431d5
data8 0x3fbd302dcae51600, 0x3ca995b83421756a
data8 0x81b533a9563310b8, 0xfe4f2005a78fb50f
data8 0x3fbd7099cc155180, 0x3caefa2f7a817d5f
data8 0x81bcc6d20cf4f373, 0xfe47b35c3b0caaeb
data8 0x3fbdb107acb5ae80, 0x3cb455fc372dd026
data8 0x81c46b4f2f3d6e68, 0xfe40365f20b316d6
data8 0x3fbdf177710518c0, 0x3cbee3dcc5b01434
data8 0x81cc2126b53c1144, 0xfe38a90ce72abf36
data8 0x3fbe31e91d439620, 0x3cb3e131c950aebd
data8 0x81d3e85ea5bd8ee2, 0xfe310b6419c9c33a
data8 0x3fbe725cb5b24900, 0x3c01d3fac6029027
data8 0x81dbc0fd1637b9c1, 0xfe295d6340932d15
data8 0x3fbeb2d23e937300, 0x3c6304cc44aeedd1
data8 0x81e3ab082ad5a0a4, 0xfe219f08e03580b3
data8 0x3fbef349bc2a77e0, 0x3cac1d2d6abe9c72
data8 0x81eba6861683cb97, 0xfe19d0537a0946e2
data8 0x3fbf33c332bbe020, 0x3ca0909dba4e96ca
data8 0x81f3b37d1afc9979, 0xfe11f1418c0f94e2
data8 0x3fbf743ea68d5b60, 0x3c937fc12a2a779a
data8 0x81fbd1f388d4be45, 0xfe0a01d190f09063
data8 0x3fbfb4bc1be5c340, 0x3cbf51a504b55813
data8 0x820401efbf87e248, 0xfe020201fff9efea
data8 0x3fbff53b970d1e80, 0x3ca625444b260078
data8 0x82106ad2ffdca049, 0xfdf5e3940a49135e
data8 0x3fc02aff52065460, 0x3c9125d113e22a57
data8 0x8221343d6ea1d3e2, 0xfde581a45429b0a0
data8 0x3fc06b84f8e03220, 0x3caccf362295894b
data8 0x82324434adbf99c2, 0xfdd4de1a001fb775
data8 0x3fc0ac0ed1fe7240, 0x3cc22f676096b0af
data8 0x82439aee8d0c7747, 0xfdc3f8e8269d1f03
data8 0x3fc0ec9cee9e4820, 0x3cca147e2886a628
data8 0x825538a1d0fcb2f0, 0xfdb2d201a9b1ba66
data8 0x3fc12d2f6006f0a0, 0x3cc72b36633bc2d4
data8 0x82671d86345c5cee, 0xfda1695934d723e7
data8 0x3fc16dc63789de60, 0x3cb11f9c47c7b83f
data8 0x827949d46a121770, 0xfd8fbee13cbbb823
data8 0x3fc1ae618682e620, 0x3cce1b59020cef8e
data8 0x828bbdc61eeab9ba, 0xfd7dd28bff0c9f34
data8 0x3fc1ef015e586c40, 0x3cafec043e0225ee
data8 0x829e7995fb6de9e1, 0xfd6ba44b823ee1ca
data8 0x3fc22fa5d07b90c0, 0x3cba905409caf8e3
data8 0x82b17d7fa5bbc982, 0xfd5934119557883a
data8 0x3fc2704eee685da0, 0x3cb5ef21838a823e
data8 0x82c4c9bfc373d276, 0xfd4681cfcfb2c161
data8 0x3fc2b0fcc9a5f3e0, 0x3ccc7952c5e0e312
data8 0x82d85e93fba50136, 0xfd338d7790ca0f41
data8 0x3fc2f1af73c6ba00, 0x3cbecf5f977d1ca9
data8 0x82ec3c3af8c76b32, 0xfd2056f9fff97727
data8 0x3fc33266fe6889a0, 0x3c9d329c022ebdb5
data8 0x830062f46abf6022, 0xfd0cde480c43b327
data8 0x3fc373237b34de60, 0x3cc95806d4928adb
data8 0x8314d30108ea35f0, 0xfcf923526c1562b2
data8 0x3fc3b3e4fbe10520, 0x3cbc299fe7223d54
data8 0x83298ca29434df97, 0xfce526099d0737ed
data8 0x3fc3f4ab922e4a60, 0x3cb59d8bb8fdbccc
data8 0x833e901bd93c7009, 0xfcd0e65de39f1f7c
data8 0x3fc435774fea2a60, 0x3c9ec18b43340914
data8 0x8353ddb0b278aad8, 0xfcbc643f4b106055
data8 0x3fc4764846ee80a0, 0x3cb90402efd87ed6
data8 0x836975a60a70c52e, 0xfca79f9da4fab13a
data8 0x3fc4b71e8921b860, 0xbc58f23449ed6365
data8 0x837f5841ddfa7a46, 0xfc92986889284148
data8 0x3fc4f7fa2876fca0, 0xbc6294812bf43acd
data8 0x839585cb3e839773, 0xfc7d4e8f554ab12f
data8 0x3fc538db36ee6960, 0x3cb910b773d4c578
data8 0x83abfe8a5466246f, 0xfc67c2012cb6fa68
data8 0x3fc579c1c6953cc0, 0x3cc5ede909fc47fc
data8 0x83c2c2c861474d91, 0xfc51f2acf82041d5
data8 0x3fc5baade9860880, 0x3cac63cdfc3588e5
data8 0x83d9d2cfc2813637, 0xfc3be08165519325
data8 0x3fc5fb9fb1e8e3a0, 0x3cbf7c8466578c29
data8 0x83f12eebf397daac, 0xfc258b6ce6e6822f
data8 0x3fc63c9731f39d40, 0x3cb6d2a7ffca3e9e
data8 0x8408d76990b9296e, 0xfc0ef35db402af94
data8 0x3fc67d947be9eec0, 0x3cb1980da09e6566
data8 0x8420cc9659487cd7, 0xfbf81841c8082dc4
data8 0x3fc6be97a21daf00, 0x3cc2ac8330e59aa5
data8 0x84390ec132759ecb, 0xfbe0fa06e24cc390
data8 0x3fc6ffa0b6ef05e0, 0x3ccc1a030fee56c4
data8 0x84519e3a29df811a, 0xfbc9989a85ce0954
data8 0x3fc740afcccca000, 0x3cc19692a5301ca6
data8 0x846a7b527842d61b, 0xfbb1f3e9f8e45dc4
data8 0x3fc781c4f633e2c0, 0x3cc0e98f3868a508
data8 0x8483a65c8434b5f0, 0xfb9a0be244f4af45
data8 0x3fc7c2e045b12140, 0x3cb2a8d309754420
data8 0x849d1fabe4e97dd7, 0xfb81e070362116d1
data8 0x3fc80401cddfd120, 0x3ca7a44544aa4ce6
data8 0x84b6e795650817ea, 0xfb6971805af8411e
data8 0x3fc84529a16ac020, 0x3c9e3b709c7d6f94
data8 0x84d0fe6f0589da92, 0xfb50beff0423a2f5
data8 0x3fc88657d30c49e0, 0x3cc60d65a7f0a278
data8 0x84eb649000a73014, 0xfb37c8d84414755c
data8 0x3fc8c78c758e8e80, 0x3cc94b2ee984c2b7
data8 0x85061a50ccd13781, 0xfb1e8ef7eeaf764b
data8 0x3fc908c79bcba900, 0x3cc8540ae794a2fe
data8 0x8521200b1fb8916e, 0xfb05114998f76a83
data8 0x3fc94a0958ade6c0, 0x3ca127f49839fa9c
data8 0x853c7619f1618bf6, 0xfaeb4fb898b65d19
data8 0x3fc98b51bf2ffee0, 0x3c8c9ba7a803909a
data8 0x85581cd97f45e274, 0xfad14a3004259931
data8 0x3fc9cca0e25d4ac0, 0x3cba458e91d3bf54
data8 0x857414a74f8446b4, 0xfab7009ab1945a54
data8 0x3fca0df6d551fe80, 0x3cc78ea1d329d2b2
data8 0x85905de2341dea46, 0xfa9c72e3370d2fbc
data8 0x3fca4f53ab3b6200, 0x3ccf60dca86d57ef
data8 0x85acf8ea4e423ff8, 0xfa81a0f3e9fa0ee9
data8 0x3fca90b777580aa0, 0x3ca4c4e2ec8a867e
data8 0x85c9e62111a92e7d, 0xfa668ab6dec711b1
data8 0x3fcad2224cf814e0, 0x3c303de5980d071c
data8 0x85e725e947fbee97, 0xfa4b3015e883dbfe
data8 0x3fcb13943f7d5f80, 0x3cc29d4eefa5cb1e
data8 0x8604b8a7144cd054, 0xfa2f90fa9883a543
data8 0x3fcb550d625bc6a0, 0x3c9e01a746152daf
data8 0x86229ebff69e2415, 0xfa13ad4e3dfbe1c1
data8 0x3fcb968dc9195ea0, 0x3ccc091bd73ae518
data8 0x8640d89acf78858c, 0xf9f784f9e5a1877b
data8 0x3fcbd815874eb160, 0x3cb5f4b89875e187
data8 0x865f669fe390c7f5, 0xf9db17e65944eacf
data8 0x3fcc19a4b0a6f9c0, 0x3cc5c0bc2b0bbf14
data8 0x867e4938df7dc45f, 0xf9be65fc1f6c2e6e
data8 0x3fcc5b3b58e061e0, 0x3cc1ca70df8f57e7
data8 0x869d80d0db7e4c0c, 0xf9a16f237aec427a
data8 0x3fcc9cd993cc4040, 0x3cbae93acc85eccf
data8 0x86bd0dd45f4f8265, 0xf98433446a806e70
data8 0x3fccde7f754f5660, 0x3cb22f70e64568d0
data8 0x86dcf0b16613e37a, 0xf966b246a8606170
data8 0x3fcd202d11620fa0, 0x3c962030e5d4c849
data8 0x86fd29d7624b3d5d, 0xf948ec11a9d4c45b
data8 0x3fcd61e27c10c0a0, 0x3cc7083c91d59217
data8 0x871db9b741dbe44a, 0xf92ae08c9eca4941
data8 0x3fcda39fc97be7c0, 0x3cc9258579e57211
data8 0x873ea0c3722d6af2, 0xf90c8f9e71633363
data8 0x3fcde5650dd86d60, 0x3ca4755a9ea582a9
data8 0x875fdf6fe45529e8, 0xf8edf92dc5875319
data8 0x3fce27325d6fe520, 0x3cbc1e2b6c1954f9
data8 0x878176321154e2bc, 0xf8cf1d20f87270b8
data8 0x3fce6907cca0d060, 0x3cb6ca4804750830
data8 0x87a36580fe6bccf5, 0xf8affb5e20412199
data8 0x3fceaae56fdee040, 0x3cad6b310d6fd46c
data8 0x87c5add5417a5cb9, 0xf89093cb0b7c0233
data8 0x3fceeccb5bb33900, 0x3cc16e99cedadb20
data8 0x87e84fa9057914ca, 0xf870e64d40a15036
data8 0x3fcf2eb9a4bcb600, 0x3cc75ee47c8b09e9
data8 0x880b4b780f02b709, 0xf850f2c9fdacdf78
data8 0x3fcf70b05fb02e20, 0x3cad6350d379f41a
data8 0x882ea1bfc0f228ac, 0xf830b926379e6465
data8 0x3fcfb2afa158b8a0, 0x3cce0ccd9f829985
data8 0x885252ff21146108, 0xf810394699fe0e8e
data8 0x3fcff4b77e97f3e0, 0x3c9b30faa7a4c703
data8 0x88765fb6dceebbb3, 0xf7ef730f865f6df0
data8 0x3fd01b6406332540, 0x3cdc5772c9e0b9bd
data8 0x88ad1f69be2cc730, 0xf7bdc59bc9cfbd97
data8 0x3fd04cf8ad203480, 0x3caeef44fe21a74a
data8 0x88f763f70ae2245e, 0xf77a91c868a9c54e
data8 0x3fd08f23ce0162a0, 0x3cd6290ab3fe5889
data8 0x89431fc7bc0c2910, 0xf73642973c91298e
data8 0x3fd0d1610f0c1ec0, 0x3cc67401a01f08cf
data8 0x8990573407c7738e, 0xf6f0d71d1d7a2dd6
data8 0x3fd113b0c65d88c0, 0x3cc7aa4020fe546f
data8 0x89df0eb108594653, 0xf6aa4e6a05cfdef2
data8 0x3fd156134ada6fe0, 0x3cc87369da09600c
data8 0x8a2f4ad16e0ed78a, 0xf662a78900c35249
data8 0x3fd19888f43427a0, 0x3cc62b220f38e49c
data8 0x8a811046373e0819, 0xf619e180181d97cc
data8 0x3fd1db121aed7720, 0x3ca3ede7490b52f4
data8 0x8ad463df6ea0fa2c, 0xf5cffb504190f9a2
data8 0x3fd21daf185fa360, 0x3caafad98c1d6c1b
data8 0x8b294a8cf0488daf, 0xf584f3f54b8604e6
data8 0x3fd2606046bf95a0, 0x3cdb2d704eeb08fa
data8 0x8b7fc95f35647757, 0xf538ca65c960b582
data8 0x3fd2a32601231ec0, 0x3cc661619fa2f126
data8 0x8bd7e588272276f8, 0xf4eb7d92ff39fccb
data8 0x3fd2e600a3865760, 0x3c8a2a36a99aca4a
data8 0x8c31a45bf8e9255e, 0xf49d0c68cd09b689
data8 0x3fd328f08ad12000, 0x3cb9efaf1d7ab552
data8 0x8c8d0b520a35eb18, 0xf44d75cd993cfad2
data8 0x3fd36bf614dcc040, 0x3ccacbb590bef70d
data8 0x8cea2005d068f23d, 0xf3fcb8a23ab4942b
data8 0x3fd3af11a079a6c0, 0x3cd9775872cf037d
data8 0x8d48e837c8cd5027, 0xf3aad3c1e2273908
data8 0x3fd3f2438d754b40, 0x3ca03304f667109a
data8 0x8da969ce732f3ac7, 0xf357c60202e2fd7e
data8 0x3fd4358c3ca032e0, 0x3caecf2504ff1a9d
data8 0x8e0baad75555e361, 0xf3038e323ae9463a
data8 0x3fd478ec0fd419c0, 0x3cc64bdc3d703971
data8 0x8e6fb18807ba877e, 0xf2ae2b1c3a6057f7
data8 0x3fd4bc6369fa40e0, 0x3cbb7122ec245cf2
data8 0x8ed5843f4bda74d5, 0xf2579b83aa556f0c
data8 0x3fd4fff2af11e2c0, 0x3c9cfa2dc792d394
data8 0x8f3d29862c861fef, 0xf1ffde2612ca1909
data8 0x3fd5439a4436d000, 0x3cc38d46d310526b
data8 0x8fa6a81128940b2d, 0xf1a6f1bac0075669
data8 0x3fd5875a8fa83520, 0x3cd8bf59b8153f8a
data8 0x901206c1686317a6, 0xf14cd4f2a730d480
data8 0x3fd5cb33f8cf8ac0, 0x3c9502b5c4d0e431
data8 0x907f4ca5fe9cf739, 0xf0f186784a125726
data8 0x3fd60f26e847b120, 0x3cc8a1a5e0acaa33
data8 0x90ee80fd34aeda5e, 0xf09504ef9a212f18
data8 0x3fd65333c7e43aa0, 0x3cae5b029cb1f26e
data8 0x915fab35e37421c6, 0xf0374ef5daab5c45
data8 0x3fd6975b02b8e360, 0x3cd5aa1c280c45e6
data8 0x91d2d2f0d894d73c, 0xefd86321822dbb51
data8 0x3fd6db9d05213b20, 0x3cbecf2c093ccd8b
data8 0x9248000249200009, 0xef7840021aca5a72
data8 0x3fd71ffa3cc87fc0, 0x3cb8d273f08d00d9
data8 0x92bf3a7351f081d2, 0xef16e42021d7cbd5
data8 0x3fd7647318b1ad20, 0x3cbce099d79cdc46
data8 0x93388a8386725713, 0xeeb44dfce6820283
data8 0x3fd7a908093fc1e0, 0x3ccb033ec17a30d9
data8 0x93b3f8aa8e653812, 0xee507c126774fa45
data8 0x3fd7edb9803e3c20, 0x3cc10aedb48671eb
data8 0x94318d99d341ade4, 0xedeb6cd32f891afb
data8 0x3fd83287f0e9cf80, 0x3c994c0c1505cd2a
data8 0x94b1523e3dedc630, 0xed851eaa3168f43c
data8 0x3fd87773cff956e0, 0x3cda3b7bce6a6b16
data8 0x95334fc20577563f, 0xed1d8ffaa2279669
data8 0x3fd8bc7d93a70440, 0x3cd4922edc792ce2
data8 0x95b78f8e8f92f274, 0xecb4bf1fd2be72da
data8 0x3fd901a5b3b9cf40, 0x3cd3fea1b00f9d0d
data8 0x963e1b4e63a87c3f, 0xec4aaa6d08694cc1
data8 0x3fd946eca98f2700, 0x3cdba4032d968ff1
data8 0x96c6fcef314074fc, 0xebdf502d53d65fea
data8 0x3fd98c52f024e800, 0x3cbe7be1ab8c95c9
data8 0x97523ea3eab028b2, 0xeb72aea36720793e
data8 0x3fd9d1d904239860, 0x3cd72d08a6a22b70
data8 0x97dfeae6f4ee4a9a, 0xeb04c4096a884e94
data8 0x3fda177f63e8ef00, 0x3cd818c3c1ebfac7
data8 0x98700c7c6d85d119, 0xea958e90cfe1efd7
data8 0x3fda5d468f92a540, 0x3cdf45fbfaa080fe
data8 0x9902ae7487a9caa1, 0xea250c6224aab21a
data8 0x3fdaa32f090998e0, 0x3cd715a9353cede4
data8 0x9997dc2e017a9550, 0xe9b33b9ce2bb7638
data8 0x3fdae939540d3f00, 0x3cc545c014943439
data8 0x9a2fa158b29b649b, 0xe9401a573f8aa706
data8 0x3fdb2f65f63f6c60, 0x3cd4a63c2f2ca8e2
data8 0x9aca09f835466186, 0xe8cba69df9f0bf35
data8 0x3fdb75b5773075e0, 0x3cda310ce1b217ec
data8 0x9b672266ab1e0136, 0xe855de74266193d4
data8 0x3fdbbc28606babc0, 0x3cdc84b75cca6c44
data8 0x9c06f7579f0b7bd5, 0xe7debfd2f98c060b
data8 0x3fdc02bf3d843420, 0x3cd225d967ffb922
data8 0x9ca995db058cabdc, 0xe76648a991511c6e
data8 0x3fdc497a9c224780, 0x3cde08101c5b825b
data8 0x9d4f0b605ce71e88, 0xe6ec76dcbc02d9a7
data8 0x3fdc905b0c10d420, 0x3cb1abbaa3edf120
data8 0x9df765b9eecad5e6, 0xe6714846bdda7318
data8 0x3fdcd7611f4b8a00, 0x3cbf6217ae80aadf
data8 0x9ea2b320350540fe, 0xe5f4bab71494cd6b
data8 0x3fdd1e8d6a0d56c0, 0x3cb726e048cc235c
data8 0x9f51023562fc5676, 0xe576cbf239235ecb
data8 0x3fdd65e082df5260, 0x3cd9e66872bd5250
data8 0xa002620915c2a2f6, 0xe4f779b15f5ec5a7
data8 0x3fddad5b02a82420, 0x3c89743b0b57534b
data8 0xa0b6e21c2caf9992, 0xe476c1a233a7873e
data8 0x3fddf4fd84bbe160, 0x3cbf7adea9ee3338
data8 0xa16e9264cc83a6b2, 0xe3f4a16696608191
data8 0x3fde3cc8a6ec6ee0, 0x3cce46f5a51f49c6
data8 0xa22983528f3d8d49, 0xe3711694552da8a8
data8 0x3fde84bd099a6600, 0x3cdc78f6490a2d31
data8 0xa2e7c5d2e2e69460, 0xe2ec1eb4e1e0a5fb
data8 0x3fdeccdb4fc685c0, 0x3cdd3aedb56a4825
data8 0xa3a96b5599bd2532, 0xe265b74506fbe1c9
data8 0x3fdf15241f23b3e0, 0x3cd440f3c6d65f65
data8 0xa46e85d1ae49d7de, 0xe1ddddb499b3606f
data8 0x3fdf5d98202994a0, 0x3cd6c44bd3fb745a
data8 0xa53727ca3e11b99e, 0xe1548f662951b00d
data8 0x3fdfa637fe27bf60, 0x3ca8ad1cd33054dd
data8 0xa6036453bdc20186, 0xe0c9c9aeabe5e481
data8 0x3fdfef0467599580, 0x3cc0f1ac0685d78a
data8 0xa6d34f1969dda338, 0xe03d89d5281e4f81
data8 0x3fe01bff067d6220, 0x3cc0731e8a9ef057
data8 0xa7a6fc62f7246ff3, 0xdfafcd125c323f54
data8 0x3fe04092d1ae3b40, 0x3ccabda24b59906d
data8 0xa87e811a861df9b9, 0xdf20909061bb9760
data8 0x3fe0653df0fd9fc0, 0x3ce94c8dcc722278
data8 0xa959f2d2dd687200, 0xde8fd16a4e5f88bd
data8 0x3fe08a00c1cae320, 0x3ce6b888bb60a274
data8 0xaa3967cdeea58bda, 0xddfd8cabd1240d22
data8 0x3fe0aedba3221c00, 0x3ced5941cd486e46
data8 0xab904fd587263c84, 0xdd1f4472e1cf64ed
data8 0x3fe0e651e85229c0, 0x3cdb6701042299b1
data8 0xad686d44dd5a74bb, 0xdbf173e1f6b46e92
data8 0x3fe1309cbf4cdb20, 0x3cbf1be7bb3f0ec5
data8 0xaf524e15640ebee4, 0xdabd54896f1029f6
data8 0x3fe17b4ee1641300, 0x3ce81dd055b792f1
data8 0xb14eca24ef7db3fa, 0xd982cb9ae2f47e41
data8 0x3fe1c66b9ffd6660, 0x3cd98ea31eb5ddc7
data8 0xb35ec807669920ce, 0xd841bd1b8291d0b6
data8 0x3fe211f66db3a5a0, 0x3ca480c35a27b4a2
data8 0xb5833e4755e04dd1, 0xd6fa0bd3150b6930
data8 0x3fe25df2e05b6c40, 0x3ca4bc324287a351
data8 0xb7bd34c8000b7bd3, 0xd5ab9939a7d23aa1
data8 0x3fe2aa64b32f7780, 0x3cba67314933077c
data8 0xba0dc64d126cc135, 0xd4564563ce924481
data8 0x3fe2f74fc9289ac0, 0x3cec1a1dc0efc5ec
data8 0xbc76222cbbfa74a6, 0xd2f9eeed501125a8
data8 0x3fe344b82f859ac0, 0x3ceeef218de413ac
data8 0xbef78e31985291a9, 0xd19672e2182f78be
data8 0x3fe392a22087b7e0, 0x3cd2619ba201204c
data8 0xc19368b2b0629572, 0xd02baca5427e436a
data8 0x3fe3e11206694520, 0x3cb5d0b3143fe689
data8 0xc44b2ae8c6733e51, 0xceb975d60b6eae5d
data8 0x3fe4300c7e945020, 0x3cbd367143da6582
data8 0xc7206b894212dfef, 0xcd3fa6326ff0ac9a
data8 0x3fe47f965d201d60, 0x3ce797c7a4ec1d63
data8 0xca14e1b0622de526, 0xcbbe13773c3c5338
data8 0x3fe4cfb4b09d1a20, 0x3cedfadb5347143c
data8 0xcd2a6825eae65f82, 0xca34913d425a5ae9
data8 0x3fe5206cc637e000, 0x3ce2798b38e54193
data8 0xd06301095e1351ee, 0xc8a2f0d3679c08c0
data8 0x3fe571c42e3d0be0, 0x3ccd7cb9c6c2ca68
data8 0xd3c0d9f50057adda, 0xc70901152d59d16b
data8 0x3fe5c3c0c108f940, 0x3ceb6c13563180ab
data8 0xd74650a98cc14789, 0xc5668e3d4cbf8828
data8 0x3fe61668a46ffa80, 0x3caa9092e9e3c0e5
data8 0xdaf5f8579dcc8f8f, 0xc3bb61b3eed42d02
data8 0x3fe669c251ad69e0, 0x3cccf896ef3b4fee
data8 0xded29f9f9a6171b4, 0xc20741d7f8e8e8af
data8 0x3fe6bdd49bea05c0, 0x3cdc6b29937c575d
data8 0xe2df5765854ccdb0, 0xc049f1c2d1b8014b
data8 0x3fe712a6b76c6e80, 0x3ce1ddc6f2922321
data8 0xe71f7a9b94fcb4c3, 0xbe833105ec291e91
data8 0x3fe76840418978a0, 0x3ccda46e85432c3d
data8 0xeb96b72d3374b91e, 0xbcb2bb61493b28b3
data8 0x3fe7bea9496d5a40, 0x3ce37b42ec6e17d3
data8 0xf049183c3f53c39b, 0xbad848720223d3a8
data8 0x3fe815ea59dab0a0, 0x3cb03ad41bfc415b
data8 0xf53b11ec7f415f15, 0xb8f38b57c53c9c48
data8 0x3fe86e0c84010760, 0x3cc03bfcfb17fe1f
data8 0xfa718f05adbf2c33, 0xb70432500286b185
data8 0x3fe8c7196b9225c0, 0x3ced99fcc6866ba9
data8 0xfff200c3f5489608, 0xb509e6454dca33cc
data8 0x3fe9211b54441080, 0x3cb789cb53515688
// The following table entries are not used
//data8 0x82e138a0fac48700, 0xb3044a513a8e6132
//data8 0x3fe97c1d30f5b7c0, 0x3ce1eb765612d1d0
//data8 0x85f4cc7fc670d021, 0xb0f2fb2ea6cbbc88
//data8 0x3fe9d82ab4b5fde0, 0x3ced3fe6f27e8039
//data8 0x89377c1387d5b908, 0xaed58e9a09014d5c
//data8 0x3fea355065f87fa0, 0x3cbef481d25f5b58
//data8 0x8cad7a2c98dec333, 0xacab929ce114d451
//data8 0x3fea939bb451e2a0, 0x3c8e92b4fbf4560f
//data8 0x905b7dfc99583025, 0xaa748cc0dbbbc0ec
//data8 0x3feaf31b11270220, 0x3cdced8c61bd7bd5
//data8 0x9446d8191f80dd42, 0xa82ff92687235baf
//data8 0x3feb53de0bcffc20, 0x3cbe1722fb47509e
//data8 0x98758ba086e4000a, 0xa5dd497a9c184f58
//data8 0x3febb5f571cb0560, 0x3ce0c7774329a613
//data8 0x9cee6c7bf18e4e24, 0xa37be3c3cd1de51b
//data8 0x3fec197373bc7be0, 0x3ce08ebdb55c3177
//data8 0xa1b944000a1b9440, 0xa10b2101b4f27e03
//data8 0x3fec7e6bd023da60, 0x3ce5fc5fd4995959
//data8 0xa6defd8ba04d3e38, 0x9e8a4b93cad088ec
//data8 0x3fece4f404e29b20, 0x3cea3413401132b5
//data8 0xac69dd408a10c62d, 0x9bf89d5d17ddae8c
//data8 0x3fed4d2388f63600, 0x3cd5a7fb0d1d4276
//data8 0xb265c39cbd80f97a, 0x99553d969fec7beb
//data8 0x3fedb714101e0a00, 0x3cdbda21f01193f2
//data8 0xb8e081a16ae4ae73, 0x969f3e3ed2a0516c
//data8 0x3fee22e1da97bb00, 0x3ce7231177f85f71
//data8 0xbfea427678945732, 0x93d5990f9ee787af
//data8 0x3fee90ac13b18220, 0x3ce3c8a5453363a5
//data8 0xc79611399b8c90c5, 0x90f72bde80febc31
//data8 0x3fef009542b712e0, 0x3ce218fd79e8cb56
//data8 0xcffa8425040624d7, 0x8e02b4418574ebed
//data8 0x3fef72c3d2c57520, 0x3cd32a717f82203f
//data8 0xd93299cddcf9cf23, 0x8af6ca48e9c44024
//data8 0x3fefe762b77744c0, 0x3ce53478a6bbcf94
//data8 0xe35eda760af69ad9, 0x87d1da0d7f45678b
//data8 0x3ff02f511b223c00, 0x3ced6e11782c28fc
//data8 0xeea6d733421da0a6, 0x84921bbe64ae029a
//data8 0x3ff06c5c6f8ce9c0, 0x3ce71fc71c1ffc02
//data8 0xfb3b2c73fc6195cc, 0x813589ba3a5651b6
//data8 0x3ff0aaf2613700a0, 0x3cf2a72d2fd94ef3
//data8 0x84ac1fcec4203245, 0xfb73a828893df19e
//data8 0x3ff0eb367c3fd600, 0x3cf8054c158610de
//data8 0x8ca50621110c60e6, 0xf438a14c158d867c
//data8 0x3ff12d51caa6b580, 0x3ce6bce9748739b6
//data8 0x95b8c2062d6f8161, 0xecb3ccdd37b369da
//data8 0x3ff1717418520340, 0x3ca5c2732533177c
//data8 0xa0262917caab4ad1, 0xe4dde4ddc81fd119
//data8 0x3ff1b7d59dd40ba0, 0x3cc4c7c98e870ff5
//data8 0xac402c688b72f3f4, 0xdcae469be46d4c8d
//data8 0x3ff200b93cc5a540, 0x3c8dd6dc1bfe865a
//data8 0xba76968b9eabd9ab, 0xd41a8f3df1115f7f
//data8 0x3ff24c6f8f6affa0, 0x3cf1acb6d2a7eff7
//data8 0xcb63c87c23a71dc5, 0xcb161074c17f54ec
//data8 0x3ff29b5b338b7c80, 0x3ce9b5845f6ec746
//data8 0xdfe323b8653af367, 0xc19107d99ab27e42
//data8 0x3ff2edf6fac7f5a0, 0x3cf77f961925fa02
//data8 0xf93746caaba3e1f1, 0xb777744a9df03bff
//data8 0x3ff344df237486c0, 0x3cf6ddf5f6ddda43
//data8 0x8ca77052f6c340f0, 0xacaf476f13806648
//data8 0x3ff3a0dfa4bb4ae0, 0x3cfee01bbd761bff
//data8 0xa1a48604a81d5c62, 0xa11575d30c0aae50
//data8 0x3ff4030b73c55360, 0x3cf1cf0e0324d37c
//data8 0xbe45074b05579024, 0x9478e362a07dd287
//data8 0x3ff46ce4c738c4e0, 0x3ce3179555367d12
//data8 0xe7a08b5693d214ec, 0x8690e3575b8a7c3b
//data8 0x3ff4e0a887c40a80, 0x3cfbd5d46bfefe69
//data8 0x94503d69396d91c7, 0xedd2ce885ff04028
//data8 0x3ff561ebd9c18cc0, 0x3cf331bd176b233b
//data8 0xced1d96c5bb209e6, 0xc965278083808702
//data8 0x3ff5f71d7ff42c80, 0x3ce3301cc0b5a48c
//data8 0xabac2cee0fc24e20, 0x9c4eb1136094cbbd
//data8 0x3ff6ae4c63222720, 0x3cf5ff46874ee51e
//data8 0x8040201008040201, 0xb4d7ac4d9acb1bf4
//data8 0x3ff7b7d33b928c40, 0x3cfacdee584023bb
LOCAL_OBJECT_END(T_table)



.align 16

LOCAL_OBJECT_START(poly_coeffs)
       // C_3
data8 0xaaaaaaaaaaaaaaab, 0x0000000000003ffc
       // C_5
data8 0x999999999999999a, 0x0000000000003ffb
       // C_7, C_9
data8 0x3fa6db6db6db6db7, 0x3f9f1c71c71c71c8
       // pi/2 (low, high)
data8 0x3C91A62633145C07, 0x3FF921FB54442D18
       // C_11, C_13
data8 0x3f96e8ba2e8ba2e9, 0x3f91c4ec4ec4ec4e
       // C_15, C_17
data8 0x3f8c99999999999a, 0x3f87a87878787223
       // pi (low, high)
data8 0x3CA1A62633145C07, 0x400921FB54442D18
LOCAL_OBJECT_END(poly_coeffs)


R_DBL_S = r21
R_EXP0 = r22
R_EXP = r15
R_SGNMASK = r23
R_TMP = r24
R_TMP2 = r25
R_INDEX = r26
R_TMP3 = r27
R_TMP03 = r27
R_TMP4 = r28
R_TMP5 = r23
R_TMP6 = r22
R_TMP7 = r21
R_T = r29
R_BIAS = r20

F_T = f6
F_1S2 = f7
F_1S2_S = f9
F_INV_1T2 = f10
F_SQRT_1T2 = f11
F_S2T2 = f12
F_X = f13
F_D = f14
F_2M64 = f15

F_CS2 = f32
F_CS3 = f33
F_CS4 = f34
F_CS5 = f35
F_CS6 = f36
F_CS7 = f37
F_CS8 = f38
F_CS9 = f39
F_S23 = f40
F_S45 = f41
F_S67 = f42
F_S89 = f43
F_S25 = f44
F_S69 = f45
F_S29 = f46
F_X2 = f47
F_X4 = f48
F_TSQRT = f49
F_DTX = f50
F_R = f51
F_R2 = f52
F_R3 = f53
F_R4 = f54

F_C3 = f55
F_C5 = f56
F_C7 = f57
F_C9 = f58
F_P79 = f59
F_P35 = f60
F_P39 = f61

F_ATHI = f62
F_ATLO = f63

F_T1 = f64
F_Y = f65
F_Y2 = f66
F_ANDMASK = f67
F_ORMASK = f68
F_S = f69
F_05 = f70
F_SQRT_1S2 = f71
F_DS = f72
F_Z = f73
F_1T2 = f74
F_DZ = f75
F_ZE = f76
F_YZ = f77
F_Y1S2 = f78
F_Y1S2X = f79
F_1X = f80
F_ST = f81
F_1T2_ST = f82
F_TSS = f83
F_Y1S2X2 = f84
F_DZ_TERM = f85
F_DTS = f86
F_DS2X = f87
F_T2 = f88
F_ZY1S2S = f89
F_Y1S2_1X = f90
F_TS = f91
F_PI2_LO = f92
F_PI2_HI = f93
F_S19 = f94
F_INV1T2_2 = f95
F_CORR = f96
F_DZ0 = f97

F_C11 = f98
F_C13 = f99
F_C15 = f100
F_C17 = f101
F_P1113 = f102
F_P1517 = f103
F_P1117 = f104
F_P317 = f105
F_R8 = f106
F_HI = f107
F_1S2_HI = f108
F_DS2 = f109
F_Y2_2 = f110
//F_S2 = f111
//F_S_DS2 = f112
F_S_1S2S = f113
F_XL = f114
F_2M128 = f115
F_1AS = f116
F_AS = f117



.section .text
GLOBAL_LIBM_ENTRY(acosl)

{.mfi
       // get exponent, mantissa (rounded to double precision) of s
       getf.d R_DBL_S = f8
       // 1-s^2
       fnma.s1 F_1S2 = f8, f8, f1
       // r2 = pointer to T_table
       addl r2 = @ltoff(T_table), gp
}

{.mfi
       // sign mask
       mov R_SGNMASK = 0x20000
       nop.f 0
       // bias-63-1
       mov R_TMP03 = 0xffff-64;;
}


{.mfi
       // get exponent of s
       getf.exp R_EXP = f8
       nop.f 0
       // R_TMP4 = 2^45
       shl R_TMP4 = R_SGNMASK, 45-17
}

{.mlx
       // load bias-4
       mov R_TMP = 0xffff-4
       // load RU(sqrt(2)/2) to integer register (in double format, shifted left by 1)
       movl R_TMP2 = 0x7fcd413cccfe779a;;
}


{.mfi
       // load 2^{-64} in FP register
       setf.exp F_2M64 = R_TMP03
       nop.f 0
       // index = (0x7-exponent)|b1 b2.. b6
       extr.u R_INDEX = R_DBL_S, 46, 9
}

{.mfi
       // get t = sign|exponent|b1 b2.. b6 1 x.. x
       or R_T = R_DBL_S, R_TMP4
       nop.f 0
       // R_TMP4 = 2^45-1
       sub R_TMP4 = R_TMP4, r0, 1;;
}


{.mfi
       // get t = sign|exponent|b1 b2.. b6 1 0.. 0
       andcm R_T = R_T, R_TMP4
       nop.f 0
       // eliminate sign from R_DBL_S (shift left by 1)
       shl R_TMP3 = R_DBL_S, 1
}

{.mfi
       // R_BIAS = 3*2^6
       mov R_BIAS = 0xc0
       nop.f 0
       // eliminate sign from R_EXP
       andcm R_EXP0 = R_EXP, R_SGNMASK;;
}



{.mfi
       // load start address for T_table
       ld8 r2 = [r2]
       nop.f 0
       // p8 = 1 if |s|> = sqrt(2)/2
       cmp.geu p8, p0 = R_TMP3, R_TMP2
}

{.mlx
       // p7 = 1 if |s|<2^{-4} (exponent of s<bias-4)
       cmp.lt p7, p0 = R_EXP0, R_TMP
       // sqrt coefficient cs8 = -33*13/128
       movl R_TMP2 = 0xc0568000;;
}



{.mbb
       // load t in FP register
       setf.d F_T = R_T
       // if |s|<2^{-4}, take alternate path
 (p7) br.cond.spnt SMALL_S
       // if |s|> = sqrt(2)/2, take alternate path
 (p8) br.cond.sptk LARGE_S
}

{.mlx
       // index = (4-exponent)|b1 b2.. b6
       sub R_INDEX = R_INDEX, R_BIAS
       // sqrt coefficient cs9 = 55*13/128
       movl R_TMP = 0x40b2c000;;
}


{.mfi
       // sqrt coefficient cs8 = -33*13/128
       setf.s F_CS8 = R_TMP2
       nop.f 0
       // shift R_INDEX by 5
       shl R_INDEX = R_INDEX, 5
}

{.mfi
       // sqrt coefficient cs3 = 0.5 (set exponent = bias-1)
       mov R_TMP4 = 0xffff - 1
       nop.f 0
       // sqrt coefficient cs6 = -21/16
       mov R_TMP6 = 0xbfa8;;
}


{.mlx
       // table index
       add r2 = r2, R_INDEX
       // sqrt coefficient cs7 = 33/16
       movl R_TMP2 = 0x40040000;;
}


{.mmi
       // load cs9 = 55*13/128
       setf.s F_CS9 = R_TMP
       // sqrt coefficient cs5 = 7/8
       mov R_TMP3 = 0x3f60
       // sqrt coefficient cs6 = 21/16
       shl R_TMP6 = R_TMP6, 16;;
}


{.mmi
       // load significand of 1/(1-t^2)
       ldf8 F_INV_1T2 = [r2], 8
       // sqrt coefficient cs7 = 33/16
       setf.s F_CS7 = R_TMP2
       // sqrt coefficient cs4 = -5/8
       mov R_TMP5 = 0xbf20;;
}


{.mmi
       // load significand of sqrt(1-t^2)
       ldf8 F_SQRT_1T2 = [r2], 8
       // sqrt coefficient cs6 = 21/16
       setf.s F_CS6 = R_TMP6
       // sqrt coefficient cs5 = 7/8
       shl R_TMP3 = R_TMP3, 16;;
}


{.mmi
       // sqrt coefficient cs3 = 0.5 (set exponent = bias-1)
       setf.exp F_CS3 = R_TMP4
       // r3 = pointer to polynomial coefficients
       addl r3 = @ltoff(poly_coeffs), gp
       // sqrt coefficient cs4 = -5/8
       shl R_TMP5 = R_TMP5, 16;;
}


{.mfi
       // sqrt coefficient cs5 = 7/8
       setf.s F_CS5 = R_TMP3
       // d = s-t
       fms.s1 F_D = f8, f1, F_T
       // set p6 = 1 if s<0, p11 = 1 if s> = 0
       cmp.ge p6, p11 = R_EXP, R_DBL_S
}

{.mfi
       // r3 = load start address to polynomial coefficients
       ld8 r3 = [r3]
       // s+t
       fma.s1 F_S2T2 = f8, f1, F_T
       nop.i 0;;
}


{.mfi
       // sqrt coefficient cs4 = -5/8
       setf.s F_CS4 = R_TMP5
       // s^2-t^2
       fma.s1 F_S2T2 = F_S2T2, F_D, f0
       nop.i 0;;
}


{.mfi
       // load C3
       ldfe F_C3 = [r3], 16
       // 0.5/(1-t^2) = 2^{-64}*(2^63/(1-t^2))
       fma.s1 F_INV_1T2 = F_INV_1T2, F_2M64, f0
       nop.i 0;;
}

{.mfi
       // load C_5
       ldfe F_C5 = [r3], 16
       // set correct exponent for sqrt(1-t^2)
       fma.s1 F_SQRT_1T2 = F_SQRT_1T2, F_2M64, f0
       nop.i 0;;
}


{.mfi
       // load C_7, C_9
       ldfpd F_C7, F_C9 = [r3], 16
       // x = -(s^2-t^2)/(1-t^2)/2
       fnma.s1 F_X = F_INV_1T2, F_S2T2, f0
       nop.i 0;;
}


{.mmf
       // load asin(t)_high, asin(t)_low
       ldfpd F_ATHI, F_ATLO = [r2]
	   // load pi/2
	   ldfpd F_PI2_LO, F_PI2_HI = [r3]
       // t*sqrt(1-t^2)
       fma.s1 F_TSQRT = F_T, F_SQRT_1T2, f0;;
}


{.mfi
       nop.m 0
       // cs9*x+cs8
       fma.s1 F_S89 = F_CS9, F_X, F_CS8
       nop.i 0
}

{.mfi
       nop.m 0
       // cs7*x+cs6
       fma.s1 F_S67 = F_CS7, F_X, F_CS6
       nop.i 0;;
}

{.mfi
       nop.m 0
       // cs5*x+cs4
       fma.s1 F_S45 = F_CS5, F_X, F_CS4
       nop.i 0
}

{.mfi
       nop.m 0
       // x*x
       fma.s1 F_X2 = F_X, F_X, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (s-t)-t*x
       fnma.s1 F_DTX = F_T, F_X, F_D
       nop.i 0
}

{.mfi
       nop.m 0
       // cs3*x+cs2 (cs2 = -0.5 = -cs3)
       fms.s1 F_S23 = F_CS3, F_X, F_CS3
       nop.i 0;;
}

{.mfi
  nop.m 0
  // if sign is negative, negate table values: asin(t)_low
  (p6) fnma.s1 F_ATLO = F_ATLO, f1, f0
  nop.i 0
}

{.mfi
  nop.m 0
  // if sign is negative, negate table values: asin(t)_high
  (p6) fnma.s1 F_ATHI = F_ATHI, f1, f0
  nop.i 0;;
}


{.mfi
       nop.m 0
       // cs9*x^3+cs8*x^2+cs7*x+cs6
       fma.s1 F_S69 = F_S89, F_X2, F_S67
       nop.i 0
}

{.mfi
       nop.m 0
       // x^4
       fma.s1 F_X4 = F_X2, F_X2, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // t*sqrt(1-t^2)*x^2
       fma.s1 F_TSQRT = F_TSQRT, F_X2, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // cs5*x^3+cs4*x^2+cs3*x+cs2
       fma.s1 F_S25 = F_S45, F_X2, F_S23
       nop.i 0;;
}


{.mfi
       nop.m 0
       // ((s-t)-t*x)*sqrt(1-t^2)
       fma.s1 F_DTX = F_DTX, F_SQRT_1T2, f0
       nop.i 0;;
}

{.mfi
       nop.m 0
       // (pi/2)_high - asin(t)_high
       fnma.s1 F_ATHI = F_ATHI, f1, F_PI2_HI
       nop.i 0
}

{.mfi
       nop.m 0
       // asin(t)_low - (pi/2)_low
       fnma.s1 F_ATLO = F_PI2_LO, f1, F_ATLO
	   nop.i 0;;
}


{.mfi
       nop.m 0
       // PS29 = cs9*x^7+..+cs5*x^3+cs4*x^2+cs3*x+cs2
       fma.s1 F_S29 = F_S69, F_X4, F_S25
       nop.i 0;;
}



{.mfi
       nop.m 0
       // R = ((s-t)-t*x)*sqrt(1-t^2)-t*sqrt(1-t^2)*x^2*PS29
       fnma.s1 F_R = F_S29, F_TSQRT, F_DTX
       nop.i 0;;
}


{.mfi
       nop.m 0
       // R^2
       fma.s1 F_R2 = F_R, F_R, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c7+c9*R^2
       fma.s1 F_P79 = F_C9, F_R2, F_C7
       nop.i 0
}

{.mfi
       nop.m 0
       // c3+c5*R^2
       fma.s1 F_P35 = F_C5, F_R2, F_C3
       nop.i 0;;
}

{.mfi
       nop.m 0
       // R^3
       fma.s1 F_R4 = F_R2, F_R2, f0
       nop.i 0;;
}

{.mfi
       nop.m 0
       // R^3
       fma.s1 F_R3 = F_R2, F_R, f0
       nop.i 0;;
}



{.mfi
       nop.m 0
       // c3+c5*R^2+c7*R^4+c9*R^6
       fma.s1 F_P39 = F_P79, F_R4, F_P35
       nop.i 0;;
}


{.mfi
       nop.m 0
       // asin(t)_low+R^3*(c3+c5*R^2+c7*R^4+c9*R^6)
       fma.s1 F_P39 = F_P39, F_R3, F_ATLO
       nop.i 0;;
}


{.mfi
       nop.m 0
       // R+asin(t)_low+R^3*(c3+c5*R^2+c7*R^4+c9*R^6)
       fma.s1 F_P39 = F_P39, f1, F_R
       nop.i 0;;
}


{.mfb
       nop.m 0
       // result = (pi/2)-asin(t)_high+R+asin(t)_low+R^3*(c3+c5*R^2+c7*R^4+c9*R^6)
       fnma.s0 f8 = F_P39, f1, F_ATHI
       // return
       br.ret.sptk b0;;
}




LARGE_S:

{.mfi
       // bias-1
       mov R_TMP3 = 0xffff - 1
       // y ~ 1/sqrt(1-s^2)
       frsqrta.s1 F_Y, p7 = F_1S2
       // c9 = 55*13*17/128
       mov R_TMP4 = 0x10af7b
}

{.mlx
       // c8 = -33*13*15/128
       mov R_TMP5 = 0x184923
       movl R_TMP2 = 0xff00000000000000;;
}

{.mfi
       // set p6 = 1 if s<0, p11 = 1 if s>0
       cmp.ge p6, p11 = R_EXP, R_DBL_S
       // 1-s^2
       fnma.s1 F_1S2 = f8, f8, f1
       // set p9 = 1
       cmp.eq p9, p0 = r0, r0;;
}


{.mfi
       // load 0.5
       setf.exp F_05 = R_TMP3
       // (1-s^2) rounded to single precision
       fnma.s.s1 F_1S2_S = f8, f8, f1
       // c9 = 55*13*17/128
       shl R_TMP4 = R_TMP4, 10
}

{.mlx
       // AND mask for getting t ~ sqrt(1-s^2)
       setf.sig F_ANDMASK = R_TMP2
       // OR mask
       movl R_TMP2 = 0x0100000000000000;;
}

.pred.rel "mutex", p6, p11
{.mfi
       nop.m 0
	   // 1-|s|
 (p6)  fma.s1 F_1AS = f8, f1, f1
       nop.i 0
}

{.mfi
       nop.m 0
       // 1-|s|
 (p11) fnma.s1 F_1AS = f8, f1, f1
       nop.i 0;;
}


{.mfi
       // c9 = 55*13*17/128
       setf.s F_CS9 = R_TMP4
	   // |s|
 (p6)  fnma.s1 F_AS = f8, f1, f0
       // c8 = -33*13*15/128
       shl R_TMP5 = R_TMP5, 11
}

{.mfi
       // c7 = 33*13/16
       mov R_TMP4 = 0x41d68
	   // |s|
 (p11) fma.s1 F_AS = f8, f1, f0
       nop.i 0;;
}


{.mfi
       setf.sig F_ORMASK = R_TMP2
       // y^2
       fma.s1 F_Y2 = F_Y, F_Y, f0
       // c7 = 33*13/16
       shl R_TMP4 = R_TMP4, 12
}

{.mfi
       // c6 = -33*7/16
       mov R_TMP6 = 0xc1670
       // y' ~ sqrt(1-s^2)
       fma.s1 F_T1 = F_Y, F_1S2, f0
       // c5 = 63/8
       mov R_TMP7 = 0x40fc;;
}


{.mlx
       // load c8 = -33*13*15/128
       setf.s F_CS8 = R_TMP5
       // c4 = -35/8
       movl R_TMP5 = 0xc08c0000;;
}

{.mfi
       // r3 = pointer to polynomial coefficients
       addl r3 = @ltoff(poly_coeffs), gp
       // 1-s-(1-s^2)_s
       fnma.s1 F_DS = F_1S2_S, f1, F_1AS
       // p9 = 0 if p7 = 1 (p9 = 1 for special cases only)
 (p7) cmp.ne p9, p0 = r0, r0
}

{.mlx
       // load c7 = 33*13/16
       setf.s F_CS7 = R_TMP4
       // c3 = 5/2
       movl R_TMP4 = 0x40200000;;
}


{.mlx
       // load c4 = -35/8
       setf.s F_CS4 = R_TMP5
       // c2 = -3/2
       movl R_TMP5 = 0xbfc00000;;
}


{.mfi
       // load c3 = 5/2
       setf.s F_CS3 = R_TMP4
       // x = (1-s^2)_s*y^2-1
       fms.s1 F_X = F_1S2_S, F_Y2, f1
       // c6 = -33*7/16
       shl R_TMP6 = R_TMP6, 12
}

{.mfi
       nop.m 0
       // y^2/2
       fma.s1 F_Y2_2 = F_Y2, F_05, f0
       nop.i 0;;
}


{.mfi
       // load c6 = -33*7/16
       setf.s F_CS6 = R_TMP6
       // eliminate lower bits from y'
       fand F_T = F_T1, F_ANDMASK
       // c5 = 63/8
       shl R_TMP7 = R_TMP7, 16
}


{.mfb
       // r3 = load start address to polynomial coefficients
       ld8 r3 = [r3]
       // 1-(1-s^2)_s-s^2
       fma.s1 F_DS = F_AS, F_1AS, F_DS
       // p9 = 1 if s is a special input (NaN, or |s|> = 1)
 (p9) br.cond.spnt acosl_SPECIAL_CASES;;
}

{.mmf
       // get exponent, significand of y' (in single prec.)
       getf.s R_TMP = F_T1
       // load c3 = -3/2
       setf.s F_CS2 = R_TMP5
       // y*(1-s^2)
       fma.s1 F_Y1S2 = F_Y, F_1S2, f0;;
}



{.mfi
       nop.m 0
       // if s<0, set s = -s
 (p6) fnma.s1 f8 = f8, f1, f0
       nop.i 0;;
}


{.mfi
       // load c5 = 63/8
       setf.s F_CS5 = R_TMP7
       // x = (1-s^2)_s*y^2-1+(1-(1-s^2)_s-s^2)*y^2
       fma.s1 F_X = F_DS, F_Y2, F_X
       // for t = 2^k*1.b1 b2.., get 7-k|b1.. b6
       extr.u R_INDEX = R_TMP, 17, 9;;
}


{.mmi
       // index = (4-exponent)|b1 b2.. b6
       sub R_INDEX = R_INDEX, R_BIAS
       nop.m 0
       // get exponent of y
       shr.u R_TMP2 = R_TMP, 23;;
}

{.mmi
       // load C3
       ldfe F_C3 = [r3], 16
       // set p8 = 1 if y'<2^{-4}
       cmp.gt p8, p0 = 0x7b, R_TMP2
       // shift R_INDEX by 5
       shl R_INDEX = R_INDEX, 5;;
}


{.mfb
       // get table index for sqrt(1-t^2)
       add r2 = r2, R_INDEX
       // get t = 2^k*1.b1 b2.. b7 1
       for F_T = F_T, F_ORMASK
 (p8) br.cond.spnt VERY_LARGE_INPUT;;
}



{.mmf
       // load C5
       ldfe F_C5 = [r3], 16
       // load 1/(1-t^2)
       ldfp8 F_INV_1T2, F_SQRT_1T2 = [r2], 16
       // x = ((1-s^2)*y^2-1)/2
       fma.s1 F_X = F_X, F_05, f0;;
}



{.mmf
       nop.m 0
       // C7, C9
       ldfpd F_C7, F_C9 = [r3], 16
       // set correct exponent for t
       fmerge.se F_T = F_T1, F_T;;
}



{.mfi
       // get address for loading pi
	   add r3 = 48, r3
       // c9*x+c8
       fma.s1 F_S89 = F_X, F_CS9, F_CS8
       nop.i 0
}

{.mfi
       nop.m 0
       // x^2
       fma.s1 F_X2 = F_X, F_X, f0
       nop.i 0;;
}


{.mfi
       // pi (low, high)
       ldfpd F_PI2_LO, F_PI2_HI = [r3]
       // y*(1-s^2)*x
       fma.s1 F_Y1S2X = F_Y1S2, F_X, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // c7*x+c6
       fma.s1 F_S67 = F_X, F_CS7, F_CS6
       nop.i 0;;
}


{.mfi
       nop.m 0
       // 1-x
       fnma.s1 F_1X = F_X, f1, f1
       nop.i 0
}

{.mfi
       nop.m 0
       // c3*x+c2
       fma.s1 F_S23 = F_X, F_CS3, F_CS2
       nop.i 0;;
}


{.mfi
       nop.m 0
       // 1-t^2
       fnma.s1 F_1T2 = F_T, F_T, f1
       nop.i 0
}

{.mfi
       // load asin(t)_high, asin(t)_low
       ldfpd F_ATHI, F_ATLO = [r2]
       // c5*x+c4
       fma.s1 F_S45 = F_X, F_CS5, F_CS4
       nop.i 0;;
}



{.mfi
       nop.m 0
       // t*s
       fma.s1 F_TS = F_T, f8, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // 0.5/(1-t^2)
       fma.s1 F_INV_1T2 = F_INV_1T2, F_2M64, f0
       nop.i 0;;
}

{.mfi
       nop.m 0
       // z~sqrt(1-t^2), rounded to 24 significant bits
       fma.s.s1 F_Z = F_SQRT_1T2, F_2M64, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // sqrt(1-t^2)
       fma.s1 F_SQRT_1T2 = F_SQRT_1T2, F_2M64, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // y*(1-s^2)*x^2
       fma.s1 F_Y1S2X2 = F_Y1S2, F_X2, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // x^4
       fma.s1 F_X4 = F_X2, F_X2, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // s*t rounded to 24 significant bits
       fma.s.s1 F_TSS = F_T, f8, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // c9*x^3+..+c6
       fma.s1 F_S69 = F_X2, F_S89, F_S67
       nop.i 0;;
}


{.mfi
       nop.m 0
       // ST = (t^2-1+s^2) rounded to 24 significant bits
       fms.s.s1 F_ST = f8, f8, F_1T2
       nop.i 0
}

{.mfi
       nop.m 0
       // c5*x^3+..+c2
       fma.s1 F_S25 = F_X2, F_S45, F_S23
       nop.i 0;;
}


{.mfi
       nop.m 0
       // 0.25/(1-t^2)
       fma.s1 F_INV1T2_2 = F_05, F_INV_1T2, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // t*s-sqrt(1-t^2)*(1-s^2)*y
       fnma.s1 F_TS = F_Y1S2, F_SQRT_1T2, F_TS
       nop.i 0;;
}


{.mfi
       nop.m 0
       // z*0.5/(1-t^2)
       fma.s1 F_ZE = F_INV_1T2, F_SQRT_1T2, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // z^2+t^2-1
       fms.s1 F_DZ0 = F_Z, F_Z, F_1T2
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (1-s^2-(1-s^2)_s)*x
       fma.s1 F_DS2X = F_X, F_DS, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // t*s-(t*s)_s
       fms.s1 F_DTS = F_T, f8, F_TSS
       nop.i 0
}

{.mfi
       nop.m 0
       // c9*x^7+..+c2
       fma.s1 F_S29 = F_X4, F_S69, F_S25
       nop.i 0;;
}


{.mfi
       nop.m 0
       // y*z
       fma.s1 F_YZ = F_Z, F_Y, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // t^2
       fma.s1 F_T2 = F_T, F_T, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // 1-t^2+ST
       fma.s1 F_1T2_ST = F_ST, f1, F_1T2
       nop.i 0;;
}


{.mfi
       nop.m 0
       // y*(1-s^2)(1-x)
       fma.s1 F_Y1S2_1X = F_Y1S2, F_1X, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // dz ~ sqrt(1-t^2)-z
       fma.s1 F_DZ = F_DZ0, F_ZE, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // -1+correction for sqrt(1-t^2)-z
       fnma.s1 F_CORR = F_INV1T2_2, F_DZ0, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (PS29*x^2+x)*y*(1-s^2)
       fma.s1 F_S19 = F_Y1S2X2, F_S29, F_Y1S2X
       nop.i 0;;
}

{.mfi
       nop.m 0
       // z*y*(1-s^2)_s
       fma.s1 F_ZY1S2S = F_YZ, F_1S2_S, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // s^2-(1-t^2+ST)
       fms.s1 F_1T2_ST = f8, f8, F_1T2_ST
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (t*s-(t*s)_s)+z*y*(1-s^2-(1-s^2)_s)*x
       fma.s1 F_DTS = F_YZ, F_DS2X, F_DTS
       nop.i 0
}

{.mfi
       nop.m 0
       // dz*y*(1-s^2)*(1-x)
       fma.s1 F_DZ_TERM = F_DZ, F_Y1S2_1X, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // R = t*s-sqrt(1-t^2)*(1-s^2)*y+sqrt(1-t^2)*(1-s^2)*y*PS19
       // (used for polynomial evaluation)
       fma.s1 F_R = F_S19, F_SQRT_1T2, F_TS
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (PS29*x^2)*y*(1-s^2)
       fma.s1 F_S29 = F_Y1S2X2, F_S29, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // apply correction to dz*y*(1-s^2)*(1-x)
       fma.s1 F_DZ_TERM = F_DZ_TERM, F_CORR, F_DZ_TERM
       nop.i 0;;
}


{.mfi
       nop.m 0
       // R^2
       fma.s1 F_R2 = F_R, F_R, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (t*s-(t*s)_s)+z*y*(1-s^2-(1-s^2)_s)*x+dz*y*(1-s^2)*(1-x)
       fma.s1 F_DZ_TERM = F_DZ_TERM, f1, F_DTS
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c7+c9*R^2
       fma.s1 F_P79 = F_C9, F_R2, F_C7
       nop.i 0
}

{.mfi
       nop.m 0
       // c3+c5*R^2
       fma.s1 F_P35 = F_C5, F_R2, F_C3
       nop.i 0;;
}

{.mfi
       nop.m 0
       // asin(t)_low-(pi)_low (if s<0)
 (p6)  fms.s1 F_ATLO = F_ATLO, f1, F_PI2_LO
       nop.i 0
}

{.mfi
       nop.m 0
       // R^4
       fma.s1 F_R4 = F_R2, F_R2, f0
       nop.i 0;;
}

{.mfi
       nop.m 0
       // R^3
       fma.s1 F_R3 = F_R2, F_R, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (t*s)_s-t^2*y*z
       fnma.s1 F_TSS = F_T2, F_YZ, F_TSS
       nop.i 0
}

{.mfi
       nop.m 0
       // d(ts)+z*y*d(1-s^2)*x+dz*y*(1-s^2)*(1-x)+z*y*(s^2-1+t^2-ST)
       fma.s1 F_DZ_TERM = F_YZ, F_1T2_ST, F_DZ_TERM
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (pi)_hi-asin(t)_hi (if s<0)
 (p6)  fms.s1 F_ATHI = F_PI2_HI, f1, F_ATHI
       nop.i 0
}

{.mfi
       nop.m 0
       // c3+c5*R^2+c7*R^4+c9*R^6
       fma.s1 F_P39 = F_P79, F_R4, F_P35
       nop.i 0;;
}


{.mfi
       nop.m 0
       // d(ts)+z*y*d(1-s^2)*x+dz*y*(1-s^2)*(1-x)+z*y*(s^2-1+t^2-ST)+
       // + sqrt(1-t^2)*y*(1-s^2)*x^2*PS29
       fma.s1 F_DZ_TERM = F_SQRT_1T2, F_S29, F_DZ_TERM
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (t*s)_s-t^2*y*z+z*y*ST
       fma.s1 F_TSS = F_YZ, F_ST, F_TSS
       nop.i 0
}

{.mfi
       nop.m 0
       // -asin(t)_low+R^3*(c3+c5*R^2+c7*R^4+c9*R^6)
       fms.s1 F_P39 = F_P39, F_R3, F_ATLO
       nop.i 0;;
}


{.mfi
       nop.m 0
       // d(ts)+z*y*d(1-s^2)*x+dz*y*(1-s^2)*(1-x)+z*y*(s^2-1+t^2-ST) +
       // + sqrt(1-t^2)*y*(1-s^2)*x^2*PS29 +
       // - asin(t)_low+R^3*(c3+c5*R^2+c7*R^4+c9*R^6)
       fma.s1 F_DZ_TERM = F_P39, f1, F_DZ_TERM
       nop.i 0;;
}


{.mfi
       nop.m 0
       // d(ts)+z*y*d(1-s^2)*x+dz*y*(1-s^2)*(1-x)+z*y*(s^2-1+t^2-ST) +
       // + sqrt(1-t^2)*y*(1-s^2)*x^2*PS29 + z*y*(1-s^2)_s*x +
       // - asin(t)_low+R^3*(c3+c5*R^2+c7*R^4+c9*R^6)
       fma.s1 F_DZ_TERM = F_ZY1S2S, F_X, F_DZ_TERM
       nop.i 0;;
}


{.mfi
       nop.m 0
       // d(ts)+z*y*d(1-s^2)*x+dz*y*(1-s^2)*(1-x)+z*y*(s^2-1+t^2-ST) +
       // + sqrt(1-t^2)*y*(1-s^2)*x^2*PS29 + z*y*(1-s^2)_s*x +
       // - asin(t)_low+R^3*(c3+c5*R^2+c7*R^4+c9*R^6) +
       // + (t*s)_s-t^2*y*z+z*y*ST
       fma.s1 F_DZ_TERM = F_TSS, f1, F_DZ_TERM
       nop.i 0;;
}


.pred.rel "mutex", p6, p11
{.mfi
       nop.m 0
       // result: add high part of table value
       // s>0 in this case
 (p11) fnma.s0 f8 = F_DZ_TERM, f1, F_ATHI
       nop.i 0
}

{.mfb
       nop.m 0
       // result: add high part of pi-table value
       // if s<0
 (p6)  fma.s0 f8 = F_DZ_TERM, f1, F_ATHI
       br.ret.sptk b0;;
}






SMALL_S:

       // use 15-term polynomial approximation

{.mmi
       // r3 = pointer to polynomial coefficients
       addl r3 = @ltoff(poly_coeffs), gp;;
       // load start address for coefficients
       ld8 r3 = [r3]
       mov R_TMP = 0x3fbf;;
}


{.mmi
       add r2 = 64, r3
       ldfe F_C3 = [r3], 16
       // p7 = 1 if |s|<2^{-64} (exponent of s<bias-64)
       cmp.lt p7, p0 = R_EXP0, R_TMP;;
}

{.mmf
       ldfe F_C5 = [r3], 16
       ldfpd F_C11, F_C13 = [r2], 16
	   nop.f 0;;
}

{.mmf
       ldfpd F_C7, F_C9 = [r3], 16
       ldfpd F_C15, F_C17 = [r2]
       nop.f 0;;
}



{.mfb
       // load pi/2
       ldfpd F_PI2_LO, F_PI2_HI = [r3]
       // s^2
       fma.s1 F_R2 = f8, f8, f0
	   // |s|<2^{-64}
  (p7) br.cond.spnt  RETURN_PI2;;
}


{.mfi
       nop.m 0
       // s^3
       fma.s1 F_R3 = f8, F_R2, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // s^4
       fma.s1 F_R4 = F_R2, F_R2, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c3+c5*s^2
       fma.s1 F_P35 = F_C5, F_R2, F_C3
       nop.i 0
}

{.mfi
       nop.m 0
       // c11+c13*s^2
       fma.s1 F_P1113 = F_C13, F_R2, F_C11
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c7+c9*s^2
       fma.s1 F_P79 = F_C9, F_R2, F_C7
       nop.i 0
}

{.mfi
       nop.m 0
       // c15+c17*s^2
       fma.s1 F_P1517 = F_C17, F_R2, F_C15
       nop.i 0;;
}

{.mfi
       nop.m 0
	   // (pi/2)_high-s_high
	   fnma.s1 F_T = f8, f1, F_PI2_HI
	   nop.i 0
}
{.mfi
       nop.m 0
       // s^8
       fma.s1 F_R8 = F_R4, F_R4, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c3+c5*s^2+c7*s^4+c9*s^6
       fma.s1 F_P39 = F_P79, F_R4, F_P35
       nop.i 0
}

{.mfi
       nop.m 0
       // c11+c13*s^2+c15*s^4+c17*s^6
       fma.s1 F_P1117 = F_P1517, F_R4, F_P1113
       nop.i 0;;
}

{.mfi
       nop.m 0
	   // -s_high
	   fms.s1 F_S = F_T, f1, F_PI2_HI
	   nop.i 0;;
}

{.mfi
       nop.m 0
       // c3+..+c17*s^14
       fma.s1 F_P317 = F_R8, F_P1117, F_P39
       nop.i 0;;
}

{.mfi
       nop.m 0
	   // s_low
	   fma.s1 F_DS = f8, f1, F_S
	   nop.i 0;;
}

{.mfi
       nop.m 0
       // (pi/2)_low-s^3*(c3+..+c17*s^14)
       fnma.s0 F_P317 = F_P317, F_R3, F_PI2_LO
	   nop.i 0;;
}

{.mfi
       nop.m 0
	   // (pi/2)_low-s_low-s^3*(c3+..+c17*s^14)
	   fms.s1 F_P317 = F_P317, f1, F_DS
	   nop.i 0;;
}

{.mfb
       nop.m 0
	   // result: pi/2-s-c3*s^3-..-c17*s^17
	   fma.s0 f8 = F_T, f1, F_P317
       br.ret.sptk b0;;
}





RETURN_PI2:

{.mfi
       nop.m 0
       // (pi/2)_low-s
	   fms.s0 F_PI2_LO = F_PI2_LO, f1, f8
	   nop.i 0;;
}

{.mfb
       nop.m 0
	   // (pi/2)-s
	   fma.s0 f8 = F_PI2_HI, f1, F_PI2_LO
	   br.ret.sptk b0;;
}





VERY_LARGE_INPUT:


{.mmf
       // pointer to pi_low, pi_high
	   add r2 = 80, r3
       // load C5
       ldfe F_C5 = [r3], 16
       // x = ((1-(s^2)_s)*y^2-1)/2-(s^2-(s^2)_s)*y^2/2
       fma.s1 F_X = F_X, F_05, f0;;
}

.pred.rel "mutex", p6, p11
{.mmf
       // load pi (low, high), if s<0
 (p6)  ldfpd F_PI2_LO, F_PI2_HI = [r2]
       // C7, C9
       ldfpd F_C7, F_C9 = [r3], 16
	   // if s>0, set F_PI2_LO=0
 (p11) fma.s1 F_PI2_HI = f0, f0, f0;;
}

{.mfi
       nop.m 0
 (p11) fma.s1 F_PI2_LO = f0, f0, f0
       nop.i 0;;
}

{.mfi
       // adjust address for C_11
	   add r3 = 16, r3
       // c9*x+c8
       fma.s1 F_S89 = F_X, F_CS9, F_CS8
       nop.i 0
}

{.mfi
       nop.m 0
       // x^2
       fma.s1 F_X2 = F_X, F_X, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // y*(1-s^2)*x
       fma.s1 F_Y1S2X = F_Y1S2, F_X, f0
       nop.i 0
}

{.mfi
       // C11, C13
       ldfpd F_C11, F_C13 = [r3], 16
       // c7*x+c6
       fma.s1 F_S67 = F_X, F_CS7, F_CS6
       nop.i 0;;
}


{.mfi
       // C15, C17
       ldfpd F_C15, F_C17 = [r3], 16
       // c3*x+c2
       fma.s1 F_S23 = F_X, F_CS3, F_CS2
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c5*x+c4
       fma.s1 F_S45 = F_X, F_CS5, F_CS4
       nop.i 0;;
}




{.mfi
       nop.m 0
       // y*(1-s^2)*x^2
       fma.s1 F_Y1S2X2 = F_Y1S2, F_X2, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // x^4
       fma.s1 F_X4 = F_X2, F_X2, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c9*x^3+..+c6
       fma.s1 F_S69 = F_X2, F_S89, F_S67
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c5*x^3+..+c2
       fma.s1 F_S25 = F_X2, F_S45, F_S23
       nop.i 0;;
}



{.mfi
       nop.m 0
       // (pi)_high-y*(1-s^2)_s
       fnma.s1 F_HI = F_Y, F_1S2_S, F_PI2_HI
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c9*x^7+..+c2
       fma.s1 F_S29 = F_X4, F_S69, F_S25
       nop.i 0;;
}


{.mfi
       nop.m 0
       // -(y*(1-s^2)_s)_high
       fms.s1 F_1S2_HI = F_HI, f1, F_PI2_HI
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (PS29*x^2+x)*y*(1-s^2)
       fma.s1 F_S19 = F_Y1S2X2, F_S29, F_Y1S2X
       nop.i 0;;
}


{.mfi
       nop.m 0
       // y*(1-s^2)_s-(y*(1-s^2))_high
       fma.s1 F_DS2 = F_Y, F_1S2_S, F_1S2_HI
       nop.i 0;;
}



{.mfi
       nop.m 0
       // R ~ sqrt(1-s^2)
       // (used for polynomial evaluation)
       fnma.s1 F_R = F_S19, f1, F_Y1S2
       nop.i 0;;
}


{.mfi
       nop.m 0
       // y*(1-s^2)-(y*(1-s^2))_high
       fma.s1 F_DS2 = F_Y, F_DS, F_DS2
       nop.i 0
}

{.mfi
       nop.m 0
       // (pi)_low+(PS29*x^2)*y*(1-s^2)
       fma.s1 F_S29 = F_Y1S2X2, F_S29, F_PI2_LO
       nop.i 0;;
}


{.mfi
       nop.m 0
       // R^2
       fma.s1 F_R2 = F_R, F_R, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
	   // if s<0
       // (pi)_low+(PS29*x^2)*y*(1-s^2)-(y*(1-s^2)-(y*(1-s^2))_high)
       fms.s1 F_S29 = F_S29, f1, F_DS2
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c7+c9*R^2
       fma.s1 F_P79 = F_C9, F_R2, F_C7
       nop.i 0
}

{.mfi
       nop.m 0
       // c3+c5*R^2
       fma.s1 F_P35 = F_C5, F_R2, F_C3
       nop.i 0;;
}



{.mfi
       nop.m 0
       // R^4
       fma.s1 F_R4 = F_R2, F_R2, f0
       nop.i 0
}

{.mfi
       nop.m 0
       // R^3
       fma.s1 F_R3 = F_R2, F_R, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c11+c13*R^2
       fma.s1 F_P1113 = F_C13, F_R2, F_C11
       nop.i 0
}

{.mfi
       nop.m 0
       // c15+c17*R^2
       fma.s1 F_P1517 = F_C17, F_R2, F_C15
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (pi)_low+(PS29*x^2)*y*(1-s^2)-(y*(1-s^2)-(y*(1-s^2))_high)+y*(1-s^2)*x
       fma.s1 F_S29 = F_Y1S2, F_X, F_S29
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c11+c13*R^2+c15*R^4+c17*R^6
       fma.s1 F_P1117 = F_P1517, F_R4, F_P1113
       nop.i 0
}

{.mfi
       nop.m 0
       // c3+c5*R^2+c7*R^4+c9*R^6
       fma.s1 F_P39 = F_P79, F_R4, F_P35
       nop.i 0;;
}



{.mfi
       nop.m 0
       // R^8
       fma.s1 F_R8 = F_R4, F_R4, f0
       nop.i 0;;
}


{.mfi
       nop.m 0
       // c3+c5*R^2+c7*R^4+c9*R^6+..+c17*R^14
       fma.s1 F_P317 = F_P1117, F_R8, F_P39
       nop.i 0;;
}


{.mfi
       nop.m 0
       // (pi)_low-(PS29*x^2)*y*(1-s^2)-(y*(1-s^2)-
       // -(y*(1-s^2))_high)+y*(1-s^2)*x - P3, 17
       fnma.s1 F_S29 = F_P317, F_R3, F_S29
       nop.i 0;;
}

.pred.rel "mutex", p6, p11
{.mfi
       nop.m 0
       // Result (if s<0):
       // (pi)_low-(PS29*x^2)*y*(1-s^2)-(y*(1-s^2)-
       // -(y*(1-s^2))_high)+y*(1-s^2)*x - P3, 17
       // +(pi)_high-(y*(1-s^2))_high
 (p6)  fma.s0 f8 = F_S29, f1, F_HI
       nop.i 0
}

{.mfb
       nop.m 0
	   // Result (if s>0):
       // (PS29*x^2)*y*(1-s^2)-
       // -y*(1-s^2)*x + P3, 17
       // +(y*(1-s^2))
 (p11) fms.s0 f8 = F_Y, F_1S2_S, F_S29
       br.ret.sptk b0;;
}






acosl_SPECIAL_CASES:

{.mfi
       alloc r32 = ar.pfs, 1, 4, 4, 0
       // check if the input is a NaN, or unsupported format
       // (i.e. not infinity or normal/denormal)
       fclass.nm p7, p8 = f8, 0x3f
       // pointer to pi/2
       add r3 = 96, r3;;
}


{.mfi
       // load pi/2
       ldfpd F_PI2_HI, F_PI2_LO = [r3]
       // get |s|
       fmerge.s F_S = f0, f8
       nop.i 0
}

{.mfb
       nop.m 0
       // if NaN, quietize it, and return
 (p7) fma.s0 f8 = f8, f1, f0
 (p7) br.ret.spnt b0;;
}


{.mfi
       nop.m 0
       // |s| = 1 ?
       fcmp.eq.s0 p9, p10 = F_S, f1
       nop.i 0
}

{.mfi
       nop.m 0
       // load FR_X
       fma.s1 FR_X = f8, f1, f0
       // load error tag
       mov GR_Parameter_TAG = 57;;
}


{.mfi
       nop.m 0
       // if s = 1, result is 0
 (p9)  fma.s0 f8 = f0, f0, f0
       // set p6=0 for |s|>1
 (p10) cmp.ne p6, p0 = r0, r0;;
}


{.mfb
       nop.m 0
       //  if s = -1, result is pi
 (p6) fma.s0 f8 = F_PI2_HI, f1, F_PI2_LO
       // return if |s| = 1
 (p9) br.ret.sptk b0;;
}


{.mfi
       nop.m 0
       // get Infinity
       frcpa.s1 FR_RESULT, p0 = f1, f0
       nop.i 0;;
}


{.mfb
       nop.m 0
       // return QNaN indefinite (0*Infinity)
       fma.s0 FR_RESULT = f0, FR_RESULT, f0
       nop.b 0;;
}


GLOBAL_LIBM_END(acosl)
libm_alias_ldouble_other (acos, acos)


LOCAL_LIBM_ENTRY(__libm_error_region)
.prologue
// (1)
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


// (2)
{ .mmi
        stfe [GR_Parameter_Y] = f1,16         // Store Parameter 2 on stack
        add GR_Parameter_X = 16,sp            // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                     // Save b0
};;

.body
// (3)
{ .mib
        stfe [GR_Parameter_X] = FR_X              // Store Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y
        nop.b 0                                 // Parameter 3 address
}
{ .mib
        stfe [GR_Parameter_Y] = FR_RESULT             // Store Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#   // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;

// (4)
{ .mmi
        ldfe  f8 = [GR_Parameter_RESULT]       // Get return result off stack
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
