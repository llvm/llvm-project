.file "log2l.s"


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
// 09/25/00 Initial version
// 11/22/00 Fixed accuracy bug (for mantissas near 1, 2)
// 12/07/00 Fixed C_1l constant, eliminated rounding errors in
//          reduced argument (x*frcpa(x)-1)
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// long double log2l(long double)
//
// Overview of operation
//==============================================================
// Background
//
// Implementation
//
// Let x = 2^l * m, where     m=1.b1 b2 ... b8 b9 ... b52
//     y=frcpa(m),   r=m*y-1, f=b1 b2 .. b8
// T_hi is a table that stores the 24 most significant bits of log2(1/y)
// (in entries 1..255) in single precision format
// T_low is a table that stores (log2(1/y)-T_high), rounded to double
// precision
//
// f is used as an index; T_high[255]=T_low[255]=0
//
// If f=0 and b9=0, r is set to 2^{-8}* 0.b9 b10 ... b52 = m-1 (fractional part of m),
//                  and 0 is used instead of T_high[0], T_low[0]
//                  (polynomial evaluation only, for m=1+r, 0<=r<2^{-9})
// If f=255, r is set to (m-2)/2  (T[255]=0, and only polynomial evaluation is used
//                                 for m=2(1-r'), 0<=r'<2^{-9})
//
// If 2^{-9}<=m<2-2^{-8} or (input not near 1), let C1r=(2^{16}+C1*r)-2^{16}
//                       and let E=((RN(m*y)-1)-r)+(m*y-RN(m*y))
// Else let C1r=C1*r (rounded to 64 significant bits)  and let  E=0
//
// Let D=C1*r-C1r
//
//
// log2l(x) is approximated as
//     (l+T_high[f]+C1r) + (D+r*(c1+c2*r+c3*r^2...+c8*r^7)+(T_low[f]+C_1*E))
//


// Special values
//==============================================================
//  log2l(0)=-inf, raises Divide by Zero
//  log2l(+inf)=inf
//  log2l(x)=NaN,  raises Invalid if x<0
//


// Registers used
//==============================================================
//   f6-f15, f32-f36
//   r2-r3, r23-r23
//   p6,p7,p8,p12
//


GR_SAVE_B0                    = r33
GR_SAVE_PFS                   = r34
GR_SAVE_GP                    = r35 // This reg. can safely be used
GR_SAVE_SP                    = r36

GR_Parameter_X                = r37
GR_Parameter_Y                = r38
GR_Parameter_RESULT           = r39
GR_Parameter_TAG              = r40

FR_X             = f10
FR_Y             = f1
FR_RESULT        = f8




// Data tables
//==============================================================

RODATA

.align 16

LOCAL_OBJECT_START(poly_coeffs)

data8 0xb8aa3b295c17f0bc, 0x00003fff  // C_1
data8 0x3fca61762a7aded9, 0xbfc71547652b82fe // C_7, C_8
data8 0x3fd2776c50ef9bfe, 0xbfcec709dc3a03fd // C_5, C_6
data8 0x3fdec709dc3a03fd, 0xbfd71547652b82fe  // C_3, C_4
//data8 0xd871319ff0342580, 0x0000bfbd	// C_1l (low part of C1)
data8 0x82f0025f2dc582ee, 0x0000bfbe   // C_1l (low part of C1)
data8 0xb8aa3b295c17f0bc, 0x0000bffe  // C_2
LOCAL_OBJECT_END(poly_coeffs)




LOCAL_OBJECT_START(T_table)

data4 0x3b38d875, 0x3c0ae7f4, 0x3c67f738, 0x3ca2b253
data4 0x3ccbb91d, 0x3cfac91e, 0x3d1504a5, 0x3d29c4a0
data4 0x3d419264, 0x3d567aa6, 0x3d6e76ca, 0x3d81c3f7
data4 0x3d8c5630, 0x3d9876e9, 0x3da31e0a, 0x3dadcf09
data4 0x3db889f9, 0x3dc34eec, 0x3dce1df5, 0x3dd8f726
data4 0x3de3da94, 0x3deec851, 0x3df82ea4, 0x3e0197dd
data4 0x3e071dad, 0x3e0ca8ca, 0x3e116d6e, 0x3e170281
data4 0x3e1bcfbc, 0x3e216ee9, 0x3e2644dc, 0x3e2b1ee1
data4 0x3e30cd12, 0x3e35affd, 0x3e3a970f, 0x3e3f824f
data4 0x3e4544c0, 0x3e4a3926, 0x3e4f31d1, 0x3e542ec7
data4 0x3e593012, 0x3e5e35b7, 0x3e633fbf, 0x3e677625
data4 0x3e6c884b, 0x3e719eea, 0x3e76ba0a, 0x3e7bd9b2
data4 0x3e80111d, 0x3e82a523, 0x3e84ccec, 0x3e876533
data4 0x3e89ffd1, 0x3e8c2d22, 0x3e8e5c18, 0x3e90fd0a
data4 0x3e932fa9, 0x3e95d506, 0x3e980b5a, 0x3e9a4361
data4 0x3e9c7d1f, 0x3e9f2b16, 0x3ea168a0, 0x3ea3a7ea
data4 0x3ea5e8f5, 0x3ea82bc4, 0x3eaa705b, 0x3eacb6bb
data4 0x3eaefee7, 0x3eb148e3, 0x3eb394b1, 0x3eb5e255
data4 0x3eb831d0, 0x3eba8327, 0x3ebcd65c, 0x3ebeb3e0
data4 0x3ec10a7a, 0x3ec362f9, 0x3ec5bd63, 0x3ec7a0b3
data4 0x3ec9fe96, 0x3ecc5e6c, 0x3ece4619, 0x3ed0a978
data4 0x3ed293fe, 0x3ed4faf1, 0x3ed6e859, 0x3ed952eb
data4 0x3edb433c, 0x3eddb178, 0x3edfa4bc, 0x3ee19953
data4 0x3ee40cee, 0x3ee60484, 0x3ee7fd73, 0x3ee9f7bb
data4 0x3eec7280, 0x3eee6fda, 0x3ef06e94, 0x3ef26eb1
data4 0x3ef47031, 0x3ef67317, 0x3ef8f8b2, 0x3efafec5
data4 0x3efd0644, 0x3eff0f32, 0x3f008cc8, 0x3f0192b0
data4 0x3f029952, 0x3f03a0b0, 0x3f0466b2, 0x3f056f5a
data4 0x3f0678c0, 0x3f0782e6, 0x3f088dcc, 0x3f099973
data4 0x3f0aa5dd, 0x3f0b6fac, 0x3f0c7d6d, 0x3f0d8bf4
data4 0x3f0e575b, 0x3f0f673e, 0x3f1077e9, 0x3f1144ef
data4 0x3f1256fc, 0x3f1369d6, 0x3f143880, 0x3f154cc1
data4 0x3f161c7a, 0x3f173227, 0x3f1802f2, 0x3f191a0f
data4 0x3f19ebee, 0x3f1b047e, 0x3f1bd775, 0x3f1cf17b
data4 0x3f1dc58e, 0x3f1ee10f, 0x3f1fb63f, 0x3f208bea
data4 0x3f21a98f, 0x3f22805c, 0x3f2357a7, 0x3f247778
data4 0x3f254fe9, 0x3f2628d9, 0x3f270249, 0x3f2824fb
data4 0x3f28ff97, 0x3f29dab4, 0x3f2ab654, 0x3f2b9277
data4 0x3f2cb8c8, 0x3f2d961e, 0x3f2e73fa, 0x3f2f525b
data4 0x3f303143, 0x3f3110b1, 0x3f31f0a7, 0x3f32d125
data4 0x3f33b22b, 0x3f3493bc, 0x3f3575d6, 0x3f36587b
data4 0x3f373bab, 0x3f381f68, 0x3f3903b1, 0x3f39e888
data4 0x3f3acdec, 0x3f3bb3e0, 0x3f3c9a63, 0x3f3d8177
data4 0x3f3e1bd4, 0x3f3f03d9, 0x3f3fec71, 0x3f40d59b
data4 0x3f41bf59, 0x3f42a9ab, 0x3f434635, 0x3f443180
data4 0x3f451d61, 0x3f4609d9, 0x3f46a7d3, 0x3f479549
data4 0x3f488357, 0x3f492261, 0x3f4a1171, 0x3f4b011c
data4 0x3f4ba139, 0x3f4c91e8, 0x3f4d8334, 0x3f4e246a
data4 0x3f4f16be, 0x3f5009b1, 0x3f50ac02, 0x3f51a001
data4 0x3f524305, 0x3f533812, 0x3f53dbca, 0x3f54d1e7
data4 0x3f55c8a8, 0x3f566d85, 0x3f57655b, 0x3f580af0
data4 0x3f58b0d0, 0x3f59aa2c, 0x3f5a50c7, 0x3f5b4b3c
data4 0x3f5bf294, 0x3f5cee26, 0x3f5d963c, 0x3f5e92ed
data4 0x3f5f3bc3, 0x3f5fe4e7, 0x3f60e32d, 0x3f618d13
data4 0x3f623748, 0x3f63372a, 0x3f63e223, 0x3f648d6b
data4 0x3f658eee, 0x3f663afe, 0x3f66e75e, 0x3f67ea86
data4 0x3f6897b0, 0x3f69452c, 0x3f69f2f9, 0x3f6af847
data4 0x3f6ba6e2, 0x3f6c55d0, 0x3f6d0510, 0x3f6e0c8d
data4 0x3f6ebc9f, 0x3f6f6d04, 0x3f701dbe, 0x3f70cecd
data4 0x3f718030, 0x3f728ae6, 0x3f733d20, 0x3f73efaf
data4 0x3f74a296, 0x3f7555d3, 0x3f760967, 0x3f76bd53
data4 0x3f777197, 0x3f7880a1, 0x3f7935c2, 0x3f79eb3c
data4 0x3f7aa10f, 0x3f7b573b, 0x3f7c0dc2, 0x3f7cc4a3
data4 0x3f7d7bdf, 0x3f7e3376, 0x3f7eeb68, 0x00000000
LOCAL_OBJECT_END(T_table)



LOCAL_OBJECT_START(T_low)


data8 0x3dc0b97f689876ef, 0x3dfd5d906028ac01
data8 0x3df8b9cbb8d7240b, 0x3de0c941a2f220cd
data8 0x3e09c6aecba15936, 0x3dfa6d528241827c
data8 0x3dd0bad25714903c, 0x3e2776b01dc036a2
data8 0x3e2b914bc77f158b, 0x3e1c0fafd29dc74a
data8 0x3e28dadc119cd3de, 0x3e3bca869da085be
data8 0x3e19d1e700f2200a, 0x3e3e13530cc37504
data8 0x3e3936464d9c41ee, 0x3e3c3fa21c9499d0
data8 0x3e3259e079b6c6e8, 0x3e2a364069c4f7f3
data8 0x3e1274c84f6c6364, 0x3e3796170159f454
data8 0x3e26e1e389f4364e, 0x3e28cedda8c7f658
data8 0x3e376c2028433268, 0x3e4aee6d650c82e1
data8 0x3e33e65094fbeeb4, 0x3e4c7d125aa92c5d
data8 0x3e1559a4b69691d8, 0x3e18efabeb7d7221
data8 0x3e4c2b255abaa8de, 0x3e37436952a4538b
data8 0x3e4e6807f4ba00b8, 0x3e33ff5964190e42
data8 0x3e4f5d798cead43c, 0x3e4f3676443bf453
data8 0x3e4660f8d5bc1bf5, 0x3e2d4f9f3ab04f36
data8 0x3e357f7a64ccd537, 0x3e394caf7c9b05af
data8 0x3e225c7d17ab29b0, 0x3e4eb202f6d55a12
data8 0x3e32faa68b19bcd2, 0x3e45ee1c9b566a8b
data8 0x3e4770a67de054ff, 0x3e42234fb9de6d6b
data8 0x3e4ad139825c6e19, 0x3e47f3d334814a93
data8 0x3e2af1ec402867b6, 0x3e2bfbda0c956e3d
data8 0x3e4287b831e77ff2, 0x3e54bf0eb77f7b89
data8 0x3e5b9259a1029607, 0x3e4a764b015e699d
data8 0x3e4d0b68ea883ab5, 0x3e33e829ecdadf46
data8 0x3e52f27efef3031b, 0x3e3073979e4af89e
data8 0x3e3b980f2cd6c253, 0x3e2a5f0f5f7f66a9
data8 0x3e37788738117b02, 0x3e58aa29a784d52f
data8 0x3e4f5504c4ff2466, 0x3e002d40340fa647
data8 0x3e5f53b64592f4c3, 0x3e543f222c526802
data8 0x3e5680e547a872fa, 0x3e5e234bd1154450
data8 0x3e3000edc18b6d21, 0x3e1c3c1f000942a8
data8 0x3e51eeae0e442d6e, 0x3e4fb265376623f2
data8 0x3e57b5941782d830, 0x3e3a4b83f24ae52c
data8 0x3e5a5fb4f23978de, 0x3e51ed071563fb02
data8 0x3e49e2071f51a7a8, 0x3e5e43ae5b924234
data8 0x3dfa2be9aedf374a, 0x3e56dea3dbba67d5
data8 0x3e3375fe732b3c3e, 0x3e5a0c6f91f2e77e
data8 0x3e55e1bf1c969e41, 0x3e30a5a5166b8eee
data8 0x3e53e6e9a539d46c, 0x3e542981b3d7b0e6
data8 0x3e595fd8ff36ad64, 0x3e5edeb9e65cbbb4
data8 0x3e46aeab4d3434c1, 0x3e4ea3ff0564b010
data8 0x3e59b00be2e3c25a, 0x3e5b887cd7b0821f
data8 0x3e5f666668547b4d, 0x3e4d0733a805273f
data8 0x3e26a2ff21c4aec5, 0x3e4c336f7a3a78f3
data8 0x3e11ad12b628e2d0, 0x3e56d43ff3f0ea64
data8 0x3e238809433cccd2, 0x3e40d9734147d40f
data8 0x3e54245fe3e24e06, 0x3e251441fce4d48c
data8 0x3e517114efc5d1f9, 0x3e5e9a99154b0d82
data8 0x3e442a71337970f8, 0x3e420c7c69211fdf
data8 0x3e537e7d5d43c6a7, 0x3e4376c66ad9ad8b
data8 0x3e49054d678a4f1c, 0x3e5d23cb3bc19f18
data8 0x3e6ebcd449dcab2b, 0x3e67f5fc2849c88a
data8 0x3e63f388395d3e84, 0x3e65c1103b0ad7e9
data8 0x3e6d5d1dd031f353, 0x3e5a159dae75c4d0
data8 0x3e4d5e22aa75f71d, 0x3e5e379ee62e1e35
data8 0x3e4df082213cb2dc, 0x3e6bfa06c156f521
data8 0x3e66e2d3c19b517b, 0x3e426b7098590071
data8 0x3e541bd027e9854e, 0x3e5061dd924b0ac0
data8 0x3e6dae01df373a03, 0x3e3baec80b207b0b
data8 0x3e6b6a6fe06bebac, 0x3e61aebcfc3ab5d1
data8 0x3e584ee3e7c79d83, 0x3e6b3c1b2840cb40
data8 0x3e6c842085d6befd, 0x3e6ac04fd7b141e0
data8 0x3e6c48250474141d, 0x3e2d889b86125f69
data8 0x3e6e74740225dad0, 0x3e45940d31d50a7c
data8 0x3e695476a6c39ddc, 0x3e6d9a6d857a060a
data8 0x3e4a3e9bb4b69337, 0x3e484f3ce4707ed6
data8 0x3e39dd125d25fc27, 0x3e563fb400de8732
data8 0x3e5fdd6d0ee28b48, 0x3e669d15b869bb07
data8 0x3e40687cfad7964d, 0x3e69317990d43957
data8 0x3e633d57e24ae1bd, 0x3e618bf03710eabb
data8 0x3e4b4df6fccd1160, 0x3e3fb26ddaa1ec45
data8 0x3e3810a5e1817fd4, 0x3e6857373642fa5c
data8 0x3e673db6193add31, 0x3e63200c8acbc9c3
data8 0x3e3d2dee448ebb62, 0x3e6a19723a80db6a
data8 0x3e5e7cdab8fd3e6a, 0x3e671855cd660672
data8 0x3e473c3c78a85ecd, 0x3e5f5e23056a7cf2
data8 0x3e52538519527367, 0x3e4b573bcf2580e9
data8 0x3e6d6f856fe90c60, 0x3e2d932a8487642e
data8 0x3e5236fc78b6174c, 0x3e50cb91d406db50
data8 0x3e650e8bd562aa57, 0x3e424ee3d9a82f2e
data8 0x3e59363960e1e3d9, 0x3e379604c1150a3e
data8 0x3e6d914f6c2ac258, 0x3e62967a451a7b48
data8 0x3e684b5f01139cb2, 0x3e448bbfbf6d292c
data8 0x3e6227e7fb487e73, 0x3e6d39d50290f458
data8 0x3e58368342b4b668, 0x3e65dc0c25bd1763
data8 0x3e61b7dc362e22b5, 0x3e671691f094bb80
data8 0x3e5011642d5123f2, 0x3e4c4eb7f11e41be
data8 0x3e5dcee36ca242cf, 0x3e6791cefff688f1
data8 0x3e60e23c8dda4ecd, 0x3e48e6a22fe78cfe
data8 0x3e6d703f244adc86, 0x3e6a281a85a5049d
data8 0x3e570f20e6403d9e, 0x3e2211518a12956f
data8 0x3e6737d1e54d71df, 0x3e66b1881476f5e9
data8 0x3e6e1bbeef085376, 0x3e47cad4944a32be
data8 0x3e527f2c738e7ee9, 0x3e699883a4b9fb29
data8 0x3e5c17d1108740d9, 0x3e5d4a9c79a43389
data8 0x3e49fdc24462ba3b, 0x3e24dbb3a60cceb2
data8 0x3e5c5bf618780748, 0x3e5c38005b0c778c
data8 0x3e6be168dd6dd3fe, 0x3e633ab9370693b0
data8 0x3dd290556b0ae339, 0x3e607c317927096a
data8 0x3e59651353b3d90e, 0x3e4d8751e5e0ae0d
data8 0x3e46c81023272a85, 0x3e6b23c988f391b2
data8 0x3e608741d215209c, 0x3e60b8ba506d758f
data8 0x3e62ddbe74803297, 0x3e5dbb8b5087587d
data8 0x3e642aa529048131, 0x3e3dcbda6835dcf4
data8 0x3e6db503ce854d2a, 0x3e6dd00b49bc6849
data8 0x3e4db2f11243bc84, 0x3e3b9848efc2ea97
data8 0x3e58f18e17c82609, 0x3e6ed8645e16c312
data8 0x3e4065bdb60a5dd4, 0x3e490453c6e6c30a
data8 0x3e62373994aa31ba, 0x3e56305f0e6b2a95
data8 0x3e68c1601a6614ee, 0x3e614e204f19d93f
data8 0x3e6e5037ca773299, 0x3e693f98892561a6
data8 0x3e639de4f4bf700d, 0x3e416c071e93fd97
data8 0x3e65466991b415ef, 0x3e6896a324afac9d
data8 0x3e44f64802e2f11c, 0x3e64d7d747e2191a
data8 0x3e6174b7581de84c, 0x3e44c7b946e1d43c
data8 0x3e6a3bcbe30512ec, 0x3e5d3ed411c95ce4
data8 0x3e3e5b5735cfaf8e, 0x3e6e538ab34efb51
data8 0x3e514e204f19d93f, 0x3e5a88e6550c89a4
data8 0x3e66b97a5d9dfd8b, 0x3e5f46b1e14ebaf3
data8 0x3e357665f6893f5d, 0x3e6bbf633078d1d5
data8 0x3e5e7337a212c417, 0x3e3570fde15fc8cc
data8 0x3e21119402da92b4, 0x3e6566e830d1ff3b
data8 0x3e558883e480e220, 0x3e589ca3a68da411
data8 0x3e44eb66df73d648, 0x3e1a0a629b1b7e68
data8 0x3e54cc207b8c1116, 0x0000000000000000
LOCAL_OBJECT_END(T_low)


.section .text
GLOBAL_IEEE754_ENTRY(log2l)

{ .mfi
  alloc r32=ar.pfs,1,4,4,0
  // normalize x
  // y=frcpa(x)
  frcpa.s1 f41,p0=f1,f8
  // r26=bias-1
  mov r26=0xfffe
}
{.mfi
  // r23=bias+16
  mov r23=0xffff+16
  fma.s1 f7=f8,f1,f0
  // r2 = pointer to C_1...C_6 followed by T_table
  addl r2 = @ltoff(poly_coeffs), gp;;
}
{.mfi
  // get significand
  getf.sig r25=f8
  // f8 denormal ?
  fclass.m p8,p10=f8,0x9
  // r24=bias-8
  mov r24=0xffff-8;;
}
{.mfi
  setf.exp f36=r26
  nop.f 0
  // r27=bias
  mov r27=0xffff;;
}

{.mmf
  getf.exp r29=f8
  // load start address for C_1...C_7 followed by T_table
  ld8 r2=[r2]
  // will continue only for positive normal/unnormal numbers
  fclass.m.unc p0,p12 = f8, 0x19;;
}


.pred.rel "mutex",p8,p10
{.mfi
  // denormal input, repeat get significand (after normalization)
  (p8) getf.sig r25=f7
  // x=1 ?
  fcmp.eq.s0 p6,p0=f8,f1
  // get T_index
  (p10) shr.u r28=r25,63-8
}
{.mfi
  // f32=2^16
  setf.exp f32=r23
  nop.f 0
  mov r26=0x804;;
}

{.mfi
  // denormal input, repeat get exponent (after normalization)
  (p8) getf.exp r29=f7
  // f33=0
  mov f33=f0
  // r26=0x80400...0 (threshold for using polynomial approximation)
  shl r26=r26,64-12;;
}

{.mfb
  add r3=16,r2
  // r=x*y-1
  fms.s1 f6=f41,f8,f1
  (p12) br.cond.spnt SPECIAL_log2l
}
{.mfi
  // load C_1
  ldfe f14=[r2],48
  // RN(x*y)
  fma.s1 f43=f41,f8,f0
  mov r23=0xff;;
}

{.mmi
  // load C_7, C_8
  ldfpd f10,f11=[r3],16
  // load C_3,C_4
  ldfpd f15,f42=[r2],16
  (p8) shr.u r28=r25,63-8;;
}


{.mfi
  // load C_5, C_6
  ldfpd f12,f13=[r3]
  // pseudo-zero ?
  fcmp.eq.s0 p7,p0=f7,f0
  // if first 9 bits after leading 1 are all zero, then p8=1
  cmp.ltu p8,p12=r25,r26
}
{.mfi
  // load C1l
  ldfe f34=[r2],16
  fmerge.se f7=f1,f7
  // get T_index
  and r28=r28,r23;;
}
{.mfi
  // r29=exponent-bias
  sub r29=r29,r27
  // if first 8 bits after leading bit are 0, use polynomial approx. only
  (p8) fms.s1 f6=f7,f1,f1
  // start address of T_low
  add r3=1024+16,r2
}
{.mfi
  // load C_2
  ldfe f35=[r2],16
  // x=1, return 0
  (p6) fma.s0 f8=f0,f0,f0
  // first 8 bits after leading 1 are all ones ?
  cmp.eq p10,p0=r23,r28;;
}

{.mfb
  // if first 8 bits after leading 1 are all ones, use polynomial approx. only
  // add 1 to the exponent additive term, and estimate log2(1-r)
  (p10) add r29=1,r29
  nop.f 0
  (p7) br.cond.spnt LOG2_PSEUDO_ZERO
}
{.mfi
  // get T_low address
  shladd r3=r28,3,r3
  // if first 8 bits after leading 1 are all ones, use polynomial approx. only
  (p10) fms.s1 f6=f7,f36,f1
  // p10 --> p8=1, p12=0
  (p10) cmp.eq p8,p12=r0,r0;;
}

{.mfi
  // get T_high address
  shladd r2=r28,2,r2
  // L(x*y)=x*y-RN(x*y)
  fms.s1 f41=f41,f8,f43
  nop.i 0
}
{.mfi
  // p13=p12
  (p12) cmp.eq.unc p13,p0=r0,r0
  // RtH=RN(x*y)-1  (will eliminate rounding errors in r)
  fms.s1 f43=f43,f1,f1
  nop.i 0;;
}

.pred.rel "mutex",p8,p12
{.mfb
  // load T_high (unless first 9 bits after leading 1 are 0)
  (p12) ldfs f7=[r2]
  // set T_high=0 (if first 9 bits after leading 1 are 0)
  (p8) fma.s1 f7=f0,f0,f0
  // x=1, return
  (p6) br.ret.spnt b0
}
.pred.rel "mutex",p8,p12
{.mfi
  // p12: load T_low
  (p12) ldfd f36=[r3]
  // p8: set T_low=0
  (p8) fma.s1 f36=f0,f0,f0
  (p8) cmp.eq p8,p12=r29,r0;; //nop.i 0;;
}

.pred.rel "mutex",p8,p12
{.mfi
  // f8=expon - bias
  setf.sig f8=r29
  // general case: 2^{16}+C1*r
  (p12) fma.s1 f33=f6,f14,f32
  nop.i 0
}
{.mfi
  // r26=1
  mov r26=1
  // p8 (mantissa is close to 1, or close to 2): 2^{-8}+C1*r
  (p8) fma.s1 f32=f6,f14,f33
  nop.i 0;;
}

{.mfi
  nop.m 0
  // P78=C_7+C_8*r
  fma.s1 f10=f11,f6,f10
  // r26=2^{63}
  shl r26=r26,63
}
{.mfi
  nop.m 0
  // P34=C_3+r*C_4
  fma.s1 f15=f42,f6,f15
  nop.i 0;;
}
{.mfi
  nop.m 0
  // r2=r*r
  fma.s1 f11=f6,f6,f0
  nop.i 0
}
{.mfi
  nop.m 0
  // P56=C_5+C_6*r
  fma.s1 f13=f13,f6,f12
  nop.i 0;;
}

{.mfi
  nop.m 0
  // Rth-r
  (p13) fms.s1 f43=f43,f1,f6
  nop.i 0
}
{.mfi
  // significand(x)=1 ?
  cmp.eq p0,p6=r25,r26
  // P12=C1l+C_2*r
  fma.s1 f34=f35,f6,f34
  nop.i 0;;
}

.pred.rel "mutex",p8,p12
{.mfi
  nop.m 0
  // p12: C1r=(2^{16}+C1*r)-2^{16}
  (p12) fms.s1 f32=f33,f1,f32
  nop.i 0
}
{.mfi
  nop.m 0
  // p8: C1r=C1*r (double extended)
  (p8) fms.s1 f32=f32,f1,f33
  nop.i 0;;
}

{.mfi
  nop.m 0
  // L(x*y)*C_1+T_low
  (p13) fma.s1 f36=f41,f14,f36
  nop.i 0
}
{.mfi
  nop.m 0
  // P58=P56+r2*P78
  fma.s1 f13=f11,f10,f13
  nop.i 0;;
}
{.mfi
  nop.m 0
  // P14=P12+r2*P34
  fma.s1 f15=f15,f11,f34
  nop.i 0
}
{.mfi
  nop.m 0
  // r4=r2*r2
  fma.s1 f11=f11,f11,f0
  nop.i 0;;
}

{.mfi
  nop.m 0
  // normalize additive term (l=exponent of x)
  fcvt.xf f8=f8
  nop.i 0;;
}


{.mfi
  nop.m 0
  // D=C1*r-C1r
  (p6) fms.s1 f12=f14,f6,f32
  nop.i 0;;
}

{.mfi
  nop.m 0
  // T_low'=(Rth-r)*C1+(L(x*y)*C1+T_low)
  (p13) fma.s1 f36=f43,f14,f36
  nop.i 0;;
}
{.mfi
  nop.m 0
  // P18=P14+r4*P58
  (p6) fma.s1 f13=f11,f13,f15
  nop.i 0;;
}

{.mfi
  nop.m 0
  // add T_high+l
  (p6) fma.s1 f8=f8,f1,f7
  nop.i 0;;
}


{.mfi
  nop.m 0
  // D+T_low
  (p6) fma.s1 f12=f12,f1,f36
  nop.i 0;;
}


{.mfi
  nop.m 0
  // (T_high+l)+C1r
  (p6) fma.s1 f8=f8,f1,f32
  nop.i 0
}
{.mfi
  nop.m 0
  // (D+T_low)+r*P18
  (p6) fma.s1 f13=f13,f6,f12
  nop.i 0;;
}

//{.mfb
//nop.m 0
//mov f8=f36
//fma.s0 f8=f13,f6,f0
//br.ret.sptk b0;;
//}


{.mfb
  nop.m 0
  // result=((T_high+l)+C1r)+((D+T_low)+r*P18)
  (p6) fma.s0 f8=f13,f1,f8
  // return
  br.ret.sptk b0;;
}


SPECIAL_log2l:
{.mfi
  nop.m 0
  mov FR_X=f8
  nop.i 0
}
{.mfi
  nop.m 0
  // x=+Infinity ?
  fclass.m p7,p0=f8,0x21
  nop.i 0;;
}
{.mfi
  nop.m 0
  // x=+/-Zero ?
  fclass.m p8,p0=f7,0x7
  nop.i 0;;
}
{.mfi
  nop.m 0
  // x=-Infinity, -normal, -denormal ?
  fclass.m p6,p0=f8,0x3a
  nop.i 0;;
}
{.mfb
  nop.m 0
  // log2l(+Infinity)=+Infinity
  nop.f 0
  (p7) br.ret.spnt b0;;
}
{.mfi
  (p8) mov GR_Parameter_TAG = 168
  // log2l(+/-0)=-infinity, raises Divide by Zero
  // set f8=-0
  (p8) fmerge.ns f8=f0,f8
  nop.i 0;;
}
{.mfb
  nop.m 0
  (p8) frcpa.s0 f8,p0=f1,f8
  (p8) br.cond.sptk __libm_error_region;;
}
{.mfb
  (p6) mov GR_Parameter_TAG = 169
  // x<0: return NaN, raise Invalid
  (p6) frcpa.s0 f8,p0=f0,f0
  (p6) br.cond.sptk __libm_error_region;;
}


{.mfb
  nop.m 0
  // Remaining cases: NaNs
  fma.s0 f8=f8,f1,f0
  br.ret.sptk b0;;
}

LOG2_PSEUDO_ZERO:

{.mfi
  nop.m 0
  mov FR_X=f8
  nop.i 0
}
{.mfi
  mov GR_Parameter_TAG = 168
  // log2l(+/-0)=-infinity, raises Divide by Zero
  // set f8=-0
  fmerge.ns f8=f0,f8
  nop.i 0;;
}
{.mfb
  nop.m 0
  frcpa.s0 f8,p0=f1,f8
  br.cond.sptk __libm_error_region;;
}


GLOBAL_IEEE754_END(log2l)
libm_alias_ldouble_other (__log2, log2)


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
        add sp=-64,sp                           // Create new stack
        nop.f 0
        mov GR_SAVE_GP=gp                       // Save gp
};;
{ .mmi
        stfe [GR_Parameter_Y] = FR_Y,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfe [GR_Parameter_X] = FR_X                  // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
	nop.b 0
}
{ .mib
        stfe [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#         // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
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
