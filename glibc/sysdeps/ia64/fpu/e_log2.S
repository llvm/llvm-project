.file "log2.s"


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
//=================================================================
// 09/11/00 Initial version
// 03/19/01 Added one polynomial coefficient, to improve accuracy
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
// 04/18/03 Reformatted T[255]
//
// API
//=================================================================
// double log2(double)
//
// Overview of operation
//=================================================================
// Background
//
// Implementation
//
// Let x = 2^l * m, where     m=1.b1 b2 ... b8 b9 ... b52
//     y=frcpa(m),   r=m*y-1, f=b1 b2 .. b8 (table index)
// j=0 if f<128; j=1 if f>=128
// T is a table that stores log2(1/y) (in entries 1..255) rounded to
//   double extended precision; f is used as an index; T[255]=0
//
// If f=0 and b9=0, r is set to 2^{-8}* 0.b9 b10 ... b52 = m-1 (fractional part of m),
//                  and 0 is used instead of T[0]
//                  (polynomial evaluation only, for m=1+r, 0<=r<2^{-9})
// If f=255, r is set to (m-2)/2  (T[255]=0, and only polynomial evaluation is used
//                                 for m=2(1-r'), 0<=r'<2^{-9})
//
// log2(x) is approximated as
//     (l-j) + T[f] + (c1*r+c2*r^2+...+c7*r^7), if f>0
//


// Special values
//=================================================================
//  log2(0)=-inf, raises Divide by Zero
//  log2(+inf)=inf
//  log2(x)=NaN,  raises Invalid if x<0
//


// Registers used
//==============================================================
//   f6-f15, f32-f33
//   r2-r3, r23-r30
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

data8 0xbfd0000000000000, 0x3fc999999999999a //C_4, C_5
data8 0xbfc5555555555555, 0x3fc2492492492492 //C_6, C_7
data8 0xb8aa3b295c17f0bc, 0x00003fff  // C_1
data8 0xaaaaaaaaaaaaaaab, 0x00003ffd  // C_3=1/3
LOCAL_OBJECT_END(poly_coeffs)


LOCAL_OBJECT_START(T_table)

data8 0xb8d8752172fed131, 0x00003ff6
data8 0x8ae7f475764180a3, 0x00003ff8
data8 0xe7f73862e72ee35d, 0x00003ff8
data8 0xa2b25310c941a2f2, 0x00003ff9
data8 0xcbb91d671abb2e85, 0x00003ff9
data8 0xfac91e34daa50483, 0x00003ff9
data8 0x9504a5042eb495c5, 0x00003ffa
data8 0xa9c4a0bbb580ee02, 0x00003ffa
data8 0xc19264dc8a5e3bf9, 0x00003ffa
data8 0xd67aa6703ebf4a77, 0x00003ffa
data8 0xee76cac6d6e08ce7, 0x00003ffa
data8 0x81c3f7de5434ed04, 0x00003ffb
data8 0x8c563033a3ce01e4, 0x00003ffb
data8 0x9876e9f09a98661c, 0x00003ffb
data8 0xa31e0ac9b2326ce2, 0x00003ffb
data8 0xadcf09e1fd10e4a5, 0x00003ffb
data8 0xb889f992cf03cdb6, 0x00003ffb
data8 0xc34eec68d901a714, 0x00003ffb
data8 0xce1df524e9909ed9, 0x00003ffb
data8 0xd8f726bcb0b80ad0, 0x00003ffb
data8 0xe3da945b878e27d1, 0x00003ffb
data8 0xeec851633b76a320, 0x00003ffb
data8 0xf82ea4bb6101421a, 0x00003ffb
data8 0x8197ddd7736b2864, 0x00003ffc
data8 0x871dad4f994253f0, 0x00003ffc
data8 0x8ca8cae3e892d549, 0x00003ffc
data8 0x916d6e1559a4b697, 0x00003ffc
data8 0x97028118efabeb7d, 0x00003ffc
data8 0x9bcfbce1592ad5d5, 0x00003ffc
data8 0xa16ee95d0da54a91, 0x00003ffc
data8 0xa644dcf3403fa5d0, 0x00003ffc
data8 0xab1ee14ffd659064, 0x00003ffc
data8 0xb0cd12faebcc6757, 0x00003ffc
data8 0xb5affdf9b3b221e0, 0x00003ffc
data8 0xba970fb307c6ade1, 0x00003ffc
data8 0xbf824f3a9f3e7561, 0x00003ffc
data8 0xc544c055fde99333, 0x00003ffc
data8 0xca39266532bdf26c, 0x00003ffc
data8 0xcf31d124b8fa2f56, 0x00003ffc
data8 0xd42ec7f59017b6ab, 0x00003ffc
data8 0xd930124bea9a2c67, 0x00003ffc
data8 0xde35b7af70e4dab3, 0x00003ffc
data8 0xe33fbfbb8533ef03, 0x00003ffc
data8 0xe77625911a7dcef3, 0x00003ffc
data8 0xec884bd689cc12e3, 0x00003ffc
data8 0xf19eeabf9e99a40a, 0x00003ffc
data8 0xf6ba0a35e3d88051, 0x00003ffc
data8 0xfbd9b237f7b4192b, 0x00003ffc
data8 0x80111d4a1ee0c79e, 0x00003ffd
data8 0x82a523a5f875bbfc, 0x00003ffd
data8 0x84ccecdc92cd0815, 0x00003ffd
data8 0x87653369d92c057a, 0x00003ffd
data8 0x89ffd1742da3aa21, 0x00003ffd
data8 0x8c2d2227d053d9b6, 0x00003ffd
data8 0x8e5c189793f7f798, 0x00003ffd
data8 0x90fd0a20e72f3c96, 0x00003ffd
data8 0x932fa937301e59ae, 0x00003ffd
data8 0x95d5061a5f0f5f7f, 0x00003ffd
data8 0x980b5a2ef10e7023, 0x00003ffd
data8 0x9a4361c5514d3c27, 0x00003ffd
data8 0x9c7d1f7d541313fd, 0x00003ffd
data8 0x9f2b16040b500d04, 0x00003ffd
data8 0xa168a0fa9db22c98, 0x00003ffd
data8 0xa3a7eaa1f9116293, 0x00003ffd
data8 0xa5e8f5b4072a3d44, 0x00003ffd
data8 0xa82bc4f11a5e88aa, 0x00003ffd
data8 0xaa705b2001db8317, 0x00003ffd
data8 0xacb6bb0e1e0f8005, 0x00003ffd
data8 0xaefee78f75707221, 0x00003ffd
data8 0xb148e37ec994dd99, 0x00003ffd
data8 0xb394b1bdaca0bc17, 0x00003ffd
data8 0xb5e255349707e496, 0x00003ffd
data8 0xb831d0d2fda791cc, 0x00003ffd
data8 0xba83278f6838ab20, 0x00003ffd
data8 0xbcd65c67881c7d47, 0x00003ffd
data8 0xbeb3e0f21d72dc92, 0x00003ffd
data8 0xc10a7a03457d35dc, 0x00003ffd
data8 0xc362f9b6f51eddd3, 0x00003ffd
data8 0xc5bd6326ebfce656, 0x00003ffd
data8 0xc7a0b3d0637c8f97, 0x00003ffd
data8 0xc9fe96af0df8e4b5, 0x00003ffd
data8 0xcc5e6c214b4a2cd7, 0x00003ffd
data8 0xce46199f374d29cf, 0x00003ffd
data8 0xd0a978a14c0d9ebe, 0x00003ffd
data8 0xd293fecafec7f9b5, 0x00003ffd
data8 0xd4faf1f6f5cf32e6, 0x00003ffd
data8 0xd6e8595abaad34d1, 0x00003ffd
data8 0xd952eb7a8ffc1593, 0x00003ffd
data8 0xdb433ccd805f171e, 0x00003ffd
data8 0xddb178dc43e6bd84, 0x00003ffd
data8 0xdfa4bcfb333342a4, 0x00003ffd
data8 0xe19953741ccea015, 0x00003ffd
data8 0xe40cee16a2ff21c5, 0x00003ffd
data8 0xe6048470cdbde8ea, 0x00003ffd
data8 0xe7fd7308d6895b14, 0x00003ffd
data8 0xe9f7bbb6a1ff9f87, 0x00003ffd
data8 0xec7280138809433d, 0x00003ffd
data8 0xee6fda4365cd051f, 0x00003ffd
data8 0xf06e94a122ff1f12, 0x00003ffd
data8 0xf26eb1151441fce5, 0x00003ffd
data8 0xf470318b88a77e2f, 0x00003ffd
data8 0xf67317f4d4c8aa58, 0x00003ffd
data8 0xf8f8b250a9c4cde6, 0x00003ffd
data8 0xfafec54831f1a484, 0x00003ffd
data8 0xfd06449bf3eaea1e, 0x00003ffd
data8 0xff0f324ddb19ab67, 0x00003ffd
data8 0x808cc8320a9acf15, 0x00003ffe
data8 0x8192b0748f2cef06, 0x00003ffe
data8 0x829952f5e6a24ee5, 0x00003ffe
data8 0x83a0b0bfafe1424e, 0x00003ffe
data8 0x8466b29f9c41caea, 0x00003ffe
data8 0x856f5aae0881d857, 0x00003ffe
data8 0x8678c0eae8ee8190, 0x00003ffe
data8 0x8782e6685676b9d7, 0x00003ffe
data8 0x888dcc3abc4554ec, 0x00003ffe
data8 0x89997378de7b98b8, 0x00003ffe
data8 0x8aa5dd3be1044279, 0x00003ffe
data8 0x8b6facdfd0360ab8, 0x00003ffe
data8 0x8c7d6db7169e0cdb, 0x00003ffe
data8 0x8d8bf424d6e130b2, 0x00003ffe
data8 0x8e575b506f409fa6, 0x00003ffe
data8 0x8f673e418776492c, 0x00003ffe
data8 0x9077e9ed700ef9ba, 0x00003ffe
data8 0x9144ef1baec80b20, 0x00003ffe
data8 0x9256fcdb537f035f, 0x00003ffe
data8 0x9369d68d75e7e1d6, 0x00003ffe
data8 0x943880613b8f9f1e, 0x00003ffe
data8 0x954cc1d9e0d94206, 0x00003ffe
data8 0xd3c70a37bdf7a294, 0x0000bffd
data8 0xd19bb053fb0284ec, 0x0000bffd
data8 0xcffa1a3b7dafb8bf, 0x0000bffd
data8 0xcdcbe1e2776479ee, 0x0000bffd
data8 0xcc282218b8bfdda2, 0x0000bffd
data8 0xc9f703a9afcb38ac, 0x0000bffd
data8 0xc851146ab89593c6, 0x0000bffd
data8 0xc61d08265927a860, 0x0000bffd
data8 0xc474e39705912d26, 0x0000bffd
data8 0xc23de19ec30c6e3e, 0x0000bffd
data8 0xc09381cc45db45b4, 0x0000bffd
data8 0xbee82b4e025ff90c, 0x0000bffd
data8 0xbcace101149788ec, 0x0000bffd
data8 0xbaff46962ea47964, 0x0000bffd
data8 0xb950b1be5e0c14a2, 0x0000bffd
data8 0xb7110e6ce866f2bc, 0x0000bffd
data8 0xb5602ccc2a81db52, 0x0000bffd
data8 0xb3ae4ce740fc8ef1, 0x0000bffd
data8 0xb1fb6d92c8240ccc, 0x0000bffd
data8 0xafb609c09b244abc, 0x0000bffd
data8 0xae00d1cfdeb43cfd, 0x0000bffd
data8 0xac4a967a8c8c9bd0, 0x0000bffd
data8 0xaa93568c249e6c52, 0x0000bffd
data8 0xa8db10cdff375343, 0x0000bffd
data8 0xa68e6fc5a42376e3, 0x0000bffd
data8 0xa4d3c25e68dc57f2, 0x0000bffd
data8 0xa3180b0c192a3816, 0x0000bffd
data8 0xa15b488e7aa329a0, 0x0000bffd
data8 0x9f9d79a30f0e1d5f, 0x0000bffd
data8 0x9dde9d050ee7d4ac, 0x0000bffd
data8 0x9c1eb16d63d7356c, 0x0000bffd
data8 0x9a5db592a310c36a, 0x0000bffd
data8 0x989ba82907a9016f, 0x0000bffd
data8 0x96d887e26cd57b79, 0x0000bffd
data8 0x9514536e481c3a4f, 0x0000bffd
data8 0x934f0979a3715fc9, 0x0000bffd
data8 0x9188a8af1742a9d5, 0x0000bffd
data8 0x8fc12fb6c470995f, 0x0000bffd
data8 0x8df89d364e34f8f1, 0x0000bffd
data8 0x8c2eefd0d3f67dd6, 0x0000bffd
data8 0x8a642626eb093d54, 0x0000bffd
data8 0x88983ed6985bae58, 0x0000bffd
data8 0x86cb387b4a0feec6, 0x0000bffd
data8 0x84fd11add101024b, 0x0000bffd
data8 0x83c856dd81804b78, 0x0000bffd
data8 0x81f84c2c62afd6f1, 0x0000bffd
data8 0x80271d3e4be5ea5a, 0x0000bffd
data8 0xfca991447e7b485d, 0x0000bffc
data8 0xf90299c904793a3c, 0x0000bffc
data8 0xf559511d2dc1ed69, 0x0000bffc
data8 0xf2e72afee9bd2aee, 0x0000bffc
data8 0xef39ff1d8a40770e, 0x0000bffc
data8 0xeb8a7a2311c935dc, 0x0000bffc
data8 0xe7d8990dc620012f, 0x0000bffc
data8 0xe560b1e3b86e44b6, 0x0000bffc
data8 0xe1aadb38caee80c4, 0x0000bffc
data8 0xddf2a051f81b76a4, 0x0000bffc
data8 0xdb7678bafcaf4b5f, 0x0000bffc
data8 0xd7ba3a8f0df19bfc, 0x0000bffc
data8 0xd3fb8fdbdd5cebdb, 0x0000bffc
data8 0xd17b191905c35652, 0x0000bffc
data8 0xcdb85d29cefd7121, 0x0000bffc
data8 0xc9f32c3c88221ef6, 0x0000bffc
data8 0xc76e5741a95b5dae, 0x0000bffc
data8 0xc3a506d80d38c718, 0x0000bffc
data8 0xbfd938ccef8b68c1, 0x0000bffc
data8 0xbd4ff63e82eef78c, 0x0000bffc
data8 0xb97ffa2b563865bd, 0x0000bffc
data8 0xb6f3eb3011eddcea, 0x0000bffc
data8 0xb31fb7d64898b3e6, 0x0000bffc
data8 0xb090d63a409e7880, 0x0000bffc
data8 0xacb8623c7ffa4f39, 0x0000bffc
data8 0xa8dd5c83d2e45246, 0x0000bffc
data8 0xa649e998a8d91f2e, 0x0000bffc
data8 0xa26a93fed6faa94f, 0x0000bffc
data8 0x9fd43df079d0db1f, 0x0000bffc
data8 0x9d3cbe69aecac4c2, 0x0000bffc
data8 0x99574f13c570d0fb, 0x0000bffc
data8 0x96bce349bf7ee6c7, 0x0000bffc
data8 0x92d30c9b86cee18e, 0x0000bffc
data8 0x9035adef17c5bd5c, 0x0000bffc
data8 0x8c4765e8e8b5f251, 0x0000bffc
data8 0x89a70da448316ffa, 0x0000bffc
data8 0x85b44a24474af78a, 0x0000bffc
data8 0x8310f17aab5adf70, 0x0000bffc
data8 0x806c6388d0965f29, 0x0000bffc
data8 0xf8e69092bf0c5ead, 0x0000bffb
data8 0xf397608bfd2d90e6, 0x0000bffb
data8 0xee45be24d0eedbc4, 0x0000bffb
data8 0xe646af233db881e9, 0x0000bffb
data8 0xe0eee4e1ce3d06fb, 0x0000bffb
data8 0xdb94a049e6e87a4f, 0x0000bffb
data8 0xd3888ef9a4249f5a, 0x0000bffb
data8 0xce280e6fbac39194, 0x0000bffb
data8 0xc8c50b72319ad574, 0x0000bffb
data8 0xc0abcd39f41e329b, 0x0000bffb
data8 0xbb4279cfa7f9667b, 0x0000bffb
data8 0xb5d69bac77ec398a, 0x0000bffb
data8 0xb068306bf20d6233, 0x0000bffb
data8 0xa83dc1b019ddb6a8, 0x0000bffb
data8 0xa2c8eb1886c2d024, 0x0000bffb
data8 0x9d517ee93f8e16c0, 0x0000bffb
data8 0x97d77aae659b92fb, 0x0000bffb
data8 0x8f9b91da5736d415, 0x0000bffb
data8 0x8a1b06b09b7fd1d1, 0x0000bffb
data8 0x8497daca0a2e077a, 0x0000bffb
data8 0xfe241745a453f10c, 0x0000bffa
data8 0xf3132d6708d723c5, 0x0000bffa
data8 0xe7fcf2e21a0e7d77, 0x0000bffa
data8 0xd75198b04afb8da9, 0x0000bffa
data8 0xcc2dfe1a4a8ca305, 0x0000bffa
data8 0xc10500d63aa65882, 0x0000bffa
data8 0xb5d69bac77ec398a, 0x0000bffa
data8 0xaaa2c95dc66abcde, 0x0000bffa
data8 0x9f6984a342d13101, 0x0000bffa
data8 0x942ac82e5387ac51, 0x0000bffa
data8 0x88e68ea899a0976c, 0x0000bffa
data8 0xefebc4409ccf872e, 0x0000bff9
data8 0xd947b0c6642ef69e, 0x0000bff9
data8 0xc2987d51e043d407, 0x0000bff9
data8 0xabde1eeee6bfd257, 0x0000bff9
data8 0x95188a9917cf2e01, 0x0000bff9
data8 0xfc8f6a777c1b7f1e, 0x0000bff8
data8 0xced727635c59725c, 0x0000bff8
data8 0xa108358a4c904615, 0x0000bff8
data8 0xe644fcbeb3ac9c90, 0x0000bff7
data8 0x8a4bd667bf08e7de, 0x0000bff7
data8 0x0000000000000000 // T[255] Low
data8 0x0000000000000000 // T[255] High
LOCAL_OBJECT_END(T_table)



.section .text
WEAK_LIBM_ENTRY(log2)

{ .mfi
  alloc r32=ar.pfs,1,4,4,0
  // y=frcpa(x)
  frcpa.s1 f6,p0=f1,f8
  // will form significand of 1.5 (to test whether the index is 128 or above)
  mov r24=0xc
}
{.mfi
  nop.m 0
  // normalize x
  fma.s1 f7=f8,f1,f0
  // r2 = pointer to C_1...C_6 followed by T_table
  addl r2 = @ltoff(poly_coeffs), gp;;
}
{.mfi
  // get significand
  getf.sig r25=f8
  // f8 denormal ?
  fclass.m p8,p10=f8,0x9
  // will form significand of 1.5 (to test whether the index is 128 or above)
  shl r24=r24,60
}
{.mfi
  mov r26=0x804
  nop.f 0
  // r23=bias-1
  mov r23=0xfffe;;
}

{.mmf
  getf.exp r29=f8
  // load start address for C_1...C_6 followed by T_table
  ld8 r2=[r2]
  // will continue only for positive normal/denormal numbers
  fclass.nm.unc p12,p7 = f8, 0x19 ;;
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
  // f32=0.5
  setf.exp f32=r23
  nop.f 0
  // r27=bias
  mov r27=0xffff;;
}

{.mmi
  // denormal input, repeat get exponent (after normalization)
  (p8) getf.exp r29=f7
  mov r23=0xff
  // r26=0x80400...0 (threshold for using polynomial approximation)
  shl r26=r26,64-12;;
}

{.mfb
  add r3=48,r2
  // r=1-x*y
  fms.s1 f6=f6,f8,f1
  (p12) br.cond.spnt SPECIAL_LOG2
}
{.mfi
  // load C_4, C_5
  ldfpd f10,f11=[r2],16
  nop.f 0
  cmp.geu p12,p0=r25,r24;;
}

{.mmi
  // load C_6, C_7
  ldfpd f12,f13=[r2],16
  // r27=bias-1 (if index >=128, will add exponent+1)
  (p12) mov r27=0xfffe
  (p8) shr.u r28=r25,63-8;;
}


{.mfi
  // load C_1
  ldfe f14=[r2],32
  fmerge.se f7=f1,f7
  // if first 9 bits after leading 1 are all zero, then p8=1
  cmp.ltu p8,p12=r25,r26
}
{.mfi
  // load C_3
  ldfe f15=[r3]
  nop.f 0
  // get T_index
  and r28=r28,r23;;
}
{.mfi
  // r29=exponent-bias
  sub r29=r29,r27
  // x=1, return 0
  (p6) fma.d.s0 f8=f0,f0,f0
  // get T address
  shladd r2=r28,4,r2
}
{.mfb
  // first 8 bits after leading 1 are all ones ?
  cmp.eq p10,p0=r23,r28
  // if first 8 bits after leading bit are 0, use polynomial approx. only
  (p8) fms.s1 f6=f7,f1,f1
  // x=1, return
  (p6) br.ret.spnt b0;;
}
{.mfi
  // r26=1
  mov r26=1
  // if first 8 bits after leading 1 are all ones, use polynomial approx. only
  (p10) fms.s1 f6=f7,f32,f1
  nop.i 0;;
}

.pred.rel "mutex",p8,p12
{.mmf
  // load T (unless first 9 bits after leading 1 are 0)
  (p12) ldfe f33=[r2]
  // f8=expon - bias
  setf.sig f8=r29
  // set T=0 (if first 9 bits after leading 1 are 0)
  (p8) fma.s1 f33=f0,f0,f0;;
}

{.mfi
  nop.m 0
  // P12=1-0.5*r
  fnma.s1 f32=f32,f6,f1
  // r26=2^{63}
  shl r26=r26,63
}
{.mfi
  nop.m 0
  // r2=r*r
  fma.s1 f7=f6,f6,f0
  nop.i 0;;
}
{.mfi
  // significand(x)=1 ?
  cmp.eq p0,p6=r26,r25
  // P67=C_6+C_7*r
  fma.s1 f13=f13,f6,f12
  nop.i 0
}
{.mfi
  nop.m 0
  // P45=C_4+C_5*r
  fma.s1 f10=f11,f6,f10
  nop.i 0;;
}

{.mfi
  nop.m 0
  // C_1*r
  (p6) fma.s1 f14=f14,f6,f0
  nop.i 0;;
}
{.mfi
  nop.m 0
  // normalize additive term (l=exponent of x)
  fcvt.xf f8=f8
  nop.i 0
}
{.mfi
  nop.m 0
  // P13=1-0.5*r+C_3*r^2
  (p6) fma.s1 f15=f15,f7,f32
  nop.i 0;;
}

{.mfi
  nop.m 0
  // P47=P45+r2*P67
  (p6) fma.s1 f13=f13,f7,f10
  // if significand(x)=1, return exponent (l)
  nop.i 0
}
{.mfi
  nop.m 0
  // r3=r^3
  (p6) fma.s1 f7=f7,f6,f0
  nop.i 0;;
}

{.mfi
  nop.m 0
  // add T+l
  (p6) fma.s1 f8=f8,f1,f33
  nop.i 0
}
{.mfi
  nop.m 0
  // P17=P13+r3*P47
  (p6) fma.s1 f13=f13,f7,f15
  nop.i 0;;
}

{.mfb
  nop.m 0
  // result=T+l+(C_1*r)*P16
  (p6) fma.d.s0 f8=f13,f14,f8
  // return
  br.ret.sptk b0;;
}


SPECIAL_LOG2:
{.mfi
  nop.m 0
  // x=+Infinity ?
  fclass.m p7,p0=f8,0x21
  nop.i 0;;
}
{.mfi
  nop.m 0
  // x=+/-Zero ?
  fclass.m p8,p0=f8,0x7
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
  // log2(+Infinity)=+Infinity
  nop.f 0
  (p7) br.ret.spnt b0;;
}
{.mfi
  (p8) mov GR_Parameter_TAG = 170
  // log2(+/-0)=-infinity, raises Divide by Zero
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
  (p6) mov GR_Parameter_TAG = 171
  // x<0: return NaN, raise Invalid
  (p6) frcpa.s0 f8,p0=f0,f0
  (p6) br.cond.sptk __libm_error_region;;
}


{.mfb
  nop.m 0
  // Remaining cases: NaNs
  fma.d.s0 f8=f8,f1,f0
  br.ret.sptk b0;;
}

WEAK_LIBM_END(log2)
libm_alias_double_other (__log2, log2)
#ifdef SHARED
.symver log2,log2@@GLIBC_2.29
.weak __log2_compat
.set __log2_compat,__log2
.symver __log2_compat,log2@GLIBC_2.2
#endif


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
        stfd [GR_Parameter_Y] = FR_Y,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfd [GR_Parameter_X] = FR_X                  // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
	nop.b 0
}
{ .mib
        stfd [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#         // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
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
