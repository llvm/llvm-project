/* lgammal expanding around zeros.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <fenv_private.h>

static const long double lgamma_zeros[][2] =
  {
    { -0x2.74ff92c01f0d82abec9f315f1ap+0L, -0x7.12c334804d9a79cb5d46094d46p-112L },
    { -0x2.bf6821437b20197995a4b4641fp+0L, 0x5.140b4ff4b7d6069e1bd7acc196p-108L },
    { -0x3.24c1b793cb35efb8be699ad3dap+0L, 0x4.59abab3480539f1c0e926287cp-108L },
    { -0x3.f48e2a8f85fca170d456129123p+0L, -0x6.cc320a4887d1cb4c711828a75ep-108L },
    { -0x4.0a139e16656030c39f0b0de182p+0L, 0xe.d53e84029416e1242006b2b3dp-108L },
    { -0x4.fdd5de9bbabf3510d0aa407698p+0L, -0x8.501d7d78125286f78d1e501f14p-108L },
    { -0x5.021a95fc2db6432a4c56e5953ap+0L, 0xb.2133950fbcf2b01a8b9058dcccp-108L },
    { -0x5.ffa4bd647d0357dd4ed62cbd32p+0L, 0x1.2071c071a2145d2982428f2269p-108L },
    { -0x6.005ac9625f233b607c2d96d164p+0L, 0x7.a347953a96cbf30e1a0db20856p-108L },
    { -0x6.fff2fddae1bbff3d626b65c24p+0L, 0x2.de0bfcff5c457ebcf4d3ad9674p-108L },
    { -0x7.000cff7b7f87adf4482dcdb988p+0L, 0x7.d54d99e35a74d6407b80292df2p-108L },
    { -0x7.fffe5fe05673c3ca9e82b522bp+0L, -0xc.a9d2e8837cd1f14bd3d05002e4p-108L },
    { -0x8.0001a01459fc9f60cb3cec1cecp+0L, -0x8.576677ca538d88084310983b8p-108L },
    { -0x8.ffffd1c425e80ffc864e957494p+0L, 0x1.a6181dfdef1807e3087e4bb163p-104L },
    { -0x9.00002e3bb47d86d6d843fedc34p+0L, -0x1.1deb7ad09ec5e9d6e8ae2d548bp-104L },
    { -0x9.fffffb606bdfdcd062ae77a504p+0L, -0x1.47c69d2eb6f33d170fce38ff818p-104L },
    { -0xa.0000049f93bb9927b45d95e154p+0L, -0x4.1e03086db9146a9287bd4f2172p-108L },
    { -0xa.ffffff9466e9f1b36dacd2adbcp+0L, -0x1.18d05a4e458062f3f95345a4dap-104L },
    { -0xb.0000006b9915315d965a6ffea4p+0L, -0xe.4bea39000dcc1848023c5f6bdcp-112L },
    { -0xb.fffffff7089387387de41acc3cp+0L, -0x1.3c978bd839c8c428b5efcf91ef8p-104L },
    { -0xc.00000008f76c7731567c0f025p+0L, -0xf.387920df5675833859190eb128p-108L },
    { -0xc.ffffffff4f6dcf617f97a5ffc8p+0L, 0xa.82ab72d76f32eaee2d1a42ed5p-108L },
    { -0xd.00000000b092309c06683dd1b8p+0L, -0x1.03e3700857a15c19ac5a611de98p-104L },
    { -0xd.fffffffff36345ab9e184a3e08p+0L, -0x1.d1176dc48e47f62d917973dd45p-104L },
    { -0xe.000000000c9cba545e94e75ec4p+0L, -0x1.718f753e2501e757a17cf2ecbfp-104L },
    { -0xe.ffffffffff28c060c6604ef304p+0L, 0x8.e0762c8ca8361c23e8393919c4p-108L },
    { -0xf.0000000000d73f9f399bd0e42p+0L, -0xf.85e9ee31b0b890744fc0e3fbcp-108L },
    { -0xf.fffffffffff28c060c6621f514p+0L, 0x1.18d1b2eec9d960bd9adc5be5f6p-104L },
    { -0x1.000000000000d73f9f399da1428p+4L, 0x3.406c46e0e88305d2800f0e414cp-104L },
    { -0x1.0ffffffffffff3569c47e7a93ep+4L, -0x1.c46a08a2e008a998ebabb8087fp-104L },
    { -0x1.1000000000000ca963b81856888p+4L, -0x7.6ca5a3a64ec15db0a95caf2cap-108L },
    { -0x1.1fffffffffffff4bec3ce23413p+4L, -0x2.d08b2b726187c841cb92cd5222p-104L },
    { -0x1.20000000000000b413c31dcbec8p+4L, -0x2.4c3b2ffacbb4932f18dceedfd7p-104L },
    { -0x1.2ffffffffffffff685b25cbf5f8p+4L, 0x2.ba3126cd1c7b7a0822d694705cp-104L },
    { -0x1.30000000000000097a4da340a08p+4L, -0x2.b81b7b1f1f001c72bf914141efp-104L },
    { -0x1.3fffffffffffffff86af516ff8p+4L, 0x8.9429818df2a87abafd48248a2p-108L },
    { -0x1.40000000000000007950ae9008p+4L, -0x8.9413ccc8a353fda263f8ce973cp-108L },
    { -0x1.4ffffffffffffffffa391c4249p+4L, 0x3.d5c63022b62b5484ba346524dbp-104L },
    { -0x1.500000000000000005c6e3bdb7p+4L, -0x3.d5c62f55ed5322b2685c5e9a52p-104L },
    { -0x1.5fffffffffffffffffbcc71a49p+4L, -0x2.01eb5aeb96c74d7ad25e060529p-104L },
    { -0x1.6000000000000000004338e5b7p+4L, 0x2.01eb5aec04b2f2eb663e4e3d8ap-104L },
    { -0x1.6ffffffffffffffffffd13c97d8p+4L, -0x1.d38fcc4d08d6fe5aa56ab04308p-104L },
    { -0x1.70000000000000000002ec36828p+4L, 0x1.d38fcc4d090cee2f5d0b69a99cp-104L },
    { -0x1.7fffffffffffffffffffe0d31p+4L, 0x1.972f577cca4b4c8cb1dc14001bp-104L },
    { -0x1.800000000000000000001f2cfp+4L, -0x1.972f577cca4b3442e35f0040b38p-104L },
    { -0x1.8ffffffffffffffffffffec0c3p+4L, -0x3.22e9a0572b1bb5b95f346a92d6p-104L },
    { -0x1.90000000000000000000013f3dp+4L, 0x3.22e9a0572b1bb5c371ddb35617p-104L },
    { -0x1.9ffffffffffffffffffffff3b88p+4L, -0x3.d01cad8d32e386fd783e97296dp-104L },
    { -0x1.a0000000000000000000000c478p+4L, 0x3.d01cad8d32e386fd7c1ab8c1fep-104L },
    { -0x1.afffffffffffffffffffffff8b8p+4L, -0x1.538f48cc5737d5979c39db806c8p-104L },
    { -0x1.b00000000000000000000000748p+4L, 0x1.538f48cc5737d5979c3b3a6bdap-104L },
    { -0x1.bffffffffffffffffffffffffcp+4L, 0x2.862898d42174dcf171470d8c8cp-104L },
    { -0x1.c0000000000000000000000004p+4L, -0x2.862898d42174dcf171470d18bap-104L },
    { -0x1.dp+4L, 0x2.4b3f31686b15af57c61ceecdf4p-104L },
    { -0x1.dp+4L, -0x2.4b3f31686b15af57c61ceecdd1p-104L },
    { -0x1.ep+4L, 0x1.3932c5047d60e60caded4c298ap-108L },
    { -0x1.ep+4L, -0x1.3932c5047d60e60caded4c29898p-108L },
    { -0x1.fp+4L, 0xa.1a6973c1fade2170f7237d36p-116L },
    { -0x1.fp+4L, -0xa.1a6973c1fade2170f7237d36p-116L },
    { -0x2p+4L, 0x5.0d34b9e0fd6f10b87b91be9bp-120L },
    { -0x2p+4L, -0x5.0d34b9e0fd6f10b87b91be9bp-120L },
    { -0x2.1p+4L, 0x2.73024a9ba1aa36a7059bff52e8p-124L },
    { -0x2.1p+4L, -0x2.73024a9ba1aa36a7059bff52e8p-124L },
    { -0x2.2p+4L, 0x1.2710231c0fd7a13f8a2b4af9d68p-128L },
    { -0x2.2p+4L, -0x1.2710231c0fd7a13f8a2b4af9d68p-128L },
    { -0x2.3p+4L, 0x8.6e2ce38b6c8f9419e3fad3f03p-136L },
    { -0x2.3p+4L, -0x8.6e2ce38b6c8f9419e3fad3f03p-136L },
    { -0x2.4p+4L, 0x3.bf30652185952560d71a254e4fp-140L },
    { -0x2.4p+4L, -0x3.bf30652185952560d71a254e4fp-140L },
    { -0x2.5p+4L, 0x1.9ec8d1c94e85af4c78b15c3d8ap-144L },
    { -0x2.5p+4L, -0x1.9ec8d1c94e85af4c78b15c3d8ap-144L },
    { -0x2.6p+4L, 0xa.ea565ce061d57489e9b8527628p-152L },
    { -0x2.6p+4L, -0xa.ea565ce061d57489e9b8527628p-152L },
    { -0x2.7p+4L, 0x4.7a6512692eb37804111dabad3p-156L },
    { -0x2.7p+4L, -0x4.7a6512692eb37804111dabad3p-156L },
    { -0x2.8p+4L, 0x1.ca8ed42a12ae3001a07244abadp-160L },
    { -0x2.8p+4L, -0x1.ca8ed42a12ae3001a07244abadp-160L },
    { -0x2.9p+4L, 0xb.2f30e1ce812063f12e7e8d8d98p-168L },
    { -0x2.9p+4L, -0xb.2f30e1ce812063f12e7e8d8d98p-168L },
    { -0x2.ap+4L, 0x4.42bd49d4c37a0db136489772e4p-172L },
    { -0x2.ap+4L, -0x4.42bd49d4c37a0db136489772e4p-172L },
    { -0x2.bp+4L, 0x1.95db45257e5122dcbae56def37p-176L },
    { -0x2.bp+4L, -0x1.95db45257e5122dcbae56def37p-176L },
    { -0x2.cp+4L, 0x9.3958d81ff63527ecf993f3fb7p-184L },
    { -0x2.cp+4L, -0x9.3958d81ff63527ecf993f3fb7p-184L },
    { -0x2.dp+4L, 0x3.47970e4440c8f1c058bd238c99p-188L },
    { -0x2.dp+4L, -0x3.47970e4440c8f1c058bd238c99p-188L },
    { -0x2.ep+4L, 0x1.240804f65951062ca46e4f25c6p-192L },
    { -0x2.ep+4L, -0x1.240804f65951062ca46e4f25c6p-192L },
    { -0x2.fp+4L, 0x6.36a382849fae6de2d15362d8a4p-200L },
    { -0x2.fp+4L, -0x6.36a382849fae6de2d15362d8a4p-200L },
    { -0x3p+4L, 0x2.123680d6dfe4cf4b9b1bcb9d8cp-204L },
  };

static const long double e_hi = 0x2.b7e151628aed2a6abf7158809dp+0L;
static const long double e_lo = -0xb.0c389d18e9f0c74b25a9587b28p-112L;

/* Coefficients B_2k / 2k(2k-1) of x^-(2k-1) in Stirling's
   approximation to lgamma function.  */

static const long double lgamma_coeff[] =
  {
    0x1.555555555555555555555555558p-4L,
    -0xb.60b60b60b60b60b60b60b60b6p-12L,
    0x3.4034034034034034034034034p-12L,
    -0x2.7027027027027027027027027p-12L,
    0x3.72a3c5631fe46ae1d4e700dca9p-12L,
    -0x7.daac36664f1f207daac36664f2p-12L,
    0x1.a41a41a41a41a41a41a41a41a4p-8L,
    -0x7.90a1b2c3d4e5f708192a3b4c5ep-8L,
    0x2.dfd2c703c0cfff430edfd2c704p-4L,
    -0x1.6476701181f39edbdb9ce625988p+0L,
    0xd.672219167002d3a7a9c886459cp+0L,
    -0x9.cd9292e6660d55b3f712eb9e08p+4L,
    0x8.911a740da740da740da740da74p+8L,
    -0x8.d0cc570e255bf59ff6eec24b48p+12L,
    0xa.8d1044d3708d1c219ee4fdc448p+16L,
    -0xe.8844d8a169abbc406169abbc4p+20L,
    0x1.6d29a0f6433b79890cede624338p+28L,
    -0x2.88a233b3c8cddaba9809357126p+32L,
    0x5.0dde6f27500939a85c40939a86p+36L,
    -0xb.4005bde03d4642a243581714bp+40L,
    0x1.bc8cd6f8f1f755c78753cdb5d6p+48L,
    -0x4.bbebb143bb94de5a0284fa7ec4p+52L,
    0xe.2e1337f5af0bed90b6b0a352d4p+56L,
    -0x2.e78250162b62405ad3e4bfe61bp+64L,
    0xa.5f7eef9e71ac7c80326ab4cc8cp+68L,
    -0x2.83be0395e550213369924971b2p+76L,
  };

#define NCOEFF (sizeof (lgamma_coeff) / sizeof (lgamma_coeff[0]))

/* Polynomial approximations to (|gamma(x)|-1)(x-n)/(x-x0), where n is
   the integer end-point of the half-integer interval containing x and
   x0 is the zero of lgamma in that half-integer interval.  Each
   polynomial is expressed in terms of x-xm, where xm is the midpoint
   of the interval for which the polynomial applies.  */

static const long double poly_coeff[] =
  {
    /* Interval [-2.125, -2] (polynomial degree 21).  */
    -0x1.0b71c5c54d42eb6c17f30b7aa9p+0L,
    -0xc.73a1dc05f34951602554c6d76cp-4L,
    -0x1.ec841408528b51473e6c42f1c58p-4L,
    -0xe.37c9da26fc3c9a3c1844c04b84p-4L,
    -0x1.03cd87c519305703b00b046ce4p-4L,
    -0xe.ae9ada65e09aa7f1c817c91048p-4L,
    0x9.b11855a4864b571b6a4f571c88p-8L,
    -0xe.f28c133e697a95ba2dabb97584p-4L,
    0x2.6ec14a1c586a7ddb6c4be90fe1p-4L,
    -0xf.57cab973e14496f0900851c0d4p-4L,
    0x4.5b0fc25f16b0df37175495c70cp-4L,
    -0xf.f50e59f1a8fb8c402091e3cd3cp-4L,
    0x6.5f5eae1681d1e50e575c3d4d36p-4L,
    -0x1.0d2422dac7ea8a52db6bf0d14fp+0L,
    0x8.820008f221eae5a36e15913bacp-4L,
    -0x1.1f492eec53b9481ea23a7e944ep+0L,
    0xa.cb55b4d662945e8cf1f81ee5b4p-4L,
    -0x1.3616863983e131d7935700ccd48p+0L,
    0xd.43c783ebab66074d18709d5cap-4L,
    -0x1.51d5dbc56bc85976871c6e51f78p+0L,
    0x1.06253af656eb6b2ed998387aabp+0L,
    -0x1.7d910a0aadc63d7a1ef7690dbb8p+0L,
    /* Interval [-2.25, -2.125] (polynomial degree 22).  */
    -0xf.2930890d7d675a80c36afb0fd4p-4L,
    -0xc.a5cfde054eab5c6770daeca684p-4L,
    0x3.9c9e0fdebb07cdf89c61d434adp-4L,
    -0x1.02a5ad35605fcf4af65a67fe8a8p+0L,
    0x9.6e9b1185bb48be9de18d8bbeb8p-4L,
    -0x1.4d8332f3cfbfa116fdf648372cp+0L,
    0x1.1c0c8cb4d9f4b1d495142b53ebp+0L,
    -0x1.c9a6f5ae9130ccfb9b7e39136f8p+0L,
    0x1.d7e9307fd58a2e85209d0e83eap+0L,
    -0x2.921cb3473d96462f22c171712fp+0L,
    0x2.e8d59113b6f3fc1ed3b556b62cp+0L,
    -0x3.cbab931624e3b6cf299cea1213p+0L,
    0x4.7d9f0f05d2c4cf91e41ea1f048p+0L,
    -0x5.ade9cba31affa276fe516135eep+0L,
    0x6.dc983a62cf6ddc935ae3c5b9ap+0L,
    -0x8.8d9ed100b2a7813f82cbd83e3cp+0L,
    0xa.6fa0926892835a9a29c9b8db8p+0L,
    -0xc.ebc90aff4ffe319d70bef0d61p+0L,
    0xf.d69cf50ab226bacece014c0b44p+0L,
    -0x1.389964ac7cfef4578eec028e5c8p+4L,
    0x1.7ff0d2090164e25901f97cab3bp+4L,
    -0x1.e9e6d282da6bd004619d073071p+4L,
    0x2.5d719ab6ad4be8b5c32b0fba2ap+4L,
    /* Interval [-2.375, -2.25] (polynomial degree 24).  */
    -0xd.7d28d505d6181218a25f31d5e4p-4L,
    -0xe.69649a3040985140cdf946827cp-4L,
    0xb.0d74a2827d053a8d4459500f88p-4L,
    -0x1.924b0922853617cac181b097e48p+0L,
    0x1.d49b12bccf0a568582e2dbf8ep+0L,
    -0x3.0898bb7d8c4093e6360d26bbc5p+0L,
    0x4.207a6cac711cb538684f74619ep+0L,
    -0x6.39ee63ea4fb1dcac86ab337e3cp+0L,
    0x8.e2e2556a797b64a1b9328a3978p+0L,
    -0xd.0e83ac82552ee5596df1706ff4p+0L,
    0x1.2e4525e0ce666e48fac68ddcdep+4L,
    -0x1.b8e350d6a8f6597ed2eb3c2eff8p+4L,
    0x2.805cd69b9197ee0089dd1b1c46p+4L,
    -0x3.a42585423e4d00db075f2d687ep+4L,
    0x5.4b4f409f874e2a7dcd8aa4a62ap+4L,
    -0x7.b3c5829962ca1b95535db9cc4ep+4L,
    0xb.33b7b928986ec6b219e2e15a98p+4L,
    -0x1.04b76dec4115106bb16316d9cd8p+8L,
    0x1.7b366d8d46f179d5c5302d6534p+8L,
    -0x2.2799846ddc54813d40da622b99p+8L,
    0x3.2253a862c1078a3ccabac65bebp+8L,
    -0x4.8d92cebc90a4a29816f4952f4ep+8L,
    0x6.9ebb8f9d72c66c80c4f4492e7ap+8L,
    -0xa.2850a483f9ba0e43f5848b5cd8p+8L,
    0xe.e1b6bdce83b27944edab8c428p+8L,
    /* Interval [-2.5, -2.375] (polynomial degree 25).  */
    -0xb.74ea1bcfff94b2c01afba9daa8p-4L,
    -0x1.2a82bd590c37538cab143308e3p+0L,
    0x1.88020f828b966fec66b8648d16p+0L,
    -0x3.32279f040eb694970e9db0308bp+0L,
    0x5.57ac82517767e68a72142041b4p+0L,
    -0x9.c2aedcfe22833de438786dc658p+0L,
    0x1.12c132f1f5577f99dbfb7ecb408p+4L,
    -0x1.ea94e26628a3de3557dc349db8p+4L,
    0x3.66b4ac4fa582f5cbe7e19d10c6p+4L,
    -0x6.0cf746a9cf4cbcb0004cb01f66p+4L,
    0xa.c102ef2c20d5a313cbfd37f5b8p+4L,
    -0x1.31ebff06e8f08f58d1c35eacfdp+8L,
    0x2.1fd6f0c0e788660ba1f1573722p+8L,
    -0x3.c6d760404305e75356a86a11d6p+8L,
    0x6.b6d18e0c31a2ba4d5b5ac78676p+8L,
    -0xb.efaf5426343e6b41a823ed6c44p+8L,
    0x1.53852db2fe01305b9f336d132d8p+12L,
    -0x2.5b977cb2b568382e71ca93a36bp+12L,
    0x4.310d090a6119c7d85a2786a616p+12L,
    -0x7.73a518387ef1d4d04917dfb25cp+12L,
    0xd.3f965798601aabd24bdaa6e68cp+12L,
    -0x1.78db20b0b166480c93cf0031198p+16L,
    0x2.9be0068b65cf13bd1cf71f0eccp+16L,
    -0x4.a221230466b9cd51d5b811d6b6p+16L,
    0x8.f6f8c13e2b52aa3e30a4ce6898p+16L,
    -0x1.02145337ff16b44fa7c2adf7f28p+20L,
    /* Interval [-2.625, -2.5] (polynomial degree 26).  */
    -0x3.d10108c27ebafad533c20eac33p-4L,
    0x1.cd557caff7d2b2085f41dbec538p+0L,
    0x3.819b4856d399520dad9776ebb9p+0L,
    0x6.8505cbad03dc34c5e42e89c4b4p+0L,
    0xb.c1b2e653a9e38f82b3997134a8p+0L,
    0x1.50a53a38f1481381051544750ep+4L,
    0x2.57ae00cbe5232cbeef4e94eb2cp+4L,
    0x4.2b156301b8604db82856d5767p+4L,
    0x7.6989ed23ca3ca751fc9c32eb88p+4L,
    0xd.2dd29765579396f3a456772c44p+4L,
    0x1.76e1c3430eb8630991d1aa8a248p+8L,
    0x2.9a77bf548873743fe65d025f56p+8L,
    0x4.a0d62ed7266389753842d7be74p+8L,
    0x8.3a6184dd32d31ec73fc6f2d37cp+8L,
    0xe.a0ade153a3bf0247db49e11ae8p+8L,
    0x1.a01359fa74d4eaf8858bbc35f68p+12L,
    0x2.e3b0a32845cbc135bae4a5216cp+12L,
    0x5.23012653815fe88456170a7dc6p+12L,
    0x9.21c92dcde748ec199bc9c65738p+12L,
    0x1.03c0f3621b4c67d2d86e5e813d8p+16L,
    0x1.cdc884edcc9f5404f2708551cb8p+16L,
    0x3.35025f0b1624d1ffc86688bf03p+16L,
    0x5.b3bd9562ebf2409c5ce99929ep+16L,
    0xa.1a229b1986d9f89cb80abccfdp+16L,
    0x1.1e69136ebd520146d51837f3308p+20L,
    0x2.2d2738c72449db2524171b9271p+20L,
    0x4.036e80cc6621b836f94f426834p+20L,
    /* Interval [-2.75, -2.625] (polynomial degree 24).  */
    -0x6.b5d252a56e8a75458a27ed1c2ep-4L,
    0x1.28d60383da3ac721aed3c57949p+0L,
    0x1.db6513ada8a66ea77d87d9a796p+0L,
    0x2.e217118f9d348a27f7506c4b4fp+0L,
    0x4.450112c5cbf725a0fb982fc44cp+0L,
    0x6.4af99151eae7810a75a5fceac8p+0L,
    0x9.2db598b4a97a7f69ab7be31128p+0L,
    0xd.62bef9c22471f5f17955733c6p+0L,
    0x1.379f294e412bd6255506135f4a8p+4L,
    0x1.c5827349d8865d858d4f85f3c38p+4L,
    0x2.93a7e7a75b755bbea1785a1349p+4L,
    0x3.bf9bb882afed66a08b22ed7a45p+4L,
    0x5.73c737828d2044aca95fdef33ep+4L,
    0x7.ee46534920f1c81574db260f0ep+4L,
    0xb.891c6b837b513eaf1592fe78ccp+4L,
    0x1.0c775d815bf741526a3dd66ded8p+8L,
    0x1.867ee44cf11f26455a8924a56bp+8L,
    0x2.37fe968baa1018e55cae680f1dp+8L,
    0x3.3a2c557f686679eb5d8e960fd1p+8L,
    0x4.b1ba0539d4d80cc9174738b992p+8L,
    0x6.d3fd80155b6d2211956cb6bc5ap+8L,
    0x9.eb5a96b0ee3d9ca523f5fbc1fp+8L,
    0xe.6b37429c1acc7dc19ef312dda4p+8L,
    0x1.621132d6aa138b203a28e4792fp+12L,
    0x2.09610219270e2ce11a985d4d36p+12L,
    /* Interval [-2.875, -2.75] (polynomial degree 23).  */
    -0x8.a41b1e4f36ff88dc820815607cp-4L,
    0xc.da87d3b69dc0f2f9c6f368b8c8p-4L,
    0x1.1474ad5c36158a7bea04fd30b28p+0L,
    0x1.761ecb90c555df6555b7dbb9ce8p+0L,
    0x1.d279bff9ae291caf6c4b17497f8p+0L,
    0x2.4e5d00559a6e2b9b5d7e35b575p+0L,
    0x2.d57545a75cee8743b1ff6e22b8p+0L,
    0x3.8514eee3aac88b89d2d4ddef4ep+0L,
    0x4.5235e3b6e1891fd9c975383318p+0L,
    0x5.562acdb10eef3c14a780490e3cp+0L,
    0x6.8ec8965c76f0b261bc41b5e532p+0L,
    0x8.15251aca144a98a1e1c0981388p+0L,
    0x9.f08d56ab9e7eee9515a457214cp+0L,
    0xc.3dbbeda2620d5be4fe8621ce6p+0L,
    0xf.0f5bfd65b3feb6d745a2cdbf9cp+0L,
    0x1.28a6ccd8dd27fb90fcaa31d37dp+4L,
    0x1.6d0a3a3091c3d64cfd1a3c5769p+4L,
    0x1.c1570107e02d5ab0b8bea6d6c98p+4L,
    0x2.28fc9b295b583fa469de7acceap+4L,
    0x2.a8a4cac0217026bbdbce34f4adp+4L,
    0x3.4532c98bce75262ac0ede53edep+4L,
    0x4.062fd9ba18e00e55c25a4f0688p+4L,
    0x5.22e00e6d9846a3451fad5587f8p+4L,
    0x6.5d0f7ce92a0bf928d4a30e92c6p+4L,
    /* Interval [-3, -2.875] (polynomial degree 22).  */
    -0xa.046d667e468f3e44dcae1afcc8p-4L,
    0x9.70b88dcc006c214d8d996fdf7p-4L,
    0xa.a8a39421c86d3ff24931a093c4p-4L,
    0xd.2f4d1363f324da2b357c850124p-4L,
    0xd.ca9aa1a3a5c00de11bf5d7047p-4L,
    0xf.cf09c31eeb52a45dfb25e50ebcp-4L,
    0x1.04b133a39ed8a096914cc78812p+0L,
    0x1.22b547a06edda9447f516a2ee7p+0L,
    0x1.2c57fce7db86a91c8d0f12077b8p+0L,
    0x1.4aade4894708fb8b78365e9bf88p+0L,
    0x1.579c8b7b67ec5179ecc4e9c7dp+0L,
    0x1.776820e7fc7361c50e7ef40a88p+0L,
    0x1.883ab28c72ef238ada6c480ab18p+0L,
    0x1.aa2ef6e1d11b9fcea06a1dcab1p+0L,
    0x1.bf4ad50f2dd2aeb02395ea08648p+0L,
    0x1.e40206a5477615838e02279dfc8p+0L,
    0x1.fdcbcfd4b0777fb173b85d5b398p+0L,
    0x2.25e32b3b3c89e833029169a17bp+0L,
    0x2.44ce344ff0bda6570fe3d0a76dp+0L,
    0x2.70bfba6fa079faf2dbf31d2216p+0L,
    0x2.953e22a97725cc179ad21024fap+0L,
    0x2.d8ccc51524659a499eee0f267p+0L,
    0x3.080fbb09c14936c2171c8a51bcp+0L,
  };

static const size_t poly_deg[] =
  {
    21,
    22,
    24,
    25,
    26,
    24,
    23,
    22,
  };

static const size_t poly_end[] =
  {
    21,
    44,
    69,
    95,
    122,
    147,
    171,
    194,
  };

/* Compute sin (pi * X) for -0.25 <= X <= 0.5.  */

static long double
lg_sinpi (long double x)
{
  if (x <= 0.25L)
    return __sinl (M_PIl * x);
  else
    return __cosl (M_PIl * (0.5L - x));
}

/* Compute cos (pi * X) for -0.25 <= X <= 0.5.  */

static long double
lg_cospi (long double x)
{
  if (x <= 0.25L)
    return __cosl (M_PIl * x);
  else
    return __sinl (M_PIl * (0.5L - x));
}

/* Compute cot (pi * X) for -0.25 <= X <= 0.5.  */

static long double
lg_cotpi (long double x)
{
  return lg_cospi (x) / lg_sinpi (x);
}

/* Compute lgamma of a negative argument -48 < X < -2, setting
   *SIGNGAMP accordingly.  */

long double
__lgamma_negl (long double x, int *signgamp)
{
  /* Determine the half-integer region X lies in, handle exact
     integers and determine the sign of the result.  */
  int i = floorl (-2 * x);
  if ((i & 1) == 0 && i == -2 * x)
    return 1.0L / 0.0L;
  long double xn = ((i & 1) == 0 ? -i / 2 : (-i - 1) / 2);
  i -= 4;
  *signgamp = ((i & 2) == 0 ? -1 : 1);

  SET_RESTORE_ROUNDL (FE_TONEAREST);

  /* Expand around the zero X0 = X0_HI + X0_LO.  */
  long double x0_hi = lgamma_zeros[i][0], x0_lo = lgamma_zeros[i][1];
  long double xdiff = x - x0_hi - x0_lo;

  /* For arguments in the range -3 to -2, use polynomial
     approximations to an adjusted version of the gamma function.  */
  if (i < 2)
    {
      int j = floorl (-8 * x) - 16;
      long double xm = (-33 - 2 * j) * 0.0625L;
      long double x_adj = x - xm;
      size_t deg = poly_deg[j];
      size_t end = poly_end[j];
      long double g = poly_coeff[end];
      for (size_t j = 1; j <= deg; j++)
	g = g * x_adj + poly_coeff[end - j];
      return __log1pl (g * xdiff / (x - xn));
    }

  /* The result we want is log (sinpi (X0) / sinpi (X))
     + log (gamma (1 - X0) / gamma (1 - X)).  */
  long double x_idiff = fabsl (xn - x), x0_idiff = fabsl (xn - x0_hi - x0_lo);
  long double log_sinpi_ratio;
  if (x0_idiff < x_idiff * 0.5L)
    /* Use log not log1p to avoid inaccuracy from log1p of arguments
       close to -1.  */
    log_sinpi_ratio = __ieee754_logl (lg_sinpi (x0_idiff)
				      / lg_sinpi (x_idiff));
  else
    {
      /* Use log1p not log to avoid inaccuracy from log of arguments
	 close to 1.  X0DIFF2 has positive sign if X0 is further from
	 XN than X is from XN, negative sign otherwise.  */
      long double x0diff2 = ((i & 1) == 0 ? xdiff : -xdiff) * 0.5L;
      long double sx0d2 = lg_sinpi (x0diff2);
      long double cx0d2 = lg_cospi (x0diff2);
      log_sinpi_ratio = __log1pl (2 * sx0d2
				  * (-sx0d2 + cx0d2 * lg_cotpi (x_idiff)));
    }

  long double log_gamma_ratio;
  long double y0 = 1 - x0_hi;
  long double y0_eps = -x0_hi + (1 - y0) - x0_lo;
  long double y = 1 - x;
  long double y_eps = -x + (1 - y);
  /* We now wish to compute LOG_GAMMA_RATIO
     = log (gamma (Y0 + Y0_EPS) / gamma (Y + Y_EPS)).  XDIFF
     accurately approximates the difference Y0 + Y0_EPS - Y -
     Y_EPS.  Use Stirling's approximation.  First, we may need to
     adjust into the range where Stirling's approximation is
     sufficiently accurate.  */
  long double log_gamma_adj = 0;
  if (i < 18)
    {
      int n_up = (19 - i) / 2;
      long double ny0, ny0_eps, ny, ny_eps;
      ny0 = y0 + n_up;
      ny0_eps = y0 - (ny0 - n_up) + y0_eps;
      y0 = ny0;
      y0_eps = ny0_eps;
      ny = y + n_up;
      ny_eps = y - (ny - n_up) + y_eps;
      y = ny;
      y_eps = ny_eps;
      long double prodm1 = __lgamma_productl (xdiff, y - n_up, y_eps, n_up);
      log_gamma_adj = -__log1pl (prodm1);
    }
  long double log_gamma_high
    = (xdiff * __log1pl ((y0 - e_hi - e_lo + y0_eps) / e_hi)
       + (y - 0.5L + y_eps) * __log1pl (xdiff / y) + log_gamma_adj);
  /* Compute the sum of (B_2k / 2k(2k-1))(Y0^-(2k-1) - Y^-(2k-1)).  */
  long double y0r = 1 / y0, yr = 1 / y;
  long double y0r2 = y0r * y0r, yr2 = yr * yr;
  long double rdiff = -xdiff / (y * y0);
  long double bterm[NCOEFF];
  long double dlast = rdiff, elast = rdiff * yr * (yr + y0r);
  bterm[0] = dlast * lgamma_coeff[0];
  for (size_t j = 1; j < NCOEFF; j++)
    {
      long double dnext = dlast * y0r2 + elast;
      long double enext = elast * yr2;
      bterm[j] = dnext * lgamma_coeff[j];
      dlast = dnext;
      elast = enext;
    }
  long double log_gamma_low = 0;
  for (size_t j = 0; j < NCOEFF; j++)
    log_gamma_low += bterm[NCOEFF - 1 - j];
  log_gamma_ratio = log_gamma_high + log_gamma_low;

  return log_sinpi_ratio + log_gamma_ratio;
}
