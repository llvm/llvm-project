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

static const _Float128 lgamma_zeros[][2] =
  {
    { L(-0x2.74ff92c01f0d82abec9f315f1a08p+0), L(0xe.d3ccb7fb2658634a2b9f6b2ba81p-116) },
    { L(-0x2.bf6821437b20197995a4b4641eaep+0), L(-0xb.f4b00b4829f961e428533e6ad048p-116) },
    { L(-0x3.24c1b793cb35efb8be699ad3d9bap+0), L(-0x6.5454cb7fac60e3f16d9d7840c2ep-116) },
    { L(-0x3.f48e2a8f85fca170d4561291236cp+0), L(-0xc.320a4887d1cb4c711828a75d5758p-116) },
    { L(-0x4.0a139e16656030c39f0b0de18114p+0), L(0x1.53e84029416e1242006b2b3d1cfp-112) },
    { L(-0x4.fdd5de9bbabf3510d0aa40769884p+0), L(-0x1.01d7d78125286f78d1e501f14966p-112) },
    { L(-0x5.021a95fc2db6432a4c56e595394cp+0), L(-0x1.ecc6af0430d4fe5746fa7233356fp-112) },
    { L(-0x5.ffa4bd647d0357dd4ed62cbd31ecp+0), L(-0x1.f8e3f8e5deba2d67dbd70dd96ce1p-112) },
    { L(-0x6.005ac9625f233b607c2d96d16384p+0), L(-0x1.cb86ac569340cf1e5f24df7aab7bp-112) },
    { L(-0x6.fff2fddae1bbff3d626b65c23fd4p+0), L(0x1.e0bfcff5c457ebcf4d3ad9674167p-112) },
    { L(-0x7.000cff7b7f87adf4482dcdb98784p+0), L(0x1.54d99e35a74d6407b80292df199fp-112) },
    { L(-0x7.fffe5fe05673c3ca9e82b522b0ccp+0), L(0x1.62d177c832e0eb42c2faffd1b145p-112) },
    { L(-0x8.0001a01459fc9f60cb3cec1cec88p+0), L(0x2.8998835ac7277f7bcef67c47f188p-112) },
    { L(-0x8.ffffd1c425e80ffc864e95749258p+0), L(-0x1.e7e20210e7f81cf781b44e9d2b02p-112) },
    { L(-0x9.00002e3bb47d86d6d843fedc352p+0), L(0x2.14852f613a16291751d2ab751f7ep-112) },
    { L(-0x9.fffffb606bdfdcd062ae77a50548p+0), L(0x3.962d1490cc2e8f031c7007eaa1ap-116) },
    { L(-0xa.0000049f93bb9927b45d95e1544p+0), L(-0x1.e03086db9146a9287bd4f2172d5ap-112) },
    { L(-0xa.ffffff9466e9f1b36dacd2adbd18p+0), L(-0xd.05a4e458062f3f95345a4d9c9b6p-116) },
    { L(-0xb.0000006b9915315d965a6ffea41p+0), L(0x1.b415c6fff233e7b7fdc3a094246fp-112) },
    { L(-0xb.fffffff7089387387de41acc3d4p+0), L(0x3.687427c6373bd74a10306e10a28ep-112) },
    { L(-0xc.00000008f76c7731567c0f0250fp+0), L(-0x3.87920df5675833859190eb128ef6p-112) },
    { L(-0xc.ffffffff4f6dcf617f97a5ffc758p+0), L(0x2.ab72d76f32eaee2d1a42ed515d3ap-116) },
    { L(-0xd.00000000b092309c06683dd1b9p+0), L(-0x3.e3700857a15c19ac5a611de9688ap-112) },
    { L(-0xd.fffffffff36345ab9e184a3e09dp+0), L(-0x1.176dc48e47f62d917973dd44e553p-112) },
    { L(-0xe.000000000c9cba545e94e75ec57p+0), L(-0x1.8f753e2501e757a17cf2ecbeeb89p-112) },
    { L(-0xe.ffffffffff28c060c6604ef3037p+0), L(-0x1.f89d37357c9e3dc17c6c6e63becap-112) },
    { L(-0xf.0000000000d73f9f399bd0e420f8p+0), L(-0x5.e9ee31b0b890744fc0e3fbc01048p-116) },
    { L(-0xf.fffffffffff28c060c6621f512e8p+0), L(0xd.1b2eec9d960bd9adc5be5f5fa5p-116) },
    { L(-0x1.000000000000d73f9f399da1424cp+4), L(0x6.c46e0e88305d2800f0e414c506a8p-116) },
    { L(-0x1.0ffffffffffff3569c47e7a93e1cp+4), L(-0x4.6a08a2e008a998ebabb8087efa2cp-112) },
    { L(-0x1.1000000000000ca963b818568887p+4), L(-0x6.ca5a3a64ec15db0a95caf2c9ffb4p-112) },
    { L(-0x1.1fffffffffffff4bec3ce234132dp+4), L(-0x8.b2b726187c841cb92cd5221e444p-116) },
    { L(-0x1.20000000000000b413c31dcbeca5p+4), L(0x3.c4d005344b6cd0e7231120294abcp-112) },
    { L(-0x1.2ffffffffffffff685b25cbf5f54p+4), L(-0x5.ced932e38485f7dd296b8fa41448p-112) },
    { L(-0x1.30000000000000097a4da340a0acp+4), L(0x7.e484e0e0ffe38d406ebebe112f88p-112) },
    { L(-0x1.3fffffffffffffff86af516ff7f7p+4), L(-0x6.bd67e720d57854502b7db75e1718p-112) },
    { L(-0x1.40000000000000007950ae900809p+4), L(0x6.bec33375cac025d9c073168c5d9p-112) },
    { L(-0x1.4ffffffffffffffffa391c4248c3p+4), L(0x5.c63022b62b5484ba346524db607p-112) },
    { L(-0x1.500000000000000005c6e3bdb73dp+4), L(-0x5.c62f55ed5322b2685c5e9a51e6a8p-112) },
    { L(-0x1.5fffffffffffffffffbcc71a492p+4), L(-0x1.eb5aeb96c74d7ad25e060528fb5p-112) },
    { L(-0x1.6000000000000000004338e5b6ep+4), L(0x1.eb5aec04b2f2eb663e4e3d8a018cp-112) },
    { L(-0x1.6ffffffffffffffffffd13c97d9dp+4), L(-0x3.8fcc4d08d6fe5aa56ab04307ce7ep-112) },
    { L(-0x1.70000000000000000002ec368263p+4), L(0x3.8fcc4d090cee2f5d0b69a99c353cp-112) },
    { L(-0x1.7fffffffffffffffffffe0d30fe7p+4), L(0x7.2f577cca4b4c8cb1dc14001ac5ecp-112) },
    { L(-0x1.800000000000000000001f2cf019p+4), L(-0x7.2f577cca4b3442e35f0040b3b9e8p-112) },
    { L(-0x1.8ffffffffffffffffffffec0c332p+4), L(-0x2.e9a0572b1bb5b95f346a92d67a6p-112) },
    { L(-0x1.90000000000000000000013f3ccep+4), L(0x2.e9a0572b1bb5c371ddb3561705ap-112) },
    { L(-0x1.9ffffffffffffffffffffff3b8bdp+4), L(-0x1.cad8d32e386fd783e97296d63dcbp-116) },
    { L(-0x1.a0000000000000000000000c4743p+4), L(0x1.cad8d32e386fd7c1ab8c1fe34c0ep-116) },
    { L(-0x1.afffffffffffffffffffffff8b95p+4), L(-0x3.8f48cc5737d5979c39db806c5406p-112) },
    { L(-0x1.b00000000000000000000000746bp+4), L(0x3.8f48cc5737d5979c3b3a6bda06f6p-112) },
    { L(-0x1.bffffffffffffffffffffffffbd8p+4), L(0x6.2898d42174dcf171470d8c8c6028p-112) },
    { L(-0x1.c000000000000000000000000428p+4), L(-0x6.2898d42174dcf171470d18ba412cp-112) },
    { L(-0x1.cfffffffffffffffffffffffffdbp+4), L(-0x4.c0ce9794ea50a839e311320bde94p-112) },
    { L(-0x1.d000000000000000000000000025p+4), L(0x4.c0ce9794ea50a839e311322f7cf8p-112) },
    { L(-0x1.dfffffffffffffffffffffffffffp+4), L(0x3.932c5047d60e60caded4c298a174p-112) },
    { L(-0x1.e000000000000000000000000001p+4), L(-0x3.932c5047d60e60caded4c298973ap-112) },
    { L(-0x1.fp+4), L(0xa.1a6973c1fade2170f7237d35fe3p-116) },
    { L(-0x1.fp+4), L(-0xa.1a6973c1fade2170f7237d35fe08p-116) },
    { L(-0x2p+4), L(0x5.0d34b9e0fd6f10b87b91be9aff1p-120) },
    { L(-0x2p+4), L(-0x5.0d34b9e0fd6f10b87b91be9aff0cp-120) },
    { L(-0x2.1p+4), L(0x2.73024a9ba1aa36a7059bff52e844p-124) },
    { L(-0x2.1p+4), L(-0x2.73024a9ba1aa36a7059bff52e844p-124) },
    { L(-0x2.2p+4), L(0x1.2710231c0fd7a13f8a2b4af9d6b7p-128) },
    { L(-0x2.2p+4), L(-0x1.2710231c0fd7a13f8a2b4af9d6b7p-128) },
    { L(-0x2.3p+4), L(0x8.6e2ce38b6c8f9419e3fad3f0312p-136) },
    { L(-0x2.3p+4), L(-0x8.6e2ce38b6c8f9419e3fad3f0312p-136) },
    { L(-0x2.4p+4), L(0x3.bf30652185952560d71a254e4eb8p-140) },
    { L(-0x2.4p+4), L(-0x3.bf30652185952560d71a254e4eb8p-140) },
    { L(-0x2.5p+4), L(0x1.9ec8d1c94e85af4c78b15c3d89d3p-144) },
    { L(-0x2.5p+4), L(-0x1.9ec8d1c94e85af4c78b15c3d89d3p-144) },
    { L(-0x2.6p+4), L(0xa.ea565ce061d57489e9b85276274p-152) },
    { L(-0x2.6p+4), L(-0xa.ea565ce061d57489e9b85276274p-152) },
    { L(-0x2.7p+4), L(0x4.7a6512692eb37804111dabad30ecp-156) },
    { L(-0x2.7p+4), L(-0x4.7a6512692eb37804111dabad30ecp-156) },
    { L(-0x2.8p+4), L(0x1.ca8ed42a12ae3001a07244abad2bp-160) },
    { L(-0x2.8p+4), L(-0x1.ca8ed42a12ae3001a07244abad2bp-160) },
    { L(-0x2.9p+4), L(0xb.2f30e1ce812063f12e7e8d8d96e8p-168) },
    { L(-0x2.9p+4), L(-0xb.2f30e1ce812063f12e7e8d8d96e8p-168) },
    { L(-0x2.ap+4), L(0x4.42bd49d4c37a0db136489772e428p-172) },
    { L(-0x2.ap+4), L(-0x4.42bd49d4c37a0db136489772e428p-172) },
    { L(-0x2.bp+4), L(0x1.95db45257e5122dcbae56def372p-176) },
    { L(-0x2.bp+4), L(-0x1.95db45257e5122dcbae56def372p-176) },
    { L(-0x2.cp+4), L(0x9.3958d81ff63527ecf993f3fb6f48p-184) },
    { L(-0x2.cp+4), L(-0x9.3958d81ff63527ecf993f3fb6f48p-184) },
    { L(-0x2.dp+4), L(0x3.47970e4440c8f1c058bd238c9958p-188) },
    { L(-0x2.dp+4), L(-0x3.47970e4440c8f1c058bd238c9958p-188) },
    { L(-0x2.ep+4), L(0x1.240804f65951062ca46e4f25c608p-192) },
    { L(-0x2.ep+4), L(-0x1.240804f65951062ca46e4f25c608p-192) },
    { L(-0x2.fp+4), L(0x6.36a382849fae6de2d15362d8a394p-200) },
    { L(-0x2.fp+4), L(-0x6.36a382849fae6de2d15362d8a394p-200) },
    { L(-0x3p+4), L(0x2.123680d6dfe4cf4b9b1bcb9d8bdcp-204) },
    { L(-0x3p+4), L(-0x2.123680d6dfe4cf4b9b1bcb9d8bdcp-204) },
    { L(-0x3.1p+4), L(0xa.d21786ff5842eca51fea0870919p-212) },
    { L(-0x3.1p+4), L(-0xa.d21786ff5842eca51fea0870919p-212) },
    { L(-0x3.2p+4), L(0x3.766dedc259af040be140a68a6c04p-216) },
  };

static const _Float128 e_hi = L(0x2.b7e151628aed2a6abf7158809cf4p+0);
static const _Float128 e_lo = L(0xf.3c762e7160f38b4da56a784d9048p-116);


/* Coefficients B_2k / 2k(2k-1) of x^-(2k-1) in Stirling's
   approximation to lgamma function.  */

static const _Float128 lgamma_coeff[] =
  {
    L(0x1.5555555555555555555555555555p-4),
    L(-0xb.60b60b60b60b60b60b60b60b60b8p-12),
    L(0x3.4034034034034034034034034034p-12),
    L(-0x2.7027027027027027027027027028p-12),
    L(0x3.72a3c5631fe46ae1d4e700dca8f2p-12),
    L(-0x7.daac36664f1f207daac36664f1f4p-12),
    L(0x1.a41a41a41a41a41a41a41a41a41ap-8),
    L(-0x7.90a1b2c3d4e5f708192a3b4c5d7p-8),
    L(0x2.dfd2c703c0cfff430edfd2c703cp-4),
    L(-0x1.6476701181f39edbdb9ce625987dp+0),
    L(0xd.672219167002d3a7a9c886459cp+0),
    L(-0x9.cd9292e6660d55b3f712eb9e07c8p+4),
    L(0x8.911a740da740da740da740da741p+8),
    L(-0x8.d0cc570e255bf59ff6eec24b49p+12),
    L(0xa.8d1044d3708d1c219ee4fdc446ap+16),
    L(-0xe.8844d8a169abbc406169abbc406p+20),
    L(0x1.6d29a0f6433b79890cede62433b8p+28),
    L(-0x2.88a233b3c8cddaba9809357125d8p+32),
    L(0x5.0dde6f27500939a85c40939a85c4p+36),
    L(-0xb.4005bde03d4642a243581714af68p+40),
    L(0x1.bc8cd6f8f1f755c78753cdb5d5c9p+48),
    L(-0x4.bbebb143bb94de5a0284fa7ec424p+52),
    L(0xe.2e1337f5af0bed90b6b0a352d4fp+56),
    L(-0x2.e78250162b62405ad3e4bfe61b38p+64),
    L(0xa.5f7eef9e71ac7c80326ab4cc8bfp+68),
    L(-0x2.83be0395e550213369924971b21ap+76),
    L(0xa.8ebfe48da17dd999790760b0cep+80),
  };

#define NCOEFF (sizeof (lgamma_coeff) / sizeof (lgamma_coeff[0]))

/* Polynomial approximations to (|gamma(x)|-1)(x-n)/(x-x0), where n is
   the integer end-point of the half-integer interval containing x and
   x0 is the zero of lgamma in that half-integer interval.  Each
   polynomial is expressed in terms of x-xm, where xm is the midpoint
   of the interval for which the polynomial applies.  */

static const _Float128 poly_coeff[] =
  {
    /* Interval [-2.125, -2] (polynomial degree 23).  */
    L(-0x1.0b71c5c54d42eb6c17f30b7aa8f5p+0),
    L(-0xc.73a1dc05f34951602554c6d7506p-4),
    L(-0x1.ec841408528b51473e6c425ee5ffp-4),
    L(-0xe.37c9da26fc3c9a3c1844c8c7f1cp-4),
    L(-0x1.03cd87c519305703b021fa33f827p-4),
    L(-0xe.ae9ada65e09aa7f1c75216128f58p-4),
    L(0x9.b11855a4864b5731cf85736015a8p-8),
    L(-0xe.f28c133e697a95c28607c9701dep-4),
    L(0x2.6ec14a1c586a72a7cc33ee569d6ap-4),
    L(-0xf.57cab973e14464a262fc24723c38p-4),
    L(0x4.5b0fc25f16e52997b2886bbae808p-4),
    L(-0xf.f50e59f1a9b56e76e988dac9ccf8p-4),
    L(0x6.5f5eae15e9a93369e1d85146c6fcp-4),
    L(-0x1.0d2422daac459e33e0994325ed23p+0),
    L(0x8.82000a0e7401fb1117a0e6606928p-4),
    L(-0x1.1f492f178a3f1b19f58a2ca68e55p+0),
    L(0xa.cb545f949899a04c160b19389abp-4),
    L(-0x1.36165a1b155ba3db3d1b77caf498p+0),
    L(0xd.44c5d5576f74302e5cf79e183eep-4),
    L(-0x1.51f22e0cdd33d3d481e326c02f3ep+0),
    L(0xf.f73a349c08244ac389c007779bfp-4),
    L(-0x1.73317bf626156ba716747c4ca866p+0),
    L(0x1.379c3c97b9bc71e1c1c4802dd657p+0),
    L(-0x1.a72a351c54f902d483052000f5dfp+0),
    /* Interval [-2.25, -2.125] (polynomial degree 24).  */
    L(-0xf.2930890d7d675a80c36afb0fd5e8p-4),
    L(-0xc.a5cfde054eab5c6770daeca577f8p-4),
    L(0x3.9c9e0fdebb07cdf89c61d41c9238p-4),
    L(-0x1.02a5ad35605fcf4af65a6dbacb84p+0),
    L(0x9.6e9b1185bb48be9de1918e00a2e8p-4),
    L(-0x1.4d8332f3cfbfa116fd611e9ce90dp+0),
    L(0x1.1c0c8cb4d9f4b1d490e1a41fae4dp+0),
    L(-0x1.c9a6f5ae9130cd0299e293a42714p+0),
    L(0x1.d7e9307fd58a2ea997f29573a112p+0),
    L(-0x2.921cb3473d96178ca2a11d2a8d46p+0),
    L(0x2.e8d59113b6f3409ff8db226e9988p+0),
    L(-0x3.cbab931625a1ae2b26756817f264p+0),
    L(0x4.7d9f0f05d5296d18663ca003912p+0),
    L(-0x5.ade9cba12a14ea485667b7135bbp+0),
    L(0x6.dc983a5da74fb48e767b7fec0a3p+0),
    L(-0x8.8d9ed454ae31d9e138dd8ee0d1a8p+0),
    L(0xa.6fa099d4e7c202e0c0fd6ed8492p+0),
    L(-0xc.ebc552a8090a0f0115e92d4ebbc8p+0),
    L(0xf.d695e4772c0d829b53fba9ca5568p+0),
    L(-0x1.38c32ae38e5e9eb79b2a4c5570a9p+4),
    L(0x1.8035145646cfab49306d0999a51bp+4),
    L(-0x1.d930adbb03dd342a4c2a8c4e1af6p+4),
    L(0x2.45c2edb1b4943ddb3686cd9c6524p+4),
    L(-0x2.e818ebbfafe2f916fa21abf7756p+4),
    L(0x3.9804ce51d0fb9a430a711fd7307p+4),
    /* Interval [-2.375, -2.25] (polynomial degree 25).  */
    L(-0xd.7d28d505d6181218a25f31d5e45p-4),
    L(-0xe.69649a3040985140cdf946829fap-4),
    L(0xb.0d74a2827d053a8d44595012484p-4),
    L(-0x1.924b0922853617cac181afbc08ddp+0),
    L(0x1.d49b12bccf0a568582e2d3c410f3p+0),
    L(-0x3.0898bb7d8c4093e636279c791244p+0),
    L(0x4.207a6cac711cb53868e8a5057eep+0),
    L(-0x6.39ee63ea4fb1dcab0c9144bf3ddcp+0),
    L(0x8.e2e2556a797b649bf3f53bd26718p+0),
    L(-0xd.0e83ac82552ef12af508589e7a8p+0),
    L(0x1.2e4525e0ce6670563c6484a82b05p+4),
    L(-0x1.b8e350d6a8f2b222fa390a57c23dp+4),
    L(0x2.805cd69b919087d8a80295892c2cp+4),
    L(-0x3.a42585424a1b7e64c71743ab014p+4),
    L(0x5.4b4f409f98de49f7bfb03c05f984p+4),
    L(-0x7.b3c5827fbe934bc820d6832fb9fcp+4),
    L(0xb.33b7b90cc96c425526e0d0866e7p+4),
    L(-0x1.04b77047ac4f59ee3775ca10df0dp+8),
    L(0x1.7b366f5e94a34f41386eac086313p+8),
    L(-0x2.2797338429385c9849ca6355bfc2p+8),
    L(0x3.225273cf92a27c9aac1b35511256p+8),
    L(-0x4.8f078aa48afe6cb3a4e89690f898p+8),
    L(0x6.9f311d7b6654fc1d0b5195141d04p+8),
    L(-0x9.a0c297b6b4621619ca9bacc48ed8p+8),
    L(0xe.ce1f06b6f90d92138232a76e4cap+8),
    L(-0x1.5b0e6806fa064daf011613e43b17p+12),
    /* Interval [-2.5, -2.375] (polynomial degree 27).  */
    L(-0xb.74ea1bcfff94b2c01afba9daa7d8p-4),
    L(-0x1.2a82bd590c37538cab143308de4dp+0),
    L(0x1.88020f828b966fec66b8649fd6fcp+0),
    L(-0x3.32279f040eb694970e9db24863dcp+0),
    L(0x5.57ac82517767e68a721005853864p+0),
    L(-0x9.c2aedcfe22833de43834a0a6cc4p+0),
    L(0x1.12c132f1f5577f99e1a0ed3538e1p+4),
    L(-0x1.ea94e26628a3de3597f7bb55a948p+4),
    L(0x3.66b4ac4fa582f58b59f96b2f7c7p+4),
    L(-0x6.0cf746a9cf4cba8c39afcc73fc84p+4),
    L(0xa.c102ef2c20d75a342197df7fedf8p+4),
    L(-0x1.31ebff06e8f14626782df58db3b6p+8),
    L(0x2.1fd6f0c0e710994e059b9dbdb1fep+8),
    L(-0x3.c6d76040407f447f8b5074f07706p+8),
    L(0x6.b6d18e0d8feb4c2ef5af6a40ed18p+8),
    L(-0xb.efaf542c529f91e34217f24ae6a8p+8),
    L(0x1.53852d873210e7070f5d9eb2296p+12),
    L(-0x2.5b977c0ddc6d540717173ac29fc8p+12),
    L(0x4.310d452ae05100eff1e02343a724p+12),
    L(-0x7.73a5d8f20c4f986a7dd1912b2968p+12),
    L(0xd.3f5ea2484f3fca15eab1f4d1a218p+12),
    L(-0x1.78d18aac156d1d93a2ffe7e08d3fp+16),
    L(0x2.9df49ca75e5b567f5ea3e47106cp+16),
    L(-0x4.a7149af8961a08aa7c3233b5bb94p+16),
    L(0x8.3db10ffa742c707c25197d989798p+16),
    L(-0xe.a26d6dd023cadd02041a049ec368p+16),
    L(0x1.c825d90514e7c57c7fa5316f947cp+20),
    L(-0x3.34bb81e5a0952df8ca1abdc6684cp+20),
    /* Interval [-2.625, -2.5] (polynomial degree 28).  */
    L(-0x3.d10108c27ebafad533c20eac32bp-4),
    L(0x1.cd557caff7d2b2085f41dbec5106p+0),
    L(0x3.819b4856d399520dad9776ea2cacp+0),
    L(0x6.8505cbad03dc34c5e42e8b12eb78p+0),
    L(0xb.c1b2e653a9e38f82b399c94e7f08p+0),
    L(0x1.50a53a38f148138105124df65419p+4),
    L(0x2.57ae00cbe5232cbeeed34d89727ap+4),
    L(0x4.2b156301b8604db85a601544bfp+4),
    L(0x7.6989ed23ca3ca7579b3462592b5cp+4),
    L(0xd.2dd2976557939517f831f5552cc8p+4),
    L(0x1.76e1c3430eb860969bce40cd494p+8),
    L(0x2.9a77bf5488742466db3a2c7c1ec6p+8),
    L(0x4.a0d62ed7266e8eb36f725a8ebcep+8),
    L(0x8.3a6184dd3021067df2f8b91e99c8p+8),
    L(0xe.a0ade1538245bf55d39d7e436b1p+8),
    L(0x1.a01359fae8617b5826dd74428e9p+12),
    L(0x2.e3b0a32caae77251169acaca1ad4p+12),
    L(0x5.2301257c81589f62b38fb5993ee8p+12),
    L(0x9.21c9275db253d4e719b73b18cb9p+12),
    L(0x1.03c104bc96141cda3f3fa4b112bcp+16),
    L(0x1.cdc8ed65119196a08b0c78f1445p+16),
    L(0x3.34f31d2eaacf34382cdb0073572ap+16),
    L(0x5.b37628cadf12bf0000907d0ef294p+16),
    L(0xa.22d8b332c0b1e6a616f425dfe5ap+16),
    L(0x1.205b01444804c3ff922cd78b4c42p+20),
    L(0x1.fe8f0cea9d1e0ff25be2470b4318p+20),
    L(0x3.8872aebeb368399aee02b39340aep+20),
    L(0x6.ebd560d351e84e26a4381f5b293cp+20),
    L(0xc.c3644d094b0dae2fbcbf682cd428p+20),
    /* Interval [-2.75, -2.625] (polynomial degree 26).  */
    L(-0x6.b5d252a56e8a75458a27ed1c2dd4p-4),
    L(0x1.28d60383da3ac721aed3c5794da9p+0),
    L(0x1.db6513ada8a66ea77d87d9a8827bp+0),
    L(0x2.e217118f9d348a27f7506a707e6ep+0),
    L(0x4.450112c5cbf725a0fb9802396c9p+0),
    L(0x6.4af99151eae7810a75df2a0303c4p+0),
    L(0x9.2db598b4a97a7f69aeef32aec758p+0),
    L(0xd.62bef9c22471f5ee47ea1b9c0b5p+0),
    L(0x1.379f294e412bd62328326d4222f9p+4),
    L(0x1.c5827349d8865f1e8825c37c31c6p+4),
    L(0x2.93a7e7a75b7568cc8cbe8c016c12p+4),
    L(0x3.bf9bb882afe57edb383d41879d3ap+4),
    L(0x5.73c737828cee095c43a5566731c8p+4),
    L(0x7.ee4653493a7f81e0442062b3823cp+4),
    L(0xb.891c6b83fc8b55bd973b5d962d6p+4),
    L(0x1.0c775d7de3bf9b246c0208e0207ep+8),
    L(0x1.867ee43ec4bd4f4fd56abc05110ap+8),
    L(0x2.37fe9ba6695821e9822d8c8af0a6p+8),
    L(0x3.3a2c667e37c942f182cd3223a936p+8),
    L(0x4.b1b500eb59f3f782c7ccec88754p+8),
    L(0x6.d3efd3b65b3d0d8488d30b79fa4cp+8),
    L(0x9.ee8224e65bed5ced8b75eaec609p+8),
    L(0xe.72416e510cca77d53fc615c1f3dp+8),
    L(0x1.4fb538b0a2dfe567a8904b7e0445p+12),
    L(0x1.e7f56a9266cf525a5b8cf4cb76cep+12),
    L(0x2.f0365c983f68c597ee49d099cce8p+12),
    L(0x4.53aa229e1b9f5b5e59625265951p+12),
    /* Interval [-2.875, -2.75] (polynomial degree 24).  */
    L(-0x8.a41b1e4f36ff88dc820815607d68p-4),
    L(0xc.da87d3b69dc0f2f9c6f368b8ca1p-4),
    L(0x1.1474ad5c36158a7bea04fd2f98c6p+0),
    L(0x1.761ecb90c555df6555b7dba955b6p+0),
    L(0x1.d279bff9ae291caf6c4b4bcb3202p+0),
    L(0x2.4e5d00559a6e2b9b5d7fe1f6689cp+0),
    L(0x2.d57545a75cee8743ae2b17bc8d24p+0),
    L(0x3.8514eee3aac88b89bec2307021bap+0),
    L(0x4.5235e3b6e1891ffeb87fed9f8a24p+0),
    L(0x5.562acdb10eef3c9a773b3e27a864p+0),
    L(0x6.8ec8965c76efe03c26bff60b1194p+0),
    L(0x8.15251aca144877af32658399f9b8p+0),
    L(0x9.f08d56aba174d844138af782c0f8p+0),
    L(0xc.3dbbeda2679e8a1346ccc3f6da88p+0),
    L(0xf.0f5bfd5eacc26db308ffa0556fa8p+0),
    L(0x1.28a6ccd84476fbc713d6bab49ac9p+4),
    L(0x1.6d0a3ae2a3b1c8ff400641a3a21fp+4),
    L(0x1.c15701b28637f87acfb6a91d33b5p+4),
    L(0x2.28fbe0eccf472089b017651ca55ep+4),
    L(0x2.a8a453004f6e8ffaacd1603bc3dp+4),
    L(0x3.45ae4d9e1e7cd1a5dba0e4ec7f6cp+4),
    L(0x4.065fbfacb7fad3e473cb577a61e8p+4),
    L(0x4.f3d1473020927acac1944734a39p+4),
    L(0x6.54bb091245815a36fb74e314dd18p+4),
    L(0x7.d7f445129f7fb6c055e582d3f6ep+4),
    /* Interval [-3, -2.875] (polynomial degree 23).  */
    L(-0xa.046d667e468f3e44dcae1afcc648p-4),
    L(0x9.70b88dcc006c214d8d996fdf5ccp-4),
    L(0xa.a8a39421c86d3ff24931a0929fp-4),
    L(0xd.2f4d1363f324da2b357c8b6ec94p-4),
    L(0xd.ca9aa1a3a5c00de11bf60499a97p-4),
    L(0xf.cf09c31eeb52a45dfa7ebe3778dp-4),
    L(0x1.04b133a39ed8a09691205660468bp+0),
    L(0x1.22b547a06edda944fcb12fd9b5ecp+0),
    L(0x1.2c57fce7db86a91df09602d344b3p+0),
    L(0x1.4aade4894708f84795212fe257eep+0),
    L(0x1.579c8b7b67ec4afed5b28c8bf787p+0),
    L(0x1.776820e7fc80ae5284239733078ap+0),
    L(0x1.883ab28c7301fde4ca6b8ec26ec8p+0),
    L(0x1.aa2ef6e1ae52eb42c9ee83b206e3p+0),
    L(0x1.bf4ad50f0a9a9311300cf0c51ee7p+0),
    L(0x1.e40206e0e96b1da463814dde0d09p+0),
    L(0x1.fdcbcffef3a21b29719c2bd9feb1p+0),
    L(0x2.25e2e8948939c4d42cf108fae4bep+0),
    L(0x2.44ce14d2b59c1c0e6bf2cfa81018p+0),
    L(0x2.70ee80bbd0387162be4861c43622p+0),
    L(0x2.954b64d2c2ebf3489b949c74476p+0),
    L(0x2.c616e133a811c1c9446105208656p+0),
    L(0x3.05a69dfe1a9ba1079f90fcf26bd4p+0),
    L(0x3.410d2ad16a0506de29736e6aafdap+0),
  };

static const size_t poly_deg[] =
  {
    23,
    24,
    25,
    27,
    28,
    26,
    24,
    23,
  };

static const size_t poly_end[] =
  {
    23,
    48,
    74,
    102,
    131,
    158,
    183,
    207,
  };

/* Compute sin (pi * X) for -0.25 <= X <= 0.5.  */

static _Float128
lg_sinpi (_Float128 x)
{
  if (x <= L(0.25))
    return __sinl (M_PIl * x);
  else
    return __cosl (M_PIl * (L(0.5) - x));
}

/* Compute cos (pi * X) for -0.25 <= X <= 0.5.  */

static _Float128
lg_cospi (_Float128 x)
{
  if (x <= L(0.25))
    return __cosl (M_PIl * x);
  else
    return __sinl (M_PIl * (L(0.5) - x));
}

/* Compute cot (pi * X) for -0.25 <= X <= 0.5.  */

static _Float128
lg_cotpi (_Float128 x)
{
  return lg_cospi (x) / lg_sinpi (x);
}

/* Compute lgamma of a negative argument -50 < X < -2, setting
   *SIGNGAMP accordingly.  */

_Float128
__lgamma_negl (_Float128 x, int *signgamp)
{
  /* Determine the half-integer region X lies in, handle exact
     integers and determine the sign of the result.  */
  int i = floorl (-2 * x);
  if ((i & 1) == 0 && i == -2 * x)
    return L(1.0) / L(0.0);
  _Float128 xn = ((i & 1) == 0 ? -i / 2 : (-i - 1) / 2);
  i -= 4;
  *signgamp = ((i & 2) == 0 ? -1 : 1);

  SET_RESTORE_ROUNDL (FE_TONEAREST);

  /* Expand around the zero X0 = X0_HI + X0_LO.  */
  _Float128 x0_hi = lgamma_zeros[i][0], x0_lo = lgamma_zeros[i][1];
  _Float128 xdiff = x - x0_hi - x0_lo;

  /* For arguments in the range -3 to -2, use polynomial
     approximations to an adjusted version of the gamma function.  */
  if (i < 2)
    {
      int j = floorl (-8 * x) - 16;
      _Float128 xm = (-33 - 2 * j) * L(0.0625);
      _Float128 x_adj = x - xm;
      size_t deg = poly_deg[j];
      size_t end = poly_end[j];
      _Float128 g = poly_coeff[end];
      for (size_t j = 1; j <= deg; j++)
	g = g * x_adj + poly_coeff[end - j];
      return __log1pl (g * xdiff / (x - xn));
    }

  /* The result we want is log (sinpi (X0) / sinpi (X))
     + log (gamma (1 - X0) / gamma (1 - X)).  */
  _Float128 x_idiff = fabsl (xn - x), x0_idiff = fabsl (xn - x0_hi - x0_lo);
  _Float128 log_sinpi_ratio;
  if (x0_idiff < x_idiff * L(0.5))
    /* Use log not log1p to avoid inaccuracy from log1p of arguments
       close to -1.  */
    log_sinpi_ratio = __ieee754_logl (lg_sinpi (x0_idiff)
				      / lg_sinpi (x_idiff));
  else
    {
      /* Use log1p not log to avoid inaccuracy from log of arguments
	 close to 1.  X0DIFF2 has positive sign if X0 is further from
	 XN than X is from XN, negative sign otherwise.  */
      _Float128 x0diff2 = ((i & 1) == 0 ? xdiff : -xdiff) * L(0.5);
      _Float128 sx0d2 = lg_sinpi (x0diff2);
      _Float128 cx0d2 = lg_cospi (x0diff2);
      log_sinpi_ratio = __log1pl (2 * sx0d2
				  * (-sx0d2 + cx0d2 * lg_cotpi (x_idiff)));
    }

  _Float128 log_gamma_ratio;
  _Float128 y0 = 1 - x0_hi;
  _Float128 y0_eps = -x0_hi + (1 - y0) - x0_lo;
  _Float128 y = 1 - x;
  _Float128 y_eps = -x + (1 - y);
  /* We now wish to compute LOG_GAMMA_RATIO
     = log (gamma (Y0 + Y0_EPS) / gamma (Y + Y_EPS)).  XDIFF
     accurately approximates the difference Y0 + Y0_EPS - Y -
     Y_EPS.  Use Stirling's approximation.  First, we may need to
     adjust into the range where Stirling's approximation is
     sufficiently accurate.  */
  _Float128 log_gamma_adj = 0;
  if (i < 20)
    {
      int n_up = (21 - i) / 2;
      _Float128 ny0, ny0_eps, ny, ny_eps;
      ny0 = y0 + n_up;
      ny0_eps = y0 - (ny0 - n_up) + y0_eps;
      y0 = ny0;
      y0_eps = ny0_eps;
      ny = y + n_up;
      ny_eps = y - (ny - n_up) + y_eps;
      y = ny;
      y_eps = ny_eps;
      _Float128 prodm1 = __lgamma_productl (xdiff, y - n_up, y_eps, n_up);
      log_gamma_adj = -__log1pl (prodm1);
    }
  _Float128 log_gamma_high
    = (xdiff * __log1pl ((y0 - e_hi - e_lo + y0_eps) / e_hi)
       + (y - L(0.5) + y_eps) * __log1pl (xdiff / y) + log_gamma_adj);
  /* Compute the sum of (B_2k / 2k(2k-1))(Y0^-(2k-1) - Y^-(2k-1)).  */
  _Float128 y0r = 1 / y0, yr = 1 / y;
  _Float128 y0r2 = y0r * y0r, yr2 = yr * yr;
  _Float128 rdiff = -xdiff / (y * y0);
  _Float128 bterm[NCOEFF];
  _Float128 dlast = rdiff, elast = rdiff * yr * (yr + y0r);
  bterm[0] = dlast * lgamma_coeff[0];
  for (size_t j = 1; j < NCOEFF; j++)
    {
      _Float128 dnext = dlast * y0r2 + elast;
      _Float128 enext = elast * yr2;
      bterm[j] = dnext * lgamma_coeff[j];
      dlast = dnext;
      elast = enext;
    }
  _Float128 log_gamma_low = 0;
  for (size_t j = 0; j < NCOEFF; j++)
    log_gamma_low += bterm[NCOEFF - 1 - j];
  log_gamma_ratio = log_gamma_high + log_gamma_low;

  return log_sinpi_ratio + log_gamma_ratio;
}
