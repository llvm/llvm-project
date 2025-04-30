/* Extended-precision floating point sine and cosine tables.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Based on quad-precision tables by Jakub Jelinek <jj@ultra.linux.cz>

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

/* For 0.1484375 + n/128.0, n=0..82 this table contains
   first 64 bits of cosine, then at least 64 additional
   bits and the same for sine.
   0.1484375+82.0/128.0 is the smallest number among above defined numbers
   larger than pi/4.
   Computed using MPFR:

   #include <stdio.h>
   #include <mpfr.h>

   int
   main (void)
   {
     int j;
     mpfr_t t, b, i, rs, rc, ts, tc, tsl, tcl;
     mpfr_init2 (b, 64);
     mpfr_init2 (i, 64);
     mpfr_init2 (t, 64);
     mpfr_set_str (b, "0.1484375", 0, MPFR_RNDN);
     mpfr_set_str (i, "0x1p-7", 0, MPFR_RNDN);
     mpfr_init2 (rs, 300);
     mpfr_init2 (rc, 300);
     mpfr_init2 (ts, 64);
     mpfr_init2 (tc, 64);
     mpfr_init2 (tsl, 64);
     mpfr_init2 (tcl, 64);
     for (j = 0; j <= 82; j++)
       {
         mpfr_mul_ui (t, i, j, MPFR_RNDN);
         mpfr_add (t, t, b, MPFR_RNDN);
         printf ("  /" "* x = 0.1484375 + %d/128.  *" "/\n", j);
         mpfr_cos (rc, t, MPFR_RNDN);
         mpfr_sin (rs, t, MPFR_RNDN);
         mpfr_set (tc, rc, MPFR_RNDN);
         mpfr_set (ts, rs, MPFR_RNDN);
         mpfr_sub (tcl, rc, tc, MPFR_RNDN);
         mpfr_sub (tsl, rs, ts, MPFR_RNDN);
         mpfr_printf ("  %.17RaL,\n", tc);
         mpfr_printf ("  %.17RaL,\n", tcl);
         mpfr_printf ("  %.17RaL,\n", ts);
         mpfr_printf ("  %.17RaL,\n", tsl);
       }
     return 0;
   }

*/

const long double __sincosl_table[] = {
  /* x = 0.1484375 + 0/128.  */
  0xf.d2f5320e1b7902100p-4L,
  -0x6.4b225d06708635580p-68L,
  0x2.5dc50bc95711d0d80p-4L,
  0x1.787d108fd438cf5a0p-68L,
  /* x = 0.1484375 + 1/128.  */
  0xf.ce1a053e621438b00p-4L,
  0x6.d60c76e8c45bf0a80p-68L,
  0x2.7d66258bacd96a400p-4L,
  -0x1.4cca4c9a3782a6bc0p-68L,
  /* x = 0.1484375 + 2/128.  */
  0xf.c8ffa01ba68074100p-4L,
  0x7.e05962b0d9fdf2000p-68L,
  0x2.9cfd49b8be4f66540p-4L,
  -0x1.89354fe340fbd96c0p-68L,
  /* x = 0.1484375 + 3/128.  */
  0xf.c3a6170f767ac7300p-4L,
  0x5.d63d99a9d439e1d80p-68L,
  0x2.bc89f9f424de54840p-4L,
  0x1.de7ce03b2514952c0p-68L,
  /* x = 0.1484375 + 4/128.  */
  0xf.be0d7f7fef11e7100p-4L,
  -0x5.5bc47540b095ba800p-68L,
  0x2.dc0bb80b49a97ffc0p-4L,
  -0xc.b1722e07246208500p-72L,
  /* x = 0.1484375 + 5/128.  */
  0xf.b835efcf670dd2d00p-4L,
  -0x1.90186db968115ec20p-68L,
  0x2.fb8205f75e56a2b40p-4L,
  0x1.6a1c4792f85625880p-68L,
  /* x = 0.1484375 + 6/128.  */
  0xf.b21f7f5c156696b00p-4L,
  0xa.c1fe28ac5fd766700p-76L,
  0x3.1aec65df552876f80p-4L,
  0x2.ece9a235671324700p-72L,
  /* x = 0.1484375 + 7/128.  */
  0xf.abca467fb3cb8f200p-4L,
  -0x2.f960fe2715cc521c0p-68L,
  0x3.3a4a5a19d86246700p-4L,
  0x1.0f602c44df4fa5140p-68L,
  /* x = 0.1484375 + 8/128.  */
  0xf.a5365e8f1d3ca2800p-4L,
  -0x4.1e24a289519b26800p-68L,
  0x3.599b652f40ec999c0p-4L,
  0x1.f12a0a4c8561de160p-68L,
  /* x = 0.1484375 + 9/128.  */
  0xf.9e63e1d9e8b6f6f00p-4L,
  0x2.e296bae5b5ed9c100p-68L,
  0x3.78df09db8c332ce00p-4L,
  0xd.2b53d865582e45200p-72L,
  /* x = 0.1484375 + 10/128.  */
  0xf.9752eba9fff6b9900p-4L,
  -0x7.bd415254fab56cd00p-68L,
  0x3.9814cb10513453cc0p-4L,
  -0x6.84de43e3595cc8500p-72L,
  /* x = 0.1484375 + 11/128.  */
  0xf.90039843324f9b900p-4L,
  0x4.0416c1984b6cbed00p-68L,
  0x3.b73c2bf6b4b9f6680p-4L,
  0xe.f9499c81f0d965100p-72L,
  /* x = 0.1484375 + 12/128.  */
  0xf.887604e2c39dbb200p-4L,
  0xe.4ec5825059a78a000p-72L,
  0x3.d654aff15cb457a00p-4L,
  0xf.ca854698aba330400p-72L,
  /* x = 0.1484375 + 13/128.  */
  0xf.80aa4fbef750ba800p-4L,
  -0x7.c2cc346a06b075c00p-68L,
  0x3.f55dda9e62aed7500p-4L,
  0x1.3bd7b8e6a3d1635e0p-68L,
  /* x = 0.1484375 + 14/128.  */
  0xf.78a098069792dab00p-4L,
  -0x4.3611bda6e483a5980p-68L,
  0x4.14572fd94556e6480p-4L,
  -0xc.29dfd8ec7722b8400p-72L,
  /* x = 0.1484375 + 15/128.  */
  0xf.7058fde0788dfc800p-4L,
  0x5.b8fe88789e4f42500p-72L,
  0x4.334033bcd90d66080p-4L,
  -0x3.0a0c93e2b47bbae40p-68L,
  /* x = 0.1484375 + 16/128.  */
  0xf.67d3a26af7d07aa00p-4L,
  0x4.bd6d42af8c0068000p-68L,
  0x4.52186aa5377ab2080p-4L,
  0x3.bf2524f52e3a06a80p-68L,
  /* x = 0.1484375 + 17/128.  */
  0xf.5f10a7bb77d3dfa00p-4L,
  0xc.1da8b578427832800p-72L,
  0x4.70df5931ae1d94600p-4L,
  0x7.6fe0dcff47fe31b80p-72L,
  /* x = 0.1484375 + 18/128.  */
  0xf.561030ddd7a789600p-4L,
  0xe.a9f4a32c652155500p-72L,
  0x4.8f948446abcd6b100p-4L,
  -0x8.0334eff185e4d9100p-72L,
  /* x = 0.1484375 + 19/128.  */
  0xf.4cd261d3e6c15bb00p-4L,
  0x3.69c8758630d2ac000p-68L,
  0x4.ae37710fad27c8a80p-4L,
  0x2.9c4cf96c03519b9c0p-68L,
  /* x = 0.1484375 + 20/128.  */
  0xf.43575f94d4f6b2700p-4L,
  0x2.f5fb76b14d2a64ac0p-68L,
  0x4.ccc7a50127e1de100p-4L,
  -0x3.494bf3cfd39ae0840p-68L,
  /* x = 0.1484375 + 21/128.  */
  0xf.399f500c9e9fd3800p-4L,
  -0x5.166a8d9c254778900p-68L,
  0x4.eb44a5da74f600200p-4L,
  0x7.aaa090f0734e28880p-72L,
  /* x = 0.1484375 + 22/128.  */
  0xf.2faa5a1b74e82fd00p-4L,
  0x6.1fa05f9177380e900p-68L,
  0x5.09adf9a7b9a5a0f80p-4L,
  -0x1.c75705c59f5e66be0p-68L,
  /* x = 0.1484375 + 23/128.  */
  0xf.2578a595224dd2e00p-4L,
  0x6.bfa2eb2f99cc67500p-68L,
  0x5.280326c3cf4818200p-4L,
  0x3.ba6bb08eac82c2080p-68L,
  /* x = 0.1484375 + 24/128.  */
  0xf.1b0a5b406b526d900p-4L,
  -0x7.93aa0152372f23380p-68L,
  0x5.4643b3da29de9b380p-4L,
  -0x2.8eaa110f0ccd04c00p-68L,
  /* x = 0.1484375 + 25/128.  */
  0xf.105fa4d66b607a600p-4L,
  0x7.d44e0427252044380p-68L,
  0x5.646f27e8bd65cbe00p-4L,
  0x3.a5d61ff0657229100p-68L,
  /* x = 0.1484375 + 26/128.  */
  0xf.0578ad01ede708000p-4L,
  -0x5.c63f6239467b50100p-68L,
  0x5.82850a41e1dd46c80p-4L,
  -0x9.fd15dbb3244403200p-76L,
  /* x = 0.1484375 + 27/128.  */
  0xe.fa559f5ec3aec3a00p-4L,
  0x4.eb03319278a2d4200p-68L,
  0x5.a084e28e35fda2780p-4L,
  -0x9.202444aace28b3100p-72L,
  /* x = 0.1484375 + 28/128.  */
  0xe.eef6a879146af0c00p-4L,
  -0x6.46a15d15f53f2c200p-72L,
  0x5.be6e38ce809554280p-4L,
  0x3.c14ee9da0d3648400p-68L,
  /* x = 0.1484375 + 29/128.  */
  0xe.e35bf5ccac8905300p-4L,
  -0x3.26e2248cb2c5b81c0p-68L,
  0x5.dc40955d9084f4880p-4L,
  0x2.94675a2498de5d840p-68L,
  /* x = 0.1484375 + 30/128.  */
  0xe.d785b5c44741b4500p-4L,
  -0x6.c3a943462cc75eb00p-68L,
  0x5.f9fb80f21b5364a00p-4L,
  -0x3.bcdabf5af1dd3ad00p-68L,
  /* x = 0.1484375 + 31/128.  */
  0xe.cb7417b8d4ee3ff00p-4L,
  -0x3.c8545bf8c55b70e00p-68L,
  0x6.179e84a09a5258a80p-4L,
  -0x3.f164a0531fc1ada00p-68L,
  /* x = 0.1484375 + 32/128.  */
  0xe.bf274bf0bda4f6200p-4L,
  0x4.47e56a09362679900p-68L,
  0x6.352929dd264bd4480p-4L,
  0x2.02ea766325d8aa8c0p-68L,
  /* x = 0.1484375 + 33/128.  */
  0xe.b29f839f201fd1400p-4L,
  -0x4.6c8697d86e9587100p-68L,
  0x6.529afa7d51b129600p-4L,
  0x3.1ec197c0a840a11c0p-68L,
  /* x = 0.1484375 + 34/128.  */
  0xe.a5dcf0e30cf03e700p-4L,
  -0x6.8910f4e13d9aea080p-68L,
  0x6.6ff380ba014410a00p-4L,
  -0x1.c65cdf4f5c05a02a0p-68L,
  /* x = 0.1484375 + 35/128.  */
  0xe.98dfc6c6be031e600p-4L,
  0xd.d3089cbdd18a75b00p-72L,
  0x6.8d324731433279700p-4L,
  0x3.bc712bcc4ccddc480p-68L,
  /* x = 0.1484375 + 36/128.  */
  0xe.8ba8393eca7821b00p-4L,
  -0x5.a9c27cb6e49efee80p-68L,
  0x6.aa56d8e8249db4e80p-4L,
  0x3.60a761fe3f9e559c0p-68L,
  /* x = 0.1484375 + 37/128.  */
  0xe.7e367d2956cfb1700p-4L,
  -0x4.955ee1abe632ffa80p-68L,
  0x6.c760c14c8585a5200p-4L,
  -0x2.42cb99f5193ad5380p-68L,
  /* x = 0.1484375 + 38/128.  */
  0xe.708ac84d4172a3e00p-4L,
  0x2.737662213429e1400p-68L,
  0x6.e44f8c36eb10a1c80p-4L,
  -0xa.d2f6c3ff0b2b84600p-72L,
  /* x = 0.1484375 + 39/128.  */
  0xe.62a551594b970a700p-4L,
  0x7.0b15d41d4c0e48400p-68L,
  0x7.0122c5ec5028c8d00p-4L,
  -0xc.c540b02cbf333c800p-76L,
  /* x = 0.1484375 + 40/128.  */
  0xe.54864fe33e8575d00p-4L,
  -0x5.40a42f1a30e4e5780p-68L,
  0x7.1dd9fb1ff46778500p-4L,
  0x3.acb970a9f6729c700p-68L,
  /* x = 0.1484375 + 41/128.  */
  0xe.462dfc670d421ab00p-4L,
  0x3.d1a15901228f146c0p-68L,
  0x7.3a74b8f52947b6800p-4L,
  0x1.baf6928eb3fb02180p-68L,
  /* x = 0.1484375 + 42/128.  */
  0xe.379c9045f29d51800p-4L,
  -0x3.b7f755b683dfa84c0p-68L,
  0x7.56f28d011d9852880p-4L,
  0x2.44a75fc29c779bd80p-68L,
  /* x = 0.1484375 + 43/128.  */
  0xe.28d245c58baef7200p-4L,
  0x2.25e232abc003c4380p-68L,
  0x7.7353054ca72690d80p-4L,
  -0x3.391e8e0266194c600p-68L,
  /* x = 0.1484375 + 44/128.  */
  0xe.19cf580eeec046b00p-4L,
  -0x5.ebdd058b7f8131080p-68L,
  0x7.8f95b0560a9a3bd80p-4L,
  -0x1.2084267e23c739ee0p-68L,
  /* x = 0.1484375 + 45/128.  */
  0xe.0a94032dbea7cee00p-4L,
  -0x4.222625d0505267a80p-68L,
  0x7.abba1d12c17bfa200p-4L,
  -0x2.6d0f26c09f2126680p-68L,
  /* x = 0.1484375 + 46/128.  */
  0xd.fb20840f3a9b36f00p-4L,
  0x7.ae2c515342890b600p-68L,
  0x7.c7bfdaf13e5ed1700p-4L,
  0x2.12f8a7525bfb113c0p-68L,
  /* x = 0.1484375 + 47/128.  */
  0xd.eb7518814a7a93200p-4L,
  -0x4.433773ef632be3b00p-68L,
  0x7.e3a679daaf25c6780p-4L,
  -0x1.abd434bfd72f69be0p-68L,
  /* x = 0.1484375 + 48/128.  */
  0xd.db91ff31879917300p-4L,
  -0x4.2dbad2f5c7760ae80p-68L,
  0x7.ff6d8a34bd5e8fa80p-4L,
  -0x2.b368b7d24aea62100p-68L,
  /* x = 0.1484375 + 49/128.  */
  0xd.cb7777ac420705100p-4L,
  0x6.8f31e3eb780ce9c80p-68L,
  0x8.1b149ce34caa5a500p-4L,
  -0x1.9af072f602b295580p-68L,
  /* x = 0.1484375 + 50/128.  */
  0xd.bb25c25b8260c1500p-4L,
  -0x9.1843671366e48f400p-72L,
  0x8.369b434a372da7f00p-4L,
  -0x4.a3758e01c931e1f80p-68L,
  /* x = 0.1484375 + 51/128.  */
  0xd.aa9d2086082706400p-4L,
  -0x2.1ae3f617aa166cd00p-72L,
  0x8.52010f4f080052100p-4L,
  0x3.78bd8dd614753d080p-68L,
  /* x = 0.1484375 + 52/128.  */
  0xd.99ddd44e44a43d500p-4L,
  -0x2.b5c5c126adfbef900p-68L,
  0x8.6d45935ab396cb500p-4L,
  -0x1.bde17dd211ab0caa0p-68L,
  /* x = 0.1484375 + 53/128.  */
  0xd.88e820b1526311e00p-4L,
  -0x2.a9e1043f3e565ac80p-68L,
  0x8.8868625b4e1dbb200p-4L,
  0x3.13310133022527200p-68L,
  /* x = 0.1484375 + 54/128.  */
  0xd.77bc4985e93a60800p-4L,
  -0x3.6279746f944394400p-68L,
  0x8.a3690fc5bfc11c000p-4L,
  -0x6.aca1d8c657aed0b80p-68L,
  /* x = 0.1484375 + 55/128.  */
  0xd.665a937b4ef2b1f00p-4L,
  0x6.d51bad6d988a44180p-68L,
  0x8.be472f9776d809b00p-4L,
  -0xd.477e8edbc29c29900p-72L,
  /* x = 0.1484375 + 56/128.  */
  0xd.54c3441844897fd00p-4L,
  -0x7.07ac0f9aa0e459680p-68L,
  0x8.d902565817ee78400p-4L,
  -0x6.431c32ed7f9fee680p-68L,
  /* x = 0.1484375 + 57/128.  */
  0xd.42f6a1b9f0168ce00p-4L,
  -0xf.ce3d09c3726cfb200p-72L,
  0x8.f39a191b2ba612300p-4L,
  -0x5.c05b0be2a5c002c00p-68L,
  /* x = 0.1484375 + 58/128.  */
  0xd.30f4f392c357ab000p-4L,
  0x6.61c5fa8a7d9b26600p-68L,
  0x9.0e0e0d81ca6787900p-4L,
  0x6.cc92c8ea8c2815c00p-68L,
  /* x = 0.1484375 + 59/128.  */
  0xd.1ebe81a95ee752e00p-4L,
  0x4.8a26bcd32d6e92300p-68L,
  0x9.285dc9bc45dd9ea00p-4L,
  0x3.d02457bcce59c4180p-68L,
  /* x = 0.1484375 + 60/128.  */
  0xd.0c5394d7722281900p-4L,
  0x5.e25736c0357470800p-68L,
  0x9.4288e48bd0335fc00p-4L,
  0x4.1c4cbd2920497a900p-68L,
  /* x = 0.1484375 + 61/128.  */
  0xc.f9b476c897c25c600p-4L,
  -0x4.018af22c0cf715080p-68L,
  0x9.5c8ef544210ec0c00p-4L,
  -0x6.e3b642d55f617ae80p-68L,
  /* x = 0.1484375 + 62/128.  */
  0xc.e6e171f92f2e27f00p-4L,
  0x3.2225327ec440ddb00p-68L,
  0x9.766f93cd18413a700p-4L,
  -0x5.503e303903d754480p-68L,
  /* x = 0.1484375 + 63/128.  */
  0xc.d3dad1b5328a2e400p-4L,
  0x5.9f993f4f510881a00p-68L,
  0x9.902a58a45e27bed00p-4L,
  0x6.8412b426b675ed500p-68L,
  /* x = 0.1484375 + 64/128.  */
  0xc.c0a0e21709883a400p-4L,
  -0xf.f6ee1ee5f811c4300p-76L,
  0x9.a9bedcdf01b38da00p-4L,
  -0x6.c0c287df87e21d700p-68L,
  /* x = 0.1484375 + 65/128.  */
  0xc.ad33f00658fe5e800p-4L,
  0x2.04bbc0f3a66a0e6c0p-68L,
  0x9.c32cba2b14156ef00p-4L,
  0x5.256c4f857991ca680p-72L,
  /* x = 0.1484375 + 66/128.  */
  0xc.99944936cf48c8900p-4L,
  0x1.1ff93fe64b3ddb7a0p-68L,
  0x9.dc738ad14204e6900p-4L,
  -0x6.53a7d2f07a7d9a700p-68L,
  /* x = 0.1484375 + 67/128.  */
  0xc.85c23c26ed7b6f000p-4L,
  0x1.4ef546c4792968220p-68L,
  0x9.f592e9b66a9cf9000p-4L,
  0x6.a3c7aa3c101998480p-68L,
  /* x = 0.1484375 + 68/128.  */
  0xc.71be181ecd6875d00p-4L,
  -0x1.d25a9ea5fc335df80p-68L,
  0xa.0e8a725d33c828c00p-4L,
  0x1.1fa50fd9e9a15ffe0p-68L,
  /* x = 0.1484375 + 69/128.  */
  0xc.5d882d2ee48030c00p-4L,
  0x7.c07d28e981e348080p-68L,
  0xa.2759c0e79c3558200p-4L,
  0x5.27c32b55f5405c180p-68L,
  /* x = 0.1484375 + 70/128.  */
  0xc.4920cc2ec38fb8900p-4L,
  0x1.b38827db08884fc60p-68L,
  0xa.400072188acf49d00p-4L,
  -0x2.94e8c7da1fc7cb900p-68L,
  /* x = 0.1484375 + 71/128.  */
  0xc.348846bbd36313400p-4L,
  -0x7.001d401622ec7e600p-68L,
  0xa.587e23555bb080800p-4L,
  0x6.d02b9c662cdd29300p-68L,
  /* x = 0.1484375 + 72/128.  */
  0xc.1fbeef380e4ffdd00p-4L,
  0x5.a613ec8722f644000p-68L,
  0xa.70d272a76a8d4b700p-4L,
  -0x2.5f136f8ed448b7480p-68L,
  /* x = 0.1484375 + 73/128.  */
  0xc.0ac518c8b6ae71100p-4L,
  -0x4.5c85c1146f34ea500p-68L,
  0xa.88fcfebd9a8dd4800p-4L,
  -0x1.d0c3891061dbc66e0p-68L,
  /* x = 0.1484375 + 74/128.  */
  0xb.f59b17550a4406800p-4L,
  0x7.5969296567cf3e380p-68L,
  0xa.a0fd66eddb9212300p-4L,
  0x2.c28520d3911b8a040p-68L,
  /* x = 0.1484375 + 75/128.  */
  0xb.e0413f84f2a771c00p-4L,
  0x6.14946a88cbf4da200p-68L,
  0xa.b8d34b36acd987200p-4L,
  0x1.0ed343ec65d7e3ae0p-68L,
  /* x = 0.1484375 + 76/128.  */
  0xb.cab7e6bfb2a14aa00p-4L,
  -0x4.edd3a8b5c89413680p-68L,
  0xa.d07e4c409d08c5000p-4L,
  -0x5.c56fa844f53db4780p-68L,
  /* x = 0.1484375 + 77/128.  */
  0xb.b4ff632a908f73f00p-4L,
  -0x3.eae7c6346266c4b00p-68L,
  0xa.e7fe0b5fc786b2e00p-4L,
  -0x6.991e2950ebf5b7780p-68L,
  /* x = 0.1484375 + 78/128.  */
  0xb.9f180ba77dd075100p-4L,
  0x6.28e135a9508299000p-68L,
  0xa.ff522a954f2ba1700p-4L,
  -0x2.621023be91cc0a180p-68L,
  /* x = 0.1484375 + 79/128.  */
  0xb.890237d3bb3c28500p-4L,
  -0x4.9eb5fac6fe9405f00p-68L,
  0xb.167a4c90d63c42400p-4L,
  0x4.cf5493b7cc23bd400p-68L,
  /* x = 0.1484375 + 80/128.  */
  0xb.72be40067aaf2c000p-4L,
  0x5.0dbdb7a14c3d7d500p-68L,
  0xb.2d7614b1f3aaa2500p-4L,
  -0x2.0d291df5881e35c00p-68L,
  /* x = 0.1484375 + 81/128.  */
  0xb.5c4c7d4f7dae91600p-4L,
  -0x5.3879330b4e5b67300p-68L,
  0xb.44452709a59752900p-4L,
  0x5.913765434a59d1100p-72L,
  /* x = 0.1484375 + 82/128.  */
  0xb.45ad4975b1294cb00p-4L,
  -0x2.35b30bf1370dd5980p-68L,
  0xb.5ae7285bc10cf5100p-4L,
  0x5.753847e8f8b7a3100p-68L,
};
