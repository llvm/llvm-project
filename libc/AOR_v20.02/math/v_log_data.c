/*
 * Lookup table for double-precision log(x) vector function.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "v_log.h"
#if WANT_VMATH

#define N (1 << V_LOG_TABLE_BITS)

/* Algorithm:

	x = 2^k z
	log(x) = k ln2 + log(c) + poly(z/c - 1)

where z is in [a;2a) which is split into N subintervals (a=0x1.69009p-1,N=128)
and log(c) and 1/c for the ith subinterval comes from a lookup table:

	tab[i].invc = 1/c
	tab[i].logc = (double)log(c)

where c is near the center of the subinterval and is chosen by trying several
floating point invc candidates around 1/center and selecting one for which
the error in (double)log(c) is minimized (< 0x1p-74), except the subinterval
that contains 1 and the previous one got tweaked to avoid cancellation.  */
const struct v_log_data __v_log_data[N] = {
{0x1.6a133d0dec120p+0, -0x1.62fe995eb963ap-2},
{0x1.6815f2f3e42edp+0, -0x1.5d5a48dad6b67p-2},
{0x1.661e39be1ac9ep+0, -0x1.57bde257d2769p-2},
{0x1.642bfa30ac371p+0, -0x1.52294fbf2af55p-2},
{0x1.623f1d916f323p+0, -0x1.4c9c7b598aa38p-2},
{0x1.60578da220f65p+0, -0x1.47174fc5ff560p-2},
{0x1.5e75349dea571p+0, -0x1.4199b7fa7b5cap-2},
{0x1.5c97fd387a75ap+0, -0x1.3c239f48cfb99p-2},
{0x1.5abfd2981f200p+0, -0x1.36b4f154d2aebp-2},
{0x1.58eca051dc99cp+0, -0x1.314d9a0ff32fbp-2},
{0x1.571e526d9df12p+0, -0x1.2bed85cca3cffp-2},
{0x1.5554d555b3fcbp+0, -0x1.2694a11421af9p-2},
{0x1.539015e2a20cdp+0, -0x1.2142d8d014fb2p-2},
{0x1.51d0014ee0164p+0, -0x1.1bf81a2c77776p-2},
{0x1.50148538cd9eep+0, -0x1.16b452a39c6a4p-2},
{0x1.4e5d8f9f698a1p+0, -0x1.11776ffa6c67ep-2},
{0x1.4cab0edca66bep+0, -0x1.0c416035020e0p-2},
{0x1.4afcf1a9db874p+0, -0x1.071211aa10fdap-2},
{0x1.495327136e16fp+0, -0x1.01e972e293b1bp-2},
{0x1.47ad9e84af28fp+0, -0x1.f98ee587fd434p-3},
{0x1.460c47b39ae15p+0, -0x1.ef5800ad716fbp-3},
{0x1.446f12b278001p+0, -0x1.e52e160484698p-3},
{0x1.42d5efdd720ecp+0, -0x1.db1104b19352ep-3},
{0x1.4140cfe001a0fp+0, -0x1.d100ac59e0bd6p-3},
{0x1.3fafa3b421f69p+0, -0x1.c6fced287c3bdp-3},
{0x1.3e225c9c8ece5p+0, -0x1.bd05a7b317c29p-3},
{0x1.3c98ec29a211ap+0, -0x1.b31abd229164fp-3},
{0x1.3b13442a413fep+0, -0x1.a93c0edadb0a3p-3},
{0x1.399156baa3c54p+0, -0x1.9f697ee30d7ddp-3},
{0x1.38131639b4cdbp+0, -0x1.95a2efa9aa40ap-3},
{0x1.36987540fbf53p+0, -0x1.8be843d796044p-3},
{0x1.352166b648f61p+0, -0x1.82395ecc477edp-3},
{0x1.33adddb3eb575p+0, -0x1.7896240966422p-3},
{0x1.323dcd99fc1d3p+0, -0x1.6efe77aca8c55p-3},
{0x1.30d129fefc7d2p+0, -0x1.65723e117ec5cp-3},
{0x1.2f67e6b72fe7dp+0, -0x1.5bf15c0955706p-3},
{0x1.2e01f7cf8b187p+0, -0x1.527bb6c111da1p-3},
{0x1.2c9f518ddc86ep+0, -0x1.491133c939f8fp-3},
{0x1.2b3fe86e5f413p+0, -0x1.3fb1b90c7fc58p-3},
{0x1.29e3b1211b25cp+0, -0x1.365d2cc485f8dp-3},
{0x1.288aa08b373cfp+0, -0x1.2d13758970de7p-3},
{0x1.2734abcaa8467p+0, -0x1.23d47a721fd47p-3},
{0x1.25e1c82459b81p+0, -0x1.1aa0229f25ec2p-3},
{0x1.2491eb1ad59c5p+0, -0x1.117655ddebc3bp-3},
{0x1.23450a54048b5p+0, -0x1.0856fbf83ab6bp-3},
{0x1.21fb1bb09e578p+0, -0x1.fe83fabbaa106p-4},
{0x1.20b415346d8f7p+0, -0x1.ec6e8507a56cdp-4},
{0x1.1f6fed179a1acp+0, -0x1.da6d68c7cc2eap-4},
{0x1.1e2e99b93c7b3p+0, -0x1.c88078462be0cp-4},
{0x1.1cf011a7a882ap+0, -0x1.b6a786a423565p-4},
{0x1.1bb44b97dba5ap+0, -0x1.a4e2676ac7f85p-4},
{0x1.1a7b3e66cdd4fp+0, -0x1.9330eea777e76p-4},
{0x1.1944e11dc56cdp+0, -0x1.8192f134d5ad9p-4},
{0x1.18112aebb1a6ep+0, -0x1.70084464f0538p-4},
{0x1.16e013231b7e9p+0, -0x1.5e90bdec5cb1fp-4},
{0x1.15b1913f156cfp+0, -0x1.4d2c3433c5536p-4},
{0x1.14859cdedde13p+0, -0x1.3bda7e219879ap-4},
{0x1.135c2dc68cfa4p+0, -0x1.2a9b732d27194p-4},
{0x1.12353bdb01684p+0, -0x1.196eeb2b10807p-4},
{0x1.1110bf25b85b4p+0, -0x1.0854be8ef8a7ep-4},
{0x1.0feeafd2f8577p+0, -0x1.ee998cb277432p-5},
{0x1.0ecf062c51c3bp+0, -0x1.ccadb79919fb9p-5},
{0x1.0db1baa076c8bp+0, -0x1.aae5b1d8618b0p-5},
{0x1.0c96c5bb3048ep+0, -0x1.89413015d7442p-5},
{0x1.0b7e20263e070p+0, -0x1.67bfe7bf158dep-5},
{0x1.0a67c2acd0ce3p+0, -0x1.46618f83941bep-5},
{0x1.0953a6391e982p+0, -0x1.2525df1b0618ap-5},
{0x1.0841c3caea380p+0, -0x1.040c8e2f77c6ap-5},
{0x1.07321489b13eap+0, -0x1.c62aad39f738ap-6},
{0x1.062491aee9904p+0, -0x1.847fe3bdead9cp-6},
{0x1.05193497a7cc5p+0, -0x1.43183683400acp-6},
{0x1.040ff6b5f5e9fp+0, -0x1.01f31c4e1d544p-6},
{0x1.0308d19aa6127p+0, -0x1.82201d1e6b69ap-7},
{0x1.0203beedb0c67p+0, -0x1.00dd0f3e1bfd6p-7},
{0x1.010037d38bcc2p+0, -0x1.ff6fe1feb4e53p-9},
{1.0, 0.0},
{0x1.fc06d493cca10p-1, 0x1.fe91885ec8e20p-8},
{0x1.f81e6ac3b918fp-1, 0x1.fc516f716296dp-7},
{0x1.f44546ef18996p-1, 0x1.7bb4dd70a015bp-6},
{0x1.f07b10382c84bp-1, 0x1.f84c99b34b674p-6},
{0x1.ecbf7070e59d4p-1, 0x1.39f9ce4fb2d71p-5},
{0x1.e91213f715939p-1, 0x1.7756c0fd22e78p-5},
{0x1.e572a9a75f7b7p-1, 0x1.b43ee82db8f3ap-5},
{0x1.e1e0e2c530207p-1, 0x1.f0b3fced60034p-5},
{0x1.de5c72d8a8be3p-1, 0x1.165bd78d4878ep-4},
{0x1.dae50fa5658ccp-1, 0x1.3425d2715ebe6p-4},
{0x1.d77a71145a2dap-1, 0x1.51b8bd91b7915p-4},
{0x1.d41c51166623ep-1, 0x1.6f15632c76a47p-4},
{0x1.d0ca6ba0bb29fp-1, 0x1.8c3c88ecbe503p-4},
{0x1.cd847e8e59681p-1, 0x1.a92ef077625dap-4},
{0x1.ca4a499693e00p-1, 0x1.c5ed5745fa006p-4},
{0x1.c71b8e399e821p-1, 0x1.e27876de1c993p-4},
{0x1.c3f80faf19077p-1, 0x1.fed104fce4cdcp-4},
{0x1.c0df92dc2b0ecp-1, 0x1.0d7bd9c17d78bp-3},
{0x1.bdd1de3cbb542p-1, 0x1.1b76986cef97bp-3},
{0x1.baceb9e1007a3p-1, 0x1.295913d24f750p-3},
{0x1.b7d5ef543e55ep-1, 0x1.37239fa295d17p-3},
{0x1.b4e749977d953p-1, 0x1.44d68dd78714bp-3},
{0x1.b20295155478ep-1, 0x1.52722ebe5d780p-3},
{0x1.af279f8e82be2p-1, 0x1.5ff6d12671f98p-3},
{0x1.ac5638197fdf3p-1, 0x1.6d64c2389484bp-3},
{0x1.a98e2f102e087p-1, 0x1.7abc4da40fddap-3},
{0x1.a6cf5606d05c1p-1, 0x1.87fdbda1e8452p-3},
{0x1.a4197fc04d746p-1, 0x1.95295b06a5f37p-3},
{0x1.a16c80293dc01p-1, 0x1.a23f6d34abbc5p-3},
{0x1.9ec82c4dc5bc9p-1, 0x1.af403a28e04f2p-3},
{0x1.9c2c5a491f534p-1, 0x1.bc2c06a85721ap-3},
{0x1.9998e1480b618p-1, 0x1.c903161240163p-3},
{0x1.970d9977c6c2dp-1, 0x1.d5c5aa93287ebp-3},
{0x1.948a5c023d212p-1, 0x1.e274051823fa9p-3},
{0x1.920f0303d6809p-1, 0x1.ef0e656300c16p-3},
{0x1.8f9b698a98b45p-1, 0x1.fb9509f05aa2ap-3},
{0x1.8d2f6b81726f6p-1, 0x1.04041821f37afp-2},
{0x1.8acae5bb55badp-1, 0x1.0a340a49b3029p-2},
{0x1.886db5d9275b8p-1, 0x1.105a7918a126dp-2},
{0x1.8617ba567c13cp-1, 0x1.1677819812b84p-2},
{0x1.83c8d27487800p-1, 0x1.1c8b405b40c0ep-2},
{0x1.8180de3c5dbe7p-1, 0x1.2295d16cfa6b1p-2},
{0x1.7f3fbe71cdb71p-1, 0x1.28975066318a2p-2},
{0x1.7d055498071c1p-1, 0x1.2e8fd855d86fcp-2},
{0x1.7ad182e54f65ap-1, 0x1.347f83d605e59p-2},
{0x1.78a42c3c90125p-1, 0x1.3a666d1244588p-2},
{0x1.767d342f76944p-1, 0x1.4044adb6f8ec4p-2},
{0x1.745c7ef26b00ap-1, 0x1.461a5f077558cp-2},
{0x1.7241f15769d0fp-1, 0x1.4be799e20b9c8p-2},
{0x1.702d70d396e41p-1, 0x1.51ac76a6b79dfp-2},
{0x1.6e1ee3700cd11p-1, 0x1.57690d5744a45p-2},
{0x1.6c162fc9cbe02p-1, 0x1.5d1d758e45217p-2},
};
#endif
