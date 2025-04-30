.file "log2f.s"


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
// 09/11/00 Initial version
// 05/20/02 Cleaned up namespace and sf0 syntax
// 02/10/03 Reordered header: .section, .global, .proc, .align
//
// API
//==============================================================
// float log2f(float)
//
// Overview of operation
//==============================================================
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
// log2f(x) is approximated as
//     (l-j) + T[f] + (c1*r+c2*r^2+...+c6*r^6), if f>0
//


// Special values
//==============================================================
//  log2f(0)=-inf, raises Divide by Zero
//  log2f(+inf)=inf
//  log2f(x)=NaN,  raises Invalid if x<0
//


// Registers used
//==============================================================
//   f6-f14
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

data8 0x3fdec709dc3a03fd, 0xbfd71547652b82fe //C_3 and C_4
data8 0xb8aa3b295c17f0bc, 0x00003fff  // C_1
data8 0xb8aa3b295c17f0bc, 0x0000bffe  // C_2
LOCAL_OBJECT_END(poly_coeffs)


LOCAL_OBJECT_START(T_table)

data8 0x3f671b0ea42e5fda, 0x3f815cfe8eaec830
data8 0x3f8cfee70c5ce5dc, 0x3f94564a62192834
data8 0x3f997723ace35766, 0x3f9f5923c69b54a1
data8 0x3fa2a094a085d693, 0x3fa538941776b01e
data8 0x3fa8324c9b914bc7, 0x3faacf54ce07d7e9
data8 0x3fadced958dadc12, 0x3fb0387efbca869e
data8 0x3fb18ac6067479c0, 0x3fb30edd3e13530d
data8 0x3fb463c15936464e, 0x3fb5b9e13c3fa21d
data8 0x3fb7113f3259e07a, 0x3fb869dd8d1b2035
data8 0x3fb9c3bea49d3214, 0x3fbb1ee4d7961701
data8 0x3fbc7b528b70f1c5, 0x3fbdd90a2c676ed4
data8 0x3fbf05d4976c2028, 0x3fc032fbbaee6d65
data8 0x3fc0e3b5a9f3284a, 0x3fc195195c7d125b
data8 0x3fc22dadc2ab3497, 0x3fc2e050231df57d
data8 0x3fc379f79c2b255b, 0x3fc42ddd2ba1b4a9
data8 0x3fc4c89b9e6807f5, 0x3fc563dc29ffacb2
data8 0x3fc619a25f5d798d, 0x3fc6b5ffbf367644
data8 0x3fc752e1f660f8d6, 0x3fc7f049e753e7cf
data8 0x3fc8a8980abfbd32, 0x3fc94724cca657be
data8 0x3fc9e63a24971f46, 0x3fca85d8feb202f7
data8 0x3fcb2602497d5346, 0x3fcbc6b6f5ee1c9b
data8 0x3fcc67f7f770a67e, 0x3fcceec4b2234fba
data8 0x3fcd91097ad13982, 0x3fce33dd57f3d335
data8 0x3fced74146bc7b10, 0x3fcf7b3646fef683
data8 0x3fd00223a943dc19, 0x3fd054a474bf0eb7
data8 0x3fd0999d9b9259a1, 0x3fd0eca66d3b2581
data8 0x3fd13ffa2e85b475, 0x3fd185a444fa0a7b
data8 0x3fd1cb8312f27eff, 0x3fd21fa1441ce5e8
data8 0x3fd265f526e603cb, 0x3fd2baa0c34be1ec
data8 0x3fd3016b45de21ce, 0x3fd3486c38aa29a8
data8 0x3fd38fa3efaa8262, 0x3fd3e562c0816a02
data8 0x3fd42d141f53b646, 0x3fd474fd543f222c
data8 0x3fd4bd1eb680e548, 0x3fd505789e234bd1
data8 0x3fd54e0b64003b70, 0x3fd596d761c3c1f0
data8 0x3fd5dfdcf1eeae0e, 0x3fd6291c6fd9329c
data8 0x3fd6729637b59418, 0x3fd6bc4aa692e0fd
data8 0x3fd7063a1a5fb4f2, 0x3fd75064f1ed0715
data8 0x3fd79acb8cf10390, 0x3fd7d67c1e43ae5c
data8 0x3fd8214f4068afa7, 0x3fd86c5f36dea3dc
data8 0x3fd8b7ac64dd7f9d, 0x3fd8f4167a0c6f92
data8 0x3fd93fd2d5e1bf1d, 0x3fd98bcd84296946
data8 0x3fd9c8c333e6e9a5, 0x3fda152f142981b4
data8 0x3fda527fd95fd8ff, 0x3fda9f5e3edeb9e6
data8 0x3fdadd0b2b5755a7, 0x3fdb2a5d6f51ff83
data8 0x3fdb686799b00be3, 0x3fdbb62f1b887cd8
data8 0x3fdbf4979f666668, 0x3fdc332a6e8399d4
data8 0x3fdc819dc2d45fe4, 0x3fdcc0908e19b7bd
data8 0x3fdcffae611ad12b, 0x3fdd3ef776d43ff4
data8 0x3fdd8e5002710128, 0x3fddcdfb486cb9a1
data8 0x3fde0dd294245fe4, 0x3fde4dd622a28840
data8 0x3fde8e06317114f0, 0x3fdece62fe9a9915
data8 0x3fdf1f164a15389a, 0x3fdf5fd8a9063e35
data8 0x3fdfa0c8937e7d5d, 0x3fdfe1e649bb6335
data8 0x3fe011990641535a, 0x3fe032560e91e59e
data8 0x3fe0532a5ebcd44a, 0x3fe0741617f5fc28
data8 0x3fe08cd653f38839, 0x3fe0adeb55c1103b
data8 0x3fe0cf181d5d1dd0, 0x3fe0f05ccd0aced7
data8 0x3fe111b9875788ab, 0x3fe1332e6f1bcf73
data8 0x3fe154bba77c2088, 0x3fe16df59bfa06c1
data8 0x3fe18fadb6e2d3c2, 0x3fe1b17e849adc26
data8 0x3fe1caeb6a0de814, 0x3fe1ece7c830eec9
data8 0x3fe20efd3dae01df, 0x3fe2289de375d901
data8 0x3fe24adf9b6a6fe0, 0x3fe26d3ad1aebcfc
data8 0x3fe287100c2771f4, 0x3fe2a9983b3c1b28
data8 0xbfda78e146f7bef4, 0xbfda33760a7f6051
data8 0xbfd9ff43476fb5f7, 0xbfd9b97c3c4eec8f
data8 0xbfd98504431717fc, 0xbfd93ee07535f967
data8 0xbfd90a228d5712b2, 0xbfd8c3a104cb24f5
data8 0xbfd88e9c72e0b226, 0xbfd847bc33d8618e
data8 0xbfd812703988bb69, 0xbfd7dd0569c04bff
data8 0xbfd7959c202292f1, 0xbfd75fe8d2c5d48f
data8 0xbfd72a1637cbc183, 0xbfd6e221cd9d0cde
data8 0xbfd6ac059985503b, 0xbfd675c99ce81f92
data8 0xbfd63f6db2590482, 0xbfd5f6c138136489
data8 0xbfd5c01a39fbd688, 0xbfd58952cf519193
data8 0xbfd5526ad18493ce, 0xbfd51b6219bfe6ea
data8 0xbfd4d1cdf8b4846f, 0xbfd49a784bcd1b8b
data8 0xbfd4630161832547, 0xbfd42b6911cf5465
data8 0xbfd3f3af3461e1c4, 0xbfd3bbd3a0a1dcfb
data8 0xbfd383d62dac7ae7, 0xbfd34bb6b2546218
data8 0xbfd313750520f520, 0xbfd2db10fc4d9aaf
data8 0xbfd2a28a6dc90387, 0xbfd269e12f346e2c
data8 0xbfd2311515e2e855, 0xbfd1f825f6d88e13
data8 0xbfd1bf13a6c9c69f, 0xbfd185ddfa1a7ed0
data8 0xbfd14c84c4dd6128, 0xbfd11307dad30b76
data8 0xbfd0d9670f6941fe, 0xbfd09fa235ba2020
data8 0xbfd0790adbb03009, 0xbfd03f09858c55fb
data8 0xbfd004e3a7c97cbd, 0xbfcf9532288fcf69
data8 0xbfcf205339208f27, 0xbfceab2a23a5b83e
data8 0xbfce5ce55fdd37a5, 0xbfcde73fe3b1480f
data8 0xbfcd714f44623927, 0xbfccfb1321b8c400
data8 0xbfccac163c770dc9, 0xbfcc355b67195dd0
data8 0xbfcbbe540a3f036f, 0xbfcb6ecf175f95e9
data8 0xbfcaf74751e1be33, 0xbfca7f71fb7bab9d
data8 0xbfca2f632320b86b, 0xbfc9b70ba539dfae
data8 0xbfc93e6587910444, 0xbfc8edcae8352b6c
data8 0xbfc874a0db01a719, 0xbfc7fb27199df16d
data8 0xbfc7a9fec7d05ddf, 0xbfc72fff456ac70d
data8 0xbfc6de7d66023dbc, 0xbfc663f6fac91316
data8 0xbfc6121ac74813cf, 0xbfc5970c478fff4a
data8 0xbfc51bab907a5c8a, 0xbfc4c93d33151b24
data8 0xbfc44d527fdadf55, 0xbfc3fa87be0f3a1b
data8 0xbfc3a797cd35d959, 0xbfc32ae9e278ae1a
data8 0xbfc2d79c6937efdd, 0xbfc25a619370d9dc
data8 0xbfc206b5bde2f8b8, 0xbfc188ecbd1d16be
data8 0xbfc134e1b489062e, 0xbfc0b6894488e95f
data8 0xbfc0621e2f556b5c, 0xbfc00d8c711a12cc
data8 0xbfbf1cd21257e18c, 0xbfbe72ec117fa5b2
data8 0xbfbdc8b7c49a1ddb, 0xbfbcc8d5e467b710
data8 0xbfbc1ddc9c39c7a1, 0xbfbb7294093cdd0f
data8 0xbfba7111df348494, 0xbfb9c501cdf75872
data8 0xbfb918a16e46335b, 0xbfb81579a73e83c6
data8 0xbfb7684f39f4ff2d, 0xbfb6bad3758efd87
data8 0xbfb60d060d7e41ac, 0xbfb507b836033bb7
data8 0xbfb4591d6310d85a, 0xbfb3aa2fdd27f1c3
data8 0xbfb2faef55ccb372, 0xbfb1f3723b4ae6db
data8 0xbfb14360d6136ffa, 0xbfb092fb594145c1
data8 0xbfafc482e8b48a7e, 0xbfae6265ace11ae4
data8 0xbfacff9e5c4341d0, 0xbfaaea3316095f72
data8 0xbfa985bfc3495194, 0xbfa820a01ac754cb
data8 0xbfa6bad3758efd87, 0xbfa554592bb8cd58
data8 0xbfa3ed3094685a26, 0xbfa2855905ca70f6
data8 0xbfa11cd1d5133413, 0xbf9dfd78881399f1
data8 0xbf9b28f618cc85df, 0xbf98530faa3c087b
data8 0xbf957bc3dddcd7fa, 0xbf92a3115322f9e6
data8 0xbf8f91ed4eef8370, 0xbf89dae4ec6b8b2e
data8 0xbf842106b1499209, 0xbf7cc89f97d67594
data8 0xbf71497accf7e11d, 0x0000000000000000
LOCAL_OBJECT_END(T_table)


.section .text
WEAK_LIBM_ENTRY(log2f)

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
  // f12=0.5
  setf.exp f12=r23
  nop.f 0
  // r27=bias
  mov r27=0xffff;;
}

{.mfb
  // denormal input, repeat get exponent (after normalization)
  (p8) getf.exp r29=f7
  nop.f 0
  (p12) br.cond.spnt SPECIAL_log2f
}
{.mfi
  cmp.geu p12,p0=r25,r24
  nop.f 0
  mov r23=0xff;;
}

{.mfi
  add r3=32,r2
  // r=1-x*y
  fms.s1 f6=f6,f8,f1
  // r26=0x80400...0 (threshold for using polynomial approximation)
  shl r26=r26,64-12
}
{.mfi
  // load C_3, C_4
  ldfpd f10,f11=[r2],16
  nop.f 0
  // r27=bias-1 (if index >=128, will add exponent+1)
  (p12) mov r27=0xfffe;;
}

{.mfi
  // load C_1
  ldfe f14=[r2],32
  // x=1, return 0
  (p6) fma.s.s0 f8=f0,f0,f0
  (p8) shr.u r28=r25,63-8
}
{.mib
  // load C_2
  ldfe f13=[r3]
  // r29=exponent-bias
  sub r29=r29,r27
  // x=1, return
  (p6) br.ret.spnt b0;;
}


{.mfi
  // get T_index
  and r28=r28,r23
  fmerge.se f7=f1,f7
  // if first 9 bits after leading 1 are all zero, then p8=1
  cmp.ltu p8,p12=r25,r26;;
}
{.mfi
  // f8=expon - bias
  setf.sig f8=r29
  nop.f 0
  // get T address
  shladd r2=r28,3,r2
}
{.mfi
  // first 8 bits after leading 1 are all ones ?
  cmp.eq p10,p0=r23,r28
  // if first 8 bits after leading bit are 0, use polynomial approx. only
  (p8) fms.s1 f6=f7,f1,f1
  nop.i 0;;
}
{.mfi
  //r26=1
  mov r26=1
  // if first 8 bits after leading 1 are all ones, use polynomial approx. only
  (p10) fms.s1 f6=f7,f12,f1
  nop.i 0;;
}

.pred.rel "mutex",p8,p12
{.mmf
  // load T (unless first 9 bits after leading 1 are 0)
  (p12) ldfd f12=[r2]
  nop.m 0
  // set T=0 (if first 9 bits after leading 1 are 0)
  (p8) fma.s1 f12=f0,f0,f0;;
}

{.mfi
  nop.m 0
  // P34=C_3+C_4*r
  fma.s1 f10=f11,f6,f10
  // r26=2^{63}
  shl r26=r26,63
}
{.mfi
  nop.m 0
  // r2=r*r
  fma.s1 f11=f6,f6,f0
  nop.i 0;;
}
{.mfi
  // significand of x is 1 ?
  cmp.eq p0,p6=r25,r26
  // P12=C_1+C_2*r
  fma.s1 f14=f13,f6,f14
  nop.i 0;;
}
{.mfi
  nop.m 0
  // normalize additive term (l=exponent of x)
  fcvt.xf f8=f8
  // if significand(x)=1, return exponent (l)
  nop.i 0;;
}
{.mfi
  nop.m 0
  // add T+l
  (p6) fma.s1 f8=f8,f1,f12
  nop.i 0
}
{.mfi
  nop.m 0
  // P14=P12+r2*P34
  (p6) fma.s1 f13=f10,f11,f14
  nop.i 0;;
}

{.mfb
  nop.m 0
  // result=T+l+r*P14
  (p6) fma.s.s0 f8=f13,f6,f8
  // return
  br.ret.sptk b0;;
}


SPECIAL_log2f:
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
  // log2f(+Infinity)=+Infinity
  nop.f 0
  (p7) br.ret.spnt b0;;
}
{.mfi
  (p8) mov GR_Parameter_TAG = 172
  // log2f(+/-0)=-infinity, raises Divide by Zero
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
  (p6) mov GR_Parameter_TAG = 173
  // x<0: return NaN, raise Invalid
  (p6) frcpa.s0 f8,p0=f0,f0
  (p6) br.cond.sptk __libm_error_region;;
}


{.mfb
  nop.m 0
  // Remaining cases: NaNs
  fma.s.s0 f8=f8,f1,f0
  br.ret.sptk b0;;
}

WEAK_LIBM_END(log2f)
libm_alias_float_other (__log2, log2)
#ifdef SHARED
.symver log2f,log2f@@GLIBC_2.27
.weak __log2f_compat
.set __log2f_compat,__log2f
.symver __log2f_compat,log2f@GLIBC_2.2
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
        stfs [GR_Parameter_Y] = FR_Y,16         // STORE Parameter 2 on stack
        add GR_Parameter_X = 16,sp              // Parameter 1 address
.save   b0, GR_SAVE_B0
        mov GR_SAVE_B0=b0                       // Save b0
};;
.body
{ .mib
        stfs [GR_Parameter_X] = FR_X                  // STORE Parameter 1 on stack
        add   GR_Parameter_RESULT = 0,GR_Parameter_Y  // Parameter 3 address
	nop.b 0
}
{ .mib
        stfs [GR_Parameter_Y] = FR_RESULT             // STORE Parameter 3 on stack
        add   GR_Parameter_Y = -16,GR_Parameter_Y
        br.call.sptk b0=__libm_error_support#         // Call error handling function
};;
{ .mmi
        nop.m 0
        nop.m 0
        add   GR_Parameter_RESULT = 48,sp
};;
{ .mmi
        ldfs  f8 = [GR_Parameter_RESULT]       // Get return result off stack
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
