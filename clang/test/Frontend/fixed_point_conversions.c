// RUN: %clang_cc1 -ffixed-point -S -emit-llvm %s -o - | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang_cc1 -ffixed-point -S -emit-llvm %s -o - -fpadding-on-unsigned-fixed-point | FileCheck %s -check-prefix=SAME

void TestFixedPointCastSameType() {
  _Accum a = 2.5k;
  _Accum a2 = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a2, align 4

  a2 = (_Accum)a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a2, align 4
}

void TestFixedPointCastDown() {
  long _Accum la = 2.5lk;
  _Accum a = la;
  // DEFAULT:      [[LACCUM:%[0-9a-z]+]] = load i64, i64* %la, align 8
  // DEFAULT-NEXT: [[ACCUM_AS_I64:%[0-9a-z]+]] = ashr i64 [[LACCUM]], 16
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = trunc i64 [[ACCUM_AS_I64]] to i32
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  a = (_Accum)la;
  // DEFAULT:      [[LACCUM:%[0-9a-z]+]] = load i64, i64* %la, align 8
  // DEFAULT-NEXT: [[ACCUM_AS_I64:%[0-9a-z]+]] = ashr i64 [[LACCUM]], 16
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = trunc i64 [[ACCUM_AS_I64]] to i32
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  short _Accum sa = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[SACCUM_AS_I32:%[0-9a-z]+]] = ashr i32 [[ACCUM]], 8
  // DEFAULT-NEXT: [[SACCUM:%[0-9a-z]+]] = trunc i32 [[SACCUM_AS_I32]] to i16
  // DEFAULT-NEXT: store i16 [[SACCUM]], i16* %sa, align 2

  sa = (short _Accum)a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[SACCUM_AS_I32:%[0-9a-z]+]] = ashr i32 [[ACCUM]], 8
  // DEFAULT-NEXT: [[SACCUM:%[0-9a-z]+]] = trunc i32 [[SACCUM_AS_I32]] to i16
  // DEFAULT-NEXT: store i16 [[SACCUM]], i16* %sa, align 2
}

void TestFixedPointCastUp() {
  short _Accum sa = 2.5hk;
  _Accum a = sa;
  // DEFAULT:      [[SACCUM:%[0-9a-z]+]] = load i16, i16* %sa, align 2
  // DEFAULT-NEXT: [[SACCUM_BUFF:%[0-9a-z]+]] = sext i16 [[SACCUM]] to i32
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[SACCUM_BUFF]], 8
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  long _Accum la = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[ACCUM_BUFF:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // DEFAULT-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_BUFF]], 16
  // DEFAULT-NEXT: store i64 [[LACCUM]], i64* %la, align 8

  a = (_Accum)sa;
  // DEFAULT:      [[SACCUM:%[0-9a-z]+]] = load i16, i16* %sa, align 2
  // DEFAULT-NEXT: [[SACCUM_BUFF:%[0-9a-z]+]] = sext i16 [[SACCUM]] to i32
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[SACCUM_BUFF]], 8
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  la = (long _Accum)a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[ACCUM_BUFF:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // DEFAULT-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_BUFF]], 16
  // DEFAULT-NEXT: store i64 [[LACCUM]], i64* %la, align 8
}

void TestFixedPointCastSignedness() {
  _Accum a = 2.5k;
  unsigned _Accum ua = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[UACCUM:%[0-9a-z]+]] = shl i32 [[ACCUM]], 1
  // DEFAULT-NEXT: store i32 [[UACCUM]], i32* %ua, align 4
  // SAME:      TestFixedPointCastSignedness
  // SAME:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // SAME-NEXT: store i32 [[ACCUM]], i32* %ua, align 4

  a = ua;
  // DEFAULT:      [[UACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = lshr i32 [[UACCUM]], 1
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4
  // SAME:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // SAME-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  ua = (unsigned _Accum)a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[UACCUM:%[0-9a-z]+]] = shl i32 [[ACCUM]], 1
  // DEFAULT-NEXT: store i32 [[UACCUM]], i32* %ua, align 4

  a = (_Accum)ua;
  // DEFAULT:      [[UACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = lshr i32 [[UACCUM]], 1
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  _Accum a2;
  unsigned long _Accum ula = a2;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a2, align 4
  // DEFAULT-NEXT: [[ACCUM_EXT:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // DEFAULT-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_EXT]], 17
  // DEFAULT-NEXT: store i64 [[LACCUM]], i64* %ula, align 8
  // SAME:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a2, align 4
  // SAME-NEXT: [[ACCUM_EXT:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // SAME-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_EXT]], 16
  // SAME-NEXT: store i64 [[LACCUM]], i64* %ula, align 8
}

void TestFixedPointCastSaturation() {
  _Accum a;
  _Sat short _Accum sat_sa;
  _Sat _Accum sat_a;
  _Sat long _Accum sat_la;
  _Sat unsigned short _Accum sat_usa;
  _Sat unsigned _Accum sat_ua;
  _Sat unsigned long _Accum sat_ula;
  _Sat short _Fract sat_sf;
  _Sat _Fract sat_f;
  _Sat long _Fract sat_lf;

  // Casting down between types
  sat_sa = sat_a;
  // DEFAULT:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 8
  // DEFAULT-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 32767
  // DEFAULT-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 32767, i32 [[ACCUM]]
  // DEFAULT-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], -32768
  // DEFAULT-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 -32768, i32 [[RESULT]]
  // DEFAULT-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // DEFAULT-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_sa, align 2

  // Accum to Fract, decreasing scale
  sat_sf = sat_a;
  // DEFAULT:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // DEFAULT-NEXT: [[FRACT:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 8
  // DEFAULT-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[FRACT]], 127
  // DEFAULT-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 127, i32 [[FRACT]]
  // DEFAULT-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], -128
  // DEFAULT-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 -128, i32 [[RESULT]]
  // DEFAULT-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i8
  // DEFAULT-NEXT: store i8 [[RESULT_TRUNC]], i8* %sat_sf, align 1

  // Accum to Fract, same scale
  sat_f = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 32767
  // DEFAULT-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 32767, i32 [[ACCUM]]
  // DEFAULT-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], -32768
  // DEFAULT-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 -32768, i32 [[RESULT]]
  // DEFAULT-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // DEFAULT-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_f, align 2

  // Accum to Fract, increasing scale
  sat_lf = sat_a;
  // DEFAULT:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = sext i32 [[OLD_ACCUM]] to i48
  // DEFAULT-NEXT: [[FRACT:%[0-9a-z]+]] = shl i48 [[ACCUM]], 16
  // DEFAULT-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i48 [[FRACT]], 2147483647
  // DEFAULT-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i48 2147483647, i48 [[FRACT]]
  // DEFAULT-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i48 [[RESULT]], -2147483648
  // DEFAULT-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i48 -2147483648, i48 [[RESULT]]
  // DEFAULT-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i48 [[RESULT2]] to i32
  // DEFAULT-NEXT: store i32 [[RESULT_TRUNC]], i32* %sat_lf, align 4

  // Signed to unsigned, decreasing scale
  _Sat _Accum sat_a2;
  sat_usa = sat_a2;
  // DEFAULT:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a2, align 4
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 7
  // DEFAULT-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 65535
  // DEFAULT-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 65535, i32 [[ACCUM]]
  // DEFAULT-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], 0
  // DEFAULT-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 0, i32 [[RESULT]]
  // DEFAULT-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // DEFAULT-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_usa, align 2
  // SAME:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a2, align 4
  // SAME-NEXT: [[ACCUM:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 8
  // SAME-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 32767
  // SAME-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 32767, i32 [[ACCUM]]
  // SAME-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], 0
  // SAME-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 0, i32 [[RESULT]]
  // SAME-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // SAME-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_usa, align 2

  // Signed to unsigned, increasing scale
  sat_ua = sat_a;
  // DEFAULT:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // DEFAULT-NEXT: [[ACCUM_EXT:%[0-9a-z]+]] = sext i32 [[OLD_ACCUM]] to i33
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i33 [[ACCUM_EXT]], 1
  // DEFAULT-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i33 [[ACCUM]], 0
  // DEFAULT-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i33 0, i33 [[ACCUM]]
  // DEFAULT-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i33 [[RESULT2]] to i32
  // DEFAULT-NEXT: store i32 [[RESULT_TRUNC]], i32* %sat_ua, align 4
  // SAME:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // SAME-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[ACCUM]], 0
  // SAME-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 0, i32 [[ACCUM]]
  // SAME-NEXT: store i32 [[RESULT]], i32* %sat_ua, align 4

  // Nothing when saturating to the same type and size
  sat_a = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %sat_a, align 4

  // Nothing when assigning back
  a = sat_a;
  // DEFAULT:      [[SAT_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // DEFAULT-NEXT: store i32 [[SAT_ACCUM]], i32* %a, align 4

  // No overflow when casting from fract to signed accum
  sat_a = sat_f;
  // DEFAULT:      [[FRACT:%[0-9a-z]+]] = load i16, i16* %sat_f, align 2
  // DEFAULT-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i16 [[FRACT]] to i32
  // DEFAULT-NEXT: store i32 [[FRACT_EXT]], i32* %sat_a, align 4

  // Only get overflow checking if signed fract to unsigned accum
  sat_ua = sat_sf;
  // DEFAULT:      [[FRACT:%[0-9a-z]+]] = load i8, i8* %sat_sf, align 1
  // DEFAULT-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i8 [[FRACT]] to i17
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i17 [[FRACT_EXT]], 9
  // DEFAULT-NEXT: [[IS_NEG:%[0-9a-z]+]] = icmp slt i17 [[ACCUM]], 0
  // DEFAULT-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[IS_NEG]], i17 0, i17 [[ACCUM]]
  // DEFAULT-NEXT: [[RESULT_EXT:%[0-9a-z]+]] = sext i17 [[RESULT]] to i32
  // DEFAULT-NEXT: store i32 [[RESULT_EXT]], i32* %sat_ua, align 4
  // SAME:      [[FRACT:%[0-9a-z]+]] = load i8, i8* %sat_sf, align 1
  // SAME-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i8 [[FRACT]] to i16
  // SAME-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i16 [[FRACT_EXT]], 8
  // SAME-NEXT: [[IS_NEG:%[0-9a-z]+]] = icmp slt i16 [[ACCUM]], 0
  // SAME-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[IS_NEG]], i16 0, i16 [[ACCUM]]
  // SAME-NEXT: [[RESULT_EXT:%[0-9a-z]+]] = sext i16 [[RESULT]] to i32
  // SAME-NEXT: store i32 [[RESULT_EXT]], i32* %sat_ua, align 4
}

void TestFixedPointCastBetFractAccum() {
  short _Accum sa;
  _Accum a;
  long _Accum la;
  short _Fract sf;
  _Fract f;
  long _Fract lf;
  unsigned _Accum ua;
  unsigned _Fract uf;

  // To lower scale
  sf = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[FRACT:%[0-9a-z]+]] = ashr i32 [[ACCUM]], 8
  // DEFAULT-NEXT: [[FRACT_TRUNC:%[0-9a-z]+]] = trunc i32 [[FRACT]] to i8
  // DEFAULT-NEXT: store i8 [[FRACT_TRUNC]], i8* %sf, align 1

  // To higher scale
  a = sf;
  // DEFAULT:      [[FRACT:%[0-9a-z]+]] = load i8, i8* %sf, align 1
  // DEFAULT-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i8 [[FRACT]] to i32
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[FRACT_EXT]], 8
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  // To same scale
  f = a;
  // DEFAULT:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // DEFAULT-NEXT: [[FRACT:%[0-9a-z]+]] = trunc i32 [[ACCUM]] to i16
  // DEFAULT-NEXT: store i16 [[FRACT]], i16* %f, align 2

  a = f;
  // DEFAULT:      [[FRACT:%[0-9a-z]+]] = load i16, i16* %f, align 2
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = sext i16 [[FRACT]] to i32
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  // To unsigned
  ua = uf;
  // DEFAULT:      [[FRACT:%[0-9a-z]+]] = load i16, i16* %uf, align 2
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = zext i16 [[FRACT]] to i32
  // DEFAULT-NEXT: store i32 [[ACCUM]], i32* %ua, align 4
  // SAME:      [[FRACT:%[0-9a-z]+]] = load i16, i16* %uf, align 2
  // SAME-NEXT: [[ACCUM:%[0-9a-z]+]] = zext i16 [[FRACT]] to i32
  // SAME-NEXT: store i32 [[ACCUM]], i32* %ua, align 4

  uf = ua;
  // DEFAULT:      [[FRACT:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // DEFAULT-NEXT: [[ACCUM:%[0-9a-z]+]] = trunc i32 [[FRACT]] to i16
  // DEFAULT-NEXT: store i16 [[ACCUM]], i16* %uf, align 2
  // SAME:      [[FRACT:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // SAME-NEXT: [[ACCUM:%[0-9a-z]+]] = trunc i32 [[FRACT]] to i16
  // SAME-NEXT: store i16 [[ACCUM]], i16* %uf, align 2
}
