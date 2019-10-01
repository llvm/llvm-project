// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - -fpadding-on-unsigned-fixed-point | FileCheck %s --check-prefixes=CHECK,UNSIGNED

// Between different fixed point types
short _Accum sa_const = 2.5hk; // CHECK-DAG: @sa_const  = {{.*}}global i16 320, align 2
_Accum a_const = 2.5hk;        // CHECK-DAG: @a_const   = {{.*}}global i32 81920, align 4
short _Accum sa_const2 = 2.5k; // CHECK-DAG: @sa_const2 = {{.*}}global i16 320, align 2

short _Accum sa_from_f_const = 0.5r; // CHECK-DAG: sa_from_f_const = {{.*}}global i16 64, align 2
_Fract f_from_sa_const = 0.5hk;      // CHECK-DAG: f_from_sa_const = {{.*}}global i16 16384, align 2

unsigned short _Accum usa_const = 2.5uk;
unsigned _Accum ua_const = 2.5uhk;
// SIGNED-DAG: @usa_const  = {{.*}}global i16 640, align 2
// SIGNED-DAG: @ua_const   = {{.*}}global i32 163840, align 4
// UNSIGNED-DAG:    @usa_const  = {{.*}}global i16 320, align 2
// UNSIGNED-DAG:    @ua_const   = {{.*}}global i32 81920, align 4

// FixedPoint to integer
int i_const = -128.0hk;  // CHECK-DAG: @i_const  = {{.*}}global i32 -128, align 4
int i_const2 = 128.0hk;  // CHECK-DAG: @i_const2 = {{.*}}global i32 128, align 4
int i_const3 = -128.0k;  // CHECK-DAG: @i_const3 = {{.*}}global i32 -128, align 4
int i_const4 = 128.0k;   // CHECK-DAG: @i_const4 = {{.*}}global i32 128, align 4
short s_const = -128.0k; // CHECK-DAG: @s_const  = {{.*}}global i16 -128, align 2
short s_const2 = 128.0k; // CHECK-DAG: @s_const2 = {{.*}}global i16 128, align 2

// Integer to fixed point
short _Accum sa_const5 = 2;    // CHECK-DAG: @sa_const5 = {{.*}}global i16 256, align 2
short _Accum sa_const6 = -2;   // CHECK-DAG: @sa_const6 = {{.*}}global i16 -256, align 2
short _Accum sa_const7 = -256; // CHECK-DAG: @sa_const7 = {{.*}}global i16 -32768, align 2

// Signedness
unsigned short _Accum usa_const2 = 2.5hk;
// SIGNED-DAG: @usa_const2  = {{.*}}global i16 640, align 2
// UNSIGNED-DAG:    @usa_const2  = {{.*}}global i16 320, align 2
short _Accum sa_const3 = 2.5hk; // CHECK-DAG: @sa_const3 = {{.*}}global i16 320, align 2

int i_const5 = 128.0uhk;
unsigned int ui_const = 128.0hk;
// CHECK-DAG: @i_const5  = {{.*}}global i32 128, align 4
// CHECK-DAG: @ui_const  = {{.*}}global i32 128, align 4

short _Accum sa_const9 = 2u; // CHECK-DAG: @sa_const9 = {{.*}}global i16 256, align 2
unsigned short _Accum usa_const3 = 2;
// SIGNED-DAG: @usa_const3 = {{.*}}global i16 512, align 2
// UNSIGNED-DAG:    @usa_const3 = {{.*}}global i16 256, align 2

// Overflow (this is undefined but allowed)
short _Accum sa_const4 = 256.0k;
unsigned int ui_const2 = -2.5hk;
short _Accum sa_const8 = 256;
unsigned short _Accum usa_const4 = -2;

// Saturation
_Sat short _Accum sat_sa_const = 2.5hk;   // CHECK-DAG: @sat_sa_const  = {{.*}}global i16 320, align 2
_Sat short _Accum sat_sa_const2 = 256.0k; // CHECK-DAG: @sat_sa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const = -1.0hk;
// CHECK-DAG: @sat_usa_const = {{.*}}global i16 0, align 2
_Sat unsigned short _Accum sat_usa_const2 = 256.0k;
// SIGNED-DAG: @sat_usa_const2 = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG:    @sat_usa_const2 = {{.*}}global i16 32767, align 2

_Sat short _Accum sat_sa_const3 = 256;  // CHECK-DAG: @sat_sa_const3 = {{.*}}global i16 32767, align 2
_Sat short _Accum sat_sa_const4 = -257; // CHECK-DAG: @sat_sa_const4 = {{.*}}global i16 -32768, align 2
_Sat unsigned short _Accum sat_usa_const3 = -1;
// CHECK-DAG: @sat_usa_const3 = {{.*}}global i16 0, align 2
_Sat unsigned short _Accum sat_usa_const4 = 256;
// SIGNED-DAG: @sat_usa_const4 = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG:    @sat_usa_const4 = {{.*}}global i16 32767, align 2

void TestFixedPointCastSameType() {
  _Accum a = 2.5k;
  _Accum a2 = a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a2, align 4

  a2 = (_Accum)a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a2, align 4
}

void TestFixedPointCastDown() {
  long _Accum la = 2.5lk;
  _Accum a = la;
  // CHECK:      [[LACCUM:%[0-9a-z]+]] = load i64, i64* %la, align 8
  // CHECK-NEXT: [[ACCUM_AS_I64:%[0-9a-z]+]] = ashr i64 [[LACCUM]], 16
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = trunc i64 [[ACCUM_AS_I64]] to i32
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  a = (_Accum)la;
  // CHECK:      [[LACCUM:%[0-9a-z]+]] = load i64, i64* %la, align 8
  // CHECK-NEXT: [[ACCUM_AS_I64:%[0-9a-z]+]] = ashr i64 [[LACCUM]], 16
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = trunc i64 [[ACCUM_AS_I64]] to i32
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  short _Accum sa = a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[SACCUM_AS_I32:%[0-9a-z]+]] = ashr i32 [[ACCUM]], 8
  // CHECK-NEXT: [[SACCUM:%[0-9a-z]+]] = trunc i32 [[SACCUM_AS_I32]] to i16
  // CHECK-NEXT: store i16 [[SACCUM]], i16* %sa, align 2

  sa = (short _Accum)a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[SACCUM_AS_I32:%[0-9a-z]+]] = ashr i32 [[ACCUM]], 8
  // CHECK-NEXT: [[SACCUM:%[0-9a-z]+]] = trunc i32 [[SACCUM_AS_I32]] to i16
  // CHECK-NEXT: store i16 [[SACCUM]], i16* %sa, align 2
}

void TestFixedPointCastUp() {
  short _Accum sa = 2.5hk;
  _Accum a = sa;
  // CHECK:      [[SACCUM:%[0-9a-z]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[SACCUM_BUFF:%[0-9a-z]+]] = sext i16 [[SACCUM]] to i32
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[SACCUM_BUFF]], 8
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  long _Accum la = a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[ACCUM_BUFF:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // CHECK-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_BUFF]], 16
  // CHECK-NEXT: store i64 [[LACCUM]], i64* %la, align 8

  a = (_Accum)sa;
  // CHECK:      [[SACCUM:%[0-9a-z]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[SACCUM_BUFF:%[0-9a-z]+]] = sext i16 [[SACCUM]] to i32
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[SACCUM_BUFF]], 8
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  la = (long _Accum)a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[ACCUM_BUFF:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // CHECK-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_BUFF]], 16
  // CHECK-NEXT: store i64 [[LACCUM]], i64* %la, align 8
}

void TestFixedPointCastSignedness() {
  _Accum a = 2.5k;
  unsigned _Accum ua = a;
  // SIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // SIGNED-NEXT: [[UACCUM:%[0-9a-z]+]] = shl i32 [[ACCUM]], 1
  // SIGNED-NEXT: store i32 [[UACCUM]], i32* %ua, align 4
  // UNSIGNED:      TestFixedPointCastSignedness
  // UNSIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // UNSIGNED-NEXT: store i32 [[ACCUM]], i32* %ua, align 4

  a = ua;
  // SIGNED:      [[UACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // SIGNED-NEXT: [[ACCUM:%[0-9a-z]+]] = lshr i32 [[UACCUM]], 1
  // SIGNED-NEXT: store i32 [[ACCUM]], i32* %a, align 4
  // UNSIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // UNSIGNED-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  ua = (unsigned _Accum)a;
  // SIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // SIGNED-NEXT: [[UACCUM:%[0-9a-z]+]] = shl i32 [[ACCUM]], 1
  // SIGNED-NEXT: store i32 [[UACCUM]], i32* %ua, align 4
  // UNSIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // UNSIGNED-NEXT: store i32 [[ACCUM]], i32* %ua, align 4

  a = (_Accum)ua;
  // SIGNED:      [[UACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // SIGNED-NEXT: [[ACCUM:%[0-9a-z]+]] = lshr i32 [[UACCUM]], 1
  // SIGNED-NEXT: store i32 [[ACCUM]], i32* %a, align 4
  // UNSIGNED:      [[UACCUM:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // UNSIGNED-NEXT: store i32 [[UACCUM]], i32* %a, align 4

  _Accum a2;
  unsigned long _Accum ula = a2;
  // SIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a2, align 4
  // SIGNED-NEXT: [[ACCUM_EXT:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // SIGNED-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_EXT]], 17
  // SIGNED-NEXT: store i64 [[LACCUM]], i64* %ula, align 8
  // UNSIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a2, align 4
  // UNSIGNED-NEXT: [[ACCUM_EXT:%[0-9a-z]+]] = sext i32 [[ACCUM]] to i64
  // UNSIGNED-NEXT: [[LACCUM:%[0-9a-z]+]] = shl i64 [[ACCUM_EXT]], 16
  // UNSIGNED-NEXT: store i64 [[LACCUM]], i64* %ula, align 8
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
  // CHECK:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 8
  // CHECK-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 32767
  // CHECK-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 32767, i32 [[ACCUM]]
  // CHECK-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], -32768
  // CHECK-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 -32768, i32 [[RESULT]]
  // CHECK-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // CHECK-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_sa, align 2

  // Accum to Fract, decreasing scale
  sat_sf = sat_a;
  // CHECK:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // CHECK-NEXT: [[FRACT:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 8
  // CHECK-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[FRACT]], 127
  // CHECK-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 127, i32 [[FRACT]]
  // CHECK-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], -128
  // CHECK-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 -128, i32 [[RESULT]]
  // CHECK-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i8
  // CHECK-NEXT: store i8 [[RESULT_TRUNC]], i8* %sat_sf, align 1

  // Accum to Fract, same scale
  sat_f = a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 32767
  // CHECK-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 32767, i32 [[ACCUM]]
  // CHECK-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], -32768
  // CHECK-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 -32768, i32 [[RESULT]]
  // CHECK-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // CHECK-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_f, align 2

  // Accum to Fract, increasing scale
  sat_lf = sat_a;
  // CHECK:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // CHECK-NEXT: [[RESIZE:%[0-9a-z]+]] = sext i32 [[OLD_ACCUM]] to i48
  // CHECK-NEXT: [[FRACT:%[0-9a-z]+]] = shl i48 [[RESIZE]], 16
  // CHECK-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i48 [[FRACT]], 2147483647
  // CHECK-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i48 2147483647, i48 [[FRACT]]
  // CHECK-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i48 [[RESULT]], -2147483648
  // CHECK-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i48 -2147483648, i48 [[RESULT]]
  // CHECK-NEXT: [[TRUNC:%[0-9a-z]+]] = trunc i48 [[RESULT2]] to i32
  // CHECK-NEXT: store i32 [[TRUNC]], i32* %sat_lf, align 4

  // Signed to unsigned, decreasing scale
  _Sat _Accum sat_a2;
  sat_usa = sat_a2;
  // SIGNED:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a2, align 4
  // SIGNED-NEXT: [[ACCUM:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 7
  // SIGNED-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 65535
  // SIGNED-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 65535, i32 [[ACCUM]]
  // SIGNED-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], 0
  // SIGNED-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 0, i32 [[RESULT]]
  // SIGNED-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // SIGNED-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_usa, align 2
  // UNSIGNED:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a2, align 4
  // UNSIGNED-NEXT: [[ACCUM:%[0-9a-z]+]] = ashr i32 [[OLD_ACCUM]], 8
  // UNSIGNED-NEXT: [[USE_MAX:%[0-9a-z]+]] = icmp sgt i32 [[ACCUM]], 32767
  // UNSIGNED-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MAX]], i32 32767, i32 [[ACCUM]]
  // UNSIGNED-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[RESULT]], 0
  // UNSIGNED-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 0, i32 [[RESULT]]
  // UNSIGNED-NEXT: [[RESULT_TRUNC:%[0-9a-z]+]] = trunc i32 [[RESULT2]] to i16
  // UNSIGNED-NEXT: store i16 [[RESULT_TRUNC]], i16* %sat_usa, align 2

  // Signed to unsigned, increasing scale
  sat_ua = sat_a;
  // SIGNED:      [[OLD_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // SIGNED-NEXT: [[RESIZE:%[0-9a-z]+]] = sext i32 [[OLD_ACCUM]] to i33
  // SIGNED-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i33 [[RESIZE]], 1
  // SIGNED-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i33 [[ACCUM]], 0
  // SIGNED-NEXT: [[RESULT2:%[0-9a-z]+]] = select i1 [[USE_MIN]], i33 0, i33 [[ACCUM]]
  // SIGNED-NEXT: [[TRUNC:%[0-9a-z]+]] = trunc i33 [[RESULT2]] to i32
  // SIGNED-NEXT: store i32 [[TRUNC]], i32* %sat_ua, align 4
  // UNSIGNED:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // UNSIGNED-NEXT: [[USE_MIN:%[0-9a-z]+]] = icmp slt i32 [[ACCUM]], 0
  // UNSIGNED-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[USE_MIN]], i32 0, i32 [[ACCUM]]
  // UNSIGNED-NEXT: store i32 [[RESULT]], i32* %sat_ua, align 4

  // Nothing when saturating to the same type and size
  sat_a = a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %sat_a, align 4

  // Nothing when assigning back
  a = sat_a;
  // CHECK:      [[SAT_ACCUM:%[0-9a-z]+]] = load i32, i32* %sat_a, align 4
  // CHECK-NEXT: store i32 [[SAT_ACCUM]], i32* %a, align 4

  // No overflow when casting from fract to signed accum
  sat_a = sat_f;
  // CHECK:      [[FRACT:%[0-9a-z]+]] = load i16, i16* %sat_f, align 2
  // CHECK-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i16 [[FRACT]] to i32
  // CHECK-NEXT: store i32 [[FRACT_EXT]], i32* %sat_a, align 4

  // Only get overflow checking if signed fract to unsigned accum
  sat_ua = sat_sf;
  // SIGNED:      [[FRACT:%[0-9a-z]+]] = load i8, i8* %sat_sf, align 1
  // SIGNED-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i8 [[FRACT]] to i32
  // SIGNED-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[FRACT_EXT]], 9
  // SIGNED-NEXT: [[IS_NEG:%[0-9a-z]+]] = icmp slt i32 [[ACCUM]], 0
  // SIGNED-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[IS_NEG]], i32 0, i32 [[ACCUM]]
  // SIGNED-NEXT: store i32 [[RESULT]], i32* %sat_ua, align 4
  // UNSIGNED:      [[FRACT:%[0-9a-z]+]] = load i8, i8* %sat_sf, align 1
  // UNSIGNED-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i8 [[FRACT]] to i32
  // UNSIGNED-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[FRACT_EXT]], 8
  // UNSIGNED-NEXT: [[IS_NEG:%[0-9a-z]+]] = icmp slt i32 [[ACCUM]], 0
  // UNSIGNED-NEXT: [[RESULT:%[0-9a-z]+]] = select i1 [[IS_NEG]], i32 0, i32 [[ACCUM]]
  // UNSIGNED-NEXT: store i32 [[RESULT]], i32* %sat_ua, align 4
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
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[FRACT:%[0-9a-z]+]] = ashr i32 [[ACCUM]], 8
  // CHECK-NEXT: [[FRACT_TRUNC:%[0-9a-z]+]] = trunc i32 [[FRACT]] to i8
  // CHECK-NEXT: store i8 [[FRACT_TRUNC]], i8* %sf, align 1

  // To higher scale
  a = sf;
  // CHECK:      [[FRACT:%[0-9a-z]+]] = load i8, i8* %sf, align 1
  // CHECK-NEXT: [[FRACT_EXT:%[0-9a-z]+]] = sext i8 [[FRACT]] to i32
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = shl i32 [[FRACT_EXT]], 8
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  // To same scale
  f = a;
  // CHECK:      [[ACCUM:%[0-9a-z]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[FRACT:%[0-9a-z]+]] = trunc i32 [[ACCUM]] to i16
  // CHECK-NEXT: store i16 [[FRACT]], i16* %f, align 2

  a = f;
  // CHECK:      [[FRACT:%[0-9a-z]+]] = load i16, i16* %f, align 2
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = sext i16 [[FRACT]] to i32
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %a, align 4

  // To unsigned
  ua = uf;
  // CHECK:      [[FRACT:%[0-9a-z]+]] = load i16, i16* %uf, align 2
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = zext i16 [[FRACT]] to i32
  // CHECK-NEXT: store i32 [[ACCUM]], i32* %ua, align 4

  uf = ua;
  // CHECK:      [[FRACT:%[0-9a-z]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[ACCUM:%[0-9a-z]+]] = trunc i32 [[FRACT]] to i16
  // CHECK-NEXT: store i16 [[ACCUM]], i16* %uf, align 2
}

void TestFixedPointToInt() {
  int i;
  short _Accum sa;
  unsigned short _Accum usa;

  // Will need to check for negative values
  i = sa;
  // CHECK:      [[FX:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[NEG:%[0-9]+]] = icmp slt i16 [[FX]], 0
  // CHECK-NEXT: [[ROUNDED:%[0-9]+]] = add i16 [[FX]], 127
  // CHECK-NEXT: [[VAL:%[0-9]+]] = select i1 [[NEG]], i16 [[ROUNDED]], i16 [[FX]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = ashr i16 [[VAL]], 7
  // CHECK-NEXT: [[RES2:%[a-z0-9]+]] = sext i16 [[RES]] to i32
  // CHECK-NEXT: store i32 [[RES2]], i32* %i, align 4

  // No check needed for unsigned fixed points. Can just right shift.
  i = usa;
  // SIGNED:      [[FX:%[0-9]+]] = load i16, i16* %usa, align 2
  // SIGNED-NEXT: [[INT:%[a-z0-9]+]] = lshr i16 [[FX]], 8
  // SIGNED-NEXT: [[RES:%[a-z0-9]+]] = zext i16 [[INT]] to i32
  // SIGNED-NEXT: store i32 [[RES]], i32* %i, align 4
  // UNSIGNED:      [[FX:%[0-9]+]] = load i16, i16* %usa, align 2
  // UNSIGNED-NEXT: [[INT:%[a-z0-9]+]] = lshr i16 [[FX]], 7
  // UNSIGNED-NEXT: [[RES:%[a-z0-9]+]] = zext i16 [[INT]] to i32
  // UNSIGNED-NEXT: store i32 [[RES]], i32* %i, align 4
}

void TestIntToFixedPoint() {
  short s;
  int i, i2;
  unsigned int ui;
  short _Accum sa;
  long _Accum la;
  unsigned short _Accum usa;
  _Sat short _Accum sat_sa;
  _Sat unsigned short _Accum sat_usa;

  sa = i;
  // CHECK:      [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = trunc i32 [[I]] to i16
  // CHECK-NEXT: [[FX:%[a-z0-9]+]] = shl i16 [[I_EXT]], 7
  // CHECK-NEXT: store i16 [[FX]], i16* %sa, align 2

  sa = ui;
  // CHECK:      [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = trunc i32 [[I]] to i16
  // CHECK-NEXT: [[FX:%[a-z0-9]+]] = shl i16 [[I_EXT]], 7
  // CHECK-NEXT: store i16 [[FX]], i16* %sa, align 2

  usa = i2;
  // SIGNED:      [[I:%[0-9]+]] = load i32, i32* %i2, align 4
  // SIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = trunc i32 [[I]] to i16
  // SIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i16 [[I_EXT]], 8
  // SIGNED-NEXT: store i16 [[FX]], i16* %usa, align 2
  // UNSIGNED:      [[I:%[0-9]+]] = load i32, i32* %i2, align 4
  // UNSIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = trunc i32 [[I]] to i16
  // UNSIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i16 [[I_EXT]], 7
  // UNSIGNED-NEXT: store i16 [[FX]], i16* %usa, align 2

  usa = ui;
  // SIGNED:      [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // SIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = trunc i32 [[I]] to i16
  // SIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i16 [[I_EXT]], 8
  // SIGNED-NEXT: store i16 [[FX]], i16* %usa, align 2
  // UNSIGNED:      [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // UNSIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = trunc i32 [[I]] to i16
  // UNSIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i16 [[I_EXT]], 7
  // UNSIGNED-NEXT: store i16 [[FX]], i16* %usa, align 2

  la = s;
  // CHECK:      [[I:%[0-9]+]] = load i16, i16* %s, align 2
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i16 [[I]] to i64
  // CHECK-NEXT: [[FX:%[a-z0-9]+]] = shl i64 [[I_EXT]], 31
  // CHECK-NEXT: store i64 [[FX]], i64* %la, align 8
}

void TestIntToSatFixedPoint() {
  int i, i2;
  unsigned int ui;
  _Sat short _Accum sat_sa;
  _Sat unsigned short _Accum sat_usa;

  sat_sa = i;
  // CHECK:      [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[FX:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // CHECK-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i39 [[FX]], 32767
  // CHECK-NEXT: [[SATMAX:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[FX]]
  // CHECK-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i39 [[SATMAX]], -32768
  // CHECK-NEXT: [[SATMIN:%[a-z0-9]+]] = select i1 [[USE_MIN]], i39 -32768, i39 [[SATMAX]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = trunc i39 [[SATMIN]] to i16
  // CHECK-NEXT: store i16 [[RES]], i16* %sat_sa, align 2

  sat_sa = ui;
  // CHECK:      [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // CHECK-NEXT: [[FX:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // CHECK-NEXT: [[USE_MAX:%[0-9]+]] = icmp ugt i39 [[FX]], 32767
  // CHECK-NEXT: [[SATMAX:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[FX]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = trunc i39 [[SATMAX]] to i16
  // CHECK-NEXT: store i16 [[RES]], i16* %sat_sa, align 2

  sat_usa = i2;
  // SIGNED:      [[I:%[0-9]+]] = load i32, i32* %i2, align 4
  // SIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i40
  // SIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i40 [[I_EXT]], 8
  // SIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i40 [[FX]], 65535
  // SIGNED-NEXT: [[SATMAX:%[a-z0-9]+]] = select i1 [[USE_MAX]], i40 65535, i40 [[FX]]
  // SIGNED-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i40 [[SATMAX]], 0
  // SIGNED-NEXT: [[SATMIN:%[a-z0-9]+]] = select i1 [[USE_MIN]], i40 0, i40 [[SATMAX]]
  // SIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i40 [[SATMIN]] to i16
  // SIGNED-NEXT: store i16 [[RES]], i16* %sat_usa, align 2
  // UNSIGNED:      [[I:%[0-9]+]] = load i32, i32* %i2, align 4
  // UNSIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // UNSIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // UNSIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i39 [[FX]], 32767
  // UNSIGNED-NEXT: [[SATMAX:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[FX]]
  // UNSIGNED-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i39 [[SATMAX]], 0
  // UNSIGNED-NEXT: [[SATMIN:%[a-z0-9]+]] = select i1 [[USE_MIN]], i39 0, i39 [[SATMAX]]
  // UNSIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i39 [[SATMIN]] to i16
  // UNSIGNED-NEXT: store i16 [[RES]], i16* %sat_usa, align 2

  sat_usa = ui;
  // SIGNED:      [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // SIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // SIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i40 [[I_EXT]], 8
  // SIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp ugt i40 [[FX]], 65535
  // SIGNED-NEXT: [[SATMAX:%[a-z0-9]+]] = select i1 [[USE_MAX]], i40 65535, i40 [[FX]]
  // SIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i40 [[SATMAX]] to i16
  // SIGNED-NEXT: store i16 [[RES]], i16* %sat_usa, align 2
  // UNSIGNED:      [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // UNSIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // UNSIGNED-NEXT: [[FX:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // UNSIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp ugt i39 [[FX]], 32767
  // UNSIGNED-NEXT: [[SATMAX:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[FX]]
  // UNSIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i39 [[SATMAX]] to i16
  // UNSIGNED-NEXT: store i16 [[RES]], i16* %sat_usa, align 2
}
