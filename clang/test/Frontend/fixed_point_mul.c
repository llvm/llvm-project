// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -fpadding-on-unsigned-fixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

// Multiplication between different fixed point types
short _Accum sa_const = 2.0hk * 2.0hk;  // CHECK-DAG: @sa_const  = {{.*}}global i16 512, align 2
_Accum a_const = 3.0hk * 2.0k;          // CHECK-DAG: @a_const   = {{.*}}global i32 196608, align 4
long _Accum la_const = 4.0hk * 2.0lk;   // CHECK-DAG: @la_const  = {{.*}}global i64 17179869184, align 8
short _Accum sa_const2 = 0.5hr * 2.0hk; // CHECK-DAG: @sa_const2  = {{.*}}global i16 128, align 2
short _Accum sa_const3 = 0.5r * 3.0hk;  // CHECK-DAG: @sa_const3  = {{.*}}global i16 192, align 2
short _Accum sa_const4 = 0.5lr * 4.0hk; // CHECK-DAG: @sa_const4  = {{.*}}global i16 256, align 2

// Unsigned multiplication
unsigned short _Accum usa_const = 1.0uhk * 2.0uhk;
// CHECK-SIGNED-DAG:   @usa_const = {{.*}}global i16 768, align 2
// CHECK-UNSIGNED-DAG: @usa_const = {{.*}}global i16 384, align 2

// Unsigned * signed
short _Accum sa_const5 = 20.0uhk * 3.0hk;
// CHECK-DAG: @sa_const5 = {{.*}}global i16 7680, align 2

// Multiplication with negative number
short _Accum sa_const6 = 0.5hr * (-2.0hk);
// CHECK-DAG: @sa_const6 = {{.*}}global i16 -128, align 2

// Int multiplication
unsigned short _Accum usa_const2 = 5 * 10.5uhk;
// CHECK-SIGNED-DAG:   @usa_const2 = {{.*}}global i16 640, align 2
// CHECK-UNSIGNED-DAG: @usa_const2 = {{.*}}global i16 320, align 2
short _Accum sa_const7 = 3 * (-0.5hk);   // CHECK-DAG: @sa_const7 = {{.*}}global i16 -192, align 2
short _Accum sa_const8 = 100 * (-2.0hk); // CHECK-DAG: @sa_const8 = {{.*}}global i16 -25600, align 2
long _Fract lf_const = -0.25lr * 3;      // CHECK-DAG: @lf_const  = {{.*}}global i32 -1610612736, align 4

// Saturated multiplication
_Sat short _Accum sat_sa_const = (_Sat short _Accum)128.0hk * 3.0hk;
// CHECK-DAG: @sat_sa_const = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const = (_Sat unsigned short _Accum)128.0uhk * 128.0uhk;
// CHECK-SIGNED-DAG:   @sat_usa_const = {{.*}}global i16 65535, align 2
// CHECK-UNSIGNED-DAG: @sat_usa_const = {{.*}}global i16 32767, align 2
_Sat short _Accum sat_sa_const2 = (_Sat short _Accum)128.0hk * -128;
// CHECK-DAG: @sat_sa_const2 = {{.*}}global i16 -32768, align 2
_Sat unsigned short _Accum sat_usa_const2 = (_Sat unsigned short _Accum)128.0uhk * 30;
// CHECK-SIGNED-DAG:   @sat_usa_const2 = {{.*}}global i16 65535, align 2
// CHECK-UNSIGNED-DAG: @sat_usa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const3 = (_Sat unsigned short _Accum)0.5uhk * (-2);
// CHECK-DAG:   @sat_usa_const3 = {{.*}}global i16 0, align 2

void SignedMultiplication() {
  // CHECK-LABEL: SignedMultiplication
  short _Accum sa;
  _Accum a, b, c, d;
  long _Accum la;
  unsigned short _Accum usa;
  unsigned _Accum ua;
  unsigned long _Accum ula;

  short _Fract sf;
  _Fract f;
  long _Fract lf;
  unsigned short _Fract usf;
  unsigned _Fract uf;
  unsigned long _Fract ulf;

  // Same type
  // CHECK:       [[TMP0:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:  [[TMP1:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:  [[TMP2:%.*]] = call i16 @llvm.smul.fix.i16(i16 [[TMP0]], i16 [[TMP1]], i32 7)
  // CHECK-NEXT:  store i16 [[TMP2]], i16* %sa, align 2
  sa = sa * sa;

  // To larger scale and larger width
  // CHECK:       [[TMP3:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:  [[TMP4:%.*]] = load i32, i32* %a, align 4
  // CHECK-NEXT:  [[RESIZE:%.*]] = sext i16 [[TMP3]] to i32
  // CHECK-NEXT:  [[UPSCALE:%.*]] = shl i32 [[RESIZE]], 8
  // CHECK-NEXT:  [[TMP5:%.*]] = call i32 @llvm.smul.fix.i32(i32 [[UPSCALE]], i32 [[TMP4]], i32 15)
  // CHECK-NEXT:  store i32 [[TMP5]], i32* %a, align 4
  a = sa * a;

  // To same scale and smaller width
  // CHECK:       [[TMP6:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:  [[TMP7:%.*]] = load i8, i8* %sf, align 1
  // CHECK-NEXT:  [[RESIZE1:%.*]] = sext i8 [[TMP7]] to i16
  // CHECK-NEXT:  [[TMP8:%.*]] = call i16 @llvm.smul.fix.i16(i16 [[TMP6]], i16 [[RESIZE1]], i32 7)
  // CHECK-NEXT:  store i16 [[TMP8]], i16* %sa, align 2
  sa = sa * sf;

  // To smaller scale and same width.
  // CHECK:       [[TMP9:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:  [[TMP10:%.*]] = load i16, i16* %f, align 2
  // CHECK-NEXT:  [[RESIZE2:%.*]] = sext i16 [[TMP9]] to i24
  // CHECK-NEXT:  [[UPSCALE3:%.*]] = shl i24 [[RESIZE2]], 8
  // CHECK-NEXT:  [[RESIZE4:%.*]] = sext i16 [[TMP10]] to i24
  // CHECK-NEXT:  [[TMP11:%.*]] = call i24 @llvm.smul.fix.i24(i24 [[UPSCALE3]], i24 [[RESIZE4]], i32 15)
  // CHECK-NEXT:  [[DOWNSCALE:%.*]] = ashr i24 [[TMP11]], 8
  // CHECK-NEXT:  [[RESIZE5:%.*]] = trunc i24 [[DOWNSCALE]] to i16
  // CHECK-NEXT:  store i16 [[RESIZE5]], i16* %sa, align 2
  sa = sa * f;

  // To smaller scale and smaller width
  // CHECK:       [[TMP12:%.*]] = load i32, i32* %a, align 4
  // CHECK-NEXT:  [[TMP13:%.*]] = load i8, i8* %sf, align 1
  // CHECK-NEXT:  [[RESIZE6:%.*]] = sext i8 [[TMP13]] to i32
  // CHECK-NEXT:  [[UPSCALE7:%.*]] = shl i32 [[RESIZE6]], 8
  // CHECK-NEXT:  [[TMP14:%.*]] = call i32 @llvm.smul.fix.i32(i32 [[TMP12]], i32 [[UPSCALE7]], i32 15)
  // CHECK-NEXT:  store i32 [[TMP14]], i32* %a, align 4
  a = a * sf;

  // To larger scale and same width
  // CHECK:       [[TMP15:%.*]] = load i32, i32* %a, align 4
  // CHECK-NEXT:  [[TMP16:%.*]] = load i32, i32* %lf, align 4
  // CHECK-NEXT:  [[RESIZE8:%.*]] = sext i32 [[TMP15]] to i48
  // CHECK-NEXT:  [[UPSCALE9:%.*]] = shl i48 [[RESIZE8]], 16
  // CHECK-NEXT:  [[RESIZE10:%.*]] = sext i32 [[TMP16]] to i48
  // CHECK-NEXT:  [[TMP17:%.*]] = call i48 @llvm.smul.fix.i48(i48 [[UPSCALE9]], i48 [[RESIZE10]], i32 31)
  // CHECK-NEXT:  [[DOWNSCALE11:%.*]] = ashr i48 [[TMP17]], 16
  // CHECK-NEXT:  [[RESIZE12:%.*]] = trunc i48 [[DOWNSCALE11]] to i32
  // CHECK-NEXT:  store i32 [[RESIZE12]], i32* %a, align 4
  a = a * lf;

  // With corresponding unsigned type
  // CHECK:        [[TMP18:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:   [[TMP19:%.*]] = load i16, i16* %usa, align 2
  // SIGNED-NEXT:  [[RESIZE13:%.*]] = sext i16 [[TMP18]] to i17
  // SIGNED-NEXT:  [[UPSCALE14:%.*]] = shl i17 [[RESIZE13]], 1
  // SIGNED-NEXT:  [[RESIZE15:%.*]] = zext i16 [[TMP19]] to i17
  // SIGNED-NEXT:  [[TMP20:%.*]] = call i17 @llvm.smul.fix.i17(i17 [[UPSCALE14]], i17 [[RESIZE15]], i32 8)
  // SIGNED-NEXT:  [[DOWNSCALE16:%.*]] = ashr i17 [[TMP20]], 1
  // SIGNED-NEXT:  [[RESIZE17:%.*]] = trunc i17 [[DOWNSCALE16]] to i16
  // SIGNED-NEXT:  store i16 [[RESIZE17]], i16* %sa, align 2
  // UNSIGNED-NEXT:[[TMP20:%.*]] = call i16 @llvm.smul.fix.i16(i16 [[TMP18]], i16 [[TMP19]], i32 7)
  // UNSIGNED-NEXT:store i16 [[TMP20]], i16* %sa, align 2
  sa = sa * usa;

  // With unsigned of larger scale
  // CHECK:        [[TMP21:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:   [[TMP22:%.*]] = load i32, i32* %ua, align 4
  // SIGNED-NEXT:  [[RESIZE18:%.*]] = sext i16 [[TMP21]] to i33
  // SIGNED-NEXT:  [[UPSCALE19:%.*]] = shl i33 [[RESIZE18]], 9
  // SIGNED-NEXT:  [[RESIZE20:%.*]] = zext i32 [[TMP22]] to i33
  // SIGNED-NEXT:  [[TMP23:%.*]] = call i33 @llvm.smul.fix.i33(i33 [[UPSCALE19]], i33 [[RESIZE20]], i32 16)
  // SIGNED-NEXT:  [[DOWNSCALE21:%.*]] = ashr i33 [[TMP23]], 1
  // SIGNED-NEXT:  [[RESIZE22:%.*]] = trunc i33 [[DOWNSCALE21]] to i32
  // SIGNED-NEXT:  store i32 [[RESIZE22]], i32* %a, align 4
  // UNSIGNED-NEXT:[[RESIZE13:%.*]] = sext i16 [[TMP21]] to i32
  // UNSIGNED-NEXT:[[UPSCALE14:%.*]] = shl i32 [[RESIZE13]], 8
  // UNSIGNED-NEXT:[[TMP23:%.*]] = call i32 @llvm.smul.fix.i32(i32 [[UPSCALE14]], i32 [[TMP22]], i32 15)
  // UNSIGNED-NEXT:store i32 [[TMP23]], i32* %a, align 4
  a = sa * ua;

  // With unsigned of smaller width
  // CHECK:        [[TMP24:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:   [[TMP25:%.*]] = load i8, i8* %usf, align 1
  // SIGNED-NEXT:  [[RESIZE23:%.*]] = sext i16 [[TMP24]] to i17
  // SIGNED-NEXT:  [[UPSCALE24:%.*]] = shl i17 [[RESIZE23]], 1
  // SIGNED-NEXT:  [[RESIZE25:%.*]] = zext i8 [[TMP25]] to i17
  // SIGNED-NEXT:  [[TMP26:%.*]] = call i17 @llvm.smul.fix.i17(i17 [[UPSCALE24]], i17 [[RESIZE25]], i32 8)
  // SIGNED-NEXT:  [[DOWNSCALE26:%.*]] = ashr i17 [[TMP26]], 1
  // SIGNED-NEXT:  [[RESIZE27:%.*]] = trunc i17 [[DOWNSCALE26]] to i16
  // SIGNED-NEXT:  store i16 [[RESIZE27]], i16* %sa, align 2
  // UNSIGNED-NEXT:[[RESIZE15:%.*]] = zext i8 [[TMP25]] to i16
  // UNSIGNED-NEXT:[[TMP26:%.*]] = call i16 @llvm.smul.fix.i16(i16 [[TMP24]], i16 [[RESIZE15]], i32 7)
  // UNSIGNED-NEXT:store i16 [[TMP26]], i16* %sa, align 2
  sa = sa * usf;

  // With unsigned of larger width and smaller scale
  // CHECK:        [[TMP27:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:   [[TMP28:%.*]] = load i32, i32* %ulf, align 4
  // SIGNED-NEXT:  [[RESIZE28:%.*]] = sext i16 [[TMP27]] to i41
  // SIGNED-NEXT:  [[UPSCALE29:%.*]] = shl i41 [[RESIZE28]], 25
  // SIGNED-NEXT:  [[RESIZE30:%.*]] = zext i32 [[TMP28]] to i41
  // SIGNED-NEXT:  [[TMP29:%.*]] = call i41 @llvm.smul.fix.i41(i41 [[UPSCALE29]], i41 [[RESIZE30]], i32 32)
  // SIGNED-NEXT:  [[DOWNSCALE31:%.*]] = ashr i41 [[TMP29]], 25
  // SIGNED-NEXT:  [[RESIZE32:%.*]] = trunc i41 [[DOWNSCALE31]] to i16
  // SIGNED-NEXT:  store i16 [[RESIZE32]], i16* %sa, align 2
  // UNSIGNED-NEXT:[[RESIZE16:%.*]] = sext i16 [[TMP27]] to i40
  // UNSIGNED-NEXT:[[UPSCALE17:%.*]] = shl i40 [[RESIZE16]], 24
  // UNSIGNED-NEXT:[[RESIZE18:%.*]] = zext i32 [[TMP28]] to i40
  // UNSIGNED-NEXT:[[TMP29:%.*]] = call i40 @llvm.smul.fix.i40(i40 [[UPSCALE17]], i40 [[RESIZE18]], i32 31)
  // UNSIGNED-NEXT:[[DOWNSCALE19:%.*]] = ashr i40 [[TMP29]], 24
  // UNSIGNED-NEXT:[[RESIZE20:%.*]] = trunc i40 [[DOWNSCALE19]] to i16
  // UNSIGNED-NEXT:store i16 [[RESIZE20]], i16* %sa, align 2
  sa = sa * ulf;

  // Chained multiplications of the same signed type should result in the same
  // CHECK:       [[TMP30:%.*]] = load i32, i32* %a, align 4
  // CHECK-NEXT:  [[TMP31:%.*]] = load i32, i32* %b, align 4
  // CHECK-NEXT:  [[TMP32:%.*]] = call i32 @llvm.smul.fix.i32(i32 [[TMP30]], i32 [[TMP31]], i32 15)
  // CHECK-NEXT:  [[TMP33:%.*]] = load i32, i32* %c, align 4
  // CHECK-NEXT:  [[TMP34:%.*]] = call i32 @llvm.smul.fix.i32(i32 [[TMP32]], i32 [[TMP33]], i32 15)
  // CHECK-NEXT:  [[TMP35:%.*]] = load i32, i32* %d, align 4
  // CHECK-NEXT:  [[TMP36:%.*]] = call i32 @llvm.smul.fix.i32(i32 [[TMP34]], i32 [[TMP35]], i32 15)
  // CHECK-NEXT:  store i32 [[TMP36]], i32* %a, align 4
  a = a * b * c * d;
}


void UnsignedMultiplication() {
  // CHECK-LABEL: UnsignedMultiplication
  unsigned short _Accum usa;
  unsigned _Accum ua;
  unsigned long _Accum ula;

  unsigned short _Fract usf;
  unsigned _Fract uf;
  unsigned long _Fract ulf;

  // CHECK:         [[TMP0:%.*]] = load i16, i16* %usa, align 2
  // CHECK-NEXT:    [[TMP1:%.*]] = load i16, i16* %usa, align 2
  // SIGNED-NEXT:   [[TMP2:%.*]] = call i16 @llvm.umul.fix.i16(i16 [[TMP0]], i16 [[TMP1]], i32 8)
  // UNSIGNED-NEXT: [[TMP2:%.*]] = call i16 @llvm.umul.fix.i16(i16 [[TMP0]], i16 [[TMP1]], i32 7)
  // CHECK-NEXT:    store i16 [[TMP2]], i16* %usa, align 2
  usa = usa * usa;

  // CHECK:         [[TMP3:%.*]] = load i16, i16* %usa, align 2
  // CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* %ua, align 4
  // CHECK-NEXT:    [[RESIZE:%.*]] = zext i16 [[TMP3]] to i32
  // CHECK-NEXT:    [[UPSCALE:%.*]] = shl i32 [[RESIZE]], 8
  // SIGNED-NEXT:   [[TMP5:%.*]] = call i32 @llvm.umul.fix.i32(i32 [[UPSCALE]], i32 [[TMP4]], i32 16)
  // UNSIGNED-NEXT: [[TMP5:%.*]] = call i32 @llvm.umul.fix.i32(i32 [[UPSCALE]], i32 [[TMP4]], i32 15)
  // CHECK-NEXT:    store i32 [[TMP5]], i32* %ua, align 4
  ua = usa * ua;

  // CHECK:         [[TMP6:%.*]] = load i16, i16* %usa, align 2
  // CHECK-NEXT:    [[TMP7:%.*]] = load i8, i8* %usf, align 1
  // CHECK-NEXT:    [[RESIZE1:%.*]] = zext i8 [[TMP7]] to i16
  // SIGNED-NEXT:   [[TMP8:%.*]] = call i16 @llvm.umul.fix.i16(i16 [[TMP6]], i16 [[RESIZE1]], i32 8)
  // UNSIGNED-NEXT: [[TMP8:%.*]] = call i16 @llvm.umul.fix.i16(i16 [[TMP6]], i16 [[RESIZE1]], i32 7)
  // CHECK-NEXT:    store i16 [[TMP8]], i16* %usa, align 2
  usa = usa * usf;

  // CHECK:         [[TMP9:%.*]] = load i16, i16* %usa, align 2
  // CHECK-NEXT:    [[TMP10:%.*]] = load i16, i16* %uf, align 2
  // CHECK-NEXT:    [[RESIZE2:%.*]] = zext i16 [[TMP9]] to i24
  // CHECK-NEXT:    [[UPSCALE3:%.*]] = shl i24 [[RESIZE2]], 8
  // CHECK-NEXT:    [[RESIZE4:%.*]] = zext i16 [[TMP10]] to i24
  // SIGNED-NEXT:   [[TMP11:%.*]] = call i24 @llvm.umul.fix.i24(i24 [[UPSCALE3]], i24 [[RESIZE4]], i32 16)
  // UNSIGNED-NEXT: [[TMP11:%.*]] = call i24 @llvm.umul.fix.i24(i24 [[UPSCALE3]], i24 [[RESIZE4]], i32 15)
  // CHECK-NEXT:    [[DOWNSCALE:%.*]] = lshr i24 [[TMP11]], 8
  // CHECK-NEXT:    [[RESIZE5:%.*]] = trunc i24 [[DOWNSCALE]] to i16
  // CHECK-NEXT:    store i16 [[RESIZE5]], i16* %usa, align 2
  usa = usa * uf;
}

void IntMultiplication() {
  // CHECK-LABEL: IntMultiplication
  short _Accum sa;
  _Accum a;
  unsigned short _Accum usa;
  int i;
  unsigned int ui;
  long _Fract lf;
  _Bool b;

  // CHECK:         [[TMP0:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* %i, align 4
  // CHECK-NEXT:    [[RESIZE:%.*]] = sext i16 [[TMP0]] to i39
  // CHECK-NEXT:    [[RESIZE1:%.*]] = sext i32 [[TMP1]] to i39
  // CHECK-NEXT:    [[UPSCALE:%.*]] = shl i39 [[RESIZE1]], 7
  // CHECK-NEXT:    [[TMP2:%.*]] = call i39 @llvm.smul.fix.i39(i39 [[RESIZE]], i39 [[UPSCALE]], i32 7)
  // CHECK-NEXT:    [[RESIZE2:%.*]] = trunc i39 [[TMP2]] to i16
  // CHECK-NEXT:    store i16 [[RESIZE2]], i16* %sa, align 2
  sa = sa * i;

  // CHECK:         [[TMP3:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* %ui, align 4
  // CHECK-NEXT:    [[RESIZE3:%.*]] = sext i16 [[TMP3]] to i40
  // CHECK-NEXT:    [[RESIZE4:%.*]] = zext i32 [[TMP4]] to i40
  // CHECK-NEXT:    [[UPSCALE5:%.*]] = shl i40 [[RESIZE4]], 7
  // CHECK-NEXT:    [[TMP5:%.*]] = call i40 @llvm.smul.fix.i40(i40 [[RESIZE3]], i40 [[UPSCALE5]], i32 7)
  // CHECK-NEXT:    [[RESIZE6:%.*]] = trunc i40 [[TMP5]] to i16
  // CHECK-NEXT:    store i16 [[RESIZE6]], i16* %sa, align 2
  sa = sa * ui;

  // CHECK:         [[TMP6:%.*]] = load i16, i16* %usa, align 2
  // CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* %i, align 4
  // SIGNED-NEXT:   [[RESIZE7:%.*]] = zext i16 [[TMP6]] to i40
  // SIGNED-NEXT:   [[RESIZE8:%.*]] = sext i32 [[TMP7]] to i40
  // SIGNED-NEXT:   [[UPSCALE9:%.*]] = shl i40 [[RESIZE8]], 8
  // SIGNED-NEXT:   [[TMP8:%.*]] = call i40 @llvm.umul.fix.i40(i40 [[RESIZE7]], i40 [[UPSCALE9]], i32 8)
  // SIGNED-NEXT:   [[RESIZE10:%.*]] = trunc i40 [[TMP8]] to i16
  // UNSIGNED-NEXT: [[RESIZE7:%.*]] = zext i16 [[TMP6]] to i39
  // UNSIGNED-NEXT: [[RESIZE8:%.*]] = sext i32 [[TMP7]] to i39
  // UNSIGNED-NEXT: [[UPSCALE9:%.*]] = shl i39 [[RESIZE8]], 7
  // UNSIGNED-NEXT: [[TMP8:%.*]] = call i39 @llvm.umul.fix.i39(i39 [[RESIZE7]], i39 [[UPSCALE9]], i32 7)
  // UNSIGNED-NEXT: [[RESIZE10:%.*]] = trunc i39 [[TMP8]] to i16
  // CHECK-NEXT:    store i16 [[RESIZE10]], i16* %usa, align 2
  usa = usa * i;

  // CHECK:         [[TMP9:%.*]] = load i16, i16* %usa, align 2
  // CHECK-NEXT:    [[TMP10:%.*]] = load i32, i32* %ui, align 4
  // SIGNED-NEXT:   [[RESIZE11:%.*]] = zext i16 [[TMP9]] to i40
  // SIGNED-NEXT:   [[RESIZE12:%.*]] = zext i32 [[TMP10]] to i40
  // SIGNED-NEXT:   [[UPSCALE13:%.*]] = shl i40 [[RESIZE12]], 8
  // SIGNED-NEXT:   [[TMP11:%.*]] = call i40 @llvm.umul.fix.i40(i40 [[RESIZE11]], i40 [[UPSCALE13]], i32 8)
  // SIGNED-NEXT:   [[RESIZE14:%.*]] = trunc i40 [[TMP11]] to i16
  // UNSIGNED-NEXT: [[RESIZE11:%.*]] = zext i16 [[TMP9]] to i39
  // UNSIGNED-NEXT: [[RESIZE12:%.*]] = zext i32 [[TMP10]] to i39
  // UNSIGNED-NEXT: [[UPSCALE13:%.*]] = shl i39 [[RESIZE12]], 7
  // UNSIGNED-NEXT: [[TMP11:%.*]] = call i39 @llvm.umul.fix.i39(i39 [[RESIZE11]], i39 [[UPSCALE13]], i32 7)
  // UNSIGNED-NEXT: [[RESIZE14:%.*]] = trunc i39 [[TMP11]] to i16
  // CHECK-NEXT:    store i16 [[RESIZE14]], i16* %usa, align 2
  usa = usa * ui;

  // CHECK:         [[TMP12:%.*]] = load i32, i32* %lf, align 4
  // CHECK-NEXT:    [[TMP13:%.*]] = load i32, i32* %ui, align 4
  // CHECK-NEXT:    [[RESIZE15:%.*]] = sext i32 [[TMP12]] to i64
  // CHECK-NEXT:    [[RESIZE16:%.*]] = zext i32 [[TMP13]] to i64
  // CHECK-NEXT:    [[UPSCALE17:%.*]] = shl i64 [[RESIZE16]], 31
  // CHECK-NEXT:    [[TMP14:%.*]] = call i64 @llvm.smul.fix.i64(i64 [[RESIZE15]], i64 [[UPSCALE17]], i32 31)
  // CHECK-NEXT:    [[RESIZE18:%.*]] = trunc i64 [[TMP14]] to i32
  // CHECK-NEXT:    store i32 [[RESIZE18]], i32* %lf, align 4
  lf = lf * ui;

  // CHECK:         [[TMP15:%.*]] = load i32, i32* %a, align 4
  // CHECK-NEXT:    [[TMP16:%.*]] = load i8, i8* %b, align 1
  // CHECK-NEXT:    [[TOBOOL:%.*]] = trunc i8 [[TMP16]] to i1
  // CHECK-NEXT:    [[CONV:%.*]] = zext i1 [[TOBOOL]] to i32
  // CHECK-NEXT:    [[RESIZE19:%.*]] = sext i32 [[TMP15]] to i47
  // CHECK-NEXT:    [[RESIZE20:%.*]] = sext i32 [[CONV]] to i47
  // CHECK-NEXT:    [[UPSCALE21:%.*]] = shl i47 [[RESIZE20]], 15
  // CHECK-NEXT:    [[TMP17:%.*]] = call i47 @llvm.smul.fix.i47(i47 [[RESIZE19]], i47 [[UPSCALE21]], i32 15)
  // CHECK-NEXT:    [[RESIZE22:%.*]] = trunc i47 [[TMP17]] to i32
  // CHECK-NEXT:    store i32 [[RESIZE22]], i32* %a, align 4
  a = a * b;

  // CHECK:         [[TMP18:%.*]] = load i32, i32* %i, align 4
  // CHECK-NEXT:    [[TMP19:%.*]] = load i32, i32* %a, align 4
  // CHECK-NEXT:    [[RESIZE23:%.*]] = sext i32 [[TMP18]] to i47
  // CHECK-NEXT:    [[UPSCALE24:%.*]] = shl i47 [[RESIZE23]], 15
  // CHECK-NEXT:    [[RESIZE25:%.*]] = sext i32 [[TMP19]] to i47
  // CHECK-NEXT:    [[TMP20:%.*]] = call i47 @llvm.smul.fix.i47(i47 [[UPSCALE24]], i47 [[RESIZE25]], i32 15)
  // CHECK-NEXT:    [[RESIZE26:%.*]] = trunc i47 [[TMP20]] to i32
  // CHECK-NEXT:    store i32 [[RESIZE26]], i32* %a, align 4
  a = i * a;

  // CHECK:         [[TMP21:%.*]] = load i32, i32* %ui, align 4
  // CHECK-NEXT:    [[TMP22:%.*]] = load i16, i16* %usa, align 2
  // SIGNED-NEXT:   [[RESIZE27:%.*]] = zext i32 [[TMP21]] to i40
  // SIGNED-NEXT:   [[UPSCALE28:%.*]] = shl i40 [[RESIZE27]], 8
  // SIGNED-NEXT:   [[RESIZE29:%.*]] = zext i16 [[TMP22]] to i40
  // SIGNED-NEXT:   [[TMP23:%.*]] = call i40 @llvm.umul.fix.i40(i40 [[UPSCALE28]], i40 [[RESIZE29]], i32 8)
  // SIGNED-NEXT:   [[RESIZE30:%.*]] = trunc i40 [[TMP23]] to i16
  // UNSIGNED-NEXT: [[RESIZE27:%.*]] = zext i32 [[TMP21]] to i39
  // UNSIGNED-NEXT: [[UPSCALE28:%.*]] = shl i39 [[RESIZE27]], 7
  // UNSIGNED-NEXT: [[RESIZE29:%.*]] = zext i16 [[TMP22]] to i39
  // UNSIGNED-NEXT: [[TMP23:%.*]] = call i39 @llvm.umul.fix.i39(i39 [[UPSCALE28]], i39 [[RESIZE29]], i32 7)
  // UNSIGNED-NEXT: [[RESIZE30:%.*]] = trunc i39 [[TMP23]] to i16
  // CHECK-NEXT:    store i16 [[RESIZE30]], i16* %usa, align 2
  usa = ui * usa;

  // CHECK:         [[TMP27:%.*]] = load i32, i32* %ui, align 4
  // CHECK-NEXT:    [[TMP28:%.*]] = load i16, i16* %sa, align 2
  // CHECK-NEXT:    [[RESIZE33:%.*]] = zext i32 [[TMP27]] to i40
  // CHECK-NEXT:    [[UPSCALE34:%.*]] = shl i40 [[RESIZE33]], 7
  // CHECK-NEXT:    [[RESIZE35:%.*]] = sext i16 [[TMP28]] to i40
  // CHECK-NEXT:    [[TMP29:%.*]] = call i40 @llvm.smul.fix.i40(i40 [[UPSCALE34]], i40 [[RESIZE35]], i32 7)
  // CHECK-NEXT:    [[RESIZE36:%.*]] = trunc i40 [[TMP29]] to i16
  // CHECK-NEXT:    store i16 [[RESIZE36]], i16* %sa, align 2
  sa = ui * sa;
}

void SaturatedMultiplication() {
  // CHECK-LABEL: SaturatedMultiplication
  short _Accum sa;
  _Accum a;
  long _Accum la;
  unsigned short _Accum usa;
  unsigned _Accum ua;
  unsigned long _Accum ula;

  _Sat short _Accum sa_sat;
  _Sat _Accum a_sat;
  _Sat long _Accum la_sat;
  _Sat unsigned short _Accum usa_sat;
  _Sat unsigned _Accum ua_sat;
  _Sat unsigned long _Accum ula_sat;
  _Sat unsigned _Fract uf_sat;

  int i;
  unsigned int ui;

  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[SA_SAT:%[0-9]+]] = load i16, i16* %sa_sat, align 2
  // CHECK-NEXT: [[SUM:%[0-9]+]] = call i16 @llvm.smul.fix.sat.i16(i16 [[SA]], i16 [[SA_SAT]], i32 7)
  // CHECK-NEXT: store i16 [[SUM]], i16* %sa_sat, align 2
  sa_sat = sa * sa_sat;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[USA_SAT:%[0-9]+]] = load i16, i16* %usa_sat, align 2
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i16 @llvm.umul.fix.sat.i16(i16 [[USA]], i16 [[USA_SAT]], i32 8)
  // SIGNED-NEXT: store i16 [[SUM]], i16* %usa_sat, align 2
  // UNSIGNED-NEXT: [[USA_TRUNC:%[a-z0-9]+]] = trunc i16 [[USA]] to i15
  // UNSIGNED-NEXT: [[USA_SAT_TRUNC:%[a-z0-9]+]] = trunc i16 [[USA_SAT]] to i15
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i15 @llvm.umul.fix.sat.i15(i15 [[USA_TRUNC]], i15 [[USA_SAT_TRUNC]], i32 7)
  // UNSIGNED-NEXT: [[SUM_EXT:%[a-z0-9]+]] = zext i15 [[SUM]] to i16
  // UNSIGNED-NEXT: store i16 [[SUM_EXT]], i16* %usa_sat, align 2
  usa_sat = usa * usa_sat;

  // CHECK:      [[UA:%[0-9]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[USA:%[0-9]+]] = load i16, i16* %usa_sat, align 2
  // SIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i32
  // SIGNED-NEXT: [[USA:%[a-z0-9]+]] = shl i32 [[USA_EXT]], 8
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i32 @llvm.umul.fix.sat.i32(i32 [[UA]], i32 [[USA]], i32 16)
  // SIGNED-NEXT: store i32 [[SUM]], i32* %ua_sat, align 4
  // UNSIGNED-NEXT: [[UA_TRUNC:%[a-z0-9]+]] = trunc i32 [[UA]] to i31
  // UNSIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i31
  // UNSIGNED-NEXT: [[USA:%[a-z0-9]+]] = shl i31 [[USA_EXT]], 8
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i31 @llvm.umul.fix.sat.i31(i31 [[UA_TRUNC]], i31 [[USA]], i32 15)
  // UNSIGNED-NEXT: [[SUM_EXT:%[a-z0-9]+]] = zext i31 [[SUM]] to i32
  // UNSIGNED-NEXT: store i32 [[SUM_EXT]], i32* %ua_sat, align 4
  ua_sat = ua * usa_sat;

  // CHECK:      [[SA_SAT:%[0-9]+]] = load i16, i16* %sa_sat, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[SA_SAT_EXT:%[a-z0-9]+]] = sext i16 [[SA_SAT]] to i39
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // CHECK-NEXT: [[SUM:%[0-9]+]] = call i39 @llvm.smul.fix.sat.i39(i39 [[SA_SAT_EXT]], i39 [[I]], i32 7)
  // CHECK-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i39 [[SUM]], 32767
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[SUM]]
  // CHECK-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i39 [[RES]], -32768
  // CHECK-NEXT: [[RES2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i39 -32768, i39 [[RES]]
  // CHECK-NEXT: [[RES3:%[a-z0-9]+]] = trunc i39 [[RES2]] to i16
  // CHECK-NEXT: store i16 [[RES3]], i16* %sa_sat, align 2
  sa_sat = sa_sat * i;

  // CHECK:      [[SA_SAT:%[0-9]+]] = load i16, i16* %sa_sat, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // CHECK-NEXT: [[SA_SAT_EXT:%[a-z0-9]+]] = sext i16 [[SA_SAT]] to i40
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = shl i40 [[I_EXT]], 7
  // CHECK-NEXT: [[SUM:%[0-9]+]] = call i40 @llvm.smul.fix.sat.i40(i40 [[SA_SAT_EXT]], i40 [[I]], i32 7)
  // CHECK-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i40 [[SUM]], 32767
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = select i1 [[USE_MAX]], i40 32767, i40 [[SUM]]
  // CHECK-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i40 [[RES]], -32768
  // CHECK-NEXT: [[RES2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i40 -32768, i40 [[RES]]
  // CHECK-NEXT: [[RES3:%[a-z0-9]+]] = trunc i40 [[RES2]] to i16
  // CHECK-NEXT: store i16 [[RES3]], i16* %sa_sat, align 2
  sa_sat = sa_sat * ui;

  // CHECK:      [[UF_SAT:%[0-9]+]] = load i16, i16* %uf_sat, align 2
  // CHECK-NEXT: [[UF_SAT2:%[0-9]+]] = load i16, i16* %uf_sat, align 2
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i16 @llvm.umul.fix.sat.i16(i16 [[UF_SAT]], i16 [[UF_SAT2]], i32 16)
  // SIGNED-NEXT: store i16 [[SUM]], i16* %uf_sat, align 2
  // UNSIGNED-NEXT: [[UF_SAT_TRUNC:%[a-z0-9]+]] = trunc i16 [[UF_SAT]] to i15
  // UNSIGNED-NEXT: [[UF_SAT_TRUNC2:%[a-z0-9]+]] = trunc i16 [[UF_SAT2]] to i15
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i15 @llvm.umul.fix.sat.i15(i15 [[UF_SAT_TRUNC]], i15 [[UF_SAT_TRUNC2]], i32 15)
  // UNSIGNED-NEXT: [[SUM_EXT:%[a-z0-9]+]] = zext i15 [[SUM]] to i16
  // UNSIGNED-NEXT: store i16 [[SUM_EXT]], i16* %uf_sat, align 2
  uf_sat = uf_sat * uf_sat;

  // CHECK:      [[USA_SAT:%[0-9]+]] = load i16, i16* %usa_sat, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // SIGNED-NEXT: [[USA_SAT_RESIZE:%[a-z0-9]+]] = zext i16 [[USA_SAT]] to i40
  // SIGNED-NEXT: [[I_RESIZE:%[a-z0-9]+]] = sext i32 [[I]] to i40
  // SIGNED-NEXT: [[I_UPSCALE:%[a-z0-9]+]] = shl i40 [[I_RESIZE]], 8
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i40 @llvm.umul.fix.sat.i40(i40 [[USA_SAT_RESIZE]], i40 [[I_UPSCALE]], i32 8)
  // SIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i40 [[SUM]], 65535
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = select i1 [[USE_MAX]], i40 65535, i40 [[SUM]]
  // SIGNED-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i40 [[RESULT]], 0
  // SIGNED-NEXT: [[RESULT2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i40 0, i40 [[RESULT]]
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = trunc i40 [[RESULT2]] to i16
  // UNSIGNED-NEXT: [[USA_SAT_RESIZE:%[a-z0-9]+]] = zext i16 [[USA_SAT]] to i39
  // UNSIGNED-NEXT: [[I_RESIZE:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // UNSIGNED-NEXT: [[I_UPSCALE:%[a-z0-9]+]] = shl i39 [[I_RESIZE]], 7
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i39 @llvm.umul.fix.sat.i39(i39 [[USA_SAT_RESIZE]], i39 [[I_UPSCALE]], i32 7)
  // UNSIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i39 [[SUM]], 32767
  // UNSIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[SUM]]
  // UNSIGNED-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i39 [[RESULT]], 0
  // UNSIGNED-NEXT: [[RESULT2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i39 0, i39 [[RESULT]]
  // UNSIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = trunc i39 [[RESULT2]] to i16
  // CHECK-NEXT: store i16 [[RESULT]], i16* %usa_sat, align 2
  usa_sat = usa_sat * i;
}
