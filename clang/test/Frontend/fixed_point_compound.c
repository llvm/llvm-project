// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -fpadding-on-unsigned-fixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

short _Fract shf;
_Accum a;
unsigned _Fract uf;
unsigned long _Accum ula;

_Sat short _Fract sshf;
_Sat _Accum sa;
_Sat unsigned _Fract suf;
_Sat unsigned long _Accum sula;

int i;
unsigned int u;
signed char c;


// CHECK-LABEL: @Addition(
void Addition() {
// CHECK:         [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, i8* @shf, align 1
// CHECK-NEXT:    [[RESIZE:%.*]] = sext i8 [[TMP1]] to i32
// CHECK-NEXT:    [[UPSCALE:%.*]] = shl i32 [[RESIZE]], 8
// CHECK-NEXT:    [[TMP2:%.*]] = add i32 [[UPSCALE]], [[TMP0]]
// CHECK-NEXT:    [[DOWNSCALE:%.*]] = ashr i32 [[TMP2]], 8
// CHECK-NEXT:    [[RESIZE1:%.*]] = trunc i32 [[DOWNSCALE]] to i8
// CHECK-NEXT:    store i8 [[RESIZE1]], i8* @shf, align 1
  shf += a;

// CHECK:         [[TMP3:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* @a, align 4
// SIGNED-NEXT:   [[RESIZE2:%.*]] = sext i32 [[TMP4]] to i33
// SIGNED-NEXT:   [[UPSCALE3:%.*]] = shl i33 [[RESIZE2]], 1
// SIGNED-NEXT:   [[RESIZE4:%.*]] = zext i16 [[TMP3]] to i33
// SIGNED-NEXT:   [[TMP5:%.*]] = add i33 [[UPSCALE3]], [[RESIZE4]]
// SIGNED-NEXT:   [[DOWNSCALE5:%.*]] = ashr i33 [[TMP5]], 1
// SIGNED-NEXT:   [[RESIZE6:%.*]] = trunc i33 [[DOWNSCALE5]] to i32
// SIGNED-NEXT:   store i32 [[RESIZE6]], i32* @a, align 4
// UNSIGNED-NEXT: [[RESIZE2:%.*]] = zext i16 [[TMP3]] to i32
// UNSIGNED-NEXT: [[TMP5:%.*]] = add i32 [[TMP4]], [[RESIZE2]]
// UNSIGNED-NEXT: store i32 [[TMP5]], i32* @a, align 4
  a += uf;

// CHECK:         [[TMP6:%.*]] = load i64, i64* @ula, align 8
// CHECK-NEXT:    [[TMP7:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    [[RESIZE7:%.*]] = zext i16 [[TMP7]] to i64
// CHECK-NEXT:    [[UPSCALE8:%.*]] = shl i64 [[RESIZE7]], 16
// CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[UPSCALE8]], [[TMP6]]
// CHECK-NEXT:    [[DOWNSCALE9:%.*]] = lshr i64 [[TMP8]], 16
// CHECK-NEXT:    [[RESIZE10:%.*]] = trunc i64 [[DOWNSCALE9]] to i16
// CHECK-NEXT:    store i16 [[RESIZE10]], i16* @uf, align 2
  uf += ula;

// CHECK:         [[TMP9:%.*]] = load i8, i8* @shf, align 1
// CHECK-NEXT:    [[TMP10:%.*]] = load i64, i64* @ula, align 8
// SIGNED-NEXT:   [[RESIZE11:%.*]] = zext i64 [[TMP10]] to i65
// SIGNED-NEXT:   [[RESIZE12:%.*]] = sext i8 [[TMP9]] to i65
// SIGNED-NEXT:   [[UPSCALE13:%.*]] = shl i65 [[RESIZE12]], 25
// SIGNED-NEXT:   [[TMP11:%.*]] = add i65 [[RESIZE11]], [[UPSCALE13]]
// SIGNED-NEXT:   [[DOWNSCALE14:%.*]] = ashr i65 [[TMP11]], 1
// SIGNED-NEXT:   [[RESIZE15:%.*]] = trunc i65 [[DOWNSCALE14]] to i64
// SIGNED-NEXT:   [[UPSCALE16:%.*]] = shl i64 [[RESIZE15]], 1
// SIGNED-NEXT:   store i64 [[UPSCALE16]], i64* @ula, align 8
// UNSIGNED-NEXT: [[RESIZE7:%.*]] = sext i8 [[TMP9]] to i64
// UNSIGNED-NEXT: [[UPSCALE8:%.*]] = shl i64 [[RESIZE7]], 24
// UNSIGNED-NEXT: [[TMP11:%.*]] = add i64 [[TMP10]], [[UPSCALE8]]
// UNSIGNED-NEXT: store i64 [[TMP11]], i64* @ula, align 8
  ula += shf;

// CHECK:         [[TMP12:%.*]] = load i8, i8* @shf, align 1
// CHECK-NEXT:    [[TMP13:%.*]] = load i16, i16* @uf, align 2
// SIGNED-NEXT:   [[RESIZE17:%.*]] = zext i16 [[TMP13]] to i17
// SIGNED-NEXT:   [[RESIZE18:%.*]] = sext i8 [[TMP12]] to i17
// SIGNED-NEXT:   [[UPSCALE19:%.*]] = shl i17 [[RESIZE18]], 9
// SIGNED-NEXT:   [[TMP14:%.*]] = add i17 [[RESIZE17]], [[UPSCALE19]]
// SIGNED-NEXT:   [[DOWNSCALE20:%.*]] = ashr i17 [[TMP14]], 1
// SIGNED-NEXT:   [[RESIZE21:%.*]] = trunc i17 [[DOWNSCALE20]] to i16
// SIGNED-NEXT:   [[UPSCALE22:%.*]] = shl i16 [[RESIZE21]], 1
// SIGNED-NEXT:   store i16 [[UPSCALE22]], i16* @uf, align 2
// UNSIGNED-NEXT: [[RESIZE9:%.*]] = sext i8 [[TMP12]] to i16
// UNSIGNED-NEXT: [[UPSCALE10:%.*]] = shl i16 [[RESIZE9]], 8
// UNSIGNED-NEXT: [[TMP14:%.*]] = add i16 [[TMP13]], [[UPSCALE10]]
// UNSIGNED-NEXT: store i16 [[TMP14]], i16* @uf, align 2
  uf += shf;

// CHECK:         [[TMP15:%.*]] = load i8, i8* @shf, align 1
// CHECK-NEXT:    [[TMP16:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[RESIZE23:%.*]] = sext i8 [[TMP15]] to i32
// CHECK-NEXT:    [[UPSCALE24:%.*]] = shl i32 [[RESIZE23]], 8
// CHECK-NEXT:    [[TMP17:%.*]] = add i32 [[TMP16]], [[UPSCALE24]]
// CHECK-NEXT:    store i32 [[TMP17]], i32* @a, align 4
  a += shf;

// CHECK:         [[TMP18:%.*]] = load i32, i32* @i, align 4
// CHECK-NEXT:    [[TMP19:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[RESIZE25:%.*]] = sext i32 [[TMP19]] to i47
// CHECK-NEXT:    [[RESIZE26:%.*]] = sext i32 [[TMP18]] to i47
// CHECK-NEXT:    [[UPSCALE27:%.*]] = shl i47 [[RESIZE26]], 15
// CHECK-NEXT:    [[TMP20:%.*]] = add i47 [[RESIZE25]], [[UPSCALE27]]
// CHECK-NEXT:    [[RESIZE28:%.*]] = trunc i47 [[TMP20]] to i32
// CHECK-NEXT:    store i32 [[RESIZE28]], i32* @a, align 4
  a += i;

// CHECK:         [[TMP21:%.*]] = load i32, i32* @u, align 4
// CHECK-NEXT:    [[TMP22:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[RESIZE29:%.*]] = sext i32 [[TMP22]] to i48
// CHECK-NEXT:    [[RESIZE30:%.*]] = zext i32 [[TMP21]] to i48
// CHECK-NEXT:    [[UPSCALE31:%.*]] = shl i48 [[RESIZE30]], 15
// CHECK-NEXT:    [[TMP23:%.*]] = add i48 [[RESIZE29]], [[UPSCALE31]]
// CHECK-NEXT:    [[RESIZE32:%.*]] = trunc i48 [[TMP23]] to i32
// CHECK-NEXT:    store i32 [[RESIZE32]], i32* @a, align 4
  a += u;

// CHECK:         [[TMP24:%.*]] = load i32, i32* @i, align 4
// CHECK-NEXT:    [[TMP25:%.*]] = load i64, i64* @ula, align 8
// SIGNED-NEXT:   [[RESIZE33:%.*]] = zext i64 [[TMP25]] to i65
// SIGNED-NEXT:   [[RESIZE34:%.*]] = sext i32 [[TMP24]] to i65
// SIGNED-NEXT:   [[UPSCALE35:%.*]] = shl i65 [[RESIZE34]], 32
// SIGNED-NEXT:   [[TMP26:%.*]] = add i65 [[RESIZE33]], [[UPSCALE35]]
// SIGNED-NEXT:   [[RESIZE36:%.*]] = trunc i65 [[TMP26]] to i64
// SIGNED-NEXT:   store i64 [[RESIZE36]], i64* @ula, align 8
// UNSIGNED-NEXT: [[RESIZE21:%.*]] = sext i32 [[TMP24]] to i64
// UNSIGNED-NEXT: [[UPSCALE22:%.*]] = shl i64 [[RESIZE21]], 31
// UNSIGNED-NEXT: [[TMP26:%.*]] = add i64 [[TMP25]], [[UPSCALE22]]
// UNSIGNED-NEXT: store i64 [[TMP26]], i64* @ula, align 8
  ula += i;

// CHECK:         [[TMP27:%.*]] = load i64, i64* @ula, align 8
// CHECK-NEXT:    [[TMP28:%.*]] = load i32, i32* @i, align 4
// SIGNED-NEXT:   [[RESIZE37:%.*]] = sext i32 [[TMP28]] to i65
// SIGNED-NEXT:   [[UPSCALE38:%.*]] = shl i65 [[RESIZE37]], 32
// SIGNED-NEXT:   [[RESIZE39:%.*]] = zext i64 [[TMP27]] to i65
// SIGNED-NEXT:   [[TMP29:%.*]] = add i65 [[UPSCALE38]], [[RESIZE39]]
// SIGNED-NEXT:   [[RESIZE40:%.*]] = trunc i65 [[TMP29]] to i64
// SIGNED-NEXT:   [[DOWNSCALE41:%.*]] = lshr i64 [[RESIZE40]], 32
// SIGNED-NEXT:   [[RESIZE42:%.*]] = trunc i64 [[DOWNSCALE41]] to i32
// SIGNED-NEXT:   store i32 [[RESIZE42]], i32* @i, align 4
// UNSIGNED-NEXT: [[RESIZE23:%.*]] = sext i32 [[TMP28]] to i64
// UNSIGNED-NEXT: [[UPSCALE24:%.*]] = shl i64 [[RESIZE23]], 31
// UNSIGNED-NEXT: [[TMP29:%.*]] = add i64 [[UPSCALE24]], [[TMP27]]
// UNSIGNED-NEXT: [[DOWNSCALE25:%.*]] = lshr i64 [[TMP29]], 31
// UNSIGNED-NEXT: [[RESIZE26:%.*]] = trunc i64 [[DOWNSCALE25]] to i32
// UNSIGNED-NEXT: store i32 [[RESIZE26]], i32* @i, align 4
  i += ula;

// CHECK:         [[TMP30:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP31:%.*]] = load i8, i8* @c, align 1
// CHECK-NEXT:    [[CONV:%.*]] = sext i8 [[TMP31]] to i32
// CHECK-NEXT:    [[RESIZE43:%.*]] = sext i32 [[CONV]] to i47
// CHECK-NEXT:    [[UPSCALE44:%.*]] = shl i47 [[RESIZE43]], 15
// CHECK-NEXT:    [[RESIZE45:%.*]] = sext i32 [[TMP30]] to i47
// CHECK-NEXT:    [[TMP32:%.*]] = add i47 [[UPSCALE44]], [[RESIZE45]]
// CHECK-NEXT:    [[RESIZE46:%.*]] = trunc i47 [[TMP32]] to i32
// CHECK-NEXT:    [[TMP33:%.*]] = icmp slt i32 [[RESIZE46]], 0
// CHECK-NEXT:    [[TMP34:%.*]] = add i32 [[RESIZE46]], 32767
// CHECK-NEXT:    [[TMP35:%.*]] = select i1 [[TMP33]], i32 [[TMP34]], i32 [[RESIZE46]]
// CHECK-NEXT:    [[DOWNSCALE47:%.*]] = ashr i32 [[TMP35]], 15
// CHECK-NEXT:    [[RESIZE48:%.*]] = trunc i32 [[DOWNSCALE47]] to i8
// CHECK-NEXT:    store i8 [[RESIZE48]], i8* @c, align 1
  c += a;

// CHECK:         [[TMP36:%.*]] = load i32, i32* @i, align 4
// CHECK-NEXT:    [[TMP37:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[RESIZE47:%.*]] = sext i32 [[TMP37]] to i47
// CHECK-NEXT:    [[RESIZE48:%.*]] = sext i32 [[TMP36]] to i47
// CHECK-NEXT:    [[UPSCALE49:%.*]] = shl i47 [[RESIZE48]], 15
// CHECK-NEXT:    [[TMP38:%.*]] = call i47 @llvm.sadd.sat.i47(i47 [[RESIZE47]], i47 [[UPSCALE49]])
// CHECK-NEXT:    [[TMP39:%.*]] = icmp sgt i47 [[TMP38]], 2147483647
// CHECK-NEXT:    [[SATMAX:%.*]] = select i1 [[TMP39]], i47 2147483647, i47 [[TMP38]]
// CHECK-NEXT:    [[TMP40:%.*]] = icmp slt i47 [[SATMAX]], -2147483648
// CHECK-NEXT:    [[SATMIN:%.*]] = select i1 [[TMP40]], i47 -2147483648, i47 [[SATMAX]]
// CHECK-NEXT:    [[RESIZE50:%.*]] = trunc i47 [[SATMIN]] to i32
// CHECK-NEXT:    store i32 [[RESIZE50]], i32* @sa, align 4
  sa += i;

// CHECK:         [[TMP41:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[TMP42:%.*]] = load i8, i8* @c, align 1
// CHECK-NEXT:    [[CONV53:%.*]] = sext i8 [[TMP42]] to i32
// CHECK-NEXT:    [[RESIZE54:%.*]] = sext i32 [[CONV53]] to i47
// CHECK-NEXT:    [[UPSCALE55:%.*]] = shl i47 [[RESIZE54]], 15
// CHECK-NEXT:    [[RESIZE56:%.*]] = sext i32 [[TMP41]] to i47
// CHECK-NEXT:    [[TMP43:%.*]] = call i47 @llvm.sadd.sat.i47(i47 [[UPSCALE55]], i47 [[RESIZE56]])
// CHECK-NEXT:    [[TMP44:%.*]] = icmp sgt i47 [[TMP43]], 2147483647
// CHECK-NEXT:    [[SATMAX57:%.*]] = select i1 [[TMP44]], i47 2147483647, i47 [[TMP43]]
// CHECK-NEXT:    [[TMP45:%.*]] = icmp slt i47 [[SATMAX57]], -2147483648
// CHECK-NEXT:    [[SATMIN58:%.*]] = select i1 [[TMP45]], i47 -2147483648, i47 [[SATMAX57]]
// CHECK-NEXT:    [[RESIZE59:%.*]] = trunc i47 [[SATMIN58]] to i32
// CHECK-NEXT:    [[TMP46:%.*]] = icmp slt i32 [[RESIZE59]], 0
// CHECK-NEXT:    [[TMP47:%.*]] = add i32 [[RESIZE59]], 32767
// CHECK-NEXT:    [[TMP48:%.*]] = select i1 [[TMP46]], i32 [[TMP47]], i32 [[RESIZE59]]
// CHECK-NEXT:    [[DOWNSCALE60:%.*]] = ashr i32 [[TMP48]], 15
// CHECK-NEXT:    [[RESIZE61:%.*]] = trunc i32 [[DOWNSCALE60]] to i8
// CHECK-NEXT:    store i8 [[RESIZE61]], i8* @c, align 1
  c += sa;

// CHECK:         [[TMP47:%.*]] = load i32, i32* @u, align 4
// CHECK-NEXT:    [[TMP48:%.*]] = load i64, i64* @sula, align 8
// SIGNED-NEXT:   [[RESIZE55:%.*]] = zext i32 [[TMP47]] to i64
// SIGNED-NEXT:   [[UPSCALE56:%.*]] = shl i64 [[RESIZE55]], 32
// SIGNED-NEXT:   [[TMP49:%.*]] = call i64 @llvm.uadd.sat.i64(i64 [[TMP48]], i64 [[UPSCALE56]])
// SIGNED-NEXT:   store i64 [[TMP49]], i64* @sula, align 8
// UNSIGNED-NEXT: [[RESIZE39:%.*]] = trunc i64 [[TMP48]] to i63
// UNSIGNED-NEXT: [[RESIZE40:%.*]] = zext i32 [[TMP47]] to i63
// UNSIGNED-NEXT: [[UPSCALE41:%.*]] = shl i63 [[RESIZE40]], 31
// UNSIGNED-NEXT: [[TMP49:%.*]] = call i63 @llvm.uadd.sat.i63(i63 [[RESIZE39]], i63 [[UPSCALE41]])
// UNSIGNED-NEXT: [[RESIZE42:%.*]] = zext i63 [[TMP49]] to i64
// UNSIGNED-NEXT: store i64 [[RESIZE42]], i64* @sula, align 8
  sula += u;

// CHECK:         [[TMP50:%.*]] = load i16, i16* @suf, align 2
// CHECK-NEXT:    [[TMP51:%.*]] = load i8, i8* @sshf, align 1
// SIGNED-NEXT:   [[RESIZE57:%.*]] = sext i8 [[TMP51]] to i17
// SIGNED-NEXT:   [[UPSCALE58:%.*]] = shl i17 [[RESIZE57]], 9
// SIGNED-NEXT:   [[RESIZE59:%.*]] = zext i16 [[TMP50]] to i17
// SIGNED-NEXT:   [[TMP52:%.*]] = call i17 @llvm.sadd.sat.i17(i17 [[UPSCALE58]], i17 [[RESIZE59]])
// SIGNED-NEXT:   [[DOWNSCALE60:%.*]] = ashr i17 [[TMP52]], 1
// SIGNED-NEXT:   [[RESIZE61:%.*]] = trunc i17 [[DOWNSCALE60]] to i16
// SIGNED-NEXT:   [[DOWNSCALE62:%.*]] = ashr i16 [[RESIZE61]], 8
// SIGNED-NEXT:   [[RESIZE63:%.*]] = trunc i16 [[DOWNSCALE62]] to i8
// SIGNED-NEXT:   store i8 [[RESIZE63]], i8* @sshf, align 1
// UNSIGNED-NEXT: [[RESIZE43:%.*]] = sext i8 [[TMP51]] to i16
// UNSIGNED-NEXT: [[UPSCALE44:%.*]] = shl i16 [[RESIZE43]], 8
// UNSIGNED-NEXT: [[TMP52:%.*]] = call i16 @llvm.sadd.sat.i16(i16 [[UPSCALE44]], i16 [[TMP50]])
// UNSIGNED-NEXT: [[DOWNSCALE45:%.*]] = ashr i16 [[TMP52]], 8
// UNSIGNED-NEXT: [[RESIZE46:%.*]] = trunc i16 [[DOWNSCALE45]] to i8
// UNSIGNED-NEXT: store i8 [[RESIZE46]], i8* @sshf, align 1
  sshf += suf;
}

// Subtraction, multiplication and division should work about the same, so
// just make sure we can do them.

// CHECK-LABEL: @Subtraction(
void Subtraction() {
// CHECK:         [[TMP0:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @a, align 4
// SIGNED-NEXT:   [[RESIZE:%.*]] = sext i32 [[TMP1]] to i33
// SIGNED-NEXT:   [[UPSCALE:%.*]] = shl i33 [[RESIZE]], 1
// SIGNED-NEXT:   [[RESIZE1:%.*]] = zext i16 [[TMP0]] to i33
// SIGNED-NEXT:   [[TMP2:%.*]] = sub i33 [[UPSCALE]], [[RESIZE1]]
// SIGNED-NEXT:   [[DOWNSCALE:%.*]] = ashr i33 [[TMP2]], 1
// SIGNED-NEXT:   [[RESIZE2:%.*]] = trunc i33 [[DOWNSCALE]] to i32
// SIGNED-NEXT:   store i32 [[RESIZE2]], i32* @a, align 4
// UNSIGNED-NEXT: [[RESIZE:%.*]] = zext i16 [[TMP0]] to i32
// UNSIGNED-NEXT: [[TMP2:%.*]] = sub i32 [[TMP1]], [[RESIZE]]
// UNSIGNED-NEXT: store i32 [[TMP2]], i32* @a, align 4
  a -= uf;

// CHECK:         [[TMP3:%.*]] = load i32, i32* @i, align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[RESIZE3:%.*]] = sext i32 [[TMP4]] to i47
// CHECK-NEXT:    [[RESIZE4:%.*]] = sext i32 [[TMP3]] to i47
// CHECK-NEXT:    [[UPSCALE5:%.*]] = shl i47 [[RESIZE4]], 15
// CHECK-NEXT:    [[TMP5:%.*]] = sub i47 [[RESIZE3]], [[UPSCALE5]]
// CHECK-NEXT:    [[RESIZE6:%.*]] = trunc i47 [[TMP5]] to i32
// CHECK-NEXT:    store i32 [[RESIZE6]], i32* @a, align 4
  a -= i;

// CHECK:         [[TMP6:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i8, i8* @c, align 1
// CHECK-NEXT:    [[CONV:%.*]] = sext i8 [[TMP7]] to i32
// CHECK-NEXT:    [[RESIZE7:%.*]] = sext i32 [[CONV]] to i47
// CHECK-NEXT:    [[UPSCALE8:%.*]] = shl i47 [[RESIZE7]], 15
// CHECK-NEXT:    [[RESIZE9:%.*]] = sext i32 [[TMP6]] to i47
// CHECK-NEXT:    [[TMP8:%.*]] = call i47 @llvm.ssub.sat.i47(i47 [[UPSCALE8]], i47 [[RESIZE9]])
// CHECK-NEXT:    [[TMP9:%.*]] = icmp sgt i47 [[TMP8]], 2147483647
// CHECK-NEXT:    [[SATMAX:%.*]] = select i1 [[TMP9]], i47 2147483647, i47 [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = icmp slt i47 [[SATMAX]], -2147483648
// CHECK-NEXT:    [[SATMIN:%.*]] = select i1 [[TMP10]], i47 -2147483648, i47 [[SATMAX]]
// CHECK-NEXT:    [[RESIZE10:%.*]] = trunc i47 [[SATMIN]] to i32
// CHECK-NEXT:    [[TMP11:%.*]] = icmp slt i32 [[RESIZE10]], 0
// CHECK-NEXT:    [[TMP12:%.*]] = add i32 [[RESIZE10]], 32767
// CHECK-NEXT:    [[TMP13:%.*]] = select i1 [[TMP11]], i32 [[TMP12]], i32 [[RESIZE10]]
// CHECK-NEXT:    [[DOWNSCALE11:%.*]] = ashr i32 [[TMP13]], 15
// CHECK-NEXT:    [[RESIZE12:%.*]] = trunc i32 [[DOWNSCALE11]] to i8
// CHECK-NEXT:    store i8 [[RESIZE12]], i8* @c, align 1
  c -= sa;
}

// CHECK-LABEL: @Multiplication(
void Multiplication() {
// CHECK:         [[TMP0:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @a, align 4
// SIGNED-NEXT:   [[RESIZE:%.*]] = sext i32 [[TMP1]] to i33
// SIGNED-NEXT:   [[UPSCALE:%.*]] = shl i33 [[RESIZE]], 1
// SIGNED-NEXT:   [[RESIZE1:%.*]] = zext i16 [[TMP0]] to i33
// SIGNED-NEXT:   [[TMP2:%.*]] = call i33 @llvm.smul.fix.i33(i33 [[UPSCALE]], i33 [[RESIZE1]], i32 16)
// SIGNED-NEXT:   [[DOWNSCALE:%.*]] = ashr i33 [[TMP2]], 1
// SIGNED-NEXT:   [[RESIZE2:%.*]] = trunc i33 [[DOWNSCALE]] to i32
// SIGNED-NEXT:   store i32 [[RESIZE2]], i32* @a, align 4
// UNSIGNED-NEXT: [[RESIZE:%.*]] = zext i16 [[TMP0]] to i32
// UNSIGNED-NEXT: [[TMP2:%.*]] = call i32 @llvm.smul.fix.i32(i32 [[TMP1]], i32 [[RESIZE]], i32 15)
// UNSIGNED-NEXT: store i32 [[TMP2]], i32* @a, align 4
  a *= uf;

// CHECK:         [[TMP3:%.*]] = load i32, i32* @i, align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[RESIZE3:%.*]] = sext i32 [[TMP4]] to i47
// CHECK-NEXT:    [[RESIZE4:%.*]] = sext i32 [[TMP3]] to i47
// CHECK-NEXT:    [[UPSCALE5:%.*]] = shl i47 [[RESIZE4]], 15
// CHECK-NEXT:    [[TMP5:%.*]] = call i47 @llvm.smul.fix.i47(i47 [[RESIZE3]], i47 [[UPSCALE5]], i32 15)
// CHECK-NEXT:    [[RESIZE6:%.*]] = trunc i47 [[TMP5]] to i32
// CHECK-NEXT:    store i32 [[RESIZE6]], i32* @a, align 4
  a *= i;

// CHECK:         [[TMP6:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i8, i8* @c, align 1
// CHECK-NEXT:    [[CONV:%.*]] = sext i8 [[TMP7]] to i32
// CHECK-NEXT:    [[RESIZE7:%.*]] = sext i32 [[CONV]] to i47
// CHECK-NEXT:    [[UPSCALE8:%.*]] = shl i47 [[RESIZE7]], 15
// CHECK-NEXT:    [[RESIZE9:%.*]] = sext i32 [[TMP6]] to i47
// CHECK-NEXT:    [[TMP8:%.*]] = call i47 @llvm.smul.fix.sat.i47(i47 [[UPSCALE8]], i47 [[RESIZE9]], i32 15)
// CHECK-NEXT:    [[TMP9:%.*]] = icmp sgt i47 [[TMP8]], 2147483647
// CHECK-NEXT:    [[SATMAX:%.*]] = select i1 [[TMP9]], i47 2147483647, i47 [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = icmp slt i47 [[SATMAX]], -2147483648
// CHECK-NEXT:    [[SATMIN:%.*]] = select i1 [[TMP10]], i47 -2147483648, i47 [[SATMAX]]
// CHECK-NEXT:    [[RESIZE10:%.*]] = trunc i47 [[SATMIN]] to i32
// CHECK-NEXT:    [[TMP11:%.*]] = icmp slt i32 [[RESIZE10]], 0
// CHECK-NEXT:    [[TMP12:%.*]] = add i32 [[RESIZE10]], 32767
// CHECK-NEXT:    [[TMP13:%.*]] = select i1 [[TMP11]], i32 [[TMP12]], i32 [[RESIZE10]]
// CHECK-NEXT:    [[DOWNSCALE11:%.*]] = ashr i32 [[TMP13]], 15
// CHECK-NEXT:    [[RESIZE12:%.*]] = trunc i32 [[DOWNSCALE11]] to i8
// CHECK-NEXT:    store i8 [[RESIZE12]], i8* @c, align 1
  c *= sa;
}

// CHECK-LABEL: @Division(
void Division() {
// CHECK:         [[TMP0:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @a, align 4
// SIGNED-NEXT:   [[RESIZE:%.*]] = sext i32 [[TMP1]] to i33
// SIGNED-NEXT:   [[UPSCALE:%.*]] = shl i33 [[RESIZE]], 1
// SIGNED-NEXT:   [[RESIZE1:%.*]] = zext i16 [[TMP0]] to i33
// SIGNED-NEXT:   [[TMP2:%.*]] = call i33 @llvm.sdiv.fix.i33(i33 [[UPSCALE]], i33 [[RESIZE1]], i32 16)
// SIGNED-NEXT:   [[DOWNSCALE:%.*]] = ashr i33 [[TMP2]], 1
// SIGNED-NEXT:   [[RESIZE2:%.*]] = trunc i33 [[DOWNSCALE]] to i32
// SIGNED-NEXT:   store i32 [[RESIZE2]], i32* @a, align 4
// UNSIGNED-NEXT: [[RESIZE:%.*]] = zext i16 [[TMP0]] to i32
// UNSIGNED-NEXT: [[TMP2:%.*]] = call i32 @llvm.sdiv.fix.i32(i32 [[TMP1]], i32 [[RESIZE]], i32 15)
// UNSIGNED-NEXT: store i32 [[TMP2]], i32* @a, align 4
  a /= uf;

// CHECK:         [[TMP3:%.*]] = load i32, i32* @i, align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[RESIZE3:%.*]] = sext i32 [[TMP4]] to i47
// CHECK-NEXT:    [[RESIZE4:%.*]] = sext i32 [[TMP3]] to i47
// CHECK-NEXT:    [[UPSCALE5:%.*]] = shl i47 [[RESIZE4]], 15
// CHECK-NEXT:    [[TMP5:%.*]] = call i47 @llvm.sdiv.fix.i47(i47 [[RESIZE3]], i47 [[UPSCALE5]], i32 15)
// CHECK-NEXT:    [[RESIZE6:%.*]] = trunc i47 [[TMP5]] to i32
// CHECK-NEXT:    store i32 [[RESIZE6]], i32* @a, align 4
  a /= i;

// CHECK:         [[TMP6:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i8, i8* @c, align 1
// CHECK-NEXT:    [[CONV:%.*]] = sext i8 [[TMP7]] to i32
// CHECK-NEXT:    [[RESIZE7:%.*]] = sext i32 [[CONV]] to i47
// CHECK-NEXT:    [[UPSCALE8:%.*]] = shl i47 [[RESIZE7]], 15
// CHECK-NEXT:    [[RESIZE9:%.*]] = sext i32 [[TMP6]] to i47
// CHECK-NEXT:    [[TMP8:%.*]] = call i47 @llvm.sdiv.fix.sat.i47(i47 [[UPSCALE8]], i47 [[RESIZE9]], i32 15)
// CHECK-NEXT:    [[TMP9:%.*]] = icmp sgt i47 [[TMP8]], 2147483647
// CHECK-NEXT:    [[SATMAX:%.*]] = select i1 [[TMP9]], i47 2147483647, i47 [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = icmp slt i47 [[SATMAX]], -2147483648
// CHECK-NEXT:    [[SATMIN:%.*]] = select i1 [[TMP10]], i47 -2147483648, i47 [[SATMAX]]
// CHECK-NEXT:    [[RESIZE10:%.*]] = trunc i47 [[SATMIN]] to i32
// CHECK-NEXT:    [[TMP11:%.*]] = icmp slt i32 [[RESIZE10]], 0
// CHECK-NEXT:    [[TMP12:%.*]] = add i32 [[RESIZE10]], 32767
// CHECK-NEXT:    [[TMP13:%.*]] = select i1 [[TMP11]], i32 [[TMP12]], i32 [[RESIZE10]]
// CHECK-NEXT:    [[DOWNSCALE11:%.*]] = ashr i32 [[TMP13]], 15
// CHECK-NEXT:    [[RESIZE12:%.*]] = trunc i32 [[DOWNSCALE11]] to i8
// CHECK-NEXT:    store i8 [[RESIZE12]], i8* @c, align 1
  c /= sa;
}

