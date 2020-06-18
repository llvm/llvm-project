// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fpadding-on-unsigned-fixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

_Accum a;
_Fract f;
long _Fract lf;
unsigned _Accum ua;
short unsigned _Accum usa;
unsigned _Fract uf;

_Sat _Accum sa;
_Sat _Fract sf;
_Sat long _Fract slf;
_Sat unsigned _Accum sua;
_Sat short unsigned _Accum susa;
_Sat unsigned _Fract suf;

// CHECK-LABEL: @Increment(
void Increment() {
// CHECK:         [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], -32768
// CHECK-NEXT:    store i32 [[TMP1]], i32* @a, align 4
  a++;

// CHECK:         [[TMP2:%.*]] = load i16, i16* @f, align 2
// CHECK-NEXT:    [[TMP3:%.*]] = sub i16 [[TMP2]], -32768
// CHECK-NEXT:    store i16 [[TMP3]], i16* @f, align 2
  f++;

// CHECK:         [[TMP4:%.*]] = load i32, i32* @lf, align 4
// CHECK-NEXT:    [[TMP5:%.*]] = sub i32 [[TMP4]], -2147483648
// CHECK-NEXT:    store i32 [[TMP5]], i32* @lf, align 4
  lf++;

// CHECK:         [[TMP6:%.*]] = load i32, i32* @ua, align 4
// SIGNED-NEXT:   [[TMP7:%.*]] = add i32 [[TMP6]], 65536
// UNSIGNED-NEXT: [[TMP7:%.*]] = add i32 [[TMP6]], 32768
// CHECK-NEXT:    store i32 [[TMP7]], i32* @ua, align 4
  ua++;

// CHECK:         [[TMP8:%.*]] = load i16, i16* @usa, align 2
// SIGNED-NEXT:   [[TMP9:%.*]] = add i16 [[TMP8]], 256
// UNSIGNED-NEXT: [[TMP9:%.*]] = add i16 [[TMP8]], 128
// CHECK-NEXT:    store i16 [[TMP9]], i16* @usa, align 2
  usa++;

// CHECK:         [[TMP10:%.*]] = load i16, i16* @uf, align 2
// SIGNED-NEXT:   [[TMP11:%.*]] = add i16 [[TMP10]], undef
// UNSIGNED-NEXT: [[TMP11:%.*]] = add i16 [[TMP10]], -32768
// CHECK-NEXT:    store i16 [[TMP11]], i16* @uf, align 2
  uf++;

// CHECK:         [[TMP12:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[TMP13:%.*]] = call i32 @llvm.ssub.sat.i32(i32 [[TMP12]], i32 -32768)
// CHECK-NEXT:    store i32 [[TMP13]], i32* @sa, align 4
  sa++;

// CHECK:         [[TMP14:%.*]] = load i16, i16* @sf, align 2
// CHECK-NEXT:    [[TMP15:%.*]] = call i16 @llvm.ssub.sat.i16(i16 [[TMP14]], i16 -32768)
// CHECK-NEXT:    store i16 [[TMP15]], i16* @sf, align 2
  sf++;

// CHECK:         [[TMP16:%.*]] = load i32, i32* @slf, align 4
// CHECK-NEXT:    [[TMP17:%.*]] = call i32 @llvm.ssub.sat.i32(i32 [[TMP16]], i32 -2147483648)
// CHECK-NEXT:    store i32 [[TMP17]], i32* @slf, align 4
  slf++;

// CHECK:         [[TMP18:%.*]] = load i32, i32* @sua, align 4
// SIGNED-NEXT:   [[TMP19:%.*]] = call i32 @llvm.uadd.sat.i32(i32 [[TMP18]], i32 65536)
// SIGNED-NEXT:   store i32 [[TMP19]], i32* @sua, align 4
// UNSIGNED-NEXT: [[RESIZE:%.*]] = trunc i32 [[TMP18]] to i31
// UNSIGNED-NEXT: [[TMP19:%.*]] = call i31 @llvm.uadd.sat.i31(i31 [[RESIZE]], i31 32768)
// UNSIGNED-NEXT: [[RESIZE1:%.*]] = zext i31 [[TMP19]] to i32
// UNSIGNED-NEXT: store i32 [[RESIZE1]], i32* @sua, align 4
  sua++;

// CHECK:         [[TMP20:%.*]] = load i16, i16* @susa, align 2
// SIGNED-NEXT:   [[TMP21:%.*]] = call i16 @llvm.uadd.sat.i16(i16 [[TMP20]], i16 256)
// SIGNED-NEXT:   store i16 [[TMP21]], i16* @susa, align 2
// UNSIGNED-NEXT: [[RESIZE2:%.*]] = trunc i16 [[TMP20]] to i15
// UNSIGNED-NEXT: [[TMP21:%.*]] = call i15 @llvm.uadd.sat.i15(i15 [[RESIZE2]], i15 128)
// UNSIGNED-NEXT: [[RESIZE3:%.*]] = zext i15 [[TMP21]] to i16
// UNSIGNED-NEXT: store i16 [[RESIZE3]], i16* @susa, align 2
  susa++;

// CHECK:         [[TMP22:%.*]] = load i16, i16* @suf, align 2
// SIGNED-NEXT:   [[TMP23:%.*]] = call i16 @llvm.uadd.sat.i16(i16 [[TMP22]], i16 -1)
// SIGNED-NEXT:   store i16 [[TMP23]], i16* @suf, align 2
// UNSIGNED-NEXT: [[RESIZE4:%.*]] = trunc i16 [[TMP22]] to i15
// UNSIGNED-NEXT: [[TMP23:%.*]] = call i15 @llvm.uadd.sat.i15(i15 [[RESIZE4]], i15 -1)
// UNSIGNED-NEXT: [[RESIZE5:%.*]] = zext i15 [[TMP23]] to i16
// UNSIGNED-NEXT: store i16 [[RESIZE5]], i16* @suf, align 2
  suf++;
}

// CHECK-LABEL: @Decrement(
void Decrement() {
// CHECK:         [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = add i32 [[TMP0]], -32768
// CHECK-NEXT:    store i32 [[TMP1]], i32* @a, align 4
  a--;

// CHECK:         [[TMP2:%.*]] = load i16, i16* @f, align 2
// CHECK-NEXT:    [[TMP3:%.*]] = add i16 [[TMP2]], -32768
// CHECK-NEXT:    store i16 [[TMP3]], i16* @f, align 2
  f--;

// CHECK:         [[TMP4:%.*]] = load i32, i32* @lf, align 4
// CHECK-NEXT:    [[TMP5:%.*]] = add i32 [[TMP4]], -2147483648
// CHECK-NEXT:    store i32 [[TMP5]], i32* @lf, align 4
  lf--;

// CHECK:         [[TMP6:%.*]] = load i32, i32* @ua, align 4
// SIGNED-NEXT:   [[TMP7:%.*]] = sub i32 [[TMP6]], 65536
// UNSIGNED-NEXT: [[TMP7:%.*]] = sub i32 [[TMP6]], 32768
// CHECK-NEXT:    store i32 [[TMP7]], i32* @ua, align 4
  ua--;

// CHECK:         [[TMP8:%.*]] = load i16, i16* @usa, align 2
// SIGNED-NEXT:   [[TMP9:%.*]] = sub i16 [[TMP8]], 256
// UNSIGNED-NEXT: [[TMP9:%.*]] = sub i16 [[TMP8]], 128
// CHECK-NEXT:    store i16 [[TMP9]], i16* @usa, align 2
  usa--;

// CHECK:         [[TMP10:%.*]] = load i16, i16* @uf, align 2
// SIGNED-NEXT:   [[TMP11:%.*]] = sub i16 [[TMP10]], undef
// UNSIGNED-NEXT: [[TMP11:%.*]] = sub i16 [[TMP10]], -32768
// CHECK-NEXT:    store i16 [[TMP11]], i16* @uf, align 2
  uf--;

// CHECK:         [[TMP12:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[TMP13:%.*]] = call i32 @llvm.sadd.sat.i32(i32 [[TMP12]], i32 -32768)
// CHECK-NEXT:    store i32 [[TMP13]], i32* @sa, align 4
  sa--;

// CHECK:         [[TMP14:%.*]] = load i16, i16* @sf, align 2
// CHECK-NEXT:    [[TMP15:%.*]] = call i16 @llvm.sadd.sat.i16(i16 [[TMP14]], i16 -32768)
// CHECK-NEXT:    store i16 [[TMP15]], i16* @sf, align 2
  sf--;

// CHECK:         [[TMP16:%.*]] = load i32, i32* @slf, align 4
// CHECK-NEXT:    [[TMP17:%.*]] = call i32 @llvm.sadd.sat.i32(i32 [[TMP16]], i32 -2147483648)
// CHECK-NEXT:    store i32 [[TMP17]], i32* @slf, align 4
  slf--;

// CHECK:         [[TMP18:%.*]] = load i32, i32* @sua, align 4
// SIGNED-NEXT:   [[TMP19:%.*]] = call i32 @llvm.usub.sat.i32(i32 [[TMP18]], i32 65536)
// SIGNED-NEXT:   store i32 [[TMP19]], i32* @sua, align 4
// UNSIGNED-NEXT: [[RESIZE:%.*]] = trunc i32 [[TMP18]] to i31
// UNSIGNED-NEXT: [[TMP19:%.*]] = call i31 @llvm.usub.sat.i31(i31 [[RESIZE]], i31 32768)
// UNSIGNED-NEXT: [[RESIZE1:%.*]] = zext i31 [[TMP19]] to i32
// UNSIGNED-NEXT: store i32 [[RESIZE1]], i32* @sua, align 4
  sua--;

// CHECK:         [[TMP20:%.*]] = load i16, i16* @susa, align 2
// SIGNED-NEXT:   [[TMP21:%.*]] = call i16 @llvm.usub.sat.i16(i16 [[TMP20]], i16 256)
// SIGNED-NEXT:   store i16 [[TMP21]], i16* @susa, align 2
// UNSIGNED-NEXT: [[RESIZE2:%.*]] = trunc i16 [[TMP20]] to i15
// UNSIGNED-NEXT: [[TMP21:%.*]] = call i15 @llvm.usub.sat.i15(i15 [[RESIZE2]], i15 128)
// UNSIGNED-NEXT: [[RESIZE3:%.*]] = zext i15 [[TMP21]] to i16
// UNSIGNED-NEXT: store i16 [[RESIZE3]], i16* @susa, align 2
  susa--;

// CHECK:         [[TMP22:%.*]] = load i16, i16* @suf, align 2
// SIGNED-NEXT:   [[TMP23:%.*]] = call i16 @llvm.usub.sat.i16(i16 [[TMP22]], i16 -1)
// SIGNED-NEXT:   store i16 [[TMP23]], i16* @suf, align 2
// UNSIGNED-NEXT: [[RESIZE4:%.*]] = trunc i16 [[TMP22]] to i15
// UNSIGNED-NEXT: [[TMP23:%.*]] = call i15 @llvm.usub.sat.i15(i15 [[RESIZE4]], i15 -1)
// UNSIGNED-NEXT: [[RESIZE5:%.*]] = zext i15 [[TMP23]] to i16
// UNSIGNED-NEXT: store i16 [[RESIZE5]], i16* @suf, align 2
  suf--;
}

// CHECK-LABEL: @Minus(
void Minus() {
// CHECK:         [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = sub i32 0, [[TMP0]]
// CHECK-NEXT:    store i32 [[TMP1]], i32* @a, align 4
  a = -a;

// CHECK:         [[TMP2:%.*]] = load i16, i16* @f, align 2
// CHECK-NEXT:    [[TMP3:%.*]] = sub i16 0, [[TMP2]]
// CHECK-NEXT:    store i16 [[TMP3]], i16* @f, align 2
  f = -f;

// CHECK:         [[TMP4:%.*]] = load i16, i16* @usa, align 2
// CHECK-NEXT:    [[TMP5:%.*]] = sub i16 0, [[TMP4]]
// CHECK-NEXT:    store i16 [[TMP5]], i16* @usa, align 2
  usa = -usa;

// CHECK:         [[TMP6:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    [[TMP7:%.*]] = sub i16 0, [[TMP6]]
// CHECK-NEXT:    store i16 [[TMP7]], i16* @uf, align 2
  uf = -uf;

// CHECK:         [[TMP8:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    [[TMP9:%.*]] = call i32 @llvm.ssub.sat.i32(i32 0, i32 [[TMP8]])
// CHECK-NEXT:    store i32 [[TMP9]], i32* @sa, align 4
  sa = -sa;

// CHECK:         [[TMP10:%.*]] = load i16, i16* @sf, align 2
// CHECK-NEXT:    [[TMP11:%.*]] = call i16 @llvm.ssub.sat.i16(i16 0, i16 [[TMP10]])
// CHECK-NEXT:    store i16 [[TMP11]], i16* @sf, align 2
  sf = -sf;

// CHECK:         [[TMP12:%.*]] = load i16, i16* @susa, align 2
// SIGNED-NEXT:   [[TMP13:%.*]] = call i16 @llvm.usub.sat.i16(i16 0, i16 [[TMP12]])
// SIGNED-NEXT:   store i16 [[TMP13]], i16* @susa, align 2
// UNSIGNED-NEXT: [[RESIZE:%.*]] = trunc i16 [[TMP12]] to i15
// UNSIGNED-NEXT: [[TMP13:%.*]] = call i15 @llvm.usub.sat.i15(i15 0, i15 [[RESIZE]])
// UNSIGNED-NEXT: [[RESIZE1:%.*]] = zext i15 [[TMP13]] to i16
// UNSIGNED-NEXT: store i16 [[RESIZE1]], i16* @susa, align 2
  susa = -susa;

// CHECK:         [[TMP14:%.*]] = load i16, i16* @suf, align 2
// SIGNED-NEXT:   [[TMP15:%.*]] = call i16 @llvm.usub.sat.i16(i16 0, i16 [[TMP14]])
// SIGNED-NEXT:   store i16 [[TMP15]], i16* @suf, align 2
// UNSIGNED-NEXT: [[RESIZE2:%.*]] = trunc i16 [[TMP14]] to i15
// UNSIGNED-NEXT: [[TMP15:%.*]] = call i15 @llvm.usub.sat.i15(i15 0, i15 [[RESIZE2]])
// UNSIGNED-NEXT: [[RESIZE3:%.*]] = zext i15 [[TMP15]] to i16
// UNSIGNED-NEXT: store i16 [[RESIZE3]], i16* @suf, align 2
  suf = -suf;
}

// CHECK-LABEL: @Plus(
void Plus() {
// CHECK:         [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    store i32 [[TMP0]], i32* @a, align 4
  a = +a;

// CHECK:         [[TMP1:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    store i16 [[TMP1]], i16* @uf, align 2
  uf = +uf;

// CHECK:         [[TMP2:%.*]] = load i32, i32* @sa, align 4
// CHECK-NEXT:    store i32 [[TMP2]], i32* @sa, align 4
  sa = +sa;
}

// CHECK-LABEL: @Not(
void Not() {
  int i;

// CHECK:         [[TMP0:%.*]] = load i32, i32* @a, align 4
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP0]], 0
// CHECK-NEXT:    [[LNOT:%.*]] = xor i1 [[TOBOOL]], true
// CHECK-NEXT:    [[LNOT_EXT:%.*]] = zext i1 [[LNOT]] to i32
// CHECK-NEXT:    store i32 [[LNOT_EXT]], i32* %i, align 4
  i = !a;

// CHECK:         [[TMP1:%.*]] = load i16, i16* @uf, align 2
// CHECK-NEXT:    [[TOBOOL1:%.*]] = icmp ne i16 [[TMP1]], 0
// CHECK-NEXT:    [[LNOT2:%.*]] = xor i1 [[TOBOOL1]], true
// CHECK-NEXT:    [[LNOT_EXT3:%.*]] = zext i1 [[LNOT2]] to i32
// CHECK-NEXT:    store i32 [[LNOT_EXT3]], i32* %i, align 4
  i = !uf;

// CHECK:         [[TMP2:%.*]] = load i16, i16* @susa, align 2
// CHECK-NEXT:    [[TOBOOL4:%.*]] = icmp ne i16 [[TMP2]], 0
// CHECK-NEXT:    [[LNOT5:%.*]] = xor i1 [[TOBOOL4]], true
// CHECK-NEXT:    [[LNOT_EXT6:%.*]] = zext i1 [[LNOT5]] to i32
// CHECK-NEXT:    store i32 [[LNOT_EXT6]], i32* %i, align 4
  i = !susa;
}
