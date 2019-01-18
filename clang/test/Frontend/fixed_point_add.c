// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -fpadding-on-unsigned-fixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

// Addition between different fixed point types
short _Accum sa_const = 1.0hk + 2.0hk;  // CHECK-DAG: @sa_const  = {{.*}}global i16 384, align 2
_Accum a_const = 1.0hk + 2.0k;          // CHECK-DAG: @a_const   = {{.*}}global i32 98304, align 4
long _Accum la_const = 1.0hk + 2.0lk;   // CHECK-DAG: @la_const  = {{.*}}global i64 6442450944, align 8
short _Accum sa_const2 = 0.5hr + 2.0hk; // CHECK-DAG: @sa_const2  = {{.*}}global i16 320, align 2
short _Accum sa_const3 = 0.5r + 2.0hk;  // CHECK-DAG: @sa_const3  = {{.*}}global i16 320, align 2
short _Accum sa_const4 = 0.5lr + 2.0hk; // CHECK-DAG: @sa_const4  = {{.*}}global i16 320, align 2

// Unsigned addition
unsigned short _Accum usa_const = 1.0uhk + 2.0uhk;
// CHECK-SIGNED-DAG:   @usa_const = {{.*}}global i16 768, align 2
// CHECK-UNSIGNED-DAG: @usa_const = {{.*}}global i16 384, align 2

// Unsigned + signed
short _Accum sa_const5 = 1.0uhk + 2.0hk;
// CHECK-DAG: @sa_const5 = {{.*}}global i16 384, align 2

// Addition with negative number
short _Accum sa_const6 = 0.5hr + (-2.0hk);
// CHECK-DAG: @sa_const6 = {{.*}}global i16 -192, align 2

// Int addition
unsigned short _Accum usa_const2 = 2 + 0.5uhk;
// CHECK-SIGNED-DAG:   @usa_const2 = {{.*}}global i16 640, align 2
// CHECK-UNSIGNED-DAG: @usa_const2 = {{.*}}global i16 320, align 2
short _Accum sa_const7 = 2 + (-0.5hk);   // CHECK-DAG: @sa_const7 = {{.*}}global i16 192, align 2
short _Accum sa_const8 = 257 + (-2.0hk); // CHECK-DAG: @sa_const8 = {{.*}}global i16 32640, align 2
long _Fract lf_const = -0.5lr + 1;       // CHECK-DAG: @lf_const  = {{.*}}global i32 1073741824, align 4

// Saturated addition
_Sat short _Accum sat_sa_const = (_Sat short _Accum)128.0hk + 128.0hk;
// CHECK-DAG: @sat_sa_const = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const = (_Sat unsigned short _Accum)128.0uhk + 128.0uhk;
// CHECK-SIGNED-DAG:   @sat_usa_const = {{.*}}global i16 65535, align 2
// CHECK-UNSIGNED-DAG: @sat_usa_const = {{.*}}global i16 32767, align 2
_Sat short _Accum sat_sa_const2 = (_Sat short _Accum)128.0hk + 128;
// CHECK-DAG: @sat_sa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const2 = (_Sat unsigned short _Accum)128.0uhk + 128;
// CHECK-SIGNED-DAG:   @sat_usa_const2 = {{.*}}global i16 65535, align 2
// CHECK-UNSIGNED-DAG: @sat_usa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const3 = (_Sat unsigned short _Accum)0.5uhk + (-2);
// CHECK-DAG:   @sat_usa_const3 = {{.*}}global i16 0, align 2

void SignedAddition() {
  // CHECK-LABEL: SignedAddition
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
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[SA2:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i16 [[SA]], [[SA2]]
  // CHECK-NEXT: store i16 [[SUM]], i16* %sa, align 2
  sa = sa + sa;

  // To larger scale and larger width
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[EXT_SA:%[a-z0-9]+]] = sext i16 [[SA]] to i32
  // CHECK-NEXT: [[SA:%[a-z0-9]+]] = shl i32 [[EXT_SA]], 8
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i32 [[SA]], [[A]]
  // CHECK-NEXT: store i32 [[SUM]], i32* %a, align 4
  a = sa + a;

  // To same scale and smaller width
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[SF:%[0-9]+]] = load i8, i8* %sf, align 1
  // CHECK-NEXT: [[EXT_SF:%[a-z0-9]+]] = sext i8 [[SF]] to i16
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i16 [[SA]], [[EXT_SF]]
  // CHECK-NEXT: store i16 [[SUM]], i16* %sa, align 2
  sa = sa + sf;

  // To smaller scale and same width.
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[F:%[0-9]+]] = load i16, i16* %f, align 2
  // CHECK-NEXT: [[EXT_SA:%[a-z0-9]+]] = sext i16 [[SA]] to i24
  // CHECK-NEXT: [[SA:%[a-z0-9]+]] = shl i24 [[EXT_SA]], 8
  // CHECK-NEXT: [[EXT_F:%[a-z0-9]+]] = sext i16 [[F]] to i24
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i24 [[SA]], [[EXT_F]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = ashr i24 [[SUM]], 8
  // CHECK-NEXT: [[TRUNC_RES:%[a-z0-9]+]] = trunc i24 [[RES]] to i16
  // CHECK-NEXT: store i16 [[TRUNC_RES]], i16* %sa, align 2
  sa = sa + f;

  // To smaller scale and smaller width
  // CHECK:      [[A:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[SF:%[0-9]+]] = load i8, i8* %sf, align 1
  // CHECK-NEXT: [[EXT_SF:%[a-z0-9]+]] = sext i8 [[SF]] to i32
  // CHECK-NEXT: [[SF:%[a-z0-9]+]] = shl i32 [[EXT_SF]], 8
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i32 [[A]], [[SF]]
  // CHECK-NEXT: store i32 [[SUM]], i32* %a, align 4
  a = a + sf;

  // To larger scale and same width
  // CHECK:      [[A:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[LF:%[0-9]+]] = load i32, i32* %lf, align 4
  // CHECK-NEXT: [[EXT_A:%[a-z0-9]+]] = sext i32 [[A]] to i48
  // CHECK-NEXT: [[A:%[a-z0-9]+]] = shl i48 [[EXT_A]], 16
  // CHECK-NEXT: [[EXT_LF:%[a-z0-9]+]] = sext i32 [[LF]] to i48
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i48 [[A]], [[EXT_LF]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = ashr i48 [[SUM]], 16
  // CHECK-NEXT: [[TRUNC_RES:%[a-z0-9]+]] = trunc i48 [[RES]] to i32
  // CHECK-NEXT: store i32 [[TRUNC_RES]], i32* %a, align 4
  a = a + lf;

  // With corresponding unsigned type
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // SIGNED-NEXT: [[SA_EXT:%[a-z0-9]+]] = sext i16 [[SA]] to i17
  // SIGNED-NEXT: [[SA:%[a-z0-9]+]] = shl i17 [[SA_EXT]], 1
  // SIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i17
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = add i17 [[SA]], [[USA_EXT]]
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = ashr i17 [[SUM]], 1
  // SIGNED-NEXT: [[SUM:%[a-z0-9]+]] = trunc i17 [[RESULT]] to i16
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = add i16 [[SA]], [[USA]]
  // CHECK-NEXT: store i16 [[SUM]], i16* %sa, align 2
  sa = sa + usa;

  // With unsigned of larger scale
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[USA:%[0-9]+]] = load i32, i32* %ua, align 4
  // SIGNED-NEXT: [[SA_EXT:%[a-z0-9]+]] = sext i16 [[SA]] to i33
  // SIGNED-NEXT: [[SA:%[a-z0-9]+]] = shl i33 [[SA_EXT]], 9
  // SIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i32 [[USA]] to i33
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = add i33 [[SA]], [[USA_EXT]]
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = ashr i33 [[SUM]], 1
  // SIGNED-NEXT: [[SUM:%[a-z0-9]+]] = trunc i33 [[RESULT]] to i32
  // UNSIGNED-NEXT: [[EXT_SA:%[a-z0-9]+]] = sext i16 [[SA]] to i32
  // UNSIGNED-NEXT: [[SA:%[a-z0-9]+]] = shl i32 [[EXT_SA]], 8
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = add i32 [[SA]], [[USA]]
  // CHECK-NEXT: store i32 [[SUM]], i32* %a, align 4
  a = sa + ua;

  // With unsigned of smaller width
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[USF:%[0-9]+]] = load i8, i8* %usf, align 1
  // SIGNED-NEXT: [[SA_EXT:%[a-z0-9]+]] = sext i16 [[SA]] to i17
  // SIGNED-NEXT: [[SA:%[a-z0-9]+]] = shl i17 [[SA_EXT]], 1
  // SIGNED-NEXT: [[USF_EXT:%[a-z0-9]+]] = zext i8 [[USF]] to i17
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = add i17 [[SA]], [[USF_EXT]]
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = ashr i17 [[SUM]], 1
  // SIGNED-NEXT: [[SUM:%[a-z0-9]+]] = trunc i17 [[RESULT]] to i16
  // UNSIGNED-NEXT: [[EXT_USF:%[a-z0-9]+]] = zext i8 [[USF]] to i16
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = add i16 [[SA]], [[EXT_USF]]
  // CHECK-NEXT: store i16 [[SUM]], i16* %sa, align 2
  sa = sa + usf;

  // With unsigned of larger width and smaller scale
  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[ULF:%[0-9]+]] = load i32, i32* %ulf, align 4
  // SIGNED-NEXT: [[SA_EXT:%[a-z0-9]+]] = sext i16 [[SA]] to i41
  // SIGNED-NEXT: [[SA:%[a-z0-9]+]] = shl i41 [[SA_EXT]], 25
  // SIGNED-NEXT: [[ULF_EXT:%[a-z0-9]+]] = zext i32 [[ULF]] to i41
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = add i41 [[SA]], [[ULF_EXT]]
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = ashr i41 [[SUM]], 25
  // SIGNED-NEXT: [[RES_TRUNC:%[a-z0-9]+]] = trunc i41 [[RESULT]] to i16
  // UNSIGNED-NEXT: [[EXT_SA:%[a-z0-9]+]] = sext i16 [[SA]] to i40
  // UNSIGNED-NEXT: [[SA:%[a-z0-9]+]] = shl i40 [[EXT_SA]], 24
  // UNSIGNED-NEXT: [[EXT_ULF:%[a-z0-9]+]] = zext i32 [[ULF]] to i40
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = add i40 [[SA]], [[EXT_ULF]]
  // UNSIGNED-NEXT: [[RES:%[a-z0-9]+]] = ashr i40 [[SUM]], 24
  // UNSIGNED-NEXT: [[RES_TRUNC:%[a-z0-9]+]] = trunc i40 [[RES]] to i16
  // CHECK-NEXT: store i16 [[RES_TRUNC]], i16* %sa, align 2
  sa = sa + ulf;

  // Chained additions of the same signed type should result in the same
  // semantics width.
  // CHECK:      [[A:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[B:%[0-9]+]] = load i32, i32* %b, align 4
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i32 [[A]], [[B]]
  // CHECK-NEXT: [[C:%[0-9]+]] = load i32, i32* %c, align 4
  // CHECK-NEXT: [[SUM2:%[0-9]+]] = add i32 [[SUM]], [[C]]
  // CHECK-NEXT: [[D:%[0-9]+]] = load i32, i32* %d, align 4
  // CHECK-NEXT: [[SUM3:%[0-9]+]] = add i32 [[SUM2]], [[D]]
  // CHECK-NEXT: store i32 [[SUM3]], i32* %a, align 4
  a = a + b + c + d;
}

void UnsignedAddition() {
  // CHECK-LABEL: UnsignedAddition
  unsigned short _Accum usa;
  unsigned _Accum ua;
  unsigned long _Accum ula;

  unsigned short _Fract usf;
  unsigned _Fract uf;
  unsigned long _Fract ulf;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[USA2:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i16 [[USA]], [[USA2]]
  // CHECK-NEXT: store i16 [[SUM]], i16* %usa, align 2
  usa = usa + usa;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[UA:%[0-9]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[EXT_USA:%[a-z0-9]+]] = zext i16 [[USA]] to i32
  // CHECK-NEXT: [[USA:%[a-z0-9]+]] = shl i32 [[EXT_USA]], 8
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i32 [[USA]], [[UA]]
  // CHECK-NEXT: store i32 [[SUM]], i32* %ua, align 4
  ua = usa + ua;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[USF:%[0-9]+]] = load i8, i8* %usf, align 1
  // CHECK-NEXT: [[EXT_USF:%[a-z0-9]+]] = zext i8 [[USF]] to i16
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i16 [[USA]], [[EXT_USF]]
  // CHECK-NEXT: store i16 [[SUM]], i16* %usa, align 2
  usa = usa + usf;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[UF:%[0-9]+]] = load i16, i16* %uf, align 2
  // CHECK-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i24
  // CHECK-NEXT: [[USA:%[a-z0-9]+]] = shl i24 [[USA_EXT]], 8
  // CHECK-NEXT: [[UF_EXT:%[a-z0-9]+]] = zext i16 [[UF]] to i24
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i24 [[USA]], [[UF_EXT]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = lshr i24 [[SUM]], 8
  // CHECK-NEXT: [[RES_TRUNC:%[a-z0-9]+]] = trunc i24 [[RES]] to i16
  // CHECK-NEXT: store i16 [[RES_TRUNC]], i16* %usa, align 2
  usa = usa + uf;
}

void IntAddition() {
  // CHECK-LABEL: IntAddition
  short _Accum sa;
  _Accum a;
  unsigned short _Accum usa;
  _Sat short _Accum sa_sat;
  int i;
  unsigned int ui;
  long _Fract lf;
  _Bool b;

  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[SA_EXT:%[a-z0-9]+]] = sext i16 [[SA]] to i39
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i39 [[SA_EXT]], [[I]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = trunc i39 [[SUM]] to i16
  // CHECK-NEXT: store i16 [[RES]], i16* %sa, align 2
  sa = sa + i;

  // CHECK:      [[SA:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[UI:%[0-9]+]] = load i32, i32* %ui, align 4
  // CHECK-NEXT: [[SA_EXT:%[a-z0-9]+]] = sext i16 [[SA]] to i40
  // CHECK-NEXT: [[UI_EXT:%[a-z0-9]+]] = zext i32 [[UI]] to i40
  // CHECK-NEXT: [[UI:%[a-z0-9]+]] = shl i40 [[UI_EXT]], 7
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i40 [[SA_EXT]], [[UI]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = trunc i40 [[SUM]] to i16
  // CHECK-NEXT: store i16 [[RES]], i16* %sa, align 2
  sa = sa + ui;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // SIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i40
  // SIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i40
  // SIGNED-NEXT: [[I:%[a-z0-9]+]] = shl i40 [[I_EXT]], 8
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = add i40 [[USA_EXT]], [[I]]
  // SIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i40 [[SUM]] to i16
  // UNSIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i39
  // UNSIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // UNSIGNED-NEXT: [[I:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = add i39 [[USA_EXT]], [[I]]
  // UNSIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i39 [[SUM]] to i16
  // CHECK-NEXT: store i16 [[RES]], i16* %usa, align 2
  usa = usa + i;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // SIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i40
  // SIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // SIGNED-NEXT: [[I:%[a-z0-9]+]] = shl i40 [[I_EXT]], 8
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = add i40 [[USA_EXT]], [[I]]
  // SIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i40 [[SUM]] to i16
  // UNSIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i39
  // UNSIGNED-NEXT: [[I_EXT:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // UNSIGNED-NEXT: [[I:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = add i39 [[USA_EXT]], [[I]]
  // UNSIGNED-NEXT: [[RES:%[a-z0-9]+]] = trunc i39 [[SUM]] to i16
  // CHECK-NEXT: store i16 [[RES]], i16* %usa, align 2
  usa = usa + ui;

  // CHECK:      [[LF:%[0-9]+]] = load i32, i32* %lf, align 4
  // CHECK-NEXT: [[UI:%[0-9]+]] = load i32, i32* %ui, align 4
  // CHECK-NEXT: [[LF_EXT:%[a-z0-9]+]] = sext i32 [[LF]] to i64
  // CHECK-NEXT: [[UI_EXT:%[a-z0-9]+]] = zext i32 [[UI]] to i64
  // CHECK-NEXT: [[UI:%[a-z0-9]+]] = shl i64 [[UI_EXT]], 31
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i64 [[LF_EXT]], [[UI]]
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = trunc i64 [[SUM]] to i32
  // CHECK-NEXT: store i32 [[RES]], i32* %lf, align 4
  lf = lf + ui;

  // CHECK:      [[ACCUM:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[BOOL:%[0-9]+]] = load i8, i8* %b, align 1
  // CHECK-NEXT: [[AS_BOOL:%[a-z0-9]+]] = trunc i8 [[BOOL]] to i1
  // CHECK-NEXT: [[BOOL_EXT:%[a-z0-9]+]] = zext i1 [[AS_BOOL]] to i32
  // CHECK-NEXT: [[ACCUM_EXT:%[a-z0-9]+]] = sext i32 [[ACCUM]] to i47
  // CHECK-NEXT: [[BOOL:%[a-z0-9]+]] = sext i32 [[BOOL_EXT]] to i47
  // CHECK-NEXT: [[BOOL_EXT:%[a-z0-9]+]] = shl i47 [[BOOL]], 15
  // CHECK-NEXT: [[SUM:%[0-9]+]] = add i47 [[ACCUM_EXT]], [[BOOL_EXT]]
  // CHECK-NEXT: [[RESULT:%[a-z0-9]+]] = trunc i47 [[SUM]] to i32
  // CHECK-NEXT: store i32 [[RESULT]], i32* %a, align 4
  a = a + b;
}

void SaturatedAddition() {
  // CHECK-LABEL: SaturatedAddition
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
  // CHECK-NEXT: [[SUM:%[0-9]+]] = call i16 @llvm.sadd.sat.i16(i16 [[SA]], i16
  // [[SA_SAT]])
  // CHECK-NEXT: store i16 [[SUM]], i16* %sa_sat, align 2
  sa_sat = sa + sa_sat;

  // CHECK:      [[USA:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[USA_SAT:%[0-9]+]] = load i16, i16* %usa_sat, align 2
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i16 @llvm.uadd.sat.i16(i16 [[USA]], i16 [[USA_SAT]])
  // SIGNED-NEXT: store i16 [[SUM]], i16* %usa_sat, align 2
  // UNSIGNED-NEXT: [[USA_TRUNC:%[a-z0-9]+]] = trunc i16 [[USA]] to i15
  // UNSIGNED-NEXT: [[USA_SAT_TRUNC:%[a-z0-9]+]] = trunc i16 [[USA_SAT]] to i15
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i15 @llvm.uadd.sat.i15(i15 [[USA_TRUNC]], i15 [[USA_SAT_TRUNC]])
  // UNSIGNED-NEXT: [[SUM_EXT:%[a-z0-9]+]] = zext i15 [[SUM]] to i16
  // UNSIGNED-NEXT: store i16 [[SUM_EXT]], i16* %usa_sat, align 2
  usa_sat = usa + usa_sat;

  // CHECK:      [[UA:%[0-9]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[USA:%[0-9]+]] = load i16, i16* %usa_sat, align 2
  // SIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i32
  // SIGNED-NEXT: [[USA:%[a-z0-9]+]] = shl i32 [[USA_EXT]], 8
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i32 @llvm.uadd.sat.i32(i32 [[UA]], i32 [[USA]])
  // SIGNED-NEXT: store i32 [[SUM]], i32* %ua_sat, align 4
  // UNSIGNED-NEXT: [[UA_TRUNC:%[a-z0-9]+]] = trunc i32 [[UA]] to i31
  // UNSIGNED-NEXT: [[USA_EXT:%[a-z0-9]+]] = zext i16 [[USA]] to i31
  // UNSIGNED-NEXT: [[USA:%[a-z0-9]+]] = shl i31 [[USA_EXT]], 8
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i31 @llvm.uadd.sat.i31(i31 [[UA_TRUNC]], i31 [[USA]])
  // UNSIGNED-NEXT: [[SUM_EXT:%[a-z0-9]+]] = zext i31 [[SUM]] to i32
  // UNSIGNED-NEXT: store i32 [[SUM_EXT]], i32* %ua_sat, align 4
  ua_sat = ua + usa_sat;

  // CHECK:      [[SA_SAT:%[0-9]+]] = load i16, i16* %sa_sat, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[SA_SAT_EXT:%[a-z0-9]+]] = sext i16 [[SA_SAT]] to i39
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = shl i39 [[I_EXT]], 7
  // CHECK-NEXT: [[SUM:%[0-9]+]] = call i39 @llvm.sadd.sat.i39(i39 [[SA_SAT_EXT]], i39 [[I]])
  // CHECK-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i39 [[SUM]], 32767
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[SUM]]
  // CHECK-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i39 [[RES]], -32768
  // CHECK-NEXT: [[RES2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i39 -32768, i39 [[RES]]
  // CHECK-NEXT: [[RES3:%[a-z0-9]+]] = trunc i39 [[RES2]] to i16
  // CHECK-NEXT: store i16 [[RES3]], i16* %sa_sat, align 2
  sa_sat = sa_sat + i;

  // CHECK:      [[SA_SAT:%[0-9]+]] = load i16, i16* %sa_sat, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // CHECK-NEXT: [[SA_SAT_EXT:%[a-z0-9]+]] = sext i16 [[SA_SAT]] to i40
  // CHECK-NEXT: [[I_EXT:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = shl i40 [[I_EXT]], 7
  // CHECK-NEXT: [[SUM:%[0-9]+]] = call i40 @llvm.sadd.sat.i40(i40 [[SA_SAT_EXT]], i40 [[I]])
  // CHECK-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i40 [[SUM]], 32767
  // CHECK-NEXT: [[RES:%[a-z0-9]+]] = select i1 [[USE_MAX]], i40 32767, i40 [[SUM]]
  // CHECK-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i40 [[RES]], -32768
  // CHECK-NEXT: [[RES2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i40 -32768, i40 [[RES]]
  // CHECK-NEXT: [[RES3:%[a-z0-9]+]] = trunc i40 [[RES2]] to i16
  // CHECK-NEXT: store i16 [[RES3]], i16* %sa_sat, align 2
  sa_sat = sa_sat + ui;

  // CHECK:      [[UF_SAT:%[0-9]+]] = load i16, i16* %uf_sat, align 2
  // CHECK-NEXT: [[UF_SAT2:%[0-9]+]] = load i16, i16* %uf_sat, align 2
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i16 @llvm.uadd.sat.i16(i16 [[UF_SAT]], i16 [[UF_SAT2]])
  // SIGNED-NEXT: store i16 [[SUM]], i16* %uf_sat, align 2
  // UNSIGNED-NEXT: [[UF_SAT_TRUNC:%[a-z0-9]+]] = trunc i16 [[UF_SAT]] to i15
  // UNSIGNED-NEXT: [[UF_SAT_TRUNC2:%[a-z0-9]+]] = trunc i16 [[UF_SAT2]] to i15
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i15 @llvm.uadd.sat.i15(i15 [[UF_SAT_TRUNC]], i15 [[UF_SAT_TRUNC2]])
  // UNSIGNED-NEXT: [[SUM_EXT:%[a-z0-9]+]] = zext i15 [[SUM]] to i16
  // UNSIGNED-NEXT: store i16 [[SUM_EXT]], i16* %uf_sat, align 2
  uf_sat = uf_sat + uf_sat;

  // CHECK:      [[USA_SAT:%[0-9]+]] = load i16, i16* %usa_sat, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // SIGNED-NEXT: [[USA_SAT_RESIZE:%[a-z0-9]+]] = zext i16 [[USA_SAT]] to i40
  // SIGNED-NEXT: [[I_RESIZE:%[a-z0-9]+]] = sext i32 [[I]] to i40
  // SIGNED-NEXT: [[I_UPSCALE:%[a-z0-9]+]] = shl i40 [[I_RESIZE]], 8
  // SIGNED-NEXT: [[SUM:%[0-9]+]] = call i40 @llvm.uadd.sat.i40(i40 [[USA_SAT_RESIZE]], i40 [[I_UPSCALE]])
  // SIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i40 [[SUM]], 65535
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = select i1 [[USE_MAX]], i40 65535, i40 [[SUM]]
  // SIGNED-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i40 [[RESULT]], 0
  // SIGNED-NEXT: [[RESULT2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i40 0, i40 [[RESULT]]
  // SIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = trunc i40 [[RESULT2]] to i16
  // UNSIGNED-NEXT: [[USA_SAT_RESIZE:%[a-z0-9]+]] = zext i16 [[USA_SAT]] to i39
  // UNSIGNED-NEXT: [[I_RESIZE:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // UNSIGNED-NEXT: [[I_UPSCALE:%[a-z0-9]+]] = shl i39 [[I_RESIZE]], 7
  // UNSIGNED-NEXT: [[SUM:%[0-9]+]] = call i39 @llvm.uadd.sat.i39(i39 [[USA_SAT_RESIZE]], i39 [[I_UPSCALE]])
  // UNSIGNED-NEXT: [[USE_MAX:%[0-9]+]] = icmp sgt i39 [[SUM]], 32767
  // UNSIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = select i1 [[USE_MAX]], i39 32767, i39 [[SUM]]
  // UNSIGNED-NEXT: [[USE_MIN:%[0-9]+]] = icmp slt i39 [[RESULT]], 0
  // UNSIGNED-NEXT: [[RESULT2:%[a-z0-9]+]] = select i1 [[USE_MIN]], i39 0, i39 [[RESULT]]
  // UNSIGNED-NEXT: [[RESULT:%[a-z0-9]+]] = trunc i39 [[RESULT2]] to i16
  // CHECK-NEXT: store i16 [[RESULT]], i16* %usa_sat, align 2
  usa_sat = usa_sat + i;
}
