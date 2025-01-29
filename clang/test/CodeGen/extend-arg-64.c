// RUN: %clang_cc1 -DD128 -triple x86_64-apple-darwin -fextend-arguments=64  \
// RUN:            -Wno-strict-prototypes %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECKEXT

// When the option isn't selected, no effect
// RUN: %clang_cc1 -DD128 -triple x86_64-apple-darwin  \
// RUN:            -Wno-strict-prototypes %s -emit-llvm -o - | FileCheck %s \
// RUN:    --implicit-check-not "ext {{.*}}to i64"

// The option isn't supported on x86, no effect
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -fextend-arguments=64 \
// RUN:            -Wno-strict-prototypes %s -emit-llvm -o - | FileCheck %s \
// RUN:    --implicit-check-not "ext {{.*}}to i64"

// The option isn't supported on ppc, no effect
// RUN: %clang_cc1 -triple ppc64le -fextend-arguments=64 \
// RUN:            -Wno-strict-prototypes %s -emit-llvm -o - | FileCheck %s \
// RUN:    --implicit-check-not "ext {{.*}}to i64"

// The option isn't supported on ppc, no effect
// RUN: %clang_cc1 -DD128 -triple powerpc64-ibm-aix-xcoff -fextend-arguments=64 \
// RUN:            -Wno-strict-prototypes %s -emit-llvm -o - | FileCheck %s \
// RUN:    --implicit-check-not "ext {{.*}}to i64"


int vararg(int, ...);
void knr();

unsigned int u32;
int s32;
unsigned short u16;
short s16;
unsigned char u8;
signed char s8;
long long ll;
_BitInt(23) ei23;
float ff;
double dd;
#ifdef D128
__int128 i128;
#endif

int test(void) {
  // CHECK: define{{.*}} i32 @test{{.*}}

  // CHECKEXT:  [[TAG_u32:%.*]] = load i32, ptr @u32{{.*}}
  // CHECKEXT: [[CONV_u32:%.*]] = zext i32 [[TAG_u32]] to i64

  // CHECKEXT:  [[TAG_s32:%.*]] = load i32, ptr @s32
  // CHECKEXT: [[CONV_s32:%.*]] = sext i32 [[TAG_s32]] to i64

  // CHECKEXT:  [[TAG_u16:%.*]] = load i16, ptr @u16
  // CHECKEXT: [[CONV_u16:%.*]] = zext i16 [[TAG_u16]] to i64

  // CHECKEXT:  [[TAG_s16:%.*]] = load i16, ptr @s16
  // CHECKEXT: [[CONV_s16:%.*]] = sext i16 [[TAG_s16]] to i64

  // CHECKEXT:  [[TAG_u8:%.*]] = load i8, ptr @u8
  // CHECKEXT: [[CONV_u8:%.*]] = zext i8 [[TAG_u8]] to i64

  // CHECKEXT:  [[TAG_s8:%.*]] = load i8, ptr @s8
  // CHECKEXT: [[CONV_s8:%.*]] = sext i8 [[TAG_s8]] to i64
  // CHECKEXT: call{{.*}} @vararg(i32 noundef %0, i64 noundef [[CONV_u32]], i64 noundef [[CONV_s32]], i64 noundef [[CONV_u16]], i64 noundef [[CONV_s16]], i64 noundef [[CONV_u8]], i64 noundef [[CONV_s8]]

  int sum = 0;
  sum = vararg(sum, u32, s32, u16, s16, u8, s8);
  knr(ll);
  // CHECKEXT: load i64, ptr @ll
  // CHECKEXT-NEXT: call void (i64, ...) @knr

  knr(ei23);
  // CHECKEXT: load i32, ptr @ei23
  // CHECKEXT: trunc i32
  // CHECKEXT-NEXT: call void (i23, ...) @knr

  knr(ff);
  // CHECKEXT: load float
  // CHECKEXT-NEXT: fpext float {{.*}} to double
  // CHECKEXT-NEXT: call{{.*}} void (double, ...) @knr

  knr(dd);
  // CHECKEXT: load double
  // CHECKEXT-NEXT: call{{.*}} void (double, ...) @knr

#ifdef D128
  knr(i128);
  // CHECKEXT: load i128
  // CHECKEXT: call{{.*}} void (i64, i64, ...) @knr
#endif

  knr(u32, s32, u16, s16, u8, s8);
  // CHECKEXT:  [[TAg_u32:%.*]] = load i32, ptr @u32{{.*}}
  // CHECKEXT: [[CONv_u32:%.*]] = zext i32 [[TAg_u32]] to i64

  // CHECKEXT:  [[TAg_s32:%.*]] = load i32, ptr @s32
  // CHECKEXT: [[CONv_s32:%.*]] = sext i32 [[TAg_s32]] to i64

  // CHECKEXT:  [[TAg_u16:%.*]] = load i16, ptr @u16
  // CHECKEXT: [[CONv_u16:%.*]] = zext i16 [[TAg_u16]] to i64

  // CHECKEXT:  [[TAg_s16:%.*]] = load i16, ptr @s16
  // CHECKEXT: [[CONv_s16:%.*]] = sext i16 [[TAg_s16]] to i64

  // CHECKEXT:  [[TAg_u8:%.*]] = load i8, ptr @u8
  // CHECKEXT: [[CONv_u8:%.*]] = zext i8 [[TAg_u8]] to i64

  // CHECKEXT:  [[TAg_s8:%.*]] = load i8, ptr @s8
  // CHECKEXT: [[CONv_s8:%.*]] = sext i8 [[TAg_s8]] to i64
  // CHECKEXT: call{{.*}} void (i64, i64, i64, i64, i64, i64, ...) @knr
  return sum;
}
