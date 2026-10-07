// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

typedef int int4 __attribute__((ext_vector_type(4)));
typedef short short4 __attribute__((ext_vector_type(4)));

short signed_to_narrower_signed(int x) {
  // CHECK-LABEL: define{{.*}}signext i16 @signed_to_narrower_signed
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: [[LOW_RESULT:%.+]] = call i32 @llvm.smax.i32(i32 [[X]], i32 -32768)
  // CHECK-NEXT: [[HIGH_RESULT:%.+]] = call i32 @llvm.smin.i32(i32 [[LOW_RESULT]], i32 32767)
  // CHECK-NEXT: trunc i32 [[HIGH_RESULT]] to i16
  return __builtin_elementwise_saturating_cast(x, short);
}

int signed_to_wider_signed(short x) {
  // CHECK-LABEL: define{{.*}}i32 @signed_to_wider_signed
  // CHECK: [[X:%.+]] = load i16, ptr %x.addr
  // CHECK-NEXT: sext i16 [[X]] to i32
  return __builtin_elementwise_saturating_cast(x, int);
}

int signed_to_equal_signed(int x) {
  // CHECK-LABEL: define{{.*}}i32 @signed_to_equal_signed
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: ret i32 [[X]]
  return __builtin_elementwise_saturating_cast(x, int);
}

unsigned short unsigned_to_narrower_unsigned(unsigned x) {
  // CHECK-LABEL: define{{.*}}zeroext i16 @unsigned_to_narrower_unsigned
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: [[HIGH_RESULT:%.+]] = call i32 @llvm.umin.i32(i32 [[X]], i32 65535)
  // CHECK-NEXT: trunc i32 [[HIGH_RESULT]] to i16
  return __builtin_elementwise_saturating_cast(x, unsigned short);
}

unsigned int unsigned_to_wider_unsigned(unsigned short x) {
  // CHECK-LABEL: define{{.*}}i32 @unsigned_to_wider_unsigned
  // CHECK: [[X:%.+]] = load i16, ptr %x.addr
  // CHECK-NEXT: zext i16 [[X]] to i32
  return __builtin_elementwise_saturating_cast(x, unsigned int);
}

unsigned int unsigned_to_equal_unsigned(unsigned x) {
  // CHECK-LABEL: define{{.*}}i32 @unsigned_to_equal_unsigned
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: ret i32 [[X]]
  return __builtin_elementwise_saturating_cast(x, unsigned int);
}

unsigned short signed_to_narrower_unsigned(int x) {
  // CHECK-LABEL: define{{.*}}zeroext i16 @signed_to_narrower_unsigned
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: [[LOW_RESULT:%.+]] = call i32 @llvm.smax.i32(i32 [[X]], i32 0)
  // CHECK-NEXT: [[HIGH_RESULT:%.+]] = call i32 @llvm.smin.i32(i32 [[LOW_RESULT]], i32 65535)
  // CHECK-NEXT: trunc i32 [[HIGH_RESULT]] to i16
  return __builtin_elementwise_saturating_cast(x, unsigned short);
}

unsigned int signed_to_wider_unsigned(short x) {
  // CHECK-LABEL: define{{.*}}i32 @signed_to_wider_unsigned
  // CHECK: [[X:%.+]] = load i16, ptr %x.addr
  // CHECK-NEXT: [[LOW_RESULT:%.+]] = call i16 @llvm.smax.i16(i16 [[X]], i16 0)
  // CHECK-NEXT: zext i16 [[LOW_RESULT]] to i32
  return __builtin_elementwise_saturating_cast(x, unsigned int);
}

unsigned int signed_to_equal_unsigned(int x) {
  // CHECK-LABEL: define{{.*}}i32 @signed_to_equal_unsigned
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: [[LOW_RESULT:%.+]] = call i32 @llvm.smax.i32(i32 [[X]], i32 0)
  // CHECK-NEXT: ret i32 [[LOW_RESULT]]
  return __builtin_elementwise_saturating_cast(x, unsigned int);
}

short unsigned_to_narrower_signed(unsigned x) {
  // CHECK-LABEL: define{{.*}}signext i16 @unsigned_to_narrower_signed
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: [[HIGH_RESULT:%.+]] = call i32 @llvm.umin.i32(i32 [[X]], i32 32767)
  // CHECK-NEXT: trunc i32 [[HIGH_RESULT]] to i16
  return __builtin_elementwise_saturating_cast(x, short);
}

int unsigned_to_wider_signed(unsigned short x) {
  // CHECK-LABEL: define{{.*}}i32 @unsigned_to_wider_signed
  // CHECK: [[X:%.+]] = load i16, ptr %x.addr
  // CHECK-NEXT: zext i16 [[X]] to i32
  return __builtin_elementwise_saturating_cast(x, int);
}

int unsigned_to_equal_signed(unsigned x) {
  // CHECK-LABEL: define{{.*}}i32 @unsigned_to_equal_signed
  // CHECK: [[X:%.+]] = load i32, ptr %x.addr
  // CHECK-NEXT: [[HIGH_RESULT:%.+]] = call i32 @llvm.umin.i32(i32 [[X]], i32 2147483647)
  // CHECK-NEXT: ret i32 [[HIGH_RESULT]]
  return __builtin_elementwise_saturating_cast(x, int);
}

short4 vector_narrow(int4 x) {
  // CHECK-LABEL: define{{.*}}@vector_narrow
  // CHECK: [[X:%.+]] = load <4 x i32>, ptr %x.addr
  // CHECK-NEXT: [[LOW_RESULT:%.+]] = call <4 x i32> @llvm.smax.v4i32(<4 x i32> [[X]], <4 x i32> splat (i32 -32768))
  // CHECK-NEXT: [[HIGH_RESULT:%.+]] = call <4 x i32> @llvm.smin.v4i32(<4 x i32> [[LOW_RESULT]], <4 x i32> splat (i32 32767))
  // CHECK-NEXT: trunc <4 x i32> [[HIGH_RESULT]] to <4 x i16>
  return __builtin_elementwise_saturating_cast(x, short);
}
