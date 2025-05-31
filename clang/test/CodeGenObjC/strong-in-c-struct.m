// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -emit-llvm -o - -DUSESTRUCT %s | FileCheck %s

// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -include-pch %t -emit-llvm -o - -DUSESTRUCT %s | FileCheck %s

#ifndef HEADER
#define HEADER

typedef void (^BlockTy)(void);

typedef struct {
  int a[4];
} Trivial;

typedef struct {
  Trivial f0;
  id f1;
} Strong;

typedef struct {
  int i;
  id f1;
} StrongSmall;

typedef struct {
  Strong f0;
  id f1;
  double d;
} StrongOuter;

typedef struct {
  id f0;
  Strong f1;
} StrongOuter2;

typedef struct {
  int f0;
  volatile id f1;
} StrongVolatile;

typedef struct {
  BlockTy f0;
} StrongBlock;

typedef struct {
  int i;
  id f0[2][2];
} IDArray;

typedef struct {
  double d;
  Strong f0[2][2];
} StructArray;

typedef struct {
  id f0;
  int i : 9;
} Bitfield0;

typedef struct {
  char c;
  int i0 : 2;
  int i1 : 4;
  id f0;
  int i2 : 31;
  int i3 : 1;
  id f1;
  int : 0;
  int a[3];
  id f2;
  double d;
  int i4 : 1;
  volatile int i5 : 2;
  volatile char i6;
} Bitfield1;

typedef struct {
  id x;
  volatile int a[16];
} VolatileArray ;

typedef struct {
  _Bool f0[2];
  VolatileArray f1;
} StructWithBool;

#endif

#ifdef USESTRUCT

StrongSmall getStrongSmall(void);
StrongOuter getStrongOuter(void);
StrongOuter2 getStrongOuter2(void);
void calleeStrongSmall(StrongSmall);
void func(Strong *);

@interface C
- (StrongSmall)getStrongSmall;
- (void)m:(StrongSmall)s;
+ (StrongSmall)getStrongSmallClass;
@end

id g0;
StrongSmall g1, g2;

// CHECK: %[[STRUCT_STRONGSMALL:.*]] = type { i32, ptr }
// CHECK: %[[STRUCT_STRONGOUTER:.*]] = type { %[[STRUCT_STRONG:.*]], ptr, double }
// CHECK: %[[STRUCT_STRONG]] = type { %[[STRUCT_TRIVIAL:.*]], ptr }
// CHECK: %[[STRUCT_TRIVIAL]] = type { [4 x i32] }
// CHECK: %[[STRUCT_BLOCK_BYREF_T:.*]] = type { ptr, ptr, i32, i32, ptr, ptr, ptr, %[[STRUCT_STRONGOUTER]] }
// CHECK: %[[STRUCT_STRONGBLOCK:.*]] = type { ptr }
// CHECK: %[[STRUCT_BITFIELD1:.*]] = type { i8, i8, ptr, i32, ptr, [3 x i32], ptr, double, i8, i8 }

// CHECK: define{{.*}} void @test_constructor_destructor_StrongOuter()
// CHECK: %[[T:.*]] = alloca %[[STRUCT_STRONGOUTER]], align 8
// CHECK: call void @__default_constructor_8_S_s16_s24(ptr %[[T]])
// CHECK: call void @__destructor_8_S_s16_s24(ptr %[[T]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__default_constructor_8_S_s16_s24(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: call void @__default_constructor_8_s16(ptr %[[V0]])
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 %[[V2]], i8 0, i64 8, i1 false)
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__default_constructor_8_s16(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 16
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 %[[V2]], i8 0, i64 8, i1 false)
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_S_s16_s24(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: call void @__destructor_8_s16(ptr %[[V0]])
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V2]], ptr null)
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_s16(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 16
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V2]], ptr null)
// CHECK: ret void

void test_constructor_destructor_StrongOuter(void) {
  StrongOuter t;
}

// CHECK: define{{.*}} void @test_copy_constructor_StrongOuter(ptr noundef %[[S:.*]])
// CHECK: %[[S_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[T:.*]] = alloca %[[STRUCT_STRONGOUTER]], align 8
// CHECK: store ptr %[[S]], ptr %[[S_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[S_ADDR]], align 8
// CHECK: call void @__copy_constructor_8_8_S_t0w16_s16_s24_t32w8(ptr %[[T]], ptr %[[V0]])
// CHECK: call void @__destructor_8_S_s16_s24(ptr %[[T]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_S_t0w16_s16_s24_t32w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(ptr %[[V0]], ptr %[[V1]])
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 24
// CHECK: %[[V8:.*]] = load ptr, ptr %[[V6]], align 8
// CHECK: %[[V9:.*]] = call ptr @llvm.objc.retain(ptr %[[V8]])
// CHECK: store ptr %[[V9]], ptr %[[V3]], align 8
// CHECK: %[[V11:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 32
// CHECK: %[[V14:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 32
// CHECK: %[[V18:.*]] = load i64, ptr %[[V14]], align 8
// CHECK: store i64 %[[V18]], ptr %[[V11]], align 8
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w16_s16(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[V0]], ptr align 8 %[[V1]], i64 16, i1 false)
// CHECK: %[[V5:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 16
// CHECK: %[[V8:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 16
// CHECK: %[[V10:.*]] = load ptr, ptr %[[V8]], align 8
// CHECK: %[[V11:.*]] = call ptr @llvm.objc.retain(ptr %[[V10]])
// CHECK: store ptr %[[V11]], ptr %[[V5]], align 8
// CHECK: ret void

void test_copy_constructor_StrongOuter(StrongOuter *s) {
  StrongOuter t = *s;
}

/// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_S_t0w16_s16_s24_t32w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 24
// CHECK: %[[V8:.*]] = load ptr, ptr %[[V6]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V3]], ptr %[[V8]])

void test_copy_assignment_StrongOuter(StrongOuter *d, StrongOuter *s) {
  *d = *s;
}

// CHECK: define{{.*}} void @test_move_constructor_StrongOuter()
// CHECK: %[[T1:.*]] = getelementptr inbounds nuw %[[STRUCT_BLOCK_BYREF_T]], ptr %{{.*}}, i32 0, i32 7
// CHECK: call void @__default_constructor_8_S_s16_s24(ptr %[[T1]])
// CHECK: %[[T2:.*]] = getelementptr inbounds nuw %[[STRUCT_BLOCK_BYREF_T]], ptr %{{.*}}, i32 0, i32 7
// CHECK: call void @__destructor_8_S_s16_s24(ptr %[[T2]])

// CHECK: define internal void @__Block_byref_object_copy_(ptr noundef %0, ptr noundef %1)
// CHECK: call void @__move_constructor_8_8_S_t0w16_s16_s24_t32w8(

// CHECK: define linkonce_odr hidden void @__move_constructor_8_8_S_t0w16_s16_s24_t32w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: call void @__move_constructor_8_8_t0w16_s16(ptr %[[V0]], ptr %[[V1]])
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 24
// CHECK: %[[V8:.*]] = load ptr, ptr %[[V6]], align 8
// CHECK: store ptr null, ptr %[[V6]], align 8
// CHECK: store ptr %[[V8]], ptr %[[V3]], align 8

// CHECK: define internal void @__Block_byref_object_dispose_(ptr noundef %0)
// CHECK: call void @__destructor_8_S_s16_s24(

void test_move_constructor_StrongOuter(void) {
  __block StrongOuter t;
  BlockTy b = ^{ (void)t; };
}

// CHECK: define linkonce_odr hidden void @__move_assignment_8_8_S_t0w16_s16_s24_t32w8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: call void @__move_assignment_8_8_t0w16_s16(ptr %[[V0]], ptr %[[V1]])
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 24
// CHECK: %[[V8:.*]] = load ptr, ptr %[[V6]], align 8
// CHECK: store ptr null, ptr %[[V6]], align 8
// CHECK: %[[V9:.*]] = load ptr, ptr %[[V3]], align 8
// CHECK: store ptr %[[V8]], ptr %[[V3]], align 8
// CHECK: call void @llvm.objc.release(ptr %[[V9]])

void test_move_assignment_StrongOuter(StrongOuter *p) {
  *p = getStrongOuter();
}

// CHECK: define linkonce_odr hidden void @__default_constructor_8_s0_S_s24(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 %[[V0]], i8 0, i64 8, i1 false)
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: call void @__default_constructor_8_s16(ptr %[[V3]])

// CHECK: define linkonce_odr hidden void @__destructor_8_s0_S_s24(ptr noundef %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V0]], ptr null)
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: call void @__destructor_8_s16(ptr %[[V2]])

void test_constructor_destructor_StrongOuter2(void) {
  StrongOuter2 t;
}

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_s0_S_t8w16_s24(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load ptr, ptr %[[V1]], align 8
// CHECK: %[[V3:.*]] = call ptr @llvm.objc.retain(ptr %[[V2]])
// CHECK: store ptr %[[V3]], ptr %[[V0]], align 8
// CHECK: %[[V5:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: %[[V8:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(ptr %[[V5]], ptr %[[V8]])

void test_copy_constructor_StrongOuter2(StrongOuter2 *s) {
  StrongOuter2 t = *s;
}

// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_s0_S_t8w16_s24(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load ptr, ptr %[[V1]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V0]], ptr %[[V2]])
// CHECK: %[[V4:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: %[[V7:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// CHECK: call void @__copy_assignment_8_8_t0w16_s16(ptr %[[V4]], ptr %[[V7]])

void test_copy_assignment_StrongOuter2(StrongOuter2 *d, StrongOuter2 *s) {
  *d = *s;
}

// CHECK: define linkonce_odr hidden void @__move_constructor_8_8_s0_S_t8w16_s24(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load ptr, ptr %[[V1]], align 8
// CHECK: store ptr null, ptr %[[V1]], align 8
// CHECK: store ptr %[[V2]], ptr %[[V0]], align 8
// CHECK: %[[V4:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: %[[V7:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// CHECK: call void @__move_constructor_8_8_t0w16_s16(ptr %[[V4]], ptr %[[V7]])

void test_move_constructor_StrongOuter2(void) {
  __block StrongOuter2 t;
  BlockTy b = ^{ (void)t; };
}

// CHECK: define linkonce_odr hidden void @__move_assignment_8_8_s0_S_t8w16_s24(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load ptr, ptr %[[V1]], align 8
// CHECK: store ptr null, ptr %[[V1]], align 8
// CHECK: %[[V3:.*]] = load ptr, ptr %[[V0]], align 8
// CHECK: store ptr %[[V2]], ptr %[[V0]], align 8
// CHECK: call void @llvm.objc.release(ptr %[[V3]])
// CHECK: %[[V5:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: %[[V8:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// CHECK: call void @__move_assignment_8_8_t0w16_s16(ptr %[[V5]], ptr %[[V8]])

void test_move_assignment_StrongOuter2(StrongOuter2 *p) {
  *p = getStrongOuter2();
}

// CHECK: define{{.*}} void @test_parameter_StrongSmall([2 x i64] %[[A_COERCE:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: store [2 x i64] %[[A_COERCE]], ptr %[[A]], align 8
// CHECK: call void @__destructor_8_s8(ptr %[[A]])
// CHECK: ret void

void test_parameter_StrongSmall(StrongSmall a) {
}

// CHECK: define{{.*}} void @test_argument_StrongSmall([2 x i64] %[[A_COERCE:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[TEMP_LVALUE:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: store [2 x i64] %[[A_COERCE]], ptr %[[A]], align 8
// CHECK: call void @__copy_constructor_8_8_t0w4_s8(ptr %[[TEMP_LVALUE]], ptr %[[A]])
// CHECK: %[[V4:.*]] = load [2 x i64], ptr %[[TEMP_LVALUE]], align 8
// CHECK: call void @calleeStrongSmall([2 x i64] %[[V4]])
// CHECK: call void @__destructor_8_s8(ptr %[[A]])
// CHECK: ret void

void test_argument_StrongSmall(StrongSmall a) {
  calleeStrongSmall(a);
}

// CHECK: define{{.*}} [2 x i64] @test_return_StrongSmall([2 x i64] %[[A_COERCE:.*]])
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[A:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: store [2 x i64] %[[A_COERCE]], ptr %[[A]], align 8
// CHECK: call void @__copy_constructor_8_8_t0w4_s8(ptr %[[RETVAL]], ptr %[[A]])
// CHECK: call void @__destructor_8_s8(ptr %[[A]])
// CHECK: %[[V5:.*]] = load [2 x i64], ptr %[[RETVAL]], align 8
// CHECK: ret [2 x i64] %[[V5]]

StrongSmall test_return_StrongSmall(StrongSmall a) {
  return a;
}

// CHECK: define{{.*}} void @test_destructor_ignored_result()
// CHECK: %[[COERCE:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[CALL:.*]] = call [2 x i64] @getStrongSmall()
// CHECK: store [2 x i64] %[[CALL]], ptr %[[COERCE]], align 8
// CHECK: call void @__destructor_8_s8(ptr %[[COERCE]])
// CHECK: ret void

void test_destructor_ignored_result(void) {
  getStrongSmall();
}

// CHECK: define{{.*}} void @test_destructor_ignored_result2(ptr noundef %[[C:.*]])
// CHECK: %[[TMP:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[CALL:.*]] = call [2 x i64]{{.*}}@objc_msgSend
// CHECK: store [2 x i64] %[[CALL]], ptr %[[TMP]], align 8
// CHECK: call void @__destructor_8_s8(ptr %[[TMP]])

void test_destructor_ignored_result2(C *c) {
  [c getStrongSmall];
}

// CHECK: define{{.*}} void @test_copy_constructor_StrongBlock(
// CHECK: call void @__copy_constructor_8_8_sb0(
// CHECK: call void @__destructor_8_sb0(
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_sb0(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load ptr, ptr %[[V1]], align 8
// CHECK: %[[V3:.*]] = call ptr @llvm.objc.retainBlock(ptr %[[V2]])
// CHECK: store ptr %[[V3]], ptr %[[V0]], align 8
// CHECK: ret void

void test_copy_constructor_StrongBlock(StrongBlock *s) {
  StrongBlock t = *s;
}

// CHECK: define{{.*}} void @test_copy_assignment_StrongBlock(ptr noundef %[[D:.*]], ptr noundef %[[S:.*]])
// CHECK: call void @__copy_assignment_8_8_sb0(

// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_sb0(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load ptr, ptr %[[V1]], align 8
// CHECK: %[[V3:.*]] = call ptr @llvm.objc.retainBlock(ptr %[[V2]])
// CHECK: %[[V4:.*]] = load ptr, ptr %[[V0]], align 8
// CHECK: store ptr %[[V3]], ptr %[[V0]], align 8
// CHECK: call void @llvm.objc.release(ptr %[[V4]])
// CHECK: ret void

void test_copy_assignment_StrongBlock(StrongBlock *d, StrongBlock *s) {
  *d = *s;
}

// CHECK-LABEL: define{{.*}} void @test_copy_assignment_StructWithBool(
// CHECK: call void @__copy_assignment_8_8_AB0s1n2_tv0w8_AE_S_sv8_AB16s4n16_tv128w32_AE(

// CHECK-LABEL: define linkonce_odr hidden void @__copy_assignment_8_8_AB0s1n2_tv0w8_AE_S_sv8_AB16s4n16_tv128w32_AE(
// CHECK: %[[ADDR_CUR:.*]] = phi ptr
// CHECK: %[[ADDR_CUR1:.*]] = phi ptr

// CHECK: %[[V6:.*]] = load volatile i8, ptr %[[ADDR_CUR1]], align 1
// CHECK: %[[TOBOOL:.*]] = trunc i8 %[[V6]] to i1
// CHECK: %[[FROMBOOL:.*]] = zext i1 %[[TOBOOL]] to i8
// CHECK: store volatile i8 %[[FROMBOOL]], ptr %[[ADDR_CUR]], align 1

void test_copy_assignment_StructWithBool(StructWithBool *d, StructWithBool *s) {
  *d = *s;
}

// CHECK: define{{.*}} void @test_copy_constructor_StrongVolatile0(
// CHECK: call void @__copy_constructor_8_8_t0w4_sv8(
// CHECK-NOT: call
// CHECK: call void @__destructor_8_sv8(
// CHECK-NOT: call

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_sv8(
// CHECK: %[[V8:.*]] = load volatile ptr, ptr %{{.*}}, align 8
// CHECK: %[[V9:.*]] = call ptr @llvm.objc.retain(ptr %[[V8]])
// CHECK: store volatile ptr %[[V9]], ptr %{{.*}}, align 8

void test_copy_constructor_StrongVolatile0(StrongVolatile *s) {
  StrongVolatile t = *s;
}

// CHECK: define{{.*}} void @test_copy_constructor_StrongVolatile1(
// CHECK: call void @__copy_constructor_8_8_tv0w128_sv16(

void test_copy_constructor_StrongVolatile1(Strong *s) {
  volatile Strong t = *s;
}

// CHECK: define{{.*}} void @test_block_capture_Strong()
// CHECK: call void @__default_constructor_8_s16(
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(
// CHECK: call void @__destructor_8_s16(
// CHECK: call void @__destructor_8_s16(
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__copy_helper_block_8_32n13_8_8_t0w16_s16(ptr noundef %0, ptr noundef %1)
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destroy_helper_block_8_32n5_8_s16(
// CHECK: call void @__destructor_8_s16(
// CHECK: ret void

void test_block_capture_Strong(void) {
  Strong t;
  BlockTy b = ^(void){ (void)t; };
}

// CHECK: define{{.*}} void @test_variable_length_array(i32 noundef %[[N:.*]])
// CHECK: %[[N_ADDR:.*]] = alloca i32, align 4
// CHECK: store i32 %[[N]], ptr %[[N_ADDR]], align 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[N_ADDR]], align 4
// CHECK: %[[V1:.*]] = zext i32 %[[V0]] to i64
// CHECK: %[[VLA:.*]] = alloca %[[STRUCT_STRONG]], i64 %[[V1]], align 8
// CHECK: %[[V4:.*]] = mul nuw i64 24, %[[V1]]
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[VLA]], i64 %[[V4]]
// CHECK: br label

// CHECK: %[[DSTADDR_CUR:.*]] = phi ptr [ %[[VLA]], {{.*}} ], [ %[[V7:.*]], {{.*}} ]
// CHECK: %[[DONE:.*]] = icmp eq ptr %[[DSTADDR_CUR]], %[[V6]]
// CHECK: br i1 %[[DONE]], label

// CHECK: call void @__default_constructor_8_s16(ptr %[[DSTADDR_CUR]])
// CHECK: %[[V9:.*]] = getelementptr inbounds i8, ptr %[[DSTADDR_CUR]], i64 24
// CHECK: br label

// CHECK: call void @func(ptr noundef %[[VLA]])
// CHECK: %[[V10:.*]] = getelementptr inbounds %[[STRUCT_STRONG]], ptr %[[VLA]], i64 %[[V1]]
// CHECK: %[[ARRAYDESTROY_ISEMPTY:.*]] = icmp eq ptr %[[VLA]], %[[V10]]
// CHECK: br i1 %[[ARRAYDESTROY_ISEMPTY]], label

// CHECK: %[[ARRAYDESTROY_ELEMENTPAST:.*]] = phi ptr [ %[[V10]], {{.*}} ], [ %[[ARRAYDESTROY_ELEMENT:.*]], {{.*}} ]
// CHECK: %[[ARRAYDESTROY_ELEMENT]] = getelementptr inbounds %[[STRUCT_STRONG]], ptr %[[ARRAYDESTROY_ELEMENTPAST]], i64 -1
// CHECK: call void @__destructor_8_s16(ptr %[[ARRAYDESTROY_ELEMENT]])
// CHECK: %[[ARRAYDESTROY_DONE:.*]] = icmp eq ptr %[[ARRAYDESTROY_ELEMENT]], %[[VLA]]
// CHECK: br i1 %[[ARRAYDESTROY_DONE]], label

// CHECK: ret void

void test_variable_length_array(int n) {
  Strong a[n];
  func(a);
}

// CHECK: define linkonce_odr hidden void @__default_constructor_8_AB8s8n4_s8_AE(
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 %{{.*}}, i8 0, i64 32, i1 false)
void test_constructor_destructor_IDArray(void) {
  IDArray t;
}

// CHECK: define linkonce_odr hidden void @__default_constructor_8_AB8s24n4_S_s24_AE(
void test_constructor_destructor_StructArray(void) {
  StructArray t;
}

// Test that StructArray's field 'd' is copied before entering the loop.

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w8_AB8s24n4_S_t8w16_s24_AE(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: entry:
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V4:.*]] = load i64, ptr %[[V1]], align 8
// CHECK: store i64 %[[V4]], ptr %[[V0]], align 8

// CHECK: phi ptr
// CHECK: phi ptr

// CHECK: phi ptr
// CHECK: phi ptr

// CHECK-NOT: load i64, ptr %
// CHECK-NOT: store i64 %
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(

void test_copy_constructor_StructArray(StructArray a) {
  StructArray t = a;
}

// Check that IRGen copies the 9-bit bitfield emitting i16 load and store.

// CHECK: define{{.*}} void @test_copy_constructor_Bitfield0(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_s0_t8w2(
// CHECK: %[[V5:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK: %[[V8:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK: %[[V12:.*]] = load i16, ptr %[[V8]], align 8
// CHECK: store i16 %[[V12]], ptr %[[V5]], align 8
// CHECK: ret void

void test_copy_constructor_Bitfield0(Bitfield0 *a) {
  Bitfield0 t = *a;
}

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w2_s8_t16w4_s24_t32w12_s48_t56w9_tv513w2_tv520w8
// CHECK: %[[V4:.*]] = load i16, ptr %{{.*}}, align 8
// CHECK: store i16 %[[V4]], ptr %{{.*}}, align 8
// CHECK: %[[V21:.*]] = load i32, ptr %{{.*}}, align 8
// CHECK: store i32 %[[V21]], ptr %{{.*}}, align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i64 12, i1 false)
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i64 9, i1 false)
// CHECK: %[[I5:.*]] = getelementptr inbounds nuw %[[STRUCT_BITFIELD1]], ptr %[[V0:.*]], i32 0, i32 8
// CHECK: %[[I51:.*]] = getelementptr inbounds nuw %[[STRUCT_BITFIELD1]], ptr %[[V1:.*]], i32 0, i32 8
// CHECK: %[[BF_LOAD:.*]] = load volatile i8, ptr %[[I51]], align 8
// CHECK: %[[BF_SHL:.*]] = shl i8 %[[BF_LOAD]], 5
// CHECK: %[[BF_ASHR:.*]] = ashr i8 %[[BF_SHL]], 6
// CHECK: %[[BF_CAST:.*]] = sext i8 %[[BF_ASHR]] to i32
// CHECK: %[[V56:.*]] = trunc i32 %[[BF_CAST]] to i8
// CHECK: %[[BF_LOAD2:.*]] = load volatile i8, ptr %[[I5]], align 8
// CHECK: %[[BF_VALUE:.*]] = and i8 %[[V56]], 3
// CHECK: %[[BF_SHL3:.*]] = shl i8 %[[BF_VALUE]], 1
// CHECK: %[[BF_CLEAR:.*]] = and i8 %[[BF_LOAD2]], -7
// CHECK: %[[BF_SET:.*]] = or i8 %[[BF_CLEAR]], %[[BF_SHL3]]
// CHECK: store volatile i8 %[[BF_SET]], ptr %[[I5]], align 8
// CHECK: %[[I6:.*]] = getelementptr inbounds nuw %[[STRUCT_BITFIELD1]], ptr %[[V0]], i32 0, i32 9
// CHECK: %[[I64:.*]] = getelementptr inbounds nuw %[[STRUCT_BITFIELD1]], ptr %[[V1]], i32 0, i32 9
// CHECK: %[[V59:.*]] = load volatile i8, ptr %[[I64]], align 1
// CHECK: store volatile i8 %[[V59]], ptr %[[I6]], align 1

void test_copy_constructor_Bitfield1(Bitfield1 *a) {
  Bitfield1 t = *a;
}

// CHECK: define{{.*}} void @test_copy_constructor_VolatileArray(
// CHECK: call void @__copy_constructor_8_8_s0_AB8s4n16_tv64w32_AE(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_s0_AB8s4n16_tv64w32_AE(
// CHECK: %[[ADDR_CUR:.*]] = phi ptr
// CHECK: %[[ADDR_CUR1:.*]] = phi ptr
// CHECK: %[[V14:.*]] = load volatile i32, ptr %[[ADDR_CUR1]], align 4
// CHECK: store volatile i32 %[[V14]], ptr %[[ADDR_CUR]], align 4

void test_copy_constructor_VolatileArray(VolatileArray *a) {
  VolatileArray t = *a;
}

// CHECK: define{{.*}} void @test_compound_literal0(
// CHECK: %[[P:.*]] = alloca ptr, align 8
// CHECK: %[[_COMPOUNDLITERAL:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[CLEANUP_COND:.*]] = alloca i1, align 1
// CHECK: %[[_COMPOUNDLITERAL1:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[CLEANUP_COND4:.*]] = alloca i1, align 1

// CHECK: %[[I:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 0
// CHECK: store i32 1, ptr %[[I]], align 8
// CHECK: %[[F1:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 1
// CHECK: store ptr null, ptr %[[F1]], align 8
// CHECK: store i1 true, ptr %[[CLEANUP_COND]], align 1

// CHECK: %[[I2:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL1]], i32 0, i32 0
// CHECK: store i32 2, ptr %[[I2]], align 8
// CHECK: %[[F13:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL1]], i32 0, i32 1
// CHECK: store ptr null, ptr %[[F13]], align 8
// CHECK: store i1 true, ptr %[[CLEANUP_COND4]], align 1

// CHECK: %[[COND:.*]] = phi ptr [ %[[_COMPOUNDLITERAL]], %{{.*}} ], [ %[[_COMPOUNDLITERAL1]], %{{.*}} ]
// CHECK: store ptr %[[COND]], ptr %[[P]], align 8
// CHECK: call void @func(

// CHECK: call void @__destructor_8_s8(ptr %[[_COMPOUNDLITERAL1]])

// CHECK: call void @__destructor_8_s8(ptr %[[_COMPOUNDLITERAL]])

void test_compound_literal0(int c) {
  StrongSmall *p = c ? &(StrongSmall){ 1, 0 } : &(StrongSmall){ 2, 0 };
  func(0);
}

// Check that there is only one destructor call, which destructs 't'.

// CHECK: define{{.*}} void @test_compound_literal1(
// CHECK: %[[T:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8

// CHECK: %[[I:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[T]], i32 0, i32 0
// CHECK: store i32 1, ptr %[[I]], align 8
// CHECK: %[[F1:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[T]], i32 0, i32 1
// CHECK: store ptr null, ptr %[[F1]], align 8

// CHECK: %[[I1:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[T]], i32 0, i32 0
// CHECK: store i32 2, ptr %[[I1]], align 8
// CHECK: %[[F12:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[T]], i32 0, i32 1
// CHECK: store ptr null, ptr %[[F12]], align 8

// CHECK: call void @func(
// CHECK-NOT: call void
// CHECK: call void @__destructor_8_s8(ptr %[[T]])
// CHECK-NOT: call void

void test_compound_literal1(int c) {
  StrongSmall t = c ? (StrongSmall){ 1, 0 } : (StrongSmall){ 2, 0 };
  func(0);
}

// CHECK: define{{.*}} void @test_compound_literal2(
// CHECK: %[[P_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[_COMPOUNDLITERAL:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[CLEANUP_COND:.*]] = alloca i1, align 1
// CHECK: %[[_COMPOUNDLITERAL1:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[CLEANUP_COND4:.*]] = alloca i1, align 1
// CHECK: %[[V0:.*]] = load ptr, ptr %[[P_ADDR]], align 8

// CHECK: %[[I:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 0
// CHECK: store i32 1, ptr %[[I]], align 8
// CHECK: %[[F1:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 1
// CHECK: store ptr null, ptr %[[F1]], align 8
// CHECK: store i1 true, ptr %[[CLEANUP_COND]], align 1
// CHECK: call void @__copy_assignment_8_8_t0w4_s8(ptr %[[V0]], ptr %[[_COMPOUNDLITERAL]])

// CHECK: %[[I2:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL1]], i32 0, i32 0
// CHECK: store i32 2, ptr %[[I2]], align 8
// CHECK: %[[F13:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONGSMALL]], ptr %[[_COMPOUNDLITERAL1]], i32 0, i32 1
// CHECK: store ptr null, ptr %[[F13]], align 8
// CHECK: store i1 true, ptr %[[CLEANUP_COND4]], align 1
// CHECK: call void @__copy_assignment_8_8_t0w4_s8(ptr %[[V0]], ptr %[[_COMPOUNDLITERAL1]])

// CHECK: call void @func(

// CHECK: call void @__destructor_8_s8(ptr %[[_COMPOUNDLITERAL1]])

// CHECK: call void @__destructor_8_s8(ptr %[[_COMPOUNDLITERAL]])

void test_compound_literal2(int c, StrongSmall *p) {
  *p = c ? (StrongSmall){ 1, 0 } : (StrongSmall){ 2, 0 };
  func(0);
}

// CHECK: define{{.*}} void @test_member_access(
// CHECK: %[[TMP:.*]] = alloca %[[STRUCT_STRONGSMALL]],
// CHECK: call void @__destructor_8_s8(ptr %[[TMP]])
// CHECK: call void @func(

void test_member_access(void) {
  g0 = getStrongSmall().f1;
  func(0);
}

// CHECK: define{{.*}} void @test_member_access2(ptr noundef %[[C:.*]])
// CHECK: %[[COERCE:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: call void @__destructor_8_s8(ptr %[[COERCE]])
// CHECK: call void @func(

void test_member_access2(C *c) {
  g0 = [c getStrongSmall].f1;
  func(0);
}

// CHECK: define{{.*}} void @test_member_access3(
// CHECK: %[[COERCE:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: call void @__destructor_8_s8(ptr %[[COERCE]])
// CHECK: call void @func(

void test_member_access3(void) {
  g0 = [C getStrongSmallClass].f1;
  func(0);
}

// CHECK: define{{.*}} void @test_member_access4()
// CHECK: %[[COERCE:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: call void @__destructor_8_s8(ptr %[[COERCE]])
// CHECK: call void @func(

void test_member_access4(void) {
  g0 = ^{ StrongSmall s; return s; }().f1;
  func(0);
}

// CHECK: define{{.*}} void @test_volatile_variable_reference(
// CHECK: %[[AGG_TMP_ENSURED:.*]] = alloca %[[STRUCT_STRONGSMALL]],
// CHECK: call void @__copy_constructor_8_8_tv0w32_sv8(ptr %[[AGG_TMP_ENSURED]], ptr %{{.*}})
// CHECK: call void @__destructor_8_s8(ptr %[[AGG_TMP_ENSURED]])
// CHECK: call void @func(

void test_volatile_variable_reference(volatile StrongSmall *a) {
  (void)*a;
  func(0);
}

struct ZeroBitfield {
  int : 0;
  id strong;
};


// CHECK: define linkonce_odr hidden void @__default_constructor_8_sv0
// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_sv0
void test_zero_bitfield(void) {
  struct ZeroBitfield volatile a, b;
  a = b;
}

// CHECK-LABEL: define{{.*}} ptr @test_conditional0(
// CHECK: %[[TMP:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8

// CHECK: call void @__copy_constructor_8_8_t0w4_s8(ptr %[[TMP]], ptr @g2)

// CHECK: call void @__copy_constructor_8_8_t0w4_s8(ptr %[[TMP]], ptr @g1)

// CHECK: call void @__destructor_8_s8(ptr %[[TMP]])
// CHECK: @llvm.objc.autoreleaseReturnValue

id test_conditional0(int c) {
  return (c ? g2 : g1).f1;
}

// CHECK-LABEL: define{{.*}} void @test_conditional1(
// CHECK-NOT: call void @__destructor

void test_conditional1(int c) {
  calleeStrongSmall(c ? g2 : g1);
}

// CHECK-LABEL: define{{.*}} ptr @test_assignment0(
// CHECK: %[[TMP:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: call void @__copy_assignment_8_8_t0w4_s8(ptr @g2, ptr @g1)
// CHECK: call void @__copy_constructor_8_8_t0w4_s8(ptr %[[TMP]], ptr @g2)
// CHECK: call void @__destructor_8_s8(ptr %[[TMP]])

id test_assignment0(void) {
  return (g2 = g1).f1;
}

// CHECK-LABEL: define{{.*}} void @test_assignment1(
// CHECK-NOT: call void @__destructor

void test_assignment1(void) {
  calleeStrongSmall(g2 = g1);
}

// CHECK-LABEL: define{{.*}} void @test_null_reveiver(
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: br i1

// CHECK: %[[V8:.*]] = load [2 x i64], ptr %[[AGG_TMP]], align 8
// CHECK: call void @objc_msgSend({{.*}}, [2 x i64] %[[V8]])
// CHECK: br

// CHECK: call void @__destructor_8_s8(ptr %[[AGG_TMP]]) #4
// CHECK: br

void test_null_reveiver(C *c) {
  [c m:getStrongSmall()];
}

#endif /* USESTRUCT */
