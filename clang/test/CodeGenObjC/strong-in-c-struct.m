// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -emit-llvm -o - -DUSESTRUCT -I %S/Inputs %s | FileCheck %s

// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -emit-pch -I %S/Inputs -o %t %s
// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -include-pch %t -emit-llvm -o - -DUSESTRUCT -I %S/Inputs %s | FileCheck %s

#ifndef HEADER
#define HEADER
#include "strong_in_union.h"

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

#endif

#ifdef USESTRUCT

StrongSmall getStrongSmall(void);
StrongOuter getStrongOuter(void);
void calleeStrongSmall(StrongSmall);
void func(Strong *);

// CHECK: %[[STRUCT_BITFIELD1:.*]] = type { i8, i8, i8*, i32, i8*, [3 x i32], i8*, double, i8, i8 }

// CHECK: define void @test_constructor_destructor_StrongOuter()
// CHECK: %[[T:.*]] = alloca %[[STRUCT_STRONGOUTER:.*]], align 8
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONGOUTER]]* %[[T]] to i8**
// CHECK: call void @__default_constructor_8_s16_s24(i8** %[[V0]])
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONGOUTER]]* %[[T]] to i8**
// CHECK: call void @__destructor_8_s16_s24(i8** %[[V1]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__default_constructor_8_s16_s24(i8** %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: call void @__default_constructor_8_s16(i8** %[[V0]])
// CHECK: %[[V1:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, i8* %[[V1]], i64 24
// CHECK: %[[V3:.*]] = bitcast i8* %[[V2]] to i8**
// CHECK: %[[V4:.*]] = bitcast i8** %[[V3]] to i8*
// CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 %[[V4]], i8 0, i64 8, i1 false)
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__default_constructor_8_s16(i8** %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, i8* %[[V1]], i64 16
// CHECK: %[[V3:.*]] = bitcast i8* %[[V2]] to i8**
// CHECK: %[[V4:.*]] = bitcast i8** %[[V3]] to i8*
// CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 %[[V4]], i8 0, i64 8, i1 false)
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_s16_s24(i8** %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: call void @__destructor_8_s16(i8** %[[V0]])
// CHECK: %[[V1:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, i8* %[[V1]], i64 24
// CHECK: %[[V3:.*]] = bitcast i8* %[[V2]] to i8**
// CHECK: call void @objc_storeStrong(i8** %[[V3]], i8* null)
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__destructor_8_s16(i8** %[[DST:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V2:.*]] = getelementptr inbounds i8, i8* %[[V1]], i64 16
// CHECK: %[[V3:.*]] = bitcast i8* %[[V2]] to i8**
// CHECK: call void @objc_storeStrong(i8** %[[V3]], i8* null)
// CHECK: ret void

void test_constructor_destructor_StrongOuter(void) {
  StrongOuter t;
}

// CHECK: define void @test_copy_constructor_StrongOuter(%[[STRUCT_STRONGOUTER:.*]]* %[[S:.*]])
// CHECK: %[[S_ADDR:.*]] = alloca %[[STRUCT_STRONGOUTER]]*, align 8
// CHECK: %[[T:.*]] = alloca %[[STRUCT_STRONGOUTER]], align 8
// CHECK: store %[[STRUCT_STRONGOUTER]]* %[[S]], %[[STRUCT_STRONGOUTER]]** %[[S_ADDR]], align 8
// CHECK: %[[V0:.*]] = load %[[STRUCT_STRONGOUTER]]*, %[[STRUCT_STRONGOUTER]]** %[[S_ADDR]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONGOUTER]]* %[[T]] to i8**
// CHECK: %[[V2:.*]] = bitcast %[[STRUCT_STRONGOUTER]]* %[[V0]] to i8**
// CHECK: call void @__copy_constructor_8_8_t0w16_s16_s24_t32w8(i8** %[[V1]], i8** %[[V2]])
// CHECK: %[[V3:.*]] = bitcast %[[STRUCT_STRONGOUTER]]* %[[T]] to i8**
// CHECK: call void @__destructor_8_s16_s24(i8** %[[V3]])
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w16_s16_s24_t32w8(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(i8** %[[V0]], i8** %[[V1]])
// CHECK: %[[V2:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, i8* %[[V2]], i64 24
// CHECK: %[[V4:.*]] = bitcast i8* %[[V3]] to i8**
// CHECK: %[[V5:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 24
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// CHECK: %[[V8:.*]] = load i8*, i8** %[[V7]], align 8
// CHECK: %[[V9:.*]] = call i8* @objc_retain(i8* %[[V8]])
// CHECK: store i8* %[[V9]], i8** %[[V4]], align 8
// CHECK: %[[V10:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V11:.*]] = getelementptr inbounds i8, i8* %[[V10]], i64 32
// CHECK: %[[V12:.*]] = bitcast i8* %[[V11]] to i8**
// CHECK: %[[V13:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: %[[V14:.*]] = getelementptr inbounds i8, i8* %[[V13]], i64 32
// CHECK: %[[V15:.*]] = bitcast i8* %[[V14]] to i8**
// CHECK: %[[V16:.*]] = bitcast i8** %[[V12]] to i64*
// CHECK: %[[V17:.*]] = bitcast i8** %[[V15]] to i64*
// CHECK: %[[V18:.*]] = load i64, i64* %[[V17]], align 8
// CHECK: store i64 %[[V18]], i64* %[[V16]], align 8
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w16_s16(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V3:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %[[V2]], i8* align 8 %[[V3]], i64 16, i1 false)
// CHECK: %[[V4:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V5:.*]] = getelementptr inbounds i8, i8* %[[V4]], i64 16
// CHECK: %[[V6:.*]] = bitcast i8* %[[V5]] to i8**
// CHECK: %[[V7:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: %[[V8:.*]] = getelementptr inbounds i8, i8* %[[V7]], i64 16
// CHECK: %[[V9:.*]] = bitcast i8* %[[V8]] to i8**
// CHECK: %[[V10:.*]] = load i8*, i8** %[[V9]], align 8
// CHECK: %[[V11:.*]] = call i8* @objc_retain(i8* %[[V10]])
// CHECK: store i8* %[[V11]], i8** %[[V6]], align 8
// CHECK: ret void

void test_copy_constructor_StrongOuter(StrongOuter *s) {
  StrongOuter t = *s;
}

/// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_t0w16_s16_s24_t32w8(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, i8* %[[V2]], i64 24
// CHECK: %[[V4:.*]] = bitcast i8* %[[V3]] to i8**
// CHECK: %[[V5:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 24
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// CHECK: %[[V8:.*]] = load i8*, i8** %[[V7]], align 8
// CHECK: call void @objc_storeStrong(i8** %[[V4]], i8* %[[V8]])

void test_copy_assignment_StrongOuter(StrongOuter *d, StrongOuter *s) {
  *d = *s;
}

// CHECK: define void @test_move_constructor_StrongOuter()
// CHECK: %[[T1:.*]] = getelementptr inbounds %[[STRUCT_BLOCK_BYREF_T:.*]], %[[STRUCT_BLOCK_BYREF_T]]* %{{.*}}, i32 0, i32 7
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONGOUTER]]* %[[T1]] to i8**
// CHECK: call void @__default_constructor_8_s16_s24(i8** %[[V1]])
// CHECK: %[[T2:.*]] = getelementptr inbounds %[[STRUCT_BLOCK_BYREF_T]], %[[STRUCT_BLOCK_BYREF_T]]* %{{.*}}, i32 0, i32 7
// CHECK: %[[V9:.*]] = bitcast %[[STRUCT_STRONGOUTER]]* %[[T2]] to i8**
// CHECK: call void @__destructor_8_s16_s24(i8** %[[V9]])

// CHECK: define internal void @__Block_byref_object_copy_(i8*, i8*)
// CHECK: call void @__move_constructor_8_8_t0w16_s16_s24_t32w8(

// CHECK: define linkonce_odr hidden void @__move_constructor_8_8_t0w16_s16_s24_t32w8(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: call void @__move_constructor_8_8_t0w16_s16(i8** %[[V0]], i8** %[[V1]])
// CHECK: %[[V2:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, i8* %[[V2]], i64 24
// CHECK: %[[V4:.*]] = bitcast i8* %[[V3]] to i8**
// CHECK: %[[V5:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 24
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// CHECK: %[[V8:.*]] = load i8*, i8** %[[V7]], align 8
// CHECK: store i8* null, i8** %[[V7]], align 8
// CHECK: store i8* %[[V8]], i8** %[[V4]], align 8

// CHECK: define internal void @__Block_byref_object_dispose_(i8*)
// CHECK: call void @__destructor_8_s16_s24(

void test_move_constructor_StrongOuter(void) {
  __block StrongOuter t;
  BlockTy b = ^{ (void)t; };
}

// CHECK: define linkonce_odr hidden void @__move_assignment_8_8_t0w16_s16_s24_t32w8(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: call void @__move_assignment_8_8_t0w16_s16(i8** %[[V0]], i8** %[[V1]])
// CHECK: %[[V2:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V3:.*]] = getelementptr inbounds i8, i8* %[[V2]], i64 24
// CHECK: %[[V4:.*]] = bitcast i8* %[[V3]] to i8**
// CHECK: %[[V5:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 24
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// CHECK: %[[V8:.*]] = load i8*, i8** %[[V7]], align 8
// CHECK: store i8* null, i8** %[[V7]], align 8
// CHECK: %[[V9:.*]] = load i8*, i8** %[[V4]], align 8
// CHECK: store i8* %[[V8]], i8** %[[V4]], align 8
// CHECK: call void @objc_release(i8* %[[V9]])

void test_move_assignment_StrongOuter(StrongOuter *p) {
  *p = getStrongOuter();
}

// CHECK: define void @test_parameter_StrongSmall([2 x i64] %[[A_COERCE:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_STRONG:.*]], align 8
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONG]]* %[[A]] to [2 x i64]*
// CHECK: store [2 x i64] %[[A_COERCE]], [2 x i64]* %[[V0]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONG]]* %[[A]] to i8**
// CHECK: call void @__destructor_8_s8(i8** %[[V1]])
// CHECK: ret void

void test_parameter_StrongSmall(StrongSmall a) {
}

// CHECK: define void @test_argument_StrongSmall([2 x i64] %[[A_COERCE:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_STRONGSMALL:.*]], align 8
// CHECK: %[[TEMP_LVALUE:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[A]] to [2 x i64]*
// CHECK: store [2 x i64] %[[A_COERCE]], [2 x i64]* %[[V0]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[TEMP_LVALUE]] to i8**
// CHECK: %[[V2:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[A]] to i8**
// CHECK: call void @__copy_constructor_8_8_t0w4_s8(i8** %[[V1]], i8** %[[V2]])
// CHECK: %[[V3:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[TEMP_LVALUE]] to [2 x i64]*
// CHECK: %[[V4:.*]] = load [2 x i64], [2 x i64]* %[[V3]], align 8
// CHECK: call void @calleeStrongSmall([2 x i64] %[[V4]])
// CHECK: %[[V5:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[A]] to i8**
// CHECK: call void @__destructor_8_s8(i8** %[[V5]])
// CHECK: ret void

void test_argument_StrongSmall(StrongSmall a) {
  calleeStrongSmall(a);
}

// CHECK: define [2 x i64] @test_return_StrongSmall([2 x i64] %[[A_COERCE:.*]])
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_STRONGSMALL:.*]], align 8
// CHECK: %[[A:.*]] = alloca %[[STRUCT_STRONGSMALL]], align 8
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[A]] to [2 x i64]*
// CHECK: store [2 x i64] %[[A_COERCE]], [2 x i64]* %[[V0]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[RETVAL]] to i8**
// CHECK: %[[V2:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[A]] to i8**
// CHECK: call void @__copy_constructor_8_8_t0w4_s8(i8** %[[V1]], i8** %[[V2]])
// CHECK: %[[V3:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[A]] to i8**
// CHECK: call void @__destructor_8_s8(i8** %[[V3]])
// CHECK: %[[V4:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[RETVAL]] to [2 x i64]*
// CHECK: %[[V5:.*]] = load [2 x i64], [2 x i64]* %[[V4]], align 8
// CHECK: ret [2 x i64] %[[V5]]

StrongSmall test_return_StrongSmall(StrongSmall a) {
  return a;
}

// CHECK: define void @test_destructor_ignored_result()
// CHECK: %[[COERCE:.*]] = alloca %[[STRUCT_STRONGSMALL:.*]], align 8
// CHECK: %[[CALL:.*]] = call [2 x i64] @getStrongSmall()
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[COERCE]] to [2 x i64]*
// CHECK: store [2 x i64] %[[CALL]], [2 x i64]* %[[V0]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONGSMALL]]* %[[COERCE]] to i8**
// CHECK: call void @__destructor_8_s8(i8** %[[V1]])
// CHECK: ret void

void test_destructor_ignored_result(void) {
  getStrongSmall();
}

// CHECK: define void @test_copy_constructor_StrongBlock(
// CHECK: call void @__copy_constructor_8_8_sb0(
// CHECK: call void @__destructor_8_sb0(
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_sb0(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load i8*, i8** %[[V1]], align 8
// CHECK: %[[V3:.*]] = call i8* @objc_retainBlock(i8* %[[V2]])
// CHECK: store i8* %[[V3]], i8** %[[V0]], align 8
// CHECK: ret void

void test_copy_constructor_StrongBlock(StrongBlock *s) {
  StrongBlock t = *s;
}

// CHECK: define void @test_copy_assignment_StrongBlock(%[[STRUCT_STRONGBLOCK:.*]]* %[[D:.*]], %[[STRUCT_STRONGBLOCK]]* %[[S:.*]])
// CHECK: call void @__copy_assignment_8_8_sb0(

// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_sb0(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V2:.*]] = load i8*, i8** %[[V1]], align 8
// CHECK: %[[V3:.*]] = call i8* @objc_retainBlock(i8* %[[V2]])
// CHECK: %[[V4:.*]] = load i8*, i8** %[[V0]], align 8
// CHECK: store i8* %[[V3]], i8** %[[V0]], align 8
// CHECK: call void @objc_release(i8* %[[V4]])
// CHECK: ret void

void test_copy_assignment_StrongBlock(StrongBlock *d, StrongBlock *s) {
  *d = *s;
}

// CHECK: define void @test_copy_constructor_StrongVolatile0(
// CHECK: call void @__copy_constructor_8_8_t0w4_sv8(
// CHECK: call void @__destructor_8_sv8(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_sv8(
// CHECK: %[[V8:.*]] = load volatile i8*, i8** %{{.*}}, align 8
// CHECK: %[[V9:.*]] = call i8* @objc_retain(i8* %[[V8]])
// CHECK: store volatile i8* %[[V9]], i8** %{{.*}}, align 8

void test_copy_constructor_StrongVolatile0(StrongVolatile *s) {
  StrongVolatile t = *s;
}

// CHECK: define void @test_copy_constructor_StrongVolatile1(
// CHECK: call void @__copy_constructor_8_8_tv0w128_sv16(

void test_copy_constructor_StrongVolatile1(Strong *s) {
  volatile Strong t = *s;
}

// CHECK: define void @test_block_capture_Strong()
// CHECK: call void @__default_constructor_8_s16(
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(
// CHECK: call void @__destructor_8_s16(
// CHECK: call void @__destructor_8_s16(
// CHECK: ret void

// CHECK: define internal void @__copy_helper_block_.1(i8*, i8*)
// CHECK: call void @__copy_constructor_8_8_t0w16_s16(
// CHECK: ret void

// CHECK: define internal void @__destroy_helper_block_.2(
// CHECK: call void @__destructor_8_s16(
// CHECK: ret void

void test_block_capture_Strong(void) {
  Strong t;
  BlockTy b = ^(){ (void)t; };
}

// CHECK: define void @test_variable_length_array(i32 %[[N:.*]])
// CHECK: %[[N_ADDR:.*]] = alloca i32, align 4
// CHECK: store i32 %[[N]], i32* %[[N_ADDR]], align 4
// CHECK: %[[V0:.*]] = load i32, i32* %[[N_ADDR]], align 4
// CHECK: %[[V1:.*]] = zext i32 %[[V0]] to i64
// CHECK: %[[VLA:.*]] = alloca %[[STRUCT_STRONG:.*]], i64 %[[V1]], align 8
// CHECK: %[[V3:.*]] = bitcast %[[STRUCT_STRONG]]* %[[VLA]] to i8**
// CHECK: %[[V4:.*]] = mul nuw i64 24, %[[V1]]
// CHECK: %[[V5:.*]] = bitcast i8** %[[V3]] to i8*
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 %[[V4]]
// CHECK: %[[DSTARRAY_END:.*]] = bitcast i8* %[[V6]] to i8**
// CHECK: br label

// CHECK: %[[DSTADDR_CUR:.*]] = phi i8** [ %[[V3]], {{.*}} ], [ %[[V7:.*]], {{.*}} ]
// CHECK: %[[DONE:.*]] = icmp eq i8** %[[DSTADDR_CUR]], %[[DSTARRAY_END]]
// CHECK: br i1 %[[DONE]], label

// CHECK: call void @__default_constructor_8_s16(i8** %[[DSTADDR_CUR]])
// CHECK: %[[V8:.*]] = bitcast i8** %[[DSTADDR_CUR]] to i8*
// CHECK: %[[V9:.*]] = getelementptr inbounds i8, i8* %[[V8]], i64 24
// CHECK: %[[V7]] = bitcast i8* %[[V9]] to i8**
// CHECK: br label

// CHECK: call void @func(%[[STRUCT_STRONG]]* %[[VLA]])
// CHECK: %[[V10:.*]] = getelementptr inbounds %[[STRUCT_STRONG]], %[[STRUCT_STRONG]]* %[[VLA]], i64 %[[V1]]
// CHECK: %[[ARRAYDESTROY_ISEMPTY:.*]] = icmp eq %[[STRUCT_STRONG]]* %[[VLA]], %[[V10]]
// CHECK: br i1 %[[ARRAYDESTROY_ISEMPTY]], label

// CHECK: %[[ARRAYDESTROY_ELEMENTPAST:.*]] = phi %[[STRUCT_STRONG]]* [ %[[V10]], {{.*}} ], [ %[[ARRAYDESTROY_ELEMENT:.*]], {{.*}} ]
// CHECK: %[[ARRAYDESTROY_ELEMENT]] = getelementptr inbounds %[[STRUCT_STRONG]], %[[STRUCT_STRONG]]* %[[ARRAYDESTROY_ELEMENTPAST]], i64 -1
// CHECK: %[[V11:.*]] = bitcast %[[STRUCT_STRONG]]* %[[ARRAYDESTROY_ELEMENT]] to i8**
// CHECK: call void @__destructor_8_s16(i8** %[[V11]])
// CHECK: %[[ARRAYDESTROY_DONE:.*]] = icmp eq %[[STRUCT_STRONG]]* %[[ARRAYDESTROY_ELEMENT]], %[[VLA]]
// CHECK: br i1 %[[ARRAYDESTROY_DONE]], label

// CHECK: ret void

void test_variable_length_array(int n) {
  Strong a[n];
  func(a);
}

// CHECK: define linkonce_odr hidden void @__default_constructor_8_AB8s8n4_s8_AE(
// CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 %{{.*}}, i8 0, i64 32, i1 false)
void test_constructor_destructor_IDArray(void) {
  IDArray t;
}

// CHECK: define linkonce_odr hidden void @__default_constructor_8_AB8s24n4_s24_AE(
void test_constructor_destructor_StructArray(void) {
  StructArray t;
}

// Check that IRGen copies the 9-bit bitfield emitting i16 load and store.

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_s0_t8w2(
// CHECK: %[[V4:.*]] = bitcast i8** %{{.*}} to i8*
// CHECK: %[[V5:.*]] = getelementptr inbounds i8, i8* %[[V4]], i64 8
// CHECK: %[[V6:.*]] = bitcast i8* %[[V5]] to i8**
// CHECK: %[[V7:.*]] = bitcast i8** %{{.*}} to i8*
// CHECK: %[[V8:.*]] = getelementptr inbounds i8, i8* %[[V7]], i64 8
// CHECK: %[[V9:.*]] = bitcast i8* %[[V8]] to i8**
// CHECK: %[[V10:.*]] = bitcast i8** %[[V6]] to i16*
// CHECK: %[[V11:.*]] = bitcast i8** %[[V9]] to i16*
// CHECK: %[[V12:.*]] = load i16, i16* %[[V11]], align 8
// CHECK: store i16 %[[V12]], i16* %[[V10]], align 8
// CHECK: ret void

void test_copy_constructor_Bitfield0(Bitfield0 *a) {
  Bitfield0 t = *a;
}

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w2_s8_t16w4_s24_t32w12_s48_t56w9_tv513w2_tv520w8
// CHECK: %[[V4:.*]] = load i16, i16* %{{.*}}, align 8
// CHECK: store i16 %[[V4]], i16* %{{.*}}, align 8
// CHECK: %[[V21:.*]] = load i32, i32* %{{.*}}, align 8
// CHECK: store i32 %[[V21]], i32* %{{.*}}, align 8
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %{{.*}}, i8* align 8 %{{.*}}, i64 12, i1 false)
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %{{.*}}, i8* align 8 %{{.*}}, i64 9, i1 false)
// CHECK: %[[V54:.*]] = bitcast i8** %[[V0:.*]] to %[[STRUCT_BITFIELD1]]*
// CHECK: %[[I5:.*]] = getelementptr inbounds %[[STRUCT_BITFIELD1]], %[[STRUCT_BITFIELD1]]* %[[V54]], i32 0, i32 8
// CHECK: %[[V55:.*]] = bitcast i8** %[[V1:.*]] to %[[STRUCT_BITFIELD1]]*
// CHECK: %[[I51:.*]] = getelementptr inbounds %[[STRUCT_BITFIELD1]], %[[STRUCT_BITFIELD1]]* %[[V55]], i32 0, i32 8
// CHECK: %[[BF_LOAD:.*]] = load volatile i8, i8* %[[I51]], align 8
// CHECK: %[[BF_SHL:.*]] = shl i8 %[[BF_LOAD]], 5
// CHECK: %[[BF_ASHR:.*]] = ashr i8 %[[BF_SHL]], 6
// CHECK: %[[BF_CAST:.*]] = sext i8 %[[BF_ASHR]] to i32
// CHECK: %[[V56:.*]] = trunc i32 %[[BF_CAST]] to i8
// CHECK: %[[BF_LOAD2:.*]] = load volatile i8, i8* %[[I5]], align 8
// CHECK: %[[BF_VALUE:.*]] = and i8 %[[V56]], 3
// CHECK: %[[BF_SHL3:.*]] = shl i8 %[[BF_VALUE]], 1
// CHECK: %[[BF_CLEAR:.*]] = and i8 %[[BF_LOAD2]], -7
// CHECK: %[[BF_SET:.*]] = or i8 %[[BF_CLEAR]], %[[BF_SHL3]]
// CHECK: store volatile i8 %[[BF_SET]], i8* %[[I5]], align 8
// CHECK: %[[V57:.*]] = bitcast i8** %[[V0]] to %[[STRUCT_BITFIELD1]]*
// CHECK: %[[I6:.*]] = getelementptr inbounds %[[STRUCT_BITFIELD1]], %[[STRUCT_BITFIELD1]]* %[[V57]], i32 0, i32 9
// CHECK: %[[V58:.*]] = bitcast i8** %[[V1]] to %[[STRUCT_BITFIELD1]]*
// CHECK: %[[I64:.*]] = getelementptr inbounds %[[STRUCT_BITFIELD1]], %[[STRUCT_BITFIELD1]]* %[[V58]], i32 0, i32 9
// CHECK: %[[V59:.*]] = load volatile i8, i8* %[[I64]], align 1
// CHECK: store volatile i8 %[[V59]], i8* %[[I6]], align 1

void test_copy_constructor_Bitfield1(Bitfield1 *a) {
  Bitfield1 t = *a;
}

// CHECK: define void @test_strong_in_union()
// CHECK: alloca %{{.*}}
// CHECK-NEXT: ret void

void test_strong_in_union() {
  U t;
}

// CHECK: define void @test_copy_constructor_VolatileArray(
// CHECK: call void @__copy_constructor_8_8_s0_AB8s4n16_tv64w32_AE(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_s0_AB8s4n16_tv64w32_AE(
// CHECK: %[[ADDR_CUR:.*]] = phi i8**
// CHECK: %[[ADDR_CUR1:.*]] = phi i8**
// CHECK: %[[V12:.*]] = bitcast i8** %[[ADDR_CUR]] to i32*
// CHECK: %[[V13:.*]] = bitcast i8** %[[ADDR_CUR1]] to i32*
// CHECK: %[[V14:.*]] = load volatile i32, i32* %[[V13]], align 4
// CHECK: store volatile i32 %[[V14]], i32* %[[V12]], align 4

void test_copy_constructor_VolatileArray(VolatileArray *a) {
  VolatileArray t = *a;
}

#endif /* USESTRUCT */
