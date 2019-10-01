// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

struct test1 { int x; int y; };
struct test2 { int x; int y; } __attribute__((aligned (16)));
struct test3 { int x; int y; } __attribute__((aligned (32)));
struct test4 { int x; int y; int z; };
struct test5 { int x[17]; };
struct test6 { int x[17]; } __attribute__((aligned (16)));
struct test7 { int x[17]; } __attribute__((aligned (32)));

// CHECK: define void @test1(i32 signext %x, i64 %y.coerce)
void test1 (int x, struct test1 y)
{
}

// CHECK: define void @test2(i32 signext %x, [1 x i128] %y.coerce)
void test2 (int x, struct test2 y)
{
}

// CHECK: define void @test3(i32 signext %x, [2 x i128] %y.coerce)
void test3 (int x, struct test3 y)
{
}

// CHECK: define void @test4(i32 signext %x, [2 x i64] %y.coerce)
void test4 (int x, struct test4 y)
{
}

// CHECK: define void @test5(i32 signext %x, %struct.test5* byval(%struct.test5) align 8 %y)
void test5 (int x, struct test5 y)
{
}

// CHECK: define void @test6(i32 signext %x, %struct.test6* byval(%struct.test6) align 16 %y)
void test6 (int x, struct test6 y)
{
}

// This case requires run-time realignment of the incoming struct
// CHECK-LABEL: define void @test7(i32 signext %x, %struct.test7* byval(%struct.test7) align 16)
// CHECK: %y = alloca %struct.test7, align 32
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
void test7 (int x, struct test7 y)
{
}

// CHECK: define void @test1va(%struct.test1* noalias sret %[[AGG_RESULT:.*]], i32 signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 8
// CHECK: store i8* %[[NEXT]], i8** %ap
// CHECK: [[T0:%.*]] = bitcast i8* %[[CUR]] to %struct.test1*
// CHECK: [[DEST:%.*]] = bitcast %struct.test1* %[[AGG_RESULT]] to i8*
// CHECK: [[SRC:%.*]] = bitcast %struct.test1* [[T0]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[DEST]], i8* align 8 [[SRC]], i64 8, i1 false)
struct test1 test1va (int x, ...)
{
  struct test1 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test1);
  va_end(ap);
  return y;
}

// CHECK: define void @test2va(%struct.test2* noalias sret %[[AGG_RESULT:.*]], i32 signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK: %[[TMP0:[^ ]+]] = ptrtoint i8* %[[CUR]] to i64
// CHECK: %[[TMP1:[^ ]+]] = add i64 %[[TMP0]], 15
// CHECK: %[[TMP2:[^ ]+]] = and i64 %[[TMP1]], -16
// CHECK: %[[ALIGN:[^ ]+]] = inttoptr i64 %[[TMP2]] to i8*
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[ALIGN]], i64 16
// CHECK: store i8* %[[NEXT]], i8** %ap
// CHECK: [[T0:%.*]] = bitcast i8* %[[ALIGN]] to %struct.test2*
// CHECK: [[DEST:%.*]] = bitcast %struct.test2* %[[AGG_RESULT]] to i8*
// CHECK: [[SRC:%.*]] = bitcast %struct.test2* [[T0]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[DEST]], i8* align 16 [[SRC]], i64 16, i1 false)
struct test2 test2va (int x, ...)
{
  struct test2 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test2);
  va_end(ap);
  return y;
}

// CHECK: define void @test3va(%struct.test3* noalias sret %[[AGG_RESULT:.*]], i32 signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK: %[[TMP0:[^ ]+]] = ptrtoint i8* %[[CUR]] to i64
// CHECK: %[[TMP1:[^ ]+]] = add i64 %[[TMP0]], 15
// CHECK: %[[TMP2:[^ ]+]] = and i64 %[[TMP1]], -16
// CHECK: %[[ALIGN:[^ ]+]] = inttoptr i64 %[[TMP2]] to i8*
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[ALIGN]], i64 32
// CHECK: store i8* %[[NEXT]], i8** %ap
// CHECK: [[T0:%.*]] = bitcast i8* %[[ALIGN]] to %struct.test3*
// CHECK: [[DEST:%.*]] = bitcast %struct.test3* %[[AGG_RESULT]] to i8*
// CHECK: [[SRC:%.*]] = bitcast %struct.test3* [[T0]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 32 [[DEST]], i8* align 16 [[SRC]], i64 32, i1 false)
struct test3 test3va (int x, ...)
{
  struct test3 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test3);
  va_end(ap);
  return y;
}

// CHECK: define void @test4va(%struct.test4* noalias sret %[[AGG_RESULT:.*]], i32 signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 16
// CHECK: store i8* %[[NEXT]], i8** %ap
// CHECK: [[T0:%.*]] = bitcast i8* %[[CUR]] to %struct.test4*
// CHECK: [[DEST:%.*]] = bitcast %struct.test4* %[[AGG_RESULT]] to i8*
// CHECK: [[SRC:%.*]] = bitcast %struct.test4* [[T0]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[DEST]], i8* align 8 [[SRC]], i64 12, i1 false)
struct test4 test4va (int x, ...)
{
  struct test4 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test4);
  va_end(ap);
  return y;
}

// CHECK: define void @testva_longdouble(%struct.test_longdouble* noalias sret %[[AGG_RESULT:.*]], i32 signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 16
// CHECK: store i8* %[[NEXT]], i8** %ap
// CHECK: [[T0:%.*]] = bitcast i8* %[[CUR]] to %struct.test_longdouble*
// CHECK: [[DEST:%.*]] = bitcast %struct.test_longdouble* %[[AGG_RESULT]] to i8*
// CHECK: [[SRC:%.*]] = bitcast %struct.test_longdouble* [[T0]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[DEST]], i8* align 8 [[SRC]], i64 16, i1 false)
struct test_longdouble { long double x; };
struct test_longdouble testva_longdouble (int x, ...)
{
  struct test_longdouble y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test_longdouble);
  va_end(ap);
  return y;
}

// CHECK: define void @testva_vector(%struct.test_vector* noalias sret %[[AGG_RESULT:.*]], i32 signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK: %[[TMP0:[^ ]+]] = ptrtoint i8* %[[CUR]] to i64
// CHECK: %[[TMP1:[^ ]+]] = add i64 %[[TMP0]], 15
// CHECK: %[[TMP2:[^ ]+]] = and i64 %[[TMP1]], -16
// CHECK: %[[ALIGN:[^ ]+]] = inttoptr i64 %[[TMP2]] to i8*
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[ALIGN]], i64 16
// CHECK: store i8* %[[NEXT]], i8** %ap
// CHECK: [[T0:%.*]] = bitcast i8* %[[ALIGN]] to %struct.test_vector*
// CHECK: [[DEST:%.*]] = bitcast %struct.test_vector* %[[AGG_RESULT]] to i8*
// CHECK: [[SRC:%.*]] = bitcast %struct.test_vector* [[T0]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[DEST]], i8* align 16 [[SRC]], i64 16, i1 false)
struct test_vector { vector int x; };
struct test_vector testva_vector (int x, ...)
{
  struct test_vector y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test_vector);
  va_end(ap);
  return y;
}

