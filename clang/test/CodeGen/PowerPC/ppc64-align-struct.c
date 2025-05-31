// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

struct test1 { int x; int y; };
struct test2 { int x; int y; } __attribute__((aligned (16)));
struct test3 { int x; int y; } __attribute__((aligned (32)));
struct test4 { int x; int y; int z; };
struct test5 { int x[17]; };
struct test6 { int x[17]; } __attribute__((aligned (16)));
struct test7 { int x[17]; } __attribute__((aligned (32)));
struct test8 { char x; };
struct test9 { _Complex char x; };

// CHECK: define{{.*}} void @test1(i32 noundef signext %x, i64 %y.coerce)
void test1 (int x, struct test1 y)
{
}

// CHECK: define{{.*}} void @test2(i32 noundef signext %x, [1 x i128] %y.coerce)
void test2 (int x, struct test2 y)
{
}

// CHECK: define{{.*}} void @test3(i32 noundef signext %x, [2 x i128] %y.coerce)
void test3 (int x, struct test3 y)
{
}

// CHECK: define{{.*}} void @test4(i32 noundef signext %x, [2 x i64] %y.coerce)
void test4 (int x, struct test4 y)
{
}

// CHECK: define{{.*}} void @test5(i32 noundef signext %x, ptr noundef byval(%struct.test5) align 8 %y)
void test5 (int x, struct test5 y)
{
}

// CHECK: define{{.*}} void @test6(i32 noundef signext %x, ptr noundef byval(%struct.test6) align 16 %y)
void test6 (int x, struct test6 y)
{
}

// This case requires run-time realignment of the incoming struct
// CHECK-LABEL: define{{.*}} void @test7(i32 noundef signext %x, ptr noundef byval(%struct.test7) align 16 %0)
// CHECK: %y = alloca %struct.test7, align 32
// CHECK: call void @llvm.memcpy.p0.p0.i64
void test7 (int x, struct test7 y)
{
}

// CHECK: define{{.*}} void @test8(i32 noundef signext %x, i8 %y.coerce)
void test8 (int x, struct test8 y)
{
}

// CHECK: define{{.*}} void @test9(i32 noundef signext %x, i16 %y.coerce)
void test9 (int x, struct test9 y)
{
}

// CHECK: define{{.*}} void @test1va(ptr dead_on_unwind noalias writable sret(%struct.test1) align 4 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i64 8
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[AGG_RESULT]], ptr align 8 %[[CUR]], i64 8, i1 false)
struct test1 test1va (int x, ...)
{
  struct test1 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test1);
  va_end(ap);
  return y;
}

// CHECK: define{{.*}} void @test2va(ptr dead_on_unwind noalias writable sret(%struct.test2) align 16 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[TMP0:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i32 15
// CHECK: %[[ALIGN:[^ ]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[TMP0]], i64 -16)
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[ALIGN]], i64 16
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[AGG_RESULT]], ptr align 16 %[[ALIGN]], i64 16, i1 false)
struct test2 test2va (int x, ...)
{
  struct test2 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test2);
  va_end(ap);
  return y;
}

// CHECK: define{{.*}} void @test3va(ptr dead_on_unwind noalias writable sret(%struct.test3) align 32 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[TMP0:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i32 15
// CHECK: %[[ALIGN:[^ ]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[TMP0]], i64 -16)
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[ALIGN]], i64 32
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 32 %[[AGG_RESULT]], ptr align 16 %[[ALIGN]], i64 32, i1 false)
struct test3 test3va (int x, ...)
{
  struct test3 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test3);
  va_end(ap);
  return y;
}

// CHECK: define{{.*}} void @test4va(ptr dead_on_unwind noalias writable sret(%struct.test4) align 4 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i64 16
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[AGG_RESULT]], ptr align 8 %[[CUR]], i64 12, i1 false)
struct test4 test4va (int x, ...)
{
  struct test4 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test4);
  va_end(ap);
  return y;
}

// CHECK: define{{.*}} void @test8va(ptr dead_on_unwind noalias writable sret(%struct.test8) align 1 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i64 8
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: [[T0:%.*]] = getelementptr inbounds i8, ptr %[[CUR]], i64 7
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[AGG_RESULT]], ptr align 1 [[T0]], i64 1, i1 false)
struct test8 test8va (int x, ...)
{
  struct test8 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test8);
  va_end(ap);
  return y;
}

// CHECK: define{{.*}} void @test9va(ptr dead_on_unwind noalias writable sret(%struct.test9) align 1 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i64 8
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: [[T0:%.*]] = getelementptr inbounds i8, ptr %[[CUR]], i64 6
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[AGG_RESULT]], ptr align 2 [[T0]], i64 2, i1 false)
struct test9 test9va (int x, ...)
{
  struct test9 y;
  va_list ap;
  va_start(ap, x);
  y = va_arg (ap, struct test9);
  va_end(ap);
  return y;
}

// CHECK: define{{.*}} void @testva_longdouble(ptr dead_on_unwind noalias writable sret(%struct.test_longdouble) align 16 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i64 16
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[AGG_RESULT]], ptr align 8 %[[CUR]], i64 16, i1 false)
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

// CHECK: define{{.*}} void @testva_vector(ptr dead_on_unwind noalias writable sret(%struct.test_vector) align 16 %[[AGG_RESULT:.*]], i32 noundef signext %x, ...)
// CHECK: %[[CUR:[^ ]+]] = load ptr, ptr %ap
// CHECK: %[[TMP0:[^ ]+]] = getelementptr inbounds i8, ptr %[[CUR]], i32 15
// CHECK: %[[ALIGN:[^ ]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %[[TMP0]], i64 -16)
// CHECK: %[[NEXT:[^ ]+]] = getelementptr inbounds i8, ptr %[[ALIGN]], i64 16
// CHECK: store ptr %[[NEXT]], ptr %ap
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[AGG_RESULT]], ptr align 16 %[[ALIGN]], i64 16, i1 false)
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

