// RUN: %clang_cc1 -triple wasm32-unknown-unknown -o - -emit-llvm %s | FileCheck %s

#include <stdarg.h>

int test_i32(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  int v = va_arg(va, int);
  va_end(va);

  return v;
}

// CHECK-LABEL: define i32 @test_i32(i8*{{.*}} %fmt, ...) {{.*}} {
// CHECK:   [[FMT_ADDR:%[^,=]+]] = alloca i8*, align 4
// CHECK:   [[VA:%[^,=]+]] = alloca i8*, align 4
// CHECK:   [[V:%[^,=]+]] = alloca i32, align 4
// CHECK:   store i8* %fmt, i8** [[FMT_ADDR]], align 4
// CHECK:   [[VA1:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_start(i8* [[VA1]])
// CHECK:   [[ARGP_CUR:%[^,=]+]] = load i8*, i8** [[VA]], align 4
// CHECK:   [[ARGP_NEXT:%[^,=]+]] = getelementptr inbounds i8, i8* [[ARGP_CUR]], i32 4
// CHECK:   store i8* [[ARGP_NEXT]], i8** [[VA]], align 4
// CHECK:   [[R3:%[^,=]+]] = bitcast i8* [[ARGP_CUR]] to i32*
// CHECK:   [[R4:%[^,=]+]] = load i32, i32* [[R3]], align 4
// CHECK:   store i32 [[R4]], i32* [[V]], align 4
// CHECK:   [[VA2:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_end(i8* [[VA2]])
// CHECK:   [[R5:%[^,=]+]] = load i32, i32* [[V]], align 4
// CHECK:   ret i32 [[R5]]
// CHECK: }

long long test_i64(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  long long v = va_arg(va, long long);
  va_end(va);

  return v;
}

// CHECK-LABEL: define i64 @test_i64(i8*{{.*}} %fmt, ...) {{.*}} {
// CHECK:   [[FMT_ADDR:%[^,=]+]] = alloca i8*, align 4
// CHECK:   [[VA:%[^,=]+]] = alloca i8*, align 4
// CHECK:   [[V:%[^,=]+]] = alloca i64, align 8
// CHECK:   store i8* %fmt, i8** [[FMT_ADDR]], align 4
// CHECK:   [[VA1:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_start(i8* [[VA1]])
// CHECK:   [[ARGP_CUR:%[^,=]+]] = load i8*, i8** [[VA]], align 4
// CHECK:   [[R0:%[^,=]+]] = ptrtoint i8* [[ARGP_CUR]] to i32
// CHECK:   [[R1:%[^,=]+]] = add i32 [[R0]], 7
// CHECK:   [[R2:%[^,=]+]] = and i32 [[R1]], -8
// CHECK:   [[ARGP_CUR_ALIGNED:%[^,=]+]] = inttoptr i32 [[R2]] to i8*
// CHECK:   [[ARGP_NEXT:%[^,=]+]] = getelementptr inbounds i8, i8* [[ARGP_CUR_ALIGNED]], i32 8
// CHECK:   store i8* [[ARGP_NEXT]], i8** [[VA]], align 4
// CHECK:   [[R3:%[^,=]+]] = bitcast i8* [[ARGP_CUR_ALIGNED]] to i64*
// CHECK:   [[R4:%[^,=]+]] = load i64, i64* [[R3]], align 8
// CHECK:   store i64 [[R4]], i64* [[V]], align 8
// CHECK:   [[VA2:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_end(i8* [[VA2]])
// CHECK:   [[R5:%[^,=]+]] = load i64, i64* [[V]], align 8
// CHECK:   ret i64 [[R5]]
// CHECK: }

struct S {
  int x;
  int y;
  int z;
};

struct S test_struct(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  struct S v = va_arg(va, struct S);
  va_end(va);

  return v;
}

// CHECK: define void @test_struct([[STRUCT_S:%[^,=]+]]*{{.*}} noalias sret [[AGG_RESULT:%.*]], i8*{{.*}} %fmt, ...) {{.*}} {
// CHECK:   [[FMT_ADDR:%[^,=]+]] = alloca i8*, align 4
// CHECK:   [[VA:%[^,=]+]] = alloca i8*, align 4
// CHECK:   store i8* %fmt, i8** [[FMT_ADDR]], align 4
// CHECK:   [[VA1:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_start(i8* [[VA1]])
// CHECK:   [[ARGP_CUR:%[^,=]+]] = load i8*, i8** [[VA]], align 4
// CHECK:   [[ARGP_NEXT:%[^,=]+]] = getelementptr inbounds i8, i8* [[ARGP_CUR]], i32 12
// CHECK:   store i8* [[ARGP_NEXT]], i8** [[VA]], align 4
// CHECK:   [[R3:%[^,=]+]] = bitcast i8* [[ARGP_CUR]] to [[STRUCT_S]]*
// CHECK:   [[R4:%[^,=]+]] = bitcast [[STRUCT_S]]* [[AGG_RESULT]] to i8*
// CHECK:   [[R5:%[^,=]+]] = bitcast [[STRUCT_S]]* [[R3]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[R4]], i8* align 4 [[R5]], i32 12, i1 false)
// CHECK:   [[VA2:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_end(i8* [[VA2]])
// CHECK:   ret void
// CHECK: }
