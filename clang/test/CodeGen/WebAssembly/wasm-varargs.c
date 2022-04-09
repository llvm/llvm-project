// RUN: %clang_cc1 -no-opaque-pointers -triple wasm32-unknown-unknown -o - -emit-llvm %s | FileCheck %s

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

// CHECK:      define void @test_struct([[STRUCT_S:%[^,=]+]]*{{.*}} noalias sret({{.*}}) align 4 [[AGG_RESULT:%.*]], i8*{{.*}} %fmt, ...) {{.*}} {
// CHECK:        [[FMT_ADDR:%[^,=]+]] = alloca i8*, align 4
// CHECK-NEXT:   [[VA:%[^,=]+]] = alloca i8*, align 4
// CHECK-NEXT:   store i8* %fmt, i8** [[FMT_ADDR]], align 4
// CHECK-NEXT:   [[VA1:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:   call void @llvm.va_start(i8* [[VA1]])
// CHECK-NEXT:   [[ARGP_CUR:%[^,=]+]] = load i8*, i8** [[VA]], align 4
// CHECK-NEXT:   [[ARGP_NEXT:%[^,=]+]] = getelementptr inbounds i8, i8* [[ARGP_CUR]], i32 4
// CHECK-NEXT:   store i8* [[ARGP_NEXT]], i8** [[VA]], align 4
// CHECK-NEXT:   [[R3:%[^,=]+]] = bitcast i8* [[ARGP_CUR]] to [[STRUCT_S]]**
// CHECK-NEXT:   [[R4:%[^,=]+]] = load [[STRUCT_S]]*, [[STRUCT_S]]** [[R3]], align 4
// CHECK-NEXT:   [[R5:%[^,=]+]] = bitcast [[STRUCT_S]]* [[AGG_RESULT]] to i8*
// CHECK-NEXT:   [[R6:%[^,=]+]] = bitcast [[STRUCT_S]]* [[R4]] to i8*
// CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[R5]], i8* align 4 [[R6]], i32 12, i1 false)
// CHECK-NEXT:   [[VA2:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:   call void @llvm.va_end(i8* [[VA2]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

struct Z {};

struct S test_empty_struct(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  struct Z u = va_arg(va, struct Z);
  struct S v = va_arg(va, struct S);
  va_end(va);

  return v;
}

// CHECK:      define void @test_empty_struct([[STRUCT_S:%[^,=]+]]*{{.*}} noalias sret([[STRUCT_S]]) align 4 [[AGG_RESULT:%.*]], i8*{{.*}} %fmt, ...) {{.*}} {
// CHECK:        [[FMT_ADDR:%[^,=]+]] = alloca i8*, align 4
// CHECK-NEXT:   [[VA:%[^,=]+]] = alloca i8*, align 4
// CHECK-NEXT:   [[U:%[^,=]+]] = alloca [[STRUCT_Z:%[^,=]+]], align 1
// CHECK-NEXT:   store i8* %fmt, i8** [[FMT_ADDR]], align 4
// CHECK-NEXT:   [[VA1:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:   call void @llvm.va_start(i8* [[VA1]])
// CHECK-NEXT:   [[ARGP_CUR:%[^,=]+]] = load i8*, i8** [[VA]], align 4
// CHECK-NEXT:   [[ARGP_NEXT:%[^,=]+]] = getelementptr inbounds i8, i8* [[ARGP_CUR]], i32 0
// CHECK-NEXT:   store i8* [[ARGP_NEXT]], i8** [[VA]], align 4
// CHECK-NEXT:   [[R0:%[^,=]+]] = bitcast i8* [[ARGP_CUR]] to [[STRUCT_Z]]*
// CHECK-NEXT:   [[R1:%[^,=]+]] = bitcast [[STRUCT_Z]]* [[U]] to i8*
// CHECK-NEXT:   [[R2:%[^,=]+]] = bitcast [[STRUCT_Z]]* [[R0]] to i8*
// CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 [[R1]], i8* align 4 [[R2]], i32 0, i1 false)
// CHECK-NEXT:   [[ARGP_CUR2:%[^,=]+]] = load i8*, i8** [[VA]], align 4
// CHECK-NEXT:   [[ARGP_NEXT2:%[^,=]+]] = getelementptr inbounds i8, i8* [[ARGP_CUR2]], i32 4
// CHECK-NEXT:   store i8* [[ARGP_NEXT2]], i8** [[VA]], align 4
// CHECK-NEXT:   [[R3:%[^,=]+]] = bitcast i8* [[ARGP_CUR2]] to [[STRUCT_S]]**
// CHECK-NEXT:   [[R4:%[^,=]+]] = load [[STRUCT_S]]*, [[STRUCT_S]]** [[R3]], align 4
// CHECK-NEXT:   [[R5:%[^,=]+]] = bitcast [[STRUCT_S]]* [[AGG_RESULT]] to i8*
// CHECK-NEXT:   [[R6:%[^,=]+]] = bitcast [[STRUCT_S]]* [[R4]] to i8*
// CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[R5]], i8* align 4 [[R6]], i32 12, i1 false)
// CHECK-NEXT:   [[VA2:%[^,=]+]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:   call void @llvm.va_end(i8* [[VA2]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
