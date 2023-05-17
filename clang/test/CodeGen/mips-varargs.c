// RUN: %clang_cc1 -triple mips-unknown-linux -o - -emit-llvm %s | FileCheck %s -check-prefixes=ALL,O32 -enable-var-scope
// RUN: %clang_cc1 -triple mipsel-unknown-linux -o - -emit-llvm %s | FileCheck %s -check-prefixes=ALL,O32 -enable-var-scope
// RUN: %clang_cc1 -triple mips64-unknown-linux -o - -emit-llvm  -target-abi n32 %s | FileCheck %s -check-prefixes=ALL,N32,NEW -enable-var-scope
// RUN: %clang_cc1 -triple mips64-unknown-linux -o - -emit-llvm  -target-abi n32 %s | FileCheck %s -check-prefixes=ALL,N32,NEW -enable-var-scope
// RUN: %clang_cc1 -triple mips64-unknown-linux -o - -emit-llvm %s | FileCheck %s -check-prefixes=ALL,N64,NEW -enable-var-scope
// RUN: %clang_cc1 -triple mips64el-unknown-linux -o - -emit-llvm %s | FileCheck %s -check-prefixes=ALL,N64,NEW -enable-var-scope

#include <stdarg.h>

typedef int v4i32 __attribute__ ((__vector_size__ (16)));

int test_i32(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  int v = va_arg(va, int);
  va_end(va);

  return v;
}

// O32-LABEL: define{{.*}} i32 @test_i32(ptr{{.*}} %fmt, ...)
// N32-LABEL: define{{.*}} signext i32 @test_i32(ptr{{.*}} %fmt, ...)
// N64-LABEL: define{{.*}} signext i32 @test_i32(ptr{{.*}} %fmt, ...)
//
// O32:   %va = alloca ptr, align [[$PTRALIGN:4]]
// N32:   %va = alloca ptr, align [[$PTRALIGN:4]]
// N64:   %va = alloca ptr, align [[$PTRALIGN:8]]
// ALL:   [[V:%.*]] = alloca i32, align 4
// NEW:   [[PROMOTION_TEMP:%.*]] = alloca i32, align 4
//
// ALL:   call void @llvm.va_start(ptr %va)
// ALL:   [[AP_CUR:%.+]] = load ptr, ptr %va, align [[$PTRALIGN]]
// O32:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], [[$INTPTR_T:i32]] [[$CHUNKSIZE:4]]
// NEW:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], [[$INTPTR_T:i32|i64]] [[$CHUNKSIZE:8]]
//
// ALL:   store ptr [[AP_NEXT]], ptr %va, align [[$PTRALIGN]]
//
// O32:   [[ARG:%.+]] = load i32, ptr [[AP_CUR]], align [[CHUNKALIGN:4]]
//
// N32:   [[TMP:%.+]] = load i64, ptr [[AP_CUR]], align [[CHUNKALIGN:8]]
// N64:   [[TMP:%.+]] = load i64, ptr [[AP_CUR]], align [[CHUNKALIGN:8]]
// NEW:   [[TMP2:%.+]] = trunc i64 [[TMP]] to i32
// NEW:   store i32 [[TMP2]], ptr [[PROMOTION_TEMP]], align 4
// NEW:   [[ARG:%.+]] = load i32, ptr [[PROMOTION_TEMP]], align 4
// ALL:   store i32 [[ARG]], ptr [[V]], align 4
//
// ALL:   call void @llvm.va_end(ptr %va)
// ALL: }

long long test_i64(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  long long v = va_arg(va, long long);
  va_end(va);

  return v;
}

// ALL-LABEL: define{{.*}} i64 @test_i64(ptr{{.*}} %fmt, ...)
//
// ALL:   %va = alloca ptr, align [[$PTRALIGN]]
// ALL:   call void @llvm.va_start(ptr %va)
// ALL:   [[AP_CUR:%.+]] = load ptr, ptr %va, align [[$PTRALIGN]]
//
// i64 is 8-byte aligned, while this is within O32's stack alignment there's no
// guarantee that the offset is still 8-byte aligned after earlier reads.
// O32:   [[TMP1:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], i32 7
// O32:   [[AP_CUR:%.+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP1]], i32 -8)
//
// ALL:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], [[$INTPTR_T]] 8
// ALL:   store ptr [[AP_NEXT]], ptr %va, align [[$PTRALIGN]]
//
// ALL:   [[ARG:%.+]] = load i64, ptr [[AP_CUR]], align 8
//
// ALL:   call void @llvm.va_end(ptr %va)
// ALL: }

char *test_ptr(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  char *v = va_arg(va, char *);
  va_end(va);

  return v;
}

// ALL-LABEL: define{{.*}} ptr @test_ptr(ptr{{.*}} %fmt, ...)
//
// ALL:   %va = alloca ptr, align [[$PTRALIGN]]
// ALL:   [[V:%.*]] = alloca ptr, align [[$PTRALIGN]]
// N32:   [[AP_CAST:%.+]] = alloca ptr, align 4
// ALL:   call void @llvm.va_start(ptr %va)
// ALL:   [[AP_CUR:%.+]] = load ptr, ptr %va, align [[$PTRALIGN]]
// ALL:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], [[$INTPTR_T]] [[$CHUNKSIZE]]
// ALL:   store ptr [[AP_NEXT]], ptr %va, align [[$PTRALIGN]]
//
// When the chunk size matches the pointer size, this is easy.
// Otherwise we need a promotion temporary.
// N32:   [[TMP2:%.+]] = load i64, ptr [[AP_CUR]], align 8
// N32:   [[TMP3:%.+]] = trunc i64 [[TMP2]] to i32
// N32:   [[PTR:%.+]] = inttoptr i32 [[TMP3]] to ptr
// N32:   store ptr [[PTR]], ptr [[AP_CAST]], align 4
// N32:   [[ARG:%.+]] = load ptr, ptr [[AP_CAST]], align [[$PTRALIGN]]
//
// O32:   [[ARG:%.+]] = load ptr, ptr [[AP_CUR]], align [[$PTRALIGN]]
// N64:   [[ARG:%.+]] = load ptr, ptr [[AP_CUR]], align [[$PTRALIGN]]
// ALL:   store ptr [[ARG]], ptr [[V]], align [[$PTRALIGN]]
//
// ALL:   call void @llvm.va_end(ptr %va)
// ALL: }

int test_v4i32(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  v4i32 v = va_arg(va, v4i32);
  va_end(va);

  return v[0];
}

// O32-LABEL: define{{.*}} i32 @test_v4i32(ptr{{.*}} %fmt, ...)
// N32-LABEL: define{{.*}} signext i32 @test_v4i32(ptr{{.*}} %fmt, ...)
// N64-LABEL: define{{.*}} signext i32 @test_v4i32(ptr{{.*}} %fmt, ...)
//
// ALL:   %va = alloca ptr, align [[$PTRALIGN]]
// ALL:   [[V:%.+]] = alloca <4 x i32>, align 16
// ALL:   call void @llvm.va_start(ptr %va)
// ALL:   [[AP_CUR:%.+]] = load ptr, ptr %va, align [[$PTRALIGN]]
//
// Vectors are 16-byte aligned, however the O32 ABI has a maximum alignment of
// 8-bytes since the base of the stack is 8-byte aligned.

// O32:   [[TMP1:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], i32 7
// O32:   [[AP_CUR:%.+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP1]], i32 -8)

// N32:   [[TMP1:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], i32 15
// N32:   [[AP_CUR:%.+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP1]], i32 -16)

// N64:   [[TMP1:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], i32 15
// N64:   [[AP_CUR:%.+]] = call ptr @llvm.ptrmask.p0.i64(ptr [[TMP1]], i64 -16)

//
// ALL:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, ptr [[AP_CUR]], [[$INTPTR_T]] 16
// ALL:   store ptr [[AP_NEXT]], ptr %va, align [[$PTRALIGN]]
//
// O32:   [[ARG:%.+]] = load <4 x i32>, ptr [[AP_CUR]], align 8
// N64:   [[ARG:%.+]] = load <4 x i32>, ptr [[AP_CUR]], align 16
// N32:   [[ARG:%.+]] = load <4 x i32>, ptr [[AP_CUR]], align 16
// ALL:   store <4 x i32> [[ARG]], ptr [[V]], align 16
//
// ALL:   call void @llvm.va_end(ptr %va)
// ALL:   [[VECEXT:%.+]] = extractelement <4 x i32> {{.*}}, i32 0
// ALL:   ret i32 [[VECEXT]]
// ALL: }
