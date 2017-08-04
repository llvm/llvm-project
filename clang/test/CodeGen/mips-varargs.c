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

// ALL-LABEL: define i32 @test_i32(i8*{{.*}} %fmt, ...)
//
// O32:   %va = alloca i8*, align [[$PTRALIGN:4]]
// N32:   %va = alloca i8*, align [[$PTRALIGN:4]]
// N64:   %va = alloca i8*, align [[$PTRALIGN:8]]
// ALL:   [[V:%.*]] = alloca i32, align 4
// NEW:   [[PROMOTION_TEMP:%.*]] = alloca i32, align 4
//
// ALL:   [[VA:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA]])
// ALL:   [[AP_CUR:%.+]] = load i8*, i8** %va, align [[$PTRALIGN]]
// O32:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, i8* [[AP_CUR]], [[$INTPTR_T:i32]] [[$CHUNKSIZE:4]]
// NEW:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, i8* [[AP_CUR]], [[$INTPTR_T:i32|i64]] [[$CHUNKSIZE:8]]
//
// ALL:   store i8* [[AP_NEXT]], i8** %va, align [[$PTRALIGN]]
//
// O32:   [[AP_CAST:%.+]] = bitcast i8* [[AP_CUR]] to [[CHUNK_T:i32]]*
// O32:   [[ARG:%.+]] = load i32, i32* [[AP_CAST]], align [[CHUNKALIGN:4]]
//
// N32:   [[AP_CAST:%.+]] = bitcast i8* [[AP_CUR]] to [[CHUNK_T:i64]]*
// N32:   [[TMP:%.+]] = load i64, i64* [[AP_CAST]], align [[CHUNKALIGN:8]]
// N64:   [[AP_CAST:%.+]] = bitcast i8* [[AP_CUR]] to [[CHUNK_T:i64]]*
// N64:   [[TMP:%.+]] = load i64, i64* [[AP_CAST]], align [[CHUNKALIGN:8]]
// NEW:   [[TMP2:%.+]] = trunc i64 [[TMP]] to i32
// NEW:   store i32 [[TMP2]], i32* [[PROMOTION_TEMP]], align 4
// NEW:   [[ARG:%.+]] = load i32, i32* [[PROMOTION_TEMP]], align 4
// ALL:   store i32 [[ARG]], i32* [[V]], align 4
//
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL: }

long long test_i64(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  long long v = va_arg(va, long long);
  va_end(va);

  return v;
}

// ALL-LABEL: define i64 @test_i64(i8*{{.*}} %fmt, ...)
//
// ALL:   %va = alloca i8*, align [[$PTRALIGN]]
// ALL:   [[VA:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA]])
// ALL:   [[AP_CUR:%.+]] = load i8*, i8** %va, align [[$PTRALIGN]]
//
// i64 is 8-byte aligned, while this is within O32's stack alignment there's no
// guarantee that the offset is still 8-byte aligned after earlier reads.
// O32:   [[TMP1:%.+]] = ptrtoint i8* [[AP_CUR]] to i32
// O32:   [[TMP2:%.+]] = add i32 [[TMP1]], 7
// O32:   [[TMP3:%.+]] = and i32 [[TMP2]], -8
// O32:   [[AP_CUR:%.+]] = inttoptr i32 [[TMP3]] to i8*
//
// ALL:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, i8* [[AP_CUR]], [[$INTPTR_T]] 8
// ALL:   store i8* [[AP_NEXT]], i8** %va, align [[$PTRALIGN]]
//
// ALL:   [[AP_CAST:%.*]] = bitcast i8* [[AP_CUR]] to i64*
// ALL:   [[ARG:%.+]] = load i64, i64* [[AP_CAST]], align 8
//
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL: }

char *test_ptr(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  char *v = va_arg(va, char *);
  va_end(va);

  return v;
}

// ALL-LABEL: define i8* @test_ptr(i8*{{.*}} %fmt, ...)
//
// ALL:   %va = alloca i8*, align [[$PTRALIGN]]
// ALL:   [[V:%.*]] = alloca i8*, align [[$PTRALIGN]]
// N32:   [[AP_CAST:%.+]] = alloca i8*, align 4
// ALL:   [[VA:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA]])
// ALL:   [[AP_CUR:%.+]] = load i8*, i8** %va, align [[$PTRALIGN]]
// ALL:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, i8* [[AP_CUR]], [[$INTPTR_T]] [[$CHUNKSIZE]]
// ALL:   store i8* [[AP_NEXT]], i8** %va, align [[$PTRALIGN]]
//
// When the chunk size matches the pointer size, this is easy.
// O32:   [[AP_CAST:%.+]] = bitcast i8* [[AP_CUR]] to i8**
// N64:   [[AP_CAST:%.+]] = bitcast i8* [[AP_CUR]] to i8**
// Otherwise we need a promotion temporary.
// N32:   [[TMP1:%.+]] = bitcast i8* [[AP_CUR]] to i64*
// N32:   [[TMP2:%.+]] = load i64, i64* [[TMP1]], align 8
// N32:   [[TMP3:%.+]] = trunc i64 [[TMP2]] to i32
// N32:   [[PTR:%.+]] = inttoptr i32 [[TMP3]] to i8*
// N32:   store i8* [[PTR]], i8** [[AP_CAST]], align 4
//
// ALL:   [[ARG:%.+]] = load i8*, i8** [[AP_CAST]], align [[$PTRALIGN]]
// ALL:   store i8* [[ARG]], i8** [[V]], align [[$PTRALIGN]]
//
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL: }

int test_v4i32(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  v4i32 v = va_arg(va, v4i32);
  va_end(va);

  return v[0];
}

// ALL-LABEL: define i32 @test_v4i32(i8*{{.*}} %fmt, ...)
//
// ALL:   %va = alloca i8*, align [[$PTRALIGN]]
// ALL:   [[V:%.+]] = alloca <4 x i32>, align 16
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA1]])
// ALL:   [[AP_CUR:%.+]] = load i8*, i8** %va, align [[$PTRALIGN]]
//
// Vectors are 16-byte aligned, however the O32 ABI has a maximum alignment of
// 8-bytes since the base of the stack is 8-byte aligned.
// O32:   [[TMP1:%.+]] = ptrtoint i8* [[AP_CUR]] to i32
// O32:   [[TMP2:%.+]] = add i32 [[TMP1]], 7
// O32:   [[TMP3:%.+]] = and i32 [[TMP2]], -8
// O32:   [[AP_CUR:%.+]] = inttoptr i32 [[TMP3]] to i8*
//
// NEW:   [[TMP1:%.+]] = ptrtoint i8* [[AP_CUR]] to [[$INTPTR_T]]
// NEW:   [[TMP2:%.+]] = add [[$INTPTR_T]] [[TMP1]], 15
// NEW:   [[TMP3:%.+]] = and [[$INTPTR_T]] [[TMP2]], -16
// NEW:   [[AP_CUR:%.+]] = inttoptr [[$INTPTR_T]] [[TMP3]] to i8*
//
// ALL:   [[AP_NEXT:%.+]] = getelementptr inbounds i8, i8* [[AP_CUR]], [[$INTPTR_T]] 16
// ALL:   store i8* [[AP_NEXT]], i8** %va, align [[$PTRALIGN]]
//
// ALL:   [[AP_CAST:%.+]] = bitcast i8* [[AP_CUR]] to <4 x i32>*
// O32:   [[ARG:%.+]] = load <4 x i32>, <4 x i32>* [[AP_CAST]], align 8
// N64:   [[ARG:%.+]] = load <4 x i32>, <4 x i32>* [[AP_CAST]], align 16
// N32:   [[ARG:%.+]] = load <4 x i32>, <4 x i32>* [[AP_CAST]], align 16
// ALL:   store <4 x i32> [[ARG]], <4 x i32>* [[V]], align 16
//
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL:   [[VECEXT:%.+]] = extractelement <4 x i32> {{.*}}, i32 0
// ALL:   ret i32 [[VECEXT]]
// ALL: }
