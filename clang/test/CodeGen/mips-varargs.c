// RUN: %clang_cc1 -triple mips-unknown-linux -O3 -o - -emit-llvm %s | FileCheck %s -check-prefix=ALL -check-prefix=O32
// RUN: %clang_cc1 -triple mips64-unknown-linux -O3 -o - -emit-llvm  -target-abi n32 %s | FileCheck %s -check-prefix=ALL -check-prefix=N32
// RUN: %clang_cc1 -triple mips64-unknown-linux -O3 -o - -emit-llvm %s | FileCheck %s -check-prefix=ALL -check-prefix=N64

#include <stdarg.h>

typedef int v4i32 __attribute__ ((__vector_size__ (16)));

int test_v4i32(char *fmt, ...) {
  va_list va;

  va_start(va, fmt);
  v4i32 v = va_arg(va, v4i32);
  va_end(va);

  return v[0];
}

// ALL: define i32 @test_v4i32(i8*{{.*}} %fmt, ...)
//
// O32:   %va = alloca i8*, align [[PTRALIGN:4]]
// N32:   %va = alloca i8*, align [[PTRALIGN:4]]
// N64:   %va = alloca i8*, align [[PTRALIGN:8]]
//
// ALL:   [[VA1:%.+]] = bitcast i8** %va to i8*
// ALL:   call void @llvm.va_start(i8* [[VA1]])
// ALL:   [[AP_CUR:%.+]] = load i8** %va, align [[PTRALIGN]]
//
// O32:   [[PTR0:%.+]] = ptrtoint i8* [[AP_CUR]] to [[PTRTYPE:i32]]
// N32:   [[PTR0:%.+]] = ptrtoint i8* [[AP_CUR]] to [[PTRTYPE:i32]]
// N64:   [[PTR0:%.+]] = ptrtoint i8* [[AP_CUR]] to [[PTRTYPE:i64]]
//
// Vectors are 16-byte aligned, however the O32 ABI has a maximum alignment of
// 8-bytes since the base of the stack is 8-byte aligned.
// O32:   [[PTR1:%.+]] = add i32 [[PTR0]], 7
// O32:   [[PTR2:%.+]] = and i32 [[PTR1]], -8
//
// N32:   [[PTR1:%.+]] = add i32 [[PTR0]], 15
// N32:   [[PTR2:%.+]] = and i32 [[PTR1]], -16
//
// N64:   [[PTR1:%.+]] = add i64 [[PTR0]], 15
// N64:   [[PTR2:%.+]] = and i64 [[PTR1]], -16
//
// ALL:   [[PTR3:%.+]] = inttoptr [[PTRTYPE]] [[PTR2]] to <4 x i32>*
// ALL:   [[PTR4:%.+]] = inttoptr [[PTRTYPE]] [[PTR2]] to i8*
// ALL:   [[AP_NEXT:%.+]] = getelementptr i8* [[PTR4]], [[PTRTYPE]] 16
// ALL:   store i8* [[AP_NEXT]], i8** %va, align [[PTRALIGN]]
// ALL:   [[PTR5:%.+]] = load <4 x i32>* [[PTR3]], align 16
// ALL:   call void @llvm.va_end(i8* [[VA1]])
// ALL:   [[VECEXT:%.+]] = extractelement <4 x i32> [[PTR5]], i32 0
// ALL:   ret i32 [[VECEXT]]
// ALL: }
