// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

void testva (int n, ...)
{
  va_list ap;

  _Complex int i   = va_arg(ap, _Complex int);
  // CHECK: %[[VAR40:[A-Za-z0-9.]+]] = load i8*, i8** %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR41:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR40]], i64 16
  // CHECK-NEXT: store i8* %[[VAR41]], i8** %[[VAR100]]
  // CHECK-NEXT: %[[VAR3:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR40]], i64 8
  // CHECK-NEXT: %[[VAR4:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR40]] to i32*
  // CHECK-NEXT: %[[VAR5:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR3]] to i32*
  // CHECK-NEXT: %[[VAR6:[A-Za-z0-9.]+]] = load i32, i32* %[[VAR4]], align 8
  // CHECK-NEXT: %[[VAR7:[A-Za-z0-9.]+]] = load i32, i32* %[[VAR5]], align 8
  // CHECK-NEXT: %[[VAR8:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* %[[VAR0:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR9:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* %[[VAR0]], i32 0, i32 1
  // CHECK-NEXT: store i32 %[[VAR6]], i32* %[[VAR8]]
  // CHECK-NEXT: store i32 %[[VAR7]], i32* %[[VAR9]]

  _Complex short s = va_arg(ap, _Complex short);
  // CHECK: %[[VAR50:[A-Za-z0-9.]+]] = load i8*, i8** %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR51:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR50]], i64 16
  // CHECK-NEXT: store i8* %[[VAR51]], i8** %[[VAR100]]
  // CHECK-NEXT: %[[VAR13:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR50]], i64 8
  // CHECK-NEXT: %[[VAR14:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR50]] to i16*
  // CHECK-NEXT: %[[VAR15:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR13]] to i16*
  // CHECK-NEXT: %[[VAR16:[A-Za-z0-9.]+]] = load i16, i16* %[[VAR14]], align 8
  // CHECK-NEXT: %[[VAR17:[A-Za-z0-9.]+]] = load i16, i16* %[[VAR15]], align 8
  // CHECK-NEXT: %[[VAR18:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, { i16, i16 }* %[[VAR10:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR19:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, { i16, i16 }* %[[VAR10]], i32 0, i32 1
  // CHECK-NEXT: store i16 %[[VAR16]], i16* %[[VAR18]]
  // CHECK-NEXT: store i16 %[[VAR17]], i16* %[[VAR19]]

  _Complex char c  = va_arg(ap, _Complex char);
  // CHECK: %[[VAR60:[A-Za-z0-9.]+]] = load i8*, i8** %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR61:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR60]], i64 16
  // CHECK-NEXT: store i8* %[[VAR61]], i8** %[[VAR100]]
  // CHECK-NEXT: %[[VAR25:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR60]], i64 8
  // CHECK-NEXT: %[[VAR26:[A-Za-z0-9.]+]] = load i8, i8* %[[VAR60]], align 8
  // CHECK-NEXT: %[[VAR27:[A-Za-z0-9.]+]] = load i8, i8* %[[VAR25]], align 8
  // CHECK-NEXT: %[[VAR28:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %[[VAR20:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR29:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %[[VAR20]], i32 0, i32 1
  // CHECK-NEXT: store i8 %[[VAR26]], i8* %[[VAR28]]
  // CHECK-NEXT: store i8 %[[VAR27]], i8* %[[VAR29]]

  _Complex float f = va_arg(ap, _Complex float);
  // CHECK: %[[VAR70:[A-Za-z0-9.]+]] = load i8*, i8** %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR71:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR70]], i64 16
  // CHECK-NEXT: store i8* %[[VAR71]], i8** %[[VAR100]]
  // CHECK-NEXT: %[[VAR33:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR70]], i64 8
  // CHECK-NEXT: %[[VAR34:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR70]] to float*
  // CHECK-NEXT: %[[VAR35:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR33]] to float*
  // CHECK-NEXT: %[[VAR36:[A-Za-z0-9.]+]] = load float, float* %[[VAR34]], align 8
  // CHECK-NEXT: %[[VAR37:[A-Za-z0-9.]+]] = load float, float* %[[VAR35]], align 8
  // CHECK-NEXT: %[[VAR38:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, { float, float }* %[[VAR30:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR39:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, { float, float }* %[[VAR30]], i32 0, i32 1
  // CHECK-NEXT: store float %[[VAR36]], float* %[[VAR38]]
  // CHECK-NEXT: store float %[[VAR37]], float* %[[VAR39]]
}
