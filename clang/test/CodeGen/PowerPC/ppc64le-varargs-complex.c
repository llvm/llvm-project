// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

void testva (int n, ...)
{
  va_list ap;

  _Complex int i   = va_arg(ap, _Complex int);
  // CHECK: %[[VAR40:[A-Za-z0-9.]+]] = load ptr, ptr %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR41:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR40]], i64 16
  // CHECK-NEXT: store ptr %[[VAR41]], ptr %[[VAR100]]
  // CHECK-NEXT: %[[VAR3:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR40]], i64 8
  // CHECK-NEXT: %[[VAR6:[A-Za-z0-9.]+]] = load i32, ptr %[[VAR40]], align 8
  // CHECK-NEXT: %[[VAR7:[A-Za-z0-9.]+]] = load i32, ptr %[[VAR3]], align 8
  // CHECK-NEXT: %[[VAR8:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, ptr %[[VAR0:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR9:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, ptr %[[VAR0]], i32 0, i32 1
  // CHECK-NEXT: store i32 %[[VAR6]], ptr %[[VAR8]]
  // CHECK-NEXT: store i32 %[[VAR7]], ptr %[[VAR9]]

  _Complex short s = va_arg(ap, _Complex short);
  // CHECK: %[[VAR50:[A-Za-z0-9.]+]] = load ptr, ptr %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR51:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR50]], i64 16
  // CHECK-NEXT: store ptr %[[VAR51]], ptr %[[VAR100]]
  // CHECK-NEXT: %[[VAR13:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR50]], i64 8
  // CHECK-NEXT: %[[VAR16:[A-Za-z0-9.]+]] = load i16, ptr %[[VAR50]], align 8
  // CHECK-NEXT: %[[VAR17:[A-Za-z0-9.]+]] = load i16, ptr %[[VAR13]], align 8
  // CHECK-NEXT: %[[VAR18:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, ptr %[[VAR10:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR19:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, ptr %[[VAR10]], i32 0, i32 1
  // CHECK-NEXT: store i16 %[[VAR16]], ptr %[[VAR18]]
  // CHECK-NEXT: store i16 %[[VAR17]], ptr %[[VAR19]]

  _Complex char c  = va_arg(ap, _Complex char);
  // CHECK: %[[VAR60:[A-Za-z0-9.]+]] = load ptr, ptr %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR61:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR60]], i64 16
  // CHECK-NEXT: store ptr %[[VAR61]], ptr %[[VAR100]]
  // CHECK-NEXT: %[[VAR25:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR60]], i64 8
  // CHECK-NEXT: %[[VAR26:[A-Za-z0-9.]+]] = load i8, ptr %[[VAR60]], align 8
  // CHECK-NEXT: %[[VAR27:[A-Za-z0-9.]+]] = load i8, ptr %[[VAR25]], align 8
  // CHECK-NEXT: %[[VAR28:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, ptr %[[VAR20:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR29:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, ptr %[[VAR20]], i32 0, i32 1
  // CHECK-NEXT: store i8 %[[VAR26]], ptr %[[VAR28]]
  // CHECK-NEXT: store i8 %[[VAR27]], ptr %[[VAR29]]

  _Complex float f = va_arg(ap, _Complex float);
  // CHECK: %[[VAR70:[A-Za-z0-9.]+]] = load ptr, ptr %[[VAR100:[A-Za-z0-9.]+]]
  // CHECK-NEXT: %[[VAR71:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR70]], i64 16
  // CHECK-NEXT: store ptr %[[VAR71]], ptr %[[VAR100]]
  // CHECK-NEXT: %[[VAR33:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR70]], i64 8
  // CHECK-NEXT: %[[VAR36:[A-Za-z0-9.]+]] = load float, ptr %[[VAR70]], align 8
  // CHECK-NEXT: %[[VAR37:[A-Za-z0-9.]+]] = load float, ptr %[[VAR33]], align 8
  // CHECK-NEXT: %[[VAR38:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, ptr %[[VAR30:[A-Za-z0-9.]+]], i32 0, i32 0
  // CHECK-NEXT: %[[VAR39:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, ptr %[[VAR30]], i32 0, i32 1
  // CHECK-NEXT: store float %[[VAR36]], ptr %[[VAR38]]
  // CHECK-NEXT: store float %[[VAR37]], ptr %[[VAR39]]
}
