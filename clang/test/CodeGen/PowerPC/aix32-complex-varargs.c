// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

void testva (int n, ...)
{
  va_list ap;

  _Complex int i   = va_arg(ap, _Complex int);
// CHECK:  %[[VAR40:[A-Za-z0-9.]+]] = load ptr, ptr %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR41:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR40]]
// CHECK-NEXT:  store ptr %[[VAR41]], ptr %[[VAR100]], align 4
// CHECK-NEXT:  %[[VAR6:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[VAR40]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR7:[A-Za-z0-9.]+]] = load i32, ptr %[[VAR6]]
// CHECK-NEXT:  %[[VAR8:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[VAR40]], i32 0, i32 1
// CHECK-NEXT:  %[[VAR9:[A-Za-z0-9.]+]] = load i32, ptr %[[VAR8]]
// CHECK-NEXT:  %[[VAR10:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[VARINT:[A-Za-z0-9.]+]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR11:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[VARINT]], i32 0, i32 1
// CHECK-NEXT:  store i32 %[[VAR7]], ptr %[[VAR10]]
// CHECK-NEXT:  store i32 %[[VAR9]], ptr %[[VAR11]]

  _Complex short s = va_arg(ap, _Complex short);
// CHECK:  %[[VAR50:[A-Za-z0-9.]+]] = load ptr, ptr %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR51:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR50]]
// CHECK-NEXT:  store ptr %[[VAR51]], ptr %[[VAR100]], align 4
// CHECK-NEXT:  %[[VAR12:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR50]], i32 2
// CHECK-NEXT:  %[[VAR13:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR50]], i32 6
// CHECK-NEXT:  %[[VAR16:[A-Za-z0-9.]+]] = load i16, ptr %[[VAR12]], align 2
// CHECK-NEXT:  %[[VAR17:[A-Za-z0-9.]+]] = load i16, ptr %[[VAR13]], align 2
// CHECK-NEXT:  %[[VAR18:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i16, i16 }, ptr %[[VAR19:[A-Za-z0-9.]+]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR20:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i16, i16 }, ptr %[[VAR19]], i32 0, i32 1
// CHECK-NEXT:  store i16 %[[VAR16]], ptr %[[VAR18]]
// CHECK-NEXT:  store i16 %[[VAR17]], ptr %[[VAR20]]


  _Complex char c  = va_arg(ap, _Complex char);
// CHECK:  %[[VAR60:[A-Za-z0-9.]+]] = load ptr, ptr %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR61:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR60]]
// CHECK-NEXT:  store ptr %[[VAR61]], ptr %[[VAR100]], align 4
// CHECK-NEXT:  %[[VAR21:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR60]], i32 3
// CHECK-NEXT:  %[[VAR22:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR60]], i32 7
// CHECK-NEXT:  %[[VAR23:[A-Za-z0-9.]+]] = load i8, ptr %[[VAR21]]
// CHECK-NEXT:  %[[VAR24:[A-Za-z0-9.]+]] = load i8, ptr %[[VAR22]]
// CHECK-NEXT:  %[[VAR25:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i8, i8 }, ptr %[[VAR26:[A-Za-z0-9.]+]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR27:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { i8, i8 }, ptr %[[VAR26]], i32 0, i32 1
// CHECK-NEXT:  store i8 %[[VAR23]], ptr %[[VAR25]]
// CHECK-NEXT:  store i8 %[[VAR24]], ptr %[[VAR27]]


  _Complex float f = va_arg(ap, _Complex float);
// CHECK:  %[[VAR70:[A-Za-z0-9.]+]] = getelementptr inbounds i8, ptr %[[VAR71:[A-Za-z0-9.]+]], i32 8
// CHECK-NEXT:  store ptr %[[VAR70]], ptr %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR29:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { float, float }, ptr %[[VAR71]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR30:[A-Za-z0-9.]+]] = load float, ptr %[[VAR29]]
// CHECK-NEXT:  %[[VAR31:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { float, float }, ptr %[[VAR71]], i32 0, i32 1
// CHECK-NEXT:  %[[VAR32:[A-Za-z0-9.]+]] = load float, ptr %[[VAR31]]
// CHECK-NEXT:  %[[VAR33:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { float, float }, ptr %f, i32 0, i32 0
// CHECK-NEXT:  %[[VAR34:[A-Za-z0-9.]+]] = getelementptr inbounds nuw { float, float }, ptr %f, i32 0, i32 1
// CHECK-NEXT:  store float %[[VAR30]], ptr %[[VAR33]]
// CHECK-NEXT:  store float %[[VAR32]], ptr %[[VAR34]]
}
