// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-ibm-aix-xcoff -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

void testva (int n, ...)
{
  va_list ap;

  _Complex int i   = va_arg(ap, _Complex int);
// CHECK:  %[[VAR40:[A-Za-z0-9.]+]] = load i8*, i8** %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR41:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR40]]
// CHECK-NEXT:  store i8* %[[VAR41]], i8** %[[VAR100]], align 4
// CHECK-NEXT:  %[[VAR4:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR40]] to { i32, i32 }*
// CHECK-NEXT:  %[[VAR6:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* %[[VAR4]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR7:[A-Za-z0-9.]+]] = load i32, i32* %[[VAR6]]
// CHECK-NEXT:  %[[VAR8:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* %[[VAR4]], i32 0, i32 1
// CHECK-NEXT:  %[[VAR9:[A-Za-z0-9.]+]] = load i32, i32* %[[VAR8]]
// CHECK-NEXT:  %[[VAR10:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* %[[VARINT:[A-Za-z0-9.]+]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR11:[A-Za-z0-9.]+]] = getelementptr inbounds { i32, i32 }, { i32, i32 }* %[[VARINT]], i32 0, i32 1
// CHECK-NEXT:  store i32 %[[VAR7]], i32* %[[VAR10]]
// CHECK-NEXT:  store i32 %[[VAR9]], i32* %[[VAR11]]

  _Complex short s = va_arg(ap, _Complex short);
// CHECK:  %[[VAR50:[A-Za-z0-9.]+]] = load i8*, i8** %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR51:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR50]]
// CHECK-NEXT:  store i8* %[[VAR51]], i8** %[[VAR100]], align 4
// CHECK-NEXT:  %[[VAR12:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR50]], i32 2
// CHECK-NEXT:  %[[VAR13:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR50]], i32 6
// CHECK-NEXT:  %[[VAR14:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR12]] to i16*
// CHECK-NEXT:  %[[VAR15:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR13]] to i16*
// CHECK-NEXT:  %[[VAR16:[A-Za-z0-9.]+]] = load i16, i16* %[[VAR14]], align 2
// CHECK-NEXT:  %[[VAR17:[A-Za-z0-9.]+]] = load i16, i16* %[[VAR15]], align 2
// CHECK-NEXT:  %[[VAR18:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, { i16, i16 }* %[[VAR19:[A-Za-z0-9.]+]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR20:[A-Za-z0-9.]+]] = getelementptr inbounds { i16, i16 }, { i16, i16 }* %[[VAR19]], i32 0, i32 1
// CHECK-NEXT:  store i16 %[[VAR16]], i16* %[[VAR18]]
// CHECK-NEXT:  store i16 %[[VAR17]], i16* %[[VAR20]]


  _Complex char c  = va_arg(ap, _Complex char);
// CHECK:  %[[VAR60:[A-Za-z0-9.]+]] = load i8*, i8** %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR61:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR60]]
// CHECK-NEXT:  store i8* %[[VAR61]], i8** %[[VAR100]], align 4
// CHECK-NEXT:  %[[VAR21:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR60]], i32 3
// CHECK-NEXT:  %[[VAR22:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR60]], i32 7
// CHECK-NEXT:  %[[VAR23:[A-Za-z0-9.]+]] = load i8, i8* %[[VAR21]]
// CHECK-NEXT:  %[[VAR24:[A-Za-z0-9.]+]] = load i8, i8* %[[VAR22]]
// CHECK-NEXT:  %[[VAR25:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %[[VAR26:[A-Za-z0-9.]+]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR27:[A-Za-z0-9.]+]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %[[VAR26]], i32 0, i32 1
// CHECK-NEXT:  store i8 %[[VAR23]], i8* %[[VAR25]]
// CHECK-NEXT:  store i8 %[[VAR24]], i8* %[[VAR27]]


  _Complex float f = va_arg(ap, _Complex float);
// CHECK:  %[[VAR70:[A-Za-z0-9.]+]] = getelementptr inbounds i8, i8* %[[VAR71:[A-Za-z0-9.]+]], i32 8
// CHECK-NEXT:  store i8* %[[VAR70]], i8** %[[VAR100:[A-Za-z0-9.]+]]
// CHECK-NEXT:  %[[VAR28:[A-Za-z0-9.]+]] = bitcast i8* %[[VAR71]] to { float, float }*
// CHECK-NEXT:  %[[VAR29:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, { float, float }* %[[VAR28]], i32 0, i32 0
// CHECK-NEXT:  %[[VAR30:[A-Za-z0-9.]+]] = load float, float* %[[VAR29]]
// CHECK-NEXT:  %[[VAR31:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, { float, float }* %[[VAR28]], i32 0, i32 1
// CHECK-NEXT:  %[[VAR32:[A-Za-z0-9.]+]] = load float, float* %[[VAR31]]
// CHECK-NEXT:  %[[VAR33:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, { float, float }* %f, i32 0, i32 0
// CHECK-NEXT:  %[[VAR34:[A-Za-z0-9.]+]] = getelementptr inbounds { float, float }, { float, float }* %f, i32 0, i32 1
// CHECK-NEXT:  store float %[[VAR30]], float* %[[VAR33]]
// CHECK-NEXT:  store float %[[VAR32]], float* %[[VAR34]]
}
