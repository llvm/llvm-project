// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-openbsd -fblocks -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

id testObject(va_list ap) {
  return va_arg(ap, id);
}
// CHECK: @testObject
// CHECK: using_regs:
// CHECK-NEXT: getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{[0-9]+}}, i32 0, i32 4
// CHECK-NEXT: load i8*, i8** %{{[0-9]+}}, align 4
// CHECK-NEXT: mul i8 %numUsedRegs, 4
// CHECK-NEXT: getelementptr inbounds i8, i8* %{{[0-9]+}}, i8 %{{[0-9]+}}
// CHECK-NEXT: bitcast i8* %{{[0-9]+}} to i8**
// CHECK-NEXT: add i8 %numUsedRegs, 1
// CHECK-NEXT: store i8 %{{[0-9]+}}, i8* %gpr, align 4
// CHECK-NEXT: br label %cont

typedef void (^block)(void);
block testBlock(va_list ap) {
  return va_arg(ap, block);
}
// CHECK: @testBlock
// CHECK: using_regs:
// CHECK-NEXT: getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{[0-9]+}}, i32 0, i32 4
// CHECK-NEXT: load i8*, i8** %{{[0-9]+}}, align 4
// CHECK-NEXT: mul i8 %numUsedRegs, 4
// CHECK-NEXT: getelementptr inbounds i8, i8* %{{[0-9]+}}, i8 %{{[0-9]+}}
// CHECK-NEXT: bitcast i8* %{{[0-9]+}} to void ()**
// CHECK-NEXT: add i8 %numUsedRegs, 1
// CHECK-NEXT: store i8 %{{[0-9]+}}, i8* %gpr, align 4
// CHECK-NEXT: br label %cont
