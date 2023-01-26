// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-openbsd -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

class something;
typedef void (something::*method)();

method test(va_list ap) {
  return va_arg(ap, method);
}
// CHECK: using_regs:
// CHECK-NEXT: getelementptr inbounds %struct.__va_list_tag, ptr %{{[0-9]+}}, i32 0, i32 4
// CHECK-NEXT: load ptr, ptr %{{[0-9]+}}, align 4
// CHECK-NEXT: mul i8 %numUsedRegs, 4
// CHECK-NEXT: getelementptr inbounds i8, ptr %{{[0-9]+}}, i8 %{{[0-9]+}}
// CHECK-NEXT: add i8 %numUsedRegs, 1
// CHECK-NEXT: store i8 %{{[0-9]+}}, ptr %gpr, align 4
// CHECK-NEXT: br label %cont
