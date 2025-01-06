// RUN:  %clang_cc1 -triple aarch64_be-linux-gnu -ffreestanding -emit-llvm -O0 -o - %s | FileCheck %s

#include <stdarg.h>

// A single member HFA must be aligned just like a non-HFA register argument.
double callee(int a, ...) {
// CHECK: [[REGPP:%.*]] = getelementptr inbounds nuw %struct.__va_list, ptr [[VA:%.*]], i32 0, i32 2
// CHECK: [[REGP:%.*]] = load ptr, ptr [[REGPP]], align 8
// CHECK: [[OFFSET0:%.*]] = getelementptr inbounds i8, ptr [[REGP]], i32 {{.*}}
// CHECK: [[OFFSET1:%.*]] = getelementptr inbounds i8, ptr [[OFFSET0]], i64 8

// CHECK: [[MEMPP:%.*]] = getelementptr inbounds nuw %struct.__va_list, ptr [[VA:%.*]], i32 0, i32 0
// CHECK: [[MEMP:%.*]] = load ptr, ptr [[MEMPP]], align 8
// CHECK: [[NEXTP:%.*]] = getelementptr inbounds i8, ptr [[MEMP]], i64 8
// CHECK: store ptr [[NEXTP]], ptr [[MEMPP]], align 8
  va_list vl;
  va_start(vl, a);
  double result = va_arg(vl, struct { double a; }).a;
  va_end(vl);
  return result;
}
