// RUN:  %clang_cc1 -triple aarch64_be-linux-gnu -ffreestanding -emit-llvm -O0 -o - %s | FileCheck %s

#include <stdarg.h>

// A single member HFA must be aligned just like a non-HFA register argument.
double callee(int a, ...) {
// CHECK: [[REGPP:%.*]] = getelementptr inbounds %"struct.std::__va_list", %"struct.std::__va_list"* [[VA:%.*]], i32 0, i32 2
// CHECK: [[REGP:%.*]] = load i8*, i8** [[REGPP]], align 8
// CHECK: [[OFFSET0:%.*]] = getelementptr inbounds i8, i8* [[REGP]], i32 {{.*}}
// CHECK: [[OFFSET1:%.*]] = getelementptr inbounds i8, i8* [[OFFSET0]], i64 8

// CHECK: [[MEMPP:%.*]] = getelementptr inbounds %"struct.std::__va_list", %"struct.std::__va_list"* [[VA:%.*]], i32 0, i32 0
// CHECK: [[MEMP:%.*]] = load i8*, i8** [[MEMPP]], align 8
// CHECK: [[NEXTP:%.*]] = getelementptr inbounds i8, i8* [[MEMP]], i64 8
// CHECK: store i8* [[NEXTP]], i8** [[MEMPP]], align 8
  va_list vl;
  va_start(vl, a);
  double result = va_arg(vl, struct { double a; }).a;
  va_end(vl);
  return result;
}
