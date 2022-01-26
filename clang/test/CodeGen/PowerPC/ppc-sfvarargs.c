// RUN: %clang -O0 --target=powerpc-unknown-linux-gnu -EB -msoft-float -S -emit-llvm %s -o - | FileCheck %s

#include <stdarg.h>
void test(char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  va_arg(ap, double);
  va_end(ap);
}

void foo() {
  double a;
  test("test",a);
}
// CHECK: %{{[0-9]+}} = add i8 %{{[0-9]+|numUsedRegs}}, 1
// CHECK: %{{[0-9]+}} = and i8 %{{[0-9]+}}, -2
// CHECK: %{{[0-9]+}} = mul i8 %{{[0-9]+}}, 4
