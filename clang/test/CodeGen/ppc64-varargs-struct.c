// REQUIRES: ppc64-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

struct x {
  long a;
  double b;
};

void testva (int n, ...)
{
  va_list ap;

  struct x t = va_arg (ap, struct x);
// CHECK: bitcast i8* %{{[a-z.0-9]*}} to %struct.x*
// CHECK: bitcast %struct.x* %t to i8*
// CHECK: bitcast %struct.x* %{{[0-9]+}} to i8*
// CHECK: call void @llvm.memcpy

  int v = va_arg (ap, int);
// CHECK: ptrtoint i8* %{{[a-z.0-9]*}} to i64
// CHECK: add i64 %{{[0-9]+}}, 4
// CHECK: inttoptr i64 %{{[0-9]+}} to i8*
// CHECK: bitcast i8* %{{[0-9]+}} to i32*

  __int128_t u = va_arg (ap, __int128_t);
// CHECK: bitcast i8* %{{[a-z.0-9]+}} to i128*
// CHECK-NEXT: load i128* %{{[0-9]+}}
}
