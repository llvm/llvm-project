// RUN: %clang_cc1 -triple arm64-windows-msvc -emit-llvm -o - %s | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-A64
// RUN: %clang_cc1 -triple arm64ec-windows-msvc -emit-llvm -o - %s | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-EC

#include <stdarg.h>

int simple_int(va_list ap) {
// CHECK-LABEL: define dso_local i32 @simple_int
  return va_arg(ap, int);
// CHECK: [[RESULT:%[a-z_0-9]+]] = load i32, ptr %argp.cur
// CHECK: ret i32 [[RESULT]]
}

struct bigstruct {
  int item[4];
};
struct bigstruct big_struct(va_list ap) {
// CHECK-LABEL: define dso_local [2 x i64] @big_struct
  return va_arg(ap, struct bigstruct);
// CHECK-EC: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i64 8
// CHECK-EC: [[PTR:%[0-9]+]] = load ptr, ptr %argp.cur, align 8
// CHECK-EC: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 [[PTR]], i64 16, i1 false)
// CHECK-A64: %argp.next = getelementptr inbounds i8, ptr %argp.cur, i64 16
// CHECK-A64: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 8 %argp.cur, i64 16, i1 false)
// CHECK: [[RESULT:%[0-9]+]] = load [2 x i64], ptr %coerce.dive
// CHECK: ret [2 x i64] [[RESULT]]
}
