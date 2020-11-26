// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 -mabi=ieeelongdouble \
// RUN:   -o - %s | FileCheck %s -check-prefix=IEEE
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 \
// RUN:   -o - %s | FileCheck %s -check-prefix=IBM

#include <stdarg.h>

// IEEE-LABEL: define fp128 @f128(i32 signext %n, ...)
// IEEE: call void @llvm.va_start(i8* %{{[0-9a-zA-Z_.]+}})
// IEEE: %[[P1:[0-9a-zA-Z_.]+]] = add i64 %{{[0-9a-zA-Z_.]+}}, 15
// IEEE: %[[P2:[0-9a-zA-Z_.]+]] = and i64 %[[P1]], -16
// IEEE: %[[P3:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[P2]] to i8*
// IEEE: %[[P4:[0-9a-zA-Z_.]+]] = bitcast i8* %[[P3]] to fp128*
// IEEE: %{{[0-9a-zA-Z_.]+}} = load fp128, fp128* %[[P4]], align 16
// IEEE: call void @llvm.va_end(i8* %{{[0-9a-zA-Z_.]+}})
__float128 f128(int n, ...) {
  va_list ap;
  va_start(ap, n);
  __float128 x = va_arg(ap, __float128);
  va_end(ap);
  return x;
}

// IEEE-LABEL: define fp128 @long_double(i32 signext %n, ...)
// IEEE: call void @llvm.va_start(i8* %{{[0-9a-zA-Z_.]+}})
// IEEE: %[[P1:[0-9a-zA-Z_.]+]] = add i64 %{{[0-9a-zA-Z_.]+}}, 15
// IEEE: %[[P2:[0-9a-zA-Z_.]+]] = and i64 %[[P1]], -16
// IEEE: %[[P3:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[P2]] to i8*
// IEEE: %[[P4:[0-9a-zA-Z_.]+]] = bitcast i8* %[[P3]] to fp128*
// IEEE: %{{[0-9a-zA-Z_.]+}} = load fp128, fp128* %[[P4]], align 16
// IEEE: call void @llvm.va_end(i8* %{{[0-9a-zA-Z_.]+}})

// IBM-LABEL: define ppc_fp128 @long_double(i32 signext %n, ...)
// IBM: call void @llvm.va_start(i8* %{{[0-9a-zA-Z_.]+}})
// IBM: %[[P1:[0-9a-zA-Z_.]+]] = add i64 %{{[0-9a-zA-Z_.]+}}, 15
// IBM: %[[P2:[0-9a-zA-Z_.]+]] = and i64 %[[P1]], -16
// IBM: %[[P3:[0-9a-zA-Z_.]+]] = inttoptr i64 %[[P2]] to i8*
// IBM: %[[P4:[0-9a-zA-Z_.]+]] = bitcast i8* %[[P3]] to ppc_fp128*
// IBM: %{{[0-9a-zA-Z_.]+}} = load ppc_fp128, ppc_fp128* %[[P4]], align 16
// IBM: call void @llvm.va_end(i8* %{{[0-9a-zA-Z_.]+}})
long double long_double(int n, ...) {
  va_list ap;
  va_start(ap, n);
  long double x = va_arg(ap, long double);
  va_end(ap);
  return x;
}
