// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_MulHigh --implicit-check-not=@_MulUnsignedHigh
// RUN: %clang_cc1 -ffreestanding -triple thumbv7-linux-gnueabihf -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefix=NOMS

// _MulHigh and _MulUnsignedHigh are the high 32 bits of the 64-bit product.

#ifdef _MSC_VER
// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

long test_MulHigh(long a, long b) { return _MulHigh(a, b); }
// CHECK-LABEL: define dso_local {{.*}}i32 @test_MulHigh(
// CHECK: [[A:%.*]] = sext i32 %{{.*}} to i64
// CHECK: [[B:%.*]] = sext i32 %{{.*}} to i64
// CHECK: [[M:%.*]] = mul nsw i64 [[A]], [[B]]
// CHECK: [[H:%.*]] = ashr i64 [[M]], 32
// CHECK: [[R:%.*]] = trunc i64 [[H]] to i32
// CHECK: ret i32 [[R]]

unsigned long test_MulUnsignedHigh(unsigned long a, unsigned long b) {
  return _MulUnsignedHigh(a, b);
}
// CHECK-LABEL: define dso_local {{.*}}i32 @test_MulUnsignedHigh(
// CHECK: [[A:%.*]] = zext i32 %{{.*}} to i64
// CHECK: [[B:%.*]] = zext i32 %{{.*}} to i64
// CHECK: [[M:%.*]] = mul nuw i64 [[A]], [[B]]
// CHECK: [[H:%.*]] = lshr i64 [[M]], 32
// CHECK: [[R:%.*]] = trunc i64 [[H]] to i32
// CHECK: ret i32 [[R]]
#else
// Without the Microsoft extensions the names are ordinary identifiers.
unsigned long _MulUnsignedHigh(unsigned long a, unsigned long b) { return 42; }
unsigned long use(unsigned long a) { return _MulUnsignedHigh(a, a); }
// NOMS: define {{.*}}i32 @_MulUnsignedHigh(
// NOMS: ret i32 42
// NOMS: call {{.*}}i32 @_MulUnsignedHigh(
#endif
