// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -O1 -disable-llvm-passes \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr7 | FileCheck %s --check-prefix=64BIT
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -O1 -disable-llvm-passes \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s --check-prefix=64BIT
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -O1 -disable-llvm-passes \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr7 | FileCheck %s --check-prefix=64BIT
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -O1 -disable-llvm-passes \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr7 | FileCheck %s --check-prefix=32BIT
// RUN: %clang_cc1 -triple powerpcle-unknown-linux-gnu -O1 -disable-llvm-passes \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s --check-prefix=32BIT
// RUN: %clang_cc1 -triple powerpc-unknown-aix -O1 -disable-llvm-passes \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr7 | FileCheck %s --check-prefix=32BIT

// 64BIT-LABEL: @testbuiltin_expect(
// 64BIT:         [[EXPVAL:%.*]] = call i64 @llvm.expect.i64(i64 {{%.*}}, i64 23)
// 64BIT-NEXT:    [[CMP:%.*]] = icmp eq i64 [[EXPVAL]], 23
// 64BIT-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
// 64BIT-NEXT:    [[CONV1:%.*]] = sext i32 [[CONV]] to i64
// 64BIT-NEXT:    ret i64 [[CONV1]]
//
// 32BIT-LABEL: @testbuiltin_expect(
// 32BIT:         [[EXPVAL:%.*]] = call i32 @llvm.expect.i32(i32 {{%.*}}, i32 23)
// 32BIT-NEXT:    [[CMP:%.*]] = icmp eq i32 [[EXPVAL]], 23
// 32BIT-NEXT:    [[CONV:%.*]] = zext i1 [[CMP]] to i32
// 32BIT-NEXT:    ret i32 [[CONV]]
//
long testbuiltin_expect(long expression) {
  // The second parameter is a long constant.
  return __builtin_expect(expression, 23) == 23;
}
