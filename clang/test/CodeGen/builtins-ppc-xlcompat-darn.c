// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s
// RUN: %clang_cc1 -triple powerpcle-unknown-unknown \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s

// The darn class of builtins are Power 9 and up and only darn_32 works in
// 32 bit mode.

// CHECK-LABEL: @testdarn(
// CHECK:         [[TMP0:%.*]] = call i64 @llvm.ppc.darn()
// CHECK-NEXT:    ret i64 [[TMP0]]
//
long long testdarn(void) {
  return __darn();
}

// CHECK-LABEL: @testdarn_raw(
// CHECK:         [[TMP0:%.*]] = call i64 @llvm.ppc.darnraw()
// CHECK-NEXT:    ret i64 [[TMP0]]
//
long long testdarn_raw(void) {
  return __darn_raw();
}

// CHECK-LABEL: @testdarn_32(
// CHECK:         [[TMP0:%.*]] = call i32 @llvm.ppc.darn32()
// CHECK-NEXT:    ret i32 [[TMP0]]
//
int testdarn_32(void) {
  return __darn_32();
}
