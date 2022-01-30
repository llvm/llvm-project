// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | \
// RUN:    FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr9 | \
// RUN:    FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | \
// RUN:    FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s
// RUN: %clang_cc1 -triple powerpcle-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr9 | FileCheck %s

// The darn class of builtins are Power 9 and up and only darn_32 works in
// 32 bit mode.

#ifdef __PPC64__
// CHECK-64-LABEL: @testdarn(
// CHECK-64:         [[TMP0:%.*]] = call i64 @llvm.ppc.darn()
// CHECK-64-NEXT:    ret i64 [[TMP0]]
//
long long testdarn(void) {
  return __darn();
}

// CHECK-64-LABEL: @testdarn_raw(
// CHECK-64:         [[TMP0:%.*]] = call i64 @llvm.ppc.darnraw()
// CHECK-64-NEXT:    ret i64 [[TMP0]]
//
long long testdarn_raw(void) {
  return __darn_raw();
}
#endif

// CHECK-LABEL: @testdarn_32(
// CHECK:         [[TMP0:%.*]] = call i32 @llvm.ppc.darn32()
// CHECK-NEXT:    ret i32 [[TMP0]]
//
int testdarn_32(void) {
  return __darn_32();
}
