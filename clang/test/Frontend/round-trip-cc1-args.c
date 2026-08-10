// RUN: %clang_cc1 %s -no-round-trip-args -Rround-trip-cc1-args 2>&1 | FileCheck %s -check-prefix=CHECK-WITHOUT-ROUND-TRIP -allow-empty
// RUN: %clang_cc1 %s -round-trip-args 2>&1 | FileCheck %s -check-prefix=CHECK-ROUND-TRIP-WITHOUT-REMARKS -allow-empty
// RUN: %clang_cc1 %s -round-trip-args -Rround-trip-cc1-args 2>&1 | FileCheck %s -check-prefix=CHECK-ROUND-TRIP-WITH-REMARKS

// CHECK-WITHOUT-ROUND-TRIP-NOT: remark:
// CHECK-ROUND-TRIP-WITHOUT-REMARKS-NOT: remark:
// CHECK-ROUND-TRIP-WITH-REMARKS: remark: generated arguments #{{.*}} in round-trip: {{.*}}
