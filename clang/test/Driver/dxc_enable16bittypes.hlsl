// RUN: %clang_dxc -enable-16bit-types -T lib_6_7 %s -### %s 2>&1 | FileCheck %s

// Make sure enable-16bit-types flag translates into '-fnative-half-type' and 'fnative-int16-type'
// CHECK: "-fnative-half-type"
// CHECK-SAME: "-fnative-int16-type"

// expected-no-diagnostics
