// Check that we compile correctly with multiple ARM -arch options.
//
// RUN: %clang -target arm7-apple-darwin10 -### \
// RUN:   -arch armv7 -arch armv7s %s 2>&1 | FileCheck %s

// CHECK: "-cc1" "-triple" "thumbv7-apple-ios5.0.0"
// CHECK-SAME: "-o" "[[CC_OUT1:[^"]*]]"
// CHECK:ld{{(\.exe)?}}" {{.*}} "-o" "[[LD_OUT1:[^"]*]]" {{.*}} "[[CC_OUT1]]"
// CHECK:"-cc1" "-triple" "thumbv7s-apple-ios5.0.0"
// CHECK-SAME: "-o" "[[CC_OUT2:[^"]*]]"
// CHECK:ld{{(\.exe)?}}" {{.*}} "-o" "[[LD_OUT2:[^"]*]]" {{.*}} "[[CC_OUT2]]"
// CHECK:lipo{{(\.exe)?}}"
// CHECK-DAG: "[[LD_OUT1]]"
// CHECK-DAG: "[[LD_OUT2]]"
