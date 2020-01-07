// RUN: touch %t.o

// RUN: %clang -target x86_64-apple-ios13.3-macabi -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-ios13.3-macabi -isysroot %S/Inputs/MacOSX10.14.versioned.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAPPED-SDK %s
// RUN: %clang -target x86_64-apple-ios12.0-macabi -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-OLD %s

// CHECK: "-platform_version" "mac catalyst" "13.3.0" "0.0.0"
// CHECK-MAPPED-SDK: "-platform_version" "mac catalyst" "13.3.0" "12.0"
// CHECK-OLD: "-platform_version" "mac catalyst" "13.0" "0.0.0"
