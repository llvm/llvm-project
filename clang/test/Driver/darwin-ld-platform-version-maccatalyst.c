// RUN: touch %t.o

// RUN: %clang -target x86_64-apple-ios13.3-macabi -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-ios13.3-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAPPED-SDK %s

// CHECK: "-platform_version" "mac catalyst" "13.3.0" "13.1"
// CHECK-MAPPED-SDK: "-platform_version" "mac catalyst" "13.3.0" "13.1"
