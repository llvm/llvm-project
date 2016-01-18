// Check that we don't try to forward -Xclang or -mlinker-version to GCC.
// PR12920 -- Check also we may not forward W_Group options to GCC.
//
// RUN: %clang -target powerpc-unknown-unknown \
// RUN:   %s \
// RUN:   -Wall -Wdocumentation \
// RUN:   -Xclang foo-bar \
// RUN:   -march=x86_64 \
// RUN:   -mlinker-version=10 -### 2> %t
// RUN: FileCheck < %t %s
//
// clang-cc1
// CHECK: "-Wall" "-Wdocumentation"
// CHECK: "-o" "{{[^"]+}}.o"
//
// gcc-ld
// CHECK: gcc{{[^"]*}}"
// CHECK-NOT: "-mlinker-version=10"
// CHECK-NOT: "-Xclang"
// CHECK-NOT: "foo-bar"
// CHECK-NOT: "-Wall"
// CHECK-NOT: "-Wdocumentation"
// CHECK: -march
// CHECK-NOT: "-mlinker-version=10"
// CHECK-NOT: "-Xclang"
// CHECK-NOT: "foo-bar"
// CHECK-NOT: "-Wall"
// CHECK-NOT: "-Wdocumentation"
// CHECK: "-o" "a.out"
