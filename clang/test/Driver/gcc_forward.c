// Check that we don't try to forward -Xclang or -mlinker-version to GCC.
//
// RUN: %clang -target powerpc-unknown-unknown \
// RUN:   -c %s \
// RUN:   -Xclang foo-bar \
// RUN:   -mlinker-version=10 -### 2> %t
// RUN: FileCheck < %t %s
//
// CHECK: gcc{{.*}}"
// CHECK-NOT: "-mlinker-version=10"
// CHECK-NOT: "-Xclang"
// CHECK-NOT: "foo-bar"
// CHECK: gcc_forward
