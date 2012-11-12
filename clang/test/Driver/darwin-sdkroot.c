// Check that SDKROOT is used to define the default for -isysroot on Darwin.
//
// RUN: rm -rf %t.tmpdir
// RUN: mkdir -p %t.tmpdir
// RUN: env SDKROOT=%t.tmpdir %clang -target x86_64-apple-darwin10 \
// RUN:   -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-BASIC < %t.log %s
//
// CHECK-BASIC: clang
// CHECK-BASIC: "-cc1"
// CHECK-BASIC: "-isysroot" "{{.*tmpdir}}"

// Check that we don't use SDKROOT as the default if it is not a valid path.

// RUN: rm -rf %t.nonpath
// RUN: env SDKROOT=%t.nonpath %clang -target x86_64-apple-darwin10 \
// RUN:   -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-NONPATH < %t.log %s
//
// CHECK-NONPATH: clang
// CHECK-NONPATH: "-cc1"
// CHECK-NONPATH-NOT: "-isysroot"
