// RUN: %clang -### -Wlarge-by-value-copy %s 2>&1 | FileCheck -check-prefix=LARGE_VALUE_COPY_DEFAULT %s
// LARGE_VALUE_COPY_DEFAULT: -Wlarge-by-value-copy=64
// RUN: %clang -### -Wlarge-by-value-copy=128 %s 2>&1 | FileCheck -check-prefix=LARGE_VALUE_COPY_JOINED %s
// LARGE_VALUE_COPY_JOINED: -Wlarge-by-value-copy=128

// Check that -isysroot warns on nonexistent paths.
// RUN: %clang -### -c -target i386-apple-darwin10 -isysroot %t/warning-options %s 2>&1 | FileCheck --check-prefix=CHECK-ISYSROOT %s
// CHECK-ISYSROOT: warning: no such sysroot directory: '{{.*}}/warning-options'
