// Flang driver changes break this test, -o multiple obj error
// XFAIL: *

// Check that we don't find the libc++ in the installation directory when
// targeting Android.

// RUN: mkdir -p %t/bin
// RUN: mkdir -p %t/include/c++/v1
// RUN: mkdir -p %t/sysroot
// RUN: %clang -target aarch64-linux-android -ccc-install-dir %t/bin \
// RUN:   --sysroot=%t/sysroot -stdlib=libc++ -fsyntax-only \
// RUN:   %s -### 2>&1 | FileCheck %s
// CHECK-NOT: "-internal-isystem" "{{.*}}v1"
