// Check that we don't find the libc++ in the installation directory when
// targeting Android.

// RUN: mkdir -p %t/bin
// RUN: mkdir -p %t/include/c++/v1
// RUN: %clang -target aarch64-linux-android -ccc-install-dir %t/bin \
// RUN:   -stdlib=libc++ -fsyntax-only %s -### 2>&1 | FileCheck %s
// CHECK-NOT: "-internal-isystem" "{{.*}}v1"
