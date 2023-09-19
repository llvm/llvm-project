// Check that we only find libc++ in the installation directory when it contains
// an Android-specific include directory.

// RUN: mkdir -p %t1/bin
// RUN: mkdir -p %t1/include/c++/v1
// RUN: mkdir -p %t1/sysroot
// RUN: %clang -target aarch64-none-linux-android -ccc-install-dir %t1/bin \
// RUN:   --sysroot=%t1/sysroot -stdlib=libc++ -fsyntax-only \
// RUN:   %s -### 2>&1 | FileCheck %s
// CHECK-NOT: "-internal-isystem" "{{.*}}v1"

// RUN: mkdir -p %t2/bin
// RUN: mkdir -p %t2/include/c++/v1
// RUN: mkdir -p %t2/sysroot
// RUN: mkdir -p %t2/include/aarch64-none-linux-android/c++/v1

// RUN: %clang -target aarch64-none-linux-android -ccc-install-dir %/t2/bin \
// RUN:   --sysroot=%t2/sysroot -stdlib=libc++ -fsyntax-only \
// RUN:   %s -### 2>&1 | FileCheck --check-prefix=ANDROID-DIR -DDIR=%/t2/bin %s

// RUN: %clang -target aarch64-none-linux-android21 -ccc-install-dir %/t2/bin \
// RUN:   --sysroot=%t2/sysroot -stdlib=libc++ -fsyntax-only \
// RUN:   %s -### 2>&1 | FileCheck --check-prefix=ANDROID-DIR -DDIR=%/t2/bin %s

// ANDROID-DIR: "-internal-isystem" "[[DIR]][[SEP:/|\\\\]]..[[SEP]]include[[SEP]]aarch64-none-linux-android[[SEP]]c++[[SEP]]v1"
// ANDROID-DIR-SAME: "-internal-isystem" "[[DIR]][[SEP]]..[[SEP]]include[[SEP]]c++[[SEP]]v1"
