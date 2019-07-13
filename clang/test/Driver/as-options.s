// PR21000: Test that -I is passed to both external and integrated assemblers.

// RUN: %clang -target x86_64-linux-gnu -c -no-integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target x86_64-linux-gnu -c -no-integrated-as %s \
// RUN:   -I foo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target x86_64-linux-gnu -c -integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target x86_64-linux-gnu -c -integrated-as %s \
// RUN:   -I foo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// Other GNU targets

// RUN: %clang -target aarch64-linux-gnu -c -no-integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target aarch64-linux-gnu -c -integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target armv7-linux-gnueabihf -c -no-integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target armv7-linux-gnueabihf -c -integrated-as %s \
// RUN:   -Ifoo_dir -### 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-I" "foo_dir"
