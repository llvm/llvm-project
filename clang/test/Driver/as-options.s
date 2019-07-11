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

// Test that assembler options don't cause warnings when there's no assembler
// stage.

// RUN: %clang -mincremental-linker-compatible -E \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s
// RUN: %clang -mincremental-linker-compatible -E \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s
// RUN: %clang -mimplicit-it=always -target armv7-linux-gnueabi -E \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s
// RUN: %clang -mimplicit-it=always -target armv7-linux-gnueabi -E \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s
// RUN: %clang -Wa,-mbig-obj -target i386-pc-windows -E \
// RUN:   -o /dev/null -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s
// RUN: %clang -Wa,-mbig-obj -target i386-pc-windows -E \
// RUN:   -o /dev/null -x assembler-with-cpp %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN --allow-empty %s
// WARN-NOT: unused
