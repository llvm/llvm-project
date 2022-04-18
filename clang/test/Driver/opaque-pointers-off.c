// UNSUPPORTED: enable-opaque-pointers
/// Test -DCLANG_ENABLE_OPAQUE_POINTERS=OFF.

// RUN: %clang -### --target=aarch64-linux-gnu %s 2>&1 | FileCheck %s
/// User -Xclang -opaque-pointers overrides the default.
// RUN: %clang -### --target=aarch64-linux-gnu -Xclang -opaque-pointers %s 2>&1 | FileCheck %s --check-prefix=CHECK2

// CHECK:       "-no-opaque-pointers"

// CHECK2:      "-no-opaque-pointers"
// CHECK2-SAME: "-opaque-pointers"
