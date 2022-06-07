// REQUIRES: enable-opaque-pointers
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto 2> %t
// RUN: FileCheck %s < %t

// CHECK-NOT: -plugin-opt=no-opaque-pointers
