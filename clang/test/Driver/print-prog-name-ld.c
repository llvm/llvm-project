// Test that -print-prog-name=ld correctly obeys -fuse-ld=...

// RUN: %clang -print-prog-name=ld -fuse-ld=lld 2>&1 | FileCheck %s
// CHECK:{{.*ld(64)?\.lld}}
