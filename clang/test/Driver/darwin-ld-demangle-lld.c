// With -fuse-ld=lld, -demangle is always passed to the linker on Darwin.
// REQUIRES: shell

// FIXME: Remove this test case when we remove the lld.darwinold backend.
// RUN: %clang --target=x86_64-apple-darwin -### \
// RUN:   -fuse-ld=lld.darwinold -B%S/Inputs/lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang --target=x86_64-apple-darwin -### \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-demangle"
