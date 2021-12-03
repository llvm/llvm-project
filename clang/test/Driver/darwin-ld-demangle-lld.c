// With -fuse-ld=lld, -demangle is always passed to the linker on Darwin.
// REQUIRES: shell

// RUN: %clang --target=x86_64-apple-darwin -### \
// RUN:   -fuse-ld=lld -B%S/Inputs/lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-demangle"
