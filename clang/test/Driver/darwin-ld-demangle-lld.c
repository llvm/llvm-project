// With -fuse-ld=lld, -demangle is always passed to the linker on Darwin.
// REQUIRES: shell

// RUN: %clang --target=x86_64-apple-darwin -### -fuse-ld=lld \
// RUN:   -B%S/Inputs/lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang --target=x86_64-apple-darwin -### -fuse-ld=lld \
// RUN:   --ld-path=%S/Inputs/lld/ld64.lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-demangle"
