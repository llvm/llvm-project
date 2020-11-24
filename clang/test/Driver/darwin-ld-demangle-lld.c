// With -fuse-ld=lld, -demangle is always passed to the linker on Darwin.

// RUN: %clang -### -fuse-ld=lld %s 2>&1 | FileCheck %s
// FIXME: Remove ld.darwinnew once it's the default (and only) mach-o lld.
// RUN: %clang -### -fuse-ld=lld.darwinnew %s 2>&1 | FileCheck %s
// CHECK: -demangle
