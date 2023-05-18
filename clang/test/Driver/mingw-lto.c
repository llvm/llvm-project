// The default linker doesn't support LLVM bitcode
// RUN: not %clang --target=i686-pc-windows-gnu %s -flto -fuse-ld=bfd
// When using lld, this is allowed though.
// RUN: %clang --target=i686-pc-windows-gnu -### %s -flto -fuse-ld=lld -B%S/Inputs/lld -femulated-tls 2>&1 | FileCheck %s

// CHECK: "-plugin-opt=-emulated-tls"
