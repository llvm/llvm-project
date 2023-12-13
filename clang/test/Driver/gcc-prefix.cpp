// UNSUPPORTED: system-windows

// RUN: %clang --target=x86_64-redhat-linux-gnu \
// RUN: --sysroot=%S/Inputs/fedora_39_tree --gcc-triple=x86_64-redhat-linux -v 2>&1 | \
// RUN: FileCheck %s

// CHECK: {{^}}Selected GCC installation:
// CHECK: fedora_39_tree/usr/lib/gcc/x86_64-redhat-linux/13{{$}}
