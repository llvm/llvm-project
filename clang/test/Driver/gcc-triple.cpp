// UNSUPPORTED: system-windows

// RUN: %clang --target=x86_64-redhat-linux-gnu \
// RUN:   --sysroot=%S/Inputs/fedora_39_tree --gcc-triple=x86_64-redhat-linux -v 2>&1 | \
// RUN:   FileCheck %s --check-prefix=TRIPLE_EXISTS

// TRIPLE_EXISTS: {{^}}Selected GCC installation:
// TRIPLE_EXISTS: fedora_39_tree/usr/lib/gcc/x86_64-redhat-linux/13{{$}}

// RUN: %clang --target=x86_64-redhat-linux-gnu \
// RUN:   --sysroot=%S/Inputs/fedora_39_tree --gcc-triple=x86_64-gentoo-linux -v 2>&1 | \
// RUN:   FileCheck %s --check-prefix=TRIPLE_DOESNT_EXIST

// TRIPLE_DOESNT_EXIST-NOT: x86_64-gentoo-linux
