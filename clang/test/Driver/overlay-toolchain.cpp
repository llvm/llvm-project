// RUN: %clangxx %s -### --target=powerpc64le-linux-gnu \
// RUN:   --overlay-platform-toolchain=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-11.2.0 \
// RUN:   2>&1 | FileCheck %s --check-prefix=OVERLAY
// RUN: %clangxx %s -### --target=powerpc64le-linux-gnu \
// RUN:   --sysroot=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-8.3.0 \
// RUN:   --overlay-platform-toolchain=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-11.2.0 \
// RUN:   2>&1 | FileCheck %s --check-prefixes=OVERLAY,ROOT

// OVERLAY: "-internal-externc-isystem"
// OVERLAY: "[[TOOLCHAIN:[^"]+]]/powerpc64le-linux-gnu-tree/gcc-11.2.0/include"
// ROOT: "-internal-externc-isystem"
// ROOT: "[[TOOLCHAIN]]/powerpc64le-linux-gnu-tree/gcc-8.3.0/include"
// OVERLAY: "-dynamic-linker"
// OVERLAY: "[[TOOLCHAIN]]/powerpc64le-linux-gnu-tree/gcc-11.2.0/lib64/ld64.so.2"
// OVERLAY: "-L[[TOOLCHAIN]]/powerpc64le-linux-gnu-tree/gcc-11.2.0/lib/../lib64"
// ROOT: "-L[[TOOLCHAIN]]/powerpc64le-linux-gnu-tree/gcc-8.3.0/lib/../lib64"
// OVERLAY: "-L[[TOOLCHAIN]]/powerpc64le-linux-gnu-tree/gcc-11.2.0/lib"
// ROOT: "-L[[TOOLCHAIN]]/powerpc64le-linux-gnu-tree/gcc-8.3.0/lib"
