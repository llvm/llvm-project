// UNSUPPORTED: system-windows

// RUN: env COMPILER_PATH=cpath1:cpath2 %clang %s -target x86_64-pc-freebsd --sysroot=%S/Inputs/basic_freebsd64_tree \
// RUN:   -B b1 -B b2 -print-search-dirs | FileCheck %s
// CHECK:      programs: =b1:b2:cpath1:cpath2:{{.*}}
// CHECK-NEXT: libraries: ={{.*}}Inputs/basic_freebsd64_tree/usr/lib
