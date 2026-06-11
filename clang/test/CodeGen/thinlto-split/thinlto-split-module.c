// UNSUPPORTED: system-windows
// REQUIRES: aarch64-registered-target

// Distributed ThinLTO (DTLTO)
// RUN: %clang -flto=thin -c %s -o %t.o
// RUN: %clang -flto=thin -fuse-ld=lld -Wl,--thinlto-index-only %t.o
// RUN: not --crash %clang %t.o -c -fthinlto-index=%t.o.thinlto.bc \
// RUN:                            -mllvm -lto-split-by-callgraph=true \
// RUN:                            -mllvm -lto-split-partitions=2
//
// Regular ThinLTO
// RUN: %clang -flto=thin -fuse-ld=lld -shared \
// RUN:   -o %t.o %s \
// RUN:   -Wl,-mllvm,-lto-split-by-callgraph=true \
// RUN:   -Wl,-mllvm,-lto-split-partitions=2 \
// RUN:   -Wl,--save-temps=prelink
// RUN: llvm-nm %t.o.lto.o | FileCheck %s --check-prefix=CHECK0
// RUN: llvm-nm %t.o.lto.1.o | FileCheck %s --check-prefix=CHECK1

// CHECK0-DAG: T caller_b
// CHECK0-DAG: T {{promoted_internal[.][0-9a-f]+}}

// CHECK1-DAG: T caller_a
// CHECK1-DAG: U {{promoted_internal[.][0-9a-f]+}}

static void promoted_internal(void) {}

void caller_a(void) {
    promoted_internal();
}

void caller_b(void) {
    promoted_internal();
}