// UNSUPPORTED: system-windows
// REQUIRES: aarch64-registered-target

// RUN: %clang -flto=full -fuse-ld=lld -shared \
// RUN:   -o %t.o %s \
// RUN:   -Wl,-mllvm,-lto-split-by-callgraph=true \
// RUN:   -Wl,--lto-partitions=2 \
// RUN:   -Wl,--save-temps=prelink
// RUN: llvm-nm %t.o.lto.o | FileCheck %s --check-prefix=CHECK0
// RUN: llvm-nm %t.o.lto.1.o | FileCheck %s --check-prefix=CHECK1

// CHECK0-DAG: T caller_b
// CHECK0-DAG: T promoted_internal

// CHECK1-DAG: T caller_a
// CHECK1-DAG: U promoted_internal

static void promoted_internal(void) {}

void caller_a(void) {
    promoted_internal();
}

void caller_b(void) {
    promoted_internal();
}