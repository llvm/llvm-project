// RUN: %clang -E -xassembler-with-cpp %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: warning: \u used with no following hex digits
// CHECK: .word \u

    .macro foo, u
        .word \u
    .endm
