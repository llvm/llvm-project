/// Test that no secondary entry points are created for basic block labels used
/// by branches.
// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt -print-cfg -o %t.null %t 2>&1 | FileCheck %s

// CHECK: Binary Function "_start" after building cfg {
// CHECK: IsMultiEntry: 0
// CHECK: beq t0, t1, .Ltmp0
// CHECK: {{^}}.Ltmp0
// CHECK: ret

    .globl _start
_start:
    beq t0, t1, 1f
1:
    ret
    .size _start, .-_start

