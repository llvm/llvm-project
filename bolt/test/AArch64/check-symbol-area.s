// This test checks that when looking for a function corresponding to a
// symbol, BOLT is not looking through a data area (constant island).

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s

// Before adding a check for constant islands, BOLT would exit with an error
// of the form: "symbol not found" and throw an LLVM UNREACHABLE error.
# CHECK-NOT: symbol not found
# CHECK-NOT: UNREACHABLE

// Now BOLT throws a warning and does not crash.
# CHECK: BOLT-WARNING: corrupted control flow detected in function main{{.*}}:
# CHECK-SAME: an external branch/call targets an invalid instruction in
# CHECK-SAME: function first_block at address {{.*}}; ignoring both functions

# CHECK: BOLT-WARNING: ignoring entry point at address 0x{{[0-9a-f]+}}
# CHECK-SAME: in constant island of function first_block

.text
.global main
.type main, %function
main:
        add     x0, x1, x1
        bl      first_block
        ret

.global first_block
$d:
first_block:
        add     x0, x1, x1
        bl      second_block
        ret

$x:
second_block:
        add     x0, x1, x1
        ret
