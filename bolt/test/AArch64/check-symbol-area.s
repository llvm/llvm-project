// This test checks that when looking for a function
// corresponding to a symbol, BOLT is not looking 
// through a data area (constant island).

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s

// Before adding a check for constant islands, BOLT would exit with an error
// of the form: "symbol not found" and throw an LLVM UNREACHABLE error.
# CHECK-NOT: symbol not found
# CHECK-NOT: UNREACHABLE

// Now BOLT throws a warning and does not crash.
# CHECK: BOLT-WARNING: symbol third_block/1  is in data region of function first_block(0x{{[0-9a-f]+}}).

.text
.global main
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
second_block:
        add     x0, x1, x1
        bl      third_block
        ret
$x:
third_block:
        add     x0, x1, x1
        ret
