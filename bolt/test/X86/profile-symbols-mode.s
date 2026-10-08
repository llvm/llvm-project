# Test the 'symbols' fdata subformat. Each line lists a single symbol name.
# Matching functions (compared against restored names, i.e. with the local
# symbol disambiguation suffix stripped) receive an execution count of 1.
#
# main.s and other.s each define a local function named loc. Local names are not
# unique, so BOLT disambiguates them as loc/1 and loc/2 (with long-form aliases
# loc/main.c/N and loc/other.c/N from the file symbols). A single "loc" entry in
# the symbols profile matches both via the restored name. foo (global) is also
# listed; bar (global) is omitted and stays cold.

# REQUIRES: system-linux

# RUN: rm -rf %t && split-file %s %t
# RUN: %clang %cflags %t/main.s %t/other.s -o %t/main.exe -Wl,-q
# RUN: llvm-bolt %t/main.exe --data %t/symbols.fdata -o %t/main.bolt \
# RUN:     --print-cfg 2>&1 | FileCheck %s

# foo and both loc functions (matched by the same restored name) each receive an
# execution count of 1.
# CHECK: Binary Function "foo"
# CHECK: Exec Count  : 1
# CHECK: Binary Function "loc/1
# CHECK: Exec Count  : 1
# CHECK: Binary Function "loc/2
# CHECK: Exec Count  : 1

#--- main.s
    .file "main.c"
    .text
    .globl _start
    .type _start, %function
_start:
    call foo
    call bar
    retq
    .size _start, .-_start

    .globl foo
    .type foo, %function
foo:
    retq
    .size foo, .-foo

# First local symbol named loc.
    .local loc
    .type loc, %function
loc:
    retq
    .size loc, .-loc

    .globl bar
    .type bar, %function
bar:
    retq
    .size bar, .-bar

#--- other.s
    .file "other.c"
    .text
# Second local symbol also named loc; BOLT disambiguates it from the one in
# main.s as loc/2.
    .local loc
    .type loc, %function
loc:
    retq
    .size loc, .-loc

#--- symbols.fdata
symbols
foo
loc
