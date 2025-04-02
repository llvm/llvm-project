// REQUIRES: x86
// This test intentionally checks for fatal errors, and fatal errors aren't supported for testing when main is run twice.
// XFAIL: main-run-twice

// Check bad archive error reporting with --whole-archive
// and without it.

// RUN: echo "!<arch>" > %t.a
// RUN: echo "foo" >> %t.a
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: not ld.lld %t.o %t.a -o /dev/null 2>&1 | FileCheck -DFILE=%t.a %s
// RUN: not ld.lld %t.o --whole-archive %t.a -o /dev/null 2>&1 | FileCheck -DFILE=%t.a %s
// CHECK: error: [[FILE]]: failed to parse archive: truncated or malformed archive (remaining size of archive too small for next archive member header at offset 8)

.globl _start
_start:
