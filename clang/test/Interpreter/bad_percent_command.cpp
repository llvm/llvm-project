// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s
%foobar
// CHECK: Invalid % command "%foobar", use "%help" to list commands
%quit
