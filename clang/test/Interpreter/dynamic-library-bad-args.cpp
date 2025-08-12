// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s
%lib
// CHECK: %lib expects 1 argument: the path to a dynamic library
%quit
