// RUN: cat %s | clang-repl | FileCheck %s
%help
// CHECK: %help   list clang-repl %commands
// CHECK-NEXT: %undo   undo the previous input
// CHECK-NEXT: %lib  <path>  link a dynamic library
// CHECK-NEXT: %quit   exit clang-repl
%quit
