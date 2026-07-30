// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


char * foo(void) { return "\\begin{"; }
