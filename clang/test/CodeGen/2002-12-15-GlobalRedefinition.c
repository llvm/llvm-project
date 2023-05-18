// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

extern char algbrfile[9];
char algbrfile[9] = "abcdefgh";

