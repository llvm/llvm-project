// RUN: %clang_cc1 -emit-llvm %s -o - -std=gnu89
// rdar://7208839

extern inline int f1 (void) {return 1;}
int f3 (void) {return f1();}
int f1 (void) {return 0;}
