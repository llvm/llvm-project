// RUN: %clang_cc1 %s -emit-llvm -o -

struct S {int a, b;} x;
void a(struct S* b) {*b = (r(), x);}
