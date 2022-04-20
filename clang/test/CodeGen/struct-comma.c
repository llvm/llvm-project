// RUN: %clang_cc1 %s -emit-llvm -o -

struct S {int a, b;} x;
extern int r(void);
void a(struct S* b) {*b = (r(), x);}
