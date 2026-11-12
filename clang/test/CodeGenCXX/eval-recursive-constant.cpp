// RUN: %clang_cc1 %s -emit-llvm-only

extern const int a,b;
const int a=b,b=a;
int c() { if (a) return 1; return 0; }
