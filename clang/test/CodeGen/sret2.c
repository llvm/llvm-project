// RUN: %clang_cc1 %s -emit-llvm -o - | grep sret | grep -v 'sret2.c' | count 1

struct abc {
 long a;
 long b;
 long c;
 long d;
 long e;
};
 
struct abc foo2(void){}
