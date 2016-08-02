// RUN: %clang_cc1 %s -emit-llvm -o - | grep sret | count 2

struct abc {
 long a;
 long b;
 long c;
 long d;
 long e;
};
 
struct abc foo2(){}
