// RUN: %clang_cc1 %s -emit-llvm -O0 -o - | grep sret | count 2

struct abc {
 long a;
 long b;
 long c;
};
 
struct abc foo2(){}
