// RUN: %clang_cc1 %s -emit-llvm -o - | grep sret | count 5

struct abc {
 long a;
 long b;
 long c;
 long d;
 long e;
};
 
struct abc foo1(void);
struct abc foo2();

void bar() {
  struct abc dummy1 = foo1();
  struct abc dummy2 = foo2();
}
