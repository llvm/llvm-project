// RUN: %clang_cc1 %s -Wno-strict-prototypes -emit-llvm -o - | grep sret | grep -v 'sret.c' | count 4

struct abc {
 long a;
 long b;
 long c;
 long d;
 long e;
};
 
struct abc foo1(void);
struct abc foo2();

void bar(void) {
  struct abc dummy1 = foo1();
  struct abc dummy2 = foo2();
}
