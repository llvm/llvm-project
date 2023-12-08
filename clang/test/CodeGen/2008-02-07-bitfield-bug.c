// RUN: %clang_cc1 %s -emit-llvm -o %t
// PR1990

struct test {
  char a[3];
  unsigned char b:1;
};

void f(struct test *t) {
  t->b = 1;
}
