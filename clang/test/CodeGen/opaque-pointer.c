// RUN: %clang_cc1 %s -emit-llvm -o -
struct test;

typedef void (*my_func) (struct test *);
my_func handler;

struct test {
  char a;
};

char f(struct test *t) {
  return t->a;
}
