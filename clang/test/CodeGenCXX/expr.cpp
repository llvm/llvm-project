// RUN: %clang_cc1 -emit-llvm -x c++ < %s

void test0(int x) {
          if (x != 0) return;
}


// PR5211
void test1() {
  char *xpto;
  while ( true && xpto[0] );
}

// PR5514
int a;
void test2() { ++a+=10; }

// PR7892
int test3(const char*);
int test3g = test3(__PRETTY_FUNCTION__);


// PR7889
struct test4A {
  int j : 2;
};
int test4() {
  test4A a;
  (a.j = 2) = 3;
}

// Incomplete type in conditional operator.
// Check operations on incomplete types.
struct s5;
struct s5 &f5_0(bool cond, struct s5 &a, struct s5 &b) {
  return cond ? a : b;
}
