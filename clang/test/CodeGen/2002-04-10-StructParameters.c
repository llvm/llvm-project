// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


typedef struct {
  char p;
  short q;
  char r;
  int X;
  short Y, Z;
  int Q;
} foo;

int test(foo X, float);
int testE(char,short,char,int,int,float);
void test3(foo *X) {
  X->q = 1;
}

void test2(foo Y) {
  testE(Y.p, Y.q, Y.r, Y.X, Y.Y, 0.1f);
  test(Y, 0.1f);
  test2(Y);
  test3(&Y);
}

