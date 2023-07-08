// RUN: %clang_cc1 -emit-llvm %s -o %t
// PR1824

int foo(int x, short y) {
  return x ?: y;
}

float test(float x, int Y) {
  return Y != 0 ? : x;
}

extern void abort(void);
void  test1 (void) {
  char x[1];
  char *y = x ? : 0;

  if (x != y)
    abort();
}

_Complex int getComplex(_Complex int val) {
  static int count;
  if (count++)
    abort();
  return val;
}

_Complex int complx(void) {
    _Complex int cond;
    _Complex int rhs;

    return getComplex(1+2i) ? : rhs;
}
