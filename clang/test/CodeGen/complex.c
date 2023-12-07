// RUN: %clang_cc1 -emit-llvm-only %s

int main(void)
{
  double _Complex a = 5;
  double _Complex b = 42;

  return a * b != b * a;
}

_Complex double bar(int);
void test(_Complex double*);
void takecomplex(_Complex double);

void test2(int c) {
  _Complex double X;
  X = bar(1);
  test(&X);
  takecomplex(X);
}

_Complex double g1, g2;
_Complex float cf;
double D;

void test3(void) {
  g1 = g1 + g2;
  g1 = g1 - g2;
  g1 = g1 * g2;
  g1 = +-~g1;

  double Gr = __real g1;

  cf += D;
  D += cf;
  cf /= g1;
  g1 = g1 + D;
  g1 = D + g1;
}

__complex__ int ci1, ci2;
__complex__ short cs;
int i;
void test3int(void) {
  ci1 = ci1 + ci2;
  ci1 = ci1 - ci2;
  ci1 = ci1 * ci2;
  ci1 = +-~ci1;

  i = __real ci1;

  cs += i;
  D += cf;
  cs /= ci1;
  ci1 = ci1 + i;
  ci1 = i + ci1;
}

void t1(void) {
  (__real__ cf) = 4.0;
}

void t2(void) {
  (__imag__ cf) = 4.0;
}

// PR1960
void t3(void) {
  __complex__ long long v = 2;
}

// PR3131
float _Complex t4(void);

void t5(void) {
  float _Complex x = t4();
}

void t6(void) {
  g1++;
  g1--;
  ++g1;
  --g1;
  ci1++;
  ci1--;
  ++ci1;
  --ci1;
}

double t7(double _Complex c) {
  return __builtin_fabs(__real__(c));
}

void t8(void) {
  __complex__ int *x = &(__complex__ int){1};
}

const _Complex double test9const = 0;
_Complex double test9func(void) { return test9const; }

// D6217
void t91(void) {
  // Check for proper type promotion of conditional expression
  char c[(int)(sizeof(typeof((0 ? 2.0f : (_Complex double) 2.0f))) - sizeof(_Complex double))];
  // Check for proper codegen
  (0 ? 2.0f : (_Complex double) 2.0f);
}

void t92(void) {
  // Check for proper type promotion of conditional expression
  char c[(int)(sizeof(typeof((0 ? (_Complex double) 2.0f : 2.0f))) - sizeof(_Complex double))];
  // Check for proper codegen
  (0 ? (_Complex double) 2.0f : 2.0f);
}

