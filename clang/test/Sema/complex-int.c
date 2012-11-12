// RUN: %clang_cc1 %s -verify -fsyntax-only

void a() {
__complex__ int arr;
__complex__ short brr;
__complex__ unsigned xx;
__complex__ signed yy;
__complex__ int result;
int ii;
int aa = 1 + 1.0iF;

result = arr*ii;
result = ii*brr;

result = arr*brr;
result = xx*yy;

switch (arr) { // expected-error{{statement requires expression of integer type ('_Complex int' invalid)}}
  case brr: ; // expected-error{{expression is not an integer constant expression}}
  case xx: ; // expected-error{{expression is not an integer constant expression}}
}
}

void Tester() {
__complex short a1;
__complex int a2;
__complex float a3;
__complex double a4;
short a5;
int a6;
float a7;
double a8;
#define TestPair(m,n) int x##m##n = a##m+a##n;
#define TestPairs(m) TestPair(m,1) TestPair(m,2) \
                    TestPair(m,3) TestPair(m,4) \
                    TestPair(m,5) TestPair(m,6) \
                    TestPair(m,7) TestPair(m,8)
TestPairs(1); TestPairs(2);
TestPairs(3); TestPairs(4);
TestPairs(5); TestPairs(6);
TestPairs(7); TestPairs(8);
}

// rdar://6097730
void test3(_Complex int *x) {
  *x = ~*x;
}

void test4(_Complex float *x) {
  *x = ~*x;
}

void test5(_Complex int *x) {
  (*x)++;
}

int i1[(2+3i)*(5+7i) == 29i-11 ? 1 : -1];
int i2[(29i-11)/(5+7i) == 2+3i ? 1 : -1];
int i3[-(2+3i) == +(-3i-2) ? 1 : -1];
int i4[~(2+3i) == 2-3i ? 1 : -1];
int i5[(3i == -(-3i) ? ((void)3, 1i - 1) : 0) == 1i - 1 ? 1 : -1];

int f1[(2.0+3.0i)*(5.0+7.0i) == 29.0i-11.0 ? 1 : -1];
int f2[(29.0i-11.0)/(5.0+7.0i) == 2.0+3.0i ? 1 : -1];
int f3[-(2.0+3.0i) == +(-3.0i-2.0) ? 1 : -1];
int f4[~(2.0+3.0i) == 2.0-3.0i ? 1 : -1];
int f5[(3.0i == -(-3.0i) ? ((void)3.0, __extension__ (1.0i - 1.0)) : 0) == 1.0i - 1.0 ? 1 : -1];
