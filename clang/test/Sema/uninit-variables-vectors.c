// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -Wuninitialized -fsyntax-only %s -verify

typedef int __v4si __attribute__((__vector_size__(16)));
typedef float __m128 __attribute__((__vector_size__(16)));
__m128 _mm_xor_ps(__m128 a, __m128 b);
__m128 _mm_loadu_ps(const float *p);

void test1(float *input) {
  __m128 x, y, z, w, X; // expected-note {{variable 'x' is declared here}} expected-note {{variable 'y' is declared here}} expected-note {{variable 'w' is declared here}}  expected-note {{variable 'z' is declared here}}
  x = _mm_xor_ps(x,x); // expected-warning {{variable 'x' is uninitialized when used here}}
  y = _mm_xor_ps(y,y); // expected-warning {{variable 'y' is uninitialized when used here}}
  z = _mm_xor_ps(z,z); // expected-warning {{variable 'z' is uninitialized when used here}}
  w = _mm_xor_ps(w,w); // expected-warning {{variable 'w' is uninitialized when used here}}
  X = _mm_loadu_ps(&input[0]);
  X = _mm_xor_ps(X,X); // no-warning
}

