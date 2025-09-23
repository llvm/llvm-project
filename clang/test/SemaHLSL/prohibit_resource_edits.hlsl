// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -o - -fsyntax-only %s -verify

RWBuffer<float> A[10]; // expected-note {{variable 'A' is declared here}} expected-note {{variable 'A' is declared here}} // expected-note {{variable 'A' is declared here}}
RWBuffer<float> B;     // expected-note {{variable 'B' is declared here}}
RWBuffer<float> C[10];

void test() {
  // expected-error@+1{{assignment to global resource variable 'A' is not allowed}}
  A = C; 

  // expected-error@+1{{assignment to global resource variable 'B' is not allowed}}
  B = C[0];

  // expected-error@+1{{assignment to global resource variable 'A' is not allowed}}
  A[1] = B;

  // expected-error@+1{{assignment to global resource variable 'A' is not allowed}}
  A[1] = C[2];

  // local resources are assignable 
  RWBuffer<float> LocalA[10] = A; // ok
  RWBuffer<float> LocalB = B; // ok

  // read-write resources can be written to
  A[0][0] = 1.0f; // ok
  B[0] = 2.0f; // ok
}
