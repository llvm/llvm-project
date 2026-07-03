// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

void increment(inout int Arr[2]) {
  for (int I = 0; I < 2; I++)
    Arr[0] += 2;
}

export int wrongSize() {
  int A[3] = { 0, 1, 2 };
  increment(A);
  // expected-error@-1 {{no matching function for call to 'increment'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'int[3]' to 'int[2]' for 1st argument}}
  return A[0];
}

export int wrongSize2() {
  int A[1] = { 0 };
  increment(A);
  // expected-error@-1 {{no matching function for call to 'increment'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'int[1]' to 'int[2]' for 1st argument}}
  return A[0];
}

export void tooFewArgs() {
  increment();
  // expected-error@-1 {{no matching function for call to 'increment'}}
  // expected-note@*:* {{candidate function not viable: requires single argument 'Arr', but no arguments were provided}}
}

export float wrongType() {
  float A[2] = { 0, 1 };
  increment(A);
  // expected-error@-1 {{no matching function for call to 'increment'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'float[2]' to 'int[2]' for 1st argument}}
  return A[0];
}

export int wrongType2() {
  increment(5);
  // expected-error@-1 {{no matching function for call to 'increment'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'int' to 'int[2]' for 1st argument}}
  return 1;
}

export void tooManyArgs() {
  int A[2] = { 0, 1 };
  int B[2] = { 2, 3 };
  increment(A, B);
  // expected-error@-1 {{no matching function for call to 'increment'}}
  // expected-note@*:* {{candidate function not viable: requires single argument 'Arr', but 2 arguments were provided}}
}
