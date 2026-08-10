// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -cl-std=CL2.0 -verify %s

kernel int bar()  { // expected-error {{kernel must have void return type}}
  return 6;
}

kernel void main() { // expected-error {{kernel cannot be called 'main'}}

}

int main() { // expected-error {{function cannot be called 'main'}}
  return 0;
}

int* global x(int* x) { // expected-error {{return type cannot be qualified with address space}}
  return x + 1;
}

int* local x(int* x) { // expected-error {{return type cannot be qualified with address space}}
  return x + 1;
}

int* constant x(int* x) { // expected-error {{return type cannot be qualified with address space}}
  return x + 1;
}
