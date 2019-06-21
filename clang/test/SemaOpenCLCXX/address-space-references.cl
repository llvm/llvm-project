// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -pedantic -verify -fsyntax-only

__global const int& f(__global float &ref) {
  return ref; // expected-error{{reference of type 'const __global int &' cannot bind to a temporary object because of address space mismatch}}
}

int bar(const __global unsigned int &i); // expected-note{{passing argument to parameter 'i' here}}
//FIXME: With the overload below the call should be resolved
// successfully. However, current overload resolution logic
// can't detect this case and therefore fails.
int bar(const unsigned int &i);

void foo() {
  bar(1) // expected-error{{binding reference of type 'const __global unsigned int' to value of type 'int' changes address space}}
}
