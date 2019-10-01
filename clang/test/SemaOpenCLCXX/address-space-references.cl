// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -pedantic -verify -fsyntax-only

__global const int& f(__global float &ref) {
  return ref; // expected-error{{reference of type 'const __global int &' cannot bind to a temporary object because of address space mismatch}}
}
