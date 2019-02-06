//RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -pedantic -verify

struct C {
  void m1() __local __local; //expected-warning{{multiple identical address spaces specified for type}}
  //expected-note@-1{{candidate function}}
  void m1() __global;
  //expected-note@-1{{candidate function}}
  void m2() __global __local; //expected-error{{multiple address spaces specified for type}}
};

__global C c_glob;

__kernel void bar() {
  __local C c_loc;
  C c_priv;

  c_glob.m1();
  c_loc.m1();
  c_priv.m1(); //expected-error{{no matching member function for call to 'm1'}}
}
