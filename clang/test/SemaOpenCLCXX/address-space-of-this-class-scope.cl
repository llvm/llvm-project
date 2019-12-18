//RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -pedantic -verify

struct C {
  auto fGlob() __global -> decltype(this);
  auto fGen() -> decltype(this);
  auto fErr() __global __local -> decltype(this); //expected-error{{multiple address spaces specified for type}}
};

void bar(__local C*);
// expected-note@-1{{candidate function not viable: cannot pass pointer to address space '__global' as a pointer to address space '__local' in 1st argument}}
// expected-note@-2{{candidate function not viable: cannot pass pointer to default address space as a pointer to address space '__local' in 1st argument}}

__global C Glob;
void foo(){
bar(Glob.fGlob()); // expected-error{{no matching function for call to 'bar'}}
// FIXME: AS of 'this' below should be correctly deduced to generic
bar(Glob.fGen()); // expected-error{{no matching function for call to 'bar'}}
}
