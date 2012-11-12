// RUN: %clang_cc1 -fsyntax-only -verify %s

void nondecl(int (*f)(int x = 5)) // expected-error {{default arguments can only be specified}}
{
  void (*f2)(int = 17)  // expected-error {{default arguments can only be specified}}
  = (void (*)(int = 42))f; // expected-error {{default arguments can only be specified}}
}

struct X0 {
  int (*f)(int = 17); // expected-error{{default arguments can only be specified for parameters in a function declaration}}
  
  void mem8(int (*fp)(int) = (int (*)(int = 17))0); // expected-error{{default arguments can only be specified for parameters in a function declaration}}  
};
