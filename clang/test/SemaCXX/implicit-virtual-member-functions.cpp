// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify -std=c++11 %s

struct A {
  virtual ~A();
#if __cplusplus >= 201103L
// expected-note@-2 3 {{overridden virtual function is here}}
#endif
};

struct B : A {
#if __cplusplus <= 199711L
// expected-error@-2 {{no suitable member 'operator delete' in 'B'}}
#else
// expected-error@-4 {{deleted function '~B' cannot override a non-deleted function}}
// expected-note@-5  {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
#ifdef MSABI
// expected-note@-7 {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
#endif
#endif
  virtual void f();

  void operator delete (void *, int);
#if __cplusplus <= 199711L
// expected-note@-2 {{'operator delete' declared here}}
#endif
};

#ifdef MSABI
B b;
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'B' first required here}}
#else
// expected-error@-4 {{attempt to use a deleted function}}
#endif

#else
void B::f() {
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'B' first required here}}
#endif
}
#endif

struct C : A {
#if __cplusplus <= 199711L
// expected-error@-2 {{no suitable member 'operator delete' in 'C'}}
#else
// expected-error@-4 {{deleted function '~C' cannot override a non-deleted function}}
// expected-note@-5  {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
#endif

  C();
  void operator delete(void *, int);
#if __cplusplus <= 199711L
// expected-note@-2 {{'operator delete' declared here}}
#endif
};

C::C() { }
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'C' first required here}}
#endif

struct D : A {
#if __cplusplus <= 199711L
// expected-error@-2 {{no suitable member 'operator delete' in 'D'}}
#else
// expected-error@-4 {{deleted function '~D' cannot override a non-deleted function}}
// expected-note@-5  {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
#endif
  void operator delete(void *, int);
#if __cplusplus <= 199711L
// expected-note@-2 {{'operator delete' declared here}}
#endif
}; 

void f() {
  new D;
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'D' first required here}}
#endif
}
