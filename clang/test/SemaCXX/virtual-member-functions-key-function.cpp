// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct A {
  virtual ~A();
#if __cplusplus >= 201103L
// expected-note@-2 2 {{overridden virtual function is here}}
#endif
};

struct B : A {
#if __cplusplus <= 199711L
// expected-error@-2 {{no suitable member 'operator delete' in 'B'}}
#else
// expected-error@-4 {{deleted function '~B' cannot override a non-deleted function}}
// expected-note@-5  {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
#endif
  B() { }
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'B' first required here}}
#endif

  void operator delete(void *, int);
#if __cplusplus <= 199711L
// expected-note@-2 {{'operator delete' declared here}}
#endif
}; 

struct C : A {
#if __cplusplus <= 199711L
// expected-error@-2 {{no suitable member 'operator delete' in 'C'}}
#else
// expected-error@-4 {{deleted function '~C' cannot override a non-deleted function}}
// expected-note@-5  {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
#endif
  void operator delete(void *, int);
#if __cplusplus <= 199711L
// expected-note@-2 {{'operator delete' declared here}}
#endif
}; 

void f() {
  (void)new B; 
  (void)new C;
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'C' first required here}}
#endif
}

// Make sure that the key-function computation is consistent when the
// first virtual member function of a nested class has an inline body.
struct Outer {
  struct Inner {
    virtual void f() { }
    void g();
  };
};

void Outer::Inner::g() { }
