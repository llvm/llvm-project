// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify -std=c++11 %s
// PR7800

// The Microsoft ABI doesn't have the concept of key functions, so we have different
// expectations about when functions are first required for that case.

class NoDestroy { ~NoDestroy(); };
#if __cplusplus <= 199711L
// expected-note@-2 3 {{declared private here}}
#ifdef MSABI
// expected-note@-4 3 {{declared private here}}
#endif
#endif

struct A {
  virtual ~A();
#if __cplusplus >= 201103L
  // expected-note@-2 3 {{overridden virtual function is here}}
#endif
};

struct B : public virtual A {
#if __cplusplus >= 201103L
// expected-error@-2 {{deleted function '~B' cannot override a non-deleted function}}
// expected-note@-3 {{overridden virtual function is here}}
#endif

  NoDestroy x;
#if __cplusplus <= 199711L
  // expected-error@-2 {{field of type 'NoDestroy' has private destructor}}
#ifdef MSABI
  // expected-error@-4 {{field of type 'NoDestroy' has private destructor}}
#endif
#else
  // expected-note@-7 {{destructor of 'B' is implicitly deleted because field 'x' has an inaccessible destructor}}
#ifdef MSABI
  // expected-note@-9 {{default constructor of 'B' is implicitly deleted because field 'x' has an inaccessible destructor}}
#endif
#endif
};

struct D : public virtual B {
#if __cplusplus <= 199711L
#ifdef MSABI
// expected-note@-3 {{implicit default constructor for 'B' first required here}}
// expected-note@-4 {{implicit destructor for 'B' first required here}}
#endif
#else
#ifdef MSABI
// expected-note@-8 {{default constructor of 'D' is implicitly deleted because base class 'B' has a deleted default constructor}}
#endif
#endif
  virtual void foo();
  ~D();
#if __cplusplus >= 201103L
  //expected-error@-2 {{non-deleted function '~D' cannot override a deleted function}}
#endif
};

#ifdef MSABI
D d;
#if __cplusplus <= 199711L
// expected-note@-2 2{{implicit default constructor for 'D' first required here}}
#else
// expected-error@-4 {{call to implicitly-deleted default constructor of 'D'}}
#endif
#else
void D::foo() {
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'B' first required here}}
#endif
}
#endif

struct E : public virtual A {
#if __cplusplus >= 201103L
// expected-error@-2 {{deleted function '~E' cannot override a non-deleted function}}
// expected-note@-3 {{overridden virtual function is here}}
#endif

  NoDestroy x;
#if __cplusplus <= 199711L
  // expected-error@-2 {{field of type 'NoDestroy' has private destructor}}
#ifdef MSABI
  // expected-error@-4 {{field of type 'NoDestroy' has private destructor}}
#endif
#else
  // expected-note@-7 {{destructor of 'E' is implicitly deleted because field 'x' has an inaccessible destructor}}
#ifdef MSABI
  // expected-note@-9 {{default constructor of 'E' is implicitly deleted because field 'x' has an inaccessible destructor}}
#endif
#endif
};

struct F : public E {
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'E' first required here}}
#ifdef MSABI
// expected-note@-4 {{implicit default constructor for 'E' first required here}}
#endif
#else
// expected-error@-7 {{non-deleted function '~F' cannot override a deleted function}}
// expected-note@-8 {{while declaring the implicit destructor for 'F'}}
// expected-note@-9 {{overridden virtual function is here}}
#ifdef MSABI
// expected-note@-11 {{default constructor of 'F' is implicitly deleted because base class 'E' has a deleted default constructor}}
#endif
#endif
};


struct G : public virtual F {
#ifdef MSABI
#if __cplusplus <= 199711L
// expected-note@-3 {{implicit default constructor for 'F' first required here}}
// expected-note@-4 {{implicit destructor for 'F' first required here}}
#else
// expected-note@-6 {{default constructor of 'G' is implicitly deleted because base class 'F' has a deleted default constructor}}
#endif
#endif

  virtual void foo();
  ~G();
#if __cplusplus >= 201103L
  //expected-error@-2 {{non-deleted function '~G' cannot override a deleted function}}
#endif
};

#ifdef MSABI
G g;
#if __cplusplus <= 199711L
// expected-note@-2 2{{implicit default constructor for 'G' first required here}}
#else
// expected-error@-4 {{call to implicitly-deleted default constructor of 'G'}}
#endif
#else
void G::foo() {
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'F' first required here}}
#endif
}
#endif

struct H : public virtual A {
#if __cplusplus >= 201103L
// expected-error@-2 {{deleted function '~H' cannot override a non-deleted function}}
// expected-note@-3 {{overridden virtual function is here}}
#endif

  NoDestroy x;
#if __cplusplus <= 199711L
  // expected-error@-2 {{field of type 'NoDestroy' has private destructor}}
#ifdef MSABI
  // expected-error@-4 {{field of type 'NoDestroy' has private destructor}}
#endif
#else
  // expected-note@-7 {{destructor of 'H' is implicitly deleted because field 'x' has an inaccessible destructor}}
#ifdef MSABI
  // expected-note@-9 {{default constructor of 'H' is implicitly deleted because field 'x' has an inaccessible destructor}}
#endif
#endif
};

struct I : public virtual H {
#ifdef MSABI
#if __cplusplus > 199711L
// expected-note@-3 {{default constructor of 'I' is implicitly deleted because base class 'H' has a deleted default constructor}}
#endif
#endif

  ~I();
#if __cplusplus >= 201103L
// expected-error@-2 {{non-deleted function '~I' cannot override a deleted function}}
#endif
};

struct J : public I {
#ifdef MSABI
#if __cplusplus <= 199711L
// expected-note@-3 {{implicit default constructor for 'H' first required here}}
// expected-note@-4 {{implicit destructor for 'H' first required here}}
#else
// expected-note@-6 {{default constructor of 'J' is implicitly deleted because base class 'I' has a deleted default constructor}}
#endif
#endif

  virtual void foo();
  ~J();
};

#ifdef MSABI
J j;
#if __cplusplus <= 199711L
// expected-note@-2 2{{implicit default constructor for 'J' first required here}}
#else
// expected-error@-4 {{call to implicitly-deleted default constructor of 'J'}}
#endif

#else
void J::foo() {
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor for 'H' first required here}}
#endif
}
#endif
