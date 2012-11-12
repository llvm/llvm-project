// RUN: %clang_cc1 -Wreturn-type -fsyntax-only -std=c++11 -verify %s

class A {
public:
  A(const A&);
};

struct S {
  int i;
  double d;

  virtual void B() {}
};

union U {
  struct {
    int i;
    virtual void B() {} // Can only do this in C++11
  } t;
};

struct S2 {
  int i;
  double d;
};

extern "C" U f3( void ); // expected-warning {{'f3' has C-linkage specified, but returns user-defined type 'U' which is incompatible with C}}
extern "C" S f0(void); // expected-warning {{'f0' has C-linkage specified, but returns user-defined type 'S' which is incompatible with C}}
extern "C" A f4( void ); // expected-warning {{'f4' has C-linkage specified, but returns user-defined type 'A' which is incompatible with C}}

// These should all be fine
extern "C" S2 f5( void );
extern "C" void f2( A x );
extern "C" void f6( S s );
extern "C" void f7( U u );
extern "C" double f8(void);
extern "C" long long f11( void );
extern "C" A *f10( void );

extern "C" struct mypodstruct f12(); // expected-warning {{'f12' has C-linkage specified, but returns incomplete type 'struct mypodstruct' which could be incompatible with C}}
