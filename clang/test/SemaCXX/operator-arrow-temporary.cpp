// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR9615

struct Resource {
  void doit();
};

template<int x> struct Lock {
  ~Lock() { int a[x]; } // expected-error {{declared as an array with a negative size}}
  Resource* operator->() { return 0; }
};

struct Accessor {
  Lock<-1> operator->();
};

// Make sure we try to instantiate the destructor for Lock here
void f() { Accessor acc; acc->doit(); } // expected-note {{requested here}}

