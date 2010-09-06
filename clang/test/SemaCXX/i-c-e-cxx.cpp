// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++-specific tests for integral constant expressions.

const int c = 10;
int ar[c];

struct X0 {
  static const int value = static_cast<int>(4.0);
};

void f() {
  if (const int value = 17) {
    int array[value];
  }
}

int a() {
  const int t=t;
  switch(1) { // expected-warning {{no case matching constant switch condition '1'}}
    case t:; // expected-error {{not an integer constant expression}}
  }
}

// PR6206:  out-of-line definitions are legit
namespace pr6206 {
  class Foo {
  public:
    static const int kBar;
  };

  const int Foo::kBar = 20;
  
  char Test() {
    char str[Foo::kBar];
    str[0] = '0';
    return str[0];
  }
}

// PR6373:  default arguments don't count.
void pr6373(const unsigned x = 0) {
  unsigned max = 80 / x;
}
