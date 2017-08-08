// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -std=gnu++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -std=gnu++11 %s

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
  const int t=t; // expected-note {{declared here}}
#if __cplusplus <= 199711L
  // expected-note@-2 {{read of object outside its lifetime}}
#endif

  switch(1) {
#if __cplusplus <= 199711L
  // expected-warning@-2 {{no case matching constant switch condition '1'}}
#endif
    case t:; // expected-note {{initializer of 't' is not a constant expression}}
#if __cplusplus <= 199711L
    // expected-error@-2 {{not an integral constant expression}}
#else
    // expected-error@-4 {{case value is not a constant expression}}
#endif
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


// rdar://9204520
namespace rdar9204520 {
  
struct A {
  static const int B = int(0.75 * 1000 * 1000);
#if __cplusplus <= 199711L
  // expected-warning@-2 {{not a constant expression; folding it to a constant is a GNU extension}}
#endif
};

int foo() { return A::B; }
}

// PR11040
const int x = 10;
int* y = reinterpret_cast<const char&>(x); // expected-error {{cannot initialize}}

// This isn't an integral constant expression, but make sure it folds anyway.
struct PR8836 { char _; long long a; };
#if __cplusplus <= 199711L
// expected-warning@-2 {{'long long' is a C++11 extension}}
#endif

int PR8836test[(__typeof(sizeof(int)))&reinterpret_cast<const volatile char&>((((PR8836*)0)->a))];
// expected-warning@-1 {{folded to constant array as an extension}}
// expected-note@-2 {{cast that performs the conversions of a reinterpret_cast is not allowed in a constant expression}}

const int nonconst = 1.0;
#if __cplusplus <= 199711L
// expected-note@-2 {{declared here}}
#endif
int arr[nonconst];
#if __cplusplus <= 199711L
// expected-warning@-2 {{folded to constant array as an extension}}
// expected-note@-3 {{initializer of 'nonconst' is not a constant expression}}
#endif

const int castfloat = static_cast<int>(1.0);
int arr2[castfloat]; // ok
