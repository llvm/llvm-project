// RUN: %clang_cc1 -fsyntax-only -verify %s 
namespace A { // expected-note 2 {{previous definition is here}}
  int A;
  void f() { A = 0; }
}

void f() { A = 0; } // expected-error {{unexpected namespace name 'A': expected expression}}
int A; // expected-error {{redefinition of 'A' as different kind of symbol}}
class A; // expected-error {{redefinition of 'A' as different kind of symbol}}

class B {}; // expected-note {{previous definition is here}} \
            // expected-note{{candidate function (the implicit copy assignment operator)}}

void C(); // expected-note {{previous definition is here}}
namespace C {} // expected-error {{redefinition of 'C' as different kind of symbol}}

namespace D {
  class D {};
}

namespace S1 {
  int x;

  namespace S2 {

    namespace S3 {
      B x;
    }
  }
}

namespace S1 {
  void f() {
    x = 0;
  }

  namespace S2 {
    
    namespace S3 {
      void f() {
        x = 0; // expected-error {{no viable overloaded '='}}
      }
    }

    int y;
  }
}

namespace S1 {
  namespace S2 {
    namespace S3 {
      void f3() {
        y = 0;
      }
    }
  }
}

namespace B {} // expected-error {{redefinition of 'B' as different kind of symbol}}


namespace foo {
  enum x {
    Y
  };
}

static foo::x  test1;  // ok

static foo::X  test2;  // typo: expected-error {{no type named 'X' in}}

namespace PR6620 {
  namespace numeric {
    namespace op {
      struct greater {};
    }
    namespace {
      extern op::greater const greater;
    }
  }

  namespace numeric {
    namespace {
      op::greater const greater = op::greater();
    }

    template<typename T, typename U>
    int f(T& l, U& r)
    { numeric::greater(l, r); }

  }
}
