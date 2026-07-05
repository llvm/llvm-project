// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
  short i; // expected-note 2{{candidate found by name lookup is 'A::i'}}
  namespace B {
    long i; // expected-note{{candidate found by name lookup is 'A::B::i'}}
    void f() {} // expected-note{{candidate function}}
    int k;
    namespace E {} // \
      expected-note{{candidate found by name lookup is 'A::B::E'}}
  }

  namespace E {} // expected-note{{candidate found by name lookup is 'A::E'}}

  namespace C {
    using namespace B;
    namespace E {} // \
      expected-note{{candidate found by name lookup is 'A::C::E'}}
  }

  void f() {} // expected-note{{candidate function}}

  class K1 {
    void foo();
  };

  void local_i() {
    char i;
    using namespace A;
    using namespace B;
    int a[sizeof(i) == sizeof(char)? 1 : -1]; // okay
  }
  namespace B {
    int j;
  }

  void ambig_i() {
    using namespace A;
    using namespace A::B;
    (void) i; // expected-error{{reference to 'i' is ambiguous}}
    f(); // expected-error{{call to 'f' is ambiguous}}
    (void) j; // okay
    using namespace C;
    (void) k; // okay
    using namespace E; // expected-error{{reference to 'E' is ambiguous}}
  }

  struct K2 {}; // expected-note 2{{candidate found by name lookup is 'A::K2'}}
}

struct K2 {}; // expected-note 2{{candidate found by name lookup is 'K2'}}

using namespace A;

void K1::foo() {} // okay

struct K2 *k2; // expected-error{{reference to 'K2' is ambiguous}}

K2 *k3; // expected-error{{reference to 'K2' is ambiguous}}

class X { // expected-note{{candidate found by name lookup is 'X'}}
  // FIXME: produce a suitable error message for this
  using namespace A; // expected-error{{not allowed}}
};

namespace N {
  struct K2;
  struct K2 { };
}

namespace Ni {
 int i(); // expected-note{{candidate found by name lookup is 'Ni::i'}}
}

namespace NiTest {
 using namespace A;
 using namespace Ni;

 int test() {
   return i; // expected-error{{reference to 'i' is ambiguous}}
 }
}

namespace OneTag {
  struct X; // expected-note{{candidate found by name lookup is 'OneTag::X'}}
}

namespace OneFunction {
  void X(); // expected-note{{candidate found by name lookup is 'OneFunction::X'}}
}

namespace TwoTag {
  struct X; // expected-note{{candidate found by name lookup is 'TwoTag::X'}}
}

namespace FuncHidesTagAmbiguity {
  using namespace OneTag;
  using namespace OneFunction;
  using namespace TwoTag;

  void test() {
    (void)X(); // expected-error{{reference to 'X' is ambiguous}}
  }
}

// PR5479
namespace Aliased {
  void inAliased();
}
namespace Alias = Aliased;
using namespace Alias;
void testAlias() {
  inAliased();
}

namespace N { void f2(int); }

extern "C++" {
  using namespace N;
  void f3() { f2(1); }
}

void f4() { f2(1); }

// PR7517
using namespace std; // expected-warning{{using directive refers to implicitly-defined namespace 'std'}}
using namespace ::std; // expected-warning{{using directive refers to implicitly-defined namespace 'std'}}

namespace test1 {
  namespace ns { typedef int test1; }
  template <class T> using namespace ns; // expected-error {{cannot template a using directive}}

  // Test that we recovered okay.
  test1 x;
}
