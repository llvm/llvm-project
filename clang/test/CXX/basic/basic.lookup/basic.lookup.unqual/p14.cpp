// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// C++0x [basic.lookup.unqual]p14:
//   If a variable member of a namespace is defined outside of the
//   scope of its namespace then any name used in the definition of
//   the variable member (after the declarator-id) is looked up as if
//   the definition of the variable member occurred in its namespace.

namespace N { 
  struct S {};
  S i; 
  extern S j;
  extern S j2;
} 

int i = 2; 
N::S N::j = i;
N::S N::j2(i);

// <rdar://problem/13317030>
namespace M {
  class X { };
  inline X operator-(int, X);

  template<typename T>
  class Y { };

  typedef Y<float> YFloat;

  namespace yfloat {
    YFloat operator-(YFloat, YFloat);
  }
  using namespace yfloat;
}

using namespace M;

namespace M {

class Other {
  void foo(YFloat a, YFloat b);
};

}

void Other::foo(YFloat a, YFloat b) {
  YFloat c = a - b;
}

// <rdar://problem/13540899>
namespace Other {
  void other_foo();
}

namespace M2 {
  using namespace Other;

  extern "C" {
    namespace MInner {
      extern "C" {
        class Bar { 
          void bar();
        };
      }
    }
  }
}

void M2::MInner::Bar::bar() {
  other_foo();
}
