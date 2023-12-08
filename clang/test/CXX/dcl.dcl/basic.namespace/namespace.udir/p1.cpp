// RUN: %clang_cc1 -fsyntax-only -verify %s

// (this actually occurs before paragraph 1)
namespace test0 {
  namespace A {}
  class B {
    using namespace A; // expected-error {{'using namespace' is not allowed in classes}}
  };
}


struct opaque0 {};
struct opaque1 {};

// Test that names appear as if in deepest common ancestor.
namespace test1 {
  namespace A {
    namespace B {
      opaque0 foo(); // expected-note {{candidate}}
    }
  }

  namespace C {
    opaque1 foo(); // expected-note {{candidate}}

    opaque1 test() {
      using namespace A::B;
      return foo(); // C::foo
    }
  }

  opaque1 test() {
    using namespace A::B;
    using namespace C;
    return foo(); // expected-error {{call to 'foo' is ambiguous}}
  }
}

// Same thing, but with the directives in namespaces.
namespace test2 {
  namespace A {
    namespace B {
      opaque0 foo(); // expected-note {{candidate}}
    }
  }

  namespace C {
    opaque1 foo(); // expected-note {{candidate}}

    namespace test {
      using namespace A::B;

      opaque1 test() {
        return foo(); // C::foo
      }
    }
  }

  namespace test {
    using namespace A::B;
    using namespace C;
    
    opaque1 test() {
      return foo(); // expected-error {{call to 'foo' is ambiguous}}
    }
  }
}

// Transitivity.
namespace test3 {
  namespace A {
    namespace B {
      opaque0 foo();
    }
  }
  namespace C {
    using namespace A;
  }

  opaque0 test0() {
    using namespace C;
    using namespace B;
    return foo();
  }

  namespace D {
    using namespace C;
  }
  namespace A {
    opaque1 foo();
  }

  opaque1 test1() {
    using namespace D;
    return foo();
  }
}

// Transitivity acts like synthetic using directives.
namespace test4 {
  namespace A {
    namespace B {
      opaque0 foo(); // expected-note {{candidate}}
    }
  }
  
  namespace C {
    using namespace A::B;
  }

  opaque1 foo(); // expected-note {{candidate}}

  namespace A {
    namespace D {
      using namespace C;
    }

    opaque0 test() {
      using namespace D;
      return foo();
    }
  }

  opaque0 test() {
    using namespace A::D;
    return foo(); // expected-error {{call to 'foo' is ambiguous}}
  }
}

// Bug: using directives should be followed when parsing default
// arguments in scoped declarations.
class test5 {
  int inc(int x);
};
namespace Test5 {
  int default_x = 0;
}
using namespace Test5;
int test5::inc(int x = default_x) {
  return x+1;
}
