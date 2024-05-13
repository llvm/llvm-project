// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
  class X { }; // expected-note{{candidate found by name lookup is 'A::X'}}
               // expected-note@-1{{candidate found by name lookup is 'A::X'}}
}
namespace B {
  void X(int); // expected-note{{candidate found by name lookup is 'B::X'}}
               // expected-note@-1{{candidate found by name lookup is 'B::X'}}
}

// Using directive doesn't cause A::X to be hidden, so X is ambiguous.
namespace Test1a {
  using namespace A;
  using namespace B;

  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test1b {
  using namespace B;
  using namespace A;

  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

// The behaviour here should be the same as using namespaces A and B directly
namespace Test2a {
  namespace C {
    using A::X; // expected-note{{candidate found by name lookup is 'Test2a::C::X'}}
  }
  namespace D {
    using B::X; // expected-note{{candidate found by name lookup is 'Test2a::D::X'}}
  }
  using namespace C;
  using namespace D;

  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test2b {
  namespace C {
    using A::X; // expected-note{{candidate found by name lookup is 'Test2b::C::X'}}
  }
  namespace D {
    using B::X; // expected-note{{candidate found by name lookup is 'Test2b::D::X'}}
  }
  using namespace D;
  using namespace C;

  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

// Defining a function X inside C should hide using A::X in C but not D, so the result is ambiguous.
namespace Test3a {
  namespace C {
    using A::X;
    void X(int); // expected-note{{candidate found by name lookup is 'Test3a::C::X'}}
  }
  namespace D {
    using A::X; // expected-note{{candidate found by name lookup is 'Test3a::D::X'}}
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test3b {
  namespace C {
    using A::X;
    void X(int); // expected-note{{candidate found by name lookup is 'Test3b::C::X'}}
  }
  namespace D {
    using A::X; // expected-note{{candidate found by name lookup is 'Test3b::D::X'}}
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test3c {
  namespace C {
    void X(int); // expected-note{{candidate found by name lookup is 'Test3c::C::X'}}
    using A::X;
  }
  namespace D {
    using A::X; // expected-note{{candidate found by name lookup is 'Test3c::D::X'}}
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test3d {
  namespace C {
    void X(int); // expected-note{{candidate found by name lookup is 'Test3d::C::X'}}
    using A::X;
  }
  namespace D {
    using A::X; // expected-note{{candidate found by name lookup is 'Test3d::D::X'}}
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

// A::X hidden in both C and D by overloaded function, so the result is not ambiguous.
namespace Test4a {
  namespace C {
    using A::X;
    void X(int);
  }
  namespace D {
    using A::X;
    void X(int, int);
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1);
  }
}

namespace Test4b {
  namespace C {
    using A::X;
    void X(int);
  }
  namespace D {
    using A::X;
    void X(int, int);
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1);
  }
}

namespace Test4c {
  namespace C {
    void X(int);
    using A::X;
  }
  namespace D {
    void X(int, int);
    using A::X;
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1);
  }
}

namespace Test4d {
  namespace C {
    void X(int);
    using A::X;
  }
  namespace D {
    void X(int, int);
    using A::X;
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1);
  }
}

// B::X hides class X in C, so the the result is not ambiguous
namespace Test5a {
  namespace C {
    using B::X;
    class X { };
  }
  namespace D {
    using B::X;
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1);
  }
}

namespace Test5b {
  namespace C {
    using B::X;
    class X { };
  }
  namespace D {
    using B::X;
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1);
  }
}

namespace Test5c {
  namespace C {
    class X { };
    using B::X;
  }
  namespace D {
    using B::X;
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1);
  }
}

namespace Test5d {
  namespace C {
    class X { };
    using B::X;
  }
  namespace D {
    using B::X;
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1);
  }
}

// B::X hides class X declared in both C and D, so the result is not ambiguous.
namespace Test6a {
  namespace C {
    class X { };
    using B::X;
  }
  namespace D {
    class X { };
    using B::X;
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1);
  }
}

namespace Test6b {
  namespace C {
    class X { };
    using B::X;
  }
  namespace D {
    class X { };
    using B::X;
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1);
  }
}

namespace Test6c {
  namespace C {
    using B::X;
    class X { };
  }
  namespace D {
    using B::X;
    class X { };
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1);
  }
}

namespace Test6d {
  namespace C {
    using B::X;
    class X { };
  }
  namespace D {
    using B::X;
    class X { };
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1);
  }
}

// function X inside C should hide class X in C but not D.
namespace Test7a {
  namespace C {
    class X;
    void X(int); // expected-note{{candidate found by name lookup is 'Test7a::C::X'}}
  }
  namespace D {
    class X; // expected-note{{candidate found by name lookup is 'Test7a::D::X'}}
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test7b {
  namespace C {
    class X;
    void X(int); // expected-note{{candidate found by name lookup is 'Test7b::C::X'}}
  }
  namespace D {
    class X; // expected-note{{candidate found by name lookup is 'Test7b::D::X'}}
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test7c {
  namespace C {
    void X(int); // expected-note{{candidate found by name lookup is 'Test7c::C::X'}}
    class X;
  }
  namespace D {
    class X; // expected-note{{candidate found by name lookup is 'Test7c::D::X'}}
  }
  using namespace C;
  using namespace D;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}

namespace Test7d {
  namespace C {
    void X(int); // expected-note{{candidate found by name lookup is 'Test7d::C::X'}}
    class X;
  }
  namespace D {
    class X; // expected-note{{candidate found by name lookup is 'Test7d::D::X'}}
  }
  using namespace D;
  using namespace C;
  void f() {
    X(1); // expected-error{{reference to 'X' is ambiguous}}
  }
}
