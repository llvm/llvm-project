// RUN: %clang_cc1 -fsyntax-only -verify %s

void f1() {
  struct X {
    struct Y;
  };

  struct X::Y {
    void f() {}
  };
}

void f2() {
  struct X {
    struct Y;

    struct Y {
      void f() {}
    };
  };
}

// A class nested within a local class is a local class.
void f3(int a) { // expected-note{{'a' declared here}}
  struct X {
    struct Y {
      int f() { return a; } // expected-error{{reference to local variable 'a' declared in enclosing function 'f3'}}
    };
  };
}

void f4() {
  {
    struct A {
      struct B;
      struct B {};
    };
  }
  {
    struct A { struct B; };
    struct A::B {};
  }
  {
    struct A { struct B; };
    {
      struct A { struct B; };
      struct A::B {};
    }
    struct A::B {};
  }
  {
    struct A { struct B; };
    struct A::B { struct C; };
    struct A::B::C {};
  }
  {
    struct A {
      void f() {
        struct B { struct C; };
        struct B::C {};
      }
    };
  }
  {
    struct A { struct B; }; // expected-note {{'A' declared here}}
    {
      struct A::B {}; // expected-error {{nested local class 'B' must be defined in the same block scope as its parent class 'A'}}
    }
  }
  {
    struct A { struct B { struct C; }; }; // expected-note {{'A' declared here}}
    {
      struct A::B::C {}; // expected-error {{nested local class 'C' must be defined in the same block scope as its parent class 'A'}}
    }
  }
  {
    struct A { struct B; }; // expected-note {{'A' declared here}}
    using AliasForA = A;
    {
      struct AliasForA::B {}; // expected-error {{nested local class 'B' must be defined in the same block scope as its parent class 'A'}}
    }
  }
}

template <typename>
void f5() {
  struct A { struct B; }; // expected-note {{'A' declared here}}
  {
    struct A::B {}; // expected-error {{nested local class 'B' must be defined in the same block scope as its parent class 'A'}}
  }
}
