// RUN: %clang_cc1 -fsyntax-only -verify -verify-directives %s

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
void f3(int a) { // #a-param
  struct X {
    struct Y {
      int f() { return a; }
      // expected-error@-1 {{reference to local variable 'a' declared in enclosing function 'f3'}}
       // expected-note@#a-param {{'a' declared here}}
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
    struct A { struct B; }; // #A1
    {
      struct A::B {};
      // expected-error@-1 {{nested local class 'B' must be defined in the same block scope as its parent class 'A'}}
      //   expected-note@#A1 {{'A' declared here}}
    }
  }
  {
    struct A { struct B { struct C; }; }; // #A2
    {
      struct A::B::C {};
      // expected-error@-1 {{nested local class 'C' must be defined in the same block scope as its parent class 'A'}}
      //   expected-note@#A2 {{'A' declared here}}
    }
  }
  {
    struct A { struct B; }; // #A3
    using AliasForA = A;
    {
      struct AliasForA::B {}; 
      // expected-error@-1 {{nested local class 'B' must be defined in the same block scope as its parent class 'A'}}
      //   expected-note@#A3 {{'A' declared here}}
    }
  }
}

template <typename>
void f5() {
  struct A { struct B; }; // #A4
  {
    struct A::B {};
    // expected-error@-1 {{nested local class 'B' must be defined in the same block scope as its parent class 'A'}}
    //   expected-note@#A4 {{'A' declared here}}
  }
}
