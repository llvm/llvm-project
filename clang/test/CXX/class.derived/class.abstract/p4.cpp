// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR6631 {
  struct A { 
    virtual void f() = 0;
  };

  struct B : virtual A { };

  struct C : virtual A { 
    virtual void f();
  };

  struct D : public B, public C { 
    virtual void f();
  };

  void f() {
    (void)new D; // okay
  }
}

// Check cases where we have a virtual function that is pure in one
// subobject but not pure in another subobject.
namespace PartlyPure {
  struct A { 
    virtual void f() = 0; // expected-note{{unimplemented pure virtual method}}
  };

  struct B : A {
    virtual void f();
  };

  struct C : virtual A { };

  struct D : B, C { };

  void f() {
    (void) new D; // expected-error{{abstract class}}
  }
}

namespace NonPureAlongOnePath {
  struct A { 
    virtual void f() = 0;
  };

  struct B : virtual A {
    virtual void f();
  };

  struct C : virtual A { };

  struct D : B, C { };

  void f() {
    (void) new D; // okay
  }  
}

namespace NonPureAlongOnePath2 {
  struct Aprime { 
    virtual void f() = 0;
  };

  struct A : Aprime {
  };

  struct B : virtual A {
    virtual void f();
  };

  struct C : virtual A { };

  struct D : B, C { };

  void f() {
    (void) new D; // okay
  }  
}
