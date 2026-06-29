// RUN: %clang_cc1                                         -verify=ref,both      -std=c++26 %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both -std=c++26 %s

namespace Simple {

  struct S {};
  constexpr S ss;
  constexpr int foo = (dynamic_cast<const S &>(ss), 0);

  struct S1 { virtual void a(); };
  struct S2 : S1 {};
  constexpr S2 s{};
  static_assert(dynamic_cast<const S2*>(static_cast<const S1*>(&s)) == &s);
}

namespace Failing {
}

namespace NotSoSimple {
  struct A2 { virtual void a2(); };
  struct D {
    virtual void d();
  };
  struct F : A2, D {
    void *f = dynamic_cast<void*>( static_cast<D*>(this) );
  };
  constexpr F g;
  static_assert(g.f == (void*)(F*)&g);

  constexpr D* d = (D*)&g;
  constexpr void* f = dynamic_cast<void*>(d);
  static_assert(f == &g);
}


namespace Again {
  struct A3 {};
  struct A4 {};

  struct A2 : A3, A4 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct C2 { virtual void c2(); };
  struct C : A, C2 { A4 *c = dynamic_cast<A4*>(static_cast<C2*>(this)); };

  struct D { virtual void d(); };

  struct F : C, D{};
  struct G : F {};
  constexpr G g;
  static_assert(g.c == (C*)&g);
}

namespace FailedReference {
  struct A {};
  struct K : A {};
  struct P : A {};

  struct L{virtual void p();};

  struct S : P, K, L {
    virtual void p();
  };
  constexpr S s{};
  static_assert(&dynamic_cast<const A&>((L&)s) == nullptr); // both-error {{not an integral constant expression}} \
                                                            // both-note {{reference dynamic_cast failed: 'A' is an ambiguous base class of dynamic type 'FailedReference::S' of operand}}
}

namespace FailedReference2 {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 { A *c = dynamic_cast<A*>(static_cast<C2*>(this)); };
  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D, private E { void *f = dynamic_cast<void*>(static_cast<D*>(this)); };
  struct G : F {};

  constexpr G g;
  static_assert(&dynamic_cast<A&>((D&)g) == nullptr); // both-error {{not an integral constant expression}} \
                                                      // both-note {{reference dynamic_cast failed: 'A' is an ambiguous base class of dynamic type 'FailedReference2::G' of operand}}
}

namespace FailedPtr {
  struct A  {};

  struct B : A {};

  struct C2 { virtual void c2(); };

  struct C : A, C2 {};
  struct F : B, C {};
  constexpr F g;
  static_assert(dynamic_cast<const A*>(static_cast<const C2*>(&g)) == nullptr);
  static_assert(dynamic_cast<const B*>(static_cast<const C2*>(&g)) != nullptr);
}

namespace Initializing {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 { A *c = dynamic_cast<A*>(static_cast<C2*>(this)); };

  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D {};
  struct Padding { virtual void padding(); };
  struct G : Padding, F {};

  constexpr G g;

  // During construction of C, A is unambiguous subobject of dynamic type C.
  static_assert(g.c == (C*)&g);
}

namespace SimpleDowncast {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 {};
  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D {};
  struct Padding { virtual void padding(); };
  struct G : Padding, F {};

  constexpr G g;
  // Can navigate from A2 to its A...
  static_assert(&dynamic_cast<A&>((A2&)(B&)g) == &(A&)(B&)g);
}

namespace ActuallyADerived2BaseCast {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 {};
  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D {};
  struct Padding { virtual void padding(); };
  struct G : Padding, F {};
  constexpr G g;
  // ... and from B to its A ...
  static_assert(&dynamic_cast<A2&>((B&)g) == &(A2&)(B&)g);
}

namespace ProperLimitedPtrInVoidCast {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 {};
  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D, private E { void *f = dynamic_cast<void*>(static_cast<D*>(this)); };
  struct Padding { virtual void padding(); };
  struct G : Padding, F {};

  constexpr G g;
  static_assert(g.f == (void*)(F*)&g);
}

namespace Unrelated {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 {};
  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D, private E {};
  struct Padding { virtual void padding(); };
  struct G : Padding, F {};

  constexpr G g;
  struct Unrelated { virtual void unrelated(); };
  constexpr int b_unrelated = (dynamic_cast<Unrelated&>((B&)g), 0); // both-error {{must be initialized by a constant expression}} \
                                                                    // both-note {{reference dynamic_cast failed: dynamic type 'Unrelated::G' of operand does not have a base class of type 'Unrelated'}}
  constexpr int e_unrelated = (dynamic_cast<Unrelated&>((E&)g), 0); // both-error {{must be initialized by a constant expression}} \
                                                                    // both-note {{reference dynamic_cast failed: dynamic type 'Unrelated::G' of operand does not have a base class of type 'Unrelated'}}
  static_assert(dynamic_cast<Unrelated*>((B*)&g) == nullptr);
  static_assert(dynamic_cast<Unrelated*>((E*)&g) == nullptr);
}

namespace PrivateSibling {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 {};
  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D, private E {};
  struct Padding { virtual void padding(); };
  struct G : Padding, F {};

  constexpr G g;
  // Cannot cast from B to private sibling E.
  constexpr int b_e = (dynamic_cast<E&>((B&)g), 0); // both-error {{must be initialized by a constant expression}} \
                                                    // both-note {{reference dynamic_cast failed: 'E' is a non-public base class of dynamic type 'PrivateSibling::G' of operand}}
  static_assert(dynamic_cast<E*>((B*)&g) == nullptr);
}

namespace Field {
  struct X {
    mutable int n = 0;
    virtual constexpr ~X() {}
  };
  struct Y : X {
  };
  struct Z {
    mutable Y y;
  };
  constexpr Z z;
  constexpr const X *pz = &z.y;
  constexpr const Y *qz = dynamic_cast<const Y*>(pz);
  static_assert(qz != nullptr);
}

/// The entire DynamicCast test from constant-expression-cxx2a.cpp but the g variable is a field.
namespace Field2 {
  struct A2 { virtual void a2(); };
  struct A : A2 { virtual void a(); };
  struct B : A {};
  struct C2 { virtual void c2(); };
  struct C : A, C2 { A *c = dynamic_cast<A*>(static_cast<C2*>(this)); };
  struct D { virtual void d(); };
  struct E { virtual void e(); };
  struct F : B, C, D, private E { void *f = dynamic_cast<void*>(static_cast<D*>(this)); };
  struct Padding { virtual void padding(); };
  struct G : Padding, F {};


  struct SomeStruct {
    int a;
    int b;
    G g;
  };

  constexpr SomeStruct ss{};

  // During construction of C, A is unambiguous subobject of dynamic type C.
  static_assert(ss.g.c == (C*)&ss.g);
  // ... but in the complete object, the same is not true, so the runtime fails.
  static_assert(dynamic_cast<const A*>(static_cast<const C2*>(&ss.g)) == nullptr);

  // dynamic_cast<void*> produces a pointer to the object of the dynamic type.
  static_assert(ss.g.f == (void*)(F*)&ss.g);
  static_assert(dynamic_cast<const void*>(static_cast<const D*>(&ss.g)) == &ss.g);

  // both-note@+1 {{reference dynamic_cast failed: 'A' is an ambiguous base class of dynamic type 'Field2::G' of operand}}
  constexpr int d_a = (dynamic_cast<const A&>(static_cast<const D&>(ss.g)), 0); // both-error {{}}

  // Can navigate from A2 to its A...
  static_assert(&dynamic_cast<A&>((A2&)(B&)ss.g) == &(A&)(B&)ss.g);
  // ... and from B to its A ...
  static_assert(&dynamic_cast<A&>((B&)ss.g) == &(A&)(B&)ss.g);
  // ... but not from D.
  // both-note@+1 {{reference dynamic_cast failed: 'A' is an ambiguous base class of dynamic type 'Field2::G' of operand}}
  static_assert(&dynamic_cast<A&>((D&)ss.g) == &(A&)(B&)ss.g); // both-error {{}}

  // Can cast from A2 to sibling class D.
  static_assert(&dynamic_cast<D&>((A2&)(B&)ss.g) == &(D&)ss.g);

  // Cannot cast from private base E to derived class F.
  // both-note@+1 {{reference dynamic_cast failed: static type 'Field2::E' of operand is a non-public base class of dynamic type 'Field2::G'}}
  constexpr int e_f = (dynamic_cast<F&>((E&)ss.g), 0); // both-error {{}}

  // Cannot cast from B to private sibling E.
  // both-note@+1 {{reference dynamic_cast failed: 'E' is a non-public base class of dynamic type 'Field2::G' of operand}}
  constexpr int b_e = (dynamic_cast<E&>((B&)ss.g), 0); // both-error {{}}

  struct Unrelated { virtual void unrelated(); };
  // both-note@+1 {{reference dynamic_cast failed: dynamic type 'Field2::G' of operand does not have a base class of type 'Unrelated'}}
  constexpr int b_unrelated = (dynamic_cast<Unrelated&>((B&)ss.g), 0); // both-error {{}}
  // both-note@+1 {{reference dynamic_cast failed: dynamic type 'Field2::G' of operand does not have a base class of type 'Unrelated'}}
  constexpr int e_unrelated = (dynamic_cast<Unrelated&>((E&)ss.g), 0); // both-error {{}}
}

namespace UnrelatedAndRootPtr{
  struct A {
    virtual void d() {}
  };
  struct B final : A { };
  struct C { };

  constexpr bool f() {
    B b;
    C *bb = dynamic_cast<C*>(&b);
    return !bb;
  }
  static_assert(f());
}

namespace UnrelatedAndRootReference {
  struct A {
    virtual void foo();
  };
  struct B1 : A {};

  struct B2 : A {};
  struct C : B2 {};

  constexpr C c;
  static_assert(&dynamic_cast<B2 &>((B1 &)c), ""); // both-error {{not an integral constant expression}} \
                                                   // both-note {{cast that performs the conversions of a reinterpret_cast is not allowed in a constant expression}}
}

namespace Invalid {
  struct S { virtual void s(); };
  struct A : S {};
  struct B : A {};
  constexpr __UINTPTR_TYPE__ g = 0;
  static_assert(&dynamic_cast<A&>((S&)(B&)g) == &(A&)(B&)g); // both-error {{not an integral constant expression}} \
                                                             // both-note {{cast that performs the conversions of a reinterpret_cast is not allowed in a constant expression}}

  struct X : S { : ; }; // both-error {{expected expression}} \
                        // both-error {{a type specifier is required for all declarations}}
  constexpr X x; // both-error {{must be initialized by a constant expression}} \
                 // both-note {{declared here}}
  static_assert(&dynamic_cast<S&>((X&)x), ""); // both-error {{not an integral constant expression}} \
                                               // both-note {{initializer of 'x' is not a constant expression}}
}

namespace UnrelatedInitializingPtr {
  struct A {
    virtual void foo();
  };
  struct B : A {};
  struct C : A {};
  struct D : B {};

  constexpr D d;
  constexpr A &a = (B &)d;
  constexpr auto p = dynamic_cast<C &>(a); // both-error {{must be initialized by a constant expression}} \
                                           // both-note {{reference dynamic_cast failed: dynamic type 'UnrelatedInitializingPtr::D' of operand does not have a base class of type 'C'}}
}
