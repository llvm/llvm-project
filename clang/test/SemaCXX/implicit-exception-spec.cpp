// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify -std=c++11 -Wall -Wno-unused-local-typedefs %s

template<bool b> struct ExceptionIf { static int f(); };
template<> struct ExceptionIf<false> { typedef int f; };

// The exception specification of a defaulted default constructor depends on
// the contents of in-class member initializers. However, the in-class member
// initializers can depend on the exception specification of the constructor,
// since the class is considered complete within them. We reject any such cases.
namespace InClassInitializers {
  // Noexcept::Noexcept() is implicitly declared as noexcept(false), because it
  // directly invokes ThrowSomething(). However...
  //
  // If noexcept(Noexcept()) is false, then Noexcept() is a constant expression,
  // so noexcept(Noexcept()) is true. But if noexcept(Noexcept()) is true, then
  // Noexcept::Noexcept is not declared constexpr, therefore noexcept(Noexcept())
  // is false.
  bool ThrowSomething() noexcept(false);
  struct ConstExpr { // expected-error {{default member initializer for 'b' needed}}
    bool b = // expected-note {{declared here}}
      noexcept(ConstExpr()) && ThrowSomething(); // expected-note {{in evaluation of exception spec}}
  };

  // Much more obviously broken: we can't parse the initializer without already
  // knowing whether it produces a noexcept expression.
  struct TemplateArg { // expected-error {{default member initializer for 'n' needed}}
    int n = // expected-note {{declared here}}
      ExceptionIf<noexcept(TemplateArg())>::f(); // expected-note {{in evaluation of exception spec}}
  };

  // And within a nested class.
  struct Nested {
    struct Inner { // expected-error {{default member initializer for 'n' needed}}
      int n = // expected-note {{declared here}}
        ExceptionIf<noexcept(Nested())>::f(); // expected-note {{in evaluation of exception spec}}
    } inner; // expected-note {{in evaluation of exception spec}}
  };

  struct Nested2 {
    struct Inner;
    int n = Inner().n; // expected-note {{in evaluation of exception spec}}
    struct Inner { // expected-error {{initializer for 'n' needed}}
      int n = ExceptionIf<noexcept(Nested2())>::f(); // expected-note {{declared here}}
    } inner;
  };
}

namespace ExceptionSpecification {
  struct Nested {
    struct T {
      T() noexcept(!noexcept(Nested())); // expected-note {{in evaluation of exception spec}}
    } t; // expected-error{{exception specification is not available until end of class definition}}
  };
}

namespace DefaultArgument {
  // FIXME: We should detect and diagnose the cyclic dependence of
  // noexcept(Default()) on itself here.
  struct Default {
    struct T {
      T(int = ExceptionIf<noexcept(Default())>::f());
    } t;
  };
}

namespace ImplicitDtorExceptionSpec {
  struct A {
    virtual ~A();

    struct Inner {
      ~Inner() throw();
    };
    Inner inner;
  };

  struct B {
    virtual ~B() {} // expected-note {{here}}
  };

  struct C : B {
    virtual ~C() {}
    A a;
  };

  struct D : B {
    ~D(); // expected-error {{more lax than base}}
    struct E {
      ~E();
      struct F {
        ~F() throw(A);
      } f;
    } e;
  };
}

struct nothrow_t {} nothrow;
void *operator new(decltype(sizeof(0)), nothrow_t) noexcept;

namespace PotentiallyConstructed {
  template<bool NE> struct A {
    A() noexcept(NE);
    A(const A&) noexcept(NE);
    A(A&&) noexcept(NE);
    A &operator=(const A&) noexcept(NE);
    A &operator=(A&&) noexcept(NE);
    ~A() noexcept(NE);
  };

  template<bool NE> struct B : virtual A<NE> {};

  template<bool NE> struct C : virtual A<NE> {
    virtual void f() = 0; // expected-note 2{{unimplemented}}
  };

  template<bool NE> struct D final : C<NE> {
    void f();
  };

  template<typename T, bool A, bool B, bool C, bool D, bool E, bool F> void check() {
    T *p = nullptr;
    T &a = *p;
    static_assert(noexcept(a = a) == D, "");
    static_assert(noexcept(a = static_cast<T&&>(a)) == E, "");
    static_assert(noexcept(delete &a) == F, "");

    // These are last because the first failure here causes instantiation to bail out.
    static_assert(noexcept(new (nothrow) T()) == A, ""); // expected-error 2{{abstract}}
    static_assert(noexcept(new (nothrow) T(a)) == B, "");
    static_assert(noexcept(new (nothrow) T(static_cast<T&&>(a))) == C, "");
  }

  template void check<A<false>, 0, 0, 0, 0, 0, 0>();
  template void check<A<true >, 1, 1, 1, 1, 1, 1>();
  template void check<B<false>, 0, 0, 0, 0, 0, 0>();
  template void check<B<true >, 1, 1, 1, 1, 1, 1>();
  template void check<C<false>, 1, 1, 1, 0, 0, 0>(); // expected-note {{instantiation}}
  template void check<C<true >, 1, 1, 1, 1, 1, 1>(); // expected-note {{instantiation}}
  template void check<D<false>, 0, 0, 0, 0, 0, 0>();
  template void check<D<true >, 1, 1, 1, 1, 1, 1>();

  // ... the above trick doesn't work for this case...
  struct Cfalse : virtual A<false> {
    virtual void f() = 0;

    Cfalse() noexcept;
    Cfalse(const Cfalse&) noexcept;
    Cfalse(Cfalse&&) noexcept;
  };
  Cfalse::Cfalse() noexcept = default;
  Cfalse::Cfalse(const Cfalse&) noexcept = default;
  Cfalse::Cfalse(Cfalse&&) noexcept = default;
}
