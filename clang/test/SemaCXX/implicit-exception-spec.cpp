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
  struct ConstExpr {
    bool b = // expected-note {{declared here}}
      noexcept(ConstExpr()) && ThrowSomething(); // expected-error {{default member initializer for 'b' needed}}
  };

  // Much more obviously broken: we can't parse the initializer without already
  // knowing whether it produces a noexcept expression.
  struct TemplateArg {
    int n = // expected-note {{declared here}}
      ExceptionIf<noexcept(TemplateArg())>::f(); // expected-error {{default member initializer for 'n' needed}}
  };

  // And within a nested class.
  struct Nested {
    struct Inner {
      int n = // expected-note {{declared here}}
        ExceptionIf<noexcept(Nested())>::f();
    } inner; // expected-error {{default member initializer for 'n' needed}}
  };

  struct Nested2 {
    struct Inner;
    int n = Inner().n; // expected-error {{initializer for 'n' needed}}
    struct Inner {
      int n = ExceptionIf<noexcept(Nested2())>::f(); // expected-note {{declared here}}
    } inner;
  };
}

namespace ExceptionSpecification {
  // FIXME: This diagnostic is quite useless; we should indicate whose
  // exception specification we were looking for and why.
  struct Nested {
    struct T {
      T() noexcept(!noexcept(Nested()));
    } t; // expected-error{{exception specification is not available until end of class definition}}
  };
}

namespace DefaultArgument {
  struct Default {
    struct T {
      T(int = ExceptionIf<noexcept(Default())::f()); // expected-error {{call to implicitly-deleted default constructor}}
    } t; // expected-note {{has no default constructor}}
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
