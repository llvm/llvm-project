// RUN: %clang_cc1 -fsyntax-only -verify -Wundefined-func-template %s

namespace GH74016 {
  template <typename T> class B {
  public:
    constexpr void foo(const T &) { bar(1); }
    virtual constexpr void bar(unsigned int) = 0;
  };

  template <typename T> class D : public B<T> {
  public:
    constexpr void bar(unsigned int) override {}
  };

  void test() {
    auto t = D<int>();
    t.foo(0);
  }
};

namespace call_pure_virtual_function_from_virtual {
  template <typename T> class B {
  public:
    const void foo(const T &) { B::bar(1); } // expected-warning {{instantiation of function 'call_pure_virtual_function_from_virtual::B<int>::bar' required here, but no definition is available}}
    // expected-note@-1 {{add an explicit instantiation declaration to suppress this warning if 'call_pure_virtual_function_from_virtual::B<int>::bar' is explicitly instantiated in another translation unit}}
    virtual const void bar(unsigned int) = 0; // expected-note {{forward declaration of template entity is here}}
  };

  template <typename T> class D : public B<T> {
  public:
    const void bar(unsigned int) override {}
  };

  void test() {
    auto t = D<int>();
    t.foo(0); // expected-note {{in instantiation of member function 'call_pure_virtual_function_from_virtual::B<int>::foo' requested here}}
  }
};

namespace non_pure_virtual_function {
  template <typename T> class B {
  public:
    constexpr void foo(const T &) { bar(1); }

    virtual constexpr void bar(unsigned int); // expected-warning {{inline function 'non_pure_virtual_function::B<int>::bar' is not defined}}
  };

  template <typename T> class D : public B<T> { // expected-note@ {{used here}}

  public:
    constexpr void bar(unsigned int) override { }
  };

  void test() {
    auto t = D<int>();
    t.foo(0);
  }
};
