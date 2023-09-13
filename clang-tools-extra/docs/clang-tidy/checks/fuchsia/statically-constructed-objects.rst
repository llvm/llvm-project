.. title:: clang-tidy - fuchsia-statically-constructed-objects

fuchsia-statically-constructed-objects
======================================

Warns if global, non-trivial objects with static storage are constructed, unless
the object is statically initialized with a ``constexpr`` constructor or has no
explicit constructor.

For example:

.. code-block:: c++

  class A {};

  class B {
  public:
    B(int Val) : Val(Val) {}
  private:
    int Val;
  };

  class C {
  public:
    constexpr C(int Val) : Val(Val) {}
    C(int Val1, int Val2) : Val(Val1+Val2) {}

  private:
    int Val;
  };

  static A a;         // No warning, as there is no explicit constructor
  static C c(0);      // No warning, as constructor is constexpr

  static B b(0);      // Warning, as constructor is not constexpr
  static C c2(0, 1);  // Warning, as constructor is not constexpr

  static int i;       // No warning, as it is trivial

  extern int get_i();
  static C c3(get_i());// Warning, as the constructor is dynamically initialized

See the features disallowed in Fuchsia at https://fuchsia.dev/fuchsia-src/development/languages/c-cpp/cxx?hl=en
