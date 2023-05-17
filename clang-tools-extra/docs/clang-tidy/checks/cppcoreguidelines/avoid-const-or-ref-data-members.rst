.. title:: clang-tidy - cppcoreguidelines-avoid-const-or-ref-data-members

cppcoreguidelines-avoid-const-or-ref-data-members
=================================================

This check warns when structs or classes have const-qualified or reference
(lvalue or rvalue) data members. Having such members is rarely useful, and
makes the class only copy-constructible but not copy-assignable.

Examples:

.. code-block:: c++

  // Bad, const-qualified member
  struct Const {
    const int x;
  }

  // Good:
  class Foo {
   public:
    int get() const { return x; }
   private:
    int x;
  };

  // Bad, lvalue reference member
  struct Ref {
    int& x;
  };

  // Good:
  struct Foo {
    int* x;
    std::unique_ptr<int> x;
    std::shared_ptr<int> x;
    gsl::not_null<int> x;
  };

  // Bad, rvalue reference member
  struct RefRef {
    int&& x;
  };

The check implements
`rule C.12 of C++ Core Guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#c12-dont-make-data-members-const-or-references>`_.

Further reading:
`Data members: Never const <https://quuxplusone.github.io/blog/2022/01/23/dont-const-all-the-things/#data-members-never-const>`_.
