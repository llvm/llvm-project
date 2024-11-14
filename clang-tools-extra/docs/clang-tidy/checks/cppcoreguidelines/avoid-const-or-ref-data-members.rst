.. title:: clang-tidy - cppcoreguidelines-avoid-const-or-ref-data-members

cppcoreguidelines-avoid-const-or-ref-data-members
=================================================

This check warns when structs or classes that are copyable or movable, and have
const-qualified or reference (lvalue or rvalue) data members. Having such
members is rarely useful, and makes the class only copy-constructible but not
copy-assignable.

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
    gsl::not_null<int*> x;
  };

  // Bad, rvalue reference member
  struct RefRef {
    int&& x;
  };

This check implements `C.12
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-constref>`_
from the C++ Core Guidelines.

Further reading:
`Data members: Never const <https://quuxplusone.github.io/blog/2022/01/23/dont-const-all-the-things/#data-members-never-const>`_.
