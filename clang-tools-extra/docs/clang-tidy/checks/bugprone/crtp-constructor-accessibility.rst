.. title:: clang-tidy - bugprone-crtp-constructor-accessibility

bugprone-crtp-constructor-accessibility
=======================================

Finds Curiously Recurring Template Pattern used in an error-prone way.

The CRTP is an idiom, in which a class derives from a template class, where 
itself is the template argument. It should be ensured that if a class is
intended to be a base class in this idiom, it can only be instantiated if
the derived class is it's template argument.

Example:

.. code-block:: c++

  template <typename T> class CRTP {
  private:
    CRTP() = default;
    friend T;
  };

  class Derived : CRTP<Derived> {};

Below can be seen some common mistakes that will allow the breaking of the idiom.

If the constructor of a class intended to be used in a CRTP is public, then
it allows users to construct that class on its own.

Example:

.. code-block:: c++

  template <typename T> class CRTP {
  public:
    CRTP() = default;
  };

  class Good : CRTP<Good> {};
  Good GoodInstance;

  CRTP<int> BadInstance;

If the constructor is protected, the possibility of an accidental instantiation
is prevented, however it can fade an error, when a different class is used as
the template parameter instead of the derived one.

Example:

.. code-block:: c++

  template <typename T> class CRTP {
  protected:
    CRTP() = default;
  };

  class Good : CRTP<Good> {};
  Good GoodInstance;

  class Bad : CRTP<Good> {};
  Bad BadInstance;

To ensure that no accidental instantiation happens, the best practice is to make
the constructor private and declare the derived class as friend. Note that as a tradeoff, 
this also gives the derived class access to every other private members of the CRTP.

Example:

.. code-block:: c++

  template <typename T> class CRTP {
    CRTP() = default;
    friend T;
  };

  class Good : CRTP<Good> {};
  Good GoodInstance;

  class Bad : CRTP<Good> {};
  Bad CompileTimeError;

  CRTP<int> AlsoCompileTimeError;
