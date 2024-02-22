.. title:: clang-tidy - bugprone-crtp-constructor-accessibility

bugprone-crtp-constructor-accessibility
=======================================

Finds CRTP used in an error-prone way.

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
the constructor private and declare the derived class as friend.

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
