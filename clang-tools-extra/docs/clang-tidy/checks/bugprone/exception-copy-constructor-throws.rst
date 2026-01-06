.. title:: clang-tidy - bugprone-exception-copy-constructor-throws

bugprone-exception-copy-constructor-throws
==========================================

Checks whether a thrown object's copy constructor can throw.

Exception objects are required to be copy constructible in C++. However, an
exception's copy constructor should not throw to avoid potential issues when
unwinding the stack. If an exception is thrown during stack unwinding (such
as from a copy constructor of an exception object), the program will
terminate via ``std::terminate``.

.. code-block:: c++

  class SomeException {
  public:
    SomeException() = default;
    SomeException(const SomeException&) { /* may throw */ }
  };

  void f() {
    throw SomeException();  // warning: thrown exception type's copy constructor can throw
  }

References
----------

This check corresponds to the CERT C++ Coding Standard rule
`ERR60-CPP. Exception objects must be nothrow copy constructible
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/ERR60-CPP.+Exception+objects+must+be+nothrow+copy+constructible>`_.
