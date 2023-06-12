.. title:: clang-tidy - bugprone-unique-ptr-array-mismatch

bugprone-unique-ptr-array-mismatch
==================================

Finds initializations of C++ unique pointers to non-array type that are
initialized with an array.

If a pointer ``std::unique_ptr<T>`` is initialized with a new-expression
``new T[]`` the memory is not deallocated correctly. A plain ``delete`` is used
in this case to deallocate the target memory. Instead a ``delete[]`` call is
needed. A ``std::unique_ptr<T[]>`` uses the correct delete operator. The check
does not emit warning if an ``unique_ptr`` with user-specified deleter type is
used.

The check offers replacement of ``unique_ptr<T>`` to ``unique_ptr<T[]>`` if it
is used at a single variable declaration (one variable in one statement).

Example:

.. code-block:: c++

  std::unique_ptr<Foo> x(new Foo[10]); // -> std::unique_ptr<Foo[]> x(new Foo[10]);
  //                     ^ warning: unique pointer to non-array is initialized with array
  std::unique_ptr<Foo> x1(new Foo), x2(new Foo[10]); // no replacement
  //                                   ^ warning: unique pointer to non-array is initialized with array

  D d;
  std::unique_ptr<Foo, D> x3(new Foo[10], d); // no warning (custom deleter used)

  struct S {
    std::unique_ptr<Foo> x(new Foo[10]); // no replacement in this case
    //                     ^ warning: unique pointer to non-array is initialized with array
  };

This check partially covers the CERT C++ Coding Standard rule
`MEM51-CPP. Properly deallocate dynamically allocated resources
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM51-CPP.+Properly+deallocate+dynamically+allocated+resources>`_
However, only the ``std::unique_ptr`` case is detected by this check.
