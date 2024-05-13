.. title:: clang-tidy - bugprone-shared-ptr-array-mismatch

bugprone-shared-ptr-array-mismatch
==================================

Finds initializations of C++ shared pointers to non-array type that are
initialized with an array.

If a shared pointer ``std::shared_ptr<T>`` is initialized with a new-expression
``new T[]`` the memory is not deallocated correctly. The pointer uses plain
``delete`` in this case to deallocate the target memory. Instead a ``delete[]``
call is needed. A ``std::shared_ptr<T[]>`` calls the correct delete operator.

The check offers replacement of ``shared_ptr<T>`` to ``shared_ptr<T[]>`` if it
is used at a single variable declaration (one variable in one statement).

Example:

.. code-block:: c++

  std::shared_ptr<Foo> x(new Foo[10]); // -> std::shared_ptr<Foo[]> x(new Foo[10]);
  //                     ^ warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  std::shared_ptr<Foo> x1(new Foo), x2(new Foo[10]); // no replacement
  //                                   ^ warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]

  std::shared_ptr<Foo> x3(new Foo[10], [](const Foo *ptr) { delete[] ptr; }); // no warning

  struct S {
    std::shared_ptr<Foo> x(new Foo[10]); // no replacement in this case
    //                     ^ warning: shared pointer to non-array is initialized with array [bugprone-shared-ptr-array-mismatch]
  };

This check partially covers the CERT C++ Coding Standard rule
`MEM51-CPP. Properly deallocate dynamically allocated resources
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM51-CPP.+Properly+deallocate+dynamically+allocated+resources>`_
However, only the ``std::shared_ptr`` case is detected by this check.
