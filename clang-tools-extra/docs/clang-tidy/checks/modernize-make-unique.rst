.. title:: clang-tidy - modernize-make-unique

modernize-make-unique
=====================

This check finds the creation of ``std::unique_ptr`` objects by explicitly
calling the constructor and a ``new`` expression, and replaces it with a call
to ``std::make_unique``, introduced in C++14.

.. code-block:: c++

  auto my_ptr = std::unique_ptr<MyPair>(new MyPair(1, 2));

  // becomes

  auto my_ptr = std::make_unique<MyPair>(1, 2);

This check also finds calls to ``std::unique_ptr::reset()`` with a ``new``
expression, and replaces it with a call to ``std::make_unique``.

.. code-block:: c++

  my_ptr.reset(new MyPair(1, 2));

  // becomes

  my_ptr = std::make_unique<MyPair>(1, 2);
