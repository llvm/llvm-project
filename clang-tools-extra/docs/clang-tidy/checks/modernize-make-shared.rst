.. title:: clang-tidy - modernize-make-shared

modernize-make-shared
=====================

This check finds the creation of ``std::shared_ptr`` objects by explicitly
calling the constructor and a ``new`` expression, and replaces it with a call
to ``std::make_shared``.

.. code-block:: c++

  auto my_ptr = std::shared_ptr<MyPair>(new MyPair(1, 2));

  // becomes

  auto my_ptr = std::make_shared<MyPair>(1, 2);

This check also finds calls to ``std::shared_ptr::reset()`` with a ``new``
expression, and replaces it with a call to ``std::make_shared``.

.. code-block:: c++

  my_ptr.reset(new MyPair(1, 2));

  // becomes

  my_ptr = std::make_shared<MyPair>(1, 2);
