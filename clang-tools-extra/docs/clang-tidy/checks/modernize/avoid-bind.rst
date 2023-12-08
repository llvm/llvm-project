.. title:: clang-tidy - modernize-avoid-bind

modernize-avoid-bind
====================

The check finds uses of ``std::bind`` and ``boost::bind`` and replaces them
with lambdas. Lambdas will use value-capture unless reference capture is
explicitly requested with ``std::ref`` or ``boost::ref``.

It supports arbitrary callables including member functions, function objects,
and free functions, and all variations thereof. Anything that you can pass
to the first argument of ``bind`` should be diagnosable. Currently, the only
known case where a fix-it is unsupported is when the same placeholder is
specified multiple times in the parameter list.

Given:

.. code-block:: c++

  int add(int x, int y) { return x + y; }

Then:

.. code-block:: c++

  void f() {
    int x = 2;
    auto clj = std::bind(add, x, _1);
  }

is replaced by:

.. code-block:: c++

  void f() {
    int x = 2;
    auto clj = [=](auto && arg1) { return add(x, arg1); };
  }

``std::bind`` can be hard to read and can result in larger object files and
binaries due to type information that will not be produced by equivalent
lambdas.

Options
-------

.. option:: PermissiveParameterList

  If the option is set to `true`, the check will append ``auto&&...`` to the end
  of every placeholder parameter list. Without this, it is possible for a fix-it
  to perform an incorrect transformation in the case where the result of the ``bind``
  is used in the context of a type erased functor such as ``std::function`` which
  allows mismatched arguments. For example:


.. code-block:: c++

  int add(int x, int y) { return x + y; }
  int foo() {
    std::function<int(int,int)> ignore_args = std::bind(add, 2, 2);
    return ignore_args(3, 3);
  }

is valid code, and returns `4`. The actual values passed to ``ignore_args`` are
simply ignored. Without ``PermissiveParameterList``, this would be transformed into

.. code-block:: c++

  int add(int x, int y) { return x + y; }
  int foo() {
    std::function<int(int,int)> ignore_args = [] { return add(2, 2); }
    return ignore_args(3, 3);
  }

which will *not* compile, since the lambda does not contain an ``operator()``
that accepts 2 arguments. With permissive parameter list, it instead generates

.. code-block:: c++

  int add(int x, int y) { return x + y; }
  int foo() {
    std::function<int(int,int)> ignore_args = [](auto&&...) { return add(2, 2); }
    return ignore_args(3, 3);
  }

which is correct.

This check requires using C++14 or higher to run.
