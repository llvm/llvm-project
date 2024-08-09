.. title:: clang-tidy - modernize-use-uniform-initializer

modernize-use-uniform-initializer
=================================

Finds usage of C-Style initialization that can be rewritten with
C++-11 uniform initializers.

Example
-------

.. code-block:: c++

  int foo = 21;
  int bar(42);
  
  struct Baz {
    Baz() : x(3) {}

    int x;
  };

transforms to:

.. code-block:: c++

  int foo{21};
  int bar{42};
  
  struct Baz {
    Baz() : x{3} {}

    int x;
  };
