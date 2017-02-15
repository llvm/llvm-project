.. title:: clang-tidy - modernize-return-braced-init-list

modernize-return-braced-init-list
=================================

Replaces explicit calls to the constructor in a return with a braced
initializer list. This way the return type is not needlessly duplicated in the
function definition and the return statement.

.. code:: c++

  Foo bar() {
    Baz baz;
    return Foo(baz);
  }

  // transforms to:

  Foo bar() {
    Baz baz;
    return {baz};
  }
