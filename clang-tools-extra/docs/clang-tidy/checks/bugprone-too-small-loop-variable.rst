.. title:: clang-tidy - bugprone-too-small-loop-variable

bugprone-too-small-loop-variable
================================

Detects those ``for`` loops that have a loop variable with a "too small" type
which means this type can't represent all values which are part of the
iteration range.

.. code-block:: c++

  int main() {
    long size = 294967296l;
    for (short i = 0; i < size; ++i) {}
  }

This ``for`` loop is an infinite loop because the ``short`` type can't represent
all values in the ``[0..size]`` interval.

In a real use case size means a container's size which depends on the user input.

.. code-block:: c++

  int doSomething(const std::vector& items) {
    for (short i = 0; i < items.size(); ++i) {}
  }

This algorithm works for small amount of objects, but will lead to freeze for a
a larger user input.
