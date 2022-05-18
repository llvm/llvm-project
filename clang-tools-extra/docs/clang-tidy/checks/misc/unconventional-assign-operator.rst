.. title:: clang-tidy - misc-unconventional-assign-operator

misc-unconventional-assign-operator
===================================


Finds declarations of assign operators with the wrong return and/or argument
types and definitions with good return type but wrong ``return`` statements.

  * The return type must be ``Class&``.
  * The assignment may be from the class type by value, const lvalue
    reference, non-const rvalue reference, or from a completely different
    type (e.g. ``int``).
  * Private and deleted operators are ignored.
  * The operator must always return ``*this``.
