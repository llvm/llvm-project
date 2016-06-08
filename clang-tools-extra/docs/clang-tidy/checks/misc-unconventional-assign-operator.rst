.. title:: clang-tidy - misc-unconventional-assign-operator

misc-unconventional-assign-operator
===================================


Finds declarations of assign operators with the wrong return and/or argument
types and definitions with good return type but wrong ``return`` statements.

  * The return type must be ``Class&``.
  * Works with move-assign and assign by value.
  * Private and deleted operators are ignored.
  * The operator must always return ``*this``.
