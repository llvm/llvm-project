.. title:: clang-tidy - misc-assign-operator-signature

misc-assign-operator-signature
==============================


Finds declarations of assign operators with the wrong return and/or argument
types.

  * The return type must be ``Class&``.
  * Works with move-assign and assign by value.
  * Private and deleted operators are ignored.
