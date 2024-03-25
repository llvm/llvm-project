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

This check implements `AUTOSAR C++14 Rule A13-2-1
<https://www.autosar.org/fileadmin/standards/R22-11/AP/AUTOSAR_RS_CPP14Guidelines.pdf>`_.
