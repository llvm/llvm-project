.. title:: clang-tidy - readability-avoid-nested-conditional-operator

readability-avoid-nested-conditional-operator
=============================================

Identifies instances of nested conditional operators in the code.

Nested conditional operators, also known as ternary operators, can contribute
to reduced code readability and comprehension. So they should be split as
several statements and stored the intermediate results in temporary variable.

Examples:

.. code-block:: c++

  int NestInConditional = (condition1 ? true1 : false1) ? true2 : false2;
  int NestInTrue = condition1 ? (condition2 ? true1 : false1) : false2;
  int NestInFalse = condition1 ? true1 : condition2 ? true2 : false1;

This check implements part of `AUTOSAR C++14 Rule A5-16-1
<https://www.autosar.org/fileadmin/standards/R22-11/AP/AUTOSAR_RS_CPP14Guidelines.pdf>`_.
