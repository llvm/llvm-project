.. title:: clang-tidy - readability-avoid-nested-conditional-operator

readability-avoid-nested-conditional-operator
=================================================

Finds nested conditional operator.

Nested conditional operators lead code hard to understand, so they should be
splited as several statement and stored in temporary varibale.

Examples:

.. code-block:: c++

  int NestInConditional = (condition1 ? true1 : false1) ? true2 : false2;
  int NestInTrue = condition1 ? (condition2 ? true1 : false1) : false2;
  int NestInFalse = condition1 ? true1 : condition2 ? true2 : false1;

This check implements part of `AUTOSAR C++14 Rule A5-16-1
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-constref>`_.
