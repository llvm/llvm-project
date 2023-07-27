.. title:: clang-tidy - bugprone-switch-missing-default-case

bugprone-switch-missing-default-case
====================================

Ensures that switch statements without default cases are flagged, focuses only
on covering cases with non-enums where the compiler may not issue warnings.

Switch statements without a default case can lead to unexpected
behavior and incomplete handling of all possible cases. When a switch statement
lacks a default case, if a value is encountered that does not match any of the
specified cases, the program will continue execution without any defined
behavior or handling.

This check helps identify switch statements that are missing a default case,
allowing developers to ensure that all possible cases are handled properly.
Adding a default case allows for graceful handling of unexpected or unmatched
values, reducing the risk of program errors and unexpected behavior.

Example:

.. code-block:: c++

  // Example 1:
  // warning: switching on non-enum value without default case may not cover all cases
  switch (i) {
  case 0:
    break;
  }

  // Example 2:
  enum E { eE1 };
  E e = eE1;
  switch (e) { // no-warning
  case eE1:
    break;
  }

  // Example 3:
  int i = 0;
  switch (i) { // no-warning
  case 0:
    break;
  default:
    break;
  }

.. note::
   Enum types are already covered by compiler warnings (comes under -Wswitch)
   when a switch statement does not handle all enum values. This check focuses
   on non-enum types where the compiler warnings may not be present.

.. seealso::
   The `CppCoreGuideline ES.79 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-default>`_
   provide guidelines on switch statements, including the recommendation to
   always provide a default case.
