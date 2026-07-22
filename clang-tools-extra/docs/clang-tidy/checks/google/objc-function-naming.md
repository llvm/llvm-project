.. title:: clang-tidy - google-objc-function-naming

google-objc-function-naming
===========================

Finds function declarations in Objective-C files that do not follow the pattern
described in the Google Objective-C Style Guide.

The corresponding style guide rule can be found here:
https://google.github.io/styleguide/objcguide.html#function-names

All function names should be in Pascal case. Functions whose storage class is
not static should have an appropriate prefix.

The following code sample does not follow this pattern:

.. code-block:: objc

  static bool is_positive(int i) { return i > 0; }
  bool IsNegative(int i) { return i < 0; }

The sample above might be corrected to the following code:

.. code-block:: objc

  static bool IsPositive(int i) { return i > 0; }
  bool *ABCIsNegative(int i) { return i < 0; }
