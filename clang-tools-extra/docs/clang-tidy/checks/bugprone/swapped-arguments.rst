.. title:: clang-tidy - bugprone-swapped-arguments

bugprone-swapped-arguments
==========================

Finds potentially swapped arguments by examining implicit conversions.
It analyzes the types of the arguments being passed to a function and compares
them to the expected types of the corresponding parameters. If there is a
mismatch or an implicit conversion that indicates a potential swap, a warning
is raised.

.. code-block:: c++

  void printNumbers(int a, float b);

  int main() {
    // Swapped arguments: float passed as int, int as float)
    printNumbers(10.0f, 5);
    return 0;
  }

Covers a wide range of implicit conversions, including:
- User-defined conversions
- Conversions from floating-point types to boolean or integral types
- Conversions from integral types to boolean or floating-point types
- Conversions from boolean to integer types or floating-point types
- Conversions from (member) pointers to boolean

It is important to note that for most argument swaps, the types need to match
exactly. However, there are exceptions to this rule. Specifically, when the
swapped argument is of integral type, an exact match is not always necessary.
Implicit casts from other integral types are also accepted. Similarly, when
dealing with floating-point arguments, implicit casts between different
floating-point types are considered acceptable.

To avoid confusion, swaps where both swapped arguments are of integral types or
both are of floating-point types do not trigger the warning. In such cases, it's
assumed that the developer intentionally used different integral or
floating-point types and does not raise a warning. This approach prevents false
positives and provides flexibility in handling situations where varying integral
or floating-point types are intentionally utilized.
