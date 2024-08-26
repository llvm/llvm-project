.. title:: clang-tidy - bugprone-suspicious-pointer-arithmetics-using-sizeof

bugprone-suspicious-pointer-arithmetics-using-sizeof
====================================================

Finds suspicious pointer arithmetic calculations where the pointer is offset by a ``sizeof()`` expression.

Pointer arithmetic expressions implicitly scale the offset added to or subtracted from the address by the size of the pointee type.
Scaling the offset expression manually effectively results in a squared offset, which creates an invalid pointer that points beyond the end of the intended array.

.. code-block:: c++

  void printEveryEvenIndexElement(int *Array, size_t N) {
    int *P = Array;
    while (P <= Array + N * sizeof(int)) { // Suspicious pointer arithmetics using sizeof()!
      printf("%d ", *P);

      P += 2 * sizeof(int); // Suspicious pointer arithmetics using sizeof()!
    }
  }

The above example should be in the following, correct form:

.. code-block:: c++

  void printEveryEvenIndexElement(int *Array, size_t N) {
    int *P = Array;
    while (P <= Array + N) {
      printf("%d ", *P);

      P += 2;
    }
  }

`cert-arr39-c` redirects here as an alias of this check.

This check corresponds to the CERT C Coding Standard rule
`ARR39-C. Do not add or subtract a scaled integer to a pointer
<http://wiki.sei.cmu.edu/confluence/display/c/ARR39-C.+Do+not+add+or+subtract+a+scaled+integer+to+a+pointer>`_.

Limitations
-----------

While incorrect from a technically rigorous point of view, the check does not warn for pointer arithmetics where the pointee type is ``char`` (``sizeof(char) == 1``, by definition) on purpose.
