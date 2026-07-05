.. title:: clang-tidy - bugprone-misplaced-pointer-arithmetic-in-alloc

bugprone-misplaced-pointer-arithmetic-in-alloc
===============================================

Finds cases where an integer expression is added to or subtracted from the
result of a memory allocation function (``malloc()``, ``calloc()``,
``realloc()``, ``alloca()``) instead of its argument. The check detects error
cases even if one of these functions is called by a constant function pointer.

Example code:

.. code-block:: c

  void bad_malloc(int n) {
    char *p = (char*) malloc(n) + 10;
  }


The suggested fix is to add the integer expression to the argument of
``malloc`` and not to its result. In the example above the fix would be

.. code-block:: c

  char *p = (char*) malloc(n + 10);
