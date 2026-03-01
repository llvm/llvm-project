.. title:: clang-tidy - readability-pointer-to-ref

readability-pointer-to-ref
==========================

Finds function parameters that are pointers but are always dereferenced without
null checks, suggesting they should be references instead.

Using references instead of pointers for non-nullable parameters makes the
contract explicit: the caller must provide a valid object. This improves code
readability and can help prevent null pointer bugs.

.. code-block:: c++

  // Before -- pointer that is never null-checked
  void process(Foo *P) {
    P->bar();
    P->X = 42;
  }

  // After -- use a reference instead
  void process(Foo &P) {
    P.bar();
    P.X = 42;
  }

The check will not flag a parameter if it:

- is passed to another function as a pointer argument,
- is used in pointer arithmetic or array subscript,
- is a ``void`` pointer or pointer to an incomplete type,
- is a parameter of a virtual method or an ``extern "C"`` function,
- is used in a ``delete`` expression,
- is returned from the function,
- is stored to another variable or captured by a lambda, or
- has its address taken (``&P``).

.. note::

   This check does not provide fix-its because changing a parameter from pointer
   to reference requires updating all call sites, which may span multiple
   translation units.
