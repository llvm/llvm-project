.. title:: clang-tidy - readability-pointer-to-ref

readability-pointer-to-ref
==========================

Finds function parameters that are pointers but are always
dereferenced without null checks, suggesting they should be
references instead.

Using references instead of pointers for non-nullable parameters
makes the contract explicit: the caller must provide a valid object.
This improves readability and can prevent null pointer bugs.

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

- is checked for null (``if (P)``, ``P != nullptr``, etc.),
- is passed to another function as a pointer argument,
- is used in pointer arithmetic or array subscript,
- is a ``void`` pointer or pointer to an incomplete type,
- is a parameter of a virtual method,
- is a function pointer,
- is unnamed, or
- is never dereferenced in the function body.

.. note::

   This check does not provide fix-its because changing a
   parameter from pointer to reference requires updating all
   call sites, which may span multiple translation units.
