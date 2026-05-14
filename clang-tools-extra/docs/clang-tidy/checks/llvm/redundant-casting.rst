.. title:: clang-tidy - llvm-redundant-casting

llvm-redundant-casting
======================

Points out uses of ``cast<>``, ``dyn_cast<>`` and their ``or_null`` variants
that are unnecessary because the argument already is of the target type, or a
derived type thereof.

.. code-block:: c++

  struct A {};
  A a;
  // Finds:
  A x = cast<A>(a);
  // replaced by:
  A x = a;

  struct B : public A {};
  B b;
  // Finds:
  A y = cast<A>(b);
  // replaced by:
  A y = b;

Supported functions:
 - ``llvm::cast``
 - ``llvm::cast_or_null``
 - ``llvm::cast_if_present``
 - ``llvm::dyn_cast``
 - ``llvm::dyn_cast_or_null``
 - ``llvm::dyn_cast_if_present``
