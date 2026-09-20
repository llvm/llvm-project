.. title:: clang-tidy - llvm-type-switch-case-types

llvm-type-switch-case-types
===========================

Finds ``llvm::TypeSwitch::Case`` calls with redundant explicit template
arguments that can be inferred from the lambda parameter type.

This check identifies two patterns:

1. **Redundant explicit type**: When the lambda parameter type matches the
   ``Case`` template argument, the explicit type can be removed.

2. **Auto parameter with explicit type**: When a lambda uses ``auto`` but
   ``Case`` has an explicit template argument, suggests using an explicit
   type in the lambda instead.

Example
-------

.. code-block:: c++

  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *a) { return 1; }) // Redundant.
      .Case<DerivedB>([](auto b) { return 2; });     // `auto` with explicit type.

Transforms to:

.. code-block:: c++

  llvm::TypeSwitch<Base *, int>(base)
      .Case([](DerivedA *a) { return 1; })       // Type inferred from lambda.
      .Case<DerivedB>([](auto b) { return 2; }); // Warning only.

Note: The second case (``auto`` parameter) only emits a warning without a
fix-it, because the deduced type of ``auto`` depends on ``dyn_cast`` behavior
which varies between pointer types and MLIR handle types.
