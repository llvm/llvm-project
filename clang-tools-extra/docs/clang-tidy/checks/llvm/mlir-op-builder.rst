.. title:: clang-tidy - llvm-mlir-op-builder

llvm-mlir-op-builder
====================

Flags usage of old form of invoking create on MLIR's ``OpBuilder`` and suggests
new form.

Example
-------

.. code-block:: c++

  builder.create<FooOp>(builder.getUnknownLoc(), "baz");


Transforms to:

.. code-block:: c++

  FooOp::create(builder, builder.getUnknownLoc(), "baz");
