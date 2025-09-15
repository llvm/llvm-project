.. title:: clang-tidy - llvm-use-new-mlir-op-builder

llvm-mlir-op-builder
====================

Checks for uses of MLIR's old/to be deprecated ``OpBuilder::create<T>`` form
and suggests using ``T::create`` instead.

Example
-------

.. code-block:: c++

  builder.create<FooOp>(builder.getUnknownLoc(), "baz");


Transforms to:

.. code-block:: c++

  FooOp::create(builder, builder.getUnknownLoc(), "baz");
