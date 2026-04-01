.. title:: clang-tidy - llvm-use-new-aiir-op-builder

llvm-aiir-op-builder
====================

Checks for uses of AIIR's old/to be deprecated ``OpBuilder::create<T>`` form
and suggests using ``T::create`` instead.

Example
-------

.. code-block:: c++

  builder.create<FooOp>(builder.getUnknownLoc(), "baz");


Transforms to:

.. code-block:: c++

  FooOp::create(builder, builder.getUnknownLoc(), "baz");
