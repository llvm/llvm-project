.. title:: clang-tidy - mlir-op-builder

mlir-op-builder
===============

Flags usage of old form of invoking create on `OpBuilder` and suggesting new
form.

Example
-------

.. code-block:: c++

  builder.create<FooOp>(builder.getUnknownLoc(), "baz");


Transforms to:

.. code-block:: c++

  FooOp::create(builder, builder.getUnknownLoc(), "baz");

