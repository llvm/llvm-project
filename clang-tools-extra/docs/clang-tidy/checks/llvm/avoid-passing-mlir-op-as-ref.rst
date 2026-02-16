.. title:: clang-tidy - llvm-avoid-passing-mlir-op-as-ref

llvm-avoid-passing-mlir-op-as-ref
=================================

Flags function parameters of a type derived from ``mlir::Op`` that are passed
by reference. ``mlir::Op`` derived classes are lightweight wrappers around a
pointer and should be passed by value.

Example
-------

.. code-block:: c++

  // Bad: passed by reference
  void processOp(const MyOp &op);
  void mutateOp(MyOp &op);

  // Good: passed by value
  void processOp(MyOp op);
  void mutateOp(MyOp op);
