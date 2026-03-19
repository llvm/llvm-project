.. title:: llvm-avoid-passing-as-ref

Flags function parameters of types that should be passed by value, but are passed
by reference. By default, it flags types derived from ``mlir::Op`` as ``mlir::Op``
derived classes are lightweight wrappers around a pointer and should be passed
by value. This check can be customized to flag other types using the `ClassNames`
option.

Example:

.. code-block:: c++

  // Bad: passed by reference
  void processOp(const MyOp &op);
  void mutateOp(MyOp &op);

  // Good: passed by value
  void processOp(MyOp op);
  void mutateOp(MyOp op);

Options
-------

.. option:: ClassNames

   A semicolon-separated list of fully qualified class names that should be
   passed by value. Default is ``"::mlir::Op"``.
