```{title} clang-tidy - llvm-mlir-use-after-erase
```

# llvm-mlir-use-after-erase

Detects uses of an `mlir::Operation` after it has been erased.

Once an operation has been erased its memory is no longer valid, so any
subsequent access to it is a use-after-free. This check flags such uses when
the operation is invalidated through:

- `mlir::Operation` member functions `erase()` and `destroy()`, or
- `mlir::RewriterBase` helpers `eraseOp()`, `eraseOpResults()`,
  `replaceOp()` and `replaceOpWithNewOp()`.
- `mlir::OpState` wrappers that provide `operator->` or `getOperation()`.
  This includes derived ops from `mlir::Op`, which inherits `mlir::OpState`.

The analysis is control-flow aware: only uses that can actually be reached
after the erasing call are reported, and reassigning the variable to a new
operation clears the diagnostic.

```cpp
void example(mlir::RewriterBase &rewriter, mlir::Operation *op) {
  rewriter.eraseOp(op);
  op->dump(); // warning: operation 'op' is used after it was erased
}
```

Uses that happen *before* the operation is erased, or after it has been reassigned, are not reported:

```cpp
void ok(mlir::Operation *op) {
  op->dump(); // no warning, 'op' is used before it is erased
  op->erase();

  op = ...;
  op->dump(); // no warning, op is reassigned to a new operation
}
```
