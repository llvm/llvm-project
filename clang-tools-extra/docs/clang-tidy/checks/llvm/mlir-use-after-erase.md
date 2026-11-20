```{title} clang-tidy - llvm-mlir-use-after-erase
```

# llvm-mlir-use-after-erase

Finds uses of an MLIR operation after it has been erased or replaced.

Erasing or replacing an `mlir::Operation` destroys the underlying object, so any
later access through a pointer or wrapper that still refers to it is a
use-after-free. This check flags such accesses, including uses in later loop
iterations and after fallthrough in `switch` statements.

```c++
void example(mlir::RewriterBase &rewriter, mlir::Operation *op) {
  rewriter.eraseOp(op);
  op->dump();  // warning: 'op' used after it was invalidated by 'eraseOp'
}
```

An operation is treated as invalidated when it is passed to one of the
configured invalidation functions, for example `mlir::Operation::erase`,
`mlir::Operation::destroy`, `mlir::RewriterBase::eraseOp`,
`mlir::RewriterBase::eraseOpResults`, or `mlir::RewriterBase::replaceOp`.
Reassigning the variable to a different operation before the use suppresses the
warning.

Because operations are almost always used through a pointer or an
`mlir::OpState` wrapper, only accesses of the object are reported, that is, a
member access (`op->m`), a dereference (`*op`), or a subscript (`op[i]`). Merely
passing the pointer as an argument, comparing it, or copying it is not reported.
Accessor calls such as `mlir::OpState::operator->` and
`mlir::OpState::getOperation` are unwrapped, so the check tracks the wrapper
variable through them.

This check is implemented as a preconfigured instance of
{doc}`bugprone-use-after-move <../bugprone/use-after-move>`. See that check's
documentation for the meaning of the `InvalidationFunctions`,
`ReportAccessOnlyUseForTypes`, and `HandleAccessorFunctions` options, which
are used to model the MLIR API.

## Options

```{option} InvalidationFunctions
Default is `::mlir::Operation::erase$;::mlir::Operation::destroy$;::mlir::RewriterBase::eraseOp(0);::mlir::RewriterBase::eraseOpResults(0);::mlir::RewriterBase::replaceOp(0)`.
```

```{option} ReportAccessOnlyUseForTypes
Default is `::mlir::Operation;::mlir::OpState`.
```

```{option} HandleAccessorFunctions
Default is `::mlir::OpState::operator->;::mlir::OpState::getOperation`.
```
