# Implementation Plan - Fix mlir-opt Crash in Shape Dialect

## Problem Analysis
The `mlir-opt` tool crashes with an assertion failure in `llvm::cast` during the `--canonicalize` pass. This occurs when `shape.broadcast` or other shape operations consume a `ub.poison` value (or any attribute that is not a `DenseIntElementsAttr`). The folding logic assumes that if an operand is not null, it must be a `DenseIntElementsAttr`, leading to an invalid cast.

**Affected File:** `mlir/lib/Dialect/Shape/IR/Shape.cpp`

**Locations:**
1.  `BroadcastOp::fold` (lines ~647-681)
2.  `hasAtMostSingleNonScalar` helper function (lines ~978-988)
3.  `CstrBroadcastableOp::fold` (lines ~990-1023)

## Proposed Changes

### 1. Update `BroadcastOp::fold`
Modify the loop that iterates over shapes. Instead of just checking `if (!next)`, check if `next` can be cast to `DenseIntElementsAttr`.

**Current Code:**
```cpp
if (!adaptor.getShapes().front())
  return nullptr;
// ...
for (auto next : adaptor.getShapes().drop_front()) {
  if (!next)
    return nullptr;
  // cast ...
}
```

**New Code:**
```cpp
auto firstAttr = llvm::dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getShapes().front());
if (!firstAttr)
  return nullptr;
// ...
for (auto next : adaptor.getShapes().drop_front()) {
  auto nextAttr = llvm::dyn_cast_if_present<DenseIntElementsAttr>(next);
  if (!nextAttr)
    return nullptr;
  // use nextAttr ...
}
```

### 2. Update `hasAtMostSingleNonScalar`
Add a check for `DenseIntElementsAttr` before accessing `getNumElements()`.

**Current Code:**
```cpp
if (!a || llvm::cast<DenseIntElementsAttr>(a).getNumElements() != 0) {
```

**New Code:**
```cpp
auto denseAttr = llvm::dyn_cast_if_present<DenseIntElementsAttr>(a);
if (!denseAttr || denseAttr.getNumElements() != 0) {
```

### 3. Update `CstrBroadcastableOp::fold`
Similar to `BroadcastOp`, ensure `dyn_cast` succeeds before accessing values.

**Current Code:**
```cpp
if (!operand)
  return false;
extents.push_back(llvm::to_vector<6>(
    llvm::cast<DenseIntElementsAttr>(operand).getValues<int64_t>()));
```

**New Code:**
```cpp
auto denseAttr = llvm::dyn_cast_if_present<DenseIntElementsAttr>(operand);
if (!denseAttr)
  return false;
extents.push_back(llvm::to_vector<6>(
    denseAttr.getValues<int64_t>()));
```

## Validation Plan
1.  **Build:** Run `ninja mlir-opt` (or `ninja check-mlir-shape` if available targets).
2.  **Reproduction:** Run the `reproduce_issue.mlir` with `mlir-opt --canonicalize` to ensure it no longer crashes.
3.  **Regression Test:** Add the reproduction case to `mlir/test/Dialect/Shape/canonicalize.mlir` or a new test file.
