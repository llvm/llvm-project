// Flow-sensitive nullability and Objective-C block pointers.
//
// Documents CURRENT behavior. The analysis gates on QualType::isPointerType(),
// which is FALSE for BlockPointerType (e.g. 'void (^)(void)'). There is no
// isBlockPointerType() gate and no block-invoke handler in FlowNullability.cpp,
// so block pointers receive NO narrowing, NO invoke checking, and NO
// nullable-argument checking. Invoking a nil block IS a crash in ObjC (unlike
// messaging nil), so this is a real gap — captured here with FIXMEs.
//
// RUN: %clang_cc1 -fsyntax-only -fblocks -fflow-sensitive-nullability -fnullability-default=nullable -Wno-unused-value %s -verify

// --- Baseline: raw pointer still analyzed under -fblocks --------------------
void raw_pointer_under_fblocks(int * _Nullable p) {
  *p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  if (p)
    *p; // OK — narrowed
}

// --- Block pointers: NOT analyzed today -------------------------------------

// Invoking a nullable block. Calling a nil block crashes at runtime, so ideally
// this would warn; today it does not (BlockPointerType is not isPointerType()).
void invoke_nullable_block(void (^_Nullable blk)(void)) {
  blk(); // OK — no warning today
  // FIXME: block pointers are not yet flow-analyzed (gated on isPointerType;
  // BlockPointerType is excluded). Invoking a nil block crashes, so this SHOULD
  // warn once isBlockPointerType() is handled.
}

void invoke_nullable_block_guarded(void (^_Nullable blk)(void)) {
  if (blk)
    blk(); // OK — would be the narrowed-safe case once blocks are analyzed
}

void invoke_nonnull_block(void (^_Nonnull blk)(void)) {
  blk(); // OK — _Nonnull
}

// Block with a return value, invoked via the call path.
int invoke_nullable_block_with_result(int (^_Nullable compute)(int)) {
  return compute(0); // OK — no warning today
  // FIXME: should warn on invoking a nullable block once blocks are analyzed.
}

// Passing a nullable block to a _Nonnull block parameter. The argument check is
// gated on Param->getType()->isPointerType(), false for BlockPointerType, so no
// warning fires.
void takesNonnullBlock(void (^_Nonnull cb)(void));
void pass_nullable_block_to_nonnull_param(void (^_Nullable blk)(void)) {
  takesNonnullBlock(blk); // OK — no warning today
  // FIXME: nullable-block-to-_Nonnull-block-param is not diagnosed
  // (Param->getType()->isPointerType() is false for BlockPointerType).
}

// Local block variable with annotations.
void local_block_annotations(void) {
  void (^_Nullable blk)(void) = (void (^)(void))0;
  blk(); // OK — no warning today
  // FIXME: block pointer locals are not flow-analyzed (isPointerType gate).
}
