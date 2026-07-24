// Flow-sensitive nullability in Objective-C.
//
// This test documents the CURRENT behavior of the analysis with respect to
// Objective-C constructs. The analysis gates pointer handling on
// QualType::isPointerType() (FlowNullability.cpp), which is TRUE only for C/C++
// raw PointerType and FALSE for ObjC object pointers (ObjCObjectPointerType,
// e.g. 'NSFoo *') and block pointers (BlockPointerType). Therefore:
//   - C raw pointers in a .m file ARE analyzed (proves the pass runs in ObjC).
//   - ObjC object pointers and ObjC message sends / property access are NOT
//     analyzed today (no ObjCObjectPointerType gate, no ObjCMessageExpr /
//     ObjCPropertyRefExpr / ObjCIvarRefExpr handler in FlowNullability.cpp).
//   - 'NSError **' is a raw PointerType at its OUTER level, so the C-pointer
//     deref path DOES fire on it.
//
// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fflow-sensitive-nullability -fnullability-default=nullable -Wno-unused-value %s -verify

__attribute__((objc_root_class))
@interface NSObject
@end

__attribute__((objc_root_class))
@interface NSError
@end

@interface Widget : NSObject
@property (nullable) Widget *child;
@property (nullable) NSError *lastError;
- (nullable Widget *)maybeChild;
- (nonnull Widget *)requiredChild;
- (void)consume:(nonnull Widget *)w;
@end

// --- Baseline: prove the analysis runs in ObjC mode -------------------------
// A C raw pointer (PointerType) in a .m file. isPointerType() is TRUE, so the
// existing dereference checking and narrowing apply exactly as in C/C++.

void raw_pointer_is_analyzed(int * _Nullable p) {
  *p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  if (p) {
    *p; // OK — narrowed
  }
}

void raw_pointer_nonnull(int * _Nonnull safe) {
  *safe; // OK — _Nonnull
}

// --- ObjC object pointers: NOT analyzed today -------------------------------
// 'Widget *' is an ObjCObjectPointerType, not a PointerType, so none of the
// isPointerType()-gated logic applies. No narrowing, no dereference/argument
// checking. We assert NO diagnostics to lock in current behavior.

// Message send to a nullable receiver. In ObjC, messaging nil is DEFINED
// (returns nil / zero), so a nil receiver is NOT a crash. The analysis
// correctly does NOT warn here — and also would not warn even if it tracked
// ObjC pointers, because message-send-to-nil is safe by language semantics.
void message_send_to_nullable(Widget * _Nullable w) {
  [w requiredChild]; // OK — messaging nil is defined; no warning expected
  // FIXME: ObjC object pointers are not yet flow-analyzed (gated on
  // isPointerType); should narrow/track once isObjCObjectPointerType is handled.
}

// Property access on a nullable object pointer. Property access lowers to a
// message send, so it is likewise nil-safe and not analyzed.
void property_access_on_nullable(Widget * _Nullable w) {
  Widget *c = w.child; // OK — no warning
  (void)c;
  // FIXME: ObjC object pointers / @property access are not yet flow-analyzed
  // (gated on isPointerType; no ObjCPropertyRefExpr handler in
  // FlowNullability.cpp). Should narrow/track once isObjCObjectPointerType is
  // handled.
}

// Passing a nullable ObjC pointer to a _Nonnull ObjC parameter. The C-pointer
// argument check (handleNullableArgument) is gated on Param->getType()->
// isPointerType(), which is FALSE for 'Widget *', so no warning fires today.
void pass_nullable_to_nonnull_objc_param(Widget * _Nullable w, Widget *receiver) {
  [receiver consume:w]; // OK — no warning (ObjC params not checked)
  // FIXME: nullable-argument-to-_Nonnull-ObjC-param is not yet diagnosed
  // (Param->getType()->isPointerType() is false for ObjCObjectPointerType).
}

// Local ObjC object pointer with _Nullable / _Nonnull annotations. Neither
// declared-type seeding nor narrowing applies to ObjCObjectPointerType today.
void objc_local_annotations(void) {
  Widget * _Nullable maybe = (Widget *)0;
  [maybe requiredChild]; // OK — no warning
  Widget * _Nonnull req = maybe; // OK — no nullable-to-nonnull flow warning here
  (void)req;
  // FIXME: ObjC object pointer locals are not flow-analyzed (isPointerType gate).
}

// --- NSError ** out-param idiom ---------------------------------------------
// 'NSError **' is a raw PointerType at the OUTER level (pointer to an ObjC
// object pointer), so isPointerType() is TRUE and the C-pointer deref path
// DOES apply to '*error'. Under -fnullability-default=nullable an unannotated
// 'NSError **' parameter is treated as nullable, so dereferencing it without a
// guard warns — and a null check narrows it.

void out_param_deref_unguarded(NSError ** _Nullable error) {
  *error = (NSError *)0; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void out_param_deref_guarded(NSError ** _Nullable error) {
  if (error) {
    *error = (NSError *)0; // OK — narrowed
  }
}

void out_param_nonnull(NSError ** _Nonnull error) {
  *error = (NSError *)0; // OK — _Nonnull
}
