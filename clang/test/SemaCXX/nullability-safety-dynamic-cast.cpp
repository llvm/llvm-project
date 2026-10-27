// A pointer dynamic_cast yields null whenever the runtime check fails, so its
// result is nullable in both nullability-default modes. Shared expectations
// use the expected prefix; mode-specific ones use nullable / nonnull.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 %s -verify=expected,nullable
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -Wno-nullable-to-nonnull-conversion -std=c++17 %s -verify=expected,nonnull
// RUN: rm -f %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nullable -std=c++17 %s \
// RUN:   --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu --ssaf-tu-summary-file=%t.json
// RUN: %python %S/../Analysis/Scalable/NullabilitySafety/Inputs/decode-summary.py %t.json | FileCheck %s --check-prefix=EVIDENCE

struct Base {
  virtual ~Base();
};
struct Derived : Base {
  int value;
};

void takesNonnull(Derived *_Nonnull);

Derived *returnDynamic(Base *_Nonnull p) {
  return dynamic_cast<Derived *>(p);
}

Derived *returnStatic(Derived *_Nonnull p) {
  return static_cast<Derived *>(p);
}

void dynamicCastPropagation(Base *_Nonnull p) {
  Derived *d = dynamic_cast<Derived *>(p);
  d->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  takesNonnull(dynamic_cast<Derived *>(p)); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
}

void preservingCastControls(Derived *_Nonnull p) {
  Derived *d = static_cast<Derived *>(p);
  d->value = 1;
  takesNonnull(static_cast<Derived *>(p));
}

void narrowedDynamicCastIsSafe(Base *_Nonnull p) {
  if (Derived *d = dynamic_cast<Derived *>(p))
    d->value = 1;
}

// returnDynamic's unannotated return type is what the caller sees, so only the
// nullable default reports this dereference.
void callerStillWarns(Base *_Nonnull p) {
  returnDynamic(p)->value = 1; // nullable-warning{{dereference of nullable pointer}} nullable-note{{add a null check}}
}

// ===----------------------------------------------------------------------===//
// dynamic_cast initializer is nullable, never narrowing
// ===----------------------------------------------------------------------===//
// Under -fnullability-default=nonnull the cast's unannotated result type is
// not nullable by type, so the initializer itself must carry the taint;
// judging the (non-null) cast source would wrongly narrow the variable.

void initFromDynamicCastWarns(Base *_Nonnull p) {
  Derived *q = dynamic_cast<Derived *>(p);
  q->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void initInConditionNarrows(Base *_Nonnull p) {
  if (auto *q = dynamic_cast<Derived *>(p))
    q->value = 1;
}

void initThenCheckNarrows(Base *_Nonnull p) {
  Derived *q = dynamic_cast<Derived *>(p);
  if (q)
    q->value = 1;
}

void reassignmentClearsTaint(Base *_Nonnull p) {
  Derived local;
  Derived *q = dynamic_cast<Derived *>(p);
  q = &local;
  q->value = 1;
}

void nonnullInitFromDynamicCastWarns(Base *_Nonnull p) {
  Derived *_Nonnull q = dynamic_cast<Derived *>(p); // expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check before assigning}}
  q->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void assignFromDynamicCastWarns(Base *_Nonnull p) {
  Derived local;
  Derived *q = &local;
  q = dynamic_cast<Derived *>(p);
  q->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

struct Holder {
  Derived *d;
};

void memberAssignFromDynamicCastWarns(Holder &h, Base *_Nonnull p) {
  h.d = dynamic_cast<Derived *>(p);
  h.d->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// ===----------------------------------------------------------------------===//
// A dynamic_cast nested under arithmetic or another cast is still the origin
// ===----------------------------------------------------------------------===//
// Unwrapping at a dereference site must stop at the dynamic_cast rather than
// judging its (non-null) source.

void derefDynamicCastPlusOne(Base *_Nonnull p) {
  (*(dynamic_cast<Derived *>(p) + 1)).value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void arrowDynamicCastPlusOne(Base *_Nonnull p) {
  (dynamic_cast<Derived *>(p) + 1)->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void staticCastOfDynamicCastPlusOne(Base *_Nonnull p) {
  static_cast<Derived *>(dynamic_cast<Derived *>(p) + 1)->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void staticCastOfDynamicCast(Base *_Nonnull p) {
  static_cast<Derived *>(dynamic_cast<Derived *>(p))->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  (*static_cast<Derived *>(dynamic_cast<Derived *>(p))).value = 2; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void starDynamicCast(Base *_Nonnull p) {
  (*dynamic_cast<Derived *>(p)).value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void narrowedDynamicCastArithmeticIsSafe(Base *_Nonnull p) {
  if (auto *q = dynamic_cast<Derived *>(p)) {
    (*(q + 1)).value = 1;
    (q + 1)->value = 2;
    static_cast<Derived *>(q + 1)->value = 3;
  }
}

// EVIDENCE-NOT:  {{.}}
// EVIDENCE:      c:@F@callerStillWarns#*$@S@Base# NonnullEvidence c:@F@returnDynamic#*$@S@Base# param 1
// EVIDENCE-NEXT: c:@F@dynamicCastPropagation#*$@S@Base# NullableEvidence c:@F@takesNonnull#*$@S@Derived# param 1
// EVIDENCE-NEXT: c:@F@memberAssignFromDynamicCastWarns#&$@S@Holder#*$@S@Base# NullableEvidence c:@S@Holder@FI@d
// EVIDENCE-NEXT: c:@F@preservingCastControls#*$@S@Derived# NonnullEvidence c:@F@takesNonnull#*$@S@Derived# param 1
// EVIDENCE-NEXT: c:@F@returnDynamic#*$@S@Base# NullableEvidence c:@F@returnDynamic#*$@S@Base# return
// EVIDENCE-NEXT: c:@F@returnStatic#*$@S@Derived# AllReturnsNonnull c:@F@returnStatic#*$@S@Derived# return
// EVIDENCE-NEXT: c:@F@returnStatic#*$@S@Derived# NonnullEvidence c:@F@returnStatic#*$@S@Derived# return
// EVIDENCE-NOT:  {{.}}
