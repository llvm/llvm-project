// RUN: rm -rf %t.json
// RUN: %clang_cc1 -fsyntax-only %s --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu-1 --ssaf-tu-summary-file=%t.json
// RUN: %python %S/Inputs/decode-summary.py %t.json | FileCheck %s

struct S { int *f; };
void take(int *p);
int *_Nonnull get(int *_Nullable unused) { static int x; return &x; }
void g(struct S *s, int *_Nullable q) {
  take(q);
  take(get(q));
  s->f = 0;
}

// CHECK-NOT:  {{.}}
// CHECK:      c:@F@g NonnullEvidence c:@F@take param 1
// CHECK-NEXT: c:@F@g NullableEvidence c:@F@get param 1
// CHECK-NEXT: c:@F@g NullableEvidence c:@F@take param 1
// CHECK-NEXT: c:@F@g NullableEvidence c:@S@S@FI@f
// CHECK-NEXT: c:@F@get AllReturnsNonnull c:@F@get return
// CHECK-NEXT: c:@F@get NonnullEvidence c:@F@get return
// CHECK-NOT:  {{.}}
