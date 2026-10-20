// _Nonnull is written only when every value stored to the entity is proven
// non-null, across translation units. A store of an unknown value vetoes it;
// a copy of another entity counts once that entity is inferred; a cycle of
// copies with no proven store is not inferred; parameters of virtual methods
// are never inferred (overriders are called through the base). An aggregate
// initializer can veto but never makes a field a candidate: a struct filled in
// by a static table is often also filled in by code outside the link unit.

// REQUIRES: clang-apply-replacements
// RUN: rm -rf %t && split-file %s %t && mkdir -p %t/edits %t/merged
// DEFINE: %{cc} = %clang_cc1 -fsyntax-only -fnullability-default=nonnull -std=c++17
// DEFINE: %{tu} = unset
// DEFINE: %{extract} = %{cc} %t/%{tu}.cpp \
// DEFINE:   --ssaf-extract-summaries=PointerFlow,NullabilitySafety \
// DEFINE:   --ssaf-compilation-unit-id=%{tu}.cu --ssaf-tu-summary-file=%t/%{tu}.json
// DEFINE: %{transform} = %{cc} %t/%{tu}.cpp \
// DEFINE:   --ssaf-source-transformation=nullability-annotations \
// DEFINE:   --ssaf-global-scope-analysis-result=%t/wpa.json \
// DEFINE:   --ssaf-src-edit-file=%t/edits/%{tu}.yaml \
// DEFINE:   --ssaf-transformation-report-file=%t/%{tu}.sarif \
// DEFINE:   --ssaf-compilation-unit-id=%{tu}.cu --ssaf-link-unit-id=lu

// REDEFINE: %{tu} = a
// RUN: %{extract}
// REDEFINE: %{tu} = b
// RUN: %{extract}
// RUN: clang-ssaf-linker %t/a.json %t/b.json -o %t/lu.json
// RUN: clang-ssaf-analyzer %t/lu.json -o %t/wpa.json \
// RUN:   -a NullabilityInferenceAnalysisResult
// REDEFINE: %{tu} = a
// RUN: %{transform}
// REDEFINE: %{tu} = b
// RUN: %{transform}
// RUN: clang-ssaf-src-edit-merge %t/edits/a.yaml %t/edits/b.yaml \
// RUN:   -o %t/merged/merged.yaml
// RUN: clang-apply-replacements %t/merged
// RUN: FileCheck --match-full-lines --input-file=%t/shared.h %s

// CHECK:      struct S { int *unknown_store; int *_Nonnull proven; int *_Nonnull via_param; int *via_unknown_param; int *agg_unknown; int *agg_only; };
// CHECK-NEXT: struct Cycle { int *a; int *b; };
// CHECK-NEXT: struct V { virtual void f(int *p); void g(int *_Nonnull q); };
// CHECK-NEXT: int *mystery();
// CHECK-NEXT: void set_proven(S *s, int *_Nonnull p);
// CHECK-NEXT: void set_maybe(S *s, int *p);
// CHECK-NEXT: int *_Nonnull get_proven();
// CHECK-NEXT: int *get_forward();

//--- shared.h
struct S { int *unknown_store; int *proven; int *via_param; int *via_unknown_param; int *agg_unknown; int *agg_only; };
struct Cycle { int *a; int *b; };
struct V { virtual void f(int *p); void g(int *q); };
int *mystery();
void set_proven(S *s, int *p);
void set_maybe(S *s, int *p);
int *get_proven();
int *get_forward();

//--- a.cpp
#include "shared.h"
static int v;
void set_proven(S *s, int *p) { s->via_param = p; }
void set_maybe(S *s, int *p) { s->via_unknown_param = p; }
int *get_proven() { return &v; }
int *get_forward() { return mystery(); }
void V::f(int *p) { v = *p; }
void V::g(int *q) { v = *q; }
void swap_cycle(Cycle *c) { c->a = c->b; c->b = c->a; }

//--- b.cpp
#include "shared.h"
static int w;
void stores(S *s) {
  s->unknown_store = &w;
  s->unknown_store = mystery();
  s->proven = &w;
  s->proven = get_proven();
  set_proven(s, &w);
  set_maybe(s, &w);
  set_maybe(s, mystery());
  S t = {&w, &w, &w, &w, mystery(), &w};
  (void)t;
}
void calls(V *o) {
  o->f(&w);
  o->g(&w);
}
