// End to end: extract, link, infer, write annotations back, merge the
// per-TU edits (the shared header is edited once), apply. Only _Nonnull is
// written: inferred _Nullable parameters and returns are reported as
// suggestions, and stores of null into fields are reported.

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

// RUN: FileCheck --check-prefix=HEADER --match-full-lines --input-file=%t/shared.h %s
// RUN: FileCheck --check-prefix=A --match-full-lines --input-file=%t/a.cpp %s
// RUN: FileCheck --check-prefix=REPORT --input-file=%t/a.sarif %s
// RUN: FileCheck --check-prefix=STORE --input-file=%t/b.sarif %s
// RUN: %{cc} -fnullability-safety -Werror %t/a.cpp
// RUN: %{cc} -fnullability-safety -Werror %t/b.cpp

// HEADER:      struct S { int *_Nonnull nonnull_field; int *nullable_field; int *reachable; };
// HEADER-NEXT: void take(int *_Nonnull p);
// HEADER-NEXT: void take_nullable(int *p);
// HEADER-NEXT: int *_Nonnull get(void);
// HEADER-NEXT: void set_reachable(struct S *s, int *p);

// A:      void take(int *_Nonnull p) { (void)*p; }
// A-NEXT: void take_nullable(int *p) {}
// A-NEXT: int *_Nonnull get() { static int v; return &v; }
// A-NEXT: void set_reachable(S *s, int *p) { s->reachable = p; }
// A-NEXT: void spelled_tight(int*_Nonnull q);
// A-NEXT: void pointer_to_pointer(int **_Nonnull pp);
// A-NEXT: void parenthesized(int (*_Nonnull p));
// A-NEXT: void function_pointer(void (*_Nonnull fp)(int));
// A-NEXT: void through_typedef(IntP _Nonnull p);
// A-NEXT: void const_pointer(int *_Nonnull const p);
// A-NEXT: auto trailing() -> int *_Nonnull { static int v; return &v; }
// A-NEXT: void through_macro(PTR p);
// A-NEXT: void already_nonnull(int *_Nonnull p);
// A-NEXT: void written_unspecified(int *_Null_unspecified p);
// A-NEXT: auto deduced() { static int v; return &v; }

// REPORT-DAG: "text": "a nullable value may reach this pointer through pointer flow; left unannotated"
// REPORT-DAG: "text": "pointer spelled through a macro is not annotated"
// REPORT-DAG: "text": "the declared type is not spelled with a '*' or a typedef name"
// REPORT-DAG: "text": "a null value reaches this pointer on every path from some caller or store; consider _Nullable (not written)"

// A field that receives null keeps its default (a field is typically null
// only in some lifecycle state); each null store is reported instead.
// STORE: "text": "stores null into field 'nullable_field', which is left unannotated and treated as non-null; check that it is not read while null"

//--- shared.h
struct S { int *nonnull_field; int *nullable_field; int *reachable; };
void take(int *p);
void take_nullable(int *p);
int *get(void);
void set_reachable(struct S *s, int *p);

//--- a.cpp
#include "shared.h"
#define PTR int *
typedef int *IntP;
void take(int *p) { (void)*p; }
void take_nullable(int *p) {}
int *get() { static int v; return &v; }
void set_reachable(S *s, int *p) { s->reachable = p; }
void spelled_tight(int*q);
void pointer_to_pointer(int **pp);
void parenthesized(int (*p));
void function_pointer(void (*fp)(int));
void through_typedef(IntP p);
void const_pointer(int *const p);
auto trailing() -> int * { static int v; return &v; }
void through_macro(PTR p);
void already_nonnull(int *_Nonnull p);
void written_unspecified(int *_Null_unspecified p);
auto deduced() { static int v; return &v; }

//--- b.cpp
#include "shared.h"
void spelled_tight(int *q);
void pointer_to_pointer(int **pp);
void parenthesized(int *p);
void function_pointer(void (*fp)(int));
void through_typedef(int *p);
void const_pointer(int *p);
int *trailing();
void through_macro(int *p);
void already_nonnull(int *p);
void written_unspecified(int *p);
int *deduced();
void fn(int);
void use(S *s) {
  static int v;
  s->nonnull_field = &v;
  s->nullable_field = nullptr;
  s->reachable = &v;
  take(get());
  take_nullable(nullptr);
  set_reachable(s, nullptr);
  int *vp = &v;
  spelled_tight(&v);
  pointer_to_pointer(&vp);
  parenthesized(&v);
  function_pointer(fn);
  through_typedef(&v);
  const_pointer(&v);
  through_macro(&v);
  already_nonnull(&v);
  written_unspecified(&v);
  (void)trailing();
  (void)deduced();
}
