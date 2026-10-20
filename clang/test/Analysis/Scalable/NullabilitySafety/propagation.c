// Evidence from two TUs, linked, then propagated over the pointer-flow graph.
// Propagation only vetoes: S::x has nonnull evidence but is reachable from the
// nullable parameter setx.p, so it is neither Nonnull nor Nullable.

// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nonnull %t/a.c \
// RUN:   --ssaf-extract-summaries=PointerFlow,NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=a.cu --ssaf-tu-summary-file=%t/a.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nonnull %t/b.c \
// RUN:   --ssaf-extract-summaries=PointerFlow,NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=b.cu --ssaf-tu-summary-file=%t/b.json
// RUN: clang-ssaf-linker %t/a.json %t/b.json -o %t/lu.json
// RUN: clang-ssaf-analyzer %t/lu.json -o %t/wpa.json \
// RUN:   -a NullabilityInferenceAnalysisResult
// RUN: FileCheck %s --input-file=%t/wpa.json
// RUN: FileCheck %s --input-file=%t/wpa.json --check-prefix=VETO

// CHECK-DAG: "id": [[S_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "1",[[:space:]]+"usr": }}"c:@F@setx"
// CHECK-DAG: "id": [[P_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "2",[[:space:]]+"usr": }}"c:@F@setx"
// CHECK-DAG: "id": [[Q_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "1",[[:space:]]+"usr": }}"c:@F@use"
// CHECK-DAG: "id": [[X_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "",[[:space:]]+"usr": }}"c:@S@S@FI@x"

// VETO:     "id": [[X_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "",[[:space:]]+"usr": }}"c:@S@S@FI@x"
// VETO:     "analysis_name": "NullabilityInferenceAnalysisResult",
// VETO:     "Nonnull": [
// VETO-NOT: "@": [[X_ID]]
// VETO:     "Nullable": [

// CHECK:      "analysis_name": "NullabilityInferenceAnalysisResult",
// CHECK:      "Nonnull": [
// CHECK-DAG:  "@": [[S_ID]]
// CHECK-DAG:  "@": [[Q_ID]]
// CHECK:      "Nullable": [
// CHECK-NEXT:   [
// CHECK-NEXT:     {
// CHECK-NEXT:       "@": [[P_ID]]
// CHECK-NEXT:     },
// CHECK-NEXT:     1
// CHECK-NEXT:   ]
// CHECK-NEXT: ],
// CHECK:      "NullableReachable": [
// CHECK-DAG:  "@": [[P_ID]]
// CHECK-DAG:  "@": [[X_ID]]
// CHECK:      "analysis_name":

//--- a.c
struct S { int *x; };
void setx(struct S *s, int *p) { s->x = p; }
int use(int *q) { return *q; }

//--- b.c
struct S { int *x; };
void setx(struct S *s, int *p);
int use(int *q);
int main(void) {
  static int v;
  struct S s;
  setx(&s, 0);
  s.x = &v;
  return use(&v);
}
