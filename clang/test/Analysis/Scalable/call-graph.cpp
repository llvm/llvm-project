// RUN: rm -rf %t.summary.json
// RUN: %clang_cc1 -fsyntax-only %s \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-tu-summary-file=%t.summary.json
// RUN: FileCheck %s --match-full-lines --input-file=%t.summary.json

// caller() has a direct callee and no virtual callees.
//   CHECK-LABEL: "entity_summary": {
//   CHECK-DAG:     "def": {
//   CHECK-DAG:       "col": {{[0-9]+}},
//   CHECK-DAG:       "file": "{{.+}}",
//   CHECK-DAG:       "line": {{[0-9]+}}
//   CHECK-DAG:     "pretty_name": "caller()",
//   CHECK-DAG:     "direct_callees": [
//   CHECK-DAG:     "virtual_callees": []

// polymorphic() has a virtual callee and no direct callees.
//   CHECK-LABEL: "entity_summary": {
//   CHECK-DAG:     "def": {
//   CHECK-DAG:       "col": {{[0-9]+}},
//   CHECK-DAG:       "file": "{{.+}}",
//   CHECK-DAG:       "line": {{[0-9]+}}
//   CHECK-DAG:     "pretty_name": "polymorphic(Base &)",
//   CHECK-DAG:     "direct_callees": [],
//   CHECK-DAG:     "virtual_callees": [

struct Base {
  virtual ~Base();
  virtual void vmethod();
};

void callee();

void caller() {
  callee();
}

void polymorphic(Base &b) {
  b.vmethod();
}
