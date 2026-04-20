// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fsyntax-only %t/caller.cpp \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-tu-summary-file=%t/caller.summary.json
// RUN: FileCheck --match-full-lines --check-prefix=CALLER %t/caller.cpp --input-file=%t/caller.summary.json

// RUN: %clang_cc1 -fsyntax-only %t/polymorphic.cpp \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-tu-summary-file=%t/polymorphic.summary.json
// RUN: FileCheck --match-full-lines --check-prefix=POLYMORPHIC %t/polymorphic.cpp --input-file=%t/polymorphic.summary.json

//--- caller.cpp
// polymorphic() has a virtual callee and no direct callees.
//   CALLER-DAG:     "def": {
//   CALLER-DAG:       "col": {{[0-9]+}},
//   CALLER-DAG:       "file": "{{.+}}",
//   CALLER-DAG:       "line": {{[0-9]+}}
//   CALLER-DAG:     "pretty_name": "caller()",
//   CALLER-DAG:     "direct_callees": [
//   CALLER-DAG:     "virtual_callees": []
void callee();
void caller() {
  callee();
}

//--- polymorphic.cpp
// polymorphic() has a virtual callee and no direct callees.
//   POLYMORPHIC-DAG:     "def": {
//   POLYMORPHIC-DAG:       "col": {{[0-9]+}},
//   POLYMORPHIC-DAG:       "file": "{{.+}}",
//   POLYMORPHIC-DAG:       "line": {{[0-9]+}}
//   POLYMORPHIC-DAG:     "pretty_name": "polymorphic(Base &)",
//   POLYMORPHIC-DAG:     "direct_callees": [],
//   POLYMORPHIC-DAG:     "virtual_callees": [
struct Base {
  virtual ~Base();
  virtual void vmethod();
};

void polymorphic(Base &b) {
  b.vmethod();
}
