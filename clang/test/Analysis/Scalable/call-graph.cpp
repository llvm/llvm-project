// RUN: rm -rf %t.summary.json
// RUN: %clang_cc1 -fsyntax-only %s \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-tu-summary-file=%t.summary.json

// Check that the JSON validation passes.
// TODO: Enable the next line once the LinkageTable is populated.
// R U N: clang-ssaf-format --type=tu %t.summary.json

// Check that the JSON has plausible content irrespective of the order of the fields.
// RUN: FileCheck %s --match-full-lines --input-file=%t.summary.json
//   CHECK-DAG: "direct_callees": [
//   CHECK-DAG: "pretty_name": "example()",
//   CHECK-DAG: "virtual_callees": []
//   CHECK-DAG: "summary_name": "CallGraph"

void no_body();
void example() {
  no_body();
}
