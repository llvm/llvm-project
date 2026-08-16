// Test that UnsafeBufferReachableAnalysis excludes type-constrained pointers

// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang_cc1 -fsyntax-only %s \
// RUN:   --ssaf-extract-summaries=PointerFlow,UnsafeBufferUsage,TypeConstrainedPointers \
// RUN:   --ssaf-tu-summary-file=%t/tu.summary.json \
// RUN:   --ssaf-compilation-unit-id="tu-1"

// RUN: clang-ssaf-linker %t/tu.summary.json -o %t/lu.json

// RUN: clang-ssaf-analyzer %t/lu.json -o %t/wpa.json \
// RUN:   -a UnsafeBufferReachableAnalysisResult

// The CHECK lines below use readable tokens instead of inline FileCheck regex.
// Expand the tokens into regex, then run FileCheck on the expanded copy:
//   $NS - skip the "namespace" array, up to its closing ']'.
//   $WS - whitespace, possibly spanning newlines.
//   $PTR_L1 - a reachable-set entry closing as "}, 1 ]", i.e. pointer level 1.
// RUN: sed -e 's|$NS|{{([^]]\|[[:space:]])+\\],}}|g' \
// RUN:     -e 's|$WS|{{[[:space:]]+}}|g' \
// RUN:     -e 's|$PTR_L1|{{[[:space:]]+\\},[[:space:]]+1[[:space:]]+\\]}}|g' \
// RUN:     %s > %t/checks.txt
// RUN: FileCheck %t/checks.txt --input-file=%t/wpa.json


int main(int argc, char **argv) {
  argv[5] = 0; // unsafe use of a type-constrained pointer
  return 0;
}

// 'q' is an ordinary unsafe pointer parameter and must remain in the result.
void foo(int *q) {
  q[5] = 0;
}

// CHECK-DAG: "id": [[Q_ID:[0-9]+]],$NS$WS"suffix": "1",$WS"usr": "c:@F@foo#*I#"
// CHECK-DAG: "id": [[ARGV_ID:[0-9]+]],$NS$WS"suffix": "2",$WS"usr": "c:@F@main{{.*}}"

// Contributor function ids:
// CHECK-DAG: "id": [[CONTRIBUTOR_FOO:[0-9]+]],$NS$WS"suffix": "",$WS"usr": "c:@F@foo#*I#"
// CHECK-DAG: "id": [[CONTRIBUTOR_MAIN:[0-9]+]],$NS$WS"suffix": "",$WS"usr": "c:@F@main{{.*}}"

// 'argv' is reported as type-constrained.
// CHECK: "analysis_name": "TypeConstrainedPointersAnalysisResult"
// CHECK: "@": [[ARGV_ID]]

// In the reachable result 'q' is present but 'argv' is not.
// CHECK: "analysis_name": "UnsafeBufferReachableAnalysisResult"

// 'foo' contributes unsafe pointer 'q'.
// CHECK: "@": [[CONTRIBUTOR_FOO]]$WS},$WS[
// CHECK: "@": [[Q_ID]]$PTR_L1
// CHECK-NOT: "@":

// 'main' contributes nothing: 'argv' is type-constrained and excluded.
// CHECK: "@": [[CONTRIBUTOR_MAIN]]$WS},$WS[
// CHECK-NOT: "@": [[ARGV_ID]]

// CHECK: "analysis_name"
