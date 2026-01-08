// This test ensures that the --quiet flag only suppresses the "X warnings generated" 
// message while keeping all diagnostic information including caret indicators (^).

// RUN: clang-tidy -checks=-*,readability-magic-numbers,clang-diagnostic-sign-compare %s -- \
// RUN:   -Wsign-compare 2>&1 | FileCheck %s --check-prefix=CHECK-NORMAL
// RUN: clang-tidy -checks=-*,readability-magic-numbers,clang-diagnostic-sign-compare -quiet %s -- \
// RUN:   -Wsign-compare 2>&1 | FileCheck %s --check-prefix=CHECK-QUIET

// CHECK-NORMAL: 2 warnings generated
// CHECK-NORMAL-DAG: warning: 42 is a magic number
// CHECK-NORMAL-DAG: {{[ ]*\^}}
// CHECK-NORMAL-DAG: warning: comparison of integers of different signs
// CHECK-NORMAL-DAG: {{[ ]*~ \^ ~}}

// CHECK-QUIET-NOT: {{[0-9]+}} warning{{s?}} generated
// CHECK-QUIET-DAG: warning: 42 is a magic number
// CHECK-QUIET-DAG: {{[ ]*\^}}
// CHECK-QUIET-DAG: warning: comparison of integers of different signs
// CHECK-QUIET-DAG: {{[ ]*~ \^ ~}}

int main() {
  const int CONST_VAL = 10;
  int x = 42; // trigger 'readability-magic-numbers' with caret: ^
  unsigned int y = CONST_VAL;
  return x < y; // trigger 'clang-diagnostic-sign-compare' with caret: ^
}
