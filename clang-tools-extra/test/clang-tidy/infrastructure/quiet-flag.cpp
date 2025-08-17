// RUN: clang-tidy -checks=-*,readability-magic-numbers %s -- 2>&1 | FileCheck %s --check-prefix=CHECK-NORMAL
// RUN: clang-tidy -checks=-*,readability-magic-numbers -quiet %s -- 2>&1 | FileCheck %s --check-prefix=CHECK-QUIET

// Normal mode should show warning count
// CHECK-NORMAL: 1 warning generated
// CHECK-NORMAL: warning: 42 is a magic number

// Quiet mode should suppress warning count but show diagnostic
// CHECK-QUIET: warning: 42 is a magic number
// CHECK-QUIET-NOT: {{[0-9]+}} warning{{s?}} generated

int main() {
  int x = 42; // This will trigger readability-magic-numbers
}
