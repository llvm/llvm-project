// RUN: clang-tidy -checks=-*,readability-magic-numbers %t.cpp -- 2>&1 | FileCheck %s --check-prefix=CHECK-NORMAL
// RUN: clang-tidy -checks=-*,readability-magic-numbers -quiet %t.cpp -- 2>&1 | FileCheck %s --check-prefix=CHECK-QUIET

// Normal mode should show warning count
// CHECK-NORMAL: 1 warning generated
// CHECK-NORMAL: warning: 42 is a magic number

// Quiet mode should suppress warning count 
// CHECK-QUIET-NOT: 1 warning generated
// CHECK-QUIET: warning: 42 is a magic number

int main() {
  int x = 42;
}
