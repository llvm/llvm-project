// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t/test.cpp\",\"file\":\"%/t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: echo "Checks: '-*,readability-magic-numbers'" > %t/.clang-tidy
// RUN: cp "%s" "%t/test.cpp"
// RUN: cd "%t"

// RUN: %run_clang_tidy -quiet -hide-progress "test.cpp" 2>&1 | FileCheck %s --check-prefix=CHECK-RUN-QUIET
// CHECK-RUN-QUIET-NOT: Running clang-tidy in {{[1-9][0-9]*}} threads for
// CHECK-RUN-QUIET-NOT: {{[0-9]+}} warning{{s?}} generated
// CHECK-RUN-QUIET-NOT: [1/1]
// CHECK-RUN-QUIET: 42 is a magic number;

// REQUIRES: shell
// RUN: sed 's/42/99/' %s > %t-diff.cpp

// RUN: not diff -U0 %s %t-diff.cpp | %clang_tidy_diff -checks=-*,readability-magic-numbers -quiet -hide-progress -- -std=c++11 2>&1 | FileCheck %s --check-prefix=CHECK-DIFF-QUIET
// CHECK-DIFF-QUIET-NOT: Running clang-tidy in {{[1-9][0-9]*}} threads...
// CHECK-DIFF-QUIET-NOT: {{[0-9]+}} warning{{s?}} generated
// CHECK-DIFF-QUIET: 99 is a magic number;

int main() {
  int x = 42;
}
