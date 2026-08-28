// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macro-expansion-kind.c %s | FileCheck %s

// Test that macro expansions are marked as MacroExpansion in the coverage
// mapping dump, distinguishing them from regular Expansion regions (#include).

#define SIMPLE_MACRO(x) ((x) > 0 ? (x) : 0)

// CHECK-LABEL: test_func:
// CHECK: File 0, [[@LINE+1]]:22 -> [[@LINE+13]]:2 = #0
void test_func(void) {
  // CHECK: MacroExpansion,File 0, [[@LINE+1]]:16 -> [[@LINE+1]]:28
  int result = SIMPLE_MACRO(42);

  // CHECK: MacroExpansion,File 0, [[@LINE+1]]:13 -> [[@LINE+1]]:25
  int val = SIMPLE_MACRO(1);

  // CHECK: File 1,
  int x = 0;
  (void)result;
  (void)val;
  (void)x;
}
