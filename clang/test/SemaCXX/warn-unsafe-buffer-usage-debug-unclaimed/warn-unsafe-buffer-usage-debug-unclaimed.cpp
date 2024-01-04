// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions \
// RUN:            -mllvm -debug-only=SafeBuffers \
// RUN:            -std=c++20 -verify=expected %s

// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions \
// RUN:            -mllvm -debug-only=SafeBuffers \
// RUN:            -std=c++20 %s                  \
// RUN:            2>&1 | grep 'The unclaimed DRE trace:' \
// RUN:                 | sed 's/^The unclaimed DRE trace://' \
// RUN:                 | %analyze_safe_buffer_debug_notes \
// RUN:                 | FileCheck %s

// This debugging facility is only available in debug builds.
//
// REQUIRES: asserts
// REQUIRES: shell

void test_unclaimed_use(int *p) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  p++;           //  expected-note{{used in pointer arithmetic here}} \
                     expected-note{{safe buffers debug: failed to produce fixit for 'p' : has an unclaimed use\n \
 The unclaimed DRE trace: DeclRefExpr, UnaryOperator(++), CompoundStmt}}
  *((p + 1) + 1); // expected-warning{{unsafe pointer arithmetic}}                      \
                     expected-note{{used in pointer arithmetic here}}			\
		     expected-note{{safe buffers debug: failed to produce fixit for 'p' : has an unclaimed use\n \
  The unclaimed DRE trace: DeclRefExpr, ImplicitCastExpr(LValueToRValue), BinaryOperator(+), ParenExpr, BinaryOperator(+), ParenExpr, UnaryOperator(*), CompoundStmt}}
  p -= 1;         // expected-note{{used in pointer arithmetic here}} \
		     expected-note{{safe buffers debug: failed to produce fixit for 'p' : has an unclaimed use\n \
  The unclaimed DRE trace: DeclRefExpr, BinaryOperator(-=), CompoundStmt}}
  p--;            // expected-note{{used in pointer arithmetic here}} \
 		     expected-note{{safe buffers debug: failed to produce fixit for 'p' : has an unclaimed use\n \
  The unclaimed DRE trace: DeclRefExpr, UnaryOperator(--), CompoundStmt}}
  p[5] = 5;       // expected-note{{used in buffer access here}}
}

// CHECK: Root # 1
// CHECK: |- DeclRefExpr # 4
// CHECK: |-- UnaryOperator(++) # 1
// CHECK: |--- CompoundStmt # 1
// CHECK: |-- ImplicitCastExpr(LValueToRValue) # 1
// CHECK: |--- BinaryOperator(+) # 1
// CHECK: |---- ParenExpr # 1
// CHECK: |----- BinaryOperator(+) # 1
// CHECK: |------ ParenExpr # 1
// CHECK: |------- UnaryOperator(*) # 1
// CHECK: |-------- CompoundStmt # 1
// CHECK: |-- BinaryOperator(-=) # 1
// CHECK: |--- CompoundStmt # 1
// CHECK: |-- UnaryOperator(--) # 1
// CHECK: |--- CompoundStmt # 1
