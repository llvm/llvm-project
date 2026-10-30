// RUN: %clang_analyze_cc1 -fexperimental-overflow-behavior-types -triple x86_64-unknown-linux-gnu -analyzer-checker=debug.DumpCFG %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// RUN: %clang_analyze_cc1 -fexperimental-overflow-behavior-types -triple x86_64-unknown-linux-gnu -analyzer-checker=core,debug.ExprInspection -verify %s

// An overflow checked operation ends its block with an overflow branch: the
// continuation is the first successor and the handler label the second.

void clang_analyzer_warnIfReached(void);
void fr(int *);

// CHECK-LABEL: int function_handler(__ob_trap int a)
// CHECK:      [B1]
// CHECK-NEXT:  __overflow:
// CHECK:      [B3]
// CHECK:        5: [B3.2] + [B3.4]
// CHECK-NEXT:   T: (Branch to the overflow handler if the check fails)
// CHECK-NEXT:   Preds (1): B4
// CHECK-NEXT:   Succs (2): B2 B1
int function_handler(int __ob_trap a) {
  a = a + 1;
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return a;
__overflow:
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return -1;
}

// Cleanups of the scopes left run on the way to the handler.
// CHECK-LABEL: int cleanup_scope(__ob_trap int a)
// CHECK:      [B1]
// CHECK-NEXT:  __overflow:
// CHECK:      [B3]
// CHECK:        8: [B3.4] + [B3.7]
// CHECK-NEXT:   T: (Branch to the overflow handler if the check fails)
// CHECK-NEXT:   Preds (1): B5
// CHECK-NEXT:   Succs (2): B2 B4
// CHECK:      [B4]
// CHECK-NEXT:   1: CleanupFunction (fr)
// CHECK-NEXT:   Preds (1): B3
// CHECK-NEXT:   Succs (1): B1
int cleanup_scope(int __ob_trap a) {
  {
    int __attribute__((cleanup(fr))) x = 0;
    a = a + x;
  }
  return a;
__overflow:
  // Not checked for reachability: the analyzer stops at cleanup functions.
  return -1;
}

// No handler: the operation does not split its block.
// CHECK-LABEL: int no_handler(__ob_trap int a)
// CHECK:      [B1]
// CHECK:        [B1.2] + [B1.4]
// CHECK:        return
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
int no_handler(int __ob_trap a) {
  return a + 1;
}

// __builtin_overflow_goto branches to its own label.
int builtin_handler(int __ob_trap a) {
  a = __builtin_overflow_goto(a + 1, l);
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return a;
l:
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return -1;
}
