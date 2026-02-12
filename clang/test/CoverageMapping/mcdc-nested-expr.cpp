// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1 | FileCheck %s

// CHECK-NOT: warning: unsupported MC/DC boolean expression; contains an operation with a nested boolean expression.

// "Split-nest" -- boolean expressions within boolean expressions.
extern bool bar(bool);
// CHECK: func_split_nest{{.*}}:
bool func_split_nest(bool a, bool b, bool c, bool d, bool e, bool f, bool g) {
  bool res = a && b && c && bar(d && e) && f && g;
  // CHECK:  Decision,File 0, [[@LINE-1]]:14 -> [[#L:@LINE-1]]:50 = M:10, C:6
  // CHECK:  Branch,File 0, [[#L]]:14 -> [[#L]]:15 = #9, (#0 - #9) [1,6,0]
  // CHECK:  Branch,File 0, [[#L]]:19 -> [[#L]]:20 = #10, (#9 - #10) [6,5,0]
  // CHECK:  Branch,File 0, [[#L]]:24 -> [[#L]]:25 = #8, (#7 - #8) [5,4,0]
  // CHECK:  Branch,File 0, [[#L]]:29 -> [[#L]]:40 = #6, (#5 - #6) [4,3,0]

  // The inner expr -- "d && e" (w/o parentheses)
  // CHECK:  Decision,File 0, [[#L]]:33 -> [[#L]]:39 = M:3, C:2
  // CHECK:  Branch,File 0, [[#L]]:33 -> [[#L]]:34 = #11, (#5 - #11) [1,2,0]
  // CHECK:  Branch,File 0, [[#L]]:38 -> [[#L]]:39 = #12, (#11 - #12) [2,0,0]

  // CHECK:  Branch,File 0, [[#L]]:44 -> [[#L]]:45 = #4, (#3 - #4) [3,2,0]
  // CHECK:  Branch,File 0, [[#L]]:49 -> [[#L]]:50 = #2, (#1 - #2) [2,0,0]
  return bar(res);
}

// The inner expr begins with the same Loc as the outer expr
// CHECK: func_condop{{.*}}:
bool func_condop(bool a, bool b, bool c) {
  return (a && b ? true : false) && c;
  // CHECK:  Decision,File 0, [[@LINE-1]]:10 -> [[#L:@LINE-1]]:38 = M:6, C:2
  // This covers outer parenthses.
  // CHECK:  Branch,File 0, [[#L]]:10 -> [[#L]]:33 = #1, (#0 - #1) [1,2,0]

  // The inner expr "a && b" (w/o parenthses)
  // CHECK:  Decision,File 0, [[#L]]:11 -> [[#L]]:17 = M:3, C:2
  // CHECK:  Branch,File 0, [[#L]]:11 -> [[#L]]:12 = #4, (#0 - #4) [1,2,0]
  // CHECK:  Branch,File 0, [[#L]]:16 -> [[#L]]:17 = #5, (#4 - #5) [2,0,0]

  // CHECK:  Branch,File 0, [[#L]]:37 -> [[#L]]:38 = #2, (#1 - #2) [2,0,0]
}

// __builtin_expect
// Treated as parentheses.
// CHECK: func_expect{{.*}}:
bool func_expect(bool a, bool b, bool c) {
  return a || __builtin_expect(b && c, true);
  // CHECK:  Decision,File 0, [[@LINE-1]]:10 -> [[#L:@LINE-1]]:45 = M:4, C:3
  // CHECK:  Branch,File 0, [[#L]]:10 -> [[#L]]:11 = (#0 - #1), #1 [1,0,2]
  // CHECK:  Branch,File 0, [[#L]]:32 -> [[#L]]:33 = #2, (#1 - #2) [2,3,0]
  // CHECK:  Branch,File 0, [[#L]]:37 -> [[#L]]:38 = #3, (#2 - #3) [3,0,0]
}

// LNot among BinOp(s)
// Doesn't split exprs.
// CHECK: func_lnot{{.*}}:
bool func_lnot(bool a, bool b, bool c, bool d) {
  return !(a || b) && !(c && d);
  // CHECK:  Decision,File 0, [[@LINE-1]]:10 -> [[#L:@LINE-1]]:32 = M:5, C:4
  // CHECK:  Branch,File 0, [[#L]]:12 -> [[#L]]:13 = (#0 - #2), #2 [1,0,3]
  // CHECK:  Branch,File 0, [[#L]]:17 -> [[#L]]:18 = (#2 - #3), #3 [3,0,2]
  // CHECK:  Branch,File 0, [[#L]]:25 -> [[#L]]:26 = #4, (#1 - #4) [2,4,0]
  // CHECK:  Branch,File 0, [[#L]]:30 -> [[#L]]:31 = #5, (#4 - #5) [4,0,0]
}
