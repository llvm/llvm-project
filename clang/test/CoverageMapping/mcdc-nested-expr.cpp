// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2> %t.stderr.txt | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARN < %t.stderr.txt

// "Split-nest" -- boolean expressions within boolean expressions.
extern bool bar(bool);
// CHECK: func_split_nest{{.*}}:
bool func_split_nest(bool a, bool b, bool c, bool d, bool e, bool f, bool g) {
  // WARN: :[[@LINE+1]]:14: warning: unsupported MC/DC boolean expression; contains an operation with a nested boolean expression.
  bool res = a && b && c && bar(d && e) && f && g;
  return bar(res);
}

// The inner expr begins with the same Loc as the outer expr
// CHECK: func_condop{{.*}}:
bool func_condop(bool a, bool b, bool c) {
  // WARN: :[[@LINE+1]]:10: warning: unsupported MC/DC boolean expression; contains an operation with a nested boolean expression.
  return (a && b ? true : false) && c;
}

// __builtin_expect
// Treated as parentheses.
// CHECK: func_expect{{.*}}:
bool func_expect(bool a, bool b, bool c) {
  // WARN: :[[@LINE+1]]:10: warning: unsupported MC/DC boolean expression; contains an operation with a nested boolean expression.
  return a || __builtin_expect(b && c, true);
}

// LNot among BinOp(s)
// Doesn't split exprs.
// CHECK: func_lnot{{.*}}:
bool func_lnot(bool a, bool b, bool c, bool d) {
  // WARN: :[[@LINE+1]]:10: warning: unsupported MC/DC boolean expression; contains an operation with a nested boolean expression.
  return !(a || b) && !(c && d);
}
